# talk

CLI-first voice assistant with a managed background daemon. Speaks text via TTS, listens for a response via STT with barge-in support, and shows a live TUI.

The daemon keeps models loaded between invocations so subsequent calls are instant. It also runs a macOS audio-io subprocess (VoiceProcessingIO with hardware echo cancellation) that persists across calls.

## Quick start

```bash
# Single mode: speak, listen, show TUI, exit
uv run talk/server.py "Hello, how are you?"

# Continuous mode: interactive loop (default when no text and tty)
uv run talk/server.py

# Read text from stdin (single mode)
echo "What is the weather?" | uv run talk/server.py

# Explicit mode selection
uv run talk/server.py --mode continuous "Hello"

# Listen only (no TTS)
uv run talk/server.py -t 10s

# Stop the daemon
uv run talk/server.py --kill
```

### TUI output

The CLI shows a live TUI with TTS and STT streams:

```
14:32:01 ⏺ Hello, how are you?
14:32:04 ❯ I'm doing great
```

- **⏺** prefix: TTS text appears word-by-word as audio plays
- **❯** prefix: STT text streams in-place, updating as recognition progresses (gray background)
- Timestamps are dimmed and italic

### Modes

- **single**: One exchange (speak text, listen for response, exit). Auto-selected when text is provided or stdin is piped.
- **continuous**: Interactive loop reading stdin lines. Type new input to interrupt current TTS. Between exchanges, STT streams live (the "(waiting for input or voice...)" prompt is replaced in-place by incoming STT). Auto-selected when running interactively with no text argument.

### Input modes (`--input`)

- **line** (default): Text is sent when you press Enter.
- **unbuffered**: Text is sent as it arrives (with 300ms debounce). Useful for piped input where upstream writes character-by-character.

## Dependencies

All dependencies are declared inline in `server.py` (PEP 723 script metadata) and resolved automatically by `uv run`:

- **[pocket-tts-mlx](https://pypi.org/project/pocket-tts-mlx/)** -- MLX text-to-speech
- **[moshi-mlx](https://pypi.org/project/moshi-mlx/)** -- MLX speech-to-text (Moshi 1B)
- **starlette** + **uvicorn** -- HTTP server
- **sphn** -- Audio I/O utilities
- **httpx** -- CLI HTTP client

No local dependencies or editable installs required.

## Architecture

```
CLI invocation                   Background daemon (persists)
─────────────────               ──────────────────────────────
./talk/server.py "Hi"  ──POST /talk──>  Starlette HTTP server
                                            │
                       <── NDJSON stream ── ├── TTS (pocket-tts-mlx)
                                            ├── STT (Moshi 1B MLX)
                                            └── audio-io subprocess
                                                 ├── stdin:  TTS audio → speakers
                                                 ├── stdout: mic (echo-cancelled) → STT
                                                 └── SIGUSR1: flush playback (barge-in)
```

### Components

- **pocket-tts-mlx** -- Text-to-speech. Generates 24kHz Int16 PCM streamed to audio-io stdin. Predefined voices: alba, marius, javert, jean, fantine, cosette, eponine, azelma.
- **Moshi STT (MLX)** -- Speech-to-text using `kyutai/stt-1b-en_fr-mlx`. Runs the Moshi LM with `dep_q=0` (text-only output from audio, no depformer). Encodes mic audio with the MLX Mimi encoder.
- **audio-io** -- Embedded Swift binary (auto-built on first run, cached at `~/.cache/voice_server/audio-io-v1/`). Uses macOS VoiceProcessingIO AudioUnit for hardware echo cancellation. Converts between 24kHz wire rate and 48kHz device rate.
- **Silence-frame VAD** -- The STT model has `extra_heads_num_heads=0` (no pause score heads), so VAD uses a simple counter: 19 consecutive frames (~1.5s) with no text tokens triggers end-of-speech, followed by 7 flush frames and a 0.4s delay before finalizing.
- **Single `_model_lock`** -- Both TTS and STT share one asyncio lock since MLX may not be thread-safe across models. TTS releases the lock between generation and playback so STT can interleave (~80ms latency for barge-in detection).

## CLI reference

```
talk/server.py [text] [options]

Positional:
  text                         Text to speak (reads stdin if omitted and not a tty)

Options:
  --mode {single,continuous}   CLI mode (auto-detected if omitted)
  --input {line,unbuffered}    Input mode: line (send on Enter) or unbuffered (send as available)
  -t, --timeout DURATION       Listening timeout (e.g. 10s, 1m, 500ms, inf)
  --post-timeout DURATION      Keep listening after return (default: 30s)
  --lookbehind DURATION        Rolling window for between-call speech (default: 30s)
  -v, --voice NAME             TTS voice (default: cosette)
  --voice-file FILE            WAV file for voice cloning (overrides --voice)
  --port PORT                  Daemon port (default: 8078)
  --foreground                 Run server in foreground (don't daemonize)
  --kill                       Stop the running daemon
  --preload [stt] [tts]        Preload models on startup (foreground mode)
  --host HOST                  Bind address (default: 127.0.0.1)
  --stt-repo REPO              HuggingFace STT model repo
  --stt-quantize {4,8}         Quantize STT model weights
  --log-level LEVEL            Logging level (default: INFO)
```

## HTTP endpoints

All endpoints are available on the daemon (default `http://127.0.0.1:8078`).

### `POST /talk`

Full-duplex speak + listen. Supports JSON (default) or streaming NDJSON.

```json
// Request
{
  "text": "Hello!",          // optional -- text to speak
  "voice": "cosette",        // optional
  "timeout": 10,             // seconds to wait for speech (null = forever)
  "post_timeout": 30.0,      // keep listening after return (between-call speech)
  "lookbehind": 30.0,        // rolling window for queued between-call speech
  "stream": false            // set true for NDJSON streaming
}

// JSON Response (stream: false)
{
  "user_speech": "I'm good!",
  "interrupted_at": null,     // char index where TTS was interrupted, or null
  "timed_out": false
}
```

#### Streaming mode (`"stream": true`)

Returns `application/x-ndjson` with one event per line:

```
{"type": "tts_start", "text": "Hello, how are you?"}
{"type": "tts_word", "word": "Hello,", "offset": 0}
{"type": "tts_word", "word": "how", "offset": 7}
{"type": "tts_done", "interrupted_at": null}
{"type": "stt_partial", "text": "I'm doing"}
{"type": "stt_final", "text": "I'm doing great"}
{"type": "done", "timed_out": false}
```

If queued speech exists: `{"type": "queued_speech", "text": "...", "interrupted_at": 0}` followed by `done`.

If the user spoke between calls (within `lookbehind` window), the queued speech is returned immediately and TTS is skipped.

### `POST /v1/audio/transcriptions`

OpenAI-compatible speech-to-text. Accepts multipart form with an audio file.

```bash
curl -X POST http://localhost:8078/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "response_format=verbose_json"
```

Response formats: `json` (default), `text`, `verbose_json` (includes duration, processing time, RTF).

### `POST /v1/audio/speech`

OpenAI-compatible text-to-speech. Returns WAV audio (24kHz, mono, Int16).

```bash
curl -X POST http://localhost:8078/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello!", "voice": "cosette"}' \
  -o speech.wav
```

### `GET /health`

```json
{"status": "ok", "models": {"tts": true, "stt": true}}
```

## Daemon management

The CLI auto-manages a background daemon:

- **Auto-start**: First CLI call spawns the daemon, waits for `/health` to respond (up to 60s for model loading), then sends the request.
- **Reuse**: Subsequent calls read the lockfile (`~/.cache/voice_server/daemon.json`), verify the PID is alive, and reuse the daemon.
- **Auto-restart**: If `server.py` is modified (mtime changes), the old daemon is killed and a new one starts automatically.
- **Manual stop**: `uv run talk/server.py --kill` sends SIGTERM, waits, removes the lockfile.
- **Logs**: Daemon stdout/stderr go to `~/.cache/voice_server/daemon.log`.

## Requirements

- macOS (for audio-io / VoiceProcessingIO)
- Python 3.10+
- Apple Silicon recommended (MLX)
- Xcode Command Line Tools (`xcode-select --install`) for building audio-io on first run
- [uv](https://docs.astral.sh/uv/) (recommended, for `uv run` with automatic dependency resolution)
