#!/usr/bin/env -S uv run --index https://pypi.org/simple --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pocket-tts-mlx>=0.2.0",
#     "moshi-mlx>=0.3.0",
#     "starlette>=0.46.0",
#     "uvicorn>=0.34.0",
#     "sphn>=0.2.0",
#     "sentencepiece>=0.2.1",
#     "huggingface-hub>=0.24",
#     "numpy>=2",
#     "httpx>=0.27",
#     "python-multipart>=0.0.9",
#     "websockets>=14",
#     "mlx",
# ]
#
# [tool.uv]
# override-dependencies = ["sentencepiece>=0.2.1"]
# ///
"""Voice server: CLI-first tool with managed daemon for TTS + STT.

CLI usage:
    uv run voice_server/server.py "Hello!"              # Speak, listen, print JSON
    echo "Hello" | uv run voice_server/server.py        # Read from stdin
    uv run voice_server/server.py -t 10s "Hi"           # With timeout
    uv run voice_server/server.py --kill                 # Stop daemon
    uv run voice_server/server.py --foreground           # Run in foreground

Daemon mode (default for CLI):
    The CLI spawns a background daemon that keeps models loaded. Subsequent
    invocations reuse the running daemon for instant responses.

HTTP endpoints (daemon):
    GET  /           — Info
    GET  /health     — Health check
    POST /talk       — Speak + listen (JSON)
    POST /v1/audio/transcriptions — OpenAI-compatible STT
    POST /v1/audio/speech         — OpenAI-compatible TTS
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import base64
import datetime
import io
import json
import logging
import os
import platform
import re
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import sentencepiece
from huggingface_hub import hf_hub_download
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from moshi_mlx import models, utils
from moshi_mlx.models.lm import LmConfig, DepFormerConfig
from moshi_mlx.models.mimi import Mimi, mimi_202407
from moshi_mlx.modules.transformer import TransformerConfig

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

SAMPLE_RATE = 24000
FRAME_RATE = 12.5
FRAME_SIZE = 1920  # samples per frame at 24kHz (80ms)
FRAME_BYTES = FRAME_SIZE * 2  # Int16
SILENCE_TOKENS = np.array([948, 243, 1178, 546, 1736, 1030, 1978, 2008], dtype=np.int64)
STT_SPECIAL_TOKENS = {0, 3}  # BOS, UNK
DEFAULT_STT_REPO = "kyutai/stt-1b-en_fr-mlx"
DEFAULT_VOICE = "cosette"
DEFAULT_PORT = 8078

# VAD constants (silence-frame based since STT model has extra_heads_num_heads=0)
PAUSE_FRAMES = 19       # ~1.5s at 12.5fps consecutive silence → end of speech
FLUSH_FRAMES = 7        # zero-audio frames to flush STT pipeline after end-of-speech
SAVE_FLUSH_FRAMES = 19  # deeper flush when saving in-progress speech (~1.5s, matches audio_delay + margin)
STT_DELAY_SEC = 0.4     # delay after flush before finalizing
SKIP_INITIAL_FRAMES = 12  # skip initial frames after reset
BARGE_IN_GRACE_MS = 1000  # ms to suppress barge-in after TTS starts

# ── STT Engine (inlined from moshi_stt_mlx) ─────────────────────────────────


def _lm_config_from_dict(data: dict) -> LmConfig:
    """Build LmConfig from config.json, handling STT-specific fields."""
    dim = data["dim"]
    hidden_scale = data.get("hidden_scale", 4.0)
    norm = data.get("norm", "rms_norm")
    if norm == "rms_norm_f32":
        norm = "rms_norm"

    transformer = TransformerConfig(
        d_model=dim,
        num_heads=data["num_heads"],
        num_layers=data["num_layers"],
        dim_feedforward=int(hidden_scale * dim),
        causal=data["causal"],
        norm_first=True,
        bias_ff=False,
        bias_attn=False,
        layer_scale=data.get("layer_scale"),
        context=data["context"],
        max_period=int(data["max_period"]),
        use_conv_block=False,
        use_conv_bias=True,
        cross_attention=data.get("cross_attention", False),
        gating=True,
        norm=norm,
        positional_embedding=data["positional_embedding"],
        conv_layout=False,
        conv_kernel_size=3,
        kv_repeat=1,
        max_seq_len=4096,
    )

    dep_q = data.get("dep_q", 0)
    depformer_dim = data.get("depformer_dim", 1024)
    depformer_dim_ff = data.get("depformer_dim_feedforward")
    if depformer_dim_ff is None:
        depformer_dim_ff = depformer_dim * 4

    depformer = DepFormerConfig(
        transformer=TransformerConfig(
            d_model=depformer_dim,
            num_heads=data.get("depformer_num_heads", 16),
            num_layers=data.get("depformer_num_layers", 6),
            dim_feedforward=depformer_dim_ff,
            causal=data.get("depformer_causal", True),
            norm_first=True,
            bias_ff=False,
            bias_attn=False,
            layer_scale=None,
            context=data.get("depformer_context", max(dep_q, 1)),
            max_period=data.get("depformer_max_period", 10000),
            use_conv_block=False,
            use_conv_bias=True,
            cross_attention=False,
            gating=True,
            norm="rms_norm",
            positional_embedding=data.get("depformer_pos_emb", "none"),
            conv_layout=False,
            conv_kernel_size=3,
            kv_repeat=1,
            max_seq_len=4096,
        ),
        num_slices=dep_q,
        weights_per_step_schedule=data.get("depformer_weights_per_step_schedule"),
        low_rank_embeddings=data.get("depformer_low_rank_embeddings"),
    )

    conditioners: dict = {}
    if "conditioners" in data:
        from moshi_mlx.modules.conditioner import (
            LutConditionerConfig,
            TensorConditionerConfig,
        )
        for _name, _cfg in data["conditioners"].items():
            if _cfg["type"] == "lut":
                c = _cfg["lut"]
                conditioners[_name] = LutConditionerConfig(
                    n_bins=c["n_bins"], dim=c["dim"],
                    tokenizer=c["tokenizer"], possible_values=c["possible_values"],
                )
            elif _cfg["type"] == "tensor":
                c = _cfg["tensor"]
                conditioners[_name] = TensorConditionerConfig(dim=c["dim"])

    return LmConfig(
        transformer=transformer,
        depformer=depformer,
        text_in_vocab_size=data["text_card"] + 1,
        text_out_vocab_size=data["text_card"],
        audio_vocab_size=data["card"] + 1,
        audio_delays=data["delays"][1:],
        audio_codebooks=data["n_q"],
        demux_second_stream=data.get("demux_second_stream", False),
        conditioners=conditioners,
        extra_heads_dim=data.get("extra_heads_dim", 6),
        extra_heads_num_heads=data.get("extra_heads_num_heads", 0),
    )


class STTEngine:
    """Moshi-based Speech-to-Text engine (MLX)."""

    def __init__(self, hf_repo: str = DEFAULT_STT_REPO, quantized: int | None = None):
        self.hf_repo = hf_repo
        self.quantized = quantized
        self._load_model()

    def _load_model(self):
        mx.random.seed(299792458)

        config_path = hf_hub_download(self.hf_repo, "config.json")
        with open(config_path) as f:
            self.config_dict = json.load(f)

        moshi_name = self.config_dict.get("moshi_name", "model.safetensors")
        mimi_name = self.config_dict.get("mimi_name", "tokenizer-e351c8d8-checkpoint125.safetensors")
        tokenizer_name = self.config_dict.get("tokenizer_name", "tokenizer_spm_32k_3.model")
        self.stt_config = self.config_dict.get("stt_config", {})
        lm_gen_config = self.config_dict.get("lm_gen_config", {})

        moshi_weight = hf_hub_download(self.hf_repo, moshi_name)
        mimi_weight_path = hf_hub_download(self.hf_repo, mimi_name)
        tokenizer_path = hf_hub_download(self.hf_repo, tokenizer_name)

        self.text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)

        lm_config = _lm_config_from_dict(self.config_dict)
        self.lm_config = lm_config

        self.text_temp = lm_gen_config.get("temp_text", 0.0)
        self.text_top_k = lm_gen_config.get("top_k_text", 50)

        model = models.Lm(lm_config)
        model.set_dtype(mx.bfloat16)

        if self.quantized is not None:
            group_size = 32 if self.quantized == 4 else 64
            nn.quantize(model, bits=self.quantized, group_size=group_size)

        model.load_weights(moshi_weight, strict=True)

        self._ct = None
        if model.condition_provider is not None:
            self._ct = model.condition_provider.condition_tensor("description", "very_good")

        model.warmup(self._ct)
        self.model = model

        self.other_codebooks = lm_config.other_codebooks
        self.main_codebooks = lm_config.generated_codebooks

        # MLX Mimi encoder
        mimi_codebooks = max(self.main_codebooks, self.other_codebooks)
        mimi_cfg = mimi_202407(mimi_codebooks)
        self.mimi = Mimi(mimi_cfg)
        self.mimi.load_pytorch_weights(mimi_weight_path, strict=True)
        # Warmup
        dummy = mx.zeros((1, 1, FRAME_SIZE * 4))
        mx.eval(self.mimi.encode(dummy))
        logger.info("STT engine ready: %s (codebooks=%d)", self.hf_repo, self.other_codebooks)

    def new_session(self) -> "STTSession":
        """Create a new streaming STT session."""
        for c in self.model.transformer_cache:
            c.reset()
        self.mimi.reset_all()

        gen = models.LmGen(
            model=self.model,
            max_steps=65536,  # ~87 minutes at 12.5fps
            text_sampler=utils.Sampler(temp=self.text_temp, top_k=self.text_top_k),
            audio_sampler=utils.Sampler(temp=0.8, top_k=250),
            cfg_coef=1.0,
            check=False,
        )
        return STTSession(self, gen)

    def transcribe_batch(self, audio: np.ndarray) -> dict:
        """Batch transcribe audio (float32, 24kHz mono)."""
        orig_duration = len(audio) / SAMPLE_RATE

        # Pad per STT config
        pad_left = int(self.stt_config.get("audio_silence_prefix_seconds", 0.0) * SAMPLE_RATE)
        pad_right = int((self.stt_config.get("audio_delay_seconds", 0.0) + 1.0) * SAMPLE_RATE)
        audio = np.pad(audio, (pad_left, pad_right), mode="constant")

        remainder = len(audio) % FRAME_SIZE
        if remainder > 0:
            audio = np.concatenate([audio, np.zeros(FRAME_SIZE - remainder, dtype=np.float32)])

        num_frames = len(audio) // FRAME_SIZE
        t0 = time.time()

        pcm_tensor = mx.array(audio[np.newaxis, np.newaxis, :])
        self.mimi.reset_all()
        all_codes = self.mimi.encode(pcm_tensor)
        mx.eval(all_codes)
        all_codes = all_codes[:, :self.other_codebooks, :]

        for c in self.model.transformer_cache:
            c.reset()
        gen = models.LmGen(
            model=self.model, max_steps=num_frames + 10,
            text_sampler=utils.Sampler(temp=self.text_temp, top_k=self.text_top_k),
            audio_sampler=utils.Sampler(temp=0.8, top_k=250),
            cfg_coef=1.0, check=False,
        )

        text_tokens = []
        for idx in range(num_frames):
            text_token, _ = gen.step(
                all_codes[:, :, idx][0], ct=self._ct,
            )
            val = text_token[0].item()
            if val not in STT_SPECIAL_TOKENS:
                text_tokens.append(int(val))

        elapsed = time.time() - t0
        text = self.text_tokenizer.decode(text_tokens) if text_tokens else ""
        return {
            "text": text.strip(),
            "duration_s": round(orig_duration, 2),
            "processing_time_s": round(elapsed, 3),
            "rtf": round(elapsed / orig_duration, 4) if orig_duration > 0 else 0,
        }


class STTSession:
    """Streaming STT session: feed PCM frames, get text pieces."""

    def __init__(self, engine: STTEngine, gen: models.LmGen):
        self.engine = engine
        self.gen = gen
        self.tokens: list[int] = []
        self._buffer = np.zeros(0, dtype=np.float32)
        # Initialize Mimi encoder caches for streaming
        for c in engine.mimi.encoder_cache:
            c.reset()

    def feed_pcm(self, pcm_int16: bytes) -> list[str]:
        """Feed raw PCM int16 bytes. Returns list of new text pieces."""
        samples = np.frombuffer(pcm_int16, dtype=np.int16).astype(np.float32) / 32768.0
        self._buffer = np.concatenate([self._buffer, samples])

        pieces = []
        while len(self._buffer) >= FRAME_SIZE:
            frame = self._buffer[:FRAME_SIZE]
            self._buffer = self._buffer[FRAME_SIZE:]

            # Encode with MLX Mimi streaming encoder
            pcm_mx = mx.array(frame[np.newaxis, np.newaxis, :])
            codes = self.engine.mimi.encode_step(pcm_mx)
            # codes shape: (1, codebooks, 1) → squeeze time dim, index batch
            codes = codes[:, :self.engine.other_codebooks, 0]  # (1, codebooks)

            text_token, _ = self.gen.step(
                codes[0], ct=self.engine._ct,
            )
            val = text_token[0].item()
            self.tokens.append(val)

            if val not in STT_SPECIAL_TOKENS:
                piece = self.engine.text_tokenizer.id_to_piece(val).replace("\u2581", " ")
                pieces.append(piece)

        return pieces

    def finish(self) -> str:
        """Flush remaining buffer and return final text."""
        # Pad remaining float32 buffer to frame boundary and convert to int16
        if len(self._buffer) > 0:
            pad_len = FRAME_SIZE - len(self._buffer)
            padded = np.concatenate([self._buffer, np.zeros(pad_len, dtype=np.float32)])
            padded_int16 = (padded * 32768).astype(np.int16)
            self._buffer = np.zeros(0, dtype=np.float32)
            self.feed_pcm(padded_int16.tobytes())

        # Feed padding frames for STT delay
        pad_frames = int((self.engine.stt_config.get("audio_delay_seconds", 0.5) + 1.0) * FRAME_RATE)
        silence_bytes = np.zeros(FRAME_SIZE, dtype=np.int16).tobytes()
        for _ in range(pad_frames):
            self.feed_pcm(silence_bytes)

        valid = [t for t in self.tokens if t not in STT_SPECIAL_TOKENS]
        return self.engine.text_tokenizer.decode(valid).strip() if valid else ""


# ── TTS Engine ───────────────────────────────────────────────────────────────

_tts_model = None
_tts_voice_states: dict[str, dict] = {}
_stt_engine = None

# MLX models are NOT thread-safe; serialize all model access.
_model_lock = asyncio.Lock()


def get_tts_model():
    global _tts_model
    if _tts_model is None:
        from pocket_tts_mlx.models.tts_model import TTSModel
        logger.info("Loading TTS model...")
        _tts_model = TTSModel.load_model()
        # Warmup with default voice
        state = _get_tts_voice_state(DEFAULT_VOICE)
        for _ in _tts_model.generate_audio_stream(state, "Hello."):
            pass
        logger.info("TTS model ready")
    return _tts_model


def _get_tts_voice_state(voice: str) -> dict:
    """Get (cached) voice state for TTS generation.

    ``voice`` can be a predefined name (e.g. "cosette") or a file path to a
    WAV audio prompt for voice cloning.
    """
    if voice not in _tts_voice_states:
        model = get_tts_model()
        # If it looks like a file path, pass as Path for voice cloning
        voice_key: str | Path = voice
        if "/" in voice or voice.endswith(".wav"):
            voice_key = Path(voice)
        _tts_voice_states[voice] = model.get_state_for_audio_prompt(voice_key)
    return _tts_voice_states[voice]


def get_stt_engine():
    global _stt_engine
    if _stt_engine is None:
        logger.info("Loading STT engine...")
        _stt_engine = STTEngine()
        logger.info("STT engine ready")
    return _stt_engine


# ── Embedded Swift audio-io binary (macOS VoiceProcessingIO + echo cancellation) ──

AUDIO_IO_PACKAGE_SWIFT = """\
// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "audio-io",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(
            name: "audio-io",
            path: "Sources",
            linkerSettings: [
                .linkedFramework("AudioToolbox"),
                .linkedFramework("CoreAudio"),
            ]
        ),
    ]
)
"""

AUDIO_IO_MAIN_SWIFT = """\
import Foundation

/// audio-io: Bidirectional audio I/O with echo cancellation via VoiceProcessingIO
///
/// stdin:  Int16 PCM 24kHz mono → speakers (with AEC reference)
/// stdout: echo-cancelled mic → Int16 PCM 24kHz mono
/// stderr: logs + "DRAINED\\n" when playback buffer empties
/// SIGUSR1: flush playback buffer (barge-in)

func log(_ message: String) {
    FileHandle.standardError.write("[audio-io] \\(message)\\n".data(using: .utf8)!)
}

// Parse sample rate from args (default 48000)
let deviceRate: Float64 = {
    let args = CommandLine.arguments
    if let idx = args.firstIndex(of: "--device-rate"), idx + 1 < args.count {
        return Float64(args[idx + 1]) ?? 48000.0
    }
    return 48000.0
}()

log("Starting with device rate: \\(deviceRate)Hz, wire rate: 24000Hz")

// Create engine
let engine: VoiceIOEngine
do {
    engine = try VoiceIOEngine(deviceRate: deviceRate)
} catch {
    log("Failed to create audio engine: \\(error)")
    exit(1)
}

// Handle SIGUSR1 for barge-in (flush playback)
signal(SIGUSR1) { _ in
    engine.flushPlayback()
}

// Start the audio unit
do {
    try engine.start()
} catch {
    log("Failed to start audio engine: \\(error)")
    exit(1)
}

// Stdin reader thread: read Int16 PCM 24kHz, convert to Float32, feed to engine
let stdinThread = Thread {
    let stdinHandle = FileHandle.standardInput
    let frameSize = 1920 * 2 // 80ms at 24kHz, Int16 = 2 bytes per sample
    var accumulator = Data()

    while !engine.shouldStop {
        let data = stdinHandle.availableData
        if data.isEmpty {
            // EOF on stdin
            log("stdin EOF")
            engine.shouldStop = true
            break
        }

        accumulator.append(data)

        // Process complete frames
        while accumulator.count >= frameSize {
            let frameData = accumulator.prefix(frameSize)
            accumulator = Data(accumulator.dropFirst(frameSize))

            // Convert Int16 to Float32
            let sampleCount = frameData.count / 2
            var float32 = [Float32](repeating: 0, count: sampleCount)

            frameData.withUnsafeBytes { raw in
                let int16 = raw.bindMemory(to: Int16.self)
                for i in 0..<sampleCount {
                    float32[i] = Float32(int16[i]) / 32768.0
                }
            }

            float32.withUnsafeBufferPointer { engine.feedPlayback($0) }
        }
    }
}
stdinThread.name = "stdin-reader"
stdinThread.start()

// Stdout writer thread: read captured audio, convert to Int16, write to stdout
let stdoutThread = Thread {
    let stdoutHandle = FileHandle.standardOutput
    let frameSize = 1920 // 80ms at 24kHz
    var float32Buffer = [Float32](repeating: 0, count: frameSize)

    while !engine.shouldStop {
        float32Buffer.withUnsafeMutableBufferPointer { dest in
            let read = engine.readCapture(into: dest)
            if read > 0 {
                // Convert Float32 to Int16
                var int16 = [Int16](repeating: 0, count: read)
                for i in 0..<read {
                    let clamped = max(-1.0, min(1.0, dest[i]))
                    int16[i] = Int16(clamped * 32767.0)
                }

                int16.withUnsafeBytes { raw in
                    stdoutHandle.write(Data(raw))
                }
            }
        }

        // Check for drain
        engine.checkDrain()

        // Sleep briefly to avoid busy-waiting (~80ms frame period)
        Thread.sleep(forTimeInterval: 0.02)
    }
}
stdoutThread.name = "stdout-writer"
stdoutThread.start()

// Handle SIGINT/SIGTERM
signal(SIGINT) { _ in
    engine.shouldStop = true
}
signal(SIGTERM) { _ in
    engine.shouldStop = true
}

// Run loop to keep process alive
log("Ready")
while !engine.shouldStop {
    RunLoop.current.run(mode: .default, before: Date(timeIntervalSinceNow: 0.1))
}

engine.stop()
log("Exiting")
exit(0)
"""

AUDIO_IO_VOICEIO_SWIFT = """\
import AudioToolbox
import CoreAudio
import Foundation

/// Core audio engine using VoiceProcessingIO AudioUnit for echo cancellation.
///
/// - stdin: Int16 PCM 24kHz mono → speakers (with AEC reference)
/// - stdout: echo-cancelled mic capture → Int16 PCM 24kHz mono
/// - stderr: "DRAINED\\n" when playback buffer empties after receiving data
final class VoiceIOEngine {
    fileprivate var audioUnit: AudioComponentInstance?
    private let playbackBuffer = RingBuffer(capacity: 48000 * 60) // ~60s at 48kHz device rate
    fileprivate let captureBuffer = RingBuffer(capacity: 48000 * 2) // ~2s at 48kHz

    private var upsampleConverter: SampleRateConverter?
    fileprivate var downsampleConverter: SampleRateConverter?

    private let deviceRate: Float64
    private let wireRate: Float64 = 24000.0

    /// Set after first audio is written to playback buffer
    private var hasReceivedPlaybackData = false
    /// Set when drain signal has been sent (reset on new data)
    private var drainSignalSent = false
    /// Set to true to stop the engine
    var shouldStop = false

    init(deviceRate: Float64 = 48000.0) throws {
        self.deviceRate = deviceRate

        // Create sample rate converters
        if deviceRate != wireRate {
            upsampleConverter = try SampleRateConverter(from: wireRate, to: deviceRate)
            downsampleConverter = try SampleRateConverter(from: deviceRate, to: wireRate)
        }

        try setupAudioUnit()
    }

    private func setupAudioUnit() throws {
        var desc = AudioComponentDescription(
            componentType: kAudioUnitType_Output,
            componentSubType: kAudioUnitSubType_VoiceProcessingIO,
            componentManufacturer: kAudioUnitManufacturer_Apple,
            componentFlags: 0,
            componentFlagsMask: 0
        )

        guard let component = AudioComponentFindNext(nil, &desc) else {
            throw AudioIOError.audioUnitSetupFailed(-1)
        }

        var unit: AudioComponentInstance?
        var status = AudioComponentInstanceNew(component, &unit)
        guard status == noErr, let audioUnit = unit else {
            throw AudioIOError.audioUnitSetupFailed(status)
        }
        self.audioUnit = audioUnit

        // Enable input (mic capture) on bus 1
        var enableInput: UInt32 = 1
        status = AudioUnitSetProperty(
            audioUnit,
            kAudioOutputUnitProperty_EnableIO,
            kAudioUnitScope_Input,
            1, // input bus
            &enableInput,
            UInt32(MemoryLayout<UInt32>.size)
        )
        guard status == noErr else {
            throw AudioIOError.audioUnitSetupFailed(status)
        }

        // Set stream format for both buses (device rate, Float32, mono)
        var streamFormat = AudioStreamBasicDescription(
            mSampleRate: deviceRate,
            mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
            mBytesPerPacket: 4,
            mFramesPerPacket: 1,
            mBytesPerFrame: 4,
            mChannelsPerFrame: 1,
            mBitsPerChannel: 32,
            mReserved: 0
        )

        // Output bus 0 (speakers) format
        status = AudioUnitSetProperty(
            audioUnit,
            kAudioUnitProperty_StreamFormat,
            kAudioUnitScope_Input,
            0, // output bus
            &streamFormat,
            UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        )
        guard status == noErr else {
            throw AudioIOError.audioUnitSetupFailed(status)
        }

        // Input bus 1 (mic) format
        status = AudioUnitSetProperty(
            audioUnit,
            kAudioUnitProperty_StreamFormat,
            kAudioUnitScope_Output,
            1, // input bus
            &streamFormat,
            UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        )
        guard status == noErr else {
            throw AudioIOError.audioUnitSetupFailed(status)
        }

        // Set render callbacks
        var renderCallback = AURenderCallbackStruct(
            inputProc: playbackCallback,
            inputProcRefCon: Unmanaged.passUnretained(self).toOpaque()
        )
        status = AudioUnitSetProperty(
            audioUnit,
            kAudioUnitProperty_SetRenderCallback,
            kAudioUnitScope_Input,
            0,
            &renderCallback,
            UInt32(MemoryLayout<AURenderCallbackStruct>.size)
        )
        guard status == noErr else {
            throw AudioIOError.audioUnitSetupFailed(status)
        }

        // Set input callback for mic capture
        var inputCallback = AURenderCallbackStruct(
            inputProc: captureCallback,
            inputProcRefCon: Unmanaged.passUnretained(self).toOpaque()
        )
        status = AudioUnitSetProperty(
            audioUnit,
            kAudioOutputUnitProperty_SetInputCallback,
            kAudioUnitScope_Global,
            1,
            &inputCallback,
            UInt32(MemoryLayout<AURenderCallbackStruct>.size)
        )
        guard status == noErr else {
            throw AudioIOError.audioUnitSetupFailed(status)
        }

        status = AudioUnitInitialize(audioUnit)
        guard status == noErr else {
            throw AudioIOError.audioUnitSetupFailed(status)
        }
    }

    func start() throws {
        guard let audioUnit = audioUnit else { return }
        let status = AudioOutputUnitStart(audioUnit)
        guard status == noErr else {
            throw AudioIOError.audioUnitSetupFailed(status)
        }
        log("Audio engine started (device rate: \\(deviceRate)Hz)")
    }

    func stop() {
        guard let audioUnit = audioUnit else { return }
        AudioOutputUnitStop(audioUnit)
        AudioComponentInstanceDispose(audioUnit)
        self.audioUnit = nil
        log("Audio engine stopped")
    }

    /// Feed playback audio (at wire rate 24kHz). Called from stdin reader thread.
    /// Blocks if buffer is nearly full (backpressure to stdin pipe).
    func feedPlayback(_ samples: UnsafeBufferPointer<Float32>) {
        // Calculate required space (account for upsampling)
        let requiredSpace = upsampleConverter != nil
            ? Int(Double(samples.count) * (deviceRate / wireRate)) + 10
            : samples.count

        // Block while buffer is too full (keep ~1s headroom for jitter)
        let headroom = Int(deviceRate) // 1 second at device rate
        while playbackBuffer.free < requiredSpace + headroom && !shouldStop {
            Thread.sleep(forTimeInterval: 0.01) // 10ms poll
        }

        if shouldStop { return }

        if let converter = upsampleConverter {
            let upsampled = converter.convert(samples)
            upsampled.withUnsafeBufferPointer { _ = playbackBuffer.write($0) }
        } else {
            _ = playbackBuffer.write(samples)
        }
        hasReceivedPlaybackData = true
        drainSignalSent = false
    }

    /// Read captured audio (at wire rate 24kHz). Called from stdout writer thread.
    func readCapture(into dest: UnsafeMutableBufferPointer<Float32>) -> Int {
        return captureBuffer.read(into: dest)
    }

    /// Flush playback buffer immediately (barge-in via SIGUSR1).
    func flushPlayback() {
        playbackBuffer.flush()
        log("Playback flushed (barge-in)")
    }

    /// Check if playback has drained and signal if needed.
    func checkDrain() {
        if hasReceivedPlaybackData && !drainSignalSent && playbackBuffer.available == 0 {
            drainSignalSent = true
            hasReceivedPlaybackData = false
            FileHandle.standardError.write("DRAINED\\n".data(using: .utf8)!)
        }
    }

    // MARK: - Audio Callbacks

    /// Playback callback: reads from playbackBuffer, outputs silence if empty.
    fileprivate func renderPlayback(
        _ ioData: UnsafeMutablePointer<AudioBufferList>,
        frameCount: UInt32
    ) {
        let buffer = UnsafeMutableAudioBufferListPointer(ioData)
        guard let data = buffer[0].mData?.assumingMemoryBound(to: Float32.self) else { return }
        let dest = UnsafeMutableBufferPointer(start: data, count: Int(frameCount))

        let read = playbackBuffer.read(into: dest)

        // Fill remainder with silence
        if read < Int(frameCount) {
            for i in read..<Int(frameCount) {
                dest[i] = 0
            }
        }
    }

    /// Capture callback: renders mic input, writes to captureBuffer.
    fileprivate func renderCapture(
        _ audioUnit: AudioUnit,
        _ ioData: UnsafeMutablePointer<AudioBufferList>,
        frameCount: UInt32,
        timestamp: UnsafePointer<AudioTimeStamp>
    ) {
        // Render mic input
        var bufferList = AudioBufferList(
            mNumberBuffers: 1,
            mBuffers: AudioBuffer(
                mNumberChannels: 1,
                mDataByteSize: frameCount * 4,
                mData: nil
            )
        )

        // Allocate temp buffer for capture
        let tempBuffer = UnsafeMutablePointer<Float32>.allocate(capacity: Int(frameCount))
        bufferList.mBuffers.mData = UnsafeMutableRawPointer(tempBuffer)

        let status = AudioUnitRender(audioUnit, nil, timestamp, 1, frameCount, &bufferList)
        if status == noErr {
            let captured = UnsafeBufferPointer(start: tempBuffer, count: Int(frameCount))

            if let converter = downsampleConverter {
                let downsampled = converter.convert(captured)
                downsampled.withUnsafeBufferPointer { _ = captureBuffer.write($0) }
            } else {
                _ = captureBuffer.write(captured)
            }
        }

        tempBuffer.deallocate()
    }
}

// MARK: - C-compatible callbacks

private let playbackCallback: AURenderCallback = {
    (inRefCon, _, _, _, inNumberFrames, ioData) -> OSStatus in

    let engine = Unmanaged<VoiceIOEngine>.fromOpaque(inRefCon).takeUnretainedValue()
    if let ioData = ioData {
        engine.renderPlayback(ioData, frameCount: inNumberFrames)
    }
    return noErr
}

private let captureCallback: AURenderCallback = {
    (inRefCon, _, inTimeStamp, inBusNumber, inNumberFrames, ioData) -> OSStatus in

    let engine = Unmanaged<VoiceIOEngine>.fromOpaque(inRefCon).takeUnretainedValue()
    guard let au = engine.audioUnit else { return noErr }

    var bufferList = AudioBufferList(
        mNumberBuffers: 1,
        mBuffers: AudioBuffer(
            mNumberChannels: 1,
            mDataByteSize: inNumberFrames * 4,
            mData: nil
        )
    )

    let tempBuffer = UnsafeMutablePointer<Float32>.allocate(capacity: Int(inNumberFrames))
    bufferList.mBuffers.mData = UnsafeMutableRawPointer(tempBuffer)

    let status = AudioUnitRender(au, nil, inTimeStamp, 1, inNumberFrames, &bufferList)
    if status == noErr {
        let captured = UnsafeBufferPointer(start: tempBuffer, count: Int(inNumberFrames))

        if let converter = engine.downsampleConverter {
            let downsampled = converter.convert(captured)
            downsampled.withUnsafeBufferPointer { _ = engine.captureBuffer.write($0) }
        } else {
            _ = engine.captureBuffer.write(captured)
        }
    }

    tempBuffer.deallocate()
    return noErr
}
"""

AUDIO_IO_RINGBUFFER_SWIFT = """\
import Foundation

/// Lock-free single-producer single-consumer ring buffer for Float32 audio samples.
final class RingBuffer {
    private let buffer: UnsafeMutableBufferPointer<Float32>
    private let capacity: Int
    private var writePos: Int = 0  // only modified by producer
    private var readPos: Int = 0   // only modified by consumer
    /// Total samples ever written (for drain detection)
    private(set) var totalWritten: Int = 0

    init(capacity: Int) {
        self.capacity = capacity
        let ptr = UnsafeMutablePointer<Float32>.allocate(capacity: capacity)
        ptr.initialize(repeating: 0, count: capacity)
        self.buffer = UnsafeMutableBufferPointer(start: ptr, count: capacity)
    }

    deinit {
        buffer.baseAddress?.deallocate()
    }

    /// Number of samples available to read.
    var available: Int {
        let w = writePos
        let r = readPos
        return w >= r ? w - r : capacity - r + w
    }

    /// Number of free slots for writing.
    var free: Int {
        return capacity - 1 - available
    }

    /// Write samples into the ring buffer. Returns number of samples actually written.
    @discardableResult
    func write(_ samples: UnsafeBufferPointer<Float32>) -> Int {
        let count = min(samples.count, free)
        guard count > 0 else { return 0 }

        let wp = writePos
        let firstChunk = min(count, capacity - wp)
        let secondChunk = count - firstChunk

        buffer.baseAddress!.advanced(by: wp).update(from: samples.baseAddress!, count: firstChunk)
        if secondChunk > 0 {
            buffer.baseAddress!.update(from: samples.baseAddress!.advanced(by: firstChunk), count: secondChunk)
        }

        totalWritten += count
        writePos = (wp + count) % capacity
        return count
    }

    /// Read samples from the ring buffer into the destination. Returns number of samples read.
    @discardableResult
    func read(into dest: UnsafeMutableBufferPointer<Float32>) -> Int {
        let count = min(dest.count, available)
        guard count > 0 else { return 0 }

        let rp = readPos
        let firstChunk = min(count, capacity - rp)
        let secondChunk = count - firstChunk

        dest.baseAddress!.update(from: buffer.baseAddress!.advanced(by: rp), count: firstChunk)
        if secondChunk > 0 {
            dest.baseAddress!.advanced(by: firstChunk).update(from: buffer.baseAddress!, count: secondChunk)
        }

        readPos = (rp + count) % capacity
        return count
    }

    /// Discard all data in the ring buffer.
    func flush() {
        readPos = writePos
    }
}
"""

AUDIO_IO_SRC_CONVERTER_SWIFT = """\
import AudioToolbox
import Foundation

/// Converts between sample rates using AudioConverter.
final class SampleRateConverter {
    private var converter: AudioConverterRef?
    let sourceRate: Float64
    let destRate: Float64
    private let ratio: Float64

    init(from sourceRate: Float64, to destRate: Float64) throws {
        self.sourceRate = sourceRate
        self.destRate = destRate
        self.ratio = destRate / sourceRate

        var srcFormat = AudioStreamBasicDescription(
            mSampleRate: sourceRate,
            mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
            mBytesPerPacket: 4,
            mFramesPerPacket: 1,
            mBytesPerFrame: 4,
            mChannelsPerFrame: 1,
            mBitsPerChannel: 32,
            mReserved: 0
        )

        var dstFormat = AudioStreamBasicDescription(
            mSampleRate: destRate,
            mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
            mBytesPerPacket: 4,
            mFramesPerPacket: 1,
            mBytesPerFrame: 4,
            mChannelsPerFrame: 1,
            mBitsPerChannel: 32,
            mReserved: 0
        )

        let status = AudioConverterNew(&srcFormat, &dstFormat, &converter)
        guard status == noErr else {
            throw AudioIOError.converterCreationFailed(status)
        }
    }

    deinit {
        if let converter = converter {
            AudioConverterDispose(converter)
        }
    }

    /// Convert samples. Returns converted Float32 array.
    func convert(_ input: UnsafeBufferPointer<Float32>) -> [Float32] {
        guard let converter = converter else { return [] }

        let outputFrames = Int(ceil(Float64(input.count) * ratio))
        var output = [Float32](repeating: 0, count: outputFrames)
        let inputSize = UInt32(input.count * 4)

        var packetCount = UInt32(outputFrames)
        var userData = ConverterUserData(buffer: input, bytesRemaining: inputSize)

        let actualFrames: Int = output.withUnsafeMutableBytes { outputRaw in
            var outputBufferList = AudioBufferList(
                mNumberBuffers: 1,
                mBuffers: AudioBuffer(
                    mNumberChannels: 1,
                    mDataByteSize: UInt32(outputFrames * 4),
                    mData: outputRaw.baseAddress
                )
            )

            let status = withUnsafeMutablePointer(to: &userData) { userDataPtr in
                AudioConverterFillComplexBuffer(
                    converter,
                    converterInputProc,
                    userDataPtr,
                    &packetCount,
                    &outputBufferList,
                    nil
                )
            }

            if status != noErr && status != kAudioConverterErr_InputDataExhausted {
                log("SRC conversion error: \\(status)")
            }

            return Int(outputBufferList.mBuffers.mDataByteSize / 4)
        }

        return Array(output.prefix(actualFrames))
    }
}

private struct ConverterUserData {
    var buffer: UnsafeBufferPointer<Float32>
    var bytesRemaining: UInt32
}

private let kAudioConverterErr_InputDataExhausted: OSStatus = -66567

private let converterInputProc: AudioConverterComplexInputDataProc = {
    (converter, ioNumPackets, ioData, outPacketDesc, inUserData) -> OSStatus in

    guard let userData = inUserData?.assumingMemoryBound(to: ConverterUserData.self).pointee else {
        ioNumPackets.pointee = 0
        return kAudioConverterErr_InputDataExhausted
    }

    if userData.bytesRemaining == 0 {
        ioNumPackets.pointee = 0
        return kAudioConverterErr_InputDataExhausted
    }

    let availablePackets = userData.bytesRemaining / 4
    let packetsToProvide = min(ioNumPackets.pointee, availablePackets)

    ioData.pointee.mNumberBuffers = 1
    ioData.pointee.mBuffers.mData = UnsafeMutableRawPointer(mutating: userData.buffer.baseAddress)
    ioData.pointee.mBuffers.mDataByteSize = packetsToProvide * 4
    ioData.pointee.mBuffers.mNumberChannels = 1

    ioNumPackets.pointee = packetsToProvide

    // Mark all as consumed
    inUserData?.assumingMemoryBound(to: ConverterUserData.self).pointee.bytesRemaining = 0

    return noErr
}

enum AudioIOError: Error {
    case converterCreationFailed(OSStatus)
    case audioUnitSetupFailed(OSStatus)
}
"""

AUDIO_IO_SOURCES = {
    "Package.swift": AUDIO_IO_PACKAGE_SWIFT,
    "Sources/main.swift": AUDIO_IO_MAIN_SWIFT,
    "Sources/VoiceIOEngine.swift": AUDIO_IO_VOICEIO_SWIFT,
    "Sources/RingBuffer.swift": AUDIO_IO_RINGBUFFER_SWIFT,
    "Sources/SampleRateConverter.swift": AUDIO_IO_SRC_CONVERTER_SWIFT,
}

AUDIO_IO_VERSION = "v1"


def _get_audio_io_cache_dir() -> Path:
    """Return cache directory for audio-io, creating if needed."""
    return Path.home() / ".cache" / "voice_server" / f"audio-io-{AUDIO_IO_VERSION}"


def _extract_audio_io_sources(target_dir: Path) -> None:
    """Extract embedded Swift sources to target directory."""
    for rel_path, content in AUDIO_IO_SOURCES.items():
        file_path = target_dir / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)


def ensure_audio_io() -> Path | None:
    """Return path to audio-io binary, building from embedded source on first run."""
    if platform.system() != "Darwin":
        return None

    cache_dir = _get_audio_io_cache_dir()
    binary_path = cache_dir / ".build" / "release" / "audio-io"

    if binary_path.exists():
        return binary_path

    if not shutil.which("swift"):
        logger.warning("swift not found — install Xcode Command Line Tools "
                       "(xcode-select --install) to enable audio-io")
        return None

    logger.info("Building audio-io from embedded source (first run)...")
    cache_dir.mkdir(parents=True, exist_ok=True)
    _extract_audio_io_sources(cache_dir)

    try:
        subprocess.run(
            ["swift", "build", "-c", "release", "--quiet"],
            cwd=cache_dir, check=True,
        )
    except subprocess.CalledProcessError:
        logger.error("Failed to build audio-io")
        return None

    if not binary_path.exists():
        binary_path = cache_dir / ".build" / "arm64-apple-macosx" / "release" / "audio-io"

    if not binary_path.exists():
        logger.error("audio-io binary not found after build")
        return None

    logger.info("audio-io built successfully: %s", binary_path)
    return binary_path


# ── TalkSession ──────────────────────────────────────────────────────────────

@dataclass
class TalkSession:
    """State for the full-duplex audio pipeline."""

    # TTS state
    tts_playing: bool = False
    tts_write_done: bool = False
    tts_char_offset: int = 0
    playback_drained: asyncio.Event = field(default_factory=asyncio.Event)

    # STT state
    stt_session: STTSession | None = None
    current_transcript: str = ""
    silence_frames: int = 0
    frames_since_reset: int = 0
    is_flushing: bool = False
    stt_needs_reset: bool = False
    flush_timer: asyncio.TimerHandle | None = None
    barge_in_suppressed_until: float = 0.0

    # Audio I/O
    audio_process: subprocess.Popen | None = None
    capture_task: asyncio.Task | None = None
    drain_task: asyncio.Task | None = None
    tts_task: asyncio.Task | None = None

    # Result coordination
    speech_ready_event: asyncio.Event | None = None
    final_transcript: str = ""
    interrupted_at: int | None = None

    # Between-call listening
    queued_speech: list[tuple[float, str]] = field(default_factory=list)
    post_listen_task: asyncio.Task | None = None
    post_timeout: float = 0
    lookbehind: float = 0

    # Lifecycle
    shutting_down: bool = False

    # Streaming events (set only during streaming /talk or WS requests)
    event_queue: asyncio.Queue | None = None

    # WebSocket external audio: callback to send TTS audio over WS
    ws_send_audio: Any = None  # async callable(bytes) | None
    audio_format: str = "pcm_f32le"  # "pcm_f32le" or "pcm_s16le"


# Singleton session for the daemon
_session: TalkSession | None = None


def get_session() -> TalkSession:
    global _session
    if _session is None:
        _session = TalkSession()
    return _session


def _emit(session: TalkSession, event: dict) -> None:
    """Emit an event to the session's event queue if streaming is active."""
    if session.event_queue is not None:
        session.event_queue.put_nowait(event)


# ── Audio Pipeline ───────────────────────────────────────────────────────────

async def start_audio_pipeline(session: TalkSession) -> None:
    """Spawn audio-io subprocess and start capture + drain loops."""
    audio_io_bin = ensure_audio_io()
    if audio_io_bin is None:
        logger.error("audio-io not available — no audio pipeline")
        return

    session.audio_process = subprocess.Popen(
        [str(audio_io_bin)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    session.capture_task = asyncio.create_task(
        audio_capture_loop(session, session.audio_process.stdout)
    )
    session.drain_task = asyncio.create_task(
        drain_monitor_loop(session, session.audio_process.stderr)
    )
    logger.info("Audio pipeline started (pid=%d)", session.audio_process.pid)


async def audio_capture_loop(session: TalkSession, stdout) -> None:
    """Read audio from audio-io stdout, feed to STT session, run VAD."""
    loop = asyncio.get_event_loop()
    frame_count = 0

    try:
        while not session.shutting_down:
            data = await loop.run_in_executor(None, stdout.read, FRAME_BYTES)
            if not data:
                logger.warning("Audio capture: stdout EOF — audio-io may have died")
                break

            frame_count += 1

            if session.stt_session is None:
                continue  # discard audio when not listening

            # Feed audio to STT under model lock
            async with _model_lock:
                stt = session.stt_session
                if stt is None:
                    continue  # session cleared while waiting for lock
                if session.stt_needs_reset:
                    logger.debug("Resetting STT session (post-finalization)")
                    engine = get_stt_engine()
                    session.stt_session = engine.new_session()
                    session.frames_since_reset = 0
                    session.stt_needs_reset = False
                    continue
                try:
                    pieces = stt.feed_pcm(data)
                except Exception as e:
                    if "max-steps" in str(e) or "max_steps" in str(e):
                        logger.warning("STT session hit max-steps, resetting")
                        engine = get_stt_engine()
                        session.stt_session = engine.new_session()
                        session.frames_since_reset = 0
                        continue
                    raise

            session.frames_since_reset += 1

            if session.frames_since_reset <= SKIP_INITIAL_FRAMES:
                continue

            # Periodic alive log (every ~10s = 125 frames at 12.5fps)
            if frame_count % 125 == 0:
                logger.debug(
                    "Capture loop alive: frame=%d, silence=%d, transcript='%s', "
                    "speech_ready=%s, stt=%s, tts_playing=%s",
                    frame_count, session.silence_frames,
                    session.current_transcript[:40],
                    session.speech_ready_event is not None and not session.speech_ready_event.is_set(),
                    session.stt_session is not None,
                    session.tts_playing,
                )

            if pieces:
                # Got text → speech active
                session.silence_frames = 0
                for piece in pieces:
                    _handle_text_piece(session, piece)
            else:
                # No text → silence frame
                session.silence_frames += 1
                _check_end_of_speech(session)
    except asyncio.CancelledError:
        logger.debug("Audio capture loop cancelled")
    except Exception as e:
        if not session.shutting_down:
            logger.error("Audio capture error: %s", e)
    logger.info("Audio capture loop exited (frame_count=%d)", frame_count)


def _handle_text_piece(session: TalkSession, piece: str) -> None:
    """Process a text piece from STT."""
    # During TTS playback: discard echo, only detect barge-in
    if session.tts_playing:
        if time.time() < session.barge_in_suppressed_until:
            return  # Discard echo during barge-in grace period
        # Barge-in: stop TTS, then accumulate this piece as user speech
        logger.info("Barge-in detected on '%s' at char %d", piece.strip(), session.tts_char_offset)
        session.interrupted_at = session.tts_char_offset
        session.tts_playing = False
        _flush_audio_output(session)

    if not session.current_transcript:
        logger.debug("Speech detected (after %d silence frames): ['%s']",
                      session.silence_frames, piece.strip())

    # Cancel pending flush — user still talking
    if session.flush_timer:
        session.flush_timer.cancel()
        session.flush_timer = None
        session.is_flushing = False

    # Accumulate transcript
    is_first = session.current_transcript == ""
    is_punct = piece and piece[0] in ".,!?;:'\")]}>"
    sep = "" if (is_first or is_punct) else ""
    session.current_transcript += sep + piece

    _emit(session, {"type": "stt_partial", "text": session.current_transcript})


def _check_end_of_speech(session: TalkSession) -> None:
    """Check if enough silence frames have passed to trigger end-of-speech."""
    if (
        session.current_transcript.strip()
        and session.silence_frames >= PAUSE_FRAMES
        and not session.is_flushing
    ):
        session.is_flushing = True
        logger.debug("End-of-speech triggered after %d silence frames", session.silence_frames)

        # Feed zero frames to flush STT pipeline
        stt = session.stt_session
        if stt is not None:
            silence_bytes = np.zeros(FRAME_SIZE, dtype=np.int16).tobytes()
            for _ in range(FLUSH_FRAMES):
                pieces = stt.feed_pcm(silence_bytes)
                for piece in pieces:
                    is_first = session.current_transcript == ""
                    is_punct = piece and piece[0] in ".,!?;:'\")]}>"
                    sep = "" if (is_first or is_punct) else ""
                    session.current_transcript += sep + piece
                    _emit(session, {"type": "stt_partial", "text": session.current_transcript})

        # After delay, finalize transcript
        loop = asyncio.get_event_loop()
        session.flush_timer = loop.call_later(
            STT_DELAY_SEC, _finalize_transcript, session
        )


def _finalize_transcript(session: TalkSession) -> None:
    """Called after flush delay — deliver final transcript."""
    session.flush_timer = None
    transcript = session.current_transcript.strip()
    session.current_transcript = ""
    session.silence_frames = 0
    session.is_flushing = False
    session.frames_since_reset = 0

    if not transcript:
        return

    logger.info("Finalized transcript: '%s'", transcript[:80])

    _emit(session, {"type": "stt_final", "text": transcript})

    if session.speech_ready_event:
        # Active /talk call waiting
        session.final_transcript = transcript
        session.speech_ready_event.set()
        _emit(session, {"type": "done", "timed_out": False})
    else:
        # No active call — queue it (post-listen)
        logger.info("Queued between-call speech: '%s'", transcript[:80])
        session.queued_speech.append((time.time(), transcript))
        # Flag capture loop to reset STT session (prevents ghost transcripts
        # from residual model state without dropping frames)
        session.stt_needs_reset = True


async def drain_monitor_loop(session: TalkSession, stderr) -> None:
    """Monitor audio-io stderr for DRAINED signals."""
    loop = asyncio.get_event_loop()
    try:
        while not session.shutting_down:
            line = await loop.run_in_executor(None, stderr.readline)
            if not line:
                logger.warning("Drain monitor: stderr EOF — audio-io may have died")
                break
            text = line.decode().strip()
            if text == "DRAINED":
                logger.debug("DRAINED received (tts_write_done=%s, tts_playing=%s)",
                             session.tts_write_done, session.tts_playing)
                session.playback_drained.set()
                if session.tts_write_done and session.tts_playing:
                    session.tts_playing = False
    except asyncio.CancelledError:
        logger.debug("Drain monitor cancelled")
    except Exception as e:
        logger.error("Drain monitor error: %s", e)
    logger.info("Drain monitor exited")


async def send_audio_to_output(session: TalkSession, audio_int16_bytes: bytes) -> None:
    """Write TTS audio to audio-io stdin."""
    if session.audio_process and session.audio_process.stdin:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, session.audio_process.stdin.write, audio_int16_bytes)
        await loop.run_in_executor(None, session.audio_process.stdin.flush)


def _flush_audio_output(session: TalkSession) -> None:
    """Send SIGUSR1 to audio-io for barge-in flush."""
    if session.audio_process:
        try:
            session.audio_process.send_signal(signal.SIGUSR1)
        except Exception:
            pass


def stop_audio_pipeline(session: TalkSession) -> None:
    """Terminate audio-io subprocess and cancel async tasks."""
    session.shutting_down = True
    if session.capture_task and not session.capture_task.done():
        session.capture_task.cancel()
    if session.drain_task and not session.drain_task.done():
        session.drain_task.cancel()
    if session.post_listen_task and not session.post_listen_task.done():
        session.post_listen_task.cancel()
    if session.tts_task and not session.tts_task.done():
        session.tts_task.cancel()
    if session.audio_process:
        try:
            session.audio_process.terminate()
            session.audio_process.wait(timeout=2)
        except Exception:
            try:
                session.audio_process.kill()
            except Exception:
                pass
        session.audio_process = None


# ── TTS Generation ───────────────────────────────────────────────────────────

_SENTENCE_RE = re.compile(r'[^.!?\n]+[.!?\n]+\s*|[^.!?\n]+$')


async def generate_and_play_tts(session: TalkSession, text: str, voice: str) -> None:
    """Generate TTS audio and feed to audio-io or WS output.

    Splits text into sentences for progressive generation — first sentence
    starts playing quickly, model lock is released between sentences for STT.
    """
    try:
        model = get_tts_model()
        loop = asyncio.get_event_loop()

        # Split text into sentences for progressive generation
        sentences = [m.group() for m in _SENTENCE_RE.finditer(text) if m.group().strip()]
        if not sentences:
            sentences = [text]

        # Pre-compute word boundaries for full text (karaoke offsets)
        words = [(m.start(), m.group()) for m in re.finditer(r'\S+', text)]
        word_idx = 0
        tts_started = False

        # Track cumulative char offset across sentences
        sent_char_base = 0

        for sent_idx, sentence in enumerate(sentences):
            if not session.tts_playing:
                break

            # Generate one sentence under model lock
            def _generate_sentence(sent=sentence):
                voice_state = _get_tts_voice_state(voice)
                chunks = []
                for audio_chunk in model.generate_audio_stream(voice_state, sent):
                    chunk_np = np.array(audio_chunk)
                    chunk_np = np.clip(chunk_np, -1, 1)
                    chunk_int16 = (chunk_np * 32767).astype(np.int16)
                    chunks.append(chunk_int16.tobytes())
                return chunks

            async with _model_lock:
                audio_bytes_list = await loop.run_in_executor(None, _generate_sentence)

            if not session.tts_playing:
                logger.debug("TTS: tts_playing=False after generation, skipping playback")
                break

            if not tts_started:
                # Set barge-in grace and emit start on first sentence
                session.barge_in_suppressed_until = time.time() + BARGE_IN_GRACE_MS / 1000
                _emit(session, {"type": "tts_start", "text": text})
                tts_started = True

            # Find the offset of this sentence in the full text
            sent_start = text.find(sentence.strip(), sent_char_base)
            if sent_start < 0:
                sent_start = sent_char_base
            sent_end = sent_start + len(sentence.rstrip())

            # Write audio in sub-chunks
            combined = b"".join(audio_bytes_list)
            total_len = len(combined)
            sent_chars = len(sentence.rstrip())
            SUB_CHUNK = FRAME_BYTES * 4  # ~320ms
            logger.debug("TTS: sentence %d/%d: %d bytes (%d chars) '%s'",
                          sent_idx + 1, len(sentences), total_len, sent_chars,
                          sentence.strip()[:40])

            sent_play_start = time.monotonic()
            for i in range(0, total_len, SUB_CHUNK):
                if not session.tts_playing:
                    logger.debug("TTS: barge-in during write at byte %d/%d", i, total_len)
                    break

                chunk_data = combined[i:i + SUB_CHUNK]
                # Proportional char offset within this sentence
                frac = min((i + SUB_CHUNK) / total_len, 1.0) if total_len else 1.0
                prev_char_offset = session.tts_char_offset
                session.tts_char_offset = sent_start + int(sent_chars * frac)

                if session.ws_send_audio:
                    # External mode: send binary frame with char position header
                    header = struct.pack('<II', prev_char_offset, session.tts_char_offset)
                    if session.audio_format == "pcm_f32le":
                        pcm = np.frombuffer(chunk_data, dtype=np.int16).astype(np.float32) / 32767.0
                        await session.ws_send_audio(header + pcm.tobytes())
                    else:
                        await session.ws_send_audio(header + chunk_data)
                else:
                    await send_audio_to_output(session, chunk_data)

                # Emit word events as char offset crosses word boundaries
                while word_idx < len(words) and words[word_idx][0] < session.tts_char_offset:
                    offset, word = words[word_idx]
                    _emit(session, {"type": "tts_word", "word": word, "offset": offset})
                    word_idx += 1

                # Pace to match playback: sleep until this chunk would be heard
                target = sent_play_start + (i + len(chunk_data)) / (2 * SAMPLE_RATE)
                delay = target - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)

            sent_char_base = sent_end + 1  # advance past sentence + separator

        # Emit any remaining words
        while word_idx < len(words):
            offset, word = words[word_idx]
            _emit(session, {"type": "tts_word", "word": word, "offset": offset})
            word_idx += 1

        # Done writing — wait for DRAINED only in builtin audio mode
        if session.tts_playing and not session.ws_send_audio:
            session.tts_write_done = True
            session.playback_drained.clear()
            logger.debug("TTS: all audio written, waiting for DRAINED (timeout=30s)")
            try:
                await asyncio.wait_for(session.playback_drained.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning("TTS: DRAINED timeout after 30s")
            if session.tts_playing:
                session.tts_playing = False
                logger.info("TTS playback complete")
        elif session.tts_playing:
            # External mode: no DRAINED, just mark done
            session.tts_playing = False

        if tts_started:
            _emit(session, {"type": "tts_done", "interrupted_at": session.interrupted_at})
            # If TTS ended naturally (no barge-in), reset STT to clear echo residuals.
            # The AEC doesn't fully cancel speaker output; residual audio in the STT
            # pipeline would otherwise be transcribed as ghost speech.
            if session.interrupted_at is None and session.stt_session is not None:
                session.current_transcript = ""
                session.silence_frames = 0
                session.is_flushing = False
                session.frames_since_reset = 0  # skip frames during transition
                if session.flush_timer:
                    session.flush_timer.cancel()
                    session.flush_timer = None
                engine = get_stt_engine()
                async with _model_lock:
                    session.stt_session = engine.new_session()
                session.frames_since_reset = 0
                logger.debug("STT session reset after TTS to clear echo residuals")
    except asyncio.CancelledError:
        logger.debug("TTS task cancelled")
        session.tts_playing = False
        session.tts_write_done = False
        raise
    except Exception as e:
        logger.exception("TTS generation/playback failed")
        session.tts_playing = False
    finally:
        session.tts_write_done = False


# ── Post-call listening ─────────────────────────────────────────────────────

async def _post_listen(session: TalkSession, timeout: float) -> None:
    """Keep STT session alive after a /talk call to capture between-call speech."""
    logger.info("Post-listen started (%.1fs window)", timeout)
    try:
        await asyncio.sleep(timeout)
        logger.info("Post-listen expired (no new call within %.1fs)", timeout)
    except asyncio.CancelledError:
        logger.info("Post-listen cancelled (new /talk call)")
    finally:
        # Clean up STT session
        session.stt_session = None
        session.post_listen_task = None


# ── HTTP Endpoints ──────────────────────────────────────────────────────────

async def index(request: Request) -> JSONResponse:
    """Info endpoint."""
    return JSONResponse({
        "service": "voice_server",
        "endpoints": {
            "GET /": "This info endpoint",
            "GET /health": "Health check",
            "POST /talk": "Speak + listen (JSON)",
            "POST /v1/audio/transcriptions": "OpenAI-compatible STT",
            "POST /v1/audio/speech": "OpenAI-compatible TTS",
        },
    })


async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "models": {
        "tts": _tts_model is not None,
        "stt": _stt_engine is not None,
    }})


async def _talk_setup(
    session: TalkSession, text: str | None, voice: str, lookbehind: float
) -> dict | None:
    """Shared setup for /talk: cancel old TTS, flush STT, check queued speech.

    Returns a queued-speech result dict if speech was queued, else None.
    After this call, the session has a fresh STT session and is ready for TTS.
    """
    # Cancel any in-flight TTS task from a previous call
    if session.tts_task and not session.tts_task.done():
        logger.debug("Cancelling previous TTS task")
        session.tts_task.cancel()
        try:
            await session.tts_task
        except (asyncio.CancelledError, Exception):
            pass
        session.tts_task = None
    session.tts_playing = False
    session.tts_write_done = False

    # Flush STT pipeline and save any in-progress speech before cancelling post-listen.
    # Skip flush if session was just finalized (stt_needs_reset) — no in-progress speech,
    # and flushing the old session would extract ghost/duplicate text.
    if session.stt_needs_reset:
        session.stt_needs_reset = False
    elif session.stt_session is not None:
        async with _model_lock:
            stt = session.stt_session
            if stt is not None:
                silence_bytes = np.zeros(FRAME_SIZE, dtype=np.int16).tobytes()
                for _ in range(SAVE_FLUSH_FRAMES):
                    pieces = stt.feed_pcm(silence_bytes)
                    for piece in pieces:
                        is_first = session.current_transcript == ""
                        is_punct = piece and piece[0] in ".,!?;:'\")]}>"
                        sep = "" if (is_first or is_punct) else ""
                        session.current_transcript += sep + piece
        if session.current_transcript.strip():
            logger.info("Saving in-progress speech: '%s'", session.current_transcript.strip()[:80])
            session.queued_speech.append((time.time(), session.current_transcript.strip()))
        session.current_transcript = ""
        session.silence_frames = 0
        session.is_flushing = False
        if session.flush_timer:
            session.flush_timer.cancel()
            session.flush_timer = None

    # Cancel previous post-listen task
    if session.post_listen_task and not session.post_listen_task.done():
        session.post_listen_task.cancel()
        try:
            await session.post_listen_task
        except asyncio.CancelledError:
            pass
        session.post_listen_task = None

    # Check queued speech (filter by lookbehind window)
    if session.queued_speech:
        cutoff = time.time() - lookbehind
        recent = [(ts, t) for ts, t in session.queued_speech if ts >= cutoff]
        dropped = len(session.queued_speech) - len(recent)
        session.queued_speech.clear()
        if recent:
            combined = " ".join(t for _, t in recent)
            logger.info("Returning queued speech (%d items, %d dropped): '%s'",
                        len(recent), dropped, combined[:80])
            return {
                "user_speech": combined,
                "interrupted_at": 0,
                "timed_out": False,
            }
        else:
            logger.info("Queued speech expired (%d items outside lookbehind window)", dropped)

    # Create fresh STT session
    engine = get_stt_engine()
    async with _model_lock:
        session.stt_session = engine.new_session()

    # Reset VAD state
    session.speech_ready_event = asyncio.Event()
    session.final_transcript = ""
    session.interrupted_at = None
    session.current_transcript = ""
    session.silence_frames = 0
    session.frames_since_reset = 0
    session.is_flushing = False
    if session.flush_timer:
        session.flush_timer.cancel()
        session.flush_timer = None

    # Start TTS if text provided
    if text:
        session.tts_char_offset = 0
        session.tts_playing = True
        session.tts_write_done = False
        session.playback_drained.clear()
        session.barge_in_suppressed_until = float("inf")
        session.tts_task = asyncio.create_task(generate_and_play_tts(session, text, voice))

    return None


async def _talk_start_post_listen(session: TalkSession, post_timeout: float) -> None:
    """Start post-call listening with a fresh STT session."""
    if post_timeout > 0:
        engine = get_stt_engine()
        async with _model_lock:
            session.stt_session = engine.new_session()
        session.frames_since_reset = 0
        session.post_listen_task = asyncio.create_task(_post_listen(session, post_timeout))
    else:
        session.stt_session = None


async def talk_endpoint(request: Request) -> Response:
    """POST /talk — speak text, listen for response. JSON or NDJSON stream."""
    body = await request.json()
    text = body.get("text")
    voice = body.get("voice", DEFAULT_VOICE)
    timeout = body.get("timeout")  # seconds or None
    post_timeout = body.get("post_timeout", 0)
    lookbehind = body.get("lookbehind", 0)
    stream = body.get("stream", False)

    session = get_session()

    queued = await _talk_setup(session, text, voice, lookbehind)

    if stream:
        # Streaming NDJSON path
        if queued:
            # Emit queued speech as events and return
            async def queued_generator():
                yield json.dumps({"type": "queued_speech", "text": queued["user_speech"],
                                  "interrupted_at": queued["interrupted_at"]}) + "\n"
                yield json.dumps({"type": "done", "timed_out": False}) + "\n"
                await _talk_start_post_listen(session, post_timeout)
            return StreamingResponse(queued_generator(), media_type="application/x-ndjson")

        session.event_queue = asyncio.Queue()

        async def event_generator():
            tts_active = text is not None
            try:
                while True:
                    try:
                        # No timeout during TTS — only apply after tts_done
                        ev_timeout = None if tts_active else timeout
                        event = await asyncio.wait_for(session.event_queue.get(), timeout=ev_timeout)
                    except asyncio.TimeoutError:
                        # Timeout waiting for speech
                        timed_out = True
                        if session.current_transcript.strip():
                            final = session.current_transcript.strip()
                            session.current_transcript = ""
                            yield json.dumps({"type": "stt_final", "text": final}) + "\n"
                        if session.tts_playing:
                            session.tts_playing = False
                            _flush_audio_output(session)
                        yield json.dumps({"type": "done", "timed_out": True}) + "\n"
                        break
                    yield json.dumps(event) + "\n"
                    if event.get("type") == "tts_done":
                        tts_active = False
                    if event.get("type") == "done":
                        break
            finally:
                session.event_queue = None
                session.speech_ready_event = None
                await _talk_start_post_listen(session, post_timeout)

        return StreamingResponse(event_generator(), media_type="application/x-ndjson")

    # Non-streaming JSON path (original behavior)
    if queued:
        return JSONResponse(queued)

    # Wait for speech
    timed_out = False
    try:
        await asyncio.wait_for(session.speech_ready_event.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        timed_out = True
        if session.current_transcript.strip():
            session.final_transcript = session.current_transcript.strip()
            session.current_transcript = ""
        if session.tts_playing:
            session.tts_playing = False
            _flush_audio_output(session)

    result = {
        "user_speech": session.final_transcript,
        "interrupted_at": session.interrupted_at,
        "timed_out": timed_out,
    }
    session.speech_ready_event = None

    await _talk_start_post_listen(session, post_timeout)

    return JSONResponse(result)


async def openai_transcriptions(request: Request) -> Response:
    """OpenAI-compatible POST /v1/audio/transcriptions."""
    import sphn

    form = await request.form()
    upload = form.get("file")
    if not upload:
        return JSONResponse({"error": {"message": "No file provided"}}, status_code=400)

    response_format = form.get("response_format", "json")
    audio_bytes = await upload.read()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        pcm, _ = sphn.read(tmp_path, sample_rate=SAMPLE_RATE)
        audio = pcm[0].astype(np.float32)
    finally:
        os.unlink(tmp_path)

    engine = get_stt_engine()
    async with _model_lock:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, engine.transcribe_batch, audio)

    if response_format == "text":
        return PlainTextResponse(result["text"])
    elif response_format == "verbose_json":
        return JSONResponse(result)
    else:
        return JSONResponse({"text": result["text"]})


async def openai_speech(request: Request) -> Response:
    """OpenAI-compatible POST /v1/audio/speech — returns WAV audio."""
    body = await request.json()
    text = body.get("input", "")
    voice = body.get("voice", DEFAULT_VOICE)

    if not text.strip():
        return JSONResponse({"error": {"message": "No input text provided"}}, status_code=400)

    model = get_tts_model()

    def _generate():
        voice_state = _get_tts_voice_state(voice)
        return model.generate_audio(voice_state, text)

    loop = asyncio.get_event_loop()
    async with _model_lock:
        audio = await loop.run_in_executor(None, _generate)

    audio_np = np.array(audio)
    audio_np = np.clip(audio_np, -1, 1)
    audio_int16 = (audio_np * 32767).astype(np.int16)

    # Build WAV in memory
    buf = io.BytesIO()
    num_samples = len(audio_int16)
    data_size = num_samples * 2
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(audio_int16.tobytes())

    return Response(
        content=buf.getvalue(),
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="speech.wav"'},
    )


# ── WebSocket Endpoint ───────────────────────────────────────────────────────

# Guard: only one builtin-audio WS at a time
_ws_builtin_lock = asyncio.Lock()


_WS_EVENT_MAP = {
    "tts_start": "speak.started",
    "tts_word": "speak.progress",
    "tts_done": "speak.done",
    "stt_partial": "transcript.partial",
    "stt_final": "transcript.final",
}


def _map_ws_event(event: dict) -> dict | None:
    """Map internal event names to WS protocol names."""
    etype = event.get("type")
    ws_type = _WS_EVENT_MAP.get(etype)
    if ws_type is None:
        return None  # skip unmapped events (done, queued_speech, etc.)
    mapped = {"type": ws_type}
    if ws_type == "speak.started":
        mapped["text"] = event.get("text", "")
    elif ws_type == "speak.progress":
        mapped["word"] = event.get("word", "")
        mapped["offset"] = event.get("offset", 0)
        mapped["spoken_end"] = event.get("offset", 0) + len(event.get("word", ""))
    elif ws_type == "speak.done":
        mapped["interrupted_at"] = event.get("interrupted_at")
    elif ws_type in ("transcript.partial", "transcript.final"):
        mapped["text"] = event.get("text", "")
    return mapped


async def _external_capture_loop(session: TalkSession, audio_queue: asyncio.Queue) -> None:
    """Like audio_capture_loop but reads PCM from a queue (WS binary frames)."""
    try:
        while True:
            data = await audio_queue.get()
            if data is None:
                break  # WS closed
            if session.stt_session is None:
                continue
            async with _model_lock:
                stt = session.stt_session
                if stt is None:
                    continue
                if session.stt_needs_reset:
                    logger.debug("Resetting external STT session (post-finalization)")
                    engine = get_stt_engine()
                    session.stt_session = engine.new_session()
                    session.frames_since_reset = 0
                    session.stt_needs_reset = False
                    continue
                try:
                    pieces = stt.feed_pcm(data)
                except Exception as e:
                    if "max-steps" in str(e) or "max_steps" in str(e):
                        logger.warning("External STT: max-steps, resetting")
                        engine = get_stt_engine()
                        session.stt_session = engine.new_session()
                        session.frames_since_reset = 0
                        continue
                    raise
            session.frames_since_reset += 1
            if session.frames_since_reset <= SKIP_INITIAL_FRAMES:
                continue
            if pieces:
                session.silence_frames = 0
                for piece in pieces:
                    _handle_text_piece(session, piece)
            else:
                session.silence_frames += 1
                _check_end_of_speech(session)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error("External capture loop error: %s", e)


async def _ws_start_stt(session: TalkSession) -> None:
    """Initialize a fresh STT session for the WS connection."""
    engine = get_stt_engine()
    async with _model_lock:
        session.stt_session = engine.new_session()
    session.frames_since_reset = 0
    session.current_transcript = ""
    session.silence_frames = 0
    session.is_flushing = False
    if session.flush_timer:
        session.flush_timer.cancel()
        session.flush_timer = None


async def _ws_cancel_tts(session: TalkSession) -> None:
    """Cancel in-flight TTS and flush audio."""
    if session.tts_task and not session.tts_task.done():
        session.tts_task.cancel()
        try:
            await session.tts_task
        except (asyncio.CancelledError, Exception):
            pass
        session.tts_task = None
    session.tts_playing = False
    session.tts_write_done = False
    if not session.ws_send_audio:
        _flush_audio_output(session)


async def ws_endpoint(websocket: WebSocket) -> None:
    """WebSocket /ws — persistent bidirectional connection for TTS + STT."""
    await websocket.accept()

    config: dict[str, Any] = {
        "audio": "builtin",
        "voice": DEFAULT_VOICE,
        "timeout": None,
        "format": "pcm_f32le",
    }
    is_external = False
    session: TalkSession | None = None
    ws_queue: asyncio.Queue | None = None
    forwarder: asyncio.Task | None = None
    ext_capture: asyncio.Task | None = None
    ext_audio_queue: asyncio.Queue | None = None
    original_event_queue: asyncio.Queue | None = None

    async def _event_forwarder():
        """Drain event queue → send JSON over WS."""
        try:
            while True:
                event = await ws_queue.get()
                mapped = _map_ws_event(event)
                if mapped:
                    await websocket.send_json(mapped)
        except (asyncio.CancelledError, WebSocketDisconnect):
            pass

    async def _setup_session():
        nonlocal session, ws_queue, forwarder, ext_capture, ext_audio_queue
        nonlocal original_event_queue, is_external

        is_external = config["audio"] == "external"

        if is_external:
            # Per-WS session for external audio
            session = TalkSession()
            session.audio_format = config.get("format", "pcm_f32le")
            session.ws_send_audio = lambda data: websocket.send_bytes(data)
            ext_audio_queue = asyncio.Queue()
            await _ws_start_stt(session)
            ext_capture = asyncio.create_task(
                _external_capture_loop(session, ext_audio_queue)
            )
        else:
            # Builtin: use singleton session, must be exclusive
            if not _ws_builtin_lock.locked():
                await _ws_builtin_lock.acquire()
            session = get_session()
            # Ensure STT session is active for mic audio
            if session.stt_session is None:
                await _ws_start_stt(session)

        # Set up event forwarding
        ws_queue = asyncio.Queue()
        original_event_queue = session.event_queue
        session.event_queue = ws_queue
        forwarder = asyncio.create_task(_event_forwarder())

    async def _handle_message(data: dict) -> None:
        nonlocal config
        msg_type = data.get("type")

        if msg_type == "session.update":
            new_cfg = data.get("session", {})
            config.update(new_cfg)
            await _setup_session()
            await websocket.send_json({
                "type": "session.created",
                "session": {**config, "sample_rate": SAMPLE_RATE},
            })

        elif msg_type == "speak":
            if session is None:
                await websocket.send_json({"type": "error", "message": "No session — send session.update first"})
                return
            text = data.get("text", "").strip()
            if not text:
                return
            # Cancel any in-flight TTS
            await _ws_cancel_tts(session)
            # Reset STT state for new exchange
            session.speech_ready_event = asyncio.Event()
            session.final_transcript = ""
            session.interrupted_at = None
            session.tts_char_offset = 0
            session.tts_playing = True
            session.tts_write_done = False
            session.playback_drained.clear()
            session.barge_in_suppressed_until = float("inf")
            voice = config.get("voice", DEFAULT_VOICE)
            session.tts_task = asyncio.create_task(
                generate_and_play_tts(session, text, voice)
            )

        elif msg_type == "speak.cancel":
            if session:
                was_active = session.tts_task and not session.tts_task.done()
                await _ws_cancel_tts(session)
                if was_active:
                    _emit(session, {"type": "tts_done", "interrupted_at": session.tts_char_offset})

        elif msg_type == "barge_in":
            if session and is_external:
                idx = data.get("spokenCharIndex", 0)
                session.interrupted_at = idx
                session.tts_playing = False
                await _ws_cancel_tts(session)

        elif msg_type == "input_audio.clear":
            if session:
                await _ws_start_stt(session)

    try:
        while True:
            msg = await websocket.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if "text" in msg:
                data = json.loads(msg["text"])
                await _handle_message(data)
            elif "bytes" in msg and is_external and ext_audio_queue:
                raw = msg["bytes"]
                # Convert Float32 → Int16 if needed (STT expects Int16 PCM)
                if config.get("format", "pcm_f32le") == "pcm_f32le":
                    f32 = np.frombuffer(raw, dtype=np.float32)
                    int16 = (np.clip(f32, -1, 1) * 32767).astype(np.int16)
                    await ext_audio_queue.put(int16.tobytes())
                else:
                    await ext_audio_queue.put(raw)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("WS error: %s", e)
    finally:
        # CRITICAL: stop audio on disconnect
        if session:
            await _ws_cancel_tts(session)
            session.event_queue = original_event_queue
        if forwarder:
            forwarder.cancel()
        if ext_capture:
            if ext_audio_queue:
                await ext_audio_queue.put(None)  # signal loop to exit
            ext_capture.cancel()
        if not is_external and _ws_builtin_lock.locked():
            _ws_builtin_lock.release()
        logger.info("WS connection closed (audio=%s)", config.get("audio"))


# ── App Composition ──────────────────────────────────────────────────────────

def create_app(start_audio: bool = False) -> Starlette:
    routes = [
        Route("/", index),
        Route("/health", health),
        Route("/talk", talk_endpoint, methods=["POST"]),
        Route("/v1/audio/transcriptions", openai_transcriptions, methods=["POST"]),
        Route("/v1/audio/speech", openai_speech, methods=["POST"]),
        WebSocketRoute("/ws", ws_endpoint),
    ]

    async def on_startup():
        if start_audio:
            session = get_session()
            await start_audio_pipeline(session)

    return Starlette(routes=routes, on_startup=[on_startup])


# ── Daemon Management ────────────────────────────────────────────────────────

def _get_lockfile_path() -> Path:
    return Path.home() / ".cache" / "voice_server" / "daemon.json"


def _read_lockfile() -> dict | None:
    path = _get_lockfile_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _write_lockfile(pid: int, port: int) -> None:
    path = _get_lockfile_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "pid": pid,
        "port": port,
        "script_mtime": os.path.getmtime(__file__),
    }))


def _remove_lockfile() -> None:
    path = _get_lockfile_path()
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def kill_daemon() -> bool:
    """Stop the running daemon. Returns True if a daemon was killed."""
    info = _read_lockfile()
    if info is None:
        print("No daemon running", file=sys.stderr)
        return False

    pid = info["pid"]
    if _pid_alive(pid):
        print(f"Stopping daemon (pid={pid})...", file=sys.stderr)
        try:
            os.kill(pid, signal.SIGTERM)
            # Wait up to 5 seconds
            for _ in range(50):
                if not _pid_alive(pid):
                    break
                time.sleep(0.1)
            else:
                os.kill(pid, signal.SIGKILL)
        except OSError:
            pass

    _remove_lockfile()
    print("Daemon stopped", file=sys.stderr)
    return True


def ensure_daemon(port: int) -> tuple[int, int]:
    """Ensure a daemon is running. Returns (pid, port)."""
    info = _read_lockfile()

    if info is not None:
        pid = info["pid"]
        old_port = info["port"]
        old_mtime = info.get("script_mtime", 0)
        current_mtime = os.path.getmtime(__file__)

        if _pid_alive(pid) and abs(old_mtime - current_mtime) < 0.001:
            return (pid, old_port)

        # Stale or outdated daemon — kill it
        if _pid_alive(pid):
            logger.debug("Killing outdated daemon pid=%d", pid)
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(0.5)
                if _pid_alive(pid):
                    os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        _remove_lockfile()

    # Start new daemon
    log_dir = Path.home() / ".cache" / "voice_server"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(log_dir / "daemon.log", "w")

    proc = subprocess.Popen(
        [sys.executable, __file__, "--foreground", "--port", str(port)],
        stdout=log_file,
        stderr=log_file,
        start_new_session=True,  # detach from parent
    )

    _write_lockfile(proc.pid, port)
    print(f"Started daemon (pid={proc.pid}, port={port})", file=sys.stderr)

    # Poll /health until ready
    import httpx
    for attempt in range(120):  # up to 60s
        time.sleep(0.5)
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=1.0)
            if r.status_code == 200:
                print(f"Daemon ready after {(attempt + 1) * 0.5:.1f}s", file=sys.stderr)
                return (proc.pid, port)
        except Exception:
            pass
        # Check if process died
        if proc.poll() is not None:
            print(f"Daemon died during startup (exit code {proc.returncode})", file=sys.stderr)
            print(f"Check logs: {log_dir / 'daemon.log'}", file=sys.stderr)
            _remove_lockfile()
            sys.exit(1)

    print("Daemon did not become ready in 60s", file=sys.stderr)
    print(f"Check logs: {log_dir / 'daemon.log'}", file=sys.stderr)
    sys.exit(1)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_timeout(s: str) -> float:
    """Parse timeout string like '10s', '1m', '500ms', 'inf'."""
    s = s.strip().lower()
    if s == "inf":
        return float("inf")
    if s.endswith("ms"):
        return float(s[:-2]) / 1000
    elif s.endswith("s"):
        return float(s[:-1])
    elif s.endswith("m"):
        return float(s[:-1]) * 60
    else:
        return float(s)


# ── TUI Rendering ────────────────────────────────────────────────────────────

DIM = "\033[2m"
DIM_ITALIC = "\033[2;3m"
RESET = "\033[0m"
GRAY_BG = "\033[100m"


def _ts() -> str:
    return f"{DIM_ITALIC}{datetime.datetime.now():%H:%M:%S}{RESET}"


class TUIRenderer:
    """Renders NDJSON events as a TUI with karaoke TTS and streaming STT."""

    # Visual width of "HH:MM:SS " — used to align the waiting prompt
    _TS_PAD = " " * 9

    def __init__(self, tty: bool = True):
        self._tty = tty
        self._stt_started = False
        self._waiting_active = False
        # Karaoke state (tty only)
        self._tts_text: str | None = None
        self._tts_spoken_end: int = 0
        self._tts_line_from_typing: bool = False
        # Non-tty STT delta tracking: how many chars we've already printed
        self._stt_printed: int = 0
        # In-place rewrite state: cursor save/restore for multi-line text
        self._inplace_saved: bool = False

    def _start_line(self) -> None:
        """Start a new output line, clearing the waiting prompt if active."""
        if self._waiting_active:
            sys.stdout.write("\r\033[K")
            self._waiting_active = False
        else:
            sys.stdout.write("\n")

    def show_waiting(self) -> None:
        if not self._tty:
            return
        self._waiting_active = True
        sys.stdout.write(f"\n{self._TS_PAD}{DIM_ITALIC}(waiting for input or voice...){RESET}")
        sys.stdout.flush()

    def _clear_inplace(self) -> None:
        """Clear previously written in-place text using cursor restore."""
        if self._inplace_saved:
            sys.stdout.write("\033[u\033[J")  # restore cursor + clear to end of screen
        else:
            sys.stdout.write("\r\033[J")

    def _begin_inplace(self) -> None:
        """Save cursor position before writing in-place text."""
        sys.stdout.write("\033[s")
        self._inplace_saved = True

    def _end_inplace(self) -> None:
        """Finish in-place mode (after writing final line with \\n)."""
        self._inplace_saved = False

    def _rewrite_karaoke(self) -> None:
        """Rewrite the TTS text with spoken=normal, unspoken=dim.

        Text wraps naturally at terminal width. Uses cursor save/restore
        for reliable multi-line in-place updates.
        """
        if not self._tts_text:
            return
        prefix = f"{_ts()} \u23fa "

        # Collapse newlines/tabs to spaces for display
        text = self._tts_text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        end = self._tts_spoken_end

        spoken = text[:end]
        unspoken = text[end:]

        # Clear previous output, save position, write new output
        self._clear_inplace()
        self._begin_inplace()

        out = f"{prefix}{spoken}"
        if unspoken:
            out += f"{DIM}{unspoken}{RESET}"
        sys.stdout.write(out)
        sys.stdout.flush()

    # Map WS event names → internal names for unified rendering
    _WS_TO_INTERNAL = {
        "speak.started": "tts_start",
        "speak.progress": "tts_word",
        "speak.done": "tts_done",
        "transcript.partial": "stt_partial",
        "transcript.final": "stt_final",
    }

    def render(self, event: dict) -> None:
        etype = event.get("type")
        # Normalize WS event names to internal names
        etype = self._WS_TO_INTERNAL.get(etype, etype)
        # speak.progress uses spoken_end; tts_word uses offset+len(word)
        if etype == "tts_word" and "spoken_end" in event and "offset" not in event:
            event = {**event, "offset": event["spoken_end"] - len(event.get("word", ""))}

        # ── Non-tty: plain forward-only text ─────────────────────────────
        if not self._tty:
            if etype == "tts_start":
                sys.stdout.write("\u23fa ")
            elif etype == "tts_word":
                sys.stdout.write(f"{event['word']} ")
            elif etype == "tts_done":
                sys.stdout.write("\n")
                self._stt_printed = 0
            elif etype in ("stt_partial", "stt_final"):
                text = event.get("text", "").strip()
                if not self._stt_started:
                    self._stt_started = True
                    self._stt_printed = 0
                    sys.stdout.write(f"\u276f {text}")
                    self._stt_printed = len(text)
                else:
                    delta = text[self._stt_printed:]
                    if delta:
                        sys.stdout.write(delta)
                        self._stt_printed = len(text)
                if etype == "stt_final":
                    sys.stdout.write("\n")
                    self._stt_started = False
                    self._stt_printed = 0
            elif etype == "queued_speech":
                sys.stdout.write(f"\u276f {event.get('text', '').strip()}\n")
            elif etype == "done" and event.get("timed_out"):
                sys.stdout.write("(timed out)\n")
            sys.stdout.flush()
            return

        # ── TTY: full ANSI with karaoke and in-place updates ─────────────
        if etype == "tts_start":
            self._tts_text = event.get("text", "")
            self._tts_spoken_end = 0
            if not self._tts_line_from_typing:
                self._start_line()
            self._rewrite_karaoke()
        elif etype == "tts_word":
            offset = event.get("offset", 0)
            word = event.get("word", "")
            self._tts_spoken_end = event.get("spoken_end", offset + len(word))
            self._rewrite_karaoke()
        elif etype == "tts_done":
            # Clear in-place karaoke and write final line
            self._clear_inplace()
            interrupted_at = event.get("interrupted_at")
            text = (self._tts_text or "").replace("\n", " ").replace("\r", " ").replace("\t", " ")
            if interrupted_at is not None and text:
                spoken = text[:interrupted_at].rstrip()
                sys.stdout.write(f"{_ts()} \u23fa {spoken}\n")
            elif text:
                sys.stdout.write(f"{_ts()} \u23fa {text}\n")
            else:
                sys.stdout.write("\n")
            sys.stdout.flush()
            self._end_inplace()
            self._tts_text = None
            self._tts_spoken_end = 0
            self._tts_line_from_typing = False
            self._stt_started = False
        elif etype == "stt_partial":
            text = event.get("text", "").strip()
            if not self._stt_started:
                self._stt_started = True
                self._start_line()
                self._begin_inplace()
                sys.stdout.write(f"{_ts()} \u276f {text}")
            else:
                self._clear_inplace()
                self._begin_inplace()
                sys.stdout.write(f"{_ts()} \u276f {text}")
            sys.stdout.flush()
        elif etype == "stt_final":
            text = event.get("text", "").strip()
            if self._stt_started:
                self._clear_inplace()
                sys.stdout.write(f"{_ts()} \u276f {text}\n")
            else:
                self._start_line()
                sys.stdout.write(f"{_ts()} \u276f {text}\n")
            sys.stdout.flush()
            self._end_inplace()
            self._stt_started = False
        elif etype == "queued_speech":
            text = event.get("text", "").strip()
            self._start_line()
            sys.stdout.write(f"{_ts()} \u276f {text}\n")
            sys.stdout.flush()
        elif etype == "done":
            if event.get("timed_out"):
                self._start_line()
                sys.stdout.write(f"{_ts()} {DIM_ITALIC}(timed out){RESET}\n")
                sys.stdout.flush()


# ── Stream + Stdin concurrency ───────────────────────────────────────────────


async def _run_ws_with_stdin(
    ws, text: str | None,
    renderer: TUIRenderer, stdin_queue: asyncio.Queue,
    *, check_stdin_during_tts: bool = True,
) -> str | bool | None:
    """Send speak command over WS and render events while checking stdin.

    Args:
        ws: WebSocket connection (websockets client)
        text: Text to speak (None = listen-only, just wait for stdin/events)
        check_stdin_during_tts: If False, only race stdin after speak.done.
            Prevents pre-buffered stdin from cancelling TTS.

    Returns:
        str: new text from stdin (interrupted)
        None: stdin EOF
        False: exchange finished naturally (speak.done + transcript.final)
    """
    if text is not None:
        await ws.send(json.dumps({"type": "speak", "text": text}))

    tts_active = text is not None and not check_stdin_during_tts
    tts_done_seen = text is None  # no TTS → already "done"

    while True:
        ws_task = asyncio.create_task(ws.recv())

        if tts_active:
            # Don't check stdin during TTS — just consume events
            try:
                await ws_task
            except Exception:
                return False
        else:
            in_task = asyncio.create_task(stdin_queue.get())
            done_set, pending_set = await asyncio.wait(
                {ws_task, in_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for t in pending_set:
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass

            if in_task in done_set:
                result = in_task.result()
                if isinstance(result, str) and not tts_done_seen:
                    # Interrupt current TTS (only if still playing)
                    await ws.send(json.dumps({"type": "speak.cancel"}))
                return result  # str or None (EOF)

        try:
            raw = ws_task.result()
        except Exception:
            return False

        if isinstance(raw, bytes):
            continue  # skip binary audio frames in CLI mode

        event = json.loads(raw)
        renderer.render(event)

        etype = event.get("type")
        if etype in ("speak.done", "tts_done"):
            tts_active = False
            tts_done_seen = True
        if etype in ("transcript.final", "stt_final"):
            return False  # exchange complete
        if etype == "done":
            return False


# ── Stdin readers ────────────────────────────────────────────────────────────


async def _stdin_reader_line(loop, queue: asyncio.Queue) -> None:
    """Read lines from stdin, put into queue. Puts None on EOF."""
    try:
        while True:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:
                await queue.put(None)
                break
            line = line.strip()
            if line:
                await queue.put(line)
    except asyncio.CancelledError:
        pass


async def _stdin_reader_unbuffered(loop, queue: asyncio.Queue) -> None:
    """Read stdin unbuffered with 300ms debounce. Puts None on EOF."""
    import select

    def _read_debounced() -> str | None:
        buf = os.read(0, 4096)
        if not buf:
            return None
        # Debounce: accumulate more data for up to 300ms
        deadline = time.time() + 0.3
        while time.time() < deadline:
            remaining = max(0, deadline - time.time())
            ready, _, _ = select.select([0], [], [], remaining)
            if ready:
                more = os.read(0, 4096)
                if not more:
                    break
                buf += more
            else:
                break
        return buf.decode("utf-8", errors="replace")

    try:
        while True:
            text = await loop.run_in_executor(None, _read_debounced)
            if text is None:
                await queue.put(None)
                break
            text = text.strip()
            if text:
                await queue.put(text)
    except asyncio.CancelledError:
        pass


async def _stdin_reader_tty(
    loop, queue: asyncio.Queue, renderer: TUIRenderer
) -> None:
    """Read from tty in cbreak mode with live echo and waiting-prompt clearing."""
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    def _restore_terminal():
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            pass

    atexit.register(_restore_terminal)

    def _read_line_cbreak() -> str | None:
        tty.setcbreak(fd)
        buf: list[str] = []
        first_char = True
        try:
            while True:
                raw = os.read(fd, 1)
                if not raw:
                    return None
                ch = raw.decode("utf-8", errors="replace")
                if ch in ("\n", "\r"):
                    sys.stdout.write(RESET)  # end dim; NO newline — karaoke rewrites in-place
                    sys.stdout.flush()
                    renderer._tts_line_from_typing = True
                    return "".join(buf)
                elif ch in ("\x7f", "\x08"):  # backspace / delete
                    if buf:
                        buf.pop()
                        sys.stdout.write("\b \b")
                        sys.stdout.flush()
                elif ch == "\x03":  # Ctrl+C (if ISIG disabled)
                    return None
                elif ch == "\x04":  # Ctrl+D
                    if not buf:
                        return None
                elif ch >= " ":  # printable
                    if first_char:
                        # Clear waiting prompt (or start new line) and show ⏺ prefix
                        if renderer._waiting_active:
                            renderer._waiting_active = False
                            sys.stdout.write("\r\033[K")
                        else:
                            sys.stdout.write("\n")
                        sys.stdout.write(f"{_ts()} \u23fa {DIM}")
                        sys.stdout.flush()
                        first_char = False
                    buf.append(ch)
                    sys.stdout.write(ch)
                    sys.stdout.flush()
        except OSError:
            return None

    try:
        while True:
            text = await loop.run_in_executor(None, _read_line_cbreak)
            if text is None:
                await queue.put(None)
                break
            text = text.strip()
            if text:
                await queue.put(text)
    except asyncio.CancelledError:
        pass
    finally:
        _restore_terminal()


# ── CLI modes ────────────────────────────────────────────────────────────────


async def single_mode(
    port: int, text: str | None, voice: str,
    timeout_val: float | None, post_timeout: float, lookbehind_val: float,
    is_tty: bool = True,
) -> None:
    """Single mode: one exchange (speak text, listen, show TUI, exit)."""
    import websockets

    renderer = TUIRenderer(tty=is_tty)
    async with websockets.connect(f"ws://127.0.0.1:{port}/ws") as ws:
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {"audio": "builtin", "voice": voice},
        }))
        # Wait for session.created
        await ws.recv()

        if text is not None:
            await ws.send(json.dumps({"type": "speak", "text": text}))

        async for raw in ws:
            if isinstance(raw, bytes):
                continue
            event = json.loads(raw)
            renderer.render(event)
            etype = event.get("type")
            if etype in ("speak.done", "tts_done"):
                # In single mode without timeout, wait briefly for transcript
                if timeout_val is None:
                    continue
            if etype in ("transcript.final", "stt_final", "done"):
                break
    sys.stdout.write("\n")


async def continuous_mode(
    port: int, voice: str,
    timeout_val: float | None, post_timeout: float, lookbehind_val: float,
    initial_text: str | None = None, input_mode: str = "line",
    is_tty: bool = True,
) -> None:
    """Continuous mode: interactive loop with streaming TUI over WebSocket.

    WS connection stays open for the entire session. After each exchange,
    STT events continue streaming. Typing new input sends a new speak command.
    On exit, WS close → server flushes audio immediately.
    """
    import websockets

    loop = asyncio.get_event_loop()
    stdin_queue: asyncio.Queue[str | None] = asyncio.Queue()
    renderer = TUIRenderer(tty=is_tty)

    if is_tty:
        reader_task = asyncio.create_task(_stdin_reader_tty(loop, stdin_queue, renderer))
    elif input_mode == "unbuffered":
        reader_task = asyncio.create_task(_stdin_reader_unbuffered(loop, stdin_queue))
    else:
        reader_task = asyncio.create_task(_stdin_reader_line(loop, stdin_queue))
    text = initial_text

    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}/ws") as ws:
            await ws.send(json.dumps({
                "type": "session.update",
                "session": {"audio": "builtin", "voice": voice},
            }))
            # Wait for session.created
            await ws.recv()

            while True:
                # In piped mode with no text, skip listen-only and just read next line
                if text is None and not is_tty:
                    text = await stdin_queue.get()
                    if text is None:
                        return  # EOF
                    continue

                result = await _run_ws_with_stdin(
                    ws, text, renderer, stdin_queue,
                    check_stdin_during_tts=is_tty,
                )
                text = None  # consumed

                if result is None:
                    return  # stdin EOF
                elif isinstance(result, str):
                    text = result  # new text from stdin
                    continue
                else:
                    # Exchange finished → show waiting, listen for speech/stdin
                    renderer.show_waiting()
                    text = None  # next iteration: listen-only (no speak, just wait)
    finally:
        reader_task.cancel()
        try:
            await reader_task
        except asyncio.CancelledError:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Voice server: TTS + STT with CLI and managed daemon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s "Hello, how are you?"              # Single mode (speak, listen, exit)
  %(prog)s --mode continuous                   # Continuous interactive mode
  %(prog)s --loiter "Hello"                   # Keep listening between calls
  echo "Hello" | %(prog)s                     # Read from stdin (single)
  %(prog)s -t 10s "Hello"                     # With listening timeout
  %(prog)s --kill                              # Stop daemon
  %(prog)s --foreground --preload stt tts     # Run in foreground
""",
    )
    parser.add_argument("text", nargs="?", help="Text to speak (reads stdin if omitted and not a tty)")
    parser.add_argument("--mode", choices=["single", "continuous"], default=None,
                        help="CLI mode: single (one exchange, exit) or continuous (interactive loop)")
    parser.add_argument("--input", choices=["line", "unbuffered"], default=None,
                        help="Input mode: line (send on Enter, default) or unbuffered (send as available)")
    parser.add_argument("--no-tty", action="store_true",
                        help="Force non-tty mode (no cbreak, no prompt clearing)")
    parser.add_argument("-t", "--timeout", help="Listening timeout (e.g. 10s, 1m, 500ms)")
    parser.add_argument("--loiter", action="store_true",
                        help="Keep listening between calls (enables post-listen + lookbehind)")
    parser.add_argument("-v", "--voice", default=DEFAULT_VOICE, help="Voice name")
    parser.add_argument("--voice-file", default=None, help="WAV file for voice cloning")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Daemon port")
    parser.add_argument("--foreground", action="store_true", help="Run server in foreground")
    parser.add_argument("--kill", action="store_true", help="Stop the running daemon")
    parser.add_argument("--preload", nargs="*", choices=["tts", "stt"], default=[],
                        help="Preload models (foreground mode)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (foreground mode)")
    parser.add_argument("--stt-repo", default=DEFAULT_STT_REPO, help="HuggingFace STT model repo")
    parser.add_argument("--stt-quantize", type=int, choices=[4, 8], default=None)
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if args.kill:
        kill_daemon()
        return

    if args.foreground:
        # Register cleanup
        _cleanup_done = False

        def _cleanup():
            nonlocal _cleanup_done
            if _cleanup_done:
                return
            _cleanup_done = True
            session = _session
            if session:
                stop_audio_pipeline(session)
            _remove_lockfile()

        atexit.register(_cleanup)

        # Preload models
        if "stt" in args.preload:
            global _stt_engine
            _stt_engine = STTEngine(hf_repo=args.stt_repo, quantized=args.stt_quantize)
        if "tts" in args.preload:
            get_tts_model()

        # Write lockfile for CLI clients
        _write_lockfile(os.getpid(), args.port)

        import uvicorn
        app = create_app(start_audio=True)
        logger.info("Starting voice server at http://%s:%d", args.host, args.port)
        logger.info("  Health:  http://%s:%d/health", args.host, args.port)
        logger.info("  Talk:    http://%s:%d/talk", args.host, args.port)
        logger.info("  WS:      ws://%s:%d/ws", args.host, args.port)
        logger.info("  STT:     http://%s:%d/v1/audio/transcriptions", args.host, args.port)
        logger.info("  TTS:     http://%s:%d/v1/audio/speech", args.host, args.port)

        try:
            uvicorn.run(app, host=args.host, port=args.port, loop="asyncio")
        except KeyboardInterrupt:
            pass
        finally:
            _cleanup()
        return

    # CLI mode: ensure daemon, use TUI
    is_tty = sys.stdin.isatty() and sys.stdout.isatty() and not args.no_tty

    text = args.text
    timeout_val = parse_timeout(args.timeout) if args.timeout else None
    post_timeout = 30.0 if args.loiter else 0
    lookbehind_val = 30.0 if args.loiter else 0

    # Default to continuous; single only when explicitly requested
    mode = args.mode or "continuous"

    # In single mode with piped stdin, consume all of stdin up front
    if text is None and not sys.stdin.isatty() and mode == "single":
        text = sys.stdin.read().strip() or None

    input_mode = args.input or "line"

    # Resolve voice: --voice-file overrides --voice with its absolute path
    voice = args.voice
    if args.voice_file:
        vf = Path(args.voice_file).resolve()
        if not vf.exists():
            print(f"Error: voice file not found: {vf}", file=sys.stderr)
            sys.exit(1)
        voice = str(vf)

    pid, port = ensure_daemon(args.port)

    try:
        if mode == "single":
            asyncio.run(single_mode(port, text, voice, timeout_val,
                                    post_timeout, lookbehind_val,
                                    is_tty=is_tty))
        else:
            asyncio.run(continuous_mode(port, voice, timeout_val,
                                        post_timeout, lookbehind_val,
                                        initial_text=text,
                                        input_mode=input_mode,
                                        is_tty=is_tty))
    except KeyboardInterrupt:
        sys.stdout.write("\n")
        sys.exit(130)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
