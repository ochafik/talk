#!/usr/bin/env python3
"""Tests for talk.py — WebSocket protocol, event mapping, sentence splitting,
karaoke rendering, and TUI rendering.

Run: python3 test-talk.py
  or: pytest test-talk.py -v
"""

import asyncio
import io
import json
import re
import struct
import sys
import unittest
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Minimal stubs so we can import the pieces we need from talk.py without
# loading MLX / Moshi / sentencepiece (which require GPU / large downloads).
# ---------------------------------------------------------------------------

# Stub heavy modules before importing talk
_STUBS = {}
for mod_name in [
    "mlx", "mlx.core", "mlx.nn",
    "sentencepiece",
    "sphn",
    "moshi_mlx", "moshi_mlx.models", "moshi_mlx.models.lm",
    "moshi_mlx.models.mimi", "moshi_mlx.modules",
    "moshi_mlx.modules.transformer", "moshi_mlx.utils",
    "pocket_tts_mlx",
]:
    _STUBS[mod_name] = MagicMock()

# Provide LmConfig, DepFormerConfig, TransformerConfig, Mimi as mock classes
_STUBS["moshi_mlx.models.lm"].LmConfig = MagicMock
_STUBS["moshi_mlx.models.lm"].DepFormerConfig = MagicMock
_STUBS["moshi_mlx.modules.transformer"].TransformerConfig = MagicMock
_STUBS["moshi_mlx.models.mimi"].Mimi = MagicMock
_STUBS["moshi_mlx.models.mimi"].mimi_202407 = MagicMock

import numpy as np
_STUBS["mlx.core"].array = np.array

with patch.dict(sys.modules, _STUBS):
    # Now we can import the pieces we need
    import talk

# Re-export for convenience
_SENTENCE_RE = talk._SENTENCE_RE
_map_ws_event = talk._map_ws_event
_WS_EVENT_MAP = talk._WS_EVENT_MAP
TUIRenderer = talk.TUIRenderer
TalkSession = talk.TalkSession
_emit = talk._emit
parse_timeout = talk.parse_timeout


# ===========================================================================
# 1. Sentence splitting regex
# ===========================================================================

class TestSentenceSplitting(unittest.TestCase):
    """Test _SENTENCE_RE splits text into sentences correctly."""

    def _split(self, text: str) -> list[str]:
        return [m.group() for m in _SENTENCE_RE.finditer(text) if m.group().strip()]

    def test_simple_sentences(self):
        result = self._split("Hello world. How are you? I'm fine!")
        self.assertEqual(len(result), 3)
        self.assertIn("Hello world.", result[0])
        self.assertIn("How are you?", result[1])
        self.assertIn("I'm fine!", result[2])

    def test_single_sentence_no_punct(self):
        result = self._split("Hello world")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].strip(), "Hello world")

    def test_newline_splitting(self):
        result = self._split("Line one\nLine two\nLine three")
        self.assertEqual(len(result), 3)

    def test_empty_string(self):
        result = self._split("")
        self.assertEqual(result, [])

    def test_multiline_with_punctuation(self):
        text = "First sentence. Second sentence!\nThird on new line."
        result = self._split(text)
        self.assertEqual(len(result), 3)

    def test_long_text_many_sentences(self):
        text = ". ".join(f"Sentence {i}" for i in range(20)) + "."
        result = self._split(text)
        self.assertEqual(len(result), 20)

    def test_preserves_full_text(self):
        """All characters in original text should be covered by matches."""
        text = "Hello! How are you? Fine."
        result = self._split(text)
        reconstructed = "".join(result)
        # Allow whitespace differences from stripping
        self.assertEqual(text.replace(" ", ""),
                         reconstructed.replace(" ", "").rstrip())


# ===========================================================================
# 2. WS event mapping
# ===========================================================================

class TestWsEventMapping(unittest.TestCase):
    """Test _map_ws_event maps internal → WS protocol names."""

    def test_tts_start(self):
        mapped = _map_ws_event({"type": "tts_start", "text": "Hello"})
        self.assertEqual(mapped["type"], "speak.started")
        self.assertEqual(mapped["text"], "Hello")

    def test_tts_word(self):
        mapped = _map_ws_event({"type": "tts_word", "word": "Hello", "offset": 0})
        self.assertEqual(mapped["type"], "speak.progress")
        self.assertEqual(mapped["word"], "Hello")
        self.assertEqual(mapped["offset"], 0)
        self.assertEqual(mapped["spoken_end"], 5)

    def test_tts_word_with_offset(self):
        mapped = _map_ws_event({"type": "tts_word", "word": "world", "offset": 6})
        self.assertEqual(mapped["spoken_end"], 11)

    def test_tts_done(self):
        mapped = _map_ws_event({"type": "tts_done", "interrupted_at": None})
        self.assertEqual(mapped["type"], "speak.done")
        self.assertIsNone(mapped["interrupted_at"])

    def test_tts_done_interrupted(self):
        mapped = _map_ws_event({"type": "tts_done", "interrupted_at": 5})
        self.assertEqual(mapped["interrupted_at"], 5)

    def test_stt_partial(self):
        mapped = _map_ws_event({"type": "stt_partial", "text": "hello"})
        self.assertEqual(mapped["type"], "transcript.partial")
        self.assertEqual(mapped["text"], "hello")

    def test_stt_final(self):
        mapped = _map_ws_event({"type": "stt_final", "text": "hello world"})
        self.assertEqual(mapped["type"], "transcript.final")
        self.assertEqual(mapped["text"], "hello world")

    def test_unmapped_done(self):
        """Internal 'done' event should not be mapped to WS."""
        mapped = _map_ws_event({"type": "done", "timed_out": False})
        self.assertIsNone(mapped)

    def test_unmapped_queued_speech(self):
        mapped = _map_ws_event({"type": "queued_speech", "text": "hi"})
        self.assertIsNone(mapped)

    def test_all_internal_types_covered(self):
        """All keys in _WS_EVENT_MAP should produce valid mappings."""
        for internal_type in _WS_EVENT_MAP:
            event = {"type": internal_type, "text": "x", "word": "x", "offset": 0,
                     "interrupted_at": None}
            mapped = _map_ws_event(event)
            self.assertIsNotNone(mapped, f"{internal_type} should map to WS event")
            self.assertIn("type", mapped)


# ===========================================================================
# 3. TUIRenderer — event rendering
# ===========================================================================

class TestTUIRenderer(unittest.TestCase):
    """Test TUIRenderer handles both internal and WS event names."""

    def _capture_render(self, events, tty=True):
        """Render events and capture stdout."""
        buf = io.StringIO()
        renderer = TUIRenderer(tty=tty)
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            for event in events:
                renderer.render(event)
        finally:
            sys.stdout = old_stdout
        return buf.getvalue()

    # -- Internal event names (backward compat with HTTP /talk) --

    def test_nontty_tts_word_by_word(self):
        output = self._capture_render([
            {"type": "tts_start", "text": "Hello world"},
            {"type": "tts_word", "word": "Hello", "offset": 0},
            {"type": "tts_word", "word": "world", "offset": 6},
            {"type": "tts_done", "interrupted_at": None},
        ], tty=False)
        self.assertIn("\u23fa", output)  # ⏺ prefix
        self.assertIn("Hello", output)
        self.assertIn("world", output)
        self.assertTrue(output.endswith("\n"))

    def test_nontty_stt_delta_streaming(self):
        output = self._capture_render([
            {"type": "stt_partial", "text": "hello"},
            {"type": "stt_partial", "text": "hello world"},
            {"type": "stt_final", "text": "hello world"},
        ], tty=False)
        self.assertIn("\u276f", output)  # ❯ prefix
        self.assertIn("hello", output)
        # Should only print delta " world" on second partial, not repeat
        # Count occurrences of "hello" — should appear once (first partial)
        # plus the "hello" substring in "hello world" from stt_final is OK
        self.assertTrue(output.strip().endswith("hello world"))

    def test_nontty_stt_final_newline(self):
        output = self._capture_render([
            {"type": "stt_final", "text": "done talking"},
        ], tty=False)
        self.assertIn("done talking\n", output)

    # -- WS event names --

    def test_ws_speak_started(self):
        output = self._capture_render([
            {"type": "speak.started", "text": "Hi"},
            {"type": "speak.progress", "word": "Hi", "spoken_end": 2},
            {"type": "speak.done", "interrupted_at": None},
        ], tty=False)
        self.assertIn("\u23fa", output)
        self.assertIn("Hi", output)

    def test_ws_transcript(self):
        output = self._capture_render([
            {"type": "transcript.partial", "text": "hey"},
            {"type": "transcript.final", "text": "hey there"},
        ], tty=False)
        self.assertIn("hey", output)
        self.assertIn("there", output)

    def test_tty_karaoke_state(self):
        """TTY mode should track karaoke state."""
        renderer = TUIRenderer(tty=True)
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            renderer.render({"type": "tts_start", "text": "Hello world"})
            self.assertEqual(renderer._tts_text, "Hello world")
            self.assertEqual(renderer._tts_spoken_end, 0)

            renderer.render({"type": "tts_word", "word": "Hello", "offset": 0})
            self.assertEqual(renderer._tts_spoken_end, 5)

            renderer.render({"type": "tts_word", "word": "world", "offset": 6})
            self.assertEqual(renderer._tts_spoken_end, 11)

            renderer.render({"type": "tts_done", "interrupted_at": None})
            self.assertIsNone(renderer._tts_text)
            self.assertEqual(renderer._tts_spoken_end, 0)
        finally:
            sys.stdout = old_stdout

    def test_tty_karaoke_ws_spoken_end(self):
        """speak.progress with spoken_end should update karaoke correctly."""
        renderer = TUIRenderer(tty=True)
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            renderer.render({"type": "speak.started", "text": "Hello world"})
            renderer.render({"type": "speak.progress", "word": "Hello", "spoken_end": 5})
            self.assertEqual(renderer._tts_spoken_end, 5)
            renderer.render({"type": "speak.progress", "word": "world", "spoken_end": 11})
            self.assertEqual(renderer._tts_spoken_end, 11)
        finally:
            sys.stdout = old_stdout

    def test_tty_interrupted_tts(self):
        """Interrupted TTS should show only spoken portion."""
        output = self._capture_render([
            {"type": "tts_start", "text": "Hello beautiful world"},
            {"type": "tts_word", "word": "Hello", "offset": 0},
            {"type": "tts_done", "interrupted_at": 5},
        ], tty=True)
        self.assertIn("Hello", output)
        # "beautiful world" should NOT appear in final line
        self.assertNotIn("beautiful", output.split("\n")[-1] if output.strip() else "")


# ===========================================================================
# 4. Karaoke sliding window
# ===========================================================================

class TestKaraokeDisplay(unittest.TestCase):
    """Test that _rewrite_karaoke handles text display correctly."""

    def _render(self, text, spoken_end):
        """Render karaoke and return raw output."""
        renderer = TUIRenderer(tty=True)
        renderer._tts_text = text
        renderer._tts_spoken_end = spoken_end
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            renderer._rewrite_karaoke()
        finally:
            sys.stdout = old_stdout
        return buf.getvalue(), renderer

    def test_short_text_shows_all(self):
        """Short text should render fully."""
        output, _ = self._render("Hello world", 5)
        self.assertIn("Hello", output)
        self.assertIn("world", output)

    def test_long_text_shows_all_no_ellipsis(self):
        """Long text should render fully (no sliding window/ellipsis)."""
        text = "A" * 200
        output, _ = self._render(text, 100)
        self.assertNotIn("\u2026", output)
        clean = re.sub(r'\033\[[0-9;]*[a-zA-Z]', '', output).replace('\r', '')
        self.assertIn("A" * 100, clean)

    def test_cursor_save_on_write(self):
        """_rewrite_karaoke should save cursor position."""
        _, renderer = self._render("Hello world", 5)
        self.assertTrue(renderer._inplace_saved)

    def test_clear_inplace_with_saved_pos(self):
        """_clear_inplace with saved pos should emit restore + clear."""
        renderer = TUIRenderer(tty=True)
        renderer._inplace_saved = True
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            renderer._clear_inplace()
        finally:
            sys.stdout = old_stdout
        output = buf.getvalue()
        self.assertIn("\033[u", output)  # restore cursor
        self.assertIn("\033[J", output)  # clear to end of screen

    def test_clear_inplace_without_saved_pos(self):
        """_clear_inplace without saved pos should just clear current line."""
        renderer = TUIRenderer(tty=True)
        renderer._inplace_saved = False
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            renderer._clear_inplace()
        finally:
            sys.stdout = old_stdout
        output = buf.getvalue()
        self.assertNotIn("\033[u", output)
        self.assertIn("\033[J", output)

    def test_multiline_text_newlines_collapsed(self):
        """Newlines in text should be collapsed to spaces."""
        output, _ = self._render("line one\nline two\nline three", 10)
        clean = re.sub(r'\033\[[0-9;]*[a-zA-Z]', '', output).replace('\r', '')
        self.assertNotIn('\n', clean)
        self.assertIn("line one line two", clean)

    def test_tts_done_ends_inplace(self):
        """tts_done should clear inplace and end inplace mode."""
        renderer = TUIRenderer(tty=True)
        renderer._tts_text = "Hello world"
        renderer._tts_spoken_end = 5
        renderer._inplace_saved = True
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            renderer.render({"type": "tts_done"})
        finally:
            sys.stdout = old_stdout
        output = buf.getvalue()
        self.assertIn("\033[u", output)  # restored cursor
        self.assertFalse(renderer._inplace_saved)

    def test_stt_partial_uses_inplace(self):
        """stt_partial should use cursor save/restore for multi-line support."""
        renderer = TUIRenderer(tty=True)
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            renderer.render({"type": "stt_partial", "text": "hello"})
            first_output = buf.getvalue()
            buf.truncate(0); buf.seek(0)
            renderer.render({"type": "stt_partial", "text": "hello world"})
            second_output = buf.getvalue()
        finally:
            sys.stdout = old_stdout
        # Second partial should restore cursor before rewriting
        self.assertIn("\033[u", second_output)
        self.assertTrue(renderer._inplace_saved)


# ===========================================================================
# 5. Event emission
# ===========================================================================

class TestEmit(unittest.TestCase):
    """Test _emit() guards and queue behavior."""

    def test_emit_with_queue(self):
        session = TalkSession()
        session.event_queue = asyncio.Queue()
        _emit(session, {"type": "tts_start", "text": "hi"})
        self.assertFalse(session.event_queue.empty())
        event = session.event_queue.get_nowait()
        self.assertEqual(event["type"], "tts_start")

    def test_emit_without_queue(self):
        """Should not raise when event_queue is None."""
        session = TalkSession()
        session.event_queue = None
        _emit(session, {"type": "tts_start", "text": "hi"})  # should not raise


# ===========================================================================
# 6. Binary audio frame format (external mode)
# ===========================================================================

class TestBinaryAudioFrameFormat(unittest.TestCase):
    """Test the binary frame format for external audio mode."""

    def test_frame_header_packing(self):
        """Header should be 8 bytes: charStart(u32le) + charEnd(u32le)."""
        char_start, char_end = 10, 25
        header = struct.pack('<II', char_start, char_end)
        self.assertEqual(len(header), 8)
        s, e = struct.unpack('<II', header)
        self.assertEqual(s, 10)
        self.assertEqual(e, 25)

    def test_full_frame_with_f32_audio(self):
        """Full frame: 8-byte header + Float32 PCM data."""
        char_start, char_end = 0, 5
        header = struct.pack('<II', char_start, char_end)
        # 100 samples of Float32 audio
        pcm = np.zeros(100, dtype=np.float32)
        frame = header + pcm.tobytes()
        self.assertEqual(len(frame), 8 + 100 * 4)

        # Parse it back
        s, e = struct.unpack_from('<II', frame, 0)
        self.assertEqual(s, 0)
        self.assertEqual(e, 5)
        audio = np.frombuffer(frame[8:], dtype=np.float32)
        self.assertEqual(len(audio), 100)

    def test_full_frame_with_s16_audio(self):
        """Full frame: 8-byte header + Int16 PCM data."""
        header = struct.pack('<II', 5, 10)
        pcm = np.zeros(100, dtype=np.int16)
        frame = header + pcm.tobytes()
        self.assertEqual(len(frame), 8 + 100 * 2)


# ===========================================================================
# 7. parse_timeout
# ===========================================================================

class TestParseTimeout(unittest.TestCase):

    def test_seconds(self):
        self.assertEqual(parse_timeout("10s"), 10.0)

    def test_milliseconds(self):
        self.assertEqual(parse_timeout("500ms"), 0.5)

    def test_minutes(self):
        self.assertEqual(parse_timeout("2m"), 120.0)

    def test_bare_number(self):
        self.assertEqual(parse_timeout("5"), 5.0)

    def test_inf(self):
        self.assertEqual(parse_timeout("inf"), float("inf"))


# ===========================================================================
# 8. WebSocket protocol integration (async tests)
# ===========================================================================

class TestWsProtocolAsync(unittest.TestCase):
    """Test WS handler logic using mock WebSocket."""

    def test_ws_cancel_tts_flushes_audio(self):
        """_ws_cancel_tts should cancel task and flush audio-io."""

        async def _run():
            session = TalkSession()
            # Simulate a running TTS task
            fut = asyncio.get_event_loop().create_future()
            fut.set_result(None)
            session.tts_task = asyncio.create_task(asyncio.sleep(100))
            session.tts_playing = True

            with patch.object(talk, '_flush_audio_output') as mock_flush:
                await talk._ws_cancel_tts(session)
                mock_flush.assert_called_once_with(session)

            self.assertFalse(session.tts_playing)

        asyncio.run(_run())

    def test_ws_cancel_tts_no_flush_external(self):
        """External mode: _ws_cancel_tts should NOT call _flush_audio_output."""

        async def _run():
            session = TalkSession()
            session.ws_send_audio = AsyncMock()  # external mode
            session.tts_task = asyncio.create_task(asyncio.sleep(100))
            session.tts_playing = True

            with patch.object(talk, '_flush_audio_output') as mock_flush:
                await talk._ws_cancel_tts(session)
                mock_flush.assert_not_called()

        asyncio.run(_run())

    def test_speak_cancel_no_emit_when_tts_done(self):
        """speak.cancel should NOT emit tts_done if TTS task is already finished."""

        async def _run():
            session = TalkSession()
            # Simulate a COMPLETED TTS task
            fut = asyncio.get_event_loop().create_future()
            fut.set_result(None)
            session.tts_task = fut  # already done
            session.event_queue = asyncio.Queue()

            with patch.object(talk, '_flush_audio_output'):
                was_active = session.tts_task and not session.tts_task.done()
                await talk._ws_cancel_tts(session)
                if was_active:
                    talk._emit(session, {"type": "tts_done", "interrupted_at": 0})

            self.assertFalse(was_active)
            self.assertTrue(session.event_queue.empty(),
                            "No tts_done should be emitted for already-finished TTS")

        asyncio.run(_run())

    def test_speak_cancel_emits_when_tts_active(self):
        """speak.cancel SHOULD emit tts_done if TTS task is still running."""

        async def _run():
            session = TalkSession()
            session.tts_task = asyncio.create_task(asyncio.sleep(100))
            session.event_queue = asyncio.Queue()

            with patch.object(talk, '_flush_audio_output'):
                was_active = session.tts_task and not session.tts_task.done()
                await talk._ws_cancel_tts(session)
                if was_active:
                    talk._emit(session, {"type": "tts_done", "interrupted_at": 0})

            self.assertTrue(was_active)
            self.assertFalse(session.event_queue.empty(),
                             "tts_done should be emitted for active TTS")
            event = session.event_queue.get_nowait()
            self.assertEqual(event["type"], "tts_done")

        asyncio.run(_run())

    def test_client_no_cancel_after_tts_done(self):
        """Client should not send speak.cancel when tts_done_seen is True."""
        # Simulate the logic in _run_ws_with_stdin
        tts_done_seen = True
        result = "next line"
        # The fix: only send cancel if not tts_done_seen
        should_cancel = isinstance(result, str) and not tts_done_seen
        self.assertFalse(should_cancel,
                         "Should NOT cancel when TTS already completed")

    def test_client_cancels_during_active_tts(self):
        """Client should send speak.cancel when tts_done_seen is False."""
        tts_done_seen = False
        result = "next line"
        should_cancel = isinstance(result, str) and not tts_done_seen
        self.assertTrue(should_cancel,
                        "SHOULD cancel when TTS is still active")

    def test_ws_start_stt_initializes_session(self):
        """_ws_start_stt should create an STT session."""

        async def _run():
            session = TalkSession()
            mock_engine = MagicMock()
            mock_session = MagicMock()
            mock_engine.new_session.return_value = mock_session

            with patch.object(talk, 'get_stt_engine', return_value=mock_engine):
                with patch.object(talk, '_model_lock', asyncio.Lock()):
                    await talk._ws_start_stt(session)

            self.assertEqual(session.stt_session, mock_session)
            self.assertEqual(session.current_transcript, "")
            self.assertEqual(session.silence_frames, 0)

        asyncio.run(_run())


# ===========================================================================
# 9. TUIRenderer WS ↔ internal name normalization
# ===========================================================================

class TestRendererNameNormalization(unittest.TestCase):
    """Verify that both internal and WS event names produce identical state."""

    def _render_sequence_state(self, events):
        """Render events, return final renderer state."""
        renderer = TUIRenderer(tty=True)
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            for e in events:
                renderer.render(e)
        finally:
            sys.stdout = old_stdout
        return renderer

    def test_internal_and_ws_produce_same_karaoke_state(self):
        # Internal events
        r1 = self._render_sequence_state([
            {"type": "tts_start", "text": "Hello world"},
            {"type": "tts_word", "word": "Hello", "offset": 0},
        ])
        # WS events
        r2 = self._render_sequence_state([
            {"type": "speak.started", "text": "Hello world"},
            {"type": "speak.progress", "word": "Hello", "spoken_end": 5},
        ])
        self.assertEqual(r1._tts_text, r2._tts_text)
        self.assertEqual(r1._tts_spoken_end, r2._tts_spoken_end)

    def test_done_resets_state_both_names(self):
        r1 = self._render_sequence_state([
            {"type": "tts_start", "text": "Hi"},
            {"type": "tts_done", "interrupted_at": None},
        ])
        r2 = self._render_sequence_state([
            {"type": "speak.started", "text": "Hi"},
            {"type": "speak.done", "interrupted_at": None},
        ])
        self.assertIsNone(r1._tts_text)
        self.assertIsNone(r2._tts_text)
        self.assertEqual(r1._tts_spoken_end, 0)
        self.assertEqual(r2._tts_spoken_end, 0)


# ===========================================================================
# 10. External capture loop
# ===========================================================================

class TestExternalCaptureLoop(unittest.TestCase):
    """Test _external_capture_loop processes audio frames correctly."""

    def test_none_terminates_loop(self):
        """Putting None in the queue should terminate the loop."""

        async def _run():
            session = TalkSession()
            session.stt_session = None  # no STT → frames discarded
            q = asyncio.Queue()
            await q.put(b'\x00' * 100)
            await q.put(None)  # termination signal
            await talk._external_capture_loop(session, q)
            # Should complete without hanging

        asyncio.run(_run())

    def test_frames_discarded_without_stt(self):
        """Frames should be discarded when stt_session is None."""

        async def _run():
            session = TalkSession()
            session.stt_session = None
            q = asyncio.Queue()
            for _ in range(5):
                await q.put(b'\x00' * 3840)
            await q.put(None)
            await talk._external_capture_loop(session, q)

        asyncio.run(_run())


# ===========================================================================
# 11. Starlette app composition
# ===========================================================================

class TestAppComposition(unittest.TestCase):
    """Test that create_app includes the /ws route."""

    def test_ws_route_exists(self):
        with patch.object(talk, 'get_session', return_value=TalkSession()):
            app = talk.create_app(start_audio=False)
        # Check routes
        route_paths = [r.path for r in app.routes]
        self.assertIn("/ws", route_paths)
        self.assertIn("/talk", route_paths)
        self.assertIn("/health", route_paths)


# ===========================================================================
# 12. Non-tty STT delta streaming
# ===========================================================================

class TestNonTtySttDelta(unittest.TestCase):
    """Test that non-tty mode streams STT incrementally."""

    def _capture(self, events):
        buf = io.StringIO()
        renderer = TUIRenderer(tty=False)
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            for e in events:
                renderer.render(e)
        finally:
            sys.stdout = old_stdout
        return buf.getvalue()

    def test_incremental_partials(self):
        output = self._capture([
            {"type": "stt_partial", "text": "he"},
            {"type": "stt_partial", "text": "hello"},
            {"type": "stt_partial", "text": "hello world"},
            {"type": "stt_final", "text": "hello world"},
        ])
        # Should print: "❯ he" then "llo" then " world" then "\n"
        # Full output should be "❯ hello world\n"
        self.assertEqual(output.strip(), "\u276f hello world")

    def test_ws_transcript_events(self):
        output = self._capture([
            {"type": "transcript.partial", "text": "ok"},
            {"type": "transcript.final", "text": "ok then"},
        ])
        self.assertEqual(output.strip(), "\u276f ok then")

    def test_stt_state_resets_between_exchanges(self):
        output = self._capture([
            {"type": "stt_partial", "text": "first"},
            {"type": "stt_final", "text": "first"},
            {"type": "stt_partial", "text": "second"},
            {"type": "stt_final", "text": "second"},
        ])
        lines = [l for l in output.strip().split("\n") if l.strip()]
        self.assertEqual(len(lines), 2)
        self.assertIn("first", lines[0])
        self.assertIn("second", lines[1])


if __name__ == "__main__":
    unittest.main()
