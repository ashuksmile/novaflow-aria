"""Streaming voice-activity detection + end-of-turn detection (Silero VAD v6, ONNX, CPU).

The model ships inside faster-whisper, so there is no extra download. Timing is driven by
*samples*, not wall-clock, which makes the endpointing deterministic and testable.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

import numpy as np
import onnxruntime
from faster_whisper.utils import get_assets_path

SAMPLE_RATE = 16000
WINDOW = 512  # 32 ms @ 16 kHz (fixed by Silero)
CONTEXT = 64
WINDOW_MS = WINDOW * 1000 // SAMPLE_RATE

_SESSION: Optional[onnxruntime.InferenceSession] = None


def _session() -> onnxruntime.InferenceSession:
    global _SESSION
    if _SESSION is None:
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        _SESSION = onnxruntime.InferenceSession(
            f"{get_assets_path()}/silero_vad_v6.onnx",
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
    return _SESSION


@dataclass
class VADEvent:
    kind: str  # "speech_start" | "speech_end"
    at_sec: float  # stream time (seconds of audio received) when the event fired
    audio: Optional[np.ndarray] = None  # full utterance for speech_end


@dataclass
class StreamingVAD:
    """Turns a continuous 16 kHz mic stream into utterances.

    speech_start fires after `min_speech_ms` of continuous speech (raise it while the agent is
    talking so speaker echo / coughs don't trigger a barge-in). speech_end fires after
    `min_silence_ms` of silence — that silence is the end-of-turn detector.
    """

    threshold: float = 0.5
    neg_threshold: float = 0.35
    min_speech_ms: int = 160
    min_silence_ms: int = 650
    preroll_ms: int = 320
    max_utterance_s: float = 25.0

    last_prob: float = 0.0
    triggered: bool = False
    _h: np.ndarray = field(default_factory=lambda: np.zeros((1, 1, 128), np.float32))
    _c: np.ndarray = field(default_factory=lambda: np.zeros((1, 1, 128), np.float32))
    _ctx: np.ndarray = field(default_factory=lambda: np.zeros(CONTEXT, np.float32))
    _pending: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    _preroll: Deque[np.ndarray] = field(default_factory=deque)
    _utt: List[np.ndarray] = field(default_factory=list)
    _speech_run_ms: int = 0
    _silence_run_ms: int = 0
    _samples_seen: int = 0

    def reset(self, keep_model_state: bool = True) -> None:
        self.triggered = False
        self._preroll.clear()
        self._utt = []
        self._speech_run_ms = 0
        self._silence_run_ms = 0
        if not keep_model_state:
            self._h[:] = 0
            self._c[:] = 0
            self._ctx[:] = 0

    @property
    def stream_time(self) -> float:
        return self._samples_seen / SAMPLE_RATE

    def _prob(self, window: np.ndarray) -> float:
        x = np.concatenate([self._ctx, window])[None, :].astype(np.float32)
        out, self._h, self._c = _session().run(None, {"input": x, "h": self._h, "c": self._c})
        self._ctx = window[-CONTEXT:].copy()
        return float(np.asarray(out).reshape(-1)[0])

    def process(self, audio: np.ndarray) -> List[VADEvent]:
        """Feed float32 mono 16 kHz samples (any length). Returns events in order."""
        events: List[VADEvent] = []
        buf = np.concatenate([self._pending, audio.astype(np.float32, copy=False)])
        n_windows = len(buf) // WINDOW
        self._pending = buf[n_windows * WINDOW :]
        preroll_windows = max(1, self.preroll_ms // WINDOW_MS)

        for i in range(n_windows):
            win = buf[i * WINDOW : (i + 1) * WINDOW]
            self._samples_seen += WINDOW
            prob = self._prob(win)
            self.last_prob = prob

            if not self.triggered:
                self._preroll.append(win)
                while len(self._preroll) > preroll_windows + self.min_speech_ms // WINDOW_MS:
                    self._preroll.popleft()
                self._speech_run_ms = self._speech_run_ms + WINDOW_MS if prob >= self.threshold else 0
                if self._speech_run_ms >= self.min_speech_ms:
                    self.triggered = True
                    self._silence_run_ms = 0
                    self._utt = list(self._preroll)
                    self._preroll.clear()
                    events.append(VADEvent("speech_start", self.stream_time))
                continue

            self._utt.append(win)
            if prob < self.neg_threshold:
                self._silence_run_ms += WINDOW_MS
            elif prob >= self.threshold:
                self._silence_run_ms = 0

            utt_sec = len(self._utt) * WINDOW / SAMPLE_RATE
            if self._silence_run_ms >= self.min_silence_ms or utt_sec >= self.max_utterance_s:
                # Keep ~200 ms of the trailing silence; Whisper likes a little padding.
                trim = max(0, (self._silence_run_ms - 200) // WINDOW_MS)
                kept = self._utt[: len(self._utt) - trim] if trim else self._utt
                events.append(VADEvent("speech_end", self.stream_time, np.concatenate(kept)))
                self.reset()
        return events


def speech_ratio(audio_16k: np.ndarray, threshold: float = 0.5) -> float:
    """Fraction of 32 ms windows Silero classifies as speech (offline helper for diagnostics)."""
    vad = StreamingVAD(threshold=threshold)
    n = len(audio_16k) // WINDOW
    if n == 0:
        return 0.0
    hits = 0
    for i in range(n):
        vad.process(audio_16k[i * WINDOW : (i + 1) * WINDOW])
        hits += vad.last_prob >= threshold
    return hits / n
