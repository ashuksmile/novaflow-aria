"""Local CPU Speech-to-Text powered by faster-whisper (int8).

* English  -> Whisper `base` (fast, ~0.25 s for a 5 s utterance on CPU).
* Hindi    -> Whisper `small` (base writes Hindi in Urdu script and garbles words; small
              produces clean Devanagari). Used when the caller picks Hindi, or when base
              auto-detects Hindi/Urdu.
* Rejects non-speech: segments Whisper itself flags as no-speech, and the classic
  silence hallucinations ("Thank you.", "Thanks for watching") are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Dict, Optional, Tuple, Union

import numpy as np
from faster_whisper import WhisperModel

from app.audio_utils import wav_bytes_to_float32
from app.config import WHISPER_CPU_THREADS, WHISPER_HI_MODEL, WHISPER_MODEL

HALLUCINATIONS = {
    "thank you", "thanks for watching", "thank you for watching", "you", "bye",
    "please subscribe", "subtitles by the amara org community", "so", "okay",
    "धन्यवाद", "शुक्रिया",
}


@dataclass
class STTResult:
    text: str
    language: str
    language_prob: float
    latency_ms: int
    model: str
    rejected_reason: str = ""


def _clean(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


class LocalSTTEngine:
    def __init__(
        self,
        model_size: str = WHISPER_MODEL,
        hindi_model_size: str = WHISPER_HI_MODEL,
        cpu_threads: int = WHISPER_CPU_THREADS,
    ):
        self.model_size = model_size
        self.hindi_model_size = hindi_model_size or model_size
        self.cpu_threads = cpu_threads
        self._models: Dict[str, WhisperModel] = {}

    def _model(self, size: str) -> WhisperModel:
        if size not in self._models:
            self._models[size] = WhisperModel(
                size, device="cpu", compute_type="int8", cpu_threads=self.cpu_threads
            )
        return self._models[size]

    def initialize(self) -> None:
        self._model(self.model_size)
        self._model(self.hindi_model_size)

    def _run(self, size: str, audio: np.ndarray, language: Optional[str]):
        segments, info = self._model(size).transcribe(
            audio,
            language=language,
            # Hindi benefits from beam search (CER 0.29 -> 0.21 on FLEURS); English doesn't need it.
            beam_size=5 if language == "hi" else 1,
            best_of=1,
            temperature=0.0,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 350},
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        kept, dropped = [], 0
        for seg in segments:
            # Whisper's own non-speech rule (same thresholds as openai/whisper).
            if seg.no_speech_prob > 0.6 and seg.avg_logprob < -1.0:
                dropped += 1
                continue
            if seg.avg_logprob < -1.2 and seg.compression_ratio > 2.4:
                dropped += 1  # repetitive garbage
                continue
            if seg.text.strip():
                kept.append(seg.text.strip())
        return " ".join(kept).strip(), info, dropped

    def transcribe_array(self, audio: np.ndarray, language_hint: Optional[str] = "auto") -> STTResult:
        t0 = time.perf_counter()
        if len(audio) < 1600:  # < 0.1 s
            return STTResult("", language_hint if language_hint in ("en", "hi") else "en", 0.0, 0, "-", "too_short")

        hint = (language_hint or "auto").lower()
        if hint == "hi":
            text, info, dropped = self._run(self.hindi_model_size, audio, "hi")
            lang, prob, used = "hi", 1.0, self.hindi_model_size
        elif hint == "en":
            text, info, dropped = self._run(self.model_size, audio, "en")
            lang, prob, used = "en", 1.0, self.model_size
        else:
            text, info, dropped = self._run(self.model_size, audio, None)
            probs = dict(info.all_language_probs or [])
            p_en = probs.get("en", 0.0)
            p_hi = probs.get("hi", 0.0) + probs.get("ur", 0.0)
            lang = "hi" if p_hi > p_en else "en"
            prob = round(max(p_en, p_hi) / max(p_en + p_hi, 1e-6), 3)
            used = self.model_size
            if lang == "hi" and (self.hindi_model_size != self.model_size or info.language != "hi"):
                text, info, dropped = self._run(self.hindi_model_size, audio, "hi")
                used = self.hindi_model_size
            elif lang == "en" and info.language != "en":
                text, info, dropped = self._run(self.model_size, audio, "en")

        latency_ms = int((time.perf_counter() - t0) * 1000)
        reason = ""
        if not text:
            reason = "no_speech" if dropped else "empty"
        elif _clean(text) in HALLUCINATIONS:
            reason, text = "hallucination", ""
        return STTResult(text, lang, prob, latency_ms, f"whisper-{used}-int8", reason)

    def transcribe(
        self, audio: Union[bytes, np.ndarray], language_hint: Optional[str] = None
    ) -> Tuple[str, str, float, int]:
        """Back-compat API: WAV bytes (any rate) or 16 kHz float32 -> (text, lang, prob, ms)."""
        arr = wav_bytes_to_float32(audio) if isinstance(audio, (bytes, bytearray)) else audio
        r = self.transcribe_array(arr, language_hint or "auto")
        return r.text, r.language, r.language_prob, r.latency_ms
