"""Audio helpers shared by the voice pipeline and the test-suite.

Everything here is pure NumPy so it runs on CPU without ffmpeg.
"""

from __future__ import annotations

import io
import re
import unicodedata
import wave
from typing import List, Tuple

import numpy as np

STT_SAMPLE_RATE = 16000


# --------------------------------------------------------------------------- #
# PCM / WAV conversion
# --------------------------------------------------------------------------- #
def pcm16_bytes_to_float32(data: bytes) -> np.ndarray:
    """Little-endian signed 16-bit PCM -> float32 in [-1, 1]."""
    usable = len(data) - (len(data) % 2)
    return np.frombuffer(data[:usable], dtype="<i2").astype(np.float32) / 32768.0


def float32_to_pcm16_bytes(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Band-limited-enough linear resampler (box low-pass when downsampling)."""
    if src_rate == dst_rate or len(audio) == 0:
        return audio.astype(np.float32, copy=False)
    if dst_rate < src_rate:
        # Simple moving-average anti-alias filter sized to the decimation ratio.
        width = max(1, int(round(src_rate / dst_rate)))
        if width > 1:
            kernel = np.ones(width, dtype=np.float32) / width
            audio = np.convolve(audio, kernel, mode="same")
    duration = len(audio) / float(src_rate)
    target_len = int(round(duration * dst_rate))
    x_old = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, duration, num=target_len, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def wav_bytes_to_float32(wav_bytes: bytes, target_rate: int = STT_SAMPLE_RATE) -> np.ndarray:
    """Decode a PCM WAV file (any rate / channel count) into mono float32 at target_rate."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    if sample_width == 2:
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sample_width == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return resample(audio, sample_rate, target_rate)


def float32_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(float32_to_pcm16_bytes(audio))
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Signal-level speech sanity checks (used by tests + /api/diagnostics)
# --------------------------------------------------------------------------- #
def rms_dbfs(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
    return 20.0 * np.log10(max(rms, 1e-9))


def spectral_flatness(audio: np.ndarray, frame: int = 512) -> float:
    """Median spectral flatness over frames. White noise ~0.5+, voiced speech << 0.3."""
    if len(audio) < frame:
        return 1.0
    n = len(audio) // frame
    frames = audio[: n * frame].reshape(n, frame) * np.hanning(frame)
    energy = np.mean(frames**2, axis=1)
    # Only score frames that actually carry signal.
    keep = energy > (np.max(energy) * 1e-3)
    if not np.any(keep):
        return 1.0
    spec = np.abs(np.fft.rfft(frames[keep], axis=1)) ** 2 + 1e-12
    geo = np.exp(np.mean(np.log(spec), axis=1))
    arith = np.mean(spec, axis=1)
    return float(np.median(geo / arith))


def clipping_ratio(audio: np.ndarray, threshold: float = 0.999) -> float:
    return float(np.mean(np.abs(audio) >= threshold)) if len(audio) else 0.0


# --------------------------------------------------------------------------- #
# Transcript scoring (WER / CER) with light normalisation
# --------------------------------------------------------------------------- #
_NUM_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "fifteen": "15", "twenty": "20", "thirty": "30",
    "mister": "mr", "doctor": "dr",
}


def normalize_for_scoring(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = text.replace("ँ", "ं").replace("़", "")  # chandrabindu/nukta spelling variants
    text = re.sub(r"\$\s?(\d[\d,]*)", r"\1 dollars", text)
    text = re.sub(r"(\d),(\d)", r"\1\2", text)
    text = re.sub(r"(\d)\s?%", r"\1 percent", text)
    text = re.sub(r"(\d+)[:.](\d\d)\s?(am|pm|a\.m\.|p\.m\.)", r"\1 \2 \3", text)
    text = text.replace("a.m.", "am").replace("p.m.", "pm")
    text = re.sub(r"[^\w\s]", " ", text)  # drop punctuation incl. । and ,
    words = [_NUM_WORDS.get(w, w) for w in text.split()]
    return " ".join(words)


def _edit_distance(ref: List[str], hyp: List[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = normalize_for_scoring(reference).split()
    hyp = normalize_for_scoring(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / len(ref)


def char_error_rate(reference: str, hypothesis: str) -> float:
    ref = list(normalize_for_scoring(reference).replace(" ", ""))
    hyp = list(normalize_for_scoring(hypothesis).replace(" ", ""))
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / len(ref)


def chunk_pcm(audio_int16: np.ndarray, sample_rate: int, frame_ms: int) -> List[Tuple[bytes, float]]:
    """Split int16 audio into (bytes, duration_sec) frames for paced streaming."""
    step = max(1, int(sample_rate * frame_ms / 1000))
    out = []
    for i in range(0, len(audio_int16), step):
        part = audio_int16[i : i + step]
        out.append((part.astype("<i2").tobytes(), len(part) / float(sample_rate)))
    return out
