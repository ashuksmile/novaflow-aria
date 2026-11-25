"""Local CPU Text-to-Speech powered by Piper ONNX (English + Hindi).

`normalize_for_speech` rewrites symbols the voices mispronounce ($799 was read as
"dollar one point seven nine nine"; English product names inside Hindi text were garbled)
before synthesis.
"""

from __future__ import annotations

import io
import re
import time
import wave
from typing import Dict, List, Tuple

import numpy as np
from piper import PiperVoice

from app.config import PIPER_EN_MODEL, PIPER_HI_MODEL

DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।])\s+")

# English terms the LLM keeps in Latin script inside Hindi replies.
HINDI_TERMS = {
    "novaflow": "नोवाफ़्लो", "ai": "ए आई", "crm": "सी आर एम", "demo": "डेमो",
    "plan": "प्लान", "growth": "ग्रोथ", "starter": "स्टार्टर", "enterprise": "एंटरप्राइज़",
    "salesforce": "सेल्सफ़ोर्स", "hubspot": "हबस्पॉट", "zoho": "ज़ोहो", "email": "ईमेल",
    "soc": "सॉक", "gdpr": "जी डी पी आर", "dpdp": "डी पी डी पी", "roi": "आर ओ आई",
    "api": "ए पी आई", "ist": "आई एस टी", "am": "ए एम", "pm": "पी एम", "sales": "सेल्स",
    "team": "टीम", "lead": "लीड", "leads": "लीड्स", "calendar": "कैलेंडर", "slot": "स्लॉट",
    "minutes": "मिनट", "reps": "रेप्स", "voice": "वॉइस", "agent": "एजेंट", "call": "कॉल",
    "pricing": "प्राइसिंग", "integration": "इंटीग्रेशन", "dashboard": "डैशबोर्ड",
    "whatsapp": "व्हाट्सऐप", "type": "टाइप", "aria": "आर्या", "ok": "ओके", "okay": "ओके",
}
HINDI_LETTERS = {
    "a": "ए", "b": "बी", "c": "सी", "d": "डी", "e": "ई", "f": "एफ़", "g": "जी", "h": "एच",
    "i": "आई", "j": "जे", "k": "के", "l": "एल", "m": "एम", "n": "एन", "o": "ओ", "p": "पी",
    "q": "क्यू", "r": "आर", "s": "एस", "t": "टी", "u": "यू", "v": "वी", "w": "डब्ल्यू",
    "x": "एक्स", "y": "वाई", "z": "ज़ेड",
}


def clean_for_speech(text: str) -> str:
    """Strip markdown syntax so TTS sounds natural."""
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    text = re.sub(r"[*_`#>|]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_for_speech(text: str, language: str = "en") -> str:
    text = clean_for_speech(text)
    hindi = language == "hi" or bool(DEVANAGARI_RE.search(text))
    dollars, rupees, percent, per, and_ = (
        (" डॉलर", " रुपये", " प्रतिशत", " प्रति ", " और ") if hindi
        else (" dollars", " rupees", " percent", " per ", " and ")
    )
    text = re.sub(r"(\d),(?=\d{2,3}\b)", r"\1", text)  # 1,299 / 1,00,000 -> plain digits
    text = re.sub(r"(\d),(?=\d)", r"\1", text)
    text = re.sub(r"\$\s?(\d+(?:\.\d+)?)\s?([kK])\b", r"\1 thousand" + dollars, text)
    text = re.sub(r"\$\s?(\d+(?:\.\d+)?)", r"\1" + dollars, text)
    text = re.sub(r"(?:₹|\bRs\.?|\bINR)\s?(\d+(?:\.\d+)?)", r"\1" + rupees, text)
    text = re.sub(r"(\d)\s?%", r"\1" + percent, text)
    text = re.sub(r"(dollars|rupees|डॉलर|रुपये)\s?/\s?", r"\1" + per, text)
    text = re.sub(r"\b24/7\b", "24 7", text)
    text = re.sub(r"\bType (I{1,3})\b", lambda m: f"Type {len(m.group(1))}", text)
    text = re.sub(r"\bSOC\s?2\b", "sock 2" if not hindi else "SOC 2", text)
    text = re.sub(r"\b(\d{1,2}):00\b", r"\1", text)
    text = re.sub(r"\b(\d{1,2}):(\d{2})\b", r"\1 \2", text)
    text = re.sub(r"(\w)\s?/\s?(\w)", r"\1 or \2" if not hindi else r"\1 या \2", text)
    text = text.replace("&", and_)
    text = re.sub(r"\be\.g\.", "for example", text)
    text = re.sub(r"\bi\.e\.", "that is", text)
    text = re.sub(r"\s?[—–]\s?", ", ", text)
    text = re.sub(r"[\U0001F000-\U0001FFFF☀-➿]", "", text)  # emoji

    if hindi:
        def latin_word(m: re.Match) -> str:
            w = m.group(0)
            if w.lower() in HINDI_TERMS:
                return HINDI_TERMS[w.lower()]
            if w.isupper() and len(w) <= 5:  # unknown acronym -> spell it
                return " ".join(HINDI_LETTERS.get(ch.lower(), ch) for ch in w)
            return w

        text = re.sub(r"[A-Za-z]+", latin_word, text)
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> List[str]:
    """Split text into speakable sentence chunks for low-latency streaming."""
    cleaned = clean_for_speech(text)
    if not cleaned:
        return []
    parts = [p.strip() for p in SENTENCE_SPLIT_RE.split(cleaned) if p.strip()]
    return parts or [cleaned]


class LocalTTSEngine:
    """Loads Piper ONNX voices for English and Hindi and synthesizes audio on CPU."""

    def __init__(self) -> None:
        self.voices: Dict[str, PiperVoice] = {}

    def initialize(self) -> List[str]:
        if PIPER_EN_MODEL.exists():
            self.voices["en"] = PiperVoice.load(str(PIPER_EN_MODEL))
        if PIPER_HI_MODEL.exists():
            self.voices["hi"] = PiperVoice.load(str(PIPER_HI_MODEL))
        return list(self.voices)

    def _select_voice(self, text: str, language: str) -> PiperVoice:
        if DEVANAGARI_RE.search(text) and "hi" in self.voices:
            return self.voices["hi"]
        if "en" in self.voices:
            return self.voices["en"]
        return next(iter(self.voices.values()))

    def sample_rate(self, language: str = "en") -> int:
        voice = self.voices.get(language) or next(iter(self.voices.values()))
        return voice.config.sample_rate

    def synthesize_pcm(self, text: str, language: str = "en") -> Tuple[np.ndarray, int, int]:
        """Return (int16 samples, sample_rate, latency_ms) for one chunk of text."""
        if not self.voices:
            self.initialize()
        spoken = normalize_for_speech(text, language)
        if not spoken:
            return np.zeros(0, np.int16), self.sample_rate(language), 0
        t0 = time.perf_counter()
        voice = self._select_voice(spoken, language)
        parts = [chunk.audio_int16_array for chunk in voice.synthesize(spoken)]
        audio = np.concatenate(parts) if parts else np.zeros(0, np.int16)
        return audio, voice.config.sample_rate, int((time.perf_counter() - t0) * 1000)

    def synthesize_wav(self, text: str, language: str = "en") -> Tuple[bytes, int]:
        """Synthesize text into a WAV file. Returns (wav_bytes, latency_ms)."""
        audio, sr, ms = self.synthesize_pcm(text, language)
        if len(audio) == 0:
            return b"", 0
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(audio.astype("<i2").tobytes())
        return buf.getvalue(), ms
