"""Speech-to-text accuracy on REAL human recordings (not our own TTS).

English: LibriSpeech dev-clean speakers.   Hindi: Google FLEURS native speakers.
Also checks robustness: background noise, browser-style 48 kHz stereo input,
language auto-detection, and no hallucinated text on silence/noise.
"""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from app.audio_utils import (
    char_error_rate,
    float32_to_wav_bytes,
    resample,
    wav_bytes_to_float32,
    word_error_rate,
)
from tests.conftest import FIXTURES

EN_MAX_WER = 0.25
HI_MAX_CER = 0.40


def _load(clip):
    return wav_bytes_to_float32((FIXTURES / clip["file"]).read_bytes())


def _clips(human_clips, lang):
    return [c for c in human_clips if c["language"] == lang]


def test_english_human_speech(stt, human_clips):
    total_wer = []
    for clip in _clips(human_clips, "en"):
        r = stt.transcribe_array(_load(clip), "auto")
        wer = word_error_rate(clip["text"], r.text)
        total_wer.append(wer)
        print(f"\n[EN {r.latency_ms}ms] ref: {clip['text']}\n            hyp: {r.text}  (WER {wer:.2f})")
        assert r.language == "en"
        assert wer <= EN_MAX_WER, (clip["file"], r.text)
    print(f"\nmean English WER = {np.mean(total_wer):.3f}")


def test_hindi_human_speech(stt, human_clips):
    total = []
    for clip in _clips(human_clips, "hi"):
        r = stt.transcribe_array(_load(clip), "auto")
        cer = char_error_rate(clip["text"], r.text)
        total.append(cer)
        print(f"\n[HI {r.latency_ms}ms {r.model}] ref: {clip['text']}\n    hyp: {r.text}  (CER {cer:.2f})")
        assert r.language == "hi", f"auto-detect said {r.language}"
        assert not any("؀" <= ch <= "ۿ" for ch in r.text), "Hindi came back in Urdu script"
        assert cer <= HI_MAX_CER, (clip["file"], r.text)
    print(f"\nmean Hindi CER = {np.mean(total):.3f}")


@pytest.mark.parametrize("snr_db", [20, 10])
def test_english_with_background_noise(stt, human_clips, snr_db):
    clip = _clips(human_clips, "en")[0]
    clean = _load(clip)
    noise = np.random.default_rng(1).normal(0, 1, len(clean)).astype(np.float32)
    noise *= np.sqrt(np.mean(clean**2) / (10 ** (snr_db / 10))) / (np.std(noise) + 1e-9)
    r = stt.transcribe_array(clean + noise, "auto")
    wer = word_error_rate(clip["text"], r.text)
    print(f"\n[SNR {snr_db}dB] {r.text} (WER {wer:.2f})")
    assert wer <= (0.25 if snr_db >= 20 else 0.4)


def test_browser_style_48k_stereo_wav(stt, human_clips):
    """Mic WAVs from browsers are usually 48 kHz; the decoder must resample correctly."""
    clip = _clips(human_clips, "en")[1]
    mono = _load(clip)
    up = resample(mono, 16000, 48000)
    stereo = np.stack([up, up], axis=1).reshape(-1)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(48000)
        wf.writeframes((np.clip(stereo, -1, 1) * 32767).astype("<i2").tobytes())
    text, lang, _, _ = stt.transcribe(buf.getvalue(), "auto")
    assert lang == "en"
    assert word_error_rate(clip["text"], text) <= EN_MAX_WER, text


def test_forced_language_hint(stt, human_clips):
    clip = _clips(human_clips, "hi")[0]
    r = stt.transcribe_array(_load(clip), "hi")
    assert r.language == "hi" and "small" in r.model
    assert char_error_rate(clip["text"], r.text) <= HI_MAX_CER


@pytest.mark.parametrize("seconds", [0.05, 2.0])
def test_silence_gives_no_text(stt, seconds):
    r = stt.transcribe_array(np.zeros(int(16000 * seconds), np.float32), "auto")
    assert r.text == ""


def test_wav_roundtrip_helpers():
    x = (0.5 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)).astype(np.float32)
    y = wav_bytes_to_float32(float32_to_wav_bytes(x, 16000))
    assert np.max(np.abs(x - y)) < 1e-3
    assert word_error_rate("Our plan is $799 per month", "our plan is 799 dollars per month") == 0
    assert char_error_rate("गाँव", "गांव") == 0
