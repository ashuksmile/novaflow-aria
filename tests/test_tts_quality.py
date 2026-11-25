"""Is the agent actually SPEAKING, or just making noise?

Every TTS sample must pass two independent checks:
  1. Signal checks  - Silero VAD says it is speech, spectrum is speech-like (not flat noise),
                      loudness/clipping/duration/speaking-rate are sane.
  2. Intelligibility - Whisper transcribes it back to the words we asked it to say.
A negative control (white noise, hum, silence) must FAIL the same checks, proving the
checks can tell speech from noise. WAVs are saved to tests/output/ so you can listen.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.audio_utils import (
    char_error_rate,
    clipping_ratio,
    float32_to_wav_bytes,
    resample,
    rms_dbfs,
    spectral_flatness,
    word_error_rate,
)
from app.vad import speech_ratio

EN_SENTENCES = [
    "Hi, this is Aria from NovaFlow AI. What brings you here today?",
    "Our Growth plan is $799/month, billed annually for a 15% discount.",
    "It integrates with Salesforce, HubSpot and Zoho, and syncs every call automatically.",
    "Your demo is confirmed for tomorrow at 11:30 AM. I'll send the invite to your email.",
    "We are SOC 2 Type II certified and compliant with GDPR and India's DPDP Act.",
]
HI_SENTENCES = [
    "नमस्ते, मैं आर्या हूँ। आज मैं आपकी क्या मदद कर सकती हूँ?",
    "हमारा ग्रोथ प्लान आपकी टीम के लिए सबसे अच्छा रहेगा।",
    "क्या मैं आपके लिए कल सुबह ग्यारह बजे का डेमो बुक कर दूँ?",
    "आपका डेटा पूरी तरह सुरक्षित है और भारत में ही स्टोर होता है।",
]
EN_MAX_WER = 0.20
HI_MAX_CER = 0.35


def signal_report(audio_16k: np.ndarray, n_words: int) -> dict:
    dur = len(audio_16k) / 16000
    return {
        "duration_s": round(dur, 2),
        "words_per_s": round(n_words / dur, 2) if dur else 0.0,
        "rms_dbfs": round(rms_dbfs(audio_16k), 1),
        "clipping": round(clipping_ratio(audio_16k), 4),
        "flatness": round(spectral_flatness(audio_16k), 3),
        "speech_ratio": round(speech_ratio(audio_16k), 2),
    }


def assert_is_speech(rep: dict) -> None:
    assert rep["duration_s"] > 0.8, rep
    assert 1.0 <= rep["words_per_s"] <= 6.0, f"implausible speaking rate {rep}"
    assert -35 <= rep["rms_dbfs"] <= -3, f"too quiet / too loud {rep}"
    assert rep["clipping"] < 0.005, f"clipping {rep}"
    assert rep["flatness"] < 0.25, f"spectrum looks like noise {rep}"
    assert rep["speech_ratio"] >= 0.5, f"VAD does not hear speech {rep}"


def judge(stt, audio_16k, language):
    """Independent listener: Whisper-small + beam search (stronger than the realtime STT),
    so a failure means the TTS is unclear, not that the fast STT model slipped."""
    segments, info = stt._model("small").transcribe(
        audio_16k, language=language, beam_size=5, without_timestamps=True,
        condition_on_previous_text=False,
    )
    return " ".join(s.text.strip() for s in segments).strip(), info.language


def _synth_16k(tts, text, lang):
    pcm, sr, ms = tts.synthesize_pcm(text, lang)
    assert len(pcm) > 0, "TTS returned no audio"
    return resample(pcm.astype(np.float32) / 32768.0, sr, 16000), pcm, sr, ms


@pytest.mark.parametrize("text", EN_SENTENCES)
def test_english_tts_is_intelligible_speech(tts, stt, output_dir, text):
    audio, pcm, sr, ms = _synth_16k(tts, text, "en")
    idx = EN_SENTENCES.index(text)
    (output_dir / f"tts_en_{idx}.wav").write_bytes(float32_to_wav_bytes(pcm / 32768.0, sr))

    rep = signal_report(audio, len(text.split()))
    assert_is_speech(rep)

    heard, lang = judge(stt, audio, None)
    assert lang == "en", f"English TTS detected as {lang}"
    wer = word_error_rate(text, heard)
    print(f"\n[EN] tts={ms}ms {rep}\n  said : {text}\n  heard: {heard}\n  WER={wer:.2f}")
    assert wer <= EN_MAX_WER, f"WER {wer:.2f}: said {text!r}, Whisper heard {heard!r}"


@pytest.mark.parametrize("text", HI_SENTENCES)
def test_hindi_tts_is_intelligible_speech(tts, stt, output_dir, text):
    audio, pcm, sr, ms = _synth_16k(tts, text, "hi")
    idx = HI_SENTENCES.index(text)
    (output_dir / f"tts_hi_{idx}.wav").write_bytes(float32_to_wav_bytes(pcm / 32768.0, sr))

    rep = signal_report(audio, len(text.split()))
    assert_is_speech(rep)

    _, detected = judge(stt, audio, None)
    assert detected in ("hi", "ur"), f"Hindi TTS detected as {detected}"
    heard, _ = judge(stt, audio, "hi")
    cer = char_error_rate(text, heard)
    print(f"\n[HI] tts={ms}ms {rep}\n  said : {text}\n  heard: {heard}\n  CER={cer:.2f}")
    assert cer <= HI_MAX_CER, f"CER {cer:.2f}: said {text!r}, Whisper heard {heard!r}"


def test_mixed_hindi_english_terms_are_spoken(tts, stt):
    """English product words inside Hindi replies get transliterated, not garbled.
    Piper is stochastic, so average over 3 syntheses."""
    text = "NovaFlow AI आपके CRM को अपने आप अपडेट करता है।"
    as_spoken = "नोवाफ्लो ए आई आपके सी आर एम को अपने आप अपडेट करता है"
    cers = []
    for _ in range(3):
        audio, *_ = _synth_16k(tts, text, "hi")
        heard, _ = judge(stt, audio, "hi")
        cers.append(char_error_rate(as_spoken, heard.replace("CRM", "सीआरएम").replace("AI", "एआई")))
        print("\n[HI mixed] heard:", heard, f"CER={cers[-1]:.2f}")
    assert sum(cers) / len(cers) <= HI_MAX_CER, cers


@pytest.mark.parametrize(
    "name,signal",
    [
        ("white_noise", lambda n: np.random.default_rng(0).normal(0, 0.1, n).astype(np.float32)),
        ("hum_120hz", lambda n: (0.3 * np.sin(2 * np.pi * 120 * np.arange(n) / 16000)).astype(np.float32)),
        ("silence", lambda n: np.zeros(n, np.float32)),
    ],
)
def test_negative_control_noise_is_not_speech(stt, name, signal):
    """The checks above must REJECT non-speech, otherwise they prove nothing."""
    audio = signal(16000 * 3)
    rep = signal_report(audio, 8)
    with pytest.raises(AssertionError):
        assert_is_speech(rep)
    assert rep["speech_ratio"] < 0.1, rep
    assert stt.transcribe_array(audio, "auto").text == "", f"{name} produced a transcript"
