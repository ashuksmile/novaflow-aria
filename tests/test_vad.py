"""Turn-taking: Silero VAD streaming endpointing on real human speech."""

from __future__ import annotations

import numpy as np

from app.audio_utils import wav_bytes_to_float32
from app.vad import StreamingVAD
from tests.conftest import FIXTURES

SR = 16000


def _speech(human_clips, i=0):
    clip = [c for c in human_clips if c["language"] == "en"][i]
    audio = wav_bytes_to_float32((FIXTURES / clip["file"]).read_bytes())
    # Trim the recording's own leading/trailing silence so the pauses we insert are exact.
    env = np.convolve(np.abs(audio), np.ones(320) / 320, mode="same")
    voiced = np.where(env > 0.1 * env.max())[0]
    return audio[voiced[0] : voiced[-1]]


def _silence(sec):
    return np.random.default_rng(3).normal(0, 0.0005, int(sec * SR)).astype(np.float32)


def _stream(vad, audio, frame=320):  # 20 ms frames like a browser AudioWorklet would send
    events = []
    for i in range(0, len(audio), frame):
        events += vad.process(audio[i : i + frame])
    return events


def test_single_utterance_start_and_end(human_clips):
    speech = _speech(human_clips)
    vad = StreamingVAD(min_silence_ms=650)
    events = _stream(vad, np.concatenate([_silence(1.0), speech, _silence(1.5)]))
    kinds = [e.kind for e in events]
    assert kinds == ["speech_start", "speech_end"], kinds
    start, end = events
    assert 1.0 <= start.at_sec <= 1.8, start.at_sec  # fires shortly after speech onset
    speech_end_t = 1.0 + len(speech) / SR
    lag = end.at_sec - speech_end_t
    print(f"\nend-of-turn detected {lag * 1000:.0f} ms after the speaker stopped")
    assert 0.3 <= lag <= 1.3, lag
    # captured utterance covers the whole speech (pre-roll keeps the first syllable)
    assert len(end.audio) / SR >= len(speech) / SR * 0.9


def test_short_pause_does_not_end_turn_but_long_pause_does(human_clips):
    a, b = _speech(human_clips, 0), _speech(human_clips, 2)
    vad = StreamingVAD(min_silence_ms=650)
    merged = _stream(vad, np.concatenate([_silence(0.5), a, _silence(0.3), b, _silence(1.2)]))
    assert [e.kind for e in merged].count("speech_end") == 1, "300 ms pause split the turn"

    vad = StreamingVAD(min_silence_ms=650)
    split = _stream(vad, np.concatenate([_silence(0.5), a, _silence(1.5), b, _silence(1.2)]))
    assert [e.kind for e in split].count("speech_end") == 2


def test_noise_and_hum_never_trigger():
    rng = np.random.default_rng(0)
    t = np.arange(4 * SR) / SR
    for audio in (
        rng.normal(0, 0.05, 4 * SR).astype(np.float32),  # fan / static
        (0.2 * np.sin(2 * np.pi * 50 * t)).astype(np.float32),  # mains hum
        _silence(4),
    ):
        assert _stream(StreamingVAD(), audio) == []


def test_barge_in_needs_sustained_speech(human_clips):
    """While the agent talks we require 400 ms of speech, so a short blip can't interrupt it."""
    speech = _speech(human_clips)
    blip = np.concatenate([_silence(0.5), speech[: int(0.2 * SR)], _silence(1.0)])
    assert _stream(StreamingVAD(min_speech_ms=400), blip) == []
    real = np.concatenate([_silence(0.5), speech, _silence(1.0)])
    assert _stream(StreamingVAD(min_speech_ms=400), real)[0].kind == "speech_start"
