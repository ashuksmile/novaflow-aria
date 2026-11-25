"""Real-browser voice call: the actual frontend, mic, AudioWorklet and Web Audio playback.

Chromium's fake media device plays a WAV file into getUserMedia as if a caller were
speaking into the microphone. The page runs unmodified; we only tap the audio Aria sends
to the speaker so we can check it is intelligible speech.

Needs: `pip install playwright`, any installed Chromium-family browser (Chrome, Chromium,
Brave; set BROWSER_PATH to override) and a Groq key (this uses the real agent).
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

import numpy as np
import pytest

from app.audio_utils import float32_to_wav_bytes, resample, word_error_rate
from tests.conftest import ROOT

playwright_api = pytest.importorskip("playwright.sync_api")

QUESTION = "Hi, how much does your Growth plan cost per month?"


def _browser() -> str | None:
    if os.getenv("BROWSER_PATH"):
        return os.getenv("BROWSER_PATH")
    for name in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable", "brave-browser"):
        if shutil.which(name):
            return shutil.which(name)
    return None


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def server(tmp_path):
    port = _free_port()
    env = dict(os.environ, SALES_DB_PATH=str(tmp_path / "browser.db"))
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port)],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1)
                break
            except Exception:
                time.sleep(0.5)
        else:
            pytest.fail("server did not start")
        yield f"http://127.0.0.1:{port}"
    finally:
        proc.terminate()
        proc.wait(timeout=10)


@pytest.mark.skipif(_browser() is None, reason="no Chromium-family browser installed")
def test_browser_voice_call(server, tts, stt, tmp_path, output_dir):
    # Caller "mic": stay quiet while Aria greets, ask the question, then stay quiet.
    pcm, sr, _ = tts.synthesize_pcm(QUESTION, "en")
    q = resample(pcm.astype(np.float32) / 32768.0, sr, 48000)
    mic = np.concatenate([np.zeros(48000 * 11), q, np.zeros(48000 * 40)]).astype(np.float32)
    mic_wav = tmp_path / "caller_mic.wav"
    mic_wav.write_bytes(float32_to_wav_bytes(mic, 48000))

    with playwright_api.sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=_browser(),
            headless=True,
            args=[
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
                f"--use-file-for-fake-audio-capture={mic_wav}",
                "--autoplay-policy=no-user-gesture-required",
            ],
        )
        page = browser.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(server)
        page.wait_for_function("() => ws && ws.readyState === 1")
        # Tap what Aria sends to the speaker (the page still plays it normally).
        page.evaluate(
            """() => {
              window.__heard = {};
              const orig = window.playAgentAudio;
              window.playAgentAudio = (buf) => {
                const v = new DataView(buf); const turn = v.getUint32(0, true);
                window.__rate = v.getUint32(4, true);
                (window.__heard[turn] = window.__heard[turn] || []).push(...new Int16Array(buf, 8));
                orig(buf);
              };
            }"""
        )
        page.click("#call-btn")
        page.wait_for_function("() => document.querySelector('#call-btn').textContent.toLowerCase().includes('end call')")

        # Greeting is spoken, then Aria listens; the caller asks; Aria answers by voice.
        page.wait_for_function("() => [...document.querySelectorAll('#chat-log .body')].length >= 1", timeout=20000)
        page.wait_for_function(
            "() => [...document.querySelectorAll('#chat-log span')].some(s => s.textContent.includes('Prospect'))",
            timeout=40000,
        )
        page.wait_for_function(
            "() => Object.keys(window.__heard).length >= 2 && callState === 'listening'", timeout=60000
        )
        page.screenshot(path=str(output_dir / "browser_call.png"), full_page=True)

        bodies = page.eval_on_selector_all("#chat-log .body", "els => els.map(e => e.textContent)")
        v2v = page.text_content("#metric-v2v")
        heard = page.evaluate("() => window.__heard")
        rate = page.evaluate("() => window.__rate")
        page.click("#call-btn")
        browser.close()

    assert not errors, errors
    user_text = bodies[1]
    print(f"\n[browser] greeting : {bodies[0]}\n[browser] caller   : {user_text}\n"
          f"[browser] Aria     : {bodies[2] if len(bodies) > 2 else ''}\n[browser] voice-to-voice {v2v}")
    # STT got clean audio through the AudioWorklet (48 kHz -> 16 kHz) path.
    assert word_error_rate(QUESTION, user_text) <= 0.25, user_text
    # The answer that played in the browser is intelligible speech.
    turns = sorted(int(k) for k in heard)
    answer = np.array(heard[str(turns[-1])], dtype=np.float32) / 32768.0
    a16 = resample(answer, rate, 16000)
    (output_dir / "browser_answer.wav").write_bytes(float32_to_wav_bytes(answer, rate))
    segs, _ = stt._model("small").transcribe(a16, language="en", beam_size=5, without_timestamps=True)
    spoken = " ".join(s.text for s in segs)
    print(f"[browser] speaker audio transcribed: {spoken}")
    assert len(answer) / rate > 2.0
    assert word_error_rate(bodies[2], spoken) <= 0.3
