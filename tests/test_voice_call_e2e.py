"""End-to-end voice CALL tests over the real WebSocket.

A simulated caller streams 16 kHz PCM "mic" frames exactly like the browser does. We
assert on what a real caller would experience: the agent greets, hears the caller,
answers out loud, the answer audio is intelligible speech, the caller can interrupt
(barge-in), pauses mid-sentence don't cut them off, and background noise is ignored.

Deterministic tests use a scripted fake LLM; `test_live_groq_call` uses the real agent.
Agent audio is saved to tests/output/ so you can listen to what the caller heard.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import struct
import threading
import time
from typing import Callable, List, Optional

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.audio_utils import (
    float32_to_pcm16_bytes,
    float32_to_wav_bytes,
    resample,
    word_error_rate,
)
from app import voice_session

SR = 16000
FRAME = 512  # 32 ms, a typical AudioWorklet batch


# --------------------------------------------------------------------------- #
# Fakes + harness
# --------------------------------------------------------------------------- #
class ScriptedAgent:
    """Stands in for Groq: streams a fixed reply token-by-token after `think_s`."""

    def __init__(self, reply: str, think_s: float = 0.2, tool: Optional[str] = None):
        self.reply, self.think_s, self.tool = reply, think_s, tool
        self.calls: List[str] = []

    async def respond_stream(self, session_id, user_text, history, language, lead, rag_sources):
        self.calls.append(user_text)
        await asyncio.sleep(self.think_s)
        if self.tool:
            yield "tool_start", {"tool": self.tool, "args": {"query": user_text}}
            yield "tool", {"tool": self.tool, "args": {"query": user_text}, "result": {}}
        for word in self.reply.split(" "):
            yield "text", word + " "
            await asyncio.sleep(0.01)


class StubInsights:
    def analyze_turn(self, history, user_text, lead):
        return {"lead_score": 42, "sentiment_label": "Positive", "sentiment_score": 0.5}


class Caller:
    """Simulated phone caller on the other end of the WebSocket."""

    def __init__(self, ws):
        self.ws = ws
        self.events: "queue.Queue[dict]" = queue.Queue()
        self.log: List[dict] = []
        self.audio: dict = {}  # turn -> list[np.int16]
        self.sample_rate = 22050
        self._t = threading.Thread(target=self._reader, daemon=True)
        self._t.start()

    def _reader(self):
        while True:
            try:
                msg = self.ws.receive()
            except Exception:
                return
            if msg.get("type") == "websocket.close":
                return
            if msg.get("bytes") is not None:
                turn, sr = struct.unpack("<II", msg["bytes"][:8])
                self.sample_rate = sr
                self.audio.setdefault(turn, []).append(np.frombuffer(msg["bytes"][8:], "<i2"))
                ev = {"type": "_audio", "turn": turn, "t": time.perf_counter()}
            else:
                ev = json.loads(msg["text"])
                ev["t"] = time.perf_counter()
            self.log.append(ev)
            self.events.put(ev)

    def wait_for(self, pred: Callable[[dict], bool], timeout: float = 30.0) -> dict:
        deadline = time.perf_counter() + timeout
        while True:
            left = deadline - time.perf_counter()
            if left <= 0:
                seen = [(e["type"], e.get("state") or e.get("text", "")[:40]) for e in self.log[-15:]]
                raise AssertionError(f"timed out; last events: {seen}")
            try:
                ev = self.events.get(timeout=left)
            except queue.Empty:
                continue
            if pred(ev):
                return ev

    def send_json(self, payload):
        self.ws.send_text(json.dumps(payload))

    def speak(self, audio_16k: np.ndarray, realtime: bool = False, tail_silence: float = 1.0):
        noise = np.random.default_rng(7).normal(0, 0.0005, int(tail_silence * SR)).astype(np.float32)
        stream = np.concatenate([audio_16k, noise])
        for i in range(0, len(stream), FRAME):
            self.ws.send_bytes(float32_to_pcm16_bytes(stream[i : i + FRAME]))
            if realtime:
                time.sleep(FRAME / SR)

    def turn_audio(self, turn: int) -> np.ndarray:
        parts = self.audio.get(turn, [])
        return np.concatenate(parts).astype(np.float32) / 32768.0 if parts else np.zeros(0, np.float32)


def caller_voice(tts, text: str, lang: str = "en") -> np.ndarray:
    pcm, sr, _ = tts.synthesize_pcm(text, lang)
    return resample(pcm.astype(np.float32) / 32768.0, sr, SR)


def judge(stt, audio_16k, language="en"):
    segs, _ = stt._model("small").transcribe(audio_16k, language=language, beam_size=5,
                                              without_timestamps=True, condition_on_previous_text=False)
    return " ".join(s.text.strip() for s in segs).strip()


@pytest.fixture()
def app_client(tmp_path, monkeypatch):
    import app.database as database
    import app.main as main

    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(main, "insights_engine", StubInsights())
    with TestClient(main.app) as client:
        yield client, main


def _connect(client, name):
    ctx = client.websocket_connect(f"/ws/voice/{name}")
    ws = ctx.__enter__()
    caller = Caller(ws)
    caller.wait_for(lambda e: e["type"] == "connected")
    return ctx, caller


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_full_voice_turn(app_client, tts, stt, output_dir, monkeypatch):
    client, main = app_client
    reply = "Our Growth plan is 799 dollars per month and covers up to 25 reps. Would you like a quick demo?"
    agent = ScriptedAgent(reply)
    monkeypatch.setattr(main, "sales_agent", agent)
    ctx, caller = _connect(client, "e2e-turn")
    try:
        caller.send_json({"type": "start_call", "language": "auto", "barge_in": True})

        # 1. Agent greets the caller by voice.
        caller.wait_for(lambda e: e["type"] == "agent_done" and e["turn"] == 1)
        caller.wait_for(lambda e: e["type"] == "state" and e["state"] == "listening")
        greeting = caller.turn_audio(1)
        (output_dir / "call_greeting.wav").write_bytes(float32_to_wav_bytes(greeting, caller.sample_rate))
        heard = judge(stt, resample(greeting, caller.sample_rate, SR))
        print(f"\n[greeting heard] {heard}")
        assert word_error_rate(voice_session.GREETINGS["en"], heard) <= 0.2

        # 2. Caller speaks; nobody presses a button: VAD decides when they're done.
        question = "Hi, what does your growth plan cost for twenty sales reps?"
        caller.speak(caller_voice(tts, question), realtime=False)
        caller.wait_for(lambda e: e["type"] == "state" and e["state"] == "user_speaking")
        tr = caller.wait_for(lambda e: e["type"] == "transcript")
        print(f"[caller said]  {question}\n[server heard] {tr['text']}  ({tr['stt_ms']} ms)")
        assert word_error_rate(question, tr["text"]) <= 0.25

        done = caller.wait_for(lambda e: e["type"] == "agent_done" and e["turn"] == tr["turn"])
        assert agent.calls == [tr["text"]]
        m = done["metrics"]
        print(f"[latency] stt={m['stt_ms']}ms llm_first_token={m['llm_first_token_ms']}ms "
              f"tts_first={m['tts_first_ms']}ms voice_to_voice={m['voice_to_voice_ms']}ms")
        assert m["voice_to_voice_ms"] < 3000

        # 3. What came out of the speaker is the reply, spoken intelligibly.
        caller.wait_for(lambda e: e["type"] == "state" and e["state"] == "listening")
        answer = caller.turn_audio(tr["turn"])
        (output_dir / "call_answer.wav").write_bytes(float32_to_wav_bytes(answer, caller.sample_rate))
        heard = judge(stt, resample(answer, caller.sample_rate, SR))
        print(f"[answer heard] {heard}")
        assert len(answer) / caller.sample_rate > 3.0
        assert word_error_rate(reply, heard) <= 0.2

        # sentences are streamed (subtitles) before the turn completes
        sentences = [e for e in caller.log if e["type"] == "agent_sentence" and e["turn"] == tr["turn"]]
        assert len(sentences) == 2
    finally:
        ctx.__exit__(None, None, None)


def test_barge_in_interrupts_agent(app_client, tts, monkeypatch):
    client, main = app_client
    long_reply = ("Let me walk you through all three plans in detail. The Starter plan is for small teams. "
                  "The Growth plan adds CRM sync and analytics. The Enterprise plan adds custom security reviews.")
    agent = ScriptedAgent(long_reply)
    monkeypatch.setattr(main, "sales_agent", agent)
    ctx, caller = _connect(client, "e2e-bargein")
    try:
        caller.send_json({"type": "start_call", "language": "en", "barge_in": True})
        caller.wait_for(lambda e: e["type"] == "agent_done" and e["turn"] == 1)
        caller.wait_for(lambda e: e["type"] == "state" and e["state"] == "listening", timeout=20)

        caller.speak(caller_voice(tts, "Tell me about your plans."))
        tr = caller.wait_for(lambda e: e["type"] == "transcript")
        caller.wait_for(lambda e: e["type"] == "state" and e["state"] == "speaking" and e["turn"] == tr["turn"])
        time.sleep(1.0)  # agent is mid-sentence

        t_speak = time.perf_counter()
        caller.speak(caller_voice(tts, "Sorry to interrupt, just tell me the price."), realtime=True)
        intr = caller.wait_for(lambda e: e["type"] == "interrupt")
        print(f"\n[barge-in] agent stopped {int((intr['t'] - t_speak) * 1000)} ms after caller started talking")
        assert intr["turn"] == tr["turn"]
        assert intr["t"] - t_speak < 1.2

        tr2 = caller.wait_for(lambda e: e["type"] == "transcript" and e["turn"] > tr["turn"])
        assert "price" in tr2["text"].lower()
        caller.wait_for(lambda e: e["type"] == "agent_done" and e["turn"] == tr2["turn"])

        hist = main.session_histories["e2e-bargein"]
        cut = [h for h in hist if "[interrupted by caller]" in h["content"]]
        assert cut, "interrupted reply should be recorded as partial"
        assert len(cut[0]["content"]) < len(long_reply)
        assert agent.calls[-1] == tr2["text"]
    finally:
        ctx.__exit__(None, None, None)


def test_mid_sentence_pause_is_merged(app_client, tts, monkeypatch):
    """Caller pauses (end-of-turn fires) then keeps talking before the agent answers:
    the half-built answer is dropped and both halves are answered as one turn."""
    client, main = app_client
    agent = ScriptedAgent("Sure, I can help with that.", think_s=2.5)
    monkeypatch.setattr(main, "sales_agent", agent)
    ctx, caller = _connect(client, "e2e-merge")
    try:
        caller.send_json({"type": "start_call", "language": "en", "barge_in": True})
        caller.wait_for(lambda e: e["type"] == "agent_done" and e["turn"] == 1)
        caller.wait_for(lambda e: e["type"] == "state" and e["state"] == "listening", timeout=20)

        caller.speak(caller_voice(tts, "I want to know about pricing."), tail_silence=0.9)
        caller.wait_for(lambda e: e["type"] == "state" and e["state"] == "thinking")
        caller.speak(caller_voice(tts, "And also about security."), tail_silence=1.0)

        done = caller.wait_for(lambda e: e["type"] == "agent_done" and e["turn"] > 1)
        final = [e for e in caller.log if e["type"] == "transcript"][-1]["text"].lower()
        print(f"\n[merged transcript] {final}")
        assert "pricing" in final and "security" in final
        assert done["text"].startswith("Sure")
        assert "security" in agent.calls[-1].lower() and "pricing" in agent.calls[-1].lower()
    finally:
        ctx.__exit__(None, None, None)


def test_background_noise_is_ignored(app_client, monkeypatch):
    client, main = app_client
    agent = ScriptedAgent("This should never be said.")
    monkeypatch.setattr(main, "sales_agent", agent)
    ctx, caller = _connect(client, "e2e-noise")
    try:
        caller.send_json({"type": "start_call", "language": "en", "barge_in": True})
        caller.wait_for(lambda e: e["type"] == "agent_done" and e["turn"] == 1)
        caller.wait_for(lambda e: e["type"] == "state" and e["state"] == "listening", timeout=20)
        rng = np.random.default_rng(5)
        t = np.arange(3 * SR) / SR
        caller.speak(rng.normal(0, 0.05, 3 * SR).astype(np.float32))            # static
        caller.speak((0.2 * np.sin(2 * np.pi * 50 * t)).astype(np.float32))     # hum
        time.sleep(2.0)
        assert not [e for e in caller.log if e["type"] in ("transcript", "interrupt")]
        assert agent.calls == []
    finally:
        ctx.__exit__(None, None, None)


def test_echo_detector():
    agent = "Our Growth plan is 799 dollars per month and covers up to 25 reps."
    assert voice_session.looks_like_echo("growth plan is 799 dollars per month", agent)
    assert not voice_session.looks_like_echo("what about the enterprise plan security", agent)


@pytest.mark.skipif(not os.getenv("GROQ_API_KEY") and not os.path.exists(".env"), reason="needs Groq")
def test_live_groq_call(app_client, tts, stt, output_dir):
    """Real agent (Groq + RAG tools) answering a spoken pricing question, Hindi follow-up."""
    client, main = app_client
    ctx, caller = _connect(client, "e2e-live")
    try:
        caller.send_json({"type": "start_call", "language": "auto", "barge_in": True})
        caller.wait_for(lambda e: e["type"] == "agent_done" and e["turn"] == 1)
        caller.wait_for(lambda e: e["type"] == "state" and e["state"] == "listening", timeout=20)

        caller.speak(caller_voice(tts, "How much does the Growth plan cost per month?"))
        tr = caller.wait_for(lambda e: e["type"] == "transcript")
        done = caller.wait_for(lambda e: e["type"] == "agent_done" and e["turn"] == tr["turn"], timeout=60)
        caller.wait_for(lambda e: e["type"] == "state" and e["state"] == "listening", timeout=60)
        m = done["metrics"]
        print(f"\n[EN] heard: {tr['text']}\n     reply: {done['text']}\n     {m}")
        assert "search_knowledge_base" in [t["tool"] for t in done["tool_events"]]
        assert "799" in done["text"] or "seven hundred" in done["text"].lower()
        answer = resample(caller.turn_audio(tr["turn"]), caller.sample_rate, SR)
        (output_dir / "live_answer_en.wav").write_bytes(float32_to_wav_bytes(answer, SR))
        assert word_error_rate(done["text"], judge(stt, answer)) <= 0.3

        caller.speak(caller_voice(tts, "क्या आपका डेटा भारत में स्टोर होता है?", "hi"))
        tr = caller.wait_for(lambda e: e["type"] == "transcript" and e["turn"] > done["turn"])
        done = caller.wait_for(lambda e: e["type"] == "agent_done" and e["turn"] == tr["turn"], timeout=60)
        print(f"[HI] heard: {tr['text']} ({tr['language']})\n     reply: {done['text']}\n     {done['metrics']}")
        assert tr["language"] == "hi"
        assert any("ऀ" <= ch <= "ॿ" for ch in done["text"]), "agent should answer in Hindi"
        caller.wait_for(lambda e: e["type"] == "state" and e["state"] == "listening", timeout=60)
        hi_audio = resample(caller.turn_audio(tr["turn"]), caller.sample_rate, SR)
        (output_dir / "live_answer_hi.wav").write_bytes(float32_to_wav_bytes(hi_audio, SR))
    finally:
        ctx.__exit__(None, None, None)
