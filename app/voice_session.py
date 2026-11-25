"""Real-time, full-duplex voice call session.

    mic (16 kHz PCM frames, continuous) ──► Silero VAD ──► end-of-turn ──► Whisper STT
                                             │                                 │
                     barge-in (user talks over agent) ◄──┐                     ▼
                                                         │        Groq LLM (token stream + tools)
    speaker ◄── paced PCM frames ◄── Piper TTS ◄── sentence splitter ◄─────────┘

Wire protocol (one WebSocket per call)
  client -> server  binary: raw PCM16 LE mono 16 kHz mic audio, any frame size
                    json:   start_call {language, barge_in} | end_call | config | text
  server -> client  binary: 8-byte header <uint32 turn_id, uint32 sample_rate> + PCM16 mono
                    json:   state, transcript, agent_sentence, agent_done, interrupt,
                            tool, metrics, intelligence, info
"""

from __future__ import annotations

import asyncio
import json
import re
import struct
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import WebSocket

from app import database as db
from app.audio_utils import STT_SAMPLE_RATE, chunk_pcm, pcm16_bytes_to_float32
from app.config import VAD_BARGE_IN_MS, VAD_END_OF_TURN_MS
from app.vad import StreamingVAD

DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
SENTENCE_END_RE = re.compile(r"[.!?।](?=\s)")
PLAYBACK_LEAD_SEC = 0.6  # how far ahead of real time we let audio run (jitter buffer)
FRAME_MS = 100

GREETINGS = {
    "en": "Hi, this is Aria from NovaFlow AI. I help sales teams automate lead qualification "
          "and CRM updates. What brings you here today?",
    "hi": "नमस्ते, मैं नोवाफ़्लो से आर्या बोल रही हूँ। मैं सेल्स टीमों की लीड क्वालिफ़िकेशन और सीआरएम "
          "अपडेट को ऑटोमेट करने में मदद करती हूँ। बताइए, मैं आपकी क्या मदद कर सकती हूँ?",
}
FILLERS = {
    "en": {"search_knowledge_base": "Let me check that for you.",
           "check_available_slots": "Let me look at the calendar.",
           "book_appointment": "Booking that for you now."},
    "hi": {"search_knowledge_base": "एक पल, मैं देख लेती हूँ।",
           "check_available_slots": "मैं कैलेंडर देख लेती हूँ।",
           "book_appointment": "मैं अभी बुक कर देती हूँ।"},
}
FALLBACK = {
    "en": "Sorry, I had a small connection hiccup. Could you say that again?",
    "hi": "माफ़ कीजिए, कनेक्शन में थोड़ी दिक्कत हुई। क्या आप दोबारा बोल सकते हैं?",
}


def _words(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def looks_like_echo(user_text: str, agent_text: str) -> bool:
    """True when a 'user' utterance is really the agent's own voice leaking from the speaker."""
    u, a = _words(user_text), set(_words(agent_text))
    if len(u) < 3 or not a:
        return False
    return sum(w in a for w in u) / len(u) >= 0.75


class VoiceCallSession:
    def __init__(self, ws: WebSocket, session_id: str, *, stt, tts, agent, crm, insights,
                 history: List[Dict[str, str]]) -> None:
        self.ws = ws
        self.session_id = session_id
        self.stt, self.tts, self.agent, self.crm, self.insights = stt, tts, agent, crm, insights
        self.history = history
        self.loop = asyncio.get_running_loop()
        self._send_lock = asyncio.Lock()

        self.vad = StreamingVAD(min_silence_ms=VAD_END_OF_TURN_MS)
        self.language = "auto"
        self.barge_in = True
        self.in_call = False
        self.state = "idle"

        self.turn_id = 0
        self.turn_task: Optional[asyncio.Task] = None
        self.phase = "idle"  # stt | thinking | speaking (inside the current turn)
        self.pending_audio: Optional[np.ndarray] = None  # user audio not yet answered
        self.ignore_utterance = False
        self.last_agent_text = ""
        self.speech_end_at: Optional[float] = None
        self._bg: set = set()

    # ------------------------------------------------------------------ io --
    async def send(self, payload: dict) -> None:
        async with self._send_lock:
            try:
                await self.ws.send_text(json.dumps(payload, default=str))
            except Exception:
                pass

    async def send_audio(self, turn_id: int, sample_rate: int, pcm: bytes) -> None:
        async with self._send_lock:
            try:
                await self.ws.send_bytes(struct.pack("<II", turn_id, sample_rate) + pcm)
            except Exception:
                pass

    async def set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            await self.send({"type": "state", "state": state, "turn": self.turn_id})

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)
        return task

    # ------------------------------------------------------------- control --
    async def on_json(self, msg: Dict[str, Any]) -> None:
        kind = msg.get("type")
        if kind in ("start_call", "config"):
            self.language = msg.get("language", self.language) or "auto"
            self.barge_in = bool(msg.get("barge_in", self.barge_in))
        if kind == "start_call":
            await self.cancel_turn()
            self.in_call = True
            self.pending_audio = None
            self.vad.reset(keep_model_state=False)
            await self.set_state("listening")
            self.start_turn("greeting")
        elif kind == "end_call":
            self.in_call = False
            await self.cancel_turn()
            self.vad.reset()
            await self.set_state("idle")
        elif kind == "greeting":  # legacy button
            await self.cancel_turn()
            self.start_turn("greeting")
        elif kind == "text":
            text = (msg.get("text") or "").strip()
            if text:
                await self.interrupt_if_speaking()
                await self.cancel_turn()
                self.start_turn("text", text=text, language=msg.get("language"))

    async def on_audio(self, data: bytes) -> None:
        if not self.in_call:
            return
        agent_talking = self.phase == "speaking"
        # Harder to trigger while the agent talks: filters speaker echo, coughs, "mm-hmm".
        self.vad.min_speech_ms = VAD_BARGE_IN_MS if agent_talking else 160
        self.vad.threshold = 0.6 if agent_talking else 0.5

        for ev in self.vad.process(pcm16_bytes_to_float32(data)):
            if ev.kind == "speech_start":
                await self._on_speech_start()
            elif ev.kind == "speech_end":
                if self.ignore_utterance:
                    self.ignore_utterance = False
                    await self.set_state("speaking" if self.phase == "speaking" else "listening")
                    continue
                gap = np.zeros(int(0.25 * STT_SAMPLE_RATE), np.float32)
                self.pending_audio = (
                    ev.audio if self.pending_audio is None
                    else np.concatenate([self.pending_audio, gap, ev.audio])
                )
                self.speech_end_at = time.perf_counter()
                self.start_turn("audio")

    async def _on_speech_start(self) -> None:
        busy = self.turn_task is not None and not self.turn_task.done()
        if busy and self.phase == "speaking":
            if not self.barge_in:
                self.ignore_utterance = True  # half-duplex mode: ignore talk-over
                return
            await self.interrupt_if_speaking()
        elif busy:
            # Caller kept talking after a pause: drop the half-built answer, keep their audio,
            # and answer the merged utterance once they finish.
            await self.cancel_turn()
        await self.set_state("user_speaking")

    async def interrupt_if_speaking(self) -> None:
        if self.turn_task and not self.turn_task.done() and self.phase == "speaking":
            tid = self.turn_id
            await self.cancel_turn()
            await self.send({"type": "interrupt", "turn": tid})

    async def cancel_turn(self) -> None:
        task, self.turn_task = self.turn_task, None
        if task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.phase = "idle"

    def start_turn(self, kind: str, text: str = "", language: Optional[str] = None) -> None:
        if self.turn_task and not self.turn_task.done():
            self.turn_task.cancel()
        self.turn_id += 1
        self.turn_task = asyncio.create_task(self._run_turn(self.turn_id, kind, text, language))

    async def close(self) -> None:
        self.in_call = False
        await self.cancel_turn()
        for t in list(self._bg):
            t.cancel()

    # ---------------------------------------------------------------- turn --
    async def _run_turn(self, turn_id: int, kind: str, text: str, language: Optional[str]) -> None:
        metrics: Dict[str, Any] = {"turn": turn_id}
        t_ref = self.speech_end_at if kind == "audio" and self.speech_end_at else time.perf_counter()
        committed = False
        spoken: List[Tuple[str, float]] = []  # (sentence, start offset in turn audio)
        play_start: Optional[float] = None
        user_text, lang = text, "en"
        tasks: List[asyncio.Task] = []

        try:
            # 1) Speech-to-text --------------------------------------------------
            if kind == "audio":
                self.phase = "stt"
                await self.set_state("transcribing")
                audio = self.pending_audio
                r = await asyncio.to_thread(self.stt.transcribe_array, audio, self.language)
                metrics.update(stt_ms=r.latency_ms, stt_model=r.model,
                               utterance_sec=round(len(audio) / STT_SAMPLE_RATE, 2))
                if r.text and self.last_agent_text and looks_like_echo(r.text, self.last_agent_text):
                    r.rejected_reason, r.text = "echo_of_agent", ""
                if not r.text:
                    self.pending_audio = None
                    await self.send({"type": "info", "message": f"Ignored audio ({r.rejected_reason}).",
                                     "reason": r.rejected_reason})
                    self.phase = "idle"
                    await self.set_state("listening")
                    return
                user_text, lang = r.text, r.language
                await self.send({"type": "transcript", "turn": turn_id, "text": user_text,
                                 "language": lang, "confidence": r.language_prob,
                                 "stt_ms": r.latency_ms, "model": r.model})
            elif kind == "text":
                if language in ("en", "hi"):
                    lang = language
                else:
                    lang = "hi" if DEVANAGARI_RE.search(user_text) else (
                        self.language if self.language in ("en", "hi") else "en")
                await self.send({"type": "transcript", "turn": turn_id, "text": user_text,
                                 "language": lang, "confidence": 1.0, "stt_ms": 0, "typed": True})
            else:  # greeting
                lang = "hi" if self.language == "hi" else "en"

            # 2) LLM (streamed) -> sentences -> TTS -> paced audio ----------------
            self.phase = "thinking"
            await self.set_state("thinking")
            sentence_q: asyncio.Queue = asyncio.Queue()
            audio_q: asyncio.Queue = asyncio.Queue(maxsize=3)
            reply_parts: List[str] = []
            rag_sources: List[dict] = []
            tool_events: List[dict] = []
            t_llm = time.perf_counter()

            async def produce_text() -> None:
                buf = ""
                queued_any = False
                try:
                    if kind == "greeting":
                        reply_parts.append(GREETINGS[lang])
                        await sentence_q.put(GREETINGS[lang])
                        return
                    lead = db.ensure_lead(self.session_id, language=lang)
                    async for ev_kind, data in self.agent.respond_stream(
                        self.session_id, user_text, list(self.history), lang, lead, rag_sources
                    ):
                        if ev_kind == "text":
                            metrics.setdefault("llm_first_token_ms", int((time.perf_counter() - t_llm) * 1000))
                            buf += data
                            reply_parts.append(data)
                            while True:
                                m = SENTENCE_END_RE.search(buf)
                                if not m and len(buf) > 140 and ", " in buf[40:]:
                                    cut = buf.rfind(", ") + 1  # long run-on: speak up to a comma
                                elif m:
                                    cut = m.end()
                                else:
                                    break
                                sentence, buf = buf[:cut].strip(), buf[cut:].lstrip()
                                if sentence:
                                    queued_any = True
                                    await sentence_q.put(sentence)
                        elif ev_kind == "tool_start":
                            await self.send({"type": "tool", "turn": turn_id, "status": "running", **data})
                            filler = FILLERS[lang].get(data["tool"])
                            if filler and not queued_any:
                                queued_any = True
                                reply_parts.append(filler + " ")
                                await sentence_q.put(filler)
                        elif ev_kind == "tool":
                            tool_events.append(data)
                            await self.send({"type": "tool", "turn": turn_id, "status": "done",
                                             "tool": data["tool"], "args": data["args"]})
                    if buf.strip():
                        await sentence_q.put(buf.strip())
                    elif not "".join(reply_parts).strip():
                        reply_parts.append(FALLBACK[lang])
                        await sentence_q.put(FALLBACK[lang])
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # LLM/network failure -> apologise out loud
                    await self.send({"type": "info", "message": f"LLM error: {exc}"})
                    reply_parts.append(" " + FALLBACK[lang])
                    await sentence_q.put(FALLBACK[lang])
                finally:
                    metrics["llm_total_ms"] = int((time.perf_counter() - t_llm) * 1000)
                    sentence_q.put_nowait(None)

            async def synthesize() -> None:
                try:
                    while (sentence := await sentence_q.get()) is not None:
                        s_lang = "hi" if DEVANAGARI_RE.search(sentence) else lang
                        pcm, sr, ms = await asyncio.to_thread(self.tts.synthesize_pcm, sentence, s_lang)
                        metrics.setdefault("tts_first_ms", ms)
                        metrics["tts_total_ms"] = metrics.get("tts_total_ms", 0) + ms
                        if len(pcm):
                            await audio_q.put((sentence, pcm, sr))
                finally:
                    await audio_q.put(None)

            tasks = [asyncio.create_task(produce_text()), asyncio.create_task(synthesize())]

            sent_sec = 0.0
            while (item := await audio_q.get()) is not None:
                sentence, pcm, sr = item
                if not committed:
                    committed = True
                    self._commit_user_turn(kind, user_text, lang)
                if play_start is None:
                    play_start = time.perf_counter()
                    metrics["voice_to_voice_ms"] = int((play_start - t_ref) * 1000)
                    self.phase = "speaking"
                    await self.set_state("speaking")
                spoken.append((sentence, sent_sec))
                await self.send({"type": "agent_sentence", "turn": turn_id, "text": sentence,
                                 "language": "hi" if DEVANAGARI_RE.search(sentence) else lang})
                for frame, dur in chunk_pcm(pcm, sr, FRAME_MS):
                    now = time.perf_counter()
                    if now - play_start > sent_sec:  # underrun: client restarts the clock
                        play_start = now - sent_sec
                    ahead = sent_sec - (now - play_start)
                    if ahead > PLAYBACK_LEAD_SEC:
                        await asyncio.sleep(ahead - PLAYBACK_LEAD_SEC)
                    await self.send_audio(turn_id, sr, frame)
                    sent_sec += dur

            await asyncio.gather(*tasks)
            reply = "".join(reply_parts).strip()
            self.last_agent_text = reply
            if not committed:
                committed = True
                self._commit_user_turn(kind, user_text, lang)
            self._record_assistant(reply, lang, rag_sources)
            metrics["audio_sec"] = round(sent_sec, 2)
            await self.send({"type": "agent_done", "turn": turn_id, "text": reply, "language": lang,
                             "rag_sources": rag_sources, "tool_events": tool_events, "metrics": metrics})
            await self.send({"type": "metrics", **metrics})
            if tool_events:
                await self.send({"type": "intelligence", "lead": db.ensure_lead(self.session_id)})

            # 3) Stay in "speaking" until the caller has actually heard the audio.
            if play_start is not None:
                remaining = sent_sec - (time.perf_counter() - play_start)
                if remaining > 0:
                    await asyncio.sleep(remaining)
            self.phase = "idle"
            await self.set_state("listening" if self.in_call else "idle")

        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
            if committed:
                heard = (time.perf_counter() - play_start) if play_start else 0.0
                said = " ".join(s for s, start in spoken if start <= heard).strip()
                if said:
                    self.last_agent_text = said
                    self._record_assistant(said + " … [interrupted by caller]", lang, [])
            raise
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()

    # ------------------------------------------------------------ records --
    def _commit_user_turn(self, kind: str, user_text: str, lang: str) -> None:
        """The agent has started answering: the user turn is final (no more merging)."""
        if kind == "greeting":
            return
        self.pending_audio = None
        snapshot = list(self.history)
        self.history.append({"role": "user", "content": user_text})
        db.ensure_lead(self.session_id, language=lang)
        db.add_message(session_id=self.session_id, role="user", content=user_text, language=lang)
        self._spawn(self._intelligence(snapshot, user_text, lang))

    def _record_assistant(self, text: str, lang: str, rag_sources: List[dict]) -> None:
        if not text:
            return
        self.history.append({"role": "assistant", "content": text})
        db.add_message(session_id=self.session_id, role="assistant", content=text,
                       language="hi" if DEVANAGARI_RE.search(text) else lang,
                       rag_sources=rag_sources)

    async def _intelligence(self, history: List[Dict[str, str]], user_text: str, lang: str) -> None:
        try:
            lead = db.ensure_lead(self.session_id, language=lang)
            analysis = await asyncio.to_thread(self.insights.analyze_turn, history, user_text, lead)
            analysis["language"] = lang
            updated = self.crm.upsert_lead(self.session_id, analysis)
            await self.send({"type": "intelligence", "lead": updated})
        except Exception as exc:
            await self.send({"type": "info", "message": f"Insights unavailable: {exc}"})
