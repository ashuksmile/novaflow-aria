"""FastAPI backend for the Autonomous AI Sales Voice Agent."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
from typing import Dict, List
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app import database as db
from app.agent import SalesVoiceAgent
from app.config import (
    FRONTEND_DIR,
    GROQ_CHAT_MODEL,
    VAD_BARGE_IN_MS,
    VAD_END_OF_TURN_MS,
    WHISPER_HI_MODEL,
    WHISPER_MODEL,
)
from app.crm import SQLiteCRMAdapter
from app.insights import SalesInsightsEngine
from app.rag import LocalRAGEngine
from app.stt import LocalSTTEngine
from app.tts import LocalTTSEngine
from app.voice_session import VoiceCallSession

rag_engine = LocalRAGEngine()
stt_engine = LocalSTTEngine()
tts_engine = LocalTTSEngine()
crm_adapter = SQLiteCRMAdapter()
insights_engine = SalesInsightsEngine()
sales_agent = SalesVoiceAgent(rag=rag_engine, crm=crm_adapter)

# In-memory short-term conversation buffers keyed by session_id
session_histories: Dict[str, List[Dict[str, str]]] = {}
loaded_voices: List[str] = []
indexed_chunks: int = 0


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global loaded_voices, indexed_chunks
    db.init_db()
    indexed_chunks = rag_engine.initialize()
    stt_engine.initialize()
    loaded_voices = tts_engine.initialize()
    yield


app = FastAPI(
    title="NovaFlow AI — Autonomous Sales Voice Agent",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


FRONTEND_DIST = FRONTEND_DIR / "dist"


@app.get("/")
async def index():
    built = FRONTEND_DIST / "index.html"
    if built.exists():
        return FileResponse(built)
    raise HTTPException(
        status_code=503,
        detail="Frontend bundle not built. Run: cd frontend && npm install && npm run build",
    )


@app.get("/assets/{asset_path:path}")
async def frontend_assets(asset_path: str):
    assets_root = (FRONTEND_DIST / "assets").resolve()
    target = (assets_root / asset_path).resolve()
    if not str(target).startswith(str(assets_root)) or not target.is_file():
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(target)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "llm_model": GROQ_CHAT_MODEL,
        "stt_model": f"faster-whisper-{WHISPER_MODEL}-int8",
        "stt_hindi_model": f"faster-whisper-{WHISPER_HI_MODEL}-int8",
        "vad": "silero-v6",
        "end_of_turn_ms": VAD_END_OF_TURN_MS,
        "barge_in_ms": VAD_BARGE_IN_MS,
        "tts_voices": loaded_voices,
        "rag_chunks": indexed_chunks,
    }


@app.get("/api/dashboard")
async def dashboard():
    return db.get_dashboard_data()


@app.get("/api/calendar")
async def calendar_view():
    from datetime import datetime, timedelta

    today = datetime.now().date()
    dates = [(today + timedelta(days=i)).isoformat() for i in range(1, 6)]
    data = db.get_dashboard_data()
    appointments = data["appointments"]
    booked_map = {
        (a["slot_date"], a["slot_time"]): a
        for a in appointments
        if a.get("status") == "Confirmed"
    }
    days = []
    for d in dates:
        slots = []
        for t in db.DEFAULT_TIME_SLOTS:
            appt = booked_map.get((d, t))
            slots.append(
                {
                    "time": t,
                    "status": "Booked" if appt else "Open",
                    "appointment": appt,
                }
            )
        days.append({"date": d, "slots": slots})
    return {"days": days, "appointments": appointments}


@app.get("/api/kb")
async def knowledge_base():
    return {
        "count": len(rag_engine.chunks),
        "chunks": [
            {"id": c.chunk_id, "source": c.source, "heading": c.heading}
            for c in rag_engine.chunks
        ],
    }


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    lead = db.ensure_lead(session_id)
    messages = db.get_messages(session_id)
    return {"lead": lead, "messages": messages}


@app.post("/api/sessions/{session_id}/summarize")
async def summarize_session(session_id: str):
    lead = db.ensure_lead(session_id)
    messages = db.get_messages(session_id)
    summary_data = await asyncio.to_thread(
        insights_engine.generate_call_summary, messages, lead
    )
    saved = db.save_call_summary(session_id, summary_data)
    return {"summary": saved, "lead": db.ensure_lead(session_id)}


@app.websocket("/ws/voice/{session_id}")
async def voice_websocket(ws: WebSocket, session_id: str):
    """Full-duplex voice call: continuous mic PCM in, streamed agent speech out."""
    await ws.accept()
    lead = db.ensure_lead(session_id)
    call = VoiceCallSession(
        ws,
        session_id,
        stt=stt_engine,
        tts=tts_engine,
        agent=sales_agent,
        crm=crm_adapter,
        insights=insights_engine,
        history=session_histories.setdefault(session_id, []),
    )
    await call.send({"type": "connected", "session_id": session_id, "lead": lead,
                     "input_sample_rate": 16000})
    try:
        while True:
            packet = await ws.receive()
            if packet.get("type") == "websocket.disconnect":
                break
            if packet.get("bytes") is not None:
                await call.on_audio(packet["bytes"])
            elif packet.get("text"):
                try:
                    msg = json.loads(packet["text"])
                except json.JSONDecodeError:
                    continue
                await call.on_json(msg)
    except WebSocketDisconnect:
        pass
    finally:
        await call.close()
