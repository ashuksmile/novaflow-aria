# NovaFlow AI — Autonomous AI Sales Voice Agent

A functional, real-time, multilingual (**English + Hindi**) Autonomous AI Sales Voice Agent built with **FastAPI**, **Silero VAD turn-taking**, **Local CPU Whisper STT (`faster-whisper`)**, **Groq LLM (`openai/gpt-oss-120b` & `openai/gpt-oss-20b`)**, **Local RAG (`FastEmbed`)**, **Local CPU Piper TTS (ONNX)**, and a live **SQLite CRM & Analytics Dashboard**.

## Key Features
- **Real voice call, not a chat box:** press *Start Call* and talk. The mic streams continuously (AudioWorklet, 16 kHz PCM over WebSocket); **Silero VAD** detects when you start and finish speaking (no push-to-talk); Aria answers out loud as the LLM is still generating (token stream → sentence → Piper TTS → paced PCM); and you can **interrupt her mid-sentence (barge-in)**. Pauses mid-thought are merged into one turn, background noise / hum / speaker echo is ignored, and the agent says "Let me check that for you" while tools run.
- **Measured latency:** ~0.85 s voice-to-voice in English (from end of speech to first audio), ~1.8 s in Hindi, on CPU + Groq.
- **Multilingual & Contextual Memory:** Automatic English/Hindi detection, rolling conversation memory, and returning customer lookup in CRM.
- **Intelligent Lead Qualification & Sentiment Analysis:** Non-blocking async worker tracks **BANT** (Budget, Authority, Need, Timeline), **Lead Score (0–100)**, **Sentiment (-1.0 to +1.0)**, and objections in real time.
- **RAG-Based Knowledge Retrieval:** Local `FastEmbed` (`BAAI/bge-small-en-v1.5`) semantic search over NovaFlow AI product, pricing, security, ROI, and objection-handling docs with live citations.
- **Automated Appointment Booking & CRM Integration:** Agentic tool calling checks open calendar slots, books demos, and updates a pluggable `CRMAdapter` (SQLite + optional HubSpot/Salesforce webhook).
- **Call Summarisation & Business Insights:** One-click post-call executive summary, pain points, sentiment trend, recommended next action, and ready-to-send follow-up email.

## Documentation
- **1-Page Project Report:** [`REPORT.md`](REPORT.md)
- **System Architecture & 100k Scalability Diagrams:** [`docs/architecture.md`](docs/architecture.md)

## Quick Start

### 1. Environment Setup
```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

### 2. Download Local Piper Voice Models (English + Hindi)
```bash
mkdir -p models/piper
.venv/bin/python -m piper.download_voices --download-dir models/piper en_US-lessac-medium hi_IN-rohan-medium
```

### 3. Configure `.env`
```bash
cp .env.example .env
# Add your GROQ_API_KEY in .env
```

### 4. Run the Server
```bash
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Open **http://localhost:8000**, press **Start Call**, allow the mic and just talk. Use headphones for the
cleanest barge-in (or untick *Allow interrupting Aria* when on laptop speakers).

### 5. Test that it really hears and really speaks
```bash
uv pip install --python .venv/bin/python -r requirements-dev.txt
.venv/bin/python -m scripts.fetch_test_audio   # once: real human EN/HI clips -> tests/fixtures (already committed)
.venv/bin/python -m pytest -s                  # ~3 min; prints what was said vs. what was heard
```
| Suite | What it proves |
|---|---|
| `tests/test_tts_quality.py` | Every TTS sample (EN + HI) is speech, not noise: Silero VAD speech ratio, spectral flatness, loudness, clipping, speaking rate **and** an independent Whisper-small listener transcribes it back to the intended words (EN WER ≤ 0.2, HI CER ≤ 0.35). White noise / hum / silence must *fail* the same checks. |
| `tests/test_stt.py` | STT on **real human recordings** (LibriSpeech EN, Google FLEURS HI): mean EN WER ≈ 0.08, HI CER ≈ 0.25; 10–20 dB background noise; 48 kHz stereo browser input; EN/HI auto-detection; no hallucinated text on silence. |
| `tests/test_vad.py` | Turn-taking on real speech: start/end detection, 300 ms pauses don't cut the speaker off, noise never triggers, short blips can't barge in. |
| `tests/test_voice_call_e2e.py` | A simulated caller streams mic audio over the WebSocket: greeting → question → spoken answer (transcribed back), barge-in stops the agent in < 0.5 s, mid-sentence pauses merge, noise ignored, plus a live Groq EN + Hindi call. |
| `tests/test_browser_call.py` | The real page in headless Chromium/Brave with a fake microphone: AudioWorklet capture → server → Web Audio playback; the audio played to the speaker is transcribed and compared with Aria's reply. |

Listen to what was generated in `tests/output/*.wav`.

## Project Structure
```text
├── app/
│   ├── main.py          # FastAPI REST + call WebSocket
│   ├── voice_session.py # Full-duplex call loop: VAD turn-taking, barge-in, streaming STT→LLM→TTS
│   ├── vad.py           # Streaming Silero VAD (end-of-turn + barge-in detection)
│   ├── agent.py         # Groq LLM Sales Agent (token streaming + function calling tools)
│   ├── stt.py           # Local CPU faster-whisper (base for EN, small for HI, noise rejection)
│   ├── tts.py           # Local CPU Piper ONNX TTS (en_US + hi_IN) + speech text normalisation
│   ├── audio_utils.py   # PCM/WAV, resampling, speech-quality metrics, WER/CER
│   ├── rag.py           # FastEmbed ONNX + NumPy cosine RAG engine
│   ├── insights.py      # Async BANT, sentiment, & post-call summary engine
│   ├── crm.py           # Pluggable CRMAdapter (SQLite + Webhook)
│   ├── database.py      # SQLite schema, calendar slots, & analytics queries
│   └── config.py        # Environment configuration
├── frontend/
│   └── index.html       # Live Voice Console & CRM Dashboard UI
├── knowledge_base/      # Markdown product, pricing, security, ROI & objection docs
├── tests/               # Audio quality, STT, VAD, call e2e & browser tests (+ real speech fixtures)
├── scripts/fetch_test_audio.py
├── docs/architecture.md # Mermaid architecture diagrams & latency budget
└── REPORT.md            # Mandatory 1-page project report
```
