# System Architecture — Autonomous AI Sales Voice Agent

## 1. MVP Real-Time Voice Call Architecture (Local + Groq)

```mermaid
flowchart LR
    subgraph Client["Browser (Vanilla JS + Tailwind)"]
        MIC["getUserMedia (AEC/NS/AGC)<br/>AudioWorklet → 16 kHz PCM16<br/>32 ms frames, continuous"]
        SPK["Web Audio scheduler<br/>gapless playback,<br/>instant flush on barge-in"]
        UI["Call UI: state orb, live captions,<br/>latency chips, CRM dashboard"]
    end

    subgraph Call["VoiceCallSession (one per WebSocket)"]
        VAD["Silero VAD v6 (ONNX)<br/>speech_start / end-of-turn<br/>(650 ms silence)"]
        BARGE{"Agent talking and<br/>caller speaks ≥ 400 ms?"}
        STT["faster-whisper int8<br/>base (EN) · small (HI)<br/>noise / echo / hallucination filter"]
        AGENT["Groq gpt-oss-120b<br/>token stream + tool calls"]
        SPLIT["Sentence splitter<br/>+ spoken fillers during tools"]
        TTS["Piper ONNX (EN/HI)<br/>+ speech normaliser ($, %, CRM…)"]
        PACE["Real-time pacer<br/>(0.6 s jitter buffer)"]
    end

    subgraph Tools["Tools & Data"]
        RAG["FastEmbed RAG"]
        CRM["CRMAdapter (SQLite + webhook)"]
        INTEL["Async insights worker<br/>gpt-oss-20b: BANT, score, sentiment"]
    end

    MIC -- "binary PCM" --> VAD
    VAD -- "utterance" --> STT --> AGENT
    VAD -- "speech_start" --> BARGE
    BARGE -- "yes: cancel turn + 'interrupt'" --> SPK
    AGENT <--> RAG
    AGENT <--> CRM
    STT -. "committed turn" .-> INTEL --> CRM
    AGENT -- "tokens" --> SPLIT --> TTS --> PACE
    PACE -- "binary PCM + turn id" --> SPK
    Call -- "state, transcript, agent_sentence,<br/>metrics, intelligence (JSON)" --> UI
```

### Turn-taking rules
| Situation | Behaviour |
|---|---|
| Caller stops talking for 650 ms | End of turn → STT → answer (`VAD_END_OF_TURN_MS`) |
| Caller pauses, then continues before Aria speaks | Half-built answer is cancelled; both audio segments are transcribed together as one turn |
| Caller talks over Aria for ≥ 400 ms | **Barge-in**: server cancels LLM/TTS, client flushes queued audio within one frame; history stores only what was actually heard + `[interrupted by caller]` (`VAD_BARGE_IN_MS`) |
| Barge-in disabled (laptop speakers) | Speech during Aria's turn is ignored (half-duplex) |
| Noise, hum, cough, Whisper "Thank you." hallucination, Aria's own voice leaking back | Rejected (VAD, Whisper no-speech prob, hallucination list, echo-overlap check) |
| Tool call (RAG / calendar) | Aria says a short filler ("Let me check that for you") so the line is never silent |

---

## 2. Production Scalability Architecture (100,000+ Users)

```mermaid
flowchart TB
    subgraph Edge["Edge & Telephony Layer"]
        WEB["WebRTC Browser SDK"]
        TEL["SIP Trunking / Telephony<br/>(Twilio / Exotel / Plivo)"]
        LK["LiveKit / Media SFU Cluster<br/>(UDP/SRTP, Auto-scaled)"]
    end

    subgraph Compute["Stateless Application Tier (Kubernetes / HPA)"]
        GW["FastAPI / Voice Orchestrator Pods<br/>(Horizontal Pod Autoscaler)"]
        Q["Kafka / Redis Streams<br/>(Async Post-Call Analytics & Webhooks)"]
        WRK["Analytics & CRM Sync Workers<br/>(Summaries, HubSpot/Salesforce)"]
    end

    subgraph AI["AI Inference Tier"]
        STT_GPU["STT Microservice<br/>Whisper / Deepgram Streaming<br/>(GPU Autoscaling)"]
        LLM_API["LLM Router<br/>Groq LPU / vLLM Cluster<br/>(Semantic Cache + Fallback)"]
        TTS_GPU["Streaming TTS Cluster<br/>Piper / Cartesia / ElevenLabs"]
    end

    subgraph Storage["Distributed Data & Security Tier"]
        REDIS[("Redis Cluster<br/>Session State & Rate Limits")]
        PG[("PostgreSQL (Primary + Replicas)<br/>Leads, Bookings, Audit Logs")]
        VEC[("Qdrant / pgvector<br/>Multi-Tenant Knowledge Base")]
        S3[("Encrypted Object Storage (S3)<br/>AES-256 Audio & Transcripts")]
    end

    WEB & TEL --> LK
    LK --> GW
    GW <--> STT_GPU & LLM_API & TTS_GPU
    GW <--> REDIS & VEC & PG
    GW --> Q --> WRK
    WRK --> PG & S3
```

---

## 3. Voice-to-Voice Latency (measured on the MVP)

Measured by `tests/test_voice_call_e2e.py::test_live_groq_call` and the browser test (12-core laptop CPU, Groq cloud).
"Voice-to-voice" = moment the caller stops speaking (end-of-turn fired) → first agent audio leaves the server.
The 650 ms end-of-turn silence comes on top of it.

| Stage | English | Hindi | Production target (GPU / streaming) |
|---|---|---|---|
| End-of-turn detection (silence) | 650 ms | 650 ms | 300–500 ms + semantic end-of-turn model |
| STT | ~380–430 ms (Whisper base) | ~1.0–1.4 s (Whisper small, beam 5) | ~100–150 ms streaming GPU STT |
| LLM first token (Groq, reasoning_effort=low) | ~200–600 ms (1.1 s with a tool call) | ~1.2 s | ~150–250 ms |
| TTS first sentence (Piper) | ~45–80 ms | ~55 ms | ~50–80 ms |
| **Voice-to-voice** | **~0.83–0.85 s** | **~1.8 s** | **~0.5 s** |

With a tool call the filler sentence ("Let me check that for you") is spoken immediately, which keeps perceived latency at ~0.85 s even though the RAG + second LLM step takes ~1.1 s.
