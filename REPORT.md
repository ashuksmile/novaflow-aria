# Autonomous AI Sales Voice Agent — 1-Page Project Report

**Project:** NovaFlow AI — Autonomous B2B Sales Voice Agent  
**Prepared for:** Fortivex AI Private Limited (Round 3 & 4 Technical Evaluation)

---

## 1. Business Problem & Commercial Feasibility
B2B sales teams lose **35–50% of inbound leads** due to slow speed-to-lead (average response time > 4 hours), high SDR turnover, inconsistent BANT qualification, and manual CRM data entry. Hiring a 24/7 multilingual SDR team costs **$4,000–$6,000/month per rep** (or INR 50k–80k/month in India).

**NovaFlow AI** solves this with an autonomous, multilingual (English + Hindi) AI Sales Voice Agent that engages prospects in **under 2 seconds**, answers complex product/pricing/security questions via **RAG**, qualifies **BANT** + **Sentiment** in real time, **books demos automatically**, and syncs structured insights into the CRM.

* **Commercial Model:** SaaS tiers — Starter ($299/mo, 1,000 mins), Growth ($799/mo, 5,000 mins), Enterprise ($2,499+/mo).
* **Gross Margin:** At ~$0.015–$0.035/min COGS vs $0.16–$0.30/min blended revenue, the product achieves **>80% SaaS gross margins** with a 4–6 week customer payback period.

---

## 2. Technical Implementation (Working MVP)
* **Full-Duplex Voice Call (WebSocket):** Browser AudioWorklet streams continuous 16 kHz PCM; **Silero VAD** does hands-free turn-taking (650 ms end-of-turn), **barge-in** (caller can interrupt mid-sentence; playback flushed in < 0.5 s), pause merging and noise/echo rejection. LLM tokens are streamed into sentence-level TTS so Aria starts talking while still "thinking": **~0.85 s voice-to-voice (EN), ~1.8 s (HI)** measured.
* **Local CPU STT:** `faster-whisper` int8 — `base` for English (real-speech WER ≈ 0.08 on LibriSpeech), `small` for Hindi (CER ≈ 0.25 on FLEURS; `base` wrote Hindi in Urdu script), auto EN/HI detection.
* **Verified audio quality:** automated tests prove TTS output is intelligible speech (independent Whisper listener, VAD, spectral checks; noise controls must fail) and drive full calls through a real headless browser with a fake microphone.
* **Agentic LLM & Tool Calling:** Groq API (`openai/gpt-oss-120b`) executes structured function calls: `search_knowledge_base`, `check_available_slots`, `book_appointment`, and `lookup_returning_customer`.
* **Local RAG Engine:** `FastEmbed` (`BAAI/bge-small-en-v1.5` ONNX) + NumPy cosine similarity retrieves top-k product, pricing, security, ROI, and objection-handling chunks with live source citations.
* **Non-Blocking Intelligence:** A parallel async worker (`openai/gpt-oss-20b`, JSON mode) updates live BANT fields, Lead Score (0–100), Sentiment (-1.0 to +1.0), and objections without adding voice latency.
* **Local CPU TTS:** `Piper TTS` ONNX (`en_US-lessac-medium` & `hi_IN-rohan-medium`) synthesizes sentence-by-sentence PCM streamed in real time, with a speech normaliser (prices, %, times, English product names inside Hindi).
* **CRM & Post-Call Insights:** SQLite CRM (behind a pluggable `CRMAdapter` with webhook support for HubSpot/Salesforce) + automated executive call summaries and follow-up email drafts.

---

## 3. Scalability Strategy (Supporting 100,000 Users)
Assuming **100,000 total users** with **5–10% daily active** and **1,000–2,000 peak concurrent voice calls**:
1. **Media Transport:** Move the (already full-duplex, barge-in capable) WebSocket PCM stream to **WebRTC / LiveKit SFU** clusters and SIP trunking (Twilio/Exotel) for UDP transport, browser-grade echo cancellation and phone-number calling.
2. **Stateless Compute:** Containerize FastAPI orchestrators on **Kubernetes (EKS/GKE)** with HPA scaling on active WebRTC sessions; store short-term session memory in **Redis Cluster**.
3. **GPU AI Pool:** Offload STT (`faster-whisper` / Deepgram) and TTS (Piper/Cartesia) to autoscaling GPU inference pods (NVIDIA L4/T4; ~40–60 concurrent streams per GPU); use Groq LPU / vLLM with semantic caching for repeated FAQs.
4. **Data & Async Pipeline:** Migrate SQLite to **PostgreSQL (read replicas + connection pooling)**, local NumPy RAG to **Qdrant / pgvector** with tenant isolation (`tenant_id` metadata filters), and push post-call summaries/CRM webhooks through **Kafka / Redis Streams**.

---

## 4. Data Security, Infrastructure, Deployment & Operational Costs
* **Security & Compliance:** TLS 1.3 / SRTP in transit, AES-256 at rest, regex/NER **PII redaction** before LLM logging, Zero Data Retention (ZDR) inference agreements, RBAC/SSO, and regional residency (India DPDP Act 2023, GDPR, SOC 2 Type II).
* **Deployment:** Docker + GitHub Actions CI/CD -> Kubernetes with blue/green rollouts, Prometheus/Grafana latency SLAs (`p95 < 1.2s`), and OpenTelemetry tracing.
* **Estimated Operational Cost (per 1,000 call minutes at scale):**
  * **STT (Self-hosted GPU Whisper):** ~$1.50 – $2.50
  * **LLM (Groq gpt-oss-120b + gpt-oss-20b async):** ~$3.00 – $5.00
  * **TTS (Self-hosted Piper ONNX / GPU):** ~$0.50 – $1.50
  * **Infra, Vector DB, Storage & Bandwidth:** ~$4.00 – $6.00
  * **Total COGS:** **~$9 – $15 per 1,000 call minutes (~$0.012/min)** (plus ~$0.01/min if PSTN telephony is used).
