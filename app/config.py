"""Centralized configuration for the Autonomous AI Sales Voice Agent."""

from pathlib import Path
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Groq API settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-120b")
GROQ_FAST_MODEL = os.getenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b")

# Local STT (faster-whisper on CPU)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
# Hindi needs a bigger model: Whisper-base writes Hindi in Urdu script. Set to "base" to disable.
WHISPER_HI_MODEL = os.getenv("WHISPER_HI_MODEL", "small")
WHISPER_CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", "6"))

# Voice-call turn taking (Silero VAD)
VAD_END_OF_TURN_MS = int(os.getenv("VAD_END_OF_TURN_MS", "650"))   # silence that ends a user turn
VAD_BARGE_IN_MS = int(os.getenv("VAD_BARGE_IN_MS", "400"))         # speech needed to interrupt the agent

# Local TTS (Piper ONNX on CPU)
PIPER_EN_MODEL = BASE_DIR / os.getenv("PIPER_EN_MODEL", "models/piper/en_US-lessac-medium.onnx")
PIPER_HI_MODEL = BASE_DIR / os.getenv("PIPER_HI_MODEL", "models/piper/hi_IN-rohan-medium.onnx")

# Paths
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
DATA_DIR = BASE_DIR / "data"
DB_PATH = Path(os.getenv("SALES_DB_PATH", str(DATA_DIR / "sales_agent.db")))
FRONTEND_DIR = BASE_DIR / "frontend"

# Optional external CRM webhook
CRM_WEBHOOK_URL = os.getenv("CRM_WEBHOOK_URL", "")
