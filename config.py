# config.py
# Single source of truth for all configuration.
# Every other file imports from here — no more scattered os.getenv() calls.
#
# HOW TO USE IN OTHER FILES:
#   from config import MODEL_NAME, OLLAMA_URL, CHROMA_PATH
#
# TO CHANGE A SETTING:
#   Edit .env — config.py picks it up automatically on restart.

from dotenv import load_dotenv
import os
load_dotenv()

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_URL    = os.getenv("OLLAMA_URL",    "http://localhost:11434")
MODEL_NAME    = os.getenv("MODEL_NAME",    "iKhalid/ALLaM:7b")
EMBED_MODEL   = os.getenv("EMBED_MODEL",   "bge-m3:latest")

# ── Model Behavior ────────────────────────────────────────────────────────────
NUM_CTX       = int(os.getenv("NUM_CTX",       "4096"))
TEMPERATURE   = float(os.getenv("TEMPERATURE", "0.3"))
MAX_HISTORY   = int(os.getenv("MAX_HISTORY",   "10"))

# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_PATH     = os.getenv("CHROMA_PATH",     "./db/chroma") 
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "finance_concepts")

# ── Prompts ───────────────────────────────────────────────────────────────────
PROMPTS_DIR = os.getenv("PROMPTS_DIR", "./prompts")

# ── Data ──────────────────────────────────────────────────────────────────────
DOCS_PATH = os.getenv("DOCS_PATH", "./data/documents")   