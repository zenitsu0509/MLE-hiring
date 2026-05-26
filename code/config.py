"""
config.py — Central configuration for the MLE Support Triage Agent.
All tunables in one place. Secrets come from environment variables only.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env from repo root ──────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

# ── API Keys (from environment only, never hardcoded) ─────────────────────
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEYS", "")

# ── Model Configuration ──────────────────────────────────────────────────
# Primary: OpenRouter (change OPENROUTER_MODEL in .env to swap models)
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash:free")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Fallback: Gemini (if OpenRouter fails)
GEMINI_MODEL = "gemini-2.5-flash"

TEMPERATURE = 0                          # Determinism: always 0
MAX_OUTPUT_TOKENS = 4096

# ── Retriever Configuration ──────────────────────────────────────────────
TOP_K_RETRIEVAL = 5                      # Number of chunks to retrieve
CHUNK_SIZE_WORDS = 400                   # Words per chunk
CHUNK_OVERLAP_WORDS = 50                 # Overlap between chunks
MIN_BM25_SCORE = 0.1                     # Minimum score to include result

# ── Paths ────────────────────────────────────────────────────────────────
DATA_DIR = str(REPO_ROOT / "data")
SUPPORT_TICKETS_PATH = str(REPO_ROOT / "support_tickets" / "support_tickets.csv")
OUTPUT_CSV_PATH = str(REPO_ROOT / "support_tickets" / "output.csv")
API_SPECS_PATH = str(REPO_ROOT / "data" / "api_specs" / "internal_tools.json")

# ── Retry Configuration ─────────────────────────────────────────────────
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2                     # seconds; exponential: 2, 4, 8

# ── Domain Mapping ───────────────────────────────────────────────────────
COMPANY_TO_DOMAIN = {
    "devplatform": "devplatform",
    "claude": "claude",
    "visa": "visa",
}

# ── Valid Enum Values (from validate_output.py) ──────────────────────────
VALID_STATUS = {"replied", "escalated"}
VALID_REQUEST_TYPE = {"product_issue", "feature_request", "bug", "invalid"}
VALID_RISK_LEVEL = {"low", "medium", "high", "critical"}

# ── File Extensions to Index ─────────────────────────────────────────────
INDEXABLE_EXTENSIONS = {".md", ".txt", ".json"}
