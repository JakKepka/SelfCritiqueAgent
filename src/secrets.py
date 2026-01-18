import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

# Load .env file located in the repository root (one level up from src)
ROOT = Path(__file__).resolve().parents[1]
env_path = ROOT / ".env"
if load_dotenv is not None:
    load_dotenv(dotenv_path=env_path)
else:
    # fallback: rely on environment variables
    pass

# Expose expected variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = os.getenv("GEMINI_API_URL")

# https://aistudio.google.com/app/usage?timeRange=last-28-days&tab=rate-limit&project=gen-lang-client-0841350474

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
