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

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")