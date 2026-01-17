"""LLM client wrapper (genai)"""
import os
import logging
from typing import Optional

logger = logging.getLogger("selfcritique.llm")

try:
    from google import genai
except Exception:
    genai = None


def call_gemini_genai(prompt_text: str) -> Optional[str]:
    if genai is None:
        logger.error("genai package not available")
        return None
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        # try project secrets file if user configured
        try:
            from secrets import GEMINI_API_KEY
            api_key = GEMINI_API_KEY
        except Exception:
            pass
    if not api_key:
        logger.error("Brakuje GEMINI_API_KEY dla genai client")
        return None
    os.environ["GEMINI_API_KEY"] = api_key
    client = genai.Client()
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    try:
        response = client.models.generate_content(model=model_name, contents=prompt_text)
        text = getattr(response, "text", None)
        if text:
            return text
        return str(response)
    except Exception as e:
        logger.error("Błąd wywołania genai.Client(): %s", e)
        return None
