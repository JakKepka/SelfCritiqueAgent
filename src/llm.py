"""LLM client wrapper (genai)"""
import os
import logging
from typing import Optional

logger = logging.getLogger("selfcritique.llm")

try:
    from google import genai
    import openai
except Exception:
    genai = None
    openai = None


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


def call_openai_gpt(prompt_text: str) -> Optional[str]:
    """Call OpenAI GPT via `openai` package (ChatCompletion).

    Uses `OPENAI_API_KEY` from `src/secrets.py` or env. Model via `OPENAI_MODEL` env var.
    """
    if openai is None:
        logger.error("openai package not available")
        return None
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        try:
            from secrets import OPENAI_API_KEY
            api_key = OPENAI_API_KEY
        except Exception:
            pass
    if not api_key:
        logger.error("Brakuje OPENAI_API_KEY dla OpenAI client")
        return None
    openai.api_key = api_key
    from secrets import OPENAI_MODEL
    model_name = OPENAI_MODEL
    try:
        resp = openai.ChatCompletion.create(model=model_name, messages=[{"role": "user", "content": prompt_text}])
        if resp and resp.get("choices"):
            ch = resp["choices"][0]
            msg = ch.get("message") or ch.get("text")
            if isinstance(msg, dict):
                return msg.get("content")
            return msg
        return str(resp)
    except Exception as e:
        logger.error("Błąd wywołania OpenAI API: %s", e)
        return None


def call_llm(prompt_text: str, provider: Optional[str] = None) -> Optional[str]:
    """Dispatch to configured LLM provider. Provider overrides env var if given."""
    prov = (provider or os.environ.get("LLM_PROVIDER") or "gemini").lower()

    if prov == "openai":
        return call_openai_gpt(prompt_text)
    return call_gemini_genai(prompt_text)
