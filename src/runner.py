#!/usr/bin/env python3
"""Prosty lokalny runner prototypu pipeline: Agent1 -> Agent2.

To nie wykonuje wywołań LLM — generuje prostą, deterministyczną odpowiedź
opartą na tagowym retrievalu paremii, aby przetestować format i pipeline.
"""
import argparse
import json
import os
import re
import importlib.util
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import logging

from google import genai
from utils import load_jsonl as utils_load_jsonl, format_paremie_text as utils_format_paremie_text, retrieve_top_k as utils_retrieve_top_k
from llm import call_gemini_genai as llm_call_gemini_genai
from parsers import parse_agent1_output as parsers_parse_agent1_output, parse_agent2_output as parsers_parse_agent2_output, compare_agent1_with_gold as parsers_compare_agent1_with_gold, compare_agent2_with_gold as parsers_compare_agent2_with_gold
from storage import save_case_results as storage_save_case_results


try:
    import requests
    import yaml
    
except Exception:
    requests = None
    yaml = None
    genai = None

# configure logging
logging.basicConfig(level=os.environ.get("SCAFFOLD_LOG_LEVEL", "INFO"))
logger = logging.getLogger("selfcritique")

# load project secrets from src/secrets.py (to avoid colliding with stdlib `secrets`)
def _load_project_secrets():
    spec_path = Path(__file__).resolve().parent / "secrets.py"
    print("Loading project secrets from:", spec_path)
    if not spec_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("project_secrets", str(spec_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

project_secrets = _load_project_secrets()

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PAREMIE_FILE = DATA_DIR / "paremie.jsonl"
CASES_FILE = DATA_DIR / "cases.jsonl"


def load_jsonl(path: Path) -> List[Dict]:
    return utils_load_jsonl(path)


def tag_score(paremia: Dict, case_tags: List[str]) -> int:
    # prosty score: liczba wspólnych tagów
    return sum(1 for t in paremia.get("tags", []) if t in case_tags)


def retrieve_top_k(paremie: List[Dict], case_tags: List[str], k: int = 3) -> List[Dict]:
    scored = [(tag_score(p, case_tags), p) for p in paremie]
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [p for s, p in scored if s > 0][:k]
    # jeśli brak dopasowań, zwróć 3 pierwsze
    if not top:
        return paremie[:k]
    return top


def load_yaml_template(path: Path) -> Dict:
    if yaml is None:
        raise RuntimeError("PyYAML not available")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_paremie_text(paremie_list: List[Dict]) -> str:
    parts = []
    for p in paremie_list:
        pid = p.get('id') or p.get('code') or p.get('key') or 'UNKNOWN'
        head = p.get('polish') or p.get('text') or p.get('title') or p.get('label') or ''
        body = p.get('meaning') or p.get('explanation') or p.get('description') or ''
        if not head and body:
            head = (body[:120] + '...') if len(body) > 120 else body
            body = ''
        if not head and not body:
            # fallback to compact JSON representation
            try:
                head = json.dumps(p, ensure_ascii=False)
            except Exception:
                head = str(p)
        if body:
            parts.append(f"{pid}: {head} — {body}")
        else:
            parts.append(f"{pid}: {head}")
    return "\n".join(parts)


def call_gemini_genai(prompt_text: str) -> Optional[str]:
    """Call Gemini via official genai client.

    Uses `GEMINI_API_KEY` from `src/secrets.py` (already loaded) or env.
    Model can be set via env var `GEMINI_MODEL` (default: gemini-3-flash-preview).
    """
    if genai is None:
        logger.error("genai package not available")
        return None
    
    # ensure API key env var set for client
    api_key = None
    if project_secrets is not None:
        api_key = getattr(project_secrets, "GEMINI_API_KEY", None)
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("Brakuje GEMINI_API_KEY dla genai client")
        return None
    
    os.environ["GEMINI_API_KEY"] = api_key
    client = genai.Client()
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    try:
        response = client.models.generate_content(model=model_name, contents=prompt_text)
        # response has .text in example
        text = getattr(response, "text", None)
        if text:
            return text
        return str(response)
    except Exception as e:
        logger.error("Błąd wywołania genai.Client(): %s", e)
        return None


def _extract_json_substring(text: str) -> Optional[str]:
    return None


def parse_agent1_output(text: str) -> Dict:
    return parsers_parse_agent1_output(text)


def parse_agent2_output(text: str) -> Dict:
    return parsers_parse_agent2_output(text)


def compare_agent1_with_gold(case: Dict, parsed: Dict) -> Dict:
    gold_decision = case.get("label")
    pred_decision = parsed.get("decision")
    decision_match = (pred_decision == gold_decision)
    gold = set(case.get("gold_paremie", [])) if case.get("gold_paremie") else set()
    pred = set(parsed.get("used_paremie", []))
    tp = pred & gold
    precision = len(tp) / len(pred) if pred else 0.0
    recall = len(tp) / len(gold) if gold else 0.0
    return {"decision_match": decision_match, "pred_decision": pred_decision, "gold_decision": gold_decision, "precision": precision, "recall": recall, "tp": list(tp), "pred": list(pred), "gold": list(gold)}


def save_case_results(case_id: str, data: Dict):
    """Save agent outputs, parsed data and metrics under results/{case_id}/"""
    results_root = ROOT / "results"
    case_dir = results_root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    # save raw texts
    if "agent1_raw" in data and data["agent1_raw"] is not None:
        (case_dir / "agent1_raw.txt").write_text(data["agent1_raw"], encoding="utf-8")
    if "agent2_raw" in data and data["agent2_raw"] is not None:
        (case_dir / "agent2_raw.txt").write_text(data["agent2_raw"], encoding="utf-8")
    # save parsed json
    if "agent1_parsed" in data:
        with open(case_dir / "agent1_parsed.json", "w", encoding="utf-8") as f:
            json.dump(data["agent1_parsed"], f, ensure_ascii=False, indent=2)
    if "agent2_parsed" in data:
        with open(case_dir / "agent2_parsed.json", "w", encoding="utf-8") as f:
            json.dump(data["agent2_parsed"], f, ensure_ascii=False, indent=2)
    # save metrics
    if "metrics" in data:
        meta = data["metrics"].copy()
        meta.setdefault("saved_at", datetime.utcnow().isoformat() + "Z")
        with open(case_dir / "metrics_agent1.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    # save agent2 post-correction metrics if present
    if "metrics_agent2" in data:
        try:
            ma2 = data["metrics_agent2"].copy()
        except Exception:
            ma2 = data["metrics_agent2"]
        if isinstance(ma2, dict):
            ma2.setdefault("saved_at", datetime.utcnow().isoformat() + "Z")
        with open(case_dir / "metrics_agent2.json", "w", encoding="utf-8") as f:
            json.dump(ma2, f, ensure_ascii=False, indent=2)


def run_case_llm(case_id: str):
    paremie = load_jsonl(PAREMIE_FILE)
    cases = load_jsonl(CASES_FILE)
    case_map = {c["id"]: c for c in cases}
    case = case_map.get(case_id)
    if not case:
        logger.error("Case %s not found. Available: %s", case_id, ', '.join(case_map.keys()))
        return

    # load templates
    tpl1 = load_yaml_template(Path(__file__).resolve().parents[1] / "prompts" / "agent1_template.yaml")
    tpl2 = load_yaml_template(Path(__file__).resolve().parents[1] / "prompts" / "agent2_template.yaml")

    paremie_text = format_paremie_text(paremie)
    # Use safe replacement to avoid interpreting braces in example JSON inside the prompt
    prompt1 = tpl1.get("prompt", "").replace("{case_facts}", case.get("facts", "")).replace("{paremie}", paremie_text)
    logger.info("= Sprawa: %s - %s (LLM mode)", case['id'], case.get('title'))

    a1_raw = call_gemini_genai(prompt1)
    if a1_raw is None:
        logger.error("Nie udało się uzyskać odpowiedzi od Gemini. Kończę.")
        return

    logger.info("-- Agent 1 (LLM) raw output --")
    logger.info("%s", a1_raw)

    # parse Agent1 output (safe JSON-extract or heuristics)
    parsed_a1 = parse_agent1_output(a1_raw)
    logger.info("-- Agent 1 parsed output --")
    logger.info("Decision: %s", parsed_a1.get('decision'))
    logger.info("Used paremie: %s", parsed_a1.get('used_paremie'))
    logger.info("Assumptions: %s", parsed_a1.get('assumptions'))

    # compare with gold (Agent1)
    metrics = parsers_compare_agent1_with_gold(case, parsed_a1)
    logger.info("-- Automatic evaluation (Agent1 vs gold) --")
    logger.info("Decision match: %s (pred=%s, gold=%s)", metrics['decision_match'], metrics['pred_decision'], metrics['gold_decision'])
    logger.info("Paremie precision: %.2f, recall: %.2f", metrics['precision'], metrics['recall'])
    logger.info("True positives: %s", metrics['tp'])
    # if parsing failed (no decision or no paremie), try a forced JSON-generation retry
    if parsed_a1.get('decision') is None or not parsed_a1.get('used_paremie'):
        logger.warning('Agent1 output did not contain structured data; attempting forced JSON retry...')
        forced = {
            "case_facts": case.get("facts", ""),
            "paremie": paremie_text,
        }
        forced_prompt = (
            "Na podstawie poniższych faktów i listy paremii wygeneruj jedną odpowiedź w formacie JSON. "
            "Zwróć jedynie poprawny JSON, np. {\"Decyzja\": \"zasadne\", \"Uzasadnienie\": [...], \"Paremie użyte\": [\"P01\"], \"Założenia\": [...] }. "
            "Sprawa: \n" + forced['case_facts'] + "\nLista paremii:\n" + forced['paremie']
        )
        a1_raw2 = call_gemini_genai(forced_prompt)
        if a1_raw2:
            logger.info('-- Agent 1 (LLM) forced retry raw output --')
            logger.info('%s', a1_raw2)
            parsed_a1 = parse_agent1_output(a1_raw2)
            logger.info('-- Agent 1 parsed output (after retry) --')
            logger.info('Decision: %s', parsed_a1.get('decision'))
            logger.info('Used paremie: %s', parsed_a1.get('used_paremie'))
            metrics = compare_agent1_with_gold(case, parsed_a1)
            logger.info('Paremie precision: %.2f, recall: %.2f', metrics['precision'], metrics['recall'])

    # Agent 2: build prompt including Agent1 response
    prompt2 = tpl2.get("prompt", "").replace("{case_facts}", case.get("facts", "")).replace("{agent1_response}", a1_raw or "").replace("{paremie}", paremie_text)
    a2_raw = call_gemini_genai(prompt2)
    if a2_raw is None:
        logger.error("Nie udało się uzyskać odpowiedzi krytyka od Gemini. Kończę.")
        return

    logger.info("-- Agent 2 (LLM) raw output --")
    logger.info('%s', a2_raw)

    # parse Agent2 output
    parsed_a2 = parse_agent2_output(a2_raw)
    logger.info("-- Agent 2 parsed output --")
    logger.info("Rubric: %s", parsed_a2.get('rubric'))
    logger.info("Errors: %s", parsed_a2.get('errors'))
    if parsed_a2.get('suggested_fix'):
        logger.info("Suggested fix: %s", parsed_a2.get('suggested_fix'))
    # retry Agent2 forcing JSON if parser found nothing useful
    if not parsed_a2.get('rubric'):
        logger.warning('Agent2 output did not contain structured rubric; attempting forced JSON retry...')
        forced2 = (
            "Na podstawie poniższych faktów i odpowiedzi Agenta 1 wygeneruj jedną ocenę krytyczną w formacie JSON. "
            "Zwróć jedynie poprawny JSON, np. {\"Rubryka\": {...}, \"Lista błędów\": [...], \"Proponowana poprawka\": \"...\"}. "
            "Fakty: \n" + case.get('facts', '') + "\nOdpowiedź Agenta1:\n" + (a1_raw or '') + "\nLista paremii:\n" + paremie_text
        )
        a2_raw2 = call_gemini_genai(forced2)
        if a2_raw2:
            logger.info('-- Agent 2 (LLM) forced retry raw output --')
            logger.info('%s', a2_raw2)
            parsed_a2 = parse_agent2_output(a2_raw2)
            logger.info('-- Agent 2 parsed output (after retry) --')
            logger.info('Rubric: %s', parsed_a2.get('rubric'))
            logger.info('Errors: %s', parsed_a2.get('errors'))

    # compute Agent2-effect metrics (inferred corrections)
    metrics_agent2 = parsers_compare_agent2_with_gold(case, parsed_a1, parsed_a2)

    # finalize and save per-case results
    a1_raw_final = a1_raw2 if ('a1_raw2' in locals() and a1_raw2) else a1_raw
    a2_raw_final = a2_raw2 if ('a2_raw2' in locals() and a2_raw2) else a2_raw
    # enrich parsed Agent2 with recommended paremies inferred from post-correction metrics
    try:
        parsed_a2_enriched = dict(parsed_a2) if isinstance(parsed_a2, dict) else {"raw": parsed_a2}
        parsed_a2_enriched.setdefault("recommended_paremie", metrics_agent2.get("pred_after", []))
    except Exception:
        parsed_a2_enriched = parsed_a2

    results_payload = {
        "agent1_raw": a1_raw_final,
        "agent1_parsed": parsed_a1,
        "agent2_raw": a2_raw_final,
        "agent2_parsed": parsed_a2_enriched,
        "metrics": metrics,
        "metrics_agent2": metrics_agent2,
    }
    try:
        save_case_results(case_id, results_payload)
        logger.info("Saved results to results/%s/", case_id)
    except Exception as e:
        logger.error("Błąd zapisu wyników: %s", e)



def naive_agent1(case: Dict, paremie_list: List[Dict]) -> Dict:
    # retrieve
    used = retrieve_top_k(paremie_list, case.get("tags", []), k=3)
    # simple decision: use gold label if exists (to allow pipeline test)
    decision = case.get("label", "nie_dazy")
    # build explanation points from paremii
    uzasadnienie = []
    for i, p in enumerate(used, start=1):
        uzasadnienie.append(f"{i}. Zastosowano {p['id']} ({p['polish']}): {p['meaning']}")

    result = {
        "decision": decision,
        "uzasadnienie": uzasadnienie,
        "used_paremie": [p["id"] for p in used],
        "assumptions": ["Brak dodatkowych informacji; odpowiedź oparta na dostarczonym opisie faktycznym."]
    }
    return result


def naive_agent2(agent1_out: Dict, case: Dict) -> Dict:
    # compare used paremie with gold
    used = set(agent1_out.get("used_paremie", []))
    gold = set(case.get("gold_paremie", []))
    tp = used & gold
    precision = len(tp) / max(len(used), 1)

    # simple rubric scoring
    rubric = {
        "Dobór paremii": int(round(5 * precision)),
        "Interpretacja paremii": 4 if agent1_out.get("uzasadnienie") else 2,
        "Aplikacja": 4 if agent1_out.get("assumptions") else 3,
        "Pominięcia": 0 if gold - used else 5,
        "Nadmierna pewność": 5 if "Brak" in " ".join(agent1_out.get("assumptions", [])) else 4,
    }

    errors = []
    missing = list(gold - used)
    if missing:
        errors.append({"pominięcie": f"Brakuje paremi: {', '.join(missing)}"})
    extra = list(used - gold)
    if extra:
        errors.append({"dobór": f"Nadmiarowe paremie: {', '.join(extra)}"})

    out = {
        "rubric": rubric,
        "errors": errors,
        "suggested_fix": "Uzupełnić listę paremii o brakujące reguły i doprecyzować zastrzeżenia w uzasadnieniu."
    }
    return out


def run_case(case_id: str):
    paremie = load_jsonl(PAREMIE_FILE)
    cases = load_jsonl(CASES_FILE)
    case_map = {c["id"]: c for c in cases}
    case = case_map.get(case_id)
    if not case:
        print(f"Case {case_id} not found. Available: {', '.join(case_map.keys())}")
        return

    logger.info("= Sprawa: %s - %s", case['id'], case.get('title'))
    a1 = naive_agent1(case, paremie)
    logger.info("-- Agent 1 (generator) --")
    logger.info("Decyzja: %s", a1['decision'])
    logger.info("Uzasadnienie:")
    for p in a1['uzasadnienie']:
        logger.info("%s", p)
    logger.info("Paremie użyte: %s", ", ".join(a1['used_paremie']))
    logger.info("Założenia:")
    for a in a1['assumptions']:
        logger.info("- %s", a)

    a2 = naive_agent2(a1, case)
    logger.info("-- Agent 2 (krytyk) --")
    logger.info("Rubryka:")
    for k, v in a2['rubric'].items():
        logger.info("- %s: %s/5", k, v)
    if a2['errors']:
        logger.info("Lista błędów:")
        for e in a2['errors']:
            for t, m in e.items():
                logger.info("- %s: %s", t, m)
    logger.info("Sugerowana poprawka:")
    logger.info("%s", a2['suggested_fix'])
    # save naive-run results as well
    try:
        metrics = parsers_compare_agent1_with_gold(case, {"decision": a1.get("decision"), "used_paremie": a1.get("used_paremie")})
        metrics_agent2 = parsers_compare_agent2_with_gold(case, {"decision": a1.get("decision"), "used_paremie": a1.get("used_paremie")}, a2)
        try:
            a2_enriched = dict(a2) if isinstance(a2, dict) else {"raw": a2}
            a2_enriched.setdefault("recommended_paremie", metrics_agent2.get("pred_after", []))
        except Exception:
            a2_enriched = a2

        payload = {
            "agent1_raw": None,
            "agent1_parsed": a1,
            "agent2_raw": None,
            "agent2_parsed": a2_enriched,
            "metrics": metrics,
            "metrics_agent2": metrics_agent2,
        }
        save_case_results(case["id"], payload)
        logger.info("Saved results to results/%s/", case['id'])
    except Exception as e:
        logger.error("Błąd zapisu wyników (naive): %s", e)


def list_cases():
    cases = load_jsonl(CASES_FILE)
    for c in cases:
        logger.info("%s: %s (%s)", c['id'], c.get('title'), c.get('label'))


def main():
    parser = argparse.ArgumentParser(description="Runner prototypu SelfCritiqueAgent")
    parser.add_argument("command", choices=["list", "run"], help="list/run")
    parser.add_argument("case_id", nargs="?", help="ID sprawy, np. C01")
    parser.add_argument("--use-gemini", action="store_true", help="Użyć Gemini API (wymaga ustawionego GEMINI_API_KEY i GEMINI_API_URL)")
    args = parser.parse_args()
    
    if args.command == "list":
        list_cases()
    elif args.command == "run":
        if not args.case_id:
            logger.error("Podaj case_id, np.: run C01")
            return
        # choose pipeline: naive vs LLM-based
        if args.use_gemini:
            if requests is None or yaml is None:
                logger.error("Brakuje zależności 'requests' lub 'PyYAML'. Zainstaluj je: pip install -r requirements.txt")
                return
            run_case_llm(args.case_id)
        else:
            run_case(args.case_id)


if __name__ == "__main__":
    main()
