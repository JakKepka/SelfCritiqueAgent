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
from llm import call_llm
from parsers import parse_agent1_output as parsers_parse_agent1_output, parse_agent2_output as parsers_parse_agent2_output, compare_agent1_with_gold as parsers_compare_agent1_with_gold, compare_agent2_with_gold as parsers_compare_agent2_with_gold, score_justification_against_gold as parsers_score_justification
from storage import save_case_results as storage_save_case_results


try:
    import requests
    import yaml
    import openai
    
except Exception:
    requests = None
    yaml = None
    genai = None
    openai = None

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

# feature flag: whether to check justification against gold
CHECK_JUSTIFICATION = True
# LLM provider flag: 'gemini' or 'openai'
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")


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
        # if justification scores present, save separately for easy aggregation
        just = meta.get("justification")
        if just is not None:
            try:
                jd = just.copy() if isinstance(just, dict) else just
            except Exception:
                jd = just
            if isinstance(jd, dict):
                jd.setdefault("saved_at", datetime.utcnow().isoformat() + "Z")
            with open(case_dir / "metrics_decisions.json", "w", encoding="utf-8") as f:
                json.dump(jd, f, ensure_ascii=False, indent=2)
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

    a1_raw = call_llm(prompt1)
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
        a1_raw2 = call_llm(forced_prompt)
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
    a2_raw = call_llm(prompt2)
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
        a2_raw2 = call_llm(forced2)
        if a2_raw2:
            logger.info('-- Agent 2 (LLM) forced retry raw output --')
            logger.info('%s', a2_raw2)
            parsed_a2 = parse_agent2_output(a2_raw2)
            logger.info('-- Agent 2 parsed output (after retry) --')
            logger.info('Rubric: %s', parsed_a2.get('rubric'))
            logger.info('Errors: %s', parsed_a2.get('errors'))

    # compute Agent2-effect metrics (inferred corrections)
    metrics_agent2 = parsers_compare_agent2_with_gold(case, parsed_a1, parsed_a2)

    # optional: score justifications vs gold
    justification_metrics = {}
    if CHECK_JUSTIFICATION:
        just_a1 = parsers_score_justification(case, parsed_a1, key_pred='uzasadnienie')
        # prefer Uzasadnienie_sadowe or uzasadnienie_poprawione for agent2
        just_a2 = parsers_score_justification(case, parsed_a2, key_pred='Uzasadnienie_sadowe')
        if just_a2.get('score') is None:
            just_a2 = parsers_score_justification(case, parsed_a2, key_pred='uzasadnienie_poprawione')
        justification_metrics = {"agent1": just_a1, "agent2": just_a2}
        # attach to metrics
        metrics["justification"] = justification_metrics
        # If LLM available, optionally call a dedicated judge prompt to get LLM scores
        try:
            judge_tpl_path = Path(__file__).resolve().parents[1] / "prompts" / "agent_judge_template.yaml"
            if judge_tpl_path.exists():
                tpl_j = load_yaml_template(judge_tpl_path)
                # helper to obtain justification text from parsed dict or its raw_text/raw
                def _just_text_from_parsed(parsed, preferred_keys=None):
                    preferred_keys = preferred_keys or ["Uzasadnienie_sadowe", "uzasadnienie_poprawione", "Uzasadnienie_sądowe", "uzasadnienie_sadowe", "uzasadnienie"]
                    if not parsed:
                        return ""
                    # if parsed is string
                    if isinstance(parsed, str):
                        return parsed
                    # check keys
                    for k in preferred_keys:
                        v = parsed.get(k)
                        if v:
                            if isinstance(v, str):
                                return v
                            if isinstance(v, list):
                                return "\n".join(v)
                    # try raw_text / raw fields
                    for rk in ("raw_text", "raw", "raw_response", "raw_output"):
                        rv = parsed.get(rk)
                        if not rv or not isinstance(rv, str):
                            continue
                        # try parse JSON substring
                        j = None
                        try:
                            import json as _json
                            m = re.search(r"\{[\s\S]*\}", rv)
                            if m:
                                j = _json.loads(m.group(0))
                        except Exception:
                            j = None
                        if isinstance(j, dict):
                            for k in preferred_keys:
                                if k in j and j.get(k):
                                    v = j.get(k)
                                    if isinstance(v, str):
                                        return v
                                    if isinstance(v, list):
                                        return "\n".join(v)
                    return ""

                gold_text = "\n".join(case.get("gold_uzasadnienie", [])) if case.get("gold_uzasadnienie") else (case.get("gold_explanation") or case.get("gold_uzasad") or case.get("gold_rationale") or "")
                a1_text = _just_text_from_parsed(parsed_a1, preferred_keys=["uzasadnienie", "Uzasadnienie"]) 
                a2_text = _just_text_from_parsed(parsed_a2)
                judge_prompt = tpl_j.get("prompt", "").replace("{gold_uzasadnienie}", gold_text).replace("{agent1_uzasadnienie}", a1_text).replace("{agent2_uzasadnienie}", a2_text)
                judge_raw = call_llm(judge_prompt)
                logger.info("-- LLM Judge raw output --")
                logger.info("%s", judge_raw)
                if judge_raw:
                    # try JSON parse
                    try:
                        jr = json.loads(judge_raw)
                    except Exception:
                        # try to extract simple JSON substring
                        m = re.search(r"\{.*\}", judge_raw, re.S)
                        if m:
                            try:
                                jr = json.loads(m.group(0))
                            except Exception:
                                jr = {"raw": judge_raw}
                        else:
                            jr = {"raw": judge_raw}
                    # normalize expected fields
                    agent1_score = jr.get("agent1_score") if isinstance(jr.get("agent1_score"), int) else None
                    agent2_score = jr.get("agent2_score") if isinstance(jr.get("agent2_score"), int) else None
                    agent1_comment = jr.get("agent1_comment") or jr.get("agent1_commentary") or ""
                    agent2_comment = jr.get("agent2_comment") or jr.get("agent2_commentary") or ""
                    metrics.setdefault("justification", {})
                    metrics["justification"]["llm_judge"] = {
                        "agent1_score": agent1_score,
                        "agent2_score": agent2_score,
                        "agent1_comment": agent1_comment,
                        "agent2_comment": agent2_comment,
                        "raw": judge_raw
                    }
        except Exception as e:
            # non-fatal: skip judge if anything goes wrong
            logger.error("Błąd wywołania LLM Judge: %s", e)
            pass

    # finalize and save per-case results
    a1_raw_final = a1_raw2 if ('a1_raw2' in locals() and a1_raw2) else a1_raw
    a2_raw_final = a2_raw2 if ('a2_raw2' in locals() and a2_raw2) else a2_raw
    # enrich parsed Agent2 with recommended paremies inferred from post-correction metrics
    try:
        parsed_a2_enriched = dict(parsed_a2) if isinstance(parsed_a2, dict) else {"raw": parsed_a2}
        parsed_a2_enriched.setdefault("recommended_paremie", metrics_agent2.get("pred_after", []))
        # ensure Agent2 file contains a decision field (agreeing or overriding Agent1)
        parsed_a2_enriched.setdefault("decision", parsed_a1.get("decision"))
        # Ensure court-style justification is preserved if present in raw parse or raw_text
        try:
            # possible field names to look for
            candidate_fields = ["Uzasadnienie_sadowe", "uzasadnienie_poprawione", "Uzasadnienie_sądowe", "uzasadnienie_sadowe"]
            # helper to try to extract JSON from a raw string
            def _extract_json_from_text(s: str):
                if not s or not isinstance(s, str):
                    return None
                # direct json
                try:
                    return json.loads(s)
                except Exception:
                    pass
                # find first {...} that looks like JSON
                m = re.search(r"\{[\s\S]*\}", s)
                if m:
                    try:
                        return json.loads(m.group(0))
                    except Exception:
                        return None
                return None

            # check existing keys
            for f in candidate_fields:
                if f in parsed_a2_enriched and parsed_a2_enriched.get(f):
                    break

            # if not found, try to extract from common raw fields
            if not any(parsed_a2_enriched.get(f) for f in candidate_fields):
                raw_candidates = []
                for key in ("raw_text", "raw", "raw_output", "raw_response"):
                    v = parsed_a2.get(key) if isinstance(parsed_a2, dict) else None
                    if v:
                        raw_candidates.append(v)
                # also include top-level string parse
                if isinstance(parsed_a2, str):
                    raw_candidates.append(parsed_a2)
                for rc in raw_candidates:
                    j = _extract_json_from_text(rc)
                    if isinstance(j, dict):
                        for f in candidate_fields:
                            if f in j and j.get(f):
                                parsed_a2_enriched.setdefault(f, j.get(f))
                        # also check for lower-case keys
                        for k, v in j.items():
                            if k.lower() in (cf.lower() for cf in candidate_fields) and v:
                                parsed_a2_enriched.setdefault(k, v)
                        # stop if we've found at least one
                        if any(parsed_a2_enriched.get(f) for f in candidate_fields):
                            break
        except Exception:
            pass
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
        justification_metrics = {}
        if CHECK_JUSTIFICATION:
            just_a1 = parsers_score_justification(case, {"uzasadnienie": a1.get("uzasadnienie")}, key_pred='uzasadnienie')
            # for naive agent2, use suggested_fix or uzasadnienie_poprawione if present
            just_a2 = parsers_score_justification(case, a2, key_pred='Uzasadnienie_sadowe')
            if just_a2.get('score') is None:
                just_a2 = parsers_score_justification(case, a2, key_pred='uzasadnienie_poprawione')
            justification_metrics = {"agent1": just_a1, "agent2": just_a2}
            metrics["justification"] = justification_metrics
        try:
            a2_enriched = dict(a2) if isinstance(a2, dict) else {"raw": a2}
            a2_enriched.setdefault("recommended_paremie", metrics_agent2.get("pred_after", []))
            a2_enriched.setdefault("decision", a1.get("decision"))
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
    parser.add_argument("case_id", nargs="?", help="ID sprawy, np. C01 lub 'all' dla wszystkich")
    parser.add_argument("--all", action="store_true", help="Uruchomić wszystkie sprawy z datasetu")
    parser.add_argument("--use-gemini", action="store_true", help="Użyć Gemini API (wymaga ustawionego GEMINI_API_KEY i GEMINI_API_URL)")
    parser.add_argument("--use-openai", action="store_true", help="Użyć OpenAI GPT API (wymaga ustawionego OPENAI_API_KEY)")
    parser.add_argument("--check-justification", action="store_true", help="Włączyć sprawdzanie uzasadnień Agenta1/2 względem gold_uzasadnienie (skala 1-5)")
    args = parser.parse_args()
    
    if args.command == "list":
        list_cases()
    elif args.command == "run":
        # Determine which cases to run
        run_all = args.all or (args.case_id and args.case_id.lower() == "all")
        
        if not run_all and not args.case_id:
            logger.error("Podaj case_id, np.: run C01, lub użyj --all")
            return
            
        # choose pipeline: naive vs LLM-based
        # set justification check flag
        global CHECK_JUSTIFICATION
        CHECK_JUSTIFICATION = True #bool(args.check_justification)

        # mutually exclusive: prefer explicit selection
        if args.use_gemini and args.use_openai:
            logger.error("Wybierz tylko jedną z opcji --use-gemini lub --use-openai")
            return
        global LLM_PROVIDER
        if args.use_openai:
            if openai is None:
                logger.error("Brakuje pakietu 'openai'. Zainstaluj go: pip install openai")
                return
            LLM_PROVIDER = "openai"
            os.environ["LLM_PROVIDER"] = "openai"
        elif args.use_gemini:
            if requests is None or yaml is None:
                logger.error("Brakuje zależności 'requests' lub 'PyYAML'. Zainstaluj je: pip install -r requirements.txt")
                return
            LLM_PROVIDER = "gemini"
            os.environ["LLM_PROVIDER"] = "gemini"

        # Run cases
        if run_all:
            cases = load_jsonl(CASES_FILE)
            logger.info("Uruchamianie wszystkich %d spraw...", len(cases))
            for i, case in enumerate(cases, 1):
                cid = case.get("id")
                logger.info("\n[%d/%d] Przetwarzanie sprawy: %s", i, len(cases), cid)
                try:
                    if args.use_gemini or args.use_openai:
                        run_case_llm(cid)
                    else:
                        run_case(cid)
                except Exception as e:
                    logger.error("Błąd przetwarzania sprawy %s: %s", cid, e)
            logger.info("\nZakończono przetwarzanie wszystkich spraw.")
        else:
            if args.use_gemini or args.use_openai:
                run_case_llm(args.case_id)
            else:
                run_case(args.case_id)


if __name__ == "__main__":
    main()
