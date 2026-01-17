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

from google import genai

try:
    import requests
    import yaml
    
except Exception:
    requests = None
    yaml = None
    genai = None

# load project secrets from src/secrets.py (to avoid colliding with stdlib `secrets`)
def _load_project_secrets():
    spec_path = Path(__file__).resolve().parent / "secrets.py"
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
    txt = path.read_text(encoding="utf-8")
    # try fast path: line-delimited JSON objects
    items = []
    lines = [l for l in txt.splitlines() if l.strip()]
    ok = True
    for ln in lines:
        try:
            items.append(json.loads(ln))
        except Exception:
            ok = False
            break
    if ok and items:
        return items
    # fallback: extract all balanced JSON objects from the file
    return _extract_all_json_objects(txt)


def _extract_all_json_objects(text: str) -> List[Dict]:
    objs: List[Dict] = []
    i = 0
    n = len(text)
    while i < n:
        # find next { or [
        j = None
        for k in range(i, n):
            if text[k] in '{[':
                j = k
                break
        if j is None:
            break
        start_char = text[j]
        stack = []
        end = None
        for k in range(j, n):
            ch = text[k]
            if ch == '{' or ch == '[':
                stack.append(ch)
            elif ch == '}' and stack and stack[-1] == '{':
                stack.pop()
                if not stack:
                    end = k + 1
                    break
            elif ch == ']' and stack and stack[-1] == '[':
                stack.pop()
                if not stack:
                    end = k + 1
                    break
        if end is None:
            break
        candidate = text[j:end]
        try:
            obj = json.loads(candidate)
            objs.append(obj)
        except Exception:
            pass
        i = end
    return objs


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
        print("genai package not available")
        return None
    
    # ensure API key env var set for client
    api_key = None
    if project_secrets is not None:
        api_key = getattr(project_secrets, "GEMINI_API_KEY", None)
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Brakuje GEMINI_API_KEY dla genai client")
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
        print("Błąd wywołania genai.Client():", e)
        return None


def _extract_json_substring(text: str) -> Optional[str]:
    # Find the first balanced JSON object or array in text using simple heuristics
    for start_char in ('{', '['):
        start_idx = text.find(start_char)
        if start_idx == -1:
            continue
        stack = []
        for i in range(start_idx, len(text)):
            ch = text[i]
            if ch == start_char:
                stack.append(ch)
            elif ch == '}' and stack and stack[-1] == '{':
                stack.pop()
                if not stack:
                    candidate = text[start_idx:i+1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except Exception:
                        break
            elif ch == ']' and stack and stack[-1] == '[':
                stack.pop()
                if not stack:
                    candidate = text[start_idx:i+1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except Exception:
                        break
    return None


def parse_agent1_output(text: str) -> Dict:
    out = {"decision": None, "uzasadnienie": [], "used_paremie": [], "assumptions": [], "raw_text": text}
    if not text:
        return out
    jstr = _extract_json_substring(text)
    if jstr:
        try:
            data = json.loads(jstr)
            out["decision"] = data.get("Decyzja") or data.get("decision")
            out["uzasadnienie"] = data.get("Uzasadnienie") or data.get("justification") or []
            used = data.get("Paremie użyte") or data.get("used_paremie") or []
            if isinstance(used, str):
                used = re.split(r"[,;\s]+", used.strip())
            out["used_paremie"] = [u.strip().upper() for u in used if u]
            out["assumptions"] = data.get("Założenia") or data.get("assumptions") or []
            # do not return yet: if some fields (e.g. uzasadnienie) are missing
            # try to extract them from the free-text body below
            json_parsed = True
        except Exception:
            json_parsed = False
    else:
        json_parsed = False

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for i, line in enumerate(lines):
        m = re.match(r"Decyzja:\s*(zasadne|niezasadne|nie_dazy)", line, flags=re.I)
        if m and not out["decision"]:
            out["decision"] = m.group(1).lower()
        if line.lower().startswith("paremie użyte") or line.lower().startswith("paremie uzyte") or line.lower().startswith("paremie:"):
            rest = line.split(":", 1)[1] if ":" in line else ""
            candidates = []
            if rest:
                candidates = re.split(r"[,;]\s*", rest)
            j = i+1
            while j < len(lines) and (lines[j].startswith("-") or re.match(r"^P\d{2}", lines[j])):
                candidates.extend(re.findall(r"P\d{2}", lines[j]))
                j += 1
            out["used_paremie"] = [c.strip().upper() for c in candidates if c]
        if line.lower().startswith("założenia") or line.lower().startswith("zalozenia") or line.lower().startswith("założenia i niepewności"):
            j = i+1
            asum = []
            while j < len(lines) and not re.match(r"^Paremie|^Decyzja|^Uzasadnienie", lines[j], flags=re.I):
                asum.append(lines[j])
                j += 1
            out["assumptions"] = asum
        # extract numbered justification block if missing
        if (not out.get("uzasadnienie")) and re.match(r"^Uzasadnienie[:\s]*$", line, flags=re.I):
            j = i+1
            just = []
            while j < len(lines) and (re.match(r"^\d+\.|^-\s", lines[j]) or lines[j]):
                # stop on next section header
                if re.match(r"^(Paremie użyte|Paremie:|Założenia|Decyzja)", lines[j], flags=re.I):
                    break
                # strip numbering
                cleaned = re.sub(r"^\d+\.?\s*", "", lines[j]).strip()
                if cleaned:
                    just.append(cleaned)
                j += 1
            if just:
                out["uzasadnienie"] = just
                # continue parsing in case other fields are present
    return out


def parse_agent2_output(text: str) -> Dict:
    out = {"rubric": {}, "errors": [], "suggested_fix": None, "raw_text": text}
    if not text:
        return out
    jstr = _extract_json_substring(text)
    if jstr:
        try:
            data = json.loads(jstr)
            out["rubric"] = data.get("Rubryka") or data.get("rubric") or {}
            out["errors"] = data.get("Lista błędów") or data.get("errors") or []
            out["suggested_fix"] = data.get("Proponowana poprawka") or data.get("suggested_fix")
            return out
        except Exception:
            pass

    for line in text.splitlines():
        m = re.match(r"[-\s]*Dob[oó]r paremii[:\s]+(\d+)", line, flags=re.I)
        if m:
            out["rubric"]["Dobór paremii"] = int(m.group(1))
        m = re.match(r"[-\s]*Interpretacja paremii[:\s]+(\d+)", line, flags=re.I)
        if m:
            out["rubric"]["Interpretacja paremii"] = int(m.group(1))
        m = re.match(r"[-\s]*Aplikacja[:\s]+(\d+)", line, flags=re.I)
        if m:
            out["rubric"]["Aplikacja"] = int(m.group(1))
        m = re.match(r"[-\s]*Pomini[eę]cia[:\s]+(\d+)", line, flags=re.I)
        if m:
            out["rubric"]["Pominięcia"] = int(m.group(1))
        m = re.match(r"[-\s]*Nadmierna pewno[sś][cć][:\s]+(\d+)", line, flags=re.I)
        if m:
            out["rubric"]["Nadmierna pewność"] = int(m.group(1))
        m = re.match(r"[-\s]*dob[oó]r[:\s]+(.+)", line, flags=re.I)
        if m:
            out["errors"].append({"dobór": m.group(1).strip()})
        m = re.match(r"[-\s]*pomini[eę]cie[:\s]+(.+)", line, flags=re.I)
        if m:
            out["errors"].append({"pominięcie": m.group(1).strip()})
    return out


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
        with open(case_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)


def run_case_llm(case_id: str):
    paremie = load_jsonl(PAREMIE_FILE)
    cases = load_jsonl(CASES_FILE)
    case_map = {c["id"]: c for c in cases}
    case = case_map.get(case_id)
    if not case:
        print(f"Case {case_id} not found. Available: {', '.join(case_map.keys())}")
        return

    # load templates
    tpl1 = load_yaml_template(Path(__file__).resolve().parents[1] / "prompts" / "agent1_template.yaml")
    tpl2 = load_yaml_template(Path(__file__).resolve().parents[1] / "prompts" / "agent2_template.yaml")

    paremie_text = format_paremie_text(paremie)
    # Use safe replacement to avoid interpreting braces in example JSON inside the prompt
    prompt1 = tpl1.get("prompt", "").replace("{case_facts}", case.get("facts", "")).replace("{paremie}", paremie_text)
    print(f"= Sprawa: {case['id']} - {case.get('title')} (LLM mode)")

    a1_raw = call_gemini_genai(prompt1)
    if a1_raw is None:
        print("Nie udało się uzyskać odpowiedzi od Gemini. Kończę.")
        return

    print("\n-- Agent 1 (LLM) raw output --")
    print(a1_raw)

    # parse Agent1 output (safe JSON-extract or heuristics)
    parsed_a1 = parse_agent1_output(a1_raw)
    print("\n-- Agent 1 parsed output --")
    print(f"Decision: {parsed_a1.get('decision')}")
    print(f"Used paremie: {parsed_a1.get('used_paremie')}")
    print(f"Assumptions: {parsed_a1.get('assumptions')}")

    # compare with gold
    metrics = compare_agent1_with_gold(case, parsed_a1)
    print("\n-- Automatic evaluation (Agent1 vs gold) --")
    print(f"Decision match: {metrics['decision_match']} (pred={metrics['pred_decision']}, gold={metrics['gold_decision']})")
    print(f"Paremie precision: {metrics['precision']:.2f}, recall: {metrics['recall']:.2f}")
    print(f"True positives: {metrics['tp']}")
    # if parsing failed (no decision or no paremie), try a forced JSON-generation retry
    if parsed_a1.get('decision') is None or not parsed_a1.get('used_paremie'):
        print('\nAgent1 output did not contain structured data; attempting forced JSON retry...')
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
            print('\n-- Agent 1 (LLM) forced retry raw output --')
            print(a1_raw2)
            parsed_a1 = parse_agent1_output(a1_raw2)
            print('\n-- Agent 1 parsed output (after retry) --')
            print(f"Decision: {parsed_a1.get('decision')}")
            print(f"Used paremie: {parsed_a1.get('used_paremie')}")
            metrics = compare_agent1_with_gold(case, parsed_a1)
            print(f"Paremie precision: {metrics['precision']:.2f}, recall: {metrics['recall']:.2f}")

    # Agent 2: build prompt including Agent1 response
    prompt2 = tpl2.get("prompt", "").replace("{case_facts}", case.get("facts", "")).replace("{agent1_response}", a1_raw or "").replace("{paremie}", paremie_text)
    a2_raw = call_gemini_genai(prompt2)
    if a2_raw is None:
        print("Nie udało się uzyskać odpowiedzi krytyka od Gemini. Kończę.")
        return

    print("\n-- Agent 2 (LLM) raw output --")
    print(a2_raw)

    # parse Agent2 output
    parsed_a2 = parse_agent2_output(a2_raw)
    print("\n-- Agent 2 parsed output --")
    print(f"Rubric: {parsed_a2.get('rubric')}")
    print(f"Errors: {parsed_a2.get('errors')}")
    if parsed_a2.get('suggested_fix'):
        print(f"Suggested fix: {parsed_a2.get('suggested_fix')}")
    # retry Agent2 forcing JSON if parser found nothing useful
    if not parsed_a2.get('rubric'):
        print('\nAgent2 output did not contain structured rubric; attempting forced JSON retry...')
        forced2 = (
            "Na podstawie poniższych faktów i odpowiedzi Agenta 1 wygeneruj jedną ocenę krytyczną w formacie JSON. "
            "Zwróć jedynie poprawny JSON, np. {\"Rubryka\": {...}, \"Lista błędów\": [...], \"Proponowana poprawka\": \"...\"}. "
            "Fakty: \n" + case.get('facts', '') + "\nOdpowiedź Agenta1:\n" + (a1_raw or '') + "\nLista paremii:\n" + paremie_text
        )
        a2_raw2 = call_gemini_genai(forced2)
        if a2_raw2:
            print('\n-- Agent 2 (LLM) forced retry raw output --')
            print(a2_raw2)
            parsed_a2 = parse_agent2_output(a2_raw2)
            print('\n-- Agent 2 parsed output (after retry) --')
            print(f"Rubric: {parsed_a2.get('rubric')}")
            print(f"Errors: {parsed_a2.get('errors')}")

    # finalize and save per-case results
    a1_raw_final = a1_raw2 if ('a1_raw2' in locals() and a1_raw2) else a1_raw
    a2_raw_final = a2_raw2 if ('a2_raw2' in locals() and a2_raw2) else a2_raw
    results_payload = {
        "agent1_raw": a1_raw_final,
        "agent1_parsed": parsed_a1,
        "agent2_raw": a2_raw_final,
        "agent2_parsed": parsed_a2,
        "metrics": metrics,
    }
    try:
        save_case_results(case_id, results_payload)
        print(f"\nSaved results to results/{case_id}/")
    except Exception as e:
        print("Błąd zapisu wyników:", e)



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

    print(f"= Sprawa: {case['id']} - {case.get('title')}")
    a1 = naive_agent1(case, paremie)
    print("\n-- Agent 1 (generator) --")
    print(f"Decyzja: {a1['decision']}")
    print("Uzasadnienie:")
    for p in a1['uzasadnienie']:
        print(p)
    print("Paremie użyte:", ", ".join(a1['used_paremie']))
    print("Założenia:")
    for a in a1['assumptions']:
        print("- ", a)

    a2 = naive_agent2(a1, case)
    print("\n-- Agent 2 (krytyk) --")
    print("Rubryka:")
    for k, v in a2['rubric'].items():
        print(f"- {k}: {v}/5")
    if a2['errors']:
        print("Lista błędów:")
        for e in a2['errors']:
            for t, m in e.items():
                print(f"- {t}: {m}")
    print("\nSugerowana poprawka:")
    print(a2['suggested_fix'])
    # save naive-run results as well
    try:
        metrics = compare_agent1_with_gold(case, {"decision": a1.get("decision"), "used_paremie": a1.get("used_paremie")})
        payload = {
            "agent1_raw": None,
            "agent1_parsed": a1,
            "agent2_raw": None,
            "agent2_parsed": a2,
            "metrics": metrics,
        }
        save_case_results(case["id"], payload)
        print(f"\nSaved results to results/{case['id']}/")
    except Exception as e:
        print("Błąd zapisu wyników (naive):", e)


def list_cases():
    cases = load_jsonl(CASES_FILE)
    for c in cases:
        print(f"{c['id']}: {c.get('title')} ({c.get('label')})")


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
            print("Podaj case_id, np.: run C01")
            return
        # choose pipeline: naive vs LLM-based
        if args.use_gemini:
            if requests is None or yaml is None:
                print("Brakuje zależności 'requests' lub 'PyYAML'. Zainstaluj je: pip install -r requirements.txt")
                return
            run_case_llm(args.case_id)
        else:
            run_case(args.case_id)


if __name__ == "__main__":
    main()
