#!/usr/bin/env python3
"""Prosty lokalny runner prototypu pipeline: Agent1 -> Agent2.

To nie wykonuje wywołań LLM — generuje prostą, deterministyczną odpowiedź
opartą na tagowym retrievalu paremii, aby przetestować format i pipeline.
"""
import argparse
import json
import os
import importlib.util
from pathlib import Path
from typing import List, Dict, Optional

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
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


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
        parts.append(f"{p['id']}: {p['polish']} — {p['meaning']}")
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
    model_name = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
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
    prompt1 = tpl1.get("prompt", "").format(case_facts=case.get("facts", ""), paremie=paremie_text)
    print(f"= Sprawa: {case['id']} - {case.get('title')} (LLM mode)")

    a1_raw = call_gemini_genai(prompt1)
    if a1_raw is None:
        print("Nie udało się uzyskać odpowiedzi od Gemini. Kończę.")
        return

    print("\n-- Agent 1 (LLM) raw output --")
    print(a1_raw)

    # Agent 2: build prompt including Agent1 response
    prompt2 = tpl2.get("prompt", "").format(case_facts=case.get("facts", ""), agent1_response=a1_raw, paremie=paremie_text)
    a2_raw = call_gemini_genai(prompt2)
    if a2_raw is None:
        print("Nie udało się uzyskać odpowiedzi krytyka od Gemini. Kończę.")
        return

    print("\n-- Agent 2 (LLM) raw output --")
    print(a2_raw)



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
