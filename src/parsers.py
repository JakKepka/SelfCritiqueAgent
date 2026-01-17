"""Parsing helpers for Agent1 and Agent2 outputs and evaluation."""
from __future__ import annotations
import json
import re
from typing import Dict, Any


def _extract_json_substring(text: str) -> str | None:
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
        if (not out.get("uzasadnienie")) and re.match(r"^Uzasadnienie[:\s]*$", line, flags=re.I):
            j = i+1
            just = []
            while j < len(lines) and (re.match(r"^\d+\.|^-\s", lines[j]) or lines[j]):
                if re.match(r"^(Paremie użyte|Paremie:|Założenia|Decyzja)", lines[j], flags=re.I):
                    break
                cleaned = re.sub(r"^\d+\.?\s*", "", lines[j]).strip()
                if cleaned:
                    just.append(cleaned)
                j += 1
            if just:
                out["uzasadnienie"] = just
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


def _apply_agent2_corrections_to_pred(predicted: set, agent2_parsed: Dict) -> set:
    """Infer corrected set of predicted paremies after Agent2 feedback.

    Heuristics:
    - If agent2_parsed contains 'recommended_paremie' use it as the corrected set.
    - Otherwise, scan 'errors' messages for mentions of P\d{2}:
      - messages under key containing 'pomini' => add those PIDs to predicted
      - messages under key containing 'dob' or mentioning 'nadmiar' => remove those PIDs from predicted
    - Fallback: return original predicted set.
    """
    corrected = set(predicted)
    # explicit recommendation
    rec = agent2_parsed.get("recommended_paremie") or agent2_parsed.get("recommended")
    if isinstance(rec, (list, set)) and rec:
        return set([r.strip().upper() for r in rec if r])

    # scan errors
    errors = agent2_parsed.get("errors", [])
    for e in errors:
        # e may be dicts like {"pominięcie": "Brakuje P02"} or strings
        if isinstance(e, dict):
            for k, msg in e.items():
                found = re.findall(r"P\d{2}", str(msg))
                if not found:
                    continue
                key = k.lower()
                if "pomi" in key:
                    corrected.update([f.upper() for f in found])
                elif "dob" in key or "nadmi" in str(msg).lower():
                    corrected.difference_update([f.upper() for f in found])
        else:
            msg = str(e)
            found = re.findall(r"P\d{2}", msg)
            if not found:
                continue
            if re.search(r"pomini|brakuje", msg, flags=re.I):
                corrected.update([f.upper() for f in found])
            elif re.search(r"nadmiar|niepotrzeb|błędn", msg, flags=re.I):
                corrected.difference_update([f.upper() for f in found])
    return corrected


def compare_agent2_with_gold(case: Dict, agent1_parsed: Dict, agent2_parsed: Dict) -> Dict:
    """Compute metrics after applying Agent2's suggested corrections to Agent1's predictions.

    Returns metrics similar to compare_agent1_with_gold plus the corrected predicted set
    and a list of modifications inferred.
    """
    gold = set(case.get("gold_paremie", [])) if case.get("gold_paremie") else set()
    pred = set([p.upper() for p in agent1_parsed.get("used_paremie", [])])
    corrected = _apply_agent2_corrections_to_pred(pred, agent2_parsed)
    tp = corrected & gold
    precision = len(tp) / len(corrected) if corrected else 0.0
    recall = len(tp) / len(gold) if gold else 0.0
    modifications = {
        "added": sorted(list(corrected - pred)),
        "removed": sorted(list(pred - corrected)),
    }
    return {
        "pred_before": sorted(list(pred)),
        "pred_after": sorted(list(corrected)),
        "precision": precision,
        "recall": recall,
        "tp": sorted(list(tp)),
        "gold": sorted(list(gold)),
        "modifications": modifications,
    }
