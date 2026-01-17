"""Utility helpers: jsonl loading, JSON substring extraction, and paremie formatting."""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import List, Dict, Any


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


def load_jsonl(path: Path) -> List[Dict]:
    txt = path.read_text(encoding='utf-8')
    items: List[Dict] = []
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
    return _extract_all_json_objects(txt)


def tag_score(paremia: Dict, case_tags: List[str]) -> int:
    return sum(1 for t in paremia.get('tags', []) if t in case_tags)


def retrieve_top_k(paremie: List[Dict], case_tags: List[str], k: int = 3) -> List[Dict]:
    scored = [(tag_score(p, case_tags), p) for p in paremie]
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [p for s, p in scored if s > 0][:k]
    if not top:
        return paremie[:k]
    return top


def format_paremie_text(paremie_list: List[Dict]) -> str:
    parts: List[str] = []
    for p in paremie_list:
        pid = p.get('id') or p.get('code') or p.get('key') or 'UNKNOWN'
        head = p.get('polish') or p.get('text') or p.get('title') or p.get('label') or ''
        body = p.get('meaning') or p.get('explanation') or p.get('description') or ''
        if not head and body:
            head = (body[:120] + '...') if len(body) > 120 else body
            body = ''
        if not head and not body:
            try:
                head = json.dumps(p, ensure_ascii=False)
            except Exception:
                head = str(p)
        if body:
            parts.append(f"{pid}: {head} — {body}")
        else:
            parts.append(f"{pid}: {head}")
    return "\n".join(parts)
