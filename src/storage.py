"""Storage helpers for saving per-case results."""
from __future__ import annotations
from pathlib import Path
import json
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]


def save_case_results(case_id: str, data: dict):
    results_root = ROOT / "results"
    case_dir = results_root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    if "agent1_raw" in data and data["agent1_raw"] is not None:
        (case_dir / "agent1_raw.txt").write_text(data["agent1_raw"], encoding="utf-8")
    if "agent2_raw" in data and data["agent2_raw"] is not None:
        (case_dir / "agent2_raw.txt").write_text(data["agent2_raw"], encoding="utf-8")
    if "agent1_parsed" in data:
        with open(case_dir / "agent1_parsed.json", "w", encoding="utf-8") as f:
            json.dump(data["agent1_parsed"], f, ensure_ascii=False, indent=2)
    if "agent2_parsed" in data:
        with open(case_dir / "agent2_parsed.json", "w", encoding="utf-8") as f:
            json.dump(data["agent2_parsed"], f, ensure_ascii=False, indent=2)
    if "metrics" in data:
        meta = data["metrics"].copy()
        meta.setdefault("saved_at", datetime.utcnow().isoformat() + "Z")
        with open(case_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
