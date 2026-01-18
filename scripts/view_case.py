#!/usr/bin/env python3
"""
Wizualizacja szczegółowa pojedynczego case'a.
Wyświetla:
- Fakty sprawy i pytanie
- Uzasadnienia: Gold, Agent 1, Agent 2
- Paremie: użyte vs gold
- Metryki: precision, recall, scores
- Decyzje: gold vs predicted

Usage: python scripts/view_case.py C01 [--output-dir DIR]
"""
import sys
import json
from pathlib import Path
from textwrap import wrap

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Błąd: brak matplotlib. Zainstaluj: pip install matplotlib")
    sys.exit(1)

# Paths
ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
DATA_DIR = ROOT_DIR / "data"
CASES_FILE = DATA_DIR / "cases.jsonl"
VIZ_OUTPUT_DIR = RESULTS_DIR / "case_visualizations"


def load_case_data(case_id):
    """Wczytuje dane case'a z data/cases.jsonl"""
    if not CASES_FILE.exists():
        return None
    
    with open(CASES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            case = json.loads(line.strip())
            if case["id"] == case_id:
                return case
    return None


def load_case_results(case_id, output_dir="tmp"):
    """Wczytuje wyniki z results/{output_dir}/{case_id}/"""
    case_dir = RESULTS_DIR / output_dir / case_id
    if not case_dir.exists():
        return None
    
    results = {}
    
    # Agent 1
    a1_parsed = case_dir / "agent1_parsed.json"
    if a1_parsed.exists():
        with open(a1_parsed, "r", encoding="utf-8") as f:
            results["agent1_parsed"] = json.load(f)
    
    # Agent 2
    a2_parsed = case_dir / "agent2_parsed.json"
    if a2_parsed.exists():
        with open(a2_parsed, "r", encoding="utf-8") as f:
            results["agent2_parsed"] = json.load(f)
    
    # Metrics
    m1 = case_dir / "metrics_agent1.json"
    if m1.exists():
        with open(m1, "r", encoding="utf-8") as f:
            results["metrics_agent1"] = json.load(f)
    
    m2 = case_dir / "metrics_agent2.json"
    if m2.exists():
        with open(m2, "r", encoding="utf-8") as f:
            results["metrics_agent2"] = json.load(f)
    
    md = case_dir / "metrics_decisions.json"
    if md.exists():
        with open(md, "r", encoding="utf-8") as f:
            results["metrics_decisions"] = json.load(f)
    
    return results


def wrap_text(text, width=80):
    """Owijanie tekstu z zachowaniem akapitów"""
    if isinstance(text, list):
        text = "\n\n".join(text)
    
    paragraphs = text.split("\n\n")
    wrapped = []
    for p in paragraphs:
        lines = p.split("\n")
        for line in lines:
            wrapped.extend(wrap(line, width=width))
    return "\n".join(wrapped)


def visualize_case(case_id, output_dir="tmp"):
    """Tworzy dashboard wizualizacji case'a"""
    
    # Load data
    case_data = load_case_data(case_id)
    if not case_data:
        print(f"Nie znaleziono case'a {case_id} w danych.")
        return
    
    results = load_case_results(case_id, output_dir)
    if not results:
        print(f"Nie znaleziono wyników dla case'a {case_id} w results/{output_dir}/.")
        return
    
    # Create figure with custom grid
    fig = plt.figure(figsize=(24, 18))
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.6, wspace=0.4)
    
    # Title
    fig.suptitle(f"Case {case_id}: {case_data['title']}", 
                 fontsize=22, fontweight='bold', y=0.98)
    
    # === Panel 1: Fakty sprawy (top left, double width) ===
    ax_facts = fig.add_subplot(gs[0, :2])
    ax_facts.axis('off')
    
    facts_text = f"FAKTY:\n{wrap_text(case_data['facts'], 110)}\n\n"
    facts_text += f"PYTANIE:\n{wrap_text(case_data['question'], 110)}"
    
    ax_facts.text(0.02, 0.98, facts_text,
                  fontsize=13, va='top', ha='left',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # === Panel 2: Metryki numeryczne (top right) ===
    ax_metrics = fig.add_subplot(gs[0, 2])
    ax_metrics.axis('off')
    
    m1 = results.get("metrics_agent1", {})
    m2 = results.get("metrics_agent2", {})
    md = results.get("metrics_decisions", {})
    
    metrics_text = "METRYKI:\n\n"
    metrics_text += f"Agent 1:\n"
    metrics_text += f"  Precision: {m1.get('precision', 0):.3f}\n"
    metrics_text += f"  Recall: {m1.get('recall', 0):.3f}\n"
    metrics_text += f"  Decision match: {m1.get('decision_match', False)}\n"
    if md and 'llm_judge' in md:
        metrics_text += f"  LLM Judge: {md['llm_judge'].get('agent1_score', 'N/A')}/5\n"
    
    metrics_text += f"\nAgent 2:\n"
    metrics_text += f"  Precision: {m2.get('precision', 0):.3f}\n"
    metrics_text += f"  Recall: {m2.get('recall', 0):.3f}\n"
    if md and 'llm_judge' in md:
        metrics_text += f"  LLM Judge: {md['llm_judge'].get('agent2_score', 'N/A')}/5\n"
    
    ax_metrics.text(0.05, 0.95, metrics_text,
                    fontsize=13, va='top', ha='left', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # === Panel 3: Decyzje (row 2, left) ===
    ax_decisions = fig.add_subplot(gs[1, 0])
    ax_decisions.axis('off')
    
    gold_dec = case_data.get('label', 'N/A')
    a1_dec = results.get("agent1_parsed", {}).get("decision", "N/A")
    a2_dec = results.get("agent2_parsed", {}).get("decision", "N/A")
    
    dec_text = "DECYZJE:\n\n"
    dec_text += f"Gold:    {gold_dec}\n"
    dec_text += f"Agent 1: {a1_dec}\n"
    dec_text += f"Agent 2: {a2_dec}"
    
    ax_decisions.text(0.05, 0.95, dec_text,
                      fontsize=14, va='top', ha='left', family='monospace',
                      bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    # === Panel 4: Paremie (row 2, middle+right) ===
    ax_paremie = fig.add_subplot(gs[1, 1:])
    ax_paremie.axis('off')
    
    gold_par = set(case_data.get('gold_paremie', []))
    a1_par = set(results.get("agent1_parsed", {}).get("used_paremie", []))
    a2_par = set(results.get("agent2_parsed", {}).get("recommended_paremie", []))
    
    par_text = "PAREMIE:\n\n"
    par_text += f"Gold:    {', '.join(sorted(gold_par)) if gold_par else 'N/A'}\n"
    par_text += f"Agent 1: {', '.join(sorted(a1_par)) if a1_par else 'N/A'}\n"
    par_text += f"Agent 2: {', '.join(sorted(a2_par)) if a2_par else 'N/A'}\n\n"
    
    # Overlap analysis
    a1_tp = gold_par & a1_par
    a1_fp = a1_par - gold_par
    a1_fn = gold_par - a1_par
    
    par_text += f"Agent 1 vs Gold:\n"
    par_text += f"  TP (correct): {', '.join(sorted(a1_tp)) if a1_tp else 'none'}\n"
    par_text += f"  FP (extra):   {', '.join(sorted(a1_fp)) if a1_fp else 'none'}\n"
    par_text += f"  FN (missing): {', '.join(sorted(a1_fn)) if a1_fn else 'none'}\n"
    
    ax_paremie.text(0.02, 0.95, par_text,
                    fontsize=12, va='top', ha='left', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.3))
    
    # === Panel 5: Gold uzasadnienie (row 3, full width) ===
    ax_gold = fig.add_subplot(gs[2, :])
    ax_gold.axis('off')
    
    gold_expl = case_data.get('gold_explanation', 'N/A')
    gold_text = f"GOLD UZASADNIENIE:\n\n{wrap_text(gold_expl, 170)}"
    
    ax_gold.text(0.01, 0.98, gold_text,
                 fontsize=12, va='top', ha='left',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.2))
    
    # === Panel 6: Agent 1 uzasadnienie (row 4, left half) ===
    ax_a1 = fig.add_subplot(gs[3, :1])
    ax_a1.axis('off')
    
    a1_just = results.get("agent1_parsed", {}).get("uzasadnienie", [])
    a1_text = f"AGENT 1 UZASADNIENIE:\n\n{wrap_text(a1_just, 95)}"
    
    if md and 'llm_judge' in md:
        a1_comment = md['llm_judge'].get('agent1_comment', '')
        a1_text += f"\n\nLLM Judge: {wrap_text(a1_comment, 95)}"
    
    ax_a1.text(0.02, 0.98, a1_text,
               fontsize=11, va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.3))
    
    # === Panel 7: Agent 2 uzasadnienie (row 4, right half) ===
    ax_a2 = fig.add_subplot(gs[3, 1:])
    ax_a2.axis('off')
    
    a2_just = results.get("agent2_parsed", {}).get("Uzasadnienie_sadowe", [])
    a2_text = f"AGENT 2 UZASADNIENIE SĄDOWE:\n\n{wrap_text(a2_just, 95)}"
    
    if md and 'llm_judge' in md:
        a2_comment = md['llm_judge'].get('agent2_comment', '')
        a2_text += f"\n\nLLM Judge: {wrap_text(a2_comment, 95)}"
    
    ax_a2.text(0.02, 0.98, a2_text,
               fontsize=11, va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
    
    # Save
    viz_dir = VIZ_OUTPUT_DIR / output_dir
    viz_dir.mkdir(parents=True, exist_ok=True)
    output_path = viz_dir / f"{case_id}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Wizualizacja zapisana: {output_path}")
    
    # Show optionally (commented out by default)
    # plt.show()
    
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/view_case.py <CASE_ID> [--output-dir DIR]")
        print("Przykład: python scripts/view_case.py C01")
        print("Przykład: python scripts/view_case.py C01 --output-dir experiment1")
        sys.exit(1)
    
    case_id = sys.argv[1]
    output_dir = "tmp"
    
    # Parse optional --output-dir
    if len(sys.argv) > 2 and sys.argv[2] == "--output-dir" and len(sys.argv) > 3:
        output_dir = sys.argv[3]
    
    visualize_case(case_id, output_dir)


if __name__ == "__main__":
    main()
