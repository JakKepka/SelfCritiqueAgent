#!/usr/bin/env python3
"""
Skrypt agregujący wyniki ewaluacji z katalogu results/.
Oblicza średnie metryki dla Agenta 1 i Agenta 2:
- Precision / Recall (paremie)
- Decision Accuracy i F1 Score (klasyfikacja zasadne/niezasadne/niejednoznaczne)
- Justification Score (na bazie metrics_decisions.json, preferując llm_judge)

Generuje wykresy porównawcze i zapisuje raport zbiorczy.
"""
import json
import re
import math
from pathlib import Path
from collections import Counter
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
import statistics

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_BASE = ROOT_DIR / "results"

def get_results_dir(output_dir="tmp"):
    return RESULTS_BASE / output_dir

def get_report_file(output_dir="tmp"):
    return RESULTS_BASE / output_dir / "evaluation_report.json"

def get_plot_file(output_dir="tmp"):
    return RESULTS_BASE / output_dir / "evaluation_plots.png"

def load_case_metrics(case_dir: Path):
    m1 = {}
    m2 = {}
    md = {}
    
    # metrics_agent1.json
    p1 = case_dir / "metrics_agent1.json"
    if p1.exists():
        try:
            with open(p1, "r", encoding="utf-8") as f:
                m1 = json.load(f)
        except Exception:
            pass

    # metrics_agent2.json
    p2 = case_dir / "metrics_agent2.json"
    if p2.exists():
        try:
            with open(p2, "r", encoding="utf-8") as f:
                m2 = json.load(f)
        except Exception:
            pass

    # metrics_decisions.json
    pd = case_dir / "metrics_decisions.json"
    if pd.exists():
        try:
            with open(pd, "r", encoding="utf-8") as f:
                md = json.load(f)
        except Exception:
            pass
            
    return m1, m2, md

def get_justification_score(metrics_decisions, agent_key):
    """
    Pobiera ocenę uzasadnienia (1-5).
    Priorytet: 'llm_judge' -> '{agent_key}_score'
    Fallback: '{agent_key}' -> 'score' (heurystyczne)
    """
    # Try LLM judge first
    llm = metrics_decisions.get("llm_judge")
    if llm and isinstance(llm, dict):
        key = f"{agent_key}_score"
        if key in llm and isinstance(llm[key], (int, float)):
            return float(llm[key])
            
    # Fallback to heuristic score stored in agent dict
    # agent_key is typically "agent1" or "agent2"
    heuristic = metrics_decisions.get(agent_key)
    if heuristic and isinstance(heuristic, dict):
        sc = heuristic.get("score")
        if isinstance(sc, (int, float)):
            return float(sc)
            
    return None


def compute_f1_multiclass(y_true, y_pred, labels=None):
    """
    Oblicza macro-averaged F1 score dla klasyfikacji wieloklasowej.
    """
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return 0.0
    
    if labels is None:
        labels = sorted(set(y_true + y_pred))
    
    f1_scores = []
    for label in labels:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp == label)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != label and yp == label)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp != label)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
    
    return statistics.mean(f1_scores) if f1_scores else 0.0

def main(output_dir="tmp"):
    RESULTS_DIR = get_results_dir(output_dir)
    REPORT_FILE = get_report_file(output_dir)
    PLOT_FILE = get_plot_file(output_dir)
    
    if not RESULTS_DIR.exists():
        print(f"Katalog {RESULTS_DIR} nie istnieje.")
        return

    # Data collectors
    data = {
        "agent1": {
            "precision": [], 
            "recall": [], 
            "decision_match": [], 
            "justification_score": [],
            "gold_decisions": [],
            "pred_decisions": []
        },
        "agent2": {
            "precision": [], 
            "recall": [], 
            "justification_score": [],
            "gold_decisions": [],
            "pred_decisions": []
        }
    }
    
    case_dirs = [d for d in RESULTS_DIR.iterdir() if d.is_dir()]
    case_ids = []

    print(f"Znaleziono {len(case_dirs)} spraw w {RESULTS_DIR}")

    for d in sorted(case_dirs):
        m1, m2, md = load_case_metrics(d)
        cid = d.name
        case_ids.append(cid)
        
        # Agent 1 stats
        if m1:
            if "precision" in m1: data["agent1"]["precision"].append(m1["precision"])
            if "recall" in m1: data["agent1"]["recall"].append(m1["recall"])
            if "decision_match" in m1: data["agent1"]["decision_match"].append(1 if m1["decision_match"] else 0)
            # Collect decisions for F1 calculation
            if "gold_decision" in m1 and "pred_decision" in m1:
                data["agent1"]["gold_decisions"].append(m1["gold_decision"])
                data["agent1"]["pred_decisions"].append(m1["pred_decision"])
        
        j1 = get_justification_score(md, "agent1")
        if j1 is not None:
            data["agent1"]["justification_score"].append(j1)
            
        # Agent 2 stats
        if m2:
            if "precision" in m2: data["agent2"]["precision"].append(m2["precision"])
            if "recall" in m2: data["agent2"]["recall"].append(m2["recall"])
            # Agent2 doesn't have direct decision match; use gold from m1
            if m1 and "gold_decision" in m1:
                # Check if agent2_parsed has decision
                try:
                    agent2_parsed_file = d / "agent2_parsed.json"
                    if agent2_parsed_file.exists():
                        with open(agent2_parsed_file, "r", encoding="utf-8") as f:
                            a2p = json.load(f)
                            if "decision" in a2p:
                                data["agent2"]["gold_decisions"].append(m1["gold_decision"])
                                data["agent2"]["pred_decisions"].append(a2p["decision"])
                except Exception:
                    pass
            
        j2 = get_justification_score(md, "agent2")
        if j2 is not None:
            data["agent2"]["justification_score"].append(j2)

    # Compute averages and F1 scores
    averages = {}
    for agent in ["agent1", "agent2"]:
        averages[agent] = {}
        for metric, values in data[agent].items():
            if metric in ("gold_decisions", "pred_decisions"):
                continue  # Skip decision lists in averages
            avg = statistics.mean(values) if values else 0.0
            averages[agent][metric] = avg
            print(f"{agent} average {metric}: {avg:.4f} (n={len(values)})")
        
        # Compute F1 score for decision classification
        gold = data[agent]["gold_decisions"]
        pred = data[agent]["pred_decisions"]
        if gold and pred:
            f1 = compute_f1_multiclass(gold, pred)
            accuracy = sum(1 for g, p in zip(gold, pred) if g == p) / len(gold)
            averages[agent]["decision_f1"] = f1
            averages[agent]["decision_accuracy"] = accuracy
            print(f"{agent} decision accuracy: {accuracy:.4f} (n={len(gold)})")
            print(f"{agent} decision F1 (macro): {f1:.4f}")

    # Save summary
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(averages, f, indent=2)
    print(f"\nRaport zapisano w {REPORT_FILE}")

    # Plotting
    if not HAS_MATPLOTLIB:
        print("Brak biblioteki matplotlib. Wykresy nie zostały wygenerowane.")
        print("Zainstaluj: pip install matplotlib")
        return
    
    def add_value_labels(ax, bars, values, fmt='.3f'):
        """Dodaje wartości numeryczne nad słupkami."""
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{val:{fmt}}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        width = 0.35
        
        # Plot 1: Precision/Recall (0-1)
        ax1 = axes[0, 0]
        pr_vars = ["precision", "recall"]
        pr_labels = ["Precision", "Recall"]
        x_pr = range(len(pr_vars))
        
        v1_pr = [averages["agent1"].get(m, 0) for m in pr_vars]
        v2_pr = [averages["agent2"].get(m, 0) for m in pr_vars]
        
        bars1 = ax1.bar([p - width/2 for p in x_pr], v1_pr, width, label='Agent 1', color='skyblue')
        bars2 = ax1.bar([p + width/2 for p in x_pr], v2_pr, width, label='Agent 2', color='salmon')
        ax1.set_title('Trafność doboru paremii')
        ax1.set_xticks(x_pr)
        ax1.set_xticklabels(pr_labels)
        ax1.set_ylim(0, 1.1)
        ax1.set_ylabel('Score (0-1)')
        ax1.legend()
        add_value_labels(ax1, bars1, v1_pr)
        add_value_labels(ax1, bars2, v2_pr)

        # Plot 2: Decision Accuracy & F1 (0-1)
        ax2 = axes[0, 1]
        dec_vars = ["decision_accuracy", "decision_f1"]
        dec_labels = ["Accuracy", "F1 (macro)"]
        x_dec = range(len(dec_vars))
        
        v1_dec = [averages["agent1"].get(m, 0) for m in dec_vars]
        v2_dec = [averages["agent2"].get(m, 0) for m in dec_vars]
        
        bars1 = ax2.bar([p - width/2 for p in x_dec], v1_dec, width, label='Agent 1', color='skyblue')
        bars2 = ax2.bar([p + width/2 for p in x_dec], v2_dec, width, label='Agent 2', color='salmon')
        ax2.set_title('Klasyfikacja decyzji (zasadne/niezasadne)')
        ax2.set_xticks(x_dec)
        ax2.set_xticklabels(dec_labels)
        ax2.set_ylim(0, 1.1)
        ax2.set_ylabel('Score (0-1)')
        ax2.legend()
        add_value_labels(ax2, bars1, v1_dec)
        add_value_labels(ax2, bars2, v2_dec)

        # Plot 3: Justification Score (1-5)
        ax3 = axes[1, 0]
        j_vars = ["justification_score"]
        v1_j = [averages["agent1"].get("justification_score", 0)]
        v2_j = [averages["agent2"].get("justification_score", 0)]
        
        x_j = range(1)
        bars1 = ax3.bar([p - width/2 for p in x_j], v1_j, width, label='Agent 1', color='skyblue')
        bars2 = ax3.bar([p + width/2 for p in x_j], v2_j, width, label='Agent 2', color='salmon')
        ax3.set_title('Jakość uzasadnienia')
        ax3.set_xticks(x_j)
        ax3.set_xticklabels(["Score"])
        ax3.set_ylim(0, 5.5)
        ax3.set_ylabel('Score (1-5)')
        ax3.legend()
        add_value_labels(ax3, bars1, v1_j)
        add_value_labels(ax3, bars2, v2_j)
        
        # Plot 4: Combined overview (normalize to 0-1)
        ax4 = axes[1, 1]
        all_metrics = ["precision", "recall", "decision_f1", "justification_score"]
        all_labels = ["Precision", "Recall", "Decision F1", "Justification"]
        
        # Normalize justification to 0-1 scale (from 1-5)
        v1_all = [
            averages["agent1"].get("precision", 0),
            averages["agent1"].get("recall", 0),
            averages["agent1"].get("decision_f1", 0),
            (averages["agent1"].get("justification_score", 0) - 1) / 4.0  # normalize 1-5 to 0-1
        ]
        v2_all = [
            averages["agent2"].get("precision", 0),
            averages["agent2"].get("recall", 0),
            averages["agent2"].get("decision_f1", 0),
            (averages["agent2"].get("justification_score", 0) - 1) / 4.0
        ]
        
        x_all = range(len(all_metrics))
        bars1 = ax4.bar([p - width/2 for p in x_all], v1_all, width, label='Agent 1', color='skyblue')
        bars2 = ax4.bar([p + width/2 for p in x_all], v2_all, width, label='Agent 2', color='salmon')
        ax4.set_title('Zestawienie ogólne (znormalizowane 0-1)')
        ax4.set_xticks(x_all)
        ax4.set_xticklabels(all_labels, rotation=15, ha='right')
        ax4.set_ylim(0, 1.1)
        ax4.set_ylabel('Normalized Score')
        ax4.legend()
        add_value_labels(ax4, bars1, v1_all)
        add_value_labels(ax4, bars2, v2_all)
        
        plt.tight_layout()
        plt.savefig(PLOT_FILE, dpi=150)
        print(f"Wykresy zapisano w {PLOT_FILE}")
        
    except Exception as e:
        print(f"Błąd generowania wykresów: {e}")

if __name__ == "__main__":
    import sys
    output_dir = "tmp"
    if len(sys.argv) > 1 and sys.argv[1] == "--output-dir" and len(sys.argv) > 2:
        output_dir = sys.argv[2]
    elif len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        output_dir = sys.argv[1]
    main(output_dir)
