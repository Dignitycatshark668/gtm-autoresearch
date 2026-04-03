"""Review experiment journal - find best runs, analyze convergence, compare weights."""

import json
import sys
from pathlib import Path


def load_journal(path: str) -> list:
    """Load JSONL experiment journal."""
    entries = []
    with open(path) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def review(journal_path: str):
    entries = load_journal(journal_path)
    if not entries:
        print("Empty journal.")
        return

    wins = [e for e in entries if e["outcome"] == "win"]
    losses = [e for e in entries if e["outcome"] == "lose"]

    metric = entries[0].get("metric", "unknown")
    first_value = entries[0].get("baseline_value", 0)
    best = max(entries, key=lambda e: e.get("variant_value", 0))
    best_value = best["variant_value"]

    print(f"{'='*60}")
    print(f"EXPERIMENT JOURNAL REVIEW")
    print(f"{'='*60}")
    print(f"Total experiments: {len(entries)}")
    print(f"Wins: {len(wins)} ({100*len(wins)/len(entries):.1f}%)")
    print(f"Metric: {metric}")
    print(f"Start: {first_value:.4f} -> Best: {best_value:.4f} (+{best_value - first_value:.4f})")

    # Best weights
    print(f"\nBest weights ({best['experiment_id']}):")
    weights = best.get("weights_snapshot", {})
    for feat, w in sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {feat:25s} {w:>4d}")

    # Convergence: when did improvements stop?
    win_indices = [i for i, e in enumerate(entries) if e["outcome"] == "win"]
    if len(win_indices) >= 2:
        last_win = win_indices[-1]
        second_last = win_indices[-2]
        print(f"\nConvergence:")
        print(f"  Last win at experiment {last_win + 1} of {len(entries)}")
        print(f"  Gap since previous win: {last_win - second_last} experiments")
        if last_win < len(entries) * 0.5:
            print(f"  Note: converged in first half - may benefit from more exploration")

    # NRR by tier in best run
    best_metrics = best.get("metrics", {})
    if "nrr_by_tier" in best_metrics:
        print(f"\nNRR by tier (best run):")
        for tier, nrr in sorted(best_metrics["nrr_by_tier"].items()):
            print(f"  {tier}: {nrr:.1%}")

    if "logo_retention_by_tier" in best_metrics:
        print(f"\nLogo retention by tier (best run):")
        for tier, ret in sorted(best_metrics["logo_retention_by_tier"].items()):
            print(f"  {tier}: {ret:.1%}")

    if "revenue_concentration_hhi" in best_metrics:
        hhi = best_metrics["revenue_concentration_hhi"]
        print(f"\nRevenue concentration HHI: {hhi:.0f}", end="")
        if hhi > 4000:
            print(" (concentrated - one tier dominates)")
        elif hhi > 2500:
            print(" (moderate)")
        else:
            print(" (well distributed)")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "experiment_journal.jsonl"
    review(path)
