#!/usr/bin/env python3
"""
Analyze NL2SQL evaluation results from a JSON file.

Usage:
    python scripts/analyze_results.py <results_json_path>

The JSON file should contain a list of dicts representing evaluation results.
Handles both EvaluationEntry (from run_evaluation.py, no fix_count/cost_estimate)
and EvaluationResult (from src/evaluation/evaluator.py, has fix_count/cost_estimate).
Missing fields are filled with safe defaults.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `src` imports work
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.evaluation.metrics import aggregate_metrics


# ---------------------------------------------------------------------------
# Result adapter (handles both EvaluationEntry and EvaluationResult fields)
# ---------------------------------------------------------------------------

def _load_result(d: dict) -> SimpleNamespace:
    """
    Create a SimpleNamespace from a result dict, supplying defaults for
    fields that may be absent in EvaluationEntry (produced by run_evaluation.py).
    """
    return SimpleNamespace(
        question_id=d.get("question_id", -1),
        db_id=d.get("db_id", "unknown"),
        difficulty=d.get("difficulty", "unknown"),
        predicted_sql=d.get("predicted_sql", ""),
        truth_sql=d.get("truth_sql", ""),
        correct=bool(d.get("correct", False)),
        selection_method=d.get("selection_method", "unknown"),
        winner_generator=d.get("winner_generator", "unknown"),
        cluster_count=d.get("cluster_count", 0),
        fix_count=d.get("fix_count", 0),          # not in EvaluationEntry
        latency_seconds=d.get("latency_seconds", 0.0),
        cost_estimate=d.get("cost_estimate", 0.0),  # not in EvaluationEntry
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _print_table(title: str, data: dict[str, float], sort_by_value: bool = False) -> None:
    """Print a two-column table of label → EX%."""
    print(f"\n--- {title} ---")
    if not data:
        print("  (no data)")
        return

    items = list(data.items())
    if sort_by_value:
        items = sorted(items, key=lambda x: x[1])  # ascending (worst first)
    else:
        items = sorted(items, key=lambda x: x[0])  # alphabetical

    col_width = max(len(k) for k, _ in items) + 2
    for label, ex in items:
        bar_filled = int(ex * 20)
        bar = "#" * bar_filled + "." * (20 - bar_filled)
        print(f"  {label:<{col_width}} {_pct(ex):>7}  [{bar}]")


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyze(results_path: str) -> None:
    path = Path(results_path)
    if not path.exists():
        print(f"ERROR: File not found: {results_path}", file=sys.stderr)
        sys.exit(1)

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        print("ERROR: Expected a JSON array at the top level.", file=sys.stderr)
        sys.exit(1)

    results = [_load_result(d) for d in raw]
    total = len(results)

    if total == 0:
        print("No results in file.")
        return

    metrics = aggregate_metrics(results)

    # -----------------------------------------------------------------------
    # Overall
    # -----------------------------------------------------------------------
    correct_count = sum(1 for r in results if r.correct)
    print(f"\n{'='*65}")
    print(f"NL2SQL Evaluation Results: {path.name}")
    print(f"{'='*65}")
    print(f"Total questions  : {total}")
    print(f"Correct          : {correct_count}")
    print(f"Overall EX       : {_pct(metrics['overall_ex'])}")
    print(f"{'='*65}")

    # -----------------------------------------------------------------------
    # EX by difficulty
    # -----------------------------------------------------------------------
    _print_table("EX by Difficulty", metrics["ex_by_difficulty"])

    # -----------------------------------------------------------------------
    # EX by database (sorted ascending by EX to highlight worst)
    # -----------------------------------------------------------------------
    _print_table("EX by Database (worst first)", metrics["ex_by_db"], sort_by_value=True)

    # -----------------------------------------------------------------------
    # EX by selection method
    # -----------------------------------------------------------------------
    _print_table("EX by Selection Method", metrics["ex_by_selection_method"])

    # -----------------------------------------------------------------------
    # EX by winner generator
    # -----------------------------------------------------------------------
    _print_table("EX by Winner Generator", metrics["ex_by_winner_generator"])

    # -----------------------------------------------------------------------
    # Worst 3 databases with analysis
    # -----------------------------------------------------------------------
    db_results: dict[str, list] = defaultdict(list)
    for r in results:
        db_results[r.db_id].append(r)

    db_ex = {
        db_id: sum(1 for r in rs if r.correct) / len(rs)
        for db_id, rs in db_results.items()
    }
    worst_3 = sorted(db_ex.items(), key=lambda x: x[1])[:3]

    print(f"\n--- Worst 3 Databases (detailed) ---")
    for db_id, ex in worst_3:
        rs = db_results[db_id]
        n_correct = sum(1 for r in rs if r.correct)
        n_total_db = len(rs)
        avg_lat = sum(r.latency_seconds for r in rs) / n_total_db
        methods_count: dict[str, int] = defaultdict(int)
        for r in rs:
            methods_count[r.selection_method] += 1
        methods_str = ", ".join(f"{m}={c}" for m, c in sorted(methods_count.items()))

        print(f"\n  {db_id}")
        print(f"    EX        : {n_correct}/{n_total_db} = {_pct(ex)}")
        print(f"    Avg latency: {avg_lat:.1f}s")
        print(f"    Methods   : {methods_str}")

        # Show incorrect questions
        incorrect = [r for r in rs if not r.correct]
        if incorrect:
            print(f"    Incorrect questions ({len(incorrect)}):")
            for r in incorrect[:5]:  # Show at most 5
                print(f"      Q#{r.question_id} [{r.difficulty}] method={r.selection_method}")
                pred_snippet = r.predicted_sql[:80].replace("\n", " ")
                truth_snippet = r.truth_sql[:80].replace("\n", " ")
                print(f"        predicted: {pred_snippet}")
                print(f"        truth    : {truth_snippet}")
            if len(incorrect) > 5:
                print(f"      ... and {len(incorrect) - 5} more")

    # -----------------------------------------------------------------------
    # Operational metrics
    # -----------------------------------------------------------------------
    print(f"\n--- Operational Metrics ---")
    print(f"  Fast path rate   : {_pct(metrics['fast_path_rate'])}")
    print(f"  Avg fix count    : {metrics['avg_fix_count']:.2f}")
    print(f"  Full failure rate: {_pct(metrics['full_failure_rate'])}")
    print(f"  Avg latency      : {metrics['avg_latency']:.2f}s/question")
    total_cost = metrics["total_cost"]
    if total_cost > 0:
        print(f"  Total cost       : ${total_cost:.4f}")
        print(f"  Cost per question: ${total_cost / total:.4f}")

    print(f"\n{'='*65}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python scripts/analyze_results.py <results_json_path>")
        sys.exit(0 if sys.argv[1:] == ["--help"] else 1)

    analyze(sys.argv[1])


if __name__ == "__main__":
    main()
