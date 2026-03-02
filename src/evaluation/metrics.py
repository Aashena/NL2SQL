"""
Aggregate metrics computation for NL2SQL evaluation results.

Takes a list of EvaluationResult objects and returns a comprehensive
metrics dictionary broken down by difficulty, database, selection method,
and winner generator.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.evaluation.evaluator import EvaluationResult


def aggregate_metrics(results: "list[EvaluationResult]") -> dict:
    """
    Compute aggregate metrics from a list of EvaluationResult objects.

    Parameters
    ----------
    results:
        List of EvaluationResult objects (or any objects with matching
        attributes). Empty list returns a dict of zeros/empty dicts.

    Returns
    -------
    dict with keys:
        overall_ex               float  — fraction correct overall
        ex_by_difficulty         dict   — difficulty str → float
        ex_by_db                 dict   — db_id str → float
        ex_by_selection_method   dict   — selection_method str → float
        ex_by_winner_generator   dict   — winner_generator str → float
        fast_path_rate           float  — fraction where selection_method == "fast_path"
        avg_fix_count            float  — mean fix_count
        full_failure_rate        float  — fraction where predicted_sql=="" or method=="error"
        avg_latency              float  — mean latency_seconds
        total_cost               float  — sum of cost_estimate
    """
    if not results:
        return {
            "overall_ex": 0.0,
            "ex_by_difficulty": {},
            "ex_by_db": {},
            "ex_by_selection_method": {},
            "ex_by_winner_generator": {},
            "fast_path_rate": 0.0,
            "avg_fix_count": 0.0,
            "full_failure_rate": 0.0,
            "avg_latency": 0.0,
            "total_cost": 0.0,
        }

    total = len(results)
    correct_total = sum(1 for r in results if r.correct)

    # -----------------------------------------------------------------------
    # Breakdown helpers
    # -----------------------------------------------------------------------

    def _ex_by_field(field_name: str) -> dict:
        """Group by field_name and compute fraction correct for each group."""
        counts: dict[str, list[int]] = defaultdict(list)
        for r in results:
            key = getattr(r, field_name, "unknown")
            counts[key].append(1 if r.correct else 0)
        return {
            k: sum(v) / len(v)
            for k, v in sorted(counts.items())
        }

    # -----------------------------------------------------------------------
    # Fast-path rate
    # -----------------------------------------------------------------------
    fast_path_count = sum(
        1 for r in results if getattr(r, "selection_method", "") == "fast_path"
    )

    # -----------------------------------------------------------------------
    # Average fix count (handle missing field gracefully)
    # -----------------------------------------------------------------------
    fix_counts = [getattr(r, "fix_count", 0) or 0 for r in results]
    avg_fix = sum(fix_counts) / total

    # -----------------------------------------------------------------------
    # Full failure rate: predicted_sql == "" or selection_method == "error"
    # -----------------------------------------------------------------------
    full_failures = sum(
        1 for r in results
        if (getattr(r, "predicted_sql", "") == "" or
            getattr(r, "selection_method", "") == "error")
    )

    # -----------------------------------------------------------------------
    # Latency
    # -----------------------------------------------------------------------
    latencies = [getattr(r, "latency_seconds", 0.0) or 0.0 for r in results]
    avg_latency = sum(latencies) / total

    # -----------------------------------------------------------------------
    # Total cost (handle missing field gracefully)
    # -----------------------------------------------------------------------
    costs = [getattr(r, "cost_estimate", 0.0) or 0.0 for r in results]
    total_cost = sum(costs)

    return {
        "overall_ex": correct_total / total,
        "ex_by_difficulty": _ex_by_field("difficulty"),
        "ex_by_db": _ex_by_field("db_id"),
        "ex_by_selection_method": _ex_by_field("selection_method"),
        "ex_by_winner_generator": _ex_by_field("winner_generator"),
        "fast_path_rate": fast_path_count / total,
        "avg_fix_count": avg_fix,
        "full_failure_rate": full_failures / total,
        "avg_latency": avg_latency,
        "total_cost": total_cost,
    }
