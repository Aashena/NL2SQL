"""
Execution Accuracy (EX) evaluator for NL2SQL.

Key design:
- compare_results: normalizes cell values, sorts rows, compares result sets.
  Column order independence is achieved by sorting values *within* each row
  before sorting the list of rows. This is consistent with BIRD evaluation.
- compute_ex: executes both SQLs, returns bool.
- EvaluationResult: comprehensive dataclass capturing per-question metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from src.data.database import execute_sql

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Value normalization
# ---------------------------------------------------------------------------

def _normalize_cell(v) -> str:
    """
    Normalize a single result cell for comparison.

    Rules (in order):
      - None  → "NULL_SENTINEL"
      - bool  → str(int(v))       (True→"1", False→"0")
      - float → Decimal with 6 decimal places, then string
      - int   → str(v)
      - else  → str(v)

    Note: bool must be checked *before* int because bool is a subclass of int.
    """
    if v is None:
        return "NULL_SENTINEL"
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, float):
        return str(Decimal(f"{v:.6f}"))
    return str(v)


# ---------------------------------------------------------------------------
# Result set comparison
# ---------------------------------------------------------------------------

def compare_results(predicted_rows: list, truth_rows: list) -> bool:
    """
    Compare two result sets for equivalence after normalization.

    Column order independence: values within each row are sorted before the
    list of rows is sorted. This treats each row as a multiset of values,
    which is consistent with how BIRD evaluation handles column order.

    Parameters
    ----------
    predicted_rows:
        List of tuples (or lists) from the predicted query.
    truth_rows:
        List of tuples (or lists) from the ground-truth query.

    Returns
    -------
    bool
        True if the normalized, sorted result sets are identical.
    """
    def _normalize_row(row) -> tuple:
        normalized = [_normalize_cell(v) for v in row]
        return tuple(sorted(normalized))

    pred_normalized = sorted(_normalize_row(r) for r in predicted_rows)
    truth_normalized = sorted(_normalize_row(r) for r in truth_rows)
    return pred_normalized == truth_normalized


# ---------------------------------------------------------------------------
# Execution Accuracy computation
# ---------------------------------------------------------------------------

def compute_ex(predicted_sql: str, truth_sql: str, db_path: str) -> bool:
    """
    Compute Execution Accuracy for a single question.

    Parameters
    ----------
    predicted_sql:
        The SQL predicted by the model.
    truth_sql:
        The ground-truth SQL.
    db_path:
        Path to the SQLite database file.

    Returns
    -------
    bool
        True if the predicted result set matches the truth result set.
    """
    # Handle empty predicted SQL immediately
    if not predicted_sql or not predicted_sql.strip():
        return False

    # Execute predicted SQL
    pred_result = execute_sql(db_path, predicted_sql)
    if not pred_result.success:
        return False

    # Execute truth SQL
    truth_result = execute_sql(db_path, truth_sql)
    if not truth_result.success:
        logger.warning(
            "Truth SQL execution failed for db=%s: %s — skipping evaluation",
            db_path,
            truth_result.error,
        )
        return False

    return compare_results(pred_result.rows, truth_result.rows)


# ---------------------------------------------------------------------------
# EvaluationResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """
    Complete per-question evaluation record.

    Used by tests and the analyze_results.py script.
    Note: EvaluationEntry in run_evaluation.py is a lighter variant without
    fix_count and cost_estimate — both share the same JSON keys where they
    overlap.
    """
    question_id: int
    db_id: str
    difficulty: str
    predicted_sql: str
    truth_sql: str
    correct: bool
    selection_method: str        # fast_path | tournament | fallback | error
    winner_generator: str        # generator_id with most wins, or "fallback"
    cluster_count: int
    fix_count: int               # total fix iterations across all candidates
    latency_seconds: float
    cost_estimate: float         # estimated API cost in USD
