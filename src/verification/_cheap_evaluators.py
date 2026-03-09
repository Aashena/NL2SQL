"""
Mixin providing cheap (no LLM) SQL evaluation methods for QueryVerifier.

Each method performs structural or SQLite-execution checks without any API
calls, making them fast enough to run on every fix iteration.

CheapEvaluatorMixin
-------------------
Defines the five cheap test implementations as instance methods so that
QueryVerifier can inherit them while keeping this file focused on evaluation
logic only.

Methods
-------
_eval_grain      — row count within [lower, upper] bounds
_eval_null       — unexpected NULL values in result rows
_eval_duplicate  — JOIN-induced row multiplication (ratio check or exact dups)
_eval_ordering   — structural check for ORDER BY / LIMIT keywords in SQL
_eval_scale      — numeric result values within expected range
"""
from __future__ import annotations

import logging
from typing import Optional

from src.data.database import ExecutionResult, execute_sql
from src.verification._models import VerificationTestResult

logger = logging.getLogger(__name__)


class CheapEvaluatorMixin:
    """Mixin providing cheap (no-LLM) _eval_* methods for QueryVerifier."""

    def _eval_grain(
        self,
        spec,  # VerificationTestSpec
        exec_result: ExecutionResult,
        db_path: str,
    ) -> VerificationTestResult:
        """Grain test: check result row count is within [lower, upper] and
        optionally that column count matches expected."""
        actual_count = len(exec_result.rows)

        # ── 1. Resolve upper bound ──────────────────────────────────────────
        upper: Optional[int] = None

        if spec.verification_sql_upper:
            vr = execute_sql(db_path, spec.verification_sql_upper)
            if vr.success and vr.rows and isinstance(vr.rows[0][0], (int, float)):
                upper = int(vr.rows[0][0])
        elif spec.verification_sql:
            # Backward compat: old verification_sql treated as upper bound for grain
            vr = execute_sql(db_path, spec.verification_sql)
            if vr.success and vr.rows and isinstance(vr.rows[0][0], (int, float)):
                upper = int(vr.rows[0][0])
        elif spec.row_count_max is not None:
            upper = spec.row_count_max
        elif spec.expected_row_count is not None:
            # Legacy exact-match: treat as upper = expected
            upper = spec.expected_row_count

        # ── 2. Resolve lower bound (minimum is always 1) ───────────────────
        lower: int = max(spec.row_count_min if spec.row_count_min is not None else 1, 1)

        # ── 3. Row count check ─────────────────────────────────────────────
        row_count_ok: Optional[bool] = None
        row_count_msg: str = ""

        if upper is not None:
            if lower <= actual_count <= upper:
                row_count_ok = True
                row_count_msg = (
                    f"Row count {actual_count} is within range [{lower}, {upper}]."
                )
            elif actual_count < lower:
                row_count_ok = False
                row_count_msg = (
                    f"Row count {actual_count} is below minimum {lower}. "
                    "Result may be too restrictive or missing rows."
                )
            else:
                row_count_ok = False
                row_count_msg = (
                    f"Row count {actual_count} exceeds upper bound {upper}. "
                    "Possible wrong grain or missing GROUP BY / DISTINCT."
                )
        else:
            # No upper bound — only enforce lower
            if actual_count >= lower:
                row_count_ok = True
                row_count_msg = f"Row count {actual_count} meets minimum {lower}."
            else:
                row_count_ok = False
                row_count_msg = (
                    f"Row count {actual_count} is below minimum {lower} "
                    "(empty results are always wrong)."
                )

        # ── 4. Column count check (independent of row count) ───────────────
        col_count_ok: Optional[bool] = None
        col_count_msg: str = ""

        if spec.expected_column_count is not None and exec_result.rows:
            actual_cols = len(exec_result.rows[0])
            if actual_cols == spec.expected_column_count:
                col_count_ok = True
                col_count_msg = (
                    f"Column count {actual_cols} matches expected "
                    f"{spec.expected_column_count}."
                )
            else:
                col_count_ok = False
                col_count_msg = (
                    f"Column count {actual_cols} does not match expected "
                    f"{spec.expected_column_count}. Check SELECT list."
                )

        # ── 5. Combine results ─────────────────────────────────────────────
        if row_count_ok is None and col_count_ok is None:
            return VerificationTestResult(
                test_type="grain",
                status="skip",
                actual_outcome="No grain bounds or expected column count specified.",
                is_critical=True,
                computed_upper_bound=upper,
                actual_row_count=actual_count,
            )

        row_fail = row_count_ok is False
        col_fail = col_count_ok is False

        if row_fail or col_fail:
            parts = []
            if row_fail:
                parts.append(row_count_msg)
            if col_fail:
                parts.append(col_count_msg)
            return VerificationTestResult(
                test_type="grain",
                status="fail",
                actual_outcome=" | ".join(parts),
                is_critical=True,
                computed_upper_bound=upper,
                actual_row_count=actual_count,
            )

        parts = []
        if row_count_ok is True:
            parts.append(row_count_msg)
        if col_count_ok is True:
            parts.append(col_count_msg)
        return VerificationTestResult(
            test_type="grain",
            status="pass",
            actual_outcome=" | ".join(parts),
            is_critical=True,
            computed_upper_bound=upper,
            actual_row_count=actual_count,
        )

    def _eval_null(
        self,
        spec,  # VerificationTestSpec
        exec_result: ExecutionResult,
    ) -> VerificationTestResult:
        """Null test: check result rows for unexpected NULL values."""
        rows = exec_result.rows
        if not rows:
            return VerificationTestResult(
                test_type="null",
                status="skip",
                actual_outcome="No rows to check for NULLs.",
                is_critical=False,
            )

        null_count = sum(1 for row in rows for val in row if val is None)
        total_cells = sum(len(row) for row in rows)

        if null_count == 0:
            return VerificationTestResult(
                test_type="null",
                status="pass",
                actual_outcome=f"No NULL values in {len(rows)} result rows.",
                is_critical=False,
            )
        null_pct = (null_count / total_cells * 100) if total_cells > 0 else 0
        return VerificationTestResult(
            test_type="null",
            status="fail",
            actual_outcome=(
                f"Found {null_count} NULL values across {len(rows)} rows "
                f"({null_pct:.1f}% of cells). May indicate missing JOIN conditions."
            ),
            is_critical=False,
        )

    def _eval_duplicate(
        self,
        spec,  # VerificationTestSpec
        exec_result: ExecutionResult,
        db_path: str,
    ) -> VerificationTestResult:
        """Duplicate test: detect JOIN-induced row multiplication."""
        actual_count = len(exec_result.rows)

        if spec.verification_sql:
            vr = execute_sql(db_path, spec.verification_sql)
            if vr.success and vr.rows and isinstance(vr.rows[0][0], (int, float)):
                expected = int(vr.rows[0][0])
                if expected > 0:
                    ratio = actual_count / expected
                    if ratio >= 2.0:
                        return VerificationTestResult(
                            test_type="duplicate",
                            status="fail",
                            actual_outcome=(
                                f"Result has {actual_count} rows but expected ~{expected} "
                                f"distinct entities (ratio: {ratio:.1f}x). "
                                "JOIN may be multiplying rows."
                            ),
                            is_critical=True,
                        )
                    return VerificationTestResult(
                        test_type="duplicate",
                        status="pass",
                        actual_outcome=(
                            f"Row count {actual_count} vs expected {expected} "
                            "— no JOIN multiplication detected."
                        ),
                        is_critical=True,
                    )

        # Fallback: check for exact duplicate rows in result
        if exec_result.rows:
            try:
                unique_rows = len(set(exec_result.rows))
            except TypeError:
                # Rows contain unhashable types (e.g. lists) — skip
                return VerificationTestResult(
                    test_type="duplicate",
                    status="skip",
                    actual_outcome="Result rows contain unhashable values — skipped.",
                    is_critical=True,
                )
            if unique_rows < actual_count:
                dup_count = actual_count - unique_rows
                return VerificationTestResult(
                    test_type="duplicate",
                    status="fail",
                    actual_outcome=(
                        f"Found {dup_count} duplicate rows ({actual_count} total, "
                        f"{unique_rows} unique). JOIN may be creating duplicates."
                    ),
                    is_critical=True,
                )
            return VerificationTestResult(
                test_type="duplicate",
                status="pass",
                actual_outcome=f"All {actual_count} rows are unique — no duplicates.",
                is_critical=True,
            )

        return VerificationTestResult(
            test_type="duplicate",
            status="skip",
            actual_outcome="No rows to check for duplicates.",
            is_critical=True,
        )

    def _eval_ordering(
        self,
        spec,  # VerificationTestSpec
        sql: str,
    ) -> VerificationTestResult:
        """Ordering test: structural check that required SQL clauses are present."""
        required = spec.required_sql_keywords or ["ORDER BY", "LIMIT"]
        sql_upper = sql.upper()
        missing = [kw for kw in required if kw.upper() not in sql_upper]

        if not missing:
            return VerificationTestResult(
                test_type="ordering",
                status="pass",
                actual_outcome=f"All required clauses present: {required}.",
                is_critical=False,
            )
        return VerificationTestResult(
            test_type="ordering",
            status="fail",
            actual_outcome=f"Missing required SQL clauses: {missing}.",
            is_critical=False,
        )

    def _eval_scale(
        self,
        spec,  # VerificationTestSpec
        exec_result: ExecutionResult,
    ) -> VerificationTestResult:
        """Scale test: check that numeric result values are within expected range."""
        if spec.numeric_min is None and spec.numeric_max is None:
            return VerificationTestResult(
                test_type="scale",
                status="skip",
                actual_outcome="No numeric range specified.",
                is_critical=False,
            )

        rows = exec_result.rows
        if not rows:
            return VerificationTestResult(
                test_type="scale",
                status="skip",
                actual_outcome="No rows to check.",
                is_critical=False,
            )

        violations: list[str] = []
        for row in rows:
            for val in row:
                if not isinstance(val, (int, float)):
                    continue
                if spec.numeric_min is not None and val < spec.numeric_min:
                    violations.append(f"{val} < min({spec.numeric_min})")
                if spec.numeric_max is not None and val > spec.numeric_max:
                    violations.append(f"{val} > max({spec.numeric_max})")

        if not violations:
            bounds = []
            if spec.numeric_min is not None:
                bounds.append(f"min={spec.numeric_min}")
            if spec.numeric_max is not None:
                bounds.append(f"max={spec.numeric_max}")
            return VerificationTestResult(
                test_type="scale",
                status="pass",
                actual_outcome=(
                    f"All numeric values within expected range ({', '.join(bounds)})."
                ),
                is_critical=False,
            )
        sample = violations[:3]
        return VerificationTestResult(
            test_type="scale",
            status="fail",
            actual_outcome=(
                f"Out-of-range values: {sample}. May indicate wrong column or unit."
            ),
            is_critical=False,
        )
