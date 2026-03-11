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
import re
from typing import Optional

from src.data.database import ExecutionResult, execute_sql
from src.verification._models import VerificationTestResult

logger = logging.getLogger(__name__)


def _is_safe_verification_sql(sql: str) -> bool:
    """Return True for any safe read-only SELECT query (no DDL/DML).

    The COUNT requirement was removed: the grain prompt still instructs the model
    to use COUNT queries, but valid upper-bound queries such as `SELECT 1` or
    subquery-based SELECTs without a top-level COUNT token are also safe to run.
    """
    s = sql.strip().upper()
    return s.startswith("SELECT") and not any(
        kw in s
        for kw in ("INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "ATTACH")
    )


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
        # Skip the entire grain test when LLM signalled it could not determine a bound
        if spec.upper_bound_confidence == "none":
            return VerificationTestResult(
                test_type="grain",
                status="skip",
                actual_outcome="Skipped — LLM could not determine a meaningful upper bound.",
                is_critical=False,
            )

        actual_count = len(exec_result.rows)

        # ── 0. Resolve lower bound (must precede upper-bound resolution) ──────
        # Default to 1; allow explicit 0 for questions that may return no rows.
        lower: int = max(spec.row_count_min if spec.row_count_min is not None else 1, 0)

        # ── 1. Resolve upper bound ──────────────────────────────────────────
        upper: Optional[int] = None

        if spec.verification_sql_upper:
            if not _is_safe_verification_sql(spec.verification_sql_upper):
                logger.warning(
                    "Grain verification_sql_upper rejected by safety check (not a SELECT COUNT): %r",
                    spec.verification_sql_upper[:120],
                )
            else:
                vr = execute_sql(db_path, spec.verification_sql_upper)
                if vr.success and vr.rows and isinstance(vr.rows[0][0], (int, float)):
                    raw_upper = int(vr.rows[0][0])
                    if raw_upper == 0 and lower > 0:
                        # A COUNT of 0 almost certainly means the verification query
                        # itself is wrong (hallucinated table name, impossible condition).
                        # Skip the upper-bound check rather than falsely failing every
                        # non-empty result.
                        logger.warning(
                            "Grain verification_sql_upper returned 0 (lower=%d) — "
                            "skipping upper-bound check (verification query likely wrong): %r",
                            lower,
                            spec.verification_sql_upper[:120],
                        )
                    else:
                        upper = raw_upper
        elif spec.verification_sql:
            # Backward compat: old verification_sql treated as upper bound for grain
            if _is_safe_verification_sql(spec.verification_sql):
                vr = execute_sql(db_path, spec.verification_sql)
                if vr.success and vr.rows and isinstance(vr.rows[0][0], (int, float)):
                    raw_upper = int(vr.rows[0][0])
                    if raw_upper == 0 and lower > 0:
                        logger.warning(
                            "Grain verification_sql returned 0 (lower=%d) — skipping upper-bound check",
                            lower,
                        )
                    else:
                        upper = raw_upper
        elif spec.expected_row_count is not None:
            # Legacy exact-match: treat as upper = expected
            upper = spec.expected_row_count

        # ── 2. Row count check ─────────────────────────────────────────────
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
                spec.fix_hint = (
                    f"Result returned {actual_count} rows but at least {lower} are expected. "
                    "The WHERE clause may be too restrictive, a JOIN condition may eliminate "
                    "valid rows, or HAVING may filter out groups that should appear."
                )
            else:
                row_count_ok = False
                row_count_msg = (
                    f"Row count {actual_count} exceeds upper bound {upper}. "
                    "Possible wrong grain or missing GROUP BY / DISTINCT."
                )
                spec.fix_hint = (
                    f"Result returned {actual_count} rows but the grain upper bound is {upper}. "
                    "Add DISTINCT to SELECT or GROUP BY the primary entity to collapse duplicates. "
                    "Check for fan-out JOINs (one entity → many rows in joined table)."
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
                spec.fix_hint = (
                    f"Result returned {actual_count} rows but at least {lower} are expected. "
                    "The WHERE clause may be too restrictive, a JOIN condition may eliminate "
                    "valid rows, or HAVING may filter out groups that should appear."
                )

        # ── 3. Column count check (independent of row count) ───────────────
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

        # ── 4. Combine results ─────────────────────────────────────────────
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
        """Null test: check result rows for unexpected NULL values.

        When check_columns is specified, we only fail if more than 10% of cells
        across the first len(check_columns) columns contain NULLs — this avoids
        false positives from LEFT JOIN columns that are intentionally NULL in
        some rows.  When check_columns is absent, any NULL triggers a fail.
        """
        rows = exec_result.rows
        if not rows:
            return VerificationTestResult(
                test_type="null",
                status="skip",
                actual_outcome="No rows to check for NULLs.",
                is_critical=False,
            )

        check_columns = spec.check_columns or []
        null_count = sum(1 for row in rows for val in row if val is None)
        total_cells = sum(len(row) for row in rows)
        null_pct = (null_count / total_cells * 100) if total_cells > 0 else 0

        if check_columns:
            # check_columns are specified but we cannot reliably map column names to
            # result positions without SQL parsing. Instead, apply a lenient threshold
            # over ALL cells: only fail when > 10% of cells are NULL.  This tolerates
            # incidental NULLs in non-checked columns (e.g. from LEFT JOINs) while
            # still catching queries where a significant fraction of rows are missing data.
            null_threshold_pct = 10.0
            col_hint = f" (checking columns: {', '.join(check_columns)})"
            if null_count == 0:
                return VerificationTestResult(
                    test_type="null",
                    status="pass",
                    actual_outcome=f"No NULL values in {len(rows)} result rows{col_hint}.",
                    is_critical=False,
                )
            if null_pct <= null_threshold_pct:
                return VerificationTestResult(
                    test_type="null",
                    status="pass",
                    actual_outcome=(
                        f"Found {null_count} NULL values ({null_pct:.1f}% of cells) "
                        f"— below threshold{col_hint}."
                    ),
                    is_critical=False,
                )
            return VerificationTestResult(
                test_type="null",
                status="fail",
                actual_outcome=(
                    f"Found {null_count} NULL values in {len(rows)} rows "
                    f"({null_pct:.1f}% of cells){col_hint}. "
                    "May indicate missing JOIN conditions or wrong column selected."
                ),
                is_critical=False,
            )
        else:
            # No check_columns specified — fail on any NULL (strict behaviour)
            if null_count == 0:
                return VerificationTestResult(
                    test_type="null",
                    status="pass",
                    actual_outcome=f"No NULL values in {len(rows)} result rows.",
                    is_critical=False,
                )
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
        # Only require LIMIT in the fallback when an explicit order_limit count is set;
        # for superlatives without a count ('which X has the highest Y?') ORDER BY alone suffices.
        default_keywords = ["ORDER BY", "LIMIT"] if spec.order_limit else ["ORDER BY"]
        required = spec.required_sql_keywords or default_keywords
        sql_upper = sql.upper()
        missing = [kw for kw in required if kw.upper() not in sql_upper]

        if missing:
            return VerificationTestResult(
                test_type="ordering",
                status="fail",
                actual_outcome=f"Missing required SQL clauses: {missing}.",
                is_critical=False,
            )

        # Additional check: if an explicit order_limit N was specified, verify that
        # any LIMIT clause in the SQL is at least N (not a smaller value).
        if spec.order_limit and "LIMIT" in sql_upper:
            limit_match = re.search(r"\bLIMIT\s+(\d+)", sql, re.IGNORECASE)
            if limit_match:
                actual_limit = int(limit_match.group(1))
                if actual_limit < spec.order_limit:
                    return VerificationTestResult(
                        test_type="ordering",
                        status="fail",
                        actual_outcome=(
                            f"LIMIT {actual_limit} is less than the required {spec.order_limit} "
                            f"rows. Query returns fewer rows than the question demands."
                        ),
                        is_critical=False,
                    )

        return VerificationTestResult(
            test_type="ordering",
            status="pass",
            actual_outcome=f"All required clauses present: {required}.",
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
