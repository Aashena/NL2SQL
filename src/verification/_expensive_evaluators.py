"""
Mixin providing structurally-complex SQL evaluation methods for QueryVerifier.

Despite being classified as "expensive" tests (run alongside LLM-judged tests),
the two methods in this mixin make NO LLM calls:

  _eval_column_alignment  — structural check comparing len(result row) to expected
                            count; async only for interface consistency.
  _eval_symmetry          — executes a complementary SQL and compares the first
                            numeric value to the candidate result within 1% tolerance.

The third "expensive" test, _eval_boundary, calls get_client() and therefore
must remain defined in query_verifier.py so that test patches targeting
``src.verification.query_verifier.get_client`` continue to intercept it.

ExpensiveEvaluatorMixin
-----------------------
Inherited by QueryVerifier alongside CheapEvaluatorMixin.
"""
from __future__ import annotations

import logging
from typing import Optional

from src.data.database import ExecutionResult, execute_sql
from src.verification._models import VerificationTestResult

logger = logging.getLogger(__name__)


class ExpensiveEvaluatorMixin:
    """Mixin providing _eval_column_alignment and _eval_symmetry for QueryVerifier."""

    async def _eval_column_alignment(
        self,
        spec,  # VerificationTestSpec
        sql: str,
        exec_result: ExecutionResult,
    ) -> VerificationTestResult:
        """Column alignment: structural check that actual column count matches expected.

        Cheap test — no LLM call. Compares len(exec_result.rows[0]) to
        spec.expected_column_count. Returns 'skip' if expected_column_count
        is not set or if there are no rows.
        """
        logger.debug("Running cheap structural test: column_alignment")

        if spec.expected_column_count is None:
            return VerificationTestResult(
                test_type="column_alignment",
                status="skip",
                actual_outcome="expected_column_count not set — cannot verify column count.",
                is_critical=True,
            )

        if not exec_result.rows:
            return VerificationTestResult(
                test_type="column_alignment",
                status="skip",
                actual_outcome="No rows returned — column count cannot be checked.",
                is_critical=True,
            )

        actual = len(exec_result.rows[0])
        expected = spec.expected_column_count

        if actual == expected:
            return VerificationTestResult(
                test_type="column_alignment",
                status="pass",
                actual_outcome=f"Column count {actual} matches expected {expected}.",
                is_critical=True,
            )

        col_desc_str = (
            f" The SELECT list should contain exactly these columns in order: "
            f"{', '.join(spec.column_descriptions)}."
            if spec.column_descriptions
            else ""
        )
        return VerificationTestResult(
            test_type="column_alignment",
            status="fail",
            actual_outcome=(
                f"Column count mismatch: query returned {actual} column(s) but "
                f"expected {expected}.{col_desc_str} "
                "Remove extra columns or add missing ones."
            ),
            is_critical=True,
        )

    def _eval_symmetry(
        self,
        spec,  # VerificationTestSpec
        exec_result: ExecutionResult,
        db_path: str,
    ) -> VerificationTestResult:
        """Symmetry test: verify an aggregate total equals the sum of sub-groups.

        Classified as an expensive test and runs on every fix iteration alongside
        column_alignment and boundary (run_expensive is always True in QueryFixer).
        Unlike column_alignment and boundary, this test makes NO LLM calls — it
        executes a complementary SQL query against the raw database and compares
        the first numeric value to the candidate result within a 1% tolerance.

        When to use: questions like "how many total X?" where the LLM plan can
        supply a verification_sql that computes the expected total a different way
        (e.g. sum of sub-groups). Applicable to roughly <10% of BIRD questions.

        Requires spec.verification_sql to be set; returns "skip" otherwise.
        """
        logger.debug("Running expensive test: symmetry")

        if not spec.verification_sql:
            logger.debug("symmetry: skip — no verification_sql provided")
            return VerificationTestResult(
                test_type="symmetry",
                status="skip",
                actual_outcome="No verification_sql provided for symmetry check.",
                is_critical=False,
            )

        vr = execute_sql(db_path, spec.verification_sql)
        if not vr.success or not vr.rows:
            logger.warning(
                "symmetry: verification SQL failed — %s", vr.error
            )
            return VerificationTestResult(
                test_type="symmetry",
                status="error",
                actual_outcome=f"Symmetry verification SQL failed: {vr.error}",
                is_critical=False,
            )

        # Extract first numeric value from the candidate result
        main_val: Optional[float] = None
        for val in (exec_result.rows[0] if exec_result.rows else []):
            if isinstance(val, (int, float)):
                main_val = float(val)
                break

        # Extract first numeric value from the symmetry verification result
        check_val: Optional[float] = None
        for val in vr.rows[0]:
            if isinstance(val, (int, float)):
                check_val = float(val)
                break

        if main_val is None or check_val is None:
            logger.debug(
                "symmetry: skip — could not extract numeric values "
                "(main_val=%s, check_val=%s)", main_val, check_val
            )
            return VerificationTestResult(
                test_type="symmetry",
                status="skip",
                actual_outcome="Could not extract numeric values for symmetry comparison.",
                is_critical=False,
            )

        tolerance = max(abs(main_val) * 0.01, 0.01)
        if abs(main_val - check_val) <= tolerance:
            logger.debug(
                "symmetry: PASS — main=%s matches check=%s (tolerance=%.4f)",
                main_val, check_val, tolerance,
            )
            return VerificationTestResult(
                test_type="symmetry",
                status="pass",
                actual_outcome=(
                    f"Main result {main_val} matches symmetry check {check_val}."
                ),
                is_critical=False,
            )

        logger.info(
            "symmetry: FAIL — main=%s, check=%s, delta=%.4f (tolerance=%.4f). "
            "Result may be overcounting or undercounting.",
            main_val, check_val, abs(main_val - check_val), tolerance,
        )
        return VerificationTestResult(
            test_type="symmetry",
            status="fail",
            actual_outcome=(
                f"Symmetry mismatch: main={main_val}, check={check_val}. "
                "Result may be overcounting or undercounting."
            ),
            is_critical=False,
        )
