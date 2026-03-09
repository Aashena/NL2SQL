"""
Semantic Verification for SQL Candidates (integrated into Op 8 fix loop).

QueryVerifier has two responsibilities:

  1. generate_plan() — one LLM call (model_powerful) per question; returns a list
     of VerificationTestSpec objects describing applicable semantic checks.
     Called once in QueryFixer.fix_candidates() before the per-candidate loop.

  2. evaluate_candidate() — evaluates all specs against a specific SQL
     candidate's execution result; mostly free (SQLite execution + structural
     analysis); expensive LLM-judgment tests run on every iteration.

Test types:

  Cheap (no LLM cost):
    grain            — result row count matches expected entity count
    null             — result rows contain no unexpected NULL values
    duplicate        — JOIN did not multiply rows beyond expected distinct count
    ordering         — "top N / highest / ranked" questions have ORDER BY + LIMIT
    scale            — numeric result values are within expected range
    column_alignment — SELECT column count matches expected_column_count

  Expensive (1 LLM call each, run every iteration):
    boundary         — date/time constraints match the question's period
    symmetry         — aggregate total matches sum of sub-groups

Module layout
-------------
This module is intentionally kept as the public facade so that all existing
import paths (from src.verification.query_verifier import ...) continue to work
without modification.  Implementation details live in the submodules:

  _constants.py           — penalty/bonus constants, test-type frozensets
  _ordering_helpers.py    — _extract_limit_from_question, _derive_direction_from_question
  _models.py              — VerificationTestSpec, VerificationTestResult, VerificationEvaluation
  _llm_schemas.py         — _PLAN_TOOL, _COLUMN_ALIGNMENT_TOOL, _PLAN_SYSTEM, _COLUMN_ALIGNMENT_SYSTEM
  _cheap_evaluators.py    — CheapEvaluatorMixin (_eval_grain/null/duplicate/ordering/scale)
  _expensive_evaluators.py— ExpensiveEvaluatorMixin (_eval_column_alignment, _eval_symmetry)

IMPORTANT: get_client is imported here (not only in submodules) so that test
patches targeting ``src.verification.query_verifier.get_client`` continue to
intercept LLM calls made by _generate_main_plan, _generate_column_alignment_spec,
and _eval_boundary.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional

from src.config.settings import settings
from src.data.database import ExecutionResult, execute_sql
from src.llm import CacheableText, LLMError, get_client  # patch target — keep here
from src.llm.base import ToolParam, sanitize_prompt_text

# ---------------------------------------------------------------------------
# Re-export submodule symbols so all existing imports keep working
# ---------------------------------------------------------------------------

from src.verification._constants import (  # noqa: F401
    _ALL_TEST_TYPES,
    _BONUS_ALL_PASS,
    _CHEAP_TESTS,
    _EXPENSIVE_TESTS,
    _MAX_BONUS,
    _PENALTY_CRITICAL,
    _PENALTY_MINOR,
)
from src.verification._ordering_helpers import (  # noqa: F401
    _ASC_KEYWORDS,
    _DESC_KEYWORDS,
    _TOP_N_RE,
    _WORD_TO_INT,
    _derive_direction_from_question,
    _extract_limit_from_question,
)
from src.verification._models import (  # noqa: F401
    VerificationEvaluation,
    VerificationTestResult,
    VerificationTestSpec,
)
from src.verification._llm_schemas import (  # noqa: F401
    _COLUMN_ALIGNMENT_SYSTEM,
    _COLUMN_ALIGNMENT_TOOL,
    _PLAN_SYSTEM,
    _PLAN_TOOL,
)
from src.verification._cheap_evaluators import CheapEvaluatorMixin
from src.verification._expensive_evaluators import ExpensiveEvaluatorMixin

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# QueryVerifier
# ---------------------------------------------------------------------------

class QueryVerifier(CheapEvaluatorMixin, ExpensiveEvaluatorMixin):
    """
    Generates and evaluates semantic verification plans for SQL candidates.

    Cheap evaluation methods (_eval_grain, _eval_null, _eval_duplicate,
    _eval_ordering, _eval_scale) come from CheapEvaluatorMixin.

    Structural-but-async methods (_eval_column_alignment, _eval_symmetry)
    come from ExpensiveEvaluatorMixin.

    LLM-calling methods (_generate_main_plan, _generate_column_alignment_spec,
    _eval_boundary) are defined directly on this class so that test patches
    on ``src.verification.query_verifier.get_client`` intercept them correctly.

    Workflow::

        verifier = QueryVerifier()

        # Once per question (runs two LLM calls concurrently):
        specs = await verifier.generate_plan(question, evidence, schema)

        # Per candidate, per fix iteration:
        eval_result = await verifier.evaluate_candidate(
            specs, candidate_id, sql, exec_result, db_path,
            run_expensive=(iteration == beta)  # True only on final check
        )
    """

    @staticmethod
    def _default_plan_no_col_alignment() -> list[VerificationTestSpec]:
        """Fallback for the main plan when LLM returns empty: grain only."""
        return [
            VerificationTestSpec(
                test_type="grain",
                fix_hint="Ensure the query returns rows. Check WHERE clause, JOINs, and GROUP BY.",
            ),
        ]

    async def _generate_main_plan(
        self,
        question: str,
        evidence: str,
        schema: str,
    ) -> list[VerificationTestSpec]:
        """
        Generate all verification tests EXCEPT column_alignment (model_powerful).

        Returns a list containing grain and any applicable optional tests.
        Falls back to a default grain-only plan on LLM failure.
        """
        q = sanitize_prompt_text(question)
        ev = sanitize_prompt_text(evidence or "None")
        schema_abbrev = schema[:3000]

        user_msg = (
            f"Question: {q}\n"
            f"Evidence: {ev}\n\n"
            f"Database Schema:\n{schema_abbrev}\n\n"
            "Generate the verification tests for this question."
        )

        try:
            client = get_client()
            response = await client.generate(
                model=settings.model_powerful,
                system=[_PLAN_SYSTEM],
                messages=[{"role": "user", "content": user_msg}],
                tools=[_PLAN_TOOL],
                tool_choice_name="verification_plan",
                temperature=0.0,
                max_tokens=1500,
            )
            if not response.tool_inputs:
                logger.warning("Main plan LLM returned no tool inputs — using default grain plan")
                return self._default_plan_no_col_alignment()
            raw_tests = response.tool_inputs[0].get("tests", [])
            specs: list[VerificationTestSpec] = []
            for t in raw_tests:
                try:
                    spec = VerificationTestSpec(**t)
                    if spec.test_type == "column_alignment":
                        # Skip — column_alignment is handled by the dedicated call
                        continue
                    if spec.test_type in _ALL_TEST_TYPES:
                        specs.append(spec)
                    else:
                        logger.warning("Unknown test_type %r in plan — skipped", spec.test_type)
                except Exception as e:
                    logger.warning("Failed to parse VerificationTestSpec: %s", e)
            if not specs:
                logger.info(
                    "Main plan empty for question %.60r — using default grain plan",
                    question,
                )
                return self._default_plan_no_col_alignment()
            return specs
        except LLMError as exc:
            logger.warning("Main plan generation failed (LLMError): %s — using default grain plan", exc)
            return self._default_plan_no_col_alignment()
        except Exception as exc:
            logger.warning("Unexpected error in _generate_main_plan: %s — using default grain plan", exc)
            return self._default_plan_no_col_alignment()

    async def _generate_column_alignment_spec(
        self,
        question: str,
        evidence: str,
        schema: str,
    ) -> VerificationTestSpec:
        """
        Dedicated LLM call (model_fast) that reasons about the expected number of
        SELECT columns for this question.

        Returns a VerificationTestSpec with test_type='column_alignment'.
        Never raises — falls back to expected_column_count=1 on any error.
        """
        q = sanitize_prompt_text(question)
        ev = sanitize_prompt_text(evidence or "None")
        schema_abbrev = sanitize_prompt_text(schema[:2000])

        user_msg = (
            f"Question: {q}\n"
            f"Evidence: {ev}\n\n"
            f"Database Schema (abbreviated):\n{schema_abbrev}\n\n"
            "How many SELECT columns must a correct SQL answer to this question produce?"
        )

        count: Optional[int] = None
        reasoning = "Default fallback"
        column_descriptions: list[str] = []
        try:
            client = get_client()
            response = await client.generate(
                model=settings.model_fast,
                system=[_COLUMN_ALIGNMENT_SYSTEM],
                messages=[{"role": "user", "content": user_msg}],
                tools=[_COLUMN_ALIGNMENT_TOOL],
                tool_choice_name="column_alignment_spec",
                temperature=0.0,
                max_tokens=2000,
            )
            if response.tool_inputs:
                data = response.tool_inputs[0]
                count = max(1, int(data.get("expected_column_count", 1)))
                reasoning = data.get("reasoning", "")
                raw_descs = data.get("column_descriptions", [])
                column_descriptions = (
                    raw_descs
                    if isinstance(raw_descs, list) and len(raw_descs) == count
                    else []
                )
            else:
                logger.warning("Column alignment LLM returned no tool inputs — skipping count check")
        except Exception as exc:
            logger.warning(
                "column_alignment dedicated call failed: %s — skipping count check", exc
            )

        col_hint = (
            f" Expected columns: {', '.join(column_descriptions)}."
            if column_descriptions else ""
        )
        fix_hint = (
            f"Adjust SELECT to return exactly {count} column(s).{col_hint} "
            "Remove extra columns or add missing ones based on the question wording."
            if count is not None else ""
        )
        return VerificationTestSpec(
            test_type="column_alignment",
            expected_column_count=count,
            column_descriptions=column_descriptions,
            fix_hint=fix_hint,
        )

    async def generate_plan(
        self,
        question: str,
        evidence: str,
        schema: str,
    ) -> list[VerificationTestSpec]:
        """
        Generate all applicable verification tests for this question.

        Runs two LLM calls concurrently via asyncio.gather():
          1. _generate_main_plan()  — model_powerful: grain + optional tests
          2. _generate_column_alignment_spec() — model_fast: dedicated column count

        Returns specs ordered: grain first, column_alignment second, then others.
        """
        main_task = self._generate_main_plan(question, evidence, schema)
        col_task = self._generate_column_alignment_spec(question, evidence, schema)

        results = await asyncio.gather(main_task, col_task, return_exceptions=True)
        main_specs, col_spec = results[0], results[1]

        if isinstance(main_specs, Exception):
            logger.warning(
                "Main plan raised unexpectedly: %s — using default grain plan", main_specs
            )
            main_specs = self._default_plan_no_col_alignment()

        if isinstance(col_spec, Exception):
            logger.warning(
                "Column alignment spec raised unexpectedly: %s — skipping count check", col_spec
            )
            col_spec = VerificationTestSpec(
                test_type="column_alignment",
                expected_column_count=None,
            )

        # Order: grain first, column_alignment second, then remaining tests
        grain_specs = [s for s in main_specs if s.test_type == "grain"]
        other_specs = [s for s in main_specs if s.test_type not in ("grain", "column_alignment")]
        return grain_specs + [col_spec] + other_specs

    async def evaluate_candidate(
        self,
        specs: list[VerificationTestSpec],
        candidate_id: str,
        sql: str,
        exec_result: ExecutionResult,
        db_path: str,
        run_expensive: bool = False,
    ) -> VerificationEvaluation:
        """
        Evaluate all applicable specs against a SQL candidate.

        Parameters
        ----------
        specs:
            Tests generated by generate_plan().
        candidate_id:
            Generator ID for logging.
        sql:
            Current SQL (after any preceding fix iterations).
        exec_result:
            Latest execution result for this SQL.
        db_path:
            Path to the SQLite database for running verification queries.
        run_expensive:
            If True, also run LLM-judgment tests (column_alignment, boundary,
            symmetry). Always True — parameter kept for interface compatibility.
        """
        results: list[VerificationTestResult] = []

        for spec in specs:
            is_expensive = spec.test_type in _EXPENSIVE_TESTS
            if is_expensive and not run_expensive:
                continue  # Skip expensive tests on early iterations

            if not exec_result.success:
                # Cannot evaluate semantic tests without a valid execution result
                results.append(VerificationTestResult(
                    test_type=spec.test_type,
                    status="skip",
                    actual_outcome="Skipped — SQL did not execute successfully.",
                    is_critical=False,
                ))
                continue

            result = await self._evaluate_single(spec, sql, exec_result, db_path)
            results.append(result)

        all_pass = all(r.status in ("pass", "skip", "error") for r in results)
        adjustment = self._compute_adjustment(results)
        failure_hints = [
            f"{r.test_type.upper()} TEST FAILED: {r.actual_outcome}  "
            f"Hint: {self._get_hint(r.test_type, specs)}"
            for r in results if r.status == "fail"
        ]

        return VerificationEvaluation(
            candidate_id=candidate_id,
            test_results=results,
            all_pass=all_pass,
            confidence_adjustment=adjustment,
            failure_hints=failure_hints,
        )

    def _get_hint(self, test_type: str, specs: list[VerificationTestSpec]) -> str:
        for spec in specs:
            if spec.test_type == test_type:
                return spec.fix_hint
        return "Review the SQL logic."

    async def _evaluate_single(
        self,
        spec: VerificationTestSpec,
        sql: str,
        exec_result: ExecutionResult,
        db_path: str,
    ) -> VerificationTestResult:
        """Dispatch to the appropriate evaluation method."""
        tt = spec.test_type
        try:
            if tt == "grain":
                return self._eval_grain(spec, exec_result, db_path)
            elif tt == "null":
                return self._eval_null(spec, exec_result)
            elif tt == "duplicate":
                return self._eval_duplicate(spec, exec_result, db_path)
            elif tt == "ordering":
                return self._eval_ordering(spec, sql)
            elif tt == "scale":
                return self._eval_scale(spec, exec_result)
            elif tt == "column_alignment":
                return await self._eval_column_alignment(spec, sql, exec_result)
            elif tt == "boundary":
                return await self._eval_boundary(spec, sql)
            elif tt == "symmetry":
                return self._eval_symmetry(spec, exec_result, db_path)
            else:
                return VerificationTestResult(
                    test_type=tt,
                    status="skip",
                    actual_outcome=f"Unknown test type: {tt}",
                    is_critical=False,
                )
        except Exception as exc:
            logger.warning("Error evaluating %r test for candidate: %s", tt, exc)
            return VerificationTestResult(
                test_type=tt,
                status="error",
                actual_outcome=f"Evaluation error: {exc}",
                is_critical=False,
            )

    async def _eval_boundary(
        self,
        spec: VerificationTestSpec,
        sql: str,
    ) -> VerificationTestResult:
        """Boundary test: LLM checks if date/time constraints match the question.

        This is an expensive test — makes 1 LLM call (model_fast).
        Runs on every fix iteration (run_expensive is always True in QueryFixer).

        NOTE: This method must remain defined here (not in ExpensiveEvaluatorMixin)
        because it calls get_client(), which is patched via
        ``src.verification.query_verifier.get_client`` in tests.
        """
        logger.debug("Running expensive test: boundary")
        where_match = re.search(
            r"\bWHERE\b(.*?)(?:\bGROUP\s+BY\b|\bORDER\s+BY\b|\bHAVING\b|\bLIMIT\b|$)",
            sql,
            re.IGNORECASE | re.DOTALL,
        )
        where_clause = where_match.group(1).strip() if where_match else "(no WHERE clause)"

        # Structural pre-check: if the SQL contains any date/time construct,
        # treat the boundary constraint as structurally present.  Only call
        # the expensive LLM judge when no such construct can be found.
        _date_pattern = re.compile(
            r"\b(BETWEEN|STRFTIME|JULIANDAY|DATE|DATETIME|YEAR|MONTH)\b"
            r"|[><=!]=?\s*['\"]?\d{4}[-/]"  # comparisons like >= '2014-'
            r"|\bLIKE\s+['\"]%?\d{4}%?['\"]"  # LIKE '2014%'
            r"|\b\d{4}\s+AND\s+\d{4}\b",  # year BETWEEN 2014 AND 2015
            re.IGNORECASE,
        )
        if _date_pattern.search(sql):
            logger.debug(
                "boundary: PASS (structural pre-check — date construct found in SQL)"
            )
            return VerificationTestResult(
                test_type="boundary",
                status="pass",
                actual_outcome="Structural pre-check: SQL contains a date/time construct.",
                is_critical=False,
            )

        prompt = (
            f"You are an SQL analyst checking whether a SQL query correctly implements"
            f" a date/time boundary condition.\n\n"
            f"Boundary requirement: {spec.description}\n"
            f"Expected outcome: {spec.expected_outcome}\n\n"
            f"SQL WHERE clause:\n{where_clause}\n\n"
            "IMPORTANT: Judge SEMANTIC EQUIVALENCE, not syntactic form.\n"
            "Different SQL expressions can represent the same time period correctly.\n\n"
            "Examples of EQUIVALENT patterns (all should be judged PASS if they target"
            " the correct period):\n"
            "  - '2014-2015' ≡ BETWEEN 2014 AND 2015 ≡ year >= 2014 AND year <= 2015\n"
            "  - year = 2014 ≡ STRFTIME('%Y', col) = '2014' ≡ col LIKE '2014%'"
            " ≡ BETWEEN '2014-01-01' AND '2014-12-31'\n"
            "  - BETWEEN '2014-01-01' AND '2014-06-30' ≡ col >= '2014-01-01'"
            " AND col < '2014-07-01'\n\n"
            "PASS if:\n"
            "  - The WHERE clause correctly captures the intended time period,"
            " even in a different syntactic form.\n"
            "  - The year/date range matches what the question requires"
            " (off-by-one rounding is acceptable).\n"
            "  - The query uses string patterns (LIKE, prefix match) that are logically"
            " equivalent to a numeric comparison on the same period.\n\n"
            "FAIL only if:\n"
            "  - The WHERE clause filters the WRONG time period"
            " (e.g., 2013 when 2014 is required).\n"
            "  - There is NO date/time constraint at all when one is clearly required.\n"
            "  - The date range has a significant off-by-one error that changes which"
            " rows are returned (e.g., 2014-01-01 to 2014-12-30 instead of 2014-12-31"
            " is NOT a fail; 2013-01-01 to 2014-11-30 IS a fail).\n\n"
            "Reply with exactly 'PASS' or 'FAIL' on the first line,"
            " then a one-sentence explanation."
        )

        try:
            client = get_client()
            response = await client.generate(
                model=settings.model_fast,
                system=[CacheableText(text="You are an SQL quality analyst.", cache=False)],
                messages=[{"role": "user", "content": prompt}],
                tools=[],
                temperature=0.0,
                max_tokens=200,
            )
            judgment = (response.text or "").strip().upper()
            if judgment.startswith("PASS"):
                status = "pass"
                logger.debug("boundary: PASS — %s", (response.text or "")[:100])
            elif judgment.startswith("FAIL"):
                status = "fail"
                logger.info("boundary: FAIL — %s", (response.text or "")[:200])
            else:
                status = "skip"
                logger.warning(
                    "boundary: inconclusive LLM response — %s",
                    (response.text or "")[:100],
                )
            return VerificationTestResult(
                test_type="boundary",
                status=status,
                actual_outcome=f"LLM judgment: {(response.text or '')[:200]}",
                is_critical=False,
            )
        except LLMError as exc:
            logger.warning("boundary LLM call failed: %s", exc)
            return VerificationTestResult(
                test_type="boundary",
                status="error",
                actual_outcome=f"LLM error: {exc}",
                is_critical=False,
            )

    # ------------------------------------------------------------------
    # Confidence adjustment
    # ------------------------------------------------------------------

    def _compute_adjustment(
        self,
        results: list[VerificationTestResult],
    ) -> float:
        """Compute confidence score adjustment from test results."""
        if not results:
            return 0.0

        _CRITICAL = {"grain", "duplicate", "column_alignment"}
        adjustment = 0.0
        for r in results:
            if r.status == "fail":
                adjustment += _PENALTY_CRITICAL if r.test_type in _CRITICAL else _PENALTY_MINOR

        # Bonus if all non-error tests passed
        non_error = [r for r in results if r.status != "error"]
        if non_error and all(r.status in ("pass", "skip") for r in non_error):
            passed = [r for r in non_error if r.status == "pass"]
            if passed:
                adjustment = min(adjustment + _BONUS_ALL_PASS, _MAX_BONUS)

        return adjustment
