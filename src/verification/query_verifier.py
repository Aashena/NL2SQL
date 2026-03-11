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
from src.llm.base import LLMMalformedToolError
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
    _GRAIN_SYSTEM,
    _GRAIN_TOOL,
    _OPTIONAL_TESTS_SYSTEM,
    _OPTIONAL_TESTS_TOOL,
    _PLAN_SYSTEM,   # backward-compat alias
    _PLAN_TOOL,     # backward-compat alias
)
from src.verification._cheap_evaluators import CheapEvaluatorMixin
from src.verification._expensive_evaluators import ExpensiveEvaluatorMixin

logger = logging.getLogger(__name__)


def _default_grain_spec() -> VerificationTestSpec:
    """Return a bare-bones grain spec used as a fallback when LLM calls fail."""
    return VerificationTestSpec(
        test_type="grain",
        fix_hint="Ensure the query returns rows. Check WHERE clause, JOINs, and GROUP BY.",
    )


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

    LLM-calling methods (_generate_grain_spec, _generate_optional_tests,
    _generate_column_alignment_spec, _eval_boundary) are defined directly on
    this class so that test patches on
    ``src.verification.query_verifier.get_client`` intercept them correctly.

    Workflow::

        verifier = QueryVerifier()

        # Once per question (runs three LLM calls concurrently):
        specs = await verifier.generate_plan(question, evidence, schema)

        # Per candidate, per fix iteration:
        eval_result = await verifier.evaluate_candidate(
            specs, candidate_id, sql, exec_result, db_path,
            run_expensive=(iteration == beta)  # True only on final check
        )
    """

    @staticmethod
    def _default_plan_no_col_alignment() -> list[VerificationTestSpec]:
        """Backward-compat alias — returns a default grain-only spec list."""
        return [_default_grain_spec()]

    async def _generate_grain_spec(
        self,
        question: str,
        evidence: str,
        schema: str,
    ) -> VerificationTestSpec:
        """
        Dedicated LLM call (model_fast) that generates the grain test only.

        The grain tool schema has verification_sql_upper as a required field,
        so the LLM is forced to always provide a bounding COUNT query.
        Falls back to a default grain spec on any LLM failure.
        """
        q = sanitize_prompt_text(question)
        ev = sanitize_prompt_text(evidence or "None")
        # TODO: schema is truncated at 3000 chars which can cut mid-table-definition,
        # causing the LLM to hallucinate table/column names not present in the schema.
        # Future fix: pass only the most relevant tables (e.g. top-K by BM25 relevance).
        schema_abbrev = sanitize_prompt_text(schema[:3000])

        user_msg = (
            f"Question: {q}\n"
            f"Evidence: {ev}\n\n"
            f"Database Schema:\n{schema_abbrev}\n\n"
            "Generate the grain verification test for this question. "
            "Use ONLY table and column names from the Database Schema above "
            "in your verification_sql_upper. Call grain_verification now."
        )

        try:
            client = get_client()
            _grain_call_kwargs = dict(
                model=settings.model_fast,
                system=[_GRAIN_SYSTEM],
                messages=[{"role": "user", "content": user_msg}],
                tools=[_GRAIN_TOOL],
                tool_choice_name="grain_verification",
                temperature=0.0,
                max_tokens=800,
            )
            response = await client.generate(**_grain_call_kwargs)
            if not response.tool_inputs:
                logger.warning(
                    "Grain LLM returned no tool inputs (finish_reason=%s) — using default grain spec",
                    response.finish_reason,
                )
                return _default_grain_spec()
            data = response.tool_inputs[0]
            raw_sql_upper = data.get("verification_sql_upper") or None
            if "verification_sql_upper" in data and not raw_sql_upper:
                logger.warning(
                    "Grain LLM returned empty verification_sql_upper (required field) — "
                    "grain upper bound will be skipped"
                )
            _valid_grain_conf = {"high", "medium", "low", "none"}
            raw_conf = data.get("upper_bound_confidence") or "none"
            if raw_conf not in _valid_grain_conf:
                logger.warning(
                    "Grain LLM returned invalid upper_bound_confidence %r — defaulting to 'none'",
                    raw_conf,
                )
                raw_conf = "none"
            return VerificationTestSpec(
                test_type="grain",
                verification_sql_upper=raw_sql_upper,
                upper_bound_confidence=raw_conf,
                row_count_min=data.get("row_count_min"),
            )
        except LLMError as exc:
            logger.warning("Grain spec LLM failed (LLMError): %s — using default grain spec", exc)
            return _default_grain_spec()
        except Exception as exc:
            logger.warning("Unexpected error in _generate_grain_spec: %s — using default grain spec", exc)
            return _default_grain_spec()

    async def _generate_optional_tests(
        self,
        question: str,
        evidence: str,
        schema: str,
    ) -> list[VerificationTestSpec]:
        """
        Dedicated LLM call (model_powerful) for optional tests: null, duplicate,
        ordering, scale, boundary, symmetry. Grain is excluded from this call.

        Returns an empty list on any LLM failure — the grain spec is unaffected.
        """
        q = sanitize_prompt_text(question)
        ev = sanitize_prompt_text(evidence or "None")
        schema_abbrev = sanitize_prompt_text(schema[:3000])

        user_msg = (
            f"Question: {q}\n"
            f"Evidence: {ev}\n\n"
            f"Database Schema:\n{schema_abbrev}\n\n"
            "Generate the optional verification tests for this question."
        )

        try:
            client = get_client()
            response = await client.generate(
                model=settings.model_powerful,
                system=[_OPTIONAL_TESTS_SYSTEM],
                messages=[{"role": "user", "content": user_msg}],
                tools=[_OPTIONAL_TESTS_TOOL],
                tool_choice_name="verification_plan",
                temperature=0.0,
                max_tokens=1800,
            )
            if not response.tool_inputs:
                return []
            # New flat schema: each test type is a top-level key in the response dict.
            # Presence of a key = test applies; absence = test not applicable.
            raw = response.tool_inputs[0]
            specs: list[VerificationTestSpec] = []
            for test_type in ("null", "duplicate", "ordering", "scale", "boundary", "symmetry"):
                sub = raw.get(test_type)
                if not isinstance(sub, dict):
                    continue
                # Post-parse validation: discard incomplete specs rather than silently
                # accepting them with empty/missing required fields.
                if test_type == "null" and not sub.get("check_columns"):
                    logger.warning("Discarding null test spec — missing required check_columns")
                    continue
                if test_type == "duplicate" and not sub.get("verification_sql"):
                    logger.warning("Discarding duplicate test spec — missing required verification_sql")
                    continue
                if test_type == "ordering" and not sub.get("order_by_column"):
                    logger.warning("Discarding ordering test spec — missing required order_by_column")
                    continue
                if test_type == "scale" and (sub.get("numeric_min") is None or sub.get("numeric_max") is None):
                    logger.warning("Discarding scale test spec — missing required numeric_min or numeric_max")
                    continue
                if test_type == "boundary" and not (sub.get("boundary_description") and sub.get("expected_outcome")):
                    logger.warning("Discarding boundary test spec — missing required boundary_description or expected_outcome")
                    continue
                if test_type == "symmetry" and not sub.get("verification_sql"):
                    logger.warning("Discarding symmetry test spec — missing required verification_sql")
                    continue
                try:
                    t = {"test_type": test_type, **sub}
                    # The schema uses "boundary_description" (Gemini rejects "description"
                    # as a property name — it is a reserved keyword in its schema format).
                    # VerificationTestSpec also uses boundary_description, so no aliasing needed.
                    # As a safety net: if the model returns the old "description" key, map it.
                    if "description" in t and "boundary_description" not in t:
                        t["boundary_description"] = t.pop("description")
                    elif "description" in t:
                        t.pop("description")  # boundary_description already present; drop duplicate
                    specs.append(VerificationTestSpec(**t))
                except Exception as e:
                    logger.warning("Failed to parse %r test spec: %s", test_type, e)
            return specs
        except LLMMalformedToolError as exc:
            logger.warning("Optional tests MALFORMED_FUNCTION_CALL: %s — returning empty", exc)
            return []
        except LLMError as exc:
            logger.warning("Optional tests LLM failed (LLMError): %s — returning empty", exc)
            return []
        except Exception as exc:
            logger.warning("Unexpected error in _generate_optional_tests: %s — returning empty", exc)
            return []

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
        confidence: Optional[str] = "medium"
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
                raw_col_conf = data.get("confidence", "medium")
                _valid_col_conf = {"high", "medium", "low"}
                if raw_col_conf not in _valid_col_conf:
                    logger.warning(
                        "Column alignment LLM returned invalid confidence %r — defaulting to 'medium'",
                        raw_col_conf,
                    )
                    raw_col_conf = "medium"
                confidence = raw_col_conf
                raw_descs = data.get("column_descriptions", [])
                if isinstance(raw_descs, list) and len(raw_descs) == count:
                    column_descriptions = raw_descs
                elif isinstance(raw_descs, list) and raw_descs:
                    logger.warning(
                        "column_descriptions length %d != expected_column_count %d — truncating",
                        len(raw_descs),
                        count,
                    )
                    column_descriptions = raw_descs[:count]
                else:
                    column_descriptions = []
            else:
                logger.warning(
                    "Column alignment LLM returned no tool inputs (finish_reason=%s) — skipping count check",
                    response.finish_reason,
                )
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
            column_alignment_confidence=confidence,
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

        Runs three LLM calls concurrently via asyncio.gather():
          1. _generate_grain_spec()       — model_fast: grain test only
          2. _generate_optional_tests()   — model_powerful: null/duplicate/ordering/scale/boundary/symmetry
          3. _generate_column_alignment_spec() — model_fast: dedicated column count

        Returns specs ordered: grain first, column_alignment second, then optional tests.
        """
        grain_task = self._generate_grain_spec(question, evidence, schema)
        optional_task = self._generate_optional_tests(question, evidence, schema)
        col_task = self._generate_column_alignment_spec(question, evidence, schema)

        results = await asyncio.gather(grain_task, optional_task, col_task, return_exceptions=True)
        grain_spec, optional_specs, col_spec = results[0], results[1], results[2]

        if isinstance(grain_spec, Exception):
            logger.warning("Grain spec raised unexpectedly: %s — using default", grain_spec)
            grain_spec = _default_grain_spec()

        if isinstance(optional_specs, Exception):
            logger.warning("Optional tests raised unexpectedly: %s — returning empty", optional_specs)
            optional_specs = []

        if isinstance(col_spec, Exception):
            logger.warning(
                "Column alignment spec raised unexpectedly: %s — skipping count check", col_spec
            )
            col_spec = VerificationTestSpec(
                test_type="column_alignment",
                expected_column_count=None,
                column_alignment_confidence="medium",
            )

        return [grain_spec, col_spec, *optional_specs]

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
        # NOTE: YEAR and MONTH are excluded — they are common column names, not
        # SQLite functions, and matching them would cause false-positives for
        # simple year-equality filters like `WHERE year = 2022`.
        _date_pattern = re.compile(
            r"\b(BETWEEN|STRFTIME|JULIANDAY|DATE|DATETIME)\b"
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

        boundary_desc = sanitize_prompt_text((spec.boundary_description or "").replace("\n", " ")[:300])
        expected_outcome = sanitize_prompt_text((spec.expected_outcome or "").replace("\n", " ")[:300])
        where_clause = sanitize_prompt_text(where_clause)

        # Use function calling for the boundary judgment.
        # Free-text parsing (.startswith("PASS")) is unreliable with Gemini because
        # the model typically begins its response with reasoning, not the verdict.
        _boundary_judgment_tool = ToolParam(
            name="boundary_judgment",
            description=(
                "Judge whether the SQL WHERE clause correctly implements the required "
                "date/time boundary condition. "
                "Use PASS if the clause captures the exact required time period "
                "(possibly in a different syntactic form). "
                "Use FAIL if the period is wrong, wider, narrower, or absent."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "verdict": {
                        "type": "string",
                        "enum": ["PASS", "FAIL"],
                        "description": (
                            "PASS — WHERE clause captures the EXACT required time period "
                            "(syntactic variations are fine: BETWEEN, LIKE prefix, "
                            "comparison operators, STRFTIME all count as equivalent "
                            "if the time period is the same). "
                            "FAIL — time period is wrong, wider, narrower, or absent."
                        ),
                    },
                    "reason": {
                        "type": "string",
                        "description": "One sentence explaining the verdict.",
                    },
                },
                "required": ["verdict", "reason"],
            },
        )

        prompt = (
            f"Check whether the SQL WHERE clause correctly implements"
            f" the required date/time boundary.\n\n"
            f"Boundary requirement: {boundary_desc}\n"
            f"Expected outcome: {expected_outcome}\n\n"
            f"SQL WHERE clause:\n{where_clause}\n\n"
            "Judge SEMANTIC EQUIVALENCE, not syntactic form.\n"
            "These are all EQUIVALENT if they cover the EXACT same period:\n"
            "  - year BETWEEN 2014 AND 2015 ≡ year >= 2014 AND year <= 2015\n"
            "  - year = 2014 ≡ STRFTIME('%Y', col) = '2014' ≡ col LIKE '2014%'"
            " ≡ BETWEEN '2014-01-01' AND '2014-12-31'\n"
            "  - BETWEEN '2014-01-01' AND '2014-06-30' ≡ col >= '2014-01-01'"
            " AND col < '2014-07-01'\n\n"
            "PASS if the WHERE clause covers the EXACT required period.\n"
            "FAIL if: wrong year, wrong quarter, wrong month, wider/narrower range,"
            " or no date constraint at all when one is required.\n\n"
            "Call boundary_judgment with your verdict and a one-sentence reason."
        )

        try:
            client = get_client()
            response = await client.generate(
                model=settings.model_fast,
                system=[CacheableText(text="You are an SQL quality analyst.", cache=False)],
                messages=[{"role": "user", "content": prompt}],
                tools=[_boundary_judgment_tool],
                tool_choice_name="boundary_judgment",
                temperature=0.0,
                max_tokens=400,
            )
            if response.tool_inputs:
                data = response.tool_inputs[0]
                verdict = (data.get("verdict") or "").strip().upper()
                reason = (data.get("reason") or "")[:200]
                if verdict == "PASS":
                    status = "pass"
                    logger.debug("boundary: PASS — %s", reason)
                elif verdict == "FAIL":
                    status = "fail"
                    logger.info("boundary: FAIL — %s", reason)
                else:
                    status = "skip"
                    logger.warning("boundary: unexpected verdict %r — skipping", verdict)
                return VerificationTestResult(
                    test_type="boundary",
                    status=status,
                    actual_outcome=f"LLM judgment: {reason}",
                    is_critical=False,
                )
            else:
                logger.warning(
                    "boundary: LLM returned no tool inputs (finish_reason=%s) — skipping",
                    response.finish_reason,
                )
                return VerificationTestResult(
                    test_type="boundary",
                    status="skip",
                    actual_outcome="LLM returned no tool inputs for boundary judgment.",
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
