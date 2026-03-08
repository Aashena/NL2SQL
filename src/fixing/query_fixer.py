"""
Op 8: Query Fixer + Semantic Verifier (integrated)

For each SQL candidate, runs a two-stage fix loop:

  Stage A — Executability: execute the SQL; categorize errors (syntax,
    schema, type, empty result).

  Stage B — Verification: if execution succeeded, evaluate the candidate
    against the question-level VerificationPlan (all tests — cheap and
    expensive — run on every iteration).

The fix LLM (model_fast) receives combined feedback from both stages,
enabling it to correct both execution errors AND semantic quality issues
in a single prompt.

Steps:
  8.1 — Generate verification plan (1 LLM call per question, shared).
  8.2 — Execute all candidates (initial pass).
  8.3 — Fix loop: up to β=3 iterations per candidate.
        Each iteration: Stage A (exec check) → Stage B (full verif check,
        including expensive LLM-judgment tests) → if both pass, done;
        else fix with combined feedback.
  8.4 — Produce FixedCandidate with verification results attached.
  8.5 — Confidence scoring + normalization across the pool.

Fix calls for independent candidates run concurrently via asyncio.gather().
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from src.config.settings import settings
from src.data.database import ExecutionResult, execute_sql
from src.generation.base_generator import SQLCandidate, clean_sql
from src.llm import CacheableText, LLMError, get_client
from src.monitoring.fallback_tracker import FallbackEvent, get_tracker
from src.verification.query_verifier import (
    QueryVerifier,
    VerificationEvaluation,
    VerificationTestSpec,
)

if TYPE_CHECKING:
    from src.schema_linking.schema_linker import LinkedSchemas

logger = logging.getLogger(__name__)

# Maximum number of fix iterations (fix attempts) per candidate.
# The loop runs _BETA + 1 times: _BETA fix attempts + 1 final assessment.
_BETA = 3

# ---------------------------------------------------------------------------
# Error categories
# ---------------------------------------------------------------------------

_ErrorCategory = str

_ERR_SYNTAX = "syntax_error"
_ERR_SCHEMA = "schema_error"
_ERR_TYPE = "type_error"
_ERR_EMPTY = "empty_result"
_ERR_OTHER = "other_error"


def _categorize_error(
    error_message: Optional[str],
    is_empty: bool,
) -> _ErrorCategory:
    """Classify an execution error into a targeted fix category."""
    if is_empty:
        return _ERR_EMPTY
    if not error_message:
        return _ERR_OTHER
    msg_lower = error_message.lower()
    if "syntax" in msg_lower:
        return _ERR_SYNTAX
    if (
        "no such column" in msg_lower
        or "no such table" in msg_lower
        or "ambiguous column" in msg_lower
    ):
        return _ERR_SCHEMA
    if (
        "datatype mismatch" in msg_lower
        or "type mismatch" in msg_lower
        or "cannot compare" in msg_lower
    ):
        return _ERR_TYPE
    return _ERR_OTHER


def _error_instruction(
    category: _ErrorCategory,
    error_message: Optional[str],
) -> str:
    """Return a targeted fix instruction for the given error category."""
    if category == _ERR_SYNTAX:
        return f"Fix the SQL syntax error: {error_message}"
    if category == _ERR_SCHEMA:
        if error_message:
            m = re.search(r"no such column:\s*(\S+)", error_message, re.IGNORECASE)
            if m:
                return (
                    f"The column '{m.group(1)}' doesn't exist. "
                    "Check the schema and use the correct column name."
                )
            m2 = re.search(r"no such table:\s*(\S+)", error_message, re.IGNORECASE)
            if m2:
                return (
                    f"The table '{m2.group(1)}' doesn't exist. "
                    "Check the schema and use the correct table name."
                )
        return (
            f"Schema error: {error_message}. "
            "Check the schema and use the correct table/column names."
        )
    if category == _ERR_TYPE:
        return f"Fix the type mismatch error: {error_message}"
    if category == _ERR_EMPTY:
        return (
            "The query returned no rows. "
            "Review the WHERE conditions — they may be too restrictive. "
            "Or maybe you should try a different join approach."
        )
    return f"Fix the following error: {error_message}"


# ---------------------------------------------------------------------------
# FixedCandidate output dataclass
# ---------------------------------------------------------------------------

@dataclass
class FixedCandidate:
    """A SQL candidate after the two-stage fix loop has been applied."""
    original_sql: str
    final_sql: str
    generator_id: str
    fix_iterations: int                              # 0 if no fix was needed
    execution_result: ExecutionResult
    confidence_score: float                          # normalized [0, 1]
    verification_results: Optional[VerificationEvaluation] = field(
        default=None
    )
    """Semantic verification evaluation from the final fix iteration.
    None if verification was skipped (e.g. SQL never executed successfully)."""
    iteration_trace: list[dict] = field(default_factory=list)
    """Per-iteration debug data (exec + verif state, feedback sent to fixer).
    Populated only when QueryFixer is constructed with trace=True.
    Each entry corresponds to one pass through the fix loop (iteration 0…β)."""


# ---------------------------------------------------------------------------
# Confidence scoring helpers
# ---------------------------------------------------------------------------

_AGG_PATTERN = re.compile(
    r"\b(COUNT|SUM|AVG|MIN|MAX|GROUP\s+BY)\b",
    re.IGNORECASE,
)


def _is_aggregation_query(sql: str) -> bool:
    return bool(_AGG_PATTERN.search(sql))


def _raw_confidence(
    execution_result: ExecutionResult,
    fix_iterations: int,
    sql: str,
    verif_eval: Optional[VerificationEvaluation] = None,
) -> float:
    """Compute the un-normalized confidence score for a single candidate.

    Rules:
      +1.0  if execution succeeded and result is non-empty
      +0.5  plausibility bonus:
              aggregation query → exactly 1 row is plausible
              non-aggregation query → 1–100 rows is plausible
      -0.5  per fix iteration needed
      +/−   verification adjustment (from VerificationEvaluation.confidence_adjustment)
       0.0  if still failing after all fix attempts
    """
    if not execution_result.success or execution_result.is_empty:
        return 0.0

    score = 1.0

    # Plausibility bonus
    n_rows = len(execution_result.rows)
    if _is_aggregation_query(sql):
        if n_rows == 1:
            score += 0.5
    else:
        if 1 <= n_rows <= 100:
            score += 0.5

    # Fix iteration penalty
    score -= 0.5 * fix_iterations

    # Verification adjustment
    if verif_eval is not None:
        score += verif_eval.confidence_adjustment

    return score


# ---------------------------------------------------------------------------
# Iteration trace helper (used only when QueryFixer(trace=True))
# ---------------------------------------------------------------------------

def _build_iter_entry(
    *,
    iteration: int,
    is_final_assessment: bool,
    sql: str,
    prev_sql: Optional[str],
    exec_result: ExecutionResult,
    verif_eval: Optional[VerificationEvaluation],
    exec_issues_sent: Optional[list[str]],
    verif_issues_sent: Optional[list[str]],
    fix_triggered: bool,
    fix_produced_different_sql: Optional[bool],
) -> dict:
    """Build one entry for FixedCandidate.iteration_trace."""
    verif_results_list: Optional[list[dict]] = None
    if verif_eval is not None:
        verif_results_list = [
            {
                "test_type": r.test_type,
                "status": r.status,
                "actual_outcome": r.actual_outcome,
                "is_critical": r.is_critical,
            }
            for r in verif_eval.test_results
        ]

    return {
        "iteration": iteration,
        "is_final_assessment": is_final_assessment,
        "sql": sql[:500] if sql else "",
        "sql_full": sql or "",          # full SQL for post-hoc oracle evaluation
        "sql_changed_from_previous": (
            (sql != prev_sql) if prev_sql is not None else False
        ),
        # Stage A
        "exec_success": exec_result.success,
        "exec_is_empty": exec_result.is_empty,
        "exec_error": exec_result.error,
        "exec_row_count": (
            len(exec_result.rows) if exec_result.success else None
        ),
        # Stage B — expensive tests (column_alignment, boundary, symmetry) run
        # on every iteration (run_expensive=True is always passed to evaluate_candidate).
        "verif_ran": verif_eval is not None,
        "verif_run_expensive": True,
        "verif_results": verif_results_list,
        "verif_all_pass": (verif_eval.all_pass if verif_eval is not None else None),
        # Fix feedback
        "exec_issues_sent": exec_issues_sent,
        "verif_issues_sent": verif_issues_sent,
        "fix_triggered": fix_triggered,
        "fix_produced_different_sql": fix_produced_different_sql,
    }


# ---------------------------------------------------------------------------
# QueryFixer class
# ---------------------------------------------------------------------------

class QueryFixer:
    """
    Operation 8: execute all candidates, verify + fix errors, score confidence.

    Usage::

        fixer = QueryFixer()
        fixed = await fixer.fix_candidates(
            candidates=candidates,
            question=question,
            evidence=evidence,
            schemas=schemas,
            db_path=db_path,
            cell_matches=cell_matches,
        )

    The verifier can be injected for testing::

        fixer = QueryFixer(verifier=mock_verifier)

    For smoke-test analysis, enable per-iteration tracing::

        fixer = QueryFixer(trace=True)
        # After fix_candidates(), inspect fc.iteration_trace and fixer.last_verif_specs
    """

    def __init__(
        self,
        verifier: Optional[QueryVerifier] = None,
        trace: bool = False,
    ) -> None:
        self._verifier = verifier if verifier is not None else QueryVerifier()
        self._trace = trace
        self.last_verif_specs: list[VerificationTestSpec] = []
        """Verification plan specs from the most recent fix_candidates() call.
        Empty list if plan generation failed or trace=False was used.
        Populated regardless of the trace flag for smoke-test access."""

    async def fix_candidates(
        self,
        candidates: list[SQLCandidate],
        question: str,
        evidence: str,
        schemas: "LinkedSchemas",
        db_path: str,
        cell_matches: list,
    ) -> list[FixedCandidate]:
        """Execute + verify + fix all candidates concurrently; return scored results.

        Parameters
        ----------
        candidates:
            SQL candidates from the generation step (Op 7).
        question:
            Natural-language question.
        evidence:
            Auxiliary evidence string (may be empty or None).
        schemas:
            LinkedSchemas from Op 6; S₂ DDL is used as schema context.
        db_path:
            Filesystem path to the SQLite database.
        cell_matches:
            CellMatch objects from context grounding (for fix prompt context).

        Returns
        -------
        list[FixedCandidate] — one per input candidate, with confidence scores
        normalized across the pool.
        """
        # Step 8.1: Generate verification plan once for this question (1 LLM call).
        verif_specs: list[VerificationTestSpec] = []
        self.last_verif_specs = []
        try:
            verif_specs = await self._verifier.generate_plan(
                question=question,
                evidence=evidence or "",
                schema=schemas.s2_ddl,
            )
            self.last_verif_specs = verif_specs
            if verif_specs:
                logger.debug(
                    "Verification plan: %d tests for question %.60r",
                    len(verif_specs),
                    question,
                )
        except Exception as exc:
            logger.warning("Verification plan generation failed: %s", exc)
            verif_specs = []

        # Step 8.2: Execute all candidates (initial pass).
        initial_results: list[tuple[SQLCandidate, ExecutionResult]] = []
        for cand in candidates:
            result = execute_sql(db_path, cand.sql)
            initial_results.append((cand, result))

        # Step 8.3: Fix loop — run concurrently per candidate.
        fix_tasks = [
            self._fix_single(
                candidate=cand,
                initial_result=result,
                question=question,
                evidence=evidence,
                schemas=schemas,
                db_path=db_path,
                cell_matches=cell_matches,
                verif_specs=verif_specs,
            )
            for cand, result in initial_results
        ]
        fixed_candidates: list[FixedCandidate] = await asyncio.gather(*fix_tasks)

        # Step 8.6: Normalize confidence scores.
        fixed_candidates = _normalize_confidence(fixed_candidates)

        return list(fixed_candidates)

    async def _fix_single(
        self,
        candidate: SQLCandidate,
        initial_result: ExecutionResult,
        question: str,
        evidence: str,
        schemas: "LinkedSchemas",
        db_path: str,
        cell_matches: list,
        verif_specs: list[VerificationTestSpec],
    ) -> FixedCandidate:
        """Run the two-stage fix loop for a single candidate.

        Loop runs _BETA + 1 times (indices 0 … _BETA):
          - Every iteration: Stage A (exec check) + full Stage B (all tests,
            including expensive LLM-judgment tests).
          - Iterations 0 … _BETA-1: if either stage fails, fix with combined
            feedback and continue.
          - Iteration _BETA (3): final assessment — accept whatever result we
            have; no further fix is attempted even if tests fail.
        If both stages pass at any point, the loop breaks early.
        """
        current_sql = candidate.sql
        current_result = initial_result
        fix_iterations = 0
        last_verif_eval: Optional[VerificationEvaluation] = None
        iter_trace: list[dict] = []   # populated only when self._trace is True

        for iteration in range(_BETA + 1):
            is_final_assessment = (iteration == _BETA)
            sql_at_iter_start = current_sql

            # ------------------------------------------------------------------
            # Stage A: Executability check
            # ------------------------------------------------------------------
            exec_ok = current_result.success and not current_result.is_empty

            # ------------------------------------------------------------------
            # Stage B: Verification check (only when execution succeeded)
            # ------------------------------------------------------------------
            verif_eval: Optional[VerificationEvaluation] = None
            if exec_ok and verif_specs:
                try:
                    verif_eval = await self._verifier.evaluate_candidate(
                        specs=verif_specs,
                        candidate_id=candidate.generator_id,
                        sql=current_sql,
                        exec_result=current_result,
                        db_path=db_path,
                        run_expensive=True,
                    )
                except Exception as exc:
                    logger.warning(
                        "Verification evaluation failed for %r (iter %d): %s",
                        candidate.generator_id,
                        iteration,
                        exc,
                    )
                    verif_eval = None

            verif_ok = verif_eval is None or verif_eval.all_pass
            last_verif_eval = verif_eval

            # Both stages pass → done
            if exec_ok and verif_ok:
                if self._trace:
                    iter_trace.append(_build_iter_entry(
                        iteration=iteration,
                        is_final_assessment=is_final_assessment,
                        sql=sql_at_iter_start,
                        prev_sql=iter_trace[-1]["sql"] if iter_trace else None,
                        exec_result=current_result,
                        verif_eval=verif_eval,
                        exec_issues_sent=None,
                        verif_issues_sent=None,
                        fix_triggered=False,
                        fix_produced_different_sql=None,
                    ))
                break

            # Budget exhausted (final assessment iteration) → accept best result
            if is_final_assessment:
                if self._trace:
                    iter_trace.append(_build_iter_entry(
                        iteration=iteration,
                        is_final_assessment=True,
                        sql=sql_at_iter_start,
                        prev_sql=iter_trace[-1]["sql"] if iter_trace else None,
                        exec_result=current_result,
                        verif_eval=verif_eval,
                        exec_issues_sent=None,
                        verif_issues_sent=None,
                        fix_triggered=False,
                        fix_produced_different_sql=None,
                    ))
                break

            # ------------------------------------------------------------------
            # Build combined feedback for the fix prompt
            # ------------------------------------------------------------------
            exec_issue_lines: list[str] = []
            if not exec_ok:
                category = _categorize_error(
                    current_result.error, current_result.is_empty
                )
                instruction = _error_instruction(category, current_result.error)
                exec_issue_lines.append(instruction)

            # Keep failure_hints for trace storage (human-readable summary)
            verif_issue_lines: list[str] = []
            if verif_eval is not None and not verif_eval.all_pass:
                verif_issue_lines = verif_eval.failure_hints

            # ------------------------------------------------------------------
            # Build and send fix prompt (with rich per-test verification context)
            # ------------------------------------------------------------------
            fix_prompt = self._build_fix_prompt(
                sql=current_sql,
                question=question,
                evidence=evidence,
                schemas=schemas,
                cell_matches=cell_matches,
                exec_issues=exec_issue_lines,
                verif_eval=verif_eval,
                verif_specs=verif_specs,
                exec_result=current_result,
            )

            try:
                client = get_client()
                response = await client.generate(
                    model=settings.model_fast,
                    system=[
                        CacheableText(
                            text="You are an expert SQL debugger.", cache=False
                        )
                    ],
                    messages=[{"role": "user", "content": fix_prompt}],
                    tools=[],
                    temperature=0.0,
                    max_tokens=1000,
                )
                fixed_sql_raw = response.text or ""
                fixed_sql = clean_sql(fixed_sql_raw)
            except LLMError as exc:
                logger.warning(
                    "Fix iteration %d failed for candidate %r: %s",
                    iteration + 1,
                    candidate.generator_id,
                    exc,
                )
                get_tracker().record(FallbackEvent(
                    component="query_fixer",
                    trigger="llm_error",
                    action="fix_loop_break",
                    details={
                        "candidate_id": candidate.generator_id,
                        "iteration": iteration + 1,
                        "error": str(exc),
                    },
                ))
                if self._trace:
                    iter_trace.append(_build_iter_entry(
                        iteration=iteration,
                        is_final_assessment=False,
                        sql=sql_at_iter_start,
                        prev_sql=iter_trace[-1]["sql"] if iter_trace else None,
                        exec_result=current_result,
                        verif_eval=verif_eval,
                        exec_issues_sent=exec_issue_lines,
                        verif_issues_sent=verif_issue_lines,
                        fix_triggered=True,
                        fix_produced_different_sql=None,  # LLM call failed
                    ))
                break

            if not fixed_sql:
                logger.warning(
                    "Fix iteration %d produced empty SQL for candidate %r",
                    iteration + 1,
                    candidate.generator_id,
                )
                get_tracker().record(FallbackEvent(
                    component="query_fixer",
                    trigger="empty_output",
                    action="fix_loop_break",
                    details={
                        "candidate_id": candidate.generator_id,
                        "iteration": iteration + 1,
                    },
                ))
                if self._trace:
                    iter_trace.append(_build_iter_entry(
                        iteration=iteration,
                        is_final_assessment=False,
                        sql=sql_at_iter_start,
                        prev_sql=iter_trace[-1]["sql"] if iter_trace else None,
                        exec_result=current_result,
                        verif_eval=verif_eval,
                        exec_issues_sent=exec_issue_lines,
                        verif_issues_sent=verif_issue_lines,
                        fix_triggered=True,
                        fix_produced_different_sql=None,  # produced empty SQL
                    ))
                break

            if self._trace:
                iter_trace.append(_build_iter_entry(
                    iteration=iteration,
                    is_final_assessment=False,
                    sql=sql_at_iter_start,
                    prev_sql=iter_trace[-1]["sql"] if iter_trace else None,
                    exec_result=current_result,
                    verif_eval=verif_eval,
                    exec_issues_sent=exec_issue_lines,
                    verif_issues_sent=verif_issue_lines,
                    fix_triggered=True,
                    fix_produced_different_sql=(fixed_sql != sql_at_iter_start),
                ))

            # Execute the fixed SQL and continue to next iteration
            new_result = execute_sql(db_path, fixed_sql)
            fix_iterations += 1
            current_sql = fixed_sql
            current_result = new_result

        # Compute raw confidence score incorporating verification results
        raw_score = _raw_confidence(
            current_result, fix_iterations, current_sql, last_verif_eval
        )

        return FixedCandidate(
            original_sql=candidate.sql,
            final_sql=current_sql,
            generator_id=candidate.generator_id,
            fix_iterations=fix_iterations,
            execution_result=current_result,
            confidence_score=raw_score,  # will be normalized later
            verification_results=last_verif_eval,
            iteration_trace=iter_trace,
        )

    @staticmethod
    def _build_fix_prompt(
        sql: str,
        question: str,
        evidence: str,
        schemas: "LinkedSchemas",
        cell_matches: list,
        exec_issues: list[str],
        verif_eval: Optional[VerificationEvaluation] = None,
        verif_specs: Optional[list[VerificationTestSpec]] = None,
        exec_result: Optional["ExecutionResult"] = None,
    ) -> str:
        """Build the combined fix prompt with execution and verification feedback.

        For each failing verification test, includes test-type-specific context:
          - grain: the verification_sql_upper that computed the row-count bound
          - column_alignment: expected_column_count (structural column count check)
          - ordering: required SQL keywords that are missing
          - scale: numeric_min / numeric_max bounds
          - null: which columns must not be NULL
          - all tests: the spec's fix_hint

        When exec_result is provided and the SQL executed successfully, a sample
        of the first 5 result rows is appended so the fixer can see the structural
        problem (e.g. duplicate join rows for grain failures).
        """
        cell_match_lines: list[str] = []
        for cm in cell_matches:
            cell_match_lines.append(
                f"  {cm.table}.{cm.column} = {cm.matched_value!r}"
            )
        formatted_cells = (
            "\n".join(cell_match_lines) if cell_match_lines else "  (none)"
        )

        issues_block_parts: list[str] = []
        if exec_issues:
            issues_block_parts.append(
                "[Execution Error]\n" + "\n".join(f"  {line}" for line in exec_issues)
            )

        # Build rich per-test verification context
        if verif_eval is not None and not verif_eval.all_pass:
            spec_by_type: dict[str, VerificationTestSpec] = {}
            if verif_specs:
                for spec in verif_specs:
                    spec_by_type[spec.test_type] = spec

            verif_blocks: list[str] = []
            for result in verif_eval.test_results:
                if result.status != "fail":
                    continue
                severity = "critical" if result.is_critical else "minor"
                lines: list[str] = [
                    f"  ── {result.test_type.upper()} TEST FAILED ({severity}) ──",
                    f"  Observed: {result.actual_outcome}",
                ]
                spec = spec_by_type.get(result.test_type)
                if spec is not None:
                    if result.test_type == "grain":
                        if spec.verification_sql_upper:
                            lines.append(
                                f"  Upper bound computed by:\n"
                                f"    {spec.verification_sql_upper}"
                            )
                        elif spec.verification_sql:
                            lines.append(
                                f"  Bound computed by:\n"
                                f"    {spec.verification_sql}"
                            )
                    elif result.test_type == "column_alignment":
                        if spec.expected_column_count is not None:
                            lines.append(
                                f"  Expected column count: {spec.expected_column_count}"
                            )
                    elif result.test_type == "ordering":
                        if spec.required_sql_keywords:
                            lines.append(
                                f"  Required keywords: {', '.join(spec.required_sql_keywords)}"
                            )
                    elif result.test_type == "scale":
                        parts: list[str] = []
                        if spec.numeric_min is not None:
                            parts.append(f"min={spec.numeric_min}")
                        if spec.numeric_max is not None:
                            parts.append(f"max={spec.numeric_max}")
                        if parts:
                            lines.append(f"  Expected range: {', '.join(parts)}")
                    elif result.test_type == "null":
                        if spec.check_columns:
                            lines.append(
                                f"  Columns that must not be NULL: {', '.join(spec.check_columns)}"
                            )
                    if spec.fix_hint:
                        lines.append(f"  Fix hint: {spec.fix_hint}")
                verif_blocks.append("\n".join(lines))

            if verif_blocks:
                issues_block_parts.append(
                    "[Verification Issues]\n\n" + "\n\n".join(verif_blocks)
                )

        issues_block = (
            "\n\n".join(issues_block_parts)
            if issues_block_parts
            else "  (none recorded)"
        )

        # Build result sample section (first 5 rows) when SQL executed successfully
        result_sample = ""
        if exec_result is not None and exec_result.success and exec_result.rows:
            sample_rows = exec_result.rows[:5]
            row_lines: list[str] = []
            for i, row in enumerate(sample_rows, 1):
                row_str = f"  Row {i}: {tuple(row)}"
                if len(row_str) > 120:
                    row_str = row_str[:117] + "..."
                row_lines.append(row_str)
            if len(exec_result.rows) > 5:
                row_lines.append(f"  ... ({len(exec_result.rows)} rows total)")
            result_sample = (
                "\nCurrent result sample (first 5 rows):\n"
                + "\n".join(row_lines)
                + "\n"
            )

        return (
            "You are an expert SQL debugger. Fix the SQL query below.\n\n"
            "Database Schema:\n"
            f"{schemas.s2_ddl}\n\n"
            f"Question: {question}\n"
            f"Evidence: {evidence or 'None'}\n"
            f"Matched values in database:\n{formatted_cells}\n\n"
            "Current SQL:\n"
            f"{sql}\n"
            f"{result_sample}\n"
            "Issues found:\n"
            f"{issues_block}\n\n"
            "Write only the corrected SQL query. No explanation."
        )


# ---------------------------------------------------------------------------
# Confidence normalization
# ---------------------------------------------------------------------------

def _normalize_confidence(candidates: list[FixedCandidate]) -> list[FixedCandidate]:
    """Normalize confidence scores to [0, 1] across the candidate pool.

    Rules:
    - Candidates with raw_score == 0.0 (failed) keep 0.0 and are excluded
      from the normalization pool.
    - Among non-zero candidates:
        * If all share the same score → all get 1.0
        * Otherwise → (raw - min) / (max - min)
    """
    nonzero = [c for c in candidates if c.confidence_score > 0.0]
    if not nonzero:
        return candidates

    min_score = min(c.confidence_score for c in nonzero)
    max_score = max(c.confidence_score for c in nonzero)

    for c in candidates:
        if c.confidence_score <= 0.0:
            c.confidence_score = 0.0
            continue
        if max_score == min_score:
            c.confidence_score = 1.0
        else:
            c.confidence_score = (c.confidence_score - min_score) / (max_score - min_score)

    return candidates
