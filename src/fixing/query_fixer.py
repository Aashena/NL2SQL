"""
Op 8: Query Fixer

For each SQL candidate that fails execution or returns empty results, attempt
up to β=2 fix iterations using a lightweight LLM (model_fast tier).

Steps:
  8.1 — Execute all candidates, flag those needing a fix.
  8.2 — Categorize errors (syntax_error, schema_error, type_error, empty_result).
  8.3 — Fix loop: up to β=2 iterations per failing candidate.
  8.4 — Produce FixedCandidate dataclass for each input.
  8.5 — Confidence scoring + normalization across the candidate pool.

Fix calls for independent candidates run concurrently via asyncio.gather().
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from src.config.settings import settings
from src.data.database import ExecutionResult, execute_sql
from src.generation.base_generator import SQLCandidate, clean_sql
from src.llm import CacheableText, LLMError, get_client

if TYPE_CHECKING:
    from src.schema_linking.schema_linker import LinkedSchemas

logger = logging.getLogger(__name__)

# Maximum number of fix iterations per candidate.
_BETA = 2

# ---------------------------------------------------------------------------
# Error categories
# ---------------------------------------------------------------------------

_ErrorCategory = str  # one of the constants below

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
        # Try to extract the problematic column/table name
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
        return f"Schema error: {error_message}. Check the schema and use the correct table/column names."
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
    """A SQL candidate after the fix loop has been applied."""
    original_sql: str
    final_sql: str
    generator_id: str
    fix_iterations: int              # 0 if no fix was needed
    execution_result: ExecutionResult
    confidence_score: float          # normalized [0, 1] (computed by QueryFixer)


# ---------------------------------------------------------------------------
# Confidence scoring helpers
# ---------------------------------------------------------------------------

_AGG_PATTERN = re.compile(
    r"\b(COUNT|SUM|AVG|MIN|MAX|GROUP\s+BY)\b",
    re.IGNORECASE,
)


def _is_aggregation_query(sql: str) -> bool:
    """Return True if the SQL contains any aggregation keyword."""
    return bool(_AGG_PATTERN.search(sql))


def _raw_confidence(
    execution_result: ExecutionResult,
    fix_iterations: int,
    sql: str,
) -> float:
    """Compute the un-normalized confidence score for a single candidate.

    Rules:
      +1.0  if execution succeeded and result is non-empty
      +0.5  plausibility bonus:
              aggregation query → exactly 1 row is plausible
              non-aggregation query → 1–100 rows is plausible
      -0.5  per fix iteration needed (−0.5 × fix_iterations)
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

    # Penalty per fix iteration
    score -= 0.5 * fix_iterations

    return score


# ---------------------------------------------------------------------------
# QueryFixer class
# ---------------------------------------------------------------------------

class QueryFixer:
    """
    Operation 8: execute all candidates, fix errors/empty results, score.

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
    """

    async def fix_candidates(
        self,
        candidates: list[SQLCandidate],
        question: str,
        evidence: str,
        schemas: "LinkedSchemas",
        db_path: str,
        cell_matches: list,
    ) -> list[FixedCandidate]:
        """Execute + fix all candidates concurrently; return scored FixedCandidates.

        Parameters
        ----------
        candidates:
            SQL candidates from the generation step (Op 7).
        question:
            Natural-language question.
        evidence:
            Auxiliary evidence string (may be empty or None).
        schemas:
            LinkedSchemas from Op 6; S₂ DDL is used as schema context in fix prompts.
        db_path:
            Filesystem path to the SQLite database.
        cell_matches:
            CellMatch objects from context grounding (for fix prompt context).

        Returns
        -------
        list[FixedCandidate] — one per input candidate, with confidence scores
        normalized across the pool.
        """
        # Step 8.1: execute all candidates first (synchronous, fast)
        initial_results: list[tuple[SQLCandidate, ExecutionResult]] = []
        for cand in candidates:
            result = execute_sql(db_path, cand.sql)
            initial_results.append((cand, result))

        # Step 8.3: fix loop — run fix coroutines concurrently
        fix_tasks = [
            self._fix_single(
                candidate=cand,
                initial_result=result,
                question=question,
                evidence=evidence,
                schemas=schemas,
                db_path=db_path,
                cell_matches=cell_matches,
            )
            for cand, result in initial_results
        ]
        fixed_candidates: list[FixedCandidate] = await asyncio.gather(*fix_tasks)

        # Step 8.5: normalize confidence scores
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
    ) -> FixedCandidate:
        """Run the fix loop for a single candidate.

        If the initial execution already succeeded with non-empty rows, the
        candidate passes through unchanged (fix_iterations=0).
        """
        current_sql = candidate.sql
        current_result = initial_result
        fix_iterations = 0

        for iteration in range(_BETA):
            # Stop if no fix is needed
            if current_result.success and not current_result.is_empty:
                break

            # Step 8.2: categorize the error
            category = _categorize_error(current_result.error, current_result.is_empty)

            # Build the fix prompt
            error_msg = current_result.error or "Query returned no rows."
            instruction = _error_instruction(category, current_result.error)

            cell_match_lines: list[str] = []
            for cm in cell_matches:
                cell_match_lines.append(
                    f"  {cm.table}.{cm.column} = {cm.matched_value!r}"
                )
            formatted_cells = (
                "\n".join(cell_match_lines) if cell_match_lines else "  (none)"
            )

            fix_prompt = (
                "You are an expert SQL debugger. Fix the SQL query below.\n\n"
                "Database Schema:\n"
                f"{schemas.s2_ddl}\n\n"
                f"Question: {question}\n"
                f"Evidence: {evidence or 'None'}\n"
                f"Matched values in database:\n{formatted_cells}\n\n"
                "Broken SQL:\n"
                f"{current_sql}\n\n"
                f"Error: {error_msg}\n\n"
                f"Fix instruction: {instruction}\n\n"
                "Write only the corrected SQL query. No explanation."
            )

            # Step 8.3: call the fixer LLM
            try:
                client = get_client()
                response = await client.generate(
                    model=settings.model_fast,
                    system=[CacheableText(text="You are an expert SQL debugger.", cache=False)],
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
                break

            if not fixed_sql:
                logger.warning(
                    "Fix iteration %d produced empty SQL for candidate %r",
                    iteration + 1,
                    candidate.generator_id,
                )
                break

            # Execute the fixed SQL
            new_result = execute_sql(db_path, fixed_sql)
            fix_iterations += 1
            current_sql = fixed_sql
            current_result = new_result

        # Compute raw confidence score
        raw_score = _raw_confidence(current_result, fix_iterations, current_sql)

        return FixedCandidate(
            original_sql=candidate.sql,
            final_sql=current_sql,
            generator_id=candidate.generator_id,
            fix_iterations=fix_iterations,
            execution_result=current_result,
            confidence_score=raw_score,  # will be normalized later
        )


# ---------------------------------------------------------------------------
# Confidence normalization
# ---------------------------------------------------------------------------

def _normalize_confidence(candidates: list[FixedCandidate]) -> list[FixedCandidate]:
    """Normalize confidence scores to [0, 1] across the candidate pool.

    Rules:
    - Candidates with raw_score == 0.0 (failed) keep 0.0 and are excluded from
      the normalization pool.
    - Among the non-zero-score candidates:
        * If all share the same score → all get 1.0
        * Otherwise → (raw - min) / (max - min)
    """
    nonzero = [c for c in candidates if c.confidence_score > 0.0]
    if not nonzero:
        # All failed — leave all at 0.0
        return candidates

    min_score = min(c.confidence_score for c in nonzero)
    max_score = max(c.confidence_score for c in nonzero)

    for c in candidates:
        if c.confidence_score <= 0.0:
            c.confidence_score = 0.0
            continue
        if max_score == min_score:
            # All tied → all get 1.0
            c.confidence_score = 1.0
        else:
            c.confidence_score = (c.confidence_score - min_score) / (max_score - min_score)

    return candidates
