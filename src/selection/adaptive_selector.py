"""
Op 9: Adaptive SQL Selection

Selects the best SQL from the fixed candidate pool using:
  - Fast path: if all candidates agree (1 cluster), return shortest SQL immediately.
  - Tournament path: if 2+ clusters, run pairwise comparisons with model_fast.

Steps:
  9.1 — Execute and cluster candidates by result equivalence.
  9.2 — Fast path (unanimous): return shortest SQL, 0 API calls.
  9.3 — Representative selection: 1 per cluster, sorted by size then generator rank.
  9.4 — Pairwise tournament: C(m,2) comparisons via asyncio.gather().
  9.5 — Final selection: argmax wins; tiebreakers: cluster size, confidence, generator rank.
  9.6 — Fallback: if <2 executable candidates, return highest-confidence candidate.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from src.config.settings import settings
from src.data.database import ExecutionResult, execute_sql
from src.llm import CacheableText, LLMError, ToolParam, get_client, sanitize_prompt_text

if TYPE_CHECKING:
    from src.fixing.query_fixer import FixedCandidate
    from src.schema_linking.schema_linker import LinkedSchemas

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class SelectionResult:
    """Result of Op 9 adaptive selection."""
    final_sql: str
    selection_method: str        # "fast_path" | "tournament" | "fallback"
    tournament_wins: dict        # generator_id → win count
    confidence: float
    cluster_count: int
    candidates_evaluated: int


# ---------------------------------------------------------------------------
# Tool definition for pairwise comparison
# ---------------------------------------------------------------------------

_SELECT_WINNER_TOOL = ToolParam(
    name="select_winner",
    description=(
        "Select the SQL candidate that better answers the question. "
        "Consider execution results, row counts, column values, and SQL correctness."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "winner": {"type": "string", "enum": ["A", "B"]},
            "reason": {"type": "string"},
        },
        "required": ["winner"],
    },
)


# ---------------------------------------------------------------------------
# Generator performance ranking
# ---------------------------------------------------------------------------

def _generator_rank(generator_id: str) -> int:
    """
    Lower number = higher priority (better generator).

    Ranking:
      0 — reasoning_A*  (starts with "reasoning")
      1 — complex_B2* / standard_B2* / *B2* (contains "B2" or starts with "complex")
      2 — icl_C*        (starts with "icl")
      3 — standard_B1*  (starts with "standard") — lowest
    """
    gid = generator_id.lower()
    if gid.startswith("reasoning"):
        return 0
    if "b2" in gid or gid.startswith("complex"):
        return 1
    if gid.startswith("icl"):
        return 2
    return 3  # standard B1 or unknown


# ---------------------------------------------------------------------------
# Result clustering helpers
# ---------------------------------------------------------------------------

def _normalize_value(v) -> str:
    """Normalize a single cell value to a consistent string representation."""
    if v is None:
        return "NULL"
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def _cluster_key(rows: list) -> str:
    """
    Compute a stable cluster key from an execution result's rows.

    Rows are sorted before hashing so that different orderings of the same
    result set map to the same cluster.
    """
    normalized_rows = [
        tuple(_normalize_value(cell) for cell in row) for row in rows
    ]
    sorted_rows = sorted(normalized_rows)
    return str(sorted_rows)


# ---------------------------------------------------------------------------
# Execution result formatting helpers
# ---------------------------------------------------------------------------

def _extract_column_names(sql: str) -> list[str]:
    """
    Extract column names / aliases from the SELECT clause of a SQL query.

    Handles common patterns:
    - Simple columns: ``SELECT a, b FROM t`` → ``["a", "b"]``
    - Table-prefixed: ``SELECT T1.col FROM t`` → ``["col"]``
    - AS aliases: ``SELECT T1.x AS alias FROM t`` → ``["alias"]``
    - Expressions with AS: ``SELECT COUNT(id) AS cnt FROM t`` → ``["cnt"]``

    Returns an empty list if ``SELECT *`` is detected or parsing fails.
    The list is used only for display purposes in the tournament prompt.
    """
    import re

    sql_clean = sql.strip().rstrip(";")

    # Locate SELECT keyword
    m = re.match(r"SELECT\s+", sql_clean, re.IGNORECASE)
    if not m:
        return []

    start = m.end()

    # Walk to the first top-level FROM (respecting parenthesis depth)
    depth = 0
    end = len(sql_clean)
    for i in range(start, len(sql_clean)):
        c = sql_clean[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif depth == 0 and re.match(r"\bFROM\b", sql_clean[i:], re.IGNORECASE):
            end = i
            break

    sel = sql_clean[start:end].strip()

    # Handle SELECT * / SELECT DISTINCT *
    if re.fullmatch(r"DISTINCT\s+\*|\*", sel, re.IGNORECASE):
        return []
    sel = re.sub(r"^DISTINCT\s+", "", sel, flags=re.IGNORECASE)

    # Split SELECT clause on top-level commas
    exprs: list[str] = []
    buf: list[str] = []
    depth = 0
    for c in sel:
        if c == "(":
            depth += 1
            buf.append(c)
        elif c == ")":
            depth -= 1
            buf.append(c)
        elif c == "," and depth == 0:
            exprs.append("".join(buf).strip())
            buf = []
        else:
            buf.append(c)
    if buf:
        exprs.append("".join(buf).strip())

    # SQL keyword set to skip when falling back to last-identifier heuristic
    _SKIP = {
        "AS", "FROM", "WHERE", "AND", "OR", "NOT", "NULL", "IS", "IN", "LIKE",
        "BETWEEN", "CASE", "WHEN", "THEN", "ELSE", "END", "CAST", "INTEGER",
        "REAL", "TEXT", "BLOB", "NUMERIC", "DISTINCT", "TOP", "LIMIT", "OFFSET",
    }

    names: list[str] = []
    for expr in exprs:
        expr = expr.strip()
        if not expr:
            continue

        # Find AS <alias> at parenthesis depth 0 by walking character by character
        alias: Optional[str] = None
        depth = 0
        for i, c in enumerate(expr):
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            elif depth == 0:
                # Only check at a word boundary (prev char is not alphanumeric / underscore)
                prev = expr[i - 1] if i > 0 else " "
                if not (prev.isalnum() or prev == "_"):
                    am = re.match(r"AS\s+(\w+)\s*$", expr[i:], re.IGNORECASE)
                    if am:
                        alias = am.group(1)
                        break

        if alias:
            names.append(alias)
        else:
            # Fall back: table.column pattern at end of expression
            dm = re.search(r"\.(\w+)\s*$", expr)
            if dm:
                names.append(dm.group(1))
            else:
                idents = [
                    x
                    for x in re.findall(r"\b([A-Za-z_]\w*)\b", expr)
                    if x.upper() not in _SKIP
                ]
                names.append(idents[-1] if idents else f"col_{len(names) + 1}")

    return names


def _format_execution_result(
    result: "ExecutionResult",
    sql: str,
    max_rows: int = 5,
) -> str:
    """
    Format an execution result as a human-readable block for the tournament prompt.

    Includes:
    - Total row count with a truncation notice when rows are clipped
    - Column names extracted from the SQL SELECT clause
    - First ``max_rows`` rows rendered as a simple text table
    """
    if not result.success:
        return f"Execution error: {result.error}"

    if result.is_empty:
        return "Query returned 0 rows."

    total_rows = len(result.rows)
    sample = result.rows[:max_rows]

    # Determine column count from the first row
    ncols = len(sample[0]) if sample and isinstance(sample[0], (tuple, list)) else 1

    col_names = _extract_column_names(sql)
    # Pad or trim to match actual column count
    while len(col_names) < ncols:
        col_names.append(f"col_{len(col_names) + 1}")
    col_names = col_names[:ncols]

    _MAX_VAL = 20

    def _fmt(v) -> str:
        s = "NULL" if v is None else str(v)
        return s if len(s) <= _MAX_VAL else s[: _MAX_VAL - 3] + "..."

    # Compute column display widths (header vs. content, capped at _MAX_VAL)
    widths = [len(cn) for cn in col_names]
    for row in sample:
        vals = row if isinstance(row, (tuple, list)) else (row,)
        for i, v in enumerate(vals[:ncols]):
            widths[i] = min(max(widths[i], len(_fmt(v))), _MAX_VAL)

    header = " | ".join(cn.ljust(widths[i]) for i, cn in enumerate(col_names))
    sep = "-+-".join("-" * w for w in widths)

    lines: list[str] = []
    if total_rows > max_rows:
        lines.append(f"Total rows: {total_rows} (showing first {max_rows})")
    else:
        lines.append(f"Total rows: {total_rows}")
    lines.append(f"Columns: {', '.join(col_names)}")
    lines.append(header)
    lines.append(sep)

    for row in sample:
        vals = row if isinstance(row, (tuple, list)) else (row,)
        cells = [
            _fmt(vals[i] if i < len(vals) else "").ljust(widths[i])
            for i in range(ncols)
        ]
        lines.append(" | ".join(cells))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# AdaptiveSelector
# ---------------------------------------------------------------------------

class AdaptiveSelector:
    """
    Operation 9: Adaptive SQL Selection.

    Usage::

        selector = AdaptiveSelector()
        result = await selector.select(
            candidates=fixed_candidates,
            question=question,
            evidence=evidence,
            schemas=schemas,
            db_path=db_path,
        )
    """

    async def select(
        self,
        candidates: list["FixedCandidate"],
        question: str,
        evidence: str,
        schemas: "LinkedSchemas",
        db_path: str,
    ) -> SelectionResult:
        """
        Select the best SQL from a pool of fixed candidates.

        Parameters
        ----------
        candidates:
            FixedCandidate pool from Op 8 (query fixer).
        question:
            Natural-language question.
        evidence:
            Auxiliary evidence (may be empty or None).
        schemas:
            LinkedSchemas from Op 6 (for tournament prompt context).
        db_path:
            Filesystem path to the SQLite database.

        Returns
        -------
        SelectionResult with final_sql and metadata.
        """
        if not candidates:
            logger.warning("AdaptiveSelector received empty candidate list — returning fallback.")
            return SelectionResult(
                final_sql="",
                selection_method="fallback",
                tournament_wins={},
                confidence=0.0,
                cluster_count=0,
                candidates_evaluated=0,
            )

        # Step 9.6: Check if any executable candidates exist
        executable = [c for c in candidates if c.confidence_score > 0]

        if len(executable) < 2:
            # Fallback: return highest confidence candidate
            best = max(candidates, key=lambda c: c.confidence_score)
            logger.warning(
                "Fewer than 2 executable candidates (found %d); using fallback.",
                len(executable),
            )
            return SelectionResult(
                final_sql=best.final_sql,
                selection_method="fallback",
                tournament_wins={},
                confidence=best.confidence_score,
                cluster_count=len(executable),
                candidates_evaluated=len(candidates),
            )

        # Step 9.1: Execute and cluster by result equivalence
        # Re-execute each surviving candidate to get fresh results for clustering
        loop = asyncio.get_event_loop()
        exec_results: list[tuple["FixedCandidate", ExecutionResult]] = []
        for cand in executable:
            result = await loop.run_in_executor(None, execute_sql, db_path, cand.final_sql)
            exec_results.append((cand, result))

        # Build clusters: key → list of (FixedCandidate, ExecutionResult)
        clusters: dict[str, list[tuple["FixedCandidate", ExecutionResult]]] = {}
        for cand, result in exec_results:
            if result.success:
                key = _cluster_key(result.rows)
            else:
                # Failed candidates each get their own unique cluster (don't group failures)
                key = f"__failed__{cand.generator_id}"
            clusters.setdefault(key, []).append((cand, result))

        cluster_count = len(clusters)

        # Step 9.2: Fast path — unanimous agreement
        if cluster_count == 1:
            cluster_members = next(iter(clusters.values()))
            # Select shortest SQL among cluster members
            best_cand = min(cluster_members, key=lambda cr: len(cr[0].final_sql))[0]
            return SelectionResult(
                final_sql=best_cand.final_sql,
                selection_method="fast_path",
                tournament_wins={},
                confidence=best_cand.confidence_score,
                cluster_count=1,
                candidates_evaluated=len(executable),
            )

        # Step 9.3: Representative selection — 1 per cluster
        # For each cluster, pick the rep with highest confidence (ties: shortest SQL)
        representatives: list[tuple[str, "FixedCandidate", ExecutionResult, int]] = []
        # (cluster_key, rep_candidate, rep_result, cluster_size)
        for cluster_key_str, members in clusters.items():
            rep_cand, rep_result = max(
                members,
                key=lambda cr: (
                    cr[0].confidence_score,
                    -len(cr[0].final_sql),  # negative length for tie-break (shorter is better)
                ),
            )
            cluster_size = len(members)
            representatives.append((cluster_key_str, rep_cand, rep_result, cluster_size))

        # Sort representatives:
        #   Primary: cluster size descending (non-empty clusters before empty ones)
        #   Secondary: generator performance ranking (lower rank = better)
        #
        # "Empty" cluster = ALL members have is_empty=True in their execution result
        def _all_empty(cluster_key_str: str) -> bool:
            members = clusters[cluster_key_str]
            return all(r.is_empty for _, r in members)

        def _rep_sort_key(rep_tuple):
            ck, cand, result, size = rep_tuple
            is_all_empty = _all_empty(ck)
            # (all_empty bucket, -cluster_size, generator_rank)
            # is_all_empty=True → deprioritized (1 > 0)
            return (int(is_all_empty), -size, _generator_rank(cand.generator_id))

        representatives.sort(key=_rep_sort_key)

        # Step 9.4: Pairwise tournament
        # Higher-ranked rep presented as "A" (positional bias exploitation)
        m = len(representatives)
        wins: dict[str, int] = {rep[1].generator_id: 0 for rep in representatives}

        async def _compare_pair(
            idx_a: int,
            idx_b: int,
        ) -> Optional[str]:
            """Run a single pairwise comparison; return generator_id of winner."""
            _, cand_a, result_a, size_a = representatives[idx_a]
            _, cand_b, result_b, size_b = representatives[idx_b]

            result_a_str = _format_execution_result(result_a, cand_a.final_sql)
            result_b_str = _format_execution_result(result_b, cand_b.final_sql)

            # Sanitize user-supplied and LLM-generated text before embedding in a
            # tool-use prompt.  Gemini's function-call parser fails (MALFORMED_FUNCTION_CALL)
            # on backtick-quoted identifiers and control characters — the same fix
            # that was applied to schema_linker.py and context_grounder.py.
            safe_question = sanitize_prompt_text(question)
            safe_evidence = sanitize_prompt_text(evidence or "None")
            safe_sql_a = sanitize_prompt_text(cand_a.final_sql)
            safe_sql_b = sanitize_prompt_text(cand_b.final_sql)

            prompt = (
                "You are evaluating two SQL queries for a natural language question. "
                "Choose which query better answers the question based on the schema "
                "and execution results.\n\n"
                f"Question: {safe_question}\n"
                f"Evidence: {safe_evidence}\n"
                f"Database Schema (filtered):\n{schemas.s2_markdown}\n\n"
                f"Candidate A (generated by {cand_a.generator_id}):\n"
                f"{safe_sql_a}\n\n"
                f"=== Candidate A Execution Result ===\n"
                f"{result_a_str}\n\n"
                f"Candidate B (generated by {cand_b.generator_id}):\n"
                f"{safe_sql_b}\n\n"
                f"=== Candidate B Execution Result ===\n"
                f"{result_b_str}\n\n"
                "Given the question, evidence, schema, and both execution results above, "
                "which SQL better answers the question? "
                "Consider: row counts, column values, and the correctness of the SQL logic."
            )

            try:
                client = get_client()
                response = await client.generate(
                    model=settings.model_fast,
                    system=[CacheableText(text="You are an expert SQL evaluator.", cache=False)],
                    messages=[{"role": "user", "content": prompt}],
                    tools=[_SELECT_WINNER_TOOL],
                    tool_choice_name="select_winner",
                    max_tokens=256,
                    temperature=0.0,
                )
                if response.tool_inputs:
                    winner_letter = response.tool_inputs[0].get("winner", "A")
                else:
                    winner_letter = "A"

                if winner_letter not in ("A", "B"):
                    logger.warning(
                        "Pairwise comparison returned invalid winner %r; defaulting to A",
                        winner_letter,
                    )
                    winner_letter = "A"

                if winner_letter == "A":
                    return cand_a.generator_id
                else:
                    return cand_b.generator_id

            except LLMError as exc:
                logger.warning("Pairwise comparison failed: %s; defaulting to A.", exc)
                return cand_a.generator_id

        # Run all C(m,2) comparisons concurrently
        pair_tasks = [
            _compare_pair(i, j)
            for i in range(m)
            for j in range(i + 1, m)
        ]
        pair_results = await asyncio.gather(*pair_tasks)

        for winner_id in pair_results:
            if winner_id and winner_id in wins:
                wins[winner_id] += 1

        # Step 9.5: Final selection
        # Winner = argmax wins; tiebreakers: cluster size, confidence, generator rank
        def _final_sort_key(rep_tuple):
            ck, cand, result, size = rep_tuple
            gen_wins = wins.get(cand.generator_id, 0)
            # Sort descending by wins, then cluster size, then confidence, then generator rank
            return (-gen_wins, -size, -cand.confidence_score, _generator_rank(cand.generator_id))

        representatives.sort(key=_final_sort_key)
        winner_tuple = representatives[0]
        _, winner_cand, _, winner_cluster_size = winner_tuple

        return SelectionResult(
            final_sql=winner_cand.final_sql,
            selection_method="tournament",
            tournament_wins=wins,
            confidence=winner_cand.confidence_score,
            cluster_count=cluster_count,
            candidates_evaluated=len(executable),
        )
