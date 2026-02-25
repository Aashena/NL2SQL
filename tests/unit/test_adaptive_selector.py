"""
Unit tests for src/selection/adaptive_selector.py — Op 9: Adaptive Selector

All 12 tests from the spec. LLM calls are mocked; execute_sql is also patched
so no real database is needed — FixedCandidate objects carry pre-defined
ExecutionResult objects that the mock returns directly.

Patch target: src.selection.adaptive_selector.execute_sql
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data.database import ExecutionResult
from src.fixing.query_fixer import FixedCandidate
from src.llm.base import LLMResponse
from src.schema_linking.schema_linker import LinkedSchemas
from src.selection.adaptive_selector import (
    AdaptiveSelector,
    SelectionResult,
    _extract_column_names,
    _format_execution_result,
    _generator_rank,
)


# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------

def _make_exec_result(
    rows: list,
    success: bool = True,
    error: Optional[str] = None,
    is_empty: bool = False,
) -> ExecutionResult:
    return ExecutionResult(
        success=success,
        rows=rows,
        error=error,
        execution_time=0.001,
        is_empty=is_empty or (success and len(rows) == 0),
    )


def _make_candidate(
    sql: str,
    generator_id: str,
    rows: list,
    confidence: float = 1.0,
    success: bool = True,
    error: Optional[str] = None,
) -> FixedCandidate:
    """Create a FixedCandidate with a pre-defined ExecutionResult."""
    is_empty = success and len(rows) == 0
    return FixedCandidate(
        original_sql=sql,
        final_sql=sql,
        generator_id=generator_id,
        fix_iterations=0,
        execution_result=_make_exec_result(rows, success=success, error=error, is_empty=is_empty),
        confidence_score=confidence,
    )


def _make_schemas() -> LinkedSchemas:
    return LinkedSchemas(
        s1_ddl="CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT);",
        s1_markdown=(
            "## Table: t\n"
            "| Column | Type | Description |\n"
            "|--------|------|-------------|\n"
            "| id | INTEGER (PK) | id |\n"
            "| val | TEXT | value |"
        ),
        s2_ddl="CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT);",
        s2_markdown=(
            "## Table: t\n"
            "| Column | Type | Description |\n"
            "|--------|------|-------------|\n"
            "| id | INTEGER (PK) | id |\n"
            "| val | TEXT | value |"
        ),
        s1_fields=[("t", "id"), ("t", "val")],
        s2_fields=[("t", "id"), ("t", "val")],
        selection_reasoning="Selected t table columns",
    )


def _make_mock_client(winner: str = "A") -> AsyncMock:
    """Return an AsyncMock LLM client whose generate() returns free-text with FINAL: winner."""
    mock_client = AsyncMock()
    mock_client.generate = AsyncMock(
        return_value=LLMResponse(
            tool_inputs=[],
            text=f"After analysis, the better candidate is clear.\nFINAL: {winner}",
            thinking=None,
            input_tokens=100,
            output_tokens=20,
        )
    )
    return mock_client


def _patch_execute_sql(candidates: list[FixedCandidate]):
    """
    Build a side_effect function that returns the pre-defined ExecutionResult
    stored in the matching FixedCandidate (matched by final_sql).
    Falls back to a failed result if no match is found.
    """
    sql_to_result = {c.final_sql: c.execution_result for c in candidates}

    def _execute(db_path: str, sql: str, timeout: float = 30.0) -> ExecutionResult:
        return sql_to_result.get(
            sql,
            ExecutionResult(
                success=False,
                rows=[],
                error="Unknown SQL in mock",
                execution_time=0.0,
                is_empty=False,
            ),
        )

    return _execute


# ---------------------------------------------------------------------------
# Test 1: Unanimous candidates take the fast path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unanimous_takes_fast_path():
    """5 candidates with same execution result → fast_path, 0 API calls."""
    rows = [(1, "alice"), (2, "bob")]
    candidates = [
        _make_candidate(f"SELECT * FROM t -- variant {i}", f"gen_{i}", rows, confidence=0.8)
        for i in range(5)
    ]
    mock_client = _make_mock_client()

    with (
        patch("src.selection.adaptive_selector.execute_sql", side_effect=_patch_execute_sql(candidates)),
        patch("src.selection.adaptive_selector.get_client", return_value=mock_client),
    ):
        selector = AdaptiveSelector()
        result = await selector.select(
            candidates=candidates,
            question="Get all rows",
            evidence="",
            schemas=_make_schemas(),
            db_path="/fake/db.sqlite",
        )

    assert result.selection_method == "fast_path", (
        f"Expected fast_path, got {result.selection_method!r}"
    )
    # No API calls for fast path
    mock_client.generate.assert_not_called()
    assert result.cluster_count == 1


# ---------------------------------------------------------------------------
# Test 2: Fast path selects shortest SQL
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fast_path_selects_shortest_sql():
    """Among unanimous candidates, the shortest SQL is selected."""
    rows = [(1,)]
    candidates = [
        _make_candidate("SELECT id FROM t WHERE id = 1", "gen_a", rows, confidence=0.8),
        _make_candidate("SELECT id FROM t WHERE id=1", "gen_b", rows, confidence=0.9),  # shorter
        _make_candidate("SELECT id FROM t WHERE id =  1 AND 1=1", "gen_c", rows, confidence=1.0),  # longest
    ]
    mock_client = _make_mock_client()

    with (
        patch("src.selection.adaptive_selector.execute_sql", side_effect=_patch_execute_sql(candidates)),
        patch("src.selection.adaptive_selector.get_client", return_value=mock_client),
    ):
        selector = AdaptiveSelector()
        result = await selector.select(
            candidates=candidates,
            question="Get id 1",
            evidence="",
            schemas=_make_schemas(),
            db_path="/fake/db.sqlite",
        )

    assert result.selection_method == "fast_path"
    # The shortest SQL should be selected
    expected = min(candidates, key=lambda c: len(c.final_sql)).final_sql
    assert result.final_sql == expected, (
        f"Expected shortest SQL {expected!r}, got {result.final_sql!r}"
    )


# ---------------------------------------------------------------------------
# Test 3: Two distinct clusters trigger tournament
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_two_clusters_triggers_tournament():
    """2 distinct result clusters → selection_method='tournament'."""
    candidates = [
        _make_candidate("SELECT id FROM t", "reasoning_A1", [(1,), (2,)], confidence=0.9),
        _make_candidate("SELECT id FROM t WHERE id > 0", "standard_B1_s1", [(1,), (2,)], confidence=0.8),
        _make_candidate("SELECT id FROM t LIMIT 1", "standard_B1_s2", [(1,)], confidence=0.7),
    ]
    mock_client = _make_mock_client("A")  # Always pick A

    with (
        patch("src.selection.adaptive_selector.execute_sql", side_effect=_patch_execute_sql(candidates)),
        patch("src.selection.adaptive_selector.get_client", return_value=mock_client),
    ):
        selector = AdaptiveSelector()
        result = await selector.select(
            candidates=candidates,
            question="Get ids",
            evidence="",
            schemas=_make_schemas(),
            db_path="/fake/db.sqlite",
        )

    assert result.selection_method == "tournament", (
        f"Expected tournament, got {result.selection_method!r}"
    )
    assert result.cluster_count == 2


# ---------------------------------------------------------------------------
# Test 4: Tournament winner has most wins
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tournament_winner_has_most_wins():
    """In a 3-way tournament (C(3,2)=3 comparisons), winner wins 2/2 matches."""
    # 3 distinct clusters
    candidates = [
        _make_candidate("SELECT id FROM t", "reasoning_A1", [(1,), (2,)], confidence=0.9),
        _make_candidate("SELECT val FROM t", "standard_B1_s1", [("a",), ("b",)], confidence=0.7),
        _make_candidate("SELECT id, val FROM t", "icl_C1", [(1, "a"), (2, "b")], confidence=0.8),
    ]

    # Track comparison calls to control winner
    call_count = [0]

    async def controlled_generate(**kwargs):
        call_count[0] += 1
        # Always return A wins
        return LLMResponse(
            tool_inputs=[],
            text="Candidate A is clearly better here.\nFINAL: A",
            thinking=None,
            input_tokens=100,
            output_tokens=20,
        )

    mock_client = AsyncMock()
    mock_client.generate = controlled_generate

    with (
        patch("src.selection.adaptive_selector.execute_sql", side_effect=_patch_execute_sql(candidates)),
        patch("src.selection.adaptive_selector.get_client", return_value=mock_client),
    ):
        selector = AdaptiveSelector()
        result = await selector.select(
            candidates=candidates,
            question="Get data",
            evidence="",
            schemas=_make_schemas(),
            db_path="/fake/db.sqlite",
        )

    assert result.selection_method == "tournament"
    # 3 clusters → C(3,2)=3 pairwise comparisons
    assert call_count[0] == 3, f"Expected 3 comparisons, got {call_count[0]}"
    # The first representative (presented as "A" in pairs 0vs1, 0vs2) wins 2 times
    wins = result.tournament_wins
    max_wins = max(wins.values())
    assert max_wins == 2, f"Winner should have 2 wins, got wins={wins}"


# ---------------------------------------------------------------------------
# Test 5: Candidate reorganization order — largest cluster first
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_candidate_reorganization_order():
    """Representatives sorted by cluster size desc; largest cluster rep = A in first pair."""
    # Cluster 1: 3 candidates with rows [(1,), (2,)]
    # Cluster 2: 1 candidate with rows [(99,)]
    rows_large = [(1,), (2,)]
    rows_small = [(99,)]

    candidates = [
        _make_candidate("SELECT id FROM t", "reasoning_A1", rows_large, confidence=0.9),
        _make_candidate("SELECT id FROM t WHERE id > 0", "reasoning_A2", rows_large, confidence=0.8),
        _make_candidate("SELECT id FROM t LIMIT 5", "reasoning_A3", rows_large, confidence=0.7),
        _make_candidate("SELECT 99", "standard_B1_s1", rows_small, confidence=0.6),
    ]

    # Track which candidates are presented as A in each comparison
    a_candidates_in_pair: list[str] = []

    async def track_a_position(**kwargs):
        messages = kwargs.get("messages", [])
        if messages:
            content = messages[0].get("content", "")
            # Extract which generator is "Candidate A"
            import re
            m = re.search(r"Candidate A \(generated by ([^)]+)\)", content)
            if m:
                a_candidates_in_pair.append(m.group(1))
        return LLMResponse(
            tool_inputs=[],
            text="Candidate A wins this comparison.\nFINAL: A",
            thinking=None,
            input_tokens=100,
            output_tokens=20,
        )

    mock_client = AsyncMock()
    mock_client.generate = track_a_position

    with (
        patch("src.selection.adaptive_selector.execute_sql", side_effect=_patch_execute_sql(candidates)),
        patch("src.selection.adaptive_selector.get_client", return_value=mock_client),
    ):
        selector = AdaptiveSelector()
        result = await selector.select(
            candidates=candidates,
            question="Get data",
            evidence="",
            schemas=_make_schemas(),
            db_path="/fake/db.sqlite",
        )

    # The first comparison should have the large cluster rep as Candidate A
    # (i.e., representatives[0] is from the 3-candidate cluster)
    assert result.selection_method == "tournament"
    # With 2 clusters (1 with size 3, 1 with size 1), we have C(2,2)=1 comparison
    # The "A" candidate should be from the larger cluster
    assert len(a_candidates_in_pair) == 1
    # The reasoning_A* candidate (from large cluster) should be presented as A
    assert a_candidates_in_pair[0].startswith("reasoning"), (
        f"Expected reasoning rep as A, got {a_candidates_in_pair[0]!r}"
    )


# ---------------------------------------------------------------------------
# Test 6: Generator ranking as tiebreaker
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generator_ranking_as_tiebreaker():
    """Equal-size clusters: reasoning rep comes before standard rep in ordering."""
    # Two clusters of size 1 each
    candidates = [
        _make_candidate("SELECT id FROM t", "standard_B1_s1", [(1,)], confidence=0.8),
        _make_candidate("SELECT id FROM t WHERE 1=1", "reasoning_A1", [(1,)], confidence=0.8),
    ]
    # Note: rows are the same value but different SQL → they will be in same cluster
    # We need distinct rows to create different clusters
    candidates = [
        _make_candidate("SELECT id FROM t", "standard_B1_s1", [(1,)], confidence=0.8),
        _make_candidate("SELECT 999", "reasoning_A1", [(999,)], confidence=0.8),
    ]

    # Track which generator is presented as A in the comparison
    a_generators: list[str] = []

    async def track_a(**kwargs):
        messages = kwargs.get("messages", [])
        if messages:
            content = messages[0].get("content", "")
            import re
            m = re.search(r"Candidate A \(generated by ([^)]+)\)", content)
            if m:
                a_generators.append(m.group(1))
        return LLMResponse(
            tool_inputs=[],
            text="Candidate A wins this comparison.\nFINAL: A",
            thinking=None,
            input_tokens=100,
            output_tokens=20,
        )

    mock_client = AsyncMock()
    mock_client.generate = track_a

    with (
        patch("src.selection.adaptive_selector.execute_sql", side_effect=_patch_execute_sql(candidates)),
        patch("src.selection.adaptive_selector.get_client", return_value=mock_client),
    ):
        selector = AdaptiveSelector()
        result = await selector.select(
            candidates=candidates,
            question="Get data",
            evidence="",
            schemas=_make_schemas(),
            db_path="/fake/db.sqlite",
        )

    assert result.selection_method == "tournament"
    assert len(a_generators) >= 1
    # The reasoning generator should be presented first (as "A")
    assert a_generators[0].startswith("reasoning"), (
        f"reasoning_A1 should be presented as A, got {a_generators[0]!r}"
    )


# ---------------------------------------------------------------------------
# Test 7: Haiku model used for pairwise comparisons
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_haiku_used_for_pairwise():
    """Pairwise comparison API calls use settings.model_fast."""
    from src.config.settings import settings

    candidates = [
        _make_candidate("SELECT id FROM t", "reasoning_A1", [(1,)], confidence=0.9),
        _make_candidate("SELECT 2", "standard_B1_s1", [(2,)], confidence=0.8),
    ]
    mock_client = _make_mock_client("A")

    with (
        patch("src.selection.adaptive_selector.execute_sql", side_effect=_patch_execute_sql(candidates)),
        patch("src.selection.adaptive_selector.get_client", return_value=mock_client),
    ):
        selector = AdaptiveSelector()
        await selector.select(
            candidates=candidates,
            question="Get data",
            evidence="",
            schemas=_make_schemas(),
            db_path="/fake/db.sqlite",
        )

    # Verify all generate() calls used model_fast and free-text mode (tools=[])
    assert mock_client.generate.call_count >= 1
    for call_args in mock_client.generate.call_args_list:
        model_used = call_args.kwargs.get("model")
        assert model_used == settings.model_fast, (
            f"Expected model={settings.model_fast!r}, got {model_used!r}"
        )
        tools_used = call_args.kwargs.get("tools")
        assert tools_used == [], (
            f"Expected tools=[] (free-text mode), got {tools_used!r}"
        )


# ---------------------------------------------------------------------------
# Test 8: Pairwise comparison count — C(4,2) = 6
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pairwise_comparison_count():
    """With 4 clusters (4 representatives), exactly C(4,2)=6 API calls are made."""
    # 4 candidates with 4 distinct result sets
    candidates = [
        _make_candidate("SELECT 1", "reasoning_A1", [(1,)], confidence=0.9),
        _make_candidate("SELECT 2", "standard_B1_s1", [(2,)], confidence=0.8),
        _make_candidate("SELECT 3", "icl_C1", [(3,)], confidence=0.7),
        _make_candidate("SELECT 4", "standard_B1_s2", [(4,)], confidence=0.6),
    ]
    mock_client = _make_mock_client("A")

    with (
        patch("src.selection.adaptive_selector.execute_sql", side_effect=_patch_execute_sql(candidates)),
        patch("src.selection.adaptive_selector.get_client", return_value=mock_client),
    ):
        selector = AdaptiveSelector()
        result = await selector.select(
            candidates=candidates,
            question="Get single values",
            evidence="",
            schemas=_make_schemas(),
            db_path="/fake/db.sqlite",
        )

    assert result.selection_method == "tournament"
    assert mock_client.generate.call_count == 6, (
        f"Expected C(4,2)=6 API calls, got {mock_client.generate.call_count}"
    )


# ---------------------------------------------------------------------------
# Test 9: Fallback on no executable candidates
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fallback_on_no_executable_candidates():
    """When all candidates have confidence_score=0, fallback is used."""
    candidates = [
        _make_candidate(
            "SELECT bad syntax",
            "reasoning_A1",
            [],
            confidence=0.0,
            success=False,
            error="syntax error",
        ),
        _make_candidate(
            "SELECT also_bad",
            "standard_B1_s1",
            [],
            confidence=0.0,
            success=False,
            error="syntax error",
        ),
    ]
    mock_client = _make_mock_client()

    with (
        patch("src.selection.adaptive_selector.execute_sql", side_effect=_patch_execute_sql(candidates)),
        patch("src.selection.adaptive_selector.get_client", return_value=mock_client),
    ):
        selector = AdaptiveSelector()
        result = await selector.select(
            candidates=candidates,
            question="Get data",
            evidence="",
            schemas=_make_schemas(),
            db_path="/fake/db.sqlite",
        )

    assert result.selection_method == "fallback", (
        f"Expected fallback, got {result.selection_method!r}"
    )
    # No LLM calls for fallback
    mock_client.generate.assert_not_called()
    # Should return the candidate with highest confidence (both 0.0, so any)
    assert result.final_sql in {c.final_sql for c in candidates}


# ---------------------------------------------------------------------------
# Test 10: Free-text mode — no tool-use
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_free_text_mode_no_tools():
    """Pairwise comparison uses free-text mode (tools=[]), not tool-use."""
    candidates = [
        _make_candidate("SELECT id FROM t", "reasoning_A1", [(1,)], confidence=0.9),
        _make_candidate("SELECT 2", "standard_B1_s1", [(2,)], confidence=0.8),
    ]
    mock_client = _make_mock_client("A")

    with (
        patch("src.selection.adaptive_selector.execute_sql", side_effect=_patch_execute_sql(candidates)),
        patch("src.selection.adaptive_selector.get_client", return_value=mock_client),
    ):
        selector = AdaptiveSelector()
        await selector.select(
            candidates=candidates,
            question="Get data",
            evidence="",
            schemas=_make_schemas(),
            db_path="/fake/db.sqlite",
        )

    assert mock_client.generate.call_count >= 1
    for call_args in mock_client.generate.call_args_list:
        tools = call_args.kwargs.get("tools")
        assert tools == [], f"Expected tools=[] (free-text mode), got {tools!r}"
        tool_choice = call_args.kwargs.get("tool_choice_name")
        assert tool_choice is None, (
            f"Expected tool_choice_name=None in free-text mode, got {tool_choice!r}"
        )


# ---------------------------------------------------------------------------
# Test 11: Result equivalence clustering — different ordering → same cluster
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_result_equivalence_clustering():
    """Two candidates with same rows in different order → same cluster → fast path."""
    rows_a = [(1, "a"), (2, "b")]
    rows_b = [(2, "b"), (1, "a")]   # same rows, different order

    candidates = [
        _make_candidate("SELECT id, val FROM t ORDER BY id", "reasoning_A1", rows_a, confidence=0.9),
        _make_candidate("SELECT id, val FROM t ORDER BY val DESC", "standard_B1_s1", rows_b, confidence=0.8),
    ]
    mock_client = _make_mock_client()

    with (
        patch("src.selection.adaptive_selector.execute_sql", side_effect=_patch_execute_sql(candidates)),
        patch("src.selection.adaptive_selector.get_client", return_value=mock_client),
    ):
        selector = AdaptiveSelector()
        result = await selector.select(
            candidates=candidates,
            question="Get id and val",
            evidence="",
            schemas=_make_schemas(),
            db_path="/fake/db.sqlite",
        )

    # Both candidates produce the same result set → fast path
    assert result.selection_method == "fast_path", (
        f"Expected fast_path (same rows different order), got {result.selection_method!r}"
    )
    assert result.cluster_count == 1
    mock_client.generate.assert_not_called()


# ---------------------------------------------------------------------------
# Test 12: Empty result cluster deprioritized
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_result_cluster_deprioritized():
    """Non-empty result cluster comes before empty result cluster in ordering."""
    # One candidate returns rows; one returns empty
    candidates = [
        _make_candidate(
            "SELECT id FROM t WHERE id = 999",
            "standard_B1_s1",
            [],   # empty result
            confidence=0.4,
        ),
        _make_candidate(
            "SELECT id FROM t",
            "reasoning_A1",
            [(1,), (2,)],  # non-empty result
            confidence=0.9,
        ),
    ]
    # Empty candidate has confidence=0 but we need it to be in the executable pool
    # Let's set it to 0.4 but have is_empty=True
    # Actually, FixedCandidate.confidence_score > 0 check passes for both

    a_generators: list[str] = []

    async def track_a(**kwargs):
        messages = kwargs.get("messages", [])
        if messages:
            content = messages[0].get("content", "")
            import re
            m = re.search(r"Candidate A \(generated by ([^)]+)\)", content)
            if m:
                a_generators.append(m.group(1))
        return LLMResponse(
            tool_inputs=[],
            text="The non-empty result is clearly better.\nFINAL: A",
            thinking=None,
            input_tokens=100,
            output_tokens=20,
        )

    mock_client = AsyncMock()
    mock_client.generate = track_a

    with (
        patch("src.selection.adaptive_selector.execute_sql", side_effect=_patch_execute_sql(candidates)),
        patch("src.selection.adaptive_selector.get_client", return_value=mock_client),
    ):
        selector = AdaptiveSelector()
        result = await selector.select(
            candidates=candidates,
            question="Get all rows",
            evidence="",
            schemas=_make_schemas(),
            db_path="/fake/db.sqlite",
        )

    # Should go to tournament (2 different clusters: non-empty vs empty)
    assert result.selection_method == "tournament"
    # The non-empty cluster rep (reasoning_A1) should be presented as "A" (comes first)
    assert len(a_generators) == 1
    assert a_generators[0].startswith("reasoning"), (
        f"Non-empty cluster rep (reasoning_A1) should be A, got {a_generators[0]!r}"
    )
    # Final selection should prefer the non-empty result (it won the comparison)
    assert result.final_sql == "SELECT id FROM t", (
        f"Expected non-empty result winner, got {result.final_sql!r}"
    )


# ---------------------------------------------------------------------------
# Tests for _extract_column_names
# ---------------------------------------------------------------------------

def test_extract_column_names_simple():
    """Simple column list without aliases."""
    assert _extract_column_names("SELECT a, b FROM t") == ["a", "b"]


def test_extract_column_names_with_as_alias():
    """AS aliases are returned, not original expression tokens."""
    sql = "SELECT T1.molecule_id AS mid, T2.element AS elem FROM atom JOIN molecule"
    names = _extract_column_names(sql)
    assert names == ["mid", "elem"]


def test_extract_column_names_table_prefix_no_alias():
    """Table-prefixed columns without AS yield the column part only."""
    sql = "SELECT T1.col_a, T2.col_b FROM t1 JOIN t2 ON t1.id = t2.id"
    names = _extract_column_names(sql)
    assert names == ["col_a", "col_b"]


def test_extract_column_names_star_returns_empty():
    """SELECT * → empty list (caller uses positional fallback)."""
    assert _extract_column_names("SELECT * FROM t") == []
    assert _extract_column_names("SELECT DISTINCT * FROM t") == []


def test_extract_column_names_with_aggregate_alias():
    """Aggregate functions with AS alias are handled."""
    sql = "SELECT COUNT(DISTINCT id) AS cnt, name FROM t"
    names = _extract_column_names(sql)
    assert names == ["cnt", "name"]


def test_extract_column_names_cast_expression():
    """CAST with AS inside parens does not pollute alias detection."""
    sql = "SELECT CAST(x AS REAL) AS score, label FROM t"
    names = _extract_column_names(sql)
    assert names == ["score", "label"]


def test_extract_column_names_not_select_returns_empty():
    """Non-SELECT statement returns empty list."""
    assert _extract_column_names("UPDATE t SET x = 1") == []


# ---------------------------------------------------------------------------
# Tests for _format_execution_result
# ---------------------------------------------------------------------------

def test_format_execution_result_error():
    """Error results display the error message."""
    result = _make_exec_result([], success=False, error="no such table: foo")
    out = _format_execution_result(result, "SELECT * FROM foo")
    assert "no such table: foo" in out
    assert "Execution error" in out


def test_format_execution_result_empty():
    """Empty results display 0 rows."""
    result = _make_exec_result([], success=True, is_empty=True)
    out = _format_execution_result(result, "SELECT id FROM t WHERE id = 999")
    assert "generator(s) agreed" not in out
    assert "0 rows" in out


def test_format_execution_result_normal_includes_agreement():
    """Normal results include row count and column names (no agreement count)."""
    result = _make_exec_result([(1, "alice"), (2, "bob")])
    out = _format_execution_result(result, "SELECT id, name FROM t")
    assert "generator(s) agreed" not in out
    assert "Total rows: 2" in out
    assert "id" in out
    assert "name" in out
    assert "alice" in out
    assert "bob" in out


def test_format_execution_result_truncation_notice():
    """When rows exceed max_rows, a truncation notice is shown."""
    rows = [(i,) for i in range(10)]
    result = _make_exec_result(rows)
    out = _format_execution_result(result, "SELECT id FROM t", max_rows=5)
    assert "Total rows: 10" in out
    assert "showing first 5" in out
    # Only first 5 rows should appear as data
    assert "9" not in out  # row index 9 is beyond max_rows


def test_format_execution_result_no_truncation_when_within_limit():
    """When rows fit within max_rows, no truncation notice appears."""
    result = _make_exec_result([(1,), (2,), (3,)])
    out = _format_execution_result(result, "SELECT id FROM t", max_rows=5)
    assert "Total rows: 3" in out
    assert "showing first" not in out


def test_format_execution_result_null_values():
    """NULL values in rows are displayed as 'NULL'."""
    result = _make_exec_result([(1, None), (2, None)])
    out = _format_execution_result(result, "SELECT id, val FROM t")
    assert "NULL" in out


def test_format_execution_result_prompt_integration():
    """Verify that _format_execution_result renders column names and values."""
    result = _make_exec_result([(42,)])
    out = _format_execution_result(result, "SELECT COUNT(*) AS total FROM t")
    assert "generator(s) agreed" not in out
    assert "total" in out.lower() or "col_1" in out.lower()


# ---------------------------------------------------------------------------
# Tests for free-text pairwise comparison
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_final_anchor_parser_robustness():
    """'FINAL: B' at end is parsed correctly even when response starts with article 'a'."""
    candidates = [
        _make_candidate("SELECT id FROM t", "reasoning_A1", [(1,)], confidence=0.9),
        _make_candidate("SELECT val FROM t", "standard_B1_s1", [(2,)], confidence=0.8),
    ]

    async def respond_with_b(**kwargs):
        # The response begins with the English article "a" — old regex would return "A".
        # The FINAL: anchor must correctly identify B as the winner.
        return LLMResponse(
            tool_inputs=[],
            text=(
                "a careful look at the execution results shows that candidate B "
                "handles the aggregation correctly and avoids the extra JOIN.\n"
                "FINAL: B"
            ),
            thinking=None,
            input_tokens=100,
            output_tokens=40,
        )

    mock_client = AsyncMock()
    mock_client.generate = respond_with_b

    with (
        patch("src.selection.adaptive_selector.execute_sql", side_effect=_patch_execute_sql(candidates)),
        patch("src.selection.adaptive_selector.get_client", return_value=mock_client),
    ):
        selector = AdaptiveSelector()
        result = await selector.select(
            candidates=candidates,
            question="Get value",
            evidence="",
            schemas=_make_schemas(),
            db_path="/fake/db.sqlite",
        )

    assert result.final_sql == "SELECT val FROM t", (
        f"Expected B's SQL (article-'a' bug fix), got {result.final_sql!r}"
    )


@pytest.mark.asyncio
async def test_llm_error_defaults_to_a():
    """When generate() raises LLMError, the comparison defaults to candidate A."""
    from src.llm.base import LLMError

    candidates = [
        _make_candidate("SELECT id FROM t", "reasoning_A1", [(1,)], confidence=0.9),
        _make_candidate("SELECT val FROM t", "standard_B1_s1", [(2,)], confidence=0.8),
    ]

    async def always_fails(**kwargs):
        raise LLMError("network timeout")

    mock_client = AsyncMock()
    mock_client.generate = always_fails

    with (
        patch("src.selection.adaptive_selector.execute_sql", side_effect=_patch_execute_sql(candidates)),
        patch("src.selection.adaptive_selector.get_client", return_value=mock_client),
    ):
        selector = AdaptiveSelector()
        result = await selector.select(
            candidates=candidates,
            question="Get value",
            evidence="",
            schemas=_make_schemas(),
            db_path="/fake/db.sqlite",
        )

    # LLMError → winner_letter stays "A" → representative A's SQL
    assert result.final_sql == "SELECT id FROM t", (
        f"Expected A's SQL (LLMError default), got {result.final_sql!r}"
    )


@pytest.mark.asyncio
async def test_single_api_call_per_pair():
    """Each pairwise comparison makes exactly 1 API call (no multi-tier fallback)."""
    # 2 clusters → C(2,2) = 1 comparison → exactly 1 API call
    candidates = [
        _make_candidate("SELECT id FROM t", "reasoning_A1", [(1,)], confidence=0.9),
        _make_candidate("SELECT val FROM t", "standard_B1_s1", [(2,)], confidence=0.8),
    ]
    mock_client = _make_mock_client("A")

    with (
        patch("src.selection.adaptive_selector.execute_sql", side_effect=_patch_execute_sql(candidates)),
        patch("src.selection.adaptive_selector.get_client", return_value=mock_client),
    ):
        selector = AdaptiveSelector()
        result = await selector.select(
            candidates=candidates,
            question="Get data",
            evidence="",
            schemas=_make_schemas(),
            db_path="/fake/db.sqlite",
        )

    assert result.selection_method == "tournament"
    assert mock_client.generate.call_count == 1, (
        f"Expected exactly 1 API call for 1 pair, got {mock_client.generate.call_count}"
    )
