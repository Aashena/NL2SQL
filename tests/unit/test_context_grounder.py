"""
Tests for src/grounding/context_grounder.py

10 tests covering keyword extraction, cell matching, deduplication,
few-shot retrieval, error handling, and latency.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.settings import settings
from src.grounding.context_grounder import GroundingContext, ground_context
from src.indexing.example_store import ExampleEntry
from src.indexing.lsh_index import CellMatch
from src.llm.base import LLMResponse


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def _make_llm_response(literals: list[str], schema_refs: list[str]) -> LLMResponse:
    """Return an LLMResponse whose tool_inputs matches the extract_grounding schema."""
    return LLMResponse(
        tool_inputs=[{"literals": literals, "schema_references": schema_refs}],
        text=None,
        thinking=None,
        input_tokens=50,
        output_tokens=20,
    )


def _make_cell_match(table: str, column: str, value: str, score: float = 0.9) -> CellMatch:
    return CellMatch(
        table=table,
        column=column,
        matched_value=value,
        similarity_score=score,
        exact_match=(score == 1.0),
    )


def _make_example_entry(db_id: str = "other_db") -> ExampleEntry:
    return ExampleEntry(
        question_id=1,
        db_id=db_id,
        question="How many schools are there?",
        evidence="",
        sql="SELECT COUNT(*) FROM schools",
        skeleton="How many [ENTITY] are there?",
        similarity_score=0.85,
    )


def _mock_client(literals: list[str], schema_refs: list[str]):
    """Return a mock LLM client whose generate() returns a canned response."""
    mock_client = MagicMock()
    mock_client.generate = AsyncMock(
        return_value=_make_llm_response(literals, schema_refs)
    )
    return mock_client


def _mock_lsh(return_values: list[CellMatch] = None):
    mock = MagicMock()
    mock.query = MagicMock(return_value=return_values or [])
    return mock


def _mock_example_store(return_values: list[ExampleEntry] = None):
    mock = MagicMock()
    mock.query = MagicMock(return_value=return_values or [])
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_keyword_extraction_from_question():
    """Keyword extraction should surface literals like "Alameda County"."""
    mock_client = _mock_client(
        literals=["Alameda County"],
        schema_refs=[],
    )
    lsh = _mock_lsh([_make_cell_match("schools", "county", "Alameda County")])
    store = _mock_example_store()

    with patch("src.grounding.context_grounder.get_client", return_value=mock_client):
        ctx = await ground_context(
            question="How many schools are in Alameda County?",
            evidence="",
            db_id="california_schools",
            lsh_index=lsh,
            example_store=store,
        )

    assert len(ctx.matched_cells) >= 1
    values = [m.matched_value for m in ctx.matched_cells]
    assert "Alameda County" in values


@pytest.mark.asyncio
async def test_schema_reference_from_evidence():
    """Schema references extracted from evidence appear in schema_hints."""
    mock_client = _mock_client(
        literals=[],
        schema_refs=["Free Meal Count", "Enrollment"],
    )
    lsh = _mock_lsh()
    store = _mock_example_store()

    with patch("src.grounding.context_grounder.get_client", return_value=mock_client):
        ctx = await ground_context(
            question="What is the eligible free rate?",
            evidence="use Free Meal Count and Enrollment",
            db_id="california_schools",
            lsh_index=lsh,
            example_store=store,
        )

    assert "Free Meal Count" in ctx.schema_hints
    assert "Enrollment" in ctx.schema_hints


@pytest.mark.asyncio
async def test_cell_match_format():
    """Returned CellMatch objects must have non-empty table, column, matched_value."""
    mock_client = _mock_client(literals=["Alameda"], schema_refs=[])
    cell = _make_cell_match("schools", "county_name", "Alameda", score=0.95)
    lsh = _mock_lsh([cell])
    store = _mock_example_store()

    with patch("src.grounding.context_grounder.get_client", return_value=mock_client):
        ctx = await ground_context(
            question="Schools in Alameda?",
            evidence="",
            db_id="california_schools",
            lsh_index=lsh,
            example_store=store,
        )

    assert len(ctx.matched_cells) >= 1
    for match in ctx.matched_cells:
        assert match.table, "table must be non-empty"
        assert match.column, "column must be non-empty"
        assert match.matched_value, "matched_value must be non-empty"


@pytest.mark.asyncio
async def test_no_keywords_returns_empty_cells():
    """If extraction returns no literals, LSH is not called and matched_cells is empty."""
    mock_client = _mock_client(literals=[], schema_refs=[])
    lsh = _mock_lsh()
    store = _mock_example_store()

    with patch("src.grounding.context_grounder.get_client", return_value=mock_client):
        ctx = await ground_context(
            question="What is the count?",
            evidence="",
            db_id="california_schools",
            lsh_index=lsh,
            example_store=store,
        )

    assert ctx.matched_cells == []
    lsh.query.assert_not_called()


@pytest.mark.asyncio
async def test_few_shot_examples_count():
    """Returns up to top_k=8 examples."""
    examples = [_make_example_entry() for _ in range(8)]
    mock_client = _mock_client(literals=[], schema_refs=[])
    lsh = _mock_lsh()
    store = _mock_example_store(return_values=examples)

    with patch("src.grounding.context_grounder.get_client", return_value=mock_client):
        ctx = await ground_context(
            question="Count schools.",
            evidence="",
            db_id="california_schools",
            lsh_index=lsh,
            example_store=store,
        )

    assert len(ctx.few_shot_examples) == 8


@pytest.mark.asyncio
async def test_db_id_excluded_from_examples():
    """
    ExampleStore excludes same-db examples. Verify none of the returned
    examples have db_id matching the query db_id.
    """
    target_db = "california_schools"
    # ExampleStore.query already filters; mock it returning only other-db examples
    examples = [_make_example_entry(db_id="other_db") for _ in range(3)]
    mock_client = _mock_client(literals=[], schema_refs=[])
    lsh = _mock_lsh()
    store = _mock_example_store(return_values=examples)

    with patch("src.grounding.context_grounder.get_client", return_value=mock_client):
        ctx = await ground_context(
            question="Count schools.",
            evidence="",
            db_id=target_db,
            lsh_index=lsh,
            example_store=store,
        )

    for ex in ctx.few_shot_examples:
        assert ex.db_id != target_db


@pytest.mark.asyncio
async def test_duplicate_cell_matches_deduplicated():
    """Two different literals matching the same (table, column) → only one CellMatch."""
    # Both literals resolve to the same (table, column) but with different values/scores
    cell1 = _make_cell_match("schools", "county", "Alameda", score=0.9)
    cell2 = _make_cell_match("schools", "county", "Alameda County", score=0.7)

    # LSH returns cell1 for the first literal, cell2 for the second
    mock_client = _mock_client(
        literals=["Alameda", "Alameda County"],
        schema_refs=[],
    )
    lsh = MagicMock()
    # literals=["Alameda", "Alameda County"]; the grounder also queries individual
    # words of multi-word literals: "Alameda" and "County" from "Alameda County"
    # → 4 total LSH calls: "Alameda", "Alameda County", "Alameda", "County"
    lsh.query = MagicMock(side_effect=[[cell1], [cell2], [cell1], []])
    store = _mock_example_store()

    with patch("src.grounding.context_grounder.get_client", return_value=mock_client):
        ctx = await ground_context(
            question="Schools in Alameda County?",
            evidence="",
            db_id="california_schools",
            lsh_index=lsh,
            example_store=store,
        )

    # Should only have one entry for (schools, county)
    school_county_matches = [
        m for m in ctx.matched_cells
        if m.table == "schools" and m.column == "county"
    ]
    assert len(school_county_matches) == 1
    # The one with higher similarity (cell1, score=0.9) should be kept
    assert school_county_matches[0].similarity_score == 0.9


@pytest.mark.asyncio
async def test_haiku_model_used():
    """Keyword extraction call must use settings.model_fast."""
    mock_client = _mock_client(literals=["Alameda"], schema_refs=[])
    lsh = _mock_lsh([_make_cell_match("schools", "county", "Alameda")])
    store = _mock_example_store()

    with patch("src.grounding.context_grounder.get_client", return_value=mock_client):
        await ground_context(
            question="Schools in Alameda?",
            evidence="",
            db_id="california_schools",
            lsh_index=lsh,
            example_store=store,
        )

    # Verify that generate() was called with model=settings.model_fast
    mock_client.generate.assert_called_once()
    call_kwargs = mock_client.generate.call_args.kwargs
    assert call_kwargs["model"] == settings.model_fast


@pytest.mark.asyncio
async def test_evidence_none_handled():
    """Empty/None evidence must still produce a valid GroundingContext without errors."""
    mock_client = _mock_client(literals=[], schema_refs=[])
    lsh = _mock_lsh()
    store = _mock_example_store()

    with patch("src.grounding.context_grounder.get_client", return_value=mock_client):
        # Test with empty string evidence
        ctx_empty = await ground_context(
            question="What is the count?",
            evidence="",
            db_id="california_schools",
            lsh_index=lsh,
            example_store=store,
        )
        assert isinstance(ctx_empty, GroundingContext)

        # Test with None evidence
        ctx_none = await ground_context(
            question="What is the count?",
            evidence=None,
            db_id="california_schools",
            lsh_index=lsh,
            example_store=store,
        )
        assert isinstance(ctx_none, GroundingContext)


@pytest.mark.asyncio
async def test_grounding_latency():
    """With mocked API and index, grounding completes in under 2 seconds."""
    mock_client = _mock_client(
        literals=["Alameda County"],
        schema_refs=["county"],
    )
    cell = _make_cell_match("schools", "county", "Alameda County")
    lsh = _mock_lsh([cell])
    examples = [_make_example_entry() for _ in range(5)]
    store = _mock_example_store(return_values=examples)

    with patch("src.grounding.context_grounder.get_client", return_value=mock_client):
        start = time.monotonic()
        ctx = await ground_context(
            question="How many schools in Alameda County?",
            evidence="",
            db_id="california_schools",
            lsh_index=lsh,
            example_store=store,
        )
        elapsed = time.monotonic() - start

    assert elapsed < 2.0, f"Grounding took {elapsed:.2f}s, expected < 2s"
    assert isinstance(ctx, GroundingContext)
