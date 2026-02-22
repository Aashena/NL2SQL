"""
Unit tests for src/generation/standard_generator.py

Tests the Op 7B Standard + Complex SQL Generators with mocked LLM calls.
No live API calls are made.
"""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, patch

from src.llm.base import LLMResponse
from src.indexing.lsh_index import CellMatch
from src.grounding.context_grounder import GroundingContext
from src.schema_linking.schema_linker import LinkedSchemas
from src.generation.base_generator import SQLCandidate
from src.generation.standard_generator import StandardAndComplexGenerator
from src.config.settings import settings


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def schemas():
    """Minimal LinkedSchemas with Markdown tables for testing."""
    s1_md = (
        "## Table: students\n"
        "| Column | Type | Description |\n"
        "|--------|------|-------------|\n"
        "| id | INTEGER (PK) | student id |\n"
        "| gpa | REAL | grade point average |"
    )
    s2_md = (
        "## Table: students\n"
        "| Column | Type | Description |\n"
        "|--------|------|-------------|\n"
        "| id | INTEGER (PK) | student id |\n"
        "| gpa | REAL | grade point average |\n\n"
        "## Table: schools\n"
        "| Column | Type | Description |\n"
        "|--------|------|-------------|\n"
        "| id | INTEGER (PK) | school id |\n"
        "| name | TEXT | school name |"
    )
    return LinkedSchemas(
        s1_fields=[("students", "id"), ("students", "gpa")],
        s2_fields=[
            ("students", "id"), ("students", "gpa"),
            ("schools", "id"), ("schools", "name"),
        ],
        s1_ddl="CREATE TABLE students (id INTEGER PRIMARY KEY, gpa REAL);",
        s2_ddl=(
            "CREATE TABLE students (id INTEGER PRIMARY KEY, gpa REAL);\n"
            "CREATE TABLE schools (id INTEGER PRIMARY KEY, name TEXT);"
        ),
        s1_markdown=s1_md,
        s2_markdown=s2_md,
        selection_reasoning="Selected gpa for student query",
    )


@pytest.fixture
def grounding():
    """GroundingContext with one matched cell and no few-shot examples."""
    return GroundingContext(
        matched_cells=[
            CellMatch(
                table="schools",
                column="name",
                matched_value="Alameda",
                similarity_score=0.95,
                exact_match=True,
            )
        ],
        schema_hints=[("schools", "name")],
        few_shot_examples=[],
    )


def make_mock_response(
    sql_text: str = "SELECT * FROM students WHERE gpa > 3.5",
) -> LLMResponse:
    """Build a mock LLMResponse with plain SQL text."""
    return LLMResponse(
        tool_inputs=[],
        text=sql_text,
        thinking=None,
        input_tokens=100,
        output_tokens=50,
    )


def make_mock_client(response: LLMResponse | None = None) -> AsyncMock:
    """Create a mock LLM client returning the given response for every call."""
    if response is None:
        response = make_mock_response()
    mock_client = AsyncMock()
    mock_client.generate = AsyncMock(return_value=response)
    return mock_client


# ---------------------------------------------------------------------------
# Test 1: Returns exactly 4 candidates
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generates_4_candidates_total(schemas, grounding):
    """Returns exactly 4 SQLCandidate objects total."""
    mock_client = make_mock_client()

    with patch("src.generation.standard_generator.get_client", return_value=mock_client):
        gen = StandardAndComplexGenerator()
        candidates = await gen.generate(
            question="List all students with GPA above 3.5",
            evidence="",
            schemas=schemas,
            grounding=grounding,
        )

    assert len(candidates) == 4, f"Expected 4 candidates, got {len(candidates)}"
    for c in candidates:
        assert isinstance(c, SQLCandidate)


# ---------------------------------------------------------------------------
# Test 2: B1 uses fast model
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_b1_uses_fast_model(schemas, grounding):
    """B1 candidates are generated using settings.model_fast."""
    mock_client = make_mock_client()

    with patch("src.generation.standard_generator.get_client", return_value=mock_client):
        gen = StandardAndComplexGenerator()
        candidates = await gen.generate(
            question="List all students with GPA above 3.5",
            evidence="",
            schemas=schemas,
            grounding=grounding,
        )

    call_args_list = mock_client.generate.call_args_list
    assert len(call_args_list) == 4

    # B1 candidates are the first 2 calls (standard_B1_s1 and standard_B1_s2)
    b1_candidates = [c for c in candidates if "B1" in c.generator_id]
    assert len(b1_candidates) == 2, f"Expected 2 B1 candidates, got {len(b1_candidates)}"

    # Find the calls made with the fast model
    fast_model_calls = [
        ca for ca in call_args_list
        if ca.kwargs.get("model") == settings.model_fast
    ]
    assert len(fast_model_calls) == 2, (
        f"Expected 2 calls with model_fast={settings.model_fast!r}, "
        f"got {len(fast_model_calls)}"
    )


# ---------------------------------------------------------------------------
# Test 3: B2 uses powerful model
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_b2_uses_powerful_model(schemas, grounding):
    """B2 candidates are generated using settings.model_powerful."""
    mock_client = make_mock_client()

    with patch("src.generation.standard_generator.get_client", return_value=mock_client):
        gen = StandardAndComplexGenerator()
        candidates = await gen.generate(
            question="List all students with GPA above 3.5",
            evidence="",
            schemas=schemas,
            grounding=grounding,
        )

    call_args_list = mock_client.generate.call_args_list

    b2_candidates = [c for c in candidates if "B2" in c.generator_id]
    assert len(b2_candidates) == 2, f"Expected 2 B2 candidates, got {len(b2_candidates)}"

    # Find the calls made with the powerful model
    powerful_model_calls = [
        ca for ca in call_args_list
        if ca.kwargs.get("model") == settings.model_powerful
    ]
    assert len(powerful_model_calls) == 2, (
        f"Expected 2 calls with model_powerful={settings.model_powerful!r}, "
        f"got {len(powerful_model_calls)}"
    )


# ---------------------------------------------------------------------------
# Test 4: Generator IDs labeled correctly
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generator_ids_labeled(schemas, grounding):
    """B1 candidates have 'B1' in generator_id, B2 candidates have 'B2'."""
    mock_client = make_mock_client()

    with patch("src.generation.standard_generator.get_client", return_value=mock_client):
        gen = StandardAndComplexGenerator()
        candidates = await gen.generate(
            question="List all students with GPA above 3.5",
            evidence="",
            schemas=schemas,
            grounding=grounding,
        )

    ids = {c.generator_id for c in candidates}
    expected_ids = {
        "standard_B1_s1", "standard_B1_s2",
        "complex_B2_s1", "complex_B2_s2",
    }
    assert ids == expected_ids, f"Expected IDs {expected_ids}, got {ids}"

    b1_ids = [c.generator_id for c in candidates if "B1" in c.generator_id]
    b2_ids = [c.generator_id for c in candidates if "B2" in c.generator_id]
    assert len(b1_ids) == 2
    assert len(b2_ids) == 2


# ---------------------------------------------------------------------------
# Test 5: Markdown schema in system prompt
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_markdown_schema_in_system_prompt(schemas, grounding):
    """System prompt passed to generate() contains Markdown table syntax (pipe char)."""
    mock_client = make_mock_client()

    with patch("src.generation.standard_generator.get_client", return_value=mock_client):
        gen = StandardAndComplexGenerator()
        await gen.generate(
            question="List all students with GPA above 3.5",
            evidence="",
            schemas=schemas,
            grounding=grounding,
        )

    call_args_list = mock_client.generate.call_args_list
    for i, ca in enumerate(call_args_list):
        system_blocks = ca.kwargs.get("system", [])
        system_text = " ".join(block.text for block in system_blocks)
        assert "|" in system_text, (
            f"Call {i}: system prompt does not contain '|' (Markdown table syntax). "
            f"System text: {system_text[:200]!r}"
        )


# ---------------------------------------------------------------------------
# Test 6: Schema scope varies — S1 and S2 per generator type
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_schema_scope_varies(schemas, grounding):
    """One S1 and one S2 candidate per generator type (2 × S1, 2 × S2 total)."""
    mock_client = make_mock_client()

    with patch("src.generation.standard_generator.get_client", return_value=mock_client):
        gen = StandardAndComplexGenerator()
        candidates = await gen.generate(
            question="List all students with GPA above 3.5",
            evidence="",
            schemas=schemas,
            grounding=grounding,
        )

    s1_candidates = [c for c in candidates if c.schema_used == "s1"]
    s2_candidates = [c for c in candidates if c.schema_used == "s2"]
    assert len(s1_candidates) == 2, f"Expected 2 S1 candidates, got {len(s1_candidates)}"
    assert len(s2_candidates) == 2, f"Expected 2 S2 candidates, got {len(s2_candidates)}"


# ---------------------------------------------------------------------------
# Test 7: No extended thinking
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_extended_thinking(schemas, grounding):
    """The 'thinking' kwarg passed to generate() is None (no extended thinking)."""
    mock_client = make_mock_client()

    with patch("src.generation.standard_generator.get_client", return_value=mock_client):
        gen = StandardAndComplexGenerator()
        await gen.generate(
            question="List all students with GPA above 3.5",
            evidence="",
            schemas=schemas,
            grounding=grounding,
        )

    for i, ca in enumerate(mock_client.generate.call_args_list):
        thinking_arg = ca.kwargs.get("thinking")
        assert thinking_arg is None, (
            f"Call {i}: expected thinking=None, got {thinking_arg!r}"
        )


# ---------------------------------------------------------------------------
# Test 8: SQL clean extracted (no markdown code fences)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sql_clean_extracted(schemas, grounding):
    """Returned candidates' sql field has no markdown code fences."""
    fenced_sql = "```sql\nSELECT * FROM students WHERE gpa > 3.5\n```"
    mock_response = make_mock_response(sql_text=fenced_sql)
    mock_client = make_mock_client(mock_response)

    with patch("src.generation.standard_generator.get_client", return_value=mock_client):
        gen = StandardAndComplexGenerator()
        candidates = await gen.generate(
            question="List all students with GPA above 3.5",
            evidence="",
            schemas=schemas,
            grounding=grounding,
        )

    for c in candidates:
        assert "```" not in c.sql, (
            f"Candidate {c.generator_id}: SQL still contains code fences: {c.sql!r}"
        )
        assert c.sql == "SELECT * FROM students WHERE gpa > 3.5", (
            f"Expected clean SQL, got: {c.sql!r}"
        )


# ---------------------------------------------------------------------------
# Test 9: B2 system prompt mentions CTE or window functions
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_b2_system_prompt_mentions_cte(schemas, grounding):
    """B2's system prompt text contains 'CTE' or 'window'."""
    mock_client = make_mock_client()

    with patch("src.generation.standard_generator.get_client", return_value=mock_client):
        gen = StandardAndComplexGenerator()
        await gen.generate(
            question="List all students with GPA above 3.5",
            evidence="",
            schemas=schemas,
            grounding=grounding,
        )

    call_args_list = mock_client.generate.call_args_list
    # B2 calls should be the last 2 (indices 2 and 3), using model_powerful
    b2_calls = [
        ca for ca in call_args_list
        if ca.kwargs.get("model") == settings.model_powerful
    ]
    assert len(b2_calls) == 2, f"Expected 2 B2 calls, got {len(b2_calls)}"

    for i, ca in enumerate(b2_calls):
        system_blocks = ca.kwargs.get("system", [])
        system_text = " ".join(block.text for block in system_blocks).lower()
        assert "cte" in system_text or "window" in system_text, (
            f"B2 call {i}: system prompt should mention 'CTE' or 'window'. "
            f"System text: {system_text[:300]!r}"
        )


# ---------------------------------------------------------------------------
# Test 10: Parallel generation (all 4 calls complete)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_parallel_generation(schemas, grounding):
    """All 4 generate() calls complete without race conditions (asyncio.gather test)."""
    completed_calls: list[str] = []

    base_response = make_mock_response()

    async def tracked_generate(**kwargs):
        # Yield control to exercise concurrent scheduling
        await asyncio.sleep(0)
        model = kwargs.get("model", "unknown")
        completed_calls.append(model)
        return base_response

    mock_client = AsyncMock()
    mock_client.generate = tracked_generate

    with patch("src.generation.standard_generator.get_client", return_value=mock_client):
        gen = StandardAndComplexGenerator()
        candidates = await gen.generate(
            question="List all students with GPA above 3.5",
            evidence="",
            schemas=schemas,
            grounding=grounding,
        )

    assert len(candidates) == 4, f"Expected 4 candidates, got {len(candidates)}"
    assert len(completed_calls) == 4, (
        f"Expected 4 completed API calls, got {len(completed_calls)}"
    )
    # All candidates should have no error
    for c in candidates:
        assert not c.error_flag, f"Unexpected error_flag=True on {c.generator_id}"
