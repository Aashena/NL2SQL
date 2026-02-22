"""
Unit tests for src/generation/reasoning_generator.py

Tests the Op 7A Reasoning Generator with mocked LLM calls.
No live API calls are made.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, call, patch

from src.llm.base import LLMResponse, ThinkingConfig
from src.indexing.lsh_index import CellMatch
from src.grounding.context_grounder import GroundingContext
from src.schema_linking.schema_linker import LinkedSchemas
from src.generation.base_generator import SQLCandidate
from src.generation.reasoning_generator import ReasoningGenerator
from src.config.settings import settings


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def schemas():
    """Minimal LinkedSchemas with 2 tables for testing."""
    return LinkedSchemas(
        s1_fields=[("students", "id"), ("students", "name"), ("students", "gpa")],
        s2_fields=[
            ("students", "id"), ("students", "name"), ("students", "gpa"),
            ("schools", "id"), ("schools", "name"),
        ],
        s1_ddl="CREATE TABLE students (id INTEGER PRIMARY KEY, name TEXT, gpa REAL);",
        s2_ddl=(
            "CREATE TABLE students (id INTEGER PRIMARY KEY, name TEXT, gpa REAL);\n"
            "CREATE TABLE schools (id INTEGER PRIMARY KEY, name TEXT);"
        ),
        s1_markdown=(
            "## Table: students\n"
            "| Column | Type | Description |\n"
            "|--------|------|-------------|\n"
            "| id | INTEGER (PK) | student id |"
        ),
        s2_markdown=(
            "## Table: students\n"
            "| Column | Type | Description |\n"
            "|--------|------|-------------|\n"
            "| id | INTEGER (PK) | student id |"
        ),
        selection_reasoning="Selected id, name, gpa for student query",
    )


@pytest.fixture
def grounding():
    """GroundingContext with one matched cell."""
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
    sql_text: str = "```sql\nSELECT * FROM students WHERE gpa > 3.5\n```",
    thinking_text: str = "Let me analyze the schema... The question asks about students with GPA above 3.5.",
) -> LLMResponse:
    """Build a mock LLMResponse simulating extended thinking output."""
    return LLMResponse(
        tool_inputs=[],
        text=sql_text,
        thinking=thinking_text,
        input_tokens=100,
        output_tokens=50,
    )


def make_mock_client(response: LLMResponse | None = None) -> AsyncMock:
    """Create a mock LLM client that returns the given response for every call."""
    if response is None:
        response = make_mock_response()
    mock_client = AsyncMock()
    mock_client.generate = AsyncMock(return_value=response)
    return mock_client


# ---------------------------------------------------------------------------
# Test 1: Returns exactly 4 SQLCandidate objects
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generates_4_candidates(schemas, grounding):
    """Returns exactly 4 SQLCandidate objects with no error flags when LLM returns valid SQL."""
    mock_client = make_mock_client()

    with patch("src.generation.reasoning_generator.get_client", return_value=mock_client):
        generator = ReasoningGenerator()
        candidates = await generator.generate(
            question="List all students with GPA above 3.5",
            evidence="",
            schemas=schemas,
            grounding=grounding,
        )

    assert len(candidates) == 4, f"Expected 4 candidates, got {len(candidates)}"
    for c in candidates:
        assert isinstance(c, SQLCandidate)
        assert not c.error_flag, f"Unexpected error_flag=True on candidate {c.generator_id}"


# ---------------------------------------------------------------------------
# Test 2: Candidates have different schemas (2 S1, 2 S2)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_candidates_have_different_schemas(schemas, grounding):
    """2 candidates have schema_used='s1' and 2 have schema_used='s2'."""
    mock_client = make_mock_client()

    with patch("src.generation.reasoning_generator.get_client", return_value=mock_client):
        generator = ReasoningGenerator()
        candidates = await generator.generate(
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
# Test 3: SQL extracted from markdown code fences
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sql_extracted_from_response(schemas, grounding):
    """When mock response.text has ```sql fences, candidate.sql has them stripped."""
    mock_response = make_mock_response(
        sql_text="```sql\nSELECT id FROM students\n```"
    )
    mock_client = make_mock_client(mock_response)

    with patch("src.generation.reasoning_generator.get_client", return_value=mock_client):
        generator = ReasoningGenerator()
        candidates = await generator.generate(
            question="List all student IDs",
            evidence="",
            schemas=schemas,
            grounding=grounding,
        )

    for c in candidates:
        assert c.sql == "SELECT id FROM students", (
            f"Expected clean SQL without fences, got: {c.sql!r}"
        )
        # No trailing semicolons either
        assert not c.sql.endswith(";"), f"SQL should not end with semicolon: {c.sql!r}"


# ---------------------------------------------------------------------------
# Test 4: Extended thinking is enabled in the API call
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extended_thinking_enabled_in_api_call(schemas, grounding):
    """Verify generate() is called with a ThinkingConfig where enabled=True."""
    mock_client = make_mock_client()

    with patch("src.generation.reasoning_generator.get_client", return_value=mock_client):
        generator = ReasoningGenerator()
        await generator.generate(
            question="List all students with GPA above 3.5",
            evidence="",
            schemas=schemas,
            grounding=grounding,
        )

    # All 4 calls should have thinking enabled
    for i, call_args in enumerate(mock_client.generate.call_args_list):
        thinking_arg = call_args.kwargs.get("thinking")
        assert thinking_arg is not None, f"Call {i}: 'thinking' kwarg missing"
        assert isinstance(thinking_arg, ThinkingConfig), (
            f"Call {i}: expected ThinkingConfig, got {type(thinking_arg)}"
        )
        assert thinking_arg.enabled is True, (
            f"Call {i}: ThinkingConfig.enabled should be True, got {thinking_arg.enabled}"
        )


# ---------------------------------------------------------------------------
# Test 5: Generator IDs start with "reasoning_"
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generator_id_label(schemas, grounding):
    """All returned candidates have generator_id starting with 'reasoning_'."""
    mock_client = make_mock_client()

    with patch("src.generation.reasoning_generator.get_client", return_value=mock_client):
        generator = ReasoningGenerator()
        candidates = await generator.generate(
            question="List all students with GPA above 3.5",
            evidence="",
            schemas=schemas,
            grounding=grounding,
        )

    for c in candidates:
        assert c.generator_id.startswith("reasoning_"), (
            f"generator_id should start with 'reasoning_', got: {c.generator_id!r}"
        )

    # Verify the specific IDs A1-A4
    ids = {c.generator_id for c in candidates}
    expected_ids = {"reasoning_A1", "reasoning_A2", "reasoning_A3", "reasoning_A4"}
    assert ids == expected_ids, f"Expected IDs {expected_ids}, got {ids}"


# ---------------------------------------------------------------------------
# Test 6: Model used is settings.model_reasoning
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_model_used(schemas, grounding):
    """The generate() call uses settings.model_reasoning as the model."""
    mock_client = make_mock_client()

    with patch("src.generation.reasoning_generator.get_client", return_value=mock_client):
        generator = ReasoningGenerator()
        await generator.generate(
            question="List all students with GPA above 3.5",
            evidence="",
            schemas=schemas,
            grounding=grounding,
        )

    for i, call_args in enumerate(mock_client.generate.call_args_list):
        model_used = call_args.kwargs.get("model")
        assert model_used == settings.model_reasoning, (
            f"Call {i}: expected model={settings.model_reasoning!r}, got {model_used!r}"
        )


# ---------------------------------------------------------------------------
# Test 7: Cell matches appear in the prompt
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cell_matches_in_prompt(schemas, grounding):
    """The messages passed to generate() contain the matched cell value 'Alameda'."""
    mock_client = make_mock_client()

    with patch("src.generation.reasoning_generator.get_client", return_value=mock_client):
        generator = ReasoningGenerator()
        await generator.generate(
            question="List all students from Alameda",
            evidence="",
            schemas=schemas,
            grounding=grounding,
        )

    for i, call_args in enumerate(mock_client.generate.call_args_list):
        messages = call_args.kwargs.get("messages", [])
        # Concatenate all message content to search for the matched value
        all_content = " ".join(
            str(msg.get("content", "")) for msg in messages
        )
        assert "Alameda" in all_content, (
            f"Call {i}: matched cell value 'Alameda' not found in messages. "
            f"Content: {all_content[:200]!r}"
        )


# ---------------------------------------------------------------------------
# Test 8: Empty SQL handling (response.text is None)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_sql_handling(schemas, grounding):
    """When mock response.text=None, the candidate has error_flag=True and sql=''."""
    mock_response = LLMResponse(
        tool_inputs=[],
        text=None,
        thinking="Some thinking trace",
        input_tokens=100,
        output_tokens=10,
    )
    mock_client = make_mock_client(mock_response)

    with patch("src.generation.reasoning_generator.get_client", return_value=mock_client):
        generator = ReasoningGenerator()
        candidates = await generator.generate(
            question="List all students",
            evidence="",
            schemas=schemas,
            grounding=grounding,
        )

    assert len(candidates) == 4
    for c in candidates:
        assert c.error_flag is True, f"Expected error_flag=True, got {c.error_flag}"
        assert c.sql == "", f"Expected sql='', got {c.sql!r}"


# ---------------------------------------------------------------------------
# Test 9: Reasoning trace is captured from response.thinking
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reasoning_trace_captured(schemas, grounding):
    """The reasoning_trace field is populated from response.thinking."""
    thinking_text = "Let me analyze the schema... The question asks about students with GPA above 3.5."
    mock_response = make_mock_response(thinking_text=thinking_text)
    mock_client = make_mock_client(mock_response)

    with patch("src.generation.reasoning_generator.get_client", return_value=mock_client):
        generator = ReasoningGenerator()
        candidates = await generator.generate(
            question="List all students with GPA above 3.5",
            evidence="",
            schemas=schemas,
            grounding=grounding,
        )

    for c in candidates:
        assert c.reasoning_trace == thinking_text, (
            f"Expected reasoning_trace={thinking_text!r}, got {c.reasoning_trace!r}"
        )


# ---------------------------------------------------------------------------
# Test 10: Concurrent calls complete without errors
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrent_calls_possible(schemas, grounding):
    """All 4 generate() calls complete without errors (uses asyncio.gather internally)."""
    import asyncio

    mock_client = make_mock_client()
    completed_calls: list[int] = []

    original_generate = mock_client.generate

    async def tracked_generate(**kwargs):
        # Small sleep to ensure concurrency is actually exercised
        await asyncio.sleep(0)
        result = await original_generate(**kwargs)
        completed_calls.append(len(completed_calls) + 1)
        return result

    mock_client.generate = tracked_generate

    with patch("src.generation.reasoning_generator.get_client", return_value=mock_client):
        generator = ReasoningGenerator()
        candidates = await generator.generate(
            question="List all students with GPA above 3.5",
            evidence="",
            schemas=schemas,
            grounding=grounding,
        )

    assert len(candidates) == 4, f"Expected 4 candidates, got {len(candidates)}"
    assert len(completed_calls) == 4, f"Expected 4 completed calls, got {len(completed_calls)}"
    # All candidates should be error-free
    for c in candidates:
        assert not c.error_flag, f"Unexpected error_flag=True on candidate {c.generator_id}"
