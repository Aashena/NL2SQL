"""
Unit tests for src/generation/icl_generator.py

Tests the Op 7C ICL Generator with mocked LLM calls.
No live API calls are made.
"""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, patch

from src.llm.base import LLMResponse, CacheableText
from src.indexing.lsh_index import CellMatch
from src.indexing.example_store import ExampleEntry
from src.grounding.context_grounder import GroundingContext
from src.schema_linking.schema_linker import LinkedSchemas
from src.generation.base_generator import SQLCandidate
from src.generation.icl_generator import ICLGenerator
from src.config.settings import settings


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_example(n: int, sql: str = "SELECT * FROM t") -> ExampleEntry:
    """Build a minimal ExampleEntry for testing."""
    return ExampleEntry(
        question_id=n,
        db_id=f"other_db_{n}",
        question=f"Sample question {n}",
        evidence="",
        sql=sql,
        skeleton=f"Sample question [NUM]",
        similarity_score=0.9,
    )


@pytest.fixture
def few_shot_examples():
    """Three sample few-shot examples."""
    return [_make_example(i) for i in range(1, 4)]


@pytest.fixture
def schemas():
    """Minimal LinkedSchemas for testing."""
    s2_md = (
        "## Table: students\n"
        "| Column | Type | Description |\n"
        "|--------|------|-------------|\n"
        "| id | INTEGER (PK) | student id |\n"
        "| gpa | REAL | grade point average |"
    )
    return LinkedSchemas(
        s1_fields=[("students", "id"), ("students", "gpa")],
        s2_fields=[("students", "id"), ("students", "gpa")],
        s1_ddl="CREATE TABLE students (id INTEGER PRIMARY KEY, gpa REAL);",
        s2_ddl="CREATE TABLE students (id INTEGER PRIMARY KEY, gpa REAL);",
        s1_markdown=s2_md,
        s2_markdown=s2_md,
        selection_reasoning="Selected gpa for student query",
    )


@pytest.fixture
def grounding(few_shot_examples):
    """GroundingContext with a few-shot examples."""
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
        schema_hints=[],
        few_shot_examples=few_shot_examples,
    )


@pytest.fixture
def grounding_no_examples():
    """GroundingContext with no few-shot examples."""
    return GroundingContext(
        matched_cells=[],
        schema_hints=[],
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
# Test 1: Returns exactly 3 candidates
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generates_3_candidates(schemas, grounding):
    """Returns exactly 3 SQLCandidate objects."""
    mock_client = make_mock_client()

    with patch("src.generation.icl_generator.get_client", return_value=mock_client):
        gen = ICLGenerator()
        candidates = await gen.generate(
            question="List all students with GPA above 3.5",
            evidence="",
            schemas=schemas,
            grounding=grounding,
        )

    assert len(candidates) == 3, f"Expected 3 candidates, got {len(candidates)}"
    for c in candidates:
        assert isinstance(c, SQLCandidate)
        assert not c.error_flag, f"Unexpected error_flag=True on {c.generator_id}"


# ---------------------------------------------------------------------------
# Test 2: Few-shot examples appear in system prompt
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_few_shot_examples_in_prompt(schemas, grounding):
    """The formatted examples text appears in the system prompt blocks."""
    mock_client = make_mock_client()

    with patch("src.generation.icl_generator.get_client", return_value=mock_client):
        gen = ICLGenerator()
        await gen.generate(
            question="List all students with GPA above 3.5",
            evidence="",
            schemas=schemas,
            grounding=grounding,
        )

    call_args_list = mock_client.generate.call_args_list
    assert len(call_args_list) == 3

    for i, ca in enumerate(call_args_list):
        system_blocks = ca.kwargs.get("system", [])
        all_system_text = " ".join(b.text for b in system_blocks)
        # The examples should contain "## Example" blocks
        assert "## Example" in all_system_text, (
            f"Call {i}: few-shot example block not found in system prompt. "
            f"System text: {all_system_text[:300]!r}"
        )
        # The question from example 1 should appear
        assert "Sample question 1" in all_system_text, (
            f"Call {i}: example question text not found in system prompt."
        )


# ---------------------------------------------------------------------------
# Test 3: S2 schema used always
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_s2_schema_used_always(schemas, grounding):
    """All candidates have schema_used='s2'."""
    mock_client = make_mock_client()

    with patch("src.generation.icl_generator.get_client", return_value=mock_client):
        gen = ICLGenerator()
        candidates = await gen.generate(
            question="List all students with GPA above 3.5",
            evidence="",
            schemas=schemas,
            grounding=grounding,
        )

    for c in candidates:
        assert c.schema_used == "s2", (
            f"Candidate {c.generator_id}: expected schema_used='s2', got {c.schema_used!r}"
        )


# ---------------------------------------------------------------------------
# Test 4: Powerful model used
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_powerful_model_used(schemas, grounding):
    """All ICL calls use settings.model_powerful."""
    mock_client = make_mock_client()

    with patch("src.generation.icl_generator.get_client", return_value=mock_client):
        gen = ICLGenerator()
        await gen.generate(
            question="List all students with GPA above 3.5",
            evidence="",
            schemas=schemas,
            grounding=grounding,
        )

    for i, ca in enumerate(mock_client.generate.call_args_list):
        model_used = ca.kwargs.get("model")
        assert model_used == settings.model_powerful, (
            f"Call {i}: expected model={settings.model_powerful!r}, got {model_used!r}"
        )


# ---------------------------------------------------------------------------
# Test 5: Prompt caching applied to examples block
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_prompt_caching_applied_to_examples(schemas, grounding):
    """At least one CacheableText in system list has cache=True and contains the examples text."""
    mock_client = make_mock_client()

    with patch("src.generation.icl_generator.get_client", return_value=mock_client):
        gen = ICLGenerator()
        await gen.generate(
            question="List all students with GPA above 3.5",
            evidence="",
            schemas=schemas,
            grounding=grounding,
        )

    for i, ca in enumerate(mock_client.generate.call_args_list):
        system_blocks = ca.kwargs.get("system", [])
        cacheable_with_examples = [
            b for b in system_blocks
            if isinstance(b, CacheableText) and b.cache and "## Example" in b.text
        ]
        assert len(cacheable_with_examples) >= 1, (
            f"Call {i}: no CacheableText with cache=True containing examples found. "
            f"System blocks: {[(type(b).__name__, getattr(b, 'cache', None), b.text[:50]) for b in system_blocks]}"
        )


# ---------------------------------------------------------------------------
# Test 6: Token limit guard — trim to 6 examples when > 6000 tokens
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_token_limit_guard(schemas):
    """When examples exceed 6000 token estimate, only first 6 are used."""
    # Create 8 examples each with a ~3500-char SQL to force >6000 token estimate
    # 8 examples × (~3500 SQL chars + ~50 overhead) / 4 chars per token ≈ 7100 tokens > 6000
    large_sql = "SELECT " + ", ".join([f"col_{i}" for i in range(400)]) + " FROM big_table"
    large_examples = [_make_example(i, sql=large_sql) for i in range(1, 9)]

    grounding_large = GroundingContext(
        matched_cells=[],
        schema_hints=[],
        few_shot_examples=large_examples,
    )

    mock_client = make_mock_client()

    with patch("src.generation.icl_generator.get_client", return_value=mock_client):
        gen = ICLGenerator()
        await gen.generate(
            question="List all students with GPA above 3.5",
            evidence="",
            schemas=schemas,
            grounding=grounding_large,
        )

    # Check that at most 6 examples appear in the system prompt
    call_args_list = mock_client.generate.call_args_list
    assert len(call_args_list) >= 1

    for i, ca in enumerate(call_args_list):
        system_blocks = ca.kwargs.get("system", [])
        all_system_text = " ".join(b.text for b in system_blocks)

        # Count "## Example N" occurrences
        import re
        example_headers = re.findall(r"## Example \d+", all_system_text)
        assert len(example_headers) <= 6, (
            f"Call {i}: expected at most 6 examples, found {len(example_headers)}"
        )
        # Should have at least 1 but not 8 (the trimming must have happened)
        assert len(example_headers) == 6, (
            f"Call {i}: expected exactly 6 examples after trimming, found {len(example_headers)}"
        )


# ---------------------------------------------------------------------------
# Test 7: Generator ID labels
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generator_id_label(schemas, grounding):
    """All candidates have generator_id starting with 'icl_C'."""
    mock_client = make_mock_client()

    with patch("src.generation.icl_generator.get_client", return_value=mock_client):
        gen = ICLGenerator()
        candidates = await gen.generate(
            question="List all students with GPA above 3.5",
            evidence="",
            schemas=schemas,
            grounding=grounding,
        )

    expected_ids = {"icl_C1", "icl_C2", "icl_C3"}
    actual_ids = {c.generator_id for c in candidates}
    assert actual_ids == expected_ids, f"Expected IDs {expected_ids}, got {actual_ids}"

    for c in candidates:
        assert c.generator_id.startswith("icl_C"), (
            f"generator_id should start with 'icl_C', got: {c.generator_id!r}"
        )


# ---------------------------------------------------------------------------
# Test 8: C2 user prompt contains CoT instruction
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_c2_prompt_cot_instruction(schemas, grounding):
    """C2's user message contains 'identify' or 'tables and joins'."""
    mock_client = make_mock_client()

    with patch("src.generation.icl_generator.get_client", return_value=mock_client):
        gen = ICLGenerator()
        await gen.generate(
            question="List all students with GPA above 3.5",
            evidence="",
            schemas=schemas,
            grounding=grounding,
        )

    call_args_list = mock_client.generate.call_args_list
    assert len(call_args_list) == 3

    # Find C2 call — it's the second call (index 1)
    c2_call = call_args_list[1]
    messages = c2_call.kwargs.get("messages", [])
    user_content = " ".join(
        str(msg.get("content", "")) for msg in messages if msg.get("role") == "user"
    ).lower()

    assert "identify" in user_content or "tables and joins" in user_content, (
        f"C2 user message should contain 'identify' or 'tables and joins'. "
        f"Content: {user_content[:300]!r}"
    )


# ---------------------------------------------------------------------------
# Test 9: Empty examples fallback — no crash
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_examples_fallback(schemas, grounding_no_examples):
    """grounding.few_shot_examples=[] → generates 3 candidates without crashing."""
    mock_client = make_mock_client()

    with patch("src.generation.icl_generator.get_client", return_value=mock_client):
        gen = ICLGenerator()
        candidates = await gen.generate(
            question="List all students with GPA above 3.5",
            evidence="",
            schemas=schemas,
            grounding=grounding_no_examples,
        )

    assert len(candidates) == 3, f"Expected 3 candidates, got {len(candidates)}"
    for c in candidates:
        assert isinstance(c, SQLCandidate)
        assert not c.error_flag, f"Unexpected error_flag=True on {c.generator_id}"


# ---------------------------------------------------------------------------
# Test 10: C1 and C2 called with different user message content
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_different_prompts_for_c1_c2(schemas, grounding):
    """C1 and C2 are called with different user message content."""
    mock_client = make_mock_client()

    with patch("src.generation.icl_generator.get_client", return_value=mock_client):
        gen = ICLGenerator()
        await gen.generate(
            question="List all students with GPA above 3.5",
            evidence="",
            schemas=schemas,
            grounding=grounding,
        )

    call_args_list = mock_client.generate.call_args_list
    assert len(call_args_list) == 3

    def get_user_content(ca) -> str:
        messages = ca.kwargs.get("messages", [])
        return " ".join(
            str(msg.get("content", "")) for msg in messages if msg.get("role") == "user"
        )

    c1_content = get_user_content(call_args_list[0])
    c2_content = get_user_content(call_args_list[1])

    assert c1_content != c2_content, (
        f"C1 and C2 should have different user message content. "
        f"C1: {c1_content[:100]!r}\nC2: {c2_content[:100]!r}"
    )
