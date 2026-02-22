"""
Integration tests for all 3 generators running together (mocked API).
Test 5 (live oracle upper bound) is saved for Checkpoint D.
"""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, patch

from src.llm.base import LLMResponse
from src.indexing.lsh_index import CellMatch
from src.indexing.example_store import ExampleEntry
from src.grounding.context_grounder import GroundingContext
from src.schema_linking.schema_linker import LinkedSchemas
from src.generation.base_generator import SQLCandidate
from src.generation.reasoning_generator import ReasoningGenerator
from src.generation.standard_generator import StandardAndComplexGenerator
from src.generation.icl_generator import ICLGenerator


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def schemas():
    """Minimal LinkedSchemas with 2 tables for testing all generators."""
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
    """GroundingContext with sample cell matches and few-shot examples."""
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
        few_shot_examples=[
            ExampleEntry(
                question_id=1,
                db_id="other_db_1",
                question="How many students have GPA above 3.0?",
                evidence="",
                sql="SELECT COUNT(*) FROM students WHERE gpa > 3.0",
                skeleton="How many students have [ENTITY] above [NUM]?",
                similarity_score=0.88,
            ),
            ExampleEntry(
                question_id=2,
                db_id="other_db_2",
                question="List all schools in the county",
                evidence="",
                sql="SELECT name FROM schools",
                skeleton="List all schools in the [ENTITY]",
                similarity_score=0.82,
            ),
        ],
    )


def make_mock_client() -> AsyncMock:
    """Create a single mock client returned for all generators."""
    mock_response = LLMResponse(
        tool_inputs=[],
        text="SELECT * FROM students WHERE gpa > 3.5",
        thinking="I think this is the right approach",
        input_tokens=50,
        output_tokens=20,
    )
    mock_client = AsyncMock()
    mock_client.generate = AsyncMock(return_value=mock_response)
    return mock_client


# ---------------------------------------------------------------------------
# Test 1: All generators run on a real question — 11 total candidates
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_all_generators_run_on_real_question(schemas, grounding):
    """Run all 3 generators concurrently; verify 10-11 total candidates (4+4+3=11)."""
    mock_client = make_mock_client()

    with patch("src.generation.reasoning_generator.get_client", return_value=mock_client), \
         patch("src.generation.standard_generator.get_client", return_value=mock_client), \
         patch("src.generation.icl_generator.get_client", return_value=mock_client):

        reasoning_gen = ReasoningGenerator()
        standard_gen = StandardAndComplexGenerator()
        icl_gen = ICLGenerator()

        all_batches = await asyncio.gather(
            reasoning_gen.generate(
                question="List all students with GPA above 3.5 from Alameda",
                evidence="GPA means grade point average",
                schemas=schemas,
                grounding=grounding,
            ),
            standard_gen.generate(
                question="List all students with GPA above 3.5 from Alameda",
                evidence="GPA means grade point average",
                schemas=schemas,
                grounding=grounding,
            ),
            icl_gen.generate(
                question="List all students with GPA above 3.5 from Alameda",
                evidence="GPA means grade point average",
                schemas=schemas,
                grounding=grounding,
            ),
        )

        candidates = [c for batch in all_batches for c in batch]

    # 4 (reasoning) + 4 (standard+complex) + 3 (icl) = 11
    assert 10 <= len(candidates) <= 11, (
        f"Expected 10-11 total candidates, got {len(candidates)}"
    )
    assert len(candidates) == 11, (
        f"Expected exactly 11 candidates (4+4+3), got {len(candidates)}"
    )


# ---------------------------------------------------------------------------
# Test 2: All generator_id labels are unique
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generator_ids_are_unique(schemas, grounding):
    """All candidates have unique generator_id labels."""
    mock_client = make_mock_client()

    with patch("src.generation.reasoning_generator.get_client", return_value=mock_client), \
         patch("src.generation.standard_generator.get_client", return_value=mock_client), \
         patch("src.generation.icl_generator.get_client", return_value=mock_client):

        reasoning_gen = ReasoningGenerator()
        standard_gen = StandardAndComplexGenerator()
        icl_gen = ICLGenerator()

        all_batches = await asyncio.gather(
            reasoning_gen.generate(
                question="List all students with GPA above 3.5",
                evidence="",
                schemas=schemas,
                grounding=grounding,
            ),
            standard_gen.generate(
                question="List all students with GPA above 3.5",
                evidence="",
                schemas=schemas,
                grounding=grounding,
            ),
            icl_gen.generate(
                question="List all students with GPA above 3.5",
                evidence="",
                schemas=schemas,
                grounding=grounding,
            ),
        )

        candidates = [c for batch in all_batches for c in batch]

    ids = [c.generator_id for c in candidates]
    unique_ids = set(ids)

    assert len(ids) == len(unique_ids), (
        f"Duplicate generator_id labels found. "
        f"All IDs: {ids}"
    )


# ---------------------------------------------------------------------------
# Test 3: Candidates use both schema scopes (at least 3 S1 and 3 S2)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_candidates_use_both_schema_scopes(schemas, grounding):
    """At least 3 candidates use S1 and at least 3 use S2."""
    mock_client = make_mock_client()

    with patch("src.generation.reasoning_generator.get_client", return_value=mock_client), \
         patch("src.generation.standard_generator.get_client", return_value=mock_client), \
         patch("src.generation.icl_generator.get_client", return_value=mock_client):

        reasoning_gen = ReasoningGenerator()
        standard_gen = StandardAndComplexGenerator()
        icl_gen = ICLGenerator()

        all_batches = await asyncio.gather(
            reasoning_gen.generate(
                question="List all students with GPA above 3.5",
                evidence="",
                schemas=schemas,
                grounding=grounding,
            ),
            standard_gen.generate(
                question="List all students with GPA above 3.5",
                evidence="",
                schemas=schemas,
                grounding=grounding,
            ),
            icl_gen.generate(
                question="List all students with GPA above 3.5",
                evidence="",
                schemas=schemas,
                grounding=grounding,
            ),
        )

        candidates = [c for batch in all_batches for c in batch]

    s1_count = sum(1 for c in candidates if c.schema_used == "s1")
    s2_count = sum(1 for c in candidates if c.schema_used == "s2")

    assert s1_count >= 3, (
        f"Expected at least 3 S1 candidates, got {s1_count}. "
        f"IDs: {[(c.generator_id, c.schema_used) for c in candidates]}"
    )
    assert s2_count >= 3, (
        f"Expected at least 3 S2 candidates, got {s2_count}. "
        f"IDs: {[(c.generator_id, c.schema_used) for c in candidates]}"
    )


# ---------------------------------------------------------------------------
# Test 4: Generators run concurrently
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generators_run_concurrently(schemas, grounding):
    """Launch all 3 generate() coroutines with asyncio.gather(); verify all complete."""
    mock_client = make_mock_client()
    completed_generators: list[str] = []

    original_generate = mock_client.generate

    async def tracked_generate(**kwargs):
        await asyncio.sleep(0)  # Yield to event loop to enable concurrency
        return await original_generate(**kwargs)

    mock_client.generate = tracked_generate

    with patch("src.generation.reasoning_generator.get_client", return_value=mock_client), \
         patch("src.generation.standard_generator.get_client", return_value=mock_client), \
         patch("src.generation.icl_generator.get_client", return_value=mock_client):

        reasoning_gen = ReasoningGenerator()
        standard_gen = StandardAndComplexGenerator()
        icl_gen = ICLGenerator()

        # Track which generator finishes
        async def wrap_reasoning():
            result = await reasoning_gen.generate(
                question="List all students with GPA above 3.5",
                evidence="",
                schemas=schemas,
                grounding=grounding,
            )
            completed_generators.append("reasoning")
            return result

        async def wrap_standard():
            result = await standard_gen.generate(
                question="List all students with GPA above 3.5",
                evidence="",
                schemas=schemas,
                grounding=grounding,
            )
            completed_generators.append("standard")
            return result

        async def wrap_icl():
            result = await icl_gen.generate(
                question="List all students with GPA above 3.5",
                evidence="",
                schemas=schemas,
                grounding=grounding,
            )
            completed_generators.append("icl")
            return result

        all_batches = await asyncio.gather(
            wrap_reasoning(),
            wrap_standard(),
            wrap_icl(),
        )

        candidates = [c for batch in all_batches for c in batch]

    # All 3 generators should have completed
    assert set(completed_generators) == {"reasoning", "standard", "icl"}, (
        f"Not all generators completed. Completed: {completed_generators}"
    )

    # All 11 candidates should be present
    assert len(candidates) == 11, (
        f"Expected 11 candidates from 3 generators, got {len(candidates)}"
    )
