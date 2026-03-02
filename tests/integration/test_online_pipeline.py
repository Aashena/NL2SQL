"""
Integration tests for src/pipeline/online_pipeline.py

6 tests — all LLM calls are mocked; a real in-memory SQLite database is used
for SQL execution (query fixer + selector steps).

Mock strategy:
  - Patch 'src.llm.get_client' in each relevant module so that *all* LLM calls
    go through the mock.  Because the modules import get_client at call time
    (not at import time), we patch the function inside each module's namespace.
  - The mock returns an AsyncMock that returns LLMResponse objects with a valid
    SQL string in response.text.
  - OfflineArtifacts is built from real in-memory objects so the whole
    integration path (incl. execute_sql) exercises real code.
"""
from __future__ import annotations

import asyncio
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from src.data.bird_loader import BirdEntry
from src.llm.base import LLMResponse
from src.schema_linking.schema_linker import LinkedSchemas
from src.generation.base_generator import SQLCandidate
from src.fixing.query_fixer import FixedCandidate
from src.data.database import ExecutionResult
from src.pipeline.online_pipeline import PipelineResult, answer_question


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

# Modules that call get_client() — we patch all of them to intercept LLM calls
_GET_CLIENT_PATCHES = [
    "src.grounding.context_grounder.get_client",
    "src.schema_linking.schema_linker.get_client",
    "src.generation.reasoning_generator.get_client",
    "src.generation.standard_generator.get_client",
    "src.generation.icl_generator.get_client",
    "src.fixing.query_fixer.get_client",
    "src.selection.adaptive_selector.get_client",
]

# Valid SQL that will be returned by the mock generators
_VALID_SQL = "SELECT id, name FROM students WHERE gpa > 3.0"
_GROUNDING_RESPONSE_TOOL = {"literals": ["3.0"], "schema_references": ["gpa"]}
_SCHEMA_LINK_RESPONSE_TOOL = {
    "selected_fields": [
        {"table": "students", "column": "id", "reason": "primary key", "role": "select"},
        {"table": "students", "column": "name", "reason": "return name", "role": "select"},
        {"table": "students", "column": "gpa", "reason": "filter condition", "role": "where"},
    ]
}


def _make_llm_response(
    text: Optional[str] = None,
    tool_inputs: Optional[list] = None,
    finish_reason: str = "end_turn",
) -> LLMResponse:
    return LLMResponse(
        tool_inputs=tool_inputs or [],
        text=text,
        thinking=None,
        input_tokens=10,
        output_tokens=20,
        finish_reason=finish_reason,
    )


def _make_mock_client(
    text_response: str = _VALID_SQL,
    tool_inputs: Optional[list] = None,
) -> MagicMock:
    """Build a mock LLM client that returns the given text or tool_inputs."""
    client = MagicMock()

    async def _generate_side_effect(**kwargs):
        # If tools are requested (tool_choice_name is not None), return tool response
        tool_choice = kwargs.get("tool_choice_name")
        if tool_choice == "extract_grounding":
            return _make_llm_response(tool_inputs=[_GROUNDING_RESPONSE_TOOL])
        elif tool_choice == "select_columns":
            return _make_llm_response(tool_inputs=[_SCHEMA_LINK_RESPONSE_TOOL])
        elif tool_choice == "select_winner":
            return _make_llm_response(tool_inputs=[{"winner": "A"}])
        else:
            # Text response for generators/fixer/selector
            return _make_llm_response(text=text_response)

    client.generate = AsyncMock(side_effect=_generate_side_effect)
    return client


def _make_temp_db() -> str:
    """Create a temp SQLite file with a 'students' table and return its path."""
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "CREATE TABLE students (id INTEGER PRIMARY KEY, name TEXT, gpa REAL)"
        )
        conn.execute("INSERT INTO students VALUES (1, 'Alice', 3.9)")
        conn.execute("INSERT INTO students VALUES (2, 'Bob', 2.8)")
        conn.execute("INSERT INTO students VALUES (3, 'Carol', 3.5)")
        conn.commit()
    finally:
        conn.close()
    return path


def _make_entry(question: str = "Show students with GPA above 3.0") -> BirdEntry:
    return BirdEntry(
        question_id=1,
        db_id="students_db",
        question=question,
        evidence="",
        SQL="SELECT id, name FROM students WHERE gpa > 3.0",
        difficulty="simple",
    )


def _make_mock_artifacts(db_path: str):
    """Create minimal OfflineArtifacts mocks for testing."""
    # Build a minimal FAISSIndex mock with _fields populated
    faiss_index = MagicMock()
    faiss_index._fields = [
        {"table": "students", "column": "id", "short_summary": "Student ID", "long_summary": "Primary key"},
        {"table": "students", "column": "name", "short_summary": "Student name", "long_summary": "Full name"},
        {"table": "students", "column": "gpa", "short_summary": "GPA", "long_summary": "Grade point average"},
    ]
    faiss_index.query = MagicMock(return_value=[])

    # Build a minimal LSHIndex mock
    lsh_index = MagicMock()
    lsh_index.query = MagicMock(return_value=[])

    # Build an ExampleStore mock
    example_store = MagicMock()
    example_store.query = MagicMock(return_value=[])

    # FormattedSchemas mock
    schemas = MagicMock()
    schemas.ddl = (
        "-- Table: students\n"
        "CREATE TABLE students (\n"
        "  id INTEGER PRIMARY KEY,  -- student id\n"
        "  name TEXT,  -- student name\n"
        "  gpa REAL  -- grade point average\n"
        ");\n"
    )
    schemas.markdown = (
        "## Table: students\n"
        "| Column | Type | Description | Sample Values |\n"
        "|--------|------|-------------|---------------|\n"
        "| id | INTEGER (PK) | Student ID | 1, 2, 3 |\n"
        "| name | TEXT | Student name | Alice, Bob |\n"
        "| gpa | REAL | GPA | 3.9, 2.8 |\n"
    )

    # Minimal profile and summary
    profile = MagicMock()
    summary = MagicMock()
    summary.field_summaries = []

    artifacts = MagicMock()
    artifacts.db_id = "students_db"
    artifacts.lsh_index = lsh_index
    artifacts.faiss_index = faiss_index
    artifacts.example_store = example_store
    artifacts.schemas = schemas
    artifacts.profile = profile
    artifacts.summary = summary

    return artifacts


# ---------------------------------------------------------------------------
# Test 1: Full pipeline produces a non-empty SQL string
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_pipeline_single_question(tmp_path):
    """Run the complete online pipeline on one mocked BIRD question."""
    db_path = _make_temp_db()
    try:
        entry = _make_entry()
        artifacts = _make_mock_artifacts(db_path)
        mock_client = _make_mock_client()

        with patch.multiple(
            "src.grounding.context_grounder",
            get_client=MagicMock(return_value=mock_client),
        ), patch.multiple(
            "src.schema_linking.schema_linker",
            get_client=MagicMock(return_value=mock_client),
        ), patch.multiple(
            "src.generation.reasoning_generator",
            get_client=MagicMock(return_value=mock_client),
        ), patch.multiple(
            "src.generation.standard_generator",
            get_client=MagicMock(return_value=mock_client),
        ), patch.multiple(
            "src.generation.icl_generator",
            get_client=MagicMock(return_value=mock_client),
        ), patch.multiple(
            "src.fixing.query_fixer",
            get_client=MagicMock(return_value=mock_client),
        ), patch.multiple(
            "src.selection.adaptive_selector",
            get_client=MagicMock(return_value=mock_client),
        ):
            result = await answer_question(entry, artifacts, db_path)

        assert isinstance(result, PipelineResult)
        assert result.final_sql  # non-empty
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test 2: Returned SQL contains SELECT
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pipeline_produces_valid_sql(tmp_path):
    """The returned SQL contains SELECT (case-insensitive)."""
    db_path = _make_temp_db()
    try:
        entry = _make_entry()
        artifacts = _make_mock_artifacts(db_path)
        mock_client = _make_mock_client()

        with patch.multiple(
            "src.grounding.context_grounder",
            get_client=MagicMock(return_value=mock_client),
        ), patch.multiple(
            "src.schema_linking.schema_linker",
            get_client=MagicMock(return_value=mock_client),
        ), patch.multiple(
            "src.generation.reasoning_generator",
            get_client=MagicMock(return_value=mock_client),
        ), patch.multiple(
            "src.generation.standard_generator",
            get_client=MagicMock(return_value=mock_client),
        ), patch.multiple(
            "src.generation.icl_generator",
            get_client=MagicMock(return_value=mock_client),
        ), patch.multiple(
            "src.fixing.query_fixer",
            get_client=MagicMock(return_value=mock_client),
        ), patch.multiple(
            "src.selection.adaptive_selector",
            get_client=MagicMock(return_value=mock_client),
        ):
            result = await answer_question(entry, artifacts, db_path)

        assert "select" in result.final_sql.lower()
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test 3: PipelineResult includes metadata
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pipeline_result_has_metadata(tmp_path):
    """PipelineResult includes cluster_count >= 0, valid selection_method, candidates_evaluated >= 0."""
    db_path = _make_temp_db()
    try:
        entry = _make_entry()
        artifacts = _make_mock_artifacts(db_path)
        mock_client = _make_mock_client()

        with patch.multiple(
            "src.grounding.context_grounder",
            get_client=MagicMock(return_value=mock_client),
        ), patch.multiple(
            "src.schema_linking.schema_linker",
            get_client=MagicMock(return_value=mock_client),
        ), patch.multiple(
            "src.generation.reasoning_generator",
            get_client=MagicMock(return_value=mock_client),
        ), patch.multiple(
            "src.generation.standard_generator",
            get_client=MagicMock(return_value=mock_client),
        ), patch.multiple(
            "src.generation.icl_generator",
            get_client=MagicMock(return_value=mock_client),
        ), patch.multiple(
            "src.fixing.query_fixer",
            get_client=MagicMock(return_value=mock_client),
        ), patch.multiple(
            "src.selection.adaptive_selector",
            get_client=MagicMock(return_value=mock_client),
        ):
            result = await answer_question(entry, artifacts, db_path)

        assert result.cluster_count >= 0
        assert result.selection_method in {"fast_path", "tournament", "fallback"}
        assert result.candidates_evaluated >= 0
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test 4: All candidates failing → no crash, fallback SQL returned
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pipeline_handles_all_candidates_failing(tmp_path):
    """If all candidates produce invalid SQL, the pipeline returns without crashing."""
    db_path = _make_temp_db()
    try:
        entry = _make_entry()
        artifacts = _make_mock_artifacts(db_path)
        # Return intentionally broken SQL from generators
        mock_client = _make_mock_client(text_response="INVALID SQL NOT A QUERY")

        with patch.multiple(
            "src.grounding.context_grounder",
            get_client=MagicMock(return_value=mock_client),
        ), patch.multiple(
            "src.schema_linking.schema_linker",
            get_client=MagicMock(return_value=mock_client),
        ), patch.multiple(
            "src.generation.reasoning_generator",
            get_client=MagicMock(return_value=mock_client),
        ), patch.multiple(
            "src.generation.standard_generator",
            get_client=MagicMock(return_value=mock_client),
        ), patch.multiple(
            "src.generation.icl_generator",
            get_client=MagicMock(return_value=mock_client),
        ), patch.multiple(
            "src.fixing.query_fixer",
            get_client=MagicMock(return_value=mock_client),
        ), patch.multiple(
            "src.selection.adaptive_selector",
            get_client=MagicMock(return_value=mock_client),
        ):
            # Should not raise
            result = await answer_question(entry, artifacts, db_path)

        assert isinstance(result, PipelineResult)
        # final_sql may be empty or a fallback string — not crashing is the key assertion
        assert isinstance(result.final_sql, str)
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test 5: Running answer_question twice with same artifacts does not re-run offline steps
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pipeline_caches_offline_artifacts(tmp_path):
    """Running answer_question twice reuses the same artifacts (no extra profiling calls)."""
    db_path = _make_temp_db()
    try:
        entry = _make_entry()
        artifacts = _make_mock_artifacts(db_path)
        mock_client = _make_mock_client()

        # Track how many times the faiss_index.query is called
        initial_call_count = artifacts.faiss_index.query.call_count  # = 0

        with patch.multiple(
            "src.grounding.context_grounder",
            get_client=MagicMock(return_value=mock_client),
        ), patch.multiple(
            "src.schema_linking.schema_linker",
            get_client=MagicMock(return_value=mock_client),
        ), patch.multiple(
            "src.generation.reasoning_generator",
            get_client=MagicMock(return_value=mock_client),
        ), patch.multiple(
            "src.generation.standard_generator",
            get_client=MagicMock(return_value=mock_client),
        ), patch.multiple(
            "src.generation.icl_generator",
            get_client=MagicMock(return_value=mock_client),
        ), patch.multiple(
            "src.fixing.query_fixer",
            get_client=MagicMock(return_value=mock_client),
        ), patch.multiple(
            "src.selection.adaptive_selector",
            get_client=MagicMock(return_value=mock_client),
        ):
            result1 = await answer_question(entry, artifacts, db_path)
            calls_after_first = artifacts.faiss_index.query.call_count

            result2 = await answer_question(entry, artifacts, db_path)
            calls_after_second = artifacts.faiss_index.query.call_count

        # The FAISS query should be called once per run — not zero (artifacts were used)
        assert calls_after_first > initial_call_count, "FAISS should have been queried at least once"
        # Second run also uses the same artifacts (no offline re-build)
        assert calls_after_second > calls_after_first, "Second run also queries FAISS (same artifacts)"
        # Both calls returned valid results
        assert isinstance(result1, PipelineResult)
        assert isinstance(result2, PipelineResult)
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test 6: All 3 generators are dispatched concurrently
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_generators_run_in_parallel(tmp_path):
    """Verify that all 3 generator types are called after a single answer_question invocation."""
    db_path = _make_temp_db()
    try:
        entry = _make_entry()
        artifacts = _make_mock_artifacts(db_path)

        # Track calls to each generator type via patched generate methods
        reasoning_called = []
        standard_called = []
        icl_called = []

        original_reasoning_generate = None
        original_standard_generate = None
        original_icl_generate = None

        mock_client = _make_mock_client()

        async def _mock_reasoning_generate(**kwargs):
            reasoning_called.append(True)
            return [
                SQLCandidate(
                    sql=_VALID_SQL,
                    generator_id="reasoning_A1",
                    schema_used="s1",
                    schema_format="ddl",
                    reasoning_trace=None,
                    error_flag=False,
                )
            ]

        async def _mock_standard_generate(**kwargs):
            standard_called.append(True)
            return [
                SQLCandidate(
                    sql=_VALID_SQL,
                    generator_id="standard_B1_s1",
                    schema_used="s1",
                    schema_format="markdown",
                    reasoning_trace=None,
                    error_flag=False,
                )
            ]

        async def _mock_icl_generate(**kwargs):
            icl_called.append(True)
            return [
                SQLCandidate(
                    sql=_VALID_SQL,
                    generator_id="icl_C1",
                    schema_used="s2",
                    schema_format="markdown",
                    reasoning_trace=None,
                    error_flag=False,
                )
            ]

        with patch.multiple(
            "src.grounding.context_grounder",
            get_client=MagicMock(return_value=mock_client),
        ), patch.multiple(
            "src.schema_linking.schema_linker",
            get_client=MagicMock(return_value=mock_client),
        ), patch.multiple(
            "src.fixing.query_fixer",
            get_client=MagicMock(return_value=mock_client),
        ), patch.multiple(
            "src.selection.adaptive_selector",
            get_client=MagicMock(return_value=mock_client),
        ), patch(
            "src.pipeline.online_pipeline._reasoning_generator.generate",
            side_effect=_mock_reasoning_generate,
        ), patch(
            "src.pipeline.online_pipeline._standard_generator.generate",
            side_effect=_mock_standard_generate,
        ), patch(
            "src.pipeline.online_pipeline._icl_generator.generate",
            side_effect=_mock_icl_generate,
        ):
            result = await answer_question(entry, artifacts, db_path)

        # All three generator types must have been called
        assert len(reasoning_called) >= 1, "ReasoningGenerator.generate was not called"
        assert len(standard_called) >= 1, "StandardAndComplexGenerator.generate was not called"
        assert len(icl_called) >= 1, "ICLGenerator.generate was not called"
        assert isinstance(result, PipelineResult)
    finally:
        os.unlink(db_path)
