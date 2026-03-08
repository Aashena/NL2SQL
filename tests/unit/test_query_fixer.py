"""
Unit tests for src/fixing/query_fixer.py — Op 8: Query Fixer + Semantic Verifier

Original 12 tests from the spec + 11 new tests for the integrated verification.

LLM calls are mocked; a real in-memory SQLite DB is used for execution.

Setup pattern:
  - Create a real SQLite DB (in-memory), write it to a temp file.
  - Patch `src.fixing.query_fixer.get_client` to return an AsyncMock for
    fix calls.
  - Inject a no-op mock verifier via `QueryFixer(verifier=mock_verifier)` to
    isolate fix-loop behavior from plan generation API calls.
  - New tests directly construct VerificationTestSpec / VerificationEvaluation
    objects and inject them via mock_verifier.evaluate_candidate.
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

from src.data.database import ExecutionResult
from src.generation.base_generator import SQLCandidate
from src.indexing.lsh_index import CellMatch
from src.llm.base import LLMResponse
from src.schema_linking.schema_linker import LinkedSchemas
from src.verification.query_verifier import (
    QueryVerifier,
    VerificationEvaluation,
    VerificationTestResult,
    VerificationTestSpec,
)
from src.fixing.query_fixer import FixedCandidate, QueryFixer, _categorize_error


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

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


def _make_schemas() -> LinkedSchemas:
    ddl = "CREATE TABLE students (id INTEGER PRIMARY KEY, name TEXT, gpa REAL);"
    md = (
        "## Table: students\n"
        "| Column | Type | Description |\n"
        "|--------|------|-------------|\n"
        "| id | INTEGER (PK) | student id |\n"
        "| name | TEXT | student name |\n"
        "| gpa | REAL | grade point average |"
    )
    return LinkedSchemas(
        s1_ddl=ddl, s1_markdown=md,
        s2_ddl=ddl, s2_markdown=md,
        s1_fields=[("students", "id"), ("students", "name"), ("students", "gpa")],
        s2_fields=[("students", "id"), ("students", "name"), ("students", "gpa")],
        selection_reasoning="Selected students table columns",
    )


def _make_candidate(
    sql: str,
    generator_id: str = "standard_B1_s1",
    schema_used: str = "s1",
) -> SQLCandidate:
    return SQLCandidate(
        sql=sql,
        generator_id=generator_id,
        schema_used=schema_used,
        schema_format="ddl",
    )


def _make_cell_matches() -> list[CellMatch]:
    return [
        CellMatch(
            table="students",
            column="name",
            matched_value="Alice",
            similarity_score=1.0,
            exact_match=True,
        )
    ]


def _make_mock_fix_client(sql_text: str = "SELECT * FROM students") -> AsyncMock:
    """Return an AsyncMock LLM client whose .generate() returns the given SQL as fix."""
    mock_client = AsyncMock()
    mock_client.generate = AsyncMock(
        return_value=LLMResponse(
            tool_inputs=[],
            text=sql_text,
            thinking=None,
            input_tokens=50,
            output_tokens=20,
        )
    )
    return mock_client


def _make_noop_verifier(
    plan: Optional[list[VerificationTestSpec]] = None,
    evaluation: Optional[VerificationEvaluation] = None,
) -> QueryVerifier:
    """Create a mock QueryVerifier that returns empty plan / all-pass evaluation."""
    mock_verifier = MagicMock(spec=QueryVerifier)
    mock_verifier.generate_plan = AsyncMock(return_value=plan or [])

    if evaluation is None:
        evaluation = VerificationEvaluation(
            candidate_id="mock",
            test_results=[],
            all_pass=True,
            confidence_adjustment=0.0,
            failure_hints=[],
        )
    mock_verifier.evaluate_candidate = AsyncMock(return_value=evaluation)
    return mock_verifier


# ---------------------------------------------------------------------------
# Original Test 1: Valid SQL passes through unchanged
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_valid_sql_passes_through_unchanged():
    """Syntactically correct SQL that returns rows → fix_iterations=0, SQL unchanged."""
    db_path = _make_temp_db()
    try:
        fixer = QueryFixer(verifier=_make_noop_verifier())
        candidates = [_make_candidate("SELECT * FROM students WHERE gpa > 3.0")]
        mock_client = _make_mock_fix_client()

        with patch("src.fixing.query_fixer.get_client", return_value=mock_client):
            results = await fixer.fix_candidates(
                candidates=candidates,
                question="Who has GPA above 3.0?",
                evidence="",
                schemas=_make_schemas(),
                db_path=db_path,
                cell_matches=_make_cell_matches(),
            )

        assert len(results) == 1
        fc = results[0]
        assert isinstance(fc, FixedCandidate)
        assert fc.fix_iterations == 0
        assert fc.final_sql == "SELECT * FROM students WHERE gpa > 3.0"
        assert fc.original_sql == "SELECT * FROM students WHERE gpa > 3.0"
        assert fc.execution_result.success is True
        assert fc.execution_result.is_empty is False
        # No LLM fix calls should have been made
        mock_client.generate.assert_not_called()
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Original Test 2: Syntax error triggers a fix
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_syntax_error_triggers_fix():
    """SQL with syntax error (SELECT form students) triggers a fix API call."""
    db_path = _make_temp_db()
    try:
        fixer = QueryFixer(verifier=_make_noop_verifier())
        bad_sql = "SELECT form students"
        candidates = [_make_candidate(bad_sql)]
        mock_client = _make_mock_fix_client("SELECT * FROM students")

        with patch("src.fixing.query_fixer.get_client", return_value=mock_client):
            results = await fixer.fix_candidates(
                candidates=candidates,
                question="List all students",
                evidence="",
                schemas=_make_schemas(),
                db_path=db_path,
                cell_matches=[],
            )

        assert len(results) == 1
        fc = results[0]
        assert mock_client.generate.call_count >= 1
        assert fc.final_sql == "SELECT * FROM students"
        assert fc.fix_iterations == 1
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Original Test 3: Empty result triggers a fix
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_result_triggers_fix():
    """Valid SQL returning 0 rows triggers a fix call."""
    db_path = _make_temp_db()
    try:
        fixer = QueryFixer(verifier=_make_noop_verifier())
        empty_sql = "SELECT * FROM students WHERE gpa > 5.0"
        candidates = [_make_candidate(empty_sql)]
        mock_client = _make_mock_fix_client("SELECT * FROM students")

        with patch("src.fixing.query_fixer.get_client", return_value=mock_client):
            results = await fixer.fix_candidates(
                candidates=candidates,
                question="Who has GPA above 5.0?",
                evidence="",
                schemas=_make_schemas(),
                db_path=db_path,
                cell_matches=[],
            )

        assert len(results) == 1
        assert mock_client.generate.call_count >= 1
        fc = results[0]
        assert fc.final_sql == "SELECT * FROM students"
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Original Test 4: Maximum 3 iterations even when all fail
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_max_3_iterations():
    """Even if all fix iterations produce bad SQL, the loop stops at β=3."""
    db_path = _make_temp_db()
    try:
        fixer = QueryFixer(verifier=_make_noop_verifier())
        bad_sql = "SELECT form students"
        candidates = [_make_candidate(bad_sql)]
        mock_client = _make_mock_fix_client("SELECT form students")

        with patch("src.fixing.query_fixer.get_client", return_value=mock_client):
            results = await fixer.fix_candidates(
                candidates=candidates,
                question="List all students",
                evidence="",
                schemas=_make_schemas(),
                db_path=db_path,
                cell_matches=[],
            )

        # Should have tried exactly 3 fix iterations (β=3)
        assert mock_client.generate.call_count == 3, (
            f"Expected exactly 3 LLM fix calls (β=3), got {mock_client.generate.call_count}"
        )
        assert len(results) == 1
        fc = results[0]
        assert fc.fix_iterations == 3
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Original Test 5: Fix calls use the haiku model (model_fast)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fix_uses_haiku_model():
    """Fix API calls must use settings.model_fast."""
    from src.config.settings import settings

    db_path = _make_temp_db()
    try:
        fixer = QueryFixer(verifier=_make_noop_verifier())
        bad_sql = "SELECT form students"
        candidates = [_make_candidate(bad_sql)]
        mock_client = _make_mock_fix_client("SELECT * FROM students")

        with patch("src.fixing.query_fixer.get_client", return_value=mock_client):
            await fixer.fix_candidates(
                candidates=candidates,
                question="List all students",
                evidence="",
                schemas=_make_schemas(),
                db_path=db_path,
                cell_matches=[],
            )

        for ca in mock_client.generate.call_args_list:
            model_used = ca.kwargs.get("model")
            assert model_used == settings.model_fast, (
                f"Expected model={settings.model_fast!r}, got {model_used!r}"
            )
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Original Test 6: Still-failing candidate has confidence_score=0.0
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_still_failing_candidate_discarded():
    """Candidates that fail all β=2 fix attempts have confidence_score=0.0."""
    db_path = _make_temp_db()
    try:
        fixer = QueryFixer(verifier=_make_noop_verifier())
        good_sql = "SELECT * FROM students"
        bad_sql = "SELECT form students"
        candidates = [
            _make_candidate(good_sql, generator_id="good"),
            _make_candidate(bad_sql, generator_id="bad"),
        ]
        mock_client = _make_mock_fix_client("SELECT form students")

        with patch("src.fixing.query_fixer.get_client", return_value=mock_client):
            results = await fixer.fix_candidates(
                candidates=candidates,
                question="List all students",
                evidence="",
                schemas=_make_schemas(),
                db_path=db_path,
                cell_matches=[],
            )

        assert len(results) == 2
        by_id = {r.generator_id: r for r in results}
        assert by_id["bad"].confidence_score == 0.0
        assert by_id["good"].confidence_score > 0.0
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Original Test 7: Error categorization — syntax error
# ---------------------------------------------------------------------------

def test_error_categorization_syntax():
    category = _categorize_error("near 'FORM': syntax error", is_empty=False)
    assert category == "syntax_error"


# ---------------------------------------------------------------------------
# Original Test 8: Error categorization — schema error
# ---------------------------------------------------------------------------

def test_error_categorization_schema():
    category = _categorize_error("no such column: frpm.Score", is_empty=False)
    assert category == "schema_error"


# ---------------------------------------------------------------------------
# Original Test 9: Error message included in fix prompt
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_error_message_in_fix_prompt():
    """The actual error message appears in the fix prompt sent to LLM."""
    db_path = _make_temp_db()
    try:
        fixer = QueryFixer(verifier=_make_noop_verifier())
        bad_sql = "SELECT form students"
        candidates = [_make_candidate(bad_sql)]
        mock_client = _make_mock_fix_client("SELECT * FROM students")

        with patch("src.fixing.query_fixer.get_client", return_value=mock_client):
            await fixer.fix_candidates(
                candidates=candidates,
                question="List all students",
                evidence="",
                schemas=_make_schemas(),
                db_path=db_path,
                cell_matches=[],
            )

        assert mock_client.generate.call_count >= 1
        first_call = mock_client.generate.call_args_list[0]
        messages = first_call.kwargs.get("messages", [])
        assert messages
        prompt_text = messages[0]["content"]
        assert "syntax" in prompt_text.lower() or "error" in prompt_text.lower()
        assert bad_sql in prompt_text or "FORM" in prompt_text.upper()
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Original Test 10: Confidence — clean success > success after fix
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_confidence_score_higher_for_clean_success():
    """Clean success has higher confidence than success after 1 fix."""
    db_path = _make_temp_db()
    try:
        fixer = QueryFixer(verifier=_make_noop_verifier())
        good_sql = "SELECT * FROM students WHERE gpa > 3.0"
        bad_sql = "SELECT form students"
        candidates = [
            _make_candidate(good_sql, generator_id="clean"),
            _make_candidate(bad_sql, generator_id="fixed"),
        ]
        mock_client = _make_mock_fix_client("SELECT * FROM students WHERE gpa > 3.0")

        with patch("src.fixing.query_fixer.get_client", return_value=mock_client):
            results = await fixer.fix_candidates(
                candidates=candidates,
                question="Who has GPA above 3.0?",
                evidence="",
                schemas=_make_schemas(),
                db_path=db_path,
                cell_matches=[],
            )

        by_id = {r.generator_id: r for r in results}
        assert by_id["clean"].confidence_score > by_id["fixed"].confidence_score
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Original Test 11: Confidence plausibility bonus for aggregation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_confidence_score_plausibility_bonus():
    """Aggregation returning 1 row gets plausibility bonus."""
    db_path = _make_temp_db()
    try:
        fixer = QueryFixer(verifier=_make_noop_verifier())
        agg_sql = "SELECT COUNT(*) FROM students"
        non_agg_sql = "SELECT * FROM students LIMIT 5"
        candidates = [
            _make_candidate(agg_sql, generator_id="agg"),
            _make_candidate(non_agg_sql, generator_id="non_agg"),
        ]
        mock_client = _make_mock_fix_client()

        with patch("src.fixing.query_fixer.get_client", return_value=mock_client):
            results = await fixer.fix_candidates(
                candidates=candidates,
                question="How many students are there?",
                evidence="",
                schemas=_make_schemas(),
                db_path=db_path,
                cell_matches=[],
            )

        by_id = {r.generator_id: r for r in results}
        assert by_id["agg"].confidence_score > 0.0
        assert by_id["non_agg"].confidence_score > 0.0
        mock_client.generate.assert_not_called()
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Original Test 12: Parallel fixing — independent candidates run concurrently
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_parallel_fixing_independent_candidates():
    """Fix calls for multiple failing candidates run concurrently via asyncio.gather."""
    db_path = _make_temp_db()
    try:
        fixer = QueryFixer(verifier=_make_noop_verifier())
        bad_sql = "SELECT form students"
        candidates = [
            _make_candidate(bad_sql, generator_id=f"cand_{i}")
            for i in range(3)
        ]
        completed_calls: list[int] = []

        async def tracked_generate(**kwargs):
            await asyncio.sleep(0)
            completed_calls.append(len(completed_calls))
            return LLMResponse(
                tool_inputs=[],
                text="SELECT * FROM students",
                thinking=None,
                input_tokens=50,
                output_tokens=20,
            )

        mock_client = AsyncMock()
        mock_client.generate = tracked_generate

        with patch("src.fixing.query_fixer.get_client", return_value=mock_client):
            results = await fixer.fix_candidates(
                candidates=candidates,
                question="List all students",
                evidence="",
                schemas=_make_schemas(),
                db_path=db_path,
                cell_matches=[],
            )

        assert len(results) == 3
        assert len(completed_calls) == 3
        for fc in results:
            assert fc.fix_iterations == 1
            assert fc.final_sql == "SELECT * FROM students"
    finally:
        os.unlink(db_path)


# ============================================================================
# New Tests: Integrated Verification Behavior
# ============================================================================

# ---------------------------------------------------------------------------
# New Test 13: Verification plan generated once, not per-candidate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_verification_plan_generated_once_per_question():
    """generate_plan() is called exactly once regardless of candidate count."""
    db_path = _make_temp_db()
    try:
        mock_verifier = _make_noop_verifier()
        fixer = QueryFixer(verifier=mock_verifier)

        candidates = [
            _make_candidate("SELECT * FROM students", generator_id=f"cand_{i}")
            for i in range(4)
        ]
        mock_client = _make_mock_fix_client()

        with patch("src.fixing.query_fixer.get_client", return_value=mock_client):
            await fixer.fix_candidates(
                candidates=candidates,
                question="List all students",
                evidence="",
                schemas=_make_schemas(),
                db_path=db_path,
                cell_matches=[],
            )

        mock_verifier.generate_plan.assert_called_once()
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# New Test 14: Execution success triggers verification
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_exec_ok_triggers_verification():
    """Successful execution triggers a call to evaluate_candidate."""
    db_path = _make_temp_db()
    try:
        mock_verifier = _make_noop_verifier(
            plan=[
                VerificationTestSpec(
                    test_type="ordering",
                    description="test",
                    required_sql_keywords=["ORDER BY"],
                    expected_outcome="has ORDER BY",
                    fix_hint="add ORDER BY",
                    is_critical=False,
                )
            ]
        )
        fixer = QueryFixer(verifier=mock_verifier)

        candidates = [_make_candidate("SELECT * FROM students ORDER BY id")]
        mock_client = _make_mock_fix_client()

        with patch("src.fixing.query_fixer.get_client", return_value=mock_client):
            results = await fixer.fix_candidates(
                candidates=candidates,
                question="List all students ordered by id",
                evidence="",
                schemas=_make_schemas(),
                db_path=db_path,
                cell_matches=[],
            )

        # evaluate_candidate should have been called at least once
        assert mock_verifier.evaluate_candidate.call_count >= 1
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# New Test 15: Failed execution skips verification
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_exec_fail_skips_verification():
    """When SQL fails execution, verification is not evaluated (only skipped records)."""
    db_path = _make_temp_db()
    try:
        # Use a verifier that tracks evaluate_candidate calls
        mock_verifier = _make_noop_verifier(
            plan=[
                VerificationTestSpec(
                    test_type="ordering",
                    description="test",
                    required_sql_keywords=["ORDER BY"],
                    expected_outcome="has ORDER BY",
                    fix_hint="add ORDER BY",
                    is_critical=False,
                )
            ]
        )

        # Override evaluate_candidate to track calls with execution status
        called_with_exec_ok: list[bool] = []
        original_eval = mock_verifier.evaluate_candidate

        async def tracking_eval(**kwargs):
            exec_result = kwargs.get("exec_result")
            if exec_result is not None:
                called_with_exec_ok.append(
                    exec_result.success and not exec_result.is_empty
                )
            return await original_eval(**kwargs)

        mock_verifier.evaluate_candidate = tracking_eval

        fixer = QueryFixer(verifier=mock_verifier)
        # SQL always fails — mock fix also returns failing SQL
        candidates = [_make_candidate("SELECT form students", generator_id="bad")]
        mock_client = _make_mock_fix_client("SELECT form students")  # stays broken

        with patch("src.fixing.query_fixer.get_client", return_value=mock_client):
            await fixer.fix_candidates(
                candidates=candidates,
                question="List students",
                evidence="",
                schemas=_make_schemas(),
                db_path=db_path,
                cell_matches=[],
            )

        # evaluate_candidate should never have been called with exec_ok=True
        assert not any(called_with_exec_ok), (
            "evaluate_candidate should not be called when execution always fails"
        )
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# New Test 16: Fix prompt includes verification failure hints
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fix_prompt_includes_verif_failures():
    """When verification fails, fix prompt includes rich per-test context.

    Checks:
    - ordering test: required_sql_keywords and fix_hint appear in prompt
    - grain test: verification_sql_upper appears in prompt
    - column_alignment test: check_columns appear in prompt
    - result sample rows appear (exec succeeded → rows available)
    """
    db_path = _make_temp_db()
    try:
        # ── ordering failure ──────────────────────────────────────────────────
        ordering_spec = VerificationTestSpec(
            test_type="ordering",
            description="top-5 needs ORDER BY + LIMIT",
            required_sql_keywords=["ORDER BY", "LIMIT"],
            expected_outcome="ORDER BY and LIMIT present",
            fix_hint="Add ORDER BY gpa DESC LIMIT 5",
            is_critical=False,
        )
        failing_eval = VerificationEvaluation(
            candidate_id="test",
            test_results=[
                VerificationTestResult(
                    test_type="ordering",
                    status="fail",
                    actual_outcome="Missing ORDER BY and LIMIT",
                    is_critical=False,
                )
            ],
            all_pass=False,
            confidence_adjustment=-0.1,
            failure_hints=["ORDERING TEST FAILED: Missing ORDER BY and LIMIT"],
        )
        all_pass_eval = VerificationEvaluation(
            candidate_id="test",
            test_results=[],
            all_pass=True,
            confidence_adjustment=0.0,
            failure_hints=[],
        )
        mock_verifier = _make_noop_verifier(plan=[ordering_spec], evaluation=failing_eval)
        call_count = [0]

        async def mock_evaluate(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return failing_eval
            return all_pass_eval

        mock_verifier.evaluate_candidate = mock_evaluate
        fixer = QueryFixer(verifier=mock_verifier)
        candidates = [_make_candidate("SELECT * FROM students", generator_id="test")]
        mock_client = _make_mock_fix_client("SELECT * FROM students ORDER BY gpa DESC LIMIT 5")

        with patch("src.fixing.query_fixer.get_client", return_value=mock_client):
            await fixer.fix_candidates(
                candidates=candidates,
                question="Who are the top 5 students by GPA?",
                evidence="",
                schemas=_make_schemas(),
                db_path=db_path,
                cell_matches=[],
            )

        assert mock_client.generate.call_count >= 1
        fix_prompt = mock_client.generate.call_args_list[0].kwargs["messages"][0]["content"]

        # ordering: structured block header + required keywords + fix_hint
        assert "ORDERING TEST FAILED" in fix_prompt
        assert "ORDER BY" in fix_prompt
        assert "LIMIT" in fix_prompt
        assert "Add ORDER BY gpa DESC LIMIT 5" in fix_prompt  # fix_hint
        # Result sample: exec succeeded → rows from students table appear
        assert "Current result sample" in fix_prompt

        # ── grain failure: verification_sql_upper appears ─────────────────────
        grain_spec = VerificationTestSpec(
            test_type="grain",
            description="one row per student",
            verification_sql_upper="SELECT COUNT(DISTINCT id) FROM students",
            expected_outcome="row count <= distinct id count",
            fix_hint="Add GROUP BY id to collapse duplicates",
            is_critical=True,
        )
        grain_fail_eval = VerificationEvaluation(
            candidate_id="grain_test",
            test_results=[
                VerificationTestResult(
                    test_type="grain",
                    status="fail",
                    actual_outcome="Row count 9 exceeds upper bound 3",
                    is_critical=True,
                )
            ],
            all_pass=False,
            confidence_adjustment=-0.3,
            failure_hints=["GRAIN TEST FAILED"],
        )
        mock_verifier2 = _make_noop_verifier(plan=[grain_spec], evaluation=grain_fail_eval)
        mock_verifier2.evaluate_candidate = AsyncMock(return_value=grain_fail_eval)
        fixer2 = QueryFixer(verifier=mock_verifier2)
        mock_client2 = _make_mock_fix_client("SELECT id FROM students GROUP BY id")

        with patch("src.fixing.query_fixer.get_client", return_value=mock_client2):
            await fixer2.fix_candidates(
                candidates=[_make_candidate("SELECT * FROM students", generator_id="grain_test")],
                question="List one row per student",
                evidence="",
                schemas=_make_schemas(),
                db_path=db_path,
                cell_matches=[],
            )

        grain_prompt = mock_client2.generate.call_args_list[0].kwargs["messages"][0]["content"]
        assert "GRAIN TEST FAILED" in grain_prompt
        assert "SELECT COUNT(DISTINCT id) FROM students" in grain_prompt  # verification_sql_upper
        assert "Add GROUP BY id to collapse duplicates" in grain_prompt   # fix_hint

        # ── column_alignment failure: check_columns appear ────────────────────
        col_spec = VerificationTestSpec(
            test_type="column_alignment",
            description="must return name and gpa",
            check_columns=["name", "gpa"],
            expected_outcome="SELECT returns name and gpa",
            fix_hint="Update SELECT to return name and gpa columns",
            is_critical=True,
        )
        col_fail_eval = VerificationEvaluation(
            candidate_id="col_test",
            test_results=[
                VerificationTestResult(
                    test_type="column_alignment",
                    status="fail",
                    actual_outcome="SELECT columns do not match question requirements",
                    is_critical=True,
                )
            ],
            all_pass=False,
            confidence_adjustment=-0.3,
            failure_hints=["COLUMN_ALIGNMENT TEST FAILED"],
        )
        mock_verifier3 = _make_noop_verifier(plan=[col_spec], evaluation=col_fail_eval)
        mock_verifier3.evaluate_candidate = AsyncMock(return_value=col_fail_eval)
        fixer3 = QueryFixer(verifier=mock_verifier3)
        mock_client3 = _make_mock_fix_client("SELECT name, gpa FROM students")

        with patch("src.fixing.query_fixer.get_client", return_value=mock_client3):
            await fixer3.fix_candidates(
                candidates=[_make_candidate("SELECT id FROM students", generator_id="col_test")],
                question="Show student names and GPAs",
                evidence="",
                schemas=_make_schemas(),
                db_path=db_path,
                cell_matches=[],
            )

        col_prompt = mock_client3.generate.call_args_list[0].kwargs["messages"][0]["content"]
        assert "COLUMN_ALIGNMENT TEST FAILED" in col_prompt
        assert "name" in col_prompt and "gpa" in col_prompt   # check_columns
        assert "Update SELECT to return name and gpa columns" in col_prompt  # fix_hint
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# New Test 17: Fix prompt includes execution error when exec fails
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fix_prompt_includes_exec_error():
    """Fix prompt includes execution error message when SQL fails."""
    db_path = _make_temp_db()
    try:
        fixer = QueryFixer(verifier=_make_noop_verifier())
        bad_sql = "SELECT form students"
        candidates = [_make_candidate(bad_sql)]
        mock_client = _make_mock_fix_client("SELECT * FROM students")

        with patch("src.fixing.query_fixer.get_client", return_value=mock_client):
            await fixer.fix_candidates(
                candidates=candidates,
                question="List students",
                evidence="",
                schemas=_make_schemas(),
                db_path=db_path,
                cell_matches=[],
            )

        fix_prompt = mock_client.generate.call_args_list[0].kwargs["messages"][0]["content"]
        assert "Execution Error" in fix_prompt or "syntax" in fix_prompt.lower()
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# New Test 18: Both pass on iter 0 → loop breaks, no fix call
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_verif_pass_breaks_loop_early():
    """If exec and verif both pass on iteration 0, no fix LLM call is made."""
    db_path = _make_temp_db()
    try:
        all_pass_eval = VerificationEvaluation(
            candidate_id="test",
            test_results=[],
            all_pass=True,
            confidence_adjustment=0.0,
            failure_hints=[],
        )
        mock_verifier = _make_noop_verifier(evaluation=all_pass_eval)
        fixer = QueryFixer(verifier=mock_verifier)

        candidates = [_make_candidate("SELECT * FROM students WHERE gpa > 3.0")]
        mock_client = _make_mock_fix_client()

        with patch("src.fixing.query_fixer.get_client", return_value=mock_client):
            results = await fixer.fix_candidates(
                candidates=candidates,
                question="Who has GPA above 3.0?",
                evidence="",
                schemas=_make_schemas(),
                db_path=db_path,
                cell_matches=[],
            )

        # No fix calls needed — both stages pass immediately
        mock_client.generate.assert_not_called()
        assert results[0].fix_iterations == 0
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# New Test 19: Expensive tests run on every iteration (run_expensive=True always)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_expensive_tests_run_on_every_iteration():
    """evaluate_candidate is called with run_expensive=True on every iteration.

    Since expensive tests now run on all iterations, failures in column_alignment
    or boundary tests can drive fix attempts on iterations 0 and 1.
    """
    db_path = _make_temp_db()
    try:
        call_args_log: list[dict] = []

        # Always return all_pass=False so the loop exhausts its full budget
        async def mock_evaluate(**kwargs):
            call_args_log.append({
                "run_expensive": kwargs.get("run_expensive"),
            })
            return VerificationEvaluation(
                candidate_id="test",
                test_results=[
                    VerificationTestResult(
                        test_type="ordering",
                        status="fail",
                        actual_outcome="Missing ORDER BY",
                        is_critical=False,
                    )
                ],
                all_pass=False,
                confidence_adjustment=-0.1,
                failure_hints=["ORDERING TEST FAILED: Missing ORDER BY  Hint: Add ORDER BY"],
            )

        mock_verifier = MagicMock(spec=QueryVerifier)
        mock_verifier.generate_plan = AsyncMock(return_value=[
            VerificationTestSpec(
                test_type="ordering",
                description="test",
                required_sql_keywords=["ORDER BY"],
                expected_outcome="has ORDER BY",
                fix_hint="add ORDER BY",
                is_critical=False,
            )
        ])
        mock_verifier.evaluate_candidate = mock_evaluate

        fixer = QueryFixer(verifier=mock_verifier)

        # SQL executes successfully (so Stage B runs) but always fails verification
        candidates = [_make_candidate("SELECT * FROM students", generator_id="test")]
        # Fix always returns same SQL (still no ORDER BY → still fails verif)
        mock_client = _make_mock_fix_client("SELECT * FROM students")

        with patch("src.fixing.query_fixer.get_client", return_value=mock_client):
            await fixer.fix_candidates(
                candidates=candidates,
                question="List students by GPA",
                evidence="",
                schemas=_make_schemas(),
                db_path=db_path,
                cell_matches=[],
            )

        # Every evaluate_candidate call must have run_expensive=True
        assert len(call_args_log) >= 1, "Expected at least one evaluate_candidate call"
        non_expensive = [c for c in call_args_log if c["run_expensive"] is not True]
        assert len(non_expensive) == 0, (
            "Every evaluate_candidate call must use run_expensive=True; "
            f"found non-expensive calls: {non_expensive}"
        )
        # All four iterations should have evaluated (loop exhausts budget)
        assert len(call_args_log) == 4, (
            f"Expected 4 evaluate_candidate calls (_BETA+1), got {len(call_args_log)}"
        )
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# New Test 20: Confidence penalized for critical verification failure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_confidence_penalized_for_critical_verif_failure():
    """Critical test failure reduces confidence score vs. an equivalent candidate without failure."""
    db_path = _make_temp_db()
    try:
        # Candidate A: all verif pass (no adjustment)
        all_pass_eval = VerificationEvaluation(
            candidate_id="A",
            test_results=[
                VerificationTestResult(
                    test_type="grain",
                    status="pass",
                    actual_outcome="OK",
                    is_critical=True,
                )
            ],
            all_pass=True,
            confidence_adjustment=0.2,  # bonus
            failure_hints=[],
        )
        # Candidate B: critical grain failure
        crit_fail_eval = VerificationEvaluation(
            candidate_id="B",
            test_results=[
                VerificationTestResult(
                    test_type="grain",
                    status="fail",
                    actual_outcome="Wrong grain",
                    is_critical=True,
                )
            ],
            all_pass=False,
            confidence_adjustment=-0.3,  # critical penalty
            failure_hints=["GRAIN TEST FAILED: ..."],
        )

        call_index = [0]

        async def mock_evaluate(**kwargs):
            idx = call_index[0]
            call_index[0] += 1
            return all_pass_eval if idx == 0 else crit_fail_eval

        mock_verifier = MagicMock(spec=QueryVerifier)
        mock_verifier.generate_plan = AsyncMock(return_value=[
            VerificationTestSpec(
                test_type="grain",
                description="grain check",
                verification_sql="SELECT COUNT(*) FROM students",
                expected_outcome="3 rows",
                fix_hint="fix grain",
                is_critical=True,
            )
        ])
        mock_verifier.evaluate_candidate = mock_evaluate

        fixer = QueryFixer(verifier=mock_verifier)

        # Both candidates execute successfully
        candidates = [
            _make_candidate("SELECT * FROM students", generator_id="A"),
            _make_candidate("SELECT * FROM students LIMIT 10", generator_id="B"),
        ]
        mock_client = _make_mock_fix_client()

        with patch("src.fixing.query_fixer.get_client", return_value=mock_client):
            results = await fixer.fix_candidates(
                candidates=candidates,
                question="List students",
                evidence="",
                schemas=_make_schemas(),
                db_path=db_path,
                cell_matches=[],
            )

        by_id = {r.generator_id: r for r in results}
        assert by_id["A"].confidence_score > by_id["B"].confidence_score, (
            f"Candidate A (all-pass) should have higher confidence than B (critical fail). "
            f"A={by_id['A'].confidence_score:.3f}, B={by_id['B'].confidence_score:.3f}"
        )
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# New Test 21: Confidence bonus when all verif tests pass
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_confidence_bonus_all_verif_pass():
    """A candidate with all verification tests passing gets a higher confidence than one with no verification."""
    db_path = _make_temp_db()
    try:
        # Candidate A: full verification pass → bonus
        all_pass_eval = VerificationEvaluation(
            candidate_id="A",
            test_results=[
                VerificationTestResult(
                    test_type="ordering",
                    status="pass",
                    actual_outcome="OK",
                    is_critical=False,
                )
            ],
            all_pass=True,
            confidence_adjustment=0.2,
            failure_hints=[],
        )
        # Candidate B: no verification (empty specs) → adjustment=0.0
        no_verif_eval = VerificationEvaluation(
            candidate_id="B",
            test_results=[],
            all_pass=True,
            confidence_adjustment=0.0,
            failure_hints=[],
        )

        call_index = [0]

        async def mock_evaluate(**kwargs):
            cid = kwargs.get("candidate_id", "")
            return all_pass_eval if cid == "A" else no_verif_eval

        mock_verifier = MagicMock(spec=QueryVerifier)
        mock_verifier.generate_plan = AsyncMock(return_value=[
            VerificationTestSpec(
                test_type="ordering",
                description="test",
                required_sql_keywords=["ORDER BY"],
                expected_outcome="has ORDER BY",
                fix_hint="add ORDER BY",
                is_critical=False,
            )
        ])
        mock_verifier.evaluate_candidate = mock_evaluate

        fixer = QueryFixer(verifier=mock_verifier)
        candidates = [
            _make_candidate(
                "SELECT * FROM students ORDER BY gpa DESC", generator_id="A"
            ),
            _make_candidate("SELECT * FROM students", generator_id="B"),
        ]
        mock_client = _make_mock_fix_client()

        with patch("src.fixing.query_fixer.get_client", return_value=mock_client):
            results = await fixer.fix_candidates(
                candidates=candidates,
                question="List students",
                evidence="",
                schemas=_make_schemas(),
                db_path=db_path,
                cell_matches=[],
            )

        by_id = {r.generator_id: r for r in results}
        assert by_id["A"].confidence_score >= by_id["B"].confidence_score, (
            f"Candidate A (all-pass bonus) should have >= confidence than B. "
            f"A={by_id['A'].confidence_score:.3f}, B={by_id['B'].confidence_score:.3f}"
        )
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# New Test 22: FixedCandidate carries verification_results field
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fixed_candidate_has_verification_results():
    """FixedCandidate.verification_results is populated when verification runs."""
    db_path = _make_temp_db()
    try:
        null_spec = VerificationTestSpec(
            test_type="null",
            description="names should not be null",
            check_columns=["name"],
            expected_outcome="No null names",
            fix_hint="Add WHERE name IS NOT NULL",
            is_critical=False,
        )
        expected_eval = VerificationEvaluation(
            candidate_id="test",
            test_results=[
                VerificationTestResult(
                    test_type="null",
                    status="pass",
                    actual_outcome="No NULLs",
                    is_critical=False,
                )
            ],
            all_pass=True,
            confidence_adjustment=0.2,
            failure_hints=[],
        )
        # Non-empty plan so verif_specs is truthy → evaluate_candidate gets called
        mock_verifier = _make_noop_verifier(plan=[null_spec], evaluation=expected_eval)
        fixer = QueryFixer(verifier=mock_verifier)

        candidates = [_make_candidate("SELECT * FROM students")]
        mock_client = _make_mock_fix_client()

        with patch("src.fixing.query_fixer.get_client", return_value=mock_client):
            results = await fixer.fix_candidates(
                candidates=candidates,
                question="List all students",
                evidence="",
                schemas=_make_schemas(),
                db_path=db_path,
                cell_matches=[],
            )

        fc = results[0]
        assert fc.verification_results is not None
        assert fc.verification_results.all_pass is True
        assert len(fc.verification_results.test_results) == 1
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# New Test 23: Verification failure followed by successful fix
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_verification_failure_then_fixed():
    """Candidate that fails verification on iter 0 is fixed and passes on iter 1."""
    db_path = _make_temp_db()
    try:
        failing_eval = VerificationEvaluation(
            candidate_id="test",
            test_results=[
                VerificationTestResult(
                    test_type="ordering",
                    status="fail",
                    actual_outcome="Missing ORDER BY",
                    is_critical=False,
                )
            ],
            all_pass=False,
            confidence_adjustment=-0.1,
            failure_hints=["ORDERING TEST FAILED: Missing ORDER BY  Hint: Add ORDER BY gpa DESC LIMIT 5"],
        )
        passing_eval = VerificationEvaluation(
            candidate_id="test",
            test_results=[
                VerificationTestResult(
                    test_type="ordering",
                    status="pass",
                    actual_outcome="ORDER BY and LIMIT present",
                    is_critical=False,
                )
            ],
            all_pass=True,
            confidence_adjustment=0.2,
            failure_hints=[],
        )
        call_count = [0]

        async def mock_evaluate(**kwargs):
            call_count[0] += 1
            return failing_eval if call_count[0] == 1 else passing_eval

        mock_verifier = MagicMock(spec=QueryVerifier)
        mock_verifier.generate_plan = AsyncMock(return_value=[
            VerificationTestSpec(
                test_type="ordering",
                description="test",
                required_sql_keywords=["ORDER BY", "LIMIT"],
                expected_outcome="has ORDER BY and LIMIT",
                fix_hint="Add ORDER BY gpa DESC LIMIT 5",
                is_critical=False,
            )
        ])
        mock_verifier.evaluate_candidate = mock_evaluate

        fixer = QueryFixer(verifier=mock_verifier)

        # Initial SQL has no ORDER BY; fix adds it
        candidates = [_make_candidate("SELECT * FROM students", generator_id="test")]
        mock_client = _make_mock_fix_client(
            "SELECT * FROM students ORDER BY gpa DESC LIMIT 5"
        )

        with patch("src.fixing.query_fixer.get_client", return_value=mock_client):
            results = await fixer.fix_candidates(
                candidates=candidates,
                question="Who are the top 5 students?",
                evidence="",
                schemas=_make_schemas(),
                db_path=db_path,
                cell_matches=[],
            )

        fc = results[0]
        # Exactly 1 fix iteration was used
        assert fc.fix_iterations == 1
        # The final SQL is the fixed version
        assert "ORDER BY" in fc.final_sql.upper()
        # Verification should pass in the end
        assert fc.verification_results is not None
        assert fc.verification_results.all_pass is True
        assert fc.confidence_score > 0.0
    finally:
        os.unlink(db_path)
