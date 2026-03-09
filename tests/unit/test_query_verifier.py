"""
Unit tests for src/verification/query_verifier.py

All tests use a real in-memory SQLite database for SQL execution and mock
the LLM client for plan generation and expensive judgment tests.

Setup pattern:
  - Create a temp SQLite file with 'students' and 'enrollments' tables.
  - Patch 'src.verification.query_verifier.get_client' where needed.
  - Construct VerificationTestSpec objects directly (bypassing LLM) for
    evaluation-only tests.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
from unittest.mock import AsyncMock, patch

import pytest

from src.data.database import ExecutionResult, execute_sql
from src.llm.base import LLMError, LLMResponse
from src.verification.query_verifier import (
    QueryVerifier,
    VerificationEvaluation,
    VerificationTestResult,
    VerificationTestSpec,
    _derive_direction_from_question,
    _extract_limit_from_question,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_temp_db() -> str:
    """Create a temp SQLite file with students + enrollments tables."""
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "CREATE TABLE students "
            "(id INTEGER PRIMARY KEY, name TEXT, gpa REAL, age INTEGER)"
        )
        conn.execute("INSERT INTO students VALUES (1, 'Alice', 3.9, 22)")
        conn.execute("INSERT INTO students VALUES (2, 'Bob', 2.8, 24)")
        conn.execute("INSERT INTO students VALUES (3, 'Carol', 3.5, 21)")
        conn.execute(
            "CREATE TABLE enrollments "
            "(id INTEGER PRIMARY KEY, student_id INTEGER, course TEXT)"
        )
        # 2 courses per student → 6 total enrollments
        conn.execute("INSERT INTO enrollments VALUES (1, 1, 'Math')")
        conn.execute("INSERT INTO enrollments VALUES (2, 1, 'Science')")
        conn.execute("INSERT INTO enrollments VALUES (3, 2, 'Math')")
        conn.execute("INSERT INTO enrollments VALUES (4, 2, 'History')")
        conn.execute("INSERT INTO enrollments VALUES (5, 3, 'Science')")
        conn.execute("INSERT INTO enrollments VALUES (6, 3, 'History')")
        conn.commit()
    finally:
        conn.close()
    return path


def _exec_result(rows: list, success: bool = True) -> ExecutionResult:
    return ExecutionResult(
        success=success,
        rows=rows,
        error=None if success else "test error",
        execution_time=0.001,
        is_empty=(len(rows) == 0),
    )


def _make_mock_client(text_response: str = "PASS\nLooks good.") -> AsyncMock:
    mock = AsyncMock()
    mock.generate = AsyncMock(
        return_value=LLMResponse(
            tool_inputs=[],
            text=text_response,
            input_tokens=50,
            output_tokens=20,
        )
    )
    return mock


# ---------------------------------------------------------------------------
# Test 1: generate_plan returns structured test list (both calls mocked)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_plan_returns_structured_tests():
    """
    generate_plan runs two internal calls concurrently.
    Mock each internal method directly → verify merged result.
    Column order: grain first, column_alignment second, then others.
    """
    ordering_spec = VerificationTestSpec(
        test_type="ordering",
        required_sql_keywords=["ORDER BY", "LIMIT"],
        fix_hint="Add ORDER BY gpa DESC LIMIT N",
    )
    grain_spec = VerificationTestSpec(
        test_type="grain",
        fix_hint="Check GROUP BY",
    )
    col_align_spec = VerificationTestSpec(
        test_type="column_alignment",
        expected_column_count=1,
        fix_hint="Adjust SELECT to return exactly 1 column.",
    )

    verifier = QueryVerifier()
    with (
        patch.object(verifier, "_generate_main_plan", AsyncMock(return_value=[grain_spec, ordering_spec])),
        patch.object(verifier, "_generate_column_alignment_spec", AsyncMock(return_value=col_align_spec)),
    ):
        specs = await verifier.generate_plan(
            question="Who are the top 5 students by GPA?",
            evidence="",
            schema="CREATE TABLE students (id INTEGER PRIMARY KEY, name TEXT, gpa REAL);",
        )

    # Expected order: grain → column_alignment → ordering
    assert len(specs) == 3
    assert specs[0].test_type == "grain"
    assert specs[1].test_type == "column_alignment"
    assert specs[1].expected_column_count == 1
    assert specs[2].test_type == "ordering"
    assert specs[2].required_sql_keywords == ["ORDER BY", "LIMIT"]


# ---------------------------------------------------------------------------
# Test 2: generate_plan returns default on LLM error (both calls fail)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_plan_returns_default_on_llm_error():
    """LLM raises LLMError for ALL calls → fallback: grain + column_alignment (count=None). Never raises."""
    mock_client = AsyncMock()
    mock_client.generate = AsyncMock(side_effect=LLMError("API unavailable"))

    verifier = QueryVerifier()
    with patch("src.verification.query_verifier.get_client", return_value=mock_client):
        specs = await verifier.generate_plan(
            question="How many students are there?",
            evidence="",
            schema="CREATE TABLE students (id INTEGER PRIMARY KEY);",
        )

    assert len(specs) == 2
    assert specs[0].test_type == "grain"
    assert specs[1].test_type == "column_alignment"
    assert specs[1].expected_column_count is None


# ---------------------------------------------------------------------------
# Test 2b: _generate_column_alignment_spec — scalar question → count=1
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_column_alignment_spec_scalar():
    """Dedicated col-align call: scalar question → expected_column_count=1."""
    mock_client = AsyncMock()
    mock_client.generate = AsyncMock(
        return_value=LLMResponse(
            tool_inputs=[{
                "reasoning": "COUNT aggregate → 1 column",
                "expected_column_count": 1,
                "column_descriptions": ["enrollment count"],
            }],
            text=None,
            input_tokens=50,
            output_tokens=30,
        )
    )

    verifier = QueryVerifier()
    with patch("src.verification.query_verifier.get_client", return_value=mock_client):
        spec = await verifier._generate_column_alignment_spec(
            question="How many students are enrolled in the club?",
            evidence="",
            schema="CREATE TABLE students (id INTEGER PRIMARY KEY, name TEXT);",
        )

    assert spec.test_type == "column_alignment"
    assert spec.expected_column_count == 1
    assert spec.column_descriptions == ["enrollment count"]


# ---------------------------------------------------------------------------
# Test 2c: _generate_column_alignment_spec — two-column question → count=2
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_column_alignment_spec_two_col():
    """Dedicated col-align call: name+value question → expected_column_count=2."""
    mock_client = AsyncMock()
    mock_client.generate = AsyncMock(
        return_value=LLMResponse(
            tool_inputs=[{
                "reasoning": "Question asks for name AND gpa → 2 columns",
                "expected_column_count": 2,
                "column_descriptions": ["student name", "GPA"],
            }],
            text=None,
            input_tokens=50,
            output_tokens=30,
        )
    )

    verifier = QueryVerifier()
    with patch("src.verification.query_verifier.get_client", return_value=mock_client):
        spec = await verifier._generate_column_alignment_spec(
            question="List the name and GPA of each student who made the honor roll.",
            evidence="",
            schema="CREATE TABLE students (id INTEGER PRIMARY KEY, name TEXT, gpa REAL);",
        )

    assert spec.test_type == "column_alignment"
    assert spec.expected_column_count == 2
    assert spec.column_descriptions == ["student name", "GPA"]


# ---------------------------------------------------------------------------
# Test 2d: _generate_column_alignment_spec — LLM failure → graceful fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_column_alignment_spec_llm_failure():
    """LLM raises during dedicated col-align call → returns count=None gracefully (never raises)."""
    mock_client = AsyncMock()
    mock_client.generate = AsyncMock(side_effect=LLMError("timeout"))

    verifier = QueryVerifier()
    with patch("src.verification.query_verifier.get_client", return_value=mock_client):
        spec = await verifier._generate_column_alignment_spec(
            question="How many schools have an average score above 400?",
            evidence="",
            schema="CREATE TABLE schools (id INTEGER PRIMARY KEY, name TEXT);",
        )

    assert spec.test_type == "column_alignment"
    assert spec.expected_column_count is None
    assert spec.column_descriptions == []


# ---------------------------------------------------------------------------
# Test 2e: _generate_column_alignment_spec — mismatched column_descriptions length
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_column_alignment_spec_mismatched_descriptions():
    """column_descriptions with wrong length → cleared to [] gracefully."""
    mock_client = AsyncMock()
    mock_client.generate = AsyncMock(
        return_value=LLMResponse(
            tool_inputs=[{
                "reasoning": "COUNT aggregate → 1 column",
                "expected_column_count": 1,
                "column_descriptions": ["col1", "col2"],  # length 2 ≠ count 1
            }],
            text=None,
            input_tokens=50,
            output_tokens=30,
        )
    )

    verifier = QueryVerifier()
    with patch("src.verification.query_verifier.get_client", return_value=mock_client):
        spec = await verifier._generate_column_alignment_spec(
            question="How many schools have an average score above 400?",
            evidence="",
            schema="CREATE TABLE schools (id INTEGER PRIMARY KEY, name TEXT);",
        )

    assert spec.expected_column_count == 1
    assert spec.column_descriptions == []


# ---------------------------------------------------------------------------
# Test 3: grain test PASS — COUNT(*) matches expected
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_grain_test_pass():
    """Grain test passes when result row count matches verification SQL count."""
    db_path = _make_temp_db()
    try:
        verifier = QueryVerifier()
        spec = VerificationTestSpec(
            test_type="grain",
            description="One row per student",
            verification_sql="SELECT COUNT(DISTINCT id) FROM students",
            expected_outcome="Result has 3 rows (one per student)",
            fix_hint="Add GROUP BY student_id",
        )
        # Query that returns exactly 3 rows (matching the 3 distinct students)
        exec_result = execute_sql(db_path, "SELECT id, name FROM students")
        assert exec_result.success and len(exec_result.rows) == 3

        eval_result = await verifier.evaluate_candidate(
            specs=[spec],
            candidate_id="test_grain_pass",
            sql="SELECT id, name FROM students",
            exec_result=exec_result,
            db_path=db_path,
            run_expensive=False,
        )

        assert eval_result.all_pass is True
        grain_result = eval_result.test_results[0]
        assert grain_result.status == "pass"
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test 4: grain test FAIL — result has more rows than expected
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_grain_test_fail():
    """Grain test fails when result has more rows than expected entity count."""
    db_path = _make_temp_db()
    try:
        verifier = QueryVerifier()
        spec = VerificationTestSpec(
            test_type="grain",
            description="One row per student",
            verification_sql="SELECT COUNT(DISTINCT id) FROM students",  # returns 3
            expected_outcome="Result has 3 rows (one per student)",
            fix_hint="Use GROUP BY student_id to get one row per student",
        )
        # JOIN produces 6 rows (2 courses per student × 3 students)
        exec_result = execute_sql(
            db_path,
            "SELECT s.id, s.name FROM students s JOIN enrollments e ON s.id = e.student_id"
        )
        assert exec_result.success and len(exec_result.rows) == 6

        eval_result = await verifier.evaluate_candidate(
            specs=[spec],
            candidate_id="test_grain_fail",
            sql="SELECT s.id, s.name FROM students s JOIN enrollments e ON s.id = e.student_id",
            exec_result=exec_result,
            db_path=db_path,
            run_expensive=False,
        )

        assert eval_result.all_pass is False
        grain_result = eval_result.test_results[0]
        assert grain_result.status == "fail"
        assert "6" in grain_result.actual_outcome  # actual row count
        assert "3" in grain_result.actual_outcome  # expected count
        assert len(eval_result.failure_hints) >= 1
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test 5: null test PASS — no NULLs in result
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_null_test_pass():
    """Null test passes when result rows have no NULL values."""
    db_path = _make_temp_db()
    try:
        verifier = QueryVerifier()
        spec = VerificationTestSpec(
            test_type="null",
            description="Student names should not be NULL",
            check_columns=["name"],
            expected_outcome="No NULL names",
            fix_hint="Add WHERE name IS NOT NULL",
        )
        exec_result = execute_sql(db_path, "SELECT id, name FROM students")

        eval_result = await verifier.evaluate_candidate(
            specs=[spec],
            candidate_id="test_null_pass",
            sql="SELECT id, name FROM students",
            exec_result=exec_result,
            db_path=db_path,
            run_expensive=False,
        )

        assert eval_result.all_pass is True
        assert eval_result.test_results[0].status == "pass"
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test 6: null test FAIL — result contains NULL values
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_null_test_fail():
    """Null test fails when result rows contain NULL values."""
    db_path = _make_temp_db()
    try:
        # Insert a student with NULL name
        conn = sqlite3.connect(db_path)
        conn.execute("INSERT INTO students VALUES (4, NULL, 3.0, 20)")
        conn.commit()
        conn.close()

        verifier = QueryVerifier()
        spec = VerificationTestSpec(
            test_type="null",
            description="Student names should not be NULL",
            check_columns=["name"],
            expected_outcome="No NULL names",
            fix_hint="Add WHERE name IS NOT NULL",
        )
        exec_result = execute_sql(db_path, "SELECT id, name FROM students")
        assert exec_result.success

        eval_result = await verifier.evaluate_candidate(
            specs=[spec],
            candidate_id="test_null_fail",
            sql="SELECT id, name FROM students",
            exec_result=exec_result,
            db_path=db_path,
            run_expensive=False,
        )

        assert eval_result.all_pass is False
        assert eval_result.test_results[0].status == "fail"
        assert "NULL" in eval_result.test_results[0].actual_outcome
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test 7: duplicate test detects JOIN row multiplication
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_duplicate_test_detects_join_multiplication():
    """Duplicate test fails when JOIN produces more rows than distinct entities."""
    db_path = _make_temp_db()
    try:
        verifier = QueryVerifier()
        # verification_sql counts 3 distinct students; JOIN produces 6 rows
        spec = VerificationTestSpec(
            test_type="duplicate",
            description="One row per student without JOIN duplication",
            verification_sql="SELECT COUNT(DISTINCT id) FROM students",
            expected_outcome="Result rows ≈ 3 distinct students",
            fix_hint="Use DISTINCT or restructure the JOIN to avoid row multiplication",
        )
        exec_result = execute_sql(
            db_path,
            "SELECT s.id, s.name FROM students s "
            "JOIN enrollments e ON s.id = e.student_id",
        )
        assert exec_result.success and len(exec_result.rows) == 6

        eval_result = await verifier.evaluate_candidate(
            specs=[spec],
            candidate_id="test_duplicate",
            sql="SELECT s.id, s.name FROM students s JOIN enrollments e ON s.id = e.student_id",
            exec_result=exec_result,
            db_path=db_path,
            run_expensive=False,
        )

        assert eval_result.all_pass is False
        dup_result = eval_result.test_results[0]
        assert dup_result.status == "fail"
        assert "6" in dup_result.actual_outcome  # actual count
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test 8: ordering test FAIL — missing ORDER BY for top-N question
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ordering_test_structural_missing_order_by():
    """Ordering test fails when ORDER BY is absent from a top-N query."""
    verifier = QueryVerifier()
    spec = VerificationTestSpec(
        test_type="ordering",
        description="Top 5 question requires ORDER BY and LIMIT",
        required_sql_keywords=["ORDER BY", "LIMIT"],
        expected_outcome="SQL contains ORDER BY and LIMIT",
        fix_hint="Add ORDER BY gpa DESC LIMIT 5",
    )
    exec_result = _exec_result([(1, "Alice", 3.9), (2, "Bob", 2.8)])

    eval_result = await verifier.evaluate_candidate(
        specs=[spec],
        candidate_id="test_ordering_fail",
        sql="SELECT id, name, gpa FROM students",  # no ORDER BY, no LIMIT
        exec_result=exec_result,
        db_path="/dev/null",  # not needed for structural test
        run_expensive=False,
    )

    assert eval_result.all_pass is False
    ord_result = eval_result.test_results[0]
    assert ord_result.status == "fail"
    assert "ORDER BY" in ord_result.actual_outcome or "LIMIT" in ord_result.actual_outcome


# ---------------------------------------------------------------------------
# Test 9: ordering test PASS — ORDER BY and LIMIT present
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ordering_test_pass_when_keywords_present():
    """Ordering test passes when all required SQL keywords are present."""
    verifier = QueryVerifier()
    spec = VerificationTestSpec(
        test_type="ordering",
        description="Top 5 question requires ORDER BY and LIMIT",
        required_sql_keywords=["ORDER BY", "LIMIT"],
        expected_outcome="SQL contains ORDER BY and LIMIT",
        fix_hint="Add ORDER BY gpa DESC LIMIT 5",
    )
    exec_result = _exec_result([(1, "Alice", 3.9), (2, "Carol", 3.5)])

    eval_result = await verifier.evaluate_candidate(
        specs=[spec],
        candidate_id="test_ordering_pass",
        sql="SELECT id, name, gpa FROM students ORDER BY gpa DESC LIMIT 5",
        exec_result=exec_result,
        db_path="/dev/null",
        run_expensive=False,
    )

    assert eval_result.all_pass is True
    assert eval_result.test_results[0].status == "pass"


# ---------------------------------------------------------------------------
# Test 10: scale test FAIL — out-of-range percentage value
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scale_test_out_of_range_percentage():
    """Scale test fails when a numeric value exceeds the expected maximum."""
    verifier = QueryVerifier()
    spec = VerificationTestSpec(
        test_type="scale",
        description="Percentage values should be between 0 and 100",
        numeric_min=0.0,
        numeric_max=100.0,
        expected_outcome="All percentages in 0–100 range",
        fix_hint="Check the calculation — result may be a ratio not a percentage",
    )
    # Result contains 150.0 which is out of range
    exec_result = _exec_result([(1, 75.0), (2, 150.0), (3, 50.0)])

    eval_result = await verifier.evaluate_candidate(
        specs=[spec],
        candidate_id="test_scale_fail",
        sql="SELECT id, pct FROM some_table",
        exec_result=exec_result,
        db_path="/dev/null",
        run_expensive=False,
    )

    assert eval_result.all_pass is False
    scale_result = eval_result.test_results[0]
    assert scale_result.status == "fail"
    assert "150" in scale_result.actual_outcome


# ---------------------------------------------------------------------------
# Test 11: boundary (expensive) skipped when run_expensive=False;
#           column_alignment (now cheap) always runs
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_expensive_tests_skipped_when_run_expensive_false():
    """boundary is skipped when run_expensive=False; column_alignment now runs (cheap)."""
    verifier = QueryVerifier()
    specs = [
        VerificationTestSpec(
            test_type="column_alignment",
            description="SELECT columns answer the question",
            # No expected_column_count → will produce 'skip' status (not an error)
            expected_outcome="Columns are relevant",
            fix_hint="Review SELECT list",
        ),
        VerificationTestSpec(
            test_type="boundary",
            description="Date constraints match the question",
            expected_outcome="Date filter is correct",
            fix_hint="Adjust the date range",
        ),
        VerificationTestSpec(
            test_type="ordering",
            description="Needs ORDER BY and LIMIT",
            required_sql_keywords=["ORDER BY", "LIMIT"],
            expected_outcome="ORDER BY + LIMIT present",
            fix_hint="Add ORDER BY col LIMIT N",
        ),
    ]
    exec_result = _exec_result([(1, "Alice")])

    # Mock client for expensive tests — should NOT be called
    mock_client = _make_mock_client("PASS\nOK")

    with patch("src.verification.query_verifier.get_client", return_value=mock_client):
        eval_result = await verifier.evaluate_candidate(
            specs=specs,
            candidate_id="test_skip_expensive",
            sql="SELECT id, name FROM students ORDER BY id LIMIT 5",
            exec_result=exec_result,
            db_path="/dev/null",
            run_expensive=False,
        )

    evaluated_types = {r.test_type for r in eval_result.test_results}
    # ordering and column_alignment are cheap — they run
    assert "ordering" in evaluated_types
    assert "column_alignment" in evaluated_types
    # boundary is still expensive — it is skipped
    assert "boundary" not in evaluated_types
    # LLM should not have been called (column_alignment is now cheap)
    mock_client.generate.assert_not_called()
    # column_alignment has no expected_column_count → status is 'skip'
    ca_result = next(r for r in eval_result.test_results if r.test_type == "column_alignment")
    assert ca_result.status == "skip"


# ---------------------------------------------------------------------------
# Test 12: adjustment computation — critical test fail
# ---------------------------------------------------------------------------

def test_adjustment_computation_critical_fail():
    """Critical test failure yields -0.3 confidence adjustment."""
    verifier = QueryVerifier()
    results = [
        VerificationTestResult(
            test_type="grain",
            status="fail",
            actual_outcome="Wrong count",
            is_critical=True,
        ),
    ]
    adj = verifier._compute_adjustment(results)
    assert abs(adj - (-0.3)) < 1e-6, f"Expected -0.3, got {adj}"


# ---------------------------------------------------------------------------
# Test 13: adjustment computation — all tests pass → bonus
# ---------------------------------------------------------------------------

def test_adjustment_computation_all_pass_bonus():
    """All tests passing yields +0.2 bonus (capped)."""
    verifier = QueryVerifier()
    results = [
        VerificationTestResult(
            test_type="ordering",
            status="pass",
            actual_outcome="OK",
            is_critical=False,
        ),
        VerificationTestResult(
            test_type="null",
            status="pass",
            actual_outcome="OK",
            is_critical=False,
        ),
    ]
    adj = verifier._compute_adjustment(results)
    assert adj > 0.0, f"Expected positive bonus, got {adj}"
    assert adj <= 0.2, f"Bonus should be capped at 0.2, got {adj}"


# ---------------------------------------------------------------------------
# Test 14: evaluation error → "error" status, no penalty
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_evaluation_error_becomes_error_no_penalty():
    """If a test evaluation raises an exception, status is 'error' and no penalty."""
    verifier = QueryVerifier()

    # grain test with a verification_sql that will fail (invalid SQL)
    spec = VerificationTestSpec(
        test_type="grain",
        description="Count test",
        verification_sql="SELECT COUNT(*) FROM nonexistent_table_xyz",
        expected_outcome="3 rows",
        fix_hint="Fix the query",
    )
    db_path = _make_temp_db()
    try:
        exec_result = execute_sql(db_path, "SELECT id FROM students")
        eval_result = await verifier.evaluate_candidate(
            specs=[spec],
            candidate_id="test_error",
            sql="SELECT id FROM students",
            exec_result=exec_result,
            db_path=db_path,
            run_expensive=False,
        )
        # The grain test should get status="fail" (table doesn't exist → verification
        # SQL fails → vr.success=False → falls through to no-sql path → skip)
        # or "skip" — either way, all_pass should be True (no "fail" result)
        statuses = {r.status for r in eval_result.test_results}
        # Confidence adjustment should not add a critical penalty (either skip or error)
        assert eval_result.confidence_adjustment >= -0.1, (
            f"Error/skip should not apply critical penalty, got {eval_result.confidence_adjustment}"
        )
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test 15: grain range PASS — actual within [min, max]
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_grain_range_pass():
    """Grain test passes when actual row count is within [row_count_min, row_count_max]."""
    verifier = QueryVerifier()
    spec = VerificationTestSpec(
        test_type="grain",
        description="Between 2 and 5 rows expected",
        row_count_min=2,
        row_count_max=5,
        expected_outcome="Result has 2-5 rows",
        fix_hint="Review GROUP BY",
    )
    exec_result = _exec_result([(1,), (2,), (3,)])  # 3 rows

    eval_result = await verifier.evaluate_candidate(
        specs=[spec],
        candidate_id="test_range_pass",
        sql="SELECT id FROM students",
        exec_result=exec_result,
        db_path="/dev/null",
        run_expensive=False,
    )

    assert eval_result.all_pass is True
    grain_result = eval_result.test_results[0]
    assert grain_result.status == "pass"
    assert "within range [2, 5]" in grain_result.actual_outcome


# ---------------------------------------------------------------------------
# Test 16: grain range FAIL — below minimum (0 rows)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_grain_range_fail_below_min():
    """Grain test fails when result has 0 rows (always below minimum of 1)."""
    verifier = QueryVerifier()
    spec = VerificationTestSpec(
        test_type="grain",
        description="At least 1 row expected",
        row_count_min=1,
        row_count_max=5,
        expected_outcome="Non-empty result",
        fix_hint="Check WHERE clause — may filter out all rows",
    )
    exec_result = _exec_result([])  # 0 rows

    eval_result = await verifier.evaluate_candidate(
        specs=[spec],
        candidate_id="test_range_fail_below",
        sql="SELECT id FROM students WHERE 1=0",
        exec_result=exec_result,
        db_path="/dev/null",
        run_expensive=False,
    )

    assert eval_result.all_pass is False
    grain_result = eval_result.test_results[0]
    assert grain_result.status == "fail"
    assert "below minimum 1" in grain_result.actual_outcome


# ---------------------------------------------------------------------------
# Test 17: grain range FAIL — actual exceeds upper bound from verification_sql_upper
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_grain_range_fail_above_max():
    """Grain test fails when actual row count exceeds the upper bound from verification_sql_upper."""
    db_path = _make_temp_db()
    try:
        verifier = QueryVerifier()
        spec = VerificationTestSpec(
            test_type="grain",
            description="Result rows should not exceed number of distinct students",
            verification_sql_upper="SELECT COUNT(DISTINCT id) FROM students",  # → 3
            expected_outcome="At most 3 rows",
            fix_hint="Use GROUP BY student_id or DISTINCT to avoid JOIN multiplication",
        )
        # JOIN produces 6 rows (2 courses × 3 students)
        exec_result = execute_sql(
            db_path,
            "SELECT s.id, s.name FROM students s JOIN enrollments e ON s.id = e.student_id",
        )
        assert exec_result.success and len(exec_result.rows) == 6

        eval_result = await verifier.evaluate_candidate(
            specs=[spec],
            candidate_id="test_range_fail_above",
            sql="SELECT s.id, s.name FROM students s JOIN enrollments e ON s.id = e.student_id",
            exec_result=exec_result,
            db_path=db_path,
            run_expensive=False,
        )

        assert eval_result.all_pass is False
        grain_result = eval_result.test_results[0]
        assert grain_result.status == "fail"
        assert "6" in grain_result.actual_outcome
        assert "3" in grain_result.actual_outcome
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test 18: grain column count PASS
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_grain_column_count_pass():
    """Grain test passes when column count matches expected_column_count."""
    verifier = QueryVerifier()
    spec = VerificationTestSpec(
        test_type="grain",
        description="Expect 2 columns: id and name",
        row_count_min=1,
        expected_column_count=2,
        expected_outcome="2-column result",
        fix_hint="Check SELECT list",
    )
    exec_result = _exec_result([(1, "Alice"), (2, "Bob"), (3, "Carol")])  # 3 rows, 2 cols

    eval_result = await verifier.evaluate_candidate(
        specs=[spec],
        candidate_id="test_col_pass",
        sql="SELECT id, name FROM students",
        exec_result=exec_result,
        db_path="/dev/null",
        run_expensive=False,
    )

    assert eval_result.all_pass is True
    grain_result = eval_result.test_results[0]
    assert grain_result.status == "pass"
    assert "2" in grain_result.actual_outcome


# ---------------------------------------------------------------------------
# Test 19: grain column count FAIL
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_grain_column_count_fail():
    """Grain test fails when actual column count differs from expected_column_count."""
    verifier = QueryVerifier()
    spec = VerificationTestSpec(
        test_type="grain",
        description="Expect 3 columns but query returns 2",
        row_count_min=1,
        expected_column_count=3,
        expected_outcome="3-column result",
        fix_hint="Add missing column to SELECT",
    )
    exec_result = _exec_result([(1, "Alice"), (2, "Bob")])  # 2 rows, 2 cols

    eval_result = await verifier.evaluate_candidate(
        specs=[spec],
        candidate_id="test_col_fail",
        sql="SELECT id, name FROM students",
        exec_result=exec_result,
        db_path="/dev/null",
        run_expensive=False,
    )

    assert eval_result.all_pass is False
    grain_result = eval_result.test_results[0]
    assert grain_result.status == "fail"
    assert "2" in grain_result.actual_outcome   # actual cols
    assert "3" in grain_result.actual_outcome   # expected cols


# ---------------------------------------------------------------------------
# Test 20: grain COMBINED FAIL — both row count and column count wrong
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_grain_combined_fail():
    """Grain test fails with combined message when both row count and column count are wrong."""
    verifier = QueryVerifier()
    spec = VerificationTestSpec(
        test_type="grain",
        description="Expect ≤2 rows and 3 columns",
        row_count_max=2,
        expected_column_count=3,
        expected_outcome="≤2 rows, 3 cols",
        fix_hint="Fix both grain and SELECT list",
    )
    # 5 rows with 1 column each — violates both constraints
    exec_result = _exec_result([(x,) for x in range(5)])

    eval_result = await verifier.evaluate_candidate(
        specs=[spec],
        candidate_id="test_combined_fail",
        sql="SELECT id FROM students",
        exec_result=exec_result,
        db_path="/dev/null",
        run_expensive=False,
    )

    assert eval_result.all_pass is False
    grain_result = eval_result.test_results[0]
    assert grain_result.status == "fail"
    # Both violations appear in the combined message
    assert "|" in grain_result.actual_outcome
    assert "5" in grain_result.actual_outcome   # actual row count
    assert "2" in grain_result.actual_outcome   # row_count_max


# ---------------------------------------------------------------------------
# Test 21: grain COMBINED PASS — both row count and column count correct
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_grain_combined_pass():
    """Grain test passes when both row count and column count constraints are met."""
    verifier = QueryVerifier()
    spec = VerificationTestSpec(
        test_type="grain",
        description="Expect 1-5 rows and 2 columns",
        row_count_min=1,
        row_count_max=5,
        expected_column_count=2,
        expected_outcome="1-5 rows, 2 cols",
        fix_hint="Fix SELECT list or GROUP BY",
    )
    exec_result = _exec_result([(1, "Alice"), (2, "Bob"), (3, "Carol")])  # 3 rows, 2 cols

    eval_result = await verifier.evaluate_candidate(
        specs=[spec],
        candidate_id="test_combined_pass",
        sql="SELECT id, name FROM students",
        exec_result=exec_result,
        db_path="/dev/null",
        run_expensive=False,
    )

    assert eval_result.all_pass is True
    grain_result = eval_result.test_results[0]
    assert grain_result.status == "pass"
    assert "within range [1, 5]" in grain_result.actual_outcome
    assert "Column count 2" in grain_result.actual_outcome


# ---------------------------------------------------------------------------
# Test 22: grain backward compat — old verification_sql treated as upper bound
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grain_backward_compat_verification_sql():
    """Old verification_sql is treated as upper bound (backward compat path)."""
    db_path = _make_temp_db()
    try:
        verifier = QueryVerifier()
        # Only set verification_sql (old-style) — no new fields
        spec = VerificationTestSpec(
            test_type="grain",
            description="One row per student",
            verification_sql="SELECT COUNT(DISTINCT id) FROM students",  # → 3
            expected_outcome="3 rows",
            fix_hint="Add GROUP BY student_id",
        )
        exec_result = execute_sql(db_path, "SELECT id, name FROM students")
        assert exec_result.success and len(exec_result.rows) == 3

        eval_result = await verifier.evaluate_candidate(
            specs=[spec],
            candidate_id="test_compat",
            sql="SELECT id, name FROM students",
            exec_result=exec_result,
            db_path=db_path,
            run_expensive=False,
        )

        assert eval_result.all_pass is True
        grain_result = eval_result.test_results[0]
        assert grain_result.status == "pass"
        # Compat path: range [1, 3], actual=3 → pass
        assert "3" in grain_result.actual_outcome
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test 23: symmetry test PASS — verification SQL matches main result
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_symmetry_test_pass():
    """Symmetry test passes when the candidate result matches the verification SQL total."""
    db_path = _make_temp_db()
    try:
        verifier = QueryVerifier()
        # verification_sql counts all 6 enrollments; candidate counts them too → match
        spec = VerificationTestSpec(
            test_type="symmetry",
            description="Total enrollment count should equal sum of per-student counts",
            verification_sql="SELECT COUNT(*) FROM enrollments",
            expected_outcome="Main result equals 6",
            fix_hint="Check that the GROUP BY / aggregation is correct",
        )
        exec_result = execute_sql(db_path, "SELECT COUNT(*) FROM enrollments")
        assert exec_result.success and exec_result.rows[0][0] == 6

        eval_result = await verifier.evaluate_candidate(
            specs=[spec],
            candidate_id="test_symmetry_pass",
            sql="SELECT COUNT(*) FROM enrollments",
            exec_result=exec_result,
            db_path=db_path,
            run_expensive=True,
        )

        assert eval_result.all_pass is True
        sym_result = eval_result.test_results[0]
        assert sym_result.test_type == "symmetry"
        assert sym_result.status == "pass"
        assert "6" in sym_result.actual_outcome
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test 24: symmetry test FAIL — verification SQL disagrees with main result
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_symmetry_test_fail():
    """Symmetry test fails when the candidate result does not match verification SQL."""
    db_path = _make_temp_db()
    try:
        verifier = QueryVerifier()
        # Candidate says 3 (students), verification_sql says 6 (enrollments) → mismatch
        spec = VerificationTestSpec(
            test_type="symmetry",
            description="Total should match sub-group sum",
            verification_sql="SELECT COUNT(*) FROM enrollments",  # → 6
            expected_outcome="Result equals 6",
            fix_hint="Recheck aggregation — may be joining wrong table",
        )
        exec_result = execute_sql(db_path, "SELECT COUNT(*) FROM students")  # → 3
        assert exec_result.success and exec_result.rows[0][0] == 3

        eval_result = await verifier.evaluate_candidate(
            specs=[spec],
            candidate_id="test_symmetry_fail",
            sql="SELECT COUNT(*) FROM students",
            exec_result=exec_result,
            db_path=db_path,
            run_expensive=True,
        )

        assert eval_result.all_pass is False
        sym_result = eval_result.test_results[0]
        assert sym_result.test_type == "symmetry"
        assert sym_result.status == "fail"
        assert "3" in sym_result.actual_outcome
        assert "6" in sym_result.actual_outcome
        assert len(eval_result.failure_hints) >= 1
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test 25: symmetry test SKIP — no verification_sql provided
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_symmetry_test_skip_no_sql():
    """Symmetry test returns skip when no verification_sql is provided."""
    verifier = QueryVerifier()
    spec = VerificationTestSpec(
        test_type="symmetry",
        description="Total should match sub-group sum",
        # verification_sql intentionally omitted
        expected_outcome="Values match",
        fix_hint="Check aggregation",
    )
    exec_result = _exec_result([(42,)])

    eval_result = await verifier.evaluate_candidate(
        specs=[spec],
        candidate_id="test_symmetry_skip",
        sql="SELECT 42",
        exec_result=exec_result,
        db_path="/dev/null",
        run_expensive=True,
    )

    sym_result = eval_result.test_results[0]
    assert sym_result.status == "skip"
    assert eval_result.all_pass is True  # skip counts as passing


# ---------------------------------------------------------------------------
# Test 26: expensive tests (boundary, symmetry) run when run_expensive=True;
#          column_alignment (now cheap) runs regardless and makes no LLM call
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_expensive_tests_all_run_when_run_expensive_true():
    """boundary and symmetry are evaluated when run_expensive=True.
    column_alignment is now cheap (structural) — no LLM call, runs always.
    """
    db_path = _make_temp_db()
    try:
        verifier = QueryVerifier()
        specs = [
            VerificationTestSpec(
                test_type="column_alignment",
                description="Columns answer the question",
                expected_column_count=1,  # COUNT(*) returns 1 column → pass
                expected_outcome="1 column returned",
                fix_hint="Review SELECT list",
            ),
            VerificationTestSpec(
                test_type="boundary",
                description="Date filter is correct",
                expected_outcome="Date filter matches",
                fix_hint="Adjust date range",
            ),
            VerificationTestSpec(
                test_type="symmetry",
                description="Total matches sub-group sum",
                verification_sql="SELECT COUNT(*) FROM students",  # → 3
                expected_outcome="Values match",
                fix_hint="Check aggregation",
            ),
        ]
        exec_result = execute_sql(db_path, "SELECT COUNT(*) FROM students")  # → 3

        mock_client = _make_mock_client("PASS\nLooks good.")

        with patch("src.verification.query_verifier.get_client", return_value=mock_client):
            eval_result = await verifier.evaluate_candidate(
                specs=specs,
                candidate_id="test_expensive_run",
                sql="SELECT COUNT(*) FROM students",
                exec_result=exec_result,
                db_path=db_path,
                run_expensive=True,
            )

        evaluated_types = {r.test_type for r in eval_result.test_results}
        assert "column_alignment" in evaluated_types
        assert "boundary" in evaluated_types
        assert "symmetry" in evaluated_types
        # column_alignment is now cheap — only boundary calls the LLM (symmetry uses SQL)
        assert mock_client.generate.call_count == 1
        # column_alignment should pass (1 column returned, expected 1)
        ca_result = next(r for r in eval_result.test_results if r.test_type == "column_alignment")
        assert ca_result.status == "pass"
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Tests A-E: column_alignment structural column-count tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_column_alignment_pass_structural():
    """column_alignment passes when actual column count equals expected_column_count."""
    verifier = QueryVerifier()
    spec = VerificationTestSpec(
        test_type="column_alignment",
        description="Scalar aggregate — expect 1 column",
        expected_column_count=1,
        expected_outcome="1 column returned",
        fix_hint="Return only the aggregate value",
    )
    exec_result = _exec_result([(42,)])  # 1 column

    eval_result = await verifier.evaluate_candidate(
        specs=[spec],
        candidate_id="test_col_align_pass",
        sql="SELECT COUNT(*) FROM students",
        exec_result=exec_result,
        db_path="/dev/null",
        run_expensive=False,  # should run anyway — now cheap
    )

    assert eval_result.all_pass is True
    ca_result = eval_result.test_results[0]
    assert ca_result.status == "pass"
    assert "1" in ca_result.actual_outcome


@pytest.mark.asyncio
async def test_column_alignment_fail_structural():
    """column_alignment fails when actual column count differs from expected."""
    verifier = QueryVerifier()
    spec = VerificationTestSpec(
        test_type="column_alignment",
        description="Scalar aggregate — expect 1 column",
        expected_column_count=1,
        expected_outcome="1 column returned",
        fix_hint="Remove extra columns from SELECT",
    )
    exec_result = _exec_result([(1, "Alice", 3.9)])  # 3 columns, expected 1

    eval_result = await verifier.evaluate_candidate(
        specs=[spec],
        candidate_id="test_col_align_fail",
        sql="SELECT id, name, gpa FROM students",
        exec_result=exec_result,
        db_path="/dev/null",
        run_expensive=False,
    )

    assert eval_result.all_pass is False
    ca_result = eval_result.test_results[0]
    assert ca_result.status == "fail"
    assert "3" in ca_result.actual_outcome   # actual count
    assert "1" in ca_result.actual_outcome   # expected count
    assert len(eval_result.failure_hints) >= 1


@pytest.mark.asyncio
async def test_column_alignment_skip_when_no_expected_count():
    """column_alignment skips when expected_column_count is not set."""
    verifier = QueryVerifier()
    spec = VerificationTestSpec(
        test_type="column_alignment",
        description="Columns answer the question",
        # expected_column_count intentionally omitted
        expected_outcome="Correct columns",
        fix_hint="Review SELECT list",
    )
    exec_result = _exec_result([(1, "Alice")])

    eval_result = await verifier.evaluate_candidate(
        specs=[spec],
        candidate_id="test_col_align_skip",
        sql="SELECT id, name FROM students",
        exec_result=exec_result,
        db_path="/dev/null",
        run_expensive=False,
    )

    ca_result = eval_result.test_results[0]
    assert ca_result.status == "skip"
    assert eval_result.all_pass is True  # skip counts as passing


@pytest.mark.asyncio
async def test_column_alignment_skip_on_empty_result():
    """column_alignment skips when exec_result has no rows."""
    verifier = QueryVerifier()
    spec = VerificationTestSpec(
        test_type="column_alignment",
        description="Expect 1 column",
        expected_column_count=1,
        expected_outcome="1 column",
        fix_hint="Fix query",
    )
    exec_result = _exec_result([])  # no rows

    eval_result = await verifier.evaluate_candidate(
        specs=[spec],
        candidate_id="test_col_align_empty",
        sql="SELECT COUNT(*) FROM students WHERE 1=0",
        exec_result=exec_result,
        db_path="/dev/null",
        run_expensive=False,
    )

    ca_result = eval_result.test_results[0]
    assert ca_result.status == "skip"


@pytest.mark.asyncio
async def test_column_alignment_runs_when_run_expensive_false():
    """column_alignment runs even with run_expensive=False (it is now a cheap test)."""
    verifier = QueryVerifier()
    spec = VerificationTestSpec(
        test_type="column_alignment",
        description="Expect 2 columns",
        expected_column_count=2,
        expected_outcome="2 columns",
        fix_hint="Add missing column",
    )
    exec_result = _exec_result([(1, "Alice"), (2, "Bob")])  # 2 cols per row

    # No mock needed — no LLM call expected
    eval_result = await verifier.evaluate_candidate(
        specs=[spec],
        candidate_id="test_col_align_cheap",
        sql="SELECT id, name FROM students",
        exec_result=exec_result,
        db_path="/dev/null",
        run_expensive=False,
    )

    evaluated_types = {r.test_type for r in eval_result.test_results}
    assert "column_alignment" in evaluated_types
    ca_result = next(r for r in eval_result.test_results if r.test_type == "column_alignment")
    assert ca_result.status == "pass"


# ---------------------------------------------------------------------------
# Test 27: trace entry verif_run_expensive is always True in QueryFixer
# ---------------------------------------------------------------------------

def test_trace_verif_run_expensive_always_true():
    """QueryFixer trace entries always report verif_run_expensive=True.

    This was previously set to is_final_assessment, which was wrong since
    QueryFixer always passes run_expensive=True to evaluate_candidate.
    """
    from src.fixing.query_fixer import _build_iter_entry
    from src.data.database import ExecutionResult

    exec_result = ExecutionResult(
        success=True, rows=[(1,)], error=None, execution_time=0.001, is_empty=False
    )

    # Non-final iteration — should still be True
    entry = _build_iter_entry(
        iteration=0,
        is_final_assessment=False,
        sql="SELECT 1",
        prev_sql=None,
        exec_result=exec_result,
        verif_eval=None,
        exec_issues_sent=None,
        verif_issues_sent=None,
        fix_triggered=False,
        fix_produced_different_sql=None,
    )
    assert entry["verif_run_expensive"] is True

    # Final iteration — also True
    entry_final = _build_iter_entry(
        iteration=2,
        is_final_assessment=True,
        sql="SELECT 1",
        prev_sql=None,
        exec_result=exec_result,
        verif_eval=None,
        exec_issues_sent=None,
        verif_issues_sent=None,
        fix_triggered=False,
        fix_produced_different_sql=None,
    )
    assert entry_final["verif_run_expensive"] is True


# ---------------------------------------------------------------------------
# New tests: computed_upper_bound / actual_row_count populated on grain result
# ---------------------------------------------------------------------------

def test_grain_result_has_computed_upper_bound_and_actual_row_count():
    """_eval_grain populates computed_upper_bound and actual_row_count on fail."""
    db = _make_temp_db()
    try:
        verifier = QueryVerifier()
        spec = VerificationTestSpec(
            test_type="grain",
            description="Should return at most 3 rows (one per student).",
            # verification_sql_upper: SELECT COUNT(DISTINCT id) FROM students → 3
            verification_sql_upper="SELECT COUNT(DISTINCT id) FROM students",
            expected_outcome="Row count ≤ 3",
            fix_hint="Add DISTINCT or GROUP BY student id.",
        )
        # Simulate a bad query returning 6 rows (JOIN multiplication)
        exec_result = _exec_result([(1, "Alice"), (1, "Alice"), (2, "Bob"),
                                    (2, "Bob"), (3, "Carol"), (3, "Carol")])
        result = verifier._eval_grain(spec, exec_result, db)
        assert result.status == "fail"
        assert result.computed_upper_bound == 3   # 3 distinct students
        assert result.actual_row_count == 6
    finally:
        import os; os.unlink(db)


def test_grain_result_has_computed_upper_bound_on_pass():
    """_eval_grain also populates fields when the test passes."""
    db = _make_temp_db()
    try:
        verifier = QueryVerifier()
        spec = VerificationTestSpec(
            test_type="grain",
            description="One row per student.",
            verification_sql_upper="SELECT COUNT(DISTINCT id) FROM students",
            expected_outcome="Row count ≤ 3",
            fix_hint="Add GROUP BY.",
        )
        exec_result = _exec_result([(1, "Alice"), (2, "Bob"), (3, "Carol")])
        result = verifier._eval_grain(spec, exec_result, db)
        assert result.status == "pass"
        assert result.computed_upper_bound == 3
        assert result.actual_row_count == 3
    finally:
        import os; os.unlink(db)


# ---------------------------------------------------------------------------
# New test: fix prompt includes numeric bound values and duplicate-row hint
# ---------------------------------------------------------------------------

def test_fix_prompt_grain_shows_numeric_bounds_and_duplicate_hint():
    """_build_fix_prompt enriches grain failure with numeric values and dup hint."""
    from src.fixing.query_fixer import QueryFixer
    from src.schema_linking.schema_linker import LinkedSchemas
    from dataclasses import dataclass

    @dataclass
    class _FakeSchemas:
        s2_ddl: str = "CREATE TABLE students (id INT, name TEXT);"

    # Build a grain VerificationTestResult with computed fields populated
    grain_result = VerificationTestResult(
        test_type="grain",
        status="fail",
        actual_outcome="Row count 6 exceeds upper bound 3.",
        is_critical=True,
        computed_upper_bound=3,
        actual_row_count=6,
    )
    verif_eval = VerificationEvaluation(
        candidate_id="gen_a_0",
        test_results=[grain_result],
        all_pass=False,
        confidence_adjustment=-0.3,
        failure_hints=["Grain failed"],
    )
    grain_spec = VerificationTestSpec(
        test_type="grain",
        description="One row per student.",
        verification_sql_upper="SELECT COUNT(DISTINCT id) FROM students",
        expected_outcome="Row count ≤ 3",
        fix_hint="Add DISTINCT.",
    )

    # Duplicate rows in result sample
    exec_result = _exec_result([
        (1, "Alice"), (1, "Alice"), (2, "Bob"),
        (2, "Bob"), (3, "Carol"), (3, "Carol"),
    ])

    prompt = QueryFixer._build_fix_prompt(
        sql="SELECT s.id, s.name FROM students s JOIN enrollments e ON s.id = e.student_id",
        question="List all students.",
        evidence="",
        schemas=_FakeSchemas(),
        cell_matches=[],
        exec_issues=[],
        verif_eval=verif_eval,
        verif_specs=[grain_spec],
        exec_result=exec_result,
    )

    # Numeric bound values must appear
    assert "Upper bound value: 3 rows" in prompt
    assert "Your query returned: 6 rows" in prompt
    assert "3 extra rows" in prompt or "extra rows" in prompt
    # The upper bound SQL must appear
    assert "SELECT COUNT(DISTINCT id) FROM students" in prompt
    # Duplicate-row hint must appear because sample has identical rows
    assert "identical rows" in prompt
    assert "duplicated" in prompt


# ---------------------------------------------------------------------------
# Tests for ordering helper functions (_extract_limit_from_question,
# _derive_direction_from_question)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question,expected_n", [
    ("Who are the top 5 students by GPA?", 5),
    ("List the top 1 student.", 1),
    ("Find the 3 highest scoring players.", 3),
    ("What are the lowest 2 prices?", 2),
    ("List the bottom 10 products by rating.", 10),
    ("Which student has the highest GPA?", None),   # no N
    ("What is the average score?", None),            # no ordering at all
])
def test_extract_limit_from_question(question, expected_n):
    assert _extract_limit_from_question(question) == expected_n


@pytest.mark.parametrize("question,expected_dir", [
    ("Who are the top 5 students by GPA?", "DESC"),
    ("Which student has the highest score?", "DESC"),
    ("Find the best 3 players.", "DESC"),
    ("What are the 3 lowest prices?", "ASC"),
    ("List the worst performing teams.", "ASC"),
    ("Find the smallest 2 values.", "ASC"),
    ("How many students enrolled?", None),   # ambiguous / no ordering keywords
])
def test_derive_direction_from_question(question, expected_dir):
    assert _derive_direction_from_question(question) == expected_dir
