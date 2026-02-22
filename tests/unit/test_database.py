"""
Unit tests for src/data/database.py

Uses the `students_db` fixture (real SQLite file) from tests/conftest.py.
"""

import sqlite3
import time

import pytest

from src.data.database import ExecutionResult, execute_sql


# ---------------------------------------------------------------------------
# Test 1 — valid SELECT returns rows
# ---------------------------------------------------------------------------

def test_valid_select_returns_rows(students_db):
    """A well-formed SELECT should return success=True with non-empty rows."""
    result = execute_sql(students_db, "SELECT * FROM students")

    assert result.success is True, f"Expected success, got error: {result.error}"
    assert len(result.rows) > 0, "Expected non-empty rows"
    assert result.is_empty is False


# ---------------------------------------------------------------------------
# Test 2 — syntax error returns failure
# ---------------------------------------------------------------------------

def test_syntax_error_returns_failure(students_db):
    """A query with a syntax error must return success=False with an error message."""
    result = execute_sql(students_db, "SELECT * FORM students")  # typo: FORM

    assert result.success is False, "Expected failure for syntax error"
    assert result.error is not None and len(result.error) > 0, (
        "Expected a non-empty error message"
    )
    assert result.rows == []


# ---------------------------------------------------------------------------
# Test 3 — query returning no rows reports is_empty=True
# ---------------------------------------------------------------------------

def test_empty_result_returns_is_empty(students_db):
    """A valid query that matches no rows must set is_empty=True."""
    result = execute_sql(students_db, "SELECT * FROM students WHERE id = 9999")

    assert result.success is True, f"Unexpected error: {result.error}"
    assert result.is_empty is True
    assert result.rows == []


# ---------------------------------------------------------------------------
# Test 4 — execution time is measured
# ---------------------------------------------------------------------------

def test_execution_time_is_measured(students_db):
    """execution_time must be a positive float."""
    result = execute_sql(students_db, "SELECT * FROM students")

    assert isinstance(result.execution_time, float), (
        f"Expected float, got {type(result.execution_time)}"
    )
    assert result.execution_time > 0.0, "execution_time should be > 0"


# ---------------------------------------------------------------------------
# Test 5 — timeout kills a hanging query
# ---------------------------------------------------------------------------

def test_timeout_kills_hanging_query(students_db, tmp_path):
    """A very short timeout must cause a complex query to fail."""
    # Create a bigger DB to make the query take some time
    big_db = str(tmp_path / "big.db")
    conn = sqlite3.connect(big_db)
    conn.execute("CREATE TABLE nums (n INTEGER)")
    conn.executemany("INSERT INTO nums VALUES (?)", [(i,) for i in range(5000)])
    conn.commit()
    conn.close()

    # Cross join to create a slow query, with a near-zero timeout
    result = execute_sql(
        big_db,
        "SELECT a.n, b.n FROM nums a, nums b WHERE a.n + b.n > 0",
        timeout=0.001,
    )

    assert result.success is False, (
        "Expected timeout failure; query returned success=True unexpectedly"
    )


# ---------------------------------------------------------------------------
# Test 6 — cartesian product protection
# ---------------------------------------------------------------------------

def test_cartesian_product_protection(tmp_path):
    """
    A query that produces >10,000 rows without a LIMIT clause must be rejected.

    We create a table with 200 rows and cross-join it with itself → 40,000 rows.
    """
    db_path = str(tmp_path / "cross.db")
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE items (id INTEGER)")
    conn.executemany("INSERT INTO items VALUES (?)", [(i,) for i in range(200)])
    conn.commit()
    conn.close()

    # Cross join produces 200 × 200 = 40,000 rows — exceeds the 10,000 limit
    result = execute_sql(db_path, "SELECT a.id, b.id FROM items a, items b")

    assert result.success is False, (
        "Expected cartesian product protection to trigger; got success=True"
    )
    assert result.error is not None
    assert "cartesian" in result.error.lower() or "10,000" in result.error or "limit" in result.error.lower(), (
        f"Error message doesn't mention cartesian product or row limit: {result.error}"
    )
