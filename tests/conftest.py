"""
Shared pytest fixtures for the NL2SQL test suite.
"""

import sqlite3

import pytest

from src.data.bird_loader import BirdEntry
from src.evaluation.evaluator import EvaluationResult


# ---------------------------------------------------------------------------
# Database fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def students_db(tmp_path):
    """
    A small in-memory-style SQLite database written to a temp file.

    Tables
    ------
    countries (code PK, name)
    students  (id PK, name, age, gpa, country FK→countries.code)

    Sample rows include NULL values and duplicate names to exercise edge cases.
    """
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE countries (
            code TEXT PRIMARY KEY,
            name TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE students (
            id      INTEGER PRIMARY KEY,
            name    TEXT,
            age     INTEGER,
            gpa     REAL,
            country TEXT,
            FOREIGN KEY (country) REFERENCES countries(code)
        )
    """)
    conn.execute("INSERT INTO students VALUES (1, 'Alice', 20, 3.9, 'USA')")
    conn.execute("INSERT INTO students VALUES (2, 'Bob',   22, NULL, 'UK')")
    conn.execute("INSERT INTO students VALUES (3, 'Alice', 21, 3.5, 'USA')")
    conn.execute("INSERT INTO students VALUES (4, NULL,    25, 2.8, NULL)")
    conn.commit()
    conn.close()
    return str(db_path)


# ---------------------------------------------------------------------------
# BirdEntry fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_bird_entry():
    """A representative BirdEntry for unit tests."""
    return BirdEntry(
        question_id=0,
        db_id="california_schools",
        question="What is the highest eligible free rate for K-12 students?",
        evidence="Eligible free rate = Free Meal Count (K-12) / Enrollment (K-12)",
        SQL="SELECT MAX(`Free Meal Count (K-12)` / `Enrollment (K-12)`) FROM frpm",
        difficulty="simple",
    )


# ---------------------------------------------------------------------------
# Preprocessed directory fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def preprocessed_dir(tmp_path):
    """A temporary directory tree mimicking the preprocessed data layout."""
    for subdir in ["profiles", "summaries", "schemas", "indices"]:
        (tmp_path / subdir).mkdir()
    return str(tmp_path)


# ---------------------------------------------------------------------------
# EvaluationResult fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_evaluation_results():
    """
    A list of 10 EvaluationResult objects with varied attributes for unit
    testing of the metrics module.

    Coverage:
    - correct: 6 True, 4 False  → overall_ex = 0.6
    - difficulties: simple(4), moderate(3), challenging(3)
    - db_ids: california_schools(4), financial(3), superhero(3)
    - selection_methods: fast_path(3), tournament(4), fallback(2), error(1)
    - winner_generators: reasoning_A1(3), standard_B1(3), icl_C1(3), fallback(1)
    - fix_count: varies from 0 to 3
    - cost_estimate: varies from 0.0 to 0.05
    """
    return [
        EvaluationResult(
            question_id=1,
            db_id="california_schools",
            difficulty="simple",
            predicted_sql="SELECT MAX(gpa) FROM students",
            truth_sql="SELECT MAX(gpa) FROM students",
            correct=True,
            selection_method="fast_path",
            winner_generator="reasoning_A1",
            cluster_count=1,
            fix_count=0,
            latency_seconds=2.5,
            cost_estimate=0.01,
        ),
        EvaluationResult(
            question_id=2,
            db_id="california_schools",
            difficulty="moderate",
            predicted_sql="SELECT name FROM students WHERE gpa > 3.0",
            truth_sql="SELECT name FROM students WHERE gpa > 3.0",
            correct=True,
            selection_method="tournament",
            winner_generator="standard_B1",
            cluster_count=3,
            fix_count=1,
            latency_seconds=5.0,
            cost_estimate=0.02,
        ),
        EvaluationResult(
            question_id=3,
            db_id="california_schools",
            difficulty="challenging",
            predicted_sql="SELECT COUNT(*) FROM students",
            truth_sql="SELECT COUNT(DISTINCT name) FROM students",
            correct=False,
            selection_method="tournament",
            winner_generator="icl_C1",
            cluster_count=2,
            fix_count=2,
            latency_seconds=8.0,
            cost_estimate=0.03,
        ),
        EvaluationResult(
            question_id=4,
            db_id="california_schools",
            difficulty="simple",
            predicted_sql="SELECT id FROM students LIMIT 5",
            truth_sql="SELECT id FROM students LIMIT 5",
            correct=True,
            selection_method="fast_path",
            winner_generator="reasoning_A1",
            cluster_count=1,
            fix_count=0,
            latency_seconds=1.5,
            cost_estimate=0.005,
        ),
        EvaluationResult(
            question_id=5,
            db_id="financial",
            difficulty="moderate",
            predicted_sql="SELECT AVG(amount) FROM transactions",
            truth_sql="SELECT AVG(amount) FROM transactions WHERE type='credit'",
            correct=False,
            selection_method="tournament",
            winner_generator="standard_B1",
            cluster_count=3,
            fix_count=1,
            latency_seconds=6.0,
            cost_estimate=0.025,
        ),
        EvaluationResult(
            question_id=6,
            db_id="financial",
            difficulty="challenging",
            predicted_sql="",
            truth_sql="SELECT SUM(amount) FROM accounts GROUP BY type",
            correct=False,
            selection_method="error",
            winner_generator="fallback",
            cluster_count=0,
            fix_count=0,
            latency_seconds=0.5,
            cost_estimate=0.0,
        ),
        EvaluationResult(
            question_id=7,
            db_id="financial",
            difficulty="simple",
            predicted_sql="SELECT COUNT(*) FROM accounts",
            truth_sql="SELECT COUNT(*) FROM accounts",
            correct=True,
            selection_method="fallback",
            winner_generator="icl_C1",
            cluster_count=1,
            fix_count=0,
            latency_seconds=3.0,
            cost_estimate=0.015,
        ),
        EvaluationResult(
            question_id=8,
            db_id="superhero",
            difficulty="simple",
            predicted_sql="SELECT name FROM heroes WHERE power='flight'",
            truth_sql="SELECT name FROM heroes WHERE power='flight'",
            correct=True,
            selection_method="fast_path",
            winner_generator="reasoning_A1",
            cluster_count=1,
            fix_count=0,
            latency_seconds=2.0,
            cost_estimate=0.01,
        ),
        EvaluationResult(
            question_id=9,
            db_id="superhero",
            difficulty="moderate",
            predicted_sql="SELECT hero_name FROM heroes JOIN powers ON id=hero_id",
            truth_sql="SELECT DISTINCT hero_name FROM heroes JOIN powers ON id=hero_id",
            correct=False,
            selection_method="tournament",
            winner_generator="standard_B1",
            cluster_count=2,
            fix_count=3,
            latency_seconds=9.0,
            cost_estimate=0.05,
        ),
        EvaluationResult(
            question_id=10,
            db_id="superhero",
            difficulty="challenging",
            predicted_sql="SELECT h.name, COUNT(p.id) FROM heroes h LEFT JOIN powers p ON h.id=p.hero_id GROUP BY h.id",
            truth_sql="SELECT h.name, COUNT(p.id) FROM heroes h LEFT JOIN powers p ON h.id=p.hero_id GROUP BY h.id",
            correct=True,
            selection_method="fallback",
            winner_generator="icl_C1",
            cluster_count=2,
            fix_count=1,
            latency_seconds=7.5,
            cost_estimate=0.04,
        ),
    ]
