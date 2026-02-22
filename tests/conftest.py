"""
Shared pytest fixtures for the NL2SQL test suite.
"""

import sqlite3

import pytest

from src.data.bird_loader import BirdEntry


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
