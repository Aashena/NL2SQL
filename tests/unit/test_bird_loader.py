"""
Unit tests for src/data/bird_loader.py

These tests use a small temporary JSON file — no real BIRD data required.
"""

import json
import os

import pytest

from src.data.bird_loader import BirdEntry, load_bird_split


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_DIFFICULTIES = {"simple", "moderate", "challenging"}

_SAMPLE_ENTRIES = [
    {
        "question_id": 0,
        "db_id": "california_schools",
        "question": "How many students graduated in 2020?",
        "evidence": "graduation year = 2020",
        "SQL": "SELECT COUNT(*) FROM students WHERE year = 2020",
        "difficulty": "simple",
    },
    {
        "question_id": 1,
        "db_id": "california_schools",
        "question": "What is the average GPA per school?",
        "evidence": "",
        "SQL": "SELECT school_id, AVG(gpa) FROM students GROUP BY school_id",
        "difficulty": "moderate",
    },
    {
        "question_id": 2,
        "db_id": "financial",
        "question": "List all transactions above $10,000.",
        "evidence": "high-value threshold is 10000",
        "SQL": "SELECT * FROM transactions WHERE amount > 10000",
        "difficulty": "challenging",
    },
]


@pytest.fixture
def mock_bird_dir(tmp_path):
    """Create a temporary directory with a synthetic 'dev' split JSON."""
    split_dir = tmp_path / "dev"
    split_dir.mkdir()
    json_path = split_dir / "dev.json"
    json_path.write_text(json.dumps(_SAMPLE_ENTRIES), encoding="utf-8")
    return str(tmp_path)


# ---------------------------------------------------------------------------
# Test 1 — returns list[BirdEntry]
# ---------------------------------------------------------------------------

def test_load_returns_list_of_bird_entries(mock_bird_dir):
    """load_bird_split should return a list of BirdEntry instances."""
    entries = load_bird_split("dev", data_dir=mock_bird_dir)

    assert isinstance(entries, list), "Expected a list"
    assert len(entries) == len(_SAMPLE_ENTRIES), (
        f"Expected {len(_SAMPLE_ENTRIES)} entries, got {len(entries)}"
    )
    for entry in entries:
        assert isinstance(entry, BirdEntry), f"Expected BirdEntry, got {type(entry)}"


# ---------------------------------------------------------------------------
# Test 2 — all required fields are populated and non-None
# ---------------------------------------------------------------------------

def test_all_fields_populated(mock_bird_dir):
    """Every required field must be present and non-None."""
    entries = load_bird_split("dev", data_dir=mock_bird_dir)
    required_fields = ["question_id", "db_id", "question", "evidence", "SQL", "difficulty"]

    for entry in entries:
        for field in required_fields:
            value = getattr(entry, field)
            assert value is not None, (
                f"Field '{field}' is None in entry {entry.question_id}"
            )


# ---------------------------------------------------------------------------
# Test 3 — db_id is a string
# ---------------------------------------------------------------------------

def test_db_id_is_string(mock_bird_dir):
    """db_id must be a str, not an int or other type."""
    entries = load_bird_split("dev", data_dir=mock_bird_dir)
    for entry in entries:
        assert isinstance(entry.db_id, str), (
            f"db_id is {type(entry.db_id)}, expected str"
        )


# ---------------------------------------------------------------------------
# Test 4 — difficulty is one of the valid values
# ---------------------------------------------------------------------------

def test_difficulty_in_valid_set(mock_bird_dir):
    """difficulty must be 'simple', 'moderate', or 'challenging'."""
    entries = load_bird_split("dev", data_dir=mock_bird_dir)
    for entry in entries:
        assert entry.difficulty in _VALID_DIFFICULTIES, (
            f"Unexpected difficulty {entry.difficulty!r}"
        )


# ---------------------------------------------------------------------------
# Test 5 — evidence can be an empty string
# ---------------------------------------------------------------------------

def test_evidence_can_be_empty_string(mock_bird_dir):
    """BirdEntry with evidence='' must be valid (no validation error)."""
    # Entry at index 1 in _SAMPLE_ENTRIES has evidence=""
    entries = load_bird_split("dev", data_dir=mock_bird_dir)
    empty_evidence_entries = [e for e in entries if e.evidence == ""]
    assert len(empty_evidence_entries) >= 1, (
        "Expected at least one entry with empty evidence"
    )
    for entry in empty_evidence_entries:
        assert isinstance(entry.evidence, str)
        assert entry.evidence == ""


# ---------------------------------------------------------------------------
# Bonus — missing file returns empty list (not exception)
# ---------------------------------------------------------------------------

def test_missing_file_returns_empty_list(tmp_path):
    """load_bird_split must return [] when the JSON file does not exist."""
    result = load_bird_split("dev", data_dir=str(tmp_path))
    assert result == []
