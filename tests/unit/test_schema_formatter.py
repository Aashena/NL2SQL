"""
Tests for src/preprocessing/schema_formatter.py — Op 0c: Schema Formatter.

Uses a small in-memory profile + summary fixture that covers:
  - Two tables: students (3 cols) and countries (2 cols)
  - Primary key columns
  - Foreign key relationship
  - Column names with special characters
  - Sample values

All tests are pure unit tests — no database connections or API calls.
"""

import pytest

from src.preprocessing.profiler import ColumnProfile, DatabaseProfile
from src.preprocessing.summarizer import FieldSummary, DatabaseSummary
from src.preprocessing.schema_formatter import (
    FormattedSchemas,
    format_schemas,
    format_and_save_schemas,
    needs_quoting,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_test_profile() -> DatabaseProfile:
    """Small profile with two tables: students and countries."""
    cols = [
        ColumnProfile(
            "students", "id", "INTEGER",
            total_count=4, null_count=0, null_rate=0.0, distinct_count=4,
            sample_values=[["1", 1], ["2", 1]],
            min_value=1, max_value=4, avg_value=2.5,
            avg_length=None, is_primary_key=True, foreign_key_ref=None,
            minhash_bands=list(range(128)),
        ),
        ColumnProfile(
            "students", "name", "TEXT",
            total_count=4, null_count=1, null_rate=0.25, distinct_count=3,
            sample_values=[["Alice", 2], ["Bob", 1]],
            min_value=None, max_value=None, avg_value=None,
            avg_length=5.0, is_primary_key=False, foreign_key_ref=None,
            minhash_bands=list(range(128)),
        ),
        ColumnProfile(
            "students", "country", "TEXT",
            total_count=4, null_count=1, null_rate=0.25, distinct_count=2,
            sample_values=[["USA", 2], ["UK", 1]],
            min_value=None, max_value=None, avg_value=None,
            avg_length=3.0, is_primary_key=False, foreign_key_ref="countries.code",
            minhash_bands=list(range(128)),
        ),
        ColumnProfile(
            "countries", "code", "TEXT",
            total_count=3, null_count=0, null_rate=0.0, distinct_count=3,
            sample_values=[["USA", 2], ["UK", 1]],
            min_value=None, max_value=None, avg_value=None,
            avg_length=3.0, is_primary_key=True, foreign_key_ref=None,
            minhash_bands=list(range(128)),
        ),
        ColumnProfile(
            "countries", "name", "TEXT",
            total_count=3, null_count=0, null_rate=0.0, distinct_count=3,
            sample_values=[["United States", 1], ["United Kingdom", 1]],
            min_value=None, max_value=None, avg_value=None,
            avg_length=13.0, is_primary_key=False, foreign_key_ref=None,
            minhash_bands=list(range(128)),
        ),
    ]
    return DatabaseProfile(
        db_id="test_db",
        tables=["students", "countries"],
        columns=cols,
        foreign_keys=[("students", "country", "countries", "code")],
        total_tables=2,
        total_columns=5,
    )


def make_test_summary() -> DatabaseSummary:
    """Summaries matching the profile above."""
    summaries = [
        FieldSummary(
            "students", "id",
            "Unique student identifier.",
            "Auto-incremented integer primary key. Used in JOINs.",
        ),
        FieldSummary(
            "students", "name",
            "Student full name.",
            "Text name, can be NULL for anonymous records. Used in name filters.",
        ),
        FieldSummary(
            "students", "country",
            "Student country code.",
            "ISO 2-letter country code referencing countries table.",
        ),
        FieldSummary(
            "countries", "code",
            "ISO country code.",
            "2-letter ISO code. Primary key for countries lookup table.",
        ),
        FieldSummary(
            "countries", "name",
            "Full country name.",
            "Full English name of the country. E.g., United States.",
        ),
    ]
    return DatabaseSummary(db_id="test_db", field_summaries=summaries)


@pytest.fixture
def test_profile() -> DatabaseProfile:
    return make_test_profile()


@pytest.fixture
def test_summary() -> DatabaseSummary:
    return make_test_summary()


@pytest.fixture
def formatted(test_profile, test_summary) -> FormattedSchemas:
    return format_schemas(test_profile, test_summary)


# ---------------------------------------------------------------------------
# Test 1: DDL contains all tables
# ---------------------------------------------------------------------------

def test_ddl_contains_all_tables(formatted):
    """DDL output contains CREATE TABLE blocks for both tables."""
    assert "CREATE TABLE students" in formatted.ddl, (
        "DDL should contain 'CREATE TABLE students'"
    )
    assert "CREATE TABLE countries" in formatted.ddl, (
        "DDL should contain 'CREATE TABLE countries'"
    )


# ---------------------------------------------------------------------------
# Test 2: DDL contains all columns
# ---------------------------------------------------------------------------

def test_ddl_contains_all_columns(formatted):
    """Every column name appears somewhere in the DDL string."""
    for col_name in ("id", "name", "country", "code"):
        assert col_name in formatted.ddl, (
            f"Column '{col_name}' not found in DDL"
        )


# ---------------------------------------------------------------------------
# Test 3: DDL injects long summaries
# ---------------------------------------------------------------------------

def test_ddl_injects_summaries(formatted):
    """The long_summary text 'Auto-incremented integer primary key' appears in DDL."""
    assert "Auto-incremented integer primary key" in formatted.ddl, (
        "Expected long_summary text to appear in DDL column comment"
    )


# ---------------------------------------------------------------------------
# Test 4: DDL marks primary key columns
# ---------------------------------------------------------------------------

def test_ddl_primary_key_notation(formatted):
    """The DDL for the 'id' column includes PRIMARY KEY."""
    assert "PRIMARY KEY" in formatted.ddl, (
        "Expected 'PRIMARY KEY' notation in DDL for primary key columns"
    )
    # Also verify it's associated with 'id'
    lines = formatted.ddl.splitlines()
    pk_lines = [l for l in lines if "PRIMARY KEY" in l]
    assert any("id" in l for l in pk_lines), (
        "Expected PRIMARY KEY to appear on the 'id' column line"
    )


# ---------------------------------------------------------------------------
# Test 5: DDL contains foreign key comments
# ---------------------------------------------------------------------------

def test_ddl_foreign_key_comments(formatted):
    """DDL contains a 'Foreign keys:' comment mentioning countries.code."""
    assert "Foreign keys:" in formatted.ddl, (
        "Expected '-- Foreign keys:' comment in DDL"
    )
    assert "countries" in formatted.ddl, (
        "Expected 'countries' to appear in foreign key comment"
    )
    assert "code" in formatted.ddl, (
        "Expected 'code' to appear in foreign key comment"
    )


# ---------------------------------------------------------------------------
# Test 6: Markdown has header per table
# ---------------------------------------------------------------------------

def test_markdown_has_header_per_table(formatted):
    """Markdown contains '## Table:' headers for both tables."""
    assert "## Table: students" in formatted.markdown, (
        "Expected '## Table: students' header in Markdown"
    )
    assert "## Table: countries" in formatted.markdown, (
        "Expected '## Table: countries' header in Markdown"
    )


# ---------------------------------------------------------------------------
# Test 7: Markdown table row count for students
# ---------------------------------------------------------------------------

def test_markdown_table_row_count(formatted):
    """For the students table (3 columns), the markdown pipe table has exactly 3 data rows."""
    # Split the markdown by the students table section header
    # Find the students section and count data rows (lines starting with '| ')
    # that are not the header row or separator row.
    lines = formatted.markdown.splitlines()

    # Find the start of the students section
    in_students_section = False
    data_rows = 0
    for line in lines:
        if line.strip() == "## Table: students":
            in_students_section = True
            continue
        if in_students_section and line.strip().startswith("## Table:"):
            # Next table section started
            break
        if in_students_section and line.startswith("|"):
            # Check if it's a data row (not the header row or separator)
            # Header row contains "Column" in the first cell
            # Separator row contains "---" pattern
            if "---" not in line and "Column" not in line:
                data_rows += 1

    assert data_rows == 3, (
        f"Expected exactly 3 data rows for students table, got {data_rows}"
    )


# ---------------------------------------------------------------------------
# Test 8: Sample values truncation in Markdown
# ---------------------------------------------------------------------------

def test_sample_values_truncation():
    """Sample values longer than 30 chars are truncated with '...' in markdown."""
    long_value = "A very long value that exceeds thirty characters in length"
    assert len(long_value) > 30, "Precondition: test value must exceed 30 chars"

    # Build a minimal profile with this long value
    col = ColumnProfile(
        "t1", "col1", "TEXT",
        total_count=1, null_count=0, null_rate=0.0, distinct_count=1,
        sample_values=[[long_value, 1]],
        min_value=None, max_value=None, avg_value=None,
        avg_length=float(len(long_value)), is_primary_key=False,
        foreign_key_ref=None, minhash_bands=list(range(128)),
    )
    profile = DatabaseProfile(
        db_id="trunc_test", tables=["t1"], columns=[col],
        foreign_keys=[], total_tables=1, total_columns=1,
    )
    summary = DatabaseSummary(
        db_id="trunc_test",
        field_summaries=[
            FieldSummary("t1", "col1", "A column.", "A longer description.")
        ],
    )

    schemas = format_schemas(profile, summary)

    # The full long value should NOT appear in markdown
    assert long_value not in schemas.markdown, (
        "Full long sample value should not appear verbatim in Markdown"
    )
    # The truncated form (first 30 chars + "...") should appear
    truncated = long_value[:30] + "..."
    assert truncated in schemas.markdown, (
        f"Expected truncated value '{truncated}' in Markdown, "
        f"got:\n{schemas.markdown}"
    )


# ---------------------------------------------------------------------------
# Test 9: Special character column name quoting in DDL
# ---------------------------------------------------------------------------

def test_special_character_escaping():
    """Column names with spaces/parens are double-quoted in DDL output."""
    special_col_name = "Free Meal Count (K-12)"

    col = ColumnProfile(
        "frpm", special_col_name, "REAL",
        total_count=10, null_count=0, null_rate=0.0, distinct_count=10,
        sample_values=[[42.0, 1]],
        min_value=0.0, max_value=100.0, avg_value=50.0,
        avg_length=None, is_primary_key=False, foreign_key_ref=None,
        minhash_bands=list(range(128)),
    )
    profile = DatabaseProfile(
        db_id="special_test", tables=["frpm"], columns=[col],
        foreign_keys=[], total_tables=1, total_columns=1,
    )
    summary = DatabaseSummary(
        db_id="special_test",
        field_summaries=[
            FieldSummary("frpm", special_col_name, "Meal count.", "Count of students eligible for free meals.")
        ],
    )

    schemas = format_schemas(profile, summary)

    # The column name should appear quoted in DDL
    quoted_name = f'"{special_col_name}"'
    assert quoted_name in schemas.ddl, (
        f"Expected double-quoted column name {quoted_name!r} in DDL, "
        f"got:\n{schemas.ddl}"
    )
    # Confirm needs_quoting works correctly
    assert needs_quoting(special_col_name) is True
    assert needs_quoting("simple_col") is False


# ---------------------------------------------------------------------------
# Test 10: Deterministic output
# ---------------------------------------------------------------------------

def test_deterministic_output(test_profile, test_summary):
    """Calling format_schemas() twice with the same inputs produces identical output."""
    schemas1 = format_schemas(test_profile, test_summary)
    schemas2 = format_schemas(test_profile, test_summary)

    assert schemas1.ddl == schemas2.ddl, (
        "DDL output should be deterministic (byte-identical) across two calls"
    )
    assert schemas1.markdown == schemas2.markdown, (
        "Markdown output should be deterministic (byte-identical) across two calls"
    )


# ---------------------------------------------------------------------------
# Bonus: format_and_save_schemas writes files
# ---------------------------------------------------------------------------

def test_format_and_save_schemas_creates_files(test_profile, test_summary, tmp_path):
    """format_and_save_schemas() writes DDL and Markdown files to disk."""
    schemas = format_and_save_schemas(test_profile, test_summary, str(tmp_path))

    ddl_file = tmp_path / "test_db_ddl.sql"
    md_file = tmp_path / "test_db_markdown.md"

    assert ddl_file.exists(), "DDL file should be created"
    assert md_file.exists(), "Markdown file should be created"

    assert ddl_file.read_text(encoding="utf-8") == schemas.ddl
    assert md_file.read_text(encoding="utf-8") == schemas.markdown
