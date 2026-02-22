"""
Tests for src/preprocessing/summarizer.py — Op 0b: LLM Field Summarizer.

All tests mock get_client() — NO real API calls are made.

The mock fixture returns summaries for five columns:
  id, name, age, gpa, country
"""

import json
from pathlib import Path

import pytest

from src.llm.base import LLMResponse
from src.preprocessing.profiler import ColumnProfile, DatabaseProfile
from src.preprocessing.summarizer import (
    DatabaseSummary,
    FieldSummary,
    summarize_database,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_col(table: str, col: str, dtype: str = "TEXT") -> ColumnProfile:
    """Build a minimal ColumnProfile for testing."""
    return ColumnProfile(
        table_name=table,
        column_name=col,
        data_type=dtype,
        total_count=10,
        null_count=0,
        null_rate=0.0,
        distinct_count=5,
        sample_values=[["a", 3], ["b", 2]],
        min_value=None,
        max_value=None,
        avg_value=None,
        avg_length=5.0,
        is_primary_key=False,
        foreign_key_ref=None,
        minhash_bands=list(range(128)),
    )


def _make_profile(columns: list[ColumnProfile], db_id: str = "test_db") -> DatabaseProfile:
    tables = list({c.table_name for c in columns})
    return DatabaseProfile(
        db_id=db_id,
        tables=tables,
        columns=columns,
        foreign_keys=[],
        total_tables=len(tables),
        total_columns=len(columns),
    )


def _make_summaries_response(*column_names: str) -> LLMResponse:
    """Build a realistic LLMResponse with summaries for the given column names."""
    _DATA = {
        "id":      ("Unique integer identifier for each record.",
                    "This is the primary key column. Values are auto-incremented integers "
                    "starting from 1. Used in JOINs and WHERE clauses to identify specific "
                    "rows. Never NULL."),
        "name":    ("Full name of the entity.",
                    "Stores text names. Can be NULL for unknown entries. "
                    "Used in filtering by name."),
        "age":     ("Age in years.",
                    "Integer age in years. Ranges from 18-100. Used in demographic filters."),
        "gpa":     ("Grade point average.",
                    "Float GPA on 0-4.0 scale. Used in academic queries."),
        "country": ("Country code.",
                    "ISO country code. Used in geographic filters."),
    }
    summaries = []
    for col in column_names:
        short, long_ = _DATA.get(col, (f"{col} field.", f"The {col} column."))
        summaries.append({
            "column_name": col,
            "short_summary": short,
            "long_summary": long_,
        })
    return LLMResponse(tool_inputs=[{"summaries": summaries}])


# ---------------------------------------------------------------------------
# Mock fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm_client(mocker):
    """
    Patch get_client() in summarizer so no real API calls are made.
    Returns a mock LLMClient whose generate() coroutine returns summaries
    for: id, name, age, gpa, country.
    """
    mock_client = mocker.AsyncMock()
    mocker.patch("src.preprocessing.summarizer.get_client", return_value=mock_client)
    mock_client.generate.return_value = _make_summaries_response(
        "id", "name", "age", "gpa", "country"
    )
    return mock_client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBatchingGroupsByTable:
    """Test 1: columns from 2 tables → 2 API calls (one per table)."""

    async def test_batching_groups_by_table(self, mock_llm_client):
        columns = [
            make_col("table_a", "col1"),
            make_col("table_a", "col2"),
            make_col("table_a", "col3"),
            make_col("table_b", "col4"),
            make_col("table_b", "col5"),
            make_col("table_b", "col6"),
        ]
        profile = _make_profile(columns)
        await summarize_database(profile)

        # One API call per table (both tables have ≤6 cols, so one batch each)
        assert mock_llm_client.generate.call_count == 2


class TestSummaryFieldsPresent:
    """Test 2: all returned FieldSummary objects have non-empty strings."""

    async def test_summary_fields_present(self, mock_llm_client):
        columns = [
            make_col("students", "id"),
            make_col("students", "name"),
            make_col("students", "age"),
            make_col("students", "gpa"),
            make_col("students", "country"),
        ]
        profile = _make_profile(columns)
        db_summary = await summarize_database(profile)

        for fs in db_summary.field_summaries:
            assert isinstance(fs.short_summary, str) and fs.short_summary, (
                f"short_summary is empty for {fs.table_name}.{fs.column_name}"
            )
            assert isinstance(fs.long_summary, str) and fs.long_summary, (
                f"long_summary is empty for {fs.table_name}.{fs.column_name}"
            )


class TestShortSummaryLengthBound:
    """Test 3: short_summary is truncated to ≤200 chars."""

    async def test_short_summary_length_bound(self, mocker):
        mock_client = mocker.AsyncMock()
        mocker.patch("src.preprocessing.summarizer.get_client", return_value=mock_client)

        long_short = "X" * 300  # 300-char short_summary from LLM
        mock_client.generate.return_value = LLMResponse(tool_inputs=[{
            "summaries": [{
                "column_name": "id",
                "short_summary": long_short,
                "long_summary": "Normal long summary.",
            }]
        }])

        columns = [make_col("students", "id")]
        profile = _make_profile(columns)
        db_summary = await summarize_database(profile)

        assert len(db_summary.field_summaries) == 1
        assert len(db_summary.field_summaries[0].short_summary) <= 200


class TestLongSummaryLengthBound:
    """Test 4: long_summary is truncated to ≤1000 chars."""

    async def test_long_summary_length_bound(self, mocker):
        mock_client = mocker.AsyncMock()
        mocker.patch("src.preprocessing.summarizer.get_client", return_value=mock_client)

        long_long = "Y" * 1500  # 1500-char long_summary from LLM
        mock_client.generate.return_value = LLMResponse(tool_inputs=[{
            "summaries": [{
                "column_name": "id",
                "short_summary": "Short summary.",
                "long_summary": long_long,
            }]
        }])

        columns = [make_col("students", "id")]
        profile = _make_profile(columns)
        db_summary = await summarize_database(profile)

        assert len(db_summary.field_summaries) == 1
        assert len(db_summary.field_summaries[0].long_summary) <= 1000


class TestAllColumnsCovered:
    """Test 5: all 5 columns produce exactly 5 FieldSummary objects."""

    async def test_all_columns_covered(self, mock_llm_client):
        columns = [
            make_col("students", "id"),
            make_col("students", "name"),
            make_col("students", "age"),
            make_col("students", "gpa"),
            make_col("students", "country"),
        ]
        profile = _make_profile(columns)
        db_summary = await summarize_database(profile)

        assert len(db_summary.field_summaries) == 5


class TestApiErrorPropagates:
    """Test 6: when generate() raises, the error propagates out of summarize_database."""

    async def test_api_error_propagates(self, mocker):
        mock_client = mocker.AsyncMock()
        mocker.patch("src.preprocessing.summarizer.get_client", return_value=mock_client)
        mock_client.generate.side_effect = Exception("API unavailable")

        columns = [make_col("students", "id")]
        profile = _make_profile(columns)

        with pytest.raises(Exception, match="API unavailable"):
            await summarize_database(profile)


class TestCacheHitSkipsApi:
    """Test 7: pre-written cache file → 0 API calls."""

    async def test_cache_hit_skips_api(self, mock_llm_client, tmp_path):
        db_id = "cached_db"
        output_dir = str(tmp_path)

        # Write a valid JSON cache file beforehand
        cached = {
            "db_id": db_id,
            "field_summaries": [
                {
                    "table_name": "students",
                    "column_name": "id",
                    "short_summary": "Cached short.",
                    "long_summary": "Cached long summary for the id column.",
                }
            ],
        }
        cache_file = tmp_path / f"{db_id}.json"
        cache_file.write_text(json.dumps(cached))

        columns = [make_col("students", "id")]
        profile = _make_profile(columns, db_id=db_id)
        db_summary = await summarize_database(profile, output_dir=output_dir)

        # No API calls should have been made
        assert mock_llm_client.generate.call_count == 0
        assert db_summary.db_id == db_id
        assert db_summary.field_summaries[0].short_summary == "Cached short."


class TestCacheWriteOnSuccess:
    """Test 8: after a successful call, the JSON cache file is created."""

    async def test_cache_write_on_success(self, mocker, tmp_path):
        mock_client = mocker.AsyncMock()
        mocker.patch("src.preprocessing.summarizer.get_client", return_value=mock_client)
        mock_client.generate.return_value = LLMResponse(tool_inputs=[{
            "summaries": [
                {
                    "column_name": "id",
                    "short_summary": "Unique identifier.",
                    "long_summary": "Primary key.",
                },
                {
                    "column_name": "name",
                    "short_summary": "Student name.",
                    "long_summary": "Text name of the student.",
                },
            ]
        }])

        db_id = "write_test_db"
        output_dir = str(tmp_path)
        columns = [
            make_col("students", "id"),
            make_col("students", "name"),
        ]
        profile = _make_profile(columns, db_id=db_id)
        await summarize_database(profile, output_dir=output_dir)

        cache_file = tmp_path / f"{db_id}.json"
        assert cache_file.exists(), f"Cache file not found at {cache_file}"
        assert cache_file.stat().st_size > 0


class TestToolUseFormatParsing:
    """Test 9: FieldSummary correctly maps column_name to table/column."""

    async def test_tool_use_format_parsing(self, mock_llm_client):
        columns = [
            make_col("students", "id"),
            make_col("students", "name"),
            make_col("students", "age"),
            make_col("students", "gpa"),
            make_col("students", "country"),
        ]
        profile = _make_profile(columns)
        db_summary = await summarize_database(profile)

        summaries_by_col = {fs.column_name: fs for fs in db_summary.field_summaries}

        assert "id" in summaries_by_col
        assert summaries_by_col["id"].short_summary == "Unique integer identifier for each record."
        assert summaries_by_col["id"].table_name == "students"

        assert "gpa" in summaries_by_col
        assert summaries_by_col["gpa"].short_summary == "Grade point average."
        assert summaries_by_col["gpa"].table_name == "students"


class TestMissingColumnGetsDefaultSummary:
    """Test 10: if LLM omits a column, it gets a safe default summary."""

    async def test_missing_column_gets_default_summary(self, mocker):
        mock_client = mocker.AsyncMock()
        mocker.patch("src.preprocessing.summarizer.get_client", return_value=mock_client)

        # LLM returns only 2 summaries even though 3 columns were sent
        mock_client.generate.return_value = LLMResponse(tool_inputs=[{
            "summaries": [
                {
                    "column_name": "id",
                    "short_summary": "Unique id.",
                    "long_summary": "Primary key.",
                },
                {
                    "column_name": "name",
                    "short_summary": "Name field.",
                    "long_summary": "Text name.",
                },
                # "age" is missing from the response
            ]
        }])

        columns = [
            make_col("students", "id"),
            make_col("students", "name"),
            make_col("students", "age"),  # this one will be missing from LLM response
        ]
        profile = _make_profile(columns)
        db_summary = await summarize_database(profile)

        # All 3 columns should have a FieldSummary
        assert len(db_summary.field_summaries) == 3

        age_summary = next(
            fs for fs in db_summary.field_summaries if fs.column_name == "age"
        )
        # Default summary must be non-empty
        assert age_summary.short_summary, "Default short_summary must not be empty"
        assert age_summary.long_summary, "Default long_summary must not be empty"
        assert "age" in age_summary.short_summary.lower() or "age" in age_summary.long_summary.lower()
