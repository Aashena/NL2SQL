"""
Tests for src/preprocessing/profiler.py — Op 0a: Statistical Database Profiler.

Uses the `students_db` fixture from conftest.py:
  - countries table: code (PK TEXT), name (TEXT)
  - students table:  id (PK INTEGER), name (TEXT), age (INTEGER), gpa (REAL),
                     country (TEXT, FK → countries.code)
  - Rows: (1,'Alice',20,3.9,'USA'), (2,'Bob',22,NULL,'UK'),
          (3,'Alice',21,3.5,'USA'), (4,NULL,25,2.8,NULL)
"""

import dataclasses
import json
import sqlite3

import pytest

from src.preprocessing.profiler import (
    ColumnProfile,
    DatabaseProfile,
    profile_database,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_col(profile: DatabaseProfile, table: str, column: str) -> ColumnProfile:
    """Retrieve a specific ColumnProfile by table + column name."""
    for col in profile.columns:
        if col.table_name == table and col.column_name == column:
            return col
    raise KeyError(f"Column {table}.{column} not found in profile")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestColumnCountMatchesSchema:
    """Profiler returns exactly one ColumnProfile per column."""

    def test_column_count_matches_schema(self, students_db):
        # students has 5 cols (id, name, age, gpa, country)
        # countries has 2 cols (code, name)
        # Total = 7
        profile = profile_database(students_db, "test")
        assert len(profile.columns) == 7, (
            f"Expected 7 total columns, got {len(profile.columns)}"
        )

    def test_tables_list(self, students_db):
        profile = profile_database(students_db, "test")
        assert set(profile.tables) == {"students", "countries"}
        assert profile.total_tables == 2
        assert profile.total_columns == 7


class TestNullRateComputation:
    """Null rates are computed correctly for columns with NULLs."""

    def test_name_null_rate(self, students_db):
        # students.name: row 4 has NULL → 1/4 = 0.25
        profile = profile_database(students_db, "test")
        col = _get_col(profile, "students", "name")
        assert col.null_rate == pytest.approx(0.25, abs=1e-9)

    def test_gpa_null_rate(self, students_db):
        # students.gpa: row 2 has NULL → 1/4 = 0.25
        profile = profile_database(students_db, "test")
        col = _get_col(profile, "students", "gpa")
        assert col.null_rate == pytest.approx(0.25, abs=1e-9)

    def test_age_null_rate_is_zero(self, students_db):
        # students.age: no NULLs → 0.0
        profile = profile_database(students_db, "test")
        col = _get_col(profile, "students", "age")
        assert col.null_rate == pytest.approx(0.0, abs=1e-9)

    def test_null_count_values(self, students_db):
        profile = profile_database(students_db, "test")
        name_col = _get_col(profile, "students", "name")
        assert name_col.null_count == 1
        assert name_col.total_count == 4


class TestDistinctCount:
    """Distinct (non-NULL) counts are computed correctly."""

    def test_name_distinct_count(self, students_db):
        # students.name distinct non-NULL values: Alice, Bob → 2
        profile = profile_database(students_db, "test")
        col = _get_col(profile, "students", "name")
        assert col.distinct_count == 2

    def test_id_distinct_count(self, students_db):
        # students.id: 1, 2, 3, 4 → 4 distinct
        profile = profile_database(students_db, "test")
        col = _get_col(profile, "students", "id")
        assert col.distinct_count == 4


class TestSampleValuesOrdering:
    """sample_values returns top-10 by frequency, descending."""

    def test_name_sample_values_alice_first(self, students_db):
        # Alice appears twice, Bob once
        profile = profile_database(students_db, "test")
        col = _get_col(profile, "students", "name")
        assert len(col.sample_values) >= 2
        # First entry should be Alice with freq=2
        first_val, first_freq = col.sample_values[0]
        assert first_val == "Alice"
        assert first_freq == 2

    def test_name_sample_values_bob_second(self, students_db):
        profile = profile_database(students_db, "test")
        col = _get_col(profile, "students", "name")
        second_val, second_freq = col.sample_values[1]
        assert second_val == "Bob"
        assert second_freq == 1

    def test_sample_values_no_nulls(self, students_db):
        # sample_values must not include NULLs
        profile = profile_database(students_db, "test")
        for col in profile.columns:
            for val, freq in col.sample_values:
                assert val is not None, (
                    f"NULL found in sample_values for {col.table_name}.{col.column_name}"
                )

    def test_sample_values_at_most_10(self, students_db):
        profile = profile_database(students_db, "test")
        for col in profile.columns:
            assert len(col.sample_values) <= 10


class TestPrimaryKeyDetection:
    """is_primary_key is correctly identified from PRAGMA table_info."""

    def test_students_id_is_pk(self, students_db):
        profile = profile_database(students_db, "test")
        col = _get_col(profile, "students", "id")
        assert col.is_primary_key is True

    def test_students_name_is_not_pk(self, students_db):
        profile = profile_database(students_db, "test")
        col = _get_col(profile, "students", "name")
        assert col.is_primary_key is False

    def test_countries_code_is_pk(self, students_db):
        profile = profile_database(students_db, "test")
        col = _get_col(profile, "countries", "code")
        assert col.is_primary_key is True


class TestNumericStats:
    """min, max, avg are computed for numeric columns."""

    def test_age_min_value(self, students_db):
        # age values: 20, 22, 21, 25 → min = 20
        profile = profile_database(students_db, "test")
        col = _get_col(profile, "students", "age")
        assert col.min_value == 20

    def test_age_max_value(self, students_db):
        # age values: 20, 22, 21, 25 → max = 25
        profile = profile_database(students_db, "test")
        col = _get_col(profile, "students", "age")
        assert col.max_value == 25

    def test_age_avg_value_not_none(self, students_db):
        # age has no NULLs, avg should be computable
        profile = profile_database(students_db, "test")
        col = _get_col(profile, "students", "age")
        assert col.avg_value is not None
        # (20 + 22 + 21 + 25) / 4 = 22.0
        assert col.avg_value == pytest.approx(22.0, abs=1e-6)

    def test_text_column_has_no_numeric_stats(self, students_db):
        # students.name is TEXT → no min/max/avg_value
        profile = profile_database(students_db, "test")
        col = _get_col(profile, "students", "name")
        assert col.min_value is None
        assert col.max_value is None
        assert col.avg_value is None


class TestForeignKeyDetection:
    """Foreign key references are correctly extracted."""

    def test_students_country_fk(self, students_db):
        profile = profile_database(students_db, "test")
        col = _get_col(profile, "students", "country")
        assert col.foreign_key_ref == "countries.code"

    def test_students_id_no_fk(self, students_db):
        profile = profile_database(students_db, "test")
        col = _get_col(profile, "students", "id")
        assert col.foreign_key_ref is None

    def test_db_foreign_keys_list(self, students_db):
        profile = profile_database(students_db, "test")
        # Should contain at least the students.country → countries.code FK
        assert ("students", "country", "countries", "code") in profile.foreign_keys


class TestMinhashGeneration:
    """MinHash bands have exactly 128 integers and are deterministic."""

    def test_minhash_length(self, students_db):
        profile = profile_database(students_db, "test")
        for col in profile.columns:
            assert len(col.minhash_bands) == 128, (
                f"Expected 128 minhash values for {col.table_name}.{col.column_name}, "
                f"got {len(col.minhash_bands)}"
            )

    def test_minhash_all_integers(self, students_db):
        profile = profile_database(students_db, "test")
        for col in profile.columns:
            for val in col.minhash_bands:
                assert isinstance(val, int), (
                    f"Non-int minhash value {val!r} in "
                    f"{col.table_name}.{col.column_name}"
                )

    def test_minhash_deterministic(self, students_db):
        # Profile the same DB twice — MinHash for same column must be identical
        profile1 = profile_database(students_db, "test_a")
        profile2 = profile_database(students_db, "test_b")

        col1 = _get_col(profile1, "students", "name")
        col2 = _get_col(profile2, "students", "name")
        assert col1.minhash_bands == col2.minhash_bands, (
            "MinHash should be deterministic for the same column values"
        )


class TestEmptyTableHandling:
    """Empty tables produce safe zero/empty statistics."""

    def test_empty_table(self, tmp_path):
        # Create a fresh DB with an empty table
        db_path = tmp_path / "empty.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE empty_test (x INTEGER)")
        conn.commit()
        conn.close()

        profile = profile_database(str(db_path), "empty")
        assert len(profile.columns) == 1

        col = _get_col(profile, "empty_test", "x")
        assert col.total_count == 0
        assert col.null_rate == 0.0
        assert col.sample_values == []
        assert col.min_value is None
        assert col.max_value is None


class TestProfileJsonSerialization:
    """DatabaseProfile survives a JSON round-trip via dataclasses.asdict()."""

    def test_json_round_trip(self, students_db):
        profile = profile_database(students_db, "roundtrip_test")
        data = dataclasses.asdict(profile)

        # Should not raise
        json_str = json.dumps(data)

        # Parse back
        loaded = json.loads(json_str)

        # db_id survives round-trip
        assert loaded["db_id"] == "roundtrip_test"

    def test_json_has_expected_keys(self, students_db):
        profile = profile_database(students_db, "key_test")
        data = dataclasses.asdict(profile)
        required_keys = {"db_id", "tables", "columns", "foreign_keys",
                         "total_tables", "total_columns"}
        assert required_keys.issubset(data.keys())

    def test_json_columns_have_expected_fields(self, students_db):
        profile = profile_database(students_db, "field_test")
        data = dataclasses.asdict(profile)
        col_data = data["columns"][0]
        required_col_keys = {
            "table_name", "column_name", "data_type", "total_count",
            "null_count", "null_rate", "distinct_count", "sample_values",
            "min_value", "max_value", "avg_value", "avg_length",
            "is_primary_key", "foreign_key_ref", "minhash_bands",
        }
        assert required_col_keys.issubset(col_data.keys())


class TestCaching:
    """Caching saves/loads DatabaseProfile to/from disk."""

    def test_cache_write_and_load(self, students_db, tmp_path):
        output_dir = str(tmp_path / "profiles")

        # First call writes cache
        profile1 = profile_database(students_db, "cache_test", output_dir=output_dir)

        # Second call reads from cache (same result)
        profile2 = profile_database(students_db, "cache_test", output_dir=output_dir)

        assert profile1.db_id == profile2.db_id
        assert profile1.total_tables == profile2.total_tables
        assert profile1.total_columns == profile2.total_columns
        assert len(profile1.columns) == len(profile2.columns)

    def test_cache_file_created(self, students_db, tmp_path):
        from pathlib import Path
        output_dir = str(tmp_path / "profiles")

        profile_database(students_db, "file_check", output_dir=output_dir)

        cache_file = Path(output_dir) / "file_check.json"
        assert cache_file.exists()
        assert cache_file.stat().st_size > 0
