"""
Tests for src/indexing/lsh_index.py  (Op 1a)

All tests use in-memory SQLite databases (no real BIRD data required).
"""

from __future__ import annotations

import sqlite3
import tempfile
import time
from pathlib import Path

import pytest

from src.indexing.lsh_index import CellMatch, LSHIndex


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_standard_db(path: str) -> None:
    """
    Create a small SQLite database at *path* with:
      - Table ``countries``: columns id, country_name (with NULLs)
      - Table ``orders``:    columns id, status, year (numeric)
    """
    conn = sqlite3.connect(path)
    cur = conn.cursor()

    cur.executescript(
        """
        CREATE TABLE countries (
            id           INTEGER PRIMARY KEY,
            country_name TEXT
        );
        INSERT INTO countries VALUES
            (1, 'United States'),
            (2, 'United Kingdom'),
            (3, 'Germany'),
            (4, 'Japan'),
            (5, NULL),
            (6, NULL);

        CREATE TABLE orders (
            id     INTEGER PRIMARY KEY,
            status TEXT,
            year   INTEGER
        );
        INSERT INTO orders VALUES
            (1, 'pending',   2015),
            (2, 'completed', 2016),
            (3, 'cancelled', 2017),
            (4, NULL,        2018);
        """
    )
    conn.commit()
    conn.close()


@pytest.fixture()
def standard_index(tmp_path: Path) -> LSHIndex:
    """Build and return an LSHIndex for the standard test database."""
    db_file = str(tmp_path / "test.db")
    _make_standard_db(db_file)
    idx = LSHIndex()
    idx.build(db_file, db_id="test_db")
    return idx


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExactMatch:
    def test_exact_match_retrieval(self, standard_index: LSHIndex) -> None:
        """Querying 'United States' should find the exact value with sim >= 0.9."""
        results = standard_index.query("United States", top_k=5)
        assert results, "Expected at least one result"
        top = results[0]
        assert top.table == "countries"
        assert top.column == "country_name"
        assert top.matched_value == "United States"
        assert top.similarity_score >= 0.9, f"Expected sim >= 0.9, got {top.similarity_score}"
        assert top.exact_match is True


class TestFuzzyMatch:
    def test_fuzzy_match_typo(self, standard_index: LSHIndex) -> None:
        """Querying 'Untied States' (typo) should still return countries.country_name as top result."""
        results = standard_index.query("Untied States", top_k=5)
        assert results, "Expected at least one result for fuzzy query"
        # The top result should be from the countries table
        country_results = [r for r in results if r.table == "countries" and "States" in r.matched_value]
        assert country_results, (
            f"Expected a countries.country_name result, got: {results}"
        )
        top_country = country_results[0]
        # True 3-gram Jaccard("Untied States", "United States") ≈ 0.47
        # MinHash estimate may vary slightly; we accept > 0.4 as a clear signal.
        assert top_country.similarity_score > 0.4, (
            f"Expected similarity > 0.4, got {top_country.similarity_score}"
        )
        # Also verify exact_match is False for the typo
        assert top_country.exact_match is False


class TestNoMatch:
    def test_no_match_for_unrelated_query(self, standard_index: LSHIndex) -> None:
        """Querying 'xyz123' should return an empty list."""
        results = standard_index.query("xyz123", top_k=5)
        assert results == [], f"Expected empty list, got {results}"


class TestCrossColumn:
    def test_cross_column_retrieval(self, standard_index: LSHIndex) -> None:
        """Querying 'pending' should find a result from orders.status."""
        results = standard_index.query("pending", top_k=5)
        assert results, "Expected at least one result"
        order_results = [r for r in results if r.table == "orders" and r.column == "status"]
        assert order_results, (
            f"Expected orders.status result, got: {results}"
        )
        assert order_results[0].matched_value == "pending"


class TestTopKLimiting:
    def test_top_k_limiting(self, standard_index: LSHIndex) -> None:
        """top_k=2 should return at most 2 results."""
        results = standard_index.query("United", top_k=2)
        assert len(results) <= 2


class TestSerialization:
    def test_serialization_roundtrip(self, standard_index: LSHIndex, tmp_path: Path) -> None:
        """Save + load should produce an index with identical query results."""
        save_path = str(tmp_path / "index.pkl")
        standard_index.save(save_path)

        loaded = LSHIndex.load(save_path)

        original_results = standard_index.query("United States", top_k=5)
        loaded_results = loaded.query("United States", top_k=5)

        assert loaded_results, "Loaded index returned no results"
        # Check that both return the same top match
        assert original_results[0].matched_value == loaded_results[0].matched_value
        assert abs(original_results[0].similarity_score - loaded_results[0].similarity_score) < 1e-6


class TestNullValues:
    def test_null_values_excluded(self, tmp_path: Path) -> None:
        """NULL values must not appear as matches."""
        db_file = str(tmp_path / "null_test.db")
        conn = sqlite3.connect(db_file)
        conn.execute("CREATE TABLE t (v TEXT)")
        conn.execute("INSERT INTO t VALUES (NULL)")
        conn.execute("INSERT INTO t VALUES (NULL)")
        conn.commit()
        conn.close()

        idx = LSHIndex()
        idx.build(db_file, db_id="null_db")

        # A blank / NULL-like query should return nothing
        results = idx.query("", top_k=5)
        assert results == [], f"Expected empty list for NULL-only table, got {results}"

        results2 = idx.query("NULL", top_k=5)
        # If there are no matches the list is empty; if something matches it
        # must not be a NULL value stored verbatim.
        for r in results2:
            assert r.matched_value.upper() != "NULL", (
                f"NULL value leaked into index: {r}"
            )


class TestNumericValues:
    def test_integer_columns_not_indexed(self, standard_index: LSHIndex) -> None:
        """INTEGER-affinity columns must not be indexed in LSH.

        The ``orders.year`` column is declared INTEGER, so its values (2015,
        2016, …) should not appear as LSH matches.  Fuzzy text matching on
        integer fields is never useful and would bloat the index.
        """
        results = standard_index.query("2015", top_k=5)
        year_results = [r for r in results if r.column == "year"]
        assert year_results == [], (
            f"INTEGER column 'year' should not be indexed; got: {year_results}"
        )


class TestBuildSpeed:
    def test_index_build_speed(self, tmp_path: Path) -> None:
        """Building an index over 10,000 strings must complete in < 30 seconds."""
        db_file = str(tmp_path / "large.db")
        conn = sqlite3.connect(db_file)
        conn.execute("CREATE TABLE big_table (value TEXT)")
        conn.executemany(
            "INSERT INTO big_table VALUES (?)",
            [(f"record_{i}",) for i in range(10_000)],
        )
        conn.commit()
        conn.close()

        idx = LSHIndex()
        start = time.monotonic()
        idx.build(db_file, db_id="large_db")
        elapsed = time.monotonic() - start

        assert elapsed < 30.0, f"Build took {elapsed:.1f}s — exceeds 30-second limit"
        assert len(idx._minhashes) == 10_000


class TestEmptyTable:
    def test_empty_table_index(self, tmp_path: Path) -> None:
        """Building an index on an empty table must not raise any errors."""
        db_file = str(tmp_path / "empty.db")
        conn = sqlite3.connect(db_file)
        conn.execute("CREATE TABLE empty_table (id INTEGER, name TEXT)")
        conn.commit()
        conn.close()

        idx = LSHIndex()
        idx.build(db_file, db_id="empty_db")  # must not raise

        results = idx.query("anything", top_k=5)
        assert results == [], f"Expected empty list for empty-table index, got {results}"
