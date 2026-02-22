"""
Integration tests for src/pipeline/offline_pipeline.py

4 tests:
1. test_full_offline_pipeline_small_db — full pipeline returns OfflineArtifacts
2. test_all_output_files_created — all 6 artifact files exist after run
3. test_pipeline_idempotent — second run loads from cache (summarizer called once)
4. test_pipeline_handles_missing_db_gracefully — non-existent db_path raises FileNotFoundError

Uses a real temp SQLite DB for profiling but mocks summarizer API calls.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from src.data.bird_loader import BirdEntry
from src.pipeline.offline_pipeline import OfflineArtifacts, run_offline_pipeline
from src.preprocessing.summarizer import DatabaseSummary, FieldSummary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_temp_db(tmp_path: Path) -> tuple[str, str]:
    """Create a small SQLite database for testing. Returns (db_id, db_path)."""
    db_id = "test_pipeline_db"
    db_path = tmp_path / f"{db_id}.sqlite"

    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE products (
            id      INTEGER PRIMARY KEY,
            name    TEXT NOT NULL,
            price   REAL,
            stock   INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE orders (
            order_id    INTEGER PRIMARY KEY,
            product_id  INTEGER REFERENCES products(id),
            quantity    INTEGER,
            total       REAL
        )
    """)
    # Insert some rows
    for i in range(1, 6):
        conn.execute(f"INSERT INTO products VALUES ({i}, 'Product {i}', {10.0 * i}, {100 * i})")
    for i in range(1, 6):
        conn.execute(f"INSERT INTO orders VALUES ({i}, {i}, {i}, {10.0 * i * i})")
    conn.commit()
    conn.close()

    return db_id, str(db_path)


def _make_dummy_summary(db_id: str) -> DatabaseSummary:
    """Create a realistic dummy DatabaseSummary for mocking."""
    field_summaries = [
        FieldSummary(
            table_name="products",
            column_name="id",
            short_summary="Primary key for products.",
            long_summary="The id field is the unique integer primary key for the products table.",
        ),
        FieldSummary(
            table_name="products",
            column_name="name",
            short_summary="Product name.",
            long_summary="The name field stores the commercial name of each product. Used for display and search.",
        ),
        FieldSummary(
            table_name="products",
            column_name="price",
            short_summary="Product price in USD.",
            long_summary="The price field stores the unit sale price of the product in US dollars.",
        ),
        FieldSummary(
            table_name="products",
            column_name="stock",
            short_summary="Available inventory count.",
            long_summary="The stock field stores the current number of units available in inventory.",
        ),
        FieldSummary(
            table_name="orders",
            column_name="order_id",
            short_summary="Primary key for orders.",
            long_summary="The order_id field is the unique integer primary key for each order record.",
        ),
        FieldSummary(
            table_name="orders",
            column_name="product_id",
            short_summary="Foreign key to products.",
            long_summary="The product_id field links each order to a specific product in the products table.",
        ),
        FieldSummary(
            table_name="orders",
            column_name="quantity",
            short_summary="Number of units ordered.",
            long_summary="The quantity field records how many units of the product were purchased in this order.",
        ),
        FieldSummary(
            table_name="orders",
            column_name="total",
            short_summary="Total order value.",
            long_summary="The total field stores the computed total monetary value of the order.",
        ),
    ]
    return DatabaseSummary(db_id=db_id, field_summaries=field_summaries)


def _make_train_entries() -> list[BirdEntry]:
    """A small set of training entries from a different database."""
    return [
        BirdEntry(
            question_id=i,
            db_id=f"other_db_{i % 3}",
            question=f"How many records are there in table {i}?",
            evidence="",
            SQL=f"SELECT COUNT(*) FROM table_{i}",
            difficulty="simple",
        )
        for i in range(10)
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_db(tmp_path):
    """Temporary SQLite database."""
    return _make_temp_db(tmp_path)


@pytest.fixture
def train_entries():
    return _make_train_entries()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_offline_pipeline_small_db(tmp_path, temp_db, train_entries):
    """Full pipeline returns OfflineArtifacts with all fields populated."""
    db_id, db_path = temp_db
    preprocessed_dir = str(tmp_path / "preprocessed")

    dummy_summary = _make_dummy_summary(db_id)

    with patch("src.pipeline.offline_pipeline.summarize_database",
               new_callable=AsyncMock) as mock_summarize:
        mock_summarize.return_value = dummy_summary

        artifacts = await run_offline_pipeline(
            db_id=db_id,
            db_path=db_path,
            train_data=train_entries,
            preprocessed_dir=preprocessed_dir,
            force=False,
        )

    # All fields must be populated
    assert isinstance(artifacts, OfflineArtifacts)
    assert artifacts.db_id == db_id
    assert artifacts.profile is not None
    assert artifacts.summary is not None
    assert artifacts.schemas is not None
    assert artifacts.lsh_index is not None
    assert artifacts.faiss_index is not None
    assert artifacts.example_store is not None

    # Profile should contain the tables we created
    assert "products" in artifacts.profile.tables
    assert "orders" in artifacts.profile.tables

    # Summary should have field summaries for the columns
    assert len(artifacts.summary.field_summaries) > 0

    # FAISS index should support queries
    faiss_results = artifacts.faiss_index.query("product price", top_k=3)
    assert len(faiss_results) > 0

    # LSH index should be populated
    assert artifacts.lsh_index is not None


@pytest.mark.asyncio
async def test_all_output_files_created(tmp_path, temp_db, train_entries):
    """After running the pipeline, all 6 required artifact files should exist."""
    db_id, db_path = temp_db
    preprocessed_dir = str(tmp_path / "preprocessed")

    dummy_summary = _make_dummy_summary(db_id)

    with patch("src.pipeline.offline_pipeline.summarize_database",
               new_callable=AsyncMock) as mock_summarize:
        mock_summarize.return_value = dummy_summary

        await run_offline_pipeline(
            db_id=db_id,
            db_path=db_path,
            train_data=train_entries,
            preprocessed_dir=preprocessed_dir,
            force=False,
        )

    root = Path(preprocessed_dir)
    # Check all expected files exist
    expected_files = [
        root / "profiles" / f"{db_id}.json",
        root / "summaries" / f"{db_id}.json",
        root / "schemas" / f"{db_id}_ddl.sql",
        root / "indices" / f"{db_id}_lsh.pkl",
        root / "indices" / f"{db_id}_faiss.index",
        root / "indices" / f"{db_id}_faiss_fields.json",
    ]
    for f in expected_files:
        assert f.exists(), f"Expected artifact file not found: {f}"


@pytest.mark.asyncio
async def test_pipeline_idempotent(tmp_path, temp_db, train_entries):
    """Running the pipeline twice should use cache on the second run (summarizer called once)."""
    db_id, db_path = temp_db
    preprocessed_dir = str(tmp_path / "preprocessed")

    dummy_summary = _make_dummy_summary(db_id)

    with patch("src.pipeline.offline_pipeline.summarize_database",
               new_callable=AsyncMock) as mock_summarize:
        mock_summarize.return_value = dummy_summary

        # First run — builds everything
        artifacts1 = await run_offline_pipeline(
            db_id=db_id,
            db_path=db_path,
            train_data=train_entries,
            preprocessed_dir=preprocessed_dir,
            force=False,
        )

        first_call_count = mock_summarize.call_count

        # Second run — should load from cache
        artifacts2 = await run_offline_pipeline(
            db_id=db_id,
            db_path=db_path,
            train_data=train_entries,
            preprocessed_dir=preprocessed_dir,
            force=False,
        )

        second_call_count = mock_summarize.call_count

    # Summarizer should only be called during first run
    assert first_call_count == 1, f"Expected 1 summarizer call, got {first_call_count}"
    assert second_call_count == 1, (
        f"Summarizer should not be called on cache hit, "
        f"but call count increased to {second_call_count}"
    )

    # Both runs should produce valid artifacts
    assert artifacts1.db_id == artifacts2.db_id
    assert len(artifacts1.profile.tables) == len(artifacts2.profile.tables)


@pytest.mark.asyncio
async def test_pipeline_handles_missing_db_gracefully(tmp_path, train_entries):
    """Non-existent db_path should raise FileNotFoundError."""
    preprocessed_dir = str(tmp_path / "preprocessed")

    with pytest.raises(FileNotFoundError):
        await run_offline_pipeline(
            db_id="nonexistent_db",
            db_path="/nonexistent/path/database.sqlite",
            train_data=train_entries,
            preprocessed_dir=preprocessed_dir,
        )
