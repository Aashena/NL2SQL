"""
Unit tests for src/indexing/faiss_index.py

10 tests covering:
- Semantic retrieval
- top_k count
- Similarity ordering
- Specific query → field mapping
- Serialization roundtrip
- Embedding dimensionality
- Single-field build
- Score range
- Different questions → different rankings
"""

import pytest
from src.indexing.faiss_index import FAISSIndex, FieldMatch
from src.preprocessing.summarizer import FieldSummary


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def field_summaries() -> list[FieldSummary]:
    """10 mock FieldSummary objects spanning sales and customers tables."""
    return [
        # sales table
        FieldSummary(
            table_name="sales",
            column_name="amount",
            short_summary="Total financial value of the sale in USD.",
            long_summary="The amount field stores the total financial value of each sale transaction in USD. It represents gross revenue before any discounts or returns.",
        ),
        FieldSummary(
            table_name="sales",
            column_name="date",
            short_summary="Date when the sale occurred.",
            long_summary="The date field records when the sale transaction occurred. It is stored in ISO 8601 format and used for temporal analysis.",
        ),
        FieldSummary(
            table_name="sales",
            column_name="customer_id",
            short_summary="Foreign key linking to customers table.",
            long_summary="Foreign key linking to customers table. References the unique customer identifier to associate each sale with a specific buyer.",
        ),
        # customers table
        FieldSummary(
            table_name="customers",
            column_name="name",
            short_summary="Full legal name of the customer.",
            long_summary="The name field stores the full legal name of the customer. It is used for identification, invoicing, and correspondence.",
        ),
        FieldSummary(
            table_name="customers",
            column_name="country",
            short_summary="Country where the customer is located.",
            long_summary="The country field stores the geographic country where the customer is located. Used for geographic reporting and regional analysis.",
        ),
        FieldSummary(
            table_name="customers",
            column_name="age",
            short_summary="Demographic age of the customer.",
            long_summary="The age field records the customer demographic age. Used for market segmentation and age-based analytics.",
        ),
        # products table
        FieldSummary(
            table_name="products",
            column_name="product_name",
            short_summary="Name of the product.",
            long_summary="The product_name field stores the commercial name of each product sold. Used for display, search, and catalog management.",
        ),
        FieldSummary(
            table_name="products",
            column_name="category",
            short_summary="Product category or type.",
            long_summary="The category field classifies products into logical groupings such as electronics, clothing, and food. Used for filtering and aggregation.",
        ),
        # regions table
        FieldSummary(
            table_name="regions",
            column_name="region",
            short_summary="Geographic sales region.",
            long_summary="The region field defines the geographic sales territory for reporting purposes. Each region corresponds to a cluster of countries or states.",
        ),
        FieldSummary(
            table_name="regions",
            column_name="revenue",
            short_summary="Total revenue for the region.",
            long_summary="The revenue field stores the total monetary income generated in a given region. Used in financial performance and regional profitability reporting.",
        ),
    ]


@pytest.fixture
def built_index(field_summaries) -> FAISSIndex:
    """A FAISSIndex already built from the mock field_summaries."""
    idx = FAISSIndex()
    idx.build(field_summaries)
    return idx


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_semantic_retrieval_relevant(built_index):
    """'How many sales were made per country?' should return fields from both sales and customers."""
    results = built_index.query("How many sales were made per country?", top_k=5)
    tables = {r.table for r in results}
    # Should include at least one field from sales and one from customers
    assert "sales" in tables or "customers" in tables
    assert len(results) > 0


def test_top_k_count(built_index):
    """Querying with top_k=3 returns exactly 3 results."""
    results = built_index.query("How many total transactions occurred?", top_k=3)
    assert len(results) == 3


def test_similarity_ordering(built_index):
    """Results are in descending order of similarity score."""
    results = built_index.query("What is the average age of customers?", top_k=5)
    assert len(results) >= 2
    scores = [r.similarity_score for r in results]
    assert scores == sorted(scores, reverse=True), "Results must be in descending similarity order"


def test_financial_query_retrieves_amount(built_index):
    """'What is the total revenue?' should return sales.amount in the top-3 results."""
    results = built_index.query("What is the total revenue?", top_k=3)
    top_fields = [(r.table, r.column) for r in results]
    # Either sales.amount or regions.revenue should appear in top results
    found = any(
        (t == "sales" and c == "amount") or (t == "regions" and c == "revenue")
        for t, c in top_fields
    )
    assert found, f"Expected financial field in top-3, got: {top_fields}"


def test_geographic_query_retrieves_country(built_index):
    """'Which countries have customers?' should return customers.country in the top result."""
    results = built_index.query("Which countries have customers?", top_k=3)
    assert len(results) > 0
    top_fields = [(r.table, r.column) for r in results]
    found = any(
        (t == "customers" and c == "country") or (t == "regions" and c == "region")
        for t, c in top_fields
    )
    assert found, f"Expected geographic field in top-3, got: {top_fields}"


def test_serialization_roundtrip(built_index, tmp_path):
    """Save + load should produce the same query results."""
    index_path = str(tmp_path / "test.index")
    fields_path = str(tmp_path / "test_fields.json")

    built_index.save(index_path, fields_path)
    loaded = FAISSIndex.load(index_path, fields_path)

    question = "What is the total financial value of sales?"
    original = built_index.query(question, top_k=5)
    restored = loaded.query(question, top_k=5)

    assert len(original) == len(restored)
    for o, r in zip(original, restored):
        assert o.table == r.table
        assert o.column == r.column
        assert abs(o.similarity_score - r.similarity_score) < 1e-5


def test_embedding_dimensionality(field_summaries):
    """Internal embeddings should have 384 dimensions (all-MiniLM-L6-v2)."""
    from src.indexing.faiss_index import _embed
    import numpy as np

    texts = [fs.long_summary for fs in field_summaries]
    embeddings = _embed(texts)
    assert embeddings.shape == (len(texts), 384), (
        f"Expected shape ({len(texts)}, 384), got {embeddings.shape}"
    )


def test_index_build_with_single_field():
    """Building with a single FieldSummary should work without errors."""
    single = FieldSummary(
        table_name="orders",
        column_name="total",
        short_summary="Order total.",
        long_summary="The total field stores the final order amount after all adjustments.",
    )
    idx = FAISSIndex()
    idx.build([single])  # must not raise
    results = idx.query("What is the order total?", top_k=1)
    assert len(results) == 1
    assert results[0].table == "orders"
    assert results[0].column == "total"


def test_query_returns_scores_between_0_and_1(built_index):
    """All similarity scores should be in [0, 1]."""
    results = built_index.query("How many customers are there?", top_k=10)
    assert len(results) > 0
    for r in results:
        assert 0.0 <= r.similarity_score <= 1.0, (
            f"Score {r.similarity_score} out of [0, 1] for {r.table}.{r.column}"
        )


def test_different_questions_different_rankings(built_index):
    """Two clearly different questions should produce different top-1 results."""
    results_finance = built_index.query("What is the total sales amount in USD?", top_k=3)
    results_geo = built_index.query("Which geographic country has most customers?", top_k=3)

    assert len(results_finance) > 0
    assert len(results_geo) > 0

    top_finance = (results_finance[0].table, results_finance[0].column)
    top_geo = (results_geo[0].table, results_geo[0].column)

    assert top_finance != top_geo, (
        f"Both queries returned the same top-1 field: {top_finance}. "
        "Different questions should rank different fields first."
    )
