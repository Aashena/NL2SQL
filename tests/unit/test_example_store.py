"""
Unit tests for src/indexing/example_store.py

10 tests covering:
- Skeleton masking (entities, numbers)
- Structural retrieval (JOIN, aggregation)
- top_k count
- db_id exclusion
- Non-empty SQL in results
- Serialization roundtrip
- Similarity ordering
- Empty training set error
"""

import pytest
from src.indexing.example_store import ExampleStore, ExampleEntry
from src.data.bird_loader import BirdEntry


# ---------------------------------------------------------------------------
# Helper to create BirdEntry objects
# ---------------------------------------------------------------------------

def _make_entry(
    question_id: int,
    db_id: str,
    question: str,
    sql: str,
    evidence: str = "",
    difficulty: str = "simple",
) -> BirdEntry:
    return BirdEntry(
        question_id=question_id,
        db_id=db_id,
        question=question,
        evidence=evidence,
        SQL=sql,
        difficulty=difficulty,
    )


# ---------------------------------------------------------------------------
# Test data: 50 entries with 10 different db_ids (5 entries each)
# Mix of JOIN (~20), aggregation (~15), simple SELECT (~15)
# At least 5 with db_id="test_db"
# ---------------------------------------------------------------------------

def _make_training_entries() -> list[BirdEntry]:
    entries = []
    qid = 0

    # test_db — 5 entries (for exclusion test)
    test_db_entries = [
        _make_entry(qid, "test_db", "What are the names of all students?",
                    "SELECT name FROM students", difficulty="simple"),
        _make_entry(qid + 1, "test_db", "Count the number of students in Alameda County",
                    "SELECT COUNT(*) FROM students WHERE county = 'Alameda'", difficulty="simple"),
        _make_entry(qid + 2, "test_db", "Find students with GPA above 3.5",
                    "SELECT name FROM students WHERE gpa > 3.5", difficulty="simple"),
        _make_entry(qid + 3, "test_db",
                    "List students and their courses using JOIN",
                    "SELECT s.name, c.name FROM students s JOIN courses c ON s.id = c.student_id",
                    difficulty="moderate"),
        _make_entry(qid + 4, "test_db",
                    "What is the average GPA of students in each department?",
                    "SELECT dept, AVG(gpa) FROM students GROUP BY dept",
                    difficulty="moderate"),
    ]
    entries.extend(test_db_entries)
    qid += 5

    # db_sales — 5 JOIN entries
    for i in range(5):
        entries.append(_make_entry(
            qid + i, "db_sales",
            f"Show orders with customer details using JOIN {i}",
            f"SELECT o.id, c.name FROM orders_{i} o JOIN customers c ON o.customer_id = c.id",
            difficulty="moderate",
        ))
    qid += 5

    # db_hr — 5 JOIN entries
    for i in range(5):
        entries.append(_make_entry(
            qid + i, "db_hr",
            f"Find employees and their departments {i}",
            f"SELECT e.name, d.name FROM employees e JOIN departments d ON e.dept_id = d.id",
            difficulty="moderate",
        ))
    qid += 5

    # db_finance — 5 JOIN entries
    for i in range(5):
        entries.append(_make_entry(
            qid + i, "db_finance",
            f"List invoices with client information {i}",
            f"SELECT i.number, c.name FROM invoices i JOIN clients c ON i.client_id = c.id",
            difficulty="moderate",
        ))
    qid += 5

    # db_ecommerce — 5 JOIN entries
    for i in range(5):
        entries.append(_make_entry(
            qid + i, "db_ecommerce",
            f"Show products with category information {i}",
            f"SELECT p.name, cat.label FROM products p JOIN categories cat ON p.cat_id = cat.id",
            difficulty="moderate",
        ))
    qid += 5

    # db_school — 5 aggregation (COUNT/SUM/AVG)
    for i in range(5):
        entries.append(_make_entry(
            qid + i, "db_school",
            f"Count the number of students in grade {i + 1}",
            f"SELECT COUNT(*) FROM students WHERE grade = {i + 1}",
            difficulty="simple",
        ))
    qid += 5

    # db_hospital — 5 aggregation
    for i in range(5):
        entries.append(_make_entry(
            qid + i, "db_hospital",
            f"What is the average age of patients in ward {i}?",
            f"SELECT AVG(age) FROM patients WHERE ward = {i}",
            difficulty="simple",
        ))
    qid += 5

    # db_logistics — 5 aggregation
    for i in range(5):
        entries.append(_make_entry(
            qid + i, "db_logistics",
            f"What is the total weight shipped in region {i}?",
            f"SELECT SUM(weight) FROM shipments WHERE region = {i}",
            difficulty="simple",
        ))
    qid += 5

    # db_retail — 5 simple SELECT
    for i in range(5):
        entries.append(_make_entry(
            qid + i, "db_retail",
            f"List all products in category {i}",
            f"SELECT name, price FROM products WHERE category_id = {i}",
            difficulty="simple",
        ))
    qid += 5

    # db_inventory — 5 simple SELECT
    for i in range(5):
        entries.append(_make_entry(
            qid + i, "db_inventory",
            f"Find items with quantity below threshold {i}",
            f"SELECT item_name FROM inventory WHERE quantity < {i + 10}",
            difficulty="simple",
        ))

    return entries


@pytest.fixture(scope="module")
def training_entries() -> list[BirdEntry]:
    return _make_training_entries()


@pytest.fixture(scope="module")
def built_store(training_entries) -> ExampleStore:
    """An ExampleStore built from the mock training entries."""
    store = ExampleStore()
    store.build(training_entries)
    return store


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_skeleton_masking_removes_entities():
    """'Find students in Alameda County' → skeleton contains '[ENTITY]', not 'Alameda'."""
    store = ExampleStore()
    skeleton = store.mask_question("Find students in Alameda County")
    assert "[ENTITY]" in skeleton, f"Expected [ENTITY] in skeleton: {skeleton!r}"
    assert "Alameda" not in skeleton, f"'Alameda' should be masked: {skeleton!r}"


def test_skeleton_masking_removes_numbers():
    """'Score above 90' → skeleton contains '[NUM]', not '90'."""
    store = ExampleStore()
    skeleton = store.mask_question("Score above 90")
    assert "[NUM]" in skeleton, f"Expected [NUM] in skeleton: {skeleton!r}"
    assert "90" not in skeleton, f"'90' should be masked: {skeleton!r}"


def test_retrieval_finds_structurally_similar(built_store):
    """A JOIN question should retrieve JOIN examples in the top-3."""
    results = built_store.query(
        "Show me customer orders with their product details",
        db_id="other_db",
        top_k=8,
    )
    assert len(results) > 0
    top_3_sql = [r.sql.upper() for r in results[:3]]
    has_join = any("JOIN" in sql for sql in top_3_sql)
    assert has_join, f"Expected JOIN in top-3 SQL, got: {top_3_sql}"


def test_top_k_count(built_store):
    """top_k=8 returns at most 8 results."""
    results = built_store.query(
        "How many records are there in the table?",
        db_id="other_db",
        top_k=8,
    )
    assert len(results) <= 8


def test_db_id_exclusion(built_store):
    """When db_id='test_db', no returned examples should have db_id='test_db'."""
    results = built_store.query(
        "What are the names of all students?",
        db_id="test_db",
        top_k=8,
    )
    for entry in results:
        assert entry.db_id != "test_db", (
            f"Found example from excluded db_id='test_db': {entry}"
        )


def test_examples_include_sql(built_store):
    """All returned ExampleEntry objects should have non-empty sql."""
    results = built_store.query(
        "List all records from the database",
        db_id="other_db",
        top_k=5,
    )
    assert len(results) > 0
    for entry in results:
        assert entry.sql.strip(), f"Entry has empty sql: {entry}"


def test_aggregation_query_retrieves_aggregation_examples(built_store):
    """An aggregation query should retrieve examples with COUNT/SUM/AVG in SQL."""
    results = built_store.query(
        "What is the average value per group?",
        db_id="other_db",
        top_k=8,
    )
    assert len(results) > 0
    all_sqls = [r.sql.upper() for r in results]
    has_aggregation = any("COUNT" in sql or "SUM" in sql or "AVG" in sql for sql in all_sqls)
    assert has_aggregation, f"Expected aggregation SQL in top-8, got: {all_sqls}"


def test_serialization_roundtrip(built_store, tmp_path):
    """Save + load should produce the same results for the same query."""
    faiss_path = str(tmp_path / "store.faiss")
    meta_path = str(tmp_path / "store_meta.json")

    built_store.save(faiss_path, meta_path)
    loaded = ExampleStore.load(faiss_path, meta_path)

    question = "List employees and departments using join"
    original = built_store.query(question, db_id="other_db", top_k=5)
    restored = loaded.query(question, db_id="other_db", top_k=5)

    assert len(original) == len(restored)
    for o, r in zip(original, restored):
        assert o.question_id == r.question_id
        assert o.sql == r.sql
        assert abs(o.similarity_score - r.similarity_score) < 1e-5


def test_similarity_scores_ordered(built_store):
    """Results should be in descending order of similarity score."""
    results = built_store.query(
        "What is the average count per category?",
        db_id="other_db",
        top_k=8,
    )
    assert len(results) >= 2
    scores = [r.similarity_score for r in results]
    assert scores == sorted(scores, reverse=True), (
        "Results are not in descending similarity order"
    )


def test_empty_training_set_handled():
    """ExampleStore.build([]) should raise ValueError."""
    store = ExampleStore()
    with pytest.raises(ValueError, match="empty"):
        store.build([])
