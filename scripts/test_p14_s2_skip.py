"""
test_p14_s2_skip.py — Verification script for ISSUE P1-4 fix.

Tests that the S2 LLM call is correctly skipped when S1 coverage is high.

Scenario tests (mocked LLM):
  1. S1 covers 85% of candidates   → 1 LLM call
  2. 0 candidates remain            → 1 LLM call
  3. 2 candidates remain (< 3)      → 1 LLM call
  4. 5 candidates remain, 40% cov   → 2 LLM calls  (normal)
  5. 15 candidates remain, 40% cov  → 2 LLM calls  (normal)

BIRD dev validation (mocked LLM, real FAISS indices):
  Q427, Q1025, Q109, Q149, Q953 — the 5 questions identified in the inspection
  report as producing S1==S2.  With real FAISS candidates, verify that the
  skip logic now triggers for them.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.base import LLMResponse
from src.indexing.faiss_index import FieldMatch, FAISSIndex
from src.grounding.context_grounder import GroundingContext
from src.schema_linking.schema_linker import link_schema, LinkedSchemas

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

PASS = "[PASS]"
FAIL = "[FAIL]"

# Reusable tiny schema
TINY_DDL = """\
-- Table: students
CREATE TABLE students (
  id INTEGER PRIMARY KEY,  -- Student ID. PRIMARY KEY.
  name TEXT,  -- Full name.
  gpa REAL,  -- GPA.
  age INTEGER,  -- Age.
  email TEXT,  -- Email.
  department_id INTEGER,  -- FK to departments.
  year INTEGER,  -- Year.
  status TEXT  -- Status.
);
-- Foreign keys: department_id REFERENCES departments(id)
-- Table: departments
CREATE TABLE departments (
  id INTEGER PRIMARY KEY,  -- Dept ID. PRIMARY KEY.
  name TEXT,  -- Dept name.
  code TEXT,  -- Dept code.
  budget REAL  -- Budget.
);
"""
TINY_MARKDOWN = """\
## Table: students
*Students*

| Column | Type | Description | Sample Values |
|--------|------|-------------|---------------|
| id | INTEGER (PK) | Student ID | 1, 2 |
| name | TEXT | Name | Alice |
| gpa | REAL | GPA | 3.9 |
| age | INTEGER | Age | 20 |
| email | TEXT | Email | alice@uni.edu |
| department_id | INTEGER (FK) | Dept FK | 1 |
| year | INTEGER | Year | 2021 |
| status | TEXT | Status | Active |

## Table: departments
*Departments*

| Column | Type | Description | Sample Values |
|--------|------|-------------|---------------|
| id | INTEGER (PK) | Dept ID | 1 |
| name | TEXT | Name | CS |
| code | TEXT | Code | CS |
| budget | REAL | Budget | 1000000 |
"""
TINY_FIELDS = [
    ("students", "id",            "Student ID",    "Student ID. PRIMARY KEY."),
    ("students", "name",          "Name",          "Full name."),
    ("students", "gpa",           "GPA",           "GPA."),
    ("students", "age",           "Age",           "Age."),
    ("students", "email",         "Email",         "Email."),
    ("students", "department_id", "Dept FK",       "FK to departments."),
    ("students", "year",          "Year",          "Year."),
    ("students", "status",        "Status",        "Status."),
    ("departments", "id",         "Dept ID",       "Dept ID. PRIMARY KEY."),
    ("departments", "name",       "Dept name",     "Dept name."),
    ("departments", "code",       "Dept code",     "Dept code."),
    ("departments", "budget",     "Budget",        "Budget."),
]

_EMPTY_GROUNDING = GroundingContext(matched_cells=[], schema_hints=[], few_shot_examples=[])


def make_field_match(table: str, col: str, score: float = 0.80) -> FieldMatch:
    return FieldMatch(table=table, column=col, similarity_score=score,
                      short_summary=f"{col} summary", long_summary=f"{col} long.")


def make_faiss(fields: list[tuple[str, str]]) -> MagicMock:
    """Create a mock FAISSIndex that returns the given (table, col) pairs."""
    mock = MagicMock()
    mock.query.return_value = [
        make_field_match(t, c, score=1.0 - i * 0.03)
        for i, (t, c) in enumerate(fields)
    ]
    return mock


def make_s1_response(fields: list[tuple[str, str]]) -> LLMResponse:
    return LLMResponse(tool_inputs=[{
        "selected_fields": [
            {"table": t, "column": c, "reason": f"S1 reason for {t}.{c}"}
            for t, c in fields
        ]
    }])


def make_mock_client(s1_fields: list[tuple[str, str]]) -> AsyncMock:
    """Return a mock LLM client whose generate() returns s1_fields."""
    client = AsyncMock()
    client.generate = AsyncMock(return_value=make_s1_response(s1_fields))
    return client


# ---------------------------------------------------------------------------
# Scenario tests
# ---------------------------------------------------------------------------

async def run_scenario(
    label: str,
    faiss_fields: list[tuple[str, str]],
    s1_fields_selected: list[tuple[str, str]],
    expected_calls: int,
) -> bool:
    """
    Run link_schema with a mocked LLM and verify the call count.

    Parameters
    ----------
    label:             Human-readable test label.
    faiss_fields:      FAISS candidates returned by the mock index.
    s1_fields_selected: Fields the S1 LLM response selects.
    expected_calls:    Expected number of client.generate() calls.

    Returns True on pass, False on fail.
    """
    mock_faiss = make_faiss(faiss_fields)

    with patch("src.schema_linking.schema_linker.get_client") as mock_get_client:
        mock_client = make_mock_client(s1_fields_selected)
        mock_get_client.return_value = mock_client

        result = await link_schema(
            question="Test question for scenario verification.",
            evidence="",
            grounding_context=_EMPTY_GROUNDING,
            faiss_index=mock_faiss,
            full_ddl=TINY_DDL,
            full_markdown=TINY_MARKDOWN,
            available_fields=TINY_FIELDS,
        )

    call_count = mock_client.generate.call_count
    s1_set = set(result.s1_fields)
    s2_set = set(result.s2_fields)
    invariant_ok = s1_set.issubset(s2_set)

    total_cands = len(faiss_fields)
    # Approximate coverage (actual s1_extended may be larger due to PK/FK)
    selected_in_cands = sum(1 for f in s1_fields_selected if f in set(faiss_fields))

    passed = (call_count == expected_calls) and invariant_ok
    status = PASS if passed else FAIL
    print(
        f"  {status} {label}\n"
        f"       FAISS cands={total_cands}, S1 selected={len(s1_fields_selected)}, "
        f"expected_calls={expected_calls}, actual_calls={call_count}, "
        f"s1_subset_s2={invariant_ok}"
    )
    if not passed:
        if call_count != expected_calls:
            print(f"         ERROR: call_count mismatch: expected {expected_calls}, got {call_count}")
        if not invariant_ok:
            print(f"         ERROR: S1 not subset of S2: {s1_set - s2_set}")
    return passed


async def scenario_tests() -> int:
    """Run all scenario tests. Returns number of failures."""
    print("\n=== Scenario Tests (mocked LLM) ===\n")
    failures = 0

    # Scenario 1: S1 covers 85% of candidates → skip
    # 20 FAISS candidates, S1 selects 17 (17/20 = 85%) → skip
    faiss_20 = [(f"students", c) for c in [
        "id", "name", "gpa", "age", "email",
        "department_id", "year", "status",
    ]] + [(f"departments", c) for c in ["id", "name", "code", "budget"]]
    # extend to ~17 total with repeats from another table (not in available, will be filtered)
    # Use only available fields to avoid hallucination filtering
    faiss_17 = [
        ("students", "id"), ("students", "name"), ("students", "gpa"),
        ("students", "age"), ("students", "email"), ("students", "department_id"),
        ("students", "year"), ("students", "status"),
        ("departments", "id"), ("departments", "name"), ("departments", "code"),
        ("departments", "budget"),
    ]  # 12 total
    # S1 selects 10 of 12 (83%) → should skip
    s1_10_of_12 = [
        ("students", "id"), ("students", "name"), ("students", "gpa"),
        ("students", "age"), ("students", "email"), ("students", "department_id"),
        ("students", "year"), ("students", "status"),
        ("departments", "id"), ("departments", "name"),
    ]
    ok = await run_scenario(
        "Scenario 1 — S1 covers 83% of 12 candidates → skip S2 (1 call)",
        faiss_17, s1_10_of_12, expected_calls=1,
    )
    if not ok:
        failures += 1

    # Scenario 2: 0 candidates remain → skip
    faiss_3 = [("students", "id"), ("students", "gpa"), ("students", "name")]
    s1_all_3 = [("students", "id"), ("students", "gpa"), ("students", "name")]
    ok = await run_scenario(
        "Scenario 2 — 0 remaining candidates (S1 consumes all) → skip S2 (1 call)",
        faiss_3, s1_all_3, expected_calls=1,
    )
    if not ok:
        failures += 1

    # Scenario 3: 2 remaining candidates (< 3) → skip
    faiss_8 = [
        ("students", "id"), ("students", "name"), ("students", "gpa"),
        ("students", "age"), ("students", "email"), ("students", "department_id"),
        ("students", "year"), ("students", "status"),
    ]
    s1_6_of_8 = [
        ("students", "id"), ("students", "name"), ("students", "gpa"),
        ("students", "age"), ("students", "email"), ("students", "department_id"),
    ]  # leaves "year" and "status" → 2 remaining < 3 → skip
    ok = await run_scenario(
        "Scenario 3 — 2 remaining candidates (< 3 threshold) → skip S2 (1 call)",
        faiss_8, s1_6_of_8, expected_calls=1,
    )
    if not ok:
        failures += 1

    # Scenario 4: 5 candidates remain, coverage 40% → proceed with S2
    faiss_10 = [
        ("students", "id"), ("students", "name"), ("students", "gpa"),
        ("students", "age"), ("students", "email"),
        ("departments", "id"), ("departments", "name"), ("departments", "code"),
        ("departments", "budget"), ("students", "department_id"),
    ]
    s1_4_of_10 = [
        ("students", "id"), ("students", "name"),
        ("students", "gpa"), ("students", "age"),
    ]  # coverage: 4/10 = 40%; 6 remaining (≥3) → proceed
    ok = await run_scenario(
        "Scenario 4 — 5 remaining (40% coverage) → S2 proceeds (2 calls)",
        faiss_10, s1_4_of_10, expected_calls=2,
    )
    if not ok:
        failures += 1

    # Scenario 5: 15 candidates remain, coverage 40% → proceed with S2
    faiss_15 = [
        ("students", "id"), ("students", "name"), ("students", "gpa"),
        ("students", "age"), ("students", "email"), ("students", "department_id"),
        ("students", "year"), ("students", "status"),
        ("departments", "id"), ("departments", "name"), ("departments", "code"),
        ("departments", "budget"),
        ("students", "gpa"),     # duplicates after dedup → actually 12 unique
        ("departments", "name"), # duplicate
        ("students", "status"),  # duplicate
    ]
    # Unique 12 fields after dedup by link_schema's seen_keys set
    # S1 selects 5 → coverage 5/12 ≈ 42%; 7+ remaining → proceed
    s1_5 = [
        ("students", "id"), ("students", "name"), ("students", "gpa"),
        ("students", "age"), ("students", "email"),
    ]
    ok = await run_scenario(
        "Scenario 5 — 15 FAISS results (12 unique, 40% coverage) → S2 proceeds (2 calls)",
        faiss_15, s1_5, expected_calls=2,
    )
    if not ok:
        failures += 1

    return failures


# ---------------------------------------------------------------------------
# BIRD dev validation (questions Q427, Q1025, Q109, Q149, Q953)
# ---------------------------------------------------------------------------

async def bird_dev_validation() -> int:
    """
    Load the 5 affected questions from BIRD dev, use real FAISS indices,
    mock the LLM to return all candidates as S1 (simulating the worst-case
    S1==S2 scenario), and verify that skip logic triggers.

    Returns number of failures.
    """
    print("\n=== BIRD Dev Validation (real FAISS, mocked LLM) ===\n")
    failures = 0

    dev_path = PROJECT_ROOT / "data" / "bird" / "dev" / "dev.json"
    if not dev_path.exists():
        print(f"  [SKIP] BIRD dev.json not found at {dev_path}")
        return 0

    with open(dev_path) as f:
        dev_data = json.load(f)

    target_qids = {427, 1025, 109, 149, 953}
    questions = {q["question_id"]: q for q in dev_data if q["question_id"] in target_qids}

    if not questions:
        print("  [SKIP] Target question IDs not found in dev.json")
        return 0

    preprocessed_dir = PROJECT_ROOT / "data" / "preprocessed"
    schemas_dir = preprocessed_dir / "schemas"
    indices_dir = preprocessed_dir / "indices"

    # Map db_id → loaded FAISS index
    faiss_cache: dict[str, FAISSIndex | None] = {}

    def load_faiss(db_id: str) -> FAISSIndex | None:
        if db_id in faiss_cache:
            return faiss_cache[db_id]
        index_path = indices_dir / f"{db_id}_faiss.index"
        fields_path = indices_dir / f"{db_id}_faiss_fields.json"
        if not index_path.exists() or not fields_path.exists():
            print(f"    [SKIP] FAISS index for {db_id} not found")
            faiss_cache[db_id] = None
            return None
        try:
            faiss_idx = FAISSIndex.load(str(index_path), str(fields_path))
            faiss_cache[db_id] = faiss_idx
            return faiss_idx
        except Exception as exc:
            print(f"    [SKIP] Failed to load FAISS for {db_id}: {exc}")
            faiss_cache[db_id] = None
            return None

    def load_schema(db_id: str) -> tuple[str, str]:
        ddl_path = schemas_dir / f"{db_id}_ddl.sql"
        md_path = schemas_dir / f"{db_id}_markdown.md"
        ddl = ddl_path.read_text() if ddl_path.exists() else ""
        md = md_path.read_text() if md_path.exists() else ""
        return ddl, md

    def load_available_fields(db_id: str) -> list[tuple[str, str, str, str]]:
        """Load available fields from the FAISS fields JSON."""
        fields_path = indices_dir / f"{db_id}_faiss_fields.json"
        if not fields_path.exists():
            return []
        with open(fields_path) as f:
            data = json.load(f)
        # Format: list of {table, column, short_summary, long_summary}
        result = []
        for item in data:
            result.append((
                item.get("table", ""),
                item.get("column", ""),
                item.get("short_summary", ""),
                item.get("long_summary", ""),
            ))
        return result

    for qid in sorted(target_qids):
        if qid not in questions:
            print(f"  [SKIP] Q{qid} not in dev.json")
            continue

        q = questions[qid]
        db_id = q["db_id"]
        question_text = q["question"]
        evidence = q.get("evidence", "")

        print(f"  Q{qid} ({db_id}, {q['difficulty']}):")
        print(f"    Question: {question_text[:70]}")

        faiss_idx = load_faiss(db_id)
        if faiss_idx is None:
            print(f"    [SKIP] No FAISS index for {db_id}")
            continue

        ddl, md = load_schema(db_id)
        if not ddl:
            print(f"    [SKIP] No DDL schema for {db_id}")
            continue

        available_fields = load_available_fields(db_id)
        if not available_fields:
            print(f"    [SKIP] No available_fields for {db_id}")
            continue

        # Query the real FAISS index to see what candidates we'd get
        from src.config.settings import settings
        faiss_candidates = faiss_idx.query(question_text, top_k=settings.faiss_top_k)
        n_candidates = len(faiss_candidates)

        # Simulate worst-case: S1 LLM selects ALL FAISS candidates
        # (this is the scenario where S1==S2 was observed)
        s1_all_candidates = [
            {"table": fm.table, "column": fm.column, "reason": "Selected in S1"}
            for fm in faiss_candidates
        ]

        s1_response = LLMResponse(tool_inputs=[{"selected_fields": s1_all_candidates}])

        grounding = _EMPTY_GROUNDING  # no real grounding context needed for this test

        with patch("src.schema_linking.schema_linker.get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.generate = AsyncMock(return_value=s1_response)
            mock_get_client.return_value = mock_client

            result = await link_schema(
                question=question_text,
                evidence=evidence,
                grounding_context=grounding,
                faiss_index=faiss_idx,
                full_ddl=ddl,
                full_markdown=md,
                available_fields=available_fields,
            )

        call_count = mock_client.generate.call_count
        n_s1 = len(result.s1_fields)
        n_s2 = len(result.s2_fields)
        s1_coverage = n_s1 / max(n_candidates, 1)
        s2_skipped = call_count == 1
        invariant_ok = set(result.s1_fields).issubset(set(result.s2_fields))

        status = PASS if (s2_skipped and invariant_ok) else FAIL
        print(
            f"    {status} FAISS cands={n_candidates}, S1 selected all={n_candidates}, "
            f"s1_extended={n_s1}, coverage={s1_coverage:.0%}, "
            f"calls={call_count}, S2_skipped={s2_skipped}, s1_subset_s2={invariant_ok}"
        )
        if not s2_skipped:
            failures += 1
            print(f"         ERROR: expected S2 to be skipped (coverage={s1_coverage:.0%})")
        if not invariant_ok:
            failures += 1
            print(f"         ERROR: S1 not subset of S2")

    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    print("=" * 70)
    print("ISSUE P1-4 Fix Verification — S2 Skip Logic")
    print("=" * 70)

    scenario_failures = await scenario_tests()
    bird_failures = await bird_dev_validation()

    total_failures = scenario_failures + bird_failures
    print("\n" + "=" * 70)
    if total_failures == 0:
        print(f"ALL TESTS PASSED (scenario_failures=0, bird_failures=0)")
    else:
        print(
            f"FAILURES: {total_failures} "
            f"(scenario={scenario_failures}, bird={bird_failures})"
        )
    print("=" * 70)

    sys.exit(0 if total_failures == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
