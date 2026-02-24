"""
test_p13_s1_cap.py — Verify S1 field-count cap (≤20 fields) in schema_linker.

Tests:
  1. Unit: large schema (35 FAISS candidates) → s1_fields ≤ 20 and S1 ⊆ S2
  2. Unit: small schema (8 FAISS candidates)  → unchanged, S1 ⊆ S2
  3. Unit: S1 user message contains "20" or "most relevant"
  4. BIRD dev validation — QID 1025 (european_football_2)
  5. BIRD dev validation — QID 994  (formula_1)
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.base import LLMResponse
from src.indexing.faiss_index import FieldMatch
from src.grounding.context_grounder import GroundingContext
from src.schema_linking.schema_linker import link_schema

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMPTY_GROUNDING = GroundingContext(matched_cells=[], schema_hints=[], few_shot_examples=[])


def make_faiss_mock(fields: list[tuple[str, str]]) -> MagicMock:
    """Return a mock FAISSIndex whose .query() returns the given fields."""
    mock = MagicMock()
    mock.query.return_value = [
        FieldMatch(
            table=t,
            column=c,
            similarity_score=max(0.0, 1.0 - i * 0.02),
            short_summary=f"{c} summary",
            long_summary=f"{c} long.",
        )
        for i, (t, c) in enumerate(fields)
    ]
    return mock


def make_s1_response(fields: list[tuple[str, str]]) -> LLMResponse:
    """LLMResponse that selects all given fields as S1."""
    return LLMResponse(
        tool_inputs=[
            {
                "selected_fields": [
                    {"table": t, "column": c, "reason": "selected"}
                    for t, c in fields
                ]
            }
        ]
    )


def _make_schema_for_fields(fields: list[tuple[str, str]]) -> tuple[str, str, list]:
    """
    Build a minimal DDL, Markdown schema, and available_fields list
    for a given set of (table, column) pairs.

    Tables are inferred from the table part; each column is given TEXT type.
    The first column of each table is designated as PRIMARY KEY.
    """
    # Group by table
    by_table: dict[str, list[str]] = {}
    for t, c in fields:
        by_table.setdefault(t, []).append(c)

    ddl_blocks: list[str] = []
    md_blocks: list[str] = []
    available_fields: list[tuple[str, str, str, str]] = []

    for table, cols in by_table.items():
        # DDL
        col_lines = []
        for i, col in enumerate(cols):
            if i == 0:
                col_lines.append(f"    {col} TEXT PRIMARY KEY,")
            else:
                col_lines.append(f"    {col} TEXT,")
        # Fix trailing comma on last line
        last = col_lines[-1]
        if last.endswith(","):
            col_lines[-1] = last[:-1]
        ddl_blocks.append(
            f"-- Table: {table}\n"
            f"CREATE TABLE {table} (\n"
            + "\n".join(col_lines)
            + "\n);"
        )
        # Markdown
        md_rows = [f"| {col} | TEXT | {col} description | |" for col in cols]
        md_blocks.append(
            f"## Table: {table}\n"
            "| Column | Type | Description | Sample Values |\n"
            "|--------|------|-------------|---------------|\n"
            + "\n".join(md_rows)
        )
        # available_fields
        for col in cols:
            available_fields.append((table, col, f"{col} summary", f"{col} long."))

    return "\n\n".join(ddl_blocks), "\n\n".join(md_blocks), available_fields


# ---------------------------------------------------------------------------
# Test 1 — Large schema (35 candidates) → S1 capped at ≤20
# ---------------------------------------------------------------------------

async def test_large_schema_cap() -> tuple[bool, str]:
    """Mock LLM selects all 35 FAISS candidates as S1; verify len(s1) ≤ 20."""
    n_fields = 35
    fields = [(f"table{i // 7}", f"col{i}") for i in range(n_fields)]
    full_ddl, full_markdown, available_fields = _make_schema_for_fields(fields)

    faiss_mock = make_faiss_mock(fields)

    # Mock client: first call (S1) returns all 35, second call (S2) returns []
    s1_resp = make_s1_response(fields)
    s2_resp = LLMResponse(tool_inputs=[{"selected_fields": []}])

    mock_client = AsyncMock()
    mock_client.generate = AsyncMock(side_effect=[s1_resp, s2_resp])

    with patch("src.schema_linking.schema_linker.get_client", return_value=mock_client):
        result = await link_schema(
            question="How many rows in table0?",
            evidence="",
            grounding_context=EMPTY_GROUNDING,
            faiss_index=faiss_mock,
            full_ddl=full_ddl,
            full_markdown=full_markdown,
            available_fields=available_fields,
        )

    s1_count = len(result.s1_fields)
    s1_set = set(result.s1_fields)
    s2_set = set(result.s2_fields)

    if s1_count > 20:
        return False, f"FAIL: s1_fields={s1_count} > 20 (should be capped)"
    if not s1_set.issubset(s2_set):
        return False, f"FAIL: S1 ⊄ S2 — extra in S1: {s1_set - s2_set}"
    return True, f"PASS: s1_fields={s1_count} (≤20), S1 ⊆ S2 holds"


# ---------------------------------------------------------------------------
# Test 2 — Small schema (8 candidates) → unchanged, still ≤20, S1 ⊆ S2
# ---------------------------------------------------------------------------

async def test_small_schema_unchanged() -> tuple[bool, str]:
    """8 candidates; cap should not truncate anything."""
    n_fields = 8
    fields = [(f"table{i // 4}", f"col{i}") for i in range(n_fields)]
    full_ddl, full_markdown, available_fields = _make_schema_for_fields(fields)

    faiss_mock = make_faiss_mock(fields)

    s1_resp = make_s1_response(fields)
    s2_resp = LLMResponse(tool_inputs=[{"selected_fields": []}])

    mock_client = AsyncMock()
    mock_client.generate = AsyncMock(side_effect=[s1_resp, s2_resp])

    with patch("src.schema_linking.schema_linker.get_client", return_value=mock_client):
        result = await link_schema(
            question="Find all records in table0.",
            evidence="",
            grounding_context=EMPTY_GROUNDING,
            faiss_index=faiss_mock,
            full_ddl=full_ddl,
            full_markdown=full_markdown,
            available_fields=available_fields,
        )

    s1_count = len(result.s1_fields)
    s1_set = set(result.s1_fields)
    s2_set = set(result.s2_fields)

    if s1_count > 20:
        return False, f"FAIL: s1_fields={s1_count} > 20"
    if not s1_set.issubset(s2_set):
        return False, f"FAIL: S1 ⊄ S2"
    return True, f"PASS: s1_fields={s1_count} (≤20, uncapped), S1 ⊆ S2 holds"


# ---------------------------------------------------------------------------
# Test 3 — S1 user message contains "20" or "most relevant"
# ---------------------------------------------------------------------------

async def test_s1_message_contains_limit() -> tuple[bool, str]:
    """Verify the user message for S1 contains the 20-field instruction."""
    fields = [("tableA", f"col{i}") for i in range(5)]
    full_ddl, full_markdown, available_fields = _make_schema_for_fields(fields)
    faiss_mock = make_faiss_mock(fields)

    captured_messages: list = []

    async def capturing_generate(**kwargs):
        captured_messages.append(kwargs.get("messages", []))
        return LLMResponse(tool_inputs=[{"selected_fields": []}])

    mock_client = AsyncMock()
    mock_client.generate = capturing_generate

    with patch("src.schema_linking.schema_linker.get_client", return_value=mock_client):
        await link_schema(
            question="Sample question.",
            evidence="",
            grounding_context=EMPTY_GROUNDING,
            faiss_index=faiss_mock,
            full_ddl=full_ddl,
            full_markdown=full_markdown,
            available_fields=available_fields,
        )

    if not captured_messages:
        return False, "FAIL: No generate() calls captured"

    first_call_messages = captured_messages[0]
    user_content = ""
    for msg in first_call_messages:
        if msg.get("role") == "user":
            user_content = msg.get("content", "")
            break

    has_20 = "20" in user_content
    has_most_relevant = "most relevant" in user_content.lower()

    if has_20 or has_most_relevant:
        return True, f"PASS: S1 user message contains limit instruction (has_20={has_20}, has_most_relevant={has_most_relevant})"
    return False, f"FAIL: S1 user message does not contain '20' or 'most relevant'. Content: {user_content!r}"


# ---------------------------------------------------------------------------
# Test 4 & 5 — BIRD dev validation (QIDs 1025 and 994)
# ---------------------------------------------------------------------------

async def test_bird_qid(qid: int) -> tuple[bool, str]:
    """
    Load real FAISS index and schema for the given QID, mock LLM to return
    ALL candidates as S1 (worst-case over-selection), verify s1_fields ≤ 20.
    """
    dev_json = PROJECT_ROOT / "data" / "bird" / "dev" / "dev.json"
    if not dev_json.exists():
        return False, f"SKIP: dev.json not found at {dev_json}"

    with open(dev_json) as f:
        dev_data = json.load(f)

    # Find the entry with question_id == qid
    entry = None
    for item in dev_data:
        if item.get("question_id") == qid:
            entry = item
            break

    if entry is None:
        return False, f"SKIP: QID {qid} not found in dev.json"

    db_id = entry["db_id"]
    question = entry["question"]
    evidence = entry.get("evidence", "")

    # Load FAISS index
    indices_dir = PROJECT_ROOT / "data" / "preprocessed" / "indices"
    faiss_index_path = indices_dir / f"{db_id}_faiss.index"
    faiss_fields_path = indices_dir / f"{db_id}_faiss_fields.json"

    if not faiss_index_path.exists() or not faiss_fields_path.exists():
        return False, f"SKIP: FAISS index files not found for db_id={db_id}"

    from src.indexing.faiss_index import FAISSIndex
    faiss_index = FAISSIndex.load(str(faiss_index_path), str(faiss_fields_path))

    # Load schemas
    schemas_dir = PROJECT_ROOT / "data" / "preprocessed" / "schemas"
    ddl_path = schemas_dir / f"{db_id}_ddl.sql"
    md_path = schemas_dir / f"{db_id}_markdown.md"

    if not ddl_path.exists() or not md_path.exists():
        return False, f"SKIP: Schema files not found for db_id={db_id}"

    full_ddl = ddl_path.read_text()
    full_markdown = md_path.read_text()

    # Build available_fields from FAISS metadata
    fields_meta = json.loads(faiss_fields_path.read_text())
    available_fields = [
        (m["table"], m["column"], m.get("short_summary", ""), m.get("long_summary", ""))
        for m in fields_meta
    ]

    # Get all FAISS candidates for this question (real query, top_k=35 to simulate large set)
    all_candidates = faiss_index.query(question, top_k=35)
    candidate_pairs = [(fm.table, fm.column) for fm in all_candidates]

    # Mock LLM: always returns ALL FAISS candidates as S1 (worst-case)
    async def always_select_all(**kwargs):
        return LLMResponse(
            tool_inputs=[
                {
                    "selected_fields": [
                        {"table": t, "column": c, "reason": "worst-case selection"}
                        for t, c in candidate_pairs
                    ]
                }
            ]
        )

    mock_client = AsyncMock()
    mock_client.generate = always_select_all

    with patch("src.schema_linking.schema_linker.get_client", return_value=mock_client):
        result = await link_schema(
            question=question,
            evidence=evidence,
            grounding_context=EMPTY_GROUNDING,
            faiss_index=faiss_index,
            full_ddl=full_ddl,
            full_markdown=full_markdown,
            available_fields=available_fields,
        )

    s1_count = len(result.s1_fields)
    s1_set = set(result.s1_fields)
    s2_set = set(result.s2_fields)

    n_candidates = len(candidate_pairs)
    details = f"db_id={db_id}, FAISS candidates={n_candidates}, s1_fields={s1_count}"

    if s1_count > 20:
        return False, f"FAIL QID={qid}: {details} — exceeds cap of 20"
    if not s1_set.issubset(s2_set):
        return False, f"FAIL QID={qid}: {details} — S1 ⊄ S2"
    return True, f"PASS QID={qid}: {details}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    tests = [
        ("Test 1 — Large schema (35 candidates) capped at ≤20", test_large_schema_cap()),
        ("Test 2 — Small schema (8 candidates) unchanged",       test_small_schema_unchanged()),
        ("Test 3 — S1 user message contains '20' instruction",   test_s1_message_contains_limit()),
        ("Test 4 — BIRD dev QID=1025 (european_football_2)",     test_bird_qid(1025)),
        ("Test 5 — BIRD dev QID=994 (formula_1)",                test_bird_qid(994)),
    ]

    results: list[tuple[str, bool, str]] = []
    for name, coro in tests:
        try:
            passed, msg = await coro
        except Exception as exc:
            passed = False
            msg = f"ERROR: {exc}"
        results.append((name, passed, msg))
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}")
        print(f"       {msg}")

    print()
    passed_count = sum(1 for _, p, _ in results if p)
    total = len(results)
    print(f"Summary: {passed_count}/{total} tests passed")
    if passed_count < total:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
