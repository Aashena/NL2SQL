#!/usr/bin/env python3
"""
Test script for P0-1: B1 max_tokens fix (2000 → 4096).

Tests:
  1. Unit test — verifies both B1 calls pass max_tokens=4096 via mocked LLM client.
  2. BIRD dev validation — for QIDs 28, 1526, 852: loads real schema data, mocks
     the LLM to return a STOP response, and asserts all 4 candidates have error_flag=False.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Path setup — ensure project root is on sys.path
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.llm.base import LLMResponse
from src.generation.standard_generator import StandardAndComplexGenerator
from src.schema_linking.schema_linker import LinkedSchemas
from src.grounding.context_grounder import GroundingContext
from src.config.settings import settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BIRD_DEV_JSON = REPO_ROOT / "data" / "bird" / "dev" / "dev.json"
PREPROCESSED_DIR = REPO_ROOT / "data" / "preprocessed"
INDICES_DIR = PREPROCESSED_DIR / "indices"
SCHEMAS_DIR = PREPROCESSED_DIR / "schemas"

AFFECTED_QIDS = [28, 1526, 852]


def _make_mock_client(finish_reason: str = "STOP", text: str = "SELECT 1") -> MagicMock:
    """Build a mock LLM client whose generate() always returns a STOP response."""
    mock_client = MagicMock()
    mock_response = LLMResponse(
        tool_inputs=[],
        text=text,
        finish_reason=finish_reason,
    )
    mock_client.generate = AsyncMock(return_value=mock_response)
    return mock_client


def _make_minimal_schemas(s1_markdown: str = "## Table: t\n| Column | Type |\n|---|---|\n| id | INTEGER |",
                           s2_markdown: str | None = None) -> LinkedSchemas:
    """Build a minimal LinkedSchemas with non-empty markdown for both schemas."""
    if s2_markdown is None:
        s2_markdown = s1_markdown
    return LinkedSchemas(
        s1_ddl="CREATE TABLE t (id INTEGER PRIMARY KEY);",
        s1_markdown=s1_markdown,
        s2_ddl="CREATE TABLE t (id INTEGER PRIMARY KEY);",
        s2_markdown=s2_markdown,
        s1_fields=[("t", "id")],
        s2_fields=[("t", "id")],
        selection_reasoning="mock",
    )


# ---------------------------------------------------------------------------
# Test 1: Unit test — inspect call_args_list for max_tokens
# ---------------------------------------------------------------------------

async def _run_unit_test() -> tuple[bool, str]:
    """
    Patch get_client in standard_generator, run generate(), then check that
    every call whose model == settings.model_fast used max_tokens=4096.
    """
    mock_client = _make_mock_client()
    schemas = _make_minimal_schemas()
    grounding = GroundingContext(matched_cells=[], schema_hints=[], few_shot_examples=[])

    with patch("src.generation.standard_generator.get_client", return_value=mock_client):
        gen = StandardAndComplexGenerator()
        candidates = await gen.generate(
            question="What is the total count?",
            evidence="",
            schemas=schemas,
            grounding=grounding,
        )

    # Inspect all calls to client.generate
    calls = mock_client.generate.call_args_list
    if not calls:
        return False, "FAIL: mock client.generate was never called"

    failures: list[str] = []
    for call in calls:
        kwargs = call.kwargs
        model = kwargs.get("model", "")
        max_tok = kwargs.get("max_tokens", None)

        if model == settings.model_fast:
            if max_tok != 4096:
                failures.append(
                    f"  B1 call (model={model!r}) used max_tokens={max_tok}, expected 4096"
                )

    if failures:
        return False, "FAIL: B1 calls with wrong max_tokens:\n" + "\n".join(failures)

    # Also confirm B2 calls still use 4096 and were not accidentally changed
    b2_ok = True
    for call in calls:
        kwargs = call.kwargs
        model = kwargs.get("model", "")
        max_tok = kwargs.get("max_tokens", None)
        if model == settings.model_powerful and max_tok != 4096:
            b2_ok = False
            failures.append(
                f"  B2 call (model={model!r}) has unexpected max_tokens={max_tok}"
            )

    # Verify we got 4 candidates back (B1×2 + B2×2)
    if len(calls) != 4:
        return False, f"FAIL: expected 4 LLM calls, got {len(calls)}"

    # Count B1 calls (should be exactly 2)
    b1_calls = [c for c in calls if c.kwargs.get("model") == settings.model_fast]
    if len(b1_calls) != 2:
        return False, f"FAIL: expected 2 B1 calls (model_fast={settings.model_fast!r}), got {len(b1_calls)}"

    b1_max_tokens = {c.kwargs.get("max_tokens") for c in b1_calls}
    if b1_max_tokens != {4096}:
        return False, f"FAIL: B1 max_tokens values were {b1_max_tokens}, expected {{4096}}"

    return True, f"PASS: both B1 calls use max_tokens=4096  (model_fast={settings.model_fast!r})"


# ---------------------------------------------------------------------------
# Test 2: BIRD dev validation for affected QIDs
# ---------------------------------------------------------------------------

def _load_bird_question(qid: int) -> dict | None:
    if not BIRD_DEV_JSON.exists():
        return None
    with open(BIRD_DEV_JSON, encoding="utf-8") as f:
        data = json.load(f)
    for q in data:
        if q.get("question_id") == qid:
            return q
    return None


def _load_schemas_for_db(db_id: str) -> tuple[str, str] | None:
    """Return (ddl_text, markdown_text) or None if files are missing."""
    ddl_path = SCHEMAS_DIR / f"{db_id}_ddl.sql"
    md_path = SCHEMAS_DIR / f"{db_id}_markdown.md"
    if not ddl_path.exists() or not md_path.exists():
        return None
    return ddl_path.read_text(encoding="utf-8"), md_path.read_text(encoding="utf-8")


def _load_faiss_for_db(db_id: str):
    """Return a loaded FAISSIndex or None if files are missing."""
    index_path = INDICES_DIR / f"{db_id}_faiss.index"
    fields_path = INDICES_DIR / f"{db_id}_faiss_fields.json"
    if not index_path.exists() or not fields_path.exists():
        return None
    try:
        from src.indexing.faiss_index import FAISSIndex
        return FAISSIndex.load(str(index_path), str(fields_path))
    except Exception as exc:
        print(f"    [WARN] Could not load FAISS index for {db_id}: {exc}")
        return None


async def _run_bird_validation_for_qid(qid: int) -> tuple[str, str]:
    """
    Returns ("PASS"/"FAIL"/"SKIP", message).
    """
    q = _load_bird_question(qid)
    if q is None:
        return "SKIP", f"[SKIP] QID={qid}: BIRD dev.json not found at {BIRD_DEV_JSON}"

    db_id = q["db_id"]
    question = q["question"]
    evidence = q.get("evidence", "")

    schema_data = _load_schemas_for_db(db_id)
    if schema_data is None:
        return "SKIP", (
            f"[SKIP] QID={qid} ({db_id}): schema files missing in {SCHEMAS_DIR}"
        )

    full_ddl, full_markdown = schema_data
    faiss_index = _load_faiss_for_db(db_id)

    # Build available_fields list from the FAISS fields JSON
    fields_path = INDICES_DIR / f"{db_id}_faiss_fields.json"
    available_fields: list[tuple[str, str, str, str]] = []
    if fields_path.exists():
        with open(fields_path, encoding="utf-8") as f:
            raw_fields = json.load(f)
        available_fields = [
            (d["table"], d["column"], d.get("short_summary", ""), d.get("long_summary", ""))
            for d in raw_fields
        ]

    # Build minimal LinkedSchemas — use a small subset to avoid over-complex rendering
    # We'll use the first table's fields (up to 5 columns) for a representative schema
    if available_fields:
        # Use fields from the first few tables
        tables_seen: dict[str, list] = {}
        for t, c, ss, ls in available_fields:
            tables_seen.setdefault(t, []).append((t, c, ss, ls))
        # Pick up to 3 tables, up to 3 cols each, for a lightweight test schema
        s1_fields: list[tuple[str, str]] = []
        for tbl, tbl_fields in list(tables_seen.items())[:3]:
            for t, c, ss, ls in tbl_fields[:3]:
                s1_fields.append((t, c))

        # Build a minimal DDL/markdown that the generator will accept
        # (generator only reads the markdown schema from LinkedSchemas)
        s1_markdown = "\n\n".join(
            f"## Table: {t}\n| Column | Type | Description | Sample Values |\n|---|---|---|---|\n| {c} | TEXT | {ss or 'col'} | - |"
            for t, c in s1_fields
        ) or "## Table: t\n| Column | Type |\n|---|---|\n| id | INTEGER |"

        schemas = LinkedSchemas(
            s1_ddl="",
            s1_markdown=s1_markdown,
            s2_ddl="",
            s2_markdown=s1_markdown,  # Use same schema for s2 in this test
            s1_fields=s1_fields,
            s2_fields=s1_fields,
            selection_reasoning="test",
        )
    else:
        schemas = _make_minimal_schemas()

    grounding = GroundingContext(matched_cells=[], schema_hints=[], few_shot_examples=[])
    mock_client = _make_mock_client(finish_reason="STOP", text="SELECT 1")

    with patch("src.generation.standard_generator.get_client", return_value=mock_client):
        gen = StandardAndComplexGenerator()
        try:
            candidates = await gen.generate(
                question=question,
                evidence=evidence,
                schemas=schemas,
                grounding=grounding,
            )
        except Exception as exc:
            return "FAIL", f"[FAIL] QID={qid} ({db_id}): exception during generate(): {exc}"

    errored = [c for c in candidates if c.error_flag]
    if errored:
        ids = [c.generator_id for c in errored]
        return "FAIL", (
            f"[FAIL] QID={qid} ({db_id}, {q['difficulty']}): "
            f"{len(errored)}/4 candidates have error_flag=True: {ids}"
        )

    return "PASS", (
        f"[PASS] QID={qid} ({db_id}, {q['difficulty']}): "
        f"all {len(candidates)} candidates returned without error"
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

async def main() -> int:
    passed = 0
    failed = 0
    skipped = 0

    print("=" * 70)
    print("P0-1: B1 max_tokens fix verification")
    print("=" * 70)

    # --- Test 1: Unit test ---
    print("\n[Test 1] Unit test — B1 max_tokens=4096 via mock inspection")
    ok, msg = await _run_unit_test()
    print(f"  {msg}")
    if ok:
        passed += 1
    else:
        failed += 1

    # --- Test 2: BIRD dev validation ---
    print("\n[Test 2] BIRD dev validation (QIDs: 28, 1526, 852)")
    for qid in AFFECTED_QIDS:
        status, msg = await _run_bird_validation_for_qid(qid)
        print(f"  {msg}")
        if status == "PASS":
            passed += 1
        elif status == "FAIL":
            failed += 1
        else:
            skipped += 1

    # --- Summary ---
    total = passed + failed + skipped
    print("\n" + "=" * 70)
    print(f"Summary: {passed} passed / {failed} failed / {skipped} skipped  (total={total})")
    if failed > 0:
        print("RESULT: FAIL")
    else:
        print("RESULT: PASS")
    print("=" * 70)

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
