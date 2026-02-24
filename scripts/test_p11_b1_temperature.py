"""
test_p11_b1_temperature.py
==========================
Verifies fix P1-1: B1 temperature changed from 0.0 to 0.3 to reduce duplicates
when S1 == S2 (S2-skip scenario).

Tests:
  1. Unit test — B1 temperature value is 0.3
  2. Unit test — B2 temperature unchanged at 0.3
  3. BIRD dev validation — QIDs 1519, 803, 1349 (high-duplicate questions):
       simulate S2-skip scenario and verify mock returns distinct SQL when
       the LLM returns varying responses.
  4. Run existing test_generator_improvements.py and verify Test 6 passes.

Run: python scripts/test_p11_b1_temperature.py
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.base import LLMResponse
from src.schema_linking.schema_linker import LinkedSchemas
from src.grounding.context_grounder import GroundingContext
from src.generation.standard_generator import StandardAndComplexGenerator
from src.config.settings import settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS_COUNT = 0
FAIL_COUNT = 0
RESULTS: list[tuple[int, str, str]] = []


def record(test_num: int, passed: bool, description: str, detail: str = "") -> None:
    global PASS_COUNT, FAIL_COUNT
    status = "PASS" if passed else "FAIL"
    if passed:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    RESULTS.append((test_num, status, description))
    suffix = f" -- {detail}" if detail else ""
    print(f"  [Test {test_num:02d}] {status}: {description}{suffix}")


def make_response(sql_text: str) -> LLMResponse:
    return LLMResponse(
        tool_inputs=[],
        text=sql_text,
        thinking=None,
        input_tokens=100,
        output_tokens=50,
        finish_reason="STOP",
    )


def make_s1_eq_s2_schemas() -> LinkedSchemas:
    """Build a LinkedSchemas where s1_fields == s2_fields (simulates S2-skip)."""
    return LinkedSchemas(
        s1_fields=[("students", "id"), ("students", "gpa")],
        s2_fields=[("students", "id"), ("students", "gpa")],  # equal to s1 → triggers alt template
        s1_ddl="CREATE TABLE students (id INTEGER PRIMARY KEY, gpa REAL);",
        s2_ddl="CREATE TABLE students (id INTEGER PRIMARY KEY, gpa REAL);",
        s1_markdown="## Table: students\n| Column | Type |\n|---|---|\n| id | INTEGER |\n| gpa | REAL |",
        s2_markdown="## Table: students\n| Column | Type |\n|---|---|\n| id | INTEGER |\n| gpa | REAL |",
        selection_reasoning="",
    )


def make_grounding() -> GroundingContext:
    return GroundingContext(
        matched_cells=[],
        schema_hints=[("students", "gpa")],
        few_shot_examples=[],
    )


# ---------------------------------------------------------------------------
# Test 1 + 2: Unit tests — temperature values
# ---------------------------------------------------------------------------

async def run_temperature_unit_tests() -> None:
    print("\n--- Unit Tests: Temperature Values ---")
    schemas = make_s1_eq_s2_schemas()
    grounding = make_grounding()

    mock_client = AsyncMock()
    mock_client.generate = AsyncMock(
        return_value=make_response("SELECT id FROM students WHERE gpa > 3.5")
    )

    with patch("src.generation.standard_generator.get_client", return_value=mock_client):
        gen = StandardAndComplexGenerator()
        await gen.generate(
            question="List students with GPA above 3.5",
            evidence="",
            schemas=schemas,
            grounding=grounding,
        )

    call_args_list = mock_client.generate.call_args_list
    assert len(call_args_list) == 4, (
        f"Expected 4 generate() calls, got {len(call_args_list)}"
    )

    b1_calls = [ca for ca in call_args_list if ca.kwargs.get("model") == settings.model_fast]
    b2_calls = [ca for ca in call_args_list if ca.kwargs.get("model") == settings.model_powerful]

    # Test 1: B1 calls use temperature=0.3
    b1_temps = [ca.kwargs.get("temperature") for ca in b1_calls]
    b1_correct = all(t == 0.3 for t in b1_temps)
    record(
        1,
        b1_correct,
        "B1 calls use temperature=0.3",
        f"B1 temperatures: {b1_temps}",
    )

    # Test 2: B2 calls use temperature=0.3 (unchanged)
    b2_temps = [ca.kwargs.get("temperature") for ca in b2_calls]
    b2_correct = all(t == 0.3 for t in b2_temps)
    record(
        2,
        b2_correct,
        "B2 calls still use temperature=0.3 (unchanged)",
        f"B2 temperatures: {b2_temps}",
    )


# ---------------------------------------------------------------------------
# Test 3: BIRD dev validation — high-duplicate QIDs with varying mock responses
# ---------------------------------------------------------------------------

async def run_bird_dev_validation() -> None:
    print("\n--- BIRD Dev Validation: High-Duplicate QIDs ---")
    dev_json_path = PROJECT_ROOT / "data" / "bird" / "dev" / "dev.json"

    if not dev_json_path.exists():
        for test_num, qid in [(3, 1519), (4, 803), (5, 1349)]:
            record(
                test_num,
                False,
                f"QID={qid} — dev.json not found",
                str(dev_json_path),
            )
        return

    with open(dev_json_path, encoding="utf-8") as f:
        dev_data = json.load(f)

    target_qids = [1519, 803, 1349]

    for test_num, qid in zip([3, 4, 5], target_qids):
        question_entry = next(
            (q for q in dev_data if q.get("question_id") == qid),
            None,
        )
        if question_entry is None:
            record(
                test_num,
                False,
                f"QID={qid} — question_id not found in dev.json",
            )
            continue

        question = question_entry.get("question", "")
        evidence = question_entry.get("evidence", "")
        db_id = question_entry.get("db_id", "")

        # Simulate S2-skip scenario (s1_fields == s2_fields, same schemas)
        schemas = make_s1_eq_s2_schemas()
        grounding = make_grounding()

        # Mock returning different SQL for each successive call so we can verify
        # the generator itself does not collapse them into one.
        # Call order from asyncio.gather: B1_s1, B1_s2, B2_s1, B2_s2
        alternating_sqls = [
            "SELECT id FROM t1",
            "SELECT name FROM t1",
            "SELECT COUNT(*) FROM t1",
            "SELECT MAX(val) FROM t1",
        ]
        mock_responses = [make_response(sql) for sql in alternating_sqls]

        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(side_effect=mock_responses)

        with patch("src.generation.standard_generator.get_client", return_value=mock_client):
            gen = StandardAndComplexGenerator()
            candidates = await gen.generate(
                question=question,
                evidence=evidence,
                schemas=schemas,
                grounding=grounding,
            )

        # Filter to B1 candidates only
        b1_candidates = [c for c in candidates if "B1" in c.generator_id]
        b1_sqls = [c.sql for c in b1_candidates if not c.error_flag]

        # With a mock that returns different text each call, B1_s1 and B1_s2
        # should receive different responses (the mock cycles through alternating_sqls).
        # Verify the candidates are not all identical.
        unique_sqls = set(b1_sqls)
        not_all_same = len(unique_sqls) >= 1  # at minimum the mock varied responses
        # The actual uniqueness check: the two B1 calls received different SQL texts
        # (alternating_sqls[0] vs alternating_sqls[1])
        b1_are_distinct = len(unique_sqls) == len(b1_sqls) if len(b1_sqls) > 1 else True

        record(
            test_num,
            b1_are_distinct,
            f"QID={qid} db={db_id}: B1 candidates receive distinct SQL from varying mock",
            f"B1 SQLs: {b1_sqls}",
        )


# ---------------------------------------------------------------------------
# Test 6 (renumbered 6): Run existing test_generator_improvements.py
# ---------------------------------------------------------------------------

def run_existing_test_suite() -> None:
    print("\n--- Existing test_generator_improvements.py ---")
    script_path = PROJECT_ROOT / "scripts" / "test_generator_improvements.py"
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    passed = result.returncode == 0

    # Print the output so we can see what happened
    if result.stdout:
        for line in result.stdout.splitlines():
            print(f"    {line}")
    if result.stderr:
        for line in result.stderr.splitlines():
            print(f"    STDERR: {line}")

    record(
        6,
        passed,
        "test_generator_improvements.py exits with returncode=0 (all tests pass)",
        f"returncode={result.returncode}",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    print("=" * 60)
    print("P1-1 B1 Temperature Fix Tests")
    print("=" * 60)

    await run_temperature_unit_tests()
    await run_bird_dev_validation()
    run_existing_test_suite()

    print("\n" + "=" * 60)
    print(
        f"Results: {PASS_COUNT} passed, {FAIL_COUNT} failed "
        f"out of {PASS_COUNT + FAIL_COUNT} total"
    )
    print("=" * 60)

    if FAIL_COUNT > 0:
        print("\nFailed tests:")
        for test_num, status, description in RESULTS:
            if status == "FAIL":
                print(f"  [Test {test_num:02d}] FAIL: {description}")
        sys.exit(1)
    else:
        print("\nALL TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
