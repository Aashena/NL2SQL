"""
test_generator_improvements.py
================================
Verifies all fixes applied in the generator improvement pass:

  P0-1 — MAX_TOKENS truncation detection
  P1-5 — Temperature diversity (B1=0.3, B2=0.3, ICL=0.7)
  P2-6/P2-7/P2-8/P2-10 — SQL writing rules in build_base_prompt()

Run: python scripts/test_generator_improvements.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.base import LLMResponse
from src.indexing.lsh_index import CellMatch
from src.indexing.example_store import ExampleEntry
from src.grounding.context_grounder import GroundingContext
from src.schema_linking.schema_linker import LinkedSchemas
from src.generation.base_generator import SQLCandidate, build_base_prompt
from src.generation.standard_generator import StandardAndComplexGenerator
from src.generation.icl_generator import ICLGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS_COUNT = 0
FAIL_COUNT = 0
RESULTS: list[tuple[int, str, str]] = []  # (test_num, PASS/FAIL, description)


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


def make_schemas(s1_eq_s2: bool = False) -> LinkedSchemas:
    s1_md = (
        "## Table: students\n"
        "| Column | Type | Description |\n"
        "|--------|------|-------------|\n"
        "| id | INTEGER (PK) | student id |\n"
        "| gpa | REAL | grade point average |"
    )
    s2_md = (
        "## Table: students\n"
        "| Column | Type | Description |\n"
        "|--------|------|-------------|\n"
        "| id | INTEGER (PK) | student id |\n"
        "| gpa | REAL | grade point average |\n\n"
        "## Table: schools\n"
        "| Column | Type | Description |\n"
        "|--------|------|-------------|\n"
        "| id | INTEGER (PK) | school id |\n"
        "| name | TEXT | school name |"
    )
    if s1_eq_s2:
        s2_md = s1_md
        s2_fields = [("students", "id"), ("students", "gpa")]
    else:
        s2_fields = [
            ("students", "id"), ("students", "gpa"),
            ("schools", "id"), ("schools", "name"),
        ]
    return LinkedSchemas(
        s1_fields=[("students", "id"), ("students", "gpa")],
        s2_fields=s2_fields,
        s1_ddl="CREATE TABLE students (id INTEGER PRIMARY KEY, gpa REAL);",
        s2_ddl="CREATE TABLE students (id INTEGER PRIMARY KEY, gpa REAL);",
        s1_markdown=s1_md,
        s2_markdown=s2_md,
        selection_reasoning="",
    )


def make_grounding() -> GroundingContext:
    return GroundingContext(
        matched_cells=[
            CellMatch(
                table="students",
                column="gpa",
                matched_value="3.5",
                similarity_score=0.95,
                exact_match=True,
            )
        ],
        schema_hints=[("students", "gpa")],
        few_shot_examples=[
            ExampleEntry(
                question_id=1,
                db_id="other_db_1",
                question="How many students have GPA above 3.0?",
                evidence="",
                sql="SELECT COUNT(*) FROM students WHERE gpa > 3.0",
                skeleton="How many students have GPA above [NUM]?",
                similarity_score=0.88,
            )
        ],
    )


def make_mock_client_with_finish_reason(
    text: str | None,
    finish_reason: str,
) -> AsyncMock:
    """Return a mock LLM client whose generate() always returns the given response."""
    response = LLMResponse(
        tool_inputs=[],
        text=text,
        thinking=None,
        input_tokens=100,
        output_tokens=50,
        finish_reason=finish_reason,
    )
    mock_client = AsyncMock()
    mock_client.generate = AsyncMock(return_value=response)
    return mock_client


def make_normal_mock_client(text: str = "SELECT id FROM students WHERE gpa > 3.5") -> AsyncMock:
    """Return a mock that returns a normal STOP response."""
    return make_mock_client_with_finish_reason(text=text, finish_reason="STOP")


# ---------------------------------------------------------------------------
# P0-1: MAX_TOKENS truncation detection
# ---------------------------------------------------------------------------

async def run_p01_tests() -> None:
    print("\n--- P0-1: MAX_TOKENS truncation detection ---")
    schemas = make_schemas()
    grounding = make_grounding()

    # Test 1: B2 with finish_reason=MAX_TOKENS + non-empty text → error_flag=True
    mock_client = make_mock_client_with_finish_reason(
        text="SELECT id FROM students WHERE gpa > 3.5 WITH INCOMPLETE",
        finish_reason="MAX_TOKENS",
    )
    with patch("src.generation.standard_generator.get_client", return_value=mock_client):
        gen = StandardAndComplexGenerator()
        candidates = await gen.generate(
            question="List students with GPA above 3.5",
            evidence="",
            schemas=schemas,
            grounding=grounding,
        )
    b2_cands = [c for c in candidates if "B2" in c.generator_id]
    b2_all_error = all(c.error_flag for c in b2_cands)
    record(1, b2_all_error, "B2 + MAX_TOKENS + non-empty text → error_flag=True",
           f"B2 candidates error_flags: {[c.error_flag for c in b2_cands]}")

    # Test 2: B1 with finish_reason=MAX_TOKENS + non-empty text → error_flag=True
    mock_client = make_mock_client_with_finish_reason(
        text="SELECT id FROM students WHERE gpa > 3.5 TRUNCATED",
        finish_reason="MAX_TOKENS",
    )
    with patch("src.generation.standard_generator.get_client", return_value=mock_client):
        gen = StandardAndComplexGenerator()
        candidates = await gen.generate(
            question="List students with GPA above 3.5",
            evidence="",
            schemas=schemas,
            grounding=grounding,
        )
    b1_cands = [c for c in candidates if "B1" in c.generator_id]
    b1_all_error = all(c.error_flag for c in b1_cands)
    record(2, b1_all_error, "B1 + MAX_TOKENS + non-empty text → error_flag=True",
           f"B1 candidates error_flags: {[c.error_flag for c in b1_cands]}")

    # Test 3: ICL with finish_reason=MAX_TOKENS + non-empty text → error_flag=True
    mock_client = make_mock_client_with_finish_reason(
        text="SELECT id FROM students WHERE gpa > 3.5 TRUNCATED",
        finish_reason="MAX_TOKENS",
    )
    with patch("src.generation.icl_generator.get_client", return_value=mock_client):
        gen = ICLGenerator()
        candidates = await gen.generate(
            question="List students with GPA above 3.5",
            evidence="",
            schemas=schemas,
            grounding=grounding,
        )
    icl_all_error = all(c.error_flag for c in candidates)
    record(3, icl_all_error, "ICL + MAX_TOKENS + non-empty text → error_flag=True",
           f"ICL candidates error_flags: {[c.error_flag for c in candidates]}")

    # Test 4: B2 with finish_reason=STOP + valid text → error_flag=False (normal case)
    mock_client = make_mock_client_with_finish_reason(
        text="SELECT id FROM students WHERE gpa > 3.5",
        finish_reason="STOP",
    )
    with patch("src.generation.standard_generator.get_client", return_value=mock_client):
        gen = StandardAndComplexGenerator()
        candidates = await gen.generate(
            question="List students with GPA above 3.5",
            evidence="",
            schemas=schemas,
            grounding=grounding,
        )
    all_no_error = all(not c.error_flag for c in candidates)
    record(4, all_no_error, "B2 + finish_reason=STOP + valid text → error_flag=False",
           f"All candidates error_flags: {[c.error_flag for c in candidates]}")

    # Test 5: B2 with finish_reason=MAX_TOKENS + text=None → error_flag=True (graceful handling)
    mock_client = make_mock_client_with_finish_reason(
        text=None,
        finish_reason="MAX_TOKENS",
    )
    with patch("src.generation.standard_generator.get_client", return_value=mock_client):
        gen = StandardAndComplexGenerator()
        candidates = await gen.generate(
            question="List students with GPA above 3.5",
            evidence="",
            schemas=schemas,
            grounding=grounding,
        )
    all_error_none = all(c.error_flag for c in candidates)
    record(5, all_error_none, "B2 + finish_reason=MAX_TOKENS + text=None → error_flag=True",
           f"All candidates error_flags: {[c.error_flag for c in candidates]}")


# ---------------------------------------------------------------------------
# P1-5: Temperature diversity
# ---------------------------------------------------------------------------

async def run_p15_tests() -> None:
    print("\n--- P1-5: Temperature diversity ---")
    schemas = make_schemas()
    grounding = make_grounding()
    mock_client = make_normal_mock_client()

    with patch("src.generation.standard_generator.get_client", return_value=mock_client):
        gen = StandardAndComplexGenerator()
        await gen.generate(
            question="List students with GPA above 3.5",
            evidence="",
            schemas=schemas,
            grounding=grounding,
        )

    call_args_list = mock_client.generate.call_args_list
    assert len(call_args_list) == 4, f"Expected 4 calls, got {len(call_args_list)}"

    # Identify calls by model
    from src.config.settings import settings
    b1_calls = [ca for ca in call_args_list if ca.kwargs.get("model") == settings.model_fast]
    b2_calls = [ca for ca in call_args_list if ca.kwargs.get("model") == settings.model_powerful]

    # Test 6: B1 calls use temperature=0.3
    b1_temps = [ca.kwargs.get("temperature") for ca in b1_calls]
    b1_correct = all(t == 0.3 for t in b1_temps)
    record(6, b1_correct, "B1 calls use temperature=0.3",
           f"B1 temperatures: {b1_temps}")

    # Test 7: B2 calls use temperature=0.3
    b2_temps = [ca.kwargs.get("temperature") for ca in b2_calls]
    b2_correct = all(t == 0.3 for t in b2_temps)
    record(7, b2_correct, "B2 calls use temperature=0.3",
           f"B2 temperatures: {b2_temps}")

    # Test 8: ICL calls use temperature=0.7
    mock_client_icl = make_normal_mock_client()
    with patch("src.generation.icl_generator.get_client", return_value=mock_client_icl):
        icl_gen = ICLGenerator()
        await icl_gen.generate(
            question="List students with GPA above 3.5",
            evidence="",
            schemas=schemas,
            grounding=grounding,
        )

    icl_call_args = mock_client_icl.generate.call_args_list
    icl_temps = [ca.kwargs.get("temperature") for ca in icl_call_args]
    icl_correct = all(t == 0.7 for t in icl_temps)
    record(8, icl_correct, "ICL calls use temperature=0.7",
           f"ICL temperatures: {icl_temps}")


# ---------------------------------------------------------------------------
# P2-6/P2-7/P2-8/P2-10: SQL writing rules in build_base_prompt()
# ---------------------------------------------------------------------------

def run_p2_prompt_tests() -> None:
    print("\n--- P2-prompts: SQL Writing Rules in build_base_prompt() ---")

    # Use a simple call to verify the rules are present
    prompt = build_base_prompt(
        question="What is the name and id of the student with the highest GPA?",
        evidence="GPA means grade point average",
        cell_matches=[],
    )

    # Test 9: "SELECT completeness" rule present
    has_select_completeness = (
        "SELECT completeness" in prompt or
        "Return ALL columns" in prompt
    )
    record(9, has_select_completeness, "build_base_prompt() contains 'SELECT completeness' rule",
           f"Prompt snippet: {prompt[prompt.find('SQL Writing Rules'):prompt.find('SQL Writing Rules')+200]!r}")

    # Test 10: "Value mappings" rule present
    has_value_mappings = (
        "Value mappings" in prompt or
        "CASE or IIF" in prompt
    )
    record(10, has_value_mappings, "build_base_prompt() contains 'Value mappings' rule")

    # Test 11: "Ratio direction" rule present
    has_ratio = (
        "Ratio direction" in prompt or
        "how many times is A compared to B" in prompt
    )
    record(11, has_ratio, "build_base_prompt() contains 'Ratio direction' rule")

    # Test 12: "Evidence scope" rule present
    has_evidence_scope = (
        "Evidence scope" in prompt or
        "Trust the question wording" in prompt
    )
    record(12, has_evidence_scope, "build_base_prompt() contains 'Evidence scope' rule")

    # Test 13: All 4 rules present when using actual Q237 context
    #  Q237: toxicology moderate — label = '+' means carcinogenic
    dev_json_path = PROJECT_ROOT / "data" / "bird" / "dev" / "dev.json"
    if not dev_json_path.exists():
        record(13, False, "Q237 prompt test — dev.json not found (skipped as FAIL)",
               "data/bird/dev/dev.json does not exist")
        return

    with open(dev_json_path, encoding="utf-8") as f:
        dev_data = json.load(f)

    q237 = next(
        (q for q in dev_data if q.get("question_id") == 237),
        None,
    )
    if q237 is None:
        record(13, False, "Q237 prompt test — question_id=237 not found in dev.json")
        return

    q237_prompt = build_base_prompt(
        question=q237.get("question", ""),
        evidence=q237.get("evidence", ""),
        cell_matches=[],
    )

    all_rules_present = (
        ("SELECT completeness" in q237_prompt or "Return ALL columns" in q237_prompt) and
        ("Value mappings" in q237_prompt or "CASE or IIF" in q237_prompt) and
        ("Ratio direction" in q237_prompt or "how many times is A" in q237_prompt) and
        ("Evidence scope" in q237_prompt or "Trust the question wording" in q237_prompt)
    )
    record(13, all_rules_present,
           "All 4 SQL rules present in Q237 (toxicology) prompt",
           f"Q237 question: {q237.get('question', '')[:80]!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    print("=" * 60)
    print("Generator Improvement Tests")
    print("=" * 60)

    await run_p01_tests()
    await run_p15_tests()
    run_p2_prompt_tests()

    print("\n" + "=" * 60)
    print(f"Results: {PASS_COUNT} passed, {FAIL_COUNT} failed out of {PASS_COUNT + FAIL_COUNT} total")
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
