"""
Test script for ISSUE P0-2: MALFORMED_FUNCTION_CALL Grounding Fallback

Verifies that when the LLM raises LLMError (simulating MALFORMED_FUNCTION_CALL),
the context grounder falls back to keyword extraction and uses those keywords
to query the LSH index — so matched_cells is NOT empty.

Run with:
    python -m pytest scripts/test_p02_grounding_fallback.py -v
or:
    python scripts/test_p02_grounding_fallback.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure the project root is on sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.grounding.context_grounder import (
    GroundingContext,
    _extract_keywords_from_text,
    ground_context,
)
from src.indexing.lsh_index import CellMatch, LSHIndex
from src.llm.base import LLMError

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Test case definitions (question texts drawn from BIRD dev)
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        "name": "california_schools — long evidence with backtick-quoted columns",
        "db_id": "california_schools",
        "question": (
            "What is the highest eligible free rate for K-12 students in the "
            "schools in Alameda County?"
        ),
        "evidence": (
            "Eligible free rate for K-12 = `Free Meal Count (K-12)` / "
            "`Enrollment (K-12)`"
        ),
        # LSH returns these for keywords like "Alameda", "County", "K-12" etc.
        "lsh_results": [
            CellMatch(
                table="schools",
                column="County",
                matched_value="Alameda",
                similarity_score=0.85,
                exact_match=False,
            ),
            CellMatch(
                table="frpm",
                column="County Name",
                matched_value="Alameda",
                similarity_score=0.90,
                exact_match=True,
            ),
        ],
    },
    {
        "name": "european_football_2 — Q1027, players and penalties",
        "db_id": "european_football_2",
        "question": (
            "Indicate the full names of the top 10 players with the highest "
            "number of penalties."
        ),
        "evidence": (
            "full name refers to player_name; players with highest number of "
            "penalties refers to MAX(penalties);"
        ),
        "lsh_results": [
            CellMatch(
                table="Player",
                column="player_name",
                matched_value="penalties",
                similarity_score=0.72,
                exact_match=False,
            ),
        ],
    },
    {
        "name": "card_games — Q463, Angel of Mercy with special chars",
        "db_id": "card_games",
        "question": (
            'How many translations are there for the set of cards with '
            '"Angel of Mercy" in it?'
        ),
        "evidence": (
            "set of cards with \"Angel of Mercy\" in it refers to "
            "name = 'Angel of Mercy'"
        ),
        "lsh_results": [
            CellMatch(
                table="cards",
                column="name",
                matched_value="Angel of Mercy",
                similarity_score=0.95,
                exact_match=True,
            ),
        ],
    },
    {
        "name": "toxicology — Q237, label '+' means carcinogenic",
        "db_id": "toxicology",
        "question": (
            "Which molecule does the atom TR001_10 belong to? Please state "
            "whether this molecule is carcinogenic or not."
        ),
        "evidence": "TR001_10 is the atom id; label = '+' mean molecules are carcinogenic",
        "lsh_results": [
            CellMatch(
                table="atom",
                column="atom_id",
                matched_value="TR001_10",
                similarity_score=0.98,
                exact_match=True,
            ),
        ],
    },
    {
        "name": "codebase_community — Q571, posts vs votes ratio",
        "db_id": "codebase_community",
        "question": (
            "For the user No.24, how many times is the number of his/her "
            "posts compared to his/her votes?"
        ),
        "evidence": (
            "user no. 24 refers to UserId = OwnerUserId = '24'; "
            "times of his/her post than votes = Divide (Count(post.Id), Count(votes.Id))"
        ),
        "lsh_results": [
            CellMatch(
                table="users",
                column="Id",
                matched_value="24",
                similarity_score=0.80,
                exact_match=True,
            ),
        ],
    },
]


# ---------------------------------------------------------------------------
# Helper: build a mock LSH that returns results for any query term
# ---------------------------------------------------------------------------

def _make_mock_lsh(cells: list[CellMatch]) -> MagicMock:
    """
    Build a mock LSHIndex whose .query() returns `cells` for every call.
    This simulates the LSH finding matches when given keywords from the fallback.
    """
    mock = MagicMock(spec=LSHIndex)
    mock.query = MagicMock(return_value=cells)
    return mock


def _make_mock_example_store() -> MagicMock:
    mock = MagicMock()
    mock.query = MagicMock(return_value=[])
    return mock


# ---------------------------------------------------------------------------
# Unit tests for the helper function
# ---------------------------------------------------------------------------

def test_extract_keywords_basic():
    """Basic keyword extraction removes stop words and short tokens."""
    text = "What is the highest eligible free rate for K-12 students in Alameda County?"
    keywords = _extract_keywords_from_text(text)
    assert "highest" in keywords, "should include 'highest'"
    assert "eligible" in keywords, "should include 'eligible'"
    assert "Alameda" in keywords, "should include 'Alameda' (name)"
    assert "County" in keywords, "should include 'County'"
    assert "the" not in keywords, "stop word 'the' should be filtered"
    assert "is" not in keywords, "stop word 'is' should be filtered"
    assert "in" not in keywords, "stop word 'in' should be filtered"
    print("  PASS: test_extract_keywords_basic")


def test_extract_keywords_punctuation_stripped():
    """Punctuation is stripped from token boundaries."""
    text = "Angel of Mercy. 'label='+'' Mercy'"
    keywords = _extract_keywords_from_text(text)
    # "Mercy'" should become "Mercy" after stripping
    assert "Mercy" in keywords, f"Expected 'Mercy' in {keywords}"
    print("  PASS: test_extract_keywords_punctuation_stripped")


def test_extract_keywords_deduplication():
    """Duplicate tokens appear only once (case-insensitive dedup)."""
    text = "Alameda County schools in Alameda"
    keywords = _extract_keywords_from_text(text)
    count_alameda = sum(1 for k in keywords if k.lower() == "alameda")
    assert count_alameda == 1, f"Expected exactly 1 'Alameda', got {count_alameda}"
    print("  PASS: test_extract_keywords_deduplication")


def test_extract_keywords_short_token_filtered():
    """Tokens shorter than 3 characters are filtered out."""
    text = "K-12 is a grade level"
    keywords = _extract_keywords_from_text(text)
    # "a" (stop word), "is" (stop word), "K-12" should pass if length >= 3
    for kw in keywords:
        assert len(kw) >= 3, f"Token {kw!r} is shorter than 3 chars"
    print("  PASS: test_extract_keywords_short_token_filtered")


def test_extract_keywords_original_case_preserved():
    """Original case is preserved (LSH exact-match detection is case-sensitive)."""
    text = "Angel of Mercy carcinogenic TR001_10"
    keywords = _extract_keywords_from_text(text)
    # "Angel" should keep capital A, not become "angel"
    assert "Angel" in keywords, f"Expected 'Angel' (not 'angel') in {keywords}"
    assert "Mercy" in keywords, f"Expected 'Mercy' in {keywords}"
    assert "carcinogenic" in keywords, f"Expected 'carcinogenic' in {keywords}"
    print("  PASS: test_extract_keywords_original_case_preserved")


# ---------------------------------------------------------------------------
# Async integration tests: LLM failure → keyword fallback → LSH matches
# ---------------------------------------------------------------------------

async def run_fallback_test(case: dict) -> dict:
    """
    Run a single test case:
    1. Patch get_client to raise LLMError (simulating MALFORMED_FUNCTION_CALL).
    2. Call ground_context().
    3. Verify matched_cells is non-empty (LSH was called with fallback keywords).
    """
    name = case["name"]
    lsh = _make_mock_lsh(case["lsh_results"])
    store = _make_mock_example_store()

    # Build a mock client that raises LLMError on .generate()
    mock_client = MagicMock()
    mock_client.generate = AsyncMock(
        side_effect=LLMError("MALFORMED_FUNCTION_CALL")
    )

    with patch("src.grounding.context_grounder.get_client", return_value=mock_client):
        ctx = await ground_context(
            question=case["question"],
            evidence=case["evidence"],
            db_id=case["db_id"],
            lsh_index=lsh,
            example_store=store,
        )

    # Core assertion: LSH must have been called (fallback keywords were used)
    lsh_was_called = lsh.query.called
    has_matched_cells = len(ctx.matched_cells) > 0

    return {
        "name": name,
        "lsh_was_called": lsh_was_called,
        "matched_cells": len(ctx.matched_cells),
        "schema_hints": len(ctx.schema_hints),
        "passed": lsh_was_called and has_matched_cells,
    }


async def run_no_regression_test():
    """
    Verify that when the LLM SUCCEEDS, behavior is unchanged:
    - tool_inputs are used as literals (not the keyword fallback)
    - LSH is called with the LLM-provided literals
    """
    from src.llm.base import LLMResponse

    lsh = _make_mock_lsh([
        CellMatch(
            table="schools",
            column="County",
            matched_value="Alameda",
            similarity_score=0.9,
            exact_match=True,
        )
    ])
    store = _make_mock_example_store()

    mock_client = MagicMock()
    mock_client.generate = AsyncMock(
        return_value=LLMResponse(
            tool_inputs=[{"literals": ["Alameda County"], "schema_references": ["County"]}],
            text=None,
            thinking=None,
            input_tokens=50,
            output_tokens=20,
        )
    )

    with patch("src.grounding.context_grounder.get_client", return_value=mock_client):
        ctx = await ground_context(
            question="How many schools are in Alameda County?",
            evidence="",
            db_id="california_schools",
            lsh_index=lsh,
            example_store=store,
        )

    assert ctx.schema_hints == ["County"], (
        f"schema_hints should come from LLM, got {ctx.schema_hints}"
    )
    assert len(ctx.matched_cells) >= 1, "matched_cells should be non-empty"
    return {
        "name": "no-regression — LLM success path unchanged",
        "passed": True,
        "matched_cells": len(ctx.matched_cells),
        "schema_hints": len(ctx.schema_hints),
    }


async def main():
    print("=" * 70)
    print("ISSUE P0-2 — Grounding Fallback Fix Test")
    print("=" * 70)

    # --- Helper function unit tests ---
    print("\n--- Unit tests for _extract_keywords_from_text ---")
    test_extract_keywords_basic()
    test_extract_keywords_punctuation_stripped()
    test_extract_keywords_deduplication()
    test_extract_keywords_short_token_filtered()
    test_extract_keywords_original_case_preserved()

    # --- Integration tests: LLM failure → keyword fallback ---
    print("\n--- Integration tests: LLMError → keyword fallback → LSH results ---")
    results = []
    for case in TEST_CASES:
        result = await run_fallback_test(case)
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"  [{status}] {result['name']}: "
            f"lsh_called={result['lsh_was_called']}, "
            f"matched_cells={result['matched_cells']}"
        )

    # --- No-regression test ---
    print("\n--- No-regression test: LLM success path unchanged ---")
    reg_result = await run_no_regression_test()
    status = "PASS" if reg_result["passed"] else "FAIL"
    print(
        f"  [{status}] {reg_result['name']}: "
        f"matched_cells={reg_result['matched_cells']}, "
        f"schema_hints={reg_result['schema_hints']}"
    )
    results.append(reg_result)

    # --- Summary ---
    print("\n" + "=" * 70)
    n_pass = sum(1 for r in results if r["passed"])
    n_fail = sum(1 for r in results if not r["passed"])
    print(f"Results: {n_pass} passed, {n_fail} failed out of {len(results)} total")

    if n_fail > 0:
        print("\nFAILED tests:")
        for r in results:
            if not r["passed"]:
                print(f"  - {r['name']}")
        sys.exit(1)
    else:
        print("All tests PASSED.")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
