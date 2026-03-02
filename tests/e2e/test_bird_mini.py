"""
E2E Smoke Test — 66 Questions (6 per each of 11 BIRD dev databases)

Sampling: 2 simple + 2 moderate + 2 challenging per database, random.seed(42).
All tests are marked @pytest.mark.live; they make real API calls and require
ANTHROPIC_API_KEY (or GEMINI_API_KEY) set in the environment / .env file.

To run:
    pytest tests/e2e/test_bird_mini.py -v -m live --timeout=4000

The session-scoped fixture runs (or loads) the 66-question smoke test exactly
once per pytest session. Results are saved to results/smoke_test_66q/results.json
and component_summary.json so the run can be re-analysed without re-running.

If a results file already exists at the expected path the fixture skips the run
and just loads the cached results — handy for re-running analysis tests after
tweaking thresholds.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_RESULTS_DIR = _REPO_ROOT / "results" / "smoke_test_66q"
_RESULTS_FILE = _RESULTS_DIR / "results.json"
_SUMMARY_FILE = _RESULTS_DIR / "component_summary.json"
_SMOKE_SCRIPT = _REPO_ROOT / "scripts" / "run_smoke_test.py"


# ---------------------------------------------------------------------------
# pytest mark
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.live


# ---------------------------------------------------------------------------
# Session-scoped fixture: run or load the 66-question smoke test
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def smoke_results():
    """
    Run the 66-question smoke test (or load cached results if already done).

    Returns a dict with keys:
      - "results": list[dict]  — one entry per question (results.json)
      - "summary": dict        — component-level aggregated metrics
    """
    if not _RESULTS_FILE.exists():
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"\n[smoke_results] Running 66-question smoke test …")
        print(f"  Output dir: {_RESULTS_DIR}")
        proc = subprocess.run(
            [
                sys.executable,
                str(_SMOKE_SCRIPT),
                "--output_dir", str(_RESULTS_DIR),
                "--workers", "3",
            ],
            cwd=str(_REPO_ROOT),
            timeout=4000,
        )
        assert proc.returncode == 0, (
            f"Smoke test script exited with code {proc.returncode}. "
            f"Check {_RESULTS_DIR}/smoke_test.log for details."
        )

    assert _RESULTS_FILE.exists(), (
        f"Smoke test completed but results file not found at {_RESULTS_FILE}"
    )

    with open(_RESULTS_FILE, encoding="utf-8") as f:
        results = json.load(f)

    summary: dict = {}
    if _SUMMARY_FILE.exists():
        with open(_SUMMARY_FILE, encoding="utf-8") as f:
            summary = json.load(f)

    return {"results": results, "summary": summary}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _ex_pct(results: list[dict]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.get("correct", False)) / len(results)


# ---------------------------------------------------------------------------
# Test 1: Full pipeline runs on all 66 questions
# ---------------------------------------------------------------------------

def test_pipeline_runs_66_questions(smoke_results):
    """
    The full pipeline must:
      - Return 66 result entries (one per question)
      - Produce a non-empty SQL string for every question
      - Achieve >= 50% EX (basic sanity bar)
      - Have no question that took > 300 seconds
    """
    results = smoke_results["results"]
    assert len(results) == 66, (
        f"Expected 66 results, got {len(results)}. "
        "Check if all 11 databases have artifacts and all 66 questions were sampled."
    )

    # All questions must produce a non-empty SQL string
    empty_sql = [r["question_id"] for r in results if not r.get("predicted_sql", "").strip()]
    assert len(empty_sql) == 0, (
        f"{len(empty_sql)} questions returned empty SQL: {empty_sql[:10]}"
    )

    # EX >= 50%
    ex = _ex_pct(results)
    assert ex >= 0.50, (
        f"EX accuracy {ex:.1%} is below 50% threshold. "
        f"Correct: {sum(1 for r in results if r.get('correct'))}/{len(results)}. "
        f"Run: python scripts/analyze_results.py {_RESULTS_FILE}"
    )

    # No question > 300s
    slow = [r for r in results if r.get("latency_s", 0) > 300]
    assert len(slow) == 0, (
        f"{len(slow)} questions exceeded 300s: "
        + ", ".join(f"Q#{r['question_id']} ({r['latency_s']:.0f}s)" for r in slow[:5])
    )


# ---------------------------------------------------------------------------
# Test 2: Fast path rate is reasonable
# ---------------------------------------------------------------------------

def test_fast_path_rate_reasonable(smoke_results):
    """
    At least 30% of questions should trigger the fast path (unanimous agreement).
    A very low fast path rate suggests the generators are producing too much diversity.
    """
    results = smoke_results["results"]
    n_fast = sum(1 for r in results if r.get("selection_method") == "fast_path")
    rate = n_fast / len(results) if results else 0.0
    assert rate >= 0.30, (
        f"Fast path rate {rate:.1%} is below 30% threshold "
        f"({n_fast}/{len(results)} questions). "
        "Generators may be too diverse or producing errors."
    )


# ---------------------------------------------------------------------------
# Test 3: Query fixer rescue rate
# ---------------------------------------------------------------------------

def test_query_fixer_rescue_rate(smoke_results):
    """
    Among candidates that failed and were attempted to be fixed,
    at least 30% should be successfully fixed.
    """
    summary = smoke_results["summary"]
    if not summary:
        pytest.skip("component_summary.json not found — skipping fixer rate test")

    fix_stats = summary.get("op8", {})
    fix_success_rate = fix_stats.get("fix_success_rate", None)

    # If no candidates needed fixing at all, that's acceptable (pass the test)
    if fix_success_rate is None:
        pytest.skip("No fixer statistics in summary")

    # Allow 0 fix success rate if no candidates needed fixing
    candidates_needing_fix_rate = fix_stats.get("candidates_needing_fix_rate", 0.0)
    if candidates_needing_fix_rate == 0.0:
        return  # No fixes needed — pipeline is producing clean SQL

    assert fix_success_rate >= 0.30, (
        f"Query fixer rescue rate {fix_success_rate:.1%} is below 30% threshold. "
        f"Fix stats: {fix_stats}"
    )


# ---------------------------------------------------------------------------
# Test 4: Cost estimate is reasonable
# ---------------------------------------------------------------------------

def test_cost_estimate_reasonable(smoke_results):
    """
    Total API cost for 66 questions should be < $14.
    (Based on typical usage: ~$0.13/question for the full pipeline.)
    """
    summary = smoke_results["summary"]
    total_cost = summary.get("total_cost_estimate", None)
    if total_cost is None or total_cost == 0.0:
        pytest.skip(
            "Cost estimation not available (total_cost_estimate=0 or missing). "
            "This is expected when cost tracking is not implemented."
        )
    assert total_cost < 14.0, (
        f"Total cost ${total_cost:.2f} exceeds $14 budget for 66 questions. "
        f"Cost per question: ${total_cost/66:.3f}"
    )


# ---------------------------------------------------------------------------
# Test 5: Results saved to disk with correct structure
# ---------------------------------------------------------------------------

def test_results_saved_to_disk(smoke_results):
    """
    After the smoke test, a JSON results file must exist with exactly 66 entries,
    each containing the required fields for analysis.
    """
    assert _RESULTS_FILE.exists(), f"Results file not found: {_RESULTS_FILE}"
    assert _SUMMARY_FILE.exists(), f"Summary file not found: {_SUMMARY_FILE}"

    results = smoke_results["results"]
    assert len(results) == 66, f"Expected 66 entries in results.json, got {len(results)}"

    # Check required fields are present in each result
    required_fields = {
        "question_id", "db_id", "difficulty", "question",
        "predicted_sql", "truth_sql", "correct",
        "selection_method", "winner_generator", "cluster_count",
        "latency_s",
    }
    for r in results[:5]:  # spot-check first 5
        missing = required_fields - set(r.keys())
        assert not missing, (
            f"Result for Q#{r.get('question_id')} is missing fields: {missing}"
        )

    # Check that all 11 BIRD dev databases are represented
    db_ids = {r["db_id"] for r in results}
    assert len(db_ids) == 11, (
        f"Expected results from 11 databases, got {len(db_ids)}: {sorted(db_ids)}"
    )

    # Check 6 questions per database
    from collections import Counter
    db_counts = Counter(r["db_id"] for r in results)
    for db_id, count in db_counts.items():
        assert count == 6, (
            f"Database {db_id!r} has {count} questions, expected 6"
        )

    # Check difficulty distribution: 2 per difficulty per database = 22 each
    diff_counts = Counter(r["difficulty"] for r in results)
    for diff in ["simple", "moderate", "challenging"]:
        assert diff_counts[diff] == 22, (
            f"Difficulty {diff!r} has {diff_counts[diff]} questions, expected 22"
        )
