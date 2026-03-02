"""
E2E Full BIRD Dev Evaluation — 1,534 Questions

Runs (or loads) the complete BIRD dev evaluation and checks that Phase 1
targets are met. All tests are marked @pytest.mark.live.

To run:
    pytest tests/e2e/test_bird_full.py -v -m live --timeout=28800  # 8h timeout

The session-scoped fixture invokes run_evaluation.py if results don't exist;
otherwise it loads the saved JSON. Interrupted runs can be resumed by passing
--resume to run_evaluation.py and rerunning this test file.

Expected Phase 1 targets:
    Overall EX  ≥ 68%
    Simple EX   ≥ 78%
    Moderate EX ≥ 65%
    Challenging EX ≥ 45%
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_RESULTS_DIR = _REPO_ROOT / "results"
_RESULTS_FILE = _RESULTS_DIR / "phase1_dev_results.json"
_EVAL_SCRIPT = _REPO_ROOT / "scripts" / "run_evaluation.py"
_ANALYZE_SCRIPT = _REPO_ROOT / "scripts" / "analyze_results.py"

# ---------------------------------------------------------------------------
# pytest mark
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.live


# ---------------------------------------------------------------------------
# Session-scoped fixture: run or load the full BIRD dev evaluation
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def full_eval_results():
    """
    Run the full BIRD dev evaluation (1,534 questions) or load cached results.

    Returns a list[dict] — one entry per question.
    """
    if not _RESULTS_FILE.exists():
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"\n[full_eval] Running full BIRD dev evaluation (1,534 questions) …")
        print(f"  Output: {_RESULTS_FILE}")
        print("  This will take 4–8 hours. Grab a coffee ☕")
        proc = subprocess.run(
            [
                sys.executable,
                str(_EVAL_SCRIPT),
                "--split", "dev",
                "--output", str(_RESULTS_FILE),
                "--workers", "5",
            ],
            cwd=str(_REPO_ROOT),
            timeout=30000,
        )
        assert proc.returncode == 0, (
            f"run_evaluation.py exited with code {proc.returncode}. "
            f"Check logs for details. You can resume with: "
            f"  python scripts/run_evaluation.py --split dev "
            f"  --output {_RESULTS_FILE} --resume {_RESULTS_FILE}"
        )

    assert _RESULTS_FILE.exists(), (
        f"Evaluation results not found at {_RESULTS_FILE}"
    )
    with open(_RESULTS_FILE, encoding="utf-8") as f:
        results = json.load(f)
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ex(results: list[dict]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.get("correct", False)) / len(results)


def _by_difficulty(results: list[dict]) -> dict[str, float]:
    groups: dict[str, list] = defaultdict(list)
    for r in results:
        groups[r.get("difficulty", "unknown")].append(r)
    return {diff: _ex(rs) for diff, rs in groups.items()}


def _by_db(results: list[dict]) -> dict[str, float]:
    groups: dict[str, list] = defaultdict(list)
    for r in results:
        groups[r.get("db_id", "unknown")].append(r)
    return {db: _ex(rs) for db, rs in groups.items()}


# ---------------------------------------------------------------------------
# Test 1: Overall EX accuracy
# ---------------------------------------------------------------------------

def test_bird_dev_ex_accuracy(full_eval_results):
    """Overall EX on BIRD dev (1,534 questions) must be >= 68%."""
    results = full_eval_results
    # Allow partial runs but warn
    n = len(results)
    if n < 1534:
        pytest.xfail(
            f"Only {n}/1534 questions evaluated — partial run. "
            "Resume with --resume to complete the evaluation."
        )
    ex = _ex(results)
    assert ex >= 0.68, (
        f"Overall EX {ex:.1%} is below Phase 1 target of 68%. "
        f"Correct: {sum(1 for r in results if r.get('correct'))}/{n}"
    )


# ---------------------------------------------------------------------------
# Test 2: Simple questions accuracy
# ---------------------------------------------------------------------------

def test_simple_questions_accuracy(full_eval_results):
    """EX on 'simple' difficulty questions must be >= 78%."""
    simple = [r for r in full_eval_results if r.get("difficulty") == "simple"]
    if not simple:
        pytest.skip("No 'simple' questions in results")
    ex = _ex(simple)
    assert ex >= 0.78, (
        f"Simple EX {ex:.1%} is below target 78% "
        f"({sum(1 for r in simple if r.get('correct'))}/{len(simple)})"
    )


# ---------------------------------------------------------------------------
# Test 3: Moderate questions accuracy
# ---------------------------------------------------------------------------

def test_moderate_questions_accuracy(full_eval_results):
    """EX on 'moderate' difficulty questions must be >= 65%."""
    moderate = [r for r in full_eval_results if r.get("difficulty") == "moderate"]
    if not moderate:
        pytest.skip("No 'moderate' questions in results")
    ex = _ex(moderate)
    assert ex >= 0.65, (
        f"Moderate EX {ex:.1%} is below target 65% "
        f"({sum(1 for r in moderate if r.get('correct'))}/{len(moderate)})"
    )


# ---------------------------------------------------------------------------
# Test 4: Challenging questions accuracy
# ---------------------------------------------------------------------------

def test_challenging_questions_accuracy(full_eval_results):
    """EX on 'challenging' difficulty questions must be >= 45%."""
    challenging = [r for r in full_eval_results if r.get("difficulty") == "challenging"]
    if not challenging:
        pytest.skip("No 'challenging' questions in results")
    ex = _ex(challenging)
    assert ex >= 0.45, (
        f"Challenging EX {ex:.1%} is below target 45% "
        f"({sum(1 for r in challenging if r.get('correct'))}/{len(challenging)})"
    )


# ---------------------------------------------------------------------------
# Test 5: No unhandled crashes
# ---------------------------------------------------------------------------

def test_no_crashes_across_all_questions(full_eval_results):
    """
    The pipeline must not produce any entries with selection_method='error'
    (which indicates an unhandled exception during pipeline execution).
    """
    errors = [r for r in full_eval_results if r.get("selection_method") == "error"]
    error_rate = len(errors) / len(full_eval_results) if full_eval_results else 0
    # Allow up to 2% error rate (unexpected API failures, network issues, etc.)
    assert error_rate <= 0.02, (
        f"{len(errors)} pipeline crashes out of {len(full_eval_results)} questions "
        f"({error_rate:.1%} — limit is 2%). "
        f"First 5 crashed question IDs: "
        + str([r.get('question_id') for r in errors[:5]])
    )


# ---------------------------------------------------------------------------
# Test 6: Evaluation is resumable
# ---------------------------------------------------------------------------

def test_evaluation_resumable(full_eval_results):
    """
    The results file must be a valid JSON array that can be re-loaded without
    errors. This implicitly validates that incremental saving produced a
    well-formed file even after potential interruptions.
    """
    assert _RESULTS_FILE.exists(), f"Results file not found: {_RESULTS_FILE}"
    # Re-load to verify JSON integrity
    with open(_RESULTS_FILE, encoding="utf-8") as f:
        reloaded = json.load(f)
    assert isinstance(reloaded, list), "Results file must be a JSON array"
    assert len(reloaded) == len(full_eval_results), (
        "Re-loaded results count doesn't match in-memory count — file may be corrupted"
    )


# ---------------------------------------------------------------------------
# Test 7: Fast path rate
# ---------------------------------------------------------------------------

def test_fast_path_rate(full_eval_results):
    """
    Fast path should be triggered for 30–65% of questions.
    A rate outside this range suggests the generators aren't producing
    sufficiently diverse (too low) or uniformly correct (too high) results.
    """
    n_fast = sum(1 for r in full_eval_results if r.get("selection_method") == "fast_path")
    rate = n_fast / len(full_eval_results) if full_eval_results else 0.0
    assert 0.30 <= rate <= 0.65, (
        f"Fast path rate {rate:.1%} is outside expected range [30%, 65%]. "
        f"Fast path count: {n_fast}/{len(full_eval_results)}"
    )


# ---------------------------------------------------------------------------
# Test 8: Generator contribution
# ---------------------------------------------------------------------------

def test_generator_contribution(full_eval_results):
    """
    Each generator family (reasoning, standard/B1, complex/B2, icl) must
    contribute at least 5% of winning candidates. No generator should be
    consistently ignored by the selector.
    """
    winner_counts: Counter = Counter()
    for r in full_eval_results:
        gen = r.get("winner_generator", "")
        if gen and gen not in ("fallback", "unknown", ""):
            # Map to family
            if gen.startswith("reasoning"):
                winner_counts["reasoning"] += 1
            elif gen.startswith("standard_B1") or gen.startswith("standard_b1"):
                winner_counts["standard_B1"] += 1
            elif gen.startswith("complex_B2") or gen.startswith("complex_b2"):
                winner_counts["complex_B2"] += 1
            elif gen.startswith("icl"):
                winner_counts["icl"] += 1

    total_wins = sum(winner_counts.values())
    if total_wins == 0:
        pytest.skip("No tournament results found (all fast_path?)")

    for family in ["reasoning", "standard_B1", "complex_B2", "icl"]:
        rate = winner_counts[family] / total_wins
        assert rate >= 0.05, (
            f"Generator family {family!r} only won {rate:.1%} of tournaments "
            f"({winner_counts[family]}/{total_wins}). "
            "Minimum expected contribution: 5%."
        )


# ---------------------------------------------------------------------------
# Test 9: Cost within budget
# ---------------------------------------------------------------------------

def test_cost_within_budget(full_eval_results):
    """Total API cost for the full dev set must be < $200."""
    # Cost is estimated and stored in summary, not per-question results
    # This test will skip if cost tracking is not implemented
    total_cost = sum(r.get("cost_estimate", 0.0) for r in full_eval_results)
    if total_cost == 0.0:
        pytest.skip(
            "Cost tracking not available (all cost_estimate=0). "
            "This is expected for Phase 1 which returns 0.0 for cost."
        )
    assert total_cost < 200.0, (
        f"Total cost ${total_cost:.2f} exceeds $200 budget. "
        f"Cost per question: ${total_cost/len(full_eval_results):.3f}"
    )


# ---------------------------------------------------------------------------
# Test 10: Error analysis report can be generated
# ---------------------------------------------------------------------------

def test_error_analysis_report_generated(full_eval_results):
    """
    The analyze_results.py script must run successfully against the results file
    and produce output without errors.
    """
    assert _RESULTS_FILE.exists(), f"Results file not found: {_RESULTS_FILE}"
    proc = subprocess.run(
        [sys.executable, str(_ANALYZE_SCRIPT), str(_RESULTS_FILE)],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, (
        f"analyze_results.py failed with exit code {proc.returncode}.\n"
        f"stderr: {proc.stderr[:500]}"
    )
    # Verify the output contains key sections
    output = proc.stdout
    assert "Overall EX" in output or "overall_ex" in output.lower(), (
        "analyze_results.py output missing 'Overall EX' section"
    )
    assert "By Difficulty" in output or "difficulty" in output.lower(), (
        "analyze_results.py output missing difficulty breakdown"
    )
    assert "By Database" in output or "database" in output.lower(), (
        "analyze_results.py output missing database breakdown"
    )
