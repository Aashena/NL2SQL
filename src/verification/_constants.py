"""
Confidence adjustment constants and test-type sets for the semantic verification system.

These constants are used by QueryVerifier._compute_adjustment() to penalise failing
tests and reward candidates that pass all checks.  The test-type frozensets define
which checks are cheap (no LLM) versus expensive (LLM-judged), and are referenced
both by the evaluation loop and by the LLM plan-generation tool schema.
"""

# ---------------------------------------------------------------------------
# Confidence adjustment constants
# ---------------------------------------------------------------------------

_PENALTY_CRITICAL = -0.3   # per critical test failure
_PENALTY_MINOR = -0.1      # per minor test failure
_BONUS_ALL_PASS = 0.2      # bonus when ALL applicable tests pass (capped)
_MAX_BONUS = 0.2

_CHEAP_TESTS = frozenset({
    "grain", "null", "duplicate", "ordering", "scale", "column_alignment"
})
_EXPENSIVE_TESTS = frozenset({
    "boundary", "symmetry"
})
_ALL_TEST_TYPES = _CHEAP_TESTS | _EXPENSIVE_TESTS
