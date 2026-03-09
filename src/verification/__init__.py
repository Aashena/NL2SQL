"""
Semantic verification package for SQL candidates (Op 8 integrated step).

Op 8 provides two top-level operations:
  - QueryVerifier.generate_plan()       — one LLM call per question
  - QueryVerifier.evaluate_candidate()  — per-candidate structural + LLM evaluation

All public symbols are also accessible via ``src.verification.query_verifier``
for backward compatibility with existing imports in query_fixer and tests.

Package structure
-----------------
_constants.py           — penalty/bonus constants, test-type frozensets
_ordering_helpers.py    — _extract_limit_from_question, _derive_direction_from_question
_models.py              — VerificationTestSpec, VerificationTestResult, VerificationEvaluation
_llm_schemas.py         — _PLAN_TOOL, _COLUMN_ALIGNMENT_TOOL, _PLAN_SYSTEM, _COLUMN_ALIGNMENT_SYSTEM
_cheap_evaluators.py    — CheapEvaluatorMixin (_eval_grain/null/duplicate/ordering/scale)
_expensive_evaluators.py— ExpensiveEvaluatorMixin (_eval_column_alignment, _eval_symmetry)
query_verifier.py       — QueryVerifier class + backward-compat re-exports
"""
from src.verification.query_verifier import (
    QueryVerifier,
    VerificationEvaluation,
    VerificationTestResult,
    VerificationTestSpec,
    _derive_direction_from_question,
    _extract_limit_from_question,
)

__all__ = [
    "QueryVerifier",
    "VerificationEvaluation",
    "VerificationTestResult",
    "VerificationTestSpec",
    "_derive_direction_from_question",
    "_extract_limit_from_question",
]
