"""
Regex helpers and pure functions for extracting ordering intent from NL questions.

These utilities are used by the query fixer (via direct import) and by the
ordering evaluator to determine whether a SQL query's ORDER BY direction and
LIMIT clause match the question's intent (e.g. "top 5", "highest", "lowest").

Public functions
----------------
_extract_limit_from_question(question) -> Optional[int]
    Parse "top N", "N highest", etc. and return the integer N, or None.

_derive_direction_from_question(question) -> Optional[str]
    Return 'ASC', 'DESC', or None based on ordering keywords in the question.
"""
from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Ordering helpers (cheap, no LLM calls)
# ---------------------------------------------------------------------------

_TOP_N_RE = re.compile(
    r"(?:top|first|bottom|last)\s+(\d+)"
    r"|(\d+)\s+(?:highest|lowest|largest|smallest|best|worst|most|least)"
    r"|(?:highest|lowest|largest|smallest|best|worst|most|least)\s+(\d+)",
    re.IGNORECASE,
)
_WORD_TO_INT = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def _extract_limit_from_question(question: str) -> Optional[int]:
    """Extract LIMIT N from 'top N', 'N highest', etc. Returns None if not found."""
    for m in _TOP_N_RE.finditer(question):
        for g in m.groups():
            if g and g.isdigit():
                return int(g)
            if g and g.lower() in _WORD_TO_INT:
                return _WORD_TO_INT[g.lower()]
    return None


_DESC_KEYWORDS = frozenset({"highest", "largest", "best", "most", "top", "greatest", "maximum"})
_ASC_KEYWORDS = frozenset({"lowest", "smallest", "worst", "least", "bottom", "minimum"})


def _derive_direction_from_question(question: str) -> Optional[str]:
    """Scan question for ordering direction keywords. Returns 'ASC', 'DESC', or None."""
    q = question.lower()
    has_desc = any(kw in q for kw in _DESC_KEYWORDS)
    has_asc = any(kw in q for kw in _ASC_KEYWORDS)
    if has_asc and not has_desc:
        return "ASC"
    if has_desc and not has_asc:
        return "DESC"
    return None
