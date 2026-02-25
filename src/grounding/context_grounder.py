"""
Op 5: Context Grounding

Given a natural-language question and optional evidence, extracts:
  1. Matched cell values from the LSH index (Step 5.1 + 5.2)
  2. Structurally similar few-shot examples from the example store (Step 5.3)

The result is a GroundingContext consumed by downstream schema linking and
SQL generation steps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from src.llm import get_client, CacheableText, ToolParam, LLMError, sanitize_prompt_text
from src.config.settings import settings
from src.monitoring.fallback_tracker import FallbackEvent, get_tracker

if TYPE_CHECKING:
    from src.indexing.lsh_index import CellMatch, LSHIndex
    from src.indexing.example_store import ExampleEntry, ExampleStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stop words for keyword fallback (used when LLM extraction fails)
# ---------------------------------------------------------------------------

_STOP_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "for", "of", "in", "on",
    "at", "to", "by", "with", "from", "and", "or", "not", "no", "nor",
    "but", "if", "then", "than", "as", "its", "it", "this", "that",
    "these", "those", "what", "which", "who", "how", "when", "where",
    "why", "whose", "whom",
})

# ---------------------------------------------------------------------------
# Keyword extraction helper (used as LLM fallback)
# ---------------------------------------------------------------------------

def _extract_keywords_from_text(text: str) -> list[str]:
    """
    Extract meaningful keywords from *text* using simple NLP.

    This is used as a fallback when the LLM structured extraction fails
    (e.g. MALFORMED_FUNCTION_CALL from Gemini on long or special-char inputs).
    It preserves original-case tokens so that LSH matching can perform
    case-insensitive 3-gram comparisons against indexed cell values.

    Steps:
    1. Split on whitespace.
    2. Strip leading/trailing punctuation from each token.
    3. Filter tokens that are stop words (checked lowercased).
    4. Filter tokens shorter than 3 characters.
    5. Deduplicate while preserving the first-seen order.
    """
    # Split on whitespace first; each chunk may still have punctuation attached.
    raw_tokens = text.split()

    seen: set[str] = set()
    keywords: list[str] = []
    for tok in raw_tokens:
        # Strip punctuation from both ends of the token.
        stripped = tok.strip(".,;:!?'\"()[]{}\\`")
        if not stripped:
            continue
        # Use lowercase only for stop-word and length checks; keep original case
        # for the LSH query so that exact-match detection works correctly.
        lower = stripped.lower()
        if lower in _STOP_WORDS:
            continue
        if len(stripped) < 3:
            continue
        if lower not in seen:
            seen.add(lower)
            keywords.append(stripped)

    return keywords


# ---------------------------------------------------------------------------
# Tool definition for keyword/literal extraction (Step 5.1)
# ---------------------------------------------------------------------------

_EXTRACT_GROUNDING_TOOL = ToolParam(
    name="extract_grounding",
    description=(
        "Extract database literals and schema references from a question and evidence"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "literals": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Specific values that might exist in database cells"
                ),
            },
            "schema_references": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Column or table names mentioned in the evidence"
                ),
            },
        },
        "required": ["literals", "schema_references"],
    },
)

_SYSTEM_PROMPT = (
    "You are a database expert. Extract specific values and column/table references "
    "from the question and evidence that should be looked up in a database."
)


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class GroundingContext:
    matched_cells: list["CellMatch"] = field(default_factory=list)
    schema_hints: list[str] = field(default_factory=list)
    few_shot_examples: list["ExampleEntry"] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main async function
# ---------------------------------------------------------------------------

async def ground_context(
    question: str,
    evidence: str,
    db_id: str,
    lsh_index: "LSHIndex",
    example_store: "ExampleStore",
) -> GroundingContext:
    """
    Perform Op 5 context grounding.

    Parameters
    ----------
    question:
        Natural-language question from the BIRD dataset.
    evidence:
        Auxiliary evidence string (may be empty or None).
    db_id:
        Database identifier; used to exclude same-db examples in retrieval.
    lsh_index:
        Pre-built LSHIndex for the target database.
    example_store:
        Pre-built ExampleStore from training data.

    Returns
    -------
    GroundingContext with matched_cells, schema_hints, and few_shot_examples.
    """
    # Normalise evidence
    evidence_str = evidence if evidence else "None"

    # ------------------------------------------------------------------
    # Step 5.1 — Keyword / literal extraction via LLM tool call
    # ------------------------------------------------------------------
    client = get_client()
    # Sanitize to avoid Gemini MALFORMED_FUNCTION_CALL on special characters
    # (e.g. backtick-quoted identifiers, control chars in evidence strings)
    safe_question = sanitize_prompt_text(question)
    safe_evidence = sanitize_prompt_text(evidence_str)
    user_message = f"Question: {safe_question}\nEvidence: {safe_evidence}"

    try:
        response = await client.generate(
            model=settings.model_fast,
            system=[CacheableText(text=_SYSTEM_PROMPT, cache=True)],
            messages=[{"role": "user", "content": user_message}],
            tools=[_EXTRACT_GROUNDING_TOOL],
            tool_choice_name="extract_grounding",
            max_tokens=512,
            temperature=0.0,
        )
        if response.tool_inputs:
            tool_output = response.tool_inputs[0]
            raw_literals = tool_output.get("literals", [])
            raw_refs = tool_output.get("schema_references", [])
            if not isinstance(raw_literals, list):
                logger.warning(
                    "Grounding: 'literals' is not a list (%r); using []",
                    type(raw_literals).__name__,
                )
                raw_literals = []
            if not isinstance(raw_refs, list):
                logger.warning(
                    "Grounding: 'schema_references' is not a list (%r); using []",
                    type(raw_refs).__name__,
                )
                raw_refs = []
            literals: list[str] = [str(x) for x in raw_literals if x is not None]
            schema_references: list[str] = [str(x) for x in raw_refs if x is not None]
        else:
            literals = []
            schema_references = []
    except LLMError as exc:
        # Gemini occasionally returns MALFORMED_FUNCTION_CALL for complex inputs
        # (e.g. backtick-quoted column names, long evidence with special characters).
        # Instead of falling back to completely empty grounding (which leaves the
        # LSH index entirely unused), extract keywords directly from the raw text
        # and use those as literals for LSH lookup.  This ensures the schema linker
        # and SQL generators still receive cell-value anchors even when the
        # structured LLM extraction fails.
        combined_text = f"{question} {evidence_str}"
        fallback_literals = _extract_keywords_from_text(combined_text)
        logger.warning(
            "Grounding LLM error for question %r — falling back to keyword extraction "
            "(%d tokens extracted): %s",
            question[:60],
            len(fallback_literals),
            exc,
        )
        get_tracker().record(FallbackEvent(
            component="context_grounder",
            trigger="llm_error",
            action="keyword_extraction_fallback",
            details={
                "question_prefix": question[:60],
                "fallback_keyword_count": len(fallback_literals),
                "error": str(exc),
            },
        ))
        literals = fallback_literals
        schema_references = []

    logger.debug(
        "Grounding: question=%r, literals=%r, schema_refs=%r",
        question[:60],
        literals,
        schema_references,
    )

    # ------------------------------------------------------------------
    # Step 5.2 — LSH cell value retrieval + deduplication
    # ------------------------------------------------------------------
    # Dedup by (table, column) — keep highest similarity_score
    best_by_col: dict[tuple[str, str], "CellMatch"] = {}

    if literals:
        # Build a deduplicated query set: each literal + individual words of
        # multi-word literals (e.g. "Alameda County" → also query "Alameda").
        # This ensures short cell values like County Name='Alameda' are found
        # even when the full phrase matches longer strings with higher similarity.
        query_terms: list[str] = []
        for literal in literals:
            query_terms.append(literal)
            words = literal.split()
            if len(words) > 1:
                # Add individual words with length >= 3 to avoid noise
                query_terms.extend(w for w in words if len(w) >= 3)

        for term in query_terms:
            matches = lsh_index.query(term, top_k=5)
            for match in matches:
                key = (match.table, match.column)
                existing = best_by_col.get(key)
                if existing is None or match.similarity_score > existing.similarity_score:
                    best_by_col[key] = match

    matched_cells = list(best_by_col.values())

    # ------------------------------------------------------------------
    # Step 5.3 — Few-shot example retrieval
    # ------------------------------------------------------------------
    # ExampleStore internally masks the question before embedding; we just
    # pass the raw question.
    # Catch OSError (includes BrokenPipeError) in case the SentenceTransformer
    # model load writes to a broken stdout/stderr pipe. Log as CRITICAL so it
    # surfaces clearly; downstream stages continue with empty few-shot context.
    try:
        few_shot_examples = example_store.query(question, db_id=db_id, top_k=8)
    except OSError as exc:
        logger.critical(
            "ExampleStore.query() OS error (broken pipe?): %s — "
            "returning empty few-shot examples",
            exc,
        )
        few_shot_examples = []

    return GroundingContext(
        matched_cells=matched_cells,
        schema_hints=schema_references,
        few_shot_examples=few_shot_examples,
    )
