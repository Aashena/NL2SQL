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

from src.llm import get_client, CacheableText, ToolParam, LLMError
from src.config.settings import settings

if TYPE_CHECKING:
    from src.indexing.lsh_index import CellMatch, LSHIndex
    from src.indexing.example_store import ExampleEntry, ExampleStore

logger = logging.getLogger(__name__)

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
    user_message = f"Question: {question}\nEvidence: {evidence_str}"

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
            literals: list[str] = tool_output.get("literals", [])
            schema_references: list[str] = tool_output.get("schema_references", [])
        else:
            literals = []
            schema_references = []
    except LLMError as exc:
        # Gemini occasionally returns MALFORMED_FUNCTION_CALL for complex inputs
        # (e.g. dates with forward slashes, special characters). Fall back to
        # empty grounding rather than crashing the entire question pipeline.
        logger.warning(
            "Grounding LLM error for question %r — falling back to empty grounding: %s",
            question[:60],
            exc,
        )
        literals = []
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
    few_shot_examples = example_store.query(question, db_id=db_id, top_k=8)

    return GroundingContext(
        matched_cells=matched_cells,
        schema_hints=schema_references,
        few_shot_examples=few_shot_examples,
    )
