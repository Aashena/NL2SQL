"""
Op 7C: ICL (In-Context Learning) Generator.

Uses few-shot examples retrieved by ExampleStore (from GroundingContext) to generate SQL.
Always uses S2 Markdown schema (recall-oriented).
Generates 2-3 candidates (C1, C2, C3) concurrently.

Prompt caching applied to the examples block.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from src.generation.base_generator import (
    SQLCandidate,
    build_base_prompt,
    clean_sql,
)
from src.llm import CacheableText, get_client
from src.config.settings import settings

if TYPE_CHECKING:
    from src.grounding.context_grounder import GroundingContext, ExampleEntry
    from src.schema_linking.schema_linker import LinkedSchemas

logger = logging.getLogger(__name__)

# Token limit for examples block (4 chars ≈ 1 token)
_MAX_EXAMPLE_TOKENS = 6000
_MAX_EXAMPLE_CHARS = _MAX_EXAMPLE_TOKENS * 4
_MAX_EXAMPLES_KEPT = 6

# User prompt instructions for each candidate variant
_C1_INSTRUCTION = "Write the SQL query for this question."
_C2_INSTRUCTION = "First, identify which tables and joins are needed. Then write the SQL query."
_C3_INSTRUCTION = "What is the general SQL pattern for answering this type of question? Then apply it."


class ICLGenerator:
    """Generator C: Few-shot in-context learning with 3 prompt variations."""

    async def generate(
        self,
        question: str,
        evidence: str,
        schemas: "LinkedSchemas",
        grounding: "GroundingContext",
    ) -> list[SQLCandidate]:
        """Generate 3 candidates concurrently (C1, C2, C3)."""
        # Format examples with token guard
        examples = list(grounding.few_shot_examples)
        formatted_examples = self._format_examples(examples)

        # Apply token guard: trim to 6 examples if estimated tokens exceed 6000
        estimated_tokens = len(formatted_examples) // 4
        if estimated_tokens > _MAX_EXAMPLE_TOKENS:
            examples = examples[:_MAX_EXAMPLES_KEPT]
            formatted_examples = self._format_examples(examples)

        # Build the base user prompt (question + evidence + cell matches)
        base_prompt = build_base_prompt(question, evidence, grounding.matched_cells)

        # Build system prompt blocks
        # Block 1: instruction + S2 schema (not cached — schema varies per question)
        instruction_block = CacheableText(
            text=(
                "You are an expert SQL writer. Given a database schema, examples, and a question, "
                "write a correct SQL query.\n\n"
                "Database Schema:\n"
                f"{schemas.s2_markdown}"
            ),
            cache=False,
        )
        # Block 2: few-shot examples (cached — same examples reused across C1/C2/C3)
        examples_block = CacheableText(
            text=formatted_examples,
            cache=True,
        )

        system_blocks = [instruction_block, examples_block]

        tasks = [
            self._generate_one(
                candidate_id="icl_C1",
                system_blocks=system_blocks,
                base_prompt=base_prompt,
                instruction=_C1_INSTRUCTION,
            ),
            self._generate_one(
                candidate_id="icl_C2",
                system_blocks=system_blocks,
                base_prompt=base_prompt,
                instruction=_C2_INSTRUCTION,
            ),
            self._generate_one(
                candidate_id="icl_C3",
                system_blocks=system_blocks,
                base_prompt=base_prompt,
                instruction=_C3_INSTRUCTION,
            ),
        ]

        candidates = await asyncio.gather(*tasks)
        return list(candidates)

    def _format_examples(self, examples: list) -> str:
        """Format ExampleEntry list into numbered ## Example N blocks."""
        if not examples:
            return ""

        blocks: list[str] = []
        for i, ex in enumerate(examples, start=1):
            evidence_str = ex.evidence if ex.evidence else "None"
            block = (
                f"## Example {i}\n"
                f"Question: {ex.question}\n"
                f"Evidence: {evidence_str}\n"
                f"SQL: {ex.sql}"
            )
            blocks.append(block)

        return "\n\n".join(blocks)

    async def _generate_one(
        self,
        candidate_id: str,
        system_blocks: list[CacheableText],
        base_prompt: str,
        instruction: str,
    ) -> SQLCandidate:
        """Generate a single ICL candidate."""
        # Combine base prompt with per-variant instruction
        user_prompt = f"{base_prompt}\n\n{instruction}"

        client = get_client()
        try:
            response = await client.generate(
                model=settings.model_powerful,
                system=system_blocks,
                messages=[{"role": "user", "content": user_prompt}],
                tools=[],
                tool_choice_name=None,
                thinking=None,
                max_tokens=4096,  # increased from 2000: gemini-2.5-pro needs headroom for implicit reasoning
            )
        except Exception as exc:
            logger.error("ICLGenerator %s failed: %s", candidate_id, exc)
            return SQLCandidate(
                sql="",
                generator_id=candidate_id,
                schema_used="s2",
                schema_format="markdown",
                reasoning_trace=None,
                error_flag=True,
            )

        raw_text = response.text
        if raw_text is None:
            logger.warning(
                "ICLGenerator %s: response.text is None", candidate_id
            )
            return SQLCandidate(
                sql="",
                generator_id=candidate_id,
                schema_used="s2",
                schema_format="markdown",
                reasoning_trace=None,
                error_flag=True,
            )

        sql = clean_sql(raw_text)
        if not sql:
            logger.warning(
                "ICLGenerator %s: clean_sql() returned empty string", candidate_id
            )
            return SQLCandidate(
                sql="",
                generator_id=candidate_id,
                schema_used="s2",
                schema_format="markdown",
                reasoning_trace=None,
                error_flag=True,
            )

        return SQLCandidate(
            sql=sql,
            generator_id=candidate_id,
            schema_used="s2",
            schema_format="markdown",
            reasoning_trace=None,
            error_flag=False,
        )
