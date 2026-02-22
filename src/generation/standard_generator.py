"""
Ops 7B: Standard + Complex SQL Generators.

Generator B1 (model_fast, standard prompt): 2 candidates using S1 and S2 Markdown schemas.
Generator B2 (model_powerful, complex SQL prompt): 2 candidates using S1 and S2 Markdown schemas.

Both generators run all 4 calls concurrently via asyncio.gather().
SQL comes in response.text (no tool-use).
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
    from src.grounding.context_grounder import GroundingContext
    from src.schema_linking.schema_linker import LinkedSchemas

logger = logging.getLogger(__name__)

_B1_SYSTEM_TEMPLATE = (
    "You are an expert SQL writer. Given a database schema and a question, write a correct SQL query.\n"
    "Focus on accuracy. Use the schema metadata to understand column meanings.\n\n"
    "Database Schema:\n"
    "{markdown_schema}"
)

_B2_SYSTEM_TEMPLATE = (
    "You are an expert SQL writer specializing in advanced query patterns. For complex questions,\n"
    "prefer using CTEs, window functions, or subqueries to express logic clearly.\n"
    "Avoid unnested JOINs when CTEs improve readability and correctness.\n\n"
    "Database Schema:\n"
    "{markdown_schema}"
)


class StandardAndComplexGenerator:
    """Generators B1 (standard) and B2 (complex SQL) — Markdown schemas, no extended thinking."""

    async def generate(
        self,
        question: str,
        evidence: str,
        schemas: "LinkedSchemas",
        grounding: "GroundingContext",
    ) -> list[SQLCandidate]:
        """Generate 4 candidates concurrently (B1a, B1b, B2a, B2b)."""
        user_prompt = build_base_prompt(question, evidence, grounding.matched_cells)

        tasks = [
            # B1: standard prompt, fast model
            self._generate_one(
                candidate_id="standard_B1_s1",
                schema_used="s1",
                markdown_schema=schemas.s1_markdown,
                system_template=_B1_SYSTEM_TEMPLATE,
                model=settings.model_fast,
                user_prompt=user_prompt,
            ),
            self._generate_one(
                candidate_id="standard_B1_s2",
                schema_used="s2",
                markdown_schema=schemas.s2_markdown,
                system_template=_B1_SYSTEM_TEMPLATE,
                model=settings.model_fast,
                user_prompt=user_prompt,
            ),
            # B2: complex SQL prompt, powerful model
            self._generate_one(
                candidate_id="complex_B2_s1",
                schema_used="s1",
                markdown_schema=schemas.s1_markdown,
                system_template=_B2_SYSTEM_TEMPLATE,
                model=settings.model_powerful,
                user_prompt=user_prompt,
            ),
            self._generate_one(
                candidate_id="complex_B2_s2",
                schema_used="s2",
                markdown_schema=schemas.s2_markdown,
                system_template=_B2_SYSTEM_TEMPLATE,
                model=settings.model_powerful,
                user_prompt=user_prompt,
            ),
        ]

        candidates = await asyncio.gather(*tasks)
        return list(candidates)

    async def _generate_one(
        self,
        candidate_id: str,
        schema_used: str,
        markdown_schema: str,
        system_template: str,
        model: str,
        user_prompt: str,
    ) -> SQLCandidate:
        """Generate a single candidate using the given model and system template."""
        system_text = system_template.format(markdown_schema=markdown_schema)
        system_prompt = CacheableText(text=system_text, cache=True)

        client = get_client()
        try:
            response = await client.generate(
                model=model,
                system=[system_prompt],
                messages=[{"role": "user", "content": user_prompt}],
                tools=[],
                tool_choice_name=None,
                thinking=None,
                max_tokens=2000,
            )
        except Exception as exc:
            logger.error(
                "StandardAndComplexGenerator %s failed: %s", candidate_id, exc
            )
            return SQLCandidate(
                sql="",
                generator_id=candidate_id,
                schema_used=schema_used,
                schema_format="markdown",
                reasoning_trace=None,
                error_flag=True,
            )

        raw_text = response.text
        if raw_text is None:
            logger.warning(
                "StandardAndComplexGenerator %s: response.text is None", candidate_id
            )
            return SQLCandidate(
                sql="",
                generator_id=candidate_id,
                schema_used=schema_used,
                schema_format="markdown",
                reasoning_trace=None,
                error_flag=True,
            )

        sql = clean_sql(raw_text)
        if not sql:
            logger.warning(
                "StandardAndComplexGenerator %s: clean_sql() returned empty string",
                candidate_id,
            )
            return SQLCandidate(
                sql="",
                generator_id=candidate_id,
                schema_used=schema_used,
                schema_format="markdown",
                reasoning_trace=None,
                error_flag=True,
            )

        return SQLCandidate(
            sql=sql,
            generator_id=candidate_id,
            schema_used=schema_used,
            schema_format="markdown",
            reasoning_trace=None,
            error_flag=False,
        )
