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
from src.monitoring.fallback_tracker import FallbackEvent, get_tracker

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

# Alternative templates used when S₁ == S₂ (schema linker found no recall-only fields).
# Instead of feeding the same schema twice and getting duplicate SQL, these templates
# steer the model toward a structurally different solution approach.
_B1_ALT_SYSTEM_TEMPLATE = (
    "You are an expert SQL writer. Given a database schema and a question, write a correct SQL query.\n"
    "Explore an alternative approach: consider a different JOIN order, use a subquery where you\n"
    "would normally use a JOIN (or vice versa), or handle edge cases the straightforward solution\n"
    "might miss (e.g. NULLs, ties, empty groups).\n\n"
    "Database Schema:\n"
    "{markdown_schema}"
)

_B2_ALT_SYSTEM_TEMPLATE = (
    "You are an expert SQL writer specializing in advanced query patterns.\n"
    "Write an alternative SQL solution using a different structural approach: if the obvious\n"
    "solution uses a JOIN, try a correlated subquery or CTE instead; if it uses GROUP BY,\n"
    "consider a window function. Also check for edge cases such as NULLs, duplicate rows,\n"
    "or boundary conditions.\n\n"
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

        # When the schema linker's recall expansion (S₂) adds no new fields, S₁ and S₂
        # are identical. Feeding the same schema twice with the same prompt produces
        # duplicate SQL and wastes API calls. Use alternative prompt templates for the
        # s2 variants instead so the model explores a different structural approach.
        s1_eq_s2 = set(schemas.s1_fields) == set(schemas.s2_fields)
        if s1_eq_s2:
            logger.debug(
                "S1==S2 detected (%d fields); using alternative prompts for B1_s2 and B2_s2",
                len(schemas.s1_fields),
            )
        b1_s2_template = _B1_ALT_SYSTEM_TEMPLATE if s1_eq_s2 else _B1_SYSTEM_TEMPLATE
        b2_s2_template = _B2_ALT_SYSTEM_TEMPLATE if s1_eq_s2 else _B2_SYSTEM_TEMPLATE

        tasks = [
            # B1: standard prompt, fast model — temperature=0.3 for slight variation while reducing duplicates
            self._generate_one(
                candidate_id="standard_B1_s1",
                schema_used="s1",
                markdown_schema=schemas.s1_markdown,
                system_template=_B1_SYSTEM_TEMPLATE,
                model=settings.model_fast,
                user_prompt=user_prompt,
                max_tokens=4096,
                temperature=0.3,
            ),
            self._generate_one(
                candidate_id="standard_B1_s2",
                schema_used="s2",
                markdown_schema=schemas.s2_markdown,
                system_template=b1_s2_template,
                model=settings.model_fast,
                user_prompt=user_prompt,
                max_tokens=4096,
                temperature=0.3,
            ),
            # B2: complex SQL prompt, powerful model — 4096 tokens to avoid MAX_TOKENS truncation
            # temperature=0.3 for slight variation while remaining close to deterministic
            self._generate_one(
                candidate_id="complex_B2_s1",
                schema_used="s1",
                markdown_schema=schemas.s1_markdown,
                system_template=_B2_SYSTEM_TEMPLATE,
                model=settings.model_powerful_list,
                user_prompt=user_prompt,
                max_tokens=4096,
                temperature=0.3,
            ),
            self._generate_one(
                candidate_id="complex_B2_s2",
                schema_used="s2",
                markdown_schema=schemas.s2_markdown,
                system_template=b2_s2_template,
                model=settings.model_powerful_list,
                user_prompt=user_prompt,
                max_tokens=4096,
                temperature=0.3,
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
        max_tokens: int = 2000,
        temperature: float = 1.0,
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
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            logger.error(
                "StandardAndComplexGenerator %s failed: %s", candidate_id, exc
            )
            get_tracker().record(FallbackEvent(
                component="standard_generator",
                trigger="llm_error",
                action="empty_result",
                details={"candidate_id": candidate_id, "error": str(exc), "schema_used": schema_used},
                severity="error",
            ))
            return SQLCandidate(
                sql="",
                generator_id=candidate_id,
                schema_used=schema_used,
                schema_format="markdown",
                reasoning_trace=None,
                error_flag=True,
            )

        raw_text = response.text

        # Detect and discard truncated responses: finish_reason=MAX_TOKENS with
        # partial text means the SQL was cut off mid-statement.  A truncated SQL
        # always fails SQLite with "incomplete input"; better to mark error_flag
        # than to pass broken SQL to the query fixer.
        if "MAX_TOKENS" in str(response.finish_reason):
            logger.warning(
                "StandardAndComplexGenerator %s: response truncated at MAX_TOKENS "
                "(partial text discarded)",
                candidate_id,
            )
            get_tracker().record(FallbackEvent(
                component="standard_generator",
                trigger="max_tokens",
                action="empty_result",
                details={"candidate_id": candidate_id, "schema_used": schema_used},
            ))
            return SQLCandidate(
                sql="",
                generator_id=candidate_id,
                schema_used=schema_used,
                schema_format="markdown",
                reasoning_trace=None,
                error_flag=True,
            )

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
