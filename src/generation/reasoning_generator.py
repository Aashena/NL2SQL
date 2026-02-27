"""
Op 7A: Reasoning Generator — uses extended thinking for deeper analysis.

Produces 4 SQLCandidate objects (A1-A4):
- A1: S1 DDL, minimal prompt
- A2: S1 DDL, step-by-step prompt
- A3: S2 DDL, minimal prompt
- A4: S2 DDL, step-by-step prompt

Diversity via prompt variation (extended thinking forces temperature=1 internally).
All 4 calls run concurrently via asyncio.gather().
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from src.generation.base_generator import (
    SQLCandidate,
    build_base_prompt,
    clean_sql,
    validate_sql_syntax,
)
from src.llm import CacheableText, ThinkingConfig, get_client
from src.config.settings import settings
from src.monitoring.fallback_tracker import FallbackEvent, get_tracker

if TYPE_CHECKING:
    from src.grounding.context_grounder import GroundingContext
    from src.schema_linking.schema_linker import LinkedSchemas

logger = logging.getLogger(__name__)

_STEP_BY_STEP_SUFFIX = (
    "\n\nThink carefully about:\n"
    "1. Which tables and columns are needed\n"
    "2. How tables should be joined\n"
    "3. Any WHERE conditions based on matched values\n"
    "4. Aggregations or subqueries needed\n\n"
    "Write the SQL query."
)


class ReasoningGenerator:
    """Generator A: Extended thinking + DDL schema + prompt variation."""

    async def generate(
        self,
        question: str,
        evidence: str,
        schemas: "LinkedSchemas",
        grounding: "GroundingContext",
    ) -> list[SQLCandidate]:
        """Generate 4 candidates concurrently (A1-A4)."""
        budget_tokens = self._get_budget_tokens(schemas)

        tasks = [
            self._generate_one(
                candidate_id="A1",
                schema_used="s1",
                ddl_schema=schemas.s1_ddl,
                question=question,
                evidence=evidence,
                cell_matches=grounding.matched_cells,
                budget_tokens=budget_tokens,
                use_step_by_step=False,
            ),
            self._generate_one(
                candidate_id="A2",
                schema_used="s1",
                ddl_schema=schemas.s1_ddl,
                question=question,
                evidence=evidence,
                cell_matches=grounding.matched_cells,
                budget_tokens=budget_tokens,
                use_step_by_step=True,
            ),
            self._generate_one(
                candidate_id="A3",
                schema_used="s2",
                ddl_schema=schemas.s2_ddl,
                question=question,
                evidence=evidence,
                cell_matches=grounding.matched_cells,
                budget_tokens=budget_tokens,
                use_step_by_step=False,
            ),
            self._generate_one(
                candidate_id="A4",
                schema_used="s2",
                ddl_schema=schemas.s2_ddl,
                question=question,
                evidence=evidence,
                cell_matches=grounding.matched_cells,
                budget_tokens=budget_tokens,
                use_step_by_step=True,
            ),
        ]

        candidates = await asyncio.gather(*tasks)
        return list(candidates)

    def _get_budget_tokens(self, schemas: "LinkedSchemas") -> int:
        """Adaptive budget: 4000 for 1-2 tables, 6000 for 3-4, 8000 for 5+."""
        n_tables = len({t for t, _ in schemas.s2_fields})
        if n_tables <= 2:
            return 4000
        elif n_tables <= 4:
            return 6000
        else:
            return 8000

    async def _generate_one(
        self,
        candidate_id: str,
        schema_used: str,
        ddl_schema: str,
        question: str,
        evidence: str,
        cell_matches: list,
        budget_tokens: int,
        use_step_by_step: bool,
    ) -> SQLCandidate:
        """Generate a single candidate using extended thinking."""
        system_prompt = CacheableText(
            text=(
                "You are an expert SQL query writer. Given a database schema and a "
                "natural language question, write a precise SQL query that answers "
                "the question.\n\n"
                "Database Schema:\n"
                f"{ddl_schema}"
            ),
            cache=True,
        )

        user_prompt = build_base_prompt(question, evidence, cell_matches)
        if use_step_by_step:
            # Replace the final "Write a SQL query..." line with the step-by-step suffix
            # build_base_prompt ends with "Write a SQL query that answers the question."
            # We replace that with the more detailed step-by-step instruction.
            base_ending = "Write a SQL query that answers the question."
            if user_prompt.endswith(base_ending):
                user_prompt = user_prompt[: -len(base_ending)].rstrip()
            user_prompt = user_prompt + _STEP_BY_STEP_SUFFIX

        client = get_client()
        try:
            response = await client.generate(
                model=settings.model_reasoning,
                system=[system_prompt],
                messages=[{"role": "user", "content": user_prompt}],
                tools=[],
                tool_choice_name=None,
                thinking=ThinkingConfig(enabled=True, budget_tokens=budget_tokens),
                max_tokens=budget_tokens + 2000,
            )
        except Exception as exc:
            logger.error(
                "ReasoningGenerator %s failed: %s", candidate_id, exc
            )
            get_tracker().record(FallbackEvent(
                component="reasoning_generator",
                trigger="llm_error",
                action="empty_result",
                details={"candidate_id": candidate_id, "error": str(exc), "schema_used": schema_used},
                severity="error",
            ))
            return SQLCandidate(
                sql="",
                generator_id=f"reasoning_{candidate_id}",
                schema_used=schema_used,
                schema_format="ddl",
                reasoning_trace=None,
                error_flag=True,
            )

        # Extract SQL from text response (no tool-use for reasoning generator)
        raw_text = response.text
        if raw_text is None:
            logger.warning(
                "ReasoningGenerator %s: response.text is None", candidate_id
            )
            return SQLCandidate(
                sql="",
                generator_id=f"reasoning_{candidate_id}",
                schema_used=schema_used,
                schema_format="ddl",
                reasoning_trace=response.thinking,
                error_flag=True,
            )

        sql = clean_sql(raw_text)
        if not sql:
            logger.warning(
                "ReasoningGenerator %s: clean_sql() returned empty string", candidate_id
            )
            return SQLCandidate(
                sql="",
                generator_id=f"reasoning_{candidate_id}",
                schema_used=schema_used,
                schema_format="ddl",
                reasoning_trace=response.thinking,
                error_flag=True,
            )

        return SQLCandidate(
            sql=sql,
            generator_id=f"reasoning_{candidate_id}",
            schema_used=schema_used,
            schema_format="ddl",
            reasoning_trace=response.thinking,
            error_flag=False,
        )
