"""
Online Pipeline — Ops 5 → 9

Orchestrates the per-question pipeline:
  Op 5: Context Grounding
  Op 6: Adaptive Schema Linking
  Op 7: Diverse SQL Generation (3 generators in parallel)
  Op 8: Query Fixer
  Op 9: Adaptive Selection

Entry point: answer_question(entry, artifacts, db_path) -> PipelineResult

A module-level asyncio.Semaphore(10) is exposed so that callers (e.g. the
evaluation script) can use it to cap the number of concurrent answer_question
invocations to N without blocking the event loop.  The semaphore is NOT
acquired inside answer_question itself — callers wrap the call as needed.

Timeout design (Root Cause 2 & 3 fix):
  GENERATOR_TIMEOUT_S — each individual generator gets this many seconds.
    If it exceeds the limit it is cancelled; its slot returns an empty list
    so Op 8/9 proceed on whatever candidates arrived first.
  QUESTION_SOFT_TIMEOUT_S — the entire answer_question() call is wrapped
    with this deadline.  If Op 5/6 or Op 8/9 take too long the function
    returns a graceful fallback before the external safety-net fires.
    Set it well below the safety-net (1560 s) but above a normal question.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.grounding.context_grounder import ground_context
from src.schema_linking.schema_linker import link_schema
from src.generation.reasoning_generator import ReasoningGenerator
from src.generation.standard_generator import StandardAndComplexGenerator
from src.generation.icl_generator import ICLGenerator
from src.fixing.query_fixer import QueryFixer
from src.selection.adaptive_selector import AdaptiveSelector

if TYPE_CHECKING:
    from src.data.bird_loader import BirdEntry
    from src.pipeline.offline_pipeline import OfflineArtifacts

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level semaphore
# ---------------------------------------------------------------------------

#: Global semaphore that limits concurrent API calls across all pipeline runs.
#: The evaluation script should acquire this before calling answer_question.
PIPELINE_SEMAPHORE: asyncio.Semaphore = asyncio.Semaphore(10)

# ---------------------------------------------------------------------------
# Timeout constants
# ---------------------------------------------------------------------------

#: Maximum seconds to wait for a single Op 7 generator.
#: Extended thinking on complex schemas can take 60–90 s; 120 s is generous
#: but still prevents a single hung generator from blocking the pipeline.
GENERATOR_TIMEOUT_S: int = 120

#: Soft per-question deadline in seconds.  The whole answer_question() body
#: is wrapped with this limit.  If it fires, a graceful fallback PipelineResult
#: is returned.  Must be well below the external safety-net (typically 1560 s).
QUESTION_SOFT_TIMEOUT_S: int = 300


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Result of a single online pipeline run."""
    final_sql: str
    selection_method: str       # "fast_path" | "tournament" | "fallback"
    cluster_count: int
    candidates_evaluated: int
    generator_wins: dict        # generator_id → win count (from tournament)
    total_cost_estimate: float  # best-effort cost estimate (always 0.0 for now)


# ---------------------------------------------------------------------------
# Singleton generator instances (created once per process)
# ---------------------------------------------------------------------------

_reasoning_generator = ReasoningGenerator()
_standard_generator = StandardAndComplexGenerator()
_icl_generator = ICLGenerator()
_query_fixer = QueryFixer()
_adaptive_selector = AdaptiveSelector()


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------

async def answer_question(
    entry: "BirdEntry",
    artifacts: "OfflineArtifacts",
    db_path: str,
) -> PipelineResult:
    """
    Run the complete online pipeline (Ops 5–9) for a single BIRD question.

    Parameters
    ----------
    entry:
        A BirdEntry from the BIRD dataset (question, evidence, db_id, …).
    artifacts:
        Pre-built OfflineArtifacts for the entry's database.
    db_path:
        Filesystem path to the SQLite database file.

    Returns
    -------
    PipelineResult with the best SQL and selection metadata.
    """
    try:
        return await asyncio.wait_for(
            _answer_question_inner(entry, artifacts, db_path),
            timeout=QUESTION_SOFT_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        db_id = entry.db_id
        logger.warning(
            "[%s] answer_question() exceeded soft timeout of %ds — returning empty fallback",
            db_id,
            QUESTION_SOFT_TIMEOUT_S,
        )
        return PipelineResult(
            final_sql="",
            selection_method="timeout_fallback",
            cluster_count=0,
            candidates_evaluated=0,
            generator_wins={},
            total_cost_estimate=0.0,
        )


async def _answer_question_inner(
    entry: "BirdEntry",
    artifacts: "OfflineArtifacts",
    db_path: str,
) -> PipelineResult:
    """Inner implementation of answer_question (wrapped by soft timeout)."""
    question = entry.question
    evidence = entry.evidence or ""
    db_id = entry.db_id

    # -----------------------------------------------------------------------
    # Op 5: Context Grounding
    # -----------------------------------------------------------------------
    logger.info("[%s] Op 5: Context grounding …", db_id)
    grounding = await ground_context(
        question=question,
        evidence=evidence,
        db_id=db_id,
        lsh_index=artifacts.lsh_index,
        example_store=artifacts.example_store,
    )
    logger.debug(
        "[%s] Grounding: %d cell matches, %d schema hints, %d few-shot examples",
        db_id,
        len(grounding.matched_cells),
        len(grounding.schema_hints),
        len(grounding.few_shot_examples),
    )

    # -----------------------------------------------------------------------
    # Build available_fields from the FAISS index's stored field metadata.
    # Each element: (table, column, short_summary, long_summary)
    # -----------------------------------------------------------------------
    available_fields: list[tuple[str, str, str, str]] = [
        (
            f["table"],
            f["column"],
            f.get("short_summary", ""),
            f.get("long_summary", ""),
        )
        for f in artifacts.faiss_index._fields
    ]

    # -----------------------------------------------------------------------
    # Op 6: Adaptive Schema Linking
    # -----------------------------------------------------------------------
    logger.info("[%s] Op 6: Schema linking …", db_id)
    schemas = await link_schema(
        question=question,
        evidence=evidence,
        grounding_context=grounding,
        faiss_index=artifacts.faiss_index,
        full_ddl=artifacts.schemas.ddl,
        full_markdown=artifacts.schemas.markdown,
        available_fields=available_fields,
    )
    logger.debug(
        "[%s] Schema: S1=%d fields, S2=%d fields",
        db_id,
        len(schemas.s1_fields),
        len(schemas.s2_fields),
    )

    # -----------------------------------------------------------------------
    # Op 7: Diverse SQL Generation (3 generators run in parallel)
    #
    # Each generator is individually wrapped with GENERATOR_TIMEOUT_S so that
    # a slow or hung generator (e.g. ICL generator blocked on a long API call)
    # cannot prevent Op 8/9 from running on the candidates that did arrive.
    # This fixes the cascading-slowdown problem: previously a 300 s ICL call
    # would stall the whole question and starve subsequent questions.
    # -----------------------------------------------------------------------
    logger.info("[%s] Op 7: Generating SQL candidates …", db_id)

    async def _timed_generate(coro, name: str) -> list:
        """Run a generator coroutine bounded by GENERATOR_TIMEOUT_S."""
        try:
            return await asyncio.wait_for(coro, timeout=GENERATOR_TIMEOUT_S)
        except asyncio.TimeoutError:
            logger.warning(
                "[%s] Generator '%s' exceeded %ds timeout — dropping its candidates",
                db_id, name, GENERATOR_TIMEOUT_S,
            )
            return []
        except Exception as exc:
            logger.error(
                "[%s] Generator '%s' raised an unexpected error: %s", db_id, name, exc
            )
            return []

    gen_results = await asyncio.gather(
        _timed_generate(
            _reasoning_generator.generate(
                question=question,
                evidence=evidence,
                schemas=schemas,
                grounding=grounding,
            ),
            "reasoning",
        ),
        _timed_generate(
            _standard_generator.generate(
                question=question,
                evidence=evidence,
                schemas=schemas,
                grounding=grounding,
            ),
            "standard",
        ),
        _timed_generate(
            _icl_generator.generate(
                question=question,
                evidence=evidence,
                schemas=schemas,
                grounding=grounding,
            ),
            "icl",
        ),
    )

    # Flatten all candidate lists; empty lists from timed-out generators are skipped.
    candidates = [c for sublist in gen_results for c in sublist]
    logger.info(
        "[%s] Op 7: %d total candidates generated "
        "(%d reasoning, %d standard/complex, %d ICL)",
        db_id,
        len(candidates),
        len(gen_results[0]),
        len(gen_results[1]),
        len(gen_results[2]),
    )

    # -----------------------------------------------------------------------
    # Op 8: Query Fixer
    # -----------------------------------------------------------------------
    logger.info("[%s] Op 8: Fixing candidates …", db_id)
    fixed_candidates = await _query_fixer.fix_candidates(
        candidates=candidates,
        question=question,
        evidence=evidence,
        schemas=schemas,
        db_path=db_path,
        cell_matches=grounding.matched_cells,
    )
    logger.debug(
        "[%s] Op 8: %d fixed candidates",
        db_id,
        len(fixed_candidates),
    )

    # -----------------------------------------------------------------------
    # Op 9: Adaptive Selection
    # -----------------------------------------------------------------------
    logger.info("[%s] Op 9: Selecting best SQL …", db_id)
    selection = await _adaptive_selector.select(
        candidates=fixed_candidates,
        question=question,
        evidence=evidence,
        schemas=schemas,
        db_path=db_path,
    )
    logger.info(
        "[%s] Op 9: method=%s clusters=%d evaluated=%d",
        db_id,
        selection.selection_method,
        selection.cluster_count,
        selection.candidates_evaluated,
    )

    return PipelineResult(
        final_sql=selection.final_sql,
        selection_method=selection.selection_method,
        cluster_count=selection.cluster_count,
        candidates_evaluated=selection.candidates_evaluated,
        generator_wins=selection.tournament_wins,
        total_cost_estimate=0.0,
    )
