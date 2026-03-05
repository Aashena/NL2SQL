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
from typing import TYPE_CHECKING, Optional

from src.config.settings import settings
from src.grounding.context_grounder import ground_context, GroundingContext
from src.schema_linking.schema_linker import link_schema, LinkedSchemas
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
#: Configurable via GENERATOR_TIMEOUT_S env var (default 600).
#: Increase for slow local providers (e.g. MLX) where all calls are serialized.
GENERATOR_TIMEOUT_S: int = settings.generator_timeout_s

#: Soft per-question deadline in seconds.
#: Configurable via QUESTION_SOFT_TIMEOUT_S env var (default 3600).
QUESTION_SOFT_TIMEOUT_S: int = settings.question_soft_timeout_s


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


@dataclass
class QuestionContext:
    """
    Intermediate state produced by Phase 1 (Op5+6, index-heavy).

    Contains all data needed to run Phase 2 (Op7-9, LLM-only).
    Once a QuestionContext has been produced for every question in a
    database batch, the caller may free the heavy index objects
    (lsh_index, faiss_index, example_store) before starting Phase 2.
    """
    entry: "BirdEntry"
    db_path: str
    grounding: GroundingContext
    schemas: LinkedSchemas


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
            _run_full_pipeline(entry, artifacts, db_path),
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


async def _run_full_pipeline(
    entry: "BirdEntry",
    artifacts: "OfflineArtifacts",
    db_path: str,
) -> PipelineResult:
    """Convenience wrapper: Phase 1 then Phase 2 sequentially (no timeout)."""
    ctx = await _prepare_context_inner(entry, artifacts, db_path)
    return await _answer_from_context_inner(ctx)


async def prepare_context(
    entry: "BirdEntry",
    artifacts: "OfflineArtifacts",
    db_path: str,
) -> Optional[QuestionContext]:
    """
    Phase 1 — Op5 (context grounding) + Op6 (schema linking).

    Uses the heavy index objects (lsh_index, faiss_index, example_store).
    Returns a QuestionContext with all data needed by Phase 2 (Op7-9).
    Returns None if an unrecoverable error occurs (caller should treat as skip).

    Once this returns successfully for all questions in a database batch,
    the caller may free lsh_index, faiss_index, and example_store before
    calling answer_from_context — they are not accessed again.
    """
    try:
        return await _prepare_context_inner(entry, artifacts, db_path)
    except Exception as exc:
        logger.error(
            "[%s] prepare_context failed for Q#%d: %s",
            entry.db_id, entry.question_id, exc,
        )
        return None


async def answer_from_context(ctx: QuestionContext) -> PipelineResult:
    """
    Phase 2 — Op7 (generation) + Op8 (fixing) + Op9 (selection).

    Requires no artifact index access — only ctx.grounding, ctx.schemas,
    ctx.entry, and ctx.db_path.  Safe to call after indexes are freed.
    """
    try:
        return await asyncio.wait_for(
            _answer_from_context_inner(ctx),
            timeout=QUESTION_SOFT_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "[%s] answer_from_context() exceeded soft timeout of %ds",
            ctx.entry.db_id, QUESTION_SOFT_TIMEOUT_S,
        )
        return PipelineResult(
            final_sql="",
            selection_method="timeout_fallback",
            cluster_count=0,
            candidates_evaluated=0,
            generator_wins={},
            total_cost_estimate=0.0,
        )


async def _prepare_context_inner(
    entry: "BirdEntry",
    artifacts: "OfflineArtifacts",
    db_path: str,
) -> QuestionContext:
    """Op5+6 implementation (no timeout wrapper)."""
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

    return QuestionContext(
        entry=entry,
        db_path=db_path,
        grounding=grounding,
        schemas=schemas,
    )


async def _answer_from_context_inner(ctx: QuestionContext) -> PipelineResult:
    """Op7+8+9 implementation (no timeout wrapper). No artifact access."""
    entry = ctx.entry
    question = entry.question
    evidence = entry.evidence or ""
    db_id = entry.db_id
    grounding = ctx.grounding
    schemas = ctx.schemas
    db_path = ctx.db_path

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
