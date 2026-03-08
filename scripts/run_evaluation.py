#!/usr/bin/env python3
"""
Evaluation Script — run the online NL2SQL pipeline against a BIRD split.

Usage examples:
  # Evaluate full dev set (1534 questions), 4 concurrent workers
  python scripts/run_evaluation.py --split dev

  # Evaluate mini-dev (500 questions) with 8 workers
  python scripts/run_evaluation.py --split mini_dev --workers 8

  # Resume a partial run
  python scripts/run_evaluation.py --split dev --resume results/dev_results.json

  # Evaluate only one database
  python scripts/run_evaluation.py --split dev --db_filter california_schools

Features:
  - Saves results incrementally (resumable via --resume)
  - Progress bar with live EX% estimate (tqdm)
  - Summary table at end broken down by difficulty and database
  - Concurrent pipeline calls via asyncio.Semaphore(N) where N = --workers
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import ctypes.util
import gc
import json
import logging
import os
import re
import sys
import time

import psutil
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so `src` imports work whether the
# script is invoked from the repo root or from within the scripts/ directory.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Suppress noisy HuggingFace output before any HF-related imports
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# ---------------------------------------------------------------------------
# Configure logging before any project imports
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("run_evaluation")

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from src.config.settings import settings
from src.data.bird_loader import load_bird_split, BirdEntry
from src.data.database import execute_sql, ExecutionResult
from src.indexing.faiss_index import FAISSIndex
from src.indexing.lsh_index import LSHIndex
from src.indexing.example_store import ExampleStore
from src.monitoring.fallback_tracker import get_tracker
from src.pipeline.online_pipeline import (
    answer_question,
    prepare_context,
    answer_from_context,
    QuestionContext,
    PipelineResult,
    PIPELINE_SEMAPHORE,
)
from src.preprocessing.schema_formatter import FormattedSchemas

# ---------------------------------------------------------------------------
# Memory return helper — encourages the OS allocator to reclaim freed pages.
# On Linux: calls malloc_trim(0). On macOS: calls malloc_zone_pressure_relief.
# Best-effort: silently ignored if the call fails.
# ---------------------------------------------------------------------------

def _malloc_trim() -> None:
    """Ask the C allocator to return freed memory pages to the OS."""
    try:
        import sys
        if sys.platform == "darwin":
            libc = ctypes.CDLL("libc.dylib", use_errno=True)
            # malloc_zone_pressure_relief(zone=NULL, goal=0) → release all zones
            fn = libc.malloc_zone_pressure_relief
            fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            fn.restype = ctypes.c_size_t
            fn(None, 0)
        else:
            # Linux / glibc
            libc_name = ctypes.util.find_library("c") or "libc.so.6"
            libc = ctypes.CDLL(libc_name, use_errno=True)
            libc.malloc_trim(ctypes.c_size_t(0))
    except Exception:
        pass  # non-critical — just best-effort


def _mem_mib(label: str) -> float:
    """Log and return the process RSS in MiB at the given checkpoint label."""
    rss_bytes = psutil.Process().memory_info().rss
    rss_mib = rss_bytes / (1024 ** 2)
    print(f"  [MEM] {label}: {rss_mib:.1f} MiB RSS", flush=True)
    return rss_mib


try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
    logger.warning("tqdm not installed — progress bar disabled.  Install with: pip install tqdm")


# ---------------------------------------------------------------------------
# EvaluationEntry dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvaluationEntry:
    question_id: int
    db_id: str
    difficulty: str
    predicted_sql: str
    truth_sql: str
    correct: bool
    selection_method: str
    winner_generator: str       # generator_id with most wins (or "fallback")
    cluster_count: int
    latency_seconds: float


# ---------------------------------------------------------------------------
# Result normalization for EX comparison
# ---------------------------------------------------------------------------

def _normalize_value(v) -> str:
    """Normalize a single result cell for comparison."""
    if v is None:
        return "NULL"
    if isinstance(v, float):
        # Normalize 1.0 → "1", 3.14159... → "3.14" (2 decimal places)
        if v == int(v):
            return str(int(v))
        return f"{v:.2f}"
    return str(v)


def _normalize_rows(rows: list) -> list[tuple]:
    """Normalize and sort rows for result-set comparison."""
    normalized = [
        tuple(_normalize_value(cell) for cell in row)
        for row in rows
    ]
    return sorted(normalized)


def _results_match(pred_rows: list, gold_rows: list) -> bool:
    """Return True if the two result sets are equivalent after normalization."""
    return _normalize_rows(pred_rows) == _normalize_rows(gold_rows)


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

def _db_path_for_split(split: str, db_id: str) -> str:
    """Return the filesystem path to the SQLite database for a given split/db_id."""
    data_dir = Path(settings.bird_data_dir)
    # Try split-specific sub-directory first (e.g. dev/dev_databases/db_id/db_id.sqlite)
    candidates = [
        data_dir / split / f"{split}_databases" / db_id / f"{db_id}.sqlite",
        data_dir / split / "databases" / db_id / f"{db_id}.sqlite",
        data_dir / "databases" / db_id / f"{db_id}.sqlite",
        # Flat layout fallback
        data_dir / db_id / f"{db_id}.sqlite",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    # Final fallback: return first candidate (will fail at runtime with a clear error)
    return str(candidates[0])


@dataclass
class DatabaseArtifacts:
    db_id: str
    db_path: str
    lsh_index: LSHIndex
    faiss_index: FAISSIndex
    example_store: ExampleStore
    schemas: FormattedSchemas


def _load_db_artifacts(db_id: str, db_path: str, preprocessed_dir: str) -> DatabaseArtifacts:
    """
    Load the pre-built offline artifacts for a single database.

    Assumes the offline pipeline has already been run and all artifact files
    exist on disk under preprocessed_dir.
    """
    root = Path(preprocessed_dir)
    lsh_path = root / "indices" / f"{db_id}_lsh.pkl"
    faiss_index_path = root / "indices" / f"{db_id}_faiss.index"
    faiss_fields_path = root / "indices" / f"{db_id}_faiss_fields.json"
    ddl_path = root / "schemas" / f"{db_id}_ddl.sql"
    md_path = root / "schemas" / f"{db_id}_markdown.md"
    ex_faiss_path = root / "indices" / "example_store.faiss"
    ex_meta_path = root / "indices" / "example_store_metadata.json"

    lsh_index = LSHIndex.load(str(lsh_path))
    faiss_index = FAISSIndex.load(str(faiss_index_path), str(faiss_fields_path))

    full_ddl = ddl_path.read_text(encoding="utf-8")
    full_markdown = md_path.read_text(encoding="utf-8")
    schemas = FormattedSchemas(db_id=db_id, ddl=full_ddl, markdown=full_markdown)

    if ex_faiss_path.exists() and ex_meta_path.exists():
        example_store = ExampleStore.load(str(ex_faiss_path), str(ex_meta_path))
    else:
        logger.warning(
            "Example store not found at %s — using empty store",
            ex_faiss_path,
        )
        example_store = ExampleStore()

    return DatabaseArtifacts(
        db_id=db_id,
        db_path=db_path,
        lsh_index=lsh_index,
        faiss_index=faiss_index,
        example_store=example_store,
        schemas=schemas,
    )


# ---------------------------------------------------------------------------
# Thin OfflineArtifacts wrapper (satisfies answer_question's type expectations)
# ---------------------------------------------------------------------------

class _ArtifactsAdapter:
    """
    Adapts a DatabaseArtifacts to the interface expected by answer_question.

    answer_question reads:
      - artifacts.lsh_index
      - artifacts.faiss_index
      - artifacts.faiss_index._fields
      - artifacts.example_store
      - artifacts.schemas.ddl
      - artifacts.schemas.markdown
    """
    def __init__(self, da: DatabaseArtifacts):
        self.db_id = da.db_id
        self.lsh_index = da.lsh_index
        self.faiss_index = da.faiss_index
        self.example_store = da.example_store
        self.schemas = da.schemas
        self.profile = None
        self.summary = None


# ---------------------------------------------------------------------------
# Per-question evaluation coroutine
# ---------------------------------------------------------------------------

async def _evaluate_one(
    entry: BirdEntry,
    artifacts: _ArtifactsAdapter,
    db_path: str,
    semaphore: asyncio.Semaphore,
) -> EvaluationEntry:
    """Evaluate a single question and return an EvaluationEntry."""
    t0 = time.monotonic()

    async with semaphore:
        try:
            result: PipelineResult = await answer_question(entry, artifacts, db_path)
        except Exception as exc:
            logger.error(
                "answer_question failed for question_id=%d (%s): %s",
                entry.question_id,
                entry.db_id,
                exc,
                exc_info=True,
            )
            latency = time.monotonic() - t0
            return EvaluationEntry(
                question_id=entry.question_id,
                db_id=entry.db_id,
                difficulty=entry.difficulty,
                predicted_sql="",
                truth_sql=entry.SQL,
                correct=False,
                selection_method="error",
                winner_generator="",
                cluster_count=0,
                latency_seconds=latency,
            )

    latency = time.monotonic() - t0
    predicted_sql = result.final_sql
    truth_sql = entry.SQL

    # Execute predicted SQL
    pred_result: ExecutionResult = execute_sql(db_path, predicted_sql)
    gold_result: ExecutionResult = execute_sql(db_path, truth_sql)

    if pred_result.success and gold_result.success:
        correct = _results_match(pred_result.rows, gold_result.rows)
    else:
        correct = False

    # Determine winner generator from tournament_wins
    winner_gen = "fallback"
    if result.generator_wins:
        winner_gen = max(result.generator_wins, key=lambda g: result.generator_wins[g])

    return EvaluationEntry(
        question_id=entry.question_id,
        db_id=entry.db_id,
        difficulty=entry.difficulty,
        predicted_sql=predicted_sql,
        truth_sql=truth_sql,
        correct=correct,
        selection_method=result.selection_method,
        winner_generator=winner_gen,
        cluster_count=result.cluster_count,
        latency_seconds=latency,
    )


async def _evaluate_one_from_context(
    ctx: QuestionContext,
    semaphore: asyncio.Semaphore,
) -> EvaluationEntry:
    """Phase 2 evaluation: Op7-9 from a pre-built QuestionContext (no artifact access)."""
    t0 = time.monotonic()
    entry = ctx.entry

    async with semaphore:
        try:
            result: PipelineResult = await answer_from_context(ctx)
        except Exception as exc:
            logger.error(
                "answer_from_context failed for question_id=%d (%s): %s",
                entry.question_id,
                entry.db_id,
                exc,
                exc_info=True,
            )
            return EvaluationEntry(
                question_id=entry.question_id,
                db_id=entry.db_id,
                difficulty=entry.difficulty,
                predicted_sql="",
                truth_sql=entry.SQL,
                correct=False,
                selection_method="error",
                winner_generator="",
                cluster_count=0,
                latency_seconds=time.monotonic() - t0,
            )

    latency = time.monotonic() - t0
    predicted_sql = result.final_sql
    truth_sql = entry.SQL

    pred_result: ExecutionResult = execute_sql(ctx.db_path, predicted_sql)
    gold_result: ExecutionResult = execute_sql(ctx.db_path, truth_sql)

    if pred_result.success and gold_result.success:
        correct = _results_match(pred_result.rows, gold_result.rows)
    else:
        correct = False

    winner_gen = "fallback"
    if result.generator_wins:
        winner_gen = max(result.generator_wins, key=lambda g: result.generator_wins[g])

    return EvaluationEntry(
        question_id=entry.question_id,
        db_id=entry.db_id,
        difficulty=entry.difficulty,
        predicted_sql=predicted_sql,
        truth_sql=truth_sql,
        correct=correct,
        selection_method=result.selection_method,
        winner_generator=winner_gen,
        cluster_count=result.cluster_count,
        latency_seconds=latency,
    )


# ---------------------------------------------------------------------------
# Summary table printer
# ---------------------------------------------------------------------------

def _print_summary(results: list[EvaluationEntry]) -> None:
    """Print a summary table broken down by difficulty and database."""
    if not results:
        print("No results to summarize.")
        return

    total = len(results)
    correct_total = sum(1 for r in results if r.correct)
    ex_overall = correct_total / total * 100 if total else 0

    print(f"\n{'='*60}")
    print(f"OVERALL: {correct_total}/{total} = {ex_overall:.1f}% EX")
    print(f"{'='*60}")

    # By difficulty
    print("\n--- By Difficulty ---")
    for diff in ["simple", "moderate", "challenging"]:
        sub = [r for r in results if r.difficulty == diff]
        if sub:
            n_correct = sum(1 for r in sub if r.correct)
            print(f"  {diff:<12}: {n_correct:>4}/{len(sub):<4} = {n_correct/len(sub)*100:.1f}%")

    # By database
    print("\n--- By Database ---")
    db_ids = sorted({r.db_id for r in results})
    for db_id in db_ids:
        sub = [r for r in results if r.db_id == db_id]
        n_correct = sum(1 for r in sub if r.correct)
        avg_lat = sum(r.latency_seconds for r in sub) / len(sub)
        print(
            f"  {db_id:<30}: {n_correct:>4}/{len(sub):<4} = {n_correct/len(sub)*100:.1f}%"
            f"  (avg {avg_lat:.1f}s/q)"
        )

    # By selection method
    print("\n--- By Selection Method ---")
    for method in ["fast_path", "tournament", "fallback", "error"]:
        sub = [r for r in results if r.selection_method == method]
        if sub:
            n_correct = sum(1 for r in sub if r.correct)
            print(f"  {method:<12}: {n_correct:>4}/{len(sub):<4} = {n_correct/len(sub)*100:.1f}%")

    avg_lat_all = sum(r.latency_seconds for r in results) / total
    print(f"\nAverage latency: {avg_lat_all:.1f}s/question")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main async function
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace) -> None:
    split = args.split
    output_path = Path(args.output)
    preprocessed_dir = settings.preprocessed_dir
    n_workers = args.workers
    db_filter = args.db_filter
    resume_path = args.resume

    # Baseline memory measurement (after all imports + model loading)
    _mem_mib("startup baseline (imports + model loading done)")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load BIRD split
    logger.info("Loading BIRD split: %s", split)
    entries: list[BirdEntry] = load_bird_split(split, data_dir=settings.bird_data_dir)
    if not entries:
        print(f"ERROR: No entries found for split={split!r}. Check BIRD_DATA_DIR setting.")
        sys.exit(1)

    # Apply db_filter if specified
    if db_filter:
        entries = [e for e in entries if e.db_id == db_filter]
        if not entries:
            print(f"ERROR: No entries found for db_filter={db_filter!r}.")
            sys.exit(1)
        print(f"Filtered to db_id={db_filter!r}: {len(entries)} questions")

    # Load existing results if resuming
    done_ids: set[int] = set()
    existing_results: list[dict] = []
    if resume_path and Path(resume_path).exists():
        with open(resume_path, encoding="utf-8") as f:
            existing_results = json.load(f)
        done_ids = {r["question_id"] for r in existing_results}
        print(f"Resuming: {len(done_ids)} questions already done, {len(entries)-len(done_ids)} remaining")

    # Filter out already-done entries
    pending: list[BirdEntry] = [e for e in entries if e.question_id not in done_ids]

    if not pending:
        print("All questions already evaluated. Nothing to do.")
        all_results = [EvaluationEntry(**r) for r in existing_results]
        _print_summary(all_results)
        return

    # Group pending questions by db_id and determine which DBs are needed
    unique_dbs = sorted({e.db_id for e in pending})
    print(f"Found {len(unique_dbs)} databases to process ({len(pending)} questions total).")

    # Validate db paths and filter out DBs whose artifacts are missing
    db_paths: dict[str, str] = {}
    for db_id in unique_dbs:
        db_paths[db_id] = _db_path_for_split(split, db_id)

    pending_by_db: dict[str, list[BirdEntry]] = {}
    for entry in pending:
        pending_by_db.setdefault(entry.db_id, []).append(entry)

    # Count total valid tasks (after artifact pre-check below is done per-DB)
    n_total_expected = len(pending)

    # Build semaphore for concurrency control (within each DB batch)
    semaphore = asyncio.Semaphore(n_workers)

    # Shared accumulators
    n_correct = 0
    n_total = 0

    # Set up output file (write incrementally as JSON array)
    all_serialized: list[dict] = list(existing_results)

    def _save_results() -> None:
        with open(output_path, "w", encoding="utf-8") as fout:
            json.dump(all_serialized, fout, ensure_ascii=False, indent=2)

    if _HAS_TQDM:
        pbar = tqdm(total=n_total_expected, unit="q", desc="Evaluating")
    else:
        pbar = None

    print(f"\nEvaluating {len(pending)} questions ({split}) with {n_workers} workers …\n")

    # -----------------------------------------------------------------------
    # Per-DB batch loop — load artifacts, process, then free memory
    # -----------------------------------------------------------------------
    for db_id in sorted(pending_by_db.keys()):
        db_questions = pending_by_db[db_id]
        db_path = db_paths[db_id]

        # Load artifacts for this DB only
        print(f"  [{db_id}] Loading artifacts for {len(db_questions)} question(s) …")
        mem_before_load = _mem_mib(f"{db_id} before loading artifacts")
        try:
            da = _load_db_artifacts(db_id, db_path, preprocessed_dir)
            artifacts = _ArtifactsAdapter(da)
            mem_after_load = _mem_mib(f"{db_id} after loading artifacts (FAISS: {len(da.faiss_index._fields)} fields)")
            print(f"  [{db_id}] artifacts loaded — index memory added: {mem_after_load - mem_before_load:+.1f} MiB")
        except FileNotFoundError as exc:
            print(f"  [{db_id}] ERROR: artifacts not found — {exc}. Skipping {len(db_questions)} question(s).")
            if pbar is not None:
                pbar.update(len(db_questions))
            n_total += len(db_questions)
            continue

        # -------------------------------------------------------------------
        # Phase 1: Run Op5+6 (LSH + FAISS) for ALL questions in this batch.
        # Each call is gated by the semaphore to limit concurrency.
        # -------------------------------------------------------------------
        print(f"  [{db_id}] Phase 1: context grounding + schema linking …")

        async def _gated_prepare(entry: BirdEntry) -> Optional[QuestionContext]:
            async with semaphore:
                return await prepare_context(entry, artifacts, db_path)

        contexts: list[Optional[QuestionContext]] = await asyncio.gather(
            *[_gated_prepare(entry) for entry in db_questions]
        )

        # -------------------------------------------------------------------
        # Free heavy indexes — lsh_index and faiss_index are no longer needed.
        # example_store is also index-backed; free it too.
        # Only schemas (small text strings) need to stay until after Phase 2.
        #
        # Strategy: grab direct references into local variables, clear ALL
        # attribute references (both `da` and `artifacts` hold separate refs
        # to the same underlying objects), then `del` the local vars so the
        # refcount drops to zero and Python immediately frees the objects.
        # Finally, ask the OS allocator to return the freed pages now — before
        # Phase 2 starts — so Activity Monitor shows the drop during LLM calls.
        # -------------------------------------------------------------------
        mem_before_free = _mem_mib(f"{db_id} before freeing indexes")
        _lsh = da.lsh_index
        _faiss = da.faiss_index
        _ex = da.example_store
        da.lsh_index = None
        da.faiss_index = None
        da.example_store = None
        artifacts.lsh_index = None
        artifacts.faiss_index = None
        artifacts.example_store = None
        del _lsh, _faiss, _ex
        gc.collect()
        _malloc_trim()  # return freed pages to OS NOW, before Phase 2 LLM calls
        mem_after_free = _mem_mib(f"{db_id} after freeing indexes + gc.collect() + malloc_trim")
        print(
            f"  [{db_id}] Index memory freed: {mem_after_free - mem_before_free:+.1f} MiB "
            f"(before={mem_before_free:.1f}, after={mem_after_free:.1f}). "
            f"Phase 2: SQL generation + selection …"
        )

        # -------------------------------------------------------------------
        # Phase 2: Run Op7-9 (LLM + SQLite) using the pre-built contexts.
        # No artifact index access happens here.
        # -------------------------------------------------------------------
        def _record(ev: EvaluationEntry) -> None:
            nonlocal n_total, n_correct
            n_total += 1
            if ev.correct:
                n_correct += 1
            all_serialized.append(asdict(ev))
            _save_results()
            ex_pct = n_correct / n_total * 100
            if pbar is not None:
                pbar.set_postfix(EX=f"{ex_pct:.1f}%", correct=n_correct)
                pbar.update(1)
            else:
                print(
                    f"[{n_total}/{n_total_expected}] Q#{ev.question_id} "
                    f"({ev.db_id}/{ev.difficulty}) "
                    f"correct={ev.correct} method={ev.selection_method} "
                    f"lat={ev.latency_seconds:.1f}s  EX={ex_pct:.1f}%"
                )

        phase2_tasks = []
        for entry, ctx in zip(db_questions, contexts):
            if ctx is None:
                # Phase 1 failed for this entry — record as error immediately
                _record(EvaluationEntry(
                    question_id=entry.question_id,
                    db_id=entry.db_id,
                    difficulty=entry.difficulty,
                    predicted_sql="",
                    truth_sql=entry.SQL,
                    correct=False,
                    selection_method="error",
                    winner_generator="",
                    cluster_count=0,
                    latency_seconds=0.0,
                ))
            else:
                phase2_tasks.append(_evaluate_one_from_context(ctx, semaphore))

        for coro in asyncio.as_completed(phase2_tasks):
            ev: EvaluationEntry = await coro
            _record(ev)

        # Free remaining objects and reset tracker
        del contexts, phase2_tasks, da, artifacts
        gc.collect()
        _malloc_trim()
        get_tracker().reset()

        mem_end = _mem_mib(f"{db_id} after Phase 2 complete + final gc")
        print(f"  [{db_id}] Done. Memory released.")

    if pbar is not None:
        pbar.close()

    # Final summary
    all_eval = [EvaluationEntry(**r) for r in all_serialized]
    _print_summary(all_eval)
    print(f"Results saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NL2SQL online pipeline evaluation against a BIRD split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split",
        choices=["dev", "mini_dev", "train"],
        default="dev",
        help="BIRD split to evaluate.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Path to write results JSON. "
            "Defaults to results/{split}_results.json."
        ),
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="PATH",
        help="Path to an existing results JSON file to resume from.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent questions processed in parallel.",
    )
    parser.add_argument(
        "--db_filter",
        default=None,
        metavar="DB_ID",
        help="If set, only evaluate questions from this database.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.output is None:
        args.output = f"results/{args.split}_results.json"
    asyncio.run(main(args))
