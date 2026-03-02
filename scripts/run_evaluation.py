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
import json
import logging
import os
import re
import sys
import time
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
from src.pipeline.online_pipeline import answer_question, PipelineResult, PIPELINE_SEMAPHORE
from src.preprocessing.schema_formatter import FormattedSchemas

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

    # Load artifacts per database (one set per unique db_id)
    unique_dbs = sorted({e.db_id for e in pending})
    print(f"Loading artifacts for {len(unique_dbs)} databases …")
    db_artifacts: dict[str, _ArtifactsAdapter] = {}
    db_paths: dict[str, str] = {}

    for db_id in unique_dbs:
        db_path = _db_path_for_split(split, db_id)
        db_paths[db_id] = db_path
        try:
            da = _load_db_artifacts(db_id, db_path, preprocessed_dir)
            db_artifacts[db_id] = _ArtifactsAdapter(da)
            print(f"  [{db_id}] artifacts loaded (FAISS: {len(da.faiss_index._fields)} fields)")
        except FileNotFoundError as exc:
            print(f"  [{db_id}] ERROR: artifacts not found — {exc}")
            # Skip entries for this DB
            pending = [e for e in pending if e.db_id != db_id]

    if not pending:
        print("No questions to evaluate after artifact loading. Exiting.")
        sys.exit(1)

    print(f"\nEvaluating {len(pending)} questions ({split}) with {n_workers} workers …\n")

    # Build semaphore for concurrency control
    semaphore = asyncio.Semaphore(n_workers)

    # Collect completed results
    completed_results: list[EvaluationEntry] = []
    n_correct = 0
    n_total = 0

    # Set up output file (write incrementally as JSON array)
    # We maintain the full list in memory and write after each result.
    all_serialized: list[dict] = list(existing_results)

    def _save_results():
        with open(output_path, "w", encoding="utf-8") as fout:
            json.dump(all_serialized, fout, ensure_ascii=False, indent=2)

    # Build coroutine list
    tasks = []
    for entry in pending:
        if entry.db_id not in db_artifacts:
            continue  # skip entries with missing artifacts
        tasks.append(
            _evaluate_one(
                entry=entry,
                artifacts=db_artifacts[entry.db_id],
                db_path=db_paths[entry.db_id],
                semaphore=semaphore,
            )
        )

    if _HAS_TQDM:
        pbar = tqdm(total=len(tasks), unit="q", desc="Evaluating")
    else:
        pbar = None

    for coro in asyncio.as_completed(tasks):
        ev: EvaluationEntry = await coro
        completed_results.append(ev)
        n_total += 1
        if ev.correct:
            n_correct += 1

        # Save incrementally
        all_serialized.append(asdict(ev))
        _save_results()

        ex_pct = n_correct / n_total * 100 if n_total else 0
        if pbar is not None:
            pbar.set_postfix(EX=f"{ex_pct:.1f}%", correct=n_correct)
            pbar.update(1)
        else:
            print(
                f"[{n_total}/{len(tasks)}] Q#{ev.question_id} "
                f"({ev.db_id}/{ev.difficulty}) "
                f"correct={ev.correct} method={ev.selection_method} "
                f"lat={ev.latency_seconds:.1f}s  EX={ex_pct:.1f}%"
            )

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
