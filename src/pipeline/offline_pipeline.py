"""
Offline Pipeline (Ops 0 + 1)

Orchestrates the full offline preprocessing pipeline for a single database:
  Op 0a: Statistical profiling (profile_database)
  Op 0b: LLM field summarization (summarize_database)
  Op 0c: Schema formatting (format_and_save_schemas)
  Op 1a: LSH index build + save
  Op 1b: FAISS index build + save
  Op 1c: Example store build + save (shared across all DBs)

Results are cached to disk under `preprocessed_dir`:
  profiles/{db_id}.json
  summaries/{db_id}.json
  schemas/{db_id}_ddl.sql
  schemas/{db_id}_markdown.md
  indices/{db_id}_lsh.pkl
  indices/{db_id}_faiss.index
  indices/{db_id}_faiss_fields.json
  indices/example_store.faiss          (shared)
  indices/example_store_metadata.json  (shared)

If all per-DB artifacts exist and `force=False`, the pipeline short-circuits
and loads from disk without re-running any computation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.data.bird_loader import BirdEntry

from src.config.settings import settings
from src.preprocessing.profiler import profile_database, DatabaseProfile
from src.preprocessing.summarizer import summarize_database, DatabaseSummary
from src.preprocessing.schema_formatter import format_and_save_schemas, FormattedSchemas
from src.indexing.lsh_index import LSHIndex
from src.indexing.faiss_index import FAISSIndex
from src.indexing.example_store import ExampleStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class OfflineArtifacts:
    db_id: str
    profile: DatabaseProfile
    summary: DatabaseSummary
    schemas: FormattedSchemas
    lsh_index: LSHIndex
    faiss_index: FAISSIndex
    example_store: ExampleStore


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _artifact_paths(preprocessed_dir: str, db_id: str) -> dict[str, Path]:
    """Return a dict of all expected artifact paths for a given db_id."""
    root = Path(preprocessed_dir)
    return {
        "profile":       root / "profiles"  / f"{db_id}.json",
        "summary":       root / "summaries" / f"{db_id}.json",
        "ddl":           root / "schemas"   / f"{db_id}_ddl.sql",
        "markdown":      root / "schemas"   / f"{db_id}_markdown.md",
        "lsh":           root / "indices"   / f"{db_id}_lsh.pkl",
        "faiss_index":   root / "indices"   / f"{db_id}_faiss.index",
        "faiss_fields":  root / "indices"   / f"{db_id}_faiss_fields.json",
    }


def _example_store_paths(preprocessed_dir: str) -> tuple[Path, Path]:
    """Return (faiss_path, meta_path) for the shared example store."""
    root = Path(preprocessed_dir)
    return (
        root / "indices" / "example_store.faiss",
        root / "indices" / "example_store_metadata.json",
    )


def _all_per_db_artifacts_exist(paths: dict[str, Path]) -> bool:
    """Return True if all per-DB artifact files exist on disk."""
    per_db_keys = ["profile", "summary", "ddl", "lsh", "faiss_index", "faiss_fields"]
    return all(paths[k].exists() for k in per_db_keys)


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------

async def run_offline_pipeline(
    db_id: str,
    db_path: str,
    train_data: "list[BirdEntry]",
    preprocessed_dir: Optional[str] = None,
    force: bool = False,
) -> OfflineArtifacts:
    """
    Run (or load from cache) the full offline pipeline for one database.

    Parameters
    ----------
    db_id:
        Logical database identifier (e.g. "california_schools").
    db_path:
        Filesystem path to the SQLite database file.
    train_data:
        Training BirdEntry objects for the example store.
    preprocessed_dir:
        Root output directory. Defaults to `settings.preprocessed_dir`.
    force:
        If True, re-run all steps even if cached artifacts exist.

    Returns
    -------
    OfflineArtifacts with all fields populated.

    Raises
    ------
    FileNotFoundError:
        If `db_path` does not exist.
    """
    # Validate db_path first (before any other work)
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    if preprocessed_dir is None:
        preprocessed_dir = settings.preprocessed_dir

    # Ensure all output subdirectories exist
    root = Path(preprocessed_dir)
    for subdir in ["profiles", "summaries", "schemas", "indices"]:
        (root / subdir).mkdir(parents=True, exist_ok=True)

    paths = _artifact_paths(preprocessed_dir, db_id)
    ex_faiss_path, ex_meta_path = _example_store_paths(preprocessed_dir)

    # ------------------------------------------------------------------
    # Short-circuit: load everything from disk if cache is complete
    # ------------------------------------------------------------------
    if not force and _all_per_db_artifacts_exist(paths):
        logger.info("[%s] All artifacts cached — loading from disk", db_id)

        from src.preprocessing.profiler import _load_profile_from_json
        from src.preprocessing.summarizer import _load_summary_from_json
        from src.preprocessing.schema_formatter import format_schemas

        profile = _load_profile_from_json(paths["profile"])
        summary = _load_summary_from_json(paths["summary"])
        schemas = format_schemas(profile, summary)

        lsh_index = LSHIndex.load(str(paths["lsh"]))
        faiss_index = FAISSIndex.load(str(paths["faiss_index"]), str(paths["faiss_fields"]))

        # Load or (re)build example store
        if ex_faiss_path.exists() and ex_meta_path.exists():
            example_store = ExampleStore.load(str(ex_faiss_path), str(ex_meta_path))
        else:
            example_store = _build_example_store(train_data, str(ex_faiss_path), str(ex_meta_path))

        return OfflineArtifacts(
            db_id=db_id,
            profile=profile,
            summary=summary,
            schemas=schemas,
            lsh_index=lsh_index,
            faiss_index=faiss_index,
            example_store=example_store,
        )

    # ------------------------------------------------------------------
    # Op 0a: Statistical profiling
    # ------------------------------------------------------------------
    logger.info("[%s] Op 0a: Profiling database …", db_id)
    profile = profile_database(
        db_path=db_path,
        db_id=db_id,
        output_dir=str(root / "profiles"),
        force=force,
    )

    # ------------------------------------------------------------------
    # Op 0b: LLM field summarization
    # ------------------------------------------------------------------
    logger.info("[%s] Op 0b: Summarizing fields …", db_id)
    summary = await summarize_database(
        profile=profile,
        output_dir=str(root / "summaries"),
    )
    # Ensure the summary is persisted to disk even if summarize_database
    # doesn't write it (e.g. when the output_dir arg is not honoured by a mock).
    summary_path = root / "summaries" / f"{db_id}.json"
    if not summary_path.exists():
        from src.preprocessing.summarizer import _save_summary_to_json
        _save_summary_to_json(summary, summary_path)

    # ------------------------------------------------------------------
    # Op 0c: Schema formatting
    # ------------------------------------------------------------------
    logger.info("[%s] Op 0c: Formatting schemas …", db_id)
    schemas = format_and_save_schemas(
        profile=profile,
        summary=summary,
        output_dir=str(root / "schemas"),
    )

    # ------------------------------------------------------------------
    # Op 1a: LSH index
    # ------------------------------------------------------------------
    logger.info("[%s] Op 1a: Building LSH index …", db_id)
    lsh_index = LSHIndex()
    lsh_index.build(db_path=db_path, db_id=db_id)
    lsh_index.save(str(paths["lsh"]))

    # ------------------------------------------------------------------
    # Op 1b: FAISS index
    # ------------------------------------------------------------------
    logger.info("[%s] Op 1b: Building FAISS index …", db_id)
    faiss_index = FAISSIndex()
    faiss_index.build(summary.field_summaries)
    faiss_index.save(str(paths["faiss_index"]), str(paths["faiss_fields"]))

    # ------------------------------------------------------------------
    # Op 1c: Example store (shared — only build once)
    # ------------------------------------------------------------------
    if ex_faiss_path.exists() and ex_meta_path.exists() and not force:
        logger.info("Example store cached — loading from disk")
        example_store = ExampleStore.load(str(ex_faiss_path), str(ex_meta_path))
    else:
        example_store = _build_example_store(train_data, str(ex_faiss_path), str(ex_meta_path))

    return OfflineArtifacts(
        db_id=db_id,
        profile=profile,
        summary=summary,
        schemas=schemas,
        lsh_index=lsh_index,
        faiss_index=faiss_index,
        example_store=example_store,
    )


def _build_example_store(
    train_data: "list[BirdEntry]",
    faiss_path: str,
    meta_path: str,
) -> ExampleStore:
    """Build and save the shared example store from training data."""
    logger.info("Op 1c: Building example store from %d entries …", len(train_data))
    example_store = ExampleStore()
    if train_data:
        example_store.build(train_data)
        example_store.save(faiss_path, meta_path)
    else:
        logger.warning("No training data — example store will be empty")
    return example_store
