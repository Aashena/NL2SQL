#!/usr/bin/env python3
"""
Run offline preprocessing (Ops 0 + 1) for BIRD databases.

This script runs the three preparatory steps (profile → summarize → format) for
every database in the requested split.  Index construction (Op 1) will be added
in Prompt 6.

Usage:
    python scripts/run_offline_preprocessing.py --split dev --step all
    python scripts/run_offline_preprocessing.py --split dev --step profile
    python scripts/run_offline_preprocessing.py --split dev --step format --db california_schools
    python scripts/run_offline_preprocessing.py --split train --step summarize
    python scripts/run_offline_preprocessing.py --split mini_dev --step all

Flags:
    --split   dev | train | mini_dev   Which BIRD split to process.
    --step    all | profile | summarize | format | indices
              Which processing step(s) to run.  "all" runs profile → summarize
              → format in sequence (indices will be added in Prompt 6).
    --db      (optional) Process only a single database by its db_id.
    --force   Re-run even if cached output already exists.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on the path when the script is run directly
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config.settings import settings  # noqa: E402  (must be after sys.path fix)
from src.preprocessing.profiler import profile_database  # noqa: E402
from src.preprocessing.summarizer import summarize_database  # noqa: E402
from src.preprocessing.schema_formatter import format_and_save_schemas  # noqa: E402

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("offline_preprocessing")


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

# Map split name → subdirectory name inside data/bird/
_SPLIT_DIR_MAP: dict[str, str] = {
    "dev": "dev",
    "train": "train",
    "mini_dev": "mini_dev",
}

# Map split name → databases sub-directory name pattern
_DB_SUBDIR_PATTERNS: dict[str, list[str]] = {
    "dev":      ["dev_databases", "databases"],
    "train":    ["train_databases", "databases"],
    "mini_dev": ["mini_dev_databases", "dev_databases", "databases"],
}


def _find_databases_dir(split_dir: Path) -> Path:
    """Locate the directory that contains per-database sub-directories."""
    patterns = _DB_SUBDIR_PATTERNS.get(split_dir.name, ["databases"])
    for pattern in patterns:
        candidate = split_dir / pattern
        if candidate.is_dir():
            return candidate
    # Fallback: look for any immediate child that is itself a directory of
    # directories (i.e. the databases are nested one level deep)
    for child in sorted(split_dir.iterdir()):
        if child.is_dir() and any(c.is_dir() for c in child.iterdir()):
            log.debug("Using databases directory: %s", child)
            return child
    raise FileNotFoundError(
        f"Cannot locate databases directory inside {split_dir}. "
        "Expected one of: " + ", ".join(patterns)
    )


def _discover_databases(split: str, db_filter: str | None) -> list[tuple[str, Path]]:
    """
    Return list of (db_id, sqlite_path) for the requested split.

    Parameters
    ----------
    split:
        One of "dev", "train", "mini_dev".
    db_filter:
        If provided, return only the matching db_id.
    """
    bird_root = Path(settings.bird_data_dir)
    split_subdir = _SPLIT_DIR_MAP.get(split)
    if split_subdir is None:
        raise ValueError(f"Unknown split: {split!r}. Choose from {list(_SPLIT_DIR_MAP)}")

    split_dir = bird_root / split_subdir
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"Split directory not found: {split_dir}. "
            "Make sure the BIRD dataset is downloaded."
        )

    databases_dir = _find_databases_dir(split_dir)

    results: list[tuple[str, Path]] = []
    for db_dir in sorted(databases_dir.iterdir()):
        if not db_dir.is_dir():
            continue
        db_id = db_dir.name
        if db_filter and db_id != db_filter:
            continue

        # Look for a .sqlite or .db file inside the database directory
        sqlite_path: Path | None = None
        for suffix in ("*.sqlite", "*.db"):
            matches = list(db_dir.glob(suffix))
            if matches:
                sqlite_path = matches[0]
                break

        if sqlite_path is None:
            log.warning("No SQLite file found in %s — skipping", db_dir)
            continue

        results.append((db_id, sqlite_path))

    if not results:
        msg = f"No databases found for split={split!r}"
        if db_filter:
            msg += f" (filter: db={db_filter!r})"
        raise FileNotFoundError(msg)

    return results


# ---------------------------------------------------------------------------
# Output directory helpers
# ---------------------------------------------------------------------------

def _output_dirs(preprocessed_root: str) -> dict[str, Path]:
    root = Path(preprocessed_root)
    dirs = {
        "profiles":  root / "profiles",
        "summaries": root / "summaries",
        "schemas":   root / "schemas",
        "indices":   root / "indices",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# ---------------------------------------------------------------------------
# Per-database step runners
# ---------------------------------------------------------------------------

def _run_profile(db_id: str, sqlite_path: Path, dirs: dict[str, Path], force: bool) -> None:
    """Op 0a: Statistical profiling."""
    cache_file = dirs["profiles"] / f"{db_id}.json"
    if not force and cache_file.exists():
        log.debug("[%s] profile cached — skipping", db_id)
        return
    log.info("[%s] Profiling …", db_id)
    profile_database(
        db_path=str(sqlite_path),
        db_id=db_id,
        output_dir=str(dirs["profiles"]),
        force=force,
    )


def _run_summarize(db_id: str, dirs: dict[str, Path], force: bool) -> None:
    """Op 0b: LLM field summarization."""
    profile_cache = dirs["profiles"] / f"{db_id}.json"
    if not profile_cache.exists():
        raise RuntimeError(
            f"Profile for {db_id!r} not found. Run the 'profile' step first."
        )
    summary_cache = dirs["summaries"] / f"{db_id}.json"
    if not force and summary_cache.exists():
        log.debug("[%s] summary cached — skipping", db_id)
        return
    log.info("[%s] Summarizing …", db_id)

    from src.preprocessing.profiler import _load_profile_from_json  # local import
    profile = _load_profile_from_json(profile_cache)
    summarize_database(profile, output_dir=str(dirs["summaries"]))


def _run_format(db_id: str, dirs: dict[str, Path], force: bool) -> None:
    """Op 0c: Schema formatting (DDL + Markdown)."""
    profile_cache = dirs["profiles"] / f"{db_id}.json"
    summary_cache = dirs["summaries"] / f"{db_id}.json"

    if not profile_cache.exists():
        raise RuntimeError(
            f"Profile for {db_id!r} not found. Run the 'profile' step first."
        )
    if not summary_cache.exists():
        raise RuntimeError(
            f"Summary for {db_id!r} not found. Run the 'summarize' step first."
        )

    ddl_file = dirs["schemas"] / f"{db_id}_ddl.sql"
    md_file = dirs["schemas"] / f"{db_id}_markdown.md"
    if not force and ddl_file.exists() and md_file.exists():
        log.debug("[%s] schemas cached — skipping", db_id)
        return

    log.info("[%s] Formatting schemas …", db_id)

    from src.preprocessing.profiler import _load_profile_from_json  # local import
    from src.preprocessing.summarizer import _load_summary_from_json  # local import
    profile = _load_profile_from_json(profile_cache)
    summary = _load_summary_from_json(summary_cache)
    format_and_save_schemas(profile, summary, output_dir=str(dirs["schemas"]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run offline preprocessing (Ops 0+1) for BIRD databases.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=["dev", "train", "mini_dev"],
        help="Which BIRD dataset split to process.",
    )
    parser.add_argument(
        "--step",
        required=True,
        choices=["all", "profile", "summarize", "format", "indices"],
        help=(
            "Processing step to run.  'all' runs profile → summarize → format. "
            "'indices' is a placeholder (implemented in Prompt 6)."
        ),
    )
    parser.add_argument(
        "--db",
        default=None,
        metavar="DB_ID",
        help="Process only this specific database (optional).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-run step even if cached output exists.",
    )
    return parser.parse_args()


def main() -> int:  # noqa: C901  (complexity OK for a CLI entry-point)
    args = _parse_args()

    # Validate step
    if args.step == "indices":
        log.error(
            "The 'indices' step is not yet implemented (it will be added in Prompt 6). "
            "Use --step all | profile | summarize | format."
        )
        return 1

    # Determine which steps to run
    if args.step == "all":
        steps = ["profile", "summarize", "format"]
    else:
        steps = [args.step]

    # Discover databases
    try:
        databases = _discover_databases(args.split, args.db)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1

    log.info(
        "Processing %d database(s) for split=%r, steps=%s",
        len(databases),
        args.split,
        steps,
    )

    dirs = _output_dirs(settings.preprocessed_dir)

    n_processed = 0
    n_skipped = 0
    n_failed = 0

    # Wrap with tqdm if available
    db_iter = databases
    if _HAS_TQDM:
        db_iter = _tqdm(databases, desc="Databases", unit="db")

    for db_id, sqlite_path in db_iter:
        if _HAS_TQDM:
            db_iter.set_postfix(db=db_id)  # type: ignore[union-attr]

        t0 = time.perf_counter()
        try:
            ran_any = False
            for step in steps:
                if step == "profile":
                    cache_file = dirs["profiles"] / f"{db_id}.json"
                    if not args.force and cache_file.exists():
                        continue
                    _run_profile(db_id, sqlite_path, dirs, force=args.force)
                    ran_any = True

                elif step == "summarize":
                    cache_file = dirs["summaries"] / f"{db_id}.json"
                    if not args.force and cache_file.exists():
                        continue
                    _run_summarize(db_id, dirs, force=args.force)
                    ran_any = True

                elif step == "format":
                    ddl_file = dirs["schemas"] / f"{db_id}_ddl.sql"
                    md_file = dirs["schemas"] / f"{db_id}_markdown.md"
                    if not args.force and ddl_file.exists() and md_file.exists():
                        continue
                    _run_format(db_id, dirs, force=args.force)
                    ran_any = True

            elapsed = time.perf_counter() - t0
            if ran_any:
                log.info("[%s] Done in %.1fs", db_id, elapsed)
                n_processed += 1
            else:
                log.debug("[%s] All steps cached — skipped", db_id)
                n_skipped += 1

        except Exception as exc:  # noqa: BLE001
            log.error("[%s] FAILED: %s", db_id, exc, exc_info=True)
            n_failed += 1

    # Summary
    print(
        f"\n--- Preprocessing complete ---\n"
        f"  Processed : {n_processed}\n"
        f"  Skipped   : {n_skipped} (cached)\n"
        f"  Failed    : {n_failed}\n"
    )

    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
