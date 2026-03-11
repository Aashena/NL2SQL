#!/usr/bin/env python3
"""
Test the three verification LLM calls in isolation.

Picks a random BIRD dev question (seeded or truly random), builds a basic
schema string directly from the SQLite PRAGMA metadata, then calls each of
the three QueryVerifier LLM methods individually and prints the full output.

Three calls under test
----------------------
  1. _generate_grain_spec        — model_fast  + grain_verification tool
  2. _generate_column_alignment_spec — model_fast  + column_alignment_spec tool
  3. _generate_optional_tests    — model_powerful + verification_plan tool

Purpose: verify that the designed prompts and tool schemas produce valid
structured outputs without MALFORMED_FUNCTION_CALL or other errors.

Usage
-----
  # Random question (different each run)
  python scripts/test_verification_llm_calls.py

  # Reproducible (fixed seed)
  python scripts/test_verification_llm_calls.py --seed 42

  # Use a specific question ID
  python scripts/test_verification_llm_calls.py --question_id 512

  # Override which BIRD split to sample from
  python scripts/test_verification_llm_calls.py --split mini_dev
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sqlite3
import sys
from pathlib import Path
from typing import Optional

# Make sure repo root is on the path when run as a script
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.config.settings import settings
from src.data.bird_loader import BirdEntry, load_bird_split
from src.verification.query_verifier import QueryVerifier

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
# Silence noisy third-party loggers
for _noisy in ("httpx", "httpcore", "anthropic", "google", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logger = logging.getLogger("test_verification_llm_calls")

# ---------------------------------------------------------------------------
# Schema builder (no offline preprocessing required)
# ---------------------------------------------------------------------------

def _build_schema_text(db_path: str) -> str:
    """Build a minimal DDL-like schema string from PRAGMA queries on the db."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [row[0] for row in cursor.fetchall()]

    lines: list[str] = []
    for table in tables:
        cursor.execute(f"PRAGMA table_info([{table}]);")
        cols = cursor.fetchall()  # cid, name, type, notnull, dflt_value, pk
        cursor.execute(f"PRAGMA foreign_key_list([{table}]);")
        fks = cursor.fetchall()

        col_defs = [f"  {c[1]} {c[2]}" + (" NOT NULL" if c[3] else "") + (" PRIMARY KEY" if c[5] else "")
                    for c in cols]

        lines.append(f"CREATE TABLE {table} (")
        lines.extend([d + "," for d in col_defs[:-1]])
        lines.append(col_defs[-1] if col_defs else "")
        if fks:
            for fk in fks:
                # (id, seq, table, from, to, on_update, on_delete, match)
                lines.append(f"  -- FK: {fk[3]} -> {fk[2]}.{fk[4]}")
        lines.append(");")
        lines.append("")

    conn.close()
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Question picker
# ---------------------------------------------------------------------------

def _pick_question(
    split: str,
    question_id: Optional[int],
    seed: Optional[int],
) -> BirdEntry:
    """Load BIRD split and return one entry."""
    data_dir = settings.bird_data_dir
    entries = load_bird_split(split, data_dir)
    if not entries:
        raise SystemExit(
            f"No entries found in split '{split}' under '{data_dir}'.\n"
            "Make sure the BIRD dataset is downloaded."
        )

    if question_id is not None:
        matches = [e for e in entries if e.question_id == question_id]
        if not matches:
            raise SystemExit(f"question_id={question_id} not found in split '{split}'.")
        return matches[0]

    rng = random.Random(seed)  # None seed → system randomness
    return rng.choice(entries)


# ---------------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------------

def _sep(title: str, width: int = 72) -> None:
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _print_spec(spec) -> None:
    """Print a VerificationTestSpec as formatted JSON."""
    print(json.dumps(spec.model_dump(exclude_none=True), indent=2))


# ---------------------------------------------------------------------------
# Main async runner
# ---------------------------------------------------------------------------

async def run(args: argparse.Namespace) -> None:
    entry = _pick_question(args.split, args.question_id, args.seed)
    # BIRD dev uses "dev_databases"; train uses "train_databases"; mini_dev uses "dev_databases"
    db_subdir = f"{args.split}_databases" if args.split != "mini_dev" else "dev_databases"
    db_path = str(
        Path(settings.bird_data_dir) / args.split / db_subdir / entry.db_id / f"{entry.db_id}.sqlite"
    )
    if not Path(db_path).exists():
        raise SystemExit(f"Database file not found: {db_path}")

    schema_text = _build_schema_text(db_path)

    _sep("SELECTED QUESTION")
    print(f"  question_id : {entry.question_id}")
    print(f"  db_id       : {entry.db_id}")
    print(f"  difficulty  : {entry.difficulty}")
    print(f"  question    : {entry.question}")
    print(f"  evidence    : {entry.evidence or '(none)'}")
    print(f"  gold SQL    : {entry.SQL}")

    _sep("DATABASE SCHEMA (truncated to 3 000 chars passed to LLM)")
    print(schema_text[:3000])

    print()
    print(f"  LLM provider : {settings.llm_provider}")
    print(f"  model_fast   : {settings.model_fast}")
    print(f"  model_powerful: {settings.model_powerful}")

    verifier = QueryVerifier()

    # ------------------------------------------------------------------
    # Call 1: grain test
    # ------------------------------------------------------------------
    _sep("CALL 1 — _generate_grain_spec  (model_fast + grain_verification tool)")
    print("Running...")
    try:
        grain_spec = await verifier._generate_grain_spec(
            entry.question, entry.evidence, schema_text
        )
        print("  STATUS: SUCCESS — no errors")
        _print_spec(grain_spec)
    except Exception as exc:
        print(f"  STATUS: ERROR — {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------
    # Call 2: column alignment
    # ------------------------------------------------------------------
    _sep("CALL 2 — _generate_column_alignment_spec  (model_fast + column_alignment_spec tool)")
    print("Running...")
    try:
        col_spec = await verifier._generate_column_alignment_spec(
            entry.question, entry.evidence, schema_text
        )
        print("  STATUS: SUCCESS — no errors")
        _print_spec(col_spec)
    except Exception as exc:
        print(f"  STATUS: ERROR — {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------
    # Call 3: optional tests
    # ------------------------------------------------------------------
    _sep("CALL 3 — _generate_optional_tests  (model_powerful + verification_plan tool)")
    print("Running...")
    try:
        optional_specs = await verifier._generate_optional_tests(
            entry.question, entry.evidence, schema_text
        )
        print(f"  STATUS: SUCCESS — no errors  ({len(optional_specs)} optional test(s) returned)")
        if optional_specs:
            for spec in optional_specs:
                print(f"\n  --- test_type: {spec.test_type} ---")
                _print_spec(spec)
        else:
            print("  (no optional tests applicable for this question)")
    except Exception as exc:
        print(f"  STATUS: ERROR — {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------
    # Summary via generate_plan (all three calls concurrently)
    # ------------------------------------------------------------------
    _sep("BONUS — generate_plan()  (all three calls concurrently)")
    print("Running all three in parallel via asyncio.gather()...")
    try:
        all_specs = await verifier.generate_plan(
            entry.question, entry.evidence, schema_text
        )
        print(f"  STATUS: SUCCESS — {len(all_specs)} total spec(s)")
        for spec in all_specs:
            print(f"\n  --- test_type: {spec.test_type} ---")
            _print_spec(spec)
    except Exception as exc:
        print(f"  STATUS: ERROR — {type(exc).__name__}: {exc}")

    print()
    print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test the three verification LLM calls on a random BIRD question."
    )
    parser.add_argument(
        "--split",
        default="dev",
        choices=["dev", "train", "mini_dev"],
        help="BIRD split to sample from (default: dev).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible question selection (default: no seed → different each run).",
    )
    parser.add_argument(
        "--question_id",
        type=int,
        default=None,
        help="Pick a specific question by its question_id instead of sampling randomly.",
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
