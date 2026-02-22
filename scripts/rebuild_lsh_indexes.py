"""
Rebuild LSH indexes for all BIRD dev databases using the updated LSHIndex
that filters out INTEGER and REAL columns (text-only fix).

Run from the project root:
    python scripts/rebuild_lsh_indexes.py
"""

import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indexing.lsh_index import LSHIndex

DB_ROOT = Path("data/bird/dev_20240627/dev_databases")
INDEX_DIR = Path("data/preprocessed/indices")

DATABASES = [
    "california_schools",
    "card_games",
    "codebase_community",
    "debit_card_specializing",
    "european_football_2",
    "financial",
    "formula_1",
    "student_club",
    "superhero",
    "thrombosis_prediction",
    "toxicology",
]


def fmt_size(bytes_val: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} TB"


def main() -> None:
    total_before = 0
    total_after = 0
    results = []

    print(f"{'Database':<30} {'Before':>8} {'After':>8} {'Saved':>8} {'Time':>8}")
    print("-" * 66)

    for db_id in DATABASES:
        db_path = DB_ROOT / db_id / f"{db_id}.sqlite"
        pkl_path = INDEX_DIR / f"{db_id}_lsh.pkl"

        if not db_path.exists():
            print(f"{db_id:<30} SKIPPED — SQLite not found")
            continue

        before_bytes = pkl_path.stat().st_size if pkl_path.exists() else 0
        total_before += before_bytes

        idx = LSHIndex()
        t0 = time.monotonic()
        idx.build(str(db_path), db_id=db_id)
        idx.save(str(pkl_path))
        elapsed = time.monotonic() - t0

        after_bytes = pkl_path.stat().st_size
        total_after += after_bytes
        saved = before_bytes - after_bytes
        pct = (saved / before_bytes * 100) if before_bytes else 0

        row = (db_id, before_bytes, after_bytes, saved, elapsed, pct)
        results.append(row)

        print(
            f"{db_id:<30} {fmt_size(before_bytes):>8} {fmt_size(after_bytes):>8} "
            f"{fmt_size(saved):>8} ({pct:.0f}%)  {elapsed:.0f}s"
        )
        sys.stdout.flush()

    print("-" * 66)
    total_saved = total_before - total_after
    total_pct = (total_saved / total_before * 100) if total_before else 0
    print(
        f"{'TOTAL':<30} {fmt_size(total_before):>8} {fmt_size(total_after):>8} "
        f"{fmt_size(total_saved):>8} ({total_pct:.0f}%)"
    )


if __name__ == "__main__":
    main()
