"""
BIRD dataset loader — loads questions and database schemas.

Handles the three BIRD splits:
  - dev      → dev.json
  - train    → train.json
  - mini_dev → mini_dev_sqlite.json
"""

import json
import os
import sqlite3
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, field_validator


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class BirdEntry(BaseModel):
    """A single NL→SQL question from the BIRD benchmark."""

    question_id: int
    db_id: str
    question: str
    evidence: str  # domain-specific context/hint; may be empty string
    SQL: str
    difficulty: str

    @field_validator("difficulty")
    @classmethod
    def validate_difficulty(cls, v: str) -> str:
        valid = {"simple", "moderate", "challenging"}
        if v not in valid:
            raise ValueError(f"difficulty must be one of {valid}, got {v!r}")
        return v

    @field_validator("evidence", mode="before")
    @classmethod
    def evidence_none_to_empty(cls, v) -> str:
        """Allow None in raw JSON to become an empty string."""
        if v is None:
            return ""
        return v


class DatabaseSchema(BaseModel):
    """Structural metadata extracted from an SQLite database."""

    db_id: str
    tables: list[str]
    # table_name → list of column names (in PRAGMA order)
    columns: dict[str, list[str]]
    # table_name → list of primary-key column names
    primary_keys: dict[str, list[str]]
    # (from_table, from_col, to_table, to_col)
    foreign_keys: list[tuple[str, str, str, str]]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

# Map split name → JSON filename inside the split directory
_SPLIT_FILENAMES: dict[str, str] = {
    "dev": "dev.json",
    "train": "train.json",
    "mini_dev": "mini_dev_sqlite.json",
}

# Field mapping: some BIRD JSON files use different key names
_FIELD_ALIASES: dict[str, str] = {
    "question_id": "question_id",
    "db_id": "db_id",
    "question": "question",
    "evidence": "evidence",
    "SQL": "SQL",
    "query": "SQL",          # some files use "query" instead of "SQL"
    "difficulty": "difficulty",
}


def _parse_entry(raw: dict, index: int) -> BirdEntry:
    """Convert a raw JSON dict to a BirdEntry, handling field-name variants."""
    # Normalise SQL field
    sql = raw.get("SQL") or raw.get("query") or raw.get("sql") or ""
    # Normalise question_id (some files omit it; use list index as fallback)
    question_id = raw.get("question_id", index)
    # Normalise difficulty
    difficulty = (raw.get("difficulty") or "simple").strip().lower()

    return BirdEntry(
        question_id=int(question_id),
        db_id=str(raw.get("db_id", "")),
        question=str(raw.get("question", "")),
        evidence=raw.get("evidence") or "",
        SQL=sql,
        difficulty=difficulty,
    )


def load_bird_split(
    split: str,
    data_dir: str = "./data/bird",
) -> list[BirdEntry]:
    """
    Load a BIRD split from disk.

    Parameters
    ----------
    split:
        One of "dev", "train", or "mini_dev".
    data_dir:
        Root directory that contains per-split sub-folders.

    Returns
    -------
    list[BirdEntry]
        Empty list if the JSON file is not found.
    """
    filename = _SPLIT_FILENAMES.get(split)
    if filename is None:
        raise ValueError(
            f"Unknown split {split!r}. Valid splits: {list(_SPLIT_FILENAMES)}"
        )

    # Try the split sub-directory first, then the root data_dir directly.
    candidates = [
        Path(data_dir) / split / filename,
        Path(data_dir) / filename,
    ]

    json_path: Optional[Path] = None
    for candidate in candidates:
        if candidate.exists():
            json_path = candidate
            break

    if json_path is None:
        return []

    with open(json_path, encoding="utf-8") as f:
        raw_list: list[dict] = json.load(f)

    entries: list[BirdEntry] = []
    for i, raw in enumerate(raw_list):
        try:
            entries.append(_parse_entry(raw, index=i))
        except Exception as exc:
            # Skip malformed entries but don't crash the whole load
            import warnings
            warnings.warn(f"Skipping entry {i} in {json_path}: {exc}")

    return entries


# ---------------------------------------------------------------------------
# Schema loader
# ---------------------------------------------------------------------------

def load_schema(db_id: str, databases_dir: str) -> DatabaseSchema:
    """
    Parse structural metadata from an SQLite database using PRAGMA queries.

    Parameters
    ----------
    db_id:
        The database identifier (sub-directory name under *databases_dir*).
    databases_dir:
        Directory that contains per-database sub-folders, each holding a
        ``<db_id>.sqlite`` file.

    Returns
    -------
    DatabaseSchema
    """
    db_path = Path(databases_dir) / db_id / f"{db_id}.sqlite"
    if not db_path.exists():
        # Try without sub-directory
        db_path = Path(databases_dir) / f"{db_id}.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_id} in {databases_dir}")

    conn = sqlite3.connect(str(db_path))
    try:
        tables: list[str] = []
        columns: dict[str, list[str]] = {}
        primary_keys: dict[str, list[str]] = {}
        foreign_keys: list[tuple[str, str, str, str]] = []

        # Enumerate tables (exclude SQLite internals)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]

        for table in tables:
            # Column info
            col_cursor = conn.execute(f'PRAGMA table_info("{table}")')
            col_rows = col_cursor.fetchall()
            # col_rows: (cid, name, type, notnull, dflt_value, pk)
            columns[table] = [row[1] for row in col_rows]
            primary_keys[table] = [row[1] for row in col_rows if row[5] > 0]

            # Foreign keys
            fk_cursor = conn.execute(f'PRAGMA foreign_key_list("{table}")')
            for fk_row in fk_cursor.fetchall():
                # (id, seq, table, from, to, on_update, on_delete, match)
                to_table = fk_row[2]
                from_col = fk_row[3]
                to_col = fk_row[4]
                foreign_keys.append((table, from_col, to_table, to_col))

    finally:
        conn.close()

    return DatabaseSchema(
        db_id=db_id,
        tables=tables,
        columns=columns,
        primary_keys=primary_keys,
        foreign_keys=foreign_keys,
    )
