"""
Op 0a: Statistical Database Profiler

Profiles each SQLite database to extract per-column statistics, sample values,
min/max/avg, null rates, FK references, and MinHash signatures for LSH indexing.

Results can be cached to disk as JSON for offline re-use.
"""

import base64
import json
import logging
import sqlite3
import struct
from dataclasses import dataclass, field, asdict
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional

from datasketch import MinHash

logger = logging.getLogger(__name__)

_MINHASH_SAMPLE_LIMIT = 50_000


# ---------------------------------------------------------------------------
# Type affinity normalisation
# ---------------------------------------------------------------------------

def _normalize_type_affinity(declared_type: str) -> str:
    """
    Map a SQLite declared type string to one of the five SQLite type affinities:
    TEXT, INTEGER, REAL, BLOB, NUMERIC.

    Rules follow the SQLite type affinity algorithm:
    https://www.sqlite.org/datatype3.html#type_affinity
    """
    t = declared_type.upper().strip()

    if not t:
        # Empty type → BLOB affinity
        return "BLOB"

    # INTEGER affinity
    for kw in ("INT",):
        if kw in t:
            return "INTEGER"

    # TEXT affinity
    for kw in ("CHAR", "CLOB", "TEXT"):
        if kw in t:
            return "TEXT"

    # BLOB affinity (no type or BLOB)
    if t == "BLOB" or t == "NONE" or t == "":
        return "BLOB"

    # REAL affinity
    for kw in ("REAL", "FLOA", "DOUB"):
        if kw in t:
            return "REAL"

    # NUMERIC affinity — everything else
    return "NUMERIC"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ColumnProfile:
    table_name: str
    column_name: str
    data_type: str           # TEXT, INTEGER, REAL, BLOB, NUMERIC
    total_count: int         # total rows in table
    null_count: int          # number of NULL values in this column
    null_rate: float         # null_count / total_count (0.0 if total_count == 0)
    distinct_count: int      # approximate distinct non-NULL values
    sample_values: list      # list of [value, frequency] pairs, top-10 by frequency
    min_value: Any           # for numeric/date columns; None otherwise
    max_value: Any           # for numeric/date columns; None otherwise
    avg_value: Optional[float]  # for numeric columns only
    avg_length: Optional[float] # for TEXT columns only
    is_primary_key: bool
    foreign_key_ref: Optional[str]  # "other_table.other_column" or None
    minhash_bands: list      # serialized minhash signature (128 integers)


@dataclass
class DatabaseProfile:
    db_id: str
    tables: list
    columns: list            # list[ColumnProfile]
    foreign_keys: list       # list of (from_table, from_col, to_table, to_col)
    total_tables: int
    total_columns: int


# ---------------------------------------------------------------------------
# MinHash JSON compression helpers
# ---------------------------------------------------------------------------

def _encode_minhash(values: list[int]) -> str:
    """Encode 128 uint64 minhash values as a base64 string (~1365 chars vs ~1900 in JSON array)."""
    return base64.b64encode(struct.pack("128Q", *values)).decode("ascii")


def _decode_minhash(encoded: str) -> list[int]:
    """Decode a base64 string back to 128 uint64 minhash values."""
    return list(struct.unpack("128Q", base64.b64decode(encoded)))


# ---------------------------------------------------------------------------
# JSON serialisation helpers
# ---------------------------------------------------------------------------

def _make_serializable(value: Any) -> Any:
    """Convert a value to a JSON-serializable type."""
    if value is None:
        return None
    if isinstance(value, (int, float, bool, str)):
        return value
    if isinstance(value, Decimal):
        return str(value)
    # Handle bytes
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    # Try str() for anything else (date, datetime, etc.)
    return str(value)


def _sanitize_sample_values(samples: list) -> list:
    """Ensure each [value, freq] pair is JSON-serializable."""
    return [[_make_serializable(v), int(f)] for v, f in samples]


# ---------------------------------------------------------------------------
# MinHash builder
# ---------------------------------------------------------------------------

def _build_minhash(values: list) -> list:
    """
    Build a MinHash signature from a list of string values using character 3-grams.

    Returns a list of 128 plain Python ints (the hash values array).
    datasketch.MinHash.hashvalues returns numpy uint64 values, so we explicitly
    convert each element to int to keep the result JSON-serializable and to satisfy
    isinstance(v, int) checks.
    """
    mh = MinHash(num_perm=128)
    for v in values:
        s = str(v)
        # Character 3-grams
        for i in range(len(s) - 2):
            mh.update(s[i:i+3].encode("utf-8"))
        # If string is too short to produce 3-grams, hash the whole thing
        if len(s) <= 2:
            mh.update(s.encode("utf-8"))
    # Convert numpy uint64 → plain Python int for JSON-serializability
    return [int(v) for v in mh.hashvalues]


# ---------------------------------------------------------------------------
# Core profiler
# ---------------------------------------------------------------------------

def profile_database(
    db_path: str,
    db_id: str,
    output_dir: Optional[str] = None,
    force: bool = False,
) -> DatabaseProfile:
    """
    Profile a SQLite database and return a DatabaseProfile.

    Parameters
    ----------
    db_path:
        Filesystem path to the ``.sqlite`` / ``.db`` file.
    db_id:
        Logical identifier for the database (used for caching).
    output_dir:
        If provided, save/load the profile as JSON in this directory.
    force:
        If True, recompute even if a cached JSON exists.

    Returns
    -------
    DatabaseProfile
    """
    # ------------------------------------------------------------------
    # Cache check (load)
    # ------------------------------------------------------------------
    cache_path: Optional[Path] = None
    if output_dir is not None:
        cache_path = Path(output_dir) / f"{db_id}.json"
        if not force and cache_path.exists():
            return _load_profile_from_json(cache_path)

    # ------------------------------------------------------------------
    # Connect and enumerate tables
    # ------------------------------------------------------------------
    conn = sqlite3.connect(db_path)
    conn.row_factory = None  # keep tuples

    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        )
        tables: list[str] = [row[0] for row in cursor.fetchall()]

        all_columns: list[ColumnProfile] = []
        db_foreign_keys: list[tuple] = []

        for table in tables:
            # ----------------------------------------------------------
            # Column metadata via PRAGMA table_info
            # cid, name, type, notnull, dflt_value, pk
            # ----------------------------------------------------------
            col_rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()

            # Build FK map: column_name → "to_table.to_column"
            fk_map: dict[str, str] = {}
            fk_rows = conn.execute(
                f'PRAGMA foreign_key_list("{table}")'
            ).fetchall()
            for fk in fk_rows:
                # (id, seq, table, from_col, to_col, on_update, on_delete, match)
                from_col = fk[3]
                to_table = fk[2]
                to_col = fk[4]
                fk_map[from_col] = f"{to_table}.{to_col}"
                db_foreign_keys.append((table, from_col, to_table, to_col))

            # Total rows for this table
            total_count_row = conn.execute(
                f'SELECT COUNT(*) FROM "{table}"'
            ).fetchone()
            total_count: int = total_count_row[0] if total_count_row else 0

            for col_info in col_rows:
                # cid=0, name=1, type=2, notnull=3, dflt_value=4, pk=5
                col_name: str = col_info[1]
                declared_type: str = col_info[2] or ""
                is_pk: bool = col_info[5] > 0

                affinity = _normalize_type_affinity(declared_type)
                is_numeric = affinity in ("INTEGER", "REAL", "NUMERIC")
                is_text = affinity == "TEXT"

                # Null count
                null_count_row = conn.execute(
                    f'SELECT COUNT(*) FROM "{table}" WHERE "{col_name}" IS NULL'
                ).fetchone()
                null_count: int = null_count_row[0] if null_count_row else 0
                null_rate: float = (
                    null_count / total_count if total_count > 0 else 0.0
                )

                # Distinct count (non-NULL)
                distinct_row = conn.execute(
                    f'SELECT COUNT(DISTINCT "{col_name}") FROM "{table}"'
                ).fetchone()
                distinct_count: int = distinct_row[0] if distinct_row else 0

                # Sample values: top-10 by frequency, non-NULL
                sample_rows = conn.execute(
                    f'SELECT "{col_name}", COUNT(*) AS freq '
                    f'FROM "{table}" '
                    f'WHERE "{col_name}" IS NOT NULL '
                    f'GROUP BY "{col_name}" '
                    f'ORDER BY freq DESC '
                    f'LIMIT 10'
                ).fetchall()
                sample_values = _sanitize_sample_values(
                    [(r[0], r[1]) for r in sample_rows]
                )

                # Numeric stats (min, max, avg)
                min_value: Any = None
                max_value: Any = None
                avg_value: Optional[float] = None
                avg_length: Optional[float] = None

                if is_numeric:
                    stats_row = conn.execute(
                        f'SELECT MIN("{col_name}"), MAX("{col_name}") '
                        f'FROM "{table}" '
                        f'WHERE "{col_name}" IS NOT NULL'
                    ).fetchone()
                    if stats_row:
                        min_value = _make_serializable(stats_row[0])
                        max_value = _make_serializable(stats_row[1])

                    avg_row = conn.execute(
                        f'SELECT AVG(CAST("{col_name}" AS REAL)) '
                        f'FROM "{table}" '
                        f'WHERE "{col_name}" IS NOT NULL'
                    ).fetchone()
                    if avg_row and avg_row[0] is not None:
                        avg_value = float(avg_row[0])

                elif is_text:
                    # avg character length for text columns
                    avg_len_row = conn.execute(
                        f'SELECT AVG(LENGTH(CAST("{col_name}" AS TEXT))) '
                        f'FROM "{table}" '
                        f'WHERE "{col_name}" IS NOT NULL'
                    ).fetchone()
                    if avg_len_row and avg_len_row[0] is not None:
                        avg_length = float(avg_len_row[0])

                # MinHash over non-NULL values (sampled if column is very large)
                if distinct_count > _MINHASH_SAMPLE_LIMIT:
                    logger.warning(
                        "Column %s.%s has %d distinct values (> %d); "
                        "sampling %d rows for MinHash to avoid slow computation",
                        table, col_name, distinct_count, _MINHASH_SAMPLE_LIMIT, _MINHASH_SAMPLE_LIMIT,
                    )
                    all_values_rows = conn.execute(
                        f'SELECT "{col_name}" FROM "{table}" '
                        f'WHERE "{col_name}" IS NOT NULL '
                        f'LIMIT {_MINHASH_SAMPLE_LIMIT}'
                    ).fetchall()
                else:
                    all_values_rows = conn.execute(
                        f'SELECT "{col_name}" FROM "{table}" '
                        f'WHERE "{col_name}" IS NOT NULL'
                    ).fetchall()
                all_values = [r[0] for r in all_values_rows]
                minhash_bands = _build_minhash(all_values)

                # FK reference
                fk_ref: Optional[str] = fk_map.get(col_name)

                all_columns.append(
                    ColumnProfile(
                        table_name=table,
                        column_name=col_name,
                        data_type=affinity,
                        total_count=total_count,
                        null_count=null_count,
                        null_rate=null_rate,
                        distinct_count=distinct_count,
                        sample_values=sample_values,
                        min_value=min_value,
                        max_value=max_value,
                        avg_value=avg_value,
                        avg_length=avg_length,
                        is_primary_key=is_pk,
                        foreign_key_ref=fk_ref,
                        minhash_bands=minhash_bands,
                    )
                )

    finally:
        conn.close()

    profile = DatabaseProfile(
        db_id=db_id,
        tables=tables,
        columns=all_columns,
        foreign_keys=db_foreign_keys,
        total_tables=len(tables),
        total_columns=len(all_columns),
    )

    # ------------------------------------------------------------------
    # Cache write
    # ------------------------------------------------------------------
    if cache_path is not None:
        _save_profile_to_json(profile, cache_path)

    return profile


# ---------------------------------------------------------------------------
# JSON persistence
# ---------------------------------------------------------------------------

def _save_profile_to_json(profile: DatabaseProfile, path: Path) -> None:
    """Serialise a DatabaseProfile to JSON file.

    minhash_bands (128 uint64 ints) is encoded as a base64 string to reduce
    file size (~33% smaller than a JSON integer array).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(profile)
    for col_data in data["columns"]:
        col_data["minhash_bands"] = _encode_minhash(col_data["minhash_bands"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, default=str, indent=2)


def _load_profile_from_json(path: Path) -> DatabaseProfile:
    """Deserialise a DatabaseProfile from a JSON file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    for col_data in data.get("columns", []):
        raw = col_data.get("minhash_bands", [])
        # Decode base64 string written by _save_profile_to_json; fall back
        # gracefully for old-format files that stored a plain list of ints.
        if isinstance(raw, str):
            col_data["minhash_bands"] = _decode_minhash(raw)

    columns = [ColumnProfile(**col) for col in data.get("columns", [])]

    # foreign_keys are stored as lists in JSON; convert back to tuples
    foreign_keys = [tuple(fk) for fk in data.get("foreign_keys", [])]

    return DatabaseProfile(
        db_id=data["db_id"],
        tables=data["tables"],
        columns=columns,
        foreign_keys=foreign_keys,
        total_tables=data["total_tables"],
        total_columns=data["total_columns"],
    )
