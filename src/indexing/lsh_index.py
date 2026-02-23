"""
Op 1a: LSH Cell Value Index

Builds a MinHash LSH index over all distinct non-NULL cell values in a SQLite
database.  At query time fuzzy-matching via 3-gram Jaccard similarity is used
to retrieve candidate (table, column, value) triples for a given keyword.

Design notes
------------
- MinHash with num_perm=128 and character 3-gram shingling.
- MinHashLSH threshold=0.5 (controls candidate recall vs. speed).
- Key format: ``"table.column::value"``
- Numeric values are included as their string representation.
- NULL values are excluded.
- ``MinHashLSH`` does NOT store MinHash objects, so we maintain a parallel
  dict ``self._minhashes`` for Jaccard re-ranking.
- Serialisation: pickle of the whole object (``save`` / ``load``).
"""

from __future__ import annotations

import logging
import pickle
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from datasketch import MinHash, MinHashLSH

logger = logging.getLogger(__name__)

_NUM_PERM = 128
# LSH candidate-retrieval threshold.  Set to 0.3 so that moderate-typo queries
# (e.g. "Untied States" → "United States") are still pulled as candidates.
# The true 3-gram Jaccard for that pair is ~0.47, so a threshold of 0.5 would
# miss it.  Post-retrieval re-ranking by actual Jaccard keeps precision high.
_LSH_THRESHOLD = 0.3
_NGRAM_SIZE = 3
# Maximum distinct values per column to index.  Columns with more distinct
# values (e.g. ID columns, UUIDs, free-text blobs) are too large to index
# efficiently and are rarely useful for fuzzy value matching.  This matches
# the profiler's MinHash sampling cap of 50,000 rows.
_MAX_DISTINCT_PER_COLUMN = 50_000
# SQLite type affinities that we do NOT index in LSH.  INTEGER and REAL
# columns (IDs, lat/lng, phone numbers, prices, years …) are never queried
# via fuzzy text matching — including them bloats the index 2–10× without
# adding recall.  The same affinity rules as SQLite's own algorithm are used
# (see _type_affinity() below and https://www.sqlite.org/datatype3.html).
_SKIP_AFFINITIES: frozenset[str] = frozenset({"INTEGER", "REAL"})


# ---------------------------------------------------------------------------
# Type-affinity helper (mirrors profiler._normalize_type_affinity)
# ---------------------------------------------------------------------------

def _type_affinity(declared_type: str) -> str:
    """Return the SQLite type affinity for *declared_type*.

    Follows the SQLite affinity algorithm so that we skip the same column
    types the profiler would classify as INTEGER or REAL.
    """
    t = declared_type.upper().strip()
    if not t:
        return "BLOB"
    if "INT" in t:
        return "INTEGER"
    for kw in ("CHAR", "CLOB", "TEXT"):
        if kw in t:
            return "TEXT"
    if t in ("BLOB", "NONE"):
        return "BLOB"
    for kw in ("REAL", "FLOA", "DOUB"):
        if kw in t:
            return "REAL"
    return "NUMERIC"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _char_ngrams(text: str, n: int = _NGRAM_SIZE) -> set[bytes]:
    """Return the set of character n-gram byte strings for *text*."""
    s = text.lower()
    if len(s) < n:
        # For short strings, use the whole string as a single shingle so that
        # they can still be indexed (rather than producing an empty set).
        return {s.encode("utf-8")}
    return {s[i : i + n].encode("utf-8") for i in range(len(s) - n + 1)}


def _make_minhash(value: str) -> MinHash:
    """Create a MinHash object for *value* using 3-gram shingling."""
    mh = MinHash(num_perm=_NUM_PERM)
    for gram in _char_ngrams(value):
        mh.update(gram)
    return mh


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class CellMatch:
    table: str
    column: str
    matched_value: str
    similarity_score: float
    exact_match: bool


# ---------------------------------------------------------------------------
# LSHIndex
# ---------------------------------------------------------------------------

class LSHIndex:
    """MinHash LSH index over all distinct cell values in a SQLite database."""

    def __init__(self) -> None:
        self._lsh: MinHashLSH = MinHashLSH(threshold=_LSH_THRESHOLD, num_perm=_NUM_PERM)
        # Stores MinHash objects keyed by the same key used in _lsh so that we
        # can compute exact Jaccard similarity for re-ranking.
        self._minhashes: dict[str, MinHash] = {}
        # Maps key → (table, column, value) for result assembly.
        self._key_meta: dict[str, tuple[str, str, str]] = {}

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, db_path: str, db_id: str) -> None:
        """Populate the index from all distinct non-NULL values in *db_path*."""
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.cursor()
            # Get all user tables.
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            tables = [row[0] for row in cursor.fetchall()]

            for table in tables:
                # Get column names and declared types.
                cursor.execute(f"PRAGMA table_info(\"{table}\")")
                # row: (cid, name, type, notnull, dflt_value, pk)
                column_infos: list[tuple[str, str]] = [
                    (row[1], row[2]) for row in cursor.fetchall()
                ]

                for column, declared_type in column_infos:
                    # Skip INTEGER and REAL columns — they are never useful
                    # for fuzzy text matching and would bloat the index.
                    affinity = _type_affinity(declared_type)
                    if affinity in _SKIP_AFFINITIES:
                        logger.debug(
                            "Skipping %s.%s (type=%s, affinity=%s) — not text-like",
                            table, column, declared_type, affinity,
                        )
                        continue
                    try:
                        # Count distinct values first; skip high-cardinality
                        # columns (IDs, UUIDs, free-text blobs) that would be
                        # too slow to index and are not useful for fuzzy matching.
                        cursor.execute(
                            f"SELECT COUNT(*) FROM ("
                            f"SELECT DISTINCT \"{column}\" FROM \"{table}\" "
                            f"WHERE \"{column}\" IS NOT NULL LIMIT {_MAX_DISTINCT_PER_COLUMN + 1})"
                        )
                        n_distinct = cursor.fetchone()[0]
                        if n_distinct > _MAX_DISTINCT_PER_COLUMN:
                            logger.warning(
                                "Skipping %s.%s for LSH: %d distinct values > cap %d",
                                table, column, n_distinct, _MAX_DISTINCT_PER_COLUMN,
                            )
                            continue
                        cursor.execute(
                            f"SELECT DISTINCT \"{column}\" FROM \"{table}\" "
                            f"WHERE \"{column}\" IS NOT NULL"
                        )
                        rows = cursor.fetchall()
                    except sqlite3.Error as exc:
                        logger.warning(
                            "Skipping %s.%s due to error: %s", table, column, exc
                        )
                        continue

                    for (raw_val,) in rows:
                        value_str = str(raw_val).strip()
                        if not value_str:
                            continue
                        key = f"{table}.{column}::{value_str}"
                        # Skip duplicates (can happen for numeric representations).
                        if key in self._minhashes:
                            continue
                        mh = _make_minhash(value_str)
                        try:
                            self._lsh.insert(key, mh)
                        except ValueError:
                            # datasketch raises ValueError if key already exists.
                            pass
                        self._minhashes[key] = mh
                        self._key_meta[key] = (table, column, value_str)

        finally:
            conn.close()

        logger.info(
            "LSHIndex built for db_id=%s: %d entries", db_id, len(self._minhashes)
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, keyword: str, top_k: int = 5) -> list[CellMatch]:
        """
        Return up to *top_k* CellMatch objects for *keyword*, ranked by
        Jaccard similarity (descending).
        """
        if not self._minhashes:
            return []

        query_mh = _make_minhash(keyword)
        candidate_keys: list[str] = self._lsh.query(query_mh)

        results: list[CellMatch] = []
        for key in candidate_keys:
            stored_mh = self._minhashes.get(key)
            if stored_mh is None:
                continue
            sim = query_mh.jaccard(stored_mh)
            table, column, matched_value = self._key_meta[key]
            exact = keyword.lower() == matched_value.lower()
            results.append(
                CellMatch(
                    table=table,
                    column=column,
                    matched_value=matched_value,
                    similarity_score=sim,
                    exact_match=exact,
                )
            )

        # Sort by similarity score descending; exact matches always first.
        results.sort(key=lambda m: (m.exact_match, m.similarity_score), reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Pickle-serialise the entire LSHIndex to *path*."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("LSHIndex saved to %s (%d entries)", path, len(self._minhashes))

    @classmethod
    def load(cls, path: str) -> "LSHIndex":
        """Load and return an LSHIndex previously saved with :meth:`save`."""
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not an LSHIndex: {type(obj)}")
        logger.info("LSHIndex loaded from %s (%d entries)", path, len(obj._minhashes))
        return obj
