"""
Op 1b: FAISS Field Semantic Index

Builds a vector index over field (column) long summaries using SentenceTransformer
embeddings. At query time, a question is embedded and the top-k most semantically
similar fields are returned.

Design notes
------------
- Uses `all-MiniLM-L6-v2` (384-dim) for both indexing and querying.
- L2-normalize all embeddings so inner product == cosine similarity.
- Small DBs (<=1000 fields): faiss.IndexFlatIP (exact).
- Large DBs (>1000 fields): faiss.IndexIVFFlat with nlist=min(32, n//10).
- Similarity scores clipped to [0, 1] after cosine computation.
- Metadata (table, column, summaries) stored as a parallel list keyed by index position.
- Serialisation: FAISS index to .index file, metadata to JSON.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.preprocessing.summarizer import FieldSummary

logger = logging.getLogger(__name__)

# Lazy model cache — loaded on first use to avoid slow startup
_EMBEDDING_MODEL = None
_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_EMBEDDING_DIM = 384

# Threshold for switching to IVF index
_IVF_THRESHOLD = 1000


def _get_embedding_model():
    """Lazily load the SentenceTransformer model."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        # Suppress HuggingFace/tqdm progress output before loading the model.
        # These env vars must be set before the first SentenceTransformer import
        # to avoid BrokenPipeError when stdout/stderr is a broken pipe (e.g. when
        # the calling process was launched with `| head -N` in a shell).
        import os
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        from sentence_transformers import SentenceTransformer
        _EMBEDDING_MODEL = SentenceTransformer(_EMBEDDING_MODEL_NAME)
        logger.info("Loaded SentenceTransformer model: %s", _EMBEDDING_MODEL_NAME)
    return _EMBEDDING_MODEL


def _embed(texts: list[str]) -> np.ndarray:
    """Embed a list of texts and L2-normalize the result. Returns float32 array."""
    model = _get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype(np.float32)
    # L2 normalize so that inner product == cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Guard against zero-norm vectors
    norms = np.where(norms == 0, 1.0, norms)
    embeddings = embeddings / norms
    return embeddings


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class FieldMatch:
    table: str
    column: str
    similarity_score: float
    long_summary: str
    short_summary: str


# ---------------------------------------------------------------------------
# FAISSIndex
# ---------------------------------------------------------------------------

class FAISSIndex:
    """FAISS vector index over field (column) semantic summaries."""

    def __init__(self) -> None:
        self._index = None          # faiss.Index
        self._fields: list[dict] = []  # [{table, column, long_summary, short_summary}]

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, field_summaries: list["FieldSummary"]) -> None:
        """
        Build the FAISS index from a list of FieldSummary objects.

        Parameters
        ----------
        field_summaries:
            List of FieldSummary objects with table_name, column_name,
            short_summary, and long_summary.
        """
        import faiss

        if not field_summaries:
            raise ValueError("Cannot build FAISSIndex from empty field_summaries")

        texts = [fs.long_summary for fs in field_summaries]
        embeddings = _embed(texts)
        n, d = embeddings.shape

        if n > _IVF_THRESHOLD:
            # IVFFlat for large databases
            nlist = min(32, n // 10)
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings)
            index.add(embeddings)
            logger.info("Built IndexIVFFlat (nlist=%d) for %d fields", nlist, n)
        else:
            # Exact flat index for smaller databases
            index = faiss.IndexFlatIP(d)
            index.add(embeddings)
            logger.info("Built IndexFlatIP for %d fields", n)

        self._index = index
        self._fields = [
            {
                "table": fs.table_name,
                "column": fs.column_name,
                "long_summary": fs.long_summary,
                "short_summary": fs.short_summary,
            }
            for fs in field_summaries
        ]

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, question: str, top_k: int = 30) -> list[FieldMatch]:
        """
        Return top-k FieldMatch objects for a question, sorted by similarity descending.

        Parameters
        ----------
        question:
            Natural language question to embed and search.
        top_k:
            Maximum number of results to return.

        Returns
        -------
        list[FieldMatch] sorted by similarity_score descending.
        """
        if self._index is None or not self._fields:
            return []

        # Embed and normalize query
        q_emb = _embed([question])  # shape: (1, d)
        k = min(top_k, len(self._fields))

        scores, indices = self._index.search(q_emb, k)

        results: list[FieldMatch] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for unfilled slots
                continue
            # Clip cosine similarity to [0, 1]
            sim = float(np.clip(score, 0.0, 1.0))
            field = self._fields[idx]
            results.append(
                FieldMatch(
                    table=field["table"],
                    column=field["column"],
                    similarity_score=sim,
                    long_summary=field["long_summary"],
                    short_summary=field["short_summary"],
                )
            )

        # Sort descending by similarity (FAISS already returns sorted, but let's be safe)
        results.sort(key=lambda m: m.similarity_score, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, index_path: str, fields_path: str) -> None:
        """
        Save the FAISS index and field metadata to disk.

        Parameters
        ----------
        index_path:
            Path to save the FAISS index (.index file).
        fields_path:
            Path to save the field metadata JSON.
        """
        import faiss

        if self._index is None:
            raise RuntimeError("Index has not been built — call build() first")

        # Ensure parent directories exist
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(fields_path).parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(index_path))
        with open(fields_path, "w", encoding="utf-8") as f:
            json.dump(self._fields, f, ensure_ascii=False, indent=2)

        logger.info(
            "FAISSIndex saved: index=%s fields=%s (%d entries)",
            index_path, fields_path, len(self._fields),
        )

    @classmethod
    def load(cls, index_path: str, fields_path: str) -> "FAISSIndex":
        """
        Load a FAISSIndex from disk.

        Parameters
        ----------
        index_path:
            Path to the FAISS index file.
        fields_path:
            Path to the field metadata JSON file.

        Returns
        -------
        FAISSIndex with _index and _fields populated.
        """
        import faiss

        obj = cls()
        obj._index = faiss.read_index(str(index_path))
        with open(fields_path, encoding="utf-8") as f:
            obj._fields = json.load(f)

        logger.info(
            "FAISSIndex loaded: index=%s fields=%s (%d entries)",
            index_path, fields_path, len(obj._fields),
        )
        return obj
