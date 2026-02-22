"""
Op 1c: Example (Few-Shot) Store

Builds a vector index over masked question skeletons from training data.
At query time, given a new question + db_id, returns structurally similar
examples from *other* databases (prevents schema leakage into few-shot context).

Skeleton masking
----------------
Order matters: regex first (so already-replaced tokens don't confuse spaCy NER),
then spaCy NER on the already-partially-masked text.

1. Numbers: `[NUM]`  — integers, floats, years
2. Quoted strings: `[STR]` — single or double-quoted text
3. Named entities (spaCy NER): `[ENTITY]` — persons, places, orgs, etc.

The masked skeleton is embedded with all-MiniLM-L6-v2 (same model as FAISS index).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.data.bird_loader import BirdEntry

logger = logging.getLogger(__name__)

# Lazy cache for spaCy and embedding models
_SPACY_MODEL = None
_EMBEDDING_MODEL = None
_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def _get_spacy_model():
    """Lazily load the spaCy model. Raises ImportError with helpful message if missing."""
    global _SPACY_MODEL
    if _SPACY_MODEL is None:
        try:
            import spacy
            _SPACY_MODEL = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model: en_core_web_sm")
        except OSError as e:
            raise ImportError(
                "spaCy model 'en_core_web_sm' is not installed. "
                "Run: python -m spacy download en_core_web_sm"
            ) from e
    return _SPACY_MODEL


def _get_embedding_model():
    """Lazily load the SentenceTransformer model."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDING_MODEL = SentenceTransformer(_EMBEDDING_MODEL_NAME)
        logger.info("Loaded SentenceTransformer model: %s", _EMBEDDING_MODEL_NAME)
    return _EMBEDDING_MODEL


def _embed(texts: list[str]) -> np.ndarray:
    """Embed a list of texts and L2-normalize. Returns float32 array."""
    model = _get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    embeddings = embeddings / norms
    return embeddings


# Regex patterns for skeleton masking
# 1. Numbers (integers, floats, years — standalone, not part of words)
_NUM_RE = re.compile(r'\b\d+(?:\.\d+)?\b')
# 2. Quoted strings — single or double quotes
_STR_RE = re.compile(r"""(?:\"[^\"]*\"|'[^']*')""")


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExampleEntry:
    question_id: int
    db_id: str
    question: str
    evidence: str
    sql: str
    skeleton: str
    similarity_score: float


# ---------------------------------------------------------------------------
# ExampleStore
# ---------------------------------------------------------------------------

class ExampleStore:
    """Vector store of training question skeletons for few-shot retrieval."""

    def __init__(self) -> None:
        self._index = None        # faiss.Index
        self._metadata: list[dict] = []  # list of {question_id, db_id, question, evidence, sql, skeleton}

    # ------------------------------------------------------------------
    # Public masking API
    # ------------------------------------------------------------------

    def mask_question(self, question: str) -> str:
        """
        Mask a question to a structural skeleton.

        Masking order (per spec):
        1. Numbers        → [NUM]   (regex, first to protect from NER)
        2. Quoted strings → [STR]   (regex)
        3. Named entities → [ENTITY] (spaCy NER, only on unmasked text)

        We run NER on the *original* question to get entity positions, then apply
        all replacements together in reverse order. This prevents already-masked
        placeholders (e.g. [NUM]) from being picked up by spaCy NER.

        Parameters
        ----------
        question:
            Raw natural-language question string.

        Returns
        -------
        Skeleton string with entities, numbers, and quoted strings replaced.
        """
        # Collect all replacement spans from the *original* text
        # Each entry: (start, end, replacement_text)
        replacements: list[tuple[int, int, str]] = []

        # Step 1: Numbers (on original text)
        for m in _NUM_RE.finditer(question):
            replacements.append((m.start(), m.end(), "[NUM]"))

        # Step 2: Quoted strings (on original text)
        for m in _STR_RE.finditer(question):
            replacements.append((m.start(), m.end(), "[STR]"))

        # Step 3: Named entities via spaCy NER (on original text)
        nlp = _get_spacy_model()
        doc = nlp(question)
        for ent in doc.ents:
            replacements.append((ent.start_char, ent.end_char, "[ENTITY]"))

        # Remove overlapping spans: keep the one with the smallest start (greedy left-to-right)
        # Sort by start position, then resolve overlaps
        replacements.sort(key=lambda r: r[0])
        non_overlapping: list[tuple[int, int, str]] = []
        last_end = 0
        for start, end, token in replacements:
            if start >= last_end:
                non_overlapping.append((start, end, token))
                last_end = end

        # Apply replacements in reverse order to preserve character offsets
        result = question
        for start, end, token in reversed(non_overlapping):
            result = result[:start] + token + result[end:]

        return result

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, train_entries: "list[BirdEntry]") -> None:
        """
        Build the example store from a list of training BirdEntry objects.

        Parameters
        ----------
        train_entries:
            List of BirdEntry objects from the training split.

        Raises
        ------
        ValueError:
            If train_entries is empty.
        """
        import faiss

        if not train_entries:
            raise ValueError("Cannot build ExampleStore from empty training set")

        # Build skeletons for all entries
        skeletons = [self.mask_question(entry.question) for entry in train_entries]

        # Embed all skeletons
        embeddings = _embed(skeletons)
        n, d = embeddings.shape

        # Build flat inner-product index (L2-normalized → cosine similarity)
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)

        self._index = index
        self._metadata = [
            {
                "question_id": entry.question_id,
                "db_id": entry.db_id,
                "question": entry.question,
                "evidence": entry.evidence,
                "sql": entry.SQL,
                "skeleton": skeletons[i],
            }
            for i, entry in enumerate(train_entries)
        ]

        logger.info("ExampleStore built: %d training examples", n)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, question: str, db_id: str, top_k: int = 8) -> list[ExampleEntry]:
        """
        Retrieve structurally similar examples, excluding those from db_id.

        Parameters
        ----------
        question:
            The natural-language question to find similar examples for.
        db_id:
            The database identifier of the query; examples from this DB are excluded.
        top_k:
            Maximum number of examples to return.

        Returns
        -------
        list[ExampleEntry] sorted by similarity_score descending (up to top_k items).
        """
        if self._index is None or not self._metadata:
            return []

        skeleton = self.mask_question(question)
        q_emb = _embed([skeleton])  # shape: (1, d)

        # Search more than top_k to account for db_id exclusion
        # Fetch all to be safe (FAISS is fast)
        k_search = min(len(self._metadata), max(top_k * 5, top_k + 20))
        scores, indices = self._index.search(q_emb, k_search)

        results: list[ExampleEntry] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self._metadata[idx]
            # Exclude examples from the same database
            if meta["db_id"] == db_id:
                continue
            sim = float(np.clip(score, 0.0, 1.0))
            results.append(
                ExampleEntry(
                    question_id=meta["question_id"],
                    db_id=meta["db_id"],
                    question=meta["question"],
                    evidence=meta["evidence"],
                    sql=meta["sql"],
                    skeleton=meta["skeleton"],
                    similarity_score=sim,
                )
            )
            if len(results) >= top_k:
                break

        # Sort descending (should already be sorted, but ensure)
        results.sort(key=lambda e: e.similarity_score, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, faiss_path: str, meta_path: str) -> None:
        """
        Save the FAISS index and metadata to disk.

        Parameters
        ----------
        faiss_path:
            Path to write the FAISS index.
        meta_path:
            Path to write the metadata JSON.
        """
        import faiss

        if self._index is None:
            raise RuntimeError("ExampleStore has not been built — call build() first")

        Path(faiss_path).parent.mkdir(parents=True, exist_ok=True)
        Path(meta_path).parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(faiss_path))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False, indent=2)

        logger.info(
            "ExampleStore saved: faiss=%s meta=%s (%d entries)",
            faiss_path, meta_path, len(self._metadata),
        )

    @classmethod
    def load(cls, faiss_path: str, meta_path: str) -> "ExampleStore":
        """
        Load an ExampleStore from disk.

        Parameters
        ----------
        faiss_path:
            Path to the FAISS index file.
        meta_path:
            Path to the metadata JSON file.

        Returns
        -------
        ExampleStore with _index and _metadata populated.
        """
        import faiss

        obj = cls()
        obj._index = faiss.read_index(str(faiss_path))
        with open(meta_path, encoding="utf-8") as f:
            obj._metadata = json.load(f)

        logger.info(
            "ExampleStore loaded: faiss=%s meta=%s (%d entries)",
            faiss_path, meta_path, len(obj._metadata),
        )
        return obj
