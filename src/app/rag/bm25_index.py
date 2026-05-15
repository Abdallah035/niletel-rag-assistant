"""
BM25 keyword retrieval index for the NileTel RAG pipeline.

Wraps `rank_bm25.BM25Okapi` and the parallel list of `Chunk` objects.
Provides build / search / save / load symmetric to `DenseIndex`.

BM25 complements dense retrieval on exact keyword matches —
technical terms (RSRP, OSS, P1), codes (2026, Level 3), and rare words
that embedding models tend to underweight.
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.core.logging import get_logger
from app.rag.chunking import Chunk


log = get_logger(__name__)


_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Lowercase + extract Unicode word tokens. Works for English and Arabic."""
    return _TOKEN_RE.findall(text.lower())


@dataclass
class BM25SearchHit:
    """One result from the BM25 index."""
    chunk: Chunk
    score: float


class BM25Index:
    """rank_bm25.BM25Okapi + parallel chunk list. Build once, reuse."""

    INDEX_FILE = "bm25.pkl"
    CHUNKS_FILE = "chunks.pkl"

    def __init__(self) -> None:
        self.bm25: BM25Okapi | None = None
        self.chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        """Tokenize all chunk texts and fit the BM25 model."""
        if not chunks:
            raise ValueError("Cannot build a BM25 index from zero chunks.")
        log.info(f"Building BM25 index from {len(chunks)} chunks")

        tokenized_corpus = [_tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.chunks = list(chunks)
        log.info(f"BM25 index built. Documents: {len(self.chunks)}")

    def search(self, query: str, top_k: int = 20) -> list[BM25SearchHit]:
        """Return top-k chunks ranked by BM25 score."""
        if self.bm25 is None:
            raise RuntimeError("Index not built. Call build() or load() first.")

        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        top_k = min(top_k, len(self.chunks))
        if top_k <= 0:
            return []

        top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [
            BM25SearchHit(chunk=self.chunks[i], score=float(scores[i]))
            for i in top_idx
        ]

    def save(self, dir_path: Path) -> None:
        """Persist BM25 model + chunks to a directory."""
        if self.bm25 is None:
            raise RuntimeError("Cannot save an unbuilt BM25 index.")
        dir_path.mkdir(parents=True, exist_ok=True)
        with open(dir_path / self.INDEX_FILE, "wb") as f:
            pickle.dump(self.bm25, f)
        with open(dir_path / self.CHUNKS_FILE, "wb") as f:
            pickle.dump(self.chunks, f)
        log.info(f"BM25 index saved to {dir_path}")

    @classmethod
    def load(cls, dir_path: Path) -> "BM25Index":
        """Load a previously-saved BM25 index."""
        idx_path = dir_path / cls.INDEX_FILE
        chunks_path = dir_path / cls.CHUNKS_FILE
        if not idx_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(f"Missing BM25 index files in {dir_path}")
        instance = cls()
        with open(idx_path, "rb") as f:
            instance.bm25 = pickle.load(f)
        with open(chunks_path, "rb") as f:
            instance.chunks = pickle.load(f)
        log.info(f"BM25 index loaded from {dir_path} ({len(instance.chunks)} docs)")
        return instance

    @property
    def is_built(self) -> bool:
        """True if the index has been built or loaded."""
        return self.bm25 is not None
