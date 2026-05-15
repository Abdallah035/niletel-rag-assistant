"""
Dense retrieval index (FAISS) for the NileTel RAG pipeline.

Wraps a `faiss.IndexFlatIP` (exact inner-product search) plus the parallel
list of source `Chunk` objects. Provides build / search / save / load.

Inner-product is used because chunk and query vectors are L2-normalized
upstream (see `app.rag.embeddings`), so dot product == cosine similarity.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from app.core.logging import get_logger
from app.rag.chunking import Chunk
from app.rag.embeddings import Embedder


log = get_logger(__name__)


@dataclass
class DenseSearchHit:
    """One result from the dense index."""
    chunk: Chunk
    score: float


class DenseIndex:
    """FAISS IndexFlatIP + parallel chunk list. Build once, reuse."""

    INDEX_FILE = "dense.faiss"
    CHUNKS_FILE = "chunks.pkl"

    def __init__(self) -> None:
        self.index: faiss.Index | None = None
        self.chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk], embedder: Embedder) -> None:
        """Embed all chunks and populate the FAISS index."""
        if not chunks:
            raise ValueError("Cannot build a dense index from zero chunks.")
        log.info(f"Building dense index from {len(chunks)} chunks")

        texts = [c.text for c in chunks]
        vectors = embedder.embed_documents(texts)

        index = faiss.IndexFlatIP(embedder.dim)
        index.add(vectors)

        self.index = index
        self.chunks = list(chunks)
        log.info(f"Dense index built. ntotal = {index.ntotal}")

    def search(self, query_vec: np.ndarray, top_k: int = 20) -> list[DenseSearchHit]:
        """Return top-k chunks ranked by cosine similarity (= inner product)."""
        if self.index is None:
            raise RuntimeError("Index not built. Call build() or load() first.")
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        scores, ids = self.index.search(query_vec.astype("float32", copy=False), top_k)

        hits: list[DenseSearchHit] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            hits.append(DenseSearchHit(chunk=self.chunks[idx], score=float(score)))
        return hits

    def save(self, dir_path: Path) -> None:
        """Persist the index + chunks to a directory."""
        if self.index is None:
            raise RuntimeError("Cannot save an unbuilt index.")
        dir_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(dir_path / self.INDEX_FILE))
        with open(dir_path / self.CHUNKS_FILE, "wb") as f:
            pickle.dump(self.chunks, f)
        log.info(f"Dense index saved to {dir_path}")

    @classmethod
    def load(cls, dir_path: Path) -> "DenseIndex":
        """Load a previously-saved index. Returns a ready-to-search instance."""
        idx_path = dir_path / cls.INDEX_FILE
        chunks_path = dir_path / cls.CHUNKS_FILE
        if not idx_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(f"Missing index files in {dir_path}")
        instance = cls()
        instance.index = faiss.read_index(str(idx_path))
        with open(chunks_path, "rb") as f:
            instance.chunks = pickle.load(f)
        log.info(f"Dense index loaded from {dir_path} (ntotal={instance.index.ntotal})")
        return instance

    @property
    def is_built(self) -> bool:
        """True if the index has been built or loaded."""
        return self.index is not None
