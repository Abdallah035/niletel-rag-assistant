"""
Embedding wrapper for the NileTel RAG pipeline.

Wraps `sentence-transformers` and enforces the e5 model's prefix
protocol — `"passage: "` for indexed text, `"query: "` for searches.
Without these prefixes e5 silently produces lower-quality embeddings.
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.logging import get_logger


log = get_logger(__name__)

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


class Embedder:
    """Loads the embedding model once and exposes batch / single-query encoding."""

    def __init__(self, model_name: str | None = None, device: str = "cpu") -> None:
        name = model_name or settings.embedding_model
        log.info(f"Loading embedding model: {name} (device={device})")
        self.model = SentenceTransformer(name, device=device)
        self._dim: int = int(self.model.get_embedding_dimension())
        log.info(f"Embedding model loaded. Dim = {self._dim}")

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed many passages. Returns shape (N, dim), L2-normalized, float32."""
        if not texts:
            return np.zeros((0, self._dim), dtype="float32")
        prefixed = [PASSAGE_PREFIX + t for t in texts]
        log.info(f"Embedding {len(prefixed)} passages (batch_size={batch_size})")
        vectors = self.model.encode(
            prefixed,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.astype("float32", copy=False)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one query string. Returns shape (dim,), L2-normalized, float32."""
        prefixed = QUERY_PREFIX + text
        vec = self.model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vec.astype("float32", copy=False)

    @property
    def dim(self) -> int:
        """Embedding dimension (e.g. 1024 for e5-large)."""
        return self._dim
