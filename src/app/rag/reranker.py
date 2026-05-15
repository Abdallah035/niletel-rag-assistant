"""
Cross-encoder reranker for the NileTel RAG pipeline.

Wraps `sentence_transformers.CrossEncoder` with `BAAI/bge-reranker-v2-m3`,
a multilingual cross-encoder fine-tuned for reranking. Unlike the bi-encoder
embedder, a cross-encoder reads the query and candidate chunk *together*,
producing a single relevance score per pair — slower per pair, but
significantly more accurate. We run it only on the top-K hybrid candidates.
"""
from __future__ import annotations

from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.core.logging import get_logger
from app.rag.chunking import Chunk


log = get_logger(__name__)


@dataclass
class RerankedHit:
    """One result from the cross-encoder reranker."""
    chunk: Chunk
    rerank_score: float


class Reranker:
    """Loads the cross-encoder once and exposes a rerank() method."""

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        name = model_name or settings.reranker_model
        requested = device or settings.reranker_device

        # Auto-fall-back: if cuda requested but unavailable, drop to cpu.
        resolved = requested
        if requested == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    log.warning("RERANKER_DEVICE=cuda requested but no GPU available — falling back to cpu")
                    resolved = "cpu"
            except ImportError:
                log.warning("torch not importable — reranker on cpu")
                resolved = "cpu"

        log.info(f"Loading reranker model: {name} (device={resolved})")
        self.model = CrossEncoder(name, device=resolved)
        self.device = resolved
        log.info(f"Reranker model loaded: {name} (device={resolved})")

    def rerank(
        self,
        query: str,
        candidates: list[Chunk],
        top_k: int = 5,
        batch_size: int = 16,
    ) -> list[RerankedHit]:
        """
        Score each candidate chunk against the query and return the top-k.

        Args:
            query:      The user's query string.
            candidates: Chunks to rerank (typically the output of RRF fusion).
            top_k:      Number of top-ranked chunks to return.
            batch_size: Pairs scored per forward pass.

        Returns:
            RerankedHit list, sorted by rerank_score descending.
        """
        if not candidates:
            return []

        pairs = [(query, c.text) for c in candidates]
        log.info(
            f"Reranking {len(pairs)} candidates "
            f"(batch_size={batch_size}, top_k={top_k})"
        )
        scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)

        scored = [
            RerankedHit(chunk=c, rerank_score=float(s))
            for c, s in zip(candidates, scores)
        ]
        scored.sort(key=lambda h: h.rerank_score, reverse=True)
        return scored[:top_k]
