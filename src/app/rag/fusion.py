"""
Reciprocal Rank Fusion (RRF) for combining multiple retrieval rankings.

Given several ranked lists of chunks (e.g. one from dense FAISS, one from
BM25), produce a single combined ranking using the formula:

    RRF_score(chunk) = Σ over retrievers R: 1 / (k + rank_R(chunk))

Where rank starts at 1 (top result) and k=60 is the standard constant
from Cormack et al. (2009). Chunks not present in a retriever's output
contribute 0 for that retriever — agreement across retrievers is the
strongest possible signal.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.core.logging import get_logger
from app.rag.chunking import Chunk


log = get_logger(__name__)


@dataclass
class FusedHit:
    """One result from RRF fusion."""
    chunk: Chunk
    rrf_score: float


def _chunk_key(chunk: Chunk) -> tuple[str, int]:
    """Stable hashable identity for a Chunk: (source_filename, chunk_index)."""
    return (chunk.source, chunk.chunk_index)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Chunk]],
    k: int = 60,
    top_n: int | None = None,
) -> list[FusedHit]:
    """
    Combine multiple ranked lists of chunks into one ranking via RRF.

    Args:
        rankings: One ranked list per retriever (e.g. [dense_hits, bm25_hits]).
                  Order within each list matters: index 0 is rank 1.
        k:        RRF constant. 60 is the standard value from Cormack et al.
        top_n:    If given, return only the top N fused hits.

    Returns:
        FusedHit list, sorted by rrf_score descending.
    """
    if not rankings:
        return []

    scores: dict[tuple[str, int], float] = {}
    chunk_by_key: dict[tuple[str, int], Chunk] = {}

    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            key = _chunk_key(chunk)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            chunk_by_key.setdefault(key, chunk)

    fused = [
        FusedHit(chunk=chunk_by_key[key], rrf_score=score)
        for key, score in scores.items()
    ]
    fused.sort(key=lambda h: h.rrf_score, reverse=True)

    if top_n is not None:
        fused = fused[:top_n]

    log.info(
        f"RRF fused {len(rankings)} rankings -> {len(fused)} unique chunks "
        f"(returned {len(fused)})"
    )
    return fused
