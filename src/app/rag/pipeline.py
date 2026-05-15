"""
End-to-end RAG retrieval pipeline.

Composes the individual components (Embedder, DenseIndex, BM25Index, Reranker)
into a single `RAGPipeline.retrieve(query)` call that:

    1. embeds the query
    2. fans out to dense + BM25 retrievers (top-K each)
    3. fuses the rankings via RRF
    4. reranks the fused candidates with a cross-encoder
    5. returns the top-N with all per-stage scores attached for debugging

The pipeline persists its indexes under `artifacts/dense/` and `artifacts/bm25/`
so subsequent startups load from disk instead of rebuilding.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger
from app.rag.chunking import Chunk, chunk_directory
from app.rag.embeddings import Embedder
from app.rag.dense_index import DenseIndex
from app.rag.bm25_index import BM25Index
from app.rag.fusion import reciprocal_rank_fusion
from app.rag.reranker import Reranker, RerankedHit


log = get_logger(__name__)


def _index_dir(name: str) -> Path:
    """Path to a sub-directory under settings.artifacts_path, e.g. 'dense' or 'bm25'."""
    return settings.artifacts_path / name


@dataclass
class RetrievalResult:
    """Final output of the pipeline: the chunks the LLM should see plus debug info."""
    query: str
    hits: list[RerankedHit]
    candidate_count: int = 0
    timings_ms: dict[str, float] = field(default_factory=dict)


class RAGPipeline:
    """Holds embedder + dense + bm25 + reranker; exposes retrieve()."""

    def __init__(
        self,
        embedder: Embedder,
        dense: DenseIndex,
        bm25: BM25Index,
        reranker: Reranker | None,
    ) -> None:
        self.embedder = embedder
        self.dense = dense
        self.bm25 = bm25
        self.reranker = reranker

    def build_or_load(self, chunks: list[Chunk] | None = None) -> None:
        """Load indexes from disk if present; otherwise build from chunks and save."""
        dense_dir = _index_dir("dense")
        bm25_dir = _index_dir("bm25")

        # Try the fast path: load from disk
        try:
            self.dense = DenseIndex.load(dense_dir)
            self.bm25 = BM25Index.load(bm25_dir)
            log.info("Indexes loaded from disk.")
            return
        except FileNotFoundError:
            log.info("No saved indexes found — building from scratch.")

        # Slow path: chunk if needed, embed, fit, persist
        if chunks is None:
            chunks = chunk_directory(settings.data_path, max_chars=settings.chunk_max_chars)

        self.dense.build(chunks, self.embedder)
        self.bm25.build(chunks)
        self.dense.save(dense_dir)
        self.bm25.save(bm25_dir)
        log.info("Indexes built and saved to disk.")

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        """Run the full hybrid + rerank pipeline for one query."""
        import time

        if not self.dense.is_built or not self.bm25.is_built:
            raise RuntimeError("Pipeline not built. Call build_or_load() first.")

        top_k = top_k or settings.rerank_top_k
        timings: dict[str, float] = {}

        # 1. Embed the query
        t0 = time.perf_counter()
        q_vec = self.embedder.embed_query(query)
        timings["embed"] = (time.perf_counter() - t0) * 1000

        # 2. Dense retrieval
        t0 = time.perf_counter()
        dense_hits = self.dense.search(q_vec, top_k=settings.dense_top_k)
        timings["dense_search"] = (time.perf_counter() - t0) * 1000

        # 3. BM25 retrieval
        t0 = time.perf_counter()
        bm25_hits = self.bm25.search(query, top_k=settings.bm25_top_k)
        timings["bm25_search"] = (time.perf_counter() - t0) * 1000

        # 4. RRF fusion
        t0 = time.perf_counter()
        fused = reciprocal_rank_fusion(
            [
                [h.chunk for h in dense_hits],
                [h.chunk for h in bm25_hits],
            ],
            k=settings.rrf_k,
        )
        timings["fusion"] = (time.perf_counter() - t0) * 1000

        candidates = [h.chunk for h in fused]

        # 5. Cross-encoder rerank — skipped when reranker is disabled.
        # When skipped, we map RRF top_k chunks into RerankedHit shape so the
        # downstream API contract is identical either way.
        t0 = time.perf_counter()
        if self.reranker is not None:
            reranked = self.reranker.rerank(query, candidates, top_k=top_k)
        else:
            from app.rag.reranker import RerankedHit
            reranked = [
                RerankedHit(chunk=h.chunk, rerank_score=h.rrf_score)
                for h in fused[:top_k]
            ]
        timings["rerank"] = (time.perf_counter() - t0) * 1000

        return RetrievalResult(
            query=query,
            hits=reranked,
            candidate_count=len(candidates),
            timings_ms=timings,
        )


def create_pipeline() -> RAGPipeline:
    """Construct a fully-loaded pipeline. Call once at app startup."""
    embedder = Embedder()
    dense = DenseIndex()
    bm25 = BM25Index()
    reranker: Reranker | None = Reranker() if settings.reranker_enabled else None
    if reranker is None:
        log.info("Reranker disabled (RERANKER_ENABLED=false) — using RRF top-K directly.")

    pipeline = RAGPipeline(embedder=embedder, dense=dense, bm25=bm25, reranker=reranker)
    pipeline.build_or_load()
    return pipeline
