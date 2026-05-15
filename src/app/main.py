"""FastAPI entrypoint for the NileTel RAG assistant.

Endpoints:
    GET  /          health check
    GET  /health    health check
    POST /ask       single-shot answer (JSON request, JSON response)
    POST /stream    streaming answer (JSON request, SSE response)
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.core.logging import setup_logging, get_logger
from app.core.pii import scrub_pii
from app.integrations.n8n import post_ticket
from app.llm.llm_client import LLMClient
from app.llm.prompts import (
    CHAT_REPLY,
    OUT_OF_SCOPE_REPLY,
    RAG_ANSWER_SYSTEM,
    RAG_ANSWER_TEMPLATE,
)
from app.rag.pipeline import create_pipeline
from app.routing.router import Intent, Router
from app.schemas import AskRequest, AskResponse, SourceInfo, TicketInfo


setup_logging()
log = get_logger(__name__)


# Globals — populated in lifespan startup
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    log.info("Starting NileTel API — loading models and indexes...")
    _state["llm"] = LLMClient()
    _state["pipeline"] = create_pipeline()
    _state["router"] = Router(_state["llm"])
    log.info("NileTel API ready.")
    yield
    log.info("Shutting down NileTel API.")
    _state.clear()


app = FastAPI(
    title="NileTel RAG Assistant",
    description="Bilingual telecom support: hybrid retrieval + reranker + Gemini",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- helpers ----------

def _format_context(hits) -> str:
    """Concatenate retrieved chunks for the answer prompt."""
    return "\n\n".join(h.chunk.text for h in hits)


def _hits_to_sources(hits) -> list[SourceInfo]:
    return [
        SourceInfo(
            source=h.chunk.source,
            heading_path=h.chunk.heading_path,
            rerank_score=h.rerank_score,
        )
        for h in hits
    ]


def _trigger_ticket(
    query: str, answer: str, hits, customer_email: str | None
) -> TicketInfo | None:
    """PII-scrub then POST to n8n. Returns TicketInfo or None."""
    payload = {
        "query": scrub_pii(query),
        "answer": answer,
        "sources": [h.chunk.source for h in hits],
        "customer_email": customer_email or "",
    }
    result = post_ticket(payload)
    if result is None:
        return None
    return TicketInfo(
        id=result["ticket_id"],
        created_at=result["created_at"],
        sheet_url=result.get("sheet_url"),
        email_sent=bool(result.get("email_sent", False)),
        status="open",
    )


# ---------- endpoints ----------

@app.get("/")
@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": settings.llm_model,
        "indexes_built": _state.get("pipeline") is not None
                         and _state["pipeline"].dense.is_built
                         and _state["pipeline"].bm25.is_built,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if not _state.get("llm"):
        raise HTTPException(503, "Service still initializing")

    t0 = time.perf_counter()
    router: Router = _state["router"]
    pipeline = _state["pipeline"]
    llm: LLMClient = _state["llm"]

    routing = router.classify(req.query)
    log.info(f"Routed query (tier {routing.tier}): {routing.intent.value} — {routing.reason}")

    if routing.intent == Intent.CHAT:
        return AskResponse(
            answer=CHAT_REPLY,
            intent=routing.intent.value,
            needs_action=False,
            sources=[],
            latency_ms=(time.perf_counter() - t0) * 1000,
            routing_tier=routing.tier,
        )

    if routing.intent == Intent.OUT_OF_SCOPE:
        return AskResponse(
            answer=OUT_OF_SCOPE_REPLY,
            intent=routing.intent.value,
            needs_action=False,
            sources=[],
            latency_ms=(time.perf_counter() - t0) * 1000,
            routing_tier=routing.tier,
        )

    # RAG or TICKET — both go through retrieval + LLM answer
    retrieval = pipeline.retrieve(req.query, top_k=settings.rerank_top_k)
    context = _format_context(retrieval.hits)
    prompt = RAG_ANSWER_TEMPLATE.format(context=context, query=req.query)
    answer = llm.complete(
        prompt=prompt,
        system=RAG_ANSWER_SYSTEM,
        temperature=0.2,
        max_tokens=400,
    )

    ticket: TicketInfo | None = None
    if routing.intent == Intent.TICKET:
        ticket = _trigger_ticket(req.query, answer, retrieval.hits, req.customer_email)

    return AskResponse(
        answer=answer,
        intent=routing.intent.value,
        needs_action=(routing.intent == Intent.TICKET),
        sources=_hits_to_sources(retrieval.hits),
        latency_ms=(time.perf_counter() - t0) * 1000,
        routing_tier=routing.tier,
        ticket=ticket,
    )


@app.post("/stream")
def stream(req: AskRequest) -> StreamingResponse:
    """Server-Sent Events streaming endpoint for the chat UI."""
    if not _state.get("llm"):
        raise HTTPException(503, "Service still initializing")

    router: Router = _state["router"]
    pipeline = _state["pipeline"]
    llm: LLMClient = _state["llm"]

    routing = router.classify(req.query)
    log.info(f"Stream routed (tier {routing.tier}): {routing.intent.value}")

    def event_stream():
        if routing.intent == Intent.CHAT:
            yield CHAT_REPLY
            return
        if routing.intent == Intent.OUT_OF_SCOPE:
            yield OUT_OF_SCOPE_REPLY
            return

        retrieval = pipeline.retrieve(req.query, top_k=settings.rerank_top_k)
        context = _format_context(retrieval.hits)
        prompt = RAG_ANSWER_TEMPLATE.format(context=context, query=req.query)

        full_answer = []
        for chunk in llm.stream(
            prompt=prompt,
            system=RAG_ANSWER_SYSTEM,
            temperature=0.2,
            max_tokens=400,
        ):
            full_answer.append(chunk)
            yield chunk

        if routing.intent == Intent.TICKET:
            ticket = _trigger_ticket(
                req.query, "".join(full_answer), retrieval.hits, req.customer_email
            )
            if ticket is not None:
                marker = (
                    f"\n\n[[TICKET]]"
                    f"id={ticket.id}|created={ticket.created_at}"
                    f"|email={int(ticket.email_sent)}"
                    f"|sheet={ticket.sheet_url or ''}"
                    f"[[/TICKET]]"
                )
                yield marker

    return StreamingResponse(event_stream(), media_type="text/plain; charset=utf-8")
