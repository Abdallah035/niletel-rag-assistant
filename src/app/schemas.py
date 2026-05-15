"""Pydantic request/response models for the FastAPI surface."""
from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's question")
    session_id: str | None = Field(None, description="Optional session id for memory")
    customer_email: str | None = Field(None, description="Optional email for ticket notification")


class SourceInfo(BaseModel):
    source: str
    heading_path: list[str] = []
    rerank_score: float = 0.0


class TicketInfo(BaseModel):
    id: str
    created_at: str
    sheet_url: str | None = None
    email_sent: bool = False
    status: str = "open"


class AskResponse(BaseModel):
    answer: str
    intent: str
    needs_action: bool = False
    sources: list[SourceInfo] = []
    latency_ms: float = 0.0
    routing_tier: int = 0
    ticket: TicketInfo | None = None
