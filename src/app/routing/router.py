"""
Tiered intent router for the NileTel RAG assistant.

Three tiers, cheapest first — only ambiguous queries reach the LLM:

    Tier 1 — REGEX:    greetings / thanks / goodbyes        (~1ms, free)
    Tier 2 — KEYWORDS: explicit ticket / out-of-scope words (~1ms, free)
    Tier 3 — LLM:      everything else                      (~500ms, 1 API call)

Output is a `RoutingResult` carrying intent + reason + which tier matched.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from enum import Enum

from app.core.logging import get_logger
from app.llm.llm_client import LLMClient
from app.llm.prompts import ROUTER_SYSTEM, ROUTER_USER_TEMPLATE


log = get_logger(__name__)


class Intent(str, Enum):
    """The 4 possible classifications for an incoming query."""
    CHAT = "chat"
    TICKET = "ticket"
    RAG = "rag"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass
class RoutingResult:
    intent: Intent
    reason: str = ""
    tier: int = 0          # 1, 2, or 3
    latency_ms: float = 0.0


# ---------- Tier 1 — regex patterns (greetings / thanks / goodbyes) ----------
# Anchored at start, word-boundary at end; case-insensitive; covers EN + AR.
_GREETING_RE = re.compile(
    r"^\s*(ازيك|ازيكم|ازاي\s*حالك|اهلا|اهلاً|أهلا|أهلاً|مرحبا|مرحباً|"
    r"السلام|سلام|هاي|هلا|"
    r"hi|hello|hey|good\s*(morning|evening|afternoon))\b",
    re.IGNORECASE | re.UNICODE,
)
_THANKS_RE = re.compile(
    r"^\s*(شكرا|شكراً|تسلم|تسلمي|ميرسي|متشكر|"
    r"thanks|thank\s*you|thx|ty)\b",
    re.IGNORECASE | re.UNICODE,
)
_GOODBYE_RE = re.compile(
    r"^\s*(باي|مع\s*السلامة|سلام\s*عليكم|"
    r"bye|goodbye|see\s*you)\b",
    re.IGNORECASE | re.UNICODE,
)


# ---------- Tier 2 — keyword sets (strong, unambiguous signals) ----------
# Out-of-scope: clearly non-telecom topics.
_OOS_TOKENS = frozenset({
    # AR
    "فيلم", "افلام", "أفلام", "مسلسل", "مسلسلات", "كورة", "ماتش",
    "مطعم", "اكل", "أكل", "اكلة", "سينما", "موسيقى", "اغنية", "أغنية",
    "الطقس", "الجو", "رياضة",
    # EN
    "movie", "movies", "sport", "sports", "weather", "restaurant",
    "food", "song", "music", "football", "match",
})

# Strong ticket signals — verb + noun pair anywhere in query.
_TICKET_VERBS = frozenset({
    "اعمل", "افتح", "افتحلي", "ابعت", "اطلب", "عاوز", "عايز", "محتاج",
    "create", "open", "file", "raise", "submit",
})
_TICKET_NOUNS = frozenset({
    "تذكرة", "تيكت", "شكوى", "بلاغ",
    "ticket", "complaint",
})


_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    """Lowercase token set for keyword matching."""
    return {t.lower() for t in _TOKEN_RE.findall(text)}


# ---------- Router ----------

class Router:
    """Three-tier classifier. Pass an LLMClient at construction."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def classify(self, query: str) -> RoutingResult:
        t0 = time.perf_counter()

        intent = self._tier1_regex(query)
        if intent is not None:
            return RoutingResult(
                intent=intent, reason="regex match", tier=1,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        intent = self._tier2_keywords(query)
        if intent is not None:
            return RoutingResult(
                intent=intent, reason="keyword match", tier=2,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        intent, reason = self._tier3_llm(query)
        return RoutingResult(
            intent=intent, reason=reason, tier=3,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    # ------- Tier 1 -------
    def _tier1_regex(self, query: str) -> Intent | None:
        if _GREETING_RE.search(query) or _THANKS_RE.search(query) or _GOODBYE_RE.search(query):
            return Intent.CHAT
        return None

    # ------- Tier 2 -------
    def _tier2_keywords(self, query: str) -> Intent | None:
        toks = _tokens(query)

        if toks & _OOS_TOKENS:
            return Intent.OUT_OF_SCOPE

        # Ticket = at least one verb AND at least one noun
        if (toks & _TICKET_VERBS) and (toks & _TICKET_NOUNS):
            return Intent.TICKET

        return None

    # ------- Tier 3 -------
    def _tier3_llm(self, query: str) -> tuple[Intent, str]:
        try:
            data = self.llm.complete_json(
                prompt=ROUTER_USER_TEMPLATE.format(query=query),
                system=ROUTER_SYSTEM,
                temperature=0.0,
                max_tokens=128,
            )
            label = str(data.get("intent", "")).lower()
            reason = str(data.get("reason", ""))
            if label == "ticket":
                return Intent.TICKET, reason
            if label == "rag":
                return Intent.RAG, reason
            log.warning(f"Router LLM returned unknown intent={label!r}; defaulting to rag")
            return Intent.RAG, f"unknown LLM intent {label!r}, defaulted to rag"
        except (ValueError, json.JSONDecodeError, Exception) as e:
            log.warning(f"Router LLM call failed: {e}; defaulting to rag")
            return Intent.RAG, f"LLM error ({type(e).__name__}); defaulted to rag"
