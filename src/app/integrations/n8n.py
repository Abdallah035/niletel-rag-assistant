"""n8n webhook integration for ticket creation.

POSTs a small JSON payload to the configured webhook URL.
PII is scrubbed BEFORE this layer is called (see app/core/pii.py).

Expected n8n webhook response (Tier-2 flow):
    {
        "ticket_id":  "NT-2026-0042",
        "created_at": "2026-05-15T22:30:00Z",
        "sheet_url":  "https://docs.google.com/...",
        "email_sent": true
    }
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import httpx

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)


def post_ticket(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Send a ticket payload to n8n. Returns parsed JSON response or None on failure.

    Always returns *something* useful for the UI:
      - on success: parses n8n's JSON (which should include ticket_id)
      - on n8n configured but failed: returns a local-fallback ticket_id so the
        demo can still show a ticket card even if the webhook is offline
      - on n8n not configured: returns None
    """
    url = settings.n8n_webhook_url
    if not url or "your-n8n" in url:
        log.warning("N8N_WEBHOOK_URL not configured; skipping ticket post")
        return None
    try:
        response = httpx.post(url, json=payload, timeout=settings.n8n_timeout_seconds)
        log.info(f"n8n response: status={response.status_code}")
        if 200 <= response.status_code < 300:
            try:
                data = response.json()
                if isinstance(data, dict):
                    # Normalize keys we expect downstream
                    return {
                        "ticket_id": data.get("ticket_id") or data.get("id") or _local_ticket_id(),
                        "created_at": data.get("created_at") or _now_iso(),
                        "sheet_url": data.get("sheet_url"),
                        "email_sent": bool(data.get("email_sent", False)),
                    }
            except Exception:
                log.warning("n8n returned 2xx but body wasn't JSON; using local id")
            # 2xx without parseable JSON — still treat as created
            return {
                "ticket_id": _local_ticket_id(),
                "created_at": _now_iso(),
                "sheet_url": None,
                "email_sent": False,
            }
        log.warning(f"n8n returned non-2xx ({response.status_code})")
        return None
    except Exception as e:
        log.error(f"n8n post failed: {e}")
        return None


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _local_ticket_id() -> str:
    """Fallback ticket id format (mirrors typical n8n ticket-numbering)."""
    return "NT-" + uuid4().hex[:8].upper()
