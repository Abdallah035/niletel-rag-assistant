"""
PII scrubber for outbound payloads.

Scrubs Egyptian phone numbers, national IDs, and email addresses before
forwarding user content to external systems (n8n, Google Sheets, email).
NTRA / data-privacy compliance for telecom support.

Usage:
    safe_text = scrub_pii("رقمي 01012345678 وعندي مشكلة")
    # -> "رقمي [PHONE] وعندي مشكلة"
"""
from __future__ import annotations

import re


# Egyptian mobile: optional +2 / 0020 country code, then 1[0-25]xxxxxxxx
# Tolerates spaces between digit groups.
_EG_PHONE_RE = re.compile(
    r"(?:(?:\+|00)?\s*20\s*)?0?\s*1\s*[0-25]\s*\d(?:\s*\d){7}",
)

# Egyptian national ID = exactly 14 digits
_EG_NID_RE = re.compile(r"\b\d{14}\b")

# Email
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def scrub_pii(text: str) -> str:
    """Replace PII with placeholder tags. Idempotent on already-scrubbed text."""
    if not text:
        return text
    text = _EG_NID_RE.sub("[NATIONAL_ID]", text)
    text = _EG_PHONE_RE.sub("[PHONE]", text)
    text = _EMAIL_RE.sub("[EMAIL]", text)
    return text
