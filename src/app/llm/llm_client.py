"""
LLM client wrapper for the NileTel RAG pipeline.

Three public methods that callers depend on:

    complete(prompt, system=None)       -> str        one-shot
    complete_json(prompt, system=None)  -> dict       JSON-mode (router)
    stream(prompt, system=None)         -> Iterator[str]  streaming (chat UI)

Two providers are supported, picked by `settings.llm_provider`:
  - "gemini"       — Google's google-genai SDK
  - "lightning"    — Lightning AI inference (OpenAI-compatible)
  - "openai"       — OpenAI itself or any OpenAI-compatible endpoint

The wrapper isolates SDK choice so the rest of the codebase never imports
the underlying client directly. To swap providers, edit `.env` only.
"""
from __future__ import annotations

import json
from collections.abc import Iterator

from app.core.config import settings
from app.core.logging import get_logger


log = get_logger(__name__)


def _extract_json_object(raw: str) -> dict:
    """Best-effort: parse `raw` as JSON, with recovery for common LLM quirks.

    Handles:
      - Plain valid JSON
      - JSON wrapped in ```json … ``` fences
      - Extra text before/after a JSON object (extracts first {...} block)
    Raises ValueError if no parsable object can be found.
    """
    s = (raw or "").strip()
    # Strip code fences
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
    # Try direct parse first
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # Recovery: locate the first balanced {...} block
    start = s.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(s)):
            ch = s[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = s[start : i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break
    log.warning(f"Could not parse JSON from LLM output: {raw[:200]!r}")
    raise ValueError("LLM returned no parsable JSON object")


# ============================================================
# Public class — provider dispatch happens in __init__
# ============================================================
class LLMClient:
    """Provider-agnostic LLM wrapper. Pick provider via .env (LLM_PROVIDER)."""

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.provider = (provider or settings.llm_provider).lower()
        self.model = model or settings.llm_model

        if self.provider == "gemini":
            self._impl = _GeminiImpl(
                api_key=api_key or settings.gemini_api_key,
                model=self.model,
            )
        elif self.provider in ("lightning", "openai"):
            self._impl = _OpenAICompatibleImpl(
                api_key=api_key or settings.llm_api_key,
                base_url=base_url or settings.llm_base_url,
                model=self.model,
            )
        else:
            raise RuntimeError(
                f"Unknown LLM_PROVIDER={self.provider!r}. "
                f"Use one of: gemini, lightning, openai."
            )
        log.info(f"LLM client ready (provider={self.provider}, model={self.model})")

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        return self._impl.complete(prompt, system, temperature, max_tokens)

    def complete_json(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> dict:
        return self._impl.complete_json(prompt, system, temperature, max_tokens)

    def stream(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> Iterator[str]:
        return self._impl.stream(prompt, system, temperature, max_tokens)


# ============================================================
# Gemini implementation
# ============================================================
class _GeminiImpl:
    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY required for provider=gemini.")
        from google import genai
        self._genai = genai
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def complete(self, prompt, system, temperature, max_tokens) -> str:
        from google.genai import types
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        response = self.client.models.generate_content(
            model=self.model, contents=prompt, config=config,
        )
        return (response.text or "").strip()

    def complete_json(self, prompt, system, temperature, max_tokens) -> dict:
        from google.genai import types
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
        )
        response = self.client.models.generate_content(
            model=self.model, contents=prompt, config=config,
        )
        return _extract_json_object(response.text or "")

    def stream(self, prompt, system, temperature, max_tokens) -> Iterator[str]:
        from google.genai import types
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        for chunk in self.client.models.generate_content_stream(
            model=self.model, contents=prompt, config=config,
        ):
            text = chunk.text or ""
            if text:
                yield text


# ============================================================
# OpenAI-compatible implementation (Lightning AI, OpenAI, vLLM, etc.)
# ============================================================
class _OpenAICompatibleImpl:
    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        if not api_key:
            raise RuntimeError("LLM_API_KEY required for OpenAI-compatible providers.")
        if not base_url:
            raise RuntimeError("LLM_BASE_URL required for OpenAI-compatible providers.")
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    @staticmethod
    def _build_messages(prompt: str, system: str | None) -> list[dict]:
        msgs: list[dict] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def complete(self, prompt, system, temperature, max_tokens) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(prompt, system),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()

    def complete_json(self, prompt, system, temperature, max_tokens) -> dict:
        # We DON'T use response_format here because some open models (gpt-oss
        # variants) corrupt output when forced into JSON mode. Instead we rely
        # on a strong prompt + extract-the-first-JSON-object recovery.
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(prompt, system),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw = (response.choices[0].message.content or "").strip()
        return _extract_json_object(raw)

    def stream(self, prompt, system, temperature, max_tokens) -> Iterator[str]:
        stream_resp = self.client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(prompt, system),
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream_resp:
            try:
                delta = chunk.choices[0].delta.content
            except (IndexError, AttributeError):
                delta = None
            if delta:
                yield delta
