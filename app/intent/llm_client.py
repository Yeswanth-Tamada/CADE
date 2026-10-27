"""Provider-independent LLM interface and the Ollama implementation.

To add another provider (e.g. vLLM, llama.cpp server, a hosted API), subclass
:class:`LLMClient`, implement ``generate_json`` and ``health``, and register it in
:func:`build_llm_client`. Nothing else in the pipeline needs to change.
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from app.core.config import LLMProvider, Settings
from app.core.errors import (
    LLMModelNotFoundError,
    LLMResponseError,
    LLMServerError,
    LLMTimeoutError,
    LLMUnavailableError,
)


@dataclass(frozen=True)
class LLMResponse:
    content: str  # the raw text the model produced (expected to be JSON)
    model: str  # the model that actually answered
    duration_ms: float


@dataclass(frozen=True)
class LLMHealth:
    provider: str
    model: str
    reachable: bool
    model_available: Optional[bool]  # None when it could not be determined
    detail: str

    @property
    def status(self) -> str:
        if self.provider == LLMProvider.MOCK.value:
            return "mock"
        if not self.reachable:
            return "unavailable"
        if self.model_available is False:
            return "model_missing"
        return "ok"


class LLMClient(ABC):
    provider: str
    model: str

    @abstractmethod
    async def generate_json(
        self, *, system_prompt: str, user_prompt: str, json_schema: dict[str, Any], temperature: Optional[float] = None
    ) -> LLMResponse:
        """Ask the model for a JSON answer constrained (where supported) by ``json_schema``.

        ``temperature`` overrides the client's default for this call (used for mixed-temperature
        sampling of candidate playbooks and for repeated judge samples).

        Raises LLMUnavailableError, LLMTimeoutError, LLMModelNotFoundError or LLMResponseError.
        Validation of the *content* is the caller's job.
        """

    @abstractmethod
    async def health(self) -> LLMHealth:
        """Check reachability without raising."""

    async def aclose(self) -> None:  # noqa: B027 - optional hook
        """Release network resources."""


class OllamaClient(LLMClient):
    """Talks to a local Ollama server via its HTTP API (``POST /api/chat``, ``GET /api/tags``)."""

    provider = LLMProvider.OLLAMA.value

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float,
        health_timeout_seconds: float = 3.0,
        temperature: float = 0.0,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.health_timeout_seconds = health_timeout_seconds
        self.temperature = temperature
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), transport=transport)

    async def generate_json(
        self, *, system_prompt: str, user_prompt: str, json_schema: dict[str, Any], temperature: Optional[float] = None
    ) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            # Ollama "structured outputs": constrains decoding to this JSON schema.
            "format": json_schema,
            "options": {"temperature": self.temperature if temperature is None else temperature},
        }
        started = time.perf_counter()
        try:
            response = await self._client.post("/api/chat", json=payload, timeout=self.timeout_seconds)
        except httpx.TimeoutException:
            raise LLMTimeoutError(f"the LLM did not respond within {self.timeout_seconds:g} seconds") from None
        except httpx.TransportError:
            raise LLMUnavailableError("cannot reach the Ollama server; is it running?") from None
        duration_ms = (time.perf_counter() - started) * 1000

        if response.status_code == 404:
            if "model" in _error_text(response).lower():
                raise LLMModelNotFoundError(
                    f"model '{self.model}' is not installed in Ollama; run: ollama pull {self.model}"
                )
            raise LLMResponseError("Ollama API endpoint not found; check CADE_OLLAMA_BASE_URL")
        if response.status_code >= 500:
            raise LLMServerError(f"Ollama returned HTTP {response.status_code}")
        if response.status_code >= 400:
            raise LLMResponseError(f"Ollama returned HTTP {response.status_code}")

        try:
            data = response.json()
            content = data["message"]["content"]
        except (ValueError, KeyError, TypeError):
            raise LLMResponseError("Ollama returned an unexpected response envelope") from None
        if not isinstance(content, str):
            raise LLMResponseError("Ollama returned non-text message content")
        return LLMResponse(content=content, model=str(data.get("model") or self.model), duration_ms=duration_ms)

    async def health(self) -> LLMHealth:
        try:
            response = await self._client.get("/api/tags", timeout=self.health_timeout_seconds)
            response.raise_for_status()
            names = {m.get("name") or m.get("model") for m in response.json().get("models", [])}
        except httpx.TimeoutException:
            return LLMHealth(self.provider, self.model, False, None, "Ollama health check timed out")
        except (httpx.HTTPError, ValueError, AttributeError):
            return LLMHealth(self.provider, self.model, False, None, "Ollama server is not reachable")
        available = self.model in names or f"{self.model}:latest" in names
        detail = "model available" if available else f"model not installed; run: ollama pull {self.model}"
        return LLMHealth(self.provider, self.model, True, available, detail)

    async def aclose(self) -> None:
        await self._client.aclose()


def _error_text(response: httpx.Response) -> str:
    try:
        return str(response.json().get("error", ""))
    except ValueError:
        return response.text[:200]


def build_llm_client(settings: Settings, model: Optional[str] = None) -> LLMClient:
    """A client for ``model`` (default: the intent-extraction model) from the configured provider."""
    if settings.llm_provider == LLMProvider.MOCK:
        from app.intent.mock_llm_client import MockLLMClient

        return MockLLMClient()
    return OllamaClient(
        base_url=settings.ollama_base_url,
        model=model or settings.ollama_model,
        timeout_seconds=settings.llm_request_timeout_seconds,
        health_timeout_seconds=settings.llm_health_timeout_seconds,
        temperature=settings.llm_temperature,
    )


def model_family(model: str) -> str:
    """Best-effort model family from an Ollama model name: 'llama3.2:3b' -> 'llama', 'qwen2.5:3b' -> 'qwen'."""
    name = model.split("/")[-1].split(":")[0].lower()
    return re.split(r"[\d._-]", name, maxsplit=1)[0] or name
