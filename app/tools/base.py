"""Base classes for security-tool adapters.

:class:`SecurityTool` validates every request *before* an adapter sees it (known operation,
accepted target type, canonical indicator values, allowed parameters), so adapters never
receive unvalidated input. :class:`HttpTool` adds timeouts, TLS verification, a response-size
cap and uniform mapping of HTTP failures to typed errors whose messages contain no secrets.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from app.core.errors import (
    ToolAuthError,
    ToolError,
    ToolInputError,
    ToolRateLimitedError,
    ToolTimeoutError,
    ToolUnavailableError,
)
from app.core.threads import run_sync
from app.schemas.enums import ToolStatus
from app.schemas.tools import OperationSpec, ToolObservation, ToolRequest, ToolResult
from app.tools.artifacts import validate_input
from app.tools.catalog import TOOL_DESCRIPTIONS, operations_for

DEFAULT_MAX_RESPONSE_BYTES = 2_000_000


class SecurityTool(ABC):
    name: str
    synthetic = False

    def __init__(self) -> None:
        self.operations: dict[str, OperationSpec] = operations_for(self.name)
        self.description = TOOL_DESCRIPTIONS.get(self.name, self.name)

    async def invoke(self, request: ToolRequest) -> ToolResult:
        spec = self.operations.get(request.operation)
        if spec is None:
            raise ToolInputError(f"{self.name} has no operation '{request.operation[:50]}'")
        if request.target_type not in spec.accepts:
            raise ToolInputError(f"{spec.key} does not accept a '{request.target_type[:50]}' target")
        target = validate_input(request.target_type, request.target)
        context = {}
        for artifact in spec.requires:
            if artifact not in request.context:
                raise ToolInputError(f"{spec.key} requires a '{artifact}' input")
            context[artifact] = validate_input(artifact, request.context[artifact])
        try:
            params = spec.resolve_params(request.params)
        except ValueError as exc:
            raise ToolInputError(str(exc)) from None
        started = time.perf_counter()
        result = await self.run(spec, request.target_type, target, context, params)
        return result.model_copy(update={"duration_ms": (time.perf_counter() - started) * 1000})

    @abstractmethod
    async def run(
        self, spec: OperationSpec, target_type: str, target: str, context: dict[str, str], params: dict[str, Any]
    ) -> ToolResult:
        """Perform a validated request. Raise ToolError subclasses on failure."""

    async def health(self) -> dict[str, Any]:
        return {"status": "unknown", "detail": "no health check implemented"}

    async def aclose(self) -> None:  # noqa: B027 - optional hook
        """Release network resources."""

    def result(
        self,
        spec: OperationSpec,
        target_type: str,
        *,
        status: ToolStatus = ToolStatus.SUCCESS,
        outputs: Optional[dict[str, Any]] = None,
        observations: Optional[list[ToolObservation]] = None,
        findings: Optional[list[str]] = None,
        message: Optional[str] = None,
    ) -> ToolResult:
        return ToolResult(
            tool=self.name,
            operation=spec.name,
            target_type=target_type,
            status=status,
            synthetic=self.synthetic,
            outputs=outputs or {},
            observations=observations or [],
            findings=findings or [],
            message=message,
        )


@dataclass(frozen=True)
class HttpReply:
    status_code: int
    body: bytes

    def json(self) -> Any:
        import json

        try:
            return json.loads(self.body)
        except ValueError:
            raise ToolError("the tool returned a response that is not JSON") from None


class HttpChannel:
    """One HTTP(S) endpoint with timeouts, TLS verification, a size cap and typed error mapping."""

    def __init__(
        self,
        tool_name: str,
        *,
        base_url: str,
        timeout_seconds: float,
        verify_tls: bool = True,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        headers: Optional[dict[str, str]] = None,
        auth: Optional[httpx.Auth | tuple[str, str]] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self.tool_name = tool_name
        self.timeout_seconds = timeout_seconds
        self.max_response_bytes = max_response_bytes
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            auth=auth,
            verify=verify_tls,
            timeout=timeout_seconds,
            transport=transport,
            follow_redirects=False,
        )

    async def request(self, method: str, url: str, **kwargs: Any) -> HttpReply:
        """Send a request; map failures to typed errors. 4xx other than auth/rate-limit are returned."""
        name = self.tool_name
        try:
            async with self._client.stream(method, url, **kwargs) as response:
                chunks, size = [], 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > self.max_response_bytes:
                        raise ToolError(f"{name} response exceeded {self.max_response_bytes} bytes")
                    chunks.append(chunk)
                reply = HttpReply(response.status_code, b"".join(chunks))
        except httpx.TimeoutException:
            raise ToolTimeoutError(f"{name} did not respond within {self.timeout_seconds:g} seconds") from None
        except httpx.TransportError:
            raise ToolUnavailableError(f"cannot reach {name}") from None
        if reply.status_code in (401, 403):
            raise ToolAuthError(f"{name} rejected CADE's credentials (HTTP {reply.status_code})")
        if reply.status_code == 429:
            raise ToolRateLimitedError(f"{name} rate limit reached")
        if reply.status_code >= 500:
            raise ToolError(f"{name} returned HTTP {reply.status_code}")
        return reply

    async def aclose(self) -> None:
        await self._client.aclose()


class HttpTool(SecurityTool):
    """An adapter for a tool reached over HTTP(S) through one :class:`HttpChannel`."""

    def __init__(self, **channel_options: Any) -> None:
        super().__init__()
        self.http = HttpChannel(self.name, **channel_options)

    async def request(self, method: str, url: str, **kwargs: Any) -> HttpReply:
        return await self.http.request(method, url, **kwargs)

    async def aclose(self) -> None:
        await self.http.aclose()


class RateLimiter:
    """Token bucket. Waits up to ``max_wait_seconds`` for a token, otherwise raises (never queues forever)."""

    def __init__(
        self,
        requests_per_minute: float,
        *,
        max_wait_seconds: float = 15.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Any] = asyncio.sleep,
    ) -> None:
        self.capacity = max(1.0, requests_per_minute)
        self.rate = requests_per_minute / 60.0
        self.max_wait_seconds = max_wait_seconds
        self._clock = clock
        self._sleep = sleep
        self._tokens = self.capacity
        self._updated = clock()
        self._lock = asyncio.Lock()

    async def acquire(self, tool: str) -> None:
        async with self._lock:
            now = self._clock()
            self._tokens = min(self.capacity, self._tokens + (now - self._updated) * self.rate)
            self._updated = now
            if self._tokens < 1:
                wait = (1 - self._tokens) / self.rate
                if wait > self.max_wait_seconds:
                    raise ToolRateLimitedError(f"{tool} client-side rate limit reached; retry in {wait:.0f}s")
                await self._sleep(wait)
                self._tokens, self._updated = 1.0, self._clock()
            self._tokens -= 1


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[[list[str], float], CommandResult]


def run_command(args: list[str], timeout_seconds: float) -> CommandResult:
    """Run a program with an argument list (never through a shell) and a hard timeout."""
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, shell=False
            args, capture_output=True, text=True, timeout=timeout_seconds, shell=False, check=False
        )
    except FileNotFoundError:
        raise ToolUnavailableError(f"program '{args[0]}' was not found") from None
    except subprocess.TimeoutExpired:
        raise ToolTimeoutError(f"'{args[0]}' did not finish within {timeout_seconds:g} seconds") from None
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


class CommandTool(SecurityTool):
    """An adapter for a local command-line tool, run off the event loop with a timeout."""

    def __init__(self, *, timeout_seconds: float, runner: Optional[CommandRunner] = None,
                 max_output_chars: int = DEFAULT_MAX_RESPONSE_BYTES) -> None:  # fmt: skip
        super().__init__()
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars
        self._runner = runner or run_command

    async def execute(self, args: list[str]) -> CommandResult:
        result = await run_sync(self._runner, args, self.timeout_seconds)
        if len(result.stdout) > self.max_output_chars:
            raise ToolError(f"{self.name} output exceeded {self.max_output_chars} characters")
        return result
