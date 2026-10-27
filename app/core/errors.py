"""Domain exceptions.

Each exception carries a stable ``category`` (used in logs and API responses) and the HTTP
status the API should return. Messages must be safe to show to API clients: never include
raw incident text, model output, stack traces or credentials.
"""

from __future__ import annotations

from typing import Any


class CADEError(Exception):
    category = "internal_error"
    http_status = 500

    def __init__(
        self,
        message: str,
        *,
        incident_id: str | None = None,
        details: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.incident_id = incident_id
        self.details = details or []
        # Filled in by the extractor so failed analyses can be recorded accurately.
        self.attempts = 0
        self.warnings: list[str] = []


class InvalidIncidentInputError(CADEError):
    category = "validation_error"
    http_status = 422


class UnsupportedMediaTypeError(CADEError):
    category = "unsupported_media_type"
    http_status = 415


class PayloadTooLargeError(CADEError):
    category = "payload_too_large"
    http_status = 413


class IncidentNotFoundError(CADEError):
    category = "not_found"
    http_status = 404


class NotFoundError(CADEError):
    """A non-incident object (plan, run, job, library entry) does not exist."""

    category = "not_found"
    http_status = 404


class ExecutionBlockedError(CADEError):
    """A safety rule refused the operation (already executed, invalid/unsafe DAG, ...)."""

    category = "execution_blocked"
    http_status = 409


class IncidentConflictError(CADEError):
    category = "conflict"
    http_status = 409


class AuthenticationError(CADEError):
    """No credentials, or credentials that do not identify an active API key."""

    category = "unauthenticated"
    http_status = 401


class AuthorizationError(CADEError):
    """The authenticated caller's roles do not allow the operation."""

    category = "forbidden"
    http_status = 403


class StorageError(CADEError):
    category = "storage_error"
    http_status = 503


class ConfigurationError(CADEError):
    category = "configuration_error"
    http_status = 500


# --- LLM / extraction failures -------------------------------------------------------


class LLMError(CADEError):
    """Base class for failures talking to the LLM provider."""

    category = "llm_error"
    http_status = 502


class LLMUnavailableError(LLMError):
    category = "llm_unavailable"
    http_status = 503


class LLMTimeoutError(LLMError):
    category = "llm_timeout"
    http_status = 504


class LLMModelNotFoundError(LLMError):
    category = "llm_model_not_found"
    http_status = 503


class LLMResponseError(LLMError):
    """The provider answered, but with an error status or an unusable envelope."""

    category = "llm_bad_response"
    http_status = 502


class LLMServerError(LLMResponseError):
    """The provider failed while generating (HTTP 5xx), e.g. constrained decoding hit a dead end.

    Unlike an unreachable or slow provider, this is worth one more attempt.
    """


class ExtractionFailedError(CADEError):
    """The model answered, but its output never passed schema validation within the retry budget."""

    category = "extraction_failed"
    http_status = 502


class PlaybookValidationError(CADEError):
    """A playbook is not a valid, safe DAG (details list the violated rules)."""

    category = "invalid_playbook"
    http_status = 422


class SynthesisError(CADEError):
    """Synthesis could not produce any admissible (valid and safe) playbook."""

    category = "synthesis_failed"
    http_status = 422


class NoUsableAnalysisError(CADEError):
    """The operation needs a successful (or partial) analysis of the incident first."""

    category = "no_usable_analysis"
    http_status = 409


# --- security tools -----------------------------------------------------------------------


class ToolError(CADEError):
    """A real security tool failed. Never replaced by a simulated result."""

    category = "tool_error"
    http_status = 502


class ToolUnavailableError(ToolError):
    category = "tool_unavailable"
    http_status = 503


class ToolTimeoutError(ToolError):
    category = "tool_timeout"
    http_status = 504


class ToolRateLimitedError(ToolError):
    category = "tool_rate_limited"
    http_status = 429


class ToolAuthError(ToolError):
    """The tool rejected CADE's credentials (check the configured key/password)."""

    category = "tool_auth_failed"
    http_status = 502


class ToolInputError(ToolError):
    """The input is not a valid indicator for this operation; nothing was sent to the tool."""

    category = "tool_input_invalid"
    http_status = 422
