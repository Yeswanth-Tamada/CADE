"""HTTP endpoints. Thin wrappers around :class:`~app.services.incident_service.IncidentService`.

Every route except ``/health`` requires an API key with the permission named in its
``require(...)`` dependency (see ``app/security``).
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Path, Query, Request
from fastapi.responses import JSONResponse

from app import __version__
from app.core.errors import PayloadTooLargeError
from app.schemas.incident import INCIDENT_ID_PATTERN
from app.schemas.records import IncidentSummary, IncidentWithAnalysis, StoredAnalysis, StoredIncident
from app.schemas.security import AuditEvent
from app.security.auth import Principal, optional_principal, require
from app.security.roles import Permission
from app.services.incident_service import IncidentService

router = APIRouter()

IncidentIdPath = Path(pattern=INCIDENT_ID_PATTERN, description="Incident identifier, e.g. INC-1A2B3C4D5E6F7A8B")

_ERROR_RESPONSES = {
    401: {"description": "Missing, invalid or revoked API key"},
    403: {"description": "The API key's roles do not allow this operation"},
    404: {"description": "Incident not found"},
    422: {"description": "Invalid input"},
}

_CREATE_REQUEST_BODY = {
    "requestBody": {
        "required": True,
        "content": {
            "application/json": {"schema": {"$ref": "#/components/schemas/IncidentSubmission"}},
            "text/plain": {
                "schema": {"type": "string"},
                "example": "SYNTHETIC: Analyst report - user jdoe on WS-FIN-042 opened an invoice attachment; "
                "EDR then saw powershell.exe -enc ... connecting to 203.0.113.45.",
            },
        },
    }
}


def _service(request: Request) -> IncidentService:
    return request.app.state.incident_service


@router.get("/health", tags=["health"])
async def health(request: Request, principal: Optional[Principal] = Depends(optional_principal)) -> JSONResponse:
    """API health plus the state of each dependency.

    Returns 200 while the API and its database work, even if optional dependencies (the LLM,
    the ATT&CK dataset) are unavailable -- those are reported individually and make the
    overall status ``degraded``. Returns 503 if the database is unavailable.

    Public, but callers without a valid API key only see the overall status: dependency
    details (model names, dataset versions, tool configuration) are for authenticated users.
    """
    state = request.app.state
    settings = state.settings
    db_ok = state.database.ping()
    llm = await state.llm_client.health()
    attack = state.attack_mapper.status()

    all_ok = db_ok and llm.status == "ok" and attack["status"] == "loaded"
    body: dict = {
        "status": "ok" if all_ok else ("degraded" if db_ok else "unavailable"),
        "api": {"status": "ok", "version": __version__},
    }
    if principal is not None and principal.can(Permission.READ):
        body["api"]["environment"] = settings.app_env.value
        body["dependencies"] = {
            "database": {"status": "ok" if db_ok else "unavailable", "required": True},
            "llm": {
                "status": llm.status,
                "required_for": "POST /incidents/{id}/analyze",
                "provider": llm.provider,
                "model": llm.model,
                "reachable": llm.reachable,
                "model_available": llm.model_available,
                "detail": llm.detail,
            },
            "attack_dataset": attack,
        }
        components = getattr(state, "components", None)
        if components is not None:
            body["playbooks"] = {
                "execution_mode": settings.execution_mode.value,
                "generator_model": components.generator_llm.model if components.generator_llm else None,
                "judge": components.judge_llm.model if components.judge_llm else "heuristic",
                "library_entries": len(components.library),
                "embedder": components.library.embedder.name,
                "vector_index": components.library.index.name,
                "warnings": [note for note in components.notes.values() if note],
                "tools": {t.name: t.effective for t in state.tool_gateway.registry.info()},
            }
    return JSONResponse(status_code=200 if db_ok else 503, content=body)


@router.post(
    "/incidents",
    status_code=201,
    response_model=StoredIncident,
    tags=["incidents"],
    responses={**_ERROR_RESPONSES, 409: {"description": "Incident ID already exists"},
               413: {"description": "Body too large"}, 415: {"description": "Unsupported content type"}},
    openapi_extra=_CREATE_REQUEST_BODY,
)  # fmt: skip
async def create_incident(
    request: Request, principal: Principal = require(Permission.SUBMIT_INCIDENT)
) -> StoredIncident:
    """Submit an incident as JSON (``application/json``) or free text (``text/plain``)."""
    max_bytes = request.app.state.settings.max_description_chars * 4 + 256_000
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > max_bytes:
        raise PayloadTooLargeError("request body is too large")
    body = await request.body()
    if len(body) > max_bytes:
        raise PayloadTooLargeError("request body is too large")
    return await _service(request).create_incident(body, request.headers.get("content-type"), actor=principal)


@router.get("/incidents", response_model=list[IncidentSummary], tags=["incidents"], responses=_ERROR_RESPONSES)
async def list_incidents(
    request: Request, limit: int = Query(default=50, ge=1, le=500), principal: Principal = require(Permission.READ)
) -> list[IncidentSummary]:
    """Most recent incidents first (summaries only)."""
    return await _service(request).list_incidents(limit)


@router.get(
    "/incidents/{incident_id}", response_model=IncidentWithAnalysis, tags=["incidents"], responses=_ERROR_RESPONSES
)
async def get_incident(
    request: Request, incident_id: str = IncidentIdPath, principal: Principal = require(Permission.READ)
) -> IncidentWithAnalysis:
    """Return a stored incident and its most recent analysis (if any)."""
    return await _service(request).get_incident(incident_id)


@router.post(
    "/incidents/{incident_id}/analyze",
    response_model=StoredAnalysis,
    tags=["incidents"],
    responses={
        **_ERROR_RESPONSES,
        409: {"description": "Analysis already in progress"},
        502: {"description": "The LLM returned unusable output (failure is recorded)"},
        503: {"description": "The LLM is unavailable (failure is recorded)"},
        504: {"description": "The LLM timed out (failure is recorded)"},
    },
)
async def analyze_incident(
    request: Request, incident_id: str = IncidentIdPath, principal: Principal = require(Permission.ANALYZE_INCIDENT)
) -> StoredAnalysis:
    """Run intent extraction, evidence profiling and ATT&CK mapping, then store the result."""
    return await _service(request).analyze(incident_id, actor=principal)


@router.get("/audit", response_model=list[AuditEvent], tags=["audit"], responses=_ERROR_RESPONSES)
async def list_audit(
    request: Request,
    target_id: Optional[str] = Query(default=None, max_length=100),
    action: Optional[str] = Query(default=None, max_length=100),
    limit: int = Query(default=100, ge=1, le=1000),
    principal: Principal = require(Permission.READ_AUDIT),
) -> list[AuditEvent]:
    """Most recent audit events first (append-only log of security-relevant actions)."""
    return await request.app.state.audit.list(target_id=target_id, action=action, limit=limit)
