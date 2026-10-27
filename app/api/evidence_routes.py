"""Evidence enrichment and analyst-supplied evidence. Thin wrappers around ``EvidenceService``."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Body, Request

from app.api.routes import IncidentIdPath
from app.schemas.evidence import (
    AnalystEvidenceSubmission,
    EffectiveEvidenceProfile,
    EnrichmentReport,
    EnrichmentRequest,
    EvidenceAddition,
)
from app.schemas.tools import ToolInfo
from app.security.auth import Principal, require
from app.security.roles import Permission
from app.services.evidence_service import EvidenceService

router = APIRouter(tags=["evidence"])

_RESPONSES = {
    401: {"description": "Missing, invalid or revoked API key"},
    403: {"description": "The API key's roles do not allow this operation"},
    404: {"description": "Incident not found"},
    409: {"description": "The incident has no usable analysis yet"},
}


def _service(request: Request) -> EvidenceService:
    return request.app.state.evidence_service


@router.get("/incidents/{incident_id}/evidence", response_model=EffectiveEvidenceProfile, responses=_RESPONSES)
async def get_evidence(
    request: Request, incident_id: str = IncidentIdPath, principal: Principal = require(Permission.READ)
) -> EffectiveEvidenceProfile:
    """The latest usable analysis' evidence profile plus all evidence added since.

    ``simulated`` observations come from simulated tools: they are synthetic and are never
    counted as observed evidence.
    """
    return await _service(request).effective_profile(incident_id)


@router.post("/incidents/{incident_id}/enrich", response_model=EnrichmentReport, responses=_RESPONSES)
async def enrich(
    request: Request,
    incident_id: str = IncidentIdPath,
    body: Optional[EnrichmentRequest] = Body(default=None),
    principal: Principal = require(Permission.ENRICH_EVIDENCE),
) -> EnrichmentReport:
    """Ask the configured tools (or their simulators) for missing / not-checked evidence."""
    return await _service(request).enrich(incident_id, body.categories if body else None, principal)


@router.post("/incidents/{incident_id}/evidence", status_code=201, response_model=EvidenceAddition,
             responses=_RESPONSES)  # fmt: skip
async def add_evidence(
    request: Request,
    body: AnalystEvidenceSubmission,
    incident_id: str = IncidentIdPath,
    principal: Principal = require(Permission.ENRICH_EVIDENCE),
) -> EvidenceAddition:
    """Record evidence an analyst obtained elsewhere; it is attributed to the API key's analyst."""
    return await _service(request).add_analyst_evidence(incident_id, body, principal)


@router.get("/tools", response_model=list[ToolInfo], responses={401: _RESPONSES[401], 403: _RESPONSES[403]})
async def list_tools(request: Request, principal: Principal = require(Permission.READ)) -> list[ToolInfo]:
    """Supported tools, their operations, and whether calls currently go to a real tool or a simulator."""
    return request.app.state.tool_gateway.registry.info()
