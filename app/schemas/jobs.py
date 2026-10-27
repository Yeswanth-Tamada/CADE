"""Background jobs and the outcome of a ``respond`` run."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import AwareDatetime, BaseModel, Field

from app.schemas.enums import EvidenceCategory
from app.schemas.evidence import EnrichmentReport
from app.schemas.response import PlaybookRun, ResponsePlan
from app.schemas.routing import StoredRoutingDecision


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"  # the server stopped while the job was running


class Job(BaseModel):
    job_id: str
    kind: str
    incident_id: Optional[str] = None
    status: JobStatus
    progress: Optional[str] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[dict[str, Any]] = None
    created_by: Optional[str] = None
    created_at: AwareDatetime
    updated_at: AwareDatetime


class ResponseOutcome(BaseModel):
    incident_id: str
    status: str  # "plan_created" | "awaiting_evidence"
    decisions: list[StoredRoutingDecision] = Field(default_factory=list)
    enrichment: list[EnrichmentReport] = Field(default_factory=list)
    evidence_requests: list[EvidenceCategory] = Field(default_factory=list)
    plan: Optional[ResponsePlan] = None
    forced_synthesis: bool = False


class IncidentResponse(BaseModel):
    """Everything the response pipeline recorded for one incident."""

    incident_id: str
    decisions: list[StoredRoutingDecision] = Field(default_factory=list)
    plans: list[ResponsePlan] = Field(default_factory=list)
    runs: list[PlaybookRun] = Field(default_factory=list)
