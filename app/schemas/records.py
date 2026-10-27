"""Read models returned by the repository and the API (decoupled from ORM classes)."""

from __future__ import annotations

from typing import Optional

from pydantic import AwareDatetime, BaseModel, Field, model_validator

from app.schemas.enums import ExtractionStatus, IncidentStatus
from app.schemas.incident import CanonicalIncident, NormalizedIncident


class StoredIncident(NormalizedIncident):
    status: IncidentStatus
    created_at: AwareDatetime
    updated_at: AwareDatetime
    analysis_started_at: Optional[AwareDatetime] = None


class IncidentSummary(BaseModel):
    incident_id: str
    title: Optional[str] = None
    source: str
    severity: Optional[str] = None
    status: IncidentStatus
    received_at: AwareDatetime
    created_at: AwareDatetime


class AnalysisError(BaseModel):
    category: str
    message: str


class StoredAnalysis(BaseModel):
    analysis_id: int
    incident_id: str
    extraction_status: ExtractionStatus
    provider: str
    model_name: str
    attempts: int = Field(ge=0)
    duration_ms: float = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)
    error: Optional[AnalysisError] = None
    result: Optional[CanonicalIncident] = None
    created_at: AwareDatetime

    @model_validator(mode="after")
    def _consistency(self) -> "StoredAnalysis":
        # Regression guard: a failed extraction can never carry a result, and a
        # successful one must carry one.
        if self.extraction_status == ExtractionStatus.FAILED:
            if self.result is not None or self.error is None:
                raise ValueError("failed analyses must have an error and no result")
        elif self.result is None:
            raise ValueError("successful analyses must include a result")
        return self


class IncidentWithAnalysis(BaseModel):
    incident: StoredIncident
    latest_analysis: Optional[StoredAnalysis] = None
