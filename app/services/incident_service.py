"""Use-case layer: coordinates ingestion, persistence and extraction.

API routes call this service and contain no business logic; the service never builds HTTP
responses. Blocking database calls run in a worker thread so the event loop stays free
while the LLM is working.
"""

from __future__ import annotations

import logging
import time
from datetime import timedelta
from typing import TYPE_CHECKING, Optional

from app.core.errors import CADEError
from app.core.logging import get_logger, log_event
from app.core.threads import run_sync as _run_sync
from app.ingestion.input_service import IncidentInputService
from app.intent.llm_extractor import IntentExtractor
from app.schemas.enums import ExtractionStatus
from app.schemas.records import (
    AnalysisError,
    IncidentSummary,
    IncidentWithAnalysis,
    StoredAnalysis,
    StoredIncident,
)
from app.storage.repository import IncidentRepository

if TYPE_CHECKING:
    from app.security.auth import Principal
    from app.services.audit_service import AuditService

logger = get_logger("service")


class IncidentService:
    def __init__(
        self,
        repository: IncidentRepository,
        input_service: IncidentInputService,
        extractor: IntentExtractor,
        stale_analysis_after: timedelta,
        audit: Optional["AuditService"] = None,
    ) -> None:
        self.repository = repository
        self.input_service = input_service
        self.extractor = extractor
        self.stale_analysis_after = stale_analysis_after
        self.audit = audit

    async def _audit(self, actor: Optional["Principal"], action: str, incident_id: str, outcome: str, **details) -> None:
        """Best effort: the action already happened, so a failed audit write must not mask its result.

        (Approval and execution audits are different -- they are written *before* acting and are fatal.)
        """
        if self.audit is None:
            return
        try:
            await self.audit.record(actor, action, outcome=outcome, target_type="incident", target_id=incident_id,
                                    details=details)  # fmt: skip
        except CADEError:
            pass  # StorageError is already logged by the repository

    async def create_incident(
        self, body: bytes, content_type: str | None, actor: Optional["Principal"] = None
    ) -> StoredIncident:
        started = time.perf_counter()
        incident = self.input_service.ingest(body, content_type)
        stored = await _run_sync(self.repository.create_incident, incident)
        await self._audit(actor, "incident.create", stored.incident_id, "success", source=stored.source.value)
        log_event(
            logger,
            "incident created",
            incident_id=stored.incident_id,
            operation="create_incident",
            stage="ingestion",
            outcome="success",
            duration_ms=(time.perf_counter() - started) * 1000,
            source=stored.source.value,
            description_chars=len(stored.raw_description),
        )
        return stored

    async def list_incidents(self, limit: int = 50) -> list[IncidentSummary]:
        return await _run_sync(self.repository.list_incidents, limit)

    async def get_incident(self, incident_id: str) -> IncidentWithAnalysis:
        incident = await _run_sync(self.repository.get_incident, incident_id)
        latest = await _run_sync(self.repository.get_latest_analysis, incident_id)
        return IncidentWithAnalysis(incident=incident, latest_analysis=latest)

    async def analyze(self, incident_id: str, actor: Optional["Principal"] = None) -> StoredAnalysis:
        incident = await _run_sync(
            self.repository.start_analysis, incident_id, stale_after=self.stale_analysis_after
        )
        started = time.perf_counter()
        llm = self.extractor.llm
        log_event(logger, "analysis started", incident_id=incident_id, operation="analyze", stage="extraction")

        try:
            canonical = await self.extractor.extract(incident)
        except Exception as exc:
            await self._record_failure(incident_id, exc, (time.perf_counter() - started) * 1000)
            category = exc.category if isinstance(exc, CADEError) else "internal_error"
            await self._audit(actor, "incident.analyze", incident_id, "failure", error_category=category)
            raise

        analysis = await _run_sync(
            self.repository.record_analysis,
            incident_id,
            extraction_status=canonical.analysis.extraction_status,
            provider=llm.provider,
            model_name=llm.model,
            attempts=canonical.analysis.attempts,
            duration_ms=canonical.analysis.duration_ms,
            warnings=canonical.analysis.warnings,
            result=canonical,
        )
        log_event(
            logger,
            "analysis completed",
            incident_id=incident_id,
            operation="analyze",
            stage="complete",
            outcome=analysis.extraction_status.value,
            duration_ms=analysis.duration_ms,
            intent=canonical.intent.label,
            warning_count=len(analysis.warnings),
            provider=llm.provider,
        )
        await self._audit(actor, "incident.analyze", incident_id, analysis.extraction_status.value,
                          analysis_id=analysis.analysis_id, provider=llm.provider)  # fmt: skip
        return analysis

    async def _record_failure(self, incident_id: str, exc: Exception, duration_ms: float) -> None:
        attempts, warnings = 0, []
        if isinstance(exc, CADEError):
            error = AnalysisError(category=exc.category, message=exc.message)
            attempts, warnings = exc.attempts, exc.warnings
        else:
            error = AnalysisError(category="internal_error", message="unexpected error during analysis")
        log_event(
            logger,
            "analysis failed",
            level=logging.ERROR if error.category == "internal_error" else logging.WARNING,
            incident_id=incident_id,
            operation="analyze",
            stage="extraction",
            outcome="failure",
            duration_ms=duration_ms,
            error_category=error.category,
            exception_type=type(exc).__name__,
        )
        try:
            await _run_sync(
                self.repository.record_analysis,
                incident_id,
                extraction_status=ExtractionStatus.FAILED,
                provider=self.extractor.llm.provider,
                model_name=self.extractor.llm.model,
                attempts=attempts,
                duration_ms=duration_ms,
                warnings=warnings,
                error=error,
            )
        except CADEError:
            # The original error is more useful to the caller; the storage failure is already logged.
            pass
