"""Persistence operations for incidents and analyses.

Each public method runs in its own transaction and returns Pydantic read models, never ORM
objects, so callers cannot accidentally lazy-load or mutate database state.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import or_, select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.core.errors import IncidentConflictError, IncidentNotFoundError, StorageError
from app.core.logging import get_logger, log_event
from app.schemas.enums import ExtractionStatus, IncidentStatus
from app.schemas.incident import CanonicalIncident, NormalizedIncident
from app.schemas.records import AnalysisError, IncidentSummary, StoredAnalysis, StoredIncident
from app.storage.database import Database
from app.storage.models import AnalysisRecord, IncidentRecord

logger = get_logger("storage")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class IncidentRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    # --- incidents -----------------------------------------------------------------

    def create_incident(self, incident: NormalizedIncident) -> StoredIncident:
        now = _utcnow()
        record = IncidentRecord(
            id=incident.incident_id,
            source=incident.source.value,
            title=incident.title,
            severity=incident.severity.value if incident.severity else None,
            status=IncidentStatus.RECEIVED.value,
            raw_description=incident.raw_description,
            normalized_json=incident.model_dump_json(),
            received_at=incident.received_at,
            occurred_at=incident.occurred_at,
            created_at=now,
            updated_at=now,
        )
        try:
            with self.db.session() as session:
                session.add(record)
        except IntegrityError:
            raise IncidentConflictError(
                "an incident with this ID already exists", incident_id=incident.incident_id
            ) from None
        except SQLAlchemyError as exc:
            raise self._storage_error("create_incident", exc, incident.incident_id) from None
        return self._to_incident(record)

    def get_incident(self, incident_id: str) -> StoredIncident:
        try:
            with self.db.session() as session:
                record = session.get(IncidentRecord, incident_id)
                if record is None:
                    raise IncidentNotFoundError("incident not found", incident_id=incident_id)
                return self._to_incident(record)
        except SQLAlchemyError as exc:
            raise self._storage_error("get_incident", exc, incident_id) from None

    def list_incidents(self, limit: int = 50) -> list[IncidentSummary]:
        """Most recently created first (summaries only: descriptions can be large and sensitive)."""
        try:
            with self.db.session() as session:
                records = session.scalars(
                    select(IncidentRecord).order_by(IncidentRecord.created_at.desc()).limit(limit)
                ).all()
                return [
                    IncidentSummary(incident_id=r.id, title=r.title, source=r.source, severity=r.severity,
                                    status=IncidentStatus(r.status), received_at=r.received_at, created_at=r.created_at)
                    for r in records
                ]  # fmt: skip
        except SQLAlchemyError as exc:
            raise self._storage_error("list_incidents", exc, None) from None

    def start_analysis(self, incident_id: str, *, stale_after: timedelta) -> StoredIncident:
        """Atomically mark an incident as ``analyzing``.

        Fails with a conflict if another analysis is already running, unless that one has
        been running longer than ``stale_after`` (e.g. the server crashed mid-analysis).
        """
        now = _utcnow()
        try:
            with self.db.session() as session:
                result = session.execute(
                    update(IncidentRecord)
                    .where(IncidentRecord.id == incident_id)
                    .where(
                        or_(
                            IncidentRecord.status != IncidentStatus.ANALYZING.value,
                            IncidentRecord.analysis_started_at.is_(None),
                            IncidentRecord.analysis_started_at < now - stale_after,
                        )
                    )
                    .values(status=IncidentStatus.ANALYZING.value, analysis_started_at=now, updated_at=now)
                )
                if result.rowcount == 0:
                    if session.get(IncidentRecord, incident_id) is None:
                        raise IncidentNotFoundError("incident not found", incident_id=incident_id)
                    raise IncidentConflictError("an analysis is already in progress", incident_id=incident_id)
                record = session.get(IncidentRecord, incident_id)
                session.refresh(record)
                return self._to_incident(record)
        except SQLAlchemyError as exc:
            raise self._storage_error("start_analysis", exc, incident_id) from None

    # --- analyses ------------------------------------------------------------------

    def record_analysis(
        self,
        incident_id: str,
        *,
        extraction_status: ExtractionStatus,
        provider: str,
        model_name: str,
        attempts: int,
        duration_ms: float,
        warnings: list[str],
        result: Optional[CanonicalIncident] = None,
        error: Optional[AnalysisError] = None,
    ) -> StoredAnalysis:
        """Store an analysis outcome and update the incident status in one transaction."""
        now = _utcnow()
        # Validate before touching the database: a failed analysis can never carry a result.
        StoredAnalysis(
            analysis_id=0,
            incident_id=incident_id,
            extraction_status=extraction_status,
            provider=provider,
            model_name=model_name,
            attempts=attempts,
            duration_ms=duration_ms,
            warnings=warnings,
            error=error,
            result=result,
            created_at=now,
        )
        new_status = (
            IncidentStatus.ANALYSIS_FAILED if extraction_status == ExtractionStatus.FAILED else IncidentStatus.ANALYZED
        )
        record = AnalysisRecord(
            incident_id=incident_id,
            extraction_status=extraction_status.value,
            provider=provider,
            model_name=model_name,
            attempts=attempts,
            duration_ms=duration_ms,
            warnings_json=json.dumps(warnings),
            error_category=error.category if error else None,
            error_message=error.message[:1000] if error else None,
            canonical_json=result.model_dump_json() if result else None,
            created_at=now,
        )
        try:
            with self.db.session() as session:
                incident = session.get(IncidentRecord, incident_id)
                if incident is None:
                    raise IncidentNotFoundError("incident not found", incident_id=incident_id)
                session.add(record)
                incident.status = new_status.value
                incident.updated_at = now
                incident.analysis_started_at = None
                session.flush()
                return self._to_analysis(record)
        except SQLAlchemyError as exc:
            raise self._storage_error("record_analysis", exc, incident_id) from None

    def get_latest_analysis(self, incident_id: str) -> Optional[StoredAnalysis]:
        try:
            with self.db.session() as session:
                record = session.scalars(
                    select(AnalysisRecord)
                    .where(AnalysisRecord.incident_id == incident_id)
                    .order_by(AnalysisRecord.id.desc())
                    .limit(1)
                ).first()
                return self._to_analysis(record) if record else None
        except SQLAlchemyError as exc:
            raise self._storage_error("get_latest_analysis", exc, incident_id) from None

    def get_latest_usable_analysis(self, incident_id: str) -> Optional[StoredAnalysis]:
        """The most recent analysis that produced a result (``success`` or ``partial``)."""
        try:
            with self.db.session() as session:
                record = session.scalars(
                    select(AnalysisRecord)
                    .where(AnalysisRecord.incident_id == incident_id)
                    .where(AnalysisRecord.canonical_json.is_not(None))
                    .order_by(AnalysisRecord.id.desc())
                    .limit(1)
                ).first()
                return self._to_analysis(record) if record else None
        except SQLAlchemyError as exc:
            raise self._storage_error("get_latest_usable_analysis", exc, incident_id) from None

    # --- helpers -------------------------------------------------------------------

    @staticmethod
    def _to_incident(record: IncidentRecord) -> StoredIncident:
        normalized = json.loads(record.normalized_json)
        return StoredIncident(
            **normalized,
            status=IncidentStatus(record.status),
            created_at=record.created_at,
            updated_at=record.updated_at,
            analysis_started_at=record.analysis_started_at,
        )

    @staticmethod
    def _to_analysis(record: AnalysisRecord) -> StoredAnalysis:
        return StoredAnalysis(
            analysis_id=record.id,
            incident_id=record.incident_id,
            extraction_status=ExtractionStatus(record.extraction_status),
            provider=record.provider,
            model_name=record.model_name,
            attempts=record.attempts,
            duration_ms=record.duration_ms,
            warnings=json.loads(record.warnings_json or "[]"),
            error=AnalysisError(category=record.error_category, message=record.error_message or "")
            if record.error_category
            else None,
            result=CanonicalIncident.model_validate_json(record.canonical_json) if record.canonical_json else None,
            created_at=record.created_at,
        )

    @staticmethod
    def _storage_error(operation: str, exc: Exception, incident_id: Optional[str]) -> StorageError:
        log_event(
            logger,
            "database operation failed",
            level=logging.ERROR,
            incident_id=incident_id,
            operation=operation,
            stage="storage",
            outcome="failure",
            error_category="storage_error",
            exception_type=type(exc).__name__,
        )
        return StorageError("database operation failed", incident_id=incident_id)
