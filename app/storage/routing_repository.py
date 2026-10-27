"""Persistence for routing decisions and background jobs."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select, update
from sqlalchemy.exc import SQLAlchemyError

from app.core.errors import NotFoundError, StorageError
from app.core.logging import get_logger, log_event
from app.schemas.jobs import Job, JobStatus
from app.schemas.routing import RoutingDecision, StoredRoutingDecision
from app.storage.database import Database
from app.storage.models import JobRecord, RoutingDecisionRecord

logger = get_logger("storage")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _error(operation: str, exc: Exception) -> StorageError:
    log_event(logger, "database operation failed", level=logging.ERROR, operation=operation, stage="storage",
              outcome="failure", error_category="storage_error", exception_type=type(exc).__name__)  # fmt: skip
    return StorageError("database operation failed")


class RoutingRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def record(self, decision: RoutingDecision, created_by: Optional[str]) -> StoredRoutingDecision:
        now = _utcnow()
        record = RoutingDecisionRecord(incident_id=decision.incident_id, action=decision.action.value,
                                       decision_json=decision.model_dump_json(), created_by=created_by, created_at=now)  # fmt: skip
        try:
            with self.db.session() as session:
                session.add(record)
                session.flush()
                return StoredRoutingDecision(**decision.model_dump(), decision_id=record.id, created_at=now,
                                             created_by=created_by)  # fmt: skip
        except SQLAlchemyError as exc:
            raise _error("record_routing_decision", exc) from None

    def list(self, incident_id: str) -> list[StoredRoutingDecision]:
        try:
            with self.db.session() as session:
                query = (select(RoutingDecisionRecord).where(RoutingDecisionRecord.incident_id == incident_id)
                         .order_by(RoutingDecisionRecord.id))  # fmt: skip
                return [StoredRoutingDecision(**json.loads(r.decision_json), decision_id=r.id, created_at=r.created_at,
                                              created_by=r.created_by) for r in session.scalars(query)]  # fmt: skip
        except SQLAlchemyError as exc:
            raise _error("list_routing_decisions", exc) from None


class JobRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, job_id: str, kind: str, incident_id: Optional[str], created_by: Optional[str]) -> Job:
        now = _utcnow()
        record = JobRecord(id=job_id, kind=kind, incident_id=incident_id, status=JobStatus.QUEUED.value,
                           created_by=created_by, created_at=now, updated_at=now)  # fmt: skip
        try:
            with self.db.session() as session:
                session.add(record)
                session.flush()
                return self._to_job(record)
        except SQLAlchemyError as exc:
            raise _error("create_job", exc) from None

    def update(self, job_id: str, *, status: Optional[JobStatus] = None, progress: Optional[str] = None,
               result: Optional[dict[str, Any]] = None, error: Optional[dict[str, Any]] = None) -> None:  # fmt: skip
        values: dict[str, Any] = {"updated_at": _utcnow()}
        if status is not None:
            values["status"] = status.value
        if progress is not None:
            values["progress"] = progress[:300]
        if result is not None:
            values["result_json"] = json.dumps(result, default=str)
        if error is not None:
            values["error_json"] = json.dumps(error)
        try:
            with self.db.session() as session:
                session.execute(update(JobRecord).where(JobRecord.id == job_id).values(**values))
        except SQLAlchemyError as exc:
            raise _error("update_job", exc) from None

    def get(self, job_id: str) -> Job:
        try:
            with self.db.session() as session:
                record = session.get(JobRecord, job_id)
                if record is None:
                    raise NotFoundError("job not found")
                return self._to_job(record)
        except SQLAlchemyError as exc:
            raise _error("get_job", exc) from None

    def active_for(self, incident_id: str, kind: str) -> Optional[Job]:
        try:
            with self.db.session() as session:
                record = session.scalars(select(JobRecord).where(
                    JobRecord.incident_id == incident_id, JobRecord.kind == kind,
                    JobRecord.status.in_([JobStatus.QUEUED.value, JobStatus.RUNNING.value]))).first()  # fmt: skip
                return self._to_job(record) if record else None
        except SQLAlchemyError as exc:
            raise _error("active_job", exc) from None

    def interrupt_unfinished(self) -> int:
        """At startup: jobs left queued/running by a previous process can never finish."""
        try:
            with self.db.session() as session:
                result = session.execute(update(JobRecord).where(
                    JobRecord.status.in_([JobStatus.QUEUED.value, JobStatus.RUNNING.value])).values(
                    status=JobStatus.INTERRUPTED.value, updated_at=_utcnow(),
                    error_json=json.dumps({"category": "interrupted", "message": "the server stopped during the job"})))  # fmt: skip
                return result.rowcount or 0
        except SQLAlchemyError as exc:
            raise _error("interrupt_jobs", exc) from None

    @staticmethod
    def _to_job(record: JobRecord) -> Job:
        return Job(job_id=record.id, kind=record.kind, incident_id=record.incident_id, status=JobStatus(record.status),
                   progress=record.progress, result=json.loads(record.result_json) if record.result_json else None,
                   error=json.loads(record.error_json) if record.error_json else None, created_by=record.created_by,
                   created_at=record.created_at, updated_at=record.updated_at)  # fmt: skip
