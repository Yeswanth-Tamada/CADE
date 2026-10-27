"""Persistence for response plans and playbook runs."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.exc import SQLAlchemyError

from app.core.errors import ExecutionBlockedError, IncidentNotFoundError, NotFoundError, StorageError
from app.core.logging import get_logger, log_event
from app.schemas.response import PlanStatus, PlaybookRun, ResponsePlan
from app.storage.database import Database
from app.storage.models import IncidentRecord, PlaybookRunRecord, ResponsePlanRecord

logger = get_logger("storage")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ResponseRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    # --- plans ------------------------------------------------------------------------------

    def create_plan(self, plan: ResponsePlan) -> ResponsePlan:
        try:
            with self.db.session() as session:
                if session.get(IncidentRecord, plan.incident_id) is None:
                    raise IncidentNotFoundError("incident not found", incident_id=plan.incident_id)
                session.add(ResponsePlanRecord(id=plan.plan_id, incident_id=plan.incident_id, status=plan.status.value,
                                               plan_json=plan.model_dump_json(), created_at=plan.created_at,
                                               updated_at=plan.updated_at))  # fmt: skip
            return plan
        except SQLAlchemyError as exc:
            raise self._error("create_plan", exc) from None

    def get_plan(self, plan_id: str) -> ResponsePlan:
        try:
            with self.db.session() as session:
                record = session.get(ResponsePlanRecord, plan_id)
                if record is None:
                    raise NotFoundError("response plan not found")
                return self._plan(record)
        except SQLAlchemyError as exc:
            raise self._error("get_plan", exc) from None

    def list_plans(self, incident_id: str) -> list[ResponsePlan]:
        try:
            with self.db.session() as session:
                query = select(ResponsePlanRecord).where(ResponsePlanRecord.incident_id == incident_id)
                return [self._plan(r) for r in session.scalars(query.order_by(ResponsePlanRecord.created_at))]
        except SQLAlchemyError as exc:
            raise self._error("list_plans", exc) from None

    def save_plan(self, plan: ResponsePlan, *, expected_status: Optional[set[PlanStatus]] = None) -> ResponsePlan:
        """Persist a modified plan; with ``expected_status`` the write only succeeds from those states."""
        plan = plan.model_copy(update={"updated_at": _utcnow()})
        try:
            with self.db.session() as session:
                statement = update(ResponsePlanRecord).where(ResponsePlanRecord.id == plan.plan_id)
                if expected_status is not None:
                    statement = statement.where(ResponsePlanRecord.status.in_([s.value for s in expected_status]))
                result = session.execute(statement.values(status=plan.status.value, plan_json=plan.model_dump_json(),
                                                          updated_at=plan.updated_at))  # fmt: skip
                if result.rowcount == 0:
                    if session.get(ResponsePlanRecord, plan.plan_id) is None:
                        raise NotFoundError("response plan not found")
                    raise ExecutionBlockedError("the response plan changed state; reload it and try again")
            return plan
        except SQLAlchemyError as exc:
            raise self._error("save_plan", exc) from None

    # --- runs -------------------------------------------------------------------------------

    def save_run(self, run: PlaybookRun) -> PlaybookRun:
        now = _utcnow()
        try:
            with self.db.session() as session:
                record = session.get(PlaybookRunRecord, run.run_id)
                if record is None:
                    session.add(PlaybookRunRecord(id=run.run_id, plan_id=run.plan_id, incident_id=run.incident_id,
                                                  status=run.status.value, run_json=run.model_dump_json(),
                                                  created_at=now, updated_at=now))  # fmt: skip
                else:
                    record.status, record.run_json, record.updated_at = run.status.value, run.model_dump_json(), now
            return run
        except SQLAlchemyError as exc:
            raise self._error("save_run", exc) from None

    def get_run(self, run_id: str) -> PlaybookRun:
        try:
            with self.db.session() as session:
                record = session.get(PlaybookRunRecord, run_id)
                if record is None:
                    raise NotFoundError("playbook run not found")
                return PlaybookRun.model_validate_json(record.run_json)
        except SQLAlchemyError as exc:
            raise self._error("get_run", exc) from None

    def list_runs(self, incident_id: str) -> list[PlaybookRun]:
        try:
            with self.db.session() as session:
                query = select(PlaybookRunRecord).where(PlaybookRunRecord.incident_id == incident_id)
                return [PlaybookRun.model_validate_json(r.run_json)
                        for r in session.scalars(query.order_by(PlaybookRunRecord.created_at))]  # fmt: skip
        except SQLAlchemyError as exc:
            raise self._error("list_runs", exc) from None

    @staticmethod
    def _plan(record: ResponsePlanRecord) -> ResponsePlan:
        return ResponsePlan.model_validate_json(record.plan_json).model_copy(update={"status": PlanStatus(record.status)})

    @staticmethod
    def _error(operation: str, exc: Exception) -> StorageError:
        log_event(logger, "database operation failed", level=logging.ERROR, operation=operation, stage="storage",
                  outcome="failure", error_category="storage_error", exception_type=type(exc).__name__)  # fmt: skip
        return StorageError("database operation failed")
