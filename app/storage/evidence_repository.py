"""Persistence for evidence additions and tool-invocation records."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.core.errors import IncidentNotFoundError, StorageError
from app.core.logging import get_logger, log_event
from app.schemas.enums import EvidenceAdditionKind
from app.schemas.evidence import EvidenceAddition, SimulatedObservation
from app.schemas.incident import EvidenceItem
from app.schemas.tools import ToolResult
from app.storage.database import Database
from app.storage.models import EvidenceAdditionRecord, IncidentRecord, ToolInvocationRecord

logger = get_logger("storage")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class EvidenceRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def add(
        self,
        incident_id: str,
        *,
        items: list[tuple[EvidenceAdditionKind, EvidenceItem]] = (),
        simulated: list[SimulatedObservation] = (),
        created_by: Optional[str] = None,
    ) -> list[EvidenceAddition]:
        """Store new evidence in one transaction, skipping exact duplicates already stored."""
        now = _utcnow()
        try:
            with self.db.session() as session:
                if session.get(IncidentRecord, incident_id) is None:
                    raise IncidentNotFoundError("incident not found", incident_id=incident_id)
                existing = {
                    (r.kind, r.evidence_key)
                    for r in session.scalars(
                        select(EvidenceAdditionRecord).where(EvidenceAdditionRecord.incident_id == incident_id)
                    )
                }
                records = []
                rows = [(kind, item.category.value, item.evidence_id, item.model_dump_json()) for kind, item in items]
                rows += [
                    (EvidenceAdditionKind.SIMULATED, s.category.value,
                     EvidenceItem.make_id(s.category, f"{s.tool}.{s.operation}|{s.description}"), s.model_dump_json())
                    for s in simulated
                ]  # fmt: skip
                for kind, category, key, payload in rows:
                    if (kind.value, key) in existing:
                        continue
                    existing.add((kind.value, key))
                    record = EvidenceAdditionRecord(incident_id=incident_id, kind=kind.value, category=category,
                                                    evidence_key=key, payload_json=payload, created_by=created_by,
                                                    created_at=now)  # fmt: skip
                    session.add(record)
                    records.append(record)
                session.flush()
                return [self._to_addition(r) for r in records]
        except SQLAlchemyError as exc:
            raise self._error("add_evidence", exc, incident_id) from None

    def list(self, incident_id: str) -> list[EvidenceAddition]:
        try:
            with self.db.session() as session:
                query = (select(EvidenceAdditionRecord).where(EvidenceAdditionRecord.incident_id == incident_id)
                         .order_by(EvidenceAdditionRecord.id))  # fmt: skip
                return [self._to_addition(r) for r in session.scalars(query)]
        except SQLAlchemyError as exc:
            raise self._error("list_evidence", exc, incident_id) from None

    def record_invocation(
        self,
        result: ToolResult,
        *,
        purpose: str,
        incident_id: Optional[str] = None,
        reference: Optional[str] = None,
        actor: Optional[str] = None,
    ) -> None:
        record = ToolInvocationRecord(
            incident_id=incident_id, purpose=purpose, reference=reference, tool=result.tool,
            operation=result.operation, target_type=result.target_type, status=result.status.value,
            synthetic=result.synthetic, error_category=result.error_category, duration_ms=result.duration_ms,
            actor=actor, created_at=_utcnow(),
        )  # fmt: skip
        try:
            with self.db.session() as session:
                session.add(record)
        except SQLAlchemyError as exc:
            raise self._error("record_tool_invocation", exc, incident_id) from None

    @staticmethod
    def _to_addition(record: EvidenceAdditionRecord) -> EvidenceAddition:
        kind = EvidenceAdditionKind(record.kind)
        payload: dict = {}
        if kind == EvidenceAdditionKind.SIMULATED:
            payload["simulated"] = SimulatedObservation.model_validate_json(record.payload_json)
        else:
            payload["item"] = EvidenceItem.model_validate_json(record.payload_json)
        return EvidenceAddition(addition_id=record.id, incident_id=record.incident_id, kind=kind,
                                created_at=record.created_at, created_by=record.created_by, **payload)  # fmt: skip

    @staticmethod
    def _error(operation: str, exc: Exception, incident_id: Optional[str]) -> StorageError:
        log_event(logger, "database operation failed", level=logging.ERROR, incident_id=incident_id,
                  operation=operation, stage="storage", outcome="failure", error_category="storage_error",
                  exception_type=type(exc).__name__)  # fmt: skip
        return StorageError("database operation failed", incident_id=incident_id)
