"""Persistence for library entries (metadata, counts and embeddings in the relational store)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.core.errors import StorageError
from app.core.logging import get_logger, log_event
from app.schemas.playbook import Playbook, PlaybookOrigin, RequiredEvidence
from app.storage.database import Database
from app.storage.models import PlaybookRecord

logger = get_logger("storage")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class StoredPlaybook:
    entry_id: str
    playbook: Playbook
    required: RequiredEvidence
    techniques: list[str]
    metadata: dict[str, Any]
    successes: int
    executions: int
    deprioritized: bool
    embedding_model: str
    embedding: list[float]
    structural_hash: str
    created_at: datetime
    updated_at: datetime


class LibraryRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def insert(self, entry_id: str, playbook: Playbook, *, required: RequiredEvidence, techniques: list[str],
               metadata: dict[str, Any], successes: int, executions: int, embedding_model: str,
               embedding: list[float]) -> StoredPlaybook:  # fmt: skip
        now = _utcnow()
        record = PlaybookRecord(
            id=entry_id, name=playbook.name, intent_label=playbook.intent_label, origin=playbook.origin.value,
            playbook_json=playbook.model_dump_json(), structural_hash=playbook.structural_hash(),
            required_json=required.model_dump_json(), techniques_json=json.dumps(techniques),
            metadata_json=json.dumps(metadata, default=str), success_count=successes, execution_count=executions,
            deprioritized=False, embedding_model=embedding_model, embedding_json=json.dumps(embedding),
            created_at=now, updated_at=now,
        )  # fmt: skip
        try:
            with self.db.session() as session:
                session.add(record)
                session.flush()
                return self._to_stored(record)
        except SQLAlchemyError as exc:
            raise self._error("insert_playbook", exc) from None

    def get(self, entry_id: str) -> Optional[StoredPlaybook]:
        try:
            with self.db.session() as session:
                record = session.get(PlaybookRecord, entry_id)
                return self._to_stored(record) if record else None
        except SQLAlchemyError as exc:
            raise self._error("get_playbook", exc) from None

    def list(self, intent_label: Optional[str] = None) -> list[StoredPlaybook]:
        query = select(PlaybookRecord).order_by(PlaybookRecord.created_at, PlaybookRecord.id)
        if intent_label:
            query = query.where(PlaybookRecord.intent_label == intent_label)
        try:
            with self.db.session() as session:
                return [self._to_stored(r) for r in session.scalars(query)]
        except SQLAlchemyError as exc:
            raise self._error("list_playbooks", exc) from None

    def count(self) -> int:
        try:
            with self.db.session() as session:
                return len(session.scalars(select(PlaybookRecord.id)).all())
        except SQLAlchemyError as exc:
            raise self._error("count_playbooks", exc) from None

    def record_outcome(self, entry_id: str, *, success: bool, deprioritize_after: int,
                       deprioritize_below: float, k: float) -> Optional[StoredPlaybook]:  # fmt: skip
        """Increment n (and s on success) atomically and re-evaluate de-prioritisation."""
        try:
            with self.db.session() as session:
                record = session.get(PlaybookRecord, entry_id, with_for_update=True)
                if record is None:
                    return None
                record.execution_count += 1
                record.success_count += 1 if success else 0
                smoothed = (record.success_count + k) / (record.execution_count + 2 * k)
                record.deprioritized = record.execution_count >= deprioritize_after and smoothed < deprioritize_below
                record.updated_at = _utcnow()
                session.flush()
                return self._to_stored(record)
        except SQLAlchemyError as exc:
            raise self._error("record_playbook_outcome", exc) from None

    def update_embedding(self, entry_id: str, model: str, embedding: list[float]) -> None:
        try:
            with self.db.session() as session:
                record = session.get(PlaybookRecord, entry_id)
                if record is not None:
                    record.embedding_model, record.embedding_json = model, json.dumps(embedding)
        except SQLAlchemyError as exc:
            raise self._error("update_playbook_embedding", exc) from None

    @staticmethod
    def _to_stored(record: PlaybookRecord) -> StoredPlaybook:
        playbook = Playbook.model_validate_json(record.playbook_json)
        return StoredPlaybook(
            entry_id=record.id, playbook=playbook.model_copy(update={"origin": PlaybookOrigin(record.origin)}),
            required=RequiredEvidence.model_validate_json(record.required_json),
            techniques=json.loads(record.techniques_json or "[]"), metadata=json.loads(record.metadata_json or "{}"),
            successes=record.success_count, executions=record.execution_count, deprioritized=bool(record.deprioritized),
            embedding_model=record.embedding_model, embedding=json.loads(record.embedding_json),
            structural_hash=record.structural_hash, created_at=record.created_at, updated_at=record.updated_at,
        )  # fmt: skip

    @staticmethod
    def _error(operation: str, exc: Exception) -> StorageError:
        log_event(logger, "database operation failed", level=logging.ERROR, operation=operation, stage="storage",
                  outcome="failure", error_category="storage_error", exception_type=type(exc).__name__)  # fmt: skip
        return StorageError("database operation failed")
