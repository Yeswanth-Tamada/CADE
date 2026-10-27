"""Persistence for API keys and the append-only audit log.

Like :mod:`app.storage.repository`, every method runs in its own transaction and returns
Pydantic read models. The audit repository intentionally has no update or delete methods.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.core.errors import StorageError
from app.core.logging import get_logger, log_event
from app.schemas.security import ApiKeyInfo, AuditEvent
from app.security.roles import Role
from app.storage.database import Database
from app.storage.models import ApiKeyRecord, AuditRecord

logger = get_logger("storage")

_MAX_DETAILS_CHARS = 4000


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _storage_error(operation: str, exc: Exception) -> StorageError:
    log_event(
        logger,
        "database operation failed",
        level=logging.ERROR,
        operation=operation,
        stage="storage",
        outcome="failure",
        error_category="storage_error",
        exception_type=type(exc).__name__,
    )
    return StorageError("database operation failed")


class ApiKeyRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, *, key_hash: str, key_prefix: str, analyst_id: str, roles: list[Role],
               description: Optional[str] = None) -> ApiKeyInfo:  # fmt: skip
        record = ApiKeyRecord(
            key_hash=key_hash,
            key_prefix=key_prefix,
            analyst_id=analyst_id,
            roles_json=json.dumps(sorted(r.value for r in roles)),
            description=description,
            created_at=_utcnow(),
        )
        try:
            with self.db.session() as session:
                session.add(record)
                session.flush()
                return self._to_info(record)
        except SQLAlchemyError as exc:
            raise _storage_error("create_api_key", exc) from None

    def find_by_hash(self, key_hash: str) -> Optional[ApiKeyInfo]:
        try:
            with self.db.session() as session:
                record = session.scalars(select(ApiKeyRecord).where(ApiKeyRecord.key_hash == key_hash)).first()
                return self._to_info(record) if record else None
        except SQLAlchemyError as exc:
            raise _storage_error("find_api_key", exc) from None

    def list(self) -> list[ApiKeyInfo]:
        try:
            with self.db.session() as session:
                return [self._to_info(r) for r in session.scalars(select(ApiKeyRecord).order_by(ApiKeyRecord.id))]
        except SQLAlchemyError as exc:
            raise _storage_error("list_api_keys", exc) from None

    def revoke(self, key_id: int) -> Optional[ApiKeyInfo]:
        try:
            with self.db.session() as session:
                record = session.get(ApiKeyRecord, key_id)
                if record is None:
                    return None
                if record.revoked_at is None:
                    record.revoked_at = _utcnow()
                session.flush()
                return self._to_info(record)
        except SQLAlchemyError as exc:
            raise _storage_error("revoke_api_key", exc) from None

    @staticmethod
    def _to_info(record: ApiKeyRecord) -> ApiKeyInfo:
        return ApiKeyInfo(
            key_id=record.id,
            key_prefix=record.key_prefix,
            analyst_id=record.analyst_id,
            roles=[Role(r) for r in json.loads(record.roles_json)],
            description=record.description,
            created_at=record.created_at,
            revoked_at=record.revoked_at,
        )


class AuditRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def append(
        self,
        *,
        action: str,
        outcome: str,
        auth_mode: str,
        actor: Optional[str] = None,
        key_id: Optional[int] = None,
        claimed_label: Optional[str] = None,
        target_type: Optional[str] = None,
        target_id: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> AuditEvent:
        details_json = json.dumps(details or {}, default=str, sort_keys=True)
        if len(details_json) > _MAX_DETAILS_CHARS:
            details_json = json.dumps({"truncated": True})
        record = AuditRecord(
            created_at=_utcnow(),
            actor=actor,
            key_id=key_id,
            auth_mode=auth_mode,
            claimed_label=claimed_label,
            action=action,
            target_type=target_type,
            target_id=target_id,
            outcome=outcome,
            details_json=details_json,
        )
        try:
            with self.db.session() as session:
                session.add(record)
                session.flush()
                return self._to_event(record)
        except SQLAlchemyError as exc:
            raise _storage_error("append_audit", exc) from None

    def list(self, *, target_id: Optional[str] = None, action: Optional[str] = None,
             limit: int = 200) -> list[AuditEvent]:  # fmt: skip
        query = select(AuditRecord).order_by(AuditRecord.id.desc()).limit(limit)
        if target_id:
            query = query.where(AuditRecord.target_id == target_id)
        if action:
            query = query.where(AuditRecord.action == action)
        try:
            with self.db.session() as session:
                return [self._to_event(r) for r in session.scalars(query)]
        except SQLAlchemyError as exc:
            raise _storage_error("list_audit", exc) from None

    @staticmethod
    def _to_event(record: AuditRecord) -> AuditEvent:
        return AuditEvent(
            audit_id=record.id,
            created_at=record.created_at,
            actor=record.actor,
            key_id=record.key_id,
            auth_mode=record.auth_mode,
            claimed_label=record.claimed_label,
            action=record.action,
            target_type=record.target_type,
            target_id=record.target_id,
            outcome=record.outcome,
            details=json.loads(record.details_json or "{}"),
        )
