"""Records who did what, to which object, and with what outcome.

Callers pass small, non-sensitive ``details`` (IDs, counts, flags) -- never incident text, tool
output or credentials. A failure to write an audit record raises ``StorageError``: callers of
security-relevant operations (approval, execution) must treat that as fatal and not proceed.
"""

from __future__ import annotations

from typing import Any, Optional

from app.core.threads import run_sync
from app.schemas.security import AuditEvent
from app.security.auth import Principal
from app.storage.security_repository import AuditRepository


class AuditService:
    def __init__(self, repository: AuditRepository) -> None:
        self.repository = repository

    async def record(
        self,
        actor: Optional[Principal],
        action: str,
        *,
        outcome: str,
        target_type: Optional[str] = None,
        target_id: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> AuditEvent:
        return await run_sync(
            self.repository.append,
            action=action,
            outcome=outcome,
            auth_mode=actor.auth_mode if actor else "system",
            actor=actor.analyst_id if actor else None,
            key_id=actor.key_id if actor else None,
            claimed_label=actor.claimed_label if actor else None,
            target_type=target_type,
            target_id=target_id,
            details=details,
        )

    async def list(self, *, target_id: Optional[str] = None, action: Optional[str] = None,
                   limit: int = 200) -> list[AuditEvent]:  # fmt: skip
        return await run_sync(self.repository.list, target_id=target_id, action=action, limit=limit)
