"""Read models for API keys and the audit trail."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import AwareDatetime, BaseModel, Field

from app.security.roles import Role


class ApiKeyInfo(BaseModel):
    """An API key's metadata. Never contains the key or its hash."""

    key_id: int
    key_prefix: str
    analyst_id: str
    roles: list[Role]
    description: Optional[str] = None
    created_at: AwareDatetime
    revoked_at: Optional[AwareDatetime] = None

    @property
    def active(self) -> bool:
        return self.revoked_at is None


class AuditEvent(BaseModel):
    audit_id: int
    created_at: AwareDatetime
    actor: Optional[str] = None
    key_id: Optional[int] = None
    auth_mode: str
    # Free-text label from the X-CADE-Analyst header: recorded for context, never trusted for authorization.
    claimed_label: Optional[str] = None
    action: str
    target_type: Optional[str] = None
    target_id: Optional[str] = None
    outcome: str
    details: dict[str, Any] = Field(default_factory=dict)
