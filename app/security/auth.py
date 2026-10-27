"""Request authentication and server-side authorization.

Identity comes only from an API key (``Authorization: Bearer <key>``). The optional
``X-CADE-Analyst`` header is a free-text label recorded in the audit log for context; it is
never used to identify or authorize anyone.

Routes declare what they need with ``principal: Principal = require(Permission.X)``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, Request

from app.core.errors import AuthenticationError, AuthorizationError
from app.core.logging import get_logger, log_event
from app.core.threads import run_sync
from app.security.keys import hash_key
from app.security.roles import AUTH_DISABLED_PERMISSIONS, Permission, Role, permissions_for

logger = get_logger("auth")

CLAIMED_LABEL_HEADER = "X-CADE-Analyst"
_MAX_LABEL_CHARS = 100


@dataclass(frozen=True)
class Principal:
    analyst_id: str
    roles: frozenset[Role]
    permissions: frozenset[Permission]
    auth_mode: str  # "api_key" | "auth_disabled"
    key_id: Optional[int] = None
    claimed_label: Optional[str] = None

    def can(self, permission: Permission) -> bool:
        return permission in self.permissions


def _claimed_label(request: Request) -> Optional[str]:
    raw = request.headers.get(CLAIMED_LABEL_HEADER)
    if not raw:
        return None
    cleaned = "".join(ch for ch in raw if ch.isprintable()).strip()
    return cleaned[:_MAX_LABEL_CHARS] or None


def _bearer_token(request: Request) -> Optional[str]:
    header = request.headers.get("Authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


async def authenticate(request: Request) -> Principal:
    """Identify the caller or raise :class:`AuthenticationError` (401)."""
    state = request.app.state
    label = _claimed_label(request)
    if not state.settings.auth_enabled:
        return Principal(
            analyst_id="auth-disabled",
            roles=frozenset(),
            permissions=AUTH_DISABLED_PERMISSIONS,
            auth_mode="auth_disabled",
            claimed_label=label,
        )
    token = _bearer_token(request)
    if token is None:
        _log_denied(request, "missing_credentials")
        raise AuthenticationError("authentication required: send 'Authorization: Bearer <API key>'")
    info = await run_sync(state.api_key_repository.find_by_hash, hash_key(token))
    if info is None or not info.active:
        _log_denied(request, "invalid_or_revoked_key")
        raise AuthenticationError("invalid or revoked API key")
    roles = frozenset(info.roles)
    return Principal(
        analyst_id=info.analyst_id,
        roles=roles,
        permissions=permissions_for(roles),
        auth_mode="api_key",
        key_id=info.key_id,
        claimed_label=label,
    )


async def optional_principal(request: Request) -> Optional[Principal]:
    """Like :func:`authenticate` but returns ``None`` instead of raising (for public endpoints)."""
    try:
        return await authenticate(request)
    except AuthenticationError:
        return None


def require(permission: Permission):
    """FastAPI dependency: authenticate, then check ``permission`` server-side."""

    async def dependency(request: Request) -> Principal:
        principal = await authenticate(request)
        if not principal.can(permission):
            await request.app.state.audit.record(
                principal,
                "authorization.denied",
                outcome="denied",
                target_type="endpoint",
                target_id=f"{request.method} {request.url.path}"[:100],
                details={"permission": permission.value},
            )
            raise AuthorizationError(f"this API key is not allowed to perform '{permission.value}'")
        return principal

    return Depends(dependency)


def _log_denied(request: Request, reason: str) -> None:
    log_event(
        logger,
        "authentication failed",
        level=logging.WARNING,
        operation=f"{request.method} {request.url.path}",
        outcome="denied",
        error_category="unauthenticated",
        reason=reason,
    )
