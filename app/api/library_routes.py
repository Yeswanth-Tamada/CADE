"""Playbook library and action vocabulary."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Path, Query, Request

from app.core.errors import NotFoundError
from app.core.threads import run_sync
from app.schemas.library import LibraryEntry, LibraryImport
from app.schemas.playbook import PlaybookOrigin
from app.security.auth import Principal, require
from app.security.roles import Permission

router = APIRouter(tags=["library"])

_RESPONSES = {401: {"description": "Missing, invalid or revoked API key"},
              403: {"description": "The API key's roles do not allow this operation"}}  # fmt: skip


@router.get("/library/playbooks", response_model=list[LibraryEntry], responses=_RESPONSES)
async def list_playbooks(request: Request, intent: Optional[str] = Query(default=None, max_length=64),
                         principal: Principal = require(Permission.READ)) -> list[LibraryEntry]:  # fmt: skip
    return request.app.state.components.library.entries(intent)


@router.get("/library/playbooks/{entry_id}", response_model=LibraryEntry,
            responses={**_RESPONSES, 404: {"description": "Not found"}})  # fmt: skip
async def get_playbook(request: Request, entry_id: str = Path(pattern=r"^PB-[0-9A-F]{16}$"),
                       principal: Principal = require(Permission.READ)) -> LibraryEntry:  # fmt: skip
    entry = request.app.state.components.library.get(entry_id)
    if entry is None:
        raise NotFoundError("library playbook not found")
    return entry


@router.post("/library/playbooks", status_code=201, response_model=LibraryEntry,
             responses={**_RESPONSES, 422: {"description": "Not a valid, safe DAG"}})  # fmt: skip
async def import_playbook(request: Request, body: LibraryImport,
                          principal: Principal = require(Permission.MANAGE_LIBRARY)) -> LibraryEntry:  # fmt: skip
    """Add a hand-authored playbook. It must be a valid DAG whose destructive steps are verified."""
    playbook = body.playbook.model_copy(update={"origin": PlaybookOrigin.IMPORTED,
                                                "description": body.description or body.playbook.description})  # fmt: skip
    library = request.app.state.components.library
    entry = await run_sync(library.add, playbook, metadata={"imported_by": principal.analyst_id})
    await request.app.state.audit.record(principal, "library.import", outcome="success", target_type="playbook",
                                         target_id=entry.entry_id, details={"intent": entry.intent_label})  # fmt: skip
    return entry


@router.get("/vocabulary", responses=_RESPONSES)
async def vocabulary(request: Request, principal: Principal = require(Permission.READ)) -> dict[str, Any]:
    """The controlled action vocabulary and the per-intent investigative goals."""
    vocab = request.app.state.components.vocabulary
    return {
        "version": vocab.version,
        "actions": [a.model_dump(mode="json") | {"binding_kind": a.binding_kind} for a in vocab.actions.values()],
        "goals": {intent: [g.model_dump(mode="json") for g in goals] for intent, goals in vocab.goals.items()},
    }
