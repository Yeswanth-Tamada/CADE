"""Library entries (paper Section 3): a playbook plus what is needed to retrieve and trust it."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import AwareDatetime, BaseModel, Field, model_validator

from app.schemas.playbook import Playbook, PlaybookOrigin, RequiredEvidence


class LibraryEntry(BaseModel):
    entry_id: str
    name: str
    intent_label: str
    origin: PlaybookOrigin
    playbook: Playbook
    required: RequiredEvidence
    techniques: list[str] = Field(default_factory=list)
    successes: int = Field(ge=0)  # analyst-retained, resolved executions
    executions: int = Field(ge=0)
    smoothed_success: float = Field(ge=0, le=1)  # (s + k) / (n + 2k)
    deprioritized: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: AwareDatetime
    updated_at: AwareDatetime

    @model_validator(mode="after")
    def _counts(self) -> "LibraryEntry":
        if self.successes > self.executions:
            raise ValueError("successes cannot exceed executions")
        return self


class LibraryImport(BaseModel):
    """A hand-authored playbook submitted by a library administrator."""

    playbook: Playbook
    description: Optional[str] = Field(default=None, max_length=2000)


class OutcomeRecord(BaseModel):
    """What the library did with an execution outcome."""

    entry_id: Optional[str] = None
    action: str  # "updated" | "merged" | "inserted" | "not_stored"
    reason: str
    successes: Optional[int] = None
    executions: Optional[int] = None
    deprioritized: Optional[bool] = None
