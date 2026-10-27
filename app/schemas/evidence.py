"""Evidence gathered after the initial analysis: real tool observations, analyst-supplied
evidence and (kept strictly apart) synthetic observations from simulated tools.

The Phase 1 :class:`~app.schemas.incident.EvidenceProfile` of an analysis is immutable; the
*effective* profile is that profile plus everything added since.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, computed_field, model_validator

from app.schemas.enums import (
    Criticality,
    EvidenceAdditionKind,
    EvidenceCategory,
    ExecutionMode,
    ProvenanceOrigin,
    ToolStatus,
)
from app.schemas.incident import EvidenceItem, EvidenceProfile


class SimulatedObservation(BaseModel):
    """Output of a simulated tool. Never an EvidenceItem, never 'observed'."""

    category: EvidenceCategory
    description: str = Field(min_length=1, max_length=1000)
    tool: str
    operation: str
    synthetic: Literal[True] = True


class EvidenceAddition(BaseModel):
    addition_id: int
    incident_id: str
    kind: EvidenceAdditionKind
    created_at: AwareDatetime
    created_by: Optional[str] = None
    item: Optional[EvidenceItem] = None
    simulated: Optional[SimulatedObservation] = None

    @model_validator(mode="after")
    def _consistency(self) -> "EvidenceAddition":
        if self.kind == EvidenceAdditionKind.SIMULATED:
            if self.simulated is None or self.item is not None:
                raise ValueError("simulated additions carry only a simulated observation")
            return self
        if self.item is None or self.simulated is not None:
            raise ValueError("tool and analyst additions carry exactly one evidence item")
        expected = ProvenanceOrigin.TOOL if self.kind == EvidenceAdditionKind.TOOL else ProvenanceOrigin.ANALYST_SUPPLIED
        if self.item.provenance is None or self.item.provenance.origin != expected:
            raise ValueError(f"{self.kind.value} additions must have provenance '{expected.value}'")
        return self


class EffectiveEvidenceProfile(BaseModel):
    incident_id: str
    analysis_id: int
    base: EvidenceProfile
    tool_observed: list[EvidenceItem] = Field(default_factory=list)
    analyst_supplied: list[EvidenceItem] = Field(default_factory=list)
    simulated: list[SimulatedObservation] = Field(default_factory=list)

    def observed_categories(self, *, include_simulated: bool = False) -> set[EvidenceCategory]:
        categories = {i.category for i in self.base.observed + self.tool_observed + self.analyst_supplied}
        if include_simulated:
            categories |= {s.category for s in self.simulated}
        return categories

    def outstanding_missing(self, *, include_simulated: bool = False) -> list[EvidenceItem]:
        """Missing items of the analysis whose category has not been obtained since."""
        have = self.observed_categories(include_simulated=include_simulated)
        return [i for i in self.base.missing if i.category not in have]

    def outstanding_critical_categories(self, *, include_simulated: bool = False) -> set[EvidenceCategory]:
        return {i.category for i in self.outstanding_missing(include_simulated=include_simulated)
                if i.criticality == Criticality.CRITICAL}  # fmt: skip

    @computed_field  # type: ignore[prop-decorator]
    @property
    def real_observed_categories(self) -> list[EvidenceCategory]:
        return sorted(self.observed_categories(), key=lambda c: c.value)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def simulated_categories(self) -> list[EvidenceCategory]:
        return sorted({s.category for s in self.simulated}, key=lambda c: c.value)


class AnalystEvidenceSubmission(BaseModel):
    """Evidence an analyst obtained outside CADE (e.g. pasted from a console), attributed to them."""

    model_config = ConfigDict(extra="forbid")

    category: EvidenceCategory
    description: str = Field(min_length=1, max_length=1000)
    excerpt: Optional[str] = Field(default=None, max_length=1000)


class EnrichmentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Default: the analysis' missing categories (critical first), then the not-checked ones.
    categories: Optional[list[EvidenceCategory]] = Field(default=None, max_length=len(EvidenceCategory))


class EnrichmentAttempt(BaseModel):
    tool: str
    operation: str
    target_type: str
    status: ToolStatus
    synthetic: bool
    categories: list[EvidenceCategory]
    error_category: Optional[str] = None
    message: Optional[str] = None


class EnrichmentReport(BaseModel):
    incident_id: str
    mode: ExecutionMode
    requested_categories: list[EvidenceCategory]
    attempts: list[EnrichmentAttempt] = Field(default_factory=list)
    added_observed: int = 0  # real tool observations stored as observed evidence
    added_simulated: int = 0  # synthetic observations, stored apart and never observed
    categories_obtained: list[EvidenceCategory] = Field(default_factory=list)  # by real tools only
    categories_simulated: list[EvidenceCategory] = Field(default_factory=list)
    categories_without_data: list[EvidenceCategory] = Field(default_factory=list)
    categories_without_tools: list[EvidenceCategory] = Field(default_factory=list)
    manual_tasks: list[str] = Field(default_factory=list)
