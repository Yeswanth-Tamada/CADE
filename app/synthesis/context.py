"""What synthesis knows about the incident: its abstract representation (intent + evidence profile)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.schemas.enums import Criticality, EvidenceCategory
from app.schemas.evidence import EffectiveEvidenceProfile
from app.schemas.incident import CanonicalIncident
from app.tools.artifacts import entity_artifacts


@dataclass(frozen=True)
class SynthesisContext:
    incident_id: str
    intent_label: str
    intent_description: str
    summary: str
    available: frozenset[str]  # entity artifact types the incident provides (e.g. host_id)
    observed_categories: tuple[EvidenceCategory, ...]
    missing_critical: tuple[EvidenceCategory, ...]
    missing_supporting: tuple[EvidenceCategory, ...]
    techniques: tuple[str, ...]  # ATT&CK candidates validated against the dataset
    synthetic_basis: bool = False  # simulated evidence was counted as available

    @classmethod
    def from_incident(
        cls,
        incident: CanonicalIncident,
        evidence: Optional[EffectiveEvidenceProfile] = None,
        *,
        include_simulated: bool = False,
    ) -> "SynthesisContext":
        if evidence is None:
            observed = {i.category for i in incident.evidence_profile.observed}
            # category level, as in EffectiveEvidenceProfile.outstanding_missing and routing's A
            missing = [i for i in incident.evidence_profile.missing if i.category not in observed]
            synthetic = False
        else:
            observed = evidence.observed_categories(include_simulated=include_simulated)
            missing = evidence.outstanding_missing(include_simulated=include_simulated)
            synthetic = include_simulated and bool(
                evidence.observed_categories(include_simulated=True) - evidence.observed_categories()
            )
        key = lambda c: c.value  # noqa: E731
        return cls(
            incident_id=incident.incident_id,
            intent_label=incident.intent.label,
            intent_description=incident.intent.description,
            summary=incident.summary,
            available=frozenset(k for k, v in entity_artifacts(incident.entities).items() if v),
            observed_categories=tuple(sorted(observed, key=key)),
            missing_critical=tuple(sorted({i.category for i in missing if i.criticality == Criticality.CRITICAL}, key=key)),
            missing_supporting=tuple(sorted({i.category for i in missing if i.criticality != Criticality.CRITICAL}
                                            - {i.category for i in missing if i.criticality == Criticality.CRITICAL},
                                            key=key)),  # fmt: skip
            techniques=tuple(t.technique_id for t in incident.intent.attack_techniques
                             if t.validation_status.value in ("validated", "unverified")),  # fmt: skip
            synthetic_basis=synthetic,
        )
