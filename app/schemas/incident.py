"""Incident schemas: what the API accepts, the normalised form, and the canonical output.

Data flow::

    IncidentSubmission  --(ingestion.normalizer)-->  NormalizedIncident
    NormalizedIncident  --(intent.llm_extractor)-->  CanonicalIncident

``CanonicalIncident`` is the contract later CADE phases (routing, synthesis, library) consume.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Literal, Optional, Union

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.enums import (
    AttackMappingMode,
    AttackValidationStatus,
    Criticality,
    EvidenceCategory,
    EvidenceStatus,
    ExtractionStatus,
    HashAlgorithm,
    IncidentSource,
    IncidentStatus,
    ProvenanceOrigin,
    Severity,
)
from app.schemas.indicators import canonical_domain, canonical_hash, canonical_ip

SCHEMA_VERSION = "1.0"
INCIDENT_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$"
TECHNIQUE_ID_PATTERN = r"^T\d{4}(\.\d{3})?$"
INTENT_LABEL_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"

# Hard upper bound; the configurable (smaller) limit is enforced by the normalizer.
MAX_TEXT_CHARS = 200_000

AttributeValue = Union[str, int, float, bool, None]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for value in values:
        key = value.lower()
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


# ======================================================================================
# Input
# ======================================================================================


class Observables(BaseModel):
    """Entities explicitly supplied with a structured incident (e.g. SIEM alert fields)."""

    model_config = ConfigDict(extra="forbid")

    hosts: list[str] = Field(default_factory=list, max_length=500)
    users: list[str] = Field(default_factory=list, max_length=500)
    ips: list[str] = Field(default_factory=list, max_length=500)
    domains: list[str] = Field(default_factory=list, max_length=500)
    file_hashes: list[str] = Field(default_factory=list, max_length=500)
    processes: list[str] = Field(default_factory=list, max_length=500)

    @field_validator("hosts", "users", "processes")
    @classmethod
    def _clean_names(cls, values: list[str]) -> list[str]:
        cleaned = [" ".join(v.split()) for v in values]
        if any(not v or len(v) > 512 for v in cleaned):
            raise ValueError("entries must be non-empty strings of at most 512 characters")
        return _dedupe(cleaned)

    @field_validator("ips")
    @classmethod
    def _validate_ips(cls, values: list[str]) -> list[str]:
        try:
            return _dedupe([canonical_ip(v) for v in values])
        except ValueError as exc:
            raise ValueError(f"invalid IP address: {exc}") from None

    @field_validator("domains")
    @classmethod
    def _validate_domains(cls, values: list[str]) -> list[str]:
        return _dedupe([canonical_domain(v) for v in values])

    @field_validator("file_hashes")
    @classmethod
    def _validate_hashes(cls, values: list[str]) -> list[str]:
        return _dedupe([canonical_hash(v)[1] for v in values])


class IncidentSubmission(BaseModel):
    """Structured incident input accepted by ``POST /incidents``.

    A plain-text body is converted into ``IncidentSubmission(description=<text>)``.
    Unknown top-level fields are rejected (not silently dropped); put extra SIEM fields
    into ``attributes`` so they are preserved.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "source": "siem_alert",
                    "title": "SYNTHETIC: Encoded PowerShell spawned by Word",
                    "description": "EDR alert on host WS-FIN-042: WINWORD.EXE spawned powershell.exe "
                    "with an encoded command, followed by an outbound connection to 203.0.113.45:443.",
                    "occurred_at": "2026-10-19T08:15:00Z",
                    "severity": "high",
                    "observables": {"hosts": ["WS-FIN-042"], "users": ["jdoe"]},
                    "attributes": {"rule_id": "EDR-1042", "synthetic": True},
                }
            ]
        },
    )

    incident_id: Optional[str] = Field(default=None, pattern=INCIDENT_ID_PATTERN)
    source: IncidentSource = IncidentSource.ANALYST_DESCRIPTION
    title: Optional[str] = Field(default=None, max_length=300)
    description: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    occurred_at: Optional[datetime] = None
    severity: Optional[Severity] = None
    log_excerpt: Optional[str] = Field(default=None, max_length=MAX_TEXT_CHARS)
    observables: Observables = Field(default_factory=Observables)
    attributes: dict[str, AttributeValue] = Field(default_factory=dict, max_length=100)

    @field_validator("description")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("description must contain non-whitespace text")
        return value

    @field_validator("attributes")
    @classmethod
    def _attribute_keys(cls, value: dict[str, AttributeValue]) -> dict[str, AttributeValue]:
        for key, item in value.items():
            if not key or len(key) > 100:
                raise ValueError("attribute keys must be 1-100 characters")
            if isinstance(item, str) and len(item) > 10_000:
                raise ValueError(f"attribute '{key}' exceeds 10000 characters")
        return value


class NormalizedIncident(BaseModel):
    """Output of the ingestion layer. ``raw_description`` is preserved byte-for-byte."""

    incident_id: str = Field(pattern=INCIDENT_ID_PATTERN)
    source: IncidentSource
    title: Optional[str] = None
    severity: Optional[Severity] = None
    raw_description: str
    normalized_description: str = Field(min_length=1)
    log_excerpt: Optional[str] = None
    occurred_at: Optional[AwareDatetime] = None
    received_at: AwareDatetime
    observables: Observables = Field(default_factory=Observables)
    attributes: dict[str, AttributeValue] = Field(default_factory=dict)
    normalization_warnings: list[str] = Field(default_factory=list)


# ======================================================================================
# Canonical output
# ======================================================================================


class FileHash(BaseModel):
    algorithm: HashAlgorithm
    value: str

    @model_validator(mode="after")
    def _check(self) -> "FileHash":
        algorithm, value = canonical_hash(self.value)
        if algorithm != self.algorithm:
            raise ValueError(f"hash length does not match algorithm {self.algorithm.value}")
        self.value = value
        return self


class Entities(BaseModel):
    """Entities found in the incident. Every value is guaranteed to occur in the input."""

    hosts: list[str] = Field(default_factory=list)
    users: list[str] = Field(default_factory=list)
    file_hashes: list[FileHash] = Field(default_factory=list)
    ips: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    processes: list[str] = Field(default_factory=list)

    @field_validator("ips")
    @classmethod
    def _ips(cls, values: list[str]) -> list[str]:
        return [canonical_ip(v) for v in values]


class Provenance(BaseModel):
    origin: ProvenanceOrigin
    source_ref: Optional[str] = Field(default=None, max_length=200)
    excerpt: Optional[str] = Field(default=None, max_length=1000)


_GROUNDED_ORIGINS = {
    ProvenanceOrigin.INCIDENT_TEXT,
    ProvenanceOrigin.STRUCTURED_INPUT,
    ProvenanceOrigin.PATTERN_EXTRACTION,
    ProvenanceOrigin.TOOL,
    ProvenanceOrigin.ANALYST_SUPPLIED,
}


class EvidenceItem(BaseModel):
    """One piece of evidence. ``evidence_id`` is stable for the same category + description."""

    evidence_id: str = Field(pattern=r"^ev-[0-9a-f]{12}$")
    category: EvidenceCategory
    description: str = Field(min_length=1, max_length=1000)
    status: EvidenceStatus
    criticality: Optional[Criticality] = None
    rationale: Optional[str] = Field(default=None, max_length=1000)
    provenance: Optional[Provenance] = None

    @staticmethod
    def make_id(category: EvidenceCategory, description: str) -> str:
        key = f"{category.value}|{' '.join(description.lower().split())}"
        return "ev-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]

    @model_validator(mode="after")
    def _consistency(self) -> "EvidenceItem":
        if self.status == EvidenceStatus.OBSERVED and (
            self.provenance is None or self.provenance.origin not in _GROUNDED_ORIGINS
        ):
            raise ValueError("observed evidence must have grounded provenance (input, real tool or analyst)")
        if self.status == EvidenceStatus.MISSING and self.criticality is None:
            raise ValueError("missing evidence must state its criticality (critical or supporting)")
        return self


class EvidenceProfile(BaseModel):
    """What evidence is in hand, what is hypothesised, and what is still needed.

    * ``observed``   -- facts present in the input (grounded).
    * ``unconfirmed``-- model inferences that no source has confirmed.
    * ``missing``    -- evidence needed for the investigation that the input does not contain.
                        Missing means *not yet in hand*, never "proven absent".
    * ``critical_missing`` -- IDs of missing items marked critical (derived automatically).
    * ``not_checked``-- evidence categories relevant to the intent that nothing has examined.
    * ``sources``    -- which inputs contributed evidence (Phase 2 adds tools such as Wazuh).
    """

    observed: list[EvidenceItem] = Field(default_factory=list)
    unconfirmed: list[EvidenceItem] = Field(default_factory=list)
    missing: list[EvidenceItem] = Field(default_factory=list)
    critical_missing: list[str] = Field(default_factory=list)
    not_checked: list[EvidenceCategory] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistency(self) -> "EvidenceProfile":
        for name, expected in (
            ("observed", EvidenceStatus.OBSERVED),
            ("unconfirmed", EvidenceStatus.INFERRED),
            ("missing", EvidenceStatus.MISSING),
        ):
            for item in getattr(self, name):
                if item.status != expected:
                    raise ValueError(f"items in '{name}' must have status '{expected.value}'")
        derived = [i.evidence_id for i in self.missing if i.criticality == Criticality.CRITICAL]
        if self.critical_missing and sorted(self.critical_missing) != sorted(derived):
            raise ValueError("critical_missing must list exactly the missing items marked critical")
        self.critical_missing = derived
        return self


class AttackTechniqueMapping(BaseModel):
    """A *candidate* ATT&CK technique for the incident.

    ``name`` and ``tactics`` come only from the loaded ATT&CK dataset -- never from the LLM --
    so they are empty when ``validation_status`` is ``unverified``. ``validated`` means the ID
    is real and active; whether it fits the incident is expressed only by ``confidence``.
    """

    technique_id: str = Field(pattern=TECHNIQUE_ID_PATTERN)
    name: Optional[str] = None
    tactics: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    justification: Optional[str] = Field(default=None, max_length=1000)
    validation_status: AttackValidationStatus

    @model_validator(mode="after")
    def _consistency(self) -> "AttackTechniqueMapping":
        if self.validation_status == AttackValidationStatus.UNVERIFIED and (self.name or self.tactics):
            raise ValueError("unverified mappings must not carry a technique name or tactics")
        if self.validation_status == AttackValidationStatus.VALIDATED and not self.name:
            raise ValueError("validated mappings must carry the dataset's technique name")
        return self


class Intent(BaseModel):
    label: str = Field(pattern=INTENT_LABEL_PATTERN)
    description: str = Field(min_length=1, max_length=2000)
    confidence: float = Field(ge=0.0, le=1.0)
    attack_techniques: list[AttackTechniqueMapping] = Field(default_factory=list)
    # Two-stage analysis: the ATT&CK tactic the activity serves and the model's one-line account of
    # the activity that grounds the label (None for single-stage analyses).
    tactic: Optional[str] = Field(default=None, max_length=40)
    rationale: Optional[str] = Field(default=None, max_length=1000)


class SafetyFlags(BaseModel):
    """Phase 1 never executes anything; these defaults are deliberately conservative."""

    destructive_actions_allowed: bool = False
    human_approval_required: bool = True

    @model_validator(mode="after")
    def _consistency(self) -> "SafetyFlags":
        if self.destructive_actions_allowed and not self.human_approval_required:
            raise ValueError("destructive actions always require human approval")
        return self


class AnalysisMetadata(BaseModel):
    extraction_status: ExtractionStatus
    warnings: list[str] = Field(default_factory=list)
    provider: str
    model_name: str
    prompt_version: str
    attempts: int = Field(ge=1)
    started_at: AwareDatetime
    completed_at: AwareDatetime
    duration_ms: float = Field(ge=0)
    llm_duration_ms: Optional[float] = Field(default=None, ge=0)
    attack_mapping_mode: AttackMappingMode
    attack_dataset_version: Optional[str] = None
    # How many technique candidates the model proposed before validation; distinguishes "the model
    # proposed none" from "candidates were discarded". None for analyses stored before this field.
    attack_candidates_proposed: Optional[int] = Field(default=None, ge=0)
    taxonomy_version: str

    @field_validator("extraction_status")
    @classmethod
    def _never_failed(cls, value: ExtractionStatus) -> ExtractionStatus:
        # A canonical incident only exists for a usable analysis. Failures are recorded
        # separately (see storage.models.AnalysisRecord) and never dressed up as results.
        if value == ExtractionStatus.FAILED:
            raise ValueError("a canonical incident cannot have extraction_status 'failed'")
        return value


class CanonicalIncident(BaseModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION

    # --- incident metadata
    incident_id: str = Field(pattern=INCIDENT_ID_PATTERN)
    source: IncidentSource
    title: Optional[str] = None
    severity: Optional[Severity] = None
    received_at: AwareDatetime
    occurred_at: Optional[AwareDatetime] = None
    raw_description: str
    normalized_description: str
    status: IncidentStatus

    # --- understanding
    summary: str = Field(min_length=1, max_length=4000)
    intent: Intent
    entities: Entities
    evidence_profile: EvidenceProfile
    uncertainties: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)

    # --- safety + provenance of the analysis itself
    safety: SafetyFlags = Field(default_factory=SafetyFlags)
    analysis: AnalysisMetadata
