"""Configurable intent taxonomy.

The labels are an implementation choice (see ``app/data/README.md``); the paper only gives
examples such as "ransomware investigation" and eight evaluation categories.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, model_validator

from app.core.errors import ConfigurationError
from app.schemas.enums import EvidenceCategory
from app.schemas.incident import INTENT_LABEL_PATTERN

UNKNOWN_INTENT = "unknown"

# MITRE ATT&CK Enterprise tactics (x_mitre_shortname). ATT&CK v20 split "defense-evasion" into
# "stealth" and "defense-impairment"; all three are accepted so older and newer datasets work.
ATTACK_TACTICS = frozenset({
    "reconnaissance", "resource-development", "initial-access", "execution", "persistence", "privilege-escalation",
    "defense-evasion", "stealth", "defense-impairment", "credential-access", "discovery", "lateral-movement",
    "collection", "command-and-control", "exfiltration", "impact",
})  # fmt: skip


class IntentDefinition(BaseModel):
    label: str = Field(pattern=INTENT_LABEL_PATTERN)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    # ATT&CK tactics the intent investigates (grounds the intent in the standard taxonomy)
    tactics: list[str] = Field(default_factory=list)
    # Evidence categories usually relevant to this intent; used to compute "not_checked"
    # and, in Phase 4, the Evidence Alignment (EA) metric.
    relevant_evidence: list[EvidenceCategory] = Field(default_factory=list)
    # The relevant categories the ontology marks critical; the others are supporting (paper Section 6.1)
    critical_evidence: list[EvidenceCategory] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "IntentDefinition":
        unknown = set(self.tactics) - ATTACK_TACTICS
        if unknown:
            raise ValueError(f"intent '{self.label}' names unknown ATT&CK tactics: {', '.join(sorted(unknown))}")
        if not set(self.critical_evidence) <= set(self.relevant_evidence):
            raise ValueError(f"intent '{self.label}': critical evidence must be among its relevant evidence")
        return self

    @property
    def supporting_evidence(self) -> list[EvidenceCategory]:
        return [c for c in self.relevant_evidence if c not in self.critical_evidence]


class IntentTaxonomy(BaseModel):
    version: str
    note: str = ""
    intents: list[IntentDefinition] = Field(min_length=2)

    @model_validator(mode="after")
    def _check(self) -> "IntentTaxonomy":
        labels = [i.label for i in self.intents]
        if len(labels) != len(set(labels)):
            raise ValueError("intent labels must be unique")
        if UNKNOWN_INTENT not in labels:
            raise ValueError(f"the taxonomy must include the fallback intent '{UNKNOWN_INTENT}'")
        return self

    @property
    def labels(self) -> list[str]:
        return [i.label for i in self.intents]

    def get(self, label: str) -> IntentDefinition | None:
        return next((i for i in self.intents if i.label == label), None)

    @classmethod
    def load(cls, path: Path) -> "IntentTaxonomy":
        try:
            return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except FileNotFoundError:
            raise ConfigurationError(f"intent taxonomy file not found: {path}") from None
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ConfigurationError(f"invalid intent taxonomy file {path}: {exc}") from None
