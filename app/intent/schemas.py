"""The JSON structure the LLM must return (validated before anything else uses it).

This is intentionally separate from the canonical incident schema: the model's raw answer is
*untrusted* and goes through grounding checks and ATT&CK validation before it becomes part
of a :class:`~app.schemas.incident.CanonicalIncident`.
"""

from __future__ import annotations

import copy
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.enums import EvidenceCategory


def _entity_list() -> Any:
    return Field(max_length=100)


class _LLMModel(BaseModel):
    # Extra keys are ignored rather than treated as malformed; missing/invalid keys are errors.
    model_config = ConfigDict(extra="ignore")


class LLMEntities(_LLMModel):
    hosts: list[str] = _entity_list()
    users: list[str] = _entity_list()
    ips: list[str] = _entity_list()
    domains: list[str] = _entity_list()
    file_hashes: list[str] = _entity_list()
    processes: list[str] = _entity_list()


class LLMObservedEvidence(_LLMModel):
    category: EvidenceCategory
    description: str = Field(min_length=1, max_length=1000)
    excerpt: str = Field(description="Verbatim quote from the incident that shows this evidence", max_length=1000)


class LLMUnconfirmedEvidence(_LLMModel):
    category: EvidenceCategory
    description: str = Field(min_length=1, max_length=1000)
    rationale: str = Field(max_length=1000)


class LLMMissingEvidence(_LLMModel):
    category: EvidenceCategory
    description: str = Field(min_length=1, max_length=1000)
    critical: bool
    rationale: str = Field(max_length=1000)
    # Optional: the exact sentence in which the incident itself says this evidence is absent.
    # Lets the pipeline tell a gap *stated in the input* from a gap the model inferred.
    excerpt: str = Field(
        default="",
        description="Verbatim quote from the incident stating that this evidence is not included, or empty",
        max_length=1000,
    )


class LLMTechnique(_LLMModel):
    technique_id: str = Field(max_length=20)
    confidence: float = Field(ge=0.0, le=1.0)
    justification: str = Field(max_length=1000)


class LLMExtractionOutput(_LLMModel):
    # Field order is the order the model writes the answer in (structured output follows the
    # schema). The intent label comes after the summary, the observed evidence and the ATT&CK
    # techniques, so the model has described the activity before it commits to a label; with
    # the label first, a small model under constrained decoding defaults to the first allowed
    # label (live run, llama3.2:3b: 10 of 13 incidents labelled suspicious_process_execution).
    summary: str = Field(min_length=1, max_length=4000)
    observed_evidence: list[LLMObservedEvidence] = Field(max_length=50)
    attack_techniques: list[LLMTechnique] = Field(max_length=20)
    intent_label: str
    intent_confidence: float = Field(ge=0.0, le=1.0)
    entities: LLMEntities
    unconfirmed_evidence: list[LLMUnconfirmedEvidence] = Field(max_length=50)
    missing_evidence: list[LLMMissingEvidence] = Field(max_length=50)
    uncertainties: list[str] = Field(max_length=30)
    conflicts: list[str] = Field(max_length=30)


def output_json_schema(intent_labels: list[str]) -> dict[str, Any]:
    """JSON schema sent to the model, with the intent label restricted to the taxonomy."""
    schema = copy.deepcopy(LLMExtractionOutput.model_json_schema())
    schema["properties"]["intent_label"]["enum"] = list(intent_labels)
    return schema


class IntentDecision(_LLMModel):
    """Stage 1 of a two-stage analysis (paper 968 Section 3: intent extraction, then evidence extraction).

    The model states the activity, then the ATT&CK tactic it serves, then the label -- in that
    order, so the label follows from what the model has just written.
    """

    model_config = ConfigDict(title="IntentDecision", extra="ignore")

    activity: str = Field(default="", max_length=1000)
    technique_id: str = Field(default="none", max_length=20)  # the ATT&CK technique that grounds the tactic
    tactic: str = Field(default="none", max_length=40)
    intent_label: str
    intent_confidence: float

    @field_validator("intent_confidence", mode="before")
    @classmethod
    def _unit_interval(cls, value: Any) -> Any:
        # small models sometimes answer on a 0-100 scale
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = value / 100 if 1 < value <= 100 else value
            return min(1.0, max(0.0, float(value)))
        return value


def intent_decision_schema(intent_labels: list[str], tactics: list[str],
                           techniques: Optional[list[str]] = None) -> dict[str, Any]:  # fmt: skip
    """Stage-1 schema: activity, a technique (from ``techniques`` when hints exist), a tactic, then a label."""
    technique: dict[str, Any] = {"type": "string", "enum": [*techniques, "none"]} if techniques else {"type": "string"}
    return {
        "title": "IntentDecision",
        "type": "object",
        "properties": {
            "activity": {"type": "string"},
            "technique_id": technique,
            "tactic": {"type": "string", "enum": list(tactics)},
            "intent_label": {"type": "string", "enum": list(intent_labels)},
            "intent_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["activity", "technique_id", "tactic", "intent_label", "intent_confidence"],
    }
