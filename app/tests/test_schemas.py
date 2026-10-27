"""Pydantic schema validation rules."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.core.errors import ConfigurationError
from app.schemas.enums import (
    AttackMappingMode,
    AttackValidationStatus,
    Criticality,
    EvidenceCategory,
    EvidenceStatus,
    ExtractionStatus,
    ProvenanceOrigin,
)
from app.schemas.incident import (
    AnalysisMetadata,
    AttackTechniqueMapping,
    EvidenceItem,
    EvidenceProfile,
    FileHash,
    IncidentSubmission,
    Provenance,
    SafetyFlags,
)
from app.schemas.records import AnalysisError, StoredAnalysis
from app.schemas.taxonomy import IntentTaxonomy

NOW = datetime(2026, 10, 19, tzinfo=timezone.utc)


def _item(status: EvidenceStatus, **kw) -> EvidenceItem:
    category = kw.pop("category", EvidenceCategory.PROCESS_TELEMETRY)
    description = kw.pop("description", "Process tree")
    return EvidenceItem(
        evidence_id=EvidenceItem.make_id(category, description),
        category=category,
        description=description,
        status=status,
        **kw,
    )


# --- submission -------------------------------------------------------------------------


def test_minimal_submission_is_valid():
    submission = IncidentSubmission(description="SYNTHETIC: suspicious login")
    assert submission.source.value == "analyst_description"
    assert submission.incident_id is None


@pytest.mark.parametrize(
    "payload",
    [
        {},  # description missing
        {"description": "   \n\t "},  # whitespace only
        {"description": "x", "unexpected_field": 1},  # unknown fields are rejected, not dropped
        {"description": "x", "source": "carrier_pigeon"},
        {"description": "x", "severity": "apocalyptic"},
        {"description": "x", "incident_id": "has spaces"},
        {"description": "x", "occurred_at": "yesterday-ish"},
        {"description": "x", "observables": {"ips": ["999.1.1.1"]}},
        {"description": "x", "observables": {"file_hashes": ["not-a-hash"]}},
        {"description": "x", "observables": {"domains": ["not a domain"]}},
    ],
)
def test_invalid_submissions_are_rejected(payload):
    with pytest.raises(ValidationError):
        IncidentSubmission.model_validate(payload)


def test_observables_are_canonicalised():
    obs = IncidentSubmission(
        description="x",
        observables={
            "ips": ["198.51.100[.]7", "198.51.100.7"],
            "domains": ["Evil[.]Example."],
            "file_hashes": ["A" * 64],
            "hosts": ["  WS-01  ", "ws-01"],
        },
    ).observables
    assert obs.ips == ["198.51.100.7"]
    assert obs.domains == ["evil.example"]
    assert obs.file_hashes == ["a" * 64]
    assert obs.hosts == ["WS-01"]


def test_file_hash_algorithm_must_match_length():
    assert FileHash(algorithm="md5", value="A" * 32).value == "a" * 32
    with pytest.raises(ValidationError):
        FileHash(algorithm="sha256", value="a" * 32)


# --- evidence ---------------------------------------------------------------------------


def test_evidence_id_is_stable_and_ignores_case_and_spacing():
    a = EvidenceItem.make_id(EvidenceCategory.DNS, "DNS  queries from host")
    b = EvidenceItem.make_id(EvidenceCategory.DNS, "dns queries from host")
    assert a == b and a.startswith("ev-")


def test_observed_evidence_requires_grounded_provenance():
    with pytest.raises(ValidationError, match="grounded provenance"):
        _item(EvidenceStatus.OBSERVED)
    with pytest.raises(ValidationError, match="grounded provenance"):
        _item(EvidenceStatus.OBSERVED, provenance=Provenance(origin=ProvenanceOrigin.LLM_INFERENCE))
    ok = _item(EvidenceStatus.OBSERVED, provenance=Provenance(origin=ProvenanceOrigin.INCIDENT_TEXT, excerpt="x"))
    assert ok.status == EvidenceStatus.OBSERVED


def test_missing_evidence_requires_criticality():
    with pytest.raises(ValidationError, match="criticality"):
        _item(EvidenceStatus.MISSING)


def test_critical_missing_is_derived_from_missing_items():
    critical = _item(EvidenceStatus.MISSING, criticality=Criticality.CRITICAL)
    supporting = _item(EvidenceStatus.MISSING, criticality=Criticality.SUPPORTING, description="DNS logs")
    profile = EvidenceProfile(missing=[critical, supporting])
    assert profile.critical_missing == [critical.evidence_id]


def test_inconsistent_critical_missing_is_rejected():
    supporting = _item(EvidenceStatus.MISSING, criticality=Criticality.SUPPORTING)
    with pytest.raises(ValidationError, match="critical_missing"):
        EvidenceProfile(missing=[supporting], critical_missing=[supporting.evidence_id])


def test_evidence_lists_must_hold_matching_statuses():
    inferred = _item(EvidenceStatus.INFERRED)
    with pytest.raises(ValidationError, match="observed"):
        EvidenceProfile(observed=[inferred])


# --- ATT&CK mapping ---------------------------------------------------------------------


def test_unverified_mapping_cannot_carry_a_name():
    with pytest.raises(ValidationError, match="unverified"):
        AttackTechniqueMapping(
            technique_id="T1059", name="Invented", confidence=0.5, validation_status=AttackValidationStatus.UNVERIFIED
        )


def test_validated_mapping_requires_dataset_name():
    with pytest.raises(ValidationError, match="validated"):
        AttackTechniqueMapping(technique_id="T1059", confidence=0.5, validation_status=AttackValidationStatus.VALIDATED)


@pytest.mark.parametrize("technique_id", ["T59", "T1059.1", "t1059", "TA0001", "T1059.0001"])
def test_malformed_technique_ids_are_rejected(technique_id):
    with pytest.raises(ValidationError):
        AttackTechniqueMapping(
            technique_id=technique_id, confidence=0.5, validation_status=AttackValidationStatus.UNVERIFIED
        )


# --- analysis metadata / regression guards ----------------------------------------------


def _metadata(status: ExtractionStatus) -> AnalysisMetadata:
    return AnalysisMetadata(
        extraction_status=status,
        provider="mock",
        model_name="m",
        prompt_version="v",
        attempts=1,
        started_at=NOW,
        completed_at=NOW,
        duration_ms=1,
        attack_mapping_mode=AttackMappingMode.UNVERIFIED,
        taxonomy_version="t",
    )


def test_canonical_metadata_can_never_say_failed():
    assert _metadata(ExtractionStatus.PARTIAL).extraction_status == ExtractionStatus.PARTIAL
    with pytest.raises(ValidationError, match="failed"):
        _metadata(ExtractionStatus.FAILED)


def test_failed_stored_analysis_cannot_have_a_result_and_success_needs_one():
    base = dict(analysis_id=1, incident_id="INC-1", provider="mock", model_name="m", attempts=1, duration_ms=1,
                created_at=NOW)  # fmt: skip
    with pytest.raises(ValidationError):
        StoredAnalysis(extraction_status=ExtractionStatus.FAILED, **base)  # no error given
    with pytest.raises(ValidationError):
        StoredAnalysis(extraction_status=ExtractionStatus.SUCCESS, **base)  # no result given
    failed = StoredAnalysis(
        extraction_status=ExtractionStatus.FAILED, error=AnalysisError(category="llm_timeout", message="t"), **base
    )
    assert failed.result is None


def test_safety_defaults_are_conservative():
    flags = SafetyFlags()
    assert flags.destructive_actions_allowed is False
    assert flags.human_approval_required is True
    with pytest.raises(ValidationError):
        SafetyFlags(destructive_actions_allowed=True, human_approval_required=False)


# --- taxonomy ---------------------------------------------------------------------------


def test_taxonomy_covers_required_categories(taxonomy):
    required = {
        "suspicious_process_execution", "malware_investigation", "phishing_investigation", "credential_compromise",
        "lateral_movement", "data_exfiltration", "denial_of_service", "web_application_attack",
        "command_and_control_persistence", "unknown",
    }  # fmt: skip
    assert required <= set(taxonomy.labels)


def test_taxonomy_without_unknown_is_rejected(tmp_path):
    path = tmp_path / "tax.json"
    path.write_text(
        json.dumps({"version": "x", "intents": [{"label": "a", "name": "A", "description": "d"},
                                                {"label": "b", "name": "B", "description": "d"}]}),
        encoding="utf-8",
    )  # fmt: skip
    with pytest.raises(ConfigurationError, match="unknown"):
        IntentTaxonomy.load(path)


def test_missing_taxonomy_file_is_a_configuration_error(tmp_path):
    with pytest.raises(ConfigurationError, match="not found"):
        IntentTaxonomy.load(tmp_path / "nope.json")
