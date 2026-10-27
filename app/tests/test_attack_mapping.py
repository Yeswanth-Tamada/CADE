"""MITRE ATT&CK grounding, using a small STIX fixture (app/tests/fixtures/attack_fixture.json)."""

from __future__ import annotations

import pytest

from app.intent.attack_mapper import AttackDataset, AttackDatasetError, AttackMapper
from app.intent.grounding import WarningCollector
from app.intent.schemas import LLMTechnique
from app.schemas.enums import AttackMappingMode, AttackValidationStatus
from app.tests.conftest import FIXTURES


def _candidates(*items) -> list[LLMTechnique]:
    return [LLMTechnique(technique_id=t, confidence=c, justification="test") for t, c in items]


def _map(mapper: AttackMapper, *items):
    warnings = WarningCollector()
    return mapper.map_candidates(_candidates(*items), warnings), warnings


def test_fixture_dataset_loads(attack_mapper):
    dataset = attack_mapper.dataset
    assert dataset.version == "fixture-1.0"
    assert "T1059.001" in dataset.techniques
    assert "CAPEC-1" not in dataset.techniques
    assert attack_mapper.mode == AttackMappingMode.VERIFIED


def test_known_technique_is_validated_with_dataset_name_and_tactics(attack_mapper):
    (mapping,), warnings = _map(attack_mapper, ("T1078", 0.6))
    assert mapping.validation_status == AttackValidationStatus.VALIDATED
    assert mapping.name == "Valid Accounts"
    assert set(mapping.tactics) == {"defense-evasion", "persistence", "privilege-escalation", "initial-access"}
    assert not warnings.degraded


def test_subtechnique_and_lowercase_ids(attack_mapper):
    (mapping,), _ = _map(attack_mapper, (" t1059.001 ", 0.5))
    assert mapping.technique_id == "T1059.001"
    assert mapping.name == "PowerShell"


def test_unknown_technique_is_discarded_not_invented(attack_mapper):
    mappings, warnings = _map(attack_mapper, ("T9999", 0.9), ("T1110", 0.5))
    assert [m.technique_id for m in mappings] == ["T1110"]
    assert warnings.degraded
    assert any("T9999" in w for w in warnings.messages)


def test_malformed_technique_is_discarded(attack_mapper):
    mappings, warnings = _map(attack_mapper, ("Phishing", 0.9), ("T15", 0.5))
    assert mappings == []
    assert warnings.degraded


def test_revoked_and_deprecated_are_flagged_not_validated(attack_mapper):
    mappings, warnings = _map(attack_mapper, ("T1086", 0.5), ("T1062", 0.4))
    statuses = {m.technique_id: m.validation_status for m in mappings}
    assert statuses == {"T1086": AttackValidationStatus.REVOKED, "T1062": AttackValidationStatus.DEPRECATED}
    assert any("revoked" in w for w in warnings.messages)
    assert any("deprecated" in w for w in warnings.messages)


def test_duplicate_ids_prefer_the_active_object(attack_mapper):
    (mapping,), _ = _map(attack_mapper, ("T1486", 0.5))
    assert mapping.validation_status == AttackValidationStatus.VALIDATED
    assert mapping.name == "Data Encrypted for Impact"


def test_duplicate_candidates_keep_highest_confidence_and_sort(attack_mapper):
    mappings, _ = _map(attack_mapper, ("T1110", 0.2), ("T1059", 0.5), ("T1110", 0.7))
    assert [(m.technique_id, m.confidence) for m in mappings] == [("T1110", 0.7), ("T1059", 0.5)]


# --- unverified mode --------------------------------------------------------------------


def test_missing_dataset_gives_unverified_mode(unverified_mapper):
    assert unverified_mapper.mode == AttackMappingMode.UNVERIFIED
    assert unverified_mapper.status()["status"] == "unavailable"
    (mapping,), warnings = _map(unverified_mapper, ("T1059.001", 0.7))
    assert mapping.validation_status == AttackValidationStatus.UNVERIFIED
    assert mapping.name is None and mapping.tactics == []  # nothing claimed without the dataset
    assert any("not loaded" in w for w in warnings.messages)


def test_unverified_mode_still_rejects_malformed_ids(unverified_mapper):
    mappings, warnings = _map(unverified_mapper, ("T12", 0.7))
    assert mappings == [] and warnings.degraded


def test_corrupt_dataset_file_falls_back_to_unverified(tmp_path):
    path = tmp_path / "enterprise-attack.json"
    path.write_text("{ this is not json", encoding="utf-8")
    mapper = AttackMapper.from_path(path)
    assert mapper.mode == AttackMappingMode.UNVERIFIED
    assert "cannot read" in mapper.status()["detail"]


@pytest.mark.parametrize("bundle", [{}, {"objects": "nope"}, {"objects": [{"type": "intrusion-set"}]}])
def test_invalid_bundles_are_rejected(bundle):
    with pytest.raises(AttackDatasetError):
        AttackDataset.from_stix_bundle(bundle)


def test_fixture_file_path_exists():
    assert (FIXTURES / "attack_fixture.json").exists()
