"""Regression tests for behaviour observed while manually testing Phase 1.

Two SYNTHETIC scenarios were run against a real model (llama3.2:3b):

* Test A -- Word spawning encoded PowerShell (``examples/synthetic_incidents/encoded_powershell_edr.json``);
* Test B -- a suspicious scheduled task whose report explicitly lists the evidence it lacks.

The model is replaced here by scripted ``MockLLMClient`` answers shaped like the ones observed,
so these tests pin down what the *pipeline* does with such answers. They do not (and cannot)
test the quality of a real model; ``scripts/check_live_extraction.py`` reports that separately.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from app.intent.attack_mapper import AttackMapper
from app.intent.grounding import GroundingCorpus
from app.intent.llm_extractor import IntentExtractor
from app.intent.mock_llm_client import MockLLMClient
from app.intent.prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_user_prompt
from app.schemas.enums import (
    AttackValidationStatus,
    Criticality,
    EvidenceCategory,
    ExtractionStatus,
    ProvenanceOrigin,
)
from app.schemas.incident import CanonicalIncident
from app.tests.conftest import make_incident

pytestmark = pytest.mark.anyio

POWERSHELL_DESCRIPTION = (
    "SYNTHETIC. EDR alert on host WS-FIN-042: WINWORD.EXE spawned powershell.exe -NoP -W Hidden -enc "
    "SQBFAFgAIAAoAE4AZQB3AC0A under user jdoe. Seconds later the host connected to 203.0.113.45 on port 443 "
    "and resolved cdn-update[.]example. A file dropped to C:\\Users\\jdoe\\AppData\\Local\\Temp\\upd.dll has "
    "MD5 0987ace2bf850aacc7630a3b8aacfc40."
)

SCHEDULED_TASK_DESCRIPTION = (
    "Synthetic endpoint telemetry reports that a scheduled task named DailyTelemetryCheck was created on "
    "workstation WS-HR-019 by user jdoe. The task action points to C:\\ProgramData\\Telemetry\\telemetry.exe. "
    "Later process telemetry reports taskeng.exe launching telemetry.exe from that path. The incident bundle "
    "does not include the executable's file reputation, a file hash, the full task XML, or a related security "
    "alert. This is synthetic test data; no real task was created and no executable was run."
)
GAP_SENTENCE = (
    "The incident bundle does not include the executable's file reputation, a file hash, the full task XML, "
    "or a related security alert."
)


def _extractor(responses, taxonomy, mapper: AttackMapper, **kwargs) -> IntentExtractor:
    return IntentExtractor(MockLLMClient(responses), taxonomy, mapper, max_attempts=2, **kwargs)


def _scheduled_task_output(**overrides: Any) -> dict[str, Any]:
    output: dict[str, Any] = {
        "intent_label": "command_and_control_persistence",
        "intent_confidence": 0.7,
        "summary": "A scheduled task on WS-HR-019 runs an executable from ProgramData.",
        "entities": {"hosts": ["WS-HR-019"], "users": ["jdoe"], "ips": [], "domains": [], "file_hashes": [],
                     "processes": ["taskeng.exe", "telemetry.exe"]},
        "observed_evidence": [
            {"category": "scheduled_tasks_services", "description": "Scheduled task DailyTelemetryCheck created",
             "excerpt": "a scheduled task named DailyTelemetryCheck was created on workstation WS-HR-019"},
        ],
        "unconfirmed_evidence": [],
        "missing_evidence": [],
        "attack_techniques": [],
        "uncertainties": [],
        "conflicts": [],
    }  # fmt: skip
    output.update(copy.deepcopy(overrides))
    return output


def _stated_gaps() -> list[dict[str, Any]]:
    return [
        {"category": "indicators_of_compromise", "description": "File reputation of telemetry.exe", "critical": True,
         "rationale": "stated as not included", "excerpt": GAP_SENTENCE},
        {"category": "file_activity", "description": "File hash of telemetry.exe", "critical": True,
         "rationale": "stated as not included", "excerpt": GAP_SENTENCE},
        {"category": "scheduled_tasks_services", "description": "Full task XML", "critical": False,
         "rationale": "stated as not included", "excerpt": GAP_SENTENCE},
        {"category": "security_alert", "description": "Related security alert", "critical": False,
         "rationale": "stated as not included", "excerpt": GAP_SENTENCE},
    ]  # fmt: skip


def _scheduled_task_incident():
    return make_incident(SCHEDULED_TASK_DESCRIPTION, source="siem_alert", title="Suspicious scheduled task creation")


# --- Test A: PowerShell ------------------------------------------------------------------


async def test_powershell_scenario_regression(taxonomy, attack_mapper):
    """Test A: a non-domain 'domain' is discarded, the real (defanged) domain is still found."""
    output = {
        "intent_label": "suspicious_process_execution",
        "intent_confidence": 0.8,
        "summary": "Word spawned encoded PowerShell, followed by network activity and a dropped DLL.",
        "entities": {"hosts": ["WS-FIN-042"], "users": ["jdoe"], "ips": ["203.0.113.45"],
                     "domains": ["hxxp://cdn-update[.]example/payload"],
                     "file_hashes": ["0987ace2bf850aacc7630a3b8aacfc40"], "processes": ["WINWORD.EXE", "powershell.exe"]},
        "observed_evidence": [
            {"category": "process_telemetry", "description": "Encoded PowerShell spawned by WINWORD.EXE",
             "excerpt": "powershell.exe -NoP -W Hidden -enc SQBFAFgAIAAoAE4AZQB3AC0A"},
            {"category": "network_traffic", "description": "Host connected to 203.0.113.45 on port 443",
             "excerpt": "connected to 203.0.113.45 on port 443"},
        ],
        "unconfirmed_evidence": [],
        "missing_evidence": [],
        "attack_techniques": [{"technique_id": "T1059.001", "confidence": 0.6,
                               "justification": "powershell.exe was launched with -enc (encoded command)"}],
        "uncertainties": [],
        "conflicts": [],
    }  # fmt: skip
    incident = make_incident(POWERSHELL_DESCRIPTION, source="siem_alert")
    result = await _extractor([output], taxonomy, attack_mapper).extract(incident)

    assert result.analysis.extraction_status == ExtractionStatus.PARTIAL
    assert "discarded LLM-proposed domain with invalid format" in result.analysis.warnings
    assert result.entities.domains == ["cdn-update.example"]  # from deterministic pattern extraction
    assert [h.value for h in result.entities.file_hashes] == ["0987ace2bf850aacc7630a3b8aacfc40"]
    (technique,) = result.intent.attack_techniques
    assert technique.technique_id == "T1059.001"
    assert technique.validation_status == AttackValidationStatus.VALIDATED
    assert technique.name == "PowerShell" and technique.tactics == ["execution"]
    assert "-enc" in technique.justification
    assert result.analysis.attack_candidates_proposed == 1


# --- Test B: scheduled task -- ATT&CK ----------------------------------------------------


async def test_scheduled_task_scenario_with_t1053_005(taxonomy, attack_mapper):
    output = _scheduled_task_output(
        attack_techniques=[{"technique_id": "t1053.005", "confidence": 0.7,
                            "justification": "a scheduled task was created that launches telemetry.exe"}]
    )  # fmt: skip
    result = await _extractor([output], taxonomy, attack_mapper).extract(_scheduled_task_incident())

    (technique,) = result.intent.attack_techniques
    assert technique.technique_id == "T1053.005"
    assert technique.validation_status == AttackValidationStatus.VALIDATED
    # Name and tactics come from the dataset, never from the model (which sent none).
    assert technique.name == "Scheduled Task"
    assert technique.tactics == ["execution", "persistence", "privilege-escalation"]
    assert "scheduled task" in technique.justification


async def test_empty_attack_candidates_recorded_without_discard_warning(taxonomy, attack_mapper):
    """Test B observation: an empty technique list means the model proposed none (nothing was discarded)."""
    result = await _extractor([_scheduled_task_output()], taxonomy, attack_mapper).extract(_scheduled_task_incident())
    assert result.intent.attack_techniques == []
    assert result.analysis.attack_candidates_proposed == 0
    assert not any("ATT&CK technique" in w for w in result.analysis.warnings)


async def test_discarded_candidates_are_counted_as_proposed(taxonomy, attack_mapper):
    output = _scheduled_task_output(attack_techniques=[{"technique_id": "T9999", "confidence": 0.5,
                                                         "justification": "invented"}])  # fmt: skip
    result = await _extractor([output], taxonomy, attack_mapper).extract(_scheduled_task_incident())
    assert result.intent.attack_techniques == []
    assert result.analysis.attack_candidates_proposed == 1
    assert any("T9999" in w for w in result.analysis.warnings)


async def test_attack_hints_come_from_dataset_and_never_appear_unless_model_proposes(taxonomy, attack_mapper):
    hints = attack_mapper.technique_hints(SCHEDULED_TASK_DESCRIPTION)
    assert ("T1053.005", "Scheduled Task") in hints
    assert all(attack_mapper.dataset.techniques[t].name == name for t, name in hints)

    client = MockLLMClient([_scheduled_task_output()])
    extractor = IntentExtractor(client, taxonomy, attack_mapper, attack_prompt_hints=True)
    result = await extractor.extract(_scheduled_task_incident())
    assert "T1053.005" in client.calls[0]["user_prompt"]  # offered to the model as a hint...
    assert result.intent.attack_techniques == []  # ...but never turned into a result by the pipeline


async def test_attack_hints_can_be_disabled(taxonomy, attack_mapper):
    client = MockLLMClient([_scheduled_task_output()])
    await IntentExtractor(client, taxonomy, attack_mapper, attack_prompt_hints=False).extract(_scheduled_task_incident())
    assert "T1053.005" not in client.calls[0]["user_prompt"]


def test_attack_hints_need_a_dataset_and_whole_phrases(attack_mapper, unverified_mapper):
    assert unverified_mapper.technique_hints(SCHEDULED_TASK_DESCRIPTION) == []
    assert attack_mapper.technique_hints("a PowerShellish string and scheduled tasking") == []
    assert ("T1059.001", "PowerShell") in attack_mapper.technique_hints("ran PowerShell -enc ...")


# --- Test B: observed-evidence grounding -------------------------------------------------


async def test_paraphrased_observed_excerpt_is_downgraded(taxonomy, attack_mapper):
    """Test B: the input says 'from that path'; a quote that adds the full path is not a quote."""
    output = _scheduled_task_output()
    output["observed_evidence"].append(
        {"category": "process_telemetry", "description": "taskeng.exe launched telemetry.exe",
         "excerpt": "taskeng.exe launching telemetry.exe from C:\\ProgramData\\Telemetry\\telemetry.exe"}
    )  # fmt: skip
    result = await _extractor([output], taxonomy, attack_mapper).extract(_scheduled_task_incident())
    assert "taskeng.exe launched telemetry.exe" in [e.description for e in result.evidence_profile.unconfirmed]
    assert result.analysis.extraction_status == ExtractionStatus.PARTIAL


async def test_exact_windows_path_excerpt_with_escaped_backslashes_is_grounded(taxonomy, attack_mapper):
    """Models often JSON-escape twice, returning ``C:\\\\ProgramData`` for an exact quote of ``C:\\ProgramData``."""
    output = _scheduled_task_output()
    output["observed_evidence"].append(
        {"category": "file_activity", "description": "Task action executable path",
         "excerpt": "The task action points to C:\\\\ProgramData\\\\Telemetry\\\\telemetry.exe"}
    )  # fmt: skip
    result = await _extractor([output], taxonomy, attack_mapper).extract(_scheduled_task_incident())
    assert "Task action executable path" in [e.description for e in result.evidence_profile.observed]
    assert result.analysis.extraction_status == ExtractionStatus.SUCCESS


def test_backslash_normalisation_does_not_ground_a_different_path():
    corpus = GroundingCorpus(SCHEDULED_TASK_DESCRIPTION)
    assert corpus.contains_excerpt("C:\\\\ProgramData\\\\Telemetry\\\\telemetry.exe")
    assert not corpus.contains_excerpt("C:\\ProgramData\\Other\\telemetry.exe")
    assert not corpus.contains_excerpt("C:\\\\ProgramData\\\\Telemetry\\\\other.exe")
    assert not corpus.contains_excerpt("C:ProgramDataTelemetrytelemetry.exe")


# --- Test B: stated vs. inferred missing evidence ----------------------------------------


async def test_stated_gap_missing_item_has_incident_text_provenance(taxonomy, attack_mapper):
    output = _scheduled_task_output(missing_evidence=_stated_gaps())
    result = await _extractor([output], taxonomy, attack_mapper).extract(_scheduled_task_incident())
    profile = result.evidence_profile

    assert {e.category for e in profile.missing} == {
        EvidenceCategory.INDICATORS_OF_COMPROMISE, EvidenceCategory.FILE_ACTIVITY,
        EvidenceCategory.SCHEDULED_TASKS_SERVICES, EvidenceCategory.SECURITY_ALERT,
    }  # fmt: skip
    for item in profile.missing:
        assert item.provenance.origin == ProvenanceOrigin.INCIDENT_TEXT
        assert item.provenance.excerpt == GAP_SENTENCE
    assert len(profile.critical_missing) == 2
    assert {i.criticality for i in profile.missing} == {Criticality.CRITICAL, Criticality.SUPPORTING}
    # The stated gap is touched, so it is not reported as "not checked".
    assert EvidenceCategory.SECURITY_ALERT not in profile.not_checked
    assert not any("states that evidence" in w for w in result.analysis.warnings)
    assert result.analysis.extraction_status == ExtractionStatus.SUCCESS


async def test_inferred_gap_missing_item_has_llm_provenance(taxonomy, attack_mapper):
    output = _scheduled_task_output(missing_evidence=_stated_gaps() + [
        {"category": "network_traffic", "description": "Network connections made by telemetry.exe",
         "critical": False, "rationale": "would show C2 if any"}])  # fmt: skip
    result = await _extractor([output], taxonomy, attack_mapper).extract(_scheduled_task_incident())
    inferred = next(e for e in result.evidence_profile.missing if e.category == EvidenceCategory.NETWORK_TRAFFIC)
    assert inferred.provenance.origin == ProvenanceOrigin.LLM_INFERENCE
    assert inferred.provenance.excerpt is None


async def test_unverifiable_missing_excerpt_falls_back_to_llm_inference(taxonomy, attack_mapper):
    output = _scheduled_task_output(missing_evidence=_stated_gaps() + [
        {"category": "memory", "description": "Memory image of WS-HR-019", "critical": False,
         "rationale": "x", "excerpt": "The bundle does not include a memory image."}])  # fmt: skip
    result = await _extractor([output], taxonomy, attack_mapper).extract(_scheduled_task_incident())
    memory = next(e for e in result.evidence_profile.missing if e.category == EvidenceCategory.MEMORY)
    assert memory.provenance.origin == ProvenanceOrigin.LLM_INFERENCE
    assert memory.provenance.excerpt is None
    assert any("Memory image" in w and "could not be verified" in w for w in result.analysis.warnings)
    # A gap the model merely failed to quote is not a reason to downgrade the whole analysis.
    assert result.analysis.extraction_status == ExtractionStatus.SUCCESS


async def test_a_bare_absence_fragment_does_not_make_unrelated_gaps_stated(taxonomy, attack_mapper):
    """Regression (live run, llama3.2:3b): the model quoted only "does not include" for email, registry and
    scheduled-task gaps, and the fragment occurs in the input, so all of them became stated by the incident."""
    fragment = "does not include"
    output = _scheduled_task_output(missing_evidence=[
        {"category": "email", "description": "Email events", "critical": False, "rationale": "x", "excerpt": fragment},
        {"category": "registry", "description": "Registry changes", "critical": False, "rationale": "x",
         "excerpt": fragment},
        # a fragment inside the statement that names this evidence is still a stated gap
        {"category": "security_alert", "description": "Related security alert", "critical": False, "rationale": "x",
         "excerpt": fragment},
    ])  # fmt: skip
    result = await _extractor([output], taxonomy, attack_mapper).extract(_scheduled_task_incident())
    origins = {e.category: e.provenance for e in result.evidence_profile.missing}
    assert origins[EvidenceCategory.EMAIL].origin == ProvenanceOrigin.LLM_INFERENCE
    assert origins[EvidenceCategory.REGISTRY].origin == ProvenanceOrigin.LLM_INFERENCE
    assert origins[EvidenceCategory.SECURITY_ALERT].origin == ProvenanceOrigin.INCIDENT_TEXT
    assert origins[EvidenceCategory.SECURITY_ALERT].excerpt == GAP_SENTENCE  # the whole statement, not the fragment
    assert any("no statement there names this evidence" in w for w in result.analysis.warnings)


async def test_unrecorded_stated_gap_produces_info_warning(taxonomy, attack_mapper):
    """Test B observation: the model filed the stated gaps under 'unconfirmed' instead of 'missing'."""
    output = _scheduled_task_output(
        unconfirmed_evidence=[{"category": "file_activity", "description": GAP_SENTENCE, "rationale": "lacking"}],
        missing_evidence=[{"category": "file_activity", "description": "The full task XML and related security alert.",
                           "critical": True, "rationale": "needed"}],
    )  # fmt: skip
    result = await _extractor([output], taxonomy, attack_mapper).extract(_scheduled_task_incident())
    warnings = [w for w in result.analysis.warnings if "states that evidence" in w]
    assert len(warnings) == 1
    # Info only: the pipeline does not invent missing items on the model's behalf.
    assert result.analysis.extraction_status == ExtractionStatus.SUCCESS
    assert len(result.evidence_profile.missing) == 1


def test_prompt_contains_gap_category_and_tactic_rules(taxonomy):
    assert PROMPT_VERSION == "intent-extraction-v4"
    prompt = build_user_prompt(_scheduled_task_incident(), taxonomy)
    # v3: labels are ATT&CK-grounded (tactics in brackets) and chosen by tactic
    assert "- privilege_escalation [privilege-escalation]:" in prompt
    assert "ATT&CK tactic" in SYSTEM_PROMPT
    assert prompt.index("- discovery") < prompt.index("- suspicious_process_execution")
    assert "excerpt" in prompt.split("missing_evidence", 1)[1][:200]
    assert "does not include" in SYSTEM_PROMPT
    for category in ("scheduled_tasks_services", "indicators_of_compromise", "security_alert"):
        assert category in SYSTEM_PROMPT
    assert "character for character" in SYSTEM_PROMPT


# --- persistence / compatibility ---------------------------------------------------------


async def test_stored_v1_analysis_json_still_loads(taxonomy, attack_mapper):
    result = await _extractor([_scheduled_task_output()], taxonomy, attack_mapper).extract(_scheduled_task_incident())
    data = result.model_dump(mode="json")
    del data["analysis"]["attack_candidates_proposed"]
    data["analysis"]["prompt_version"] = "intent-extraction-v1"
    restored = CanonicalIncident.model_validate(data)
    assert restored.analysis.attack_candidates_proposed is None


def test_get_after_analysis_returns_same_analysis(make_client):
    client = make_client([_scheduled_task_output(missing_evidence=_stated_gaps())])
    created = client.post(
        "/incidents",
        json={"description": SCHEDULED_TASK_DESCRIPTION, "source": "siem_alert",
              "attributes": {"original_source": "synthetic-edr"}},
    )  # fmt: skip
    assert created.status_code == 201
    incident_id = created.json()["incident_id"]
    before = client.get(f"/incidents/{incident_id}").json()
    assert before["incident"]["status"] == "received" and before["latest_analysis"] is None

    analysis = client.post(f"/incidents/{incident_id}/analyze")
    assert analysis.status_code == 200
    saved = client.get(f"/incidents/{incident_id}").json()
    assert saved["incident"]["status"] == "analyzed"
    assert saved["incident"]["analysis_started_at"] is None
    assert saved["incident"]["attributes"] == {"original_source": "synthetic-edr"}
    assert saved["latest_analysis"] == analysis.json()


def test_successful_reanalysis_appends_and_becomes_latest(make_client):
    client = make_client([_scheduled_task_output(), _scheduled_task_output(intent_confidence=0.9)])
    incident_id = client.post("/incidents", json={"description": SCHEDULED_TASK_DESCRIPTION}).json()["incident_id"]
    first = client.post(f"/incidents/{incident_id}/analyze").json()
    second = client.post(f"/incidents/{incident_id}/analyze").json()
    assert second["analysis_id"] > first["analysis_id"]
    latest = client.get(f"/incidents/{incident_id}").json()["latest_analysis"]
    assert latest["analysis_id"] == second["analysis_id"]
    assert latest["result"]["intent"]["confidence"] == 0.9
