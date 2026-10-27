"""ATT&CK / log-source ontology (paper Section 6.1): tactics, critical evidence, host log sources."""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import ValidationError

from app.core.config import PROJECT_ROOT
from app.enrichment.availability import AvailabilityProfile
from app.intent.llm_extractor import IntentExtractor
from app.intent.mock_llm_client import MockLLMClient
from app.playbook.vocabulary import Vocabulary
from app.routing.router import RetrievedCandidate, Router, StaticCandidateSource
from app.schemas.enums import EvidenceCategory as C
from app.schemas.enums import ProvenanceOrigin
from app.schemas.taxonomy import IntentDefinition
from app.schemas.tools import ToolRequest
from app.tests.conftest import good_output, make_incident
from app.tests.test_playbook_dag import pb
from app.tools import artifacts as A
from app.tools.elastic import ElasticTool
from app.tools.log_sources import default_log_sources

DATA = PROJECT_ROOT / "app" / "data"
PAPER_TACTICS = {"execution", "persistence", "privilege-escalation", "defense-evasion", "credential-access",
                 "discovery", "lateral-movement", "impact"}  # fmt: skip


@pytest.fixture(scope="module")
def vocabulary(taxonomy) -> Vocabulary:
    return Vocabulary.load(DATA / "action_vocabulary.json", DATA / "intent_goals.json", taxonomy)


# --- taxonomy ----------------------------------------------------------------------------------


def test_intents_cover_the_papers_eight_tactics_and_mark_critical_evidence(taxonomy, vocabulary):
    assert PAPER_TACTICS <= {t for intent in taxonomy.intents for t in intent.tactics}
    for label in ("privilege_escalation", "defense_evasion", "discovery", "system_impact"):
        intent = taxonomy.get(label)
        assert intent is not None and intent.critical_evidence and vocabulary.goals_for(label)
        assert set(intent.critical_evidence) | set(intent.supporting_evidence) == set(intent.relevant_evidence)


def test_taxonomy_rejects_unknown_tactics_and_critical_evidence_outside_relevant():
    base = {"label": "x_intent", "name": "X", "description": "x", "relevant_evidence": ["registry"]}
    assert IntentDefinition.model_validate({**base, "critical_evidence": ["registry"]}).supporting_evidence == []
    with pytest.raises(ValidationError):
        IntentDefinition.model_validate({**base, "tactics": ["hacking"]})
    with pytest.raises(ValidationError):
        IntentDefinition.model_validate({**base, "critical_evidence": ["dns"]})


def test_new_log_source_actions_close_the_new_categories(vocabulary):
    for category, action in ((C.REGISTRY, "query_registry_events"), (C.SCRIPT_EXECUTION, "query_script_blocks"),
                             (C.SCHEDULED_TASKS_SERVICES, "query_service_task_events")):  # fmt: skip
        assert action in {a.id for a in vocabulary.actions_consuming(category)}
        assert vocabulary.get(action).binding.startswith("elastic.")


# --- log sources ---------------------------------------------------------------------------------


def test_log_source_mapping_follows_the_paper():
    sources = default_log_sources()
    assert sources.queries(C.PROCESS_TELEMETRY) == [("Microsoft-Windows-Sysmon/Operational", ["1"]), ("Security", ["4688"])]
    assert sources.queries(C.REGISTRY) == [("Microsoft-Windows-Sysmon/Operational", ["12", "13", "14"])]
    assert sources.queries(C.SCRIPT_EXECUTION) == [("Microsoft-Windows-PowerShell/Operational", ["4104"])]
    assert sources.queries(C.AUTHENTICATION) == [("Security", ["4624", "4625", "4648"])]
    assert sources.queries(C.SCHEDULED_TASKS_SERVICES) == [("Security", ["4697", "4698"])]


def test_event_records_are_recognised_and_small_sysmon_ids_need_the_source_name():
    text = ("Sysmon EventID: 13 TargetObject=HKLM\\...\\Run\\updater\n"
            "EventCode=4104 ScriptBlockText=IEX(...)\n"
            "EventID: 1 (no source named on this line)\n"
            "event.code: 4698 task created")  # fmt: skip
    found = {(m.category, m.event_code) for m in default_log_sources().find(text)}
    assert found == {(C.REGISTRY, "13"), (C.SCRIPT_EXECUTION, "4104"), (C.SCHEDULED_TASKS_SERVICES, "4698")}


def test_event_digest_restates_records_without_adding_facts(taxonomy):
    from app.intent.prompts import DIGEST_START, build_user_prompt, render_incident
    from app.tests.conftest import make_incident

    records = ("EventCode=13 Channel=Microsoft-Windows-Sysmon/Operational Computer=WS-HR-019 User=CORP\\asmith "
               "TargetObject=HKU\\S-1\\Software\\Microsoft\\Windows\\CurrentVersion\\Run\\Upd Details=C:\\x\\upd.exe\n"
               "EventCode=1102 Channel=Security Computer=WS-HR-019 Message=The audit log was cleared.")  # fmt: skip
    digest = default_log_sources().digest(records)
    assert digest[0].startswith("Sysmon event 13 (registry value set): TargetObject=HKU\\S-1\\Software")
    assert digest[1] == "Windows Security event 1102 (audit log cleared): Message=The audit log was cleared.; Computer=WS-HR-019"
    incident = make_incident("SYNTHETIC. 2 events.", log_excerpt=records)
    prompt = build_user_prompt(incident, taxonomy)
    assert DIGEST_START in prompt and "registry value set" in prompt
    assert "registry value set" not in render_incident(incident)  # the digest is never grounding text


@pytest.mark.anyio
async def test_event_records_in_the_input_become_grounded_observed_evidence(taxonomy, attack_mapper):
    description = ("SYNTHETIC: PowerShell 4104 records from host WS-FIN-042.\n"
                   "Channel=Microsoft-Windows-PowerShell/Operational EventCode=4104 ScriptBlockText=IEX")  # fmt: skip
    output = good_output(observed_evidence=[])
    incident = await IntentExtractor(MockLLMClient([output]), taxonomy, attack_mapper).extract(make_incident(description))
    observed = [e for e in incident.evidence_profile.observed if e.category == C.SCRIPT_EXECUTION]
    assert len(observed) == 1 and observed[0].provenance.origin == ProvenanceOrigin.PATTERN_EXTRACTION
    assert observed[0].provenance.excerpt == "EventCode=4104" and observed[0].provenance.excerpt in description


@pytest.mark.anyio
async def test_elastic_event_code_search_filters_by_channel_and_code():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"hits": {"hits": [{"_source": {"event.code": "4104"}}]}})

    tool = ElasticTool(url="https://es.test:9200", api_key="k", transport=httpx.MockTransport(handler))
    result = await tool.invoke(ToolRequest(operation="script_block_events", target_type=A.HOST_ID, target="WS-1"))
    assert result.outputs[A.SCRIPT_BLOCKS] and {o.category for o in result.observations} == {C.SCRIPT_EXECUTION}
    body = json.dumps(seen["body"])
    assert '"winlog.channel": "Microsoft-Windows-PowerShell/Operational"' in body
    assert '"event.code": ["4104"]' in body and '"term": {"host.name": "WS-1"}' in body


# --- routing ---------------------------------------------------------------------------------------


def test_the_intents_critical_evidence_is_critical_for_routing(vocabulary):
    # query_dns_activity marks dns as supporting; an intent that marks dns critical makes it R_c
    playbook = pb("n1:extract_entities n2:query_dns_activity n3:generate_incident_report",
                  "n1>n2:host_id n2>n3:findings")  # fmt: skip
    router = Router(StaticCandidateSource(), vocabulary, AvailabilityProfile(version="test"))
    candidate = [RetrievedCandidate(entry_id="E1", playbook=playbook, similarity=0.9, score=0.9)]
    plain = router.score("INC-1", candidate, set(), {})
    marked = router.score("INC-1", candidate, set(), {}, critical_categories=frozenset({C.DNS}))
    assert C.DNS in plain.required_supporting and C.DNS not in plain.required_critical
    assert C.DNS in marked.required_critical and C.DNS in marked.missing_critical
