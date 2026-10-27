"""Evidence enrichment and analyst-supplied evidence, through the API (milestone M2)."""

from __future__ import annotations

import pytest

from app.schemas.enums import EvidenceCategory, ExecutionMode, ToolFailurePolicy
from app.schemas.tools import ToolObservation, ToolResult
from app.tools.base import SecurityTool
from app.tools.registry import ToolGateway, ToolRegistry
from app.enrichment.enricher import EvidenceEnricher, default_categories
from app.tests.conftest import SYNTHETIC_DESCRIPTION, auth, good_output, issue_key, with_settings

HASH = "0987ace2bf850aacc7630a3b8aacfc40"
DESCRIPTION_WITH_HASH = SYNTHETIC_DESCRIPTION + f" The dropped file has MD5 {HASH}."


class FakeVirusTotal(SecurityTool):
    """Stands in for a real, configured VirusTotal (synthetic answers, but synthetic=False like a real tool)."""

    name = "virustotal"

    def __init__(self, *, fail: Exception | None = None) -> None:
        super().__init__()
        self.fail = fail
        self.calls = 0

    async def run(self, spec, target_type, target, context, params) -> ToolResult:
        self.calls += 1
        if self.fail:
            raise self.fail
        text = f"VirusTotal: 7/70 engines flag {target_type} {target} (verdict: malicious)"
        return self.result(spec, target_type, outputs={"reputation_report": {"verdict": "malicious"}},
                           observations=[ToolObservation(category=c, description=text) for c in spec.categories],
                           findings=[text])  # fmt: skip


def _analysed(client, description: str = DESCRIPTION_WITH_HASH) -> str:
    incident_id = client.post("/incidents", json={"description": description}).json()["incident_id"]
    assert client.post(f"/incidents/{incident_id}/analyze").status_code == 200
    return incident_id


def _output_with_ioc_gap() -> dict:
    output = good_output()
    output["entities"]["file_hashes"] = [HASH]
    output["missing_evidence"].append({"category": "indicators_of_compromise", "description": "Reputation of the dropped file",
                                       "critical": True, "rationale": "needed to judge the file"})  # fmt: skip
    return output


def test_enrich_requires_a_usable_analysis(make_client):
    client = make_client()
    incident_id = client.post("/incidents", json={"description": SYNTHETIC_DESCRIPTION}).json()["incident_id"]
    response = client.post(f"/incidents/{incident_id}/enrich")
    assert response.status_code == 409
    assert response.json()["error"]["category"] == "no_usable_analysis"
    assert client.post("/incidents/INC-NOPE/enrich").status_code == 404


def test_simulated_enrichment_is_never_observed_evidence(make_client, settings):
    client = make_client([_output_with_ioc_gap()])
    incident_id = _analysed(client)
    before = client.get(f"/incidents/{incident_id}/evidence").json()

    report = client.post(f"/incidents/{incident_id}/enrich").json()
    assert report["mode"] == "simulate"
    assert report["added_observed"] == 0
    assert report["categories_obtained"] == []
    assert all(a["synthetic"] for a in report["attempts"])

    after = client.get(f"/incidents/{incident_id}/evidence").json()
    assert after["tool_observed"] == []
    assert after["real_observed_categories"] == before["real_observed_categories"]
    assert after["base"] == before["base"]  # the analysis itself is immutable
    assert len(after["simulated"]) == report["added_simulated"] > 0
    assert all(s["synthetic"] and s["description"].startswith("SYNTHETIC") for s in after["simulated"])


def test_live_enrichment_records_real_observations_with_tool_provenance(make_client, settings):
    fake = FakeVirusTotal()
    live = with_settings(settings, execution_mode="live")
    client = make_client([_output_with_ioc_gap()], app_settings=live, real_tools={"virustotal": fake})
    incident_id = _analysed(client)

    report = client.post(f"/incidents/{incident_id}/enrich", json={"categories": ["indicators_of_compromise"]}).json()
    assert report["mode"] == "live" and fake.calls >= 1
    assert report["categories_obtained"] == ["indicators_of_compromise"]
    assert report["added_simulated"] == 0

    evidence = client.get(f"/incidents/{incident_id}/evidence").json()
    (item,) = [i for i in evidence["tool_observed"] if i["category"] == "indicators_of_compromise"][:1]
    assert item["status"] == "observed"
    assert item["provenance"]["origin"] == "tool"
    assert item["provenance"]["source_ref"].startswith("virustotal.")
    assert "indicators_of_compromise" in evidence["real_observed_categories"]

    # Repeating the enrichment does not duplicate stored evidence.
    client.post(f"/incidents/{incident_id}/enrich", json={"categories": ["indicators_of_compromise"]})
    again = client.get(f"/incidents/{incident_id}/evidence").json()
    assert len(again["tool_observed"]) == len(evidence["tool_observed"])


def test_live_tool_failure_creates_manual_task_and_no_evidence(make_client, settings):
    from app.core.errors import ToolTimeoutError

    live = with_settings(settings, execution_mode="live")
    client = make_client([_output_with_ioc_gap()], app_settings=live,
                         real_tools={"virustotal": FakeVirusTotal(fail=ToolTimeoutError("slow"))})  # fmt: skip
    incident_id = _analysed(client)
    report = client.post(f"/incidents/{incident_id}/enrich", json={"categories": ["indicators_of_compromise"]}).json()
    assert report["added_observed"] == 0 and report["added_simulated"] == 0
    assert report["manual_tasks"] and all(a["status"] == "manual_task" for a in report["attempts"])
    assert {a["error_category"] for a in report["attempts"]} == {"tool_timeout"}
    evidence = client.get(f"/incidents/{incident_id}/evidence").json()
    assert evidence["tool_observed"] == [] and evidence["simulated"] == []


def test_live_mode_error_policy_reports_errors(make_client, settings):
    live = with_settings(settings, execution_mode="live", tool_failure_policy="error")
    client = make_client([good_output()], app_settings=live)
    incident_id = _analysed(client)
    report = client.post(f"/incidents/{incident_id}/enrich", json={"categories": ["dns"]}).json()
    assert report["attempts"] and {a["status"] for a in report["attempts"]} == {"error"}
    assert report["manual_tasks"] == [] and report["categories_without_data"] == ["dns"]


def test_categories_without_tools_are_reported(make_client):
    client = make_client([good_output()])
    incident_id = _analysed(client)
    report = client.post(f"/incidents/{incident_id}/enrich", json={"categories": ["memory"]}).json()
    assert report["attempts"] == [] and report["categories_without_tools"] == ["memory"]


def test_tool_invocations_and_audit_are_recorded(make_client):
    client = make_client([good_output()])
    incident_id = _analysed(client)
    client.post(f"/incidents/{incident_id}/enrich", json={"categories": ["network_traffic"]})
    from sqlalchemy import text

    with client.app.state.database.engine.connect() as conn:
        rows = conn.execute(text("SELECT tool, purpose, synthetic, actor FROM tool_invocations")).all()
    assert rows and all(r.purpose == "enrichment" and r.synthetic and r.actor == "analyst-test" for r in rows)
    (event,) = client.app.state.audit.repository.list(action="evidence.enrich")
    assert event.details["mode"] == "simulate"


def test_analyst_supplied_evidence_is_attributed(make_client):
    client = make_client([good_output()], roles=("analyst",))
    incident_id = _analysed(client)
    response = client.post(f"/incidents/{incident_id}/evidence",
                           json={"category": "memory", "description": "SYNTHETIC: memory image shows injected thread",
                                 "excerpt": "volatility malfind output"},
                           headers={"X-CADE-Analyst": "not-me"})  # fmt: skip
    assert response.status_code == 201
    item = response.json()["item"]
    assert item["status"] == "observed"
    assert item["provenance"] == {"origin": "analyst_supplied", "source_ref": "analyst:analyst-test",
                                  "excerpt": "volatility malfind output"}  # fmt: skip
    evidence = client.get(f"/incidents/{incident_id}/evidence").json()
    assert "memory" in evidence["real_observed_categories"]
    (event,) = client.app.state.audit.repository.list(action="evidence.add")
    assert event.actor == "analyst-test" and event.claimed_label == "not-me"


def test_viewer_cannot_enrich_or_add_evidence(make_client):
    client = make_client([good_output()])
    incident_id = _analysed(client)
    viewer = auth(issue_key(client, "viewer", analyst="viewer-1"))
    assert client.post(f"/incidents/{incident_id}/enrich", headers=viewer).status_code == 403
    body = {"category": "dns", "description": "x"}
    assert client.post(f"/incidents/{incident_id}/evidence", json=body, headers=viewer).status_code == 403
    assert client.get(f"/incidents/{incident_id}/evidence", headers=viewer).status_code == 200


def test_analyst_evidence_rejects_unknown_fields(make_client):
    client = make_client([good_output()])
    incident_id = _analysed(client)
    body = {"category": "dns", "description": "x", "status": "observed", "provenance": {"origin": "tool"}}
    assert client.post(f"/incidents/{incident_id}/evidence", json=body).status_code == 422


def test_tools_endpoint_reports_effective_mode(make_client):
    tools = make_client().get("/tools").json()
    assert {t["name"] for t in tools} == {"virustotal", "wazuh", "elastic", "osquery", "yara", "sigma"}
    assert {t["effective"] for t in tools} == {"simulated"}


# --- enricher unit tests --------------------------------------------------------------------


@pytest.mark.anyio
async def test_enricher_plans_calls_from_grounded_entities(taxonomy, attack_mapper):
    from app.intent.llm_extractor import IntentExtractor
    from app.intent.mock_llm_client import MockLLMClient
    from app.tests.conftest import make_incident

    incident = await IntentExtractor(MockLLMClient([_output_with_ioc_gap()]), taxonomy, attack_mapper).extract(
        make_incident(DESCRIPTION_WITH_HASH)
    )
    enricher = EvidenceEnricher(ToolGateway(ToolRegistry(ExecutionMode.SIMULATE), ToolFailurePolicy.MANUAL_TASK),
                                max_calls=50)  # fmt: skip
    calls, without_tools = enricher.plan(incident, [EvidenceCategory.INDICATORS_OF_COMPROMISE, EvidenceCategory.MEMORY])
    keys = {(spec.key, target_type, value) for spec, target_type, value in calls}
    assert ("virustotal.file_reputation", "file_hash", HASH) in keys
    assert ("virustotal.ip_reputation", "ip_address", "203.0.113.45") in keys
    assert without_tools == [EvidenceCategory.MEMORY]
    capped = EvidenceEnricher(enricher.gateway, max_calls=1).plan(incident, [EvidenceCategory.INDICATORS_OF_COMPROMISE])
    assert len(capped[0]) == 1

    ordered = default_categories(incident, set())
    assert ordered[0] in {i.category for i in incident.evidence_profile.missing if i.criticality.value == "critical"}
    assert EvidenceCategory.PROCESS_TELEMETRY not in default_categories(incident, {EvidenceCategory.PROCESS_TELEMETRY})
