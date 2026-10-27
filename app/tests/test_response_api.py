"""End-to-end response pipeline through the HTTP API, with role-separated API keys (milestone M10).

Mock LLM, hashing embedder, NumPy index and simulated tools: no network, no real tools.
"""

from __future__ import annotations

import re
import time

import pytest

from app.schemas.routing import RoutingAction
from app.tests.conftest import SYNTHETIC_DESCRIPTION, auth, good_output, issue_key, with_settings
from app.tests.test_playbook_dag import seeds

HOST = "WS-FIN-042"


@pytest.fixture
def client(make_client):
    return make_client([good_output()] * 5)


@pytest.fixture
def keys(client) -> dict[str, dict[str, str]]:
    return {role: auth(issue_key(client, role, analyst=f"{role}-1"))
            for role in ("analyst", "approver", "executor", "viewer", "library_admin")}  # fmt: skip


def analysed(client, headers) -> str:
    incident_id = client.post("/incidents", json={"description": SYNTHETIC_DESCRIPTION, "title": "SYNTHETIC test"},
                              headers=headers).json()["incident_id"]  # fmt: skip
    assert client.post(f"/incidents/{incident_id}/analyze", headers=headers).status_code == 200
    return incident_id


def wait_for(client, job_id: str, headers, timeout: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/jobs/{job_id}", headers=headers).json()
        if job["status"] in ("succeeded", "failed", "interrupted"):
            return job
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_respond_requires_a_usable_analysis(client, keys):
    incident_id = client.post("/incidents", json={"description": SYNTHETIC_DESCRIPTION}, headers=keys["analyst"]).json()["incident_id"]
    response = client.post(f"/incidents/{incident_id}/respond?wait=true", headers=keys["analyst"])
    assert response.status_code == 409 and response.json()["error"]["category"] == "no_usable_analysis"


def test_route_records_every_term(client, keys):
    incident_id = analysed(client, keys["analyst"])
    decision = client.post(f"/incidents/{incident_id}/route", headers=keys["analyst"]).json()
    assert decision["action"] in {a.value for a in RoutingAction}
    assert set(decision["terms"]) >= {"coverage", "counterfactual_coverage", "entropy", "requestability",
                                      "unmet_fraction", "smoothed_success"}  # fmt: skip
    assert decision["top_candidate"] is not None and decision["candidates"]
    assert client.post(f"/incidents/{incident_id}/route", headers=keys["viewer"]).status_code == 403


def test_full_pipeline_with_separated_roles(client, keys):
    incident_id = analysed(client, keys["analyst"])

    outcome = client.post(f"/incidents/{incident_id}/synthesize?wait=true", headers=keys["analyst"]).json()
    plan = outcome["plan"]
    assert outcome["status"] == "plan_created" and outcome["forced_synthesis"]
    assert plan["source"] == "synthesized" and plan["fitness"]["fitness"] > 0
    synthesis = plan["synthesis"]
    assert synthesis["seeding"]["k"] >= 1 and synthesis["evolution"]["history"]
    assert synthesis["generator"]["provider"] == "mock"
    assert plan["safety"] == {"destructive_actions_allowed": False, "human_approval_required": True}

    plan_id = plan["plan_id"]
    # an analyst cannot approve or execute; an approver cannot execute
    if plan["destructive_nodes"]:
        body = {"node_ids": plan["destructive_nodes"]}
        assert client.post(f"/response-plans/{plan_id}/approve", json=body, headers=keys["analyst"]).status_code == 403
        assert client.post(f"/response-plans/{plan_id}/approve", json=body, headers=keys["approver"]).status_code == 200
    assert client.post(f"/response-plans/{plan_id}/execute", headers=keys["analyst"]).status_code == 403
    assert client.post(f"/response-plans/{plan_id}/execute", headers=keys["approver"]).status_code == 403

    run = client.post(f"/response-plans/{plan_id}/execute", headers=keys["executor"]).json()
    assert run["mode"] == "simulate" and run["synthetic"] and run["started_by"] == "executor-1"
    assert run["report"]
    assert client.post(f"/response-plans/{plan_id}/execute", headers=keys["executor"]).status_code == 409

    assert client.post(f"/playbook-runs/{run['run_id']}/feedback", json={"resolved": True, "findings_retained": True},
                       headers=keys["executor"]).status_code == 403  # fmt: skip
    feedback = client.post(f"/playbook-runs/{run['run_id']}/feedback", json={"resolved": True, "findings_retained": True},
                           headers=keys["analyst"]).json()["feedback"]  # fmt: skip
    assert feedback["library_outcome"]["action"] == "not_stored"  # simulated runs do not train the library

    history = client.get(f"/incidents/{incident_id}/response", headers=keys["viewer"]).json()
    assert history["decisions"] and [p["plan_id"] for p in history["plans"]] == [plan_id]
    assert history["runs"][0]["feedback"]["submitted_by"] == "analyst-1"
    assert client.get(f"/response-plans/{plan_id}", headers=keys["viewer"]).json()["status"] == "executed"


def test_reuse_path_and_closed_loop_learning(make_client, settings):
    client = make_client([good_output()] * 3, app_settings=with_settings(settings, library_learn_from_simulated=True))
    incident_id = analysed(client, client.headers)
    outcome = client.post(f"/incidents/{incident_id}/respond?wait=true").json()
    plan = outcome["plan"]
    run = client.post(f"/response-plans/{plan['plan_id']}/execute").json()
    feedback = client.post(f"/playbook-runs/{run['run_id']}/feedback",
                           json={"resolved": True, "findings_retained": True}).json()["feedback"]  # fmt: skip
    action = feedback["library_outcome"]["action"]
    if plan["source"] == "library":
        assert action == "updated" and feedback["library_outcome"]["entry_id"] == plan["library_entry_id"]
    else:
        assert action in ("inserted", "merged")
    entries = client.get("/library/playbooks").json()
    assert any(e["executions"] >= 1 for e in entries)


def test_background_job_runs_and_reports(client, keys):
    incident_id = analysed(client, keys["analyst"])
    response = client.post(f"/incidents/{incident_id}/respond", headers=keys["analyst"])
    assert response.status_code == 202
    job = wait_for(client, response.json()["job_id"], keys["viewer"])
    assert job["status"] == "succeeded" and job["result"]["plan"]["plan_id"].startswith("PLAN-")
    assert job["created_by"] == "analyst-1"


def test_evidence_request_loop_is_bounded(client, keys, monkeypatch):
    incident_id = analysed(client, keys["analyst"])
    router = client.app.state.components.router
    original = router.decide

    def always_request(*args, **kwargs):
        decision = original(*args, **kwargs)
        return decision.model_copy(update={"action": RoutingAction.REQUEST_EVIDENCE})

    monkeypatch.setattr(router, "decide", always_request)
    waiting = client.post(f"/incidents/{incident_id}/respond?wait=true&auto_acquire=false", headers=keys["analyst"]).json()
    assert waiting["status"] == "awaiting_evidence" and waiting["plan"] is None and len(waiting["decisions"]) == 1

    outcome = client.post(f"/incidents/{incident_id}/respond?wait=true", headers=keys["analyst"]).json()
    rounds = client.app.state.settings.max_evidence_rounds
    assert len(outcome["enrichment"]) == rounds
    assert len(outcome["decisions"]) == rounds + 1  # one per request round, then the final (request-free) round
    assert outcome["plan"]["source"] == "synthesized"  # the request option was exhausted


def test_the_final_round_never_offers_an_evidence_request(client, keys):
    incident_id = analysed(client, keys["analyst"])
    outcome = client.post(f"/incidents/{incident_id}/respond?wait=true", headers=keys["analyst"]).json()
    rounds = client.app.state.settings.max_evidence_rounds
    assert len(outcome["decisions"]) <= rounds + 1
    final = outcome["decisions"][-1]
    assert final["action"] != "request_evidence"
    if final["round"] > rounds:
        assert "request_evidence" not in final["ucb"]  # masked for the bandit too
    assert all(len(d["features"]) == 9 and d["policy"] == "rule" for d in outcome["decisions"])  # still bootstrapping


def test_library_endpoints_and_import_validation(client, keys):
    entries = client.get("/library/playbooks", headers=keys["viewer"]).json()
    assert len(entries) == len(seeds()) and all(e["origin"] == "seed" for e in entries)
    one = client.get(f"/library/playbooks/{entries[0]['entry_id']}", headers=keys["viewer"]).json()
    assert one["playbook"]["nodes"]
    assert client.get("/library/playbooks/PB-0000000000000000", headers=keys["viewer"]).status_code == 404

    seed = next(s for s in seeds() if s["intent_label"] == "denial_of_service")
    next(n for n in seed["nodes"] if n["action"] == "query_ip_connections")["params"] = {"lookback_hours": 6}
    body = {"playbook": seed, "description": "SYNTHETIC imported playbook"}
    assert client.post("/library/playbooks", json=body, headers=keys["analyst"]).status_code == 403
    created = client.post("/library/playbooks", json=body, headers=keys["library_admin"])
    assert created.status_code == 201 and created.json()["origin"] == "imported"

    unsafe = {"playbook": {"name": "unsafe", "intent_label": "ransomware_investigation", "origin": "llm",
                           "nodes": [{"node_id": "n1", "action": "extract_entities"}, {"node_id": "n2", "action": "isolate_host"},
                                     {"node_id": "n3", "action": "generate_incident_report"}],
                           "edges": [{"source": "n1", "target": "n2", "artifact": "host_id"},
                                     {"source": "n2", "target": "n3", "artifact": "findings"}]}}  # fmt: skip
    refused = client.post("/library/playbooks", json=unsafe, headers=keys["library_admin"])
    assert refused.status_code == 422 and refused.json()["error"]["category"] == "invalid_playbook"


def test_vocabulary_incident_list_and_health(client, keys):
    vocabulary = client.get("/vocabulary", headers=keys["viewer"]).json()
    assert len(vocabulary["actions"]) >= 40 and "ransomware_investigation" in vocabulary["goals"]
    incident_id = analysed(client, keys["analyst"])
    listed = client.get("/incidents", headers=keys["viewer"]).json()
    assert listed[0]["incident_id"] == incident_id and "raw_description" not in listed[0]
    health = client.get("/health", headers=keys["viewer"]).json()
    assert health["playbooks"]["execution_mode"] == "simulate"
    assert health["playbooks"]["library_entries"] == len(seeds())
    assert set(health["playbooks"]["tools"].values()) == {"simulated"}


def test_interrupted_jobs_are_marked_on_startup(make_client, settings):
    first = make_client()
    first.app.state.jobs.repository.create("JOB-0123456789ABCDEF", "respond", None, "someone")
    first.app.state.jobs.repository.update("JOB-0123456789ABCDEF", progress="synthesizing")
    second = make_client()
    job = second.get("/jobs/JOB-0123456789ABCDEF").json()
    assert job["status"] == "interrupted" and job["error"]["category"] == "interrupted"


def test_console_is_served_without_external_assets(make_client):
    anonymous = make_client(roles=())
    page = anonymous.get("/console/")
    assert page.status_code == 200 and "CADE Analyst Console" in page.text
    assert not re.search(r"(src|href)=\"https?://", page.text)
    script = anonymous.get("/console/app.js").text
    assert "innerHTML" not in script and "fetch(" in script
    assert anonymous.get("/incidents").status_code == 401  # the data behind it still needs a key
