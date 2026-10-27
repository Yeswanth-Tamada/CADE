"""HTTP API behaviour, end to end through FastAPI with a mocked LLM."""

from __future__ import annotations

import json

from app.core.errors import LLMTimeoutError, LLMUnavailableError
from app.tests.conftest import EXAMPLES, SYNTHETIC_DESCRIPTION, good_output


def _create(client, **fields) -> str:
    response = client.post("/incidents", json={"description": SYNTHETIC_DESCRIPTION, **fields})
    assert response.status_code == 201, response.text
    return response.json()["incident_id"]


# --- POST /incidents --------------------------------------------------------------------


def test_create_structured_incident(make_client):
    client = make_client()
    response = client.post(
        "/incidents",
        json={"description": SYNTHETIC_DESCRIPTION, "source": "siem_alert", "severity": "high",
              "occurred_at": "2026-10-19T08:15:00Z"},
    )  # fmt: skip
    assert response.status_code == 201
    body = response.json()
    assert body["incident_id"].startswith("INC-")
    assert body["status"] == "received"
    assert body["source"] == "siem_alert"
    assert body["raw_description"] == SYNTHETIC_DESCRIPTION
    assert body["occurred_at"] == "2026-10-19T08:15:00Z"


def test_create_plain_text_incident(make_client):
    text = (EXAMPLES / "beaconing_plain_text.txt").read_text(encoding="utf-8")
    response = make_client().post("/incidents", content=text.encode(), headers={"Content-Type": "text/plain"})
    assert response.status_code == 201
    assert response.json()["raw_description"] == text
    assert response.json()["source"] == "analyst_description"


def test_duplicate_incident_id_conflicts(make_client):
    client = make_client()
    _create(client, incident_id="SYN-DUP-1")
    response = client.post("/incidents", json={"description": "x", "incident_id": "SYN-DUP-1"})
    assert response.status_code == 409
    assert response.json()["error"]["category"] == "conflict"


def test_invalid_input_returns_422_envelope_without_echoing_input(make_client):
    response = make_client().post("/incidents", json={"description": "x", "severity": "TOP-SECRET-VALUE"})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["category"] == "validation_error"
    assert error["details"][0]["loc"] == ["severity"]
    assert "TOP-SECRET-VALUE" not in response.text


def test_malformed_json_returns_422(make_client):
    response = make_client().post("/incidents", content=b"{oops", headers={"Content-Type": "application/json"})
    assert response.status_code == 422


def test_unsupported_media_type_returns_415(make_client):
    response = make_client().post("/incidents", content=b"<x/>", headers={"Content-Type": "application/xml"})
    assert response.status_code == 415
    assert response.json()["error"]["category"] == "unsupported_media_type"


def test_oversized_body_returns_413(make_client, settings):
    huge = "A" * (settings.max_description_chars * 4 + 300_000)
    response = make_client().post("/incidents", content=huge.encode(), headers={"Content-Type": "text/plain"})
    assert response.status_code == 413


# --- GET /incidents/{id} ----------------------------------------------------------------


def test_get_unknown_incident_returns_404(make_client):
    response = make_client().get("/incidents/INC-DOESNOTEXIST")
    assert response.status_code == 404
    assert response.json()["error"]["category"] == "not_found"


def test_invalid_incident_id_in_path_returns_422(make_client):
    assert make_client().get("/incidents/bad id with spaces").status_code == 422


def test_get_incident_before_analysis(make_client):
    client = make_client()
    incident_id = _create(client)
    body = client.get(f"/incidents/{incident_id}").json()
    assert body["incident"]["incident_id"] == incident_id
    assert body["latest_analysis"] is None


# --- POST /incidents/{id}/analyze -------------------------------------------------------


def test_full_workflow_submit_analyze_retrieve(make_client):
    client = make_client([good_output()])
    incident_id = _create(client)

    response = client.post(f"/incidents/{incident_id}/analyze")
    assert response.status_code == 200
    analysis = response.json()
    assert analysis["extraction_status"] == "success"
    result = analysis["result"]
    assert result["intent"]["label"] == "suspicious_process_execution"
    assert result["evidence_profile"]["critical_missing"]
    assert result["safety"] == {"destructive_actions_allowed": False, "human_approval_required": True}

    saved = client.get(f"/incidents/{incident_id}").json()
    assert saved["incident"]["status"] == "analyzed"
    assert saved["latest_analysis"]["analysis_id"] == analysis["analysis_id"]
    assert saved["latest_analysis"]["result"]["intent"] == result["intent"]


def test_analyze_unknown_incident_returns_404(make_client):
    assert make_client().post("/incidents/INC-NOPE/analyze").status_code == 404


def test_llm_unavailable_is_recorded_as_failure_not_success(make_client):
    """Regression: a failed extraction must never be reported or stored as successful."""
    client = make_client([LLMUnavailableError("cannot reach the Ollama server; is it running?")])
    incident_id = _create(client)

    response = client.post(f"/incidents/{incident_id}/analyze")
    assert response.status_code == 503
    assert response.json()["error"]["category"] == "llm_unavailable"

    saved = client.get(f"/incidents/{incident_id}").json()
    assert saved["incident"]["status"] == "analysis_failed"
    latest = saved["latest_analysis"]
    assert latest["extraction_status"] == "failed"
    assert latest["result"] is None
    assert latest["error"]["category"] == "llm_unavailable"


def test_llm_timeout_returns_504_and_is_recorded(make_client):
    client = make_client([LLMTimeoutError("the LLM did not respond within 120 seconds")])
    incident_id = _create(client)
    assert client.post(f"/incidents/{incident_id}/analyze").status_code == 504
    latest = client.get(f"/incidents/{incident_id}").json()["latest_analysis"]
    assert latest["extraction_status"] == "failed" and latest["error"]["category"] == "llm_timeout"


def test_malformed_llm_output_returns_502_and_is_recorded_as_failed(make_client):
    client = make_client(["not json", '{"still": "wrong"}'])
    incident_id = _create(client)
    response = client.post(f"/incidents/{incident_id}/analyze")
    assert response.status_code == 502
    assert response.json()["error"]["category"] == "extraction_failed"

    latest = client.get(f"/incidents/{incident_id}").json()["latest_analysis"]
    assert latest["extraction_status"] == "failed"
    assert latest["result"] is None
    assert latest["attempts"] == 2
    assert len(latest["warnings"]) == 2


def test_reanalysis_after_failure_succeeds_and_becomes_latest(make_client):
    client = make_client([LLMUnavailableError("down"), good_output()])
    incident_id = _create(client)
    assert client.post(f"/incidents/{incident_id}/analyze").status_code == 503
    second = client.post(f"/incidents/{incident_id}/analyze")
    assert second.status_code == 200
    saved = client.get(f"/incidents/{incident_id}").json()
    assert saved["incident"]["status"] == "analyzed"
    assert saved["latest_analysis"]["analysis_id"] == second.json()["analysis_id"]


def test_partial_extraction_is_labelled_partial(make_client):
    output = good_output()
    output["entities"]["hosts"].append("INVENTED-HOST")
    client = make_client([output])
    incident_id = _create(client)
    analysis = client.post(f"/incidents/{incident_id}/analyze").json()
    assert analysis["extraction_status"] == "partial"
    assert any("INVENTED-HOST" in w for w in analysis["warnings"])


def test_concurrent_analysis_is_rejected(make_client, repository):
    client = make_client([good_output()])
    incident_id = _create(client)
    from datetime import timedelta

    repository.start_analysis(incident_id, stale_after=timedelta(minutes=10))  # simulate one in progress
    response = client.post(f"/incidents/{incident_id}/analyze")
    assert response.status_code == 409


def test_unexpected_errors_are_sanitised(make_client):
    client = make_client([RuntimeError("database password is hunter2")])
    incident_id = _create(client)
    response = client.post(f"/incidents/{incident_id}/analyze")
    assert response.status_code == 500
    assert response.json() == {"error": {"category": "internal_error", "message": "internal server error"}}
    latest = client.get(f"/incidents/{incident_id}").json()["latest_analysis"]
    assert latest["extraction_status"] == "failed"
    assert "hunter2" not in json.dumps(latest)


def test_unverified_attack_mode_is_visible_in_results(make_client, unverified_mapper):
    client = make_client([good_output()], mapper=unverified_mapper)
    incident_id = _create(client)
    result = client.post(f"/incidents/{incident_id}/analyze").json()["result"]
    assert result["analysis"]["attack_mapping_mode"] == "unverified"
    assert {t["validation_status"] for t in result["intent"]["attack_techniques"]} == {"unverified"}


# --- docs -------------------------------------------------------------------------------


def test_interactive_docs_are_exposed(make_client):
    client = make_client()
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    assert "IncidentSubmission" in schema["components"]["schemas"]
    content = schema["paths"]["/incidents"]["post"]["requestBody"]["content"]
    assert {"application/json", "text/plain"} <= set(content)
