"""GET /health must distinguish API health from optional-dependency health."""

from __future__ import annotations

import httpx

from app.core.config import Settings
from app.intent.llm_client import OllamaClient


def _ollama(handler) -> OllamaClient:
    return OllamaClient(
        base_url="http://ollama.test", model="llama3.2:3b", timeout_seconds=5, transport=httpx.MockTransport(handler)
    )


def test_health_with_mock_llm_is_degraded_not_ok(make_client):
    body = make_client().get("/health").json()
    assert body["api"]["status"] == "ok"
    assert body["dependencies"]["database"]["status"] == "ok"
    # A mock is never reported as a healthy real LLM.
    assert body["dependencies"]["llm"]["status"] == "mock"
    assert body["status"] == "degraded"


def test_health_reports_unreachable_ollama_without_failing_the_api(make_client):
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    response = make_client(llm_client=_ollama(refuse)).get("/health")
    assert response.status_code == 200  # the API itself is up
    body = response.json()
    llm = body["dependencies"]["llm"]
    assert llm["status"] == "unavailable"
    assert llm["reachable"] is False
    assert body["status"] == "degraded"


def test_health_reports_missing_model(make_client):
    def tags(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "some-other-model:latest"}]})

    llm = make_client(llm_client=_ollama(tags)).get("/health").json()["dependencies"]["llm"]
    assert llm["reachable"] is True
    assert llm["model_available"] is False
    assert llm["status"] == "model_missing"


def test_health_ok_when_all_dependencies_ready(make_client):
    def tags(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": "llama3.2:3b"}]})

    body = make_client(llm_client=_ollama(tags)).get("/health").json()
    assert body["dependencies"]["llm"]["status"] == "ok"
    assert body["dependencies"]["attack_dataset"]["status"] == "loaded"
    assert body["dependencies"]["attack_dataset"]["version"] == "fixture-1.0"
    assert body["status"] == "ok"


def test_health_reports_unverified_attack_mode(make_client, unverified_mapper):
    attack = make_client(mapper=unverified_mapper).get("/health").json()["dependencies"]["attack_dataset"]
    assert attack["status"] == "unavailable"
    assert attack["mapping_mode"] == "unverified"
    assert attack["version"] is None


def test_log_level_is_case_insensitive():
    assert Settings(log_level="debug").log_level == "DEBUG"
