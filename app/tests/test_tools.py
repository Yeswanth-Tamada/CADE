"""Security-tool adapters, the simulator and the gateway's failure policy (milestone M2).

No test contacts a real tool: HTTP adapters run against ``httpx.MockTransport`` and command-line
adapters against a fake runner. All indicators are from documentation ranges / synthetic.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from app.core.errors import (
    ToolAuthError,
    ToolError,
    ToolInputError,
    ToolRateLimitedError,
    ToolTimeoutError,
    ToolUnavailableError,
)
from app.enrichment.availability import AvailabilityProfile
from app.schemas.enums import EvidenceCategory, ExecutionMode, ToolFailurePolicy, ToolStatus
from app.schemas.tools import OperationSpec, ToolRequest, ToolResult
from app.tools import artifacts as A
from app.tools.base import CommandResult, RateLimiter, SecurityTool
from app.tools.catalog import CATALOG, OPERATIONS, enrichment_operations
from app.tools.elastic import ElasticTool
from app.tools.osquery import OsqueryTool
from app.tools.registry import ToolGateway, ToolRegistry, build_real_tools
from app.tools.sigma import SigmaTool
from app.tools.simulated import SimulatedTool
from app.tools.virustotal import VirusTotalTool
from app.tools.wazuh import WazuhTool
from app.tools.yara import YaraTool
from app.tests.conftest import with_settings

pytestmark = pytest.mark.anyio

HASH = "0987ace2bf850aacc7630a3b8aacfc40"
VT_KEY = "vt-secret-key-123"


def req(operation: str, target_type: str, target: str, **kwargs) -> ToolRequest:
    return ToolRequest(operation=operation, target_type=target_type, target=target, **kwargs)


class SpyTool(SecurityTool):
    """A 'real' tool that records calls and returns a fixed result or raises."""

    def __init__(self, name: str, *, error: Exception | None = None) -> None:
        self.name = name
        super().__init__()
        self.error = error
        self.calls: list[tuple] = []

    async def run(self, spec, target_type, target, context, params) -> ToolResult:
        self.calls.append((spec.name, target_type, target, context, params))
        if self.error:
            raise self.error
        return self.result(spec, target_type, outputs={spec.produces[0]: {"real": True}},
                           observations=[], findings=[f"real {spec.key}"])  # fmt: skip


# --- catalog & input validation ------------------------------------------------------------


def test_catalog_is_consistent():
    for op in OPERATIONS:
        assert op.accepts and op.produces
        for name, allowed in op.params.items():
            assert allowed, f"{op.key}.{name} has no allowed values"
        if op.destructive:
            assert not op.enrichment and A.CONTAINMENT_STATUS in op.produces
        for artifact in op.accepts + op.requires:
            assert artifact in A.ENTITY_ARTIFACTS + (A.FILE_PATH, A.TECHNIQUE_ID)
    assert enrichment_operations(EvidenceCategory.MEMORY) == []  # no tool collects memory: manual work


@pytest.mark.parametrize(
    "artifact, value, expected",
    [
        (A.IP_ADDRESS, "203.0.113[.]45", "203.0.113.45"),
        (A.DOMAIN, "CDN-Update[.]Example.", "cdn-update.example"),
        (A.FILE_HASH, HASH.upper(), HASH),
        (A.HOST_ID, "WS-FIN-042", "WS-FIN-042"),
        (A.USER_ID, "CORP\\jdoe", "CORP\\jdoe"),
        (A.TECHNIQUE_ID, "T1053.005", "T1053.005"),
    ],
)
def test_validate_input_canonicalises(artifact, value, expected):
    assert A.validate_input(artifact, value) == expected


@pytest.mark.parametrize(
    "artifact, value",
    [
        (A.IP_ADDRESS, "999.1.1.1"),
        (A.FILE_HASH, "xyz"),
        (A.HOST_ID, "host; rm -rf /"),
        (A.HOST_ID, 'ws-1" OR 1=1'),
        (A.USER_ID, ""),
        (A.TECHNIQUE_ID, "T10"),
        (A.FILE_PATH, "C:\\a\x00b"),
        (A.REPUTATION_REPORT, "x"),
    ],
)
def test_validate_input_rejects_without_echoing(artifact, value):
    with pytest.raises(ToolInputError) as info:
        A.validate_input(artifact, value)
    if value:
        assert value not in info.value.message


async def test_invalid_requests_never_reach_the_adapter():
    tool = SpyTool("virustotal")
    bad = [
        req("no_such_op", A.FILE_HASH, HASH),
        req("file_reputation", A.IP_ADDRESS, "203.0.113.45"),  # wrong target type
        req("file_reputation", A.FILE_HASH, "not-a-hash"),
    ]
    for request in bad:
        with pytest.raises(ToolInputError):
            await tool.invoke(request)
    wazuh = SpyTool("wazuh")
    with pytest.raises(ToolInputError):  # required context missing
        await wazuh.invoke(req("block_ip", A.IP_ADDRESS, "203.0.113.45"))
    with pytest.raises(ToolInputError):  # parameter value not allowed
        await wazuh.invoke(req("host_alerts", A.HOST_ID, "WS-1", params={"lookback_hours": 99999}))
    with pytest.raises(ToolInputError):  # unknown parameter
        await wazuh.invoke(req("host_alerts", A.HOST_ID, "WS-1", params={"query": "*"}))
    assert tool.calls == [] and wazuh.calls == []


# --- simulator ----------------------------------------------------------------------------


async def test_simulated_results_are_deterministic_and_labelled():
    tool = SimulatedTool("virustotal", {EvidenceCategory.INDICATORS_OF_COMPROMISE: 1.0})
    first = await tool.invoke(req("file_reputation", A.FILE_HASH, HASH))
    second = await tool.invoke(req("file_reputation", A.FILE_HASH, HASH.upper()))
    assert first.model_dump(exclude={"duration_ms"}) == second.model_dump(exclude={"duration_ms"})
    assert first.synthetic is True and first.status == ToolStatus.SUCCESS
    assert all(f.startswith("SYNTHETIC") for f in first.findings)
    assert all(o.description.startswith("SYNTHETIC") for o in first.observations)
    assert first.outputs[A.REPUTATION_REPORT]["verdict"] in {"clean", "suspicious", "malicious"}


async def test_simulator_follows_the_availability_profile():
    never = SimulatedTool("wazuh", {EvidenceCategory.SECURITY_ALERT: 0.0})
    always = SimulatedTool("wazuh", {EvidenceCategory.SECURITY_ALERT: 1.0})
    for host in ("WS-1", "WS-2", "WS-3"):
        empty = await never.invoke(req("host_alerts", A.HOST_ID, host))
        assert empty.status == ToolStatus.NO_DATA and "not proof of absence" in empty.message
        assert (await always.invoke(req("host_alerts", A.HOST_ID, host))).status == ToolStatus.SUCCESS


async def test_simulated_destructive_operation_takes_no_action():
    result = await SimulatedTool("wazuh").invoke(req("isolate_host", A.HOST_ID, "WS-1"))
    assert result.synthetic and result.status == ToolStatus.SUCCESS
    assert result.outputs[A.CONTAINMENT_STATUS]["simulated"] is True
    assert "no real action was taken" in result.findings[0]


def test_tool_result_rejects_unlabelled_synthetic_findings_and_failed_observations():
    with pytest.raises(ValueError):
        ToolResult(tool="x", operation="y", target_type="host_id", status=ToolStatus.SUCCESS, synthetic=True,
                   findings=["looks real"])  # fmt: skip
    with pytest.raises(ValueError):
        ToolResult(tool="x", operation="y", target_type="host_id", status=ToolStatus.NO_DATA, synthetic=False,
                   observations=[{"category": "dns", "description": "d"}])  # fmt: skip


# --- registry & gateway policy --------------------------------------------------------------


async def test_simulate_mode_never_calls_real_tools():
    real = SpyTool("virustotal")
    gateway = ToolGateway(ToolRegistry(ExecutionMode.SIMULATE, {"virustotal": real}), ToolFailurePolicy.ERROR)
    result = await gateway.invoke("virustotal", req("file_reputation", A.FILE_HASH, HASH))
    assert result.synthetic is True and real.calls == []


async def test_live_mode_uses_the_real_adapter():
    real = SpyTool("virustotal")
    gateway = ToolGateway(ToolRegistry(ExecutionMode.LIVE, {"virustotal": real}), ToolFailurePolicy.ERROR)
    result = await gateway.invoke("virustotal", req("file_reputation", A.FILE_HASH, HASH))
    assert result.synthetic is False and len(real.calls) == 1


@pytest.mark.parametrize("policy, status", [(ToolFailurePolicy.MANUAL_TASK, ToolStatus.MANUAL_TASK),
                                            (ToolFailurePolicy.ERROR, ToolStatus.ERROR)])  # fmt: skip
@pytest.mark.parametrize("error", [ToolTimeoutError("t"), ToolUnavailableError("u"), ToolAuthError("a"), ToolError("e")])
async def test_real_tool_failure_is_never_replaced_by_simulation(policy, status, error):
    registry = ToolRegistry(ExecutionMode.LIVE, {"virustotal": SpyTool("virustotal", error=error)})
    result = await ToolGateway(registry, policy).invoke("virustotal", req("file_reputation", A.FILE_HASH, HASH))
    assert result.status == status
    assert result.synthetic is False
    assert result.observations == [] and result.outputs == {}
    assert result.error_category == error.category
    assert not any("SYNTHETIC" in f for f in result.findings)


async def test_unconfigured_tool_in_live_mode_becomes_manual_task():
    gateway = ToolGateway(ToolRegistry(ExecutionMode.LIVE, {}), ToolFailurePolicy.MANUAL_TASK)
    result = await gateway.invoke("wazuh", req("host_alerts", A.HOST_ID, "WS-1"))
    assert result.status == ToolStatus.MANUAL_TASK and result.error_category == "tool_unavailable"
    assert result.findings[0].startswith("Manual investigation task")


async def test_gateway_propagates_input_errors_and_unknown_tools():
    gateway = ToolGateway(ToolRegistry(ExecutionMode.SIMULATE), ToolFailurePolicy.MANUAL_TASK)
    with pytest.raises(ToolInputError):
        await gateway.invoke("virustotal", req("file_reputation", A.FILE_HASH, "bad"))
    with pytest.raises(ToolInputError):
        await gateway.invoke("nmap", req("scan", A.HOST_ID, "WS-1"))


def test_registry_reports_effective_mode():
    live = ToolRegistry(ExecutionMode.LIVE, {"virustotal": SpyTool("virustotal")}).info()
    effective = {t.name: t.effective for t in live}
    assert effective["virustotal"] == "real" and effective["wazuh"] == "unavailable"
    assert {t.effective for t in ToolRegistry(ExecutionMode.SIMULATE).info()} == {"simulated"}


def test_build_real_tools_only_from_complete_settings(settings):
    assert build_real_tools(settings) == {}
    configured = with_settings(settings, virustotal_api_key=VT_KEY, wazuh_api_url="https://wazuh.test:55000",
                               wazuh_username="cade", elastic_url="https://es.test:9200")  # fmt: skip
    tools = build_real_tools(configured)
    assert set(tools) == {"virustotal"}  # wazuh lacks a password, elastic lacks an API key
    assert VT_KEY not in repr(configured) and VT_KEY not in str(configured.model_dump())


# --- rate limiter -------------------------------------------------------------------------------


async def test_rate_limiter_waits_then_refuses():
    now = [0.0]
    slept: list[float] = []

    async def sleep(seconds: float) -> None:
        slept.append(seconds)
        now[0] += seconds

    limiter = RateLimiter(4, max_wait_seconds=15, clock=lambda: now[0], sleep=sleep)
    for _ in range(4):
        await limiter.acquire("vt")
    await limiter.acquire("vt")  # 5th request in the same minute: waits 15 s
    assert slept == [pytest.approx(15.0)]
    strict = RateLimiter(4, max_wait_seconds=1, clock=lambda: now[0], sleep=sleep)
    for _ in range(4):
        await strict.acquire("vt")
    with pytest.raises(ToolRateLimitedError):
        await strict.acquire("vt")


# --- VirusTotal -------------------------------------------------------------------------------


def _vt(handler) -> VirusTotalTool:
    limiter = RateLimiter(1000)
    return VirusTotalTool(api_key=VT_KEY, base_url="https://vt.test/api/v3", limiter=limiter,
                          transport=httpx.MockTransport(handler))  # fmt: skip


async def test_virustotal_parses_verdicts():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET" and request.url.path == f"/api/v3/files/{HASH}"
        assert request.headers["x-apikey"] == VT_KEY
        stats = {"malicious": 5, "suspicious": 1, "harmless": 10, "undetected": 54}
        return httpx.Response(200, json={"data": {"attributes": {"last_analysis_stats": stats, "reputation": -20}}})

    result = await _vt(handler).invoke(req("file_reputation", A.FILE_HASH, HASH))
    assert result.status == ToolStatus.SUCCESS and result.synthetic is False
    report = result.outputs[A.REPUTATION_REPORT]
    assert report["verdict"] == "malicious" and report["engines"] == 70
    assert result.observations[0].category == EvidenceCategory.INDICATORS_OF_COMPROMISE


async def test_virustotal_unknown_indicator_is_no_data_not_clean():
    result = await _vt(lambda r: httpx.Response(404, json={"error": {}})).invoke(req("ip_reputation", A.IP_ADDRESS, "203.0.113.45"))
    assert result.status == ToolStatus.NO_DATA and result.observations == []


@pytest.mark.parametrize(
    "response, error",
    [
        (httpx.Response(401), ToolAuthError),
        (httpx.Response(429), ToolRateLimitedError),
        (httpx.Response(503), ToolError),
        (httpx.Response(200, content=b"not json"), ToolError),
        (httpx.Response(200, content=b"x" * 2_100_000), ToolError),
    ],
)
async def test_virustotal_failures_raise_typed_errors_without_secrets(response, error):
    with pytest.raises(error) as info:
        await _vt(lambda r: response).invoke(req("domain_reputation", A.DOMAIN, "cdn-update.example"))
    assert VT_KEY not in info.value.message


async def test_http_timeouts_and_connection_errors_are_typed():
    def timeout(request):
        raise httpx.ReadTimeout("slow", request=request)

    def refused(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ToolTimeoutError):
        await _vt(timeout).invoke(req("file_reputation", A.FILE_HASH, HASH))
    with pytest.raises(ToolUnavailableError):
        await _vt(refused).invoke(req("file_reputation", A.FILE_HASH, HASH))


# --- Wazuh ----------------------------------------------------------------------------------


def _wazuh_handler(log: list, *, agents=({"id": "007", "name": "WS-1"},), expire_first=False):
    state = {"tokens": 0, "expired": expire_first}

    def handler(request: httpx.Request) -> httpx.Response:
        log.append((request.method, request.url.path, dict(request.url.params), request.content))
        if request.url.path == "/security/user/authenticate":
            assert request.headers["Authorization"].startswith("Basic ")
            state["tokens"] += 1
            return httpx.Response(200, json={"data": {"token": f"jwt-{state['tokens']}"}})
        if state["expired"]:
            state["expired"] = False
            return httpx.Response(401)
        assert request.headers["Authorization"].startswith("Bearer jwt-")
        if request.url.path == "/agents":
            return httpx.Response(200, json={"data": {"affected_items": list(agents)}})
        if request.url.path == "/syscollector/007/processes":
            return httpx.Response(200, json={"data": {"affected_items": [{"pid": 1, "name": "telemetry.exe"}]}})
        if request.url.path == "/active-response":
            return httpx.Response(200, json={"data": {"total_affected_items": 1}})
        return httpx.Response(404)

    return handler


def _wazuh(handler, **kwargs) -> WazuhTool:
    return WazuhTool(api_url="https://wazuh.test:55000", username="cade", password="pw-secret",
                     transport=httpx.MockTransport(handler), **kwargs)  # fmt: skip


async def test_wazuh_authenticates_reuses_token_and_lists_processes():
    log: list = []
    tool = _wazuh(_wazuh_handler(log))
    result = await tool.invoke(req("host_processes", A.HOST_ID, "WS-1"))
    await tool.invoke(req("host_processes", A.HOST_ID, "WS-1"))
    assert result.status == ToolStatus.SUCCESS and result.outputs[A.PROCESS_LIST][0]["name"] == "telemetry.exe"
    assert sum(1 for entry in log if entry[1] == "/security/user/authenticate") == 1


async def test_wazuh_reauthenticates_once_on_expired_token():
    log: list = []
    result = await _wazuh(_wazuh_handler(log, expire_first=True)).invoke(req("host_processes", A.HOST_ID, "WS-1"))
    assert result.status == ToolStatus.SUCCESS
    assert sum(1 for entry in log if entry[1] == "/security/user/authenticate") == 2


async def test_wazuh_unknown_agent_is_no_data():
    result = await _wazuh(_wazuh_handler([], agents=())).invoke(req("host_processes", A.HOST_ID, "WS-404"))
    assert result.status == ToolStatus.NO_DATA


async def test_wazuh_isolation_requires_a_configured_command():
    with pytest.raises(ToolUnavailableError):
        await _wazuh(_wazuh_handler([])).invoke(req("isolate_host", A.HOST_ID, "WS-1"))


async def test_wazuh_block_ip_targets_one_agent():
    log: list = []
    result = await _wazuh(_wazuh_handler(log)).invoke(
        req("block_ip", A.IP_ADDRESS, "203.0.113.45", context={A.HOST_ID: "WS-1"}))
    assert result.synthetic is False and result.outputs[A.CONTAINMENT_STATUS]["state"] == "requested"
    method, path, params, body = log[-1]
    assert (method, path, params) == ("PUT", "/active-response", {"agents_list": "007"})
    assert json.loads(body)["alert"]["data"]["srcip"] == "203.0.113.45"


async def test_wazuh_alerts_need_the_indexer():
    with pytest.raises(ToolUnavailableError):
        await _wazuh(_wazuh_handler([])).invoke(req("host_alerts", A.HOST_ID, "WS-1"))


# --- Elastic ----------------------------------------------------------------------------------


async def test_elastic_search_uses_structured_term_filters():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["auth"] = request.headers["Authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"hits": {"hits": [{"_source": {"process.name": "telemetry.exe"}}]}})

    tool = ElasticTool(url="https://es.test:9200", api_key="es-key", transport=httpx.MockTransport(handler))
    result = await tool.invoke(req("process_events", A.HOST_ID, "WS-HR-019", params={"lookback_hours": 72}))
    assert result.status == ToolStatus.SUCCESS
    assert seen["path"] == "/logs-*/_search" and seen["auth"] == "ApiKey es-key"
    body = json.dumps(seen["body"])
    assert '"term": {"host.name": "WS-HR-019"}' in body and "now-72h" in body
    assert "query_string" not in body


async def test_elastic_rows_are_flattened_for_execution_handlers():
    # Real Elasticsearch returns nested ECS objects; the execution handlers read dotted names. Before
    # flattening, a live run analysed 50 real process events (one an encoded PowerShell) as "suspicious: none".
    from app.execution.internal import analyze_auth_patterns, analyze_network_flows, analyze_process_tree

    sources = {
        "process": {"@timestamp": "2026-10-24T09:00:00Z", "user": {"name": "Administrator"},
                    "process": {"name": "powershell.exe", "command_line": "powershell.exe -Exec bypass -enc VwBy",
                                "parent": {"name": "cmd.exe"}}},
        "network": {"source": {"ip": "10.0.1.15"}, "destination": {"ip": "203.0.113.45", "port": 443}},
        "authentication": {"user": {"name": "jdoe"}, "event": {"outcome": "failure"}},
    }  # fmt: skip

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        category = next(f["term"]["event.category"] for f in body["query"]["bool"]["filter"] if "term" in f)
        return httpx.Response(200, json={"hits": {"hits": [{"_source": sources[category]}]}})

    tool = ElasticTool(url="https://es.test:9200", api_key="k", transport=httpx.MockTransport(handler))
    processes = (await tool.invoke(req("process_events", A.HOST_ID, "win-dc-397"))).outputs[A.PROCESS_LIST]
    assert processes[0]["process.name"] == "powershell.exe" and processes[0]["process.parent.name"] == "cmd.exe"
    flows = (await tool.invoke(req("network_events", A.HOST_ID, "win-dc-397"))).outputs[A.NETWORK_FLOWS]
    auth = (await tool.invoke(req("auth_events", A.USER_ID, "jdoe"))).outputs[A.AUTH_EVENTS]

    assert "suspicious: powershell.exe" in analyze_process_tree(None, {A.PROCESS_LIST: processes}).findings[0]
    assert "1 distinct destinations" in analyze_network_flows(None, {A.NETWORK_FLOWS: flows}).findings[0]
    assert "1 failures" in analyze_auth_patterns(None, {A.AUTH_EVENTS: auth}).findings[0]


async def test_elastic_no_hits_and_missing_kibana():
    tool = ElasticTool(url="https://es.test:9200", api_key="k",
                       transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"hits": {"hits": []}})))  # fmt: skip
    assert (await tool.invoke(req("dns_events", A.DOMAIN, "cdn-update.example"))).status == ToolStatus.NO_DATA
    with pytest.raises(ToolUnavailableError):
        await tool.invoke(req("isolate_host", A.HOST_ID, "WS-1"))


async def test_elastic_isolation_through_kibana():
    def kibana(request: httpx.Request) -> httpx.Response:
        assert request.headers["kbn-xsrf"] == "true"
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"metadata": {"agent": {"id": "agent-9"}}}]})
        assert json.loads(request.content)["endpoint_ids"] == ["agent-9"]
        return httpx.Response(200, json={"data": {"id": "action-1"}})

    tool = ElasticTool(url="https://es.test", api_key="k", kibana_url="https://kibana.test",
                       transport=httpx.MockTransport(lambda r: httpx.Response(500)),
                       kibana_transport=httpx.MockTransport(kibana))  # fmt: skip
    result = await tool.invoke(req("isolate_host", A.HOST_ID, "WS-1"))
    assert result.outputs[A.CONTAINMENT_STATUS]["action_id"] == "action-1"


# --- osquery / YARA / Sigma ----------------------------------------------------------------------


def _runner(calls: list, result: CommandResult):
    def run(args: list[str], timeout: float) -> CommandResult:
        calls.append(args)
        return result

    return run


async def test_osquery_only_queries_the_local_host_with_fixed_sql():
    calls: list = []
    rows = [{"name": "DailyTelemetryCheck", "action": "telemetry.exe"}]
    tool = OsqueryTool(runner=_runner(calls, CommandResult(0, json.dumps(rows), "")), local_hostnames={"cade-box"})
    with pytest.raises(ToolUnavailableError):
        await tool.invoke(req("scheduled_tasks", A.HOST_ID, "WS-HR-019"))
    assert calls == []
    result = await tool.invoke(req("scheduled_tasks", A.HOST_ID, "CADE-BOX"))
    assert result.outputs[A.SCHEDULED_TASKS] == rows
    (args,) = calls
    assert args[:2] == ["osqueryi", "--json"] and "CADE-BOX" not in args[2]


async def test_osquery_failure_is_an_error():
    tool = OsqueryTool(runner=_runner([], CommandResult(1, "", "boom")), local_hostnames={"cade-box"})
    with pytest.raises(ToolError):
        await tool.invoke(req("processes", A.HOST_ID, "cade-box"))


async def test_yara_scans_only_inside_scan_roots(tmp_path: Path):
    root = tmp_path / "quarantine"
    root.mkdir()
    sample = root / "telemetry.exe"
    sample.write_bytes(b"SYNTHETIC sample, not an executable")
    outside = tmp_path / "secret.txt"
    outside.write_text("not for scanning")
    calls: list = []
    tool = YaraTool(rules_path=tmp_path / "rules.yar", scan_roots=[root],
                    runner=_runner(calls, CommandResult(0, f"Suspicious_Telemetry {sample}\n", "")))  # fmt: skip
    for path in (str(outside), str(root / ".." / "secret.txt"), str(tmp_path / "missing.exe")):
        with pytest.raises(ToolInputError):
            await tool.invoke(req("scan_file", A.FILE_PATH, path))
    assert calls == []
    result = await tool.invoke(req("scan_file", A.FILE_PATH, str(sample)))
    assert result.outputs[A.YARA_MATCHES] == ["Suspicious_Telemetry"]
    assert calls[0][:2] == ["yara", "--"]


def _msearch_handler(calls: list, answer=lambda body: {"hits": {"hits": [{"_source": {"host.name": "WS-HR-019"}}]}}):
    """A fake Elasticsearch ``_msearch`` endpoint: one response per (header, body) pair."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/_msearch"
        lines = [json.loads(line) for line in request.content.decode().splitlines() if line]
        bodies = lines[1::2]
        calls.append(bodies)
        return httpx.Response(200, json={"responses": [answer(body) for body in bodies]})

    return handler


def _sigma_query(body: dict) -> str:
    return next(f["query_string"]["query"] for f in body["query"]["bool"]["filter"] if "query_string" in f)


async def test_sigma_runs_rules_tagged_with_the_technique(tmp_path: Path):
    (tmp_path / "schtask.yml").write_text("tags: [attack.t1053.005]")
    (tmp_path / "other.yml").write_text("tags: [attack.t1059.001]")
    calls: list = []
    elastic = ElasticTool(url="https://es.test", api_key="k", transport=httpx.MockTransport(_msearch_handler(calls)))
    tool = SigmaTool(rules_dir=tmp_path, elastic=elastic, converter=lambda text: ["process.name:schtasks.exe"],
                     tag_reader=lambda text: ["attack.t1053.005"] if "t1053.005" in text else ["attack.t1059.001"])  # fmt: skip
    result = await tool.invoke(req("hunt_technique", A.TECHNIQUE_ID, "T1053.005"))
    assert result.status == ToolStatus.SUCCESS and result.outputs[A.SIGMA_HITS][0]["rule"] == "schtask"
    assert len(calls) == 1 and len(calls[0]) == 1
    no_rules = await tool.invoke(req("hunt_technique", A.TECHNIQUE_ID, "T1486"))
    assert no_rules.status == ToolStatus.NO_DATA


async def test_sigma_runs_every_tagged_rule_in_batched_msearch_calls(tmp_path: Path):
    # SigmaHQ has 181 Windows rules for T1059.001; an early cap of 25 (in path order) hid the relevant ones.
    for i in range(120):
        (tmp_path / f"rule_{i:03}.yml").write_text(f"id: {i}\ntags: [attack.t1059.001]")
    calls: list = []
    last = lambda body: ({"hits": {"hits": [{"_source": {"host.name": "WS-1"}}]}}  # noqa: E731
                         if _sigma_query(body) == "Image:rule_119" else {"hits": {"hits": []}})  # fmt: skip
    elastic = ElasticTool(url="https://es.test", api_key="k", transport=httpx.MockTransport(_msearch_handler(calls, last)))
    tool = SigmaTool(rules_dir=tmp_path, elastic=elastic, tag_reader=lambda text: ["attack.t1059.001"],
                     converter=lambda text: [f"Image:rule_{int(text.split()[1]):03}"])  # fmt: skip
    result = await tool.invoke(req("hunt_technique", A.TECHNIQUE_ID, "T1059.001"))
    assert [len(bodies) for bodies in calls] == [50, 50, 20]
    assert result.outputs[A.SIGMA_HITS] == [{"rule": "rule_119", "host.name": "WS-1"}]
    assert result.findings == ["Sigma: 1 hit(s) from 1 of 120 rule(s) for T1059.001"]


async def test_sigma_reports_unconverted_rules_and_rejected_queries(tmp_path: Path):
    for name in ("good", "unsupported", "bad_query"):
        (tmp_path / f"{name}.yml").write_text(name)
    reject = lambda body: ({"error": {"type": "query_shard_exception"}, "status": 400}  # noqa: E731
                           if _sigma_query(body) == "bad" else {"hits": {"hits": []}})  # fmt: skip
    elastic = ElasticTool(url="https://es.test", api_key="k", transport=httpx.MockTransport(_msearch_handler([], reject)))
    tool = SigmaTool(rules_dir=tmp_path, elastic=elastic, tag_reader=lambda text: ["attack.t1053.005"],
                     converter=lambda text: {"good": ["ok"], "bad_query": ["bad"]}.get(text, []))  # fmt: skip
    result = await tool.invoke(req("hunt_technique", A.TECHNIQUE_ID, "T1053.005"))
    assert result.status == ToolStatus.SUCCESS and result.outputs[A.SIGMA_HITS] == []
    assert result.findings == ["Sigma: 3 rule(s) for T1053.005 ran with no hits; 1 could not be converted; "
                               "Elastic rejected 1 query"]  # fmt: skip


def test_sigma_default_converter_lowercases_values_but_not_field_names():
    pytest.importorskip("sigma.processing.transformations")
    from app.tools.sigma import _default_converter

    rule = """
title: Encoded PowerShell
logsource: {product: windows, category: process_creation}
detection:
    selection:
        Image|endswith: '\\PowerShell.EXE'
        CommandLine|contains: ' -ENC '
    condition: selection
"""
    [query] = _default_converter(rule)
    assert "Image:" in query and "CommandLine:" in query
    assert "powershell.exe" in query and "\\-enc" in query and "PowerShell" not in query


async def test_sigma_without_elastic_is_unavailable(tmp_path: Path):
    with pytest.raises(ToolUnavailableError):
        await SigmaTool(rules_dir=tmp_path, elastic=None).invoke(req("hunt_technique", A.TECHNIQUE_ID, "T1053.005"))


def test_availability_profile_file_is_valid(settings):
    profile = AvailabilityProfile.load(settings.resolve_path(settings.evidence_availability_path))
    assert set(profile.availability) == set(EvidenceCategory)
    assert isinstance(CATALOG["virustotal.file_reputation"], OperationSpec)
