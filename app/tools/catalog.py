"""The operations each supported security tool offers.

Real adapters implement exactly these operations; the simulator answers them with synthetic
data. Phase 3's action vocabulary binds playbook actions to ``tool.operation`` keys from here.
"""

from __future__ import annotations

from app.schemas.enums import EvidenceCategory as C
from app.schemas.tools import OperationSpec
from app.tools import artifacts as A

LOOKBACK = {"lookback_hours": (24, 1, 6, 72, 168)}
LIMIT = {"limit": (50, 10, 100)}

TOOL_DESCRIPTIONS = {
    "virustotal": "VirusTotal v3 API: reputation of file hashes, IP addresses and domains (free tier supported)",
    "wazuh": "Wazuh server API and indexer: agent alerts, process inventory, file integrity, active response",
    "elastic": "Elasticsearch / Elastic Security: ECS event search; Kibana endpoint isolation",
    "osquery": "osqueryi on the CADE host itself: processes, scheduled tasks, ports, startup items, users",
    "yara": "YARA command-line scanner against a configured rules file, restricted to allowed directories",
    "sigma": "Sigma rules tagged with an ATT&CK technique, converted with pySigma and run on Elasticsearch",
}


def _op(tool, name, description, accepts, produces, categories=(), destructive=False, params=None,
        requires=()) -> OperationSpec:  # fmt: skip
    return OperationSpec(
        tool=tool, name=name, description=description, accepts=tuple(accepts), produces=tuple(produces),
        categories=tuple(categories), destructive=destructive, params=params or {}, requires=tuple(requires),
    )  # fmt: skip


OPERATIONS: list[OperationSpec] = [
    # --- VirusTotal
    _op("virustotal", "file_reputation", "Engine verdicts for a file hash", [A.FILE_HASH], [A.REPUTATION_REPORT],
        [C.INDICATORS_OF_COMPROMISE]),
    _op("virustotal", "ip_reputation", "Engine verdicts for an IP address", [A.IP_ADDRESS], [A.REPUTATION_REPORT],
        [C.INDICATORS_OF_COMPROMISE]),
    _op("virustotal", "domain_reputation", "Engine verdicts for a domain", [A.DOMAIN], [A.REPUTATION_REPORT],
        [C.INDICATORS_OF_COMPROMISE, C.DNS]),
    # --- Wazuh
    _op("wazuh", "host_alerts", "Recent Wazuh alerts for an agent", [A.HOST_ID], [A.ALERT_LIST],
        [C.SECURITY_ALERT], params={**LOOKBACK, **LIMIT}),
    _op("wazuh", "host_processes", "Process inventory (syscollector) of an agent", [A.HOST_ID], [A.PROCESS_LIST],
        [C.PROCESS_TELEMETRY], params=LIMIT),
    _op("wazuh", "file_integrity_events", "File integrity monitoring (syscheck) events of an agent", [A.HOST_ID],
        [A.FILE_EVENTS], [C.FILE_ACTIVITY], params=LIMIT),
    _op("wazuh", "isolate_host", "Run the configured isolation active-response on an agent", [A.HOST_ID],
        [A.CONTAINMENT_STATUS], destructive=True),
    _op("wazuh", "block_ip", "Run the firewall-drop active-response for an IP on one agent", [A.IP_ADDRESS],
        [A.CONTAINMENT_STATUS], destructive=True, requires=[A.HOST_ID]),
    # --- Elastic
    _op("elastic", "process_events", "Process start events (ECS event.category=process)", [A.HOST_ID],
        [A.PROCESS_LIST], [C.PROCESS_TELEMETRY, C.COMMAND_LINE], params={**LOOKBACK, **LIMIT}),
    _op("elastic", "network_events", "Network connection events", [A.HOST_ID, A.IP_ADDRESS], [A.NETWORK_FLOWS],
        [C.NETWORK_TRAFFIC], params={**LOOKBACK, **LIMIT}),
    _op("elastic", "dns_events", "DNS query events", [A.HOST_ID, A.DOMAIN], [A.DNS_RECORDS], [C.DNS],
        params={**LOOKBACK, **LIMIT}),
    _op("elastic", "auth_events", "Authentication events", [A.USER_ID, A.HOST_ID], [A.AUTH_EVENTS],
        [C.AUTHENTICATION, C.USER_ACCOUNT], params={**LOOKBACK, **LIMIT}),
    _op("elastic", "web_events", "Web server access events", [A.HOST_ID, A.IP_ADDRESS], [A.WEB_REQUESTS],
        [C.WEB_SERVER_LOGS], params={**LOOKBACK, **LIMIT}),
    _op("elastic", "email_events", "Email gateway events for a recipient", [A.USER_ID], [A.EMAIL_MESSAGES],
        [C.EMAIL], params={**LOOKBACK, **LIMIT}),
    _op("elastic", "cloud_audit_events", "Cloud audit events for a user", [A.USER_ID], [A.AUTH_EVENTS],
        [C.CLOUD_AUDIT], params={**LOOKBACK, **LIMIT}),
    # Windows event-log searches by channel and event code (app/data/log_source_categories.json)
    _op("elastic", "registry_events", "Registry activity (Sysmon 12-14)", [A.HOST_ID], [A.REGISTRY_EVENTS],
        [C.REGISTRY], params={**LOOKBACK, **LIMIT}),
    _op("elastic", "script_block_events", "PowerShell script-block logging (PowerShell 4104)", [A.HOST_ID],
        [A.SCRIPT_BLOCKS], [C.SCRIPT_EXECUTION], params={**LOOKBACK, **LIMIT}),
    _op("elastic", "service_task_events", "Service installs and scheduled-task creation (Security 4697, 4698)",
        [A.HOST_ID], [A.SERVICE_EVENTS], [C.SCHEDULED_TASKS_SERVICES], params={**LOOKBACK, **LIMIT}),
    _op("elastic", "isolate_host", "Isolate a host with Elastic Defend (Kibana response action)", [A.HOST_ID],
        [A.CONTAINMENT_STATUS], destructive=True),
    # --- osquery (local host only)
    _op("osquery", "processes", "Running processes with command lines", [A.HOST_ID], [A.PROCESS_LIST],
        [C.PROCESS_TELEMETRY, C.COMMAND_LINE]),
    _op("osquery", "scheduled_tasks", "Scheduled tasks", [A.HOST_ID], [A.SCHEDULED_TASKS],
        [C.SCHEDULED_TASKS_SERVICES]),
    _op("osquery", "listening_ports", "Listening ports and owning processes", [A.HOST_ID], [A.LISTENING_PORTS],
        [C.NETWORK_TRAFFIC]),
    _op("osquery", "startup_items", "Autostart entries (run keys, services, startup folders)", [A.HOST_ID],
        [A.STARTUP_ITEMS], [C.REGISTRY, C.SCHEDULED_TASKS_SERVICES]),
    _op("osquery", "local_users", "Local user accounts", [A.HOST_ID], [A.USER_LIST], [C.USER_ACCOUNT]),
    # --- YARA
    _op("yara", "scan_file", "Scan one file with the configured YARA rules", [A.FILE_PATH], [A.YARA_MATCHES],
        [C.FILE_ACTIVITY, C.INDICATORS_OF_COMPROMISE]),
    # --- Sigma
    _op("sigma", "hunt_technique", "Run Sigma rules tagged with an ATT&CK technique", [A.TECHNIQUE_ID],
        [A.SIGMA_HITS], [C.SECURITY_ALERT], params=LOOKBACK),
]  # fmt: skip

CATALOG: dict[str, OperationSpec] = {op.key: op for op in OPERATIONS}
TOOLS: tuple[str, ...] = tuple(TOOL_DESCRIPTIONS)


def operations_for(tool: str) -> dict[str, OperationSpec]:
    return {op.name: op for op in OPERATIONS if op.tool == tool}


def enrichment_operations(category: C) -> list[OperationSpec]:
    """Read-only operations that can collect evidence of ``category``."""
    return [op for op in OPERATIONS if op.enrichment and category in op.categories]
