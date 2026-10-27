"""Typed artifacts that flow between tools (and, from Phase 3, between playbook DAG nodes).

An artifact type names *what* a value is ("host_id", "file_hash", "reputation_report"). Tool
operations declare the artifact types they consume and produce; a playbook edge u -> v is
valid only when an output type of u is an input type of v.

Every value that originates outside CADE (incident entities, analyst input, a previous tool's
output) is validated by :func:`validate_input` before it is sent to any tool.
"""

from __future__ import annotations

import re
from typing import Any

from app.core.errors import ToolInputError
from app.schemas.incident import Entities
from app.schemas.indicators import canonical_domain, canonical_hash, canonical_ip

# --- entity artifacts (can come straight from an analysed incident) ----------------------
INCIDENT_CONTEXT = "incident_context"
HOST_ID = "host_id"
USER_ID = "user_id"
IP_ADDRESS = "ip_address"
DOMAIN = "domain"
FILE_HASH = "file_hash"
FILE_PATH = "file_path"
PROCESS_NAME = "process_name"
TECHNIQUE_ID = "technique_id"

# --- tool outputs --------------------------------------------------------------------------
REPUTATION_REPORT = "reputation_report"
ALERT_LIST = "alert_list"
PROCESS_LIST = "process_list"
FILE_EVENTS = "file_events"
NETWORK_FLOWS = "network_flows"
DNS_RECORDS = "dns_records"
AUTH_EVENTS = "auth_events"
WEB_REQUESTS = "web_requests"
EMAIL_MESSAGES = "email_messages"
SCHEDULED_TASKS = "scheduled_tasks"
REGISTRY_EVENTS = "registry_events"
SCRIPT_BLOCKS = "script_blocks"
SERVICE_EVENTS = "service_events"
LISTENING_PORTS = "listening_ports"
STARTUP_ITEMS = "startup_items"
USER_LIST = "user_list"
YARA_MATCHES = "yara_matches"
SIGMA_HITS = "sigma_hits"
CONTAINMENT_STATUS = "containment_status"
FINDINGS = "findings"

# --- produced by CADE's own (internal) or analyst-performed playbook actions -----------------
SCOPE_ASSESSMENT = "scope_assessment"
VERIFICATION_RESULT = "verification_result"
INCIDENT_REPORT = "incident_report"

ENTITY_ARTIFACTS = (HOST_ID, USER_ID, IP_ADDRESS, DOMAIN, FILE_HASH, PROCESS_NAME)

_HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,252}$")
_USER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@\\ -]{0,127}$")
_PROCESS_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]{0,254}$")
_TECHNIQUE_RE = re.compile(r"^T\d{4}(\.\d{3})?$")
_MAX_PATH_CHARS = 1024


def validate_input(artifact_type: str, value: Any) -> str:
    """Canonicalise one indicator, or raise :class:`ToolInputError` without echoing the value."""
    if not isinstance(value, str) or not value.strip():
        raise ToolInputError(f"{artifact_type} must be a non-empty string")
    value = value.strip()
    try:
        if artifact_type == IP_ADDRESS:
            return canonical_ip(value)
        if artifact_type == DOMAIN:
            return canonical_domain(value)
        if artifact_type == FILE_HASH:
            return canonical_hash(value)[1]
    except ValueError:
        raise ToolInputError(f"invalid {artifact_type}") from None
    patterns = {HOST_ID: _HOST_RE, USER_ID: _USER_RE, PROCESS_NAME: _PROCESS_RE, TECHNIQUE_ID: _TECHNIQUE_RE}
    if artifact_type in patterns:
        if not patterns[artifact_type].match(value):
            raise ToolInputError(f"invalid {artifact_type}")
        return value
    if artifact_type == FILE_PATH:
        if len(value) > _MAX_PATH_CHARS or any(ch in value for ch in "\x00\r\n"):
            raise ToolInputError("invalid file_path")
        return value
    raise ToolInputError(f"'{artifact_type}' is not an input artifact type")


def entity_artifacts(entities: Entities) -> dict[str, list[str]]:
    """The artifact values an analysed incident provides directly (already grounded in the input)."""
    return {
        HOST_ID: list(entities.hosts),
        USER_ID: list(entities.users),
        IP_ADDRESS: list(entities.ips),
        DOMAIN: list(entities.domains),
        FILE_HASH: [h.value for h in entities.file_hashes],
        PROCESS_NAME: list(entities.processes),
    }


ARTIFACT_TYPES: frozenset[str] = frozenset(
    value for name, value in dict(globals()).items() if name.isupper() and isinstance(value, str)
)
