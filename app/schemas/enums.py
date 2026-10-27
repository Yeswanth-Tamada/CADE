"""Enumerations shared by the incident schemas.

``str``-based enums serialise as plain strings in JSON and SQLite.
"""

from __future__ import annotations

from enum import Enum


class IncidentSource(str, Enum):
    """Where an incident came from (paper §III: SIEM alert, threat-intel report, log excerpt, analyst)."""

    SIEM_ALERT = "siem_alert"
    ANALYST_DESCRIPTION = "analyst_description"
    LOG_EXCERPT = "log_excerpt"
    THREAT_INTEL_REPORT = "threat_intel_report"
    OTHER = "other"


class Severity(str, Enum):
    INFORMATIONAL = "informational"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentStatus(str, Enum):
    RECEIVED = "received"
    ANALYZING = "analyzing"
    ANALYZED = "analyzed"
    ANALYSIS_FAILED = "analysis_failed"


class ExtractionStatus(str, Enum):
    SUCCESS = "success"  # valid model output, nothing had to be discarded
    PARTIAL = "partial"  # valid model output, but some content was discarded or downgraded (see warnings)
    FAILED = "failed"  # no usable analysis was produced


class EvidenceCategory(str, Enum):
    """Evidence categories, loosely aligned with MITRE ATT&CK data sources.

    The paper's seven host log-source categories (Section 6.1) map onto process_telemetry,
    network_traffic, file_activity, registry, authentication, script_execution and
    scheduled_tasks_services (see ``app/data/log_source_categories.json``); the others extend the
    vocabulary to non-host sources (implementation choice).
    """

    PROCESS_TELEMETRY = "process_telemetry"
    COMMAND_LINE = "command_line"
    SCRIPT_EXECUTION = "script_execution"
    FILE_ACTIVITY = "file_activity"
    NETWORK_TRAFFIC = "network_traffic"
    DNS = "dns"
    AUTHENTICATION = "authentication"
    USER_ACCOUNT = "user_account"
    EMAIL = "email"
    WEB_SERVER_LOGS = "web_server_logs"
    MEMORY = "memory"
    REGISTRY = "registry"
    SCHEDULED_TASKS_SERVICES = "scheduled_tasks_services"
    CLOUD_AUDIT = "cloud_audit"
    INDICATORS_OF_COMPROMISE = "indicators_of_compromise"
    SECURITY_ALERT = "security_alert"
    OTHER = "other"


class EvidenceStatus(str, Enum):
    OBSERVED = "observed"  # explicitly present in the incident input (verified against the text)
    INFERRED = "inferred"  # model hypothesis; not confirmed by any source
    MISSING = "missing"  # needed but not present in the input -- NOT proof that it does not exist
    NOT_CHECKED = "not_checked"  # relevant to the intent, but nothing has examined it yet


class Criticality(str, Enum):
    """Paper §IV partitions required evidence into critical (R_c) and supporting (R_s) items."""

    CRITICAL = "critical"
    SUPPORTING = "supporting"


class ProvenanceOrigin(str, Enum):
    INCIDENT_TEXT = "incident_text"  # verbatim excerpt found in the description / log excerpt
    STRUCTURED_INPUT = "structured_input"  # an explicit structured field supplied with the incident
    PATTERN_EXTRACTION = "pattern_extraction"  # deterministic regex extraction from the input
    LLM_INFERENCE = "llm_inference"  # produced by the language model, not verified
    INTENT_TAXONOMY = "intent_taxonomy"  # derived from the configured taxonomy
    TOOL = "tool"  # returned by a real (non-simulated) security tool
    ANALYST_SUPPLIED = "analyst_supplied"  # added by an authenticated analyst, attributed in source_ref


class HashAlgorithm(str, Enum):
    MD5 = "md5"
    SHA1 = "sha1"
    SHA256 = "sha256"


class AttackValidationStatus(str, Enum):
    # The ID exists and is active in the loaded ATT&CK dataset. This does NOT confirm that the
    # technique applies to the incident -- relevance is still the model's (uncertain) claim.
    VALIDATED = "validated"
    UNVERIFIED = "unverified"  # well-formed ID, but no dataset was loaded to check it
    DEPRECATED = "deprecated"  # exists in the dataset but is deprecated
    REVOKED = "revoked"  # exists in the dataset but was revoked (replaced)


class AttackMappingMode(str, Enum):
    VERIFIED = "verified"  # a versioned ATT&CK dataset was loaded
    UNVERIFIED = "unverified"  # no dataset: technique IDs are format-checked only


class ExecutionMode(str, Enum):
    """``simulate`` (default): every tool call is answered by a deterministic simulator and labelled
    synthetic. ``live``: calls go to configured real tools; nothing is ever simulated."""

    SIMULATE = "simulate"
    LIVE = "live"


class ToolFailurePolicy(str, Enum):
    """What happens when a real tool is unavailable or fails (never: fall back to simulation)."""

    MANUAL_TASK = "manual_task"  # record a manual investigation task for an analyst
    ERROR = "error"  # surface the error to the caller


class ToolStatus(str, Enum):
    SUCCESS = "success"  # the tool answered with data
    NO_DATA = "no_data"  # the tool answered, but returned nothing -- NOT proof of absence
    ERROR = "error"  # the tool failed (timeout, HTTP error, bad response)
    UNAVAILABLE = "unavailable"  # the tool is not configured / not reachable
    MANUAL_TASK = "manual_task"  # failure converted into a task for an analyst (policy)


class EvidenceAdditionKind(str, Enum):
    TOOL = "tool"  # observed by a real tool
    ANALYST = "analyst"  # supplied by an analyst
    SIMULATED = "simulated"  # synthetic output of a simulated tool; never treated as observed
