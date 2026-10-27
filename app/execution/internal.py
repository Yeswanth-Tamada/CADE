"""CADE's own playbook steps (bindings ``cade.*``): deterministic, local, no external calls.

Each handler receives the analysed incident and its typed inputs (lists of values) and returns
typed outputs plus human-readable findings. Analysis handlers summarise what upstream tools
returned; they never invent indicators.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional

from app.schemas.incident import CanonicalIncident
from app.schemas.response import NodeStatus, Verdict
from app.tools import artifacts as A
from app.tools.artifacts import entity_artifacts

_SUSPICIOUS_PROCESS = re.compile(r"powershell|rundll32|mshta|regsvr32|wscript|cscript|certutil|bitsadmin|-enc", re.I)
_SCANNABLE = re.compile(r"\.(exe|dll|ps1|vbs|js|hta|scr|bat|cmd|jar)$", re.I)


@dataclass
class HandlerResult:
    status: NodeStatus
    outputs: dict[str, list[Any]] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)
    verdict: Optional[Verdict] = None
    targets: list[str] = field(default_factory=list)


@dataclass
class ExecutionContext:
    incident: CanonicalIncident
    findings_so_far: list[str]


Handler = Callable[[ExecutionContext, dict[str, list[Any]]], HandlerResult]


def _ok(outputs: dict[str, list[Any]], *findings: str) -> HandlerResult:
    return HandlerResult(NodeStatus.SUCCEEDED, {**outputs, A.FINDINGS: list(findings)}, list(findings))


def extract_entities(ctx: ExecutionContext, inputs: dict) -> HandlerResult:
    values = {k: v for k, v in entity_artifacts(ctx.incident.entities).items() if v}
    techniques = [t.technique_id for t in ctx.incident.intent.attack_techniques if t.validation_status.value == "validated"]
    if techniques:
        values[A.TECHNIQUE_ID] = techniques
    counts = ", ".join(f"{len(v)} {k}" for k, v in values.items()) or "no entities"
    return _ok(values, f"Entities in scope: {counts}")


def triage_alert(ctx: ExecutionContext, inputs: dict) -> HandlerResult:
    incident = ctx.incident
    critical = [i.category.value for i in incident.evidence_profile.missing if i.criticality and i.criticality.value == "critical"]
    severity = incident.severity.value if incident.severity else "unspecified"
    return _ok({}, f"Triage: intent {incident.intent.label} (confidence {incident.intent.confidence:.2f}), "
                   f"severity {severity}; critical evidence missing: {', '.join(critical) or 'none'}")  # fmt: skip


def analyze_process_tree(ctx: ExecutionContext, inputs: dict) -> HandlerResult:
    rows = [r for r in inputs.get(A.PROCESS_LIST, []) if isinstance(r, dict)]
    flagged = sorted({str(r.get("name") or r.get("process.name") or "") for r in rows
                      if _SUSPICIOUS_PROCESS.search(" ".join(str(v) for v in r.values()))} - {""})  # fmt: skip
    return _ok({}, f"Process analysis: {len(rows)} records, suspicious: {', '.join(flagged[:10]) or 'none'}")


def analyze_network_flows(ctx: ExecutionContext, inputs: dict) -> HandlerResult:
    rows = [r for r in inputs.get(A.NETWORK_FLOWS, []) if isinstance(r, dict)]
    destinations = {str(r.get("destination.ip")) for r in rows if r.get("destination.ip")}
    return _ok({}, f"Network analysis: {len(rows)} flows to {len(destinations)} distinct destinations")


def analyze_auth_patterns(ctx: ExecutionContext, inputs: dict) -> HandlerResult:
    rows = [r for r in inputs.get(A.AUTH_EVENTS, []) if isinstance(r, dict)]
    failures = sum(1 for r in rows if str(r.get("event.outcome", "")).lower() == "failure")
    return _ok({}, f"Authentication analysis: {len(rows)} events, {failures} failures")


def analyze_email(ctx: ExecutionContext, inputs: dict) -> HandlerResult:
    rows = [r for r in inputs.get(A.EMAIL_MESSAGES, []) if isinstance(r, dict)]
    senders = {str(r.get("email.from.address")) for r in rows if r.get("email.from.address")}
    return _ok({}, f"Email analysis: {len(rows)} messages from {len(senders)} senders")


def select_suspicious_files(ctx: ExecutionContext, inputs: dict) -> HandlerResult:
    rows = [r for r in inputs.get(A.FILE_EVENTS, []) if isinstance(r, dict)]
    paths = sorted({str(r["file"]) for r in rows if isinstance(r.get("file"), str) and _SCANNABLE.search(r["file"])})
    if not paths:
        return HandlerResult(NodeStatus.NO_DATA, {A.FINDINGS: ["No scannable files among the file events"]},
                             ["No scannable files among the file events"])  # fmt: skip
    return _ok({A.FILE_PATH: paths[:10]}, f"Selected {min(len(paths), 10)} file(s) for scanning")


def assess_scope(ctx: ExecutionContext, inputs: dict) -> HandlerResult:
    entities = ctx.incident.entities
    scope = {"hosts": len(entities.hosts), "users": len(entities.users), "findings": len(inputs.get(A.FINDINGS, []))}
    return _ok({A.SCOPE_ASSESSMENT: [scope]},
               f"Scope: {scope['hosts']} host(s), {scope['users']} user(s), {scope['findings']} upstream finding(s)")  # fmt: skip


def verify_reputation(ctx: ExecutionContext, inputs: dict) -> HandlerResult:
    reports = [r for r in inputs.get(A.REPUTATION_REPORT, []) if isinstance(r, dict)]
    malicious = sorted({str(r.get("indicator")) for r in reports if r.get("verdict") == "malicious" and r.get("indicator")})
    verdict = Verdict.CONFIRMED if malicious else Verdict.NOT_CONFIRMED
    result = {"verdict": verdict.value, "target_type": "indicator", "targets": malicious}
    finding = (f"Verification: {len(malicious)} indicator(s) confirmed malicious by reputation" if malicious
               else "Verification: no indicator has a malicious reputation verdict")  # fmt: skip
    return HandlerResult(NodeStatus.SUCCEEDED, {A.VERIFICATION_RESULT: [result], A.FINDINGS: [finding]}, [finding],
                         verdict=verdict, targets=malicious)  # fmt: skip


def generate_report(ctx: ExecutionContext, inputs: dict) -> HandlerResult:
    lines = list(dict.fromkeys(str(f) for f in inputs.get(A.FINDINGS, [])))
    return HandlerResult(NodeStatus.SUCCEEDED, {A.INCIDENT_REPORT: lines}, [f"Report compiled from {len(lines)} finding(s)"])


HANDLERS: dict[str, Handler] = {
    "extract_entities": extract_entities,
    "triage_alert": triage_alert,
    "analyze_process_tree": analyze_process_tree,
    "analyze_network_flows": analyze_network_flows,
    "analyze_auth_patterns": analyze_auth_patterns,
    "analyze_email": analyze_email,
    "select_suspicious_files": select_suspicious_files,
    "assess_scope": assess_scope,
    "verify_reputation": verify_reputation,
    "generate_report": generate_report,
}
