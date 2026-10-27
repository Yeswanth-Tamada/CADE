"""Offline stand-in for a language model -- for tests and development without Ollama.

THIS IS NOT A REAL MODEL. It has two modes:

* **scripted**: pass ``responses=[...]`` (strings, dicts or exceptions) and each call returns
  the next item -- used by the tests to simulate good, malformed and failing models;
* **heuristic**: with no script, it builds a schema-valid answer from simple keyword rules so
  the full API workflow can be exercised offline.

Every analysis produced through it is labelled ``provider="mock"`` and carries a warning.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Any, Optional, Union

from app.intent.entity_extractor import extract_pattern_entities
from app.intent.llm_client import LLMClient, LLMHealth, LLMResponse
from app.intent.prompts import extract_incident_block

MockItem = Union[str, dict, Exception]

# (intent label, keywords, observed-evidence category, technique hints, missing evidence)
_RULES: list[tuple[str, tuple[str, ...], str, list[str], list[tuple[str, str, bool]]]] = [
    ("ransomware_investigation", ("ransom", "encrypted", "vssadmin", "shadow cop"), "file_activity", ["T1486"],
     [("process_telemetry", "Process tree of the process that modified the files", True),
      ("file_activity", "Scope of affected files and shares", False)]),
    ("phishing_investigation", ("phish", "email", "attachment", "sender"), "email", ["T1566"],
     [("email", "Full message headers and the original attachment or URL", True),
      ("user_account", "Whether any recipient opened the attachment or entered credentials", False)]),
    ("credential_compromise", ("brute force", "failed login", "failed logon", "password spray", "mfa"),
     "authentication", ["T1110"],
     [("authentication", "Authentication logs for the affected account(s)", True)]),
    ("lateral_movement", ("psexec", "lateral", "remote desktop", "rdp", "admin$", "wmic"), "authentication",
     ["T1021"], [("authentication", "Logon events on the destination hosts", True)]),
    ("data_exfiltration", ("exfiltrat", "large upload", "uploaded", "outbound transfer"), "network_traffic",
     ["T1041"], [("network_traffic", "Flow records for the transfer (volume, destination, duration)", True)]),
    ("denial_of_service", ("ddos", "denial of service", "flood", "requests per second"), "network_traffic",
     ["T1498"], [("network_traffic", "Traffic volume by source over the attack window", True)]),
    ("web_application_attack", ("sql injection", "sqli", "web shell", "webshell", "union select", "path traversal"),
     "web_server_logs", ["T1190"], [("web_server_logs", "Full web server access logs for the affected application", True)]),
    ("command_and_control_persistence", ("beacon", "command and control", "c2", "scheduled task", "run key"),
     "network_traffic", ["T1071"], [("dns", "DNS resolution history for the contacted destinations", True)]),
    ("privilege_escalation", ("privilege escalation", "uac bypass", "elevated", "token manipulation", "seimpersonate"),
     "process_telemetry", ["T1548.002"], [("authentication", "Privileged logons and account changes on the host", True)]),
    ("defense_evasion", ("defender disabled", "tamper", "log cleared", "cleared the security log", "obfuscat",
                         "masquerad"), "process_telemetry", ["T1562.001"],
     [("registry", "Registry changes to security tool and policy settings", True)]),
    ("discovery", ("whoami", "net group", "net user", "nltest", "systeminfo", "enumerat"), "process_telemetry",
     ["T1087"], [("process_telemetry", "All commands run by the same session", True)]),
    ("system_impact", ("wiper", "wiped", "disk wipe", "services stopped", "stopped services", "defaced"),
     "file_activity", ["T1485"], [("process_telemetry", "Process tree of the destructive tool", True)]),
    ("malware_investigation", ("malware", "trojan", "malicious file", "quarantine"), "file_activity", [],
     [("file_activity", "A copy or hash reputation of the suspicious file", True)]),
    ("suspicious_process_execution", ("powershell", "encoded command", "rundll32", "mshta", "cmd.exe"),
     "process_telemetry", ["T1059"], [("process_telemetry", "Full parent/child process tree with command lines", True)]),
]  # fmt: skip


def heuristic_analysis(incident_text: str, allowed_labels: Optional[list[str]] = None) -> dict[str, Any]:
    lowered = incident_text.lower()
    scored = [(sum(lowered.count(k) for k in rule[1]), index) for index, rule in enumerate(_RULES)]
    hits, index = max(scored, key=lambda s: (s[0], -s[1]))
    rule = _RULES[index] if hits else None
    if rule and allowed_labels and rule[0] not in allowed_labels:
        rule = None

    pattern = extract_pattern_entities(incident_text)
    analysis: dict[str, Any] = {
        "intent_label": rule[0] if rule else "unknown",
        "intent_confidence": 0.3 if rule else 0.1,
        "summary": "MOCK ANALYSIS (no language model was used): "
        + (f"keyword rules suggest '{rule[0]}'." if rule else "no keyword rule matched."),
        "entities": {
            "hosts": [],
            "users": [],
            "ips": pattern.ips,
            "domains": pattern.domains,
            "file_hashes": pattern.file_hashes,
            "processes": pattern.processes,
        },
        "observed_evidence": [],
        "unconfirmed_evidence": [],
        "missing_evidence": [],
        "attack_techniques": [],
        "uncertainties": ["Mock analysis generated by keyword rules; it is not a language-model judgement."],
        "conflicts": [],
    }
    if rule:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", incident_text) if s.strip()]
        for sentence in sentences:
            if any(k in sentence.lower() for k in rule[1]):
                analysis["observed_evidence"].append(
                    {"category": rule[2], "description": f"Keyword match: {sentence[:120]}", "excerpt": sentence}
                )
                break
        analysis["missing_evidence"] = [
            {"category": c, "description": d, "critical": crit, "rationale": "standard need for this intent"}
            for c, d, crit in rule[4]
        ]
        analysis["attack_techniques"] = [
            {"technique_id": t, "confidence": 0.2, "justification": "keyword heuristic (mock)"} for t in rule[3]
        ]
    return analysis


MOCK_JUDGEMENT = {"tool_appropriateness": 3, "sequencing": 3, "coherence": 3, "destructive_safety": 3,
                  "rationale": "MOCK judgement: fixed scores, no language model was used"}  # fmt: skip


def mock_playbook_candidate(json_schema: dict[str, Any], user_prompt: str, temperature: Optional[float]) -> dict[str, Any]:
    """A random (but seeded) candidate built from the schema's allowed actions. Labelled as mock."""
    allowed = json_schema["$defs"]["CandidateStep"]["properties"]["action"]["enum"]
    seed = hashlib.sha256(f"{user_prompt}|{temperature}".encode("utf-8")).hexdigest()
    rng = random.Random(int(seed[:16], 16))
    middle = [a for a in allowed if a not in ("extract_entities", "generate_incident_report")]
    steps = [{"id": "s1", "action": "extract_entities", "depends_on": []}]
    for index, action in enumerate(rng.sample(middle, rng.randint(3, 8)), start=2):
        earlier = [s["id"] for s in steps]
        steps.append({"id": f"s{index}", "action": action, "depends_on": rng.sample(earlier, min(len(earlier), rng.randint(1, 2)))})
    used = {d for s in steps for d in s["depends_on"]}
    steps.append({"id": "s99", "action": "generate_incident_report",
                  "depends_on": [s["id"] for s in steps if s["id"] not in used]})  # fmt: skip
    return {"name": "MOCK candidate (no language model was used)", "rationale": "mock", "steps": steps}


class MockLLMClient(LLMClient):
    provider = "mock"

    def __init__(self, responses: Optional[list[MockItem]] = None, model: str = "mock-keyword-heuristic") -> None:
        self.model = model
        self._responses = list(responses or [])
        self.calls: list[dict[str, Any]] = []

    async def generate_json(
        self, *, system_prompt: str, user_prompt: str, json_schema: dict[str, Any], temperature: Optional[float] = None
    ) -> LLMResponse:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt, "json_schema": json_schema,
                           "temperature": temperature})  # fmt: skip
        if self._responses:
            item = self._responses.pop(0)
            if isinstance(item, Exception):
                raise item
            content = item if isinstance(item, str) else json.dumps(item)
        elif json_schema.get("title") == "PlaybookCandidate":
            content = json.dumps(mock_playbook_candidate(json_schema, user_prompt, temperature))
        elif json_schema.get("title") == "SoundnessJudgement":
            content = json.dumps(MOCK_JUDGEMENT)
        elif json_schema.get("title") == "IntentDecision":  # keyword rules, as for the full analysis
            allowed = json_schema["properties"]["intent_label"]["enum"]
            analysis = heuristic_analysis(extract_incident_block(user_prompt), allowed)
            tactics = json_schema["properties"]["tactic"]["enum"]
            content = json.dumps({"activity": "MOCK: keyword rules, no language model was used",
                                  "tactic": "none" if "none" in tactics else tactics[0],
                                  "intent_label": analysis["intent_label"] if analysis["intent_label"] in allowed else allowed[0],
                                  "intent_confidence": analysis["intent_confidence"]})  # fmt: skip
        elif json_schema.get("title") == "GapProposals":  # deterministic: the first allowed action per gap
            content = json.dumps({key: spec["enum"][0] for key, spec in json_schema["properties"].items()})
        else:
            labels = json_schema.get("properties", {}).get("intent_label", {}).get("enum")
            content = json.dumps(heuristic_analysis(extract_incident_block(user_prompt), labels))
        return LLMResponse(content=content, model=self.model, duration_ms=0.0)

    async def health(self) -> LLMHealth:
        return LLMHealth(self.provider, self.model, True, True, "mock client: output is not produced by a language model")
