"""Deterministic simulators for every tool in the catalog (the default execution mode).

Simulated results exist so the whole pipeline can run without real tools. They are:

* **labelled**: ``ToolResult.synthetic`` is True and every finding/observation starts with
  ``SYNTHETIC``; the evidence layer stores them apart from real observations;
* **deterministic**: the same request always gets the same answer (seeded by its content);
* **plausibly incomplete**: a read-only operation returns data with the probability given by
  the deployment's evidence-availability profile for its categories, otherwise ``no_data``;
* **harmless**: "destructive" operations only report what *would* have happened.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Callable, Optional

from app.schemas.enums import EvidenceCategory, ToolStatus
from app.schemas.tools import OperationSpec, ToolObservation, ToolResult
from app.tools import artifacts as A
from app.tools.base import SecurityTool

_PROCESSES = ["svchost.exe", "explorer.exe", "powershell.exe", "cmd.exe", "rundll32.exe", "msedge.exe", "taskhostw.exe"]
_DOMAINS = ["updates.example", "cdn.example.net", "login.example.org", "files.example.com"]


def _seed(*parts: Any) -> int:
    digest = hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _reputation(rng: random.Random, target: str) -> tuple[dict, str]:
    malicious = rng.choice([0, 0, 0, 1, 3, 9, 27])
    verdict = "malicious" if malicious >= 3 else "suspicious" if malicious else "clean"
    return ({"indicator": target, "malicious": malicious, "engines": 70, "verdict": verdict},
            f"{malicious}/70 engines flag the indicator (verdict: {verdict})")  # fmt: skip


def _processes(rng: random.Random, target: str) -> tuple[list, str]:
    rows = [{"pid": rng.randint(400, 9000), "name": rng.choice(_PROCESSES), "parent": rng.choice(_PROCESSES)}
            for _ in range(rng.randint(3, 8))]  # fmt: skip
    return rows, f"{len(rows)} process records"


def _records(kind: str) -> Callable[[random.Random, str], tuple[list, str]]:
    def make(rng: random.Random, target: str) -> tuple[list, str]:
        rows = [{"record": f"synthetic-{kind}-{i}"} for i in range(rng.randint(1, 6))]
        return rows, f"{len(rows)} {kind.replace('_', ' ')} records"

    return make


def _dns(rng: random.Random, target: str) -> tuple[list, str]:
    rows = [{"query": rng.choice(_DOMAINS), "answer": f"198.51.100.{rng.randint(1, 254)}"} for _ in range(rng.randint(1, 5))]
    return rows, f"{len(rows)} DNS queries"


def _alerts(rng: random.Random, target: str) -> tuple[list, str]:
    rows = [{"rule_id": rng.randint(100000, 199999), "level": rng.randint(3, 12)} for _ in range(rng.randint(1, 4))]
    return rows, f"{len(rows)} alerts (max level {max(r['level'] for r in rows)})"


_GENERATORS: dict[str, Callable[[random.Random, str], tuple[Any, str]]] = {
    A.REPUTATION_REPORT: _reputation,
    A.PROCESS_LIST: _processes,
    A.DNS_RECORDS: _dns,
    A.ALERT_LIST: _alerts,
    A.SIGMA_HITS: _alerts,
}


class SimulatedTool(SecurityTool):
    synthetic = True

    def __init__(self, name: str, availability: Optional[dict[EvidenceCategory, float]] = None) -> None:
        self.name = name
        super().__init__()
        self.availability = availability or {}

    async def run(
        self, spec: OperationSpec, target_type: str, target: str, context: dict[str, str], params: dict[str, Any]
    ) -> ToolResult:
        rng = random.Random(_seed(self.name, spec.name, target_type, target, context, params))
        label = f"SYNTHETIC (simulated {spec.key})"

        if spec.destructive:
            state = {"action": spec.name, "target_type": target_type, "target": target, "simulated": True,
                     "state": "simulated_only"}  # fmt: skip
            return self.result(
                spec, target_type,
                outputs={A.CONTAINMENT_STATUS: state},
                findings=[f"{label}: {spec.name} of the {target_type} was simulated; no real action was taken"],
            )  # fmt: skip

        probability = (
            sum(self.availability.get(c, 0.5) for c in spec.categories) / len(spec.categories) if spec.categories else 1.0
        )
        if rng.random() >= probability:
            return self.result(spec, target_type, status=ToolStatus.NO_DATA,
                               message=f"{label}: no data returned (this is not proof of absence)")  # fmt: skip

        outputs: dict[str, Any] = {}
        summaries: list[str] = []
        for artifact in spec.produces:
            generator = _GENERATORS.get(artifact, _records(artifact))
            value, summary = generator(rng, target)
            outputs[artifact] = value
            summaries.append(summary)
        summary = "; ".join(summaries)
        observations = [ToolObservation(category=c, description=f"{label}: {summary}") for c in spec.categories]
        return self.result(spec, target_type, outputs=outputs, observations=observations,
                           findings=[f"{label}: {summary}"])  # fmt: skip

    async def health(self) -> dict[str, Any]:
        return {"status": "simulated", "detail": "deterministic synthetic results; no real tool is contacted"}
