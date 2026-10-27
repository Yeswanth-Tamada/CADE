"""Walk one SYNTHETIC incident through the whole CADE pipeline against a running server.

    submit -> analyse -> respond (route: reuse / request evidence / synthesize) -> approve
           -> verify -> execute (simulate by default) -> feedback -> library

1. Start the server (offline example, PowerShell):
       $env:CADE_LLM_PROVIDER="mock"; $env:CADE_EMBEDDER="hashing"; uvicorn app.main:app
2. Issue keys (each printed once). Separate people approve and execute destructive steps:
       python -m app.security.keys create --analyst alice --roles analyst
       python -m app.security.keys create --analyst bob --roles approver
       python -m app.security.keys create --analyst carol --roles executor
3. Run:
       python scripts/demo_full_pipeline.py --api-key <alice> --approver-key <bob> --executor-key <carol>
       python scripts/demo_full_pipeline.py ... --example ransomware_analyst_report.json --verify-target <host>

Without --verify-target nobody confirms a compromise, so destructive steps are (correctly) skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "synthetic_incidents"


def check(response: httpx.Response, what: str) -> dict:
    if response.status_code >= 400:
        error = response.json().get("error", {}) if response.headers.get("content-type", "").startswith("application/json") else {}
        raise SystemExit(f"[{what}] failed with HTTP {response.status_code}: {error.get('category')}: {error.get('message')}")
    return response.json()


def main() -> int:
    parser = argparse.ArgumentParser(description="CADE full-pipeline demo client")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--example", default="encoded_powershell_edr.json")
    parser.add_argument("--api-key", default=os.environ.get("CADE_API_KEY"), help="analyst key (default: $CADE_API_KEY)")
    parser.add_argument("--approver-key", help="key with the approver role (to approve destructive steps)")
    parser.add_argument("--executor-key", help="key with the executor role (default: the analyst key)")
    parser.add_argument("--verify-target", help="entity value to confirm in verification steps (e.g. a host name)")
    parser.add_argument("--force-synthesis", action="store_true", help="synthesize even if a library playbook fits")
    parser.add_argument("--not-resolved", action="store_true", help="report that the run did not resolve the incident")
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    if not args.api_key:
        parser.error("an analyst API key is required (--api-key or CADE_API_KEY)")

    def client(key: str) -> httpx.Client:
        return httpx.Client(base_url=args.base_url, timeout=args.timeout, headers={"Authorization": f"Bearer {key}"})

    analyst = client(args.api_key)
    approver = client(args.approver_key) if args.approver_key else None
    executor = client(args.executor_key or args.api_key)

    health = check(analyst.get("/health"), "health")
    if "dependencies" not in health:
        raise SystemExit("the API key was not accepted")
    playbooks = health.get("playbooks", {})
    print(f"[health] {health['status']}; mode={playbooks.get('execution_mode')} llm={health['dependencies']['llm']['status']} "
          f"judge={playbooks.get('judge')} library={playbooks.get('library_entries')} embedder={playbooks.get('embedder')}")  # fmt: skip
    for warning in playbooks.get("warnings", []):
        print(f"[health] warning: {warning}")

    path = EXAMPLES / args.example
    data = json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else {"description": path.read_text()}
    data.pop("incident_id", None)  # always create a fresh incident
    incident_id = check(analyst.post("/incidents", json=data), "submit")["incident_id"]
    print(f"[1] submitted {path.name} -> {incident_id}")

    analysis = check(analyst.post(f"/incidents/{incident_id}/analyze"), "analyze")
    result = analysis["result"]
    print(f"[2] analysis {analysis['extraction_status']}: intent={result['intent']['label']} "
          f"({result['intent']['confidence']}), ATT&CK={[t['technique_id'] for t in result['intent']['attack_techniques']]}")  # fmt: skip

    kind = "synthesize" if args.force_synthesis else "respond"
    job = check(analyst.post(f"/incidents/{incident_id}/{kind}"), kind)
    print(f"[3] {kind} job {job['job_id']} started")
    while job["status"] in ("queued", "running"):
        time.sleep(2)
        job = check(analyst.get(f"/jobs/{job['job_id']}"), "job")
        print(f"    ... {job['status']}: {job.get('progress') or ''}")
    if job["status"] != "succeeded":
        raise SystemExit(f"[3] job {job['status']}: {job.get('error')}")
    outcome = job["result"]
    for decision in outcome["decisions"]:
        scores = decision["scores"]
        print(f"    route round {decision['round']}: {decision['action']} (reuse {scores['reuse']:.3f}, request "
              f"{scores['request_evidence']:.3f}, synthesize {scores['synthesize']:.3f})"
              f"{' [synthetic basis]' if decision['synthetic_basis'] else ''}")  # fmt: skip
    for report in outcome["enrichment"]:
        print(f"    evidence request: real {report['categories_obtained']}, simulated {report['categories_simulated']}, "
              f"none {report['categories_without_data']}")  # fmt: skip
    plan = outcome["plan"]
    print(f"[4] plan {plan['plan_id']} ({plan['source']}): {plan['playbook']['name']}")
    if plan.get("fitness"):
        f = plan["fitness"]
        print(f"    fitness {f['fitness']:.3f} = cbrt(IC {f['ic']:.2f} x EA {f['ea']:.2f} x OS {f['os']:.2f} [{f['os_source']}])")
    if plan.get("synthesis"):
        evolution, seeding = plan["synthesis"]["evolution"], plan["synthesis"]["seeding"]
        print(f"    synthesis: K={seeding['k']} seeds; {evolution['termination']} after {evolution['generations']} "
              f"generation(s); {evolution['evaluations']} evaluations; {evolution['judge_calls']} judge calls")  # fmt: skip
    for node in plan["playbook"]["nodes"]:
        marker = " [DESTRUCTIVE]" if node["node_id"] in plan["destructive_nodes"] else (
            " [VERIFICATION]" if node["node_id"] in plan["verification_nodes"] else "")  # fmt: skip
        print(f"    - {node['node_id']}: {node['action']}{marker}")

    if plan["destructive_nodes"]:
        if approver is None:
            print("[5] no --approver-key: destructive steps stay unapproved and will be skipped")
        else:
            check(approver.post(f"/response-plans/{plan['plan_id']}/approve", json={"node_ids": plan["destructive_nodes"]}),
                  "approve")  # fmt: skip
            print(f"[5] approved destructive steps {plan['destructive_nodes']}")
    if plan["verification_nodes"] and approver is not None and args.verify_target:
        for node_id in plan["verification_nodes"]:
            check(approver.post(f"/response-plans/{plan['plan_id']}/verifications",
                                json={"node_id": node_id, "verdict": "confirmed", "targets": [args.verify_target]}), "verify")  # fmt: skip
        print(f"[5] verification recorded: {args.verify_target} confirmed")

    run = check(executor.post(f"/response-plans/{plan['plan_id']}/execute"), "execute")
    print(f"[6] run {run['run_id']} ({run['mode']}{', SYNTHETIC' if run['synthetic'] else ''}): {run['status']}")
    for result in run["node_results"]:
        reason = f" -- {result['reason']}" if result.get("reason") else ""
        print(f"    {result['node_id']} {result['action']}: {result['status']}{reason}")

    feedback = check(analyst.post(f"/playbook-runs/{run['run_id']}/feedback",
                                  json={"resolved": not args.not_resolved, "findings_retained": not args.not_resolved}),
                     "feedback")["feedback"]  # fmt: skip
    outcome = feedback["library_outcome"]
    print(f"[7] feedback recorded; library: {outcome['action']} ({outcome['reason']})")
    entries = check(analyst.get("/library/playbooks"), "library")
    print(f"[8] library: {len(entries)} entries; with execution history: {sum(1 for e in entries if e['executions'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
