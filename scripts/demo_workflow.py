"""Walk through the Phase 1 workflow against a running server.

1. Start the server in another terminal, e.g. (offline, mock LLM):
       $env:CADE_LLM_PROVIDER="mock"; uvicorn app.main:app      # PowerShell
       CADE_LLM_PROVIDER=mock uvicorn app.main:app              # bash
2. Issue an API key with the analyst role (printed once):
       python -m app.security.keys create --analyst demo --roles analyst
3. Run (or set CADE_API_KEY instead of passing --api-key):
       python scripts/demo_workflow.py --api-key <key>
       python scripts/demo_workflow.py --api-key <key> --example phishing_siem_alert.json --base-url http://127.0.0.1:8000

The script submits a SYNTHETIC incident, analyses it, prints the intent and evidence
profile, and re-reads the stored analysis. It only talks to the API you point it at.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "synthetic_incidents"


def main() -> int:
    parser = argparse.ArgumentParser(description="CADE Phase 1 demo client")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--example", default="encoded_powershell_edr.json", help="file in examples/synthetic_incidents")
    parser.add_argument("--timeout", type=float, default=600, help="seconds to wait for the analysis")
    parser.add_argument("--api-key", default=os.environ.get("CADE_API_KEY"), help="API key (default: $CADE_API_KEY)")
    args = parser.parse_args()
    if not args.api_key:
        parser.error("an API key is required: --api-key <key> or set CADE_API_KEY "
                     "(create one with: python -m app.security.keys create --analyst demo --roles analyst)")

    path = EXAMPLES / args.example
    content_type = "application/json" if path.suffix == ".json" else "text/plain"
    headers = {"Authorization": f"Bearer {args.api_key}"}
    with httpx.Client(base_url=args.base_url, timeout=args.timeout, headers=headers) as client:
        health = client.get("/health").json()
        if "dependencies" not in health:
            print("[health] the API key was not accepted; check --api-key")
            return 1
        print(f"[health] overall={health['status']} llm={health['dependencies']['llm']['status']} "
              f"attack={health['dependencies']['attack_dataset']['mapping_mode']}")  # fmt: skip

        response = client.post("/incidents", content=path.read_bytes(), headers={"Content-Type": content_type})
        if response.status_code == 409:
            print("[create] incident ID already exists; remove 'incident_id' from the example or use a fresh DB")
            return 1
        response.raise_for_status()
        incident_id = response.json()["incident_id"]
        print(f"[1-2] submitted {path.name} -> incident_id={incident_id}")

        saved = client.get(f"/incidents/{incident_id}").json()
        print(f"[3] retrieved incident: status={saved['incident']['status']}, analysis={saved['latest_analysis']}")

        print("[4] analysing (this calls the configured LLM)...")
        response = client.post(f"/incidents/{incident_id}/analyze")
        analysis = response.json()
        if response.status_code != 200:
            print(f"    analysis FAILED with HTTP {response.status_code}: {json.dumps(analysis['error'])}")
            return 1

        result = analysis["result"]
        intent = result["intent"]
        profile = result["evidence_profile"]
        print(f"[5] extraction_status={analysis['extraction_status']} provider={analysis['provider']} "
              f"model={analysis['model_name']} duration_ms={analysis['duration_ms']:.0f}")  # fmt: skip
        print(f"    intent: {intent['label']} (confidence {intent['confidence']})")
        print(f"    summary: {result['summary']}")
        for technique in intent["attack_techniques"]:
            print(f"    ATT&CK {technique['technique_id']} {technique['name'] or ''} "
                  f"[{technique['validation_status']}, conf {technique['confidence']}]")  # fmt: skip
        print(f"    entities: {json.dumps(result['entities'])}")
        print(f"    observed: {[e['description'] for e in profile['observed']]}")
        print(f"    unconfirmed: {[e['description'] for e in profile['unconfirmed']]}")
        print(f"    missing: {[(e['description'], e['criticality']) for e in profile['missing']]}")
        print(f"    not_checked: {profile['not_checked']}")
        for warning in analysis["warnings"]:
            print(f"    warning: {warning}")

        again = client.get(f"/incidents/{incident_id}").json()
        same = again["latest_analysis"]["analysis_id"] == analysis["analysis_id"]
        print(f"[6] re-read: incident status={again['incident']['status']}, latest analysis matches={same}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
