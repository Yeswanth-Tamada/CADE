"""Check that the real security tools of paper 968 answer: VirusTotal, Elastic and Sigma.

The adapters' ``health()`` only reports whether settings are present; this report makes one
**read-only** call per configured tool, in live mode, and prints what came back. Nothing
destructive is ever invoked, nothing is written to the database, and no simulated result is
used: an unconfigured or failing tool shows as an error.

    python scripts/check_tools.py                                  # uses CADE_* settings / .env
    python scripts/check_tools.py --host WIN-HOST01 --technique T1059.001

Wazuh, osquery and YARA have adapters but are not part of paper 968, so they are not probed.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.core.errors import ToolInputError  # noqa: E402
from app.schemas.enums import ExecutionMode, ToolFailurePolicy, ToolStatus  # noqa: E402
from app.schemas.tools import ToolRequest  # noqa: E402
from app.tools import artifacts as A  # noqa: E402
from app.tools.registry import ToolGateway, ToolRegistry, build_real_tools  # noqa: E402

# SHA-256 of the EICAR anti-virus test file: harmless, and known to VirusTotal.
EICAR_SHA256 = "275a021bbfb6489e54d471899f7db9d1663fc695ec2fe2a2c4538aabf651fd0f"
PAPER_TOOLS = ("virustotal", "elastic", "sigma")
WIDEST_LOOKBACK = {"lookback_hours": 168}


def probes(host: str, file_hash: str, technique: str) -> dict[str, list[ToolRequest]]:
    return {
        "virustotal": [ToolRequest(operation="file_reputation", target_type=A.FILE_HASH, target=file_hash)],
        "elastic": [
            ToolRequest(operation="process_events", target_type=A.HOST_ID, target=host, params=WIDEST_LOOKBACK),
            ToolRequest(operation="script_block_events", target_type=A.HOST_ID, target=host, params=WIDEST_LOOKBACK),
        ],
        "sigma": [ToolRequest(operation="hunt_technique", target_type=A.TECHNIQUE_ID, target=technique,
                              params=WIDEST_LOOKBACK)],
    }  # fmt: skip


async def run(host: str, file_hash: str, technique: str) -> int:
    settings = get_settings()
    real = build_real_tools(settings)
    gateway = ToolGateway(ToolRegistry(ExecutionMode.LIVE, real), ToolFailurePolicy.ERROR)
    failures = 0
    try:
        for tool, requests in probes(host, file_hash, technique).items():
            if tool not in real:
                print(f"{tool:<11} not configured (see .env.example)")
                failures += 1
                continue
            print(f"{tool:<11} configured: {await real[tool].health()}")
            for request in requests:
                try:
                    result = await gateway.invoke(tool, request)
                except ToolInputError as exc:
                    print(f"  {request.operation:<20} input rejected: {exc.message}")
                    failures += 1
                    continue
                detail = result.findings[0] if result.findings else (result.message or "")
                print(f"  {request.operation:<20} {result.status.value:<8} {result.duration_ms:7.0f} ms  {detail}")
                if result.status in (ToolStatus.ERROR, ToolStatus.UNAVAILABLE, ToolStatus.MANUAL_TASK):
                    failures += 1
    finally:
        await gateway.registry.aclose()
    print("\nno_data means the tool answered but found nothing (e.g. no logs loaded for that host yet).")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="cade-lab-host", help="host.name to search in Elastic")
    parser.add_argument("--hash", default=EICAR_SHA256, help="file hash for the VirusTotal lookup")
    parser.add_argument("--technique", default="T1059.001", help="ATT&CK technique for the Sigma hunt")
    args = parser.parse_args()
    return asyncio.run(run(args.host, args.hash, args.technique))


if __name__ == "__main__":
    raise SystemExit(main())
