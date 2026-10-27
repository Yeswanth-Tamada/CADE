"""VirusTotal API v3 adapter (works with the free tier: 4 requests/minute, enforced client-side).

Only indicator *lookups* are made (GET); nothing is uploaded. A 404 means VirusTotal has no
record of the indicator, which is reported as ``no_data``, not as "clean".
"""

from __future__ import annotations

from typing import Any, Optional

import httpx

from app.schemas.enums import ToolStatus
from app.schemas.tools import OperationSpec, ToolObservation, ToolResult
from app.tools import artifacts as A
from app.tools.base import HttpTool, RateLimiter

_PATHS = {"file_reputation": "/files/{}", "ip_reputation": "/ip_addresses/{}", "domain_reputation": "/domains/{}"}


class VirusTotalTool(HttpTool):
    name = "virustotal"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://www.virustotal.com/api/v3",
        timeout_seconds: float = 20.0,
        verify_tls: bool = True,
        requests_per_minute: float = 4,
        malicious_threshold: int = 3,
        limiter: Optional[RateLimiter] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        super().__init__(base_url=base_url, timeout_seconds=timeout_seconds, verify_tls=verify_tls,
                         headers={"x-apikey": api_key, "Accept": "application/json"}, transport=transport)  # fmt: skip
        self.limiter = limiter or RateLimiter(requests_per_minute)
        self.malicious_threshold = malicious_threshold

    async def run(
        self, spec: OperationSpec, target_type: str, target: str, context: dict[str, str], params: dict[str, Any]
    ) -> ToolResult:
        await self.limiter.acquire(self.name)
        reply = await self.request("GET", _PATHS[spec.name].format(target))
        if reply.status_code == 404:
            return self.result(spec, target_type, status=ToolStatus.NO_DATA,
                               message="VirusTotal has no record of this indicator")  # fmt: skip
        if reply.status_code >= 400:
            return self.result(spec, target_type, status=ToolStatus.ERROR, message=f"HTTP {reply.status_code}")
        attributes = (reply.json().get("data") or {}).get("attributes") or {}
        stats = attributes.get("last_analysis_stats") or {}
        malicious = int(stats.get("malicious", 0))
        suspicious = int(stats.get("suspicious", 0))
        engines = sum(int(v) for v in stats.values() if isinstance(v, int))
        verdict = "malicious" if malicious >= self.malicious_threshold else "suspicious" if malicious or suspicious else "clean"
        report = {
            "indicator": target,
            "malicious": malicious,
            "suspicious": suspicious,
            "engines": engines,
            "reputation": attributes.get("reputation"),
            "verdict": verdict,
        }
        summary = f"VirusTotal: {malicious}/{engines} engines flag {target_type} {target} (verdict: {verdict})"
        return self.result(
            spec, target_type,
            outputs={A.REPUTATION_REPORT: report},
            observations=[ToolObservation(category=c, description=summary) for c in spec.categories],
            findings=[summary],
        )  # fmt: skip

    async def health(self) -> dict[str, Any]:
        return {"status": "configured", "detail": "API key set; no request is made by the health check"}
