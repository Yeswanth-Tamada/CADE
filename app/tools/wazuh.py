"""Wazuh adapter: server API (agents, syscollector, syscheck, active response) and indexer (alerts).

* Authentication: ``POST /security/user/authenticate`` with basic auth returns a JWT that is
  cached and refreshed once on HTTP 401.
* Alerts live in the Wazuh indexer (OpenSearch); ``host_alerts`` needs the indexer settings.
* Active response is destructive. ``block_ip`` uses the configured firewall command (default
  ``!firewall-drop``); ``isolate_host`` needs an explicitly configured isolation command, because
  Wazuh ships none -- without one the operation is reported unavailable, never faked.
"""

from __future__ import annotations

from typing import Any, Optional

import httpx

from app.core.errors import ToolAuthError, ToolError, ToolUnavailableError
from app.schemas.enums import ToolStatus
from app.schemas.tools import OperationSpec, ToolObservation, ToolResult
from app.tools import artifacts as A
from app.tools.base import HttpChannel, HttpReply, SecurityTool

_MAX_ROWS = 50


class WazuhTool(SecurityTool):
    name = "wazuh"

    def __init__(
        self,
        *,
        api_url: str,
        username: str,
        password: str,
        indexer_url: Optional[str] = None,
        indexer_username: Optional[str] = None,
        indexer_password: Optional[str] = None,
        isolation_command: Optional[str] = None,
        block_ip_command: str = "!firewall-drop",
        timeout_seconds: float = 20.0,
        verify_tls: bool = True,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        indexer_transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        super().__init__()
        self._credentials = (username, password)
        self._token: Optional[str] = None
        self.isolation_command = isolation_command
        self.block_ip_command = block_ip_command
        self.api = HttpChannel(self.name, base_url=api_url, timeout_seconds=timeout_seconds, verify_tls=verify_tls,
                               transport=transport)  # fmt: skip
        self.indexer = (
            HttpChannel(f"{self.name} indexer", base_url=indexer_url, timeout_seconds=timeout_seconds,
                        verify_tls=verify_tls, auth=(indexer_username or "", indexer_password or ""),
                        transport=indexer_transport)  # fmt: skip
            if indexer_url
            else None
        )

    # --- server API helpers ------------------------------------------------------------

    async def _authenticate(self) -> str:
        reply = await self.api.request("POST", "/security/user/authenticate", auth=self._credentials)
        token = ((reply.json() or {}).get("data") or {}).get("token") if reply.status_code == 200 else None
        if not token:
            raise ToolAuthError("wazuh did not issue an API token")
        self._token = token
        return token

    async def _api(self, method: str, url: str, **kwargs: Any) -> HttpReply:
        for attempt in (1, 2):
            token = self._token or await self._authenticate()
            try:
                return await self.api.request(method, url, headers={"Authorization": f"Bearer {token}"}, **kwargs)
            except ToolAuthError:
                self._token = None  # expired token: re-authenticate once
                if attempt == 2:
                    raise
        raise AssertionError("unreachable")

    @staticmethod
    def _items(reply: HttpReply) -> list[dict]:
        if reply.status_code >= 400:
            raise ToolError(f"wazuh returned HTTP {reply.status_code}")
        return list(((reply.json() or {}).get("data") or {}).get("affected_items") or [])

    async def _agent_id(self, host: str) -> Optional[str]:
        items = self._items(await self._api("GET", "/agents", params={"name": host, "select": "id,name,status"}))
        return str(items[0]["id"]) if items else None

    # --- operations --------------------------------------------------------------------

    async def run(
        self, spec: OperationSpec, target_type: str, target: str, context: dict[str, str], params: dict[str, Any]
    ) -> ToolResult:
        if spec.name == "host_alerts":
            return await self._host_alerts(spec, target_type, target, params)
        host = context.get(A.HOST_ID, target) if spec.name == "block_ip" else target
        agent_id = await self._agent_id(host)
        if agent_id is None:
            if spec.destructive:
                raise ToolError("no Wazuh agent is registered under that host name")
            return self.result(spec, target_type, status=ToolStatus.NO_DATA, message="no Wazuh agent with that name")
        if spec.name == "host_processes":
            rows = self._items(await self._api("GET", f"/syscollector/{agent_id}/processes",
                                               params={"limit": params["limit"]}))  # fmt: skip
            processes = [{"pid": r.get("pid"), "name": r.get("name"), "ppid": r.get("ppid"), "cmd": r.get("cmd")}
                         for r in rows[:_MAX_ROWS]]  # fmt: skip
            return self._listing(spec, target_type, A.PROCESS_LIST, processes, "process inventory entries")
        if spec.name == "file_integrity_events":
            rows = self._items(await self._api("GET", f"/syscheck/{agent_id}", params={"limit": params["limit"]}))
            events = [{"file": r.get("file"), "event": r.get("type"), "date": r.get("date"), "sha256": r.get("sha256")}
                      for r in rows[:_MAX_ROWS]]  # fmt: skip
            return self._listing(spec, target_type, A.FILE_EVENTS, events, "file integrity events")
        if spec.name == "isolate_host":
            if not self.isolation_command:
                raise ToolUnavailableError("no Wazuh isolation active-response command is configured")
            return await self._active_response(spec, target_type, agent_id, self.isolation_command, None)
        if spec.name == "block_ip":
            return await self._active_response(spec, target_type, agent_id, self.block_ip_command, target)
        raise ToolError(f"unsupported wazuh operation {spec.name}")

    def _listing(self, spec: OperationSpec, target_type: str, artifact: str, rows: list, noun: str) -> ToolResult:
        if not rows:
            return self.result(spec, target_type, status=ToolStatus.NO_DATA, message=f"no {noun} returned")
        summary = f"Wazuh returned {len(rows)} {noun}"
        return self.result(spec, target_type, outputs={artifact: rows},
                           observations=[ToolObservation(category=c, description=summary) for c in spec.categories],
                           findings=[summary])  # fmt: skip

    async def _host_alerts(self, spec: OperationSpec, target_type: str, host: str, params: dict) -> ToolResult:
        if self.indexer is None:
            raise ToolUnavailableError("the Wazuh indexer is not configured (needed for alerts)")
        query = {
            "size": params["limit"],
            "sort": [{"timestamp": {"order": "desc"}}],
            "query": {"bool": {"filter": [
                {"term": {"agent.name": host}},
                {"range": {"timestamp": {"gte": f"now-{params['lookback_hours']}h"}}},
            ]}},
            "_source": ["timestamp", "rule.id", "rule.level", "rule.description", "rule.mitre.id"],
        }  # fmt: skip
        reply = await self.indexer.request("POST", "/wazuh-alerts-*/_search", json=query)
        if reply.status_code >= 400:
            raise ToolError(f"wazuh indexer returned HTTP {reply.status_code}")
        hits = (((reply.json() or {}).get("hits") or {}).get("hits")) or []
        alerts = [h.get("_source", {}) for h in hits[:_MAX_ROWS]]
        return self._listing(spec, target_type, A.ALERT_LIST, alerts, "alerts")

    async def _active_response(self, spec, target_type, agent_id: str, command: str, srcip: Optional[str]) -> ToolResult:
        body: dict[str, Any] = {"command": command, "arguments": []}
        if srcip:
            body["alert"] = {"data": {"srcip": srcip}}
        reply = await self._api("PUT", "/active-response", params={"agents_list": agent_id}, json=body)
        data = (reply.json() or {}).get("data") or {}
        if reply.status_code >= 400 or not data.get("total_affected_items"):
            raise ToolError("wazuh did not confirm the active response")
        state = {"action": spec.name, "agent_id": agent_id, "command": command, "simulated": False, "state": "requested"}
        return self.result(spec, target_type, outputs={A.CONTAINMENT_STATUS: state},
                           findings=[f"Wazuh accepted active response '{command}' for agent {agent_id}"])  # fmt: skip

    async def health(self) -> dict[str, Any]:
        return {"status": "configured", "indexer": self.indexer is not None,
                "isolation_command": bool(self.isolation_command)}  # fmt: skip

    async def aclose(self) -> None:
        await self.api.aclose()
        if self.indexer is not None:
            await self.indexer.aclose()
