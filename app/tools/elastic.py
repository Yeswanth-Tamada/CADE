"""Elasticsearch / Elastic Security adapter.

Searches ECS-formatted events (``event.category``) for a host, user, IP or domain within a
look-back window, using structured ``term`` filters (no query-string interpolation). Windows
event-log operations (registry, PowerShell script blocks, services / scheduled tasks) filter by
``winlog.channel`` and ``event.code`` from the log-source mapping. Host isolation goes through
Kibana's Elastic Defend response-action API and requires ``kibana_url``.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import httpx

from app.core.errors import ToolError, ToolUnavailableError
from app.schemas.enums import EvidenceCategory, ToolStatus
from app.schemas.tools import OperationSpec, ToolObservation, ToolResult
from app.tools import artifacts as A
from app.tools.base import HttpChannel, HttpTool
from app.tools.log_sources import LogSourceMap, default_log_sources

# operation -> (ECS event.category or None, {target artifact: ECS fields that may hold it}, output artifact, fields)
_SEARCHES: dict[str, tuple[Optional[str], dict[str, tuple[str, ...]], str, tuple[str, ...]]] = {
    "process_events": ("process", {A.HOST_ID: ("host.name",)}, A.PROCESS_LIST,
                       ("@timestamp", "process.name", "process.command_line", "process.parent.name", "user.name")),
    "network_events": ("network", {A.HOST_ID: ("host.name",), A.IP_ADDRESS: ("source.ip", "destination.ip")},
                       A.NETWORK_FLOWS, ("@timestamp", "source.ip", "destination.ip", "destination.port", "network.bytes")),
    "dns_events": ("network", {A.HOST_ID: ("host.name",), A.DOMAIN: ("dns.question.name",)}, A.DNS_RECORDS,
                   ("@timestamp", "host.name", "dns.question.name", "dns.resolved_ip")),
    "auth_events": ("authentication", {A.USER_ID: ("user.name",), A.HOST_ID: ("host.name",)}, A.AUTH_EVENTS,
                    ("@timestamp", "user.name", "host.name", "source.ip", "event.outcome")),
    "web_events": ("web", {A.HOST_ID: ("host.name",), A.IP_ADDRESS: ("source.ip",)}, A.WEB_REQUESTS,
                   ("@timestamp", "source.ip", "url.original", "http.response.status_code")),
    "email_events": ("email", {A.USER_ID: ("email.to.address", "user.name")}, A.EMAIL_MESSAGES,
                     ("@timestamp", "email.from.address", "email.subject", "email.attachments.file.name")),
    "cloud_audit_events": (None, {A.USER_ID: ("user.name",)}, A.AUTH_EVENTS,
                           ("@timestamp", "cloud.provider", "event.action", "event.outcome", "source.ip")),
}  # fmt: skip

# operation -> (evidence category whose channel / event codes to search, output artifact, fields)
_EVENT_CODE_SEARCHES: dict[str, tuple[EvidenceCategory, str, tuple[str, ...]]] = {
    "registry_events": (EvidenceCategory.REGISTRY, A.REGISTRY_EVENTS,
                        ("@timestamp", "event.code", "registry.path", "registry.data.strings", "process.name")),
    "script_block_events": (EvidenceCategory.SCRIPT_EXECUTION, A.SCRIPT_BLOCKS,
                            ("@timestamp", "event.code", "powershell.file.script_block_text", "user.name")),
    "service_task_events": (EvidenceCategory.SCHEDULED_TASKS_SERVICES, A.SERVICE_EVENTS,
                            ("@timestamp", "event.code", "winlog.event_data.ServiceName",
                             "winlog.event_data.TaskName", "user.name")),
}  # fmt: skip


def flatten_source(source: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Nested ECS ``_source`` -> dotted keys (``{"process": {"name": x}}`` -> ``{"process.name": x}``).

    Execution handlers read rows by their dotted ECS names; Elasticsearch returns the nesting it
    was indexed with (Winlogbeat and Elastic Agent index nested objects). Lists are kept as values.
    """
    flat: dict[str, Any] = {}
    for key, value in source.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_source(value, f"{name}."))
        else:
            flat[name] = value
    return flat


class ElasticTool(HttpTool):
    name = "elastic"

    def __init__(
        self,
        *,
        url: str,
        api_key: str,
        index_pattern: str = "logs-*",
        kibana_url: Optional[str] = None,
        timeout_seconds: float = 20.0,
        verify_tls: bool = True,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        kibana_transport: Optional[httpx.AsyncBaseTransport] = None,
        log_sources: Optional[LogSourceMap] = None,
    ) -> None:
        headers = {"Authorization": f"ApiKey {api_key}", "Content-Type": "application/json"}
        super().__init__(base_url=url, timeout_seconds=timeout_seconds, verify_tls=verify_tls, headers=headers,
                         transport=transport)  # fmt: skip
        self.index_pattern = index_pattern
        self.log_sources = log_sources or default_log_sources()
        self.kibana = (
            HttpChannel("kibana", base_url=kibana_url, timeout_seconds=timeout_seconds, verify_tls=verify_tls,
                        headers={**headers, "kbn-xsrf": "true"}, transport=kibana_transport)  # fmt: skip
            if kibana_url
            else None
        )

    @staticmethod
    def _body(filters: list[dict], should: list[dict], size: int, fields: tuple[str, ...]) -> dict[str, Any]:
        query: dict[str, Any] = {"bool": {"filter": filters}}
        if should:
            query["bool"]["should"] = should
            query["bool"]["minimum_should_match"] = 1
        return {"size": size, "sort": [{"@timestamp": {"order": "desc"}}], "query": query, "_source": list(fields)}

    async def search(self, filters: list[dict], should: list[dict], size: int, fields: tuple[str, ...]) -> list[dict]:
        body = self._body(filters, should, size, fields)
        reply = await self.request("POST", f"/{self.index_pattern}/_search", json=body)
        if reply.status_code == 404:
            return []  # index pattern matches nothing
        if reply.status_code >= 400:
            raise ToolError(f"elastic returned HTTP {reply.status_code}")
        hits = ((reply.json() or {}).get("hits") or {}).get("hits") or []
        return [flatten_source(h.get("_source") or {}) for h in hits[:size]]

    async def search_many(self, filter_sets: list[list[dict]], size: int,
                          fields: tuple[str, ...]) -> list[Optional[list[dict]]]:  # fmt: skip
        """Run several searches in one ``_msearch`` call. A search Elastic rejects yields ``None``."""
        if not filter_sets:
            return []
        header = json.dumps({"index": self.index_pattern})
        lines = [line for filters in filter_sets for line in (header, json.dumps(self._body(filters, [], size, fields)))]
        reply = await self.request("POST", "/_msearch", content="\n".join(lines) + "\n",
                                   headers={"Content-Type": "application/x-ndjson"})  # fmt: skip
        if reply.status_code >= 400:
            raise ToolError(f"elastic returned HTTP {reply.status_code}")
        responses = (reply.json() or {}).get("responses") or []
        if len(responses) != len(filter_sets):
            raise ToolError("elastic returned an unexpected number of search responses")
        results: list[Optional[list[dict]]] = []
        for response in responses:
            if response.get("error") and response.get("status") != 404:
                results.append(None)
                continue
            hits = (response.get("hits") or {}).get("hits") or []
            results.append([flatten_source(h.get("_source") or {}) for h in hits[:size]])
        return results

    async def run(
        self, spec: OperationSpec, target_type: str, target: str, context: dict[str, str], params: dict[str, Any]
    ) -> ToolResult:
        if spec.name == "isolate_host":
            return await self._isolate(spec, target_type, target)
        filters: list[dict] = [{"range": {"@timestamp": {"gte": f"now-{params['lookback_hours']}h"}}}]
        if spec.name in _EVENT_CODE_SEARCHES:
            evidence, artifact, fields = _EVENT_CODE_SEARCHES[spec.name]
            channels = [{"bool": {"filter": [{"term": {"winlog.channel": channel}}, {"terms": {"event.code": codes}}]}}
                        for channel, codes in self.log_sources.queries(evidence)]  # fmt: skip
            filters.append({"bool": {"should": channels, "minimum_should_match": 1}})
            targets = {A.HOST_ID: ("host.name",)}
        else:
            category, targets, artifact, fields = _SEARCHES[spec.name]
            if category:
                filters.append({"term": {"event.category": category}})
            if spec.name == "cloud_audit_events":
                filters.append({"exists": {"field": "cloud.provider"}})
        should = [{"term": {field: target}} for field in targets[target_type]]
        rows = await self.search(filters, should, params["limit"], fields)
        if not rows:
            return self.result(spec, target_type, status=ToolStatus.NO_DATA, message="no matching events")
        summary = f"Elastic returned {len(rows)} {spec.name.replace('_', ' ')} for the {target_type}"
        return self.result(spec, target_type, outputs={artifact: rows},
                           observations=[ToolObservation(category=c, description=summary) for c in spec.categories],
                           findings=[summary])  # fmt: skip

    async def _isolate(self, spec: OperationSpec, target_type: str, host: str) -> ToolResult:
        if self.kibana is None:
            raise ToolUnavailableError("Kibana is not configured (needed for Elastic Defend isolation)")
        reply = await self.kibana.request("GET", "/api/endpoint/metadata",
                                          params={"kuery": f'united.endpoint.host.hostname:"{host}"'})  # fmt: skip
        endpoints = (reply.json() or {}).get("data") or [] if reply.status_code == 200 else []
        if not endpoints:
            raise ToolError("no Elastic Defend endpoint found for that host")
        agent_id = ((endpoints[0].get("metadata") or {}).get("agent") or {}).get("id")
        if not agent_id:
            raise ToolError("Elastic Defend endpoint metadata has no agent id")
        reply = await self.kibana.request("POST", "/api/endpoint/action/isolate",
                                          json={"endpoint_ids": [agent_id], "comment": "CADE playbook (approved)"})  # fmt: skip
        if reply.status_code >= 400:
            raise ToolError(f"kibana refused the isolation request (HTTP {reply.status_code})")
        action_id = ((reply.json() or {}).get("data") or {}).get("id")
        state = {"action": "isolate_host", "agent_id": agent_id, "action_id": action_id, "simulated": False,
                 "state": "requested"}  # fmt: skip
        return self.result(spec, target_type, outputs={A.CONTAINMENT_STATUS: state},
                           findings=[f"Elastic Defend accepted isolation of endpoint {agent_id}"])  # fmt: skip

    async def health(self) -> dict[str, Any]:
        return {"status": "configured", "kibana": self.kibana is not None, "index_pattern": self.index_pattern}

    async def aclose(self) -> None:
        await super().aclose()
        if self.kibana is not None:
            await self.kibana.aclose()
