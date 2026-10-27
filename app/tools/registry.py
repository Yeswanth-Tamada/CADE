"""Which adapter answers a tool call, and what happens when a real tool fails.

* ``simulate`` mode: every tool is answered by its :class:`SimulatedTool` (labelled synthetic),
  whatever real tools are configured.
* ``live`` mode: a tool is answered by its real adapter if configured. An unconfigured or failing
  real tool yields an ``error`` or ``manual_task`` result according to the failure policy --
  **never** a simulated result, so synthetic data can never pass for real evidence.

All tool calls go through :class:`ToolGateway`.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.core.config import Settings
from app.core.errors import ToolError, ToolInputError, ToolUnavailableError
from app.core.logging import get_logger, log_event
from app.enrichment.availability import AvailabilityProfile
from app.schemas.enums import ExecutionMode, ToolFailurePolicy, ToolStatus
from app.schemas.tools import ToolInfo, ToolRequest, ToolResult
from app.tools.base import SecurityTool
from app.tools.catalog import TOOL_DESCRIPTIONS, TOOLS, operations_for
from app.tools.simulated import SimulatedTool

logger = get_logger("tools")


class ToolRegistry:
    def __init__(
        self,
        mode: ExecutionMode,
        real: Optional[dict[str, SecurityTool]] = None,
        availability: Optional[AvailabilityProfile] = None,
    ) -> None:
        self.mode = mode
        self.real = dict(real or {})
        unknown = set(self.real) - set(TOOLS)
        if unknown:
            raise ValueError(f"unknown tools: {sorted(unknown)}")
        profile = availability.availability if availability else {}
        self.simulated = {name: SimulatedTool(name, profile) for name in TOOLS}

    def resolve(self, tool: str) -> Optional[SecurityTool]:
        if self.mode == ExecutionMode.SIMULATE:
            return self.simulated[tool]
        return self.real.get(tool)

    def info(self) -> list[ToolInfo]:
        def effective(name: str) -> str:
            if self.mode == ExecutionMode.SIMULATE:
                return "simulated"
            return "real" if name in self.real else "unavailable"

        return [
            ToolInfo(name=name, description=TOOL_DESCRIPTIONS[name], configured=name in self.real,
                     effective=effective(name), operations=list(operations_for(name).values()))
            for name in TOOLS
        ]  # fmt: skip

    async def aclose(self) -> None:
        for tool in self.real.values():
            await tool.aclose()


class ToolGateway:
    def __init__(self, registry: ToolRegistry, failure_policy: ToolFailurePolicy) -> None:
        self.registry = registry
        self.failure_policy = failure_policy

    @property
    def mode(self) -> ExecutionMode:
        return self.registry.mode

    async def invoke(self, tool: str, request: ToolRequest) -> ToolResult:
        """Run one tool operation. Input errors raise ``ToolInputError``; tool failures never raise."""
        if tool not in TOOLS:
            raise ToolInputError(f"unknown tool '{tool[:50]}'")
        adapter = self.registry.resolve(tool)
        if adapter is None:
            return self._failure(tool, request, ToolUnavailableError(f"{tool} is not configured for live mode"))
        try:
            result = await adapter.invoke(request)
        except ToolInputError:
            raise
        except ToolError as exc:
            return self._failure(tool, request, exc)
        log_event(logger, "tool call completed", operation=f"{tool}.{request.operation}", stage="tool",
                  outcome=result.status.value, duration_ms=result.duration_ms, synthetic=result.synthetic)  # fmt: skip
        return result

    def _failure(self, tool: str, request: ToolRequest, exc: ToolError) -> ToolResult:
        log_event(
            logger,
            "tool call failed",
            level=logging.WARNING,
            operation=f"{tool}.{request.operation}",
            stage="tool",
            outcome="failure",
            error_category=exc.category,
            policy=self.failure_policy.value,
        )
        manual = self.failure_policy == ToolFailurePolicy.MANUAL_TASK
        return ToolResult(
            tool=tool,
            operation=request.operation,
            target_type=request.target_type,
            status=ToolStatus.MANUAL_TASK if manual else ToolStatus.ERROR,
            synthetic=False,
            error_category=exc.category,
            message=exc.message[:500],
            findings=[f"Manual investigation task: perform {tool}.{request.operation} for the {request.target_type} "
                      f"by hand ({exc.category})"] if manual else [],
        )  # fmt: skip


def build_real_tools(settings: Settings) -> dict[str, SecurityTool]:
    """Instantiate the real adapters whose settings are complete. Secrets stay inside the adapters."""
    from app.tools.elastic import ElasticTool
    from app.tools.osquery import OsqueryTool
    from app.tools.sigma import SigmaTool
    from app.tools.virustotal import VirusTotalTool
    from app.tools.wazuh import WazuhTool
    from app.tools.yara import YaraTool

    common = {"timeout_seconds": settings.tool_timeout_seconds, "verify_tls": settings.tool_verify_tls}
    tools: dict[str, SecurityTool] = {}
    if settings.virustotal_api_key:
        tools["virustotal"] = VirusTotalTool(
            api_key=settings.virustotal_api_key.get_secret_value(), base_url=settings.virustotal_base_url,
            requests_per_minute=settings.virustotal_requests_per_minute, **common,
        )  # fmt: skip
    if settings.wazuh_api_url and settings.wazuh_username and settings.wazuh_password:
        tools["wazuh"] = WazuhTool(
            api_url=settings.wazuh_api_url,
            username=settings.wazuh_username,
            password=settings.wazuh_password.get_secret_value(),
            indexer_url=settings.wazuh_indexer_url,
            indexer_username=settings.wazuh_indexer_username,
            indexer_password=settings.wazuh_indexer_password.get_secret_value() if settings.wazuh_indexer_password else None,
            isolation_command=settings.wazuh_isolation_command,
            block_ip_command=settings.wazuh_block_ip_command,
            **common,
        )
    elastic = None
    if settings.elastic_url and settings.elastic_api_key:
        elastic = ElasticTool(url=settings.elastic_url, api_key=settings.elastic_api_key.get_secret_value(),
                              index_pattern=settings.elastic_index_pattern, kibana_url=settings.kibana_url, **common)  # fmt: skip
        tools["elastic"] = elastic
    if settings.osquery_binary:
        tools["osquery"] = OsqueryTool(binary=settings.osquery_binary, timeout_seconds=settings.tool_timeout_seconds)
    if settings.yara_binary and settings.yara_rules_path and settings.yara_scan_roots:
        tools["yara"] = YaraTool(binary=settings.yara_binary, rules_path=settings.resolve_path(settings.yara_rules_path),
                                 scan_roots=[settings.resolve_path(p) for p in settings.yara_scan_roots],
                                 timeout_seconds=settings.tool_timeout_seconds)  # fmt: skip
    if settings.sigma_rules_dir:
        tools["sigma"] = SigmaTool(rules_dir=settings.resolve_path(settings.sigma_rules_dir), elastic=elastic)
    return tools


def build_tool_gateway(
    settings: Settings, availability: AvailabilityProfile, real: Optional[dict[str, SecurityTool]] = None
) -> ToolGateway:
    real_tools = build_real_tools(settings) if real is None else real
    return ToolGateway(ToolRegistry(settings.execution_mode, real_tools, availability), settings.tool_failure_policy)
