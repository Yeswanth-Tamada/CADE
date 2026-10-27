"""Controlled playbook execution (paper Sections III, X; safety invariants I1-I7 of the plan).

Nodes run in topological order. Values flow along typed edges; a node whose inputs are not
available (an upstream step was skipped, failed or returned no data) is skipped with that reason
-- never run on guesses. Tool steps go through the :class:`ToolGateway` (simulated by default);
internal steps run locally; analyst steps become manual tasks.

A **destructive** node runs only if every gate passes, in this order, otherwise it is skipped or
blocked with an auditable reason:

1. an analyst with the approver role approved this node of this exact plan structure;
2. (separation of duties) the approver is not the person executing the run;
3. a verification ancestor *confirmed* the concrete target(s) -- analyst verdicts are recorded
   explicitly and never assumed, reputation verdicts must be "malicious";
4. no more confirmed targets than ``max_destructive_targets``;
5. in live mode, ``CADE_ALLOW_DESTRUCTIVE_LIVE=true`` **and** the executor holds ``execute_live``.

The attempt is written to the audit log *before* the action; if that write fails the run stops.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Optional

from app.core.errors import ToolInputError
from app.execution.internal import HANDLERS, ExecutionContext, HandlerResult
from app.playbook.graph import topological_order
from app.playbook.validator import PlaybookValidator
from app.playbook.vocabulary import ActionDefinition
from app.schemas.enums import ExecutionMode, ToolStatus
from app.schemas.incident import CanonicalIncident
from app.schemas.response import NodeResult, NodeStatus, ResponsePlan, Verdict
from app.schemas.tools import ToolRequest, ToolResult
from app.security.auth import Principal
from app.security.roles import Permission
from app.tools import artifacts as A
from app.tools.artifacts import entity_artifacts
from app.tools.registry import ToolGateway

AuditHook = Callable[[str, str, dict[str, Any]], Awaitable[None]]  # (action, outcome, details)
InvocationHook = Callable[[ToolResult], Awaitable[None]]

_INDICATOR_TYPES = {A.IP_ADDRESS, A.DOMAIN, A.FILE_HASH}


@dataclass
class _Value:
    items: list[Any]
    synthetic: bool


@dataclass
class ExecutionOutcome:
    results: list[NodeResult]
    report: list[str]
    synthetic: bool


@dataclass
class ExecutionPolicy:
    mode: ExecutionMode
    allow_destructive_live: bool = False
    require_separate_approver: bool = True
    max_targets_per_node: int = 3
    max_destructive_targets: int = 1


@dataclass
class _RunState:
    outputs: dict[str, dict[str, _Value]] = field(default_factory=dict)
    results: dict[str, NodeResult] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)


class PlaybookExecutor:
    def __init__(self, validator: PlaybookValidator, gateway: ToolGateway) -> None:
        self.validator = validator
        self.vocabulary = validator.vocabulary
        self.gateway = gateway

    async def run(
        self,
        plan: ResponsePlan,
        incident: CanonicalIncident,
        principal: Principal,
        policy: ExecutionPolicy,
        audit: AuditHook,
        record_invocation: InvocationHook,
    ) -> ExecutionOutcome:
        playbook = plan.playbook
        initial: dict[str, _Value] = {A.INCIDENT_CONTEXT: _Value([incident.incident_id], False)}
        for artifact, values in entity_artifacts(incident.entities).items():
            if values:
                initial[artifact] = _Value(list(values), False)
        state = _RunState()
        context = ExecutionContext(incident=incident, findings_so_far=state.findings)

        for node_id in topological_order(playbook) or []:
            node = playbook.node(node_id)
            action = self.vocabulary.get(node.action)
            started = time.perf_counter()
            inputs, missing = self._inputs(plan, node_id, action, initial, state)
            if action.terminal:
                # The report only compiles text, so it always runs -- documenting skipped steps too.
                missing = []
                inputs[A.FINDINGS] = _Value(list(state.findings), any(r.synthetic for r in state.results.values()))
            if missing:
                reason = f"input not available: {', '.join(missing)}"
                result = NodeResult(node_id=node_id, action=action.id, status=NodeStatus.SKIPPED,
                                    destructive=action.destructive, reason=reason,
                                    findings=[f"{action.name}: skipped -- {reason}"])  # fmt: skip
                if action.destructive:
                    await audit("execution.destructive.skipped", "skipped", {"node_id": node_id, "reason": result.reason})
                outputs: dict[str, list[Any]] = {}
            else:
                taint = any(v.synthetic for v in inputs.values())
                values = {k: v.items for k, v in inputs.items()}
                if action.destructive:
                    result, outputs = await self._destructive(plan, node_id, node.params, action, values, taint,
                                                              principal, policy, audit, record_invocation, state)  # fmt: skip
                elif action.binding_kind == "tool":
                    result, outputs = await self._tool(node_id, node.params, action, values, policy, record_invocation)
                elif action.binding_kind == "internal":
                    handled = HANDLERS[action.operation](context, values)
                    result, outputs = self._from_handler(node_id, action, handled), handled.outputs
                else:
                    result, outputs = self._analyst(plan, node_id, action, values)
                result = result.model_copy(update={"synthetic": result.synthetic or taint})
            result = result.model_copy(update={"duration_ms": (time.perf_counter() - started) * 1000})
            state.results[node_id] = result
            state.outputs[node_id] = {k: _Value(list(v), result.synthetic) for k, v in outputs.items() if v}
            state.findings.extend(result.findings)

        report = []
        for node_id, values in state.outputs.items():
            if A.INCIDENT_REPORT in values:
                report.extend(values[A.INCIDENT_REPORT].items)
        ordered = [state.results[n] for n in (topological_order(playbook) or []) if n in state.results]
        return ExecutionOutcome(ordered, report, any(r.synthetic for r in ordered))

    # --- inputs -------------------------------------------------------------------------------

    @staticmethod
    def _inputs(plan, node_id, action: ActionDefinition, initial, state) -> tuple[dict[str, _Value], list[str]]:
        inputs: dict[str, _Value] = {}
        missing: list[str] = []
        for artifact in action.inputs:
            sources = [e.source for e in plan.playbook.edges if e.target == node_id and e.artifact == artifact]
            items: list[Any] = []
            synthetic = False
            for source in sources:
                value = state.outputs.get(source, {}).get(artifact)
                if value is not None:
                    items.extend(value.items)
                    synthetic = synthetic or value.synthetic
            if not sources and artifact in initial:
                items, synthetic = list(initial[artifact].items), initial[artifact].synthetic
            if items:
                inputs[artifact] = _Value(list(dict.fromkeys(map(_hashable, items))) if artifact in A.ENTITY_ARTIFACTS
                                          else items, synthetic)  # fmt: skip
            else:
                missing.append(artifact)
        return inputs, missing

    # --- node kinds -----------------------------------------------------------------------------

    async def _tool(self, node_id, params, action: ActionDefinition, values, policy: ExecutionPolicy,
                    record_invocation: InvocationHook, targets: Optional[list[str]] = None):  # fmt: skip
        spec = action.operation_spec
        target_type = action.tool_target_type
        targets = targets if targets is not None else [str(v) for v in values.get(target_type, [])][: policy.max_targets_per_node]
        context = {artifact: str(values[artifact][0]) for artifact in spec.requires if values.get(artifact)}
        results: list[ToolResult] = []
        findings: list[str] = []
        for target in targets:
            request = ToolRequest(operation=spec.name, target_type=target_type, target=target, context=context, params=params)
            try:
                result = await self.gateway.invoke(spec.tool, request)
            except ToolInputError as exc:
                findings.append(f"{action.name}: input rejected ({exc.message})")
                continue
            await record_invocation(result)
            results.append(result)
            findings.extend(result.findings or ([result.message] if result.message else []))
        outputs: dict[str, list[Any]] = {}
        for result in results:
            if result.status == ToolStatus.SUCCESS:
                for artifact, value in result.outputs.items():
                    outputs.setdefault(artifact, []).extend(value if isinstance(value, list) else [value])
        statuses = {r.status for r in results}
        if ToolStatus.SUCCESS in statuses:
            status = NodeStatus.SUCCEEDED
        elif ToolStatus.MANUAL_TASK in statuses:
            status = NodeStatus.MANUAL_TASK
        elif statuses == {ToolStatus.NO_DATA}:
            status = NodeStatus.NO_DATA
        else:
            status = NodeStatus.FAILED
        findings = findings or [f"{action.name}: no result"]
        outputs[A.FINDINGS] = findings
        result = NodeResult(node_id=node_id, action=action.id, status=status, tool=spec.tool, targets=targets,
                            synthetic=any(r.synthetic for r in results), destructive=action.destructive, findings=findings,
                            reason=None if status == NodeStatus.SUCCEEDED else
                            "; ".join(sorted({r.error_category or r.status.value for r in results})) or "no valid target")  # fmt: skip
        return result, outputs

    def _analyst(self, plan: ResponsePlan, node_id, action: ActionDefinition, values):
        if action.verification:
            recorded = plan.verification_for(node_id)
            if recorded is None:
                finding = f"{action.name}: awaiting an analyst's verdict"
                pending = {"verdict": "pending", "target_type": None, "targets": []}
                return (NodeResult(node_id=node_id, action=action.id, status=NodeStatus.MANUAL_TASK, findings=[finding],
                                   reason="no verification has been recorded for this step"),
                        {A.VERIFICATION_RESULT: [pending], A.FINDINGS: [finding]})  # fmt: skip
            finding = (f"{action.name}: {recorded.verdict.value} by {recorded.verified_by} "
                       f"for {recorded.target_type} {', '.join(recorded.targets) or '-'}")  # fmt: skip
            outcome = {"verdict": recorded.verdict.value, "target_type": recorded.target_type, "targets": recorded.targets,
                       "verified_by": recorded.verified_by}  # fmt: skip
            return (NodeResult(node_id=node_id, action=action.id, status=NodeStatus.SUCCEEDED, verdict=recorded.verdict,
                               targets=recorded.targets, findings=[finding]),
                    {A.VERIFICATION_RESULT: [outcome], A.FINDINGS: [finding]})  # fmt: skip
        finding = f"Analyst task: {action.name} ({action.description})"
        return (NodeResult(node_id=node_id, action=action.id, status=NodeStatus.MANUAL_TASK, findings=[finding],
                           reason="performed by an analyst outside CADE"),
                {A.FINDINGS: [finding]})  # fmt: skip

    def _from_handler(self, node_id, action: ActionDefinition, handled: HandlerResult) -> NodeResult:
        return NodeResult(node_id=node_id, action=action.id, status=handled.status, verdict=handled.verdict,
                          targets=handled.targets, findings=handled.findings)  # fmt: skip

    # --- destructive gates ------------------------------------------------------------------------

    async def _destructive(self, plan: ResponsePlan, node_id, params, action: ActionDefinition, values, taint: bool,
                           principal: Principal, policy: ExecutionPolicy, audit: AuditHook,
                           record_invocation: InvocationHook, state: _RunState):  # fmt: skip
        def stop(status: NodeStatus, reason: str, targets: Optional[list[str]] = None):
            return NodeResult(node_id=node_id, action=action.id, status=status, destructive=True, targets=targets or [],
                              reason=reason, findings=[f"{action.name}: {status.value} -- {reason}"]), {}  # fmt: skip

        details: dict[str, Any] = {"node_id": node_id, "action": action.id, "mode": policy.mode.value}
        approval = plan.approval_for(node_id)
        if approval is None:
            await audit("execution.destructive.skipped", "skipped", {**details, "reason": "not_approved"})
            return stop(NodeStatus.SKIPPED, "this destructive step was not approved")
        details["approved_by"] = approval.approved_by
        if policy.require_separate_approver and approval.approved_by == principal.analyst_id:
            await audit("execution.destructive.blocked", "blocked", {**details, "reason": "self_approval"})
            return stop(NodeStatus.BLOCKED, "the approver of a destructive step cannot also execute it")

        targets = self._confirmed_targets(plan, node_id, action, values, state)
        if not targets:
            await audit("execution.destructive.skipped", "skipped", {**details, "reason": "not_verified"})
            return stop(NodeStatus.SKIPPED, "no verification step confirmed the target of this destructive step")
        if len(targets) > policy.max_destructive_targets:
            await audit("execution.destructive.blocked", "blocked", {**details, "reason": "too_many_targets",
                                                                     "targets": len(targets)})  # fmt: skip
            return stop(NodeStatus.BLOCKED, f"{len(targets)} confirmed targets exceed the limit of "
                                            f"{policy.max_destructive_targets}; act on them individually")  # fmt: skip
        if policy.mode == ExecutionMode.LIVE:
            if not policy.allow_destructive_live:
                await audit("execution.destructive.blocked", "blocked", {**details, "reason": "live_destructive_disabled"})
                return stop(NodeStatus.BLOCKED, "CADE_ALLOW_DESTRUCTIVE_LIVE is not enabled", targets)
            if not principal.can(Permission.EXECUTE_LIVE):
                await audit("execution.destructive.blocked", "blocked", {**details, "reason": "missing_live_permission"})
                return stop(NodeStatus.BLOCKED, "the executor lacks the execute_live permission", targets)

        # The attempt is recorded before anything happens; a failing audit write aborts the run.
        await audit("execution.destructive.attempt", "attempt", {**details, "targets": targets})
        if action.binding_kind == "tool":
            result, outputs = await self._tool(node_id, params, action, values, policy, record_invocation, targets=targets)
        else:
            finding = (f"Analyst task: {action.name} for {', '.join(targets)} (approved by {approval.approved_by}; "
                       "target verified)")  # fmt: skip
            result = NodeResult(node_id=node_id, action=action.id, status=NodeStatus.MANUAL_TASK, destructive=True,
                                targets=targets, findings=[finding], reason="performed by an analyst outside CADE")  # fmt: skip
            outputs = {A.CONTAINMENT_STATUS: [{"action": action.id, "targets": targets, "state": "manual_task"}],
                       A.FINDINGS: [finding]}  # fmt: skip
        result = result.model_copy(update={"destructive": True})
        await audit("execution.destructive.result", result.status.value,
                    {**details, "targets": targets, "synthetic": result.synthetic or taint})  # fmt: skip
        return result, outputs

    def _confirmed_targets(self, plan: ResponsePlan, node_id, action: ActionDefinition, values, state) -> list[str]:
        """Values of the action's target type that a verification ancestor confirmed."""
        candidates = [str(v) for v in values.get(action.target, [])]
        hosts = {str(v) for v in values.get(A.HOST_ID, [])}
        confirmed: set[str] = set()
        for ancestor in self.validator.verification_ancestors(plan.playbook, node_id):
            produced = state.outputs.get(ancestor, {}).get(A.VERIFICATION_RESULT)
            for outcome in produced.items if produced else []:
                if not isinstance(outcome, dict) or outcome.get("verdict") != Verdict.CONFIRMED.value:
                    continue
                kind, verified = outcome.get("target_type"), {str(t) for t in outcome.get("targets") or []}
                if kind == action.target or (kind == "indicator" and action.target in _INDICATOR_TYPES):
                    confirmed |= {c for c in candidates if c.lower() in {v.lower() for v in verified}}
                elif kind == A.HOST_ID and action.target != A.HOST_ID and hosts & verified:
                    confirmed |= set(candidates)  # host-scoped confirmation for a process/file on that host
        return sorted(confirmed)


def _hashable(value: Any) -> Any:
    return value if isinstance(value, (str, int, float, bool)) else str(value)
