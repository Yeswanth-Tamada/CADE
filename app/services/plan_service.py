"""Response plans: creation, approval, analyst verification, execution and feedback.

Security-relevant writes (approvals, verifications, execution attempts) are audited, and their
audit writes are fatal: if the audit record cannot be written the operation does not proceed.

Execution records IC_exec, the coverage the run achieved; feedback trains both the library
(success counts) and, with a :class:`~app.routing.learning.RoutingLearner`, the routing policy
(paper Eq. 9).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from app.core.config import Settings
from app.core.errors import (
    AuthorizationError,
    CADEError,
    ExecutionBlockedError,
    IncidentConflictError,
    InvalidIncidentInputError,
)
from app.core.threads import run_sync
from app.execution.executor import ExecutionPolicy, PlaybookExecutor
from app.library.library import PlaybookLibrary
from app.playbook.validator import PlaybookValidator
from app.routing.learning import RoutingLearner, response_chain
from app.routing.reward import execution_coverage
from app.schemas.enums import ExecutionMode
from app.schemas.incident import CanonicalIncident
from app.schemas.library import OutcomeRecord
from app.schemas.playbook import Playbook
from app.schemas.routing import RoutingAction
from app.schemas.response import (
    AnalystVerification,
    ApprovalRequest,
    FeedbackRequest,
    NodeApproval,
    NodeStatus,
    PlanSource,
    PlanStatus,
    PlaybookRun,
    ResponsePlan,
    RunFeedback,
    RunStatus,
    VerificationRequest,
)
from app.schemas.tools import ToolResult
from app.security.auth import Principal
from app.security.roles import Permission
from app.services.audit_service import AuditService
from app.services.evidence_service import EvidenceService
from app.storage.evidence_repository import EvidenceRepository
from app.storage.response_repository import ResponseRepository
from app.storage.routing_repository import RoutingRepository
from app.tools.artifacts import entity_artifacts


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PlanService:
    def __init__(
        self,
        responses: ResponseRepository,
        evidence_service: EvidenceService,
        evidence: EvidenceRepository,
        validator: PlaybookValidator,
        executor: PlaybookExecutor,
        library: PlaybookLibrary,
        audit: AuditService,
        settings: Settings,
        *,
        learner: Optional[RoutingLearner] = None,
        routing: Optional[RoutingRepository] = None,
    ) -> None:
        self.responses = responses
        self.evidence_service = evidence_service
        self.evidence = evidence
        self.validator = validator
        self.vocabulary = validator.vocabulary
        self.executor = executor
        self.library = library
        self.audit = audit
        self.settings = settings
        self.learner = learner
        self.routing = routing

    # --- creation & reads -------------------------------------------------------------------

    async def create_plan(
        self,
        incident: CanonicalIncident,
        playbook: Playbook,
        *,
        source: PlanSource,
        actor: Optional[Principal],
        library_entry_id: Optional[str] = None,
        routing_decision_id: Optional[int] = None,
        fitness: Optional[dict[str, Any]] = None,
        synthesis: Optional[dict[str, Any]] = None,
        synthetic_basis: bool = False,
    ) -> ResponsePlan:
        available = self._available(incident)
        report = self.validator.validate(playbook, available)
        if not report.valid:  # I1: an invalid or unsafe DAG never becomes a plan
            raise ExecutionBlockedError("the playbook is not a valid, safe DAG for this incident",
                                        incident_id=incident.incident_id,
                                        details=[{"code": v.code.value, "node_id": v.node_id} for v in report.violations[:20]])  # fmt: skip
        destructive = [n.node_id for n in playbook.nodes if self.vocabulary.get(n.action).destructive]
        verifications = [n.node_id for n in playbook.nodes
                         if self.vocabulary.get(n.action).verification and self.vocabulary.get(n.action).binding_kind == "analyst"]  # fmt: skip
        now = _utcnow()
        plan = ResponsePlan(
            plan_id=f"PLAN-{uuid.uuid4().hex[:16].upper()}", incident_id=incident.incident_id, source=source,
            playbook=playbook, structural_hash=playbook.structural_hash(), library_entry_id=library_entry_id,
            routing_decision_id=routing_decision_id, fitness=fitness, synthesis=synthesis,
            destructive_nodes=destructive, verification_nodes=verifications,
            status=PlanStatus.AWAITING_APPROVAL if destructive else PlanStatus.READY,
            safety=incident.safety, synthetic_basis=synthetic_basis,
            created_by=actor.analyst_id if actor else None, created_at=now, updated_at=now,
        )  # fmt: skip
        await run_sync(self.responses.create_plan, plan)
        await self.audit.record(actor, "plan.create", outcome="success", target_type="plan", target_id=plan.plan_id,
                                details={"incident_id": incident.incident_id, "source": source.value,
                                         "destructive_nodes": len(destructive)})  # fmt: skip
        return plan

    async def get_plan(self, plan_id: str) -> ResponsePlan:
        return await run_sync(self.responses.get_plan, plan_id)

    async def get_run(self, run_id: str) -> PlaybookRun:
        return await run_sync(self.responses.get_run, run_id)

    # --- approval & verification ---------------------------------------------------------------

    async def approve(self, plan_id: str, request: ApprovalRequest, actor: Principal) -> ResponsePlan:
        plan = await self.get_plan(plan_id)
        self._editable(plan)
        unknown = [n for n in request.node_ids if n not in plan.destructive_nodes]
        if unknown:
            raise InvalidIncidentInputError("only destructive steps of this plan can be approved",
                                            details=[{"node_id": n} for n in unknown])  # fmt: skip
        now = _utcnow()
        approvals = [a for a in plan.approvals if a.node_id not in request.node_ids]
        for node_id in request.node_ids:
            approvals.append(NodeApproval(node_id=node_id, action=plan.playbook.node(node_id).action,
                                          approved_by=actor.analyst_id, approved_at=now, plan_hash=plan.structural_hash,
                                          comment=request.comment))  # fmt: skip
        approved = {a.node_id for a in approvals if a.plan_hash == plan.structural_hash}
        status = PlanStatus.READY if set(plan.destructive_nodes) <= approved else PlanStatus.AWAITING_APPROVAL
        # Audit first: if the approval cannot be recorded in the audit log, it is not granted.
        await self.audit.record(actor, "plan.approve", outcome="success", target_type="plan", target_id=plan_id,
                                details={"node_ids": request.node_ids, "plan_hash": plan.structural_hash[:16]})  # fmt: skip
        updated = plan.model_copy(update={"approvals": approvals, "status": status})
        return await run_sync(self.responses.save_plan, updated,
                              expected_status={PlanStatus.AWAITING_APPROVAL, PlanStatus.READY})  # fmt: skip

    async def record_verification(self, plan_id: str, request: VerificationRequest, actor: Principal) -> ResponsePlan:
        plan = await self.get_plan(plan_id)
        self._editable(plan)
        if request.node_id not in plan.verification_nodes:
            raise InvalidIncidentInputError("this step is not an analyst verification of the plan")
        action = self.vocabulary.get(plan.playbook.node(request.node_id).action)
        target_type = next(t for t in action.inputs if t in action.verifies)
        analysis = await self.evidence_service.usable_analysis(plan.incident_id)
        known = {v.lower(): v for v in entity_artifacts(analysis.result.entities)[target_type]}
        targets = []
        for target in request.targets:
            if target.lower() not in known:
                raise InvalidIncidentInputError(f"'{target[:60]}' is not a {target_type} of this incident")
            targets.append(known[target.lower()])
        if request.verdict.value == "confirmed" and not targets:
            raise InvalidIncidentInputError("a confirmed verification must name the confirmed target(s)")
        verification = AnalystVerification(node_id=request.node_id, verdict=request.verdict, target_type=target_type,
                                           targets=targets, verified_by=actor.analyst_id, verified_at=_utcnow(),
                                           notes=request.notes)  # fmt: skip
        await self.audit.record(actor, "plan.verify", outcome=request.verdict.value, target_type="plan", target_id=plan_id,
                                details={"node_id": request.node_id, "targets": targets})  # fmt: skip
        updated = plan.model_copy(update={"verifications": plan.verifications + [verification]})
        return await run_sync(self.responses.save_plan, updated,
                              expected_status={PlanStatus.AWAITING_APPROVAL, PlanStatus.READY})  # fmt: skip

    # --- execution ----------------------------------------------------------------------------------

    async def execute(self, plan_id: str, actor: Principal) -> PlaybookRun:
        mode = self.settings.execution_mode
        needed = Permission.EXECUTE_LIVE if mode == ExecutionMode.LIVE else Permission.EXECUTE_SIMULATED
        if not actor.can(needed):
            await self.audit.record(actor, "execution.denied", outcome="denied", target_type="plan", target_id=plan_id,
                                    details={"mode": mode.value})  # fmt: skip
            raise AuthorizationError(f"executing in {mode.value} mode requires the '{needed.value}' permission")
        plan = await self.get_plan(plan_id)
        if plan.status in (PlanStatus.EXECUTING, PlanStatus.EXECUTED):
            raise ExecutionBlockedError("this plan has already been executed; create a new plan to run it again")
        analysis = await self.evidence_service.usable_analysis(plan.incident_id)
        incident = analysis.result
        report = self.validator.validate(plan.playbook, self._available(incident))
        if not report.valid or plan.structural_hash != plan.playbook.structural_hash():  # I1, re-checked at run time
            await self.audit.record(actor, "execution.blocked", outcome="blocked", target_type="plan", target_id=plan_id,
                                    details={"reason": "invalid_or_unsafe_dag"})  # fmt: skip
            raise ExecutionBlockedError("the plan's playbook is not a valid, safe DAG; it will not be executed")

        claimed = plan.model_copy(update={"status": PlanStatus.EXECUTING})
        await run_sync(self.responses.save_plan, claimed, expected_status={PlanStatus.AWAITING_APPROVAL, PlanStatus.READY})
        await self.audit.record(actor, "execution.start", outcome="started", target_type="plan", target_id=plan_id,
                                details={"mode": mode.value})  # fmt: skip
        run = PlaybookRun(run_id=f"RUN-{uuid.uuid4().hex[:16].upper()}", plan_id=plan_id, incident_id=plan.incident_id,
                          mode=mode, status=RunStatus.COMPLETED, synthetic=mode == ExecutionMode.SIMULATE,
                          started_by=actor.analyst_id, started_at=_utcnow())  # fmt: skip

        async def audit(action: str, outcome: str, details: dict[str, Any]) -> None:
            await self.audit.record(actor, action, outcome=outcome, target_type="run", target_id=run.run_id,
                                    details={**details, "plan_id": plan_id})  # fmt: skip

        async def record_invocation(result: ToolResult) -> None:
            await run_sync(self.evidence.record_invocation, result, purpose="execution", incident_id=plan.incident_id,
                           reference=run.run_id, actor=actor.analyst_id)  # fmt: skip

        policy = ExecutionPolicy(mode=mode, allow_destructive_live=self.settings.allow_destructive_live,
                                 require_separate_approver=self.settings.require_separate_approver,
                                 max_targets_per_node=self.settings.max_targets_per_node,
                                 max_destructive_targets=self.settings.max_destructive_targets)  # fmt: skip
        try:
            outcome = await self.executor.run(plan, incident, actor, policy, audit, record_invocation)
        except Exception as exc:  # noqa: BLE001 - the plan must never stay "executing"
            reason = exc.message if isinstance(exc, CADEError) else "internal error"
            run = run.model_copy(update={"status": RunStatus.BLOCKED, "block_reason": f"run aborted: {reason}",
                                         "completed_at": _utcnow()})  # fmt: skip
        else:
            clean = all(r.status in (NodeStatus.SUCCEEDED, NodeStatus.NO_DATA) for r in outcome.results)
            coverage = execution_coverage(plan.playbook, outcome.results, self.vocabulary.goals_for(incident.intent.label),
                                          self.vocabulary)  # fmt: skip
            run = run.model_copy(update={
                "node_results": outcome.results, "report": outcome.report, "synthetic": outcome.synthetic,
                "status": RunStatus.COMPLETED if clean else RunStatus.COMPLETED_WITH_ISSUES, "completed_at": _utcnow(),
                "execution_coverage": coverage,
            })  # fmt: skip
        await run_sync(self.responses.save_run, run)
        await run_sync(self.responses.save_plan, claimed.model_copy(update={"status": PlanStatus.EXECUTED}))
        await self.audit.record(actor, "execution.finish", outcome=run.status.value, target_type="run",
                                target_id=run.run_id, details={"plan_id": plan_id, "synthetic": run.synthetic})  # fmt: skip
        return run

    # --- feedback ------------------------------------------------------------------------------------

    async def feedback(self, run_id: str, request: FeedbackRequest, actor: Principal) -> PlaybookRun:
        run = await self.get_run(run_id)
        if run.feedback is not None:
            raise IncidentConflictError("feedback has already been recorded for this run")
        plan = await self.get_plan(run.plan_id)
        success = request.resolved and request.findings_retained
        if run.synthetic and not self.settings.library_learn_from_simulated:
            outcome = OutcomeRecord(action="not_stored", reason="simulated run: synthetic findings do not train the "
                                    "library (set CADE_LIBRARY_LEARN_FROM_SIMULATED=true to allow)")  # fmt: skip
        elif run.status == RunStatus.BLOCKED:
            outcome = OutcomeRecord(action="not_stored", reason="the run was blocked before it executed")
        else:
            metadata = {"incident_id": run.incident_id, "plan_id": plan.plan_id, "run_id": run.run_id,
                        "fitness": plan.fitness, "synthetic": run.synthetic}  # fmt: skip
            outcome = await run_sync(self.library.record_outcome, plan.playbook, success=success,
                                     source_entry_id=plan.library_entry_id if plan.source == PlanSource.LIBRARY else None,
                                     metadata=metadata)  # fmt: skip
        reward = await self._credit_routing(run, plan, retained=request.findings_retained)
        feedback = RunFeedback(resolved=request.resolved, findings_retained=request.findings_retained,
                               notes=request.notes, submitted_by=actor.analyst_id, submitted_at=_utcnow(),
                               library_outcome=outcome, routing_reward=reward)  # fmt: skip
        await self.audit.record(actor, "run.feedback", outcome=outcome.action, target_type="run", target_id=run_id,
                                details={"resolved": request.resolved, "retained": request.findings_retained,
                                         "library_entry": outcome.entry_id, "routing_reward": reward})  # fmt: skip
        return await run_sync(self.responses.save_run, run.model_copy(update={"feedback": feedback}))

    async def _credit_routing(self, run: PlaybookRun, plan: ResponsePlan, *, retained: bool) -> Optional[float]:
        """Train the router on this outcome (paper Eq. 9); ``None`` if nothing was credited."""
        if self.learner is None or self.routing is None or plan.routing_decision_id is None:
            return None
        if run.status == RunStatus.BLOCKED or run.completed_at is None:
            return None
        chain = response_chain(await run_sync(self.routing.list, plan.incident_id), plan.routing_decision_id)
        if not chain:
            return None
        # machine latency: routing and planning (or synthesis), then the run itself -- not the analysts' wait
        latency = ((plan.created_at - chain[0].created_at).total_seconds()
                   + (run.completed_at - run.started_at).total_seconds())  # fmt: skip
        arm = RoutingAction.REUSE if plan.source == PlanSource.LIBRARY else RoutingAction.SYNTHESIZE
        reward, stored = await run_sync(self.learner.credit, incident_id=plan.incident_id, run_id=run.run_id,
                                        chain=chain, terminal_arm=arm, retained=retained,
                                        execution_coverage=run.execution_coverage or 0.0,
                                        latency_seconds=max(0.0, latency), synthetic=run.synthetic)  # fmt: skip
        return round(reward.value, 6) if reward is not None and stored else None

    # --- helpers --------------------------------------------------------------------------------------

    @staticmethod
    def _available(incident: CanonicalIncident) -> frozenset[str]:
        return frozenset(k for k, v in entity_artifacts(incident.entities).items() if v)

    @staticmethod
    def _editable(plan: ResponsePlan) -> None:
        if plan.status in (PlanStatus.EXECUTING, PlanStatus.EXECUTED):
            raise ExecutionBlockedError("an executed plan can no longer be changed")
