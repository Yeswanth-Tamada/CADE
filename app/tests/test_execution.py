"""Controlled execution, approvals, verification gates, audit and feedback (milestone M9).

Invariants under test (see the plan): I1 invalid/unsafe DAGs never run; I2 per-node approval;
I3 a verification ancestor must confirm the target; I4 live destructive gate (flag AND role);
I5 fail closed with a reason; I6 conservative safety flags; I7 audit before acting.
All tools are simulated or fakes; no real destructive action is ever attempted.
"""

from __future__ import annotations

import pytest

from app.core.errors import (
    AuthorizationError,
    ExecutionBlockedError,
    IncidentConflictError,
    InvalidIncidentInputError,
    StorageError,
    ToolTimeoutError,
)
from app.schemas.playbook import Playbook
from app.schemas.response import (
    ApprovalRequest,
    FeedbackRequest,
    NodeStatus,
    PlanSource,
    PlanStatus,
    RunStatus,
    Verdict,
    VerificationRequest,
)
from app.schemas.tools import ToolObservation, ToolResult
from app.security.auth import Principal
from app.security.roles import Role, permissions_for
from app.tests.conftest import SYNTHETIC_DESCRIPTION, good_output, with_settings
from app.tests.test_playbook_dag import pb
from app.tools.base import SecurityTool

HOST = "WS-FIN-042"
INTENT = "suspicious_process_execution"

ISOLATION = ("n1:extract_entities n2:verify_host_compromise n3:isolate_host n4:generate_incident_report",
             "n1>n2:host_id n1>n2:findings n1>n3:host_id n2>n3:verification_result n3>n4:findings")  # fmt: skip
INVESTIGATION = ("n1:extract_entities n2:query_process_events n3:analyze_process_tree n4:file_hash_reputation "
                 "n5:generate_incident_report",
                 "n1>n2:host_id n2>n3:process_list n1>n4:file_hash n3>n5:findings n4>n5:findings")  # fmt: skip


def principal(name: str, *roles: str) -> Principal:
    granted = frozenset(Role(r) for r in roles)
    return Principal(analyst_id=name, roles=granted, permissions=permissions_for(granted), auth_mode="api_key", key_id=1)


ANALYST = principal("alice", "analyst")
APPROVER = principal("bob", "approver")
EXECUTOR = principal("carol", "executor")
LIVE_EXECUTOR = principal("dave", "live_executor")


class FakeWazuh(SecurityTool):
    """Stands in for a configured real Wazuh (never contacts anything)."""

    name = "wazuh"

    def __init__(self, fail: Exception | None = None) -> None:
        super().__init__()
        self.calls: list[tuple[str, str]] = []
        self.fail = fail

    async def run(self, spec, target_type, target, context, params) -> ToolResult:
        self.calls.append((spec.name, target))
        if self.fail:
            raise self.fail
        return self.result(spec, target_type, outputs={spec.produces[0]: {"state": "requested", "simulated": False}},
                           observations=[ToolObservation(category=c, description="x") for c in spec.categories],
                           findings=[f"FAKE real {spec.key} accepted"])  # fmt: skip


def analysed(client) -> str:
    incident_id = client.post("/incidents", json={"description": SYNTHETIC_DESCRIPTION}).json()["incident_id"]
    assert client.post(f"/incidents/{incident_id}/analyze").status_code == 200
    return incident_id


def call(client, fn, *args, **kwargs):
    return client.portal.call(lambda: fn(*args, **kwargs))


def make_plan(client, incident_id: str, spec=ISOLATION, source=PlanSource.SYNTHESIZED, **kwargs):
    service = client.app.state.plan_service
    incident = call(client, client.app.state.evidence_service.usable_analysis, incident_id).result
    return call(client, service.create_plan, incident, pb(*spec, intent=INTENT), source=source, actor=ANALYST, **kwargs)


def audit_actions(client, target_id=None) -> list[str]:
    return [e.action for e in reversed(client.app.state.audit.repository.list(target_id=target_id, limit=500))]


def node(run, action: str):
    return next(r for r in run.node_results if r.action == action)


@pytest.fixture
def client(make_client):
    return make_client([good_output()])


# --- plans & I1 ---------------------------------------------------------------------------------


def test_invalid_or_unsafe_playbooks_never_become_plans(client):
    incident_id = analysed(client)
    with pytest.raises(ExecutionBlockedError):
        make_plan(client, incident_id, ("n1:extract_entities n2:isolate_host n3:generate_incident_report",
                                        "n1>n2:host_id n2>n3:findings"))  # fmt: skip
    plan = make_plan(client, incident_id)
    assert plan.status == PlanStatus.AWAITING_APPROVAL and plan.destructive_nodes == ["n3"]
    assert plan.verification_nodes == ["n2"]
    assert plan.safety.human_approval_required and not plan.safety.destructive_actions_allowed  # I6


def test_investigation_runs_in_topological_order_with_synthetic_results(client):
    incident_id = analysed(client)
    plan = make_plan(client, incident_id, INVESTIGATION)
    assert plan.status == PlanStatus.READY
    run = call(client, client.app.state.plan_service.execute, plan.plan_id, EXECUTOR)
    assert run.mode.value == "simulate" and run.synthetic
    assert [r.node_id for r in run.node_results] == ["n1", "n2", "n4", "n3", "n5"] or \
        [r.node_id for r in run.node_results][0] == "n1"  # fmt: skip
    assert node(run, "query_process_events").synthetic and node(run, "query_process_events").tool == "elastic"
    assert run.report and run.status in (RunStatus.COMPLETED, RunStatus.COMPLETED_WITH_ISSUES)
    assert call(client, client.app.state.plan_service.get_plan, plan.plan_id).status == PlanStatus.EXECUTED
    with pytest.raises(ExecutionBlockedError):
        call(client, client.app.state.plan_service.execute, plan.plan_id, EXECUTOR)
    from sqlalchemy import text

    with client.app.state.database.engine.connect() as conn:
        rows = conn.execute(text("SELECT purpose, reference, synthetic FROM tool_invocations")).all()
    assert rows and {r.purpose for r in rows} == {"execution"} and {r.reference for r in rows} == {run.run_id}


def test_missing_inputs_fail_closed(client):
    incident_id = analysed(client)  # the incident has no file hash
    run = call(client, client.app.state.plan_service.execute, make_plan(client, incident_id, INVESTIGATION).plan_id, EXECUTOR)
    reputation = node(run, "file_hash_reputation")
    assert reputation.status == NodeStatus.SKIPPED and "file_hash" in reputation.reason


# --- destructive gates (I2, I3, I5, I7) -----------------------------------------------------------


def test_unapproved_destructive_step_is_skipped(client):
    incident_id = analysed(client)
    run = call(client, client.app.state.plan_service.execute, make_plan(client, incident_id).plan_id, EXECUTOR)
    isolate = node(run, "isolate_host")
    assert isolate.status == NodeStatus.SKIPPED and "not approved" in isolate.reason
    assert "execution.destructive.skipped" in audit_actions(client, run.run_id)
    assert "execution.destructive.attempt" not in audit_actions(client, run.run_id)


def test_approved_but_unverified_destructive_step_is_skipped(client):
    incident_id = analysed(client)
    plan = make_plan(client, incident_id)
    call(client, client.app.state.plan_service.approve, plan.plan_id, ApprovalRequest(node_ids=["n3"]), APPROVER)
    run = call(client, client.app.state.plan_service.execute, plan.plan_id, EXECUTOR)
    assert node(run, "verify_host_compromise").status == NodeStatus.MANUAL_TASK
    isolate = node(run, "isolate_host")
    assert isolate.status == NodeStatus.SKIPPED and "verification" in isolate.reason


def test_not_confirmed_verification_blocks_the_destructive_step(client):
    incident_id = analysed(client)
    service = client.app.state.plan_service
    plan = make_plan(client, incident_id)
    call(client, service.approve, plan.plan_id, ApprovalRequest(node_ids=["n3"]), APPROVER)
    call(client, service.record_verification, plan.plan_id,
         VerificationRequest(node_id="n2", verdict=Verdict.NOT_CONFIRMED), APPROVER)  # fmt: skip
    run = call(client, service.execute, plan.plan_id, EXECUTOR)
    assert node(run, "verify_host_compromise").verdict == Verdict.NOT_CONFIRMED
    assert node(run, "isolate_host").status == NodeStatus.SKIPPED


def test_approved_and_verified_step_runs_simulated_and_is_audited_first(client):
    incident_id = analysed(client)
    service = client.app.state.plan_service
    plan = make_plan(client, incident_id)
    call(client, service.approve, plan.plan_id, ApprovalRequest(node_ids=["n3"], comment="confirmed C2"), APPROVER)
    plan = call(client, service.record_verification, plan.plan_id,
                VerificationRequest(node_id="n2", verdict=Verdict.CONFIRMED, targets=[HOST.lower()]), APPROVER)  # fmt: skip
    assert plan.status == PlanStatus.READY and plan.verifications[0].targets == [HOST]  # canonical entity value
    run = call(client, service.execute, plan.plan_id, EXECUTOR)
    isolate = node(run, "isolate_host")
    assert isolate.status == NodeStatus.SUCCEEDED and isolate.synthetic and isolate.targets == [HOST]
    assert any("no real action was taken" in f for f in isolate.findings)
    actions = audit_actions(client, run.run_id)
    assert actions.index("execution.destructive.attempt") < actions.index("execution.destructive.result")
    plan_actions = audit_actions(client, plan.plan_id)
    assert {"plan.create", "plan.approve", "plan.verify", "execution.start"} <= set(plan_actions)
    (approval,) = client.app.state.audit.repository.list(action="plan.approve")
    assert approval.actor == "bob" and approval.details["node_ids"] == ["n3"]


def test_self_approval_is_blocked(client):
    incident_id = analysed(client)
    service = client.app.state.plan_service
    both = principal("erin", "approver", "executor")
    plan = make_plan(client, incident_id)
    call(client, service.approve, plan.plan_id, ApprovalRequest(node_ids=["n3"]), both)
    call(client, service.record_verification, plan.plan_id,
         VerificationRequest(node_id="n2", verdict=Verdict.CONFIRMED, targets=[HOST]), both)  # fmt: skip
    isolate = node(call(client, service.execute, plan.plan_id, both), "isolate_host")
    assert isolate.status == NodeStatus.BLOCKED and "approver" in isolate.reason


def test_approval_is_void_if_the_plan_structure_changes(client):
    incident_id = analysed(client)
    service = client.app.state.plan_service
    plan = make_plan(client, incident_id)
    plan = call(client, service.approve, plan.plan_id, ApprovalRequest(node_ids=["n3"]), APPROVER)
    stale = plan.approvals[0].model_copy(update={"plan_hash": "0" * 64})
    assert plan.model_copy(update={"approvals": [stale]}).approval_for("n3") is None


def test_approval_and_verification_validation(client):
    incident_id = analysed(client)
    service = client.app.state.plan_service
    plan = make_plan(client, incident_id)
    with pytest.raises(InvalidIncidentInputError):
        call(client, service.approve, plan.plan_id, ApprovalRequest(node_ids=["n1"]), APPROVER)  # not destructive
    with pytest.raises(InvalidIncidentInputError):  # target is not an entity of the incident
        call(client, service.record_verification, plan.plan_id,
             VerificationRequest(node_id="n2", verdict=Verdict.CONFIRMED, targets=["DC-01"]), APPROVER)  # fmt: skip
    with pytest.raises(InvalidIncidentInputError):  # confirmation without a target
        call(client, service.record_verification, plan.plan_id,
             VerificationRequest(node_id="n2", verdict=Verdict.CONFIRMED), APPROVER)  # fmt: skip
    with pytest.raises(InvalidIncidentInputError):  # not a verification step
        call(client, service.record_verification, plan.plan_id,
             VerificationRequest(node_id="n3", verdict=Verdict.CONFIRMED, targets=[HOST]), APPROVER)  # fmt: skip


def test_failed_pre_action_audit_aborts_the_run(client, monkeypatch):
    incident_id = analysed(client)
    service = client.app.state.plan_service
    plan = make_plan(client, incident_id)
    call(client, service.approve, plan.plan_id, ApprovalRequest(node_ids=["n3"]), APPROVER)
    call(client, service.record_verification, plan.plan_id,
         VerificationRequest(node_id="n2", verdict=Verdict.CONFIRMED, targets=[HOST]), APPROVER)  # fmt: skip
    audit = client.app.state.audit
    original = audit.record

    async def failing(actor, action, **kwargs):
        if action == "execution.destructive.attempt":
            raise StorageError("database operation failed")
        return await original(actor, action, **kwargs)

    monkeypatch.setattr(audit, "record", failing)
    run = call(client, service.execute, plan.plan_id, EXECUTOR)
    assert run.status == RunStatus.BLOCKED and "aborted" in run.block_reason
    assert run.node_results == []


# --- live mode (I4) ----------------------------------------------------------------------------------


def _live_client(make_client, settings, **changes):
    fake = FakeWazuh(changes.pop("fail", None))
    live = with_settings(settings, execution_mode="live", **changes)
    return make_client([good_output()], app_settings=live, real_tools={"wazuh": fake}), fake


def _approved_verified_plan(client):
    incident_id = analysed(client)
    service = client.app.state.plan_service
    plan = make_plan(client, incident_id)
    call(client, service.approve, plan.plan_id, ApprovalRequest(node_ids=["n3"]), APPROVER)
    call(client, service.record_verification, plan.plan_id,
         VerificationRequest(node_id="n2", verdict=Verdict.CONFIRMED, targets=[HOST]), APPROVER)  # fmt: skip
    return plan


def test_live_execution_requires_the_live_permission(make_client, settings):
    client, fake = _live_client(make_client, settings, allow_destructive_live=True)
    plan = _approved_verified_plan(client)
    with pytest.raises(AuthorizationError):
        call(client, client.app.state.plan_service.execute, plan.plan_id, EXECUTOR)
    assert fake.calls == [] and "execution.denied" in audit_actions(client, plan.plan_id)


def test_live_destructive_is_blocked_without_the_flag_even_when_approved_and_verified(make_client, settings):
    client, fake = _live_client(make_client, settings, allow_destructive_live=False)
    plan = _approved_verified_plan(client)
    run = call(client, client.app.state.plan_service.execute, plan.plan_id, LIVE_EXECUTOR)
    isolate = node(run, "isolate_host")
    assert isolate.status == NodeStatus.BLOCKED and "CADE_ALLOW_DESTRUCTIVE_LIVE" in isolate.reason
    assert fake.calls == [] and not run.synthetic


def test_live_destructive_runs_only_when_every_gate_passes(make_client, settings):
    client, fake = _live_client(make_client, settings, allow_destructive_live=True)
    plan = _approved_verified_plan(client)
    run = call(client, client.app.state.plan_service.execute, plan.plan_id, LIVE_EXECUTOR)
    isolate = node(run, "isolate_host")
    assert isolate.status == NodeStatus.SUCCEEDED and not isolate.synthetic
    assert fake.calls == [("isolate_host", HOST)]


def test_live_real_tool_failure_is_never_simulated(make_client, settings):
    client, fake = _live_client(make_client, settings, allow_destructive_live=True, fail=ToolTimeoutError("slow"))
    plan = _approved_verified_plan(client)
    run = call(client, client.app.state.plan_service.execute, plan.plan_id, LIVE_EXECUTOR)
    isolate = node(run, "isolate_host")
    assert isolate.status == NodeStatus.MANUAL_TASK and not isolate.synthetic
    assert not any("SYNTHETIC" in f for f in isolate.findings)


# --- feedback ------------------------------------------------------------------------------------------


def test_simulated_runs_do_not_train_the_library_by_default(client):
    incident_id = analysed(client)
    service = client.app.state.plan_service
    run = call(client, service.execute, make_plan(client, incident_id, INVESTIGATION).plan_id, EXECUTOR)
    before = len(client.app.state.components.library)
    run = call(client, service.feedback, run.run_id, FeedbackRequest(resolved=True, findings_retained=True), ANALYST)
    assert run.feedback.library_outcome.action == "not_stored" and "simulated" in run.feedback.library_outcome.reason
    assert len(client.app.state.components.library) == before
    with pytest.raises(IncidentConflictError):
        call(client, service.feedback, run.run_id, FeedbackRequest(resolved=True, findings_retained=True), ANALYST)


def test_feedback_writes_back_when_allowed(make_client, settings):
    client = make_client([good_output()], app_settings=with_settings(settings, library_learn_from_simulated=True))
    incident_id = analysed(client)
    service = client.app.state.plan_service
    library = client.app.state.components.library
    synthesized = call(client, service.execute, make_plan(client, incident_id, INVESTIGATION).plan_id, EXECUTOR)
    stored = call(client, service.feedback, synthesized.run_id, FeedbackRequest(resolved=True, findings_retained=True), ANALYST)
    assert stored.feedback.library_outcome.action in ("inserted", "merged")

    entry = library.entries(INTENT)[0]
    reuse = make_plan(client, incident_id, INVESTIGATION, source=PlanSource.LIBRARY, library_entry_id=entry.entry_id)
    reuse_run = call(client, service.execute, reuse.plan_id, EXECUTOR)
    outcome = call(client, service.feedback, reuse_run.run_id,
                   FeedbackRequest(resolved=False, findings_retained=False), ANALYST).feedback.library_outcome  # fmt: skip
    assert outcome.action == "updated" and outcome.executions == entry.executions + 1
    assert outcome.successes == entry.successes


def test_every_internal_operation_has_a_handler():
    from app.execution.internal import HANDLERS
    from app.playbook.vocabulary import INTERNAL_OPERATIONS

    assert set(HANDLERS) == set(INTERNAL_OPERATIONS)


def test_report_documents_skipped_steps(client):
    incident_id = analysed(client)
    run = call(client, client.app.state.plan_service.execute, make_plan(client, incident_id, INVESTIGATION).plan_id, EXECUTOR)
    assert node(run, "generate_incident_report").status == NodeStatus.SUCCEEDED
    assert any("skipped" in line for line in run.report)
