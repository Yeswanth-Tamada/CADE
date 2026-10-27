"""Response plans, approvals, analyst verifications, playbook runs and feedback (milestone M9)."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from app.schemas.enums import ExecutionMode
from app.schemas.incident import SafetyFlags
from app.schemas.library import OutcomeRecord
from app.schemas.playbook import Playbook


class PlanSource(str, Enum):
    LIBRARY = "library"  # reused from the playbook library
    SYNTHESIZED = "synthesized"  # generated and optimised by CADE for this incident


class PlanStatus(str, Enum):
    AWAITING_APPROVAL = "awaiting_approval"  # has destructive steps nobody has approved yet
    READY = "ready"
    EXECUTING = "executing"
    EXECUTED = "executed"


class NodeApproval(BaseModel):
    node_id: str
    action: str
    approved_by: str
    approved_at: AwareDatetime
    plan_hash: str  # the approval is void if the plan's structure changes
    comment: Optional[str] = None


class Verdict(str, Enum):
    CONFIRMED = "confirmed"
    NOT_CONFIRMED = "not_confirmed"


class AnalystVerification(BaseModel):
    node_id: str
    verdict: Verdict
    target_type: str
    targets: list[str] = Field(default_factory=list)  # the entity values the analyst confirmed
    verified_by: str
    verified_at: AwareDatetime
    notes: Optional[str] = None


class ResponsePlan(BaseModel):
    plan_id: str
    incident_id: str
    source: PlanSource
    playbook: Playbook
    structural_hash: str
    library_entry_id: Optional[str] = None
    routing_decision_id: Optional[int] = None
    fitness: Optional[dict[str, Any]] = None
    synthesis: Optional[dict[str, Any]] = None
    destructive_nodes: list[str] = Field(default_factory=list)
    verification_nodes: list[str] = Field(default_factory=list)  # analyst-performed verifications
    approvals: list[NodeApproval] = Field(default_factory=list)
    verifications: list[AnalystVerification] = Field(default_factory=list)
    status: PlanStatus
    safety: SafetyFlags = Field(default_factory=SafetyFlags)
    synthetic_basis: bool = False  # simulated evidence influenced routing / synthesis
    created_by: Optional[str] = None
    created_at: AwareDatetime
    updated_at: AwareDatetime

    def approval_for(self, node_id: str) -> Optional[NodeApproval]:
        return next((a for a in self.approvals if a.node_id == node_id and a.plan_hash == self.structural_hash), None)

    def verification_for(self, node_id: str) -> Optional[AnalystVerification]:
        matches = [v for v in self.verifications if v.node_id == node_id]
        return matches[-1] if matches else None


class ApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_ids: list[str] = Field(min_length=1, max_length=20)
    comment: Optional[str] = Field(default=None, max_length=1000)


class VerificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(max_length=40)
    verdict: Verdict
    targets: list[str] = Field(default_factory=list, max_length=20)
    notes: Optional[str] = Field(default=None, max_length=2000)


class NodeStatus(str, Enum):
    SUCCEEDED = "succeeded"
    NO_DATA = "no_data"  # the tool answered with nothing (not proof of absence)
    FAILED = "failed"
    MANUAL_TASK = "manual_task"  # handed to an analyst
    SKIPPED = "skipped"  # a precondition (input, approval, verification) was not met -- fail closed
    BLOCKED = "blocked"  # a safety gate refused the action (live-mode gate, separation of duties, ...)


class NodeResult(BaseModel):
    node_id: str
    action: str
    status: NodeStatus
    synthetic: bool = False  # produced from simulated data
    destructive: bool = False
    tool: Optional[str] = None
    targets: list[str] = Field(default_factory=list)
    verdict: Optional[Verdict] = None
    findings: list[str] = Field(default_factory=list)
    reason: Optional[str] = None
    duration_ms: float = 0.0


class RunStatus(str, Enum):
    COMPLETED = "completed"  # every node ran
    COMPLETED_WITH_ISSUES = "completed_with_issues"  # some nodes were skipped, blocked, failed or manual
    BLOCKED = "blocked"  # the run was refused before any node ran


class RunFeedback(BaseModel):
    resolved: bool
    findings_retained: bool
    notes: Optional[str] = None
    submitted_by: str
    submitted_at: AwareDatetime
    library_outcome: OutcomeRecord
    routing_reward: Optional[float] = None  # Eq. 9 reward credited to the routing decision (None: not credited)


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resolved: bool
    findings_retained: bool
    notes: Optional[str] = Field(default=None, max_length=2000)


class PlaybookRun(BaseModel):
    run_id: str
    plan_id: str
    incident_id: str
    mode: ExecutionMode
    status: RunStatus
    synthetic: bool  # any node result came from simulated data
    node_results: list[NodeResult] = Field(default_factory=list)
    report: list[str] = Field(default_factory=list)  # findings compiled by the terminal action
    # IC_exec: fraction of the intent's goals answered by an executed step that returned data (paper Eq. 9)
    execution_coverage: Optional[float] = Field(default=None, ge=0, le=1)
    block_reason: Optional[str] = None
    started_by: Optional[str] = None
    started_at: AwareDatetime
    completed_at: Optional[AwareDatetime] = None
    feedback: Optional[RunFeedback] = None
