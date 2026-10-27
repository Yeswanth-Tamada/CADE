"""Routing decisions (paper Section 4): reuse a stored playbook, request evidence, or synthesize."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import AwareDatetime, BaseModel, Field

from app.schemas.enums import EvidenceCategory


class RoutingAction(str, Enum):
    REUSE = "reuse"
    REQUEST_EVIDENCE = "request_evidence"
    SYNTHESIZE = "synthesize"


class RoutingTerms(BaseModel):
    """Every quantity of Eq. 1-5, so a decision can be audited and explained."""

    coverage: float = Field(ge=0, le=1)  # Cov(p*, A), Eq. 1
    counterfactual_coverage: float = Field(ge=0, le=1)  # Cov+(p*): coverage if requestable evidence arrived
    entropy: float = Field(ge=0, le=1)  # H, Eq. 2 (normalised)
    requestability: float = Field(ge=0, le=1)  # rho: mean availability of the missing critical categories
    unmet_fraction: float = Field(ge=0, le=1)  # M: fraction of critical requirements unmet
    smoothed_success: float = Field(ge=0, le=1)  # s-hat(p*) = (s + k) / (n + 2k)
    w: float
    k: float
    tau: float  # availability at or above which a missing category counts as obtainable for Cov+
    candidates_retrieved: int


class RoutingScores(BaseModel):
    reuse: float = Field(ge=0, le=1)  # Eq. 3
    request_evidence: float = Field(ge=0, le=1)  # Eq. 4
    synthesize: float = Field(ge=0, le=1)  # Eq. 5


class RoutingCandidate(BaseModel):
    entry_id: str
    name: str
    intent_label: str
    similarity: float
    score: float  # similarity after down-weighting de-prioritised entries (used for ranking and H)
    successes: int
    executions: int
    deprioritized: bool = False


class RoutingDecision(BaseModel):
    incident_id: str
    action: RoutingAction
    scores: RoutingScores
    terms: RoutingTerms
    top_candidate: Optional[RoutingCandidate] = None
    candidates: list[RoutingCandidate] = Field(default_factory=list)
    available_categories: list[EvidenceCategory] = Field(default_factory=list)  # A
    required_critical: list[EvidenceCategory] = Field(default_factory=list)  # R_c of p*
    required_supporting: list[EvidenceCategory] = Field(default_factory=list)  # R_s of p*
    missing_critical: list[EvidenceCategory] = Field(default_factory=list)  # R_c \ A
    requestable: list[EvidenceCategory] = Field(default_factory=list)  # missing critical with availability >= tau
    synthetic_basis: bool = False  # simulated evidence was counted in A
    round: int = Field(default=1, ge=1)
    rationale: str
    # Learned routing (paper Section 4.2): which policy chose, what the rule of Eq. 7 would have
    # chosen, the context vector x and each allowed arm's upper confidence bound (Eq. 8).
    policy: str = "rule"  # "rule" | "linucb"
    rule_action: Optional[RoutingAction] = None
    features: list[float] = Field(default_factory=list)
    ucb: dict[str, float] = Field(default_factory=dict)


class StoredRoutingDecision(RoutingDecision):
    decision_id: int
    created_at: AwareDatetime
    created_by: Optional[str] = None


class RoutingReward(BaseModel):
    """A reward credited to one routing decision (paper Eq. 9)."""

    decision_id: int
    incident_id: str
    run_id: str
    arm: RoutingAction
    terminal: bool  # False: an evidence request credited with the terminal outcome less its cost
    reward: float
    features: list[float]
    components: dict[str, Any] = Field(default_factory=dict)
    synthetic: bool = False
    created_at: AwareDatetime
