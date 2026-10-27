"""The learned routing policy (paper Sections 4.2-4.3).

:class:`RoutingLearner` owns the LinUCB policy and its training loop:

* **choosing** -- once ``bootstrap`` incidents have been rewarded, the bandit (Eq. 8) chooses the
  action; before that the hand-designed rule of Eq. 7 does, while every outcome is logged;
* **crediting** -- when the analyst reports on an executed run, the terminal decision's arm is
  updated with the Eq. 9 reward, and each evidence request that led to it with that reward less the
  request cost. Rewards are logged from the first incident, whichever policy chose.

Implementation choices: rewards are only credited for decisions that carry a context vector, a
run that was blocked before executing is not credited, and simulated runs do not train the router
unless ``learn_from_simulated`` (mirroring the library). ``L_max`` is the configured value, else
the 90th-percentile latency of the first ``bootstrap`` rewarded incidents (paper Section 6.2).

Until that L_max is known, rewards are *provisional*: they are logged (scored with the fallback
L_max and flagged ``provisional``) but not applied to the bandit. When the last bootstrap latency
arrives, every provisional reward is rescored with the final L_max and applied in the order it was
logged, so the bandit learns from exactly the paper's Eq. 9 reward. The rule chooses throughout the
bootstrap, so deferring these updates changes no routing decision.

State written by earlier versions is never changed silently. :meth:`RoutingLearner.audit` (run at
start-up; a finding is logged and reported by ``/health``) detects rewards that were applied with a
superseded L_max -- before 2026-10-26 bootstrap rewards were applied at once with the fallback -- and
stored arms that a replay of the stored rewards does not reproduce. :meth:`RoutingLearner.rebuild`
(``scripts/rebuild_bandit_state.py``, a dry run unless asked) rescores every stored reward under
the current protocol and rebuilds the arms from them.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from app.core.config import Settings
from app.core.logging import get_logger, log_event
from app.routing.bandit import LinUCBPolicy
from app.routing.reward import Reward, RewardWeights, percentile, terminal_reward
from app.schemas.routing import RoutingAction, RoutingReward, StoredRoutingDecision
from app.storage.bandit_repository import BanditRepository

logger = get_logger("routing")
_TOLERANCE = 1e-9


@dataclass(frozen=True)
class StateAudit:
    """How the stored learner state relates to the current reward protocol."""

    rewards: int  # stored rewards
    provisional: int  # logged but not yet applied (bootstrap in progress)
    stale: tuple[int, ...]  # decision ids applied with an L_max other than the protocol's
    latency_max: Optional[float]  # the protocol's L_max; None while the bootstrap latencies are incomplete
    consistent: bool  # the stored arms equal a replay of the applied stored rewards
    max_arm_difference: float

    @property
    def needs_rebuild(self) -> bool:
        return bool(self.stale) or not self.consistent

    def warning(self) -> Optional[str]:
        if not self.needs_rebuild:
            return None
        parts = []
        if self.stale:
            parts.append(f"{len(self.stale)} stored routing reward(s) were applied with a superseded L_max")
        if not self.consistent:
            parts.append(f"the stored bandit arms differ from a replay of the stored rewards (max diff {self.max_arm_difference:.3g})")
        return "; ".join(parts) + "; the learner keeps using them until scripts/rebuild_bandit_state.py is run"


@dataclass(frozen=True)
class RebuildPlan:
    """What :meth:`RoutingLearner.rebuild` changes: every stored reward under the current protocol, and the arms."""

    audit: StateAudit
    rewards: list[RoutingReward]  # the rescored rewards, oldest first
    changes: list[dict] = field(default_factory=list)  # per reward: old / new value, L_max and flag
    arms_before: dict[str, dict] = field(default_factory=dict)
    arms_after: dict[str, dict] = field(default_factory=dict)
    applied: bool = False


class RoutingLearner:
    def __init__(
        self,
        repository: BanditRepository,
        policy: LinUCBPolicy,
        *,
        enabled: bool = True,
        bootstrap: int = 20,
        size_scale: float = 50.0,
        weights: RewardWeights = RewardWeights(),
        latency_max_seconds: Optional[float] = None,
        latency_fallback_seconds: float = 600.0,
        learn_from_simulated: bool = False,
    ) -> None:
        self.repository = repository
        self.policy = policy
        self.enabled = enabled
        self.bootstrap = bootstrap
        self.size_scale = size_scale
        self.weights = weights
        self.latency_max_seconds = latency_max_seconds
        self.latency_fallback_seconds = latency_fallback_seconds
        self.learn_from_simulated = learn_from_simulated
        self.rewarded_incidents = 0
        self._latencies: list[float] = []
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls, settings: Settings, repository: BanditRepository) -> "RoutingLearner":
        learner = cls(
            repository, LinUCBPolicy(alpha=settings.bandit_alpha, ridge=settings.bandit_ridge_lambda),
            enabled=settings.routing_policy == "linucb", bootstrap=settings.bandit_bootstrap_incidents,
            size_scale=settings.bandit_library_size_scale,
            weights=RewardWeights(settings.reward_lambda_coverage, settings.reward_lambda_latency,
                                  settings.reward_lambda_request, settings.reward_request_cost),
            latency_max_seconds=settings.reward_latency_max_seconds,
            latency_fallback_seconds=settings.reward_latency_fallback_seconds,
            learn_from_simulated=settings.bandit_learn_from_simulated,
        )  # fmt: skip
        learner.load()
        return learner

    def load(self) -> None:
        self.policy.arms.update(self.repository.load_arms())
        self.rewarded_incidents = self.repository.rewarded_incidents()
        self._latencies = self.repository.first_latencies(self.bootstrap)
        if self.latency_max_final:  # e.g. stopped between the last bootstrap reward and its replay
            with self._lock:
                self._finalize_bootstrap()
        self.last_audit = self.audit()
        if self.last_audit.needs_rebuild:
            log_event(logger, "routing state needs a rebuild", operation="startup", stage="routing", outcome="stale",
                      stale_rewards=len(self.last_audit.stale), consistent=self.last_audit.consistent)  # fmt: skip

    def state_warning(self) -> Optional[str]:
        audit = getattr(self, "last_audit", None)
        return audit.warning() if audit else None

    # --- auditing and rebuilding stored state -----------------------------------------------------

    def _replay(self, rewards: list[RoutingReward]) -> LinUCBPolicy:
        policy = LinUCBPolicy(alpha=self.policy.alpha, ridge=self.policy.ridge, dimension=self.policy.dimension)
        for reward in rewards:
            policy.update(reward.arm, reward.features, reward.reward)
        return policy

    def audit(self) -> StateAudit:
        """Compare the stored rewards and arms with the current protocol (read-only)."""
        stored = self.repository.all()
        applied = [r for r in stored if not r.components.get("provisional")]
        final = self.latency_max() if self.latency_max_final else None
        stale = tuple(r.decision_id for r in applied
                      if final is None or abs(float(r.components.get("latency_max_seconds", -1.0)) - final) > _TOLERANCE)  # fmt: skip
        replay = self._replay(applied)
        difference = 0.0
        for arm, state in replay.arms.items():
            mine = self.policy.arms[arm]
            difference = max(difference, float(np.max(np.abs(np.array(mine.a) - np.array(state.a)))),
                             float(np.max(np.abs(np.array(mine.b) - np.array(state.b)))))  # fmt: skip
        pulls_match = all(self.policy.arms[arm].pulls == state.pulls for arm, state in replay.arms.items())
        return StateAudit(len(stored), len(stored) - len(applied), stale, final, difference <= 1e-6 and pulls_match, difference)

    def _rescore(self, reward: RoutingReward, latency_max: float, provisional: bool) -> RoutingReward:
        c = reward.components
        penalty = float(c.get("request_penalty", 0.0))
        terminal = terminal_reward(retained=bool(c["retained"]), coverage=float(c["execution_coverage"]),
                                   latency_seconds=float(c["latency_seconds"]), latency_max_seconds=latency_max,
                                   weights=self.weights)  # fmt: skip
        components = {**terminal.components(), "request_penalty": penalty, **({"provisional": True} if provisional else {})}
        return reward.model_copy(update={"reward": terminal.value - penalty, "components": components})

    def rebuild(self, *, apply: bool = False, force: bool = False) -> RebuildPlan:
        """Rescore every stored reward under the current protocol and rebuild the arms from them.

        While the bootstrap latencies are incomplete every reward becomes provisional and the arms return
        to their prior (nothing is applied before L_max is known); afterwards every reward is rescored
        with the final L_max and replayed oldest first. Nothing is written unless ``apply``; stored arms
        that a replay of the stored rewards does not reproduce are only overwritten with ``force``.
        """
        with self._lock:
            audit = self.audit()
            final = audit.latency_max
            stored = self.repository.all()
            rescored = [self._rescore(r, final if final is not None else self.latency_max(), provisional=final is None)
                        for r in stored]  # fmt: skip
            policy = self._replay([r for r in rescored if not r.components.get("provisional")])

            def summary(arms):
                return {a.value: {"pulls": st.pulls, "b_sum": float(np.sum(st.b))} for a, st in arms.items()}

            changes = [{"decision_id": old.decision_id, "arm": old.arm.value, "old_reward": old.reward,
                        "new_reward": new.reward, "old_latency_max": old.components.get("latency_max_seconds"),
                        "new_latency_max": new.components.get("latency_max_seconds"),
                        "old_provisional": bool(old.components.get("provisional")),
                        "new_provisional": bool(new.components.get("provisional"))}
                       for old, new in zip(stored, rescored)]  # fmt: skip
            plan = RebuildPlan(audit, rescored, changes, summary(self.policy.arms), summary(policy.arms))
            if not apply:
                return plan
            if not audit.consistent and not force:
                raise ValueError("the stored arms are not a replay of the stored rewards; rebuilding would discard "
                                 "state the reward log does not explain (use force to do it anyway)")  # fmt: skip
            self.repository.finalize(rescored, policy.arms)
            self.policy.arms.update(policy.arms)
            self.last_audit = self.audit()
        log_event(logger, "routing state rebuilt", operation="maintenance", stage="routing", outcome="rebuilt",
                  rewards=len(rescored), latency_max_seconds=final)  # fmt: skip
        return RebuildPlan(audit, rescored, changes, plan.arms_before, plan.arms_after, applied=True)

    @property
    def active(self) -> bool:
        """The bandit chooses (the bootstrap rule has been followed long enough)."""
        return self.enabled and self.rewarded_incidents >= self.bootstrap

    @property
    def latency_max_final(self) -> bool:
        """L_max no longer changes: it is configured, or every bootstrap latency has been observed."""
        return self.latency_max_seconds is not None or not self.bootstrap or len(self._latencies) >= self.bootstrap

    def latency_max(self) -> float:
        if self.latency_max_seconds is not None:
            return self.latency_max_seconds
        if self.bootstrap and len(self._latencies) >= self.bootstrap:
            return max(percentile(self._latencies, 90), 1e-3)
        return self.latency_fallback_seconds

    # --- crediting ----------------------------------------------------------------------------

    def credit(self, *, incident_id: str, run_id: str, chain: list[StoredRoutingDecision], terminal_arm: RoutingAction,
               retained: bool, execution_coverage: float, latency_seconds: float,
               synthetic: bool) -> tuple[Optional[Reward], list[RoutingReward]]:  # fmt: skip
        """Reward the decisions of one response (``chain``: its evidence requests, then the terminal decision).

        Returns the terminal reward and the stored rewards; during the bootstrap they are provisional
        until the last bootstrap latency arrives, when they are rescored (see the module docstring).
        """
        if not chain or (synthetic and not self.learn_from_simulated):
            return None, []
        now = datetime.now(timezone.utc)
        terminal, earlier = chain[-1], chain[:-1]
        with self._lock:
            provisional = not self.latency_max_final
            reward = terminal_reward(retained=retained, coverage=execution_coverage, latency_seconds=latency_seconds,
                                     latency_max_seconds=self.latency_max(), weights=self.weights)  # fmt: skip
            credits: list[tuple[StoredRoutingDecision, RoutingAction, Reward, bool]] = [(terminal, terminal_arm, reward, True)]
            credits += [(d, RoutingAction.REQUEST_EVIDENCE, reward.as_request(self.weights), False)
                        for d in earlier if d.action == RoutingAction.REQUEST_EVIDENCE]  # fmt: skip
            done = self.repository.credited([d.decision_id for d, *_ in credits])
            pending = [c for c in credits if c[0].features and c[0].decision_id not in done]
            if not pending:
                return reward, []
            if not provisional:
                for decision, arm, credit, _ in pending:
                    self.policy.update(arm, decision.features, credit.value)
            flag = {"provisional": True} if provisional else {}
            rewards = [RoutingReward(decision_id=d.decision_id, incident_id=incident_id, run_id=run_id, arm=arm,
                                     terminal=is_terminal, reward=credit.value, features=d.features,
                                     components={**credit.components(), **flag}, synthetic=synthetic, created_at=now)
                       for d, arm, credit, is_terminal in pending]  # fmt: skip
            stored = self.repository.credit(rewards, self.policy.arms)
            if any(r.terminal for r in stored):
                self.rewarded_incidents = self.repository.rewarded_incidents()
                if len(self._latencies) < self.bootstrap:
                    self._latencies.append(latency_seconds)
            if provisional and self.latency_max_final:
                final = self._finalize_bootstrap()
                stored = [final.get(r.decision_id, r) for r in stored]
                reward = terminal_reward(retained=retained, coverage=execution_coverage, latency_seconds=latency_seconds,
                                         latency_max_seconds=self.latency_max(), weights=self.weights)  # fmt: skip
        log_event(logger, "routing rewards credited", incident_id=incident_id, operation="feedback", stage="routing",
                  outcome="credited", rewards=len(stored), reward=round(reward.value, 4),
                  bandit_active=self.active, provisional=provisional and not self.latency_max_final)  # fmt: skip
        return reward, stored

    def _finalize_bootstrap(self) -> dict[int, RoutingReward]:
        """Rescore the provisional rewards with the final L_max and train the bandit on them, oldest first."""
        provisional = self.repository.provisional()
        if not provisional:
            return {}
        latency_max = self.latency_max()
        final = []
        for reward in provisional:
            rescored = self._rescore(reward, latency_max, provisional=False)
            self.policy.update(rescored.arm, rescored.features, rescored.reward)
            final.append(rescored)
        self.repository.finalize(final, self.policy.arms)
        log_event(logger, "routing bootstrap rewards finalized", operation="feedback", stage="routing",
                  outcome="finalized", rewards=len(final), latency_max_seconds=round(latency_max, 3))  # fmt: skip
        return {r.decision_id: r for r in final}


def response_chain(decisions: list[StoredRoutingDecision], terminal_id: int) -> list[StoredRoutingDecision]:
    """The decisions of the response that ended in ``terminal_id``: back to its round-1 decision."""
    ordered = sorted(decisions, key=lambda d: d.decision_id)
    index = next((i for i, d in enumerate(ordered) if d.decision_id == terminal_id), None)
    if index is None:
        return []
    start = index
    while start > 0 and ordered[start].round > 1 and ordered[start - 1].round < ordered[start].round:
        start -= 1
    return ordered[start: index + 1]
