"""The routing reward from execution telemetry (paper Section 4.3, Eq. 9).

    r(a) = 1_ret * [ lambda_c * IC_exec(p*) + lambda_l * (1 - l / L_max) ] - lambda_q * c_q(a)

* ``1_ret`` -- the analyst retained the run's findings (a gate, not a weighted term);
* ``IC_exec`` -- the coverage achieved when the playbook executed: the fraction of the intent's
  investigative goals for which at least one executed, non-destructive step returned data;
* ``l`` -- machine wall-clock time: routing and planning (or synthesis) plus the run itself,
  excluding the time the plan waited for analysts; ``L_max`` normalises it (clipped to [0, 1]);
* ``c_q(a)`` -- analyst-effort cost, non-zero only for the evidence-request arm.

The request arm does not end in an executed playbook: it is credited with the terminal action's
eventual reward less its own cost.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from app.playbook.vocabulary import IntentGoal, Vocabulary
from app.schemas.playbook import Playbook
from app.schemas.response import NodeResult, NodeStatus


@dataclass(frozen=True)
class RewardWeights:
    coverage: float = 0.6  # lambda_c
    latency: float = 0.3  # lambda_l
    request: float = 0.1  # lambda_q
    request_cost: float = 1.0  # c_q of the request arm


@dataclass(frozen=True)
class Reward:
    value: float
    retained: bool
    execution_coverage: float
    latency_seconds: float
    latency_max_seconds: float
    request_penalty: float = 0.0

    def as_request(self, weights: RewardWeights) -> "Reward":
        """The credit of an evidence request that led to this terminal outcome."""
        penalty = weights.request * weights.request_cost
        return Reward(self.value - penalty, self.retained, self.execution_coverage, self.latency_seconds,
                      self.latency_max_seconds, penalty)  # fmt: skip

    def components(self) -> dict[str, float | bool]:
        """The Eq. 9 terms, at full precision so a stored reward can be rescored exactly (bootstrap L_max)."""
        return {"retained": self.retained, "execution_coverage": self.execution_coverage,
                "latency_seconds": self.latency_seconds, "latency_max_seconds": self.latency_max_seconds,
                "request_penalty": self.request_penalty}  # fmt: skip


def execution_coverage(playbook: Playbook, results: Iterable[NodeResult], goals: Sequence[IntentGoal],
                       vocabulary: Vocabulary) -> float:  # fmt: skip
    """IC_exec: fraction of ``goals`` addressed by an executed, non-destructive step that returned data."""
    if not goals:
        return 1.0
    returned = {r.node_id for r in results if r.status == NodeStatus.SUCCEEDED and not r.destructive}
    actions = [a for a in (vocabulary.get(n.action) for n in playbook.nodes if n.node_id in returned)
               if a is not None and not a.destructive]  # fmt: skip
    return sum(1 for goal in goals if any(vocabulary.addresses(a, goal) for a in actions)) / len(goals)


def terminal_reward(*, retained: bool, coverage: float, latency_seconds: float, latency_max_seconds: float,
                    weights: RewardWeights) -> Reward:  # fmt: skip
    speed = 1.0 - min(1.0, max(0.0, latency_seconds) / latency_max_seconds)
    value = (weights.coverage * coverage + weights.latency * speed) if retained else 0.0
    return Reward(value, retained, coverage, latency_seconds, latency_max_seconds)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (``q`` in [0, 100]) of a non-empty sequence."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
