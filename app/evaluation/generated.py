"""Option A: the paper-matched routing benchmark with REAL synthesized playbooks (paper 968 Sections 6.1-6.4).

Same 96 scenarios, protocol and LinUCB as :class:`app.evaluation.paper968.PaperEnv`, with one change: when the
terminal action is GENERATE, CADE really synthesizes a playbook with the configured models (pool B, adaptive K,
evolution, LLM judge) and the reward is computed from *that* playbook, instead of assuming synthesis answers every
goal in hand.

What is real and what is not (written into every report):

* real: scenarios from Splunk Attack Data metadata, the router (rule and LinUCB), retrieval (paper embedder and
  FAISS), the synthesized playbooks (local generator and judge), synthesis latency (measured on this machine);
* SIMULATED: execution. A goal (a ground-truth category) counts as answered when the returned playbook has a step
  that reads that category, the evidence is in hand and the playbook investigates the scenario's tactic; no logs
  are loaded and no query runs. Reuse and request latencies stay the paper's Fig. 4b values (nothing executes).

Synthesis is cached per (scenario, evidence in hand), so each needed playbook is written once and every ordering
reuses it. With ``lazy_alternatives`` (default) the reward-optimality diagnostics (``alternatives``) only use
synthesized playbooks that already exist and fall back to the simulated stand-in otherwise; how many did is
reported (``estimated_alternatives``). Training rewards and accuracy never use the stand-in for GENERATE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from app.evaluation.growth import queried
from app.evaluation.paper968 import PaperEnv, PaperScenario
from app.evaluation.protocol import GENERATE, REQUEST, REUSE, Outcome
from app.evaluation.quality import QualityRun, state_key
from app.routing.learning import RoutingLearner
from app.schemas.enums import EvidenceCategory as C
from app.schemas.playbook import Playbook
from app.schemas.routing import RoutingAction, RoutingDecision

EXECUTION_NOTE = ("SIMULATED execution on REAL synthesized playbooks: a goal is answered when the returned playbook "
                  "reads that category, the evidence is in hand and it investigates the scenario's tactic; no logs, "
                  "no query runs")  # fmt: skip


@dataclass
class GeneratedPaperEnv(PaperEnv):
    quality: Optional[QualityRun] = None  # set after construction (it needs this env for retrieval)
    lazy_alternatives: bool = True
    estimated_alternatives: int = 0
    synthesized: dict[str, float] = field(default_factory=dict)  # "<scenario>|<state>" -> synthesis seconds

    def _goals(self, scenario: PaperScenario) -> frozenset[C]:
        return scenario.critical if self.goals == "critical" else scenario.recording.categories

    def check(self, scenario: PaperScenario, playbook: Playbook, in_hand: frozenset[C]) -> tuple[float, bool]:
        goals = self._goals(scenario)
        answered = goals & queried(playbook, self.vocabulary) & in_hand
        coverage = len(answered) / len(goals) if goals else 1.0
        return coverage, scenario.critical <= answered and coverage >= 0.8

    async def synthesize(self, scenario: PaperScenario, in_hand: frozenset[C]) -> tuple[float, bool, float, dict]:
        """(IC_exec, retained, synthesis seconds, diagnostics) for the playbook CADE synthesizes for this state."""
        record = await self.quality.produce("generation_only", scenario, in_hand)
        self.synthesized[f"{scenario.name}|{record.state}"] = record.latency_s
        if record.failed or not record.playbook:
            return 0.0, False, record.latency_s, {"synthesis_failed": True}
        coverage, retained = self.check(scenario, Playbook.model_validate(record.playbook), in_hand)
        return coverage, retained, record.latency_s, {"synthesis_fitness": round(record.fitness, 4)}

    def path_latency(self, first: RoutingAction, terminal: RoutingAction, synthesis_seconds: Optional[float]) -> float:
        if terminal == GENERATE and synthesis_seconds is not None:
            return synthesis_seconds + ((self.latency.request - self.latency.reuse) if first == REQUEST else 0.0)
        return self.latency.of(first, terminal)

    async def run_terminal(self, scenario: PaperScenario, first: RoutingDecision, terminal: RoutingDecision,
                           in_hand: frozenset[C]) -> tuple[float, bool, float, dict]:  # fmt: skip
        if terminal.action == GENERATE:
            coverage, retained, seconds, extra = await self.synthesize(scenario, in_hand)
            return coverage, retained, self.path_latency(first.action, terminal.action, seconds), extra
        coverage, retained = self.execute(scenario, terminal.action, terminal, in_hand)
        return coverage, retained, self.latency.of(first.action, terminal.action), {}

    async def respond(self, scenario: PaperScenario, learner: Optional[RoutingLearner] = None) -> Outcome:
        in_hand = scenario.available
        first = self.decide(scenario, in_hand, scenario.obtainable, learner, 1)
        chain, terminal = [first], first
        if first.action == REQUEST:
            in_hand = in_hand | scenario.obtainable
            terminal = self.decide(scenario, in_hand, frozenset(), learner, 2)
            chain.append(terminal)
        coverage, retained, latency, extra = await self.run_terminal(scenario, first, terminal, in_hand)
        return Outcome(first, terminal, chain, coverage, retained, latency,
                       extra={**self.diagnostics(scenario, first), **extra})  # fmt: skip

    async def _option(self, scenario: PaperScenario, first_action: RoutingAction, terminal: RoutingDecision,
                      in_hand: frozenset[C]) -> tuple[float, bool, float]:  # fmt: skip
        if terminal.action == GENERATE or first_action == GENERATE:
            cached = self.quality.cache.get("generation_only", scenario.name, state_key(in_hand))
            if cached is None and self.lazy_alternatives:
                self.estimated_alternatives += 1
                coverage, retained = self.execute(scenario, GENERATE, terminal, in_hand)
                return coverage, retained, self.latency.of(first_action, GENERATE)
            coverage, retained, seconds, _ = await self.synthesize(scenario, in_hand)
            return coverage, retained, self.path_latency(first_action, GENERATE, seconds)
        coverage, retained = self.execute(scenario, REUSE, terminal, in_hand)
        return coverage, retained, self.latency.of(first_action, REUSE)

    async def alternatives(self, scenario: PaperScenario) -> dict[RoutingAction, list[tuple[float, bool, float]]]:
        first = self.decide(scenario, scenario.available, scenario.obtainable, None, 1)
        generate = first.model_copy(update={"action": GENERATE})
        options = {GENERATE: [await self._option(scenario, GENERATE, generate, scenario.available)]}
        if first.top_candidate is not None:
            options[REUSE] = [await self._option(scenario, REUSE, first.model_copy(update={"action": REUSE}),
                                                 scenario.available)]  # fmt: skip
        if first.requestable or first.rule_action == REQUEST:
            in_hand = scenario.available | scenario.obtainable
            second = self.decide(scenario, in_hand, frozenset(), None, 2)
            terminals = [second.model_copy(update={"action": GENERATE})]
            if second.top_candidate is not None:
                terminals.append(second.model_copy(update={"action": REUSE}))
            options[REQUEST] = [await self._option(scenario, REQUEST, t, in_hand) for t in terminals]
        return options

    def stats(self) -> dict[str, Any]:
        seconds = list(self.synthesized.values())
        return {"syntheses": len(seconds), "synthesis_seconds_mean": (sum(seconds) / len(seconds)) if seconds else None,
                "estimated_alternatives": self.estimated_alternatives}  # fmt: skip
