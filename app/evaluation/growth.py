"""Library-growth experiment (paper 968 Section 6.4 and Fig. 4a): reuse rate as the library grows.

"A separate library-growth experiment does not fix library suitability by design: resolved playbooks are written
back, and each incident's correct decision is recomputed against the current library." The paper reports reuse
rising from 28% at 10 stored playbooks to 41%, 50%, 57% and 61% at 20, 30, 40 and 50.

How this run works (choices where the paper is silent are marked *(choice)*):

* The library starts with ``initial`` benchmark technique playbooks drawn at random (seeded) *(choice)*; nothing
  is withheld afterwards.
* The 96 scenarios are streamed in a shuffled order, pass after pass, until the library holds ``target``
  playbooks or ``max_passes`` passes are done *(choice: the paper does not say how many incidents it streamed)*.
* Routing uses the hand-designed rule of Eq. 7 (deterministic) *(choice: the learned router's online state
  would mix two experiments)*; at most one evidence request per incident, released evidence as constructed.
* Execution and retention use the paper's rule on the playbook actually returned: a goal (critical category) is
  answered when the playbook queries that category, the evidence is in hand and the playbook investigates the
  scenario's tactic; retained = every critical goal answered and IC_exec >= 0.8. This is a SIMULATED execution
  check (no logs), like the routing benchmark's, but applied to the real reused or synthesized playbook.
* Write-back is CADE's own (``PlaybookLibrary.record_outcome``): a reused entry's counts are updated; a retained
  synthesized playbook is generalised and stored, or merged into a near-duplicate.
* The correct decision is recomputed against the current library with Table 2's rules: REUSE when a playbook for
  the scenario's technique is stored and its critical evidence is in hand, REQUEST when critical evidence is
  missing but obtainable and a stored playbook of the technique or tactic could run once released, else GENERATE.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Any, Optional

from app.evaluation.paper968 import SEVEN, Benchmark, PaperScenario, technique_playbook
from app.evaluation.quality import QualityRun
from app.library.library import PlaybookLibrary
from app.playbook.vocabulary import Vocabulary
from app.schemas.enums import EvidenceCategory as C
from app.schemas.playbook import Playbook
from app.schemas.routing import RoutingAction

REUSE, REQUEST, GENERATE = RoutingAction.REUSE.value, RoutingAction.REQUEST_EVIDENCE.value, RoutingAction.SYNTHESIZE.value
PAPER_FIG_4A = {10: 0.28, 20: 0.41, 30: 0.50, 40: 0.57, 50: 0.61}


def queried(playbook: Playbook, vocabulary: Vocabulary) -> frozenset[C]:
    """The paper's seven evidence categories the playbook's actions read."""
    out = set()
    for node in playbook.nodes:
        action = vocabulary.get(node.action)
        out |= {u.category for u in action.consumes} if action else set()
    return frozenset(out) & frozenset(SEVEN)


def execution_check(scenario: PaperScenario, playbook: Playbook, tactic: str, in_hand: frozenset[C],
                    vocabulary: Vocabulary) -> tuple[float, bool]:  # fmt: skip
    """(IC_exec, retained) for the returned playbook -- SIMULATED (see the module docstring)."""
    goals = scenario.critical
    if tactic != scenario.tactic:
        return 0.0, False  # another tactic's playbook finds no events attributable to the technique
    answered = goals & queried(playbook, vocabulary) & in_hand
    coverage = len(answered) / len(goals) if goals else 1.0
    return coverage, goals <= answered and coverage >= 0.8


def seed_initial(benchmark: Benchmark, library: PlaybookLibrary, vocabulary: Vocabulary, initial: int,
                 rng: random.Random) -> list[str]:  # fmt: skip
    techniques = rng.sample(sorted(benchmark.library_techniques), min(initial, len(benchmark.library_techniques)))
    for technique in techniques:
        library.add(technique_playbook(benchmark, technique, vocabulary), deduplicate=False,
                    metadata={"source": "paper968 benchmark", "technique": technique,
                              "tactic": benchmark.tactic_of[technique]})  # fmt: skip
    return techniques


@dataclass
class GrowthRun:
    quality: QualityRun  # synthesis (cached) and the PaperEnv routing over the growing library
    benchmark: Benchmark
    vocabulary: Vocabulary

    @property
    def library(self) -> PlaybookLibrary:
        return self.quality.env.library

    def stored(self) -> dict[str, set[str]]:
        """technique -> tactics of the stored playbooks for it (from entry metadata)."""
        out: dict[str, set[str]] = {}
        for entry in self.library.entries():
            technique = entry.metadata.get("technique")
            if technique:
                out.setdefault(technique, set()).add(entry.metadata.get("tactic", entry.intent_label))
        return out

    def correct(self, scenario: PaperScenario) -> str:
        stored = self.stored()
        own = scenario.technique in stored
        same_tactic = any(self.benchmark.tactic_of.get(t) == scenario.tactic for t in stored)
        critical_in_hand = scenario.critical <= scenario.available
        if own and critical_in_hand:
            return REUSE
        released = scenario.available | scenario.obtainable
        if not critical_in_hand and scenario.critical <= released and (own or same_tactic):
            return REQUEST
        return GENERATE

    async def step(self, scenario: PaperScenario) -> dict[str, Any]:
        env = self.quality.env
        open_scenario = replace(scenario, withheld_techniques=frozenset())
        size, expected = len(self.library), self.correct(open_scenario)
        in_hand = scenario.available
        first = env.decide(open_scenario, in_hand, scenario.obtainable, None, 1)
        terminal = first
        if first.action.value == REQUEST:
            in_hand = in_hand | scenario.obtainable
            terminal = env.decide(open_scenario, in_hand, frozenset(), None, 2)
        row: dict[str, Any] = {"scenario": scenario.name, "library_size": size, "expected": expected,
                               "got": first.action.value, "terminal": terminal.action.value}  # fmt: skip
        if terminal.action.value == REUSE and terminal.top_candidate is not None:
            entry = self.library.get(terminal.top_candidate.entry_id)
            coverage, retained = execution_check(scenario, entry.playbook, entry.intent_label, in_hand, self.vocabulary)
            outcome = self.library.record_outcome(entry.playbook, success=retained, source_entry_id=entry.entry_id)
        else:
            record = await self.quality.produce("generation_only", scenario, in_hand)
            playbook: Optional[Playbook] = Playbook.model_validate(record.playbook) if record.playbook and not record.failed else None
            if playbook is None:
                coverage, retained, outcome = 0.0, False, None
            else:
                # stored under the benchmark's tactic intent, like the seeded technique playbooks
                playbook = playbook.model_copy(update={"intent_label": scenario.tactic})
                coverage, retained = execution_check(scenario, playbook, scenario.tactic, in_hand, self.vocabulary)
                outcome = self.library.record_outcome(playbook, success=retained,
                                                      metadata={"source": "growth write-back", "technique": scenario.technique,
                                                                "tactic": scenario.tactic})  # fmt: skip
        row.update({"execution_coverage": coverage, "retained": retained,
                    "write_back": outcome.action if outcome else "synthesis_failed", "library_size_after": len(self.library)})  # fmt: skip
        return row

    async def run(self, *, seed: int = 0, target: int = 50, max_passes: int = 3) -> list[dict[str, Any]]:
        rng = random.Random(seed)
        rows: list[dict[str, Any]] = []
        for pass_number in range(max_passes):
            order = list(self.benchmark.scenarios)
            rng.shuffle(order)
            for scenario in order:
                row = await self.step(scenario)
                rows.append({"pass": pass_number, "step": len(rows), **row})
            if len(self.library) >= target:
                break
        return rows


def reuse_by_size(rows: list[dict[str, Any]], sizes=(10, 20, 30, 40, 50), width: int = 10) -> dict[int, dict[str, Any]]:
    """Reuse rate (terminal action REUSE) among incidents routed while the library held size +- width/2 playbooks."""
    out = {}
    for size in sizes:
        window = [r for r in rows if size - width / 2 <= r["library_size"] < size + width / 2]
        out[size] = {"incidents": len(window),
                     "reuse_rate": (sum(r["terminal"] == REUSE for r in window) / len(window)) if window else None,
                     "accuracy": (sum(r["got"] == r["expected"] for r in window) / len(window)) if window else None}  # fmt: skip
    return out
