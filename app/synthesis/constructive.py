"""Randomised goal-driven construction of candidate playbooks.

Used when the language model yields no valid candidate (so synthesis never stalls) and to fill
the pool in offline/mock mode. Every candidate is built only from vocabulary actions, wired by
type, and must pass the same repair/validation as model output. It never adds destructive actions.
"""

from __future__ import annotations

import random
from typing import Optional

from app.fitness.coverage import CoverageModel
from app.playbook.builder import Draft
from app.playbook.repair import PlaybookRepairer
from app.playbook.validator import initial_artifacts
from app.schemas.playbook import Playbook, PlaybookOrigin
from app.synthesis.context import SynthesisContext
from app.tools import artifacts as A


class ConstructiveGenerator:
    def __init__(self, coverage: CoverageModel, repairer: PlaybookRepairer) -> None:
        self.coverage = coverage
        self.vocabulary = coverage.vocabulary
        self.repairer = repairer

    def build(self, context: SynthesisContext, rng: random.Random, *, goal_rate: float = 0.85,
              category_rate: float = 0.6, verify_rate: float = 0.5) -> Optional[Playbook]:  # fmt: skip
        intent = context.intent_label
        draft = Draft(self.vocabulary, name=f"Constructed playbook for {intent.replace('_', ' ')}",
                      intent_label=intent, origin=PlaybookOrigin.CONSTRUCTIVE)  # fmt: skip
        initial = initial_artifacts(context.available)
        if rng.random() < 0.7:
            draft.add_node("triage_alert")

        goals = list(self.coverage.goals(intent))
        rng.shuffle(goals)
        for goal in goals:
            if rng.random() < goal_rate:
                self._add_one(draft, [a.id for a in self.vocabulary.actions_for_goal(goal)], initial, rng)
        for category in self.coverage.relevant_categories(intent):
            if rng.random() < category_rate:
                self._add_one(draft, [a.id for a in self.vocabulary.actions_consuming(category)], initial, rng)
        if rng.random() < verify_rate:
            targets = [t for t in (A.HOST_ID, A.USER_ID) if t in context.available] or [A.HOST_ID]
            self._add_one(draft, [a.id for a in self.vocabulary.verifications_for(rng.choice(targets))], initial, rng)
        if not draft.nodes:
            return None
        draft.ensure_terminal()
        return self.repairer.repair(draft.to_playbook(), context.available, rng=rng)

    def _add_one(self, draft: Draft, choices: list[str], initial: frozenset[str], rng: random.Random) -> None:
        present = {n.action for n in draft.nodes.values()}
        choices = [c for c in choices if c not in present]
        rng.shuffle(choices)
        for action_id in choices[:3]:
            state = draft.snapshot()
            node_id = draft.add_node(action_id)
            if draft.wire_all_inputs(node_id, initial, rng=rng):
                return
            draft.restore(state)
