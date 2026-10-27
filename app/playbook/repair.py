"""Repair of invalid candidate playbooks (paper Section V-C: "invalid offspring are repaired or discarded").

Repair is conservative: it never adds a destructive action, and a destructive action that cannot
be given a proper verification step is removed rather than kept. If the result is still not
valid, the candidate is discarded (``None``).
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from typing import Optional

from app.playbook.builder import Draft
from app.playbook.validator import PlaybookValidator, initial_artifacts
from app.schemas.playbook import Playbook, PlaybookOrigin


class PlaybookRepairer:
    def __init__(self, validator: PlaybookValidator) -> None:
        self.validator = validator
        self.vocabulary = validator.vocabulary

    def repair(
        self,
        playbook: Playbook,
        available: Iterable[str] = (),
        *,
        rng: Optional[random.Random] = None,
        origin: Optional[PlaybookOrigin] = None,
    ) -> Optional[Playbook]:
        available = tuple(available)
        if self.validator.validate(playbook, available).valid:
            return playbook if origin is None else playbook.model_copy(update={"origin": origin})
        draft = Draft.from_playbook(self.vocabulary, playbook, origin=origin)
        if not draft.nodes:
            return None
        self._repair_draft(draft, initial_artifacts(available), rng)
        if not draft.nodes:
            return None
        candidate = draft.to_playbook()
        return candidate if self.validator.validate(candidate, available).valid else None

    def _repair_draft(self, draft: Draft, initial: frozenset[str], rng: Optional[random.Random]) -> None:
        # 1. keep only well-typed, non-duplicate edges that do not close a cycle (in original order)
        edges, draft.edges = draft.edges, []
        for source, target, artifact in edges:
            if not draft.add_edge(source, target, artifact):
                # try retyping a mistyped edge to an artifact the two actions actually share
                if source in draft.nodes and target in draft.nodes and source != target:
                    shared = [a for a in draft.action(source).outputs if a in draft.action(target).inputs]
                    if shared and not any(e[0] == source and e[1] == target for e in draft.edges):
                        draft.add_edge(source, target, shared[0])

        # 2. respect the node cap, leaving room for a verification step and the report
        cap = self.validator.node_cap
        while len(draft.nodes) > max(1, cap - 2):
            removable = [n for n in reversed(list(draft.nodes)) if not draft.action(n).terminal]
            draft.remove_node(removable[0] if removable else next(reversed(list(draft.nodes))))

        # 3. satisfy every input (removing nodes whose inputs cannot be met), until stable
        for _ in range(len(draft.nodes) + 1):
            changed = False
            for node_id in draft.topological_order():
                if node_id in draft.nodes and draft.unmet_inputs(node_id, initial):
                    if not draft.wire_all_inputs(node_id, initial, rng=rng):
                        draft.remove_node(node_id)
                        changed = True
            if not changed:
                break

        # 4. destructive actions need a verification of their target upstream -- or they go
        for node_id in [n for n in draft.nodes if draft.action(n).destructive]:
            if node_id in draft.nodes and not draft.ensure_verification(node_id, initial, rng=rng):
                draft.remove_node(node_id)

        # 5. a terminal action exists and every step leads to it
        if draft.nodes:
            draft.ensure_terminal()
