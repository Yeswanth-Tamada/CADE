"""A mutable working copy of a playbook with type-aware wiring.

Shared by repair, the constructive generator and the evolutionary operators so that every way
of changing a playbook wires edges the same way: an input is satisfied by an initial artifact,
by an existing upstream producer of that type (never a descendant, so no cycles), or by
inserting a producer action whose own inputs can be satisfied the same way (bounded depth).
"""

from __future__ import annotations

import copy
import random
from typing import Any, Optional

from app.playbook.vocabulary import ActionDefinition, Vocabulary
from app.schemas.playbook import Playbook, PlaybookEdge, PlaybookNode, PlaybookOrigin
from app.tools import artifacts as A

Edge = tuple[str, str, str]  # (source, target, artifact)


class Draft:
    def __init__(self, vocabulary: Vocabulary, *, name: str, intent_label: str, origin: PlaybookOrigin,
                 description: Optional[str] = None) -> None:  # fmt: skip
        self.vocabulary = vocabulary
        self.name = name
        self.intent_label = intent_label
        self.origin = origin
        self.description = description
        self.nodes: dict[str, PlaybookNode] = {}
        self.edges: list[Edge] = []
        self._counter = 0

    # --- conversion ----------------------------------------------------------------------------

    @classmethod
    def from_playbook(cls, vocabulary: Vocabulary, playbook: Playbook, *, origin: Optional[PlaybookOrigin] = None,
                      sanitize: bool = True) -> "Draft":  # fmt: skip
        """Copy a playbook. With ``sanitize``: drop unknown actions, rename duplicate IDs, reset bad params."""
        draft = cls(vocabulary, name=playbook.name, intent_label=playbook.intent_label,
                    origin=origin or playbook.origin, description=playbook.description)  # fmt: skip
        renamed: dict[str, str] = {}
        for node in playbook.nodes:
            action = vocabulary.get(node.action)
            if sanitize and action is None:
                continue
            params = dict(node.params)
            if sanitize and action is not None:
                try:
                    vocabulary.resolve_params(action, params)
                except ValueError:
                    params = {}
            node_id = node.node_id
            if node_id in draft.nodes:
                if not sanitize:
                    raise ValueError(f"duplicate node id {node_id}")
                node_id = draft.new_id()
            else:
                renamed.setdefault(node.node_id, node_id)
            draft.nodes[node_id] = PlaybookNode(node_id=node_id, action=node.action, params=params)
            draft._bump(node_id)
        for edge in playbook.edges:
            source, target = renamed.get(edge.source), renamed.get(edge.target)
            if source in draft.nodes and target in draft.nodes:
                draft.edges.append((source, target, edge.artifact))
        return draft

    def to_playbook(self) -> Playbook:
        return Playbook(
            name=self.name,
            intent_label=self.intent_label,
            description=self.description,
            origin=self.origin,
            nodes=list(self.nodes.values()),
            edges=[PlaybookEdge(source=s, target=t, artifact=a) for s, t, a in self.edges],
        )

    def snapshot(self) -> tuple:
        return copy.deepcopy(self.nodes), list(self.edges), self._counter

    def restore(self, state: tuple) -> None:
        self.nodes, self.edges, self._counter = copy.deepcopy(state[0]), list(state[1]), state[2]

    # --- basic edits ---------------------------------------------------------------------------

    def _bump(self, node_id: str) -> None:
        if node_id.startswith("n") and node_id[1:].isdigit():
            self._counter = max(self._counter, int(node_id[1:]))

    def new_id(self) -> str:
        self._counter += 1
        while f"n{self._counter}" in self.nodes:
            self._counter += 1
        return f"n{self._counter}"

    def action(self, node_id: str) -> ActionDefinition:
        return self.vocabulary.actions[self.nodes[node_id].action]

    def add_node(self, action_id: str, params: Optional[dict[str, Any]] = None) -> str:
        node_id = self.new_id()
        self.nodes[node_id] = PlaybookNode(node_id=node_id, action=action_id, params=dict(params or {}))
        return node_id

    def remove_node(self, node_id: str) -> None:
        self.nodes.pop(node_id, None)
        self.edges = [e for e in self.edges if node_id not in (e[0], e[1])]

    def add_edge(self, source: str, target: str, artifact: str) -> bool:
        """Add an edge if it is well-typed, new and keeps the graph acyclic."""
        edge = (source, target, artifact)
        if source == target or edge in self.edges or source not in self.nodes or target not in self.nodes:
            return False
        if artifact not in self.action(source).outputs or artifact not in self.action(target).inputs:
            return False
        if source in self.descendants(target) or source == target:
            return False
        self.edges.append(edge)
        return True

    # --- structure queries -------------------------------------------------------------------

    def successors(self, node_id: str) -> set[str]:
        return {t for s, t, _ in self.edges if s == node_id}

    def predecessors(self, node_id: str) -> set[str]:
        return {s for s, t, _ in self.edges if t == node_id}

    def descendants(self, node_id: str) -> set[str]:
        seen: set[str] = set()
        stack = [node_id]
        while stack:
            for nxt in self.successors(stack.pop()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    def ancestors(self, node_id: str) -> set[str]:
        seen: set[str] = set()
        stack = [node_id]
        while stack:
            for prev in self.predecessors(stack.pop()):
                if prev not in seen:
                    seen.add(prev)
                    stack.append(prev)
        return seen

    def topological_order(self) -> list[str]:
        """Topological order (stable by insertion); assumes the draft is acyclic."""
        indegree = {n: 0 for n in self.nodes}
        for _, t, _ in self.edges:
            indegree[t] += 1
        order, ready = [], [n for n in self.nodes if indegree[n] == 0]
        while ready:
            node = ready.pop(0)
            order.append(node)
            for s, t, _ in self.edges:
                if s == node:
                    indegree[t] -= 1
                    if indegree[t] == 0:
                        ready.append(t)
        return order if len(order) == len(self.nodes) else list(self.nodes)

    def supplied(self, node_id: str) -> set[str]:
        return {a for _, t, a in self.edges if t == node_id}

    def unmet_inputs(self, node_id: str, initial: frozenset[str]) -> list[str]:
        have = self.supplied(node_id) | initial
        return [a for a in self.action(node_id).inputs if a not in have]

    def verified(self, node_id: str) -> bool:
        target = self.action(node_id).target
        return any(self.action(a).verification and target in self.action(a).verifies for a in self.ancestors(node_id))

    # --- wiring --------------------------------------------------------------------------------

    def _producer_actions(self, artifact: str, consumer: str) -> list[ActionDefinition]:
        candidates = [a for a in self.vocabulary.producers.get(artifact, []) if not a.destructive and not a.terminal]
        consumer_action = self.action(consumer)
        if artifact == A.VERIFICATION_RESULT and consumer_action.destructive:
            candidates = [a for a in candidates if consumer_action.target in a.verifies]
        return sorted(candidates, key=lambda a: (len(a.inputs), a.phase.order, a.id))

    def _closure(self, nodes: set[str]) -> set[str]:
        """``nodes`` plus all their descendants."""
        result = set(nodes)
        for node in nodes:
            result |= self.descendants(node)
        return result

    def wire_input(self, consumer: str, artifact: str, initial: frozenset[str], *,
                   rng: Optional[random.Random] = None, depth: int = 2, allow_insert: bool = True,
                   forbidden: frozenset[str] = frozenset()) -> bool:  # fmt: skip
        """Satisfy one input of ``consumer``; returns False (draft unchanged) if impossible.

        ``forbidden`` nodes (and their descendants) may not supply the input: when a producer is
        inserted *for* some node, it must not draw its own inputs from that node's downstream.
        """
        if artifact in initial or artifact in self.supplied(consumer):
            return True
        blocked = self._closure({consumer} | set(forbidden))
        existing = [n for n in self.nodes if n not in blocked and artifact in self.action(n).outputs]
        consumer_action = self.action(consumer)
        if artifact == A.VERIFICATION_RESULT and consumer_action.destructive:
            existing = [n for n in existing if consumer_action.target in self.action(n).verifies]
        if existing:
            order = {n: i for i, n in enumerate(self.topological_order())}
            existing.sort(key=lambda n: order.get(n, 0), reverse=True)  # nearest upstream producer first
            source = rng.choice(existing[:2]) if rng and len(existing) > 1 else existing[0]
            return self.add_edge(source, consumer, artifact)
        if not allow_insert or depth <= 0:
            return False
        producers = self._producer_actions(artifact, consumer)
        if rng:
            head = producers[:3]
            rng.shuffle(head)
            producers = head + producers[3:]
        for producer in producers[:6]:
            state = self.snapshot()
            node_id = self.add_node(producer.id)
            upstream_only = frozenset(forbidden | {consumer})
            if all(self.wire_input(node_id, a, initial, rng=rng, depth=depth - 1, forbidden=upstream_only)
                   for a in self.unmet_inputs(node_id, initial)) and self.add_edge(node_id, consumer, artifact):  # fmt: skip
                return True
            self.restore(state)
        return False

    def wire_all_inputs(self, node_id: str, initial: frozenset[str], *, rng: Optional[random.Random] = None,
                        forbidden: frozenset[str] = frozenset()) -> bool:  # fmt: skip
        state = self.snapshot()
        for artifact in self.unmet_inputs(node_id, initial):
            if not self.wire_input(node_id, artifact, initial, rng=rng, forbidden=forbidden):
                self.restore(state)
                return False
        return True

    def ensure_verification(self, node_id: str, initial: frozenset[str], *, rng: Optional[random.Random] = None) -> bool:
        """Give a destructive node a verification ancestor that verifies its target."""
        if self.verified(node_id):
            return True
        target = self.action(node_id).target
        for verification in sorted(self.vocabulary.verifications_for(target), key=lambda a: (len(a.inputs), a.id)):
            state = self.snapshot()
            verifier = self.add_node(verification.id)
            wired = self.wire_all_inputs(verifier, initial, rng=rng, forbidden=frozenset({node_id}))
            if wired and self.add_edge(verifier, node_id, A.VERIFICATION_RESULT):
                return True
            self.restore(state)
        return False

    def ensure_terminal(self) -> None:
        """Make sure a terminal action exists and every other sink feeds it its findings."""
        terminals = [n for n in self.nodes if self.action(n).terminal]
        for terminal in terminals:  # a terminal action may not have successors
            self.edges = [e for e in self.edges if e[0] != terminal]
        if not terminals:
            report = self.vocabulary.terminal_actions()[0]
            terminals = [self.add_node(report.id)]
        terminal = terminals[0]
        for node_id in list(self.nodes):
            if node_id in terminals:
                continue
            if not (self.descendants(node_id) & set(terminals)):
                self.add_edge(node_id, terminal, A.FINDINGS)
