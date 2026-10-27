"""DAG validity (paper Section V) and the destructive-action safety filter.

A playbook is **valid** for a given set of initial artifacts when:

* every node uses a known action with allowed parameters, node IDs are unique, and the node
  count is within the cap (paper: 20);
* every edge joins two existing, distinct nodes and carries an artifact that is an output of its
  source and an input of its target ("types agree"); edges are not duplicated;
* the graph is acyclic;
* every input of every node is met by an incoming edge of that type or by an initial artifact;
* at least one terminal action exists and terminal actions have no successors;
* no node is orphaned -- interpreted here as: every node has a path to a terminal action, so
  every step contributes to the outcome (stricter than "has at least one edge");
* every destructive node has an ancestor verification action that verifies its target (the
  paper's hard safety filter; violations of this rule make a playbook *unsafe*).

Invalid or unsafe playbooks are rejected before scoring and before execution.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from app.playbook.graph import adjacency, ancestors, reachable, topological_order
from app.playbook.vocabulary import Vocabulary
from app.schemas.playbook import Playbook
from app.tools import artifacts as A

DEFAULT_NODE_CAP = 20


class ViolationCode(str, Enum):
    TOO_MANY_NODES = "too_many_nodes"
    DUPLICATE_NODE = "duplicate_node"
    UNKNOWN_ACTION = "unknown_action"
    INVALID_PARAMS = "invalid_params"
    UNKNOWN_EDGE_NODE = "unknown_edge_node"
    SELF_LOOP = "self_loop"
    DUPLICATE_EDGE = "duplicate_edge"
    TYPE_MISMATCH = "type_mismatch"
    CYCLE = "cycle"
    UNMET_INPUT = "unmet_input"
    NO_TERMINAL = "no_terminal"
    TERMINAL_HAS_SUCCESSOR = "terminal_has_successor"
    ORPHAN = "orphan"
    UNVERIFIED_DESTRUCTIVE = "unverified_destructive"


SAFETY_CODES = frozenset({ViolationCode.UNVERIFIED_DESTRUCTIVE})


class Violation(BaseModel):
    code: ViolationCode
    message: str
    node_id: Optional[str] = None


class ValidationReport(BaseModel):
    violations: list[Violation] = Field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.violations

    @property
    def safe(self) -> bool:
        return not any(v.code in SAFETY_CODES for v in self.violations)

    def codes(self) -> set[ViolationCode]:
        return {v.code for v in self.violations}


def initial_artifacts(available: Iterable[str] = ()) -> frozenset[str]:
    """Artifacts a playbook may assume: the incident context plus available entity types only.

    Gating artifacts such as ``verification_result`` can never be assumed: they must be produced
    inside the playbook.
    """
    return frozenset({A.INCIDENT_CONTEXT} | (set(available) & set(A.ENTITY_ARTIFACTS)))


class PlaybookValidator:
    def __init__(self, vocabulary: Vocabulary, node_cap: int = DEFAULT_NODE_CAP) -> None:
        self.vocabulary = vocabulary
        self.node_cap = node_cap

    def validate(self, playbook: Playbook, available: Iterable[str] = ()) -> ValidationReport:
        initial = initial_artifacts(available)
        violations: list[Violation] = []
        add = lambda code, message, node=None: violations.append(Violation(code=code, message=message, node_id=node))  # noqa: E731

        if len(playbook.nodes) > self.node_cap:
            add(ViolationCode.TOO_MANY_NODES, f"{len(playbook.nodes)} nodes exceed the cap of {self.node_cap}")

        actions = {}
        seen_ids: set[str] = set()
        for node in playbook.nodes:
            if node.node_id in seen_ids:
                add(ViolationCode.DUPLICATE_NODE, f"node id '{node.node_id}' is used twice", node.node_id)
            seen_ids.add(node.node_id)
            action = self.vocabulary.get(node.action)
            if action is None:
                add(ViolationCode.UNKNOWN_ACTION, f"unknown action '{node.action}'", node.node_id)
                continue
            actions[node.node_id] = action
            try:
                self.vocabulary.resolve_params(action, node.params)
            except ValueError as exc:
                add(ViolationCode.INVALID_PARAMS, str(exc), node.node_id)

        seen_edges: set[tuple[str, str, str]] = set()
        supplied: dict[str, set[str]] = {node_id: set() for node_id in seen_ids}
        for edge in playbook.edges:
            key = (edge.source, edge.target, edge.artifact)
            if edge.source not in seen_ids or edge.target not in seen_ids:
                add(ViolationCode.UNKNOWN_EDGE_NODE, f"edge {edge.source}->{edge.target} references an unknown node")
                continue
            if edge.source == edge.target:
                add(ViolationCode.SELF_LOOP, f"node '{edge.source}' feeds itself", edge.source)
                continue
            if key in seen_edges:
                add(ViolationCode.DUPLICATE_EDGE, f"edge {edge.source}->{edge.target} ({edge.artifact}) is duplicated")
                continue
            seen_edges.add(key)
            source, target = actions.get(edge.source), actions.get(edge.target)
            if source and target and (edge.artifact not in source.outputs or edge.artifact not in target.inputs):
                add(ViolationCode.TYPE_MISMATCH, f"edge {edge.source}->{edge.target} carries '{edge.artifact}', which is "
                    f"not an output of '{source.id}' and an input of '{target.id}'", edge.target)  # fmt: skip
                continue
            supplied[edge.target].add(edge.artifact)

        if topological_order(playbook) is None:
            add(ViolationCode.CYCLE, "the graph contains a cycle")

        for node_id, action in actions.items():
            for artifact in action.inputs:
                if artifact not in initial and artifact not in supplied.get(node_id, set()):
                    add(ViolationCode.UNMET_INPUT, f"input '{artifact}' of '{action.id}' is not supplied", node_id)

        successors, _ = adjacency(playbook)
        terminals = {node_id for node_id, action in actions.items() if action.terminal}
        if not terminals:
            add(ViolationCode.NO_TERMINAL, "the playbook has no terminal action")
        for node_id in terminals:
            if successors.get(node_id):
                add(ViolationCode.TERMINAL_HAS_SUCCESSOR, "a terminal action cannot have successors", node_id)
        if terminals:
            for node_id in actions:
                if node_id not in terminals and not (reachable(node_id, successors) & terminals):
                    add(ViolationCode.ORPHAN, f"node '{node_id}' does not lead to a terminal action", node_id)

        for node_id, action in actions.items():
            if action.destructive and not self._verified(playbook, node_id, action.target, actions):
                add(ViolationCode.UNVERIFIED_DESTRUCTIVE,
                    f"destructive action '{action.id}' has no preceding verification of its {action.target}", node_id)  # fmt: skip

        return ValidationReport(violations=violations)

    @staticmethod
    def _verified(playbook: Playbook, node_id: str, target: Optional[str], actions: dict) -> bool:
        return any(
            actions.get(a) is not None and actions[a].verification and target in actions[a].verifies
            for a in ancestors(playbook, node_id)
        )

    def verification_ancestors(self, playbook: Playbook, node_id: str) -> list[str]:
        """IDs of ancestor verification nodes that verify ``node_id``'s target (used at execution time)."""
        node = playbook.node(node_id)
        action = self.vocabulary.get(node.action) if node else None
        if action is None or not action.destructive:
            return []
        result = []
        for ancestor in ancestors(playbook, node_id):
            ancestor_node = playbook.node(ancestor)
            definition = self.vocabulary.get(ancestor_node.action) if ancestor_node else None
            if definition and definition.verification and action.target in definition.verifies:
                result.append(ancestor)
        return sorted(result)
