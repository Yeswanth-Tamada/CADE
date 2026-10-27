"""Small graph helpers over a playbook's edge list."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Optional

from app.schemas.playbook import Playbook


def adjacency(playbook: Playbook) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """``(successors, predecessors)`` keyed by node id (edges to unknown nodes are ignored)."""
    ids = {n.node_id for n in playbook.nodes}
    successors: dict[str, set[str]] = defaultdict(set)
    predecessors: dict[str, set[str]] = defaultdict(set)
    for edge in playbook.edges:
        if edge.source in ids and edge.target in ids:
            successors[edge.source].add(edge.target)
            predecessors[edge.target].add(edge.source)
    return successors, predecessors


def topological_order(playbook: Playbook) -> Optional[list[str]]:
    """Kahn's algorithm, stable with respect to node list order. ``None`` if there is a cycle."""
    successors, predecessors = adjacency(playbook)
    order_index = {n.node_id: i for i, n in enumerate(playbook.nodes)}
    indegree = {n.node_id: len(predecessors[n.node_id]) for n in playbook.nodes}
    ready = deque(sorted((n for n, d in indegree.items() if d == 0), key=order_index.get))
    order: list[str] = []
    while ready:
        node = ready.popleft()
        order.append(node)
        for successor in sorted(successors[node], key=order_index.get):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
    return order if len(order) == len(playbook.nodes) else None


def reachable(start: str, edges: dict[str, set[str]]) -> set[str]:
    seen: set[str] = set()
    stack = list(edges.get(start, ()))
    while stack:
        node = stack.pop()
        if node not in seen:
            seen.add(node)
            stack.extend(edges.get(node, ()))
    return seen


def ancestors(playbook: Playbook, node_id: str) -> set[str]:
    return reachable(node_id, adjacency(playbook)[1])


def descendants(playbook: Playbook, node_id: str) -> set[str]:
    return reachable(node_id, adjacency(playbook)[0])
