"""Variation operators (paper Section 5.3).

**Six local mutations**, each modifying one parent:

* ``add_goal_node`` -- add a node addressing a goal in U(p);
* ``add_evidence_node`` -- add a node consuming an uncovered evidence category;
* ``remove_redundant_node`` -- remove a node that repeats earlier work or adds nothing;
* ``rebind_tool`` -- rebind a node to another compatible tool (an action of the same phase reading
  the same kind of evidence through a different tool operation);
* ``rewire_edge`` -- reroute a dependency edge to another type-compatible producer;
* ``insert_verification`` -- insert a verification step before an unguarded destructive action.

The two add operators are gap-directed: with ``directed`` they close a gap drawn from U(p), and
the action is the one the generator proposed for that gap (``proposals``) when there is one;
undirected applications add a random action instead.

**Crossover** is a typed subgraph exchange adapted from GraphEvol: the donor covers items in the
recipient's U(p); a connected donor subgraph addressing them is spliced in at a cut whose incoming
and outgoing artifact types match (replacing the recipient's subgraph there) or attached where its
inputs are satisfied. Offspring over the node cap are rejected.

Every offspring goes through repair; offspring that cannot be repaired into a valid, safe DAG
are discarded (``None``). Operators never add destructive actions: they only appear when a
parent (ultimately the generator) proposed them, and repair keeps them verified or removes them.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

from app.fitness.coverage import CoverageModel, CoverageProfile, Gap
from app.playbook.builder import Draft
from app.playbook.repair import PlaybookRepairer
from app.playbook.validator import initial_artifacts
from app.schemas.playbook import Playbook, PlaybookOrigin

MAX_SUBGRAPH = 3

MUTATIONS = ("add_goal_node", "add_evidence_node", "remove_redundant_node", "rebind_tool", "rewire_edge",
             "insert_verification")  # fmt: skip


@dataclass(frozen=True)
class MutationContext:
    intent: str
    initial: frozenset[str]
    profile: CoverageProfile
    directed: bool


class Operators:
    def __init__(self, coverage: CoverageModel, repairer: PlaybookRepairer) -> None:
        self.coverage = coverage
        self.vocabulary = coverage.vocabulary
        self.repairer = repairer
        self.node_cap = repairer.validator.node_cap
        # gap -> action id proposed by the generator for closing it (filled before a run)
        self.proposals: dict[Gap, str] = {}
        self.applied: dict[str, int] = {name: 0 for name in MUTATIONS + ("crossover",)}
        self.gap_directed = 0  # add applications that closed a gap in U(p)
        self.proposal_used = 0  # ... with the generator's proposed action

    # --- entry points ----------------------------------------------------------------------

    def mutate(self, playbook: Playbook, profile: CoverageProfile, intent: str, available: frozenset[str],
               rng: random.Random, *, directed: bool) -> Optional[Playbook]:  # fmt: skip
        """Apply one local mutation, chosen uniformly among those applicable to this parent."""
        context = MutationContext(intent, initial_artifacts(available), profile, directed)
        names = list(MUTATIONS)
        rng.shuffle(names)
        parent_key = playbook.structural_hash()
        for name in names:
            draft = Draft.from_playbook(self.vocabulary, playbook, origin=PlaybookOrigin.EVOLVED)
            operator: Callable = getattr(self, name)
            if not operator(draft, context, rng):
                continue
            child = self._finish(draft, available, rng)
            if child is not None and child.structural_hash() != parent_key:
                self.applied[name] += 1
                return child
        return None

    def crossover(self, recipient: Playbook, recipient_profile: CoverageProfile, donor: Playbook,
                  donor_profile: CoverageProfile, intent: str, available: frozenset[str],
                  rng: random.Random) -> Optional[Playbook]:  # fmt: skip
        """Splice a donor subgraph that addresses items of the recipient's U(p) into the recipient."""
        wanted = set(recipient_profile.gaps()) & set(donor_profile.covered())
        if not wanted:
            return None
        initial = initial_artifacts(available)
        child = Draft.from_playbook(self.vocabulary, recipient, origin=PlaybookOrigin.EVOLVED)
        source = Draft.from_playbook(self.vocabulary, donor, origin=PlaybookOrigin.EVOLVED)
        foreign = [s for s in self._subgraphs(source)
                   if any(self.coverage.closes(source.action(n), gap, intent) for n in s for gap in wanted)]  # fmt: skip
        if not foreign:
            return None
        cuts: list[set[str]] = [set()] + self._subgraphs(child)  # attach, or replace a subgraph
        rng.shuffle(foreign)
        rng.shuffle(cuts)
        parent_key = recipient.structural_hash()
        for removed in cuts[:5]:
            # what the rest of the child can still provide once the subgraph is gone
            remaining_outputs = set(initial) | {a for n in child.nodes if n not in removed for a in child.action(n).outputs}
            consumers = [(t, a) for s, t, a in child.edges if s in removed and t not in removed]
            needed_out = {a for _, a in consumers}
            for donated in foreign[:6]:
                if len(child.nodes) - len(removed) + len(donated) > self.node_cap:
                    continue
                inputs = {a for s, t, a in source.edges if t in donated and s not in donated}
                inputs |= {a for n in donated for a in source.action(n).inputs
                           if a in initial and a not in {x for s, t, x in source.edges if t == n}}  # fmt: skip
                outputs = {a for n in donated for a in source.action(n).outputs}
                if not inputs <= remaining_outputs or not needed_out <= outputs | remaining_outputs:
                    continue  # interface types do not align
                state = child.snapshot()
                if self._swap(child, removed, source, donated, consumers, initial, rng):
                    result = self._finish(child, available, rng)
                    if (result is not None and len(result.nodes) <= self.node_cap
                            and result.structural_hash() != parent_key):  # fmt: skip
                        self.applied["crossover"] += 1
                        return result
                child.restore(state)
        return None

    # --- the six local mutations (each edits the draft in place and reports success) --------------

    def add_goal_node(self, draft: Draft, context: MutationContext, rng: random.Random) -> bool:
        return self._add(draft, context, rng, "goal")

    def add_evidence_node(self, draft: Draft, context: MutationContext, rng: random.Random) -> bool:
        return self._add(draft, context, rng, "category")

    def remove_redundant_node(self, draft: Draft, context: MutationContext, rng: random.Random) -> bool:
        """Remove a node that repeats earlier work, or whose removal costs no coverage and feeds no other step."""
        candidates = self._redundant(draft, context.intent)
        if not candidates:
            return False
        draft.remove_node(rng.choice(candidates))
        return True  # consumers that lost an input are rewired or removed by repair

    def rebind_tool(self, draft: Draft, context: MutationContext, rng: random.Random) -> bool:
        nodes = [n for n in draft.nodes if draft.action(n).binding_kind == "tool" and not draft.action(n).destructive]
        rng.shuffle(nodes)
        for node_id in nodes:
            current = draft.action(node_id)
            supplied = draft.supplied(node_id) | context.initial
            needed = {a for s, _, a in draft.edges if s == node_id}
            options = [a for a in self.vocabulary.actions.values()
                       if a.id != current.id and a.binding != current.binding and a.binding_kind == "tool"
                       and a.phase == current.phase and not a.destructive and not a.terminal
                       and a.categories & current.categories
                       and set(a.inputs) <= supplied and needed <= set(a.outputs)]  # fmt: skip
            if options:
                replacement = rng.choice(options)
                draft.nodes[node_id] = draft.nodes[node_id].model_copy(update={"action": replacement.id, "params": {}})
                # drop incoming edges the new action does not consume
                draft.edges = [e for e in draft.edges if e[1] != node_id or e[2] in replacement.inputs]
                return True
        return False

    def rewire_edge(self, draft: Draft, context: MutationContext, rng: random.Random) -> bool:
        edges = list(draft.edges)
        rng.shuffle(edges)
        for source, target, artifact in edges:
            blocked = draft.descendants(target) | {target, source}
            alternatives = [n for n in draft.nodes if n not in blocked and artifact in draft.action(n).outputs
                            and (n, target, artifact) not in draft.edges]  # fmt: skip
            if alternatives:
                draft.edges.remove((source, target, artifact))
                draft.add_edge(rng.choice(alternatives), target, artifact)
                return True
        return False

    def insert_verification(self, draft: Draft, context: MutationContext, rng: random.Random) -> bool:
        unguarded = [n for n in draft.nodes if draft.action(n).destructive and not draft.verified(n)]
        rng.shuffle(unguarded)
        for node_id in unguarded:
            if len(draft.nodes) < self.node_cap and draft.ensure_verification(node_id, context.initial, rng=rng):
                return True
        return False

    # --- helpers -------------------------------------------------------------------------------

    def _add(self, draft: Draft, context: MutationContext, rng: random.Random, kind: str) -> bool:
        present = {n.action for n in draft.nodes.values()}
        if context.directed:
            gaps = [g for g in context.profile.gaps() if g.kind == kind]
            if not gaps:
                return False
            gap = rng.choice(gaps)
            closers = [a for a in self.coverage.closers(gap, context.intent) if a.id not in present]
            rng.shuffle(closers)
            proposed = self.proposals.get(gap)
            closers.sort(key=lambda a: a.id != proposed)  # the generator's proposal first
        else:
            closers = [a for a in self.vocabulary.actions.values()
                       if not a.destructive and not a.terminal and a.id not in present]  # fmt: skip
            rng.shuffle(closers)
        for closer in closers[:4]:
            state = draft.snapshot()
            if len(draft.nodes) >= self.node_cap - 1 and not self._drop_least_useful(draft, context.intent):
                return False
            node_id = draft.add_node(closer.id)
            if draft.wire_all_inputs(node_id, context.initial, rng=rng):
                if context.directed:
                    self.gap_directed += 1
                    self.proposal_used += closer.id == proposed
                return True
            draft.restore(state)
        return False

    def _redundant(self, draft: Draft, intent: str) -> list[str]:
        base = self.coverage.profile(draft.to_playbook(), intent).vector
        seen: set[tuple] = set()
        result = []
        for node_id in draft.topological_order():
            action = draft.action(node_id)
            if action.terminal or action.verification or action.destructive or action.id == "extract_entities":
                continue
            params = json.dumps(draft.nodes[node_id].params, sort_keys=True, default=str)
            signatures = {("action", action.id, params)}
            if action.binding_kind == "tool":
                signatures.add(("operation", action.binding, action.tool_target_type, params))
            if signatures & seen:
                result.append(node_id)
            seen |= signatures
            if node_id in result or any(not draft.action(s).terminal for s in draft.successors(node_id)):
                continue
            trial = Draft.from_playbook(self.vocabulary, draft.to_playbook(), sanitize=False)
            trial.remove_node(node_id)
            if self.coverage.profile(trial.to_playbook(), intent).vector == base:
                result.append(node_id)
        return result

    def _finish(self, draft: Draft, available: frozenset[str], rng: random.Random) -> Optional[Playbook]:
        draft.ensure_terminal()
        return self.repairer.repair(draft.to_playbook(), available, rng=rng, origin=PlaybookOrigin.EVOLVED)

    def _subgraphs(self, draft: Draft) -> list[set[str]]:
        """Small connected subgraphs rooted at a non-terminal node (the node plus up to two descendants)."""
        result = []
        for root in draft.nodes:
            if draft.action(root).terminal or draft.action(root).id == "extract_entities":
                continue
            members = {root}
            frontier = [root]
            while frontier and len(members) < MAX_SUBGRAPH:
                for successor in sorted(draft.successors(frontier.pop(0))):
                    if not draft.action(successor).terminal and len(members) < MAX_SUBGRAPH:
                        members.add(successor)
                        frontier.append(successor)
            result.append(members)
        return result

    def _swap(self, child: Draft, removed: set[str], donor: Draft, donated: set[str],
              consumers: list[tuple[str, str]], initial: frozenset[str], rng: random.Random) -> bool:  # fmt: skip
        for node_id in removed:
            child.remove_node(node_id)
        mapping = {old: child.add_node(donor.nodes[old].action, donor.nodes[old].params)
                   for old in donor.topological_order() if old in donated}  # fmt: skip
        for source, target, artifact in donor.edges:
            if source in donated and target in donated:
                child.add_edge(mapping[source], mapping[target], artifact)
        for new in mapping.values():
            if not child.wire_all_inputs(new, initial, rng=rng):
                return False
        for target, artifact in consumers:
            if target in child.nodes and not child.wire_input(target, artifact, initial, rng=rng):
                return False
        return True

    def _drop_least_useful(self, draft: Draft, intent: str) -> bool:
        """Make room under the node cap by removing a node whose loss costs no coverage."""
        base = self.coverage.profile(draft.to_playbook(), intent).vector
        for node_id in reversed(list(draft.nodes)):
            action = draft.action(node_id)
            if action.terminal or action.verification or action.id == "extract_entities":
                continue
            trial = Draft.from_playbook(self.vocabulary, draft.to_playbook(), sanitize=False)
            trial.remove_node(node_id)
            if self.coverage.profile(trial.to_playbook(), intent).vector == base:
                draft.remove_node(node_id)
                return True
        return False
