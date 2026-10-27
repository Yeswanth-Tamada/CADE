"""Selection (paper Section 5.3).

**Coverage-stratified selection**: candidates are grouped into strata by covered fraction
|c(p)| / (|G| + |E|), and parents are drawn by binary tournament on fitness within a uniformly
chosen stratum, so one high-fitness lineage cannot crowd out candidates covering rarer goals.

**Elites** are the best *judged* candidates (their OS comes from the judge, or from OS_H when no
judge is configured); heuristic estimates only decide which candidates are judged and bred.

With coverage-stratified selection disabled, parents come from a binary tournament over the whole
population. ``stratified_survivors`` / ``proportional_survivors`` implement (mu + lambda) survival
for the ``generational_replacement=False`` ablation.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Optional

from app.fitness.fitness import EvaluatedCandidate

TOURNAMENT_SIZE = 2


def unique_by_structure(candidates: list[EvaluatedCandidate]) -> list[EvaluatedCandidate]:
    best: dict[str, EvaluatedCandidate] = {}
    for candidate in candidates:
        if candidate.key not in best or candidate.fitness > best[candidate.key].fitness:
            best[candidate.key] = candidate
    return sorted(best.values(), key=lambda c: c.fitness, reverse=True)


def strata(candidates: list[EvaluatedCandidate]) -> dict[int, list[EvaluatedCandidate]]:
    """Candidates grouped by |c(p)| (the covered fraction's numerator; the denominator is fixed per intent)."""
    groups: dict[int, list[EvaluatedCandidate]] = defaultdict(list)
    for candidate in sorted(candidates, key=lambda c: c.fitness, reverse=True):
        groups[candidate.profile.covered_count].append(candidate)
    return dict(groups)


def niches(candidates: list[EvaluatedCandidate]) -> dict[tuple[int, ...], list[EvaluatedCandidate]]:
    """Candidates grouped by their exact coverage vector (diversity statistic and (mu + lambda) survival)."""
    groups: dict[tuple[int, ...], list[EvaluatedCandidate]] = defaultdict(list)
    for candidate in sorted(candidates, key=lambda c: c.fitness, reverse=True):
        groups[candidate.profile.vector].append(candidate)
    return dict(groups)


def elites(candidates: list[EvaluatedCandidate], count: int) -> list[EvaluatedCandidate]:
    """The ``count`` fittest distinct candidates whose OS is judged (never a heuristic estimate)."""
    return [c for c in unique_by_structure(candidates) if c.judged][:count]


def stratified_survivors(candidates: list[EvaluatedCandidate], size: int, elite_count: int) -> list[EvaluatedCandidate]:
    ranked = unique_by_structure(candidates)
    survivors = elites(ranked, elite_count)
    chosen = {c.key for c in survivors}
    groups = sorted(niches(ranked).values(), key=lambda g: g[0].fitness, reverse=True)
    depth = 0
    while len(survivors) < size and any(len(g) > depth for g in groups):
        for group in groups:
            if len(survivors) >= size:
                break
            if len(group) > depth and group[depth].key not in chosen:
                survivors.append(group[depth])
                chosen.add(group[depth].key)
        depth += 1
    return survivors


def proportional_survivors(candidates: list[EvaluatedCandidate], size: int, elite_count: int,
                           rng: random.Random) -> list[EvaluatedCandidate]:  # fmt: skip
    ranked = unique_by_structure(candidates)
    survivors = elites(ranked, elite_count)
    chosen = {c.key for c in survivors}
    rest = [c for c in ranked if c.key not in chosen]
    while rest and len(survivors) < size:
        total = sum(c.fitness for c in rest)
        if total <= 0:
            survivors.extend(rest[: size - len(survivors)])
            break
        pick, acc = rng.random() * total, 0.0
        for index, candidate in enumerate(rest):
            acc += candidate.fitness
            if acc >= pick:
                survivors.append(rest.pop(index))
                break
    return survivors


def select_parent(population: list[EvaluatedCandidate], rng: random.Random, *, stratified: bool) -> EvaluatedCandidate:
    pool = population
    if stratified:
        pool = rng.choice(list(strata(population).values()))
    contestants = rng.sample(pool, min(TOURNAMENT_SIZE, len(pool)))
    return max(contestants, key=lambda c: c.fitness)


def select_donor(population: list[EvaluatedCandidate], recipient: EvaluatedCandidate,
                 rng: random.Random) -> Optional[EvaluatedCandidate]:  # fmt: skip
    """A crossover donor whose covered set includes items in the recipient's U(p) (binary tournament among them)."""
    gaps = set(recipient.profile.gaps())
    donors = [c for c in population if c.key != recipient.key and gaps & set(c.profile.covered())]
    if not donors:
        return None
    contestants = rng.sample(donors, min(TOURNAMENT_SIZE, len(donors)))
    return max(contestants, key=lambda c: c.fitness)
