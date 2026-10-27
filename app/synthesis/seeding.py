"""Deduplication and adaptive K (paper Section 5.2).

Each pool sample is embedded as its coverage vector c(p) (which expected goals and relevant
evidence categories of the intent it addresses), exact structural duplicates are removed, and
agglomerative hierarchical clustering with average linkage under the Jaccard distance collapses
near-duplicates: the dendrogram is cut at distance ``threshold`` (tau). Each cluster contributes
its member with the highest provisional fitness as a seed, and K is the number of clusters --
clustered samples yield a small K, dispersed samples a larger one.

``k_min`` / ``k_max`` are optional guards (the paper leaves K unclamped; ``k_max=None``).
With coverage-aware initialisation disabled, K is fixed and seeds are a random sample.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional

from app.fitness.coverage import jaccard
from app.fitness.fitness import EvaluatedCandidate, FitnessEvaluator
from app.schemas.playbook import Playbook


@dataclass
class SeedSelection:
    seeds: list[EvaluatedCandidate]
    k: int
    pool_size: int
    admissible: int
    duplicates: int
    clusters: int
    adaptive: bool


def agglomerate(vectors: list[tuple[int, ...]], threshold: float) -> list[list[int]]:
    """Average-linkage agglomerative clustering (Jaccard distance), cut where the closest pair is farther than ``threshold``."""
    clusters = [[i] for i in range(len(vectors))]
    while len(clusters) > 1:
        best = None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                pairs = [(i, j) for i in clusters[a] for j in clusters[b]]
                distance = sum(jaccard(vectors[i], vectors[j]) for i, j in pairs) / len(pairs)
                if best is None or distance < best[0]:
                    best = (distance, a, b)
        if best[0] > threshold:
            break
        _, a, b = best
        clusters[a].extend(clusters[b])
        del clusters[b]
    return clusters


class AdaptiveSeeder:
    def __init__(self, evaluator: FitnessEvaluator, *, k_min: int = 1, k_max: Optional[int] = None,
                 threshold: float = 0.3, fixed_k: int = 4) -> None:  # fmt: skip
        self.evaluator = evaluator
        self.k_min = k_min
        self.k_max = max(k_min, k_max) if k_max is not None else None
        self.threshold = threshold
        self.fixed_k = fixed_k

    def select(self, pool: list[Playbook], intent_label: str, available: Iterable[str], rng: random.Random,
               *, adaptive: bool = True) -> SeedSelection:  # fmt: skip
        available = tuple(available)
        evaluated = [c for c in (self.evaluator.evaluate(p, intent_label, available) for p in pool) if c is not None]
        unique: dict[str, EvaluatedCandidate] = {}
        for candidate in evaluated:
            kept = unique.get(candidate.key)
            if kept is None or candidate.fitness > kept.fitness:
                unique[candidate.key] = candidate
        candidates = list(unique.values())
        duplicates = len(evaluated) - len(candidates)

        if not adaptive:
            seeds = rng.sample(candidates, min(self.fixed_k, len(candidates)))
            return SeedSelection(seeds, len(seeds), len(pool), len(evaluated), duplicates, 0, False)

        clusters = agglomerate([c.profile.vector for c in candidates], self.threshold) if candidates else []
        representatives = sorted((max((candidates[i] for i in cluster), key=lambda c: c.fitness) for cluster in clusters),
                                 key=lambda c: c.fitness, reverse=True)[: self.k_max]  # fmt: skip
        if len(representatives) < self.k_min:
            chosen = {c.key for c in representatives}
            for candidate in sorted(candidates, key=lambda c: c.fitness, reverse=True):
                if len(representatives) >= self.k_min:
                    break
                if candidate.key not in chosen:
                    representatives.append(candidate)
                    chosen.add(candidate.key)
        return SeedSelection(representatives, len(representatives), len(pool), len(evaluated), duplicates,
                             len(clusters), True)  # fmt: skip
