"""Disjoint LinUCB contextual bandit for the routing decision (paper Section 4.2, Eq. 8).

Each arm a (reuse, request evidence, synthesize) keeps its own ridge-regression statistics
A_a = lambda * I + sum x x^T and b_a = sum r x, updated only from incidents where it was taken.
The layer selects

    a* = argmax_a  theta_a^T x + alpha * sqrt(x^T A_a^-1 x),   theta_a = A_a^-1 b_a

where the first term is the estimated reward and the second a confidence bonus that makes
under-explored arms get tried. The context x has d = 9 entries: the eight routing features of
:func:`context_vector` and a bias of 1. The whole policy is a few hundred numbers.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from app.schemas.routing import RoutingAction, RoutingTerms

ARMS = (RoutingAction.REUSE, RoutingAction.REQUEST_EVIDENCE, RoutingAction.SYNTHESIZE)  # also the tie-break order
FEATURES = ("coverage", "counterfactual_coverage", "entropy", "requestability", "unmet_fraction",
            "smoothed_success", "top_similarity", "library_size", "bias")  # fmt: skip
DIMENSION = len(FEATURES)


def context_vector(terms: RoutingTerms, top_similarity: float, library_size: int, size_scale: float) -> list[float]:
    """x: coverage, Cov+, H, rho, M, smoothed success, top similarity, library size and a bias.

    The library size enters as n / (n + size_scale) so every feature lies in [0, 1]
    (implementation choice; the paper does not say how it is scaled).
    """
    size = library_size / (library_size + size_scale) if library_size > 0 else 0.0
    return [terms.coverage, terms.counterfactual_coverage, terms.entropy, terms.requestability, terms.unmet_fraction,
            terms.smoothed_success, max(0.0, min(1.0, top_similarity)), size, 1.0]  # fmt: skip


@dataclass
class ArmState:
    a: list[list[float]]  # A_a (d x d)
    b: list[float]  # b_a (d)
    pulls: int = 0

    @classmethod
    def fresh(cls, dimension: int, ridge: float) -> "ArmState":
        return cls(a=(np.eye(dimension) * ridge).tolist(), b=[0.0] * dimension)


@dataclass(frozen=True)
class ArmScore:
    estimate: float  # theta_a^T x
    bonus: float  # sqrt(x^T A_a^-1 x)
    ucb: float  # estimate + alpha * bonus


@dataclass
class LinUCBPolicy:
    alpha: float = 1.0
    ridge: float = 1.0
    dimension: int = DIMENSION
    arms: dict[RoutingAction, ArmState] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for arm in ARMS:
            self.arms.setdefault(arm, ArmState.fresh(self.dimension, self.ridge))
        self._lock = threading.Lock()

    def scores(self, x: Iterable[float], allowed: Iterable[RoutingAction] = ARMS) -> dict[RoutingAction, ArmScore]:
        vector = self._vector(x)
        allowed = set(allowed)
        result = {}
        with self._lock:
            for arm in ARMS:
                if arm not in allowed:
                    continue
                state = self.arms[arm]
                inverse = np.linalg.inv(np.array(state.a))
                theta = inverse @ np.array(state.b)
                estimate = float(theta @ vector)
                bonus = float(np.sqrt(max(0.0, vector @ inverse @ vector)))
                result[arm] = ArmScore(estimate, bonus, estimate + self.alpha * bonus)
        return result

    def select(self, x: Iterable[float], allowed: Iterable[RoutingAction] = ARMS) -> RoutingAction:
        """Eq. 8 over the ``allowed`` arms; ties go to the cheaper action (reuse, request, synthesize)."""
        scores = self.scores(x, allowed)
        if not scores:
            raise ValueError("no arm is allowed")
        return max(scores, key=lambda arm: (scores[arm].ucb, -ARMS.index(arm)))

    def update(self, arm: RoutingAction, x: Iterable[float], reward: float) -> None:
        """A_a += x x^T, b_a += r x (theta_a is refit from them by ridge regression on demand)."""
        vector = self._vector(x)
        with self._lock:
            state = self.arms[arm]
            state.a = (np.array(state.a) + np.outer(vector, vector)).tolist()
            state.b = (np.array(state.b) + reward * vector).tolist()
            state.pulls += 1

    def _vector(self, x: Iterable[float]) -> np.ndarray:
        vector = np.asarray(list(x), dtype=float)
        if vector.shape != (self.dimension,):
            raise ValueError(f"context must have {self.dimension} entries")
        return vector
