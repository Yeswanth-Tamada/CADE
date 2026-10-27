"""The routing layer's quantities (paper Section 4, Eq. 1-5), as pure functions.

The paper calls these *utility scores*: heuristic multiplicative composites of normalised terms
(an action scores well only when every condition it depends on holds), not expectations under a
probability model.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

from app.schemas.enums import EvidenceCategory

_EPSILON = 1e-9


def evidence_coverage(critical: set, supporting: set, available: set, w: float) -> float:
    """Eq. 1: Cov(p*, A) = |Rc n A|/|Rc| * (w + (1 - w) * |Rs n A|/|Rs|).

    An empty requirement set is fully covered (factor 1): a playbook with no critical
    requirements can always run.
    """
    critical_fraction = len(critical & available) / len(critical) if critical else 1.0
    supporting_fraction = len(supporting & available) / len(supporting) if supporting else 1.0
    return critical_fraction * (w + (1 - w) * supporting_fraction)


def normalized_entropy(scores: Sequence[float], sharpness: float = 1.0) -> float:
    """Eq. 2: H = -1/log2(N) * sum(pi_k log2 pi_k), pi_k = s_k / sum_j s_j, over the top-N scores.

    Scores are clipped to a small positive value (cosine similarities can be negative). One
    candidate gives H = 0 (a single, unambiguous match); no candidates gives H = 1.

    ``sharpness`` = 1 is the paper's formula. Because raw cosine similarities of the top N = 10
    candidates are rarely far apart, that H is usually above 0.9, which suppresses reuse and
    evidence requests. ``sharpness`` > 1 (pi_k proportional to s_k ** sharpness) is an opt-in
    implementation option that lets a clearly better match dominate; it is off by default.
    """
    if not scores:
        return 1.0
    if len(scores) == 1:
        return 0.0
    clipped = [max(s, _EPSILON) ** sharpness for s in scores]
    total = sum(clipped)
    probabilities = [s / total for s in clipped]
    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
    return min(1.0, max(0.0, entropy / math.log2(len(scores))))


def smoothed_success(successes: int, executions: int, k: float) -> float:
    """s-hat(p) = (s + k) / (n + 2k): 0.5 for a new playbook, converging to s/n with use."""
    if executions < 0 or successes < 0 or successes > executions:
        raise ValueError("need 0 <= successes <= executions")
    return (successes + k) / (executions + 2 * k)


def unmet_fraction(critical: set, available: set) -> float:
    """M: the fraction of critical requirements not in A (0 when there are none)."""
    return len(critical - available) / len(critical) if critical else 0.0


def requestability(missing: Iterable[EvidenceCategory], availability: Mapping[EvidenceCategory, float]) -> float:
    """rho: mean per-deployment availability over the missing critical categories (0 if none are missing)."""
    missing = list(missing)
    return sum(availability.get(c, 0.0) for c in missing) / len(missing) if missing else 0.0


def obtainable(missing: Iterable[EvidenceCategory], availability: Mapping[EvidenceCategory, float],
               tau: float) -> set[EvidenceCategory]:  # fmt: skip
    return {c for c in missing if availability.get(c, 0.0) >= tau}


def utility_reuse(coverage: float, entropy: float, success: float) -> float:
    """Eq. 3: U_reuse = Cov(p*, A) * (1 - H) * s-hat(p*)."""
    return coverage * (1 - entropy) * success


def utility_request(unmet: float, rho: float, counterfactual: float, entropy: float) -> float:
    """Eq. 4: U_request = M * rho * (Cov+(p*) * (1 - H))."""
    return unmet * rho * (counterfactual * (1 - entropy))


def utility_synthesize(coverage: float, rho: float, unmet: float) -> float:
    """Eq. 5: U_gen = (1 - Cov(p*, A)) * (1 - rho * M)."""
    return (1 - coverage) * (1 - rho * unmet)
