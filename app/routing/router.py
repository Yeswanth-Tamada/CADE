"""The adaptive routing layer (paper Section 4).

Given an incident's intent and current evidence, retrieve the top-N library candidates by
embedding similarity, take the top candidate p* and compute the terms of Eq. 1-6. The
hand-designed rule (Eq. 7) takes the action with the highest utility score; with a
:class:`~app.routing.learning.RoutingLearner` the same terms (plus top similarity and library
size) form the context of a LinUCB bandit (Eq. 8), which chooses once its bootstrap period is
over. Evidence acquisition is a first-class action, scored by the counterfactual coverage Cov+
that obtainable missing evidence would unlock.

Implementation choices where the paper is silent (documented in the README):

* A is the set of evidence categories observed so far (analysis + real tools + analyst);
  simulated evidence counts only when the caller allows it, and the decision is then flagged.
* R_c / R_s of a playbook are the categories its actions consume: critical if any action marks
  the category critical or the incident's intent marks it critical in the ontology (paper
  Section 6.1), see ``Vocabulary.required_evidence``.
* Cov+ adds the missing critical categories whose availability is at least ``tau``.
* With no candidates (empty library) Cov = 0 and H = 1, so synthesis wins.
* Categories already requested without success are passed as ``unobtainable`` (availability 0).
* Ties are broken towards the cheaper action: reuse, then evidence request, then synthesis.
* ``allow_request=False`` removes the request arm once the per-incident request cap is reached;
  the bandit also only considers reuse with a candidate and a request with obtainable evidence.
* ``requirements`` optionally replaces how R_c / R_s of p* are derived (default:
  ``Vocabulary.required_evidence``); the paper-matched benchmark passes the stored playbook's own
  intent ontology (``app/evaluation/paper968.py``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Optional, Protocol

from app.enrichment.availability import AvailabilityProfile
from app.playbook.vocabulary import Vocabulary
from app.routing import utility as U
from app.routing.bandit import context_vector
from app.routing.learning import RoutingLearner
from app.schemas.enums import EvidenceCategory
from app.schemas.evidence import EffectiveEvidenceProfile
from app.schemas.incident import CanonicalIncident
from app.schemas.playbook import Playbook, RequiredEvidence
from app.schemas.routing import RoutingAction, RoutingCandidate, RoutingDecision, RoutingScores, RoutingTerms
from app.schemas.taxonomy import IntentTaxonomy


@dataclass(frozen=True)
class RetrievalQuery:
    intent_label: str
    intent_description: str
    observed: tuple[EvidenceCategory, ...]
    missing: tuple[EvidenceCategory, ...]
    techniques: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievedCandidate:
    entry_id: str
    playbook: Playbook
    similarity: float
    score: float
    successes: int = 0
    executions: int = 0
    deprioritized: bool = False

    def summary(self) -> RoutingCandidate:
        return RoutingCandidate(entry_id=self.entry_id, name=self.playbook.name, intent_label=self.playbook.intent_label,
                                similarity=self.similarity, score=self.score, successes=self.successes,
                                executions=self.executions, deprioritized=self.deprioritized)  # fmt: skip


class CandidateSource(Protocol):
    def retrieve(self, query: RetrievalQuery, top_n: int) -> list[RetrievedCandidate]:
        """Up to ``top_n`` reuse-eligible candidates, best first."""


@dataclass
class StaticCandidateSource:
    """A fixed candidate list (tests, and routing before the library exists)."""

    candidates: list[RetrievedCandidate] = field(default_factory=list)

    def retrieve(self, query: RetrievalQuery, top_n: int) -> list[RetrievedCandidate]:
        return sorted(self.candidates, key=lambda c: c.score, reverse=True)[:top_n]


def build_query(incident: CanonicalIncident, evidence: EffectiveEvidenceProfile, *, include_simulated: bool) -> RetrievalQuery:
    key = lambda c: c.value  # noqa: E731
    return RetrievalQuery(
        intent_label=incident.intent.label,
        intent_description=incident.intent.description,
        observed=tuple(sorted(evidence.observed_categories(include_simulated=include_simulated), key=key)),
        missing=tuple(sorted({i.category for i in evidence.outstanding_missing(include_simulated=include_simulated)}, key=key)),
        techniques=tuple(t.technique_id for t in incident.intent.attack_techniques),
    )


class Router:
    def __init__(
        self,
        source: CandidateSource,
        vocabulary: Vocabulary,
        availability: AvailabilityProfile,
        *,
        w: float = 0.7,
        k: float = 3,
        top_n: int = 10,
        tau: float = 0.5,
        sharpness: float = 1.0,
        taxonomy: Optional[IntentTaxonomy] = None,
        learner: Optional[RoutingLearner] = None,
        library_size: Callable[[], int] = lambda: 0,
        requirements: Optional[Callable[[Playbook, frozenset[EvidenceCategory]], RequiredEvidence]] = None,
    ) -> None:
        self.source = source
        self.vocabulary = vocabulary
        self.availability = availability
        self.w, self.k, self.top_n, self.tau = w, k, top_n, tau
        self.sharpness = sharpness
        self.taxonomy = taxonomy
        self.learner = learner
        self.library_size = library_size
        self.requirements = requirements or vocabulary.required_evidence

    def decide(
        self,
        incident: CanonicalIncident,
        evidence: EffectiveEvidenceProfile,
        *,
        include_simulated: bool = False,
        unobtainable: Iterable[EvidenceCategory] = (),
        round_number: int = 1,
        allow_request: bool = True,
    ) -> RoutingDecision:
        available = evidence.observed_categories(include_simulated=include_simulated)
        synthetic = include_simulated and bool(available - evidence.observed_categories())
        candidates = self.source.retrieve(build_query(incident, evidence, include_simulated=include_simulated), self.top_n)
        availability = dict(self.availability.availability)
        for category in unobtainable:
            availability[category] = 0.0
        definition = self.taxonomy.get(incident.intent.label) if self.taxonomy else None
        critical = frozenset(definition.critical_evidence) if definition else frozenset()
        return self.score(incident.incident_id, candidates, available, availability, synthetic, round_number,
                          critical_categories=critical, allow_request=allow_request)  # fmt: skip

    def score(
        self,
        incident_id: str,
        candidates: list[RetrievedCandidate],
        available: set[EvidenceCategory],
        availability: dict[EvidenceCategory, float],
        synthetic: bool = False,
        round_number: int = 1,
        *,
        critical_categories: frozenset[EvidenceCategory] = frozenset(),
        allow_request: bool = True,
    ) -> RoutingDecision:
        key = lambda c: c.value  # noqa: E731
        top: Optional[RetrievedCandidate] = candidates[0] if candidates else None
        if top is None:
            critical, supporting = set(), set()
            coverage, entropy, success = 0.0, 1.0, 0.0
        else:
            required = self.requirements(top.playbook, critical_categories)
            critical, supporting = set(required.critical), set(required.supporting)
            coverage = U.evidence_coverage(critical, supporting, available, self.w)
            entropy = U.normalized_entropy([c.score for c in candidates], self.sharpness)
            success = U.smoothed_success(top.successes, top.executions, self.k)
        missing = critical - available
        unmet = U.unmet_fraction(critical, available) if top else 0.0
        rho = U.requestability(missing, availability)
        requestable = U.obtainable(missing, availability, self.tau)
        counterfactual = U.evidence_coverage(critical, supporting, available | requestable, self.w) if top else 0.0

        scores = RoutingScores(
            reuse=U.utility_reuse(coverage, entropy, success),
            request_evidence=U.utility_request(unmet, rho, counterfactual, entropy),
            synthesize=U.utility_synthesize(coverage, rho, unmet),
        )
        ranked = [(scores.reuse, 0, RoutingAction.REUSE), (scores.request_evidence, 1, RoutingAction.REQUEST_EVIDENCE),
                  (scores.synthesize, 2, RoutingAction.SYNTHESIZE)]  # fmt: skip
        if not allow_request:  # the per-incident cap on evidence requests is reached
            ranked = [r for r in ranked if r[2] != RoutingAction.REQUEST_EVIDENCE]
        rule_action = max(ranked, key=lambda r: (r[0], -r[1]))[2]
        terms = RoutingTerms(coverage=coverage, counterfactual_coverage=counterfactual, entropy=entropy,
                             requestability=rho, unmet_fraction=unmet, smoothed_success=success, w=self.w, k=self.k,
                             tau=self.tau, candidates_retrieved=len(candidates))  # fmt: skip

        # Learned routing (Eq. 8) over the arms that can act: reuse needs a candidate, a request
        # needs obtainable missing evidence (and the cap not to be reached).
        action, policy, features, ucb = rule_action, "rule", [], {}
        if self.learner is not None:
            features = context_vector(terms, top.similarity if top else 0.0, self.library_size(),
                                      self.learner.size_scale)  # fmt: skip
            allowed = [RoutingAction.SYNTHESIZE] + ([RoutingAction.REUSE] if top else []) + (
                [RoutingAction.REQUEST_EVIDENCE] if allow_request and requestable else [])
            arm_scores = self.learner.policy.scores(features, allowed)
            ucb = {arm.value: round(s.ucb, 6) for arm, s in arm_scores.items()}
            if self.learner.active:
                action, policy = self.learner.policy.select(features, allowed), "linucb"
        rationale = self._rationale(action, top, terms, scores, missing, requestable)
        if policy == "linucb":
            rationale = (f"The learned policy (LinUCB) chose {action.value} (upper confidence bound "
                         f"{ucb[action.value]:.3f}); the hand-designed rule would choose {rule_action.value}. {rationale}")  # fmt: skip
        return RoutingDecision(
            incident_id=incident_id,
            action=action,
            scores=scores,
            terms=terms,
            top_candidate=top.summary() if top else None,
            candidates=[c.summary() for c in candidates],
            available_categories=sorted(available, key=key),
            required_critical=sorted(critical, key=key),
            required_supporting=sorted(supporting, key=key),
            missing_critical=sorted(missing, key=key),
            requestable=sorted(requestable, key=key),
            synthetic_basis=synthetic,
            round=round_number,
            rationale=rationale,
            policy=policy,
            rule_action=rule_action,
            features=features,
            ucb=ucb,
        )

    @staticmethod
    def _rationale(action, top, terms: RoutingTerms, scores: RoutingScores, missing, requestable) -> str:
        if top is None:
            return "No reusable playbook was retrieved (empty or ineligible library), so a new playbook must be synthesized."
        head = (f"Top candidate '{top.playbook.name}' covers {terms.coverage:.2f} of its evidence needs "
                f"(retrieval entropy {terms.entropy:.2f}, smoothed success {terms.smoothed_success:.2f}).")  # fmt: skip
        if action == RoutingAction.REUSE:
            return f"{head} Reuse scores highest ({scores.reuse:.3f})."
        if action == RoutingAction.REQUEST_EVIDENCE:
            names = ", ".join(sorted(c.value for c in requestable)) or "none"
            return (f"{head} Missing critical evidence ({', '.join(sorted(c.value for c in missing))}) is obtainable "
                    f"(requestable: {names}); with it coverage would be {terms.counterfactual_coverage:.2f}. "
                    f"Requesting evidence scores highest ({scores.request_evidence:.3f}).")  # fmt: skip
        return f"{head} The candidate does not fit and a request would not repair it; synthesis scores highest ({scores.synthesize:.3f})."
