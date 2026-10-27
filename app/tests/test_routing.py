"""Utility-based routing (paper Eq. 1-5) (milestone M7). Candidates come from a stub source here."""

from __future__ import annotations

import math

import pytest

from app.core.config import PROJECT_ROOT
from app.enrichment.availability import AvailabilityProfile
from app.intent.llm_extractor import IntentExtractor
from app.intent.mock_llm_client import MockLLMClient
from app.playbook.vocabulary import Vocabulary
from app.routing import utility as U
from app.routing.router import RetrievedCandidate, Router, StaticCandidateSource
from app.schemas.enums import EvidenceCategory as C
from app.schemas.evidence import EffectiveEvidenceProfile, SimulatedObservation
from app.schemas.playbook import Playbook
from app.schemas.routing import RoutingAction
from app.tests.conftest import good_output, make_incident
from app.tests.test_playbook_dag import seeds

DATA = PROJECT_ROOT / "app" / "data"


@pytest.fixture(scope="module")
def vocabulary(taxonomy) -> Vocabulary:
    return Vocabulary.load(DATA / "action_vocabulary.json", DATA / "intent_goals.json", taxonomy)


@pytest.fixture(scope="module")
def availability() -> AvailabilityProfile:
    return AvailabilityProfile.load(DATA / "evidence_availability.json")


def _seed(intent="suspicious_process_execution") -> Playbook:
    return Playbook.model_validate(next(s for s in seeds() if s["intent_label"] == intent))


def candidate(playbook, score=0.9, successes=0, executions=0, entry="pb-1") -> RetrievedCandidate:
    return RetrievedCandidate(entry_id=entry, playbook=playbook, similarity=score, score=score, successes=successes,
                              executions=executions)  # fmt: skip


# --- Eq. 1-5 ---------------------------------------------------------------------------------


def test_eq1_coverage():
    assert U.evidence_coverage({"a", "b"}, {"c", "d"}, {"a", "c"}, 0.7) == pytest.approx(0.5 * (0.7 + 0.3 * 0.5))
    assert U.evidence_coverage(set(), {"c", "d"}, {"c"}, 0.7) == pytest.approx(0.7 + 0.3 * 0.5)
    assert U.evidence_coverage({"a"}, set(), {"a"}, 0.7) == 1.0
    assert U.evidence_coverage({"a"}, {"c"}, set(), 0.7) == 0.0  # a missing critical item zeroes coverage
    assert U.evidence_coverage(set(), set(), set(), 0.7) == 1.0


def test_eq2_normalised_entropy():
    assert U.normalized_entropy([0.5, 0.5, 0.5, 0.5]) == pytest.approx(1.0)
    assert U.normalized_entropy([0.8, 0.2]) == pytest.approx(-(0.8 * math.log2(0.8) + 0.2 * math.log2(0.2)))
    assert U.normalized_entropy([0.9]) == 0.0
    assert U.normalized_entropy([]) == 1.0
    assert U.normalized_entropy([0.9, -0.3]) < 0.01  # negative similarities are clipped, not inverted


def test_smoothed_success():
    assert U.smoothed_success(0, 0, 3) == 0.5  # neutral for a new playbook
    assert U.smoothed_success(10, 10, 3) == pytest.approx(13 / 16)
    assert U.smoothed_success(0, 10, 3) == pytest.approx(3 / 16)
    assert U.smoothed_success(900, 1000, 3) == pytest.approx(0.9, abs=0.01)  # converges to s / n
    with pytest.raises(ValueError):
        U.smoothed_success(5, 3, 3)


def test_requestability_unmet_fraction_and_utilities():
    critical = {C.MEMORY, C.DNS, C.EMAIL}
    available = {C.MEMORY}
    assert U.unmet_fraction(critical, available) == pytest.approx(2 / 3)
    assert U.unmet_fraction(set(), available) == 0.0
    rho = U.requestability(critical - available, {C.DNS: 0.9, C.EMAIL: 0.3})
    assert rho == pytest.approx(0.6)
    assert U.requestability([], {}) == 0.0
    assert U.obtainable(critical - available, {C.DNS: 0.9, C.EMAIL: 0.3}, 0.5) == {C.DNS}
    assert U.utility_reuse(0.8, 0.25, 0.5) == pytest.approx(0.3)
    assert U.utility_request(2 / 3, 0.6, 0.9, 0.25) == pytest.approx(2 / 3 * 0.6 * 0.9 * 0.75)
    assert U.utility_synthesize(0.2, 0.6, 2 / 3) == pytest.approx(0.8 * (1 - 0.4))


# --- decisions ---------------------------------------------------------------------------------


def _router(vocabulary, availability, candidates) -> Router:
    return Router(StaticCandidateSource(candidates), vocabulary, availability, w=0.7, k=3, top_n=10, tau=0.5)


def test_reuse_when_the_top_candidate_fits_confidently(vocabulary, availability):
    playbook = _seed()
    required = vocabulary.required_evidence(playbook)
    available = set(required.critical) | set(required.supporting)
    router = _router(vocabulary, availability, [candidate(playbook, successes=8, executions=8)])
    decision = router.score("INC-1", router.source.retrieve(None, 10), available, availability.availability)
    assert decision.action == RoutingAction.REUSE
    assert decision.terms.coverage == 1.0 and decision.terms.entropy == 0.0
    assert decision.scores.reuse == pytest.approx(11 / 14)
    assert decision.scores.synthesize == 0.0 and decision.scores.request_evidence == 0.0


def test_request_evidence_when_critical_evidence_is_missing_but_obtainable(vocabulary, availability):
    playbook = _seed()
    required = vocabulary.required_evidence(playbook)
    router = _router(vocabulary, availability, [candidate(playbook)])
    decision = router.score("INC-2", router.source.retrieve(None, 10), set(required.supporting), availability.availability)
    assert decision.action == RoutingAction.REQUEST_EVIDENCE
    assert decision.terms.coverage == 0.0 and decision.terms.unmet_fraction == 1.0
    assert decision.terms.counterfactual_coverage == pytest.approx(1.0)
    assert set(decision.missing_critical) == set(required.critical) == set(decision.requestable)
    expected_rho = sum(availability.of(c) for c in required.critical) / len(required.critical)
    assert decision.scores.request_evidence == pytest.approx(expected_rho)
    assert decision.scores.synthesize == pytest.approx(1 - expected_rho)
    assert "obtainable" in decision.rationale


def test_synthesize_when_the_candidate_does_not_fit_and_evidence_is_unobtainable(vocabulary, availability):
    playbook = _seed()
    router = _router(vocabulary, availability, [candidate(playbook)])
    nothing = {c: 0.0 for c in C}
    decision = router.score("INC-3", router.source.retrieve(None, 10), set(), nothing)
    assert decision.action == RoutingAction.SYNTHESIZE
    assert decision.scores.synthesize == 1.0 and decision.scores.request_evidence == 0.0


def test_empty_library_means_synthesis(vocabulary, availability):
    decision = _router(vocabulary, availability, []).score("INC-4", [], {C.DNS}, availability.availability)
    assert decision.action == RoutingAction.SYNTHESIZE and decision.top_candidate is None
    assert decision.terms.entropy == 1.0 and decision.terms.candidates_retrieved == 0
    assert "No reusable playbook" in decision.rationale


def test_ambiguous_retrieval_suppresses_reuse(vocabulary, availability):
    playbook = _seed()
    required = vocabulary.required_evidence(playbook)
    available = set(required.critical) | set(required.supporting)
    ambiguous = [candidate(playbook, score=0.6, successes=8, executions=8, entry=f"pb-{i}") for i in range(4)]
    decision = _router(vocabulary, availability, ambiguous).score("INC-5", ambiguous, available, availability.availability)
    assert decision.terms.entropy == pytest.approx(1.0) and decision.scores.reuse == pytest.approx(0.0)


@pytest.mark.anyio
async def test_decide_uses_real_evidence_and_never_simulated_unless_allowed(vocabulary, availability, taxonomy, attack_mapper):
    incident = await IntentExtractor(MockLLMClient([good_output()]), taxonomy, attack_mapper).extract(make_incident())
    playbook = _seed()
    required = vocabulary.required_evidence(playbook)
    simulated = [SimulatedObservation(category=c, description="SYNTHETIC x", tool="t", operation="o")
                 for c in required.critical]  # fmt: skip
    evidence = EffectiveEvidenceProfile(incident_id=incident.incident_id, analysis_id=1,
                                        base=incident.evidence_profile, simulated=simulated)  # fmt: skip
    router = _router(vocabulary, availability, [candidate(playbook, successes=5, executions=5)])
    real = router.decide(incident, evidence)
    assert not real.synthetic_basis
    assert set(real.available_categories) == evidence.observed_categories()
    with_simulated = router.decide(incident, evidence, include_simulated=True)
    assert with_simulated.synthetic_basis and with_simulated.terms.coverage >= real.terms.coverage
    blocked = router.decide(incident, evidence, unobtainable=set(required.critical), round_number=2)
    assert blocked.terms.requestability == 0.0 and blocked.round == 2
    assert blocked.action != RoutingAction.REQUEST_EVIDENCE


def test_entropy_on_realistic_similarities_and_optional_sharpening():
    # top-10 cosine similarities observed in an end-to-end run: the right playbook ranks first,
    # but the paper's H is still high, which suppresses reuse and evidence requests.
    observed = [0.462, 0.348, 0.302, 0.241, 0.199, 0.167, 0.159, 0.151, 0.145, 0.116]
    paper = U.normalized_entropy(observed)
    assert paper > 0.9
    assert U.normalized_entropy(observed, 1.0) == paper  # default is exactly Eq. 2
    sharpened = U.normalized_entropy(observed, 6.0)
    assert sharpened < 0.6
    assert U.normalized_entropy([0.5] * 4, 6.0) == pytest.approx(1.0)  # equal candidates stay ambiguous
