"""Persisted LinUCB state: bootstrap replay, precision, restarts, legacy state and benchmark isolation (paper §4.2-4.3, §6.2)."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from app.core.config import PROJECT_ROOT
from app.enrichment.availability import AvailabilityProfile
from app.evaluation import protocol as P
from app.playbook.vocabulary import Vocabulary
from app.routing.bandit import ARMS, DIMENSION, LinUCBPolicy
from app.routing.learning import RoutingLearner
from app.routing.reward import RewardWeights, percentile, terminal_reward
from app.routing.router import RetrievedCandidate, Router, StaticCandidateSource
from app.schemas.routing import RoutingAction, RoutingReward, StoredRoutingDecision
from app.storage.bandit_repository import BanditRepository
from app.storage.database import Database
from app.tests.conftest import with_settings
from app.tests.test_playbook_dag import pb

DATA = PROJECT_ROOT / "app" / "data"
REUSE, REQUEST, SYNTH = RoutingAction.REUSE, RoutingAction.REQUEST_EVIDENCE, RoutingAction.SYNTHESIZE
NOW = datetime(2026, 10, 26, tzinfo=timezone.utc)
INTENT = "suspicious_process_execution"


@pytest.fixture(scope="module")
def vocabulary(taxonomy) -> Vocabulary:
    return Vocabulary.load(DATA / "action_vocabulary.json", DATA / "intent_goals.json", taxonomy)


@pytest.fixture
def repo_factory(tmp_path):
    databases = []

    def make(name: str = "bandit.db") -> BanditRepository:
        database = Database(f"sqlite:///{(tmp_path / name).as_posix()}")
        database.create_schema()
        databases.append(database)
        return BanditRepository(database)

    yield make
    for database in databases:
        database.dispose()


def _learner(repository, bootstrap: int = 20) -> RoutingLearner:
    return RoutingLearner(repository, LinUCBPolicy(), enabled=True, bootstrap=bootstrap, latency_max_seconds=None,
                          latency_fallback_seconds=600.0, learn_from_simulated=False)  # fmt: skip


def _router(vocabulary, learner) -> Router:
    return Router(StaticCandidateSource(), vocabulary, AvailabilityProfile(version="t"), learner=learner,
                  library_size=lambda: 10)  # fmt: skip


def _candidates(vocabulary, similarity: float) -> list[RetrievedCandidate]:
    playbook = pb("n1:extract_entities n2:query_process_events n3:generate_incident_report",
                  "n1>n2:host_id n2>n3:findings", intent=INTENT)  # fmt: skip
    return [RetrievedCandidate(entry_id="E1", playbook=playbook, similarity=similarity, score=similarity)]


# Twenty bootstrap incidents with non-round telemetry (full precision matters for the replay).
LATENCIES = [6.123456, 238.98765, 8.4321, 239.5, 6.000001, 241.25, 237.1, 6.5, 240.0, 7.25,
             238.0, 6.9, 236.75, 8.8, 239.9, 6.2, 242.125, 238.5, 6.05, 240.6]  # fmt: skip
COVERAGE = [1 / 3 + i / 97 for i in range(20)]


def _run_bootstrap(router, learner, vocabulary, start_id: int = 1, count: int = 20, decisions=None):
    """Route and credit ``count`` incidents; returns the stored decisions (one per incident)."""
    decisions = decisions if decisions is not None else []
    for i in range(count):
        n = start_id + i
        decision = router.score(f"INC-{n}", _candidates(vocabulary, 0.6 + 0.01 * (n % 30)), set(), {})
        stored = StoredRoutingDecision(**decision.model_dump(), decision_id=n, created_at=NOW)
        decisions.append(stored)
        index = (n - 1) % 20
        learner.credit(incident_id=f"INC-{n}", run_id=f"RUN-{n}", chain=[stored], terminal_arm=decision.action,
                       retained=True, execution_coverage=COVERAGE[index], latency_seconds=LATENCIES[index],
                       synthetic=False)  # fmt: skip
    return decisions


def _reference(decisions, latency_max: float) -> LinUCBPolicy:
    """A fresh policy fed the paper's Eq. 9 rewards directly, oldest first."""
    policy = LinUCBPolicy()
    for d in decisions:
        index = (d.decision_id - 1) % 20
        value = terminal_reward(retained=True, coverage=COVERAGE[index], latency_seconds=LATENCIES[index],
                                latency_max_seconds=latency_max, weights=RewardWeights()).value  # fmt: skip
        policy.update(d.action, d.features, value)
    return policy


def _same(a: LinUCBPolicy, b: LinUCBPolicy) -> bool:
    return all(np.allclose(a.arms[x].a, b.arms[x].a, atol=1e-12, rtol=0) and np.allclose(a.arms[x].b, b.arms[x].b, atol=1e-12, rtol=0)
               and a.arms[x].pulls == b.arms[x].pulls for x in ARMS)  # fmt: skip


# --- items 1-6: the bootstrap -----------------------------------------------------------------------


def test_bootstrap_rule_lmax_buffering_precision_and_the_state_incident_21_starts_from(repo_factory, vocabulary, monkeypatch):
    repository = repo_factory()
    learner = _learner(repository)
    router = _router(vocabulary, learner)
    applied: list[tuple[str, float]] = []
    original = learner.policy.update

    def spy(arm, x, reward):
        applied.append((arm.value, reward))
        original(arm, x, reward)

    monkeypatch.setattr(learner.policy, "update", spy)
    decisions = _run_bootstrap(router, learner, vocabulary, count=19)
    # (1) the hand-designed rule chose all of them; (3) nothing was applied, everything is logged as provisional
    assert all(d.policy == "rule" and d.action == d.rule_action for d in decisions)
    assert applied == [] and all(a.pulls == 0 for a in learner.policy.arms.values())
    assert len(repository.provisional()) == 19 and not learner.latency_max_final

    decisions = _run_bootstrap(router, learner, vocabulary, start_id=20, count=1, decisions=decisions)
    assert decisions[-1].policy == "rule"
    # (2) L_max is the 90th percentile of the observed bootstrap latencies
    latency_max = percentile(LATENCIES, 90)
    assert learner.latency_max() == pytest.approx(latency_max, abs=1e-12)
    # (4) full-precision rewards, recomputed with that L_max and applied oldest first
    expected = [terminal_reward(retained=True, coverage=COVERAGE[i], latency_seconds=LATENCIES[i],
                                latency_max_seconds=latency_max, weights=RewardWeights()).value for i in range(20)]  # fmt: skip
    assert [v for _, v in applied] == pytest.approx(expected, abs=1e-12)
    # (5) the rewards reported (stored) are exactly the rewards the bandit learned from
    stored = {r.decision_id: r.reward for r in repository.all()}
    assert [stored[d.decision_id] for d in decisions] == pytest.approx([v for _, v in applied], abs=1e-12)
    assert not repository.provisional() and all(r.components["latency_max_seconds"] == latency_max for r in repository.all())
    # (6) incident 21 is routed by LinUCB from exactly the corrected state
    assert _same(learner.policy, _reference(decisions, latency_max))
    nxt = router.score("INC-21", _candidates(vocabulary, 0.77), set(), {})
    assert nxt.policy == "linucb"
    reference = _reference(decisions, latency_max).scores(nxt.features, [SYNTH, REUSE])
    assert nxt.ucb == {arm.value: round(score.ucb, 6) for arm, score in reference.items()}


# --- items 7-9: equivalence, no duplicates, restarts ----------------------------------------------


def test_uninterrupted_restarted_and_fresh_learners_reach_the_same_state(repo_factory, vocabulary):
    straight = _learner(repo_factory("a.db"))
    decisions = _run_bootstrap(_router(vocabulary, straight), straight, vocabulary)

    repository = repo_factory("b.db")
    first = _learner(repository)
    _run_bootstrap(_router(vocabulary, first), first, vocabulary, count=11)
    resumed = _learner(repository)  # an application restart in the middle of the bootstrap
    resumed.load()
    assert len(repository.provisional()) == 11 and all(a.pulls == 0 for a in resumed.policy.arms.values())
    _run_bootstrap(_router(vocabulary, resumed), resumed, vocabulary, start_id=12, count=9)

    assert _same(straight.policy, resumed.policy)  # (7) restarted == uninterrupted
    assert _same(straight.policy, _reference(decisions, percentile(LATENCIES, 90)))  # (7) == a fresh policy fed Eq. 9


def test_replaying_twice_and_crediting_twice_never_duplicates_an_update(repo_factory, vocabulary):
    repository = repo_factory()
    learner = _learner(repository)
    router = _router(vocabulary, learner)
    decisions = _run_bootstrap(router, learner, vocabulary, count=19)
    again = decisions[3]
    learner.credit(incident_id="INC-4", run_id="RUN-4", chain=[again], terminal_arm=again.action, retained=True,
                   execution_coverage=COVERAGE[3], latency_seconds=LATENCIES[3], synthetic=False)  # same decision again
    _run_bootstrap(router, learner, vocabulary, start_id=20, count=1, decisions=decisions)
    pulls = {a: s.pulls for a, s in learner.policy.arms.items()}
    assert sum(pulls.values()) == 20  # (8) one update per incident, not 21
    learner.load()  # replay again on reload: nothing is provisional any more
    restarted = _learner(repository)
    restarted.load()
    assert {a: s.pulls for a, s in learner.policy.arms.items()} == pulls and _same(learner.policy, restarted.policy)
    # (9) a restart after the bootstrap restores the corrected state and reports nothing stale
    assert restarted.last_audit.stale == () and restarted.last_audit.consistent and restarted.state_warning() is None


# --- item 10: state written before the fix --------------------------------------------------------------


def _write_legacy_state(repository: BanditRepository, count: int) -> LinUCBPolicy:
    """What the pre-2026-10-26 learner stored: bootstrap rewards applied at once with the 600 s fallback, unflagged."""
    policy = LinUCBPolicy()
    rewards = []
    for i in range(count):
        x = [0.5] * (DIMENSION - 1) + [1.0]
        legacy = terminal_reward(retained=True, coverage=round(COVERAGE[i], 4), latency_seconds=round(LATENCIES[i], 3),
                                 latency_max_seconds=600.0, weights=RewardWeights())  # fmt: skip
        policy.update(SYNTH, x, legacy.value)
        components = {"retained": True, "execution_coverage": round(COVERAGE[i], 4), "latency_seconds": round(LATENCIES[i], 3),
                      "latency_max_seconds": 600.0, "request_penalty": 0.0}  # fmt: skip
        rewards.append(RoutingReward(decision_id=i + 1, incident_id=f"OLD-{i}", run_id=f"R{i}", arm=SYNTH, terminal=True,
                                     reward=legacy.value, features=x, components=components, created_at=NOW))  # fmt: skip
    repository.credit(rewards, policy.arms)
    return policy


def test_legacy_state_is_reported_never_changed_silently_and_rebuilt_only_on_request(repo_factory):
    repository = repo_factory()
    legacy = _write_legacy_state(repository, 20)
    learner = _learner(repository)
    learner.load()
    audit = learner.last_audit
    assert audit.stale == tuple(range(1, 21)) and audit.consistent and audit.latency_max is not None
    assert "superseded L_max" in learner.state_warning()
    assert _same(learner.policy, legacy)  # loading did not touch the stale arms ...
    legacy_values = [terminal_reward(retained=True, coverage=round(COVERAGE[i], 4), latency_seconds=round(LATENCIES[i], 3),
                                     latency_max_seconds=600.0, weights=RewardWeights()).value for i in range(20)]  # fmt: skip
    assert [r.reward for r in repository.all()] == pytest.approx(legacy_values, abs=1e-12)  # ... nor the stored rewards

    plan = learner.rebuild()  # dry run
    assert not plan.applied and _same(learner.policy, legacy) and repository.all()[0].components["latency_max_seconds"] == 600.0
    rebuilt = learner.rebuild(apply=True)
    assert rebuilt.applied and learner.state_warning() is None
    final = percentile([round(v, 3) for v in LATENCIES], 90)
    assert all(r.components["latency_max_seconds"] == final for r in repository.all())
    reference = LinUCBPolicy()
    for r in repository.all():
        reference.update(SYNTH, r.features, terminal_reward(retained=True, coverage=r.components["execution_coverage"],
                                                            latency_seconds=r.components["latency_seconds"],
                                                            latency_max_seconds=final, weights=RewardWeights()).value)  # fmt: skip
    assert _same(learner.policy, reference)
    reloaded = _learner(repository)
    reloaded.load()
    assert _same(reloaded.policy, reference) and reloaded.state_warning() is None


def test_legacy_rewards_during_an_incomplete_bootstrap_become_provisional_and_unexplained_arms_need_force(repo_factory):
    repository = repo_factory()
    _write_legacy_state(repository, 2)  # like cade_e2e.db: two rewards applied with the fallback
    learner = _learner(repository)
    learner.load()
    assert learner.last_audit.stale == (1, 2) and learner.last_audit.latency_max is None
    plan = learner.rebuild(apply=True)
    assert all(c["new_provisional"] for c in plan.changes) and all(a.pulls == 0 for a in learner.policy.arms.values())
    assert len(repository.provisional()) == 2 and learner.state_warning() is None

    other = repo_factory("tampered.db")
    _write_legacy_state(other, 3)
    tampered = _learner(other)
    tampered.load()
    tampered.policy.arms[REUSE].pulls = 7  # stored arms no longer explained by the reward log
    assert not tampered.audit().consistent
    with pytest.raises(ValueError):
        tampered.rebuild(apply=True)
    assert tampered.rebuild(apply=True, force=True).applied


def test_health_reports_stale_routing_state(make_client, settings):
    database = Database(settings.database_url)
    database.create_schema()
    _write_legacy_state(BanditRepository(database), 2)
    database.dispose()
    client = make_client([])
    warnings = client.get("/health").json()["playbooks"]["warnings"]
    assert any("superseded L_max" in w for w in warnings)


# --- benchmark isolation ------------------------------------------------------------------------


def test_benchmark_learners_are_fresh_and_never_use_the_configured_database(settings, tmp_path):
    configured = with_settings(settings, database_url=f"sqlite:///{(tmp_path / 'production.db').as_posix()}")
    a = P.fresh_learner(configured, tmp_path / "work" / "bandit_0.db")
    b = P.fresh_learner(configured, tmp_path / "work" / "bandit_1.db")
    assert not (tmp_path / "production.db").exists()
    assert a.repository is not b.repository and a.rewarded_incidents == b.rewarded_incidents == 0
    assert all(s.pulls == 0 and np.allclose(s.a, np.eye(DIMENSION)) for s in a.policy.arms.values())
    a.repository.db.dispose()
    b.repository.db.dispose()
