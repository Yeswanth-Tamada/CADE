"""Learned routing: LinUCB (paper Eq. 8), the execution-telemetry reward (Eq. 9) and the training loop."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from app.core.config import PROJECT_ROOT
from app.enrichment.availability import AvailabilityProfile
from app.playbook.vocabulary import Vocabulary
from app.routing.bandit import ARMS, DIMENSION, FEATURES, LinUCBPolicy, context_vector
from app.routing.learning import RoutingLearner, response_chain
from app.routing.reward import RewardWeights, execution_coverage, percentile, terminal_reward
from app.routing.router import RetrievedCandidate, Router, StaticCandidateSource
from app.schemas.response import NodeResult, NodeStatus
from app.schemas.routing import RoutingAction, RoutingScores, RoutingTerms, StoredRoutingDecision
from app.storage.bandit_repository import BanditRepository
from app.storage.database import Database
from app.tests.conftest import good_output, with_settings
from app.tests.test_fitness import SMALL
from app.tests.test_playbook_dag import pb
from app.tests.test_response_api import analysed

DATA = PROJECT_ROOT / "app" / "data"
INTENT = "suspicious_process_execution"
REUSE, REQUEST, SYNTH = RoutingAction.REUSE, RoutingAction.REQUEST_EVIDENCE, RoutingAction.SYNTHESIZE


@pytest.fixture(scope="module")
def vocabulary(taxonomy) -> Vocabulary:
    return Vocabulary.load(DATA / "action_vocabulary.json", DATA / "intent_goals.json", taxonomy)


@pytest.fixture
def repository(settings):
    database = Database(settings.database_url)
    database.create_schema()
    yield BanditRepository(database)
    database.dispose()


def _terms(**values) -> RoutingTerms:
    base = dict(coverage=0.8, counterfactual_coverage=0.9, entropy=0.4, requestability=0.5, unmet_fraction=0.2,
                smoothed_success=0.5, w=0.7, k=3, tau=0.5, candidates_retrieved=3)  # fmt: skip
    return RoutingTerms(**{**base, **values})


def _decision(decision_id: int, action: RoutingAction, round_number: int, *, features=None,
              created_at=None) -> StoredRoutingDecision:  # fmt: skip
    return StoredRoutingDecision(
        incident_id="INC-1", action=action, scores=RoutingScores(reuse=0, request_evidence=0, synthesize=0),
        terms=_terms(), rationale="test", round=round_number, decision_id=decision_id,
        created_at=created_at or datetime(2026, 10, 23, tzinfo=timezone.utc),
        features=[0.5] * DIMENSION if features is None else features,
    )  # fmt: skip


# --- LinUCB -----------------------------------------------------------------------------------


def test_linucb_matches_a_hand_computed_update():
    policy = LinUCBPolicy(alpha=1.0, ridge=1.0, dimension=2)
    fresh = policy.scores([1.0, 0.0])[REUSE]
    assert (fresh.estimate, fresh.bonus, fresh.ucb) == (0.0, 1.0, 1.0)  # A = I, b = 0
    policy.update(REUSE, [1.0, 0.0], 1.0)  # A = diag(2, 1), b = (1, 0) -> theta = (0.5, 0)
    after = policy.scores([1.0, 0.0])[REUSE]
    assert after.estimate == pytest.approx(0.5) and after.bonus == pytest.approx(np.sqrt(0.5))
    assert after.ucb == pytest.approx(0.5 + np.sqrt(0.5))
    assert policy.arms[REUSE].pulls == 1 and policy.arms[SYNTH].pulls == 0  # disjoint: only the taken arm learns


def test_linucb_explores_then_exploits_and_respects_allowed_arms():
    policy = LinUCBPolicy(alpha=1.0, dimension=2)
    x = [1.0, 1.0]
    assert policy.select(x) == REUSE  # all equal: ties go to the cheaper action
    for _ in range(30):
        policy.update(REUSE, x, 0.1)
        policy.update(SYNTH, x, 0.9)
    assert policy.select(x, allowed=[REUSE, SYNTH]) == SYNTH  # exploitation between the tried arms
    assert policy.select(x) == REQUEST  # the never-tried arm's bonus sqrt(2) wins: exploration
    assert policy.select(x, allowed=[REUSE]) == REUSE
    with pytest.raises(ValueError):
        policy.scores([1.0, 0.0, 0.0])


def test_context_has_eight_features_and_a_bias():
    x = context_vector(_terms(), top_similarity=0.83, library_size=50, size_scale=50.0)
    assert len(x) == DIMENSION == 9 and FEATURES[-1] == "bias" and x[-1] == 1.0
    assert x[:6] == [0.8, 0.9, 0.4, 0.5, 0.2, 0.5] and x[6] == 0.83 and x[7] == 0.5
    assert context_vector(_terms(), top_similarity=-0.2, library_size=0, size_scale=50.0)[6:8] == [0.0, 0.0]


# --- reward -------------------------------------------------------------------------------------


def test_reward_follows_equation_9():
    weights = RewardWeights()  # 0.6, 0.3, 0.1, c_q = 1
    reward = terminal_reward(retained=True, coverage=0.5, latency_seconds=30, latency_max_seconds=60, weights=weights)
    assert reward.value == pytest.approx(0.6 * 0.5 + 0.3 * 0.5)
    assert reward.as_request(weights).value == pytest.approx(reward.value - 0.1)
    slow = terminal_reward(retained=True, coverage=1.0, latency_seconds=600, latency_max_seconds=60, weights=weights)
    assert slow.value == pytest.approx(0.6)  # the latency term is clipped at 0
    rejected = terminal_reward(retained=False, coverage=1.0, latency_seconds=1, latency_max_seconds=60, weights=weights)
    assert rejected.value == 0.0 and rejected.as_request(weights).value == pytest.approx(-0.1)  # retention gates
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 90) == pytest.approx(9.1)


def test_execution_coverage_counts_goals_answered_by_steps_that_returned_data(vocabulary):
    goals = vocabulary.goals_for(INTENT)
    # SMALL covers triage (n1), process_lineage and persistence_check (n2, n3) of the intent's 6 goals
    results = [NodeResult(node_id="n1", action="extract_entities", status=NodeStatus.SUCCEEDED),
               NodeResult(node_id="n2", action="query_process_events", status=NodeStatus.NO_DATA),
               NodeResult(node_id="n3", action="analyze_process_tree", status=NodeStatus.SKIPPED)]  # fmt: skip
    assert execution_coverage(SMALL, results, goals, vocabulary) == pytest.approx(1 / 6)  # no data is no answer
    results[1] = results[1].model_copy(update={"status": NodeStatus.SUCCEEDED})
    assert execution_coverage(SMALL, results, goals, vocabulary) == pytest.approx(3 / 6)
    assert execution_coverage(SMALL, [], goals, vocabulary) == 0.0


def test_response_chain_reaches_back_to_round_one():
    decisions = [_decision(1, SYNTH, 1), _decision(2, REQUEST, 1), _decision(3, REUSE, 2), _decision(4, REUSE, 1)]
    assert [d.decision_id for d in response_chain(decisions, 3)] == [2, 3]
    assert [d.decision_id for d in response_chain(decisions, 4)] == [4]
    assert response_chain(decisions, 99) == []


# --- learner ------------------------------------------------------------------------------------


def _learner(repository, **changes) -> RoutingLearner:
    options = dict(enabled=True, bootstrap=2, latency_max_seconds=60.0, learn_from_simulated=False)
    return RoutingLearner(repository, LinUCBPolicy(), **{**options, **changes})


def test_requests_are_credited_with_the_terminal_reward_less_their_cost(repository):
    learner = _learner(repository)
    chain = [_decision(10, REQUEST, 1), _decision(11, REUSE, 2)]
    reward, stored = learner.credit(incident_id="INC-1", run_id="RUN-1", chain=chain, terminal_arm=REUSE,
                                    retained=True, execution_coverage=1.0, latency_seconds=0.0, synthetic=False)  # fmt: skip
    assert reward.value == pytest.approx(0.9)
    credited = {r.decision_id: r for r in stored}
    assert credited[11].terminal and credited[11].arm == REUSE and credited[11].reward == pytest.approx(0.9)
    assert not credited[10].terminal and credited[10].arm == REQUEST and credited[10].reward == pytest.approx(0.8)
    assert learner.policy.arms[REUSE].pulls == 1 and learner.policy.arms[REQUEST].pulls == 1
    # crediting the same decisions again changes nothing
    _, again = learner.credit(incident_id="INC-1", run_id="RUN-1", chain=chain, terminal_arm=REUSE, retained=True,
                              execution_coverage=1.0, latency_seconds=0.0, synthetic=False)  # fmt: skip
    assert again == [] and learner.policy.arms[REUSE].pulls == 1


def test_simulated_runs_and_decisions_without_context_do_not_train(repository):
    learner = _learner(repository)
    chain = [_decision(20, SYNTH, 1)]
    assert learner.credit(incident_id="INC-2", run_id="RUN-2", chain=chain, terminal_arm=SYNTH, retained=True,
                          execution_coverage=1.0, latency_seconds=1.0, synthetic=True) == (None, [])  # fmt: skip
    _, stored = learner.credit(incident_id="INC-2", run_id="RUN-2", chain=[_decision(21, SYNTH, 1, features=[])],
                               terminal_arm=SYNTH, retained=True, execution_coverage=1.0, latency_seconds=1.0,
                               synthetic=False)  # fmt: skip
    assert stored == [] and learner.rewarded_incidents == 0


def test_the_bandit_takes_over_after_the_bootstrap_and_survives_a_restart(repository, vocabulary):
    learner = _learner(repository, latency_max_seconds=None, latency_fallback_seconds=100.0)
    router = Router(StaticCandidateSource(), vocabulary, AvailabilityProfile(version="t"), learner=learner,
                    library_size=lambda: 10)  # fmt: skip
    playbook = pb("n1:extract_entities n2:query_process_events n3:generate_incident_report",
                  "n1>n2:host_id n2>n3:findings", intent=INTENT)  # fmt: skip
    candidates = [RetrievedCandidate(entry_id="E1", playbook=playbook, similarity=0.9, score=0.9)]
    before = router.score("INC-3", candidates, set(), {})
    assert before.policy == "rule" and before.action == before.rule_action and len(before.features) == DIMENSION
    assert set(before.ucb) == {"reuse", "synthesize"}  # nothing requestable: the request arm is not offered
    assert learner.latency_max() == 100.0  # fallback until the bootstrap latencies exist

    for index, latency in enumerate((10.0, 30.0)):
        learner.credit(incident_id=f"INC-B{index}", run_id=f"RUN-B{index}",
                       chain=[_decision(30 + index, SYNTH, 1, features=before.features)], terminal_arm=SYNTH,
                       retained=True, execution_coverage=1.0, latency_seconds=latency, synthetic=False)  # fmt: skip
    assert learner.active and learner.latency_max() == pytest.approx(percentile([10.0, 30.0], 90))
    after = router.score("INC-3", candidates, set(), {})
    assert after.policy == "linucb" and "learned policy" in after.rationale.lower()

    restored = _learner(repository)
    restored.load()
    assert restored.rewarded_incidents == 2 and restored.active
    assert restored.policy.arms[SYNTH].pulls == 2
    assert np.allclose(restored.policy.arms[SYNTH].a, learner.policy.arms[SYNTH].a)


def _credit(learner, decision_id, arm, latency, *, chain=None, retained=True, coverage=1.0):
    return learner.credit(incident_id=f"INC-{decision_id}", run_id=f"RUN-{decision_id}",
                          chain=chain or [_decision(decision_id, arm, 1)], terminal_arm=arm, retained=retained,
                          execution_coverage=coverage, latency_seconds=latency, synthetic=False)  # fmt: skip


def test_bootstrap_rewards_wait_for_the_final_latency_max_then_train_the_bandit(repository):
    # Paper Section 6.2: L_max is the 90th percentile of the first 20 latencies, so rewards seen before
    # then cannot be scored yet. They are logged as provisional and applied once L_max is known.
    learner = _learner(repository, bootstrap=3, latency_max_seconds=None, latency_fallback_seconds=600.0)
    weights = RewardWeights()
    reward, stored = _credit(learner, 1, SYNTH, 238.0)
    assert stored[0].components["provisional"] and reward.latency_max_seconds == 600.0
    assert learner.policy.arms[SYNTH].pulls == 0 and not learner.latency_max_final  # nothing applied yet
    request_chain = [_decision(2, REQUEST, 1), _decision(3, REUSE, 2)]
    _credit(learner, 3, REUSE, 8.4, chain=request_chain)
    assert all(a.pulls == 0 for a in learner.policy.arms.values())

    reward, stored = _credit(learner, 4, SYNTH, 238.0)  # the last bootstrap latency
    latency_max = percentile([238.0, 8.4, 238.0], 90)
    assert learner.latency_max_final and learner.latency_max() == pytest.approx(latency_max)
    assert reward.value == pytest.approx(0.6) and not stored[0].components.get("provisional")
    final = {r.decision_id: r.reward for r in repository.list()}
    reuse = weights.coverage + weights.latency * (1 - 8.4 / latency_max)
    assert final == pytest.approx({1: 0.6, 2: reuse - 0.1, 3: reuse, 4: 0.6})  # every reward uses the final L_max
    assert not repository.provisional()
    assert (learner.policy.arms[SYNTH].pulls, learner.policy.arms[REUSE].pulls, learner.policy.arms[REQUEST].pulls) == (2, 1, 1)
    expected = LinUCBPolicy()
    for arm, value in ((SYNTH, 0.6), (REQUEST, reuse - 0.1), (REUSE, reuse), (SYNTH, 0.6)):
        expected.update(arm, [0.5] * DIMENSION, value)
    for arm in ARMS:
        assert np.allclose(learner.policy.arms[arm].b, expected.arms[arm].b)

    _, after = _credit(learner, 5, REUSE, 6.0)  # after the bootstrap: applied immediately
    assert not after[0].components.get("provisional") and learner.policy.arms[REUSE].pulls == 2


def test_a_restart_during_or_right_after_the_bootstrap_keeps_the_deferred_rewards(repository, tmp_path):
    first = _learner(repository, bootstrap=2, latency_max_seconds=None)
    _credit(first, 1, SYNTH, 100.0)
    resumed = _learner(repository, bootstrap=2, latency_max_seconds=None)
    resumed.load()
    assert len(repository.provisional()) == 1 and resumed.policy.arms[SYNTH].pulls == 0
    _credit(resumed, 2, REUSE, 10.0)
    assert not repository.provisional() and resumed.policy.arms[SYNTH].pulls == 1 and resumed.policy.arms[REUSE].pulls == 1

    # stopped after storing the last bootstrap reward but before applying them: load() finishes the job
    database = Database(f"sqlite:///{(tmp_path / 'crashed.db').as_posix()}")
    database.create_schema()
    crashed = BanditRepository(database)
    crashed.credit([r.model_copy(update={"components": {**r.components, "provisional": True}}) for r in repository.list()],
                   LinUCBPolicy().arms)  # fmt: skip
    restored = _learner(crashed, bootstrap=2, latency_max_seconds=None)
    restored.load()
    assert not crashed.provisional() and sum(a.pulls for a in restored.policy.arms.values()) == 2
    database.dispose()


def test_rule_policy_never_lets_the_bandit_choose(repository, vocabulary):
    learner = _learner(repository, enabled=False, bootstrap=0)
    router = Router(StaticCandidateSource(), vocabulary, AvailabilityProfile(version="t"), learner=learner)
    candidate = [RetrievedCandidate(entry_id="E1", playbook=SMALL, similarity=0.9, score=0.9)]
    assert router.score("INC-4", candidate, set(), {}).policy == "rule"


# --- end to end ---------------------------------------------------------------------------------


def test_feedback_on_an_executed_run_trains_the_router(make_client, settings):
    app_settings = with_settings(settings, bandit_learn_from_simulated=True, bandit_bootstrap_incidents=1)
    client = make_client([good_output()] * 4, app_settings=app_settings)
    learner = client.app.state.components.learner
    incident_id = analysed(client, client.headers)
    outcome = client.post(f"/incidents/{incident_id}/respond?wait=true").json()
    assert outcome["decisions"][0]["policy"] == "rule" and not learner.active
    run = client.post(f"/response-plans/{outcome['plan']['plan_id']}/execute").json()
    assert 0.0 <= run["execution_coverage"] <= 1.0
    feedback = client.post(f"/playbook-runs/{run['run_id']}/feedback",
                           json={"resolved": True, "findings_retained": True}).json()["feedback"]  # fmt: skip
    assert feedback["routing_reward"] is not None and feedback["routing_reward"] > 0
    rewards = learner.repository.list(incident_id)
    assert rewards and rewards[0].terminal and rewards[0].synthetic
    assert learner.active  # bootstrap of one incident is over

    second = client.post(f"/incidents/{analysed(client, client.headers)}/respond?wait=true").json()
    assert second["decisions"][0]["policy"] == "linucb"


def test_simulated_feedback_does_not_train_the_router_by_default(make_client):
    client = make_client([good_output()] * 3)
    incident_id = analysed(client, client.headers)
    plan = client.post(f"/incidents/{incident_id}/respond?wait=true").json()["plan"]
    run = client.post(f"/response-plans/{plan['plan_id']}/execute").json()
    feedback = client.post(f"/playbook-runs/{run['run_id']}/feedback",
                           json={"resolved": True, "findings_retained": True}).json()["feedback"]  # fmt: skip
    assert feedback["routing_reward"] is None
    assert client.app.state.components.learner.repository.list(incident_id) == []


def test_settings_follow_table_3(settings):
    from app.core.config import Settings

    defaults = Settings(_env_file=None)
    assert (defaults.routing_policy, defaults.bandit_alpha, defaults.bandit_bootstrap_incidents) == ("linucb", 1.0, 20)
    assert (defaults.reward_lambda_coverage, defaults.reward_lambda_latency, defaults.reward_lambda_request) == (0.6, 0.3, 0.1)
    assert defaults.max_evidence_rounds == 1 and len(ARMS) == 3
