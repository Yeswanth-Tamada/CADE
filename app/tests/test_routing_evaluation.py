"""The routing evaluation protocol (paper Section 6.4) and its metrics, on a small deterministic environment."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass

import pytest

from app.evaluation import metrics as M
from app.evaluation import protocol as P
from app.routing.bandit import ARMS, DIMENSION
from app.routing.reward import RewardWeights, percentile
from app.schemas.routing import RoutingAction, RoutingDecision, RoutingScores, RoutingTerms
from app.tests.conftest import with_settings

REUSE, REQUEST, SYNTH = RoutingAction.REUSE, RoutingAction.REQUEST_EVIDENCE, RoutingAction.SYNTHESIZE


# --- metrics ------------------------------------------------------------------------------------


def test_classification_matches_a_hand_computed_confusion_matrix():
    pairs = [("reuse", "reuse")] * 3 + [("reuse", "synthesize")] + [("request_evidence", "request_evidence")] * 2 + \
            [("request_evidence", "reuse")] * 2 + [("synthesize", "synthesize")] * 4  # fmt: skip
    c = M.classification([{"expected": e, "got": g} for e, g in pairs])
    assert c["confusion"] == [[3, 0, 1], [2, 2, 0], [0, 0, 4]] and c["accuracy"] == pytest.approx(9 / 12)
    reuse, request, synth = (c["per_class"][label] for label in M.LABELS)
    assert (reuse["precision"], reuse["recall"]) == (pytest.approx(3 / 5), pytest.approx(3 / 4))
    assert reuse["f1"] == pytest.approx(2 * 0.6 * 0.75 / 1.35)
    assert (request["precision"], request["recall"], synth["precision"], synth["recall"]) == (1.0, 0.5, 0.8, 1.0)
    assert c["macro_f1"] == pytest.approx((reuse["f1"] + 2 / 3 + 2 * 0.8 / 1.8) / 3)
    assert c["confusion_row_normalized"][1] == [0.5, 0.5, 0.0]


def test_mean_std_is_the_sample_standard_deviation():
    assert M.mean_std([1.0, 2.0, 3.0, 4.0]) == {"mean": 2.5, "std": pytest.approx(1.2909944), "n": 4}
    assert M.mean_std([5.0])["std"] == 0.0 and M.relative_gain(1.14, 1.0) == pytest.approx(0.14)


# --- rewards, latencies, provenance ---------------------------------------------------------------


def test_path_latency_follows_figure_4b():
    latency = P.PathLatency()
    assert (latency.of(REUSE, REUSE), latency.of(SYNTH, SYNTH)) == (6.0, 238.0)
    assert latency.of(REQUEST, REUSE) == pytest.approx(8.4)  # the whole request path, not 8.4 + 6
    assert latency.of(REQUEST, SYNTH) == pytest.approx(240.4)
    legacy = P.PathLatency(additive_request=True)
    assert legacy.of(REQUEST, REUSE) == pytest.approx(14.4) and "ASSUMED" in latency.describe()


def test_reward_value_charges_the_request_and_gates_on_retention():
    w = RewardWeights()
    assert P.reward_value(REUSE, 1.0, True, 6.0, 240.0, w) == pytest.approx(0.6 + 0.3 * (1 - 6 / 240))
    assert P.reward_value(REQUEST, 1.0, True, 8.4, 240.0, w) == pytest.approx(0.6 + 0.3 * (1 - 8.4 / 240) - 0.1)
    assert P.reward_value(SYNTH, 1.0, False, 1.0, 240.0, w) == 0.0  # retention gates both bracketed terms
    assert P.reward_value(REQUEST, 1.0, False, 1.0, 240.0, w) == pytest.approx(-0.1)


def test_reward_preference_separates_label_accuracy_from_reward_optimality():
    assert P.reward_preference({REUSE: 0.9, SYNTH: 0.6}, REUSE) == ("unique_expected", ["reuse"])
    assert P.reward_preference({REUSE: 0.9, SYNTH: 0.6}, SYNTH) == ("unique_other", ["reuse"])
    assert P.reward_preference({REUSE: 0.0, SYNTH: 0.0, REQUEST: -0.1}, SYNTH) == ("tie_with_expected", ["reuse", "synthesize"])
    assert P.reward_preference({REUSE: 0.5, SYNTH: 0.5}, REQUEST)[0] == "tie_without_expected"
    w = RewardWeights()
    best = P.best_first_rewards({REQUEST: [(1.0, True, 8.4), (1.0, True, 240.4)], SYNTH: [(1.0, False, 238.0)]}, 240.0, w)
    assert best[REQUEST] == pytest.approx(0.6 + 0.3 * (1 - 8.4 / 240) - 0.1) and best[SYNTH] == 0.0


def test_substituted_components_are_reported_and_refused_unless_allowed():
    found = P.component_fallbacks("hashing-384", "sentence-transformers unavailable", "numpy", None)
    assert len(found) == 3 and any("FAISS" in f for f in found)
    assert P.component_fallbacks("st:sentence-transformers/all-MiniLM-L6-v2", None, "faiss", None) == ()
    with pytest.raises(SystemExit):
        P.require_real_components(found, allow_fallback=False)
    P.require_real_components(found, allow_fallback=True)
    degraded = P.Provenance("b", "hashing-384", "numpy", "a", "d", "l", fallbacks=found)
    assert degraded.status == "DEGRADED" and "SIMULATED" in degraded.as_dict()["execution"]


# --- the online protocol --------------------------------------------------------------------------


@dataclass
class ToyScenario:
    name: str
    cell: str
    expected: RoutingAction
    rule: RoutingAction
    x: list[float]


def _decision(action: RoutingAction, x: list[float], rule: RoutingAction, round_number: int = 1) -> RoutingDecision:
    terms = RoutingTerms(coverage=0.5, counterfactual_coverage=0.5, entropy=0.5, requestability=0.0, unmet_fraction=0.0,
                         smoothed_success=0.5, w=0.7, k=3, tau=0.5, candidates_retrieved=1)  # fmt: skip
    return RoutingDecision(incident_id="toy", action=action, scores=RoutingScores(reuse=0, request_evidence=0, synthesize=0),
                           terms=terms, rationale="toy", round=round_number, rule_action=rule, features=x)  # fmt: skip


class ToyEnv:
    """Reuse is retained (and best) when x[0] = 1; otherwise only synthesis is retained. A request releases nothing."""

    latency = P.PathLatency()

    def _outcome(self, scenario, terminal):
        retained = terminal == SYNTH or scenario.x[0] == 1.0
        return (1.0 if retained else 0.0), retained

    async def respond(self, scenario, learner):
        bandit = learner is not None and learner.active
        action = learner.policy.select(scenario.x, ARMS) if bandit else scenario.rule
        first = _decision(action, scenario.x, scenario.rule)
        chain, terminal = [first], first
        if action == REQUEST:
            second = learner.policy.select(scenario.x, [REUSE, SYNTH]) if bandit else SYNTH
            terminal = _decision(second, scenario.x, SYNTH, round_number=2)
            chain.append(terminal)
        coverage, retained = self._outcome(scenario, terminal.action)
        return P.Outcome(first, terminal, chain, coverage, retained, self.latency.of(action, terminal.action))

    async def alternatives(self, scenario):
        options = {}
        for first in (REUSE, REQUEST, SYNTH):
            terminals = [REUSE, SYNTH] if first == REQUEST else [first]
            options[first] = [(*self._outcome(scenario, t), self.latency.of(first, t)) for t in terminals]
        return options


def _toy(n: int = 30) -> list[ToyScenario]:
    rng = random.Random(7)
    scenarios = []
    for i in range(n):
        reusable = i % 2 == 0
        x = [1.0 if reusable else 0.0] + [round(rng.random(), 3) for _ in range(DIMENSION - 2)] + [1.0]
        # the rule always synthesizes: right for half the scenarios, never reward-optimal on the other half
        scenarios.append(ToyScenario(f"s{i}", "reusable" if reusable else "not_reusable", REUSE if reusable else SYNTH, SYNTH, x))
    return scenarios


def _run(settings, tmp_path, scenarios, orderings=3):
    settings = with_settings(settings, bandit_bootstrap_incidents=10)
    return asyncio.run(P.run(ToyEnv(), scenarios, settings=settings, orderings=orderings, workdir=tmp_path / "bandits"))


def test_the_rule_chooses_until_the_bootstrap_then_linucb_and_both_share_one_l_max(settings, tmp_path):
    result = _run(settings, tmp_path, _toy())
    rows = result["rows"]
    for row in rows:
        if row["position"] < 10:
            assert row["policy"] == "rule" and row["got"] == row["rule_got"] and row["reward"] == pytest.approx(row["rule_reward"])
        else:
            assert row["policy"] == "linucb"
    ordering0 = sorted((r for r in rows if r["ordering"] == 0), key=lambda r: r["position"])
    first10 = [r["rule_latency"] for r in ordering0[:10]]
    assert len({r["latency_max"] for r in ordering0}) == 1
    assert ordering0[0]["latency_max"] == pytest.approx(percentile(first10, 90))
    # the learner trained on exactly the reported rewards (run_ordering checks every decision) and found that
    # reusing pays off where it is retained, which the always-synthesize rule never does. Its label accuracy
    # need not rise with it: exploring request-then-reuse also beats synthesis on reward but is not the label.
    s = result["summary"]["linucb"]["mean_std"], result["summary"]["rule"]["mean_std"]
    assert s[0]["reward_after_bootstrap"]["mean"] > s[1]["reward_after_bootstrap"]["mean"]
    assert s[0]["reward_bootstrap"] == s[1]["reward_bootstrap"] and s[0]["accuracy_bootstrap"] == s[1]["accuracy_bootstrap"]


def test_orderings_are_seeded_reproducible_and_use_a_fresh_learner(settings, tmp_path):
    scenarios = _toy()
    first = _run(settings, tmp_path / "a", scenarios, orderings=2)["rows"]
    again = _run(settings, tmp_path / "b", scenarios, orderings=2)["rows"]
    strip = lambda rows: [{k: v for k, v in r.items() if k != "ucb"} for r in rows]  # noqa: E731
    assert strip(first) == strip(again)
    order = lambda rows, o: [r["scenario"] for r in sorted((r for r in rows if r["ordering"] == o), key=lambda r: r["position"])]  # noqa: E731
    expected0 = [s.name for s in scenarios]
    random.Random(0).shuffle(expected0)
    assert order(first, 0) == expected0 and order(first, 0) != order(first, 1)
    # ordering 1 alone gives the same decisions as ordering 1 after ordering 0: nothing leaks between orderings
    alone = asyncio.run(P.run_ordering(ToyEnv(), scenarios, 1, settings=with_settings(settings, bandit_bootstrap_incidents=10),
                                       workdir=tmp_path / "c", rule={s.name: asyncio.run(ToyEnv().respond(s, None)) for s in scenarios},
                                       alternatives={s.name: asyncio.run(ToyEnv().alternatives(s)) for s in scenarios}))  # fmt: skip
    assert [r["got"] for r in alone] == [r["got"] for r in sorted((r for r in first if r["ordering"] == 1), key=lambda r: r["position"])]


def test_the_legacy_bootstrap_diagnostic_reproduces_the_pre_fix_training_rewards(settings, tmp_path):
    settings = with_settings(settings, bandit_bootstrap_incidents=10)
    result = asyncio.run(P.run(ToyEnv(), _toy(), settings=settings, orderings=1, workdir=tmp_path / "w", legacy_bootstrap=True))
    rows = sorted(result["rows"], key=lambda r: r["position"])
    w = RewardWeights()
    for r in rows[:10]:  # trained at once with the 600 s fallback, reported with the 90th-percentile L_max
        expected = P.reward_value(RoutingAction(r["got"]), r["execution_coverage"], r["retained"], r["latency"], 600.0, w)
        assert r["trained_reward"] == pytest.approx(expected) and (not r["retained"] or r["trained_reward"] != pytest.approx(r["reward"]))
    assert all(r["trained_reward"] == pytest.approx(r["reward"]) for r in rows[10:])
    assert all("regret" in r and r["regret"] >= -1e-9 and len(r["features"]) == DIMENSION for r in rows)


def test_label_accuracy_and_reward_optimality_are_reported_separately(settings, tmp_path):
    # relabel the reusable scenarios as GENERATE: the rule (always synthesize) is now 100% "accurate" while
    # Eq. 9 still prefers reuse there, so its reward-optimal rate stays at one half
    scenarios = [ToyScenario(s.name, s.cell, SYNTH, SYNTH, s.x) for s in _toy()]
    s = _run(settings, tmp_path, scenarios, orderings=1)["summary"]
    assert s["rule"]["mean_std"]["accuracy"]["mean"] == 1.0
    assert s["rule"]["mean_std"]["reward_optimal_rate"]["mean"] == pytest.approx(0.5)
    assert s["reward_preference"]["counts_per_ordering"][0] == {"unique_expected": 15, "unique_other": 15}
    assert s["rule"]["by_cell"]["reusable"]["accuracy"] == 1.0
    with pytest.raises(ValueError):
        _run(settings, tmp_path / "dup", scenarios + scenarios[:1], orderings=1)
