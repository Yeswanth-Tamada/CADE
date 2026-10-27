"""The reproduction tooling: seeds, accuracy ceilings, the separate validation benchmark (V2) and labelled provenance."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import random
import sys
from collections import Counter
from types import SimpleNamespace

import pytest

from app.core.config import PROJECT_ROOT, Settings
from app.evaluation import paper968 as B
from app.evaluation import protocol as P
from app.evaluation.report import render
from app.schemas.routing import RoutingAction
from app.tests.conftest import with_settings
from app.tests.test_routing_evaluation import ToyEnv, _toy

REUSE, REQUEST, GENERATE = RoutingAction.REUSE, RoutingAction.REQUEST_EVIDENCE, RoutingAction.SYNTHESIZE
ARCHIVED = PROJECT_ROOT / "reports" / "2026-10-26_routing_benchmarks" / "paper968" / "scenarios.json"
FIVE = ("execution", "persistence", "privilege_escalation", "defense_evasion", "discovery")


def _run(settings, tmp_path, **kwargs):
    settings = with_settings(settings, bandit_bootstrap_incidents=10)
    return asyncio.run(P.run(ToyEnv(), _toy(), settings=settings, workdir=tmp_path / "bandits", **kwargs))


# --- seeds ------------------------------------------------------------------------------------------


def test_explicit_seeds_select_those_orderings_reproducibly(settings, tmp_path):
    chosen = _run(settings, tmp_path / "a", seeds=[3, 7])
    assert sorted({r["ordering"] for r in chosen["rows"]}) == [3, 7] and chosen["summary"]["seeds"] == [3, 7]
    default = _run(settings, tmp_path / "b", orderings=8)
    assert default["summary"]["seeds"] == list(range(8))
    strip = lambda rows, o: [{k: v for k, v in r.items() if k != "ucb"} for r in sorted(rows, key=lambda r: r["position"])  # noqa: E731
                             if r["ordering"] == o]
    assert strip(chosen["rows"], 3) == strip(default["rows"], 3)  # the same seed gives the same ordering and decisions
    expected = [s.name for s in _toy()]
    random.Random(7).shuffle(expected)
    assert [r["scenario"] for r in sorted(chosen["rows"], key=lambda r: r["position"]) if r["ordering"] == 7] == expected


def test_a_state_directory_is_never_reused_by_a_fresh_learner(settings, tmp_path):
    _run(settings, tmp_path, orderings=1)
    with pytest.raises(FileExistsError):  # a second run into the same state directory would inherit the first's rewards
        _run(settings, tmp_path, orderings=1)


# --- ceilings ----------------------------------------------------------------------------------------


def _row(o, position, expected, optimal, got=None):
    return {"ordering": o, "position": position, "expected": expected, "got": got or expected, "reward_optimal": optimal}


def test_ceilings_follow_their_definitions_exactly():
    rows = [_row(0, 0, "reuse", ["reuse"]), _row(0, 1, "synthesize", ["synthesize"], got="reuse"),  # B = 1 of 2
            _row(0, 2, "reuse", ["reuse"]), _row(0, 3, "request_evidence", ["request_evidence"]),  # U = 2
            _row(0, 4, "synthesize", ["reuse", "synthesize"]),  # T, credit 1/2
            _row(0, 5, "request_evidence", ["reuse", "request_evidence", "synthesize"]),  # T, credit 1/3
            _row(0, 6, "reuse", ["synthesize"]),  # X
            _row(0, 7, "request_evidence", ["reuse", "synthesize"])]  # W  # fmt: skip
    c = P.ceilings(rows, bootstrap=2)["per_ordering"][0]
    assert (c["n"], c["m"], c["B"], c["U"], c["T"], c["X"], c["W"]) == (8, 6, 1, 2, 2, 1, 1)
    assert c["T_uniform_credit"] == pytest.approx(1 / 2 + 1 / 3)
    assert c["any_policy"] == pytest.approx((1 + 6) / 8)
    assert c["reward_uniform_ties"] == pytest.approx((1 + 2 + 1 / 2 + 1 / 3) / 8)
    assert c["reward_favourable_ties"] == pytest.approx((1 + 2 + 2) / 8)
    second = [dict(r, ordering=1, got=r["expected"]) for r in rows]  # B = 2 in the second ordering
    mean = P.ceilings(rows + second, bootstrap=2)["mean"]
    assert mean["B"] == 1.5 and mean["any_policy"] == pytest.approx(((1 + 6) / 8 + (2 + 6) / 8) / 2)


def test_ceilings_bound_the_accuracy_achieved_in_a_protocol_run(settings, tmp_path):
    summary = _run(settings, tmp_path, orderings=3)["summary"]
    ceilings = summary["ceilings"]
    for o, c in ceilings["per_ordering"].items():
        assert c["U"] + c["T"] + c["X"] + c["W"] == c["m"] == c["n"] - 10
        achieved = summary["linucb"]["per_ordering"]["accuracy"][list(ceilings["per_ordering"]).index(o)]
        assert achieved <= c["any_policy"] + 1e-12
    assert ceilings["mean"]["reward_uniform_ties"] <= ceilings["mean"]["reward_favourable_ties"] <= ceilings["mean"]["any_policy"]


# --- the separate validation benchmark (V2) ------------------------------------------------------------


@pytest.fixture(scope="module")
def main_benchmark() -> B.Benchmark:
    return B.build()


@pytest.fixture(scope="module")
def validation(main_benchmark) -> B.Benchmark:
    return B.build(exclude_recordings={s.recording.id for s in main_benchmark.scenarios}, tactics=FIVE,
                   tactic_of=main_benchmark.tactic_of, critical=main_benchmark.critical, name_prefix="v2:")  # fmt: skip


def test_the_default_build_is_the_archived_96(main_benchmark):
    explicit = B.build(exclude_recordings=(), tactics=B.TACTIC_ORDER, tactic_of=main_benchmark.tactic_of,
                       critical=main_benchmark.critical)  # fmt: skip
    assert [s.summary() for s in explicit.scenarios] == [s.summary() for s in main_benchmark.scenarios]
    if ARCHIVED.exists():  # read-only comparison with the archived run (a protected file)
        assert [s.summary() for s in main_benchmark.scenarios] == json.loads(ARCHIVED.read_text(encoding="utf-8"))["scenarios"]


def _assert_construction_rules(benchmark: B.Benchmark) -> None:
    library = set(benchmark.library_techniques)
    for s in benchmark.scenarios:
        crit = benchmark.critical[s.tactic]
        kept = {t for t in library if benchmark.tactic_of[t] == s.tactic} - s.withheld_techniques
        assert s.available | s.hidden == s.recording.categories and s.obtainable <= s.hidden
        if s.suitability == "strong":
            assert s.technique in kept
        elif s.suitability == "weak":
            assert kept == {s.related} and s.related != s.technique
        else:
            assert not kept
        if s.expected == REUSE:
            assert s.critical <= s.available
        if s.expected == REQUEST:
            runner = s.technique if s.suitability == "strong" else s.related
            assert (s.critical - s.available) and (s.critical - s.available) <= s.obtainable
            assert benchmark.footprint[runner] & crit <= s.available | s.obtainable
        if s.expected == GENERATE:
            assert s.suitability == "none" or not s.obtainable


def test_the_validation_benchmark_is_separate_and_follows_the_same_rules(main_benchmark, validation):
    scenarios = validation.scenarios
    assert len(scenarios) == 60 and not validation.infeasible
    assert Counter(s.expected for s in scenarios) == {REUSE: 20, REQUEST: 20, GENERATE: 20}
    weights = {f"{s}/{e}": n for s, e, _, n in B.TABLE_2}
    for tactic in FIVE:
        assert Counter(s.cell for s in scenarios if s.tactic == tactic) == weights
    assert not {s.recording.id for s in scenarios} & {s.recording.id for s in main_benchmark.scenarios}  # disjoint recordings
    assert validation.critical == main_benchmark.critical and validation.tactic_of == main_benchmark.tactic_of  # frozen
    assert all(s.name.startswith("v2:") for s in scenarios)
    assert not {s.name for s in scenarios} & {s.name for s in main_benchmark.scenarios}
    _assert_construction_rules(validation)
    _assert_construction_rules(main_benchmark)


# --- labelled provenance and reports -----------------------------------------------------------------------


def _runner():
    spec = importlib.util.spec_from_file_location("eval_routing_paper_script", PROJECT_ROOT / "scripts" / "eval_routing_paper.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_runner_parses_seeds_and_flags_every_departure_in_the_settings_audit():
    runner = _runner()
    assert runner.parse_seeds("10-19") == list(range(10, 20)) and runner.parse_seeds("0,3") == [0, 3]
    settings = Settings(_env_file=None)
    args = SimpleNamespace(benchmark="main", goals="critical", diagnostic_legacy_bootstrap=False,
                           diagnostic_profile_requestability=False)  # fmt: skip
    audit = runner.settings_audit(settings, args, list(range(10)), 1.0, "st:sentence-transformers/all-MiniLM-L6-v2", "faiss", ())
    statuses = {row[0]: row[3] for row in audit}
    assert not any("DEPARTURE" in s for s in statuses.values()) and statuses["Orderings"] == "documented"
    assert statuses["Ridge λ"].startswith("inferred") and statuses["Critical / supporting per intent; intents"] == "inferred"
    departed = runner.settings_audit(settings.model_copy(update={"bandit_ridge_lambda": 0.01}),
                                     SimpleNamespace(**{**vars(args), "goals": "all"}), list(range(10, 20)), 20.0,
                                     "hashing-384", "numpy", ("embedder is hashing-384",))  # fmt: skip
    flagged = {row[0]: row[3] for row in departed}
    for name in ("Ridge λ", "IC_exec goals; attribution", "Entropy H", "Retrieval"):
        assert flagged[name].startswith("DEPARTURE")
    assert flagged["Orderings"].startswith("validation")


def test_reports_carry_the_evaluation_kind_settings_and_ceilings(settings, tmp_path):
    result = _run(settings, tmp_path, seeds=[10, 11])
    provenance = P.Provenance("b", "st:x", "faiss", "a", "d", "l", evaluation="V1 validation: held-out orderings",
                              settings_audit=(("Ridge λ", "not stated", "1", "inferred"),)).as_dict()  # fmt: skip
    assert provenance["evaluation"].startswith("V1") and provenance["settings_audit"][0]["status"] == "inferred"
    text = render("t", result["summary"], provenance)
    assert "seeds 10–11" in text and "V1 validation" in text and "## Settings and their status" in text
    assert "## Accuracy ceilings" in text and "CHECK FAILED" not in text
    assert "Against the paper's published values" not in text  # only the primary reproduction is compared
