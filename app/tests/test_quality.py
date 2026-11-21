"""Playbook-quality evaluation (paper 968 Table 4, Fig. 2, Fig. 4b): systems, scoring, caching, aggregation."""

from __future__ import annotations

import asyncio
import json

import pytest

from app.core.config import PROJECT_ROOT, Settings
from app.evaluation import paper968 as B
from app.evaluation import quality as Q
from app.intent.mock_llm_client import MockLLMClient
from app.library.embedder import HashingEmbedder
from app.library.index import NumpyIndex
from app.library.library import PlaybookLibrary
from app.playbook.validator import PlaybookValidator
from app.playbook.vocabulary import Vocabulary
from app.schemas.taxonomy import IntentTaxonomy
from app.storage.database import Database
from app.storage.library_repository import LibraryRepository
from app.synthesis.pipeline import run_synthesis

DATA = PROJECT_ROOT / "app" / "data"


def judge_answer(score: int = 4) -> dict:
    return {"tool_appropriateness": score, "sequencing": score, "coherence": score, "destructive_safety": score,
            "rationale": "ok"}  # fmt: skip


@pytest.fixture(scope="module")
def benchmark() -> B.Benchmark:
    return B.build()


@pytest.fixture(scope="module")
def run(benchmark, tmp_path_factory):
    settings = Settings(_env_file=None, generation_pool_size=4, population_size=8, max_generations=3)
    vocabulary = Vocabulary.load(DATA / "action_vocabulary.json", DATA / "intent_goals.json")
    root = tmp_path_factory.mktemp("quality")
    databases = []

    def database(name):
        db = Database(f"sqlite:///{(root / name).as_posix()}")
        db.create_schema()
        databases.append(db)
        return db

    library, entries = B.build_library(benchmark, settings, vocabulary, HashingEmbedder(), NumpyIndex(384), database("lib.db"))
    env = B.PaperEnv(benchmark, settings, vocabulary, library, entries)
    kit = Q.build_kit(settings, MockLLMClient(), None)
    seeds = PlaybookLibrary(LibraryRepository(database("seeds.db")), HashingEmbedder(), NumpyIndex(384), kit.vocabulary,
                            PlaybookValidator(kit.vocabulary, node_cap=settings.node_cap), k=settings.routing_k)  # fmt: skip
    seeds.seed_if_empty(Q.seed_library_path(settings))
    seeds.load()
    yield Q.QualityRun(kit, env, seeds, Q.Cache(root / "cache"))
    for db in databases:
        db.dispose()


def test_every_tactic_maps_to_an_intent_of_the_ontology(benchmark):
    taxonomy = IntentTaxonomy.load(DATA / "intent_taxonomy.json")
    assert set(Q.TACTIC_INTENT) == set(B.TACTIC_ORDER)
    assert all(taxonomy.get(intent) is not None for intent in Q.TACTIC_INTENT.values())


def test_the_incident_context_splits_missing_evidence_by_criticality(benchmark):
    scenario = next(s for s in benchmark.scenarios if s.cell == "strong/critical_missing")
    context = Q.context_for(scenario, scenario.available, "Q-1")
    assert context.intent_label == Q.TACTIC_INTENT[scenario.tactic]
    assert set(context.missing_critical) == scenario.critical - scenario.available
    assert not set(context.observed_categories) & set(context.missing_critical)
    assert context.techniques == (scenario.technique,) and scenario.technique in context.summary


@pytest.mark.parametrize("system", ["static_soar", "retrieval_only", "llm_only", "generation_only", "generation_only_osh"])
def test_each_system_returns_a_scored_playbook_and_caches_it(run, benchmark, system):
    scenario = next(s for s in benchmark.scenarios if s.cell == "strong/full")
    record = asyncio.run(run.produce(system, scenario, scenario.available))
    assert not record.failed and record.playbook is not None
    assert 0 < record.fitness <= 1 and record.fitness == pytest.approx((record.ic * record.ea * record.os) ** (1 / 3))
    if system.startswith("generation_only"):
        assert {"generation_s", "evolution_s", "total_s", "judge_calls", "judge_s"} <= set(record.info["timing"])
        assert record.info["evolution"]["history"]

    def boom(*_args, **_kwargs):
        raise AssertionError("a cached record must not be produced again")

    original, run._produce = run._produce, boom
    try:
        again = asyncio.run(run.produce(system, scenario, scenario.available))
    finally:
        run._produce = original
    assert again.as_dict() == json.loads(json.dumps(record.as_dict(), default=str))


def test_retrieval_only_never_returns_a_withheld_playbook(run, benchmark):
    for scenario in [s for s in benchmark.scenarios if s.suitability == "none"][:4]:
        record = asyncio.run(run.produce("retrieval_only", scenario, scenario.available))
        matched = record.info["matched"]
        assert matched is None or not any(matched.startswith(t + " ") for t in scenario.withheld_techniques)


def test_no_playbook_is_a_scored_failure(run, benchmark):
    record = asyncio.run(run.score("llm_only", benchmark.scenarios[0], "x", None, 1.5, {}))
    assert record.failed and (record.ic, record.ea, record.os, record.fitness) == (0, 0, 0, 0)


def test_cade_full_follows_the_routed_action(run, benchmark):
    scenario = next(s for s in benchmark.scenarios if s.cell == "strong/full")
    reused = asyncio.run(run.cade_full(scenario, "reuse", requested=False))
    synthesized = asyncio.run(run.cade_full(scenario, "synthesize", requested=False))
    base = asyncio.run(run.produce("retrieval_only", scenario, scenario.available))
    assert reused.system == "cade_full" and reused.playbook == base.playbook
    assert synthesized.info == {"terminal": "synthesize", "requested": False}


def test_table4_averages_systems_and_cade_full_over_orderings():
    def rec(system, f, failed=False, terminal=None):
        return Q.Record(system, "s", "x", None, f, f, f, "llm", f, 2.0, failed=failed,
                        info={"terminal": terminal} if terminal else {})  # fmt: skip

    table = Q.table4({"llm_only": [rec("llm_only", 0.6), rec("llm_only", 0.0, failed=True)]},
                     {0: [rec("cade_full", 0.8, terminal="reuse")], 1: [rec("cade_full", 0.6, terminal="synthesize")]})  # fmt: skip
    assert table["llm_only"]["fitness"] == pytest.approx(0.3) and table["llm_only"]["failures"] == 1
    assert table["cade_full"]["fitness"] == pytest.approx(0.7) and table["cade_full"]["orderings"] == 2
    assert table["cade_full"]["synthesis_share"] == pytest.approx(0.5)


def test_generation_curves_carry_early_stops_forward_and_costs_report_the_judge_share():
    def rec(history, timing):
        return Q.Record("generation_only", "s", "x", None, 1, 1, 1, "llm", 1, 1, info={
            "evolution": {"history": history, "generations": len(history) - 1}, "timing": timing})

    short = [{"best_fitness": 0.5, "mean_fitness": 0.4, "best_ic": 0.5}]
    long = [{"best_fitness": 0.6, "mean_fitness": 0.5, "best_ic": 0.7}, {"best_fitness": 0.8, "mean_fitness": 0.6, "best_ic": 0.9}]
    timing = {"generation_s": 10, "seeding_s": 2, "gap_proposals_s": 1, "evolution_s": 7, "total_s": 20, "judge_calls": 6, "judge_s": 13}
    curves = Q.generation_curves([rec(short, timing), rec(long, timing)])
    assert curves["best_fitness"] == pytest.approx([0.55, 0.65])  # the short run's last value is carried forward
    costs = Q.latency_breakdown([rec(short, timing), rec(long, timing)])
    assert costs["judge_share"] == pytest.approx(0.65) and costs["runs"] == 2


def test_run_synthesis_reports_judge_calls_and_time(run, benchmark):
    """The judge's call count and time are the Section 7.4 cost figures; a scripted judge makes them countable."""
    kit = run.kit
    from app.fitness.fitness import FitnessEvaluator
    from app.fitness.judge import LLMSoundnessJudge

    judge = LLMSoundnessJudge(MockLLMClient(responses=[judge_answer()] * 200), kit.coverage, kit.heuristic, samples=3)
    evaluator = FitnessEvaluator(kit.coverage, kit.validator, kit.heuristic, judge, judge_top_k=2)
    scenario = next(s for s in benchmark.scenarios if s.cell == "none/full")
    result = asyncio.run(run_synthesis(
        generator=kit.generator, seeder=kit.seeder(evaluator), new_engine=kit.engine_factory(evaluator),
        coverage=kit.coverage, settings=kit.settings, context=Q.context_for(scenario, scenario.available, "Q-2"),
        examples=[], seed=7, judge_model="mock-judge"))  # fmt: skip
    timing = result.info["timing"]
    assert timing["judge_calls"] == judge.calls > 0 and timing["judge_calls"] % 3 == 0
    assert 0 <= timing["judge_s"] <= timing["total_s"] and result.best.scores.os_source == "llm"
