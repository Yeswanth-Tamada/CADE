"""Library-growth experiment (paper 968 Fig. 4a): execution check, recomputed labels, write-back, reuse by size."""

from __future__ import annotations

import asyncio
import random

import pytest

from app.core.config import PROJECT_ROOT, Settings
from app.evaluation import growth as G
from app.evaluation import paper968 as B
from app.evaluation import quality as Q
from app.intent.mock_llm_client import MockLLMClient
from app.library.embedder import HashingEmbedder
from app.library.index import NumpyIndex
from app.library.library import PlaybookLibrary
from app.playbook.validator import PlaybookValidator
from app.playbook.vocabulary import Vocabulary
from app.storage.database import Database
from app.storage.library_repository import LibraryRepository

DATA = PROJECT_ROOT / "app" / "data"


@pytest.fixture(scope="module")
def benchmark() -> B.Benchmark:
    return B.build()


@pytest.fixture(scope="module")
def vocabulary() -> Vocabulary:
    return Vocabulary.load(DATA / "action_vocabulary.json", DATA / "intent_goals.json")


@pytest.fixture
def growth(benchmark, vocabulary, tmp_path):
    settings = Settings(_env_file=None, generation_pool_size=3, population_size=6, max_generations=2)
    database = Database(f"sqlite:///{(tmp_path / 'lib.db').as_posix()}")
    database.create_schema()
    library = PlaybookLibrary(LibraryRepository(database), HashingEmbedder(), NumpyIndex(384), vocabulary,
                              PlaybookValidator(vocabulary, node_cap=settings.node_cap), k=settings.routing_k)  # fmt: skip
    initial = G.seed_initial(benchmark, library, vocabulary, 10, random.Random(0))
    env = B.PaperEnv(benchmark, settings, vocabulary, library, {})
    quality = Q.QualityRun(Q.build_kit(settings, MockLLMClient(), None), env, library, Q.Cache(tmp_path / "cache"))
    yield G.GrowthRun(quality, benchmark, vocabulary), initial
    database.dispose()


def test_the_starting_library_holds_the_drawn_technique_playbooks(growth):
    run, initial = growth
    assert len(initial) == 10 == len(run.library) and set(run.stored()) == set(initial)


def test_execution_check_needs_the_scenarios_tactic_and_its_critical_evidence(benchmark, vocabulary):
    scenario = next(s for s in benchmark.scenarios if s.cell == "strong/full")
    playbook = B.technique_playbook(benchmark, scenario.technique, vocabulary)
    assert G.execution_check(scenario, playbook, scenario.tactic, scenario.available, vocabulary) == (1.0, True)
    other = next(t for t in B.TACTIC_ORDER if t != scenario.tactic)
    assert G.execution_check(scenario, playbook, other, scenario.available, vocabulary) == (0.0, False)
    assert G.execution_check(scenario, playbook, scenario.tactic, frozenset(), vocabulary)[1] is False


def test_the_correct_decision_is_recomputed_against_the_current_library(growth, benchmark, vocabulary):
    run, initial = growth
    full = next(s for s in benchmark.scenarios if s.critical <= s.available and s.technique not in initial)
    assert run.correct(full) == G.GENERATE
    run.library.add(B.technique_playbook(benchmark, full.technique, vocabulary), deduplicate=False,
                    metadata={"technique": full.technique, "tactic": full.tactic})  # fmt: skip
    assert run.correct(full) == G.REUSE


def test_a_step_routes_executes_and_writes_back(growth, benchmark):
    run, _ = growth
    rows = [asyncio.run(run.step(s)) for s in benchmark.scenarios[:6]]
    assert all(r["terminal"] in (G.REUSE, G.GENERATE) for r in rows)
    assert all(r["library_size_after"] >= r["library_size"] for r in rows)
    assert {r["write_back"] for r in rows} <= {"updated", "inserted", "merged", "not_stored", "synthesis_failed"}


def test_reuse_by_size_buckets_incidents_by_library_size():
    rows = [{"library_size": 9, "terminal": G.REUSE, "got": G.REUSE, "expected": G.REUSE},
            {"library_size": 12, "terminal": G.GENERATE, "got": G.GENERATE, "expected": G.REUSE},
            {"library_size": 21, "terminal": G.REUSE, "got": G.REUSE, "expected": G.REUSE}]  # fmt: skip
    by_size = G.reuse_by_size(rows, sizes=(10, 20, 30))
    assert by_size[10] == {"incidents": 2, "reuse_rate": 0.5, "accuracy": 0.5}
    assert by_size[20]["reuse_rate"] == 1.0 and by_size[30]["reuse_rate"] is None
