"""Option A routing environment: GENERATE uses a real synthesized playbook; diagnostics may fall back lazily."""

from __future__ import annotations

import asyncio

import pytest

from app.core.config import PROJECT_ROOT, Settings
from app.evaluation import paper968 as B
from app.evaluation import quality as Q
from app.evaluation.generated import GeneratedPaperEnv
from app.evaluation.protocol import GENERATE, REQUEST, REUSE
from app.intent.mock_llm_client import MockLLMClient
from app.library.embedder import HashingEmbedder
from app.library.index import NumpyIndex
from app.playbook.vocabulary import Vocabulary
from app.storage.database import Database

DATA = PROJECT_ROOT / "app" / "data"


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    settings = Settings(_env_file=None, generation_pool_size=3, population_size=6, max_generations=2)
    vocabulary = Vocabulary.load(DATA / "action_vocabulary.json", DATA / "intent_goals.json")
    root = tmp_path_factory.mktemp("generated")
    database = Database(f"sqlite:///{(root / 'lib.db').as_posix()}")
    database.create_schema()
    benchmark = B.build()
    library, entries = B.build_library(benchmark, settings, vocabulary, HashingEmbedder(), NumpyIndex(384), database)
    env = GeneratedPaperEnv(benchmark, settings, vocabulary, library, entries)
    env.quality = Q.QualityRun(Q.build_kit(settings, MockLLMClient(), None), env, library, Q.Cache(root / "cache"))
    yield env
    database.dispose()


def test_generate_is_scored_on_the_synthesized_playbook_and_cached(env):
    scenario = next(s for s in env.benchmark.scenarios if s.cell == "none/full")
    decision = env.decide(scenario, scenario.available, scenario.obtainable, None, 1)
    generate = decision.model_copy(update={"action": GENERATE})
    coverage, retained, latency, extra = asyncio.run(env.run_terminal(scenario, generate, generate, scenario.available))
    record = env.quality.cache.get("generation_only", scenario.name, Q.state_key(scenario.available))
    assert record is not None and ("synthesis_fitness" in extra or "synthesis_failed" in extra)
    assert latency == pytest.approx(record.latency_s)  # measured synthesis time, not the paper's 238 s
    playbook = B.Playbook.model_validate(record.playbook) if hasattr(B, "Playbook") else None
    if playbook is not None and not record.failed:
        assert (coverage, retained) == env.check(scenario, playbook, scenario.available)
    assert env.stats()["syntheses"] == 1


def test_request_then_generate_adds_the_evidence_wait(env):
    assert env.path_latency(REQUEST, GENERATE, 100.0) == pytest.approx(100.0 + env.latency.request - env.latency.reuse)
    assert env.path_latency(REUSE, REUSE, None) == env.latency.of(REUSE, REUSE)


def test_lazy_alternatives_do_not_synthesize_new_states(env):
    scenario = next(s for s in env.benchmark.scenarios if s.cell == "strong/full")
    before, estimated = env.stats()["syntheses"], env.estimated_alternatives
    options = asyncio.run(env.alternatives(scenario))
    assert GENERATE in options and env.stats()["syntheses"] == before
    assert env.estimated_alternatives > estimated
