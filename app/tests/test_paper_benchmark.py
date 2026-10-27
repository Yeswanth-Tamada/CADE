"""The paper-matched routing benchmark (paper 968 Section 6.1, Table 2) and its separation from the extended one."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from collections import Counter
from types import SimpleNamespace

import pytest

from app.core.config import PROJECT_ROOT
from app.enrichment.availability import AvailabilityProfile
from app.evaluation import paper968 as B
from app.evaluation import protocol as P
from app.library.embedder import HashingEmbedder
from app.library.index import NumpyIndex
from app.playbook.validator import PlaybookValidator
from app.playbook.vocabulary import Vocabulary
from app.schemas.enums import EvidenceCategory as C
from app.schemas.routing import RoutingAction
from app.storage.database import Database

DATA = PROJECT_ROOT / "app" / "data"
REUSE, REQUEST, GENERATE = RoutingAction.REUSE, RoutingAction.REQUEST_EVIDENCE, RoutingAction.SYNTHESIZE


@pytest.fixture(scope="module")
def benchmark() -> B.Benchmark:
    return B.build()


@pytest.fixture(scope="module")
def vocabulary() -> Vocabulary:
    return Vocabulary.load(DATA / "action_vocabulary.json", DATA / "intent_goals.json")


# --- construction ---------------------------------------------------------------------------------


def test_the_catalog_is_pinned_and_offline(benchmark):
    provenance = benchmark.catalog["provenance"]
    assert provenance["splunk_attack_data"]["commit"] == "115785491ca34054ac441a2ca9e87fba40b42a6b"
    assert provenance["attack"]["version"] == "20.0" and not provenance["splunk_attack_data"]["yaml_unavailable"]
    assert benchmark.catalog["event_categories"]["Security 4688"] == "process_telemetry"  # paper Section 6.1
    assert all(set(r["sources"]) <= {"sysmon", "security", "powershell"} for r in benchmark.catalog["recordings"])


def test_generation_is_deterministic(benchmark):
    again = B.build()
    assert [s.summary() for s in again.scenarios] == [s.summary() for s in benchmark.scenarios]
    assert again.library_techniques == benchmark.library_techniques


def test_table_2_composition_and_label_balance(benchmark):
    scenarios = benchmark.scenarios
    assert len(scenarios) == 96 and not benchmark.infeasible
    assert Counter(s.expected for s in scenarios) == {REUSE: 32, REQUEST: 32, GENERATE: 32}
    assert Counter(s.tactic for s in scenarios) == {t: 12 for t in B.TACTIC_ORDER}
    weights = {f"{s}/{e}": n for s, e, _, n in B.TABLE_2}
    assert list(weights.values()) == [2, 2, 1, 2, 1, 1, 1, 1, 1]
    for tactic in B.TACTIC_ORDER:
        assert Counter(s.cell for s in scenarios if s.tactic == tactic) == weights
    assert len({s.name for s in scenarios}) == 96
    for tactic in B.TACTIC_ORDER:  # recordings are distinct within a tactic unless a flagged relaxation reused one
        ids = [s.recording.id for s in scenarios if s.tactic == tactic and not (s.relaxed or "").startswith("B")]
        assert len(ids) == len(set(ids))


def test_the_ontology_follows_the_derivation_rule(benchmark):
    for tactic in B.TACTIC_ORDER:
        techniques = [t for t in benchmark.footprint if benchmark.tactic_of[t] == tactic]
        share = {c: sum(c in benchmark.footprint[t] for t in techniques) / len(techniques) for c in B.SEVEN}
        assert benchmark.critical[tactic] == {c for c, v in share.items() if v >= 0.5}


def test_every_label_follows_the_paper_construction_rules(benchmark):
    library = set(benchmark.library_techniques)
    for s in benchmark.scenarios:
        crit = benchmark.critical[s.tactic]
        same_tactic = {t for t in library if benchmark.tactic_of[t] == s.tactic}
        kept = same_tactic - s.withheld_techniques
        assert s.available | s.hidden == s.recording.categories and not s.available & s.hidden
        assert s.obtainable <= s.hidden and s.critical == s.recording.categories & crit
        if s.suitability == "strong":
            assert s.technique in kept and not s.withheld_techniques
        elif s.suitability == "weak":
            assert kept == {s.related} and s.related != s.technique and benchmark.tactic_of[s.related] == s.tactic
        else:
            assert not kept  # no suitable playbook: every playbook of the tactic is withheld
        if s.expected == REUSE:
            assert s.suitability == "strong" and s.critical <= s.available and s.evidence in ("full", "partial")
        if s.expected == REQUEST:
            assert s.critical - s.available and (s.critical - s.available) <= s.obtainable
            runner = s.technique if s.suitability == "strong" else s.related
            assert benchmark.footprint[runner] & crit <= s.available | s.obtainable  # a stored playbook could run
        if s.expected == GENERATE:
            assert s.suitability == "none" or (s.suitability == "weak" and not s.obtainable)
        if s.evidence == "partial":
            assert s.hidden and not s.hidden & crit
        if s.evidence == "critical_missing":
            assert len(s.hidden) == 1 and s.hidden <= crit
        if s.evidence.startswith("sparse"):
            assert 1 <= len(s.available) <= 2


def test_technique_playbooks_are_valid_and_split_by_their_own_intent(benchmark, vocabulary):
    validator = PlaybookValidator(vocabulary)
    required = B.requirements_by_own_intent(benchmark, vocabulary)
    for technique in benchmark.library_techniques[:20]:
        playbook = B.technique_playbook(benchmark, technique, vocabulary)
        assert validator.validate(playbook).valid and technique in playbook.name
        r = required(playbook, frozenset({C.COMMAND_LINE}))  # the incident's own critical set is not used
        crit = benchmark.critical[benchmark.tactic_of[technique]]
        assert set(r.critical) == benchmark.footprint[technique] & crit
        assert set(r.supporting) == benchmark.footprint[technique] - crit


# --- environment (hashing embedder: fast and offline; the real runs use the paper's embedder) ----------------


@pytest.fixture(scope="module")
def env(benchmark, vocabulary, tmp_path_factory):
    from app.core.config import Settings

    settings = Settings(_env_file=None)
    database = Database(f"sqlite:///{(tmp_path_factory.mktemp('paper') / 'library.db').as_posix()}")
    database.create_schema()
    library, entries = B.build_library(benchmark, settings, vocabulary, HashingEmbedder(), NumpyIndex(384), database)
    yield B.PaperEnv(benchmark, settings, vocabulary, library, entries)
    database.dispose()


def _scenario(benchmark, cell):
    return next(s for s in benchmark.scenarios if s.cell == cell)


def test_the_execution_stand_in_rewards_the_construction(env, benchmark):
    strong = _scenario(benchmark, "strong/full")
    first = env.decide(strong, strong.available, strong.obtainable, None, 1)
    reuse = env.execute(strong, REUSE, first, strong.available)
    assert first.top_candidate.name.startswith(strong.technique) and reuse == (1.0, True)
    none = _scenario(benchmark, "none/full")
    decision = env.decide(none, none.available, none.obtainable, None, 1)
    assert env.execute(none, REUSE, decision, none.available) == (0.0, False)  # another tactic's playbook
    assert env.execute(none, GENERATE, decision, none.available) == (1.0, True)
    missing = _scenario(benchmark, "strong/critical_missing")
    assert env.execute(missing, GENERATE, first, missing.available)[1] is False
    assert env.execute(missing, GENERATE, first, missing.available | missing.obtainable) == (1.0, True)


def test_reward_alternatives_single_out_request_when_critical_evidence_is_obtainable(env, benchmark):
    scenario = _scenario(benchmark, "strong/critical_missing")
    options = asyncio.run(env.alternatives(scenario))
    best = P.best_first_rewards(options, 240.0, P.RewardWeights())
    assert P.reward_preference(best, REQUEST)[0] == "unique_expected"
    sparse = _scenario(benchmark, "weak/sparse_unobtainable")  # nothing can be retained: the paper's label is a tie
    best = P.best_first_rewards(asyncio.run(env.alternatives(sparse)), 240.0, P.RewardWeights())
    assert P.reward_preference(best, GENERATE)[0] == "tie_with_expected"


def test_the_goal_model_switch_only_changes_execution(env, benchmark):
    partial = next(s for s in benchmark.scenarios if s.cell == "strong/partial" and len(s.recording.categories) == 2)
    decision = env.decide(partial, partial.available, partial.obtainable, None, 1)
    env_all = B.PaperEnv(env.benchmark, env.settings, env.vocabulary, env.library, env.entries, goals="all")
    assert env.execute(partial, GENERATE, decision, partial.available) == (1.0, True)
    assert env_all.execute(partial, GENERATE, decision, partial.available) == (0.5, False)  # IC_exec < 0.8


# --- separation from the extended benchmark ---------------------------------------------------------


def _eval_routing():
    spec = importlib.util.spec_from_file_location("eval_routing_script", PROJECT_ROOT / "scripts" / "eval_routing.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def extended(taxonomy, tmp_path_factory):
    from app.core.config import Settings
    from app.library.library import PlaybookLibrary
    from app.storage.library_repository import LibraryRepository

    module = _eval_routing()
    settings = Settings(_env_file=None)
    vocabulary = Vocabulary.load(DATA / "action_vocabulary.json", DATA / "intent_goals.json", taxonomy)
    database = Database(f"sqlite:///{(tmp_path_factory.mktemp('extended') / 'library.db').as_posix()}")
    database.create_schema()
    library = PlaybookLibrary(LibraryRepository(database), HashingEmbedder(), NumpyIndex(384), vocabulary,
                              PlaybookValidator(vocabulary), k=settings.routing_k)  # fmt: skip
    library.seed_if_empty(DATA / "seed_playbooks.json")
    library.load()
    env = module.Env(settings, taxonomy, vocabulary, AvailabilityProfile.load(DATA / "evidence_availability.json"),
                     library, 1.0, P.PathLatency())  # fmt: skip
    legacy, legacy_changes = asyncio.run(module.build_scenarios(env, None, "legacy"))
    corrected, changes = asyncio.run(module.build_scenarios(env, None, "corrected"))
    yield SimpleNamespace(legacy=legacy, legacy_changes=legacy_changes, corrected=corrected, changes=changes, env=env)
    database.dispose()


def test_the_legacy_benchmark_keeps_its_original_123_scenarios(extended):
    assert len(extended.legacy) == 123 and not extended.legacy_changes
    assert Counter(s.expected for s in extended.legacy) == {REUSE: 28, REQUEST: 41, GENERATE: 54}
    weak = [s for s in extended.legacy if s.suitability == "weak"]
    assert all(s.withheld == frozenset({s.seed_id}) for s in weak)  # the original (defective) weak construction


def test_the_corrected_benchmark_only_changes_weak_cells_and_logs_every_change(extended):
    legacy = {s.name: s for s in extended.legacy}
    corrected = {s.name: s for s in extended.corrected}
    for name, s in corrected.items():
        if s.suitability != "weak":
            assert s.withheld == legacy[name].withheld and s.expected == legacy[name].expected
    weak_legacy = {n for n, s in legacy.items() if s.suitability == "weak"}
    logged = {c["scenario"] for c in extended.changes}
    assert logged == weak_legacy and set(legacy) - set(corrected) <= weak_legacy
    for change in extended.changes:
        assert change["change"] in ("excluded", "library_changed") and (change["change"] != "excluded" or change["reason"])
    for s in corrected.values():
        if s.suitability == "weak":  # exactly one same-tactic playbook is kept: the designated related one
            kept = [e for e in extended.env.library.entries() if e.entry_id not in s.withheld
                    and extended.env.same_tactic(e.intent_label, s.intent)]  # fmt: skip
            assert [e.entry_id for e in kept] == [s.related_id]


def test_benchmarks_never_share_scenario_names(benchmark, extended):
    paper = {s.name for s in benchmark.scenarios}
    assert not paper & {s.name for s in extended.legacy} and not paper & {s.name for s in extended.corrected}
    assert all(name.split("/")[0] in B.TACTIC_ORDER for name in paper)
