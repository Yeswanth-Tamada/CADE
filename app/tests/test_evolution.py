"""CADE evolution: operators, coverage-stratified selection, termination (milestone M6, paper Section 5.3)."""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from app.core.config import PROJECT_ROOT
from app.evolution.engine import CADEEngine, EvolutionConfig, SynthesisError
from app.evolution.operators import MUTATIONS, MutationContext, Operators
from app.evolution.selection import (
    TOURNAMENT_SIZE,
    elites,
    niches,
    proportional_survivors,
    select_donor,
    select_parent,
    strata,
    stratified_survivors,
)
from app.fitness.coverage import CoverageModel
from app.fitness.fitness import FitnessEvaluator
from app.fitness.judge import HeuristicSoundness
from app.playbook.builder import Draft
from app.playbook.repair import PlaybookRepairer
from app.playbook.validator import PlaybookValidator, initial_artifacts
from app.playbook.vocabulary import Vocabulary
from app.schemas.playbook import Playbook, PlaybookOrigin
from app.synthesis.constructive import ConstructiveGenerator
from app.synthesis.context import SynthesisContext
from app.synthesis.seeding import AdaptiveSeeder
from app.tests.test_fitness import SMALL
from app.tests.test_playbook_dag import pb, seeds

DATA = PROJECT_ROOT / "app" / "data"
INTENT = "suspicious_process_execution"
AVAILABLE = frozenset({"host_id", "user_id"})

DUPLICATED = pb("n1:extract_entities n2:query_process_events n3:query_process_events n4:analyze_process_tree "
                "n5:generate_incident_report",
                "n1>n2:host_id n1>n3:host_id n2>n4:process_list n3>n5:findings n4>n5:findings", intent=INTENT)  # fmt: skip
UNGUARDED = pb("n1:extract_entities n2:query_process_events n3:isolate_host n4:generate_incident_report",
               "n1>n2:host_id n1>n3:host_id n2>n4:findings n3>n4:findings", intent=INTENT)  # fmt: skip


@pytest.fixture(scope="module")
def vocabulary(taxonomy) -> Vocabulary:
    return Vocabulary.load(DATA / "action_vocabulary.json", DATA / "intent_goals.json", taxonomy)


@pytest.fixture(scope="module")
def coverage(vocabulary, taxonomy) -> CoverageModel:
    return CoverageModel(vocabulary, taxonomy)


@pytest.fixture(scope="module")
def validator(vocabulary) -> PlaybookValidator:
    return PlaybookValidator(vocabulary)


@pytest.fixture(scope="module")
def repairer(validator) -> PlaybookRepairer:
    return PlaybookRepairer(validator)


@pytest.fixture
def evaluator(coverage, validator) -> FitnessEvaluator:
    return FitnessEvaluator(coverage, validator, HeuristicSoundness(coverage, validator))


@pytest.fixture
def operators(coverage, repairer) -> Operators:
    return Operators(coverage, repairer)


def _seed(intent: str) -> Playbook:
    return Playbook.model_validate(next(s for s in seeds() if s["intent_label"] == intent))


def _context(intent=INTENT) -> SynthesisContext:
    return SynthesisContext(incident_id="INC-T", intent_label=intent, intent_description="x", summary="SYNTHETIC",
                            available=AVAILABLE, observed_categories=(), missing_critical=(), missing_supporting=(),
                            techniques=())  # fmt: skip


def _seeds(coverage, repairer, evaluator, intent=INTENT, count=8):
    builder = ConstructiveGenerator(coverage, repairer)
    pool = [p for p in (builder.build(_context(intent), random.Random(s), goal_rate=0.4) for s in range(count)) if p]
    return AdaptiveSeeder(evaluator, k_min=2, k_max=6).select(pool, intent, AVAILABLE, random.Random(0)).seeds


def _is_destructive(vocabulary, playbook) -> bool:
    return any(vocabulary.get(n.action).destructive for n in playbook.nodes)


def _mutation_context(coverage, parent, *, directed=True, intent=INTENT) -> MutationContext:
    return MutationContext(intent, initial_artifacts(AVAILABLE), coverage.profile(parent, intent), directed)


# --- operators -------------------------------------------------------------------------------


def test_the_six_paper_mutations():
    assert MUTATIONS == ("add_goal_node", "add_evidence_node", "remove_redundant_node", "rebind_tool", "rewire_edge",
                         "insert_verification")  # fmt: skip


@pytest.mark.parametrize("name", MUTATIONS)
def test_each_mutation_yields_valid_offspring_or_nothing(name, operators, coverage, validator, vocabulary):
    parent = {"remove_redundant_node": DUPLICATED, "insert_verification": UNGUARDED,
              "rewire_edge": _seed("command_and_control_persistence")}.get(name, SMALL)  # fmt: skip
    produced = 0
    for trial in range(25):
        rng = random.Random(trial)
        draft = Draft.from_playbook(vocabulary, parent, origin=PlaybookOrigin.EVOLVED)
        if not getattr(operators, name)(draft, _mutation_context(coverage, parent, directed=trial % 2 == 0), rng):
            continue
        child = operators._finish(draft, AVAILABLE, rng)
        if child is not None:
            produced += 1
            report = validator.validate(child, AVAILABLE)
            assert report.valid and report.safe
            assert len(child.nodes) <= validator.node_cap
    assert produced > 0, f"{name} never produced a valid child"


def test_gap_directed_additions_close_a_gap_of_their_kind(operators, coverage, vocabulary):
    before = coverage.profile(SMALL, INTENT)
    for name, kind in (("add_goal_node", "goal"), ("add_evidence_node", "category")):
        for trial in range(8):
            rng = random.Random(trial)
            draft = Draft.from_playbook(vocabulary, SMALL, origin=PlaybookOrigin.EVOLVED)
            assert getattr(operators, name)(draft, _mutation_context(coverage, SMALL), rng)
            after = coverage.profile(operators._finish(draft, AVAILABLE, rng), INTENT)
            assert all(a >= b for a, b in zip(after.vector, before.vector))
            closed = set(before.gaps()) - set(after.gaps())
            assert any(gap.kind == kind for gap in closed)
    assert operators.gap_directed == 16


def test_gap_directed_additions_use_the_generators_proposal(operators, coverage, vocabulary):
    gaps = [g for g in coverage.profile(SMALL, INTENT).gaps() if g.kind == "goal"]
    operators.proposals = {gap: sorted(a.id for a in coverage.closers(gap, INTENT))[-1] for gap in gaps}
    draft = Draft.from_playbook(vocabulary, SMALL, origin=PlaybookOrigin.EVOLVED)
    assert operators.add_goal_node(draft, _mutation_context(coverage, SMALL), random.Random(4))
    added = {n.action for n in draft.nodes.values()} - set(SMALL.action_multiset())
    assert added & set(operators.proposals.values()) and operators.proposal_used == 1


def test_directed_additions_do_nothing_without_gaps(operators, coverage, vocabulary):
    complete = _seed("denial_of_service")
    assert coverage.profile(complete, "denial_of_service").complete
    context = _mutation_context(coverage, complete, intent="denial_of_service")
    for name in ("add_goal_node", "add_evidence_node"):
        draft = Draft.from_playbook(vocabulary, complete)
        assert not getattr(operators, name)(draft, context, random.Random(0))


def test_remove_redundant_node_drops_repeated_work(operators, coverage, vocabulary):
    draft = Draft.from_playbook(vocabulary, DUPLICATED)
    # n3 repeats n2; n4 adds no coverage and feeds only the report; n2 feeds a working step
    assert operators._redundant(draft, INTENT) == ["n3", "n4"]
    assert operators.remove_redundant_node(draft, _mutation_context(coverage, DUPLICATED), random.Random(0))
    assert len(draft.nodes) == 4 and "n2" in draft.nodes


def test_rebind_tool_keeps_phase_and_evidence_but_changes_the_tool(operators, coverage, vocabulary):
    draft = Draft.from_playbook(vocabulary, SMALL)
    assert operators.rebind_tool(draft, _mutation_context(coverage, SMALL), random.Random(1))
    old = vocabulary.get("query_process_events")
    new = vocabulary.get(draft.nodes["n2"].action)
    assert new.binding != old.binding and new.phase == old.phase and new.categories & old.categories


def test_insert_verification_guards_an_unguarded_destructive_step(operators, coverage, validator, vocabulary):
    assert not validator.validate(UNGUARDED, AVAILABLE).safe
    draft = Draft.from_playbook(vocabulary, UNGUARDED)
    assert operators.insert_verification(draft, _mutation_context(coverage, UNGUARDED), random.Random(0))
    assert draft.verified("n3")
    assert not operators.insert_verification(Draft.from_playbook(vocabulary, SMALL),
                                             _mutation_context(coverage, SMALL), random.Random(0))  # fmt: skip


def test_crossover_splices_donor_coverage_into_the_recipients_gaps(operators, coverage, validator):
    first, second = _seed(INTENT), _seed("malware_investigation")
    first_profile, second_profile = coverage.profile(first, INTENT), coverage.profile(second, INTENT)
    wanted = set(first_profile.gaps()) & set(second_profile.covered())
    children = [operators.crossover(first, first_profile, second, second_profile, INTENT, AVAILABLE, random.Random(t))
                for t in range(30)]  # fmt: skip
    children = [c for c in children if c is not None]
    if wanted:
        assert children
    for child in children:
        assert validator.validate(child, AVAILABLE).valid and len(child.nodes) <= validator.node_cap
        assert wanted - set(coverage.profile(child, INTENT).gaps())  # closes at least one wanted gap


def test_crossover_needs_a_donor_covering_a_gap(operators, coverage):
    profile = coverage.profile(SMALL, INTENT)
    assert operators.crossover(SMALL, profile, SMALL, profile, INTENT, AVAILABLE, random.Random(0)) is None


def test_operators_never_introduce_destructive_actions(operators, coverage, vocabulary):
    parents = [_seed(i) for i in (INTENT, "malware_investigation", "data_exfiltration")]
    rng = random.Random(5)
    for _ in range(120):
        parent = rng.choice(parents)
        child = operators.mutate(parent, coverage.profile(parent, INTENT), INTENT, AVAILABLE, rng,
                                 directed=rng.random() < 0.5)  # fmt: skip
        if child is not None:
            assert not _is_destructive(vocabulary, child)
        donor = rng.choice(parents)
        other = operators.crossover(parent, coverage.profile(parent, INTENT), donor, coverage.profile(donor, INTENT),
                                    INTENT, AVAILABLE, rng)  # fmt: skip
        if other is not None:
            assert not _is_destructive(vocabulary, other)


def test_destructive_parent_offspring_stay_verified(operators, coverage, validator):
    parent = _seed("ransomware_investigation")  # contains isolate_host after verify_host_compromise
    rng = random.Random(9)
    for _ in range(60):
        child = operators.mutate(parent, coverage.profile(parent, "ransomware_investigation"), "ransomware_investigation",
                                 AVAILABLE, rng, directed=False)  # fmt: skip
        if child is not None:
            assert validator.validate(child, AVAILABLE).safe


# --- selection --------------------------------------------------------------------------------


def _pool(coverage, repairer, evaluator, count=16):
    builder = ConstructiveGenerator(coverage, repairer)
    pool = [builder.build(_context(), random.Random(s), goal_rate=0.2 + 0.05 * s) for s in range(count)]
    return [c for c in (evaluator.evaluate(p, INTENT, AVAILABLE) for p in pool if p) if c]


def test_parents_come_from_binary_tournaments_inside_coverage_strata(coverage, repairer, evaluator):
    candidates = _pool(coverage, repairer, evaluator)
    groups = strata(candidates)
    assert len(groups) >= 2 and TOURNAMENT_SIZE == 2
    assert all(len({c.profile.covered_count for c in g}) == 1 for g in groups.values())
    picked = {select_parent(candidates, random.Random(seed), stratified=True).profile.covered_count for seed in range(60)}
    assert picked == set(groups)  # every stratum gets chosen, not just the fittest lineage
    for stratified in (True, False):
        assert select_parent(candidates, random.Random(1), stratified=stratified) in candidates


def test_donors_cover_part_of_the_recipients_gaps(coverage, repairer, evaluator):
    candidates = _pool(coverage, repairer, evaluator)
    recipient = min(candidates, key=lambda c: c.profile.covered_count)
    donor = select_donor(candidates, recipient, random.Random(0))
    assert donor is not None and set(recipient.profile.gaps()) & set(donor.profile.covered())


def test_elites_are_judged_candidates_only(coverage, repairer, evaluator):
    candidates = _pool(coverage, repairer, evaluator)
    best = max(candidates, key=lambda c: c.fitness)
    estimated = replace(best, scores=best.scores.model_copy(update={"os_source": "estimated"}))
    rest = [c for c in candidates if c.key != best.key]
    chosen = elites(rest + [estimated], 2)
    assert len(chosen) == 2 and estimated not in chosen and all(c.judged for c in chosen)


def test_mu_plus_lambda_ablation_keeps_every_niche_and_the_elites(coverage, repairer, evaluator):
    candidates = _pool(coverage, repairer, evaluator)
    groups = niches(candidates)
    assert len(groups) >= 3
    survivors = stratified_survivors(candidates, size=len(groups) + 1, elite_count=1)
    assert survivors[0] is max(candidates, key=lambda c: c.fitness)  # elite carried over unmodified
    assert {c.profile.vector for c in survivors} == set(groups)  # every coverage region represented
    proportional = proportional_survivors(candidates, size=5, elite_count=1, rng=random.Random(0))
    assert len(proportional) == 5 and proportional[0] is survivors[0]


# --- engine -----------------------------------------------------------------------------------


def _config(**changes) -> EvolutionConfig:
    return replace(EvolutionConfig(population_size=10, max_generations=6, plateau_generations=3), **changes)


@pytest.mark.anyio
async def test_engine_improves_or_keeps_the_best_and_returns_a_valid_playbook(coverage, repairer, evaluator, validator):
    seeds_ = _seeds(coverage, repairer, evaluator)
    engine = CADEEngine(evaluator, Operators(coverage, repairer), _config(fitness_threshold=1.01, plateau_epsilon=0.0))
    result = await engine.run(seeds_, INTENT, AVAILABLE, random.Random(42))
    assert validator.validate(result.best.playbook, AVAILABLE).valid
    best = [h.best_fitness for h in result.history]
    assert best == sorted(best)  # never decreases
    assert result.best.fitness >= max(s.fitness for s in seeds_)
    assert result.best.fitness == pytest.approx(best[-1])
    assert all(h.population <= 10 for h in result.history)  # generational replacement: elites + offspring
    assert result.evaluations > 0 and result.termination in {"full_coverage", "plateau", "generation_budget"}
    assert sum(result.operator_counts[name] for name in MUTATIONS) > 0


@pytest.mark.anyio
async def test_termination_on_full_coverage(coverage, repairer, evaluator):
    complete = evaluator.evaluate(_seed("denial_of_service"), "denial_of_service", AVAILABLE)
    engine = CADEEngine(evaluator, Operators(coverage, repairer), _config())
    result = await engine.run([complete], "denial_of_service", AVAILABLE, random.Random(1))
    assert result.termination == "full_coverage" and result.generations == 0
    assert result.best.profile.complete


@pytest.mark.anyio
async def test_termination_on_plateau_and_budget(coverage, repairer, evaluator):
    seeds_ = _seeds(coverage, repairer, evaluator)
    plateau = await CADEEngine(evaluator, Operators(coverage, repairer),
                               _config(fitness_threshold=1.01, plateau_generations=1, plateau_epsilon=0.5)
                               ).run(seeds_, INTENT, AVAILABLE, random.Random(2))  # fmt: skip
    assert plateau.termination == "plateau" and plateau.generations == 1
    # the window: no stop before plateau_generations generations have passed
    windowed = await CADEEngine(evaluator, Operators(coverage, repairer),
                                _config(fitness_threshold=1.01, plateau_generations=3, plateau_epsilon=0.5)
                                ).run(seeds_, INTENT, AVAILABLE, random.Random(2))  # fmt: skip
    assert windowed.termination == "plateau" and windowed.generations == 3
    budget = await CADEEngine(evaluator, Operators(coverage, repairer),
                              _config(coverage_termination=False, max_generations=3)
                              ).run(seeds_, INTENT, AVAILABLE, random.Random(2))  # fmt: skip
    assert budget.termination == "generation_budget" and budget.generations == 3


@pytest.mark.anyio
async def test_seeded_runs_are_reproducible(coverage, repairer, evaluator):
    seeds_ = _seeds(coverage, repairer, evaluator)
    runs = []
    for _ in range(2):
        engine = CADEEngine(evaluator, Operators(coverage, repairer), _config(max_generations=3))
        runs.append(await engine.run(seeds_, INTENT, AVAILABLE, random.Random(123)))
    assert runs[0].best.key == runs[1].best.key
    assert [h.best_fitness for h in runs[0].history] == [h.best_fitness for h in runs[1].history]


@pytest.mark.anyio
async def test_fitness_only_variant_runs(coverage, repairer, evaluator, validator):
    seeds_ = _seeds(coverage, repairer, evaluator)
    config = _config(coverage_init=False, coverage_variation=False, coverage_selection=False,
                     coverage_termination=False, generational_replacement=False, max_generations=3)  # fmt: skip
    result = await CADEEngine(evaluator, Operators(coverage, repairer), config).run(seeds_, INTENT, AVAILABLE, random.Random(7))
    assert result.termination == "generation_budget" and validator.validate(result.best.playbook, AVAILABLE).valid
    assert result.operator_counts["gap_directed"] == 0


@pytest.mark.anyio
async def test_no_seeds_is_a_synthesis_error(coverage, repairer, evaluator):
    with pytest.raises(SynthesisError):
        await CADEEngine(evaluator, Operators(coverage, repairer), _config()).run([], INTENT, AVAILABLE, random.Random(0))


def test_config_defaults_follow_the_paper(settings):
    from app.core.config import Settings

    config = EvolutionConfig.from_settings(Settings(_env_file=None))
    assert (config.population_size, config.max_generations, config.elite_count) == (30, 20, 2)
    assert (config.plateau_generations, config.plateau_epsilon, config.fitness_threshold) == (3, 0.01, 0.0)
    assert config.generational_replacement
    assert EvolutionConfig.from_settings(settings).population_size == settings.population_size
