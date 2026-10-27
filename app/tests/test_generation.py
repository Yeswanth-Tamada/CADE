"""LLM candidate generation, the constructive fallback and adaptive K (milestone M5)."""

from __future__ import annotations

import json
import random

import pytest

from app.core.config import PROJECT_ROOT
from app.core.errors import LLMTimeoutError, LLMUnavailableError
from app.fitness.coverage import CoverageModel, jaccard
from app.fitness.fitness import FitnessEvaluator
from app.fitness.judge import HeuristicSoundness
from app.intent.llm_client import model_family
from app.intent.llm_extractor import IntentExtractor
from app.intent.mock_llm_client import MockLLMClient
from app.playbook.repair import PlaybookRepairer
from app.playbook.validator import PlaybookValidator
from app.playbook.vocabulary import Vocabulary
from app.schemas.enums import EvidenceCategory
from app.schemas.playbook import Playbook, PlaybookOrigin
from app.synthesis.constructive import ConstructiveGenerator
from app.synthesis.context import SynthesisContext
from app.synthesis.generator import CandidateGenerator, candidate_to_playbook
from app.synthesis.prompts import DATA_END, DATA_START, PlaybookCandidate, build_generation_prompt, candidate_schema
from app.synthesis.seeding import AdaptiveSeeder, agglomerate
from app.tests.conftest import good_output, make_incident
from app.tests.test_playbook_dag import seeds

DATA = PROJECT_ROOT / "app" / "data"
INTENT = "suspicious_process_execution"


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


@pytest.fixture(scope="module")
def evaluator(coverage, validator) -> FitnessEvaluator:
    return FitnessEvaluator(coverage, validator, HeuristicSoundness(coverage, validator))


@pytest.fixture
def context() -> SynthesisContext:
    return SynthesisContext(
        incident_id="INC-TEST", intent_label=INTENT, intent_description="Suspicious process execution",
        summary="SYNTHETIC: Word spawned encoded PowerShell on WS-FIN-042.", available=frozenset({"host_id", "user_id"}),
        observed_categories=(EvidenceCategory.PROCESS_TELEMETRY,), missing_critical=(EvidenceCategory.COMMAND_LINE,),
        missing_supporting=(EvidenceCategory.DNS,), techniques=("T1059.001",),
    )  # fmt: skip


def _generator(llm, coverage, repairer, pool_size=4) -> CandidateGenerator:
    return CandidateGenerator(llm, coverage, repairer, ConstructiveGenerator(coverage, repairer), pool_size=pool_size)


def candidate(*steps, name="candidate") -> dict:
    return {"name": name, "rationale": "r", "steps": [{"id": s[0], "action": s[1], "depends_on": list(s[2:])} for s in steps]}


GOOD = candidate(("s1", "extract_entities"), ("s2", "query_process_events", "s1"), ("s3", "analyze_process_tree", "s2"),
                 ("s4", "generate_incident_report", "s3"))  # fmt: skip


# --- parsing ---------------------------------------------------------------------------------


def test_steps_become_typed_edges(vocabulary, validator):
    playbook = candidate_to_playbook(PlaybookCandidate.model_validate(GOOD), INTENT, vocabulary)
    edges = {(e.source, e.target, e.artifact) for e in playbook.edges}
    assert edges == {("n1", "n2", "host_id"), ("n2", "n3", "process_list"), ("n3", "n4", "findings")}
    assert validator.validate(playbook).valid


def test_malformed_steps_are_neutralised(vocabulary, repairer):
    weird = candidate(("s1", "extract_entities"), ("s2", "Run Mimikatz!", "s1"), ("s3", "query_dns_activity", "s9", "s1"),
                      ("s3", "triage_alert"), ("s4", "generate_incident_report", "s3", "s4"))  # fmt: skip
    playbook = candidate_to_playbook(PlaybookCandidate.model_validate(weird), INTENT, vocabulary)
    assert playbook.nodes[1].action == "unknown_action"
    assert len(playbook.nodes) == 4  # the duplicate step id is ignored
    repaired = repairer.repair(playbook)
    assert repaired is not None and "unknown_action" not in repaired.action_multiset()


def test_schema_restricts_actions_to_the_vocabulary(vocabulary):
    schema = candidate_schema(vocabulary)
    assert schema["title"] == "PlaybookCandidate"
    assert set(schema["$defs"]["CandidateStep"]["properties"]["action"]["enum"]) == set(vocabulary.actions)


def test_generation_prompt_carries_intent_evidence_examples_and_directive(context, vocabulary, coverage):
    example = Playbook.model_validate(seeds()[0])
    prompt = build_generation_prompt(context, vocabulary, coverage, [example], "Emphasis: minimal.")
    assert DATA_START in prompt and DATA_END in prompt
    assert "Missing critical evidence: command_line" in prompt
    assert "Reconstruct the process tree" in prompt  # a goal of the intent
    assert "isolate_host [containment DESTRUCTIVE]" in prompt
    assert f"Example playbook '{example.name}'" in prompt and prompt.rstrip().endswith("Emphasis: minimal.")


# --- generator -------------------------------------------------------------------------------


@pytest.mark.anyio
async def test_mixed_temperatures_and_untrusted_output_handling(context, coverage, repairer, validator):
    unsafe = candidate(("s1", "extract_entities"), ("s2", "isolate_host", "s1"), ("s3", "generate_incident_report", "s2"))
    llm = MockLLMClient([GOOD, "not json", {"name": "x", "steps": []}, unsafe])
    pool, report = await _generator(llm, coverage, repairer).generate(context, [], random.Random(1))
    assert [c["temperature"] for c in llm.calls] == [0.2, 0.6, 0.9, 1.2]
    assert [c["json_schema"]["title"] for c in llm.calls] == ["PlaybookCandidate"] * 4
    assert report.llm_calls == 4 and report.unparseable == 2 and report.llm_valid == 2
    assert report.constructive_used == 0 and report.fallback_reason is None
    for playbook in pool:
        assert validator.validate(playbook, context.available).valid
    repaired_unsafe = pool[1]
    isolate = next(n for n in repaired_unsafe.nodes if n.action == "isolate_host")
    assert validator.verification_ancestors(repaired_unsafe, isolate.node_id)


@pytest.mark.anyio
async def test_prior_playbooks_join_the_pool(context, coverage, repairer):
    example = Playbook.model_validate(next(s for s in seeds() if s["intent_label"] == INTENT))
    pool, report = await _generator(MockLLMClient([GOOD] * 4), coverage, repairer).generate(context, [example], random.Random(1))
    assert report.examples_used == 1 and any(p.origin == PlaybookOrigin.SEED for p in pool)


@pytest.mark.anyio
async def test_unavailable_generator_falls_back_to_constructive(context, coverage, repairer, validator):
    llm = MockLLMClient([LLMUnavailableError("down")])
    pool, report = await _generator(llm, coverage, repairer).generate(context, [], random.Random(2))
    assert report.llm_calls == 1 and report.llm_failures == 1  # stops asking a model that is not there
    assert "unavailable" in report.fallback_reason
    assert pool and all(p.origin == PlaybookOrigin.CONSTRUCTIVE for p in pool)
    assert all(validator.validate(p, context.available).valid for p in pool)


@pytest.mark.anyio
async def test_transient_failures_are_skipped(context, coverage, repairer):
    llm = MockLLMClient([LLMTimeoutError("slow"), GOOD, GOOD, GOOD])
    _, report = await _generator(llm, coverage, repairer).generate(context, [], random.Random(3))
    assert report.llm_calls == 4 and report.llm_failures == 1 and report.llm_valid == 3


@pytest.mark.anyio
async def test_offline_mock_generation_produces_valid_candidates(context, coverage, repairer, validator):
    pool, report = await _generator(MockLLMClient(), coverage, repairer, pool_size=6).generate(context, [], random.Random(4))
    assert report.llm_valid >= 1
    assert all(validator.validate(p, context.available).valid for p in pool)


def test_constructive_generator_is_seeded_safe_and_valid(context, coverage, repairer, validator, vocabulary):
    builder = ConstructiveGenerator(coverage, repairer)
    first = builder.build(context, random.Random(11))
    assert first == builder.build(context, random.Random(11))
    for seed in range(20):
        playbook = builder.build(context, random.Random(seed))
        assert playbook is not None and validator.validate(playbook, context.available).valid
        assert not any(vocabulary.get(n.action).destructive for n in playbook.nodes)


# --- adaptive K ------------------------------------------------------------------------------


@pytest.mark.anyio
async def test_generator_proposes_one_closing_action_per_gap(context, coverage, repairer):
    constructive = ConstructiveGenerator(coverage, repairer)
    gaps = coverage.all_gaps(INTENT)
    llm = MockLLMClient()
    proposals = await CandidateGenerator(llm, coverage, repairer, constructive).propose_gap_actions(context, gaps)
    assert set(proposals) == set(gaps) and len(llm.calls) == 1  # one structured call for all gaps
    assert all(action in {a.id for a in coverage.closers(gap, INTENT)} for gap, action in proposals.items())
    assert DATA_START in llm.calls[0]["user_prompt"]
    # answers outside a gap's closers (or no model at all) are ignored
    first = gaps[0]
    wrong = MockLLMClient([{f"{first.kind}:{first.key}": "isolate_host"}])
    assert await CandidateGenerator(wrong, coverage, repairer, constructive).propose_gap_actions(context, gaps) == {}
    assert await CandidateGenerator(None, coverage, repairer, constructive).propose_gap_actions(context, gaps) == {}
    failing = MockLLMClient([LLMTimeoutError("t")])
    assert await CandidateGenerator(failing, coverage, repairer, constructive).propose_gap_actions(context, gaps) == {}


def test_jaccard_distance_between_coverage_vectors():
    assert jaccard((1, 1, 0, 0), (1, 1, 1, 0)) == pytest.approx(1 / 3)
    assert jaccard((1, 0), (0, 1)) == 1.0 and jaccard((1, 1), (1, 1)) == 0.0
    assert jaccard((0, 0, 0), (0, 0, 0)) == 0.0  # neither covers anything: identical


def test_agglomerative_clustering_merges_near_duplicates():
    # Jaccard distances: 0-1 = 0.25 (3 of 4 shared), {0,1}-2 = 1.0 (nothing shared)
    vectors = [(1, 1, 1, 0, 0, 0, 0, 0), (1, 1, 1, 1, 0, 0, 0, 0), (0, 0, 0, 0, 1, 1, 1, 1)]
    assert sorted(map(sorted, agglomerate(vectors, 0.3))) == [[0, 1], [2]]  # the paper's tau = 0.3
    assert len(agglomerate(vectors, 0.2)) == 3 and len(agglomerate(vectors, 1.0)) == 1


def test_tight_pool_gives_small_k_and_dispersed_pool_larger_k(context, coverage, repairer, evaluator):
    seed_playbook = Playbook.model_validate(next(s for s in seeds() if s["intent_label"] == INTENT))
    seeder = AdaptiveSeeder(evaluator, k_min=1, k_max=8, threshold=0.15)
    tight = seeder.select([seed_playbook] * 5, INTENT, context.available, random.Random(0))
    assert tight.k == 1 and tight.duplicates == 4

    builder = ConstructiveGenerator(coverage, repairer)
    dispersed_pool = [builder.build(context, random.Random(s), goal_rate=0.3 + 0.07 * (s % 10)) for s in range(16)]
    dispersed = seeder.select([p for p in dispersed_pool if p], INTENT, context.available, random.Random(0))
    assert dispersed.k > tight.k and dispersed.clusters == dispersed.k
    assert [s.fitness for s in dispersed.seeds] == sorted((s.fitness for s in dispersed.seeds), reverse=True)


def test_k_is_clamped_and_inadmissible_candidates_are_excluded(context, coverage, repairer, evaluator):
    builder = ConstructiveGenerator(coverage, repairer)
    pool = [builder.build(context, random.Random(s), goal_rate=0.3 + 0.07 * (s % 10)) for s in range(16)]
    pool = [p for p in pool if p]
    capped = AdaptiveSeeder(evaluator, k_min=1, k_max=2, threshold=0.0).select(pool, INTENT, context.available, random.Random(0))
    assert capped.k == 2
    unsafe = Playbook.model_validate({"name": "u", "intent_label": INTENT, "origin": "llm",
                                      "nodes": [{"node_id": "n1", "action": "isolate_host"}], "edges": []})  # fmt: skip
    one = AdaptiveSeeder(evaluator, k_min=3, k_max=8).select([unsafe, pool[0]], INTENT, context.available, random.Random(0))
    assert one.admissible == 1 and one.k == 1  # k_min cannot be met with one admissible candidate


def test_fixed_k_without_coverage_aware_initialisation(context, coverage, repairer, evaluator):
    builder = ConstructiveGenerator(coverage, repairer)
    pool = [p for p in (builder.build(context, random.Random(s)) for s in range(10)) if p]
    selection = AdaptiveSeeder(evaluator, fixed_k=3).select(pool, INTENT, context.available, random.Random(0), adaptive=False)
    assert selection.k == 3 and selection.adaptive is False and selection.clusters == 0


# --- context & model roles --------------------------------------------------------------------


@pytest.mark.anyio
async def test_synthesis_context_from_an_analysed_incident(taxonomy, attack_mapper):
    incident = await IntentExtractor(MockLLMClient([good_output()]), taxonomy, attack_mapper).extract(make_incident())
    context = SynthesisContext.from_incident(incident)
    assert context.intent_label == INTENT
    assert {"host_id", "user_id", "ip_address", "process_name"} <= context.available
    assert EvidenceCategory.PROCESS_TELEMETRY in context.observed_categories
    # "full process tree" is missing, but process telemetry is already observed: at category level
    # (the level routing and EA work at) nothing critical is outstanding.
    assert context.missing_critical == ()
    assert context.missing_supporting == (EvidenceCategory.DNS,)
    assert "T1059.001" in context.techniques and not context.synthetic_basis
    json.dumps(context.techniques)


def test_model_family_detects_same_family_judges():
    assert model_family("llama3.2:3b") == model_family("llama3.1:8b") == "llama"
    assert model_family("qwen2.5:3b") != model_family("llama3.2:3b")
