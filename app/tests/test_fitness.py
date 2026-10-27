"""Fitness: IC, EA, the OS judges and f = cube root of IC * EA * OS (milestone M4)."""

from __future__ import annotations

import json

import pytest

from app.core.config import PROJECT_ROOT
from app.core.errors import LLMTimeoutError
from app.fitness.coverage import CoverageModel, Gap, hamming
from app.fitness.fitness import FitnessEvaluator, geometric_mean
from app.fitness.judge import HeuristicSoundness, LLMSoundnessJudge, render_playbook
from app.intent.mock_llm_client import MockLLMClient
from app.playbook.validator import PlaybookValidator
from app.playbook.vocabulary import Vocabulary
from app.schemas.enums import EvidenceCategory
from app.schemas.playbook import Playbook
from app.tests.test_playbook_dag import VALID, pb, seeds

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
def heuristic(coverage, validator) -> HeuristicSoundness:
    return HeuristicSoundness(coverage, validator)


SMALL = pb("n1:extract_entities n2:query_process_events n3:analyze_process_tree n4:generate_incident_report",
           "n1>n2:host_id n2>n3:process_list n3>n4:findings", intent=INTENT)  # fmt: skip


def judgement(**scores) -> dict:
    base = {"tool_appropriateness": 4, "sequencing": 4, "coherence": 4, "destructive_safety": 5, "rationale": "ok"}
    base.update(scores)
    return base


# --- IC / EA ------------------------------------------------------------------------------------


def test_ic_and_ea_match_hand_computed_values(coverage):
    profile = coverage.profile(SMALL, INTENT)
    # goals: triage (extract_entities), process_lineage (T1059 via process events / analyze tree) and
    # persistence_check (T1053.005 via query_process_events) of 6 goals
    assert profile.goals_covered == (True, True, False, False, True, False)
    assert profile.ic == pytest.approx(3 / 6)
    # relevant: process_telemetry, command_line, script_execution, file_activity, network_traffic, security_alert
    assert profile.ea == pytest.approx(3 / 6)
    assert {g.key for g in profile.gaps()} == {
        "payload_reputation", "network_follow_up", "compromise_verification", "script_execution", "file_activity",
        "network_traffic"}  # fmt: skip
    assert Gap("category", "file_activity") in profile.gaps()


def test_destructive_actions_never_add_coverage(coverage):
    playbook = pb(*VALID, intent="ransomware_investigation")
    without = pb("n1:extract_entities n2:query_process_events n3:verify_host_compromise n5:generate_incident_report",
                 "n1>n2:host_id n1>n3:host_id n2>n3:findings n3>n5:findings", intent="ransomware_investigation")  # fmt: skip
    assert coverage.profile(playbook, "ransomware_investigation").vector == coverage.profile(without, "ransomware_investigation").vector


def test_unknown_intent_has_full_evidence_alignment(coverage):
    assert coverage.relevant_categories("unknown") == ()
    assert coverage.profile(SMALL, "unknown").ea == 1.0


@pytest.mark.parametrize("seed", seeds(), ids=lambda s: s["intent_label"])
def test_seed_playbooks_have_full_investigative_coverage(seed, coverage):
    playbook = Playbook.model_validate(seed)
    assert coverage.profile(playbook, playbook.intent_label).ic == 1.0


def test_coverage_closers_and_hamming(coverage):
    closers = {a.id for a in coverage.closers(Gap("category", EvidenceCategory.FILE_ACTIVITY.value), INTENT)}
    assert "review_file_integrity" in closers and "quarantine_file" not in closers
    assert hamming((1, 0, 1, 1), (1, 1, 1, 0)) == 0.5 and hamming((), ()) == 0.0


# --- Eq. 6 -----------------------------------------------------------------------------------


def test_geometric_mean_collapses_on_any_zero():
    assert geometric_mean(1, 1, 1) == pytest.approx(1)
    assert geometric_mean(0.5, 0.6, 0.8) == pytest.approx(0.24 ** (1 / 3))
    assert geometric_mean(1, 1, 0) == 0 and geometric_mean(0, 1, 1) == 0
    # balance: a balanced playbook beats a lopsided one with the same arithmetic mean
    assert geometric_mean(0.6, 0.6, 0.6) > geometric_mean(1.0, 0.6, 0.2)


def test_hard_filter_rejects_invalid_and_unverified_destructive(coverage, validator, heuristic):
    evaluator = FitnessEvaluator(coverage, validator, heuristic)
    unverified = pb("n1:extract_entities n2:isolate_host n3:generate_incident_report", "n1>n2:host_id n2>n3:findings")
    assert evaluator.evaluate(unverified, INTENT) is None
    assert evaluator.evaluate(pb("n1:query_process_events"), INTENT) is None
    assert evaluator.evaluations == 0
    scored = evaluator.evaluate(SMALL, INTENT)
    assert scored is not None and scored.scores.os_source == "heuristic"
    assert scored.fitness == pytest.approx(geometric_mean(0.5, 0.5, scored.scores.os))


# --- heuristic OS ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", seeds(), ids=lambda s: s["intent_label"])
def test_curated_seed_playbooks_pass_every_rule_check(seed, heuristic):
    playbook = Playbook.model_validate(seed)
    result = heuristic.score(playbook, playbook.intent_label)
    assert result.source == "heuristic" and result.score == 1.0
    assert set(heuristic.criteria(playbook, playbook.intent_label)) == {
        "tool_fit", "evidence_support", "containment_investigated", "no_redundancy"}


def test_heuristic_soundness_is_the_fraction_of_rule_checks_passed(heuristic):
    # three identical triage steps: 2 of 3 are redundant -> 4 tool_fit + 1 tool check pass, 2 of 3 redundancy fail
    redundant = pb("n1:triage_alert n2:triage_alert n3:triage_alert n4:generate_incident_report",
                   "n1>n4:findings n2>n4:findings n3>n4:findings", intent=INTENT)  # fmt: skip
    assert heuristic.criteria(redundant, INTENT)["no_redundancy"] == pytest.approx(1 / 3)
    checks = heuristic.checks(redundant, INTENT)
    passed = sum(sum(v) for v in checks.values())
    assert heuristic.score(redundant, INTENT).score == pytest.approx(passed / sum(len(v) for v in checks.values()))
    # the same tool operation on the same target type is repeated work, even under another action id
    repeated = pb("n1:extract_entities n2:query_network_connections n3:query_ip_connections n4:query_network_connections "
                  "n5:generate_incident_report",
                  "n1>n2:host_id n1>n3:ip_address n1>n4:host_id n2>n5:findings n3>n5:findings n4>n5:findings",
                  intent=INTENT)  # fmt: skip
    assert heuristic.checks(repeated, INTENT)["no_redundancy"].count(False) == 1


def test_containment_needs_an_investigative_step_on_the_same_entity(heuristic):
    # isolation guarded only by the (mandatory) verification: no investigation of the host
    bare = pb("n1:extract_entities n2:verify_host_compromise n3:isolate_host n4:generate_incident_report",
              "n1>n2:host_id n1>n2:findings n1>n3:host_id n2>n3:verification_result n3>n4:findings",
              intent=INTENT)  # fmt: skip
    assert heuristic.checks(bare, INTENT)["containment_investigated"] == [False]
    investigated = pb(*VALID, intent=INTENT)
    assert heuristic.checks(investigated, INTENT)["containment_investigated"] == [True]
    assert heuristic.score(bare, INTENT).score < heuristic.score(investigated, INTENT).score


# --- LLM judge ------------------------------------------------------------------------------


@pytest.mark.anyio
async def test_llm_judge_averages_samples_and_caches(coverage, heuristic):
    llm = MockLLMClient([judgement(), judgement(coherence=2), judgement(sequencing=5)])
    judge = LLMSoundnessJudge(llm, coverage, heuristic, samples=3, temperature=0.7)
    result = await judge.score(SMALL, INTENT)
    # rubric sums 17, 15 and 18 out of 4..20, normalised as (sum - 4) / 16
    expected = ((17 - 4) / 16 + (15 - 4) / 16 + (18 - 4) / 16) / 3
    assert result.source == "llm" and result.samples == 3 and result.score == pytest.approx(expected)
    assert [c["temperature"] for c in llm.calls] == [0.7, 0.7, 0.7]
    assert "Analyse the process tree" in llm.calls[0]["user_prompt"]
    assert llm.calls[0]["json_schema"]["title"] == "SoundnessJudgement"
    again = await judge.score(SMALL.model_copy(update={"name": "renamed"}), INTENT)
    assert again == result and len(llm.calls) == 3  # same structure: no new calls


@pytest.mark.anyio
async def test_llm_judge_skips_bad_samples_and_falls_back(coverage, heuristic):
    llm = MockLLMClient(["not json", judgement(tool_appropriateness=9), judgement()])
    result = await LLMSoundnessJudge(llm, coverage, heuristic).score(SMALL, INTENT)
    assert result.source == "llm" and result.samples == 1
    failing = MockLLMClient([LLMTimeoutError("t"), "garbage", json.dumps({"coherence": 3})])
    fallback = await LLMSoundnessJudge(failing, coverage, heuristic).score(SMALL, INTENT)
    assert fallback.source == "heuristic_fallback"
    assert fallback.score == pytest.approx(heuristic.score(SMALL, INTENT).score)


@pytest.mark.anyio
async def test_only_the_top_k_candidates_are_judged(coverage, validator, heuristic):
    # a perfect judgement keeps the judged candidates ahead, so no extra leader judging is triggered
    llm = MockLLMClient([judgement(tool_appropriateness=5, sequencing=5, coherence=5)] * 30)
    judge = LLMSoundnessJudge(llm, coverage, heuristic, samples=1)
    evaluator = FitnessEvaluator(coverage, validator, heuristic, judge, judge_top_k=2)
    candidates = [evaluator.evaluate(Playbook.model_validate(s), s["intent_label"]) for s in seeds()[:4]]
    candidates = [evaluator.evaluate(c.playbook, INTENT) for c in candidates]
    assert all(c.scores.os_source == "estimated" for c in candidates)
    judged = await evaluator.judge_top(candidates, INTENT)
    sources = [c.scores.os_source for c in judged]
    assert sources.count("llm") == 2 and sources.count("estimated") == 2 and len(llm.calls) == 2
    best_two = sorted(candidates, key=lambda c: c.fitness, reverse=True)[:2]
    assert {c.key for c in judged if c.scores.os_source == "llm"} == {c.key for c in best_two}
    # A judged structure is reused when re-evaluated later.
    assert evaluator.evaluate(best_two[0].playbook, INTENT).scores.os_source == "llm"


def test_rendering_marks_destructive_and_verification_steps(coverage):
    text = render_playbook(pb(*VALID), coverage)
    assert "[DESTRUCTIVE]" in text and "[VERIFICATION]" in text
    assert text.splitlines()[0].startswith("1. [triage] Extract entities")


@pytest.mark.anyio
async def test_the_leader_is_always_judged_never_estimated(coverage, validator, heuristic):
    """Regression (live run): judging the top-k lowered their OS and an unjudged candidate became the best."""
    harsh = {"tool_appropriateness": 1, "sequencing": 1, "coherence": 1, "destructive_safety": 1, "rationale": "poor"}
    llm = MockLLMClient([harsh] * 40)
    judge = LLMSoundnessJudge(llm, coverage, heuristic, samples=1)
    evaluator = FitnessEvaluator(coverage, validator, heuristic, judge, judge_top_k=1)
    candidates = [evaluator.evaluate(Playbook.model_validate(s), INTENT) for s in seeds()[:5]]
    judged = await evaluator.judge_top(candidates, INTENT)
    leader = max(judged, key=lambda c: c.fitness)
    assert leader.scores.os_source == "llm"
    assert sum(1 for c in judged if c.scores.os_source == "llm") >= 2  # more than top-k were needed
