"""Playbook quality, cost and scorer ablation on the 96-scenario benchmark (paper 968 Section 6.4, Table 4, Fig. 2, Fig. 4b).

Every system returns one playbook per scenario, and every returned playbook is scored the same way: IC and EA
from the intent ontology and OS from the LLM judge (three judgments averaged), f = cube root of IC*EA*OS (Eq. 10).
"Every returned playbook is scored with the same IC, EA, and judge-based OS" (Section 6.4).

Systems (Section 6.4):

* ``static_soar``     -- the closest *hand-authored* playbook (CADE's 15 seed playbooks), not adapted;
* ``retrieval_only``  -- the top match from the scenario's benchmark library (withheld playbooks excluded);
* ``llm_only``        -- one unoptimised draft from the generator (one call, no pool, no evolution);
* ``generation_only`` -- always CADE synthesis (pool B, adaptive K, evolution, judge);
* ``generation_only_osh`` -- the scorer ablation: CADE synthesis with OS_H for every candidate, no judge;
* ``cade_full``       -- the router's actual decision per ordering (from a routing run's ``results.json``):
  reuse -> the top match given the evidence in hand, synthesize -> CADE synthesis for that evidence state.

Choices where the paper is silent are marked *(choice)*:

* The benchmark's intents are ATT&CK tactics; IC and EA use CADE's intent ontology, so each tactic is mapped to
  the CADE intent that investigates it (``TACTIC_INTENT``) *(choice)*.
* An incident is the scenario's templated one-line summary plus its evidence profile (the fixed ground-truth
  analysis of the routing benchmark); the generator sees the evidence in hand and what is missing.
* A system that returns no admissible playbook scores IC = EA = OS = 0 and is counted as a failure.
* Only ``cade_full`` depends on the ordering. The other systems' playbooks are produced once per scenario and
  cached, so their means over 10 orderings equal their means over the 96 scenarios *(choice: cached synthesis)*.

Results are cached per (system, scenario, evidence state) as JSON files so a run that takes hours can resume.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from app.core.config import PROJECT_ROOT, Settings
from app.evaluation.paper968 import PaperEnv, PaperScenario, display, ordered
from app.evolution.engine import CADEEngine, EvolutionConfig
from app.evolution.operators import Operators
from app.fitness.coverage import CoverageModel
from app.fitness.fitness import FitnessEvaluator, geometric_mean
from app.fitness.judge import HeuristicSoundness, LLMSoundnessJudge
from app.intent.llm_client import LLMClient
from app.library.library import PlaybookLibrary
from app.playbook.repair import PlaybookRepairer
from app.playbook.validator import PlaybookValidator
from app.playbook.vocabulary import Vocabulary
from app.routing.router import RetrievalQuery
from app.schemas.enums import EvidenceCategory as C
from app.schemas.playbook import Playbook
from app.schemas.routing import RoutingAction
from app.schemas.taxonomy import IntentTaxonomy
from app.synthesis.constructive import ConstructiveGenerator
from app.synthesis.context import SynthesisContext
from app.synthesis.generator import CandidateGenerator
from app.synthesis.pipeline import run_synthesis
from app.synthesis.seeding import AdaptiveSeeder

SYSTEMS = ("static_soar", "llm_only", "retrieval_only", "generation_only", "generation_only_osh", "cade_full")
LABELS = {"static_soar": "Static SOAR", "llm_only": "LLM-only", "retrieval_only": "Retrieval-only",
          "generation_only": "Generation-only", "cade_full": "CADE (full)",
          "generation_only_osh": "Generation-only, OSH only (scorer ablation)"}  # fmt: skip
TACTIC_INTENT = {
    "execution": "suspicious_process_execution", "persistence": "command_and_control_persistence",
    "privilege_escalation": "privilege_escalation", "defense_evasion": "defense_evasion",
    "credential_access": "credential_compromise", "discovery": "discovery",
    "lateral_movement": "lateral_movement", "impact": "system_impact",
}  # fmt: skip
INCIDENT_ARTIFACTS = frozenset({"host_id", "user_id"})  # Windows host recordings name a host and a user
# Paper Table 4, for side-by-side reporting only.
PAPER_TABLE_4 = {
    "static_soar": (0.61, 0.58, 0.74, 0.64, 6.1), "llm_only": (0.72, 0.69, 0.66, 0.69, 16.9),
    "retrieval_only": (0.70, 0.71, 0.77, 0.72, 6.0), "generation_only": (0.83, 0.80, 0.80, 0.81, 238.0),
    "cade_full": (0.89, 0.86, 0.83, 0.86, 88.0), "generation_only_osh": (0.83, 0.80, 0.70, 0.77, 84.0),
}  # fmt: skip


@dataclass
class QualityKit:
    """The synthesis and scoring components for one run, built from the settings and two LLM clients."""

    settings: Settings
    vocabulary: Vocabulary
    coverage: CoverageModel
    validator: PlaybookValidator
    repairer: PlaybookRepairer
    heuristic: HeuristicSoundness
    generator: CandidateGenerator
    single_draft: CandidateGenerator  # LLM-only: one call, one temperature, no examples
    judged: FitnessEvaluator  # CADE's evaluator with the LLM judge (Eq. 12)
    unjudged: FitnessEvaluator  # scorer ablation: OS_H for every candidate
    scorer: Optional[LLMSoundnessJudge]  # scores every returned playbook (None: OS_H, mock runs only)
    generator_model: Optional[str]
    judge_model: Optional[str]

    def seeder(self, evaluator: FitnessEvaluator) -> AdaptiveSeeder:
        s = self.settings
        return AdaptiveSeeder(evaluator, k_min=s.seed_k_min, k_max=s.seed_k_max, threshold=s.seed_cluster_threshold)

    def engine_factory(self, evaluator: FitnessEvaluator):
        config = EvolutionConfig.from_settings(self.settings)
        return lambda: CADEEngine(evaluator, Operators(self.coverage, self.repairer), config)


def build_kit(settings: Settings, generator_llm: Optional[LLMClient], judge_llm: Optional[LLMClient]) -> QualityKit:
    taxonomy = IntentTaxonomy.load(settings.resolve_path(settings.intent_taxonomy_path))
    vocabulary = Vocabulary.load(settings.resolve_path(settings.action_vocabulary_path),
                                 settings.resolve_path(settings.intent_goals_path), taxonomy)  # fmt: skip
    validator = PlaybookValidator(vocabulary, node_cap=settings.node_cap)
    repairer = PlaybookRepairer(validator)
    coverage = CoverageModel(vocabulary, taxonomy)
    heuristic = HeuristicSoundness(coverage, validator)

    def judge() -> Optional[LLMSoundnessJudge]:
        return (LLMSoundnessJudge(judge_llm, coverage, heuristic, samples=settings.os_judge_samples,
                                  temperature=settings.os_judge_temperature) if judge_llm is not None else None)  # fmt: skip

    constructive = ConstructiveGenerator(coverage, repairer)
    generator = CandidateGenerator(generator_llm, coverage, repairer, constructive, pool_size=settings.generation_pool_size,
                                   temperatures=tuple(settings.generation_temperatures))  # fmt: skip
    single = CandidateGenerator(generator_llm, coverage, repairer, constructive, pool_size=1,
                                temperatures=tuple(settings.generation_temperatures[:1]))  # fmt: skip
    return QualityKit(
        settings=settings, vocabulary=vocabulary, coverage=coverage, validator=validator, repairer=repairer,
        heuristic=heuristic, generator=generator, single_draft=single,
        judged=FitnessEvaluator(coverage, validator, heuristic, judge(), judge_top_k=settings.os_judge_top_k),
        unjudged=FitnessEvaluator(coverage, validator, heuristic, None),
        scorer=judge(), generator_model=getattr(generator_llm, "model", None), judge_model=getattr(judge_llm, "model", None),
    )  # fmt: skip


# --- scenarios as incidents -----------------------------------------------------------------------------


def intent_of(scenario: PaperScenario) -> str:
    return TACTIC_INTENT[scenario.tactic]


def summary_line(scenario: PaperScenario) -> str:
    return (f"Windows host telemetry records {scenario.technique_name} ({scenario.technique}), "
            f"an ATT&CK {display(scenario.tactic)} technique.")  # fmt: skip


def context_for(scenario: PaperScenario, in_hand: frozenset[C], incident_id: str) -> SynthesisContext:
    missing = scenario.recording.categories - in_hand
    return SynthesisContext(
        incident_id=incident_id, intent_label=intent_of(scenario),
        intent_description=f"{display(scenario.tactic)}: {scenario.technique_name} ({scenario.technique})",
        summary=summary_line(scenario), available=INCIDENT_ARTIFACTS, observed_categories=tuple(ordered(in_hand)),
        missing_critical=tuple(ordered(missing & scenario.critical)),
        missing_supporting=tuple(ordered(missing - scenario.critical)), techniques=(scenario.technique,),
    )  # fmt: skip


def state_key(in_hand: frozenset[C]) -> str:
    return "+".join(c.value for c in ordered(in_hand)) or "none"


# --- records and cache ----------------------------------------------------------------------------------


@dataclass
class Record:
    system: str
    scenario: str
    state: str
    playbook: Optional[dict[str, Any]]
    ic: float
    ea: float
    os: float
    os_source: str
    fitness: float
    latency_s: float
    failed: bool = False
    info: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


class Cache:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, system: str, scenario: str, state: str) -> Path:
        safe = scenario.replace("/", "__").replace(":", "_")
        return self.root / system / f"{safe}__{state}.json"

    def get(self, system: str, scenario: str, state: str) -> Optional[Record]:
        path = self._path(system, scenario, state)
        if not path.exists():
            return None
        return Record(**json.loads(path.read_text(encoding="utf-8")))

    def put(self, record: Record) -> None:
        path = self._path(record.system, record.scenario, record.state)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record.as_dict(), indent=1, default=str), encoding="utf-8")
        tmp.replace(path)


# --- the evaluator --------------------------------------------------------------------------------------


@dataclass
class QualityRun:
    kit: QualityKit
    env: PaperEnv  # the benchmark library, withheld playbooks and retrieval
    seeds: PlaybookLibrary  # CADE's hand-authored seed playbooks (static SOAR)
    cache: Cache
    synthesis_seed: int = 0  # synthesis is seeded per scenario from this, so a rerun reproduces the cache

    async def score(self, system: str, scenario: PaperScenario, state: str, playbook: Optional[Playbook],
                    latency_s: float, info: dict[str, Any]) -> Record:  # fmt: skip
        intent = intent_of(scenario)
        if playbook is None or not self.kit.validator.validate(playbook, INCIDENT_ARTIFACTS).valid:
            return Record(system, scenario.name, state, playbook.model_dump(mode="json") if playbook else None,
                          0.0, 0.0, 0.0, "none", 0.0, latency_s, failed=True, info=info)  # fmt: skip
        profile = self.kit.coverage.profile(playbook, intent)
        if self.kit.scorer is not None:
            result = await self.kit.scorer.score(playbook, intent)
            os_score, source = result.score, result.source
        else:
            os_score, source = self.kit.heuristic.score(playbook, intent).score, "heuristic"
        return Record(system, scenario.name, state, playbook.model_dump(mode="json"), profile.ic, profile.ea, os_score,
                      source, geometric_mean(profile.ic, profile.ea, os_score), latency_s, info=info)  # fmt: skip

    def _query(self, scenario: PaperScenario, in_hand: frozenset[C], intent: str) -> RetrievalQuery:
        return RetrievalQuery(intent_label=intent, intent_description=summary_line(scenario),
                              observed=tuple(ordered(in_hand)), missing=tuple(ordered(scenario.critical - in_hand)),
                              techniques=(scenario.technique,))  # fmt: skip

    def top_match(self, scenario: PaperScenario, in_hand: frozenset[C]) -> tuple[Optional[Playbook], float]:
        started = time.perf_counter()
        decision = self.env.decide(scenario, in_hand, frozenset(), None, 2)
        top = decision.top_candidate
        entry = self.env.library.get(top.entry_id) if top else None
        return (entry.playbook if entry else None), time.perf_counter() - started

    def examples(self, scenario: PaperScenario, in_hand: frozenset[C]) -> list[Playbook]:
        _, source = self.env.router(scenario)
        candidates = source.retrieve(self.env.query(scenario, in_hand), self.kit.settings.routing_top_n)
        return [c.playbook for c in candidates[: self.kit.settings.generation_examples]]

    async def produce(self, system: str, scenario: PaperScenario, in_hand: frozenset[C]) -> Record:
        state = state_key(in_hand)
        cached = self.cache.get(system, scenario.name, state)
        if cached is not None:
            return cached
        record = await self._produce(system, scenario, in_hand, state)
        self.cache.put(record)
        return record

    async def _produce(self, system: str, scenario: PaperScenario, in_hand: frozenset[C], state: str) -> Record:
        kit, intent = self.kit, intent_of(scenario)
        incident_id = f"Q-{scenario.name}"
        if system == "static_soar":
            started = time.perf_counter()
            hits = self.seeds.retrieve(self._query(scenario, in_hand, intent), 1)
            playbook = hits[0].playbook if hits else None
            return await self.score(system, scenario, state, playbook, time.perf_counter() - started,
                                    {"matched": playbook.name if playbook else None})  # fmt: skip
        if system == "retrieval_only":
            playbook, latency = self.top_match(scenario, in_hand)
            return await self.score(system, scenario, state, playbook, latency,
                                    {"matched": playbook.name if playbook else None})  # fmt: skip
        if system == "llm_only":
            import random

            started = time.perf_counter()
            pool, report = await kit.single_draft.generate(context_for(scenario, in_hand, incident_id), [],
                                                           random.Random(self.seed_for(scenario)))  # fmt: skip
            llm_drafts = [p for p in pool if p.origin.value == "llm"]
            playbook = llm_drafts[0] if llm_drafts else None
            return await self.score(system, scenario, state, playbook, time.perf_counter() - started,
                                    {"llm_valid": report.llm_valid, "unparseable": report.unparseable,
                                     "discarded": report.discarded, "fallback": report.fallback_reason})  # fmt: skip
        if system in ("generation_only", "generation_only_osh"):
            evaluator = kit.judged if system == "generation_only" else kit.unjudged
            started = time.perf_counter()
            try:
                run = await run_synthesis(generator=kit.generator, seeder=kit.seeder(evaluator),
                                          new_engine=kit.engine_factory(evaluator), coverage=kit.coverage,
                                          settings=kit.settings, context=context_for(scenario, in_hand, incident_id),
                                          examples=self.examples(scenario, in_hand), seed=self.seed_for(scenario),
                                          judge_model=kit.judge_model if system == "generation_only" else None)  # fmt: skip
            except Exception as exc:  # noqa: BLE001 - a failed synthesis is a scored failure, not a crash
                return await self.score(system, scenario, state, None, time.perf_counter() - started,
                                        {"error": f"{type(exc).__name__}: {exc}"[:300]})  # fmt: skip
            return await self.score(system, scenario, state, run.best.playbook, time.perf_counter() - started, run.info)
        raise ValueError(f"unknown system {system!r}")

    def seed_for(self, scenario: PaperScenario) -> int:
        return (self.synthesis_seed * 1_000_003 + sum(map(ord, scenario.name))) % (2**31)

    async def cade_full(self, scenario: PaperScenario, terminal: str, requested: bool) -> Record:
        """The full framework's playbook for one routed decision: reuse -> top match, synthesize -> CADE synthesis."""
        in_hand = scenario.available | (scenario.obtainable if requested else frozenset())
        if terminal == RoutingAction.REUSE.value:
            base = await self.produce("retrieval_only", scenario, in_hand)
        else:
            base = await self.produce("generation_only", scenario, in_hand)
        return Record(**{**base.as_dict(), "system": "cade_full", "info": {"terminal": terminal, "requested": requested}})


# --- aggregation ----------------------------------------------------------------------------------------


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else float("nan")


def table4(records: dict[str, list[Record]], cade_by_ordering: dict[int, list[Record]]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for system, items in records.items():
        if not items:
            continue
        rows[system] = {"ic": mean(r.ic for r in items), "ea": mean(r.ea for r in items), "os": mean(r.os for r in items),
                        "fitness": mean(r.fitness for r in items), "latency_s": mean(r.latency_s for r in items),
                        "failures": sum(r.failed for r in items), "n": len(items),
                        "os_sources": dict(sorted(_count(r.os_source for r in items).items()))}  # fmt: skip
    if cade_by_ordering:
        per = [{"ic": mean(r.ic for r in rs), "ea": mean(r.ea for r in rs), "os": mean(r.os for r in rs),
                "fitness": mean(r.fitness for r in rs), "latency_s": mean(r.latency_s for r in rs)}
               for rs in cade_by_ordering.values()]  # fmt: skip
        flat = [r for rs in cade_by_ordering.values() for r in rs]
        rows["cade_full"] = {**{k: mean(p[k] for p in per) for k in per[0]},
                             **{f"{k}_std": (statistics.pstdev([p[k] for p in per]) if len(per) > 1 else 0.0) for k in per[0]},
                             "failures": sum(r.failed for r in flat), "n": len(flat), "orderings": len(per),
                             "synthesis_share": mean(r.info.get("terminal") == RoutingAction.SYNTHESIZE.value for r in flat)}  # fmt: skip
    return rows


def _count(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for v in values:
        counts[v] += 1
    return counts


def generation_curves(records: list[Record]) -> dict[str, list[float]]:
    """Fig. 2: per generation, mean over synthesis runs of the best fitness, population-mean fitness and best IC.

    Runs that stopped early carry their last value forward, so every generation averages all runs.
    """
    histories = [r.info.get("evolution", {}).get("history") or [] for r in records if not r.failed]
    histories = [h for h in histories if h]
    if not histories:
        return {}
    length = max(len(h) for h in histories)

    def curve(key: str) -> list[float]:
        return [mean((h[min(g, len(h) - 1)].get(key) or 0.0) for h in histories) for g in range(length)]

    return {"best_fitness": curve("best_fitness"), "mean_fitness": curve("mean_fitness"), "best_ic": curve("best_ic"),
            "generations_mean": mean(r.info["evolution"]["generations"] for r in records
                                     if not r.failed and r.info.get("evolution"))}  # fmt: skip


def latency_breakdown(records: list[Record]) -> dict[str, float]:
    """Fig. 4b / Section 7.4: mean seconds per synthesis stage, judge calls and judge share of synthesis time."""
    timings = [r.info["timing"] for r in records if not r.failed and r.info.get("timing")]
    if not timings:
        return {}
    out = {k: mean(t[k] for t in timings) for k in ("generation_s", "seeding_s", "gap_proposals_s", "evolution_s",
                                                    "total_s", "judge_calls", "judge_s")}  # fmt: skip
    out["judge_share"] = out["judge_s"] / out["total_s"] if out["total_s"] else 0.0
    out["runs"] = len(timings)
    return out


def seed_library_path(settings: Settings) -> Path:
    return settings.resolve_path(settings.library_seed_path) if settings.library_seed_path else (
        PROJECT_ROOT / "app" / "data" / "seed_playbooks.json")
