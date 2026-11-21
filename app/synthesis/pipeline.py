"""One synthesis run (paper Section 5): candidate pool -> adaptive-K seeds -> CADE evolution -> best playbook.

Shared by the API (:class:`app.services.response_service.ResponseService`) and the evaluation harness
(:mod:`app.evaluation.quality`), so an evaluated synthesis is exactly the one the service runs. Stage
durations are recorded for the latency breakdown (paper Fig. 4b); judge calls and judge time come from the
evaluator's LLM judge when one is configured.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Optional

from app.core.config import Settings
from app.core.errors import SynthesisError
from app.evolution.engine import CADEEngine
from app.fitness.coverage import CoverageModel
from app.fitness.fitness import EvaluatedCandidate
from app.schemas.playbook import Playbook
from app.synthesis.context import SynthesisContext
from app.synthesis.generator import CandidateGenerator
from app.synthesis.prompts import GENERATION_PROMPT_VERSION
from app.synthesis.seeding import AdaptiveSeeder


@dataclass
class SynthesisRun:
    best: EvaluatedCandidate
    info: dict[str, Any]  # what was done: generation, seeding, evolution history, timing (JSON-serialisable)


async def run_synthesis(
    *,
    generator: CandidateGenerator,
    seeder: AdaptiveSeeder,
    new_engine: Callable[[], CADEEngine],
    coverage: CoverageModel,
    settings: Settings,
    context: SynthesisContext,
    examples: list[Playbook],
    seed: int,
    judge_model: Optional[str] = None,
) -> SynthesisRun:
    rng = random.Random(seed)
    judge = getattr(new_engine().evaluator, "judge", None)
    judge_calls, judge_seconds = (judge.calls, judge.seconds) if judge else (0, 0.0)
    started = time.perf_counter()

    pool, generation = await generator.generate(context, examples, rng)
    generated = time.perf_counter()
    selection = seeder.select(pool, context.intent_label, context.available, rng, adaptive=settings.coverage_init)
    if not selection.seeds:
        raise SynthesisError("no admissible candidate playbook could be generated", incident_id=context.incident_id)
    seeded = time.perf_counter()
    engine = new_engine()
    if settings.directed_mutation_llm and settings.coverage_variation:
        # gap-directed additions use the generator's proposed action for each gap the seeds leave open
        open_gaps = {gap for seed_candidate in selection.seeds for gap in seed_candidate.profile.gaps()}
        ordered = [gap for gap in coverage.all_gaps(context.intent_label) if gap in open_gaps]
        engine.operators.proposals = await generator.propose_gap_actions(context, ordered)
    proposed = time.perf_counter()
    result = await engine.run(selection.seeds, context.intent_label, context.available, rng,
                              incident_id=context.incident_id)  # fmt: skip
    finished = time.perf_counter()

    info = {
        "prompt_version": GENERATION_PROMPT_VERSION,
        "generator": {"provider": generation.provider, "model": generation.model},
        "judge_model": judge_model,
        "generation": asdict(generation),
        "seeding": {"pool": selection.pool_size, "admissible": selection.admissible,
                    "duplicates": selection.duplicates, "clusters": selection.clusters, "k": selection.k,
                    "adaptive": selection.adaptive},
        "gap_proposals": {f"{gap.kind}:{gap.key}": action for gap, action in engine.operators.proposals.items()},
        "evolution": {"termination": result.termination, "generations": result.generations,
                      "evaluations": result.evaluations, "judge_calls": result.judge_calls,
                      "duration_ms": round(result.duration_ms, 1), "operators": result.operator_counts,
                      "history": [asdict(h) for h in result.history],
                      "config": asdict(engine.config)},
        # seconds per stage (paper Fig. 4b); judge time is part of seeding and evolution
        "timing": {"generation_s": round(generated - started, 3), "seeding_s": round(seeded - generated, 3),
                   "gap_proposals_s": round(proposed - seeded, 3), "evolution_s": round(finished - proposed, 3),
                   "total_s": round(finished - started, 3),
                   "judge_calls": (judge.calls - judge_calls) if judge else 0,
                   "judge_s": round(judge.seconds - judge_seconds, 3) if judge else 0.0},
        "random_seed": seed,
        "examples": [p.name for p in examples],
    }  # fmt: skip
    return SynthesisRun(result.best, info)
