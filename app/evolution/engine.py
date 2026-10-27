"""The CADE evolutionary loop (paper Section 5.3).

Coverage of the intent's goals and relevant evidence steers the search at four points:

1. **initialisation** -- the adaptive-K seeds (see ``app.synthesis.seeding``) are expanded to the
   population size by mutated variants (gap-directed additions when enabled);
2. **variation** -- the add-node mutations close gaps in U(p) (``directed_mutation_probability``
   of add applications; the rest pick random targets), and crossover takes a donor covering items
   of the recipient's U(p);
3. **selection** -- binary tournaments inside coverage-fraction strata;
4. **termination** -- stop when the best judged candidate has IC = EA = 1, or when its fitness
   gained less than ``plateau_epsilon`` over ``plateau_generations`` generations; otherwise at the
   generation budget.

Each generation validates candidates, computes IC, EA and OS_H, judges the top m by provisional
fitness, keeps the best judged candidates as elites and refills the population with offspring of
selected parents (generational replacement). The best judged candidate is returned.

Each ``coverage_*`` switch can be disabled to get the corresponding non-coverage behaviour; with
all four off this is the "coverage in fitness only" variant. ``generational_replacement=False``
switches to (mu + lambda) survival. The node cap is enforced at every variation step.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from app.core.config import Settings
from app.core.errors import SynthesisError
from app.core.logging import get_logger, log_event
from app.evolution.operators import Operators
from app.evolution.selection import (
    elites,
    proportional_survivors,
    select_donor,
    select_parent,
    stratified_survivors,
    unique_by_structure,
)
from app.fitness.fitness import EvaluatedCandidate, FitnessEvaluator

logger = get_logger("evolution")


@dataclass(frozen=True)
class EvolutionConfig:
    population_size: int = 30
    max_generations: int = 20
    elite_count: int = 2
    crossover_probability: float = 0.3
    directed_mutation_probability: float = 0.7
    mutation_retries: int = 3
    generational_replacement: bool = True
    fitness_threshold: float = 0.0
    plateau_generations: int = 3
    plateau_epsilon: float = 0.01
    coverage_init: bool = True
    coverage_variation: bool = True
    coverage_selection: bool = True
    coverage_termination: bool = True

    @classmethod
    def from_settings(cls, settings: Settings) -> "EvolutionConfig":
        return cls(**{name: getattr(settings, name) for name in cls.__dataclass_fields__})


@dataclass
class GenerationStats:
    generation: int
    best_fitness: float
    mean_fitness: float
    best_ic: float
    best_ea: float
    best_os: float
    niches: int
    population: int


@dataclass
class EvolutionResult:
    best: EvaluatedCandidate
    termination: str  # "full_coverage" | "plateau" | "generation_budget"
    generations: int
    history: list[GenerationStats] = field(default_factory=list)
    evaluations: int = 0
    judge_calls: int = 0
    operator_counts: dict[str, int] = field(default_factory=dict)
    duration_ms: float = 0.0


class CADEEngine:
    def __init__(self, evaluator: FitnessEvaluator, operators: Operators, config: EvolutionConfig) -> None:
        self.evaluator = evaluator
        self.operators = operators
        self.config = config

    async def run(
        self,
        seeds: list[EvaluatedCandidate],
        intent: str,
        available: frozenset[str],
        rng: random.Random,
        *,
        incident_id: str | None = None,
    ) -> EvolutionResult:
        if not seeds:
            raise SynthesisError("no admissible candidate playbook to evolve", incident_id=incident_id)
        started = time.perf_counter()
        evaluations_before = self.evaluator.evaluations
        judge_before = self.evaluator.judge.calls if self.evaluator.judge else 0
        config = self.config

        population = await self.evaluator.judge_top(self._initial_population(seeds, intent, available, rng), intent)
        best = self._best_judged(population)
        history = [self._stats(0, population, best)]
        termination = "generation_budget"
        generation = 0

        for generation in range(1, config.max_generations + 1):
            if config.coverage_termination and self._good_enough(best):
                termination, generation = "full_coverage", generation - 1
                break
            kept = elites(population, config.elite_count)
            offspring = self._offspring(population, len(kept), intent, available, rng)
            if config.generational_replacement:
                population = unique_by_structure(kept + offspring)
            else:
                population = self._survivors(population + offspring, rng)
            population = await self.evaluator.judge_top(population, intent)
            leader = self._best_judged(population)
            if leader.fitness > best.fitness:
                best = leader
            history.append(self._stats(generation, population, best))
            if config.coverage_termination:
                if self._good_enough(best):
                    termination = "full_coverage"
                    break
                window = config.plateau_generations
                if generation >= window and best.fitness - history[generation - window].best_fitness < config.plateau_epsilon:
                    termination = "plateau"
                    break

        counts = dict(self.operators.applied)
        counts.update(gap_directed=self.operators.gap_directed, proposal_used=self.operators.proposal_used)
        result = EvolutionResult(
            best=best,
            termination=termination,
            generations=generation,
            history=history,
            evaluations=self.evaluator.evaluations - evaluations_before,
            judge_calls=(self.evaluator.judge.calls if self.evaluator.judge else 0) - judge_before,
            operator_counts=counts,
            duration_ms=(time.perf_counter() - started) * 1000,
        )
        log_event(logger, "evolution finished", incident_id=incident_id, operation="synthesize", stage="evolution",
                  outcome=termination, duration_ms=result.duration_ms, generations=generation,
                  best_fitness=round(best.fitness, 4), evaluations=result.evaluations)  # fmt: skip
        return result

    # --- stages ------------------------------------------------------------------------------

    def _initial_population(self, seeds, intent, available, rng) -> list[EvaluatedCandidate]:
        population = unique_by_structure(seeds)
        keys = {c.key for c in population}
        attempts = 0
        while len(population) < self.config.population_size and attempts < self.config.population_size * 6:
            parent = seeds[attempts % len(seeds)]
            attempts += 1
            child = self.operators.mutate(parent.playbook, parent.profile, intent, available, rng,
                                          directed=self.config.coverage_init)  # fmt: skip
            evaluated = self.evaluator.evaluate(child, intent, available) if child is not None else None
            if evaluated is not None and evaluated.key not in keys:
                population.append(evaluated)
                keys.add(evaluated.key)
        return population

    def _offspring(self, population, elite_count, intent, available, rng) -> list[EvaluatedCandidate]:
        config = self.config
        offspring: list[EvaluatedCandidate] = []
        for _ in range(max(1, config.population_size - elite_count)):
            parent = select_parent(population, rng, stratified=config.coverage_selection)
            child = None
            if len(population) > 1 and rng.random() < config.crossover_probability:
                child = self._crossover(population, parent, intent, available, rng)
            for _ in range(config.mutation_retries if child is None else 0):
                directed = config.coverage_variation and rng.random() < config.directed_mutation_probability
                mutant = self.operators.mutate(parent.playbook, parent.profile, intent, available, rng,
                                               directed=directed)  # fmt: skip
                child = self.evaluator.evaluate(mutant, intent, available) if mutant is not None else None
                if child is not None:
                    break
            offspring.append(child or parent)  # an unmutable parent is kept
        return offspring

    def _crossover(self, population, recipient, intent, available, rng):
        if self.config.coverage_variation:
            donor = select_donor(population, recipient, rng)
        else:
            donor = select_parent(population, rng, stratified=self.config.coverage_selection)
            donor = donor if donor.key != recipient.key else None
        if donor is None:
            return None
        child = self.operators.crossover(recipient.playbook, recipient.profile, donor.playbook, donor.profile,
                                         intent, available, rng)  # fmt: skip
        return self.evaluator.evaluate(child, intent, available) if child is not None else None

    def _survivors(self, candidates, rng) -> list[EvaluatedCandidate]:
        size, elite_count = self.config.population_size, self.config.elite_count
        if self.config.coverage_selection:
            return stratified_survivors(candidates, size, elite_count)
        return proportional_survivors(candidates, size, elite_count, rng)

    def _good_enough(self, candidate: EvaluatedCandidate) -> bool:
        return candidate.profile.complete and candidate.fitness >= self.config.fitness_threshold

    @staticmethod
    def _best_judged(population: list[EvaluatedCandidate]) -> EvaluatedCandidate:
        # judge_top always judges the leader, so a judged candidate exists
        return max((c for c in population if c.judged), key=lambda c: c.fitness)

    @staticmethod
    def _stats(generation: int, population: list[EvaluatedCandidate], best: EvaluatedCandidate) -> GenerationStats:
        return GenerationStats(
            generation=generation,
            best_fitness=best.fitness,
            mean_fitness=sum(c.fitness for c in population) / len(population),
            best_ic=best.scores.ic,
            best_ea=best.scores.ea,
            best_os=best.scores.os,
            niches=len({c.profile.vector for c in population}),
            population=len(population),
        )
