"""The CADE fitness function (paper Eq. 10): f(p) = cube root of IC(p) * EA(p) * OS(p).

Invalid playbooks and playbooks whose destructive actions lack a prior verification are hard-
filtered: they are rejected before scoring and never receive a fitness value.

Evaluation is two-stage (paper Section 5.4, Eq. 12): :meth:`FitnessEvaluator.evaluate` scores a
candidate with OS_H, giving its provisional fitness; :meth:`FitnessEvaluator.judge_top` replaces
OS_H with the LLM judge's OS_J for the top m candidates by provisional fitness. A candidate keeps
its judged score until variation modifies it (a new structure).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Optional

from pydantic import BaseModel, Field

from app.fitness.coverage import CoverageModel, CoverageProfile
from app.fitness.judge import HeuristicSoundness, LLMSoundnessJudge
from app.playbook.validator import PlaybookValidator
from app.schemas.playbook import Playbook


def geometric_mean(ic: float, ea: float, os: float) -> float:
    product = max(0.0, ic) * max(0.0, ea) * max(0.0, os)
    return product ** (1 / 3)


class FitnessScores(BaseModel):
    ic: float = Field(ge=0, le=1)
    ea: float = Field(ge=0, le=1)
    os: float = Field(ge=0, le=1)
    # "llm" (judged), "heuristic" (no LLM judge configured), "estimated" (proxy pending the judge),
    # "heuristic_fallback" (every judge sample was unusable)
    os_source: str
    fitness: float = Field(ge=0, le=1)


@dataclass(frozen=True)
class EvaluatedCandidate:
    playbook: Playbook
    profile: CoverageProfile
    scores: FitnessScores

    @property
    def fitness(self) -> float:
        return self.scores.fitness

    @property
    def key(self) -> str:
        return self.playbook.structural_hash()

    @property
    def judged(self) -> bool:
        """OS is final (Eq. 12): from the judge, or OS_H when no judge is configured -- not a provisional estimate."""
        return self.scores.os_source != "estimated"


class FitnessEvaluator:
    def __init__(
        self,
        coverage: CoverageModel,
        validator: PlaybookValidator,
        heuristic: HeuristicSoundness,
        judge: Optional[LLMSoundnessJudge] = None,
        *,
        judge_top_k: int = 5,
    ) -> None:
        self.coverage = coverage
        self.validator = validator
        self.heuristic = heuristic
        self.judge = judge
        self.judge_top_k = judge_top_k
        self.evaluations = 0

    def admissible(self, playbook: Playbook, available: Iterable[str] = ()) -> bool:
        """The hard filter: valid DAG and every destructive action verified."""
        report = self.validator.validate(playbook, available)
        return report.valid and report.safe

    def evaluate(self, playbook: Playbook, intent_label: str, available: Iterable[str] = ()) -> Optional[EvaluatedCandidate]:
        """Score with the deterministic OS proxy; ``None`` if the hard filter rejects the playbook."""
        if not self.admissible(playbook, available):
            return None
        self.evaluations += 1
        profile = self.coverage.profile(playbook, intent_label)
        cached = self.judge.cached(playbook, intent_label) if self.judge else None
        if cached is not None:
            os_score, source = cached.score, cached.source
        else:
            os_score = self.heuristic.score(playbook, intent_label).score
            source = "estimated" if self.judge else "heuristic"
        scores = FitnessScores(ic=profile.ic, ea=profile.ea, os=os_score, os_source=source,
                               fitness=geometric_mean(profile.ic, profile.ea, os_score))  # fmt: skip
        return EvaluatedCandidate(playbook, profile, scores)

    async def judge_top(self, candidates: list[EvaluatedCandidate], intent_label: str) -> list[EvaluatedCandidate]:
        """Replace the OS proxy with the LLM judge for the ``judge_top_k`` fittest not-yet-judged candidates."""
        if self.judge is None:
            return candidates
        ranked = sorted(candidates, key=lambda c: c.fitness, reverse=True)
        pending = [c for c in ranked if c.scores.os_source == "estimated"][: self.judge_top_k]
        judged = {candidate.key: await self._judge(candidate, intent_label) for candidate in pending}
        # cached judgements also apply to other copies of the same structure
        candidates = [judged.get(c.key, c) if c.scores.os_source == "estimated" else c for c in candidates]
        return await self.judge_leader(candidates, intent_label)

    async def judge_leader(self, candidates: list[EvaluatedCandidate], intent_label: str) -> list[EvaluatedCandidate]:
        """Judge the current leader until the leader's OS comes from the judge, not the proxy.

        Judging the top-k can lower their OS and let a candidate scored only by the proxy take the
        lead; termination and the returned playbook must never rest on an estimated OS.
        """
        if self.judge is None:
            return candidates
        candidates = list(candidates)
        for _ in range(len(candidates)):
            leader = max(candidates, key=lambda c: c.fitness)
            if leader.scores.os_source != "estimated":
                break
            judged = await self._judge(leader, intent_label)
            candidates = [judged if c.key == leader.key else c for c in candidates]
        return candidates

    async def _judge(self, candidate: EvaluatedCandidate, intent_label: str) -> EvaluatedCandidate:
        result = await self.judge.score(candidate.playbook, intent_label)
        scores = candidate.scores.model_copy(update={
            "os": result.score, "os_source": result.source,
            "fitness": geometric_mean(candidate.scores.ic, candidate.scores.ea, result.score),
        })  # fmt: skip
        return replace(candidate, scores=scores)
