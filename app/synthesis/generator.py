"""LLM generation of the candidate pool (paper Section V-B).

A base pool is sampled with mixed decoding temperatures and diversity prompting, conditioned on
the intent, the evidence profile and the most relevant prior playbooks. Every sample is untrusted:
it is parsed against a schema, converted to a typed DAG, repaired, and discarded if still invalid.
If the model yields no valid candidate at all, the pool is filled by the constructive generator
(and the report says so).
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Optional

from pydantic import ValidationError

from app.core.errors import CADEError, LLMModelNotFoundError, LLMUnavailableError
from app.core.logging import get_logger, log_event
from app.fitness.coverage import CoverageModel, Gap
from app.intent.llm_client import LLMClient
from app.playbook.repair import PlaybookRepairer
from app.playbook.vocabulary import Vocabulary
from app.schemas.playbook import ACTION_ID_PATTERN, Playbook, PlaybookEdge, PlaybookNode, PlaybookOrigin
from app.synthesis.constructive import ConstructiveGenerator
from app.synthesis.context import SynthesisContext
from app.synthesis.prompts import (
    DIVERSITY_DIRECTIVES,
    GAP_PROPOSAL_SYSTEM_PROMPT,
    GENERATION_SYSTEM_PROMPT,
    PlaybookCandidate,
    build_gap_proposal_prompt,
    build_generation_prompt,
    candidate_schema,
    gap_key,
    gap_proposal_schema,
)
from app.tools import artifacts as A

logger = get_logger("synthesis")


@dataclass
class GenerationReport:
    requested: int = 0
    provider: Optional[str] = None
    model: Optional[str] = None
    llm_calls: int = 0
    llm_failures: int = 0
    unparseable: int = 0
    llm_valid: int = 0
    discarded: int = 0
    examples_used: int = 0
    constructive_used: int = 0
    fallback_reason: Optional[str] = None
    temperatures: list[float] = field(default_factory=list)


def candidate_to_playbook(candidate: PlaybookCandidate, intent_label: str, vocabulary: Vocabulary) -> Playbook:
    """Map model steps to nodes, and ``depends_on`` to typed edges derived from the vocabulary."""
    ids: dict[str, str] = {}
    nodes: list[PlaybookNode] = []
    for index, step in enumerate(candidate.steps[:60], start=1):
        if step.id in ids:
            continue
        ids[step.id] = f"n{index}"
        # an action name that is not even well-formed becomes an unknown action, which repair drops
        action = step.action if re.fullmatch(ACTION_ID_PATTERN, step.action) else "unknown_action"
        nodes.append(PlaybookNode(node_id=f"n{index}", action=action))
    edges: list[PlaybookEdge] = []
    for step in candidate.steps[:60]:
        target = ids.get(step.id)
        target_action = vocabulary.get(step.action)
        for dependency in step.depends_on:
            source = ids.get(dependency)
            source_step = next((s for s in candidate.steps if s.id == dependency), None)
            source_action = vocabulary.get(source_step.action) if source_step else None
            if not (source and target and source != target and source_action and target_action):
                continue
            shared = [a for a in source_action.outputs if a in target_action.inputs]
            shared.sort(key=lambda a: a == A.FINDINGS)  # prefer a specific artifact over generic findings
            if shared:
                edges.append(PlaybookEdge(source=source, target=target, artifact=shared[0]))
    return Playbook(name=candidate.name[:200], intent_label=intent_label, origin=PlaybookOrigin.LLM,
                    description=candidate.rationale[:2000] or None, nodes=nodes, edges=list(dict.fromkeys(edges)))  # fmt: skip


class CandidateGenerator:
    def __init__(
        self,
        llm: Optional[LLMClient],
        coverage: CoverageModel,
        repairer: PlaybookRepairer,
        constructive: ConstructiveGenerator,
        *,
        pool_size: int = 8,
        temperatures: tuple[float, ...] = (0.2, 0.6, 0.9, 1.2),
    ) -> None:
        self.llm = llm
        self.coverage = coverage
        self.vocabulary = coverage.vocabulary
        self.repairer = repairer
        self.constructive = constructive
        self.pool_size = pool_size
        self.temperatures = tuple(temperatures) or (0.7,)
        self.schema = candidate_schema(self.vocabulary)

    async def generate(
        self, context: SynthesisContext, examples: list[Playbook], rng: random.Random
    ) -> tuple[list[Playbook], GenerationReport]:
        report = GenerationReport(requested=self.pool_size)
        pool: list[Playbook] = []
        if self.llm is not None:
            report.provider, report.model = self.llm.provider, self.llm.model
            await self._sample_llm(context, examples, rng, pool, report)
        else:
            report.fallback_reason = "no generator model configured"

        for example in examples:  # prior playbooks condition the search (adapted to this incident)
            adapted = self.repairer.repair(example, context.available, rng=rng)
            if adapted is not None:
                # the seed's own description ("hand-authored ...") would misdescribe the adapted copy
                pool.append(adapted.model_copy(update={"description": f"Adapted from library playbook '{example.name}'."}))
                report.examples_used += 1

        if report.llm_valid == 0:
            report.fallback_reason = report.fallback_reason or "the model produced no valid candidate"
            for index in range(self.pool_size):
                built = self.constructive.build(context, rng, goal_rate=0.6 + 0.4 * ((index % 4) / 3),
                                                category_rate=0.4 + 0.15 * (index % 4), verify_rate=0.5)  # fmt: skip
                if built is not None:
                    pool.append(built)
                    report.constructive_used += 1
        log_event(logger, "candidate pool generated", incident_id=context.incident_id, operation="synthesize",
                  stage="generation", outcome="success", pool=len(pool), llm_calls=report.llm_calls,
                  llm_valid=report.llm_valid, discarded=report.discarded, constructive=report.constructive_used)  # fmt: skip
        return pool, report

    async def propose_gap_actions(self, context: SynthesisContext, gaps: list[Gap]) -> dict[Gap, str]:
        """The generator's choice of action for each coverage gap (paper Section 5.3: gap-directed
        additions use the action the generator proposes for the gap).

        One structured call covers every gap; each answer must be one of the gap's closers, anything
        else is ignored. Without a model, or if the call fails, the result is empty and the
        operators fall back to a random closer.
        """
        options = {gap: [a.id for a in self.coverage.closers(gap, context.intent_label)] for gap in gaps}
        options = {gap: ids for gap, ids in options.items() if ids}
        if self.llm is None or not options:
            return {}
        try:
            response = await self.llm.generate_json(
                system_prompt=GAP_PROPOSAL_SYSTEM_PROMPT,
                user_prompt=build_gap_proposal_prompt(context, self.vocabulary, self.coverage, options),
                json_schema=gap_proposal_schema(options), temperature=0.0,
            )  # fmt: skip
            answers = json.loads(response.content)
        except (CADEError, ValueError) as exc:
            log_event(logger, "gap proposals unavailable", level=logging.WARNING, incident_id=context.incident_id,
                      operation="synthesize", stage="gap_proposals", outcome="failure",
                      error_category=getattr(exc, "category", type(exc).__name__))  # fmt: skip
            return {}
        if not isinstance(answers, dict):
            return {}
        return {gap: answers[gap_key(gap)] for gap, ids in options.items() if answers.get(gap_key(gap)) in ids}

    async def _sample_llm(self, context, examples, rng, pool, report) -> None:
        for index in range(self.pool_size):
            temperature = self.temperatures[index % len(self.temperatures)]
            directive = DIVERSITY_DIRECTIVES[index % len(DIVERSITY_DIRECTIVES)]
            prompt = build_generation_prompt(context, self.vocabulary, self.coverage, examples, directive)
            report.llm_calls += 1
            report.temperatures.append(temperature)
            try:
                response = await self.llm.generate_json(system_prompt=GENERATION_SYSTEM_PROMPT, user_prompt=prompt,
                                                        json_schema=self.schema, temperature=temperature)  # fmt: skip
            except (LLMUnavailableError, LLMModelNotFoundError) as exc:
                report.llm_failures += 1
                report.fallback_reason = f"generator model unavailable ({exc.category})"
                break  # no point retrying a model that is not there
            except CADEError as exc:
                report.llm_failures += 1
                log_event(logger, "generation sample failed", level=logging.WARNING, incident_id=context.incident_id,
                          operation="synthesize", stage="generation", outcome="failure", error_category=exc.category)  # fmt: skip
                continue
            try:
                candidate = PlaybookCandidate.model_validate(json.loads(response.content))
            except (ValueError, ValidationError):
                report.unparseable += 1
                continue
            repaired = self.repairer.repair(candidate_to_playbook(candidate, context.intent_label, self.vocabulary),
                                            context.available, rng=rng, origin=PlaybookOrigin.LLM)  # fmt: skip
            if repaired is None:
                report.discarded += 1
                continue
            pool.append(repaired)
            report.llm_valid += 1
