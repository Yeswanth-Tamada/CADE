"""End-to-end incident response orchestration (paper Section III, Fig. 1).

``respond`` routes the analysed incident (Eq. 1-5) and acts on the decision:

* **reuse** -- the top library playbook becomes the response plan;
* **request evidence** -- the obtainable missing critical evidence is requested from the tools;
  categories that yield nothing become unobtainable for this incident, and the incident is routed
  again (at most ``max_evidence_rounds`` times; then the request option is dropped). With
  ``auto_acquire=false`` the run stops and reports the evidence requests instead;
* **synthesize** -- LLM candidate pool -> adaptive-K seeds -> CADE evolution -> best playbook.

Simulated evidence counts towards routing only in simulate mode, and every decision or plan that
relied on it is flagged ``synthetic_basis``.
"""

from __future__ import annotations

import random
import secrets
from dataclasses import asdict
from typing import Optional

from app.bootstrap import CADEComponents
from app.core.config import Settings
from app.core.errors import SynthesisError
from app.core.logging import get_logger, log_event
from app.core.threads import run_sync
from app.routing.router import build_query
from app.schemas.enums import EvidenceCategory, ExecutionMode
from app.schemas.evidence import EffectiveEvidenceProfile
from app.schemas.incident import CanonicalIncident
from app.schemas.jobs import IncidentResponse, ResponseOutcome
from app.schemas.response import PlanSource, ResponsePlan
from app.schemas.routing import RoutingAction, StoredRoutingDecision
from app.security.auth import Principal
from app.services.audit_service import AuditService
from app.services.evidence_service import EvidenceService
from app.services.jobs import Progress
from app.storage.response_repository import ResponseRepository
from app.storage.routing_repository import RoutingRepository
from app.synthesis.context import SynthesisContext
from app.synthesis.prompts import GENERATION_PROMPT_VERSION
from app.tools.artifacts import entity_artifacts

logger = get_logger("response")


async def _no_progress(message: str) -> None:
    return None


class ResponseService:
    def __init__(
        self,
        components: CADEComponents,
        evidence_service: EvidenceService,
        routing: RoutingRepository,
        responses: ResponseRepository,
        audit: AuditService,
        settings: Settings,
    ) -> None:
        self.components = components
        self.evidence_service = evidence_service
        self.routing = routing
        self.responses = responses
        self.audit = audit
        self.settings = settings

    @property
    def include_simulated(self) -> bool:
        return self.settings.execution_mode == ExecutionMode.SIMULATE

    # --- routing --------------------------------------------------------------------------------

    async def route(self, incident_id: str, actor: Optional[Principal], *,
                    unobtainable: frozenset[EvidenceCategory] = frozenset(), round_number: int = 1,
                    allow_request: bool = True) -> StoredRoutingDecision:  # fmt: skip
        analysis = await self.evidence_service.usable_analysis(incident_id)
        evidence = await self.evidence_service.effective_profile(incident_id)
        decision = await run_sync(self.components.router.decide, analysis.result, evidence,
                                  include_simulated=self.include_simulated, unobtainable=unobtainable,
                                  round_number=round_number, allow_request=allow_request)  # fmt: skip
        stored = await run_sync(self.routing.record, decision, actor.analyst_id if actor else None)
        await self.audit.record(actor, "routing.decide", outcome=decision.action.value, target_type="incident",
                                target_id=incident_id,
                                details={"decision_id": stored.decision_id, "round": round_number,
                                         "scores": decision.scores.model_dump(), "policy": decision.policy,
                                         "rule_action": decision.rule_action.value if decision.rule_action else None,
                                         "synthetic_basis": decision.synthetic_basis})  # fmt: skip
        return stored

    # --- the full pipeline --------------------------------------------------------------------------

    async def respond(self, incident_id: str, actor: Optional[Principal], *, auto_acquire: bool = True,
                      force_synthesis: bool = False, progress: Progress = _no_progress) -> ResponseOutcome:  # fmt: skip
        analysis = await self.evidence_service.usable_analysis(incident_id)
        incident = analysis.result
        outcome = ResponseOutcome(incident_id=incident_id, status="plan_created", forced_synthesis=force_synthesis)
        unobtainable: set[EvidenceCategory] = set()

        for round_number in range(1, self.settings.max_evidence_rounds + 2):
            await progress(f"routing (round {round_number})")
            # Requests are capped per incident (paper: one); the final round chooses reuse or synthesis.
            decision = await self.route(incident_id, actor, unobtainable=frozenset(unobtainable), round_number=round_number,
                                        allow_request=round_number <= self.settings.max_evidence_rounds)  # fmt: skip
            outcome.decisions.append(decision)
            action = RoutingAction.SYNTHESIZE if force_synthesis else decision.action
            if action == RoutingAction.REQUEST_EVIDENCE and round_number > self.settings.max_evidence_rounds:
                action = RoutingAction.SYNTHESIZE  # defensive: the cap holds even if a router ignores it
            if action != RoutingAction.REQUEST_EVIDENCE:
                break
            if not auto_acquire:
                outcome.status = "awaiting_evidence"
                outcome.evidence_requests = list(decision.requestable)
                return outcome
            await progress(f"requesting evidence: {', '.join(c.value for c in decision.requestable)}")
            report = await self.evidence_service.enrich(incident_id, list(decision.requestable), actor)
            outcome.enrichment.append(report)
            obtained = set(report.categories_obtained) | (set(report.categories_simulated) if self.include_simulated else set())
            unobtainable |= set(decision.requestable) - obtained

        evidence = await self.evidence_service.effective_profile(incident_id)
        plan = None
        if action == RoutingAction.REUSE and decision.top_candidate is not None:
            plan = await self._reuse(incident, decision, actor)
        if plan is None:
            await progress("synthesizing a new playbook")
            plan = await self.synthesize(incident, evidence, actor, decision)
        outcome.plan = plan
        return outcome

    async def _reuse(self, incident: CanonicalIncident, decision: StoredRoutingDecision,
                     actor: Optional[Principal]) -> Optional[ResponsePlan]:  # fmt: skip
        entry = self.components.library.get(decision.top_candidate.entry_id)
        if entry is None:
            return None
        available = frozenset(k for k, v in self._entities(incident).items() if v)
        playbook = self.components.repairer.repair(entry.playbook, available)  # adapt to this incident (usually a no-op)
        if playbook is None:
            log_event(logger, "library playbook not applicable; synthesizing", incident_id=incident.incident_id,
                      operation="respond", stage="reuse", outcome="fallback")  # fmt: skip
            return None
        scored = self.components.evaluator.evaluate(playbook, incident.intent.label, available)
        return await self.components.plan_service.create_plan(
            incident, playbook, source=PlanSource.LIBRARY, actor=actor, library_entry_id=entry.entry_id,
            routing_decision_id=decision.decision_id, fitness=scored.scores.model_dump() if scored else None,
            synthetic_basis=decision.synthetic_basis,
        )  # fmt: skip

    async def synthesize(self, incident: CanonicalIncident, evidence: EffectiveEvidenceProfile,
                         actor: Optional[Principal], decision: Optional[StoredRoutingDecision] = None) -> ResponsePlan:  # fmt: skip
        components = self.components
        context = SynthesisContext.from_incident(incident, evidence, include_simulated=self.include_simulated)
        seed = self.settings.random_seed if self.settings.random_seed is not None else secrets.randbits(32)
        rng = random.Random(seed)
        query = build_query(incident, evidence, include_simulated=self.include_simulated)
        examples = [c.playbook for c in await run_sync(components.library.retrieve, query, self.settings.generation_examples)
                    if c.playbook.intent_label == incident.intent.label][: self.settings.generation_examples]  # fmt: skip

        pool, generation = await components.generator.generate(context, examples, rng)
        selection = components.seeder.select(pool, context.intent_label, context.available, rng,
                                             adaptive=self.settings.coverage_init)  # fmt: skip
        if not selection.seeds:
            raise SynthesisError("no admissible candidate playbook could be generated", incident_id=incident.incident_id)
        engine = components.new_engine()
        if self.settings.directed_mutation_llm and self.settings.coverage_variation:
            # gap-directed additions use the generator's proposed action for each gap the seeds leave open
            open_gaps = {gap for seed_candidate in selection.seeds for gap in seed_candidate.profile.gaps()}
            ordered = [gap for gap in components.coverage.all_gaps(context.intent_label) if gap in open_gaps]
            engine.operators.proposals = await components.generator.propose_gap_actions(context, ordered)
        result = await engine.run(selection.seeds, context.intent_label, context.available, rng,
                                  incident_id=incident.incident_id)  # fmt: skip
        best = result.best
        synthesis = {
            "prompt_version": GENERATION_PROMPT_VERSION,
            "generator": {"provider": generation.provider, "model": generation.model},
            "judge_model": components.judge_llm.model if components.judge_llm else None,
            "generation": asdict(generation),
            "seeding": {"pool": selection.pool_size, "admissible": selection.admissible,
                        "duplicates": selection.duplicates, "clusters": selection.clusters, "k": selection.k,
                        "adaptive": selection.adaptive},
            "gap_proposals": {f"{gap.kind}:{gap.key}": action for gap, action in engine.operators.proposals.items()},
            "evolution": {"termination": result.termination, "generations": result.generations,
                          "evaluations": result.evaluations, "judge_calls": result.judge_calls,
                          "duration_ms": round(result.duration_ms, 1), "operators": result.operator_counts,
                          "history": [asdict(h) for h in result.history],
                          "config": asdict(components.engine_config)},
            "random_seed": seed,
            "examples": [p.name for p in examples],
        }  # fmt: skip
        log_event(logger, "playbook synthesized", incident_id=incident.incident_id, operation="synthesize",
                  stage="complete", outcome=result.termination, fitness=round(best.fitness, 4),
                  generations=result.generations, k=selection.k)  # fmt: skip
        return await components.plan_service.create_plan(
            incident, best.playbook, source=PlanSource.SYNTHESIZED, actor=actor,
            routing_decision_id=decision.decision_id if decision else None, fitness=best.scores.model_dump(),
            synthesis=synthesis, synthetic_basis=context.synthetic_basis,
        )  # fmt: skip

    async def history(self, incident_id: str) -> IncidentResponse:
        await self.evidence_service.usable_analysis(incident_id)
        return IncidentResponse(
            incident_id=incident_id,
            decisions=await run_sync(self.routing.list, incident_id),
            plans=await run_sync(self.responses.list_plans, incident_id),
            runs=await run_sync(self.responses.list_runs, incident_id),
        )

    @staticmethod
    def _entities(incident: CanonicalIncident) -> dict[str, list[str]]:
        return entity_artifacts(incident.entities)
