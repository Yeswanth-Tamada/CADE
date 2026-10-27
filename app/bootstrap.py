"""Assembly of CADE's playbook components (Phases 3-9) from settings.

Kept apart from ``app.main`` so the wiring can be built and tested without the HTTP layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from app.core.config import LLMProvider, Settings
from app.core.logging import get_logger, log_event
from app.enrichment.availability import AvailabilityProfile
from app.evolution.engine import CADEEngine, EvolutionConfig
from app.evolution.operators import Operators
from app.execution.executor import PlaybookExecutor
from app.fitness.coverage import CoverageModel
from app.fitness.fitness import FitnessEvaluator
from app.fitness.judge import HeuristicSoundness, LLMSoundnessJudge
from app.intent.llm_client import LLMClient, build_llm_client, model_family
from app.library.embedder import build_embedder
from app.library.index import build_index
from app.library.library import PlaybookLibrary
from app.playbook.repair import PlaybookRepairer
from app.playbook.validator import PlaybookValidator
from app.playbook.vocabulary import Vocabulary
from app.routing.learning import RoutingLearner
from app.routing.router import Router
from app.schemas.taxonomy import IntentTaxonomy
from app.services.audit_service import AuditService
from app.services.evidence_service import EvidenceService
from app.services.plan_service import PlanService
from app.storage.bandit_repository import BanditRepository
from app.storage.database import Database
from app.storage.evidence_repository import EvidenceRepository
from app.storage.library_repository import LibraryRepository
from app.storage.response_repository import ResponseRepository
from app.storage.routing_repository import RoutingRepository
from app.synthesis.constructive import ConstructiveGenerator
from app.synthesis.generator import CandidateGenerator
from app.synthesis.seeding import AdaptiveSeeder
from app.tools.registry import ToolGateway

logger = get_logger("app")


@dataclass
class CADEComponents:
    vocabulary: Vocabulary
    validator: PlaybookValidator
    repairer: PlaybookRepairer
    coverage: CoverageModel
    evaluator: FitnessEvaluator
    generator: CandidateGenerator
    seeder: AdaptiveSeeder
    operators: Operators
    engine_config: EvolutionConfig
    library: PlaybookLibrary
    router: Router
    executor: PlaybookExecutor
    plan_service: PlanService
    generator_llm: Optional[LLMClient]
    judge_llm: Optional[LLMClient]
    learner: Optional[RoutingLearner] = None
    notes: dict[str, Optional[str]] = field(default_factory=dict)

    def new_engine(self) -> CADEEngine:
        """A fresh engine (with fresh operator counters) for one synthesis run."""
        return CADEEngine(self.evaluator, Operators(self.coverage, self.repairer), self.engine_config)

    async def aclose(self) -> None:
        for client in {id(c): c for c in (self.generator_llm, self.judge_llm) if c is not None}.values():
            await client.aclose()


def build_role_clients(settings: Settings) -> tuple[LLMClient, Optional[LLMClient], Optional[str]]:
    """Generator and judge clients (independently configured models, paper Section 3 and Table 3)."""
    generator = build_llm_client(settings, settings.generator_model)
    use_llm_judge = settings.os_judge_mode == "llm" and settings.llm_provider != LLMProvider.MOCK
    judge = build_llm_client(settings, settings.judge_model) if use_llm_judge else None
    warning = None
    if judge is not None and model_family(settings.judge_model) == model_family(settings.generator_model):
        warning = (f"the judge model ({settings.judge_model}) is from the same family as the generator "
                   f"({settings.generator_model}); the paper recommends a different family to limit self-preference bias")  # fmt: skip
        log_event(logger, "judge model family matches generator", level=logging.WARNING, operation="startup",
                  outcome="warning", generator=settings.generator_model, judge=settings.judge_model)  # fmt: skip
    return generator, judge, warning


def build_components(
    settings: Settings,
    database: Database,
    taxonomy: IntentTaxonomy,
    availability: AvailabilityProfile,
    gateway: ToolGateway,
    evidence_service: EvidenceService,
    audit: AuditService,
    *,
    generator_llm: Optional[LLMClient] = None,
    judge_llm: Optional[LLMClient] = None,
) -> CADEComponents:
    vocabulary = Vocabulary.load(settings.resolve_path(settings.action_vocabulary_path),
                                 settings.resolve_path(settings.intent_goals_path), taxonomy)  # fmt: skip
    validator = PlaybookValidator(vocabulary, node_cap=settings.node_cap)
    repairer = PlaybookRepairer(validator)
    coverage = CoverageModel(vocabulary, taxonomy)
    heuristic = HeuristicSoundness(coverage, validator)

    notes: dict[str, Optional[str]] = {}
    if generator_llm is None:
        generator_llm, built_judge, notes["judge_family_warning"] = build_role_clients(settings)
        judge_llm = judge_llm or built_judge
    judge = (LLMSoundnessJudge(judge_llm, coverage, heuristic, samples=settings.os_judge_samples,
                               temperature=settings.os_judge_temperature) if judge_llm is not None else None)  # fmt: skip
    evaluator = FitnessEvaluator(coverage, validator, heuristic, judge, judge_top_k=settings.os_judge_top_k)
    generator = CandidateGenerator(generator_llm, coverage, repairer, ConstructiveGenerator(coverage, repairer),
                                   pool_size=settings.generation_pool_size,
                                   temperatures=tuple(settings.generation_temperatures))  # fmt: skip
    seeder = AdaptiveSeeder(evaluator, k_min=settings.seed_k_min, k_max=settings.seed_k_max,
                            threshold=settings.seed_cluster_threshold)  # fmt: skip

    embedder, notes["embedder_fallback"] = build_embedder(settings.embedder, settings.embedding_model)
    index, notes["index_fallback"] = build_index(settings.vector_index, embedder.dimension)
    library = PlaybookLibrary(
        LibraryRepository(database), embedder, index, vocabulary, validator, k=settings.routing_k,
        near_duplicate_cosine=settings.library_near_duplicate_cosine,
        near_duplicate_jaccard=settings.library_near_duplicate_jaccard,
        deprioritize_after=settings.library_deprioritize_after, deprioritize_below=settings.library_deprioritize_below,
        deprioritized_weight=settings.library_deprioritized_weight,
        allow_curated_reuse=settings.library_allow_curated_reuse,
    )  # fmt: skip
    if settings.library_seed_on_startup:
        library.seed_if_empty(settings.resolve_path(settings.library_seed_path))
    library.load()
    # The learned router logs rewards from the first incident; with routing_policy="rule" it never chooses.
    learner = RoutingLearner.from_settings(settings, BanditRepository(database))
    notes["routing_state"] = learner.state_warning()  # stale state is reported, never changed silently
    router = Router(library, vocabulary, availability, w=settings.routing_w, k=settings.routing_k,
                    top_n=settings.routing_top_n, tau=settings.routing_tau,
                    sharpness=settings.routing_similarity_sharpness, taxonomy=taxonomy, learner=learner,
                    library_size=library.__len__)  # fmt: skip
    executor = PlaybookExecutor(validator, gateway)
    plan_service = PlanService(ResponseRepository(database), evidence_service, EvidenceRepository(database), validator,
                               executor, library, audit, settings, learner=learner,
                               routing=RoutingRepository(database))  # fmt: skip
    return CADEComponents(
        vocabulary=vocabulary, validator=validator, repairer=repairer, coverage=coverage, evaluator=evaluator,
        generator=generator, seeder=seeder, operators=Operators(coverage, repairer),
        engine_config=EvolutionConfig.from_settings(settings), library=library, router=router, executor=executor,
        plan_service=plan_service, generator_llm=generator_llm, judge_llm=judge_llm, learner=learner, notes=notes,
    )  # fmt: skip
