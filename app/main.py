"""FastAPI application factory.

Run the server with::

    uvicorn app.main:app --reload

Interactive API docs are served at http://127.0.0.1:8000/docs (Swagger UI) and /redoc.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.utils import get_openapi

from app import __version__
from app.api.errors import register_error_handlers
from app.api.evidence_routes import router as evidence_router
from app.api.library_routes import router as library_router
from app.api.response_routes import router as response_router
from app.api.routes import router
from app.bootstrap import build_components
from app.core.config import AppEnv, Settings, get_settings
from app.core.errors import ConfigurationError
from app.core.logging import configure_logging, get_logger, log_event
from app.enrichment.availability import AvailabilityProfile
from app.enrichment.enricher import EvidenceEnricher
from app.ingestion.input_service import IncidentInputService
from app.intent.attack_mapper import AttackMapper
from app.intent.llm_client import LLMClient, build_llm_client
from app.intent.llm_extractor import IntentExtractor
from app.schemas.incident import IncidentSubmission
from app.schemas.taxonomy import IntentTaxonomy
from app.services.audit_service import AuditService
from app.services.evidence_service import EvidenceService
from app.services.incident_service import IncidentService
from app.services.jobs import JobManager
from app.services.response_service import ResponseService
from app.storage.database import Database
from app.storage.evidence_repository import EvidenceRepository
from app.storage.repository import IncidentRepository
from app.storage.response_repository import ResponseRepository
from app.storage.routing_repository import JobRepository, RoutingRepository
from app.storage.security_repository import ApiKeyRepository, AuditRepository
from app.tools.base import SecurityTool
from app.tools.registry import build_tool_gateway

logger = get_logger("app")

DESCRIPTION = """
Phase 1 of **CADE** (Coverage-Aware DAG Evolution for adaptive incident response):
incident ingestion, LLM-based intent extraction, evidence profiling and MITRE ATT&CK grounding.

Every endpoint except `/health` requires `Authorization: Bearer <API key>`; issue keys with
`python -m app.security.keys create --analyst <id> --roles <roles>`.
"""


def create_app(
    settings: Optional[Settings] = None,
    *,
    llm_client: Optional[LLMClient] = None,
    attack_mapper: Optional[AttackMapper] = None,
    real_tools: Optional[dict[str, SecurityTool]] = None,
    generator_llm: Optional[LLMClient] = None,
    judge_llm: Optional[LLMClient] = None,
) -> FastAPI:
    """Build the app. Tests inject a mock LLM client, ATT&CK mapper or tool adapters through the arguments."""
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    production = settings.app_env == AppEnv.PRODUCTION
    if production and not settings.auth_enabled:
        raise ConfigurationError("CADE_AUTH_ENABLED=false is not allowed when CADE_APP_ENV=production")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        database = Database(settings.database_url)
        database.create_schema()
        taxonomy = IntentTaxonomy.load(settings.resolve_path(settings.intent_taxonomy_path))
        mapper = attack_mapper or AttackMapper.from_path(settings.resolve_path(settings.attack_dataset_path))
        llm = llm_client or build_llm_client(settings)
        extractor = IntentExtractor(
            llm,
            taxonomy,
            mapper,
            max_attempts=settings.llm_max_attempts,
            attack_prompt_hints=settings.attack_prompt_hints,
            two_stage=settings.intent_two_stage,
        )
        # An analysis older than this is presumed abandoned (e.g. server crash) and may be restarted.
        stale_after = timedelta(seconds=settings.llm_request_timeout_seconds * settings.llm_max_attempts + 60)

        app.state.settings = settings
        app.state.database = database
        app.state.llm_client = llm
        app.state.attack_mapper = mapper
        app.state.api_key_repository = ApiKeyRepository(database)
        app.state.audit = AuditService(AuditRepository(database))
        incidents = IncidentRepository(database)
        availability = AvailabilityProfile.load(settings.resolve_path(settings.evidence_availability_path))
        gateway = build_tool_gateway(settings, availability, real_tools)
        app.state.availability = availability
        app.state.tool_gateway = gateway
        app.state.evidence_service = EvidenceService(
            incidents,
            EvidenceRepository(database),
            EvidenceEnricher(gateway, max_calls=settings.enrichment_max_calls,
                             max_values_per_type=settings.enrichment_max_values_per_type),
            app.state.audit,
        )  # fmt: skip
        components = build_components(settings, database, taxonomy, availability, gateway, app.state.evidence_service,
                                      app.state.audit, generator_llm=generator_llm, judge_llm=judge_llm)  # fmt: skip
        app.state.components = components
        app.state.plan_service = components.plan_service
        app.state.response_service = ResponseService(components, app.state.evidence_service, RoutingRepository(database),
                                                     ResponseRepository(database), app.state.audit, settings)  # fmt: skip
        app.state.jobs = JobManager(JobRepository(database))
        interrupted = app.state.jobs.recover()
        app.state.incident_service = IncidentService(
            incidents,
            IncidentInputService(settings.max_description_chars),
            extractor,
            stale_analysis_after=stale_after,
            audit=app.state.audit,
        )
        log_event(
            logger,
            "application started",
            operation="startup",
            outcome="success",
            environment=settings.app_env.value,
            llm_provider=llm.provider,
            llm_model=llm.model,
            attack_mapping_mode=mapper.mode.value,
            taxonomy_version=taxonomy.version,
            auth_enabled=settings.auth_enabled,
            execution_mode=settings.execution_mode.value,
            real_tools=sorted(gateway.registry.real),
            library_entries=len(components.library),
            interrupted_jobs=interrupted,
            embedder=components.library.embedder.name,
            vector_index=components.library.index.name,
        )
        if not settings.auth_enabled:
            log_event(logger, "AUTHENTICATION IS DISABLED: development use only", level=logging.WARNING,
                      operation="startup", outcome="insecure_configuration")  # fmt: skip
        try:
            yield
        finally:
            await app.state.jobs.drain()
            await llm.aclose()
            await components.aclose()
            await gateway.registry.aclose()
            database.dispose()
            log_event(logger, "application stopped", operation="shutdown", outcome="success")

    # Interactive docs reveal the whole API surface; they are not served in production.
    docs = {} if not production else {"docs_url": None, "redoc_url": None, "openapi_url": None}
    app = FastAPI(title=settings.app_name, version=__version__, description=DESCRIPTION, lifespan=lifespan, **docs)
    register_error_handlers(app)
    app.include_router(router)
    app.include_router(evidence_router)
    app.include_router(response_router)
    app.include_router(library_router)
    # The console is static HTML/JS; every piece of data it shows comes from the authenticated API.
    app.mount("/console", StaticFiles(directory=Path(__file__).parent / "web" / "static", html=True), name="console")
    app.openapi = lambda: _openapi_schema(app)  # type: ignore[method-assign]
    return app


def _openapi_schema(app: FastAPI) -> dict:
    """Add IncidentSubmission to the docs (the create route reads the raw body to support text/plain)."""
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(title=app.title, version=app.version, description=app.description, routes=app.routes)
    components = schema.setdefault("components", {}).setdefault("schemas", {})
    submission = IncidentSubmission.model_json_schema(ref_template="#/components/schemas/{model}")
    components.update(submission.pop("$defs", {}))
    components["IncidentSubmission"] = submission
    app.openapi_schema = schema
    return schema


app = create_app()
