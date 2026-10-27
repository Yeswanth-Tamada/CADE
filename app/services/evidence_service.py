"""Use cases around evidence after analysis: enrichment through tools and analyst-supplied evidence."""

from __future__ import annotations

import time
from typing import Optional

from app.core.errors import NoUsableAnalysisError
from app.core.logging import get_logger, log_event
from app.core.threads import run_sync
from app.enrichment.enricher import EvidenceEnricher, default_categories
from app.schemas.enums import EvidenceAdditionKind, EvidenceCategory, EvidenceStatus, ProvenanceOrigin
from app.schemas.evidence import (
    AnalystEvidenceSubmission,
    EffectiveEvidenceProfile,
    EnrichmentReport,
    EvidenceAddition,
)
from app.schemas.incident import EvidenceItem, Provenance
from app.schemas.records import StoredAnalysis
from app.security.auth import Principal
from app.services.audit_service import AuditService
from app.storage.evidence_repository import EvidenceRepository
from app.storage.repository import IncidentRepository

logger = get_logger("evidence")


class EvidenceService:
    def __init__(
        self,
        incidents: IncidentRepository,
        evidence: EvidenceRepository,
        enricher: EvidenceEnricher,
        audit: AuditService,
    ) -> None:
        self.incidents = incidents
        self.evidence = evidence
        self.enricher = enricher
        self.audit = audit

    async def usable_analysis(self, incident_id: str) -> StoredAnalysis:
        await run_sync(self.incidents.get_incident, incident_id)  # 404 if unknown
        analysis = await run_sync(self.incidents.get_latest_usable_analysis, incident_id)
        if analysis is None or analysis.result is None:
            raise NoUsableAnalysisError("analyse the incident successfully first", incident_id=incident_id)
        return analysis

    async def effective_profile(self, incident_id: str) -> EffectiveEvidenceProfile:
        analysis = await self.usable_analysis(incident_id)
        additions = await run_sync(self.evidence.list, incident_id)
        return EffectiveEvidenceProfile(
            incident_id=incident_id,
            analysis_id=analysis.analysis_id,
            base=analysis.result.evidence_profile,
            tool_observed=[a.item for a in additions if a.kind == EvidenceAdditionKind.TOOL],
            analyst_supplied=[a.item for a in additions if a.kind == EvidenceAdditionKind.ANALYST],
            simulated=[a.simulated for a in additions if a.kind == EvidenceAdditionKind.SIMULATED],
        )

    async def enrich(
        self, incident_id: str, categories: Optional[list[EvidenceCategory]], actor: Optional[Principal]
    ) -> EnrichmentReport:
        started = time.perf_counter()
        analysis = await self.usable_analysis(incident_id)
        profile = await self.effective_profile(incident_id)
        requested = categories or default_categories(analysis.result, profile.observed_categories())
        outcome = await self.enricher.enrich(analysis.result, requested)

        actor_id = actor.analyst_id if actor else None
        for result in outcome.results:
            await run_sync(self.evidence.record_invocation, result, purpose="enrichment", incident_id=incident_id,
                           actor=actor_id)  # fmt: skip
        await run_sync(
            self.evidence.add, incident_id,
            items=[(EvidenceAdditionKind.TOOL, item) for item in outcome.observed],
            simulated=outcome.simulated, created_by=actor_id,
        )  # fmt: skip
        report = outcome.report
        await self.audit.record(actor, "evidence.enrich", outcome="success", target_type="incident",
                                target_id=incident_id,
                                details={"mode": report.mode.value, "calls": len(report.attempts),
                                         "observed": report.added_observed, "simulated": report.added_simulated,
                                         "manual_tasks": len(report.manual_tasks)})  # fmt: skip
        log_event(logger, "enrichment completed", incident_id=incident_id, operation="enrich", stage="enrichment",
                  outcome="success", duration_ms=(time.perf_counter() - started) * 1000, calls=len(report.attempts),
                  observed=report.added_observed, simulated=report.added_simulated)  # fmt: skip
        return report

    async def add_analyst_evidence(
        self, incident_id: str, submission: AnalystEvidenceSubmission, actor: Principal
    ) -> EvidenceAddition:
        await self.usable_analysis(incident_id)
        item = EvidenceItem(
            evidence_id=EvidenceItem.make_id(submission.category, submission.description),
            category=submission.category,
            description=submission.description,
            status=EvidenceStatus.OBSERVED,
            provenance=Provenance(origin=ProvenanceOrigin.ANALYST_SUPPLIED, source_ref=f"analyst:{actor.analyst_id}",
                                  excerpt=submission.excerpt),  # fmt: skip
        )
        added = await run_sync(self.evidence.add, incident_id, items=[(EvidenceAdditionKind.ANALYST, item)],
                               created_by=actor.analyst_id)  # fmt: skip
        await self.audit.record(actor, "evidence.add", outcome="success" if added else "duplicate",
                                target_type="incident", target_id=incident_id,
                                details={"category": submission.category.value})  # fmt: skip
        if not added:
            stored = [a for a in await run_sync(self.evidence.list, incident_id)
                      if a.item is not None and a.item.evidence_id == item.evidence_id]  # fmt: skip
            return stored[0]
        return added[0]
