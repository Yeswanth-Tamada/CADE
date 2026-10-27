"""Evidence enrichment: ask security tools for evidence the analysis found missing.

For each requested evidence category, the enricher picks the catalog's read-only operations
that collect it and can target one of the incident's grounded entities, and calls them through
the :class:`~app.tools.registry.ToolGateway`. It then sorts what came back:

* a **real** tool observation becomes an ``observed`` EvidenceItem with ``provenance=tool``;
* a **simulated** observation becomes a :class:`SimulatedObservation` -- never observed evidence;
* ``no_data`` is reported as "no data returned", never as proof that the evidence does not exist;
* failures become ``error``/``manual_task`` attempts (policy), never simulated stand-ins.
"""

from __future__ import annotations

from app.core.errors import ToolInputError
from app.schemas.enums import EvidenceCategory, EvidenceStatus, ProvenanceOrigin, ToolStatus
from app.schemas.evidence import EnrichmentAttempt, EnrichmentReport, SimulatedObservation
from app.schemas.incident import CanonicalIncident, EvidenceItem, Provenance
from app.schemas.tools import OperationSpec, ToolRequest, ToolResult
from app.tools.artifacts import entity_artifacts
from app.tools.catalog import enrichment_operations
from app.tools.registry import ToolGateway


class EnrichmentOutcome:
    def __init__(self, report: EnrichmentReport, observed: list[EvidenceItem], simulated: list[SimulatedObservation],
                 results: list[ToolResult]) -> None:  # fmt: skip
        self.report = report
        self.observed = observed
        self.simulated = simulated
        self.results = results


class EvidenceEnricher:
    def __init__(self, gateway: ToolGateway, *, max_calls: int = 20, max_values_per_type: int = 3) -> None:
        self.gateway = gateway
        self.max_calls = max_calls
        self.max_values_per_type = max_values_per_type

    def plan(
        self, incident: CanonicalIncident, categories: list[EvidenceCategory]
    ) -> tuple[list[tuple[OperationSpec, str, str]], list[EvidenceCategory]]:
        """``(calls, categories_without_tools)``; each call is (operation, target type, target value)."""
        values = {k: v[: self.max_values_per_type] for k, v in entity_artifacts(incident.entities).items()}
        calls: list[tuple[OperationSpec, str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        without_tools: list[EvidenceCategory] = []
        for category in categories:
            planned_for_category = False
            for spec in enrichment_operations(category):
                if spec.requires:
                    continue
                for target_type in spec.accepts:
                    for value in values.get(target_type, []):
                        planned_for_category = True
                        key = (spec.key, target_type, value)
                        if key not in seen:
                            seen.add(key)
                            calls.append((spec, target_type, value))
            if not planned_for_category:
                without_tools.append(category)
        return calls[: self.max_calls], without_tools

    async def enrich(self, incident: CanonicalIncident, categories: list[EvidenceCategory]) -> EnrichmentOutcome:
        calls, without_tools = self.plan(incident, categories)
        attempts: list[EnrichmentAttempt] = []
        observed: list[EvidenceItem] = []
        simulated: list[SimulatedObservation] = []
        results: list[ToolResult] = []
        manual_tasks: list[str] = []

        for spec, target_type, value in calls:
            request = ToolRequest(operation=spec.name, target_type=target_type, target=value)
            try:
                result = await self.gateway.invoke(spec.tool, request)
            except ToolInputError as exc:
                attempts.append(EnrichmentAttempt(tool=spec.tool, operation=spec.name, target_type=target_type,
                                                  status=ToolStatus.ERROR, synthetic=False, categories=list(spec.categories),
                                                  error_category=exc.category, message=exc.message))  # fmt: skip
                continue
            results.append(result)
            attempts.append(EnrichmentAttempt(tool=result.tool, operation=result.operation, target_type=target_type,
                                              status=result.status, synthetic=result.synthetic,
                                              categories=list(spec.categories), error_category=result.error_category,
                                              message=result.message))  # fmt: skip
            if result.status == ToolStatus.MANUAL_TASK:
                manual_tasks.extend(result.findings)
            if result.status != ToolStatus.SUCCESS:
                continue
            for observation in result.observations:
                if result.synthetic:
                    simulated.append(SimulatedObservation(category=observation.category,
                                                          description=observation.description,
                                                          tool=result.tool, operation=result.operation))  # fmt: skip
                else:
                    observed.append(EvidenceItem(
                        evidence_id=EvidenceItem.make_id(observation.category, observation.description),
                        category=observation.category,
                        description=observation.description,
                        status=EvidenceStatus.OBSERVED,
                        provenance=Provenance(origin=ProvenanceOrigin.TOOL, source_ref=f"{result.tool}.{result.operation}",
                                              excerpt=observation.description),
                    ))  # fmt: skip

        obtained = {i.category for i in observed}
        simulated_categories = {s.category for s in simulated} - obtained
        attempted = {c for a in attempts for c in a.categories if c in categories}
        report = EnrichmentReport(
            incident_id=incident.incident_id,
            mode=self.gateway.mode,
            requested_categories=categories,
            attempts=attempts,
            added_observed=len(observed),
            added_simulated=len(simulated),
            categories_obtained=sorted(obtained & set(categories), key=lambda c: c.value),
            categories_simulated=sorted(simulated_categories & set(categories), key=lambda c: c.value),
            categories_without_data=sorted(attempted - obtained - simulated_categories, key=lambda c: c.value),
            categories_without_tools=without_tools,
            manual_tasks=manual_tasks,
        )
        return EnrichmentOutcome(report, observed, simulated, results)


def default_categories(incident: CanonicalIncident, already_have: set[EvidenceCategory]) -> list[EvidenceCategory]:
    """Missing categories (critical first), then relevant categories nothing has checked yet."""
    profile = incident.evidence_profile
    ordered = [i.category for i in sorted(profile.missing, key=lambda i: i.criticality.value)]  # critical < supporting
    ordered += list(profile.not_checked)
    result: list[EvidenceCategory] = []
    for category in ordered:
        if category not in already_have and category not in result:
            result.append(category)
    return result
