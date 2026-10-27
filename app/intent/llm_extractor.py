"""LLM-based intent extraction and evidence profiling.

Pipeline for one incident::

    [two-stage] intent prompt -> LLM -> activity, ATT&CK tactic, label  (tactic-consistent; paper 968 Section 3)
    prompt -> LLM -> JSON parse -> schema validation  (retry on failure, bounded; label fixed in two-stage mode)
           -> entity grounding -> evidence grounding -> ATT&CK validation -> CanonicalIncident

Both calls see ATT&CK hints: technique names found in the text and well-known artifacts of
techniques (``app/data/attack_artifact_hints.json``). Hints only steer the model; techniques in the
result are still the model's proposals, validated against the dataset.

Malformed model output is never "repaired" into a successful analysis: if no attempt yields
valid JSON matching the schema, :class:`ExtractionFailedError` is raised.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional

from pydantic import ValidationError

from app.core.errors import CADEError, ExtractionFailedError, LLMServerError
from app.core.logging import get_logger, log_event
from app.intent.attack_mapper import ArtifactMatch, AttackMapper
from app.intent.entity_extractor import build_entities
from app.intent.evidence_profiler import build_evidence_profile
from app.intent.grounding import WarningCollector
from app.intent.llm_client import LLMClient
from app.intent.prompts import (
    INTENT_SYSTEM_PROMPT,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_intent_prompt,
    build_retry_prompt,
    build_user_prompt,
    intents_specific_first,
    render_incident,
)
from app.intent.schemas import (
    IntentDecision,
    LLMExtractionOutput,
    LLMTechnique,
    intent_decision_schema,
    output_json_schema,
)
from app.schemas.enums import ExtractionStatus, IncidentStatus
from app.schemas.incident import AnalysisMetadata, CanonicalIncident, Intent, NormalizedIncident
from app.schemas.taxonomy import ATTACK_TACTICS, IntentTaxonomy

logger = get_logger("intent")

LOW_CONFIDENCE_THRESHOLD = 0.5
# After a model-server failure the same prompt at temperature 0 would fail the same way.
RETRY_TEMPERATURE = 0.2


class OutputProblem(Exception):
    """The model's answer was unusable. The message is safe to log (no model output)."""


def parse_llm_output(content: str, allowed_labels: list[str]) -> LLMExtractionOutput:
    text = content.strip()
    if text.startswith("```"):  # tolerate a markdown fence around otherwise valid JSON
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0].strip()
    if not text:
        raise OutputProblem("the response was empty")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise OutputProblem(f"the response is not valid JSON ({exc.msg} at position {exc.pos})") from None
    if not isinstance(data, dict):
        raise OutputProblem("the top-level JSON value must be an object")
    try:
        output = LLMExtractionOutput.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or 'root'}: {err['msg']}"
            for err in exc.errors(include_url=False, include_input=False)[:5]
        )
        raise OutputProblem(f"schema validation failed: {problems}") from None
    if output.intent_label not in allowed_labels:
        raise OutputProblem(f"intent_label '{output.intent_label[:40]}' is not one of the allowed labels")
    return output


class IntentExtractor:
    def __init__(
        self,
        llm: LLMClient,
        taxonomy: IntentTaxonomy,
        attack_mapper: AttackMapper,
        max_attempts: int = 2,
        attack_prompt_hints: bool = True,
        two_stage: bool = False,
    ) -> None:
        self.llm = llm
        self.taxonomy = taxonomy
        self.attack_mapper = attack_mapper
        self.max_attempts = max_attempts
        # Offer dataset technique names / well-known artifacts found in the text as hints (never as results).
        self.attack_prompt_hints = attack_prompt_hints
        # Paper 968 Section 3: intent extraction into a canonical label, then evidence extraction.
        self.two_stage = two_stage
        self._labels = [i.label for i in intents_specific_first(taxonomy)]
        self._schema = output_json_schema(self._labels)

    async def extract(self, incident: NormalizedIncident) -> CanonicalIncident:
        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()
        rendered = render_incident(incident)
        artifacts = self.attack_mapper.artifact_matches(rendered) if self.attack_prompt_hints else []
        names = self.attack_mapper.technique_hints(rendered) if self.attack_prompt_hints else []
        hints = names + [(m.technique_id, m.name) for m in artifacts if m.technique_id not in {t for t, _ in names}]
        llm_ms = 0.0
        notes: list[str] = []

        decision: Optional[IntentDecision] = None
        if self.two_stage:
            try:
                decision, llm_ms = await self._decide_intent(incident, artifacts, names, notes)
            except CADEError as exc:
                exc.attempts = 1
                exc.incident_id = incident.incident_id
                raise
        schema = self._schema if decision is None else output_json_schema([decision.intent_label])
        base_prompt = build_user_prompt(incident, self.taxonomy, hints, decision)
        prompt = base_prompt
        problems: list[str] = []

        output = None
        attempt = 0
        temperature: Optional[float] = None  # the client's configured temperature
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = await self.llm.generate_json(
                    system_prompt=SYSTEM_PROMPT, user_prompt=prompt, json_schema=schema, temperature=temperature
                )
            except LLMServerError as exc:
                if attempt < self.max_attempts:  # e.g. constrained decoding hit a dead end: try once more
                    problems.append(f"attempt {attempt}: the model server failed while generating")
                    temperature = RETRY_TEMPERATURE
                    continue
                exc.attempts = attempt
                exc.incident_id = incident.incident_id
                raise
            except CADEError as exc:
                exc.attempts = attempt
                exc.incident_id = incident.incident_id
                raise
            llm_ms += response.duration_ms
            try:
                output = parse_llm_output(response.content, self.taxonomy.labels)
                break
            except OutputProblem as problem:
                problems.append(f"attempt {attempt}: {problem}")
                log_event(
                    logger,
                    "LLM output rejected",
                    level=logging.WARNING,
                    incident_id=incident.incident_id,
                    operation="analyze",
                    stage="llm_output_validation",
                    outcome="retry" if attempt < self.max_attempts else "failure",
                    error_category="malformed_llm_output",
                    attempt=attempt,
                    output_chars=len(response.content),
                )
                prompt = build_retry_prompt(base_prompt, str(problem))

        if output is None:
            error = ExtractionFailedError(
                f"the model did not return valid structured output after {attempt} attempt(s)",
                incident_id=incident.incident_id,
                details=[{"problem": p} for p in problems],
            )
            error.attempts = attempt
            error.warnings = problems
            raise error

        if decision is not None:  # stage 1 owns the intent; stage 2 was constrained to that label
            techniques = list(output.attack_techniques)
            grounding = decision.technique_id.strip().upper()
            if grounding not in ("", "NONE") and grounding not in {t.technique_id.strip().upper() for t in techniques}:
                # the technique that grounds the intent is a model proposal too (validated like the others)
                techniques.append(LLMTechnique(technique_id=grounding, confidence=decision.intent_confidence,
                                               justification=f"intent stage: {decision.activity}"[:1000]))  # fmt: skip
            output = output.model_copy(update={"intent_label": decision.intent_label, "attack_techniques": techniques,
                                               "intent_confidence": decision.intent_confidence})  # fmt: skip
        try:
            return self._build_canonical(incident, output, started_at, started, attempt, llm_ms, decision, notes)
        except ValidationError as exc:
            error = ExtractionFailedError(
                "the model output could not be converted into a valid canonical incident",
                incident_id=incident.incident_id,
                details=[{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors(include_input=False)[:5]],
            )
            error.attempts = attempt
            raise error from None

    async def _decide_intent(self, incident: NormalizedIncident, artifacts: list[ArtifactMatch],
                             names: list[tuple[str, str]], notes: list[str]) -> tuple[Optional[IntentDecision], float]:  # fmt: skip
        """Stage 1: activity -> ATT&CK tactic -> intent label, kept consistent with ATT&CK.

        * With artifact hints, the tactic is chosen among the hinted techniques' tactics (or "other",
          which reopens the full list), so a known artifact grounds the decision.
        * A label whose tactics share nothing with the chosen tactic or the hinted tactics is
          re-chosen among the labels that cover the tactic.
        * If stage 1 yields no usable answer, ``None`` is returned and the analysis falls back to a
          single call (noted in the warnings); transport errors propagate as usual.
        """
        all_tactics = sorted(ATTACK_TACTICS) + ["none", "other"]
        hinted = sorted({t for m in artifacts for t in m.tactics})
        llm_ms = 0.0

        techniques = [m.technique_id for m in artifacts]

        async def ask(labels: list[str], tactics: list[str]) -> Optional[IntentDecision]:
            nonlocal llm_ms
            prompt = build_intent_prompt(incident, self.taxonomy, artifacts, names, labels)
            schema = intent_decision_schema(labels, tactics, techniques)
            temperature: Optional[float] = None
            for _ in range(self.max_attempts):
                try:
                    response = await self.llm.generate_json(system_prompt=INTENT_SYSTEM_PROMPT, user_prompt=prompt,
                                                            json_schema=schema, temperature=temperature)  # fmt: skip
                except LLMServerError:
                    temperature = RETRY_TEMPERATURE  # a generation dead end: retry, then fall back
                    continue
                llm_ms += response.duration_ms
                try:
                    decision = IntentDecision.model_validate(json.loads(response.content))
                except (ValueError, ValidationError):
                    continue
                if decision.intent_label in labels:
                    return decision
            return None

        decision = await ask(self._labels, hinted + ["other"] if hinted else all_tactics)
        if decision is not None and decision.tactic == "other":
            decision = await ask(self._labels, all_tactics)
        if decision is None:
            notes.append("intent stage gave no usable answer; intent chosen in a single analysis call")
            return None, llm_ms

        label_tactics = set(self.taxonomy.get(decision.intent_label).tactics)
        fitting = [label for label in self._labels if decision.tactic in self.taxonomy.get(label).tactics]
        if (decision.tactic in ATTACK_TACTICS and fitting and decision.tactic not in label_tactics
                and not label_tactics & set(hinted)):  # fmt: skip
            original = decision.intent_label
            if len(fitting) == 1:
                decision = decision.model_copy(update={"intent_label": fitting[0]})
            else:
                retry = await ask(fitting, [decision.tactic])
                decision = retry or decision.model_copy(update={"intent_label": fitting[0]})
            notes.append(f"intent '{original}' does not cover ATT&CK tactic '{decision.tactic}'; "
                         f"re-chosen as '{decision.intent_label}'")  # fmt: skip
        log_event(logger, "intent decided", incident_id=incident.incident_id, operation="analyze", stage="intent",
                  outcome="success", tactic=decision.tactic, artifact_hints=len(artifacts))  # fmt: skip
        return decision, llm_ms

    def _build_canonical(
        self,
        incident: NormalizedIncident,
        output: LLMExtractionOutput,
        started_at: datetime,
        started: float,
        attempts: int,
        llm_ms: float,
        decision: Optional[IntentDecision] = None,
        notes: Optional[list[str]] = None,
    ) -> CanonicalIncident:
        warnings = WarningCollector()
        if self.llm.provider == "mock":
            warnings.info("MOCK analysis: produced by the offline mock LLM client, not by a language model")
        for message in incident.normalization_warnings:
            warnings.info(f"normalization: {message}")
        for message in notes or ():
            warnings.info(message)

        intent_def = self.taxonomy.get(output.intent_label)
        assert intent_def is not None  # guaranteed by parse_llm_output
        if output.intent_confidence < LOW_CONFIDENCE_THRESHOLD:
            warnings.info(f"low intent confidence ({output.intent_confidence:.2f}); treat the intent as tentative")

        incident_text = render_incident(incident)
        entities = build_entities(incident.observables, output.entities, incident_text, warnings)
        evidence = build_evidence_profile(output, incident_text, incident.source, intent_def, warnings)
        techniques = self.attack_mapper.map_candidates(output.attack_techniques, warnings)
        log_event(
            logger,
            "ATT&CK candidates mapped",
            incident_id=incident.incident_id,
            operation="analyze",
            stage="attack_mapping",
            outcome="success",
            candidates_proposed=len(output.attack_techniques),
            candidates_kept=len(techniques),
        )

        return CanonicalIncident(
            incident_id=incident.incident_id,
            source=incident.source,
            title=incident.title,
            severity=incident.severity,
            received_at=incident.received_at,
            occurred_at=incident.occurred_at,
            raw_description=incident.raw_description,
            normalized_description=incident.normalized_description,
            status=IncidentStatus.ANALYZED,
            summary=output.summary.strip(),
            intent=Intent(
                label=intent_def.label,
                description=intent_def.description,
                confidence=output.intent_confidence,
                attack_techniques=techniques,
                tactic=decision.tactic if decision else None,
                rationale=(decision.activity.strip()[:1000] or None) if decision else None,
            ),
            entities=entities,
            evidence_profile=evidence,
            uncertainties=[u.strip() for u in output.uncertainties if u.strip()],
            conflicts=[c.strip() for c in output.conflicts if c.strip()],
            analysis=AnalysisMetadata(
                extraction_status=ExtractionStatus.PARTIAL if warnings.degraded else ExtractionStatus.SUCCESS,
                warnings=warnings.messages,
                provider=self.llm.provider,
                model_name=self.llm.model,
                prompt_version=PROMPT_VERSION,
                attempts=attempts,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                duration_ms=(time.perf_counter() - started) * 1000,
                llm_duration_ms=llm_ms,
                attack_mapping_mode=self.attack_mapper.mode,
                attack_dataset_version=self.attack_mapper.version,
                attack_candidates_proposed=len(output.attack_techniques),
                taxonomy_version=self.taxonomy.version,
            ),
        )
