"""Builds the evidence profile from the model's answer, enforcing the observed/inferred split.

* An "observed" item survives only if its excerpt is really in the incident text; otherwise
  it is downgraded to "unconfirmed" (never silently kept as a fact).
* Event records of known host log sources (Sysmon, Windows Security, PowerShell; see
  ``app.tools.log_sources``) are observed evidence of their category, extracted by pattern.
* "Missing" items are things the input does not contain -- not proof they do not exist. A
  missing item is a gap the incident *states* (provenance ``incident_text``, excerpt = the whole
  statement) only when its quote lies in a sentence of the input saying evidence is not included
  and that sentence names this evidence; any other missing item is a gap the model *inferred*
  (``llm_inference``).
* ``not_checked`` lists categories the taxonomy marks as relevant for the intent that the
  analysis did not touch at all.
"""

from __future__ import annotations

import re
from typing import Optional

from app.intent.grounding import GroundingCorpus, WarningCollector, canonical_text
from app.intent.schemas import LLMExtractionOutput
from app.schemas.enums import Criticality, EvidenceCategory, EvidenceStatus, IncidentSource, ProvenanceOrigin
from app.schemas.incident import EvidenceItem, EvidenceProfile, Provenance
from app.schemas.taxonomy import IntentDefinition
from app.tools.log_sources import LogSourceMap, default_log_sources

# Phrasing with which incident reports say that some evidence is absent from the submitted data.
_STATED_ABSENCE_RE = re.compile(
    r"\b(?:does|do|did)(?: not|n't) (?:include|contain|have)\b"
    r"|\bnot (?:been )?(?:provided|available|included|collected|attached|supplied)\b"
    r"|\b(?:is|are|was|were) missing\b"
    r"|\bno (?:\w+ ){0,3}(?:was|were) (?:provided|collected|attached|available|supplied)\b",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

# Words that name each evidence category in an incident's own statement of what it lacks.
_CATEGORY_TERMS: dict[EvidenceCategory, tuple[str, ...]] = {
    EvidenceCategory.PROCESS_TELEMETRY: ("process", "parent", "child"),
    EvidenceCategory.COMMAND_LINE: ("command line", "command-line", "commandline", "argument"),
    EvidenceCategory.SCRIPT_EXECUTION: ("script", "powershell"),
    EvidenceCategory.FILE_ACTIVITY: ("file", "directory", "folder", "disk"),
    EvidenceCategory.NETWORK_TRAFFIC: ("network", "traffic", "flow", "connection", "packet", "pcap", "firewall", "proxy"),
    EvidenceCategory.DNS: ("dns", "resolution", "lookup"),
    EvidenceCategory.AUTHENTICATION: ("authentication", "logon", "login", "sign-in", "signin", "credential"),
    EvidenceCategory.USER_ACCOUNT: ("account", "user"),
    EvidenceCategory.EMAIL: ("mail", "message", "header", "attachment", "sender"),
    EvidenceCategory.WEB_SERVER_LOGS: ("web", "http", "access log", "url", "request"),
    EvidenceCategory.MEMORY: ("memory", "ram", "dump"),
    EvidenceCategory.REGISTRY: ("registry", "run key"),
    EvidenceCategory.SCHEDULED_TASKS_SERVICES: ("task", "service", "cron"),
    EvidenceCategory.CLOUD_AUDIT: ("cloud", "audit", "tenant"),
    EvidenceCategory.INDICATORS_OF_COMPROMISE: ("hash", "reputation", "indicator", "ioc", "verdict", "threat intel"),
    EvidenceCategory.SECURITY_ALERT: ("alert", "detection", "antivirus"),
}
# Description words too general to show that a statement is about this evidence.
_GENERIC_WORDS = frozenset({"events", "event", "logs", "data", "information", "evidence", "activity", "details",
                            "records", "record", "full", "related", "other", "include", "included", "provided"})  # fmt: skip
_WORD_RE = re.compile(r"[a-z][a-z-]{3,}")


def stated_absence_sentences(incident_text: str) -> list[str]:
    """Sentences of the input that explicitly say some evidence is not included."""
    sentences = (s.strip() for s in _SENTENCE_SPLIT_RE.split(incident_text))
    return [s for s in sentences if s and _STATED_ABSENCE_RE.search(s)]


def stated_gap_sentence(incident_text: str, excerpt: str, category: EvidenceCategory, description: str) -> Optional[str]:
    """The input's sentence stating that *this* evidence is absent, or ``None``.

    The quoted ``excerpt`` must lie inside a sentence that says evidence is not included, and that
    sentence must name this evidence (a term of its category or a specific word of its
    description). A bare fragment such as "does not include" therefore cannot attribute an
    unrelated gap (e.g. email for a host-telemetry incident) to the incident text.
    """
    needle = canonical_text(excerpt.strip().strip(".…").strip())
    if len(needle) < 3:
        return None
    words = {w for w in _WORD_RE.findall(description.lower()) if w not in _GENERIC_WORDS}
    for sentence in stated_absence_sentences(incident_text):
        canonical = canonical_text(sentence)
        if needle not in canonical:
            continue
        if any(term in canonical for term in _CATEGORY_TERMS.get(category, ())) or any(
                re.search(rf"\b{re.escape(w)}\b", canonical) for w in words):  # fmt: skip
            return sentence
    return None


def build_evidence_profile(
    output: LLMExtractionOutput,
    incident_text: str,
    incident_source: IncidentSource,
    intent: IntentDefinition,
    warnings: WarningCollector,
    log_sources: Optional[LogSourceMap] = None,
) -> EvidenceProfile:
    corpus = GroundingCorpus(incident_text)
    llm_provenance = Provenance(origin=ProvenanceOrigin.LLM_INFERENCE)
    seen: set[str] = set()

    def first_time(item_id: str, description: str) -> bool:
        if item_id in seen:
            warnings.degrade(f"dropped duplicate or contradictory evidence item '{description[:60]}'")
            return False
        seen.add(item_id)
        return True

    observed: list[EvidenceItem] = []
    unconfirmed: list[EvidenceItem] = []
    missing: list[EvidenceItem] = []
    stated_gap_excerpts: list[str] = []

    for ev in output.observed_evidence:
        evidence_id = EvidenceItem.make_id(ev.category, ev.description)
        if not first_time(evidence_id, ev.description):
            continue
        if corpus.contains_excerpt(ev.excerpt):
            observed.append(
                EvidenceItem(
                    evidence_id=evidence_id,
                    category=ev.category,
                    description=ev.description,
                    status=EvidenceStatus.OBSERVED,
                    provenance=Provenance(
                        origin=ProvenanceOrigin.INCIDENT_TEXT,
                        source_ref=incident_source.value,
                        excerpt=ev.excerpt.strip(),
                    ),
                )
            )
        else:
            warnings.degrade(
                f"evidence '{ev.description[:60]}' was reported as observed but its excerpt is not in the "
                "incident text; downgraded to unconfirmed"
            )
            unconfirmed.append(
                EvidenceItem(
                    evidence_id=evidence_id,
                    category=ev.category,
                    description=ev.description,
                    status=EvidenceStatus.INFERRED,
                    rationale="model claimed this was observed, but the quoted excerpt could not be verified",
                    provenance=llm_provenance,
                )
            )

    # Event records of known host log sources are observed evidence of their category, whatever
    # the model said (deterministic, grounded in the matched text).
    llm_observed = {item.category for item in observed}
    pattern_observed = 0
    for match in (log_sources or default_log_sources()).find(incident_text):
        if match.category in llm_observed:
            continue
        description = f"{match.source} event {match.event_code} in the incident input"
        evidence_id = EvidenceItem.make_id(match.category, description)
        if evidence_id in seen:
            continue
        seen.add(evidence_id)
        pattern_observed += 1
        observed.append(
            EvidenceItem(
                evidence_id=evidence_id,
                category=match.category,
                description=description,
                status=EvidenceStatus.OBSERVED,
                provenance=Provenance(
                    origin=ProvenanceOrigin.PATTERN_EXTRACTION,
                    source_ref=f"log_source:{match.source}",
                    excerpt=match.excerpt,
                ),
            )
        )

    # Observed facts take precedence: a "missing" item identical to an observed one is dropped.
    for ev in output.missing_evidence:
        evidence_id = EvidenceItem.make_id(ev.category, ev.description)
        if not first_time(evidence_id, ev.description):
            continue
        provenance = llm_provenance
        excerpt = ev.excerpt.strip()
        sentence = stated_gap_sentence(incident_text, excerpt, ev.category, ev.description) if excerpt else None
        if sentence is not None:
            # The incident itself says this evidence is not included (the whole statement is kept).
            provenance = Provenance(
                origin=ProvenanceOrigin.INCIDENT_TEXT, source_ref=incident_source.value, excerpt=sentence
            )
            stated_gap_excerpts.append(sentence)
        elif excerpt and corpus.contains_excerpt(excerpt):
            warnings.info(
                f"missing evidence '{ev.description[:60]}' quotes the input, but no statement there names this "
                "evidence as missing; recorded as a model inference"
            )
        elif excerpt:
            # Not a reason to downgrade the analysis: the gap is still a legitimate inference.
            warnings.info(
                f"missing evidence '{ev.description[:60]}' quotes an excerpt that could not be verified; "
                "recorded as a model inference"
            )
        missing.append(
            EvidenceItem(
                evidence_id=evidence_id,
                category=ev.category,
                description=ev.description,
                status=EvidenceStatus.MISSING,
                criticality=Criticality.CRITICAL if ev.critical else Criticality.SUPPORTING,
                rationale=ev.rationale or None,
                provenance=provenance,
            )
        )

    for ev in output.unconfirmed_evidence:
        evidence_id = EvidenceItem.make_id(ev.category, ev.description)
        if first_time(evidence_id, ev.description):
            unconfirmed.append(
                EvidenceItem(
                    evidence_id=evidence_id,
                    category=ev.category,
                    description=ev.description,
                    status=EvidenceStatus.INFERRED,
                    rationale=ev.rationale or None,
                    provenance=llm_provenance,
                )
            )

    _warn_about_unrecorded_stated_gaps(incident_text, stated_gap_excerpts, warnings)

    touched = {item.category for item in observed + unconfirmed + missing}
    not_checked = [category for category in intent.relevant_evidence if category not in touched]

    sources: list[str] = []
    if len(observed) > pattern_observed or stated_gap_excerpts:
        sources.append(f"{ProvenanceOrigin.INCIDENT_TEXT.value}:{incident_source.value}")
    if pattern_observed:
        sources.append(f"{ProvenanceOrigin.PATTERN_EXTRACTION.value}:log_source")
    if unconfirmed or len(stated_gap_excerpts) < len(missing):
        sources.append(ProvenanceOrigin.LLM_INFERENCE.value)

    return EvidenceProfile(
        observed=observed,
        unconfirmed=unconfirmed,
        missing=missing,
        not_checked=not_checked,
        sources=sources,
    )


def _warn_about_unrecorded_stated_gaps(incident_text: str, gap_excerpts: list[str], warnings: WarningCollector) -> None:
    """Flag input sentences that state an evidence gap which no missing item quotes.

    Informational only: deciding *which* evidence the sentence lists, and its category and
    criticality, is analysis work, so nothing is created on the model's behalf.
    """
    quoted = [canonical_text(e) for e in gap_excerpts]
    for sentence in stated_absence_sentences(incident_text):
        canonical = canonical_text(sentence)
        if not any(q in canonical or canonical in q for q in quoted):
            clipped = sentence if len(sentence) <= 80 else sentence[:77] + "..."
            warnings.info(
                f"the incident text states that evidence is not included ('{clipped}'), but no missing-evidence "
                "item quotes that statement"
            )
