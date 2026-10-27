"""Evaluate Part 2 (routing) on the extended, intent-level benchmark: does CADE choose reuse / request evidence /
synthesize as expected, and reuse the right playbook?

Two benchmarks, always run and reported separately (``--benchmark``):

* ``legacy``    -- the original 123 SYNTHETIC scenarios, unchanged. For every intent with a seed playbook,
  library suitability (strong: its seed is kept; weak: its seed is withheld; none: every seed sharing a
  tactic with the intent is withheld) is crossed with evidence availability (full, partial, one critical
  category missing but obtainable, sparse and obtainable, sparse and not obtainable). Known defect: a
  "weak" scenario only withholds the intent's own seed, so for most intents the top candidate belongs to
  another tactic and the scenario is not a weak match (reported per scenario as ``p_star_same_tactic``).
* ``corrected`` -- the same scenarios with the weak-match semantics of paper Section 6.1 enforced ("only a
  playbook for a related technique of the same tactic is kept"): one designated same-tactic playbook is
  kept and every other same-tactic seed withheld. Weak scenarios without a same-tactic playbook, and
  REQUEST scenarios whose kept playbook could not run once the evidence is released, are excluded; every
  change is listed in the output (``changes``).

Expected decisions follow the construction:

* REUSE    -- strong match and its critical evidence available (full or partial);
* REQUEST  -- critical evidence missing but obtainable and a stored playbook could run once released;
* GENERATE -- no suitable playbook, or a weak match whose missing evidence cannot be obtained.

Routing is isolated from Part 1: each incident carries a fixed, correct analysis (injected through the
mock LLM client; no model is called), so only the router is measured. Two protocols:

* default    -- the hand-designed rule (Eq. 1-7) on every scenario once;
* --learned  -- the paper's online protocol (Section 6.4, ``app/evaluation/protocol.py``): 10 random
  orderings (seeds 0-9), a fresh learner each, the rule for the first 20 incidents, then LinUCB (Eq. 8)
  trained on the Eq. 9 reward, compared with the rule over the same orderings.

Execution (Part 3) is not exercised: IC_exec and retention come from the construction-based stand-in in
:func:`outcome` (SIMULATED), and latencies are the paper's Fig. 4b means (ASSUMED).

    python scripts/eval_routing.py --benchmark legacy --learned --out reports/.../legacy.json
    python scripts/eval_routing.py --benchmark corrected --learned --out reports/.../corrected.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import Settings  # noqa: E402
from app.enrichment.availability import AvailabilityProfile  # noqa: E402
from app.evaluation import protocol as P  # noqa: E402
from app.evaluation.report import render  # noqa: E402
from app.ingestion.normalizer import normalize_submission  # noqa: E402
from app.intent.attack_mapper import AttackMapper  # noqa: E402
from app.intent.llm_extractor import IntentExtractor  # noqa: E402
from app.intent.mock_llm_client import MockLLMClient  # noqa: E402
from app.library.embedder import build_embedder  # noqa: E402
from app.library.index import build_index  # noqa: E402
from app.library.library import PlaybookLibrary  # noqa: E402
from app.playbook.validator import PlaybookValidator  # noqa: E402
from app.playbook.vocabulary import ActionDefinition, Vocabulary  # noqa: E402
from app.routing.learning import RoutingLearner  # noqa: E402
from app.routing.router import RetrievalQuery, RetrievedCandidate, Router  # noqa: E402
from app.schemas.evidence import EffectiveEvidenceProfile  # noqa: E402
from app.schemas.incident import CanonicalIncident, IncidentSubmission  # noqa: E402
from app.schemas.routing import RoutingAction, RoutingDecision  # noqa: E402
from app.schemas.taxonomy import IntentTaxonomy  # noqa: E402
from app.storage.database import Database  # noqa: E402
from app.storage.library_repository import LibraryRepository  # noqa: E402

REUSE, REQUEST, GENERATE = RoutingAction.REUSE, RoutingAction.REQUEST_EVIDENCE, RoutingAction.SYNTHESIZE
LABELS = [REUSE.value, REQUEST.value, GENERATE.value]
# (library suitability, evidence availability, expected decision) -- paper Table 2. The paper's
# "no suitable playbook, sparse" cell leaves obtainability open; it is built unobtainable here, since
# with obtainable evidence requesting before synthesis would be an equally defensible answer (the paper
# excludes ambiguous combinations).
GRID = [
    ("strong", "full", REUSE), ("strong", "partial", REUSE), ("strong", "critical_missing", REQUEST),
    ("weak", "critical_missing", REQUEST), ("weak", "sparse_obtainable", REQUEST),
    ("none", "full", GENERATE), ("none", "partial", GENERATE), ("none", "sparse_unobtainable", GENERATE),
    ("weak", "sparse_unobtainable", GENERATE),
]  # fmt: skip
RETAIN_COVERAGE = 0.8  # the paper's retention rule: IC_exec >= 0.8 (Section 6.2)


@dataclass
class FilteredSource:
    """The library with some entries withheld (library suitability of a scenario)."""

    library: PlaybookLibrary
    withheld: frozenset[str]

    def retrieve(self, query: RetrievalQuery, top_n: int) -> list[RetrievedCandidate]:
        hits = self.library.retrieve(query, top_n + len(self.withheld))
        return [c for c in hits if c.entry_id not in self.withheld][:top_n]


@dataclass
class Scenario:
    intent: str
    suitability: str
    evidence: str
    expected: RoutingAction
    seed_id: str
    withheld: frozenset[str]
    available: set
    missing: list
    unobtainable: set
    critical: frozenset  # R_c of the scenario's own playbook (the investigation's critical log sources)
    hidden_obtainable: set  # hidden categories (critical or supporting) released by an evidence request
    incident: CanonicalIncident
    related_id: Optional[str] = None  # corrected benchmark: the one same-tactic playbook kept in a weak cell
    released: dict = field(default_factory=dict)  # frozenset(released categories) -> analysed incident

    @property
    def name(self) -> str:
        return f"{self.intent}/{self.suitability}/{self.evidence}"

    @property
    def cell(self) -> str:
        return f"{self.suitability}/{self.evidence}"

    @property
    def obtainable(self) -> set:
        """Hidden categories released by an evidence request (paper Section 6.1)."""
        return self.hidden_obtainable


def evidence_variant(kind: str, critical: list, supporting: list, availability: dict, tau: float):
    """(available categories, missing critical, unobtainable) for an evidence condition, or None if impossible."""
    obtainable = [c for c in critical if availability.get(c, 0.0) >= tau]
    if kind == "full":
        return set(critical) | set(supporting), [], set()
    if kind == "partial":
        if not supporting:
            return None
        return set(critical) | set(supporting[: len(supporting) // 2]), [], set()
    if kind == "critical_missing":
        if not obtainable:
            return None
        gone = obtainable[0]
        return (set(critical) - {gone}) | set(supporting), [gone], set()
    if kind in ("sparse_obtainable", "sparse_unobtainable"):
        if len(critical) < 2 or (kind == "sparse_obtainable" and len(obtainable) < 1):
            return None
        keep = next((c for c in critical if c not in obtainable), critical[0]) if kind == "sparse_obtainable" else critical[0]
        missing = [c for c in critical if c != keep]
        if kind == "sparse_obtainable" and not all(availability.get(c, 0.0) >= tau for c in missing):
            return None  # "obtainable on request": every hidden critical category can be released
        return {keep}, missing, (set(missing) if kind == "sparse_unobtainable" else set())
    raise ValueError(kind)


async def analysed_incident(intent: str, available: set, missing: list, taxonomy: IntentTaxonomy):
    """A fixed, correct analysis for the scenario (routing is evaluated in isolation from Part 1)."""
    observed = [{"category": c.value, "description": f"{c.value} available", "excerpt": f"Evidence available: {c.value}"}
                for c in sorted(available, key=lambda c: c.value)]  # fmt: skip
    text = "SYNTHETIC routing scenario. " + ". ".join(o["excerpt"] for o in observed) + "."
    output = {"intent_label": intent, "intent_confidence": 0.9, "summary": f"SYNTHETIC {intent} scenario.",
              "entities": {"hosts": [], "users": [], "ips": [], "domains": [], "file_hashes": [], "processes": []},
              "observed_evidence": observed, "unconfirmed_evidence": [], "attack_techniques": [],
              "missing_evidence": [{"category": c.value, "description": f"{c.value} needed", "critical": True,
                                    "rationale": "required by the investigation"} for c in missing],
              "uncertainties": [], "conflicts": []}  # fmt: skip
    incident = normalize_submission(IncidentSubmission(description=text), max_description_chars=50_000)
    extractor = IntentExtractor(MockLLMClient([output]), taxonomy, AttackMapper(None), attack_prompt_hints=False)
    return await extractor.extract(incident)


def evidence_of(incident: CanonicalIncident) -> EffectiveEvidenceProfile:
    return EffectiveEvidenceProfile(incident_id=incident.incident_id, analysis_id=1, base=incident.evidence_profile)


# --- outcome model (stands in for execution telemetry until Part 3) ---------------------------------


def returns_data(action: ActionDefinition, in_hand: set, critical: frozenset) -> bool:
    """A step returns events when its critical inputs are all in hand and, if it reads only supporting
    categories, at least one of them is (steps reading no evidence category always return data)."""
    if not action.consumes:
        return True
    needed = {u.category for u in action.consumes if u.critical or u.category in critical}
    if needed:
        return needed <= in_hand
    return bool(action.categories & in_hand)


def execution_coverage(actions: list[ActionDefinition], intent: str, in_hand: set, vocabulary: Vocabulary,
                       critical: frozenset) -> float:  # fmt: skip
    """IC_exec: fraction of the intent's goals addressed by a non-destructive step that returns data."""
    goals = vocabulary.goals_for(intent)
    working = [a for a in actions if not a.destructive and returns_data(a, in_hand, critical)]
    return sum(1 for g in goals if any(vocabulary.addresses(a, g) for a in working)) / len(goals) if goals else 1.0


def outcome(action: RoutingAction, decision: RoutingDecision, scenario: Scenario, in_hand: set, env: "Env") -> tuple[float, bool]:
    """(IC_exec, retained) of a terminal action -- the SIMULATED stand-in for execution telemetry (paper Eq. 9).

    It encodes the benchmark's construction (Sections 6.1-6.2) rather than guessing tool output:

    * reuse executes the top retrieved playbook p*. Its queries return events attributable to the
      technique only when p* investigates the incident's tactic (strong or weak match; a playbook for
      another tactic does not) and its critical evidence R_c(p*) is in hand;
    * synthesis builds a playbook for the intent, which can answer the critical goals only when the
      intent's critical evidence is in hand; its IC_exec is the best any vocabulary action set reaches
      on that evidence (an optimistic oracle for CADE's coverage-driven search);
    * a request releases the hidden categories that are obtainable (Section 6.1).
    Retention follows the paper's rule: critical goals answered and IC_exec >= 0.8 (library and
    synthesized playbooks are validated, so no destructive step is unguarded). Steps that read no
    evidence (triage, verification) count as returning data -- a known difference from the paper's
    "executed query that returns attributable events".
    """
    critical = frozenset(env.taxonomy.get(scenario.intent).critical_evidence)
    if action == REUSE:
        entry = env.library.get(decision.top_candidate.entry_id) if decision.top_candidate else None
        if entry is None:
            return 0.0, False
        actions = [a for a in (env.vocabulary.get(n.action) for n in entry.playbook.nodes) if a]
        suitable = env.same_tactic(entry.intent_label, scenario.intent)
        required = set(env.vocabulary.required_evidence(entry.playbook, critical).critical)
        answers_critical = suitable and required <= in_hand
    else:
        actions = list(env.vocabulary.actions.values())
        answers_critical = scenario.critical <= in_hand
    coverage = execution_coverage(actions, scenario.intent, in_hand, env.vocabulary, critical)
    return coverage, answers_critical and coverage >= RETAIN_COVERAGE


# --- environment --------------------------------------------------------------------------------------


@dataclass
class Env:
    settings: Settings
    taxonomy: IntentTaxonomy
    vocabulary: Vocabulary
    availability: AvailabilityProfile
    library: PlaybookLibrary
    sharpness: float
    latency: P.PathLatency

    def router(self, scenario: Scenario, learner: Optional[RoutingLearner] = None) -> Router:
        s = self.settings
        return Router(FilteredSource(self.library, scenario.withheld), self.vocabulary, self.availability,
                      w=s.routing_w, k=s.routing_k, top_n=s.routing_top_n, tau=s.routing_tau, sharpness=self.sharpness,
                      taxonomy=self.taxonomy, learner=learner,
                      library_size=lambda: len(self.library) - len(scenario.withheld))  # fmt: skip

    def same_tactic(self, a: str, b: str) -> bool:
        return bool(set(self.taxonomy.get(a).tactics) & set(self.taxonomy.get(b).tactics))

    def diagnostics(self, scenario: Scenario, decision: RoutingDecision) -> dict:
        top = decision.top_candidate
        entry = self.library.get(top.entry_id) if top else None
        return {"intent": scenario.intent, "top_candidate": top.name if top else None,
                "top_similarity": round(top.similarity, 4) if top else None,
                "p_star_is_own_seed": bool(top and top.entry_id == scenario.seed_id),
                "p_star_same_tactic": bool(entry and self.same_tactic(entry.intent_label, scenario.intent)),
                "p_star_is_designated_related": bool(top and scenario.related_id and top.entry_id == scenario.related_id),
                "entropy": round(decision.terms.entropy, 4)}  # fmt: skip

    async def respond(self, scenario: Scenario, learner: Optional[RoutingLearner] = None) -> P.Outcome:
        router = self.router(scenario, learner)
        first = router.decide(scenario.incident, evidence_of(scenario.incident), unobtainable=scenario.unobtainable)
        chain, in_hand, terminal = [first], set(scenario.available), first
        if first.action == REQUEST:
            in_hand |= scenario.obtainable
            incident = await released_incident(self, scenario, scenario.obtainable)
            terminal = router.decide(incident, evidence_of(incident), unobtainable=scenario.unobtainable, round_number=2,
                                     allow_request=False)  # fmt: skip
            chain.append(terminal)
        coverage, retained = outcome(terminal.action, terminal, scenario, in_hand, self)
        return P.Outcome(first, terminal, chain, coverage, retained, self.latency.of(first.action, terminal.action),
                         extra=self.diagnostics(scenario, first))  # fmt: skip

    async def alternatives(self, scenario: Scenario) -> dict:
        router = self.router(scenario)
        first = router.decide(scenario.incident, evidence_of(scenario.incident), unobtainable=scenario.unobtainable)
        options: dict = {GENERATE: [(*outcome(GENERATE, first, scenario, set(scenario.available), self),
                                     self.latency.of(GENERATE, GENERATE))]}  # fmt: skip
        if first.top_candidate is not None:
            options[REUSE] = [(*outcome(REUSE, first, scenario, set(scenario.available), self), self.latency.of(REUSE, REUSE))]
        if first.requestable or first.rule_action == REQUEST:
            in_hand = set(scenario.available) | scenario.obtainable
            incident = await released_incident(self, scenario, scenario.obtainable)
            second = router.decide(incident, evidence_of(incident), unobtainable=scenario.unobtainable, round_number=2,
                                   allow_request=False)  # fmt: skip
            terminals = [GENERATE] + ([REUSE] if second.top_candidate is not None else [])
            options[REQUEST] = [(*outcome(t, second, scenario, in_hand, self), self.latency.of(REQUEST, t)) for t in terminals]
        return options


def build_env(args) -> tuple[Env, tuple[str, ...], str, str]:
    settings = Settings()
    taxonomy = IntentTaxonomy.load(settings.resolve_path(settings.intent_taxonomy_path))
    vocabulary = Vocabulary.load(settings.resolve_path(settings.action_vocabulary_path),
                                 settings.resolve_path(settings.intent_goals_path), taxonomy)  # fmt: skip
    availability = AvailabilityProfile.load(settings.resolve_path(settings.evidence_availability_path))
    embedder, embedder_note = build_embedder(args.embedder or settings.embedder, settings.embedding_model)
    index, index_note = build_index(settings.vector_index, embedder.dimension)
    fallbacks = P.component_fallbacks(embedder.name, embedder_note, index.name, index_note)
    P.require_real_components(fallbacks, args.allow_fallback)
    db = Database(f"sqlite:///{(Path(tempfile.mkdtemp()) / 'routing_eval.db').as_posix()}")
    db.create_schema()
    library = PlaybookLibrary(LibraryRepository(db), embedder, index, vocabulary,
                              PlaybookValidator(vocabulary, node_cap=settings.node_cap), k=settings.routing_k)  # fmt: skip
    library.seed_if_empty(settings.resolve_path(settings.library_seed_path))
    library.load()
    sharpness = args.sharpness if args.sharpness is not None else settings.routing_similarity_sharpness
    latency = P.PathLatency(additive_request=args.request_latency == "additive")
    print(f"embedder={embedder.name} index={index.name} fallbacks={list(fallbacks) or 'none'} library={len(library)} "
          f"w={settings.routing_w} k={settings.routing_k} N={settings.routing_top_n} tau={settings.routing_tau} "
          f"sharpness={sharpness} request-latency={args.request_latency}\n")  # fmt: skip
    return Env(settings, taxonomy, vocabulary, availability, library, sharpness, latency), fallbacks, embedder.name, index.name


# --- scenarios ----------------------------------------------------------------------------------------


def designated_related(env: Env, intent: str, seed, siblings: list, critical: frozenset):
    """The same-tactic playbook a corrected weak cell keeps: the largest overlap of required evidence."""
    own = env.vocabulary.required_evidence(seed.playbook, critical)
    own_set = set(own.critical) | set(own.supporting)

    def overlap(entry):
        req = env.vocabulary.required_evidence(entry.playbook, critical)
        theirs = set(req.critical) | set(req.supporting)
        return len(own_set & theirs) / len(own_set | theirs) if own_set | theirs else 0.0

    return sorted(siblings, key=lambda e: (-overlap(e), e.intent_label))[0]


async def build_scenarios(env: Env, only: Optional[list[str]], variant: str) -> tuple[list[Scenario], list[dict]]:
    """The legacy scenarios, or (``corrected``) the same with weak-match semantics enforced; plus the change log."""
    library, taxonomy, vocabulary = env.library, env.taxonomy, env.vocabulary
    seeds = {e.intent_label: e for e in library.entries()}
    scenarios, changes = [], []
    for intent, seed in sorted(seeds.items()):
        if intent == "unknown" or (only and intent not in only):
            continue
        definition = taxonomy.get(intent)
        critical_of_intent = frozenset(definition.critical_evidence)
        required = vocabulary.required_evidence(seed.playbook, critical_of_intent)
        same_tactic = {e.entry_id for e in library.entries() if set(taxonomy.get(e.intent_label).tactics) & set(definition.tactics)}
        withheld = {"strong": frozenset(), "weak": frozenset({seed.entry_id}), "none": frozenset(same_tactic | {seed.entry_id})}
        related = None
        if variant == "corrected":
            siblings = [e for e in library.entries() if e.entry_id in same_tactic and e.entry_id != seed.entry_id]
            if siblings:
                related = designated_related(env, intent, seed, siblings, critical_of_intent)
                withheld["weak"] = frozenset((same_tactic | {seed.entry_id}) - {related.entry_id})
        for suitability, evidence_kind, expected in GRID:
            variant_ev = evidence_variant(evidence_kind, list(required.critical), list(required.supporting),
                                          env.availability.availability, env.settings.routing_tau)  # fmt: skip
            if variant_ev is None:
                continue
            available, missing, unobtainable = variant_ev
            hidden = (set(required.critical) | set(required.supporting)) - available
            obtainable = {c for c in hidden - unobtainable if env.availability.availability.get(c, 0.0) >= env.settings.routing_tau}
            name = f"{intent}/{suitability}/{evidence_kind}"
            if variant == "corrected" and suitability == "weak":
                if related is None:
                    changes.append({"scenario": name, "change": "excluded", "expected": expected.value,
                                    "reason": "no other playbook shares a tactic with the intent, so no weak match can exist "
                                              "(the legacy scenario's top candidate is from another tactic)"})  # fmt: skip
                    continue
                if expected == REQUEST:
                    needs = set(vocabulary.required_evidence(related.playbook, critical_of_intent).critical)
                    if not needs <= available | obtainable:
                        changes.append({"scenario": name, "change": "excluded", "expected": expected.value,
                                        "related": related.intent_label,
                                        "reason": "the kept same-tactic playbook could not run once the evidence is released "
                                                  f"(needs {sorted(c.value for c in needs - (available | obtainable))}); the "
                                                  "label rule would say GENERATE, but requesting the obtainable critical "
                                                  "evidence first is equally defensible, so it is ambiguous (Section 6.1)"})  # fmt: skip
                        continue
                changes.append({"scenario": name, "change": "library_changed", "expected": expected.value,
                                "related": related.intent_label,
                                "legacy_withheld": sorted(seeds_by_id(library)[i] for i in {seed.entry_id}),
                                "corrected_withheld": sorted(seeds_by_id(library)[i] for i in withheld["weak"])})  # fmt: skip
            incident = await analysed_incident(intent, available, missing, taxonomy)
            scenarios.append(Scenario(intent, suitability, evidence_kind, expected, seed.entry_id, withheld[suitability],
                                      available, missing, unobtainable, frozenset(required.critical), obtainable,
                                      incident, related.entry_id if (related and suitability == "weak") else None))  # fmt: skip
    return scenarios, changes


def seeds_by_id(library: PlaybookLibrary) -> dict[str, str]:
    return {e.entry_id: e.intent_label for e in library.entries()}


async def released_incident(env: Env, scenario: Scenario, released: set) -> CanonicalIncident:
    """The scenario after an evidence request released ``released`` (same incident id)."""
    key = frozenset(released)
    if key not in scenario.released:
        missing = [c for c in scenario.missing if c not in released]  # critical categories still hidden
        incident = await analysed_incident(scenario.intent, scenario.available | released, missing, env.taxonomy)
        scenario.released[key] = incident.model_copy(update={"incident_id": scenario.incident.incident_id})
    return scenario.released[key]


# --- reporting ----------------------------------------------------------------------------------------


def report(title: str, rows: list[dict]) -> None:
    confusion, by_class = Counter(), defaultdict(lambda: [0, 0])
    for r in rows:
        confusion[(r["expected"], r["got"])] += 1
        by_class[r["expected"]][0] += r["ok"]
        by_class[r["expected"]][1] += 1
    correct = sum(r["ok"] for r in rows)
    print(f"{title}: {correct}/{len(rows)} ({100 * correct / max(len(rows), 1):.1f}%)  (a constant policy scores ~1/3)")
    for label in LABELS:
        hit, n = by_class.get(label, (0, 0))
        print(f"  expected {label:<17} {hit}/{n} ({100 * hit / max(n, 1):.1f}%)")
    print(" " * 18 + "".join(f"{label:>18}" for label in LABELS))
    for row in LABELS:
        print(f"{row:<18}" + "".join(f"{confusion[(row, col)]:>18}" for col in LABELS))


def decision_line(scenario: Scenario, decision: RoutingDecision, ok: bool, right_playbook: bool) -> str:
    t, s, top = decision.terms, decision.scores, decision.top_candidate
    return (f"{scenario.intent:<32} {scenario.suitability:<6} {scenario.evidence:<19} expected={scenario.expected.value:<16} "
            f"got={decision.action.value:<16} {'ok  ' if ok else 'FAIL'} top={top.name[:34] if top else '-':<34} "
            f"sim={top.similarity if top else 0:.3f} Cov={t.coverage:.2f} Cov+={t.counterfactual_coverage:.2f} H={t.entropy:.3f} "
            f"rho={t.requestability:.2f} M={t.unmet_fraction:.2f} U=({s.reuse:.3f},{s.request_evidence:.3f},{s.synthesize:.3f})"
            + ("" if right_playbook or scenario.expected != REUSE else "  WRONG-PLAYBOOK"))  # fmt: skip


async def run_rule(env: Env, scenarios: list[Scenario], verbose: bool) -> list[dict]:
    rows, playbook_hits, playbook_total = [], 0, 0
    for scenario in scenarios:
        decision = env.router(scenario).decide(scenario.incident, evidence_of(scenario.incident),
                                               unobtainable=scenario.unobtainable)  # fmt: skip
        ok = decision.action == scenario.expected
        top = decision.top_candidate
        right_playbook = top is not None and top.entry_id == scenario.seed_id
        if scenario.expected == REUSE:
            playbook_total += 1
            playbook_hits += ok and right_playbook
        if verbose or not ok or (scenario.expected == REUSE and not right_playbook):
            print(decision_line(scenario, decision, ok, right_playbook))
        rows.append({"intent": scenario.intent, "suitability": scenario.suitability, "evidence": scenario.evidence,
                     "expected": scenario.expected.value, "got": decision.action.value, "ok": ok,
                     "top_candidate": top.name if top else None, "right_playbook": right_playbook,
                     **env.diagnostics(scenario, decision),
                     "terms": decision.terms.model_dump(), "scores": decision.scores.model_dump()})  # fmt: skip
    print()
    report("Rule (Eq. 7) routing accuracy", rows)
    print(f"  REUSE with the right playbook: {playbook_hits}/{playbook_total}")
    weak = [r for r in rows if r["suitability"] == "weak"]
    if weak:
        print(f"  weak-match scenarios whose top candidate shares the intent's tactic: "
              f"{sum(r['p_star_same_tactic'] for r in weak)}/{len(weak)}")  # fmt: skip
    return rows


def composition(scenarios: list[Scenario]) -> dict:
    return {"scenarios": len(scenarios), "by_expected": dict(Counter(s.expected.value for s in scenarios)),
            "by_cell": dict(sorted(Counter(s.cell for s in scenarios).items())),
            "intents": len({s.intent for s in scenarios})}  # fmt: skip


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--benchmark", choices=["legacy", "corrected"], default="legacy",
                        help="legacy: the original 123 scenarios; corrected: weak-match semantics enforced")
    parser.add_argument("--embedder", choices=["sentence_transformers", "hashing"], help="override CADE_EMBEDDER")
    parser.add_argument("--allow-fallback", action="store_true", help="run even if a substitute embedder/index loads "
                                                                       "(the run is marked DEGRADED)")
    parser.add_argument("--sharpness", type=float, help="override CADE_ROUTING_SIMILARITY_SHARPNESS (1.0 = Eq. 2)")
    parser.add_argument("--request-latency", choices=["path", "additive"], default="path",
                        help="path: Fig. 4b's 8.4 s is the whole request path (default); additive: the earlier "
                             "evaluation's 8.4 s on top of the terminal path")  # fmt: skip
    parser.add_argument("--only", nargs="*", help="intent labels to evaluate")
    parser.add_argument("--learned", action="store_true", help="also run the online LinUCB protocol (paper Section 6.4)")
    parser.add_argument("--orderings", type=int, default=10, help="random orderings for --learned (paper: 10)")
    parser.add_argument("--verbose", action="store_true", help="print every scenario, not only failures")
    parser.add_argument("--out", type=Path, help="write all decisions and the summary as JSON")
    parser.add_argument("--report", type=Path, help="write the Markdown report (with --learned)")
    parser.add_argument("--state-dir", type=Path, help="keep each ordering's learner database (bandit_<seed>.db) here")
    args = parser.parse_args()

    env, fallbacks, embedder_name, index_name = build_env(args)
    scenarios, changes = await build_scenarios(env, args.only, args.benchmark)
    provenance = P.Provenance(
        benchmark=f"{args.benchmark} extended benchmark (intent-level, SYNTHETIC scenarios)", embedder=embedder_name,
        index=index_name, fallbacks=fallbacks,
        analysis="fixed ground-truth analysis per scenario, injected through MockLLMClient (no model call)",
        data="SYNTHETIC: 14 intents of app/data/intent_taxonomy.json x Table-2 cells; library = 15 seed playbooks",
        latencies=env.latency.describe(),
    )  # fmt: skip
    print(f"benchmark={args.benchmark}: {composition(scenarios)}")
    for change in changes:
        print(f"  {change['change']:<16} {change['scenario']}" + (f"  ({change['reason']})" if "reason" in change else ""))
    print()
    output = {"benchmark": args.benchmark, "provenance": provenance.as_dict(), "composition": composition(scenarios),
              "changes": changes, "rule": await run_rule(env, scenarios, args.verbose)}  # fmt: skip
    if args.learned:
        print()
        workdir = args.state_dir or Path(tempfile.mkdtemp(prefix="cade_bandit_"))
        result = await P.run(env, scenarios, settings=env.settings, orderings=args.orderings, workdir=workdir)
        output["learned"], output["summary"] = result["rows"], result["summary"]
        s = result["summary"]
        for key in ("linucb", "rule"):
            m = s[key]["mean_std"]
            print(f"{key:>6}: accuracy {100 * m['accuracy']['mean']:.1f} +/- {100 * m['accuracy']['std']:.1f}%  "
                  f"macro-F1 {m['macro_f1']['mean']:.3f}  reward {m['reward_total']['mean']:.2f} +/- {m['reward_total']['std']:.2f}  "
                  f"first 20 {100 * m['accuracy_bootstrap']['mean']:.1f}%  after {100 * m['accuracy_after_bootstrap']['mean']:.1f}%")  # fmt: skip
        print(f"relative reward gain {100 * s['relative_reward_gain']['mean_std']['mean']:+.1f} +/- "
              f"{100 * s['relative_reward_gain']['mean_std']['std']:.1f}%; labels not uniquely reward-optimal: "
              f"{sorted(set(s['reward_preference']['not_uniquely_expected_per_ordering'].values()))} of {s['scenarios']}")  # fmt: skip
        if args.report:
            notes = [f"Composition: {composition(scenarios)}."]
            if args.benchmark == "legacy":
                weak = [r for r in output["rule"] if r["suitability"] == "weak"]
                notes.append(f"KNOWN DEFECT (kept for comparability): only {sum(r['p_star_same_tactic'] for r in weak)} of "
                             f"{len(weak)} weak-match scenarios retrieve a same-tactic top candidate; the rest are not weak "
                             f"matches. See the corrected benchmark.")  # fmt: skip
            else:
                kinds = Counter(c["change"] for c in changes)
                notes.append(f"Changes from legacy: {dict(kinds)} (every change is listed in the JSON output).")
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(render(f"Routing: {args.benchmark} extended benchmark", s, provenance.as_dict(), notes=notes),
                                   encoding="utf-8")  # fmt: skip
            print(f"report written to {args.report}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(output, indent=1, default=str), encoding="utf-8")
        print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
