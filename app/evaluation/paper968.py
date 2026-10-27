"""The paper-matched routing benchmark: paper 968 Section 6.1 and Table 2, 96 scenarios = 8 tactics x 12.

Everything is derived deterministically from the frozen catalog ``app/data/paper_benchmark/catalog.json``
(Splunk Attack Data recordings and ATT&CK v20 metadata; ``scripts/build_paper_catalog.py``). Each rule
is a reading of Section 6.1; where the paper is silent the choice is marked *(choice)*.

1. **Tactic of a technique.** A recording's primary technique belongs to one of the eight tactics. A
   technique listed under several of them goes to the tactic with the fewest candidate techniques
   (ties in the paper's order) so that every tactic can reach twelve recordings *(choice)*.
2. **Ontology** ("for each intent, the ontology marks every category as either critical or
   supporting"). One intent per tactic; a category is critical for it when at least half of the
   tactic's candidate techniques can produce it, otherwise supporting *(choice: derived from data,
   decided by the user)*. The categories are the paper's seven host log-source categories.
3. **Library** ("at least one playbook per benchmark technique"): one playbook per technique used,
   built from the action vocabulary to query every category the technique produces across its
   catalogued recordings (its footprint P(t)), plus triage, a Sigma hunt for the technique, a
   verification and a report. R_c / R_s of a playbook are P(t) split by its own intent's ontology
   (the library stores each playbook's required evidence, Section 3).
4. **Library suitability.** strong: everything kept; weak: "only a playbook for a related technique
   of the same tactic is kept" -- every playbook of the tactic is withheld except one designated
   related technique (same parent technique first, then the largest overlap of categories); none:
   every playbook of the tactic is withheld.
5. **Evidence availability** (A starts as the recording's categories C; K = critical, S = supporting):
   full -- nothing hidden; partial -- ceil(|S|/2) supporting categories hidden; critical evidence
   missing -- one critical category hidden; sparse -- only one or two categories remain. Hidden
   categories are released by a request only when marked obtainable. Cell by cell:
   strong/critical-missing and weak/critical-missing hide one critical category, obtainable (weak: one
   the related playbook needs); weak/sparse keeps one category (a supporting one if any) and hides the
   rest, all obtainable or none obtainable; none/sparse keeps the critical categories (one or two) and
   hides every supporting one, not obtainable *(choice: the paper qualifies obtainability only for the
   weak-match sparse cells)*; partial cells' hidden categories are not obtainable. Strong cells require
   the recording to cover its technique's footprint, so "full" means the playbook's evidence is complete.
   Recordings are distinct within a tactic. When no catalogued recording satisfies a cell, two
   relaxations are tried in order and the scenario is flagged (``relaxed``): (A) a strong cell accepts
   a recording that holds the footprint's critical categories only; (B) a recording already used by
   another cell of the tactic is reused.
6. **Labels** by construction: REUSE (strong match, critical evidence available), REQUEST (critical
   evidence missing but obtainable and a stored playbook -- the technique's own or the related one --
   could run once it is released), GENERATE otherwise. 32 per label.
7. **Requestability** rho is the mean obtainability over the missing critical categories of p*, with
   obtainability 1 for hidden categories marked obtainable and 0 otherwise (categories a recording
   never contained cannot be obtained) *(choice)*.
8. **Execution stand-in (SIMULATED).** Ground-truth goals are the recording's critical categories
   (``goals="critical"``, decided by the user) or all its categories (``goals="all"``, sensitivity). A
   reused playbook answers a goal when it queries that category, the category is in hand and the
   playbook investigates the scenario's tactic (a playbook for another tactic finds no events
   attributable to the technique); synthesis answers every goal whose category is in hand (an
   optimistic stand-in for CADE). Retention is the paper's rule: every critical goal answered and
   IC_exec >= 0.8 (validated playbooks: no failed node, no unguarded destructive action).
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from app.core.config import PROJECT_ROOT, Settings
from app.enrichment.availability import AvailabilityProfile
from app.evaluation.protocol import GENERATE, REQUEST, REUSE, Outcome, PathLatency
from app.library.library import PlaybookLibrary
from app.playbook.builder import Draft
from app.playbook.validator import PlaybookValidator, initial_artifacts
from app.playbook.vocabulary import Vocabulary
from app.routing.learning import RoutingLearner
from app.routing.router import RetrievalQuery, RetrievedCandidate, Router
from app.schemas.enums import EvidenceCategory as C
from app.schemas.playbook import Playbook, PlaybookOrigin, RequiredEvidence
from app.schemas.routing import RoutingAction, RoutingDecision

CATALOG = PROJECT_ROOT / "app" / "data" / "paper_benchmark" / "catalog.json"
SEVEN = (C.PROCESS_TELEMETRY, C.NETWORK_TRAFFIC, C.FILE_ACTIVITY, C.REGISTRY, C.AUTHENTICATION, C.SCRIPT_EXECUTION,
         C.SCHEDULED_TASKS_SERVICES)  # fmt: skip
TACTIC_ORDER = ("execution", "persistence", "privilege_escalation", "defense_evasion", "credential_access", "discovery",
                "lateral_movement", "impact")  # fmt: skip
# Table 2: (library suitability, evidence availability, label, scenarios per tactic)
TABLE_2 = (
    ("strong", "full", REUSE, 2), ("strong", "partial", REUSE, 2), ("strong", "critical_missing", REQUEST, 1),
    ("weak", "critical_missing", REQUEST, 2), ("weak", "sparse_obtainable", REQUEST, 1),
    ("none", "full", GENERATE, 1), ("none", "partial", GENERATE, 1), ("none", "sparse", GENERATE, 1),
    ("weak", "sparse_unobtainable", GENERATE, 1),
)  # fmt: skip
FILL_ORDER = ("weak/sparse_obtainable", "weak/sparse_unobtainable", "weak/critical_missing", "none/sparse",
              "strong/partial", "none/partial", "strong/critical_missing", "strong/full", "none/full")  # fmt: skip
QUERY_ACTION = {C.PROCESS_TELEMETRY: "query_process_events", C.NETWORK_TRAFFIC: "query_network_connections",
                C.FILE_ACTIVITY: "review_file_integrity", C.REGISTRY: "query_registry_events",
                C.AUTHENTICATION: "query_host_auth_events", C.SCRIPT_EXECUTION: "query_script_blocks",
                C.SCHEDULED_TASKS_SERVICES: "query_service_task_events"}  # fmt: skip
_order = {c: i for i, c in enumerate(SEVEN)}


def ordered(categories) -> list[C]:
    return sorted(categories, key=_order.__getitem__)


def display(tactic: str) -> str:
    return tactic.replace("_", " ")


@dataclass(frozen=True)
class Recording:
    id: str
    path: str
    technique: str
    sources: tuple[str, ...]
    categories: frozenset[C]
    atomic: bool


@dataclass
class PaperScenario:
    tactic: str
    suitability: str
    evidence: str
    expected: RoutingAction
    slot: int  # 1-based index within the cell (two scenarios share some cells)
    recording: Recording
    technique: str
    technique_name: str
    critical: frozenset[C]  # K: the recording's critical categories (the ground-truth critical goals)
    available: frozenset[C]  # A at the outset
    hidden: frozenset[C]
    obtainable: frozenset[C]  # hidden categories a request releases
    withheld_techniques: frozenset[str]
    related: Optional[str] = None  # weak cells: the one same-tactic technique whose playbook is kept
    degenerate: Optional[str] = None  # the cell's evidence coincides with another cell's (thin recording)
    relaxed: Optional[str] = None  # a construction rule relaxed because no recording satisfied the cell
    prefix: str = ""  # distinguishes a separate benchmark's scenario names (e.g. the V2 validation benchmark)

    @property
    def name(self) -> str:
        return f"{self.prefix}{self.tactic}/{self.cell}/{self.slot}"

    @property
    def cell(self) -> str:
        return f"{self.suitability}/{self.evidence}"

    def summary(self) -> dict[str, Any]:
        key = lambda cs: [c.value for c in ordered(cs)]  # noqa: E731
        return {"name": self.name, "tactic": self.tactic, "cell": self.cell, "expected": self.expected.value,
                "technique": self.technique, "technique_name": self.technique_name, "recording": self.recording.path,
                "recording_id": self.recording.id, "sources": list(self.recording.sources),
                "categories": key(self.recording.categories), "critical": key(self.critical),
                "available": key(self.available), "hidden": key(self.hidden), "obtainable": key(self.obtainable),
                "related": self.related, "withheld_techniques": sorted(self.withheld_techniques),
                "degenerate": self.degenerate, "relaxed": self.relaxed}  # fmt: skip


@dataclass
class Benchmark:
    catalog: dict
    tactic_of: dict[str, str]  # technique -> its benchmark tactic
    critical: dict[str, frozenset[C]]  # tactic -> critical categories (the ontology)
    footprint: dict[str, frozenset[C]]  # technique -> P(t)
    scenarios: list[PaperScenario]
    library_techniques: list[str]
    infeasible: dict[str, str] = field(default_factory=dict)

    def technique_name(self, tid: str) -> str:
        return self.catalog["techniques"][tid]["name"]

    def ontology(self) -> dict[str, dict[str, list[str]]]:
        return {t: {"critical": [c.value for c in ordered(k)], "supporting": [c.value for c in ordered(set(SEVEN) - k)]}
                for t, k in self.critical.items()}  # fmt: skip

    def composition(self) -> dict[str, Any]:
        return {"scenarios": len(self.scenarios), "by_expected": dict(Counter(s.expected.value for s in self.scenarios)),
                "by_tactic": dict(Counter(s.tactic for s in self.scenarios)),
                "by_cell": dict(sorted(Counter(s.cell for s in self.scenarios).items())),
                "distinct_techniques": len({s.technique for s in self.scenarios}),
                "library_playbooks": len(self.library_techniques),
                "degenerate": sum(s.degenerate is not None for s in self.scenarios),
                "relaxed": dict(Counter(s.relaxed for s in self.scenarios if s.relaxed)),
                "infeasible": len(self.infeasible)}  # fmt: skip


# --- construction -------------------------------------------------------------------------------


def load_catalog(path: Path = CATALOG) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def recordings_of(catalog: dict) -> list[Recording]:
    return [Recording(r["id"], r["path"], r["technique"], tuple(r["sources"]), frozenset(C(c) for c in r["categories"]),
                      r["atomic_red_team"]) for r in catalog["recordings"]]  # fmt: skip


def assign_tactics(catalog: dict, recordings: list[Recording]) -> dict[str, str]:
    """Each technique's benchmark tactic: the candidate tactic with the fewest candidate techniques."""
    candidates = defaultdict(set)
    for r in recordings:
        for tactic in catalog["techniques"][r.technique]["tactics"]:
            candidates[tactic].add(r.technique)
    assigned: dict[str, str] = {}
    for tactic in sorted(TACTIC_ORDER, key=lambda t: (len(candidates[t]), TACTIC_ORDER.index(t))):
        for tid in sorted(candidates[tactic]):
            assigned.setdefault(tid, tactic)
    return assigned


def derive_ontology(recordings: list[Recording], tactic_of: dict[str, str]) -> tuple[dict, dict]:
    """(critical categories per tactic, footprint per technique)."""
    footprint: dict[str, set[C]] = defaultdict(set)
    for r in recordings:
        footprint[r.technique] |= r.categories
    critical = {}
    for tactic in TACTIC_ORDER:
        techniques = [t for t in footprint if tactic_of.get(t) == tactic]
        counts = Counter(c for t in techniques for c in footprint[t])
        chosen = {c for c in SEVEN if techniques and counts[c] / len(techniques) >= 0.5}
        critical[tactic] = frozenset(chosen or {max(SEVEN, key=lambda c: (counts[c], -_order[c]))})
    return critical, {t: frozenset(cs) for t, cs in footprint.items()}


def _preference(r: Recording) -> tuple:
    return (not r.atomic, r.technique, r.path, r.id)


def _relatedness(technique: str, other: str, categories: frozenset[C], footprint: dict) -> tuple:
    same_parent = technique.split(".")[0] == other.split(".")[0]
    theirs = footprint[other]
    jaccard = len(categories & theirs) / len(categories | theirs) if categories | theirs else 0.0
    return (not same_parent, -jaccard, other)


def _construct(cell: str, r: Recording, crit: frozenset[C], footprint: dict, tactic_techniques: list[str],
               critical_footprint_only: bool = False):  # fmt: skip
    """(available, hidden, obtainable, related, degenerate) for one recording in one cell, or None if infeasible."""
    cats, K = r.categories, r.categories & crit
    S = cats - K
    if not K:
        return None
    suitability, evidence = cell.split("/")
    if suitability == "strong":
        covered = (footprint[r.technique] & crit) <= cats if critical_footprint_only else footprint[r.technique] == cats
        if not covered:
            return None
    if evidence == "full":
        return cats, frozenset(), frozenset(), None, None
    if evidence == "partial":
        if not S:
            return None
        hidden = frozenset(ordered(S)[: math.ceil(len(S) / 2)])
        return cats - hidden, hidden, frozenset(), None, None
    if evidence == "sparse":  # none/sparse: keep the critical categories (one or two), hide every supporting one
        if not S or len(K) > 2:
            return None
        return K, S, frozenset(), None, ("same evidence as none/partial" if len(S) == 1 else None)
    if len(cats) < 2:
        return None
    others = sorted((t for t in tactic_techniques if t != r.technique),
                    key=lambda t: _relatedness(r.technique, t, cats, footprint))  # fmt: skip
    if evidence == "critical_missing":
        if suitability == "strong":
            hide = ordered(K)[0]
            return cats - {hide}, frozenset({hide}), frozenset({hide}), None, None
        for other in others:
            needs = footprint[other] & crit
            shared = ordered(needs & K)
            if needs and needs <= cats and shared:
                return cats - {shared[0]}, frozenset({shared[0]}), frozenset({shared[0]}), other, None
        return None
    # weak/sparse_obtainable and weak/sparse_unobtainable: one category remains
    if S:
        keep = ordered(S)[0]
    elif len(K) >= 2:
        keep = ordered(K)[0]
    else:
        return None
    hidden = cats - {keep}
    for other in others:
        needs = footprint[other] & crit
        if needs and needs <= cats and needs & hidden:
            obtainable = hidden if evidence == "sparse_obtainable" else frozenset()
            degenerate = "two-category recording: one category hidden, as in critical_missing" if len(cats) == 2 else None
            return frozenset({keep}), hidden, obtainable, other, degenerate
    return None


def build(catalog: Optional[dict] = None, *, exclude_recordings: Iterable[str] = (), tactics: Optional[Sequence[str]] = None,
          tactic_of: Optional[dict[str, str]] = None, critical: Optional[dict[str, frozenset[C]]] = None,
          name_prefix: str = "") -> Benchmark:  # fmt: skip
    """The 96 scenarios, deterministically (with the defaults).

    The keyword options build a separate validation benchmark: recordings in ``exclude_recordings`` are
    not used, only ``tactics`` are built, and the tactic assignment and ontology can be frozen from
    another benchmark (``tactic_of``, ``critical``) instead of being derived again. Technique footprints
    always come from the whole catalog; ``name_prefix`` keeps scenario names distinct.
    """
    catalog = catalog or load_catalog()
    recordings = recordings_of(catalog)
    tactic_of = tactic_of or assign_tactics(catalog, recordings)
    derived, footprint = derive_ontology(recordings, tactic_of)
    critical = critical or derived
    excluded = set(exclude_recordings)
    cells = {f"{s}/{e}": (label, n) for s, e, label, n in TABLE_2}
    scenarios, infeasible = [], {}
    for tactic in (tactics or TACTIC_ORDER):
        pool = sorted((r for r in recordings if tactic_of[r.technique] == tactic and r.id not in excluded), key=_preference)
        techniques = sorted({r.technique for r in pool})
        used_recordings: set[str] = set()
        uses: Counter = Counter()
        used_in_cell: dict[str, set[str]] = defaultdict(set)
        for cell in FILL_ORDER:
            label, count = cells[cell]
            for slot in range(1, count + 1):
                options, relaxed = [], None
                for relaxation in (None, "A: strong cell with the footprint's critical categories only",
                                   "B: recording reused from another cell of the tactic"):  # fmt: skip
                    for r in pool:
                        if r.id in used_in_cell[cell] or (r.id in used_recordings and not (relaxation or "").startswith("B")):
                            continue
                        built = _construct(cell, r, critical[tactic], footprint, techniques, critical_footprint_only=relaxation is not None)
                        if built is not None:
                            # prefer non-degenerate constructions and techniques not used yet, then the catalog order
                            options.append(((built[4] is not None, uses[r.technique], _preference(r)), r, built))
                    if options:
                        relaxed = relaxation
                        break
                if not options:
                    infeasible[f"{tactic}/{cell}/{slot}"] = "no catalogued recording satisfies the cell, even relaxed"
                    continue
                _, r, (available, hidden, obtainable, related, degenerate) = min(options, key=lambda o: o[0])
                used_recordings.add(r.id)
                used_in_cell[cell].add(r.id)
                uses[r.technique] += 1
                suitability, evidence = cell.split("/")
                same_tactic = frozenset(techniques)
                withheld = {"strong": frozenset(), "weak": same_tactic - {related}, "none": same_tactic}[suitability]
                scenarios.append(PaperScenario(
                    tactic, suitability, evidence, label, slot, r, r.technique, catalog["techniques"][r.technique]["name"],
                    r.categories & critical[tactic], available, hidden, obtainable, withheld, related, degenerate, relaxed,
                    name_prefix))  # fmt: skip
    library = sorted({s.technique for s in scenarios} | {s.related for s in scenarios if s.related})
    ordering = {name: i for i, name in enumerate(f"{t}/{c}/{k}" for t in TACTIC_ORDER for s, e, _, n in TABLE_2
                                                 for c in [f"{s}/{e}"] for k in range(1, n + 1))}  # fmt: skip
    scenarios.sort(key=lambda s: ordering[f"{s.tactic}/{s.cell}/{s.slot}"])
    return Benchmark(catalog, tactic_of, critical, footprint, scenarios, library, infeasible)


# --- library ------------------------------------------------------------------------------------


def technique_playbook(benchmark: Benchmark, technique: str, vocabulary: Vocabulary) -> Playbook:
    """A validated playbook investigating ``technique``: triage, one query per footprint category, a Sigma hunt
    for the technique, a verification and a report (a stand-in for a converted published playbook)."""
    tactic, name = benchmark.tactic_of[technique], benchmark.technique_name(technique)
    categories = ordered(benchmark.footprint[technique])
    draft = Draft(vocabulary, name=f"{technique} {name}", intent_label=tactic, origin=PlaybookOrigin.SEED,
                  description=f"Investigation playbook for ATT&CK {technique} {name} ({display(tactic)}) on Windows hosts; "
                              f"queries {', '.join(c.value.replace('_', ' ') for c in categories)}.")  # fmt: skip
    initial = initial_artifacts()
    actions = ["extract_entities", "triage_alert", *(QUERY_ACTION[c] for c in categories), "hunt_technique_sigma",
               "verify_host_compromise"]  # fmt: skip
    for action in actions:
        node = draft.add_node(action)
        if not draft.wire_all_inputs(node, initial):
            raise ValueError(f"cannot wire {action} for {technique}")
    draft.ensure_terminal()
    return draft.to_playbook()


def requirements_by_own_intent(benchmark: Benchmark, vocabulary: Vocabulary):
    """R_c / R_s of a stored playbook: the paper's seven categories it consumes, split by its own intent's ontology."""

    def required(playbook: Playbook, _incident_critical: frozenset) -> RequiredEvidence:
        consumed = {u.category for n in playbook.nodes for u in (vocabulary.get(n.action).consumes if vocabulary.get(n.action) else ())}
        consumed &= set(SEVEN)
        critical = consumed & benchmark.critical.get(playbook.intent_label, frozenset())
        return RequiredEvidence(critical=ordered(critical), supporting=ordered(consumed - critical))

    return required


def seed_library(benchmark: Benchmark, library: PlaybookLibrary, vocabulary: Vocabulary) -> dict[str, str]:
    """Add one playbook per benchmark technique; returns technique -> entry id."""
    entries = {}
    for technique in benchmark.library_techniques:
        playbook = technique_playbook(benchmark, technique, vocabulary)
        entry = library.add(playbook, deduplicate=False,
                            metadata={"source": "paper968 benchmark", "technique": technique,
                                      "tactic": benchmark.tactic_of[technique]})  # fmt: skip
        entries[technique] = entry.entry_id
    return entries


# --- environment ----------------------------------------------------------------------------------


@dataclass
class _Source:
    library: PlaybookLibrary
    withheld: frozenset[str]

    def retrieve(self, query: RetrievalQuery, top_n: int) -> list[RetrievedCandidate]:
        hits = self.library.retrieve(query, top_n + len(self.withheld))
        return [c for c in hits if c.entry_id not in self.withheld][:top_n]


@dataclass
class PaperEnv:
    benchmark: Benchmark
    settings: Settings
    vocabulary: Vocabulary
    library: PlaybookLibrary
    entries: dict[str, str]  # technique -> entry id
    latency: PathLatency = field(default_factory=PathLatency)
    sharpness: float = 1.0
    goals: str = "critical"  # "critical" (primary) or "all" (sensitivity)
    # "marked" (primary): obtainability 1 for hidden categories marked obtainable; "profile" (diagnostic): their
    # deployment availability from app/data/evidence_availability.json instead, as the extended benchmarks use
    requestability: str = "marked"

    def __post_init__(self) -> None:
        self._technique_of_entry = {e: t for t, e in self.entries.items()}
        self._requirements = requirements_by_own_intent(self.benchmark, self.vocabulary)
        self._profile = AvailabilityProfile.load(PROJECT_ROOT / "app" / "data" / "evidence_availability.json").availability

    def router(self, scenario: PaperScenario, learner: Optional[RoutingLearner] = None) -> tuple[Router, _Source]:
        s = self.settings
        withheld = frozenset(self.entries[t] for t in scenario.withheld_techniques if t in self.entries)
        source = _Source(self.library, withheld)
        router = Router(source, self.vocabulary, AvailabilityProfile(version="paper968"), w=s.routing_w, k=s.routing_k,
                        top_n=s.routing_top_n, tau=s.routing_tau, sharpness=self.sharpness, learner=learner,
                        library_size=lambda: len(self.library) - len(withheld), requirements=self._requirements)  # fmt: skip
        return router, source

    def query(self, scenario: PaperScenario, in_hand: frozenset[C]) -> RetrievalQuery:
        """The fixed, correct analysis of the incident: tactic intent, technique, observed and missing evidence."""
        summary = (f"Windows host telemetry records {scenario.technique_name} ({scenario.technique}), "
                   f"an ATT&CK {display(scenario.tactic)} technique.")  # fmt: skip
        missing = (scenario.critical - in_hand)
        return RetrievalQuery(intent_label=scenario.tactic, intent_description=summary,
                              observed=tuple(ordered(in_hand)), missing=tuple(ordered(missing)),
                              techniques=(scenario.technique,))  # fmt: skip

    def decide(self, scenario: PaperScenario, in_hand: frozenset[C], obtainable: frozenset[C],
               learner: Optional[RoutingLearner], round_number: int) -> RoutingDecision:  # fmt: skip
        router, source = self.router(scenario, learner)
        candidates = source.retrieve(self.query(scenario, in_hand), self.settings.routing_top_n)
        marked = (lambda c: 1.0) if self.requestability == "marked" else (lambda c: self._profile.get(c, 0.0))
        availability = {c: (marked(c) if c in obtainable else 0.0) for c in C}
        return router.score(scenario.name, candidates, set(in_hand), availability, round_number=round_number,
                            critical_categories=self.benchmark.critical[scenario.tactic],
                            allow_request=round_number == 1)  # fmt: skip

    def execute(self, scenario: PaperScenario, action: RoutingAction, decision: RoutingDecision,
                in_hand: frozenset[C]) -> tuple[float, bool]:  # fmt: skip
        """(IC_exec, retained) -- SIMULATED (see the module docstring, item 8)."""
        goals = scenario.critical if self.goals == "critical" else scenario.recording.categories
        if action == REUSE:
            technique = self._technique_of_entry.get(decision.top_candidate.entry_id) if decision.top_candidate else None
            if technique is None or self.benchmark.tactic_of[technique] != scenario.tactic:
                answered = frozenset()  # another tactic's playbook finds no events attributable to the technique
            else:
                answered = goals & self.benchmark.footprint[technique] & in_hand
        else:
            answered = goals & in_hand
        coverage = len(answered) / len(goals) if goals else 1.0
        return coverage, scenario.critical <= answered and coverage >= 0.8

    def diagnostics(self, scenario: PaperScenario, decision: RoutingDecision) -> dict[str, Any]:
        top = decision.top_candidate
        technique = self._technique_of_entry.get(top.entry_id) if top else None
        return {"tactic": scenario.tactic, "technique": scenario.technique, "top_candidate": top.name if top else None,
                "top_similarity": round(top.similarity, 4) if top else None,
                "p_star_is_own_technique": technique == scenario.technique,
                "p_star_same_tactic": bool(technique and self.benchmark.tactic_of[technique] == scenario.tactic),
                "p_star_is_designated_related": bool(technique and technique == scenario.related),
                "request_arm_offered": bool(decision.requestable), "entropy": round(decision.terms.entropy, 4),
                "degenerate": scenario.degenerate}  # fmt: skip

    async def respond(self, scenario: PaperScenario, learner: Optional[RoutingLearner] = None) -> Outcome:
        in_hand = scenario.available
        first = self.decide(scenario, in_hand, scenario.obtainable, learner, 1)
        chain, terminal = [first], first
        if first.action == REQUEST:
            in_hand = in_hand | scenario.obtainable
            terminal = self.decide(scenario, in_hand, frozenset(), learner, 2)
            chain.append(terminal)
        coverage, retained = self.execute(scenario, terminal.action, terminal, in_hand)
        return Outcome(first, terminal, chain, coverage, retained, self.latency.of(first.action, terminal.action),
                       extra=self.diagnostics(scenario, first))  # fmt: skip

    async def alternatives(self, scenario: PaperScenario) -> dict[RoutingAction, list[tuple[float, bool, float]]]:
        first = self.decide(scenario, scenario.available, scenario.obtainable, None, 1)
        options = {GENERATE: [(*self.execute(scenario, GENERATE, first, scenario.available), self.latency.of(GENERATE, GENERATE))]}
        if first.top_candidate is not None:
            options[REUSE] = [(*self.execute(scenario, REUSE, first, scenario.available), self.latency.of(REUSE, REUSE))]
        if first.requestable or first.rule_action == REQUEST:
            in_hand = scenario.available | scenario.obtainable
            second = self.decide(scenario, in_hand, frozenset(), None, 2)
            terminals = [GENERATE] + ([REUSE] if second.top_candidate is not None else [])
            options[REQUEST] = [(*self.execute(scenario, t, second, in_hand), self.latency.of(REQUEST, t)) for t in terminals]
        return options


def build_library(benchmark: Benchmark, settings: Settings, vocabulary: Vocabulary, embedder, index, database) -> tuple[PlaybookLibrary, dict]:
    from app.storage.library_repository import LibraryRepository

    library = PlaybookLibrary(LibraryRepository(database), embedder, index, vocabulary,
                              PlaybookValidator(vocabulary, node_cap=settings.node_cap), k=settings.routing_k)  # fmt: skip
    return library, seed_library(benchmark, library, vocabulary)


# --- the paper's published values (Section 7.2-7.3, Fig. 3) -----------------------------------------------

PAPER = {
    "linucb_accuracy": 0.90, "linucb_macro_f1": 0.90, "linucb_f1": {"reuse": 0.94, "request_evidence": 0.87, "synthesize": 0.90},
    "linucb_confusion_row_normalized": [[0.94, 0.04, 0.02], [0.05, 0.84, 0.11], [0.02, 0.06, 0.92]],
    "linucb_request_recall": 0.841, "rule_request_recall": 0.419, "rule_accuracy": 0.735,
    "linucb_accuracy_first20": 0.790, "linucb_accuracy_after20": 0.930,
    "cumulative_reward": {"linucb": 58.0, "rule": 51.0}, "relative_reward_gain": 0.14,
    "request_share_of_reward_difference": 0.869,
    "by_cell": {"strong/full": 0.97, "strong/partial": 0.91, "strong/critical_missing": 0.90, "weak/critical_missing": 0.86,
                "weak/sparse": 0.781, "none/full": 0.99, "none/partial": 0.98, "none/sparse": 0.91},
    "by_evidence": {"full": 0.975, "partial": 0.933, "sparse": 0.825},
}  # fmt: skip
