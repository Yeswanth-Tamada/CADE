"""The self-evolving playbook library (paper Section 3).

Each entry stores a playbook's DAG, intent, required evidence (critical and supporting), metadata,
success / execution counts and an embedding of its intent and evidence profile, so retrieval
matches on both what is investigated and what evidence is in hand.

* **Retrieval** (the routing layer's candidate source): embedding similarity, restricted to
  reuse-eligible entries -- curated entries (seed / imported) and entries with at least one
  analyst-retained success (paper Section 8: "conditioning reuse on recorded analyst-retained
  outcomes"). De-prioritised entries have their score multiplied by a weight < 1.
* **Write-back**: an execution outcome of a reused entry updates its counts. A synthesized
  playbook enters the library only when its run resolved the incident and the analyst retained
  its findings; a near-duplicate of a same-intent entry (embedding cosine and DAG structure both
  above thresholds) updates that entry's counts instead of adding a record.
* **De-prioritisation**: after ``deprioritize_after`` executions, an entry whose smoothed success
  falls below ``deprioritize_below`` is de-prioritised (and recovers if it improves).
"""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np

from app.core.errors import ConfigurationError, PlaybookValidationError
from app.core.logging import get_logger, log_event
from app.library.embedder import Embedder
from app.library.index import VectorIndex
from app.playbook.repair import PlaybookRepairer
from app.playbook.validator import PlaybookValidator
from app.playbook.vocabulary import Vocabulary
from app.routing.router import RetrievalQuery, RetrievedCandidate
from app.routing.utility import smoothed_success
from app.schemas.library import LibraryEntry, OutcomeRecord
from app.schemas.playbook import Playbook, PlaybookOrigin
from app.storage.library_repository import LibraryRepository, StoredPlaybook

logger = get_logger("library")

CURATED_ORIGINS = frozenset({PlaybookOrigin.SEED, PlaybookOrigin.IMPORTED})


def structural_similarity(a: Playbook, b: Playbook) -> float:
    """Jaccard similarity of the two DAGs' actions and action-level edges (IDs ignored)."""

    def features(p: Playbook) -> set[tuple]:
        actions = {n.node_id: n.action for n in p.nodes}
        nodes = {("node", action, i) for action in set(actions.values()) for i in range(list(actions.values()).count(action))}
        edges = {("edge", actions.get(e.source), actions.get(e.target), e.artifact) for e in p.edges}
        return nodes | edges

    fa, fb = features(a), features(b)
    return len(fa & fb) / len(fa | fb) if fa | fb else 1.0


class PlaybookLibrary:
    def __init__(
        self,
        repository: LibraryRepository,
        embedder: Embedder,
        index: VectorIndex,
        vocabulary: Vocabulary,
        validator: PlaybookValidator,
        *,
        k: float = 3,
        near_duplicate_cosine: float = 0.95,
        near_duplicate_jaccard: float = 0.8,
        deprioritize_after: int = 5,
        deprioritize_below: float = 0.35,
        deprioritized_weight: float = 0.5,
        allow_curated_reuse: bool = True,
    ) -> None:
        self.repository = repository
        self.embedder = embedder
        self.index = index
        self.vocabulary = vocabulary
        self.validator = validator
        self.k = k
        self.near_duplicate_cosine = near_duplicate_cosine
        self.near_duplicate_jaccard = near_duplicate_jaccard
        self.deprioritize_after = deprioritize_after
        self.deprioritize_below = deprioritize_below
        self.deprioritized_weight = deprioritized_weight
        self.allow_curated_reuse = allow_curated_reuse
        self._repairer = PlaybookRepairer(validator)
        self._entries: dict[str, StoredPlaybook] = {}
        self._lock = threading.Lock()

    # --- texts embedded for entries and queries (same template so similarity is meaningful) ----

    def _intent_text(self, label: str, description: str) -> str:
        return f"intent {label.replace('_', ' ')}. {description}"

    def entry_text(self, playbook: Playbook) -> str:
        required = self.vocabulary.required_evidence(playbook)
        techniques = sorted({t for n in playbook.nodes for t in (self.vocabulary.get(n.action).techniques
                                                                 if self.vocabulary.get(n.action) else ())})  # fmt: skip
        evidence = sorted(c.value.replace("_", " ") for c in required.critical + required.supporting)
        return (f"{self._intent_text(playbook.intent_label, playbook.description or playbook.name)} "
                f"evidence: {', '.join(evidence)}. critical: {', '.join(c.value for c in required.critical)}. "
                f"techniques: {' '.join(techniques)}")  # fmt: skip

    def query_text(self, query: RetrievalQuery) -> str:
        evidence = sorted(c.value.replace("_", " ") for c in set(query.observed) | set(query.missing))
        return (f"{self._intent_text(query.intent_label, query.intent_description)} "
                f"evidence: {', '.join(evidence)}. missing: {', '.join(c.value for c in query.missing)}. "
                f"techniques: {' '.join(query.techniques)}")  # fmt: skip

    def _embed(self, text: str) -> np.ndarray:
        return self.embedder.embed([text])[0]

    # --- lifecycle -------------------------------------------------------------------------------

    def load(self) -> int:
        """Load all entries and rebuild the vector index (re-embedding if the embedder changed)."""
        stored = self.repository.list()
        with self._lock:
            self._entries = {}
            for entry in stored:
                vector = np.asarray(entry.embedding, dtype=np.float32)
                if entry.embedding_model != self.embedder.name or vector.shape[0] != self.embedder.dimension:
                    vector = self._embed(self.entry_text(entry.playbook))
                    self.repository.update_embedding(entry.entry_id, self.embedder.name, vector.tolist())
                self.index.upsert(entry.entry_id, vector)
                self._entries[entry.entry_id] = entry
        log_event(logger, "library loaded", operation="startup", stage="library", outcome="success",
                  entries=len(stored), embedder=self.embedder.name, index=self.index.name)  # fmt: skip
        return len(stored)

    def seed_if_empty(self, path: Path) -> int:
        if self.repository.count():
            return 0
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ConfigurationError(f"cannot read seed playbooks {path}: {type(exc).__name__}") from None
        added = 0
        for raw in data.get("playbooks", []):
            playbook = Playbook.model_validate(raw).model_copy(update={"origin": PlaybookOrigin.SEED})
            self.add(playbook, metadata={"source": "seed_playbooks.json", "seed_version": data.get("version")})
            added += 1
        return added

    # --- queries ---------------------------------------------------------------------------------

    def eligible(self, entry: StoredPlaybook) -> bool:
        if entry.playbook.origin in CURATED_ORIGINS:
            return self.allow_curated_reuse
        return entry.successes >= 1

    def retrieve(self, query: RetrievalQuery, top_n: int) -> list[RetrievedCandidate]:
        vector = self._embed(self.query_text(query))
        hits = self.index.search(vector, max(top_n * 3, top_n))
        candidates = []
        with self._lock:
            for entry_id, similarity in hits:
                entry = self._entries.get(entry_id)
                if entry is None or not self.eligible(entry):
                    continue
                score = similarity * (self.deprioritized_weight if entry.deprioritized else 1.0)
                candidates.append(RetrievedCandidate(entry_id=entry_id, playbook=entry.playbook, similarity=similarity,
                                                     score=score, successes=entry.successes, executions=entry.executions,
                                                     deprioritized=entry.deprioritized))  # fmt: skip
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top_n]

    def get(self, entry_id: str) -> Optional[LibraryEntry]:
        with self._lock:
            entry = self._entries.get(entry_id)
        return self._to_entry(entry) if entry else None

    def entries(self, intent_label: Optional[str] = None) -> list[LibraryEntry]:
        with self._lock:
            stored = list(self._entries.values())
        return [self._to_entry(e) for e in stored if intent_label is None or e.playbook.intent_label == intent_label]

    def __len__(self) -> int:
        return len(self._entries)

    # --- writes ----------------------------------------------------------------------------------

    def add(self, playbook: Playbook, *, metadata: Optional[dict[str, Any]] = None, successes: int = 0,
            executions: int = 0, deduplicate: bool = True) -> LibraryEntry:  # fmt: skip
        """Store a validated playbook. A structural duplicate of an entry with the same intent is not added
        again unless ``deduplicate`` is False (distinct published playbooks seeding a benchmark library)."""
        report = self.validator.validate(playbook)
        if not report.valid:
            raise PlaybookValidationError(
                "the playbook is not a valid, safe DAG",
                details=[{"code": v.code.value, "message": v.message, "node_id": v.node_id} for v in report.violations[:20]],
            )
        with self._lock:
            duplicate = next((e for e in self._entries.values() if e.structural_hash == playbook.structural_hash()
                              and e.playbook.intent_label == playbook.intent_label), None)  # fmt: skip
        if duplicate is not None and deduplicate:
            return self._to_entry(duplicate)
        vector = self._embed(self.entry_text(playbook))
        techniques = sorted({t for n in playbook.nodes for t in self.vocabulary.get(n.action).techniques})
        stored = self.repository.insert(
            f"PB-{uuid.uuid4().hex[:16].upper()}", playbook, required=self.vocabulary.required_evidence(playbook),
            techniques=techniques, metadata=metadata or {}, successes=successes, executions=executions,
            embedding_model=self.embedder.name, embedding=vector.tolist(),
        )  # fmt: skip
        with self._lock:
            self._entries[stored.entry_id] = stored
            self.index.upsert(stored.entry_id, vector)
        return self._to_entry(stored)

    def record_outcome(self, playbook: Playbook, *, success: bool, source_entry_id: Optional[str] = None,
                       metadata: Optional[dict[str, Any]] = None) -> OutcomeRecord:  # fmt: skip
        """Feed an execution outcome back (paper Section 3)."""
        if source_entry_id is not None:
            return self._update(source_entry_id, success, "updated", "outcome of a reused library playbook")
        if not success:
            return OutcomeRecord(action="not_stored", reason="only playbooks whose findings the analyst retained "
                                                             "and that resolved the incident are written back")  # fmt: skip
        generic = self.generalize(playbook)
        if generic is None:
            return OutcomeRecord(action="not_stored", reason="the playbook could not be generalised beyond this incident")
        playbook = generic
        vector = self._embed(self.entry_text(playbook))
        with self._lock:
            same_intent = [e for e in self._entries.values() if e.playbook.intent_label == playbook.intent_label]
        identical = next((e for e in same_intent if e.structural_hash == playbook.structural_hash()), None)
        if identical is not None:  # same DAG, e.g. an adapted library example: count the success, never re-add
            return self._update(identical.entry_id, True, "merged", "identical to an existing entry")
        for entry in same_intent:
            cosine = float(np.dot(vector, np.asarray(entry.embedding, dtype=np.float32)))
            if cosine >= self.near_duplicate_cosine and structural_similarity(playbook, entry.playbook) >= self.near_duplicate_jaccard:
                return self._update(entry.entry_id, True, "merged", "near-duplicate of an existing entry")
        entry = self.add(playbook, metadata=metadata, successes=1, executions=1)
        return OutcomeRecord(entry_id=entry.entry_id, action="inserted", reason="new playbook retained by the analyst",
                             successes=1, executions=1, deprioritized=False)  # fmt: skip

    def generalize(self, playbook: Playbook) -> Optional[Playbook]:
        """Make an incident-specific playbook reusable for other incidents.

        A synthesized playbook may take entity inputs (e.g. ``host_id``) straight from its incident.
        Library entries may assume only the incident context, so such inputs are re-wired from an
        entity-producing step (repair inserts ``extract_entities`` where needed). ``None`` if impossible.
        """
        if self.validator.validate(playbook).valid:
            return playbook
        return self._repairer.repair(playbook)

    def _update(self, entry_id: str, success: bool, action: str, reason: str) -> OutcomeRecord:
        stored = self.repository.record_outcome(entry_id, success=success, deprioritize_after=self.deprioritize_after,
                                                deprioritize_below=self.deprioritize_below, k=self.k)  # fmt: skip
        if stored is None:
            return OutcomeRecord(action="not_stored", reason="the library entry no longer exists")
        with self._lock:
            self._entries[entry_id] = stored
        return OutcomeRecord(entry_id=entry_id, action=action, reason=reason, successes=stored.successes,
                             executions=stored.executions, deprioritized=stored.deprioritized)  # fmt: skip

    def _to_entry(self, stored: StoredPlaybook) -> LibraryEntry:
        return LibraryEntry(
            entry_id=stored.entry_id, name=stored.playbook.name, intent_label=stored.playbook.intent_label,
            origin=stored.playbook.origin, playbook=stored.playbook, required=stored.required,
            techniques=stored.techniques, successes=stored.successes, executions=stored.executions,
            smoothed_success=smoothed_success(stored.successes, stored.executions, self.k),
            deprioritized=stored.deprioritized, metadata=stored.metadata, created_at=stored.created_at,
            updated_at=stored.updated_at,
        )  # fmt: skip
