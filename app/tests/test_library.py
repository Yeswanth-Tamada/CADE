"""The self-evolving playbook library: embeddings, index, retrieval, write-back (milestone M8)."""

from __future__ import annotations

import numpy as np
import pytest

from app.core.config import PROJECT_ROOT
from app.core.errors import PlaybookValidationError
from app.library.embedder import HashingEmbedder, build_embedder
from app.library.index import NumpyIndex, build_index
from app.library.library import PlaybookLibrary, structural_similarity
from app.playbook.validator import PlaybookValidator
from app.playbook.vocabulary import Vocabulary
from app.routing.router import RetrievalQuery, Router
from app.enrichment.availability import AvailabilityProfile
from app.schemas.enums import EvidenceCategory as C
from app.schemas.playbook import Playbook, PlaybookOrigin
from app.schemas.routing import RoutingAction
from app.storage.database import Database
from app.storage.library_repository import LibraryRepository
from app.tests.test_playbook_dag import pb, seeds

DATA = PROJECT_ROOT / "app" / "data"
SEED_PATH = DATA / "seed_playbooks.json"


@pytest.fixture(scope="module")
def vocabulary(taxonomy) -> Vocabulary:
    return Vocabulary.load(DATA / "action_vocabulary.json", DATA / "intent_goals.json", taxonomy)


@pytest.fixture
def database(settings):
    db = Database(settings.database_url)
    db.create_schema()
    yield db
    db.dispose()


def make_library(database, vocabulary, **kwargs) -> PlaybookLibrary:
    embedder = HashingEmbedder()
    return PlaybookLibrary(LibraryRepository(database), embedder, NumpyIndex(embedder.dimension), vocabulary,
                           PlaybookValidator(vocabulary), **kwargs)  # fmt: skip


def query(intent: str, observed=(), missing=(), description: str = "") -> RetrievalQuery:
    return RetrievalQuery(intent_label=intent, intent_description=description or intent.replace("_", " "),
                          observed=tuple(observed), missing=tuple(missing))  # fmt: skip


def _seed(intent: str) -> Playbook:
    return Playbook.model_validate(next(s for s in seeds() if s["intent_label"] == intent))


# --- embeddings & index ---------------------------------------------------------------------


def test_hashing_embedder_is_deterministic_and_normalised():
    embedder = HashingEmbedder(64)
    a, b, c = embedder.embed(["ransomware file encryption", "ransomware file encryption", "phishing email link"])
    assert np.allclose(a, b) and np.isclose(np.linalg.norm(a), 1.0)
    assert float(a @ b) > float(a @ c)
    assert np.allclose(embedder.embed([""])[0], 0)


def test_missing_optional_packages_fall_back(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name in ("sentence_transformers", "faiss"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    embedder, reason = build_embedder("sentence_transformers", "any-model")
    assert isinstance(embedder, HashingEmbedder) and "hashing" in reason
    index, index_reason = build_index("faiss", 8)
    assert isinstance(index, NumpyIndex) and "NumPy" in index_reason
    assert build_embedder("hashing", "x")[1] is None and build_index("numpy", 8)[1] is None


def test_numpy_index_upsert_search_remove():
    index = NumpyIndex(3)
    index.upsert("a", np.array([1, 0, 0], dtype=np.float32))
    index.upsert("b", np.array([0, 1, 0], dtype=np.float32))
    index.upsert("a", np.array([0.6, 0.8, 0], dtype=np.float32))  # update in place
    assert len(index) == 2
    assert [i for i, _ in index.search(np.array([0, 1, 0]), 2)] == ["b", "a"]
    index.remove("b")
    assert index.search(np.array([0, 1, 0]), 5) == [("a", pytest.approx(0.8))]


# --- library ------------------------------------------------------------------------------


def test_seeding_validates_and_loads_every_seed_once(database, vocabulary):
    library = make_library(database, vocabulary)
    assert library.seed_if_empty(SEED_PATH) == len(seeds())
    assert library.seed_if_empty(SEED_PATH) == 0  # only when empty
    entries = library.entries()
    assert {e.intent_label for e in entries} == {s["intent_label"] for s in seeds()}
    assert all(e.origin == PlaybookOrigin.SEED and e.smoothed_success == 0.5 for e in entries)


def test_retrieval_prefers_the_same_intent(database, vocabulary):
    library = make_library(database, vocabulary)
    library.seed_if_empty(SEED_PATH)
    for intent in ("ransomware_investigation", "phishing_investigation", "credential_compromise",
                   "command_and_control_persistence", "denial_of_service"):  # fmt: skip
        hits = library.retrieve(query(intent, observed=[C.FILE_ACTIVITY]), 10)
        assert hits[0].playbook.intent_label == intent, intent
        assert len(hits) <= 10 and [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_index_is_rebuilt_from_the_database(database, vocabulary):
    first = make_library(database, vocabulary)
    first.seed_if_empty(SEED_PATH)
    second = make_library(database, vocabulary)
    assert second.load() == len(seeds()) and len(second.index) == len(seeds())
    assert second.retrieve(query("phishing_investigation"), 3)[0].playbook.intent_label == "phishing_investigation"


def test_reembeds_when_the_embedder_changes(database, vocabulary):
    make_library(database, vocabulary).seed_if_empty(SEED_PATH)
    other = HashingEmbedder(128)
    library = PlaybookLibrary(LibraryRepository(database), other, NumpyIndex(128), vocabulary, PlaybookValidator(vocabulary))
    library.load()
    assert {s.embedding_model for s in LibraryRepository(database).list()} == {other.name}


def test_invalid_or_unsafe_playbooks_are_refused(database, vocabulary):
    library = make_library(database, vocabulary)
    unsafe = pb("n1:extract_entities n2:isolate_host n3:generate_incident_report", "n1>n2:host_id n2>n3:findings")
    with pytest.raises(PlaybookValidationError) as info:
        library.add(unsafe)
    assert any(d["code"] == "unverified_destructive" for d in info.value.details)
    assert len(library) == 0


def test_smoothed_success_counts_and_deprioritisation(database, vocabulary):
    library = make_library(database, vocabulary, deprioritize_after=5, deprioritize_below=0.35)
    entry = library.add(_seed("ransomware_investigation"))
    for _ in range(5):
        record = library.record_outcome(entry.playbook, success=False, source_entry_id=entry.entry_id)
    assert record.action == "updated" and (record.successes, record.executions) == (0, 5)
    stored = library.get(entry.entry_id)
    assert stored.smoothed_success == pytest.approx(3 / 11) and stored.deprioritized
    for _ in range(6):
        library.record_outcome(entry.playbook, success=True, source_entry_id=entry.entry_id)
    recovered = library.get(entry.entry_id)
    assert recovered.smoothed_success == pytest.approx(9 / 17) and not recovered.deprioritized


def test_deprioritised_entries_rank_lower(database, vocabulary):
    library = make_library(database, vocabulary, deprioritize_after=1, deprioritize_below=0.5, deprioritized_weight=0.1)
    library.seed_if_empty(SEED_PATH)
    top = library.retrieve(query("phishing_investigation"), 1)[0]
    library.record_outcome(top.playbook, success=False, source_entry_id=top.entry_id)
    hits = library.retrieve(query("phishing_investigation"), len(library))  # the demoted entry may fall past the top 10
    demoted = next(h for h in hits if h.entry_id == top.entry_id)
    assert demoted.deprioritized and demoted.score == pytest.approx(demoted.similarity * 0.1)
    assert hits[0].entry_id != top.entry_id


def test_synthesized_playbooks_enter_only_when_retained(database, vocabulary):
    library = make_library(database, vocabulary)
    new = _seed("lateral_movement").model_copy(update={"origin": PlaybookOrigin.EVOLVED, "name": "evolved"})
    assert library.record_outcome(new, success=False).action == "not_stored"
    assert len(library) == 0
    inserted = library.record_outcome(new, success=True, metadata={"incident_id": "INC-1"})
    assert inserted.action == "inserted" and (inserted.successes, inserted.executions) == (1, 1)
    assert library.get(inserted.entry_id).metadata == {"incident_id": "INC-1"}


def test_near_duplicates_update_counts_instead_of_inserting(database, vocabulary):
    library = make_library(database, vocabulary)
    base = _seed("data_exfiltration").model_copy(update={"origin": PlaybookOrigin.EVOLVED})
    first = library.record_outcome(base, success=True)
    # same DAG with one parameter changed: structurally near-identical
    tweaked = base.model_copy(deep=True)
    tweaked.nodes[2].params = {"lookback_hours": 72}
    assert structural_similarity(base, tweaked) == 1.0
    merged = library.record_outcome(tweaked, success=True)
    assert merged.action == "merged" and merged.entry_id == first.entry_id
    assert (merged.successes, merged.executions) == (2, 2) and len(library) == 1
    different = _seed("web_application_attack").model_copy(update={"origin": PlaybookOrigin.EVOLVED,
                                                                    "intent_label": "data_exfiltration"})  # fmt: skip
    assert library.record_outcome(different, success=True).action == "inserted" and len(library) == 2


def test_seeding_distinct_playbooks_can_keep_structural_duplicates(database, vocabulary):
    library = make_library(database, vocabulary)
    base = _seed("discovery")
    library.add(base)
    assert library.add(base.model_copy(update={"name": "T1087 Account Discovery"})).name == base.name  # merged
    other = library.add(base.model_copy(update={"name": "T1087 Account Discovery"}), deduplicate=False)
    assert other.name == "T1087 Account Discovery" and len(library) == 2


def test_an_identical_dag_counts_a_success_even_when_its_text_differs(database, vocabulary):
    """Regression: a synthesized copy of a seed (other name/description, so another embedding) was reported as
    'inserted' while add() silently returned the seed, and the success was never counted."""
    library = make_library(database, vocabulary)
    library.seed_if_empty(SEED_PATH)
    seed = _seed("command_and_control_persistence")
    entry = next(e for e in library.entries("command_and_control_persistence"))
    copy = seed.model_copy(update={"origin": PlaybookOrigin.EVOLVED, "name": "evolved copy",
                                   "description": "Adapted from library playbook 'C2 and persistence investigation'."})  # fmt: skip
    size = len(library)
    outcome = library.record_outcome(copy, success=True)
    assert outcome.action == "merged" and outcome.entry_id == entry.entry_id
    assert (outcome.successes, outcome.executions) == (entry.successes + 1, entry.executions + 1)
    assert len(library) == size


def test_reuse_requires_a_retained_history_unless_curated(database, vocabulary):
    library = make_library(database, vocabulary, allow_curated_reuse=False)
    library.seed_if_empty(SEED_PATH)
    assert library.retrieve(query("ransomware_investigation"), 10) == []  # curated reuse disabled
    evolved = _seed("ransomware_investigation").model_copy(update={"origin": PlaybookOrigin.EVOLVED, "name": "evolved"})
    evolved.nodes[3].params = {"lookback_hours": 6}
    entry = library.add(evolved)  # stored without a track record (e.g. imported for review)
    assert library.retrieve(query("ransomware_investigation"), 10) == []
    library.record_outcome(evolved, success=True, source_entry_id=entry.entry_id)
    assert [h.entry_id for h in library.retrieve(query("ransomware_investigation"), 10)] == [entry.entry_id]


def test_structural_similarity():
    a = _seed("phishing_investigation")
    assert structural_similarity(a, a) == 1.0
    assert structural_similarity(a, _seed("denial_of_service")) < 0.5


def test_router_over_the_real_library(database, vocabulary):
    library = make_library(database, vocabulary)
    library.seed_if_empty(SEED_PATH)
    router = Router(library, vocabulary, AvailabilityProfile.load(DATA / "evidence_availability.json"))
    hits = library.retrieve(query("credential_compromise"), 10)
    required = vocabulary.required_evidence(hits[0].playbook)
    decision = router.score("INC-L", hits, set(required.critical) | set(required.supporting), router.availability.availability)
    assert decision.top_candidate.intent_label == "credential_compromise"
    assert decision.terms.candidates_retrieved == len(hits) == 10
    assert decision.action in set(RoutingAction)


def test_incident_specific_playbooks_are_generalised_before_storage(database, vocabulary):
    library = make_library(database, vocabulary)
    # valid only for an incident that provides host_id directly
    specific = pb("n1:query_process_events n2:analyze_process_tree n3:generate_incident_report",
                  "n1>n2:process_list n2>n3:findings", intent="suspicious_process_execution")  # fmt: skip
    specific = specific.model_copy(update={"origin": PlaybookOrigin.EVOLVED})
    assert not PlaybookValidator(vocabulary).validate(specific).valid
    record = library.record_outcome(specific, success=True)
    assert record.action == "inserted"
    stored = library.get(record.entry_id).playbook
    assert PlaybookValidator(vocabulary).validate(stored).valid
    assert "query_process_events" in stored.action_multiset() and len(stored.nodes) > len(specific.nodes)
