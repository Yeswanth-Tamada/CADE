"""SQLite persistence through the repository."""

from __future__ import annotations

import sqlite3
from datetime import timedelta

import pytest
from pydantic import ValidationError

from app.core.errors import IncidentConflictError, IncidentNotFoundError
from app.schemas.enums import ExtractionStatus, IncidentStatus
from app.schemas.records import AnalysisError
from app.tests.conftest import good_output, make_incident

STALE = timedelta(minutes=10)


def _failure(repository, incident_id, **kw):
    return repository.record_analysis(
        incident_id,
        extraction_status=ExtractionStatus.FAILED,
        provider="mock",
        model_name="m",
        attempts=1,
        duration_ms=5,
        warnings=[],
        error=AnalysisError(category="llm_timeout", message="timed out"),
        **kw,
    )


async def _canonical(taxonomy, attack_mapper, incident):
    from app.intent.llm_extractor import IntentExtractor
    from app.intent.mock_llm_client import MockLLMClient

    return await IntentExtractor(MockLLMClient([good_output()]), taxonomy, attack_mapper).extract(incident)


def test_create_and_get_incident_roundtrip(repository):
    incident = make_incident(incident_id="SYN-1", title="SYNTHETIC title")
    stored = repository.create_incident(incident)
    loaded = repository.get_incident("SYN-1")
    assert loaded == stored
    assert loaded.status == IncidentStatus.RECEIVED
    assert loaded.raw_description == incident.raw_description
    assert loaded.received_at.tzinfo is not None and loaded.created_at.tzinfo is not None


def test_duplicate_incident_id_raises_conflict(repository):
    repository.create_incident(make_incident(incident_id="SYN-2"))
    with pytest.raises(IncidentConflictError):
        repository.create_incident(make_incident(incident_id="SYN-2"))


def test_unknown_incident_raises_not_found(repository):
    with pytest.raises(IncidentNotFoundError):
        repository.get_incident("SYN-MISSING")
    with pytest.raises(IncidentNotFoundError):
        repository.start_analysis("SYN-MISSING", stale_after=STALE)
    assert repository.get_latest_analysis("SYN-MISSING") is None


def test_start_analysis_is_exclusive_until_stale(repository):
    repository.create_incident(make_incident(incident_id="SYN-3"))
    started = repository.start_analysis("SYN-3", stale_after=STALE)
    assert started.status == IncidentStatus.ANALYZING and started.analysis_started_at is not None
    with pytest.raises(IncidentConflictError):
        repository.start_analysis("SYN-3", stale_after=STALE)
    # An abandoned analysis (older than stale_after) can be taken over.
    assert repository.start_analysis("SYN-3", stale_after=timedelta(seconds=-1)).status == IncidentStatus.ANALYZING


@pytest.mark.anyio
async def test_record_success_updates_status_and_latest(repository, taxonomy, attack_mapper):
    incident = make_incident(incident_id="SYN-4")
    repository.create_incident(incident)
    _failure(repository, "SYN-4")
    canonical = await _canonical(taxonomy, attack_mapper, incident)
    saved = repository.record_analysis(
        "SYN-4",
        extraction_status=canonical.analysis.extraction_status,
        provider="mock",
        model_name="m",
        attempts=1,
        duration_ms=10,
        warnings=canonical.analysis.warnings,
        result=canonical,
    )
    latest = repository.get_latest_analysis("SYN-4")
    assert latest == saved
    assert latest.result == canonical  # canonical JSON round-trips losslessly
    assert repository.get_incident("SYN-4").status == IncidentStatus.ANALYZED


@pytest.mark.anyio
async def test_failed_analysis_with_result_is_refused_and_nothing_is_written(repository, taxonomy, attack_mapper):
    """Regression: a failure can never be stored together with a 'result'."""
    incident = make_incident(incident_id="SYN-5")
    repository.create_incident(incident)
    canonical = await _canonical(taxonomy, attack_mapper, incident)
    with pytest.raises(ValidationError):
        _failure(repository, "SYN-5", result=canonical)
    assert repository.get_latest_analysis("SYN-5") is None
    assert repository.get_incident("SYN-5").status == IncidentStatus.RECEIVED


def test_failure_sets_analysis_failed_and_clears_lock(repository):
    repository.create_incident(make_incident(incident_id="SYN-6"))
    repository.start_analysis("SYN-6", stale_after=STALE)
    _failure(repository, "SYN-6")
    incident = repository.get_incident("SYN-6")
    assert incident.status == IncidentStatus.ANALYSIS_FAILED
    assert incident.analysis_started_at is None


def test_rows_are_really_in_the_sqlite_file(repository, settings):
    repository.create_incident(make_incident(incident_id="SYN-7"))
    _failure(repository, "SYN-7")
    path = settings.database_url.replace("sqlite:///", "")
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT status FROM incidents WHERE id='SYN-7'").fetchone() == ("analysis_failed",)
        assert conn.execute("SELECT extraction_status, canonical_json FROM analyses").fetchall() == [("failed", None)]
