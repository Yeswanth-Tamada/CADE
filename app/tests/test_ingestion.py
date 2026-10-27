"""Input parsing and normalisation (no LLM involved)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from app.core.errors import InvalidIncidentInputError, UnsupportedMediaTypeError
from app.ingestion.input_service import IncidentInputService, parse_submission
from app.ingestion.normalizer import normalize_submission, normalize_text
from app.schemas.incident import IncidentSubmission
from app.tests.conftest import EXAMPLES

NOW = datetime(2026, 10, 19, 9, 0, tzinfo=timezone.utc)


def _normalize(**fields):
    return normalize_submission(IncidentSubmission(**fields), max_description_chars=50_000, received_at=NOW)


def test_whitespace_is_normalised_but_raw_text_is_preserved():
    raw = "﻿  SYNTHETIC:\tHost   WS-01\r\n\r\n\r\n\r\nran   cmd.exe  \x00 /c whoami  \r\n"
    incident = _normalize(description=raw)
    assert incident.raw_description == raw  # original kept byte-for-byte
    assert incident.normalized_description == "SYNTHETIC: Host WS-01\n\nran cmd.exe /c whoami"


def test_control_characters_become_spaces_not_deleted():
    assert normalize_text("evil\x00.exe") == "evil .exe"


def test_incident_id_is_generated_when_missing_and_unique():
    a = _normalize(description="x")
    b = _normalize(description="x")
    assert a.incident_id.startswith("INC-") and a.incident_id != b.incident_id


def test_supplied_incident_id_is_kept():
    assert _normalize(description="x", incident_id="SIEM-42").incident_id == "SIEM-42"


def test_timestamps_are_converted_to_utc():
    incident = _normalize(description="x", occurred_at="2026-10-17T02:13:00+05:30")
    assert incident.occurred_at == datetime(2026, 10, 16, 20, 43, tzinfo=timezone.utc)
    assert incident.normalization_warnings == []
    assert incident.received_at == NOW


def test_naive_timestamp_is_assumed_utc_with_warning():
    incident = _normalize(description="x", occurred_at="2026-10-19T08:00:00")
    assert incident.occurred_at.tzinfo is not None
    assert any("timezone" in w for w in incident.normalization_warnings)


def test_epoch_timestamp_is_accepted():
    incident = _normalize(description="x", occurred_at=1_800_000_000)
    assert incident.occurred_at == datetime.fromtimestamp(1_800_000_000, tz=timezone.utc)


def test_future_timestamp_is_kept_with_warning():
    incident = _normalize(description="x", occurred_at=NOW + timedelta(days=30))
    assert incident.occurred_at == NOW + timedelta(days=30)
    assert any("future" in w for w in incident.normalization_warnings)


def test_log_excerpt_keeps_internal_spacing():
    incident = _normalize(description="x", log_excerpt="a   b\r\nc\t\td   \r\n")
    assert incident.log_excerpt == "a   b\nc\t\td"


def test_description_over_configured_limit_is_rejected():
    with pytest.raises(InvalidIncidentInputError, match="limit"):
        normalize_submission(IncidentSubmission(description="x" * 200), max_description_chars=100)


# --- payload parsing --------------------------------------------------------------------


def test_plain_text_body_becomes_a_submission():
    submission = parse_submission(b"SYNTHETIC: odd login", "text/plain; charset=utf-8")
    assert submission.description == "SYNTHETIC: odd login"


def test_json_body_is_validated():
    body = json.dumps({"description": "SYNTHETIC", "source": "siem_alert"}).encode()
    assert parse_submission(body, "application/json").source.value == "siem_alert"


@pytest.mark.parametrize(
    "body,content_type,error",
    [
        (b"{not json", "application/json", InvalidIncidentInputError),
        (b"[1, 2]", "application/json", InvalidIncidentInputError),
        (b'{"description": ""}', "application/json", InvalidIncidentInputError),
        (b"\xff\xfe\x00bad", "text/plain", InvalidIncidentInputError),
        (b"   ", "text/plain", InvalidIncidentInputError),
        (b"x", "application/xml", UnsupportedMediaTypeError),
        (b"x", None, UnsupportedMediaTypeError),
    ],
)
def test_malformed_payloads_raise_meaningful_errors(body, content_type, error):
    with pytest.raises(error):
        parse_submission(body, content_type)


def test_validation_errors_do_not_echo_submitted_values():
    with pytest.raises(InvalidIncidentInputError) as info:
        parse_submission(json.dumps({"description": "x", "severity": "SECRET-VALUE"}).encode(), "application/json")
    assert info.value.details
    assert "SECRET-VALUE" not in json.dumps(info.value.details)


@pytest.mark.parametrize("path", sorted(EXAMPLES.iterdir()), ids=lambda p: p.name)
def test_all_synthetic_examples_ingest(path):
    content_type = "application/json" if path.suffix == ".json" else "text/plain"
    incident = IncidentInputService(max_description_chars=50_000).ingest(path.read_bytes(), content_type)
    assert "SYNTHETIC" in incident.raw_description
    assert incident.received_at.tzinfo is not None
