"""Deterministic normalisation of incident submissions (no LLM involved).

Normalisation never deletes information: the raw description is kept verbatim in
``raw_description`` and only a cleaned *copy* is produced for analysis.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta, timezone

from app.core.errors import InvalidIncidentInputError
from app.schemas.incident import IncidentSubmission, NormalizedIncident

# C0 control characters other than tab (\x09) and newline (\x0a); \r is handled separately.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HORIZONTAL_WS = re.compile(r"[ \t ]+")
_MANY_BLANK_LINES = re.compile(r"\n{3,}")


def generate_incident_id() -> str:
    return f"INC-{uuid.uuid4().hex[:16].upper()}"


def normalize_line_endings(text: str) -> str:
    return text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")


def normalize_text(text: str) -> str:
    """Collapse runs of spaces/tabs, trim each line, limit blank lines, unify line endings.

    Control characters become spaces (rather than being deleted) so that tokens on either
    side are not glued together.
    """
    text = _CONTROL_CHARS.sub(" ", normalize_line_endings(text))
    lines = [_HORIZONTAL_WS.sub(" ", line).strip() for line in text.split("\n")]
    return _MANY_BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def normalize_log_excerpt(text: str) -> str:
    """Logs keep their internal spacing (it can be meaningful); only line endings/trailing spaces change."""
    text = _CONTROL_CHARS.sub(" ", normalize_line_endings(text))
    return "\n".join(line.rstrip() for line in text.split("\n")).strip("\n")


def normalize_timestamp(value: datetime | None, field: str, warnings: list[str], now: datetime) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        warnings.append(f"{field} had no timezone; interpreted as UTC")
        value = value.replace(tzinfo=timezone.utc)
    value = value.astimezone(timezone.utc)
    if value > now + timedelta(days=1):
        warnings.append(f"{field} is more than one day in the future; kept as supplied")
    return value


def normalize_submission(
    submission: IncidentSubmission,
    *,
    max_description_chars: int,
    received_at: datetime | None = None,
) -> NormalizedIncident:
    if len(submission.description) > max_description_chars:
        raise InvalidIncidentInputError(
            f"description exceeds the configured limit of {max_description_chars} characters"
        )

    now = received_at or datetime.now(timezone.utc)
    warnings: list[str] = []

    normalized_description = normalize_text(submission.description)
    if not normalized_description:
        raise InvalidIncidentInputError("description contains no readable text after normalization")

    title = normalize_text(submission.title) if submission.title else None
    log_excerpt = normalize_log_excerpt(submission.log_excerpt) if submission.log_excerpt else None

    return NormalizedIncident(
        incident_id=submission.incident_id or generate_incident_id(),
        source=submission.source,
        title=title or None,
        severity=submission.severity,
        raw_description=submission.description,
        normalized_description=normalized_description,
        log_excerpt=log_excerpt or None,
        occurred_at=normalize_timestamp(submission.occurred_at, "occurred_at", warnings, now),
        received_at=now,
        observables=submission.observables,
        attributes=submission.attributes,
        normalization_warnings=warnings,
    )
