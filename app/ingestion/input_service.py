"""Turns raw HTTP request bodies into validated, normalised incidents.

Supported inputs:

* ``Content-Type: application/json`` -- a structured :class:`IncidentSubmission` object.
* ``Content-Type: text/plain``       -- a free-text incident description.
"""

from __future__ import annotations

import json

from pydantic import ValidationError

from app.core.errors import InvalidIncidentInputError, UnsupportedMediaTypeError
from app.ingestion.normalizer import normalize_submission
from app.schemas.incident import IncidentSubmission, NormalizedIncident


def validation_details(exc: ValidationError) -> list[dict]:
    """Pydantic errors without echoing the submitted values back (they may be sensitive)."""
    return [
        {"loc": list(err["loc"]), "msg": err["msg"], "type": err["type"]}
        for err in exc.errors(include_url=False, include_input=False, include_context=False)
    ]


def parse_submission(body: bytes, content_type: str | None) -> IncidentSubmission:
    media_type = (content_type or "").split(";")[0].strip().lower()

    if media_type == "text/plain":
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError:
            raise InvalidIncidentInputError("plain-text body must be UTF-8 encoded") from None
        return _validate({"description": text})

    if media_type == "application/json" or media_type.endswith("+json"):
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise InvalidIncidentInputError(f"malformed JSON body: {exc.__class__.__name__}") from None
        if not isinstance(payload, dict):
            raise InvalidIncidentInputError("JSON body must be an object")
        return _validate(payload)

    raise UnsupportedMediaTypeError("use Content-Type 'application/json' or 'text/plain'")


def _validate(payload: dict) -> IncidentSubmission:
    try:
        return IncidentSubmission.model_validate(payload)
    except ValidationError as exc:
        raise InvalidIncidentInputError("incident input failed validation", details=validation_details(exc)) from None


class IncidentInputService:
    def __init__(self, max_description_chars: int) -> None:
        self.max_description_chars = max_description_chars

    def ingest(self, body: bytes, content_type: str | None) -> NormalizedIncident:
        return self.normalize(parse_submission(body, content_type))

    def normalize(self, submission: IncidentSubmission) -> NormalizedIncident:
        return normalize_submission(submission, max_description_chars=self.max_description_chars)
