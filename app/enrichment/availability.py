"""The per-deployment evidence availability profile (rho in the paper's routing equations)."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.core.errors import ConfigurationError
from app.schemas.enums import EvidenceCategory


class AvailabilityProfile(BaseModel):
    version: str
    note: str = ""
    availability: dict[EvidenceCategory, float] = Field(default_factory=dict)

    @field_validator("availability")
    @classmethod
    def _probabilities(cls, value: dict[EvidenceCategory, float]) -> dict[EvidenceCategory, float]:
        if any(not 0.0 <= p <= 1.0 for p in value.values()):
            raise ValueError("availabilities must be probabilities in [0, 1]")
        return value

    def of(self, category: EvidenceCategory) -> float:
        """Unlisted categories are assumed unobtainable (conservative)."""
        return self.availability.get(category, 0.0)

    @classmethod
    def load(cls, path: Path) -> "AvailabilityProfile":
        try:
            return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except FileNotFoundError:
            raise ConfigurationError(f"evidence availability file not found: {path}") from None
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ConfigurationError(f"invalid evidence availability file {path}: {exc}") from None
