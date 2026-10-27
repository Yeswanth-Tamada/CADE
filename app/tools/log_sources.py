"""Host log sources and the evidence category of each event (paper Section 6.1).

The mapping lives in ``app/data/log_source_categories.json``. It serves three purposes:

* **Elastic queries** -- the event-code operations search a category's channel / event codes;
* **deterministic evidence** -- event records in the incident input (``EventCode=4104``,
  ``"EventID": 1`` with ``Sysmon`` on the same line, ...) are recognised as observed evidence of
  their category, grounded in the matched text (provenance ``pattern_extraction``);
* **event digest** -- each record is restated as one readable line for the intent model
  (``Sysmon event 13 (registry value set): TargetObject=...``); a small model reads that far more
  reliably than raw ``Key=value`` records.

Sysmon's event ids are small numbers that also occur in other contexts, so they count only when a
source keyword (``sysmon``) is on the same line; four- and five-digit ids unique to one source
count on their own.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.core.errors import ConfigurationError
from app.schemas.enums import EvidenceCategory

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "log_source_categories.json"

_EVENT_ID_RE = re.compile(
    r"\b(?:EventCode|EventID|Event ID|event[._]code|event[._]id|winlog\.event_id)[\"']?\s*[:=]\s*[\"']?(\d{1,5})\b",
    re.IGNORECASE,
)
_FIELD_RE = re.compile(r"(?:(?<=\s)|^)([A-Za-z][\w.]*)=")
# Event-record fields worth showing the intent model, in display order (others are left out)
DIGEST_FIELDS = (
    "Image", "CommandLine", "ParentImage", "IntegrityLevel", "TargetObject", "Details", "TargetFilename",
    "SourceImage", "TargetImage", "GrantedAccess", "DestinationIp", "DestinationPort", "QueryName", "ScriptBlockText",
    "ServiceName", "ServiceFileName", "TaskName", "TaskContent", "TargetUserName", "SubjectUserName", "User",
    "IpAddress", "LogonType", "FailureReason", "Count", "Message", "Computer",
)  # fmt: skip
_MAX_DIGEST_LINES = 40
_MAX_FIELD_CHARS = 200


class LogSource(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    channel: str
    keywords: tuple[str, ...] = ()
    events: dict[str, EvidenceCategory] = Field(min_length=1)
    names: dict[str, str] = Field(default_factory=dict)  # event id -> what happened (digest only)


class _LogSourceFile(BaseModel):
    version: str
    note: str = ""
    sources: list[LogSource] = Field(min_length=1)


@dataclass(frozen=True)
class LogSourceMatch:
    category: EvidenceCategory
    source: str
    event_code: str
    excerpt: str


class LogSourceMap:
    def __init__(self, sources: list[LogSource]) -> None:
        self.sources = sources
        self._owners: dict[str, list[LogSource]] = {}
        self._namers: dict[str, list[LogSource]] = {}
        for source in sources:
            for code in source.events:
                self._owners.setdefault(code, []).append(source)
            for code in set(source.events) | set(source.names):
                self._namers.setdefault(code, []).append(source)

    @staticmethod
    def _resolve(code: str, lowered_line: str, candidates: list[LogSource]) -> Optional[LogSource]:
        """The source a record line belongs to: named on the line, or the only owner of a 4+ digit id."""
        named = [s for s in candidates if any(k in lowered_line for k in s.keywords)]
        if len(named) == 1:
            return named[0]
        return candidates[0] if len(candidates) == 1 and len(code) >= 4 else None

    @classmethod
    def load(cls, path: Path = DEFAULT_PATH) -> "LogSourceMap":
        try:
            return cls(_LogSourceFile.model_validate(json.loads(path.read_text(encoding="utf-8"))).sources)
        except FileNotFoundError:
            raise ConfigurationError(f"log source mapping not found: {path}") from None
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ConfigurationError(f"invalid log source mapping {path}: {exc}") from None

    def queries(self, *categories: EvidenceCategory) -> list[tuple[str, list[str]]]:
        """``(channel, event codes)`` pairs whose events belong to any of ``categories``."""
        wanted = set(categories)
        result = []
        for source in self.sources:
            codes = sorted((code for code, category in source.events.items() if category in wanted), key=int)
            if codes:
                result.append((source.channel, codes))
        return result

    def find(self, text: str) -> list[LogSourceMatch]:
        """Event records in ``text``, one match per (category, source, code), in order of appearance."""
        matches: dict[tuple, LogSourceMatch] = {}
        for line in text.splitlines():
            lowered = line.lower()
            for found in _EVENT_ID_RE.finditer(line):
                code = found.group(1).lstrip("0") or "0"
                source = self._resolve(code, lowered, self._owners.get(code, []))
                if source is not None:
                    category = source.events[code]
                    matches.setdefault((category, source.name, code),
                                       LogSourceMatch(category, source.name, code, found.group(0)))  # fmt: skip
        return list(matches.values())

    def digest(self, text: str) -> list[str]:
        """One plain-language line per event record, e.g. ``Sysmon event 13 (registry value set): TargetObject=...``.

        A reading aid for the intent model: the fields are copied from the record (selected and
        reordered, never interpreted), so the digest adds no fact the input does not contain.
        """
        lines: list[str] = []
        for line in text.splitlines():
            found = _EVENT_ID_RE.search(line)
            if found is None:
                continue
            code = found.group(1).lstrip("0") or "0"
            source = self._resolve(code, line.lower(), self._namers.get(code, []))
            label = f"{source.name} event {code} ({source.names.get(code, 'event')})" if source else f"event {code}"
            fields = _record_fields(line)
            shown = [f"{key}={_clip(fields[key])}" for key in DIGEST_FIELDS if fields.get(key)]
            lines.append(f"{label}: {'; '.join(shown)}" if shown else label)
            if len(lines) >= _MAX_DIGEST_LINES:
                break
        return lines


def _record_fields(line: str) -> dict[str, str]:
    """``Key=value`` pairs of one record line (a value runs until the next ``Key=``)."""
    keys = list(_FIELD_RE.finditer(line))
    return {match.group(1): line[match.end(): keys[i + 1].start() if i + 1 < len(keys) else len(line)].strip()
            for i, match in enumerate(keys)}  # fmt: skip


def _clip(value: str) -> str:
    return value if len(value) <= _MAX_FIELD_CHARS else value[: _MAX_FIELD_CHARS - 3] + "..."


@lru_cache
def default_log_sources() -> LogSourceMap:
    return LogSourceMap.load(DEFAULT_PATH)
