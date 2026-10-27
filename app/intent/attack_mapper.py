"""MITRE ATT&CK grounding of candidate technique IDs.

The mapper loads an ATT&CK Enterprise STIX 2.x bundle (see ``app/data/README.md``). If the
file is missing or unreadable it runs in *unverified* mode: IDs are only format-checked and
are never labelled as validated. Technique names and tactics always come from the dataset,
never from the LLM.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.core.logging import get_logger, log_event
from app.intent.grounding import WarningCollector, canonical_text
from app.intent.schemas import LLMTechnique
from app.schemas.enums import AttackMappingMode, AttackValidationStatus
from app.schemas.incident import TECHNIQUE_ID_PATTERN, AttackTechniqueMapping

logger = get_logger("attack")

_TECHNIQUE_ID_RE = re.compile(TECHNIQUE_ID_PATTERN)
_MIN_HINT_NAME_CHARS = 5  # shorter names are too likely to be ordinary words


class AttackDatasetError(Exception):
    pass


@dataclass(frozen=True)
class TechniqueRecord:
    technique_id: str
    name: str
    tactics: tuple[str, ...]
    deprecated: bool
    revoked: bool

    @property
    def active(self) -> bool:
        return not (self.deprecated or self.revoked)


@dataclass(frozen=True)
class AttackDataset:
    version: Optional[str]
    techniques: dict[str, TechniqueRecord]

    @classmethod
    def from_stix_file(cls, path: Path) -> "AttackDataset":
        try:
            bundle = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AttackDatasetError(f"cannot read ATT&CK bundle: {exc.__class__.__name__}") from None
        return cls.from_stix_bundle(bundle)

    @classmethod
    def from_stix_bundle(cls, bundle: dict) -> "AttackDataset":
        if not isinstance(bundle, dict) or not isinstance(bundle.get("objects"), list):
            raise AttackDatasetError("not a STIX bundle (missing 'objects')")

        version = None
        techniques: dict[str, TechniqueRecord] = {}
        for obj in bundle["objects"]:
            if obj.get("type") == "x-mitre-collection":
                version = obj.get("x_mitre_version") or version
            if obj.get("type") != "attack-pattern":
                continue
            technique_id = next(
                (
                    ref.get("external_id")
                    for ref in obj.get("external_references", [])
                    if ref.get("source_name") == "mitre-attack"
                ),
                None,
            )
            if not technique_id or not _TECHNIQUE_ID_RE.match(technique_id) or not obj.get("name"):
                continue
            record = TechniqueRecord(
                technique_id=technique_id,
                name=obj["name"],
                tactics=tuple(
                    phase["phase_name"]
                    for phase in obj.get("kill_chain_phases", [])
                    if phase.get("kill_chain_name") == "mitre-attack" and phase.get("phase_name")
                ),
                deprecated=bool(obj.get("x_mitre_deprecated", False)),
                revoked=bool(obj.get("revoked", False)),
            )
            # The same ID can appear twice (an old revoked object and its replacement); prefer the active one.
            existing = techniques.get(technique_id)
            if existing is None or (record.active and not existing.active):
                techniques[technique_id] = record

        if not techniques:
            raise AttackDatasetError("bundle contains no ATT&CK techniques")
        return cls(version=version, techniques=techniques)


DEFAULT_ARTIFACT_HINTS_PATH = Path(__file__).resolve().parents[1] / "data" / "attack_artifact_hints.json"


@dataclass(frozen=True)
class ArtifactHint:
    """A well-known artifact of a technique (``app/data/attack_artifact_hints.json``)."""

    technique_ids: tuple[str, ...]  # candidates, newest ATT&CK numbering first
    pattern: re.Pattern
    artifact: str


@dataclass(frozen=True)
class ArtifactMatch:
    technique_id: str
    name: str  # from the dataset ("" without one)
    tactics: tuple[str, ...]  # from the dataset (empty without one)
    artifact: str
    matched: str  # the text that matched, from the incident data


def load_artifact_hints(path: Path = DEFAULT_ARTIFACT_HINTS_PATH) -> list[ArtifactHint]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        hints = [ArtifactHint(tuple(h["techniques"]), re.compile(h["pattern"], re.IGNORECASE), h["artifact"])
                 for h in data["hints"]]  # fmt: skip
    except FileNotFoundError:
        return []
    except (OSError, json.JSONDecodeError, KeyError, TypeError, re.error) as exc:
        raise AttackDatasetError(f"invalid ATT&CK artifact hints file {path.name}: {exc}") from None
    bad = [t for h in hints for t in h.technique_ids if not _TECHNIQUE_ID_RE.match(t)] + [
        h.artifact for h in hints if not h.technique_ids]
    if bad:
        raise AttackDatasetError(f"artifact hints name malformed technique ids: {', '.join(bad)}")
    return hints


class AttackMapper:
    def __init__(self, dataset: Optional[AttackDataset] = None, unavailable_reason: Optional[str] = None,
                 artifact_hints: Optional[list[ArtifactHint]] = None) -> None:  # fmt: skip
        self.dataset = dataset
        self.unavailable_reason = unavailable_reason if dataset is None else None
        self.artifact_hints = artifact_hints or []

    @classmethod
    def from_path(cls, path: Path, hints_path: Path = DEFAULT_ARTIFACT_HINTS_PATH) -> "AttackMapper":
        """Load the dataset (and the artifact hints) if present; never downloads anything."""
        hints = load_artifact_hints(hints_path)
        if not path.exists():
            reason = f"ATT&CK dataset file '{path.name}' not found; run scripts/fetch_attack_dataset.py"
        else:
            try:
                dataset = AttackDataset.from_stix_file(path)
            except AttackDatasetError as exc:
                reason = str(exc)
            else:
                log_event(
                    logger,
                    "ATT&CK dataset loaded",
                    operation="load_attack_dataset",
                    outcome="success",
                    attack_version=dataset.version,
                    technique_count=len(dataset.techniques),
                )
                return cls(dataset, artifact_hints=hints)
        log_event(
            logger,
            "ATT&CK dataset unavailable; technique mappings will be unverified",
            level=logging.WARNING,
            operation="load_attack_dataset",
            outcome="unavailable",
            error_category="attack_data_unavailable",
        )
        return cls(None, unavailable_reason=reason, artifact_hints=hints)

    def artifact_matches(self, text: str, limit: int = 8) -> list[ArtifactMatch]:
        """Techniques whose well-known artifacts occur in ``text`` (prompt hints only, like :meth:`technique_hints`).

        With a dataset, hints for unknown, deprecated or revoked techniques are dropped and the
        name and tactics come from the dataset.
        """
        found: dict[str, ArtifactMatch] = {}
        for hint in self.artifact_hints:
            match = hint.pattern.search(text)
            if match is None:
                continue
            if self.dataset is None:
                technique_id, record = hint.technique_ids[0], None
            else:
                technique_id = next((t for t in hint.technique_ids
                                     if t in self.dataset.techniques and self.dataset.techniques[t].active), None)  # fmt: skip
                if technique_id is None:
                    continue
                record = self.dataset.techniques[technique_id]
            if technique_id not in found:
                found[technique_id] = ArtifactMatch(technique_id, record.name if record else "",
                                                    record.tactics if record else (), hint.artifact,
                                                    match.group(0)[:120])  # fmt: skip
        return list(found.values())[:limit]

    def technique_hints(self, text: str, limit: int = 10) -> list[tuple[str, str]]:
        """Active techniques whose dataset name occurs as a whole phrase in ``text``.

        These are *prompt hints only* ("consider T1053.005 Scheduled Task"): the model still has to
        propose a technique and justify it from the incident, and the result is validated as usual.
        Longer (more specific) names come first. Empty without a dataset.
        """
        if self.dataset is None:
            return []
        haystack = canonical_text(text)
        found: list[tuple[str, str]] = []
        for record in self.dataset.techniques.values():
            name = " ".join(record.name.lower().split())
            if not record.active or len(name) < _MIN_HINT_NAME_CHARS or name not in haystack:
                continue
            if re.search(r"(?<![\w-])" + re.escape(name) + r"(?![\w-])", haystack):
                found.append((record.technique_id, record.name))
        found.sort(key=lambda item: (-len(item[1]), item[0]))
        return found[:limit]

    @property
    def mode(self) -> AttackMappingMode:
        return AttackMappingMode.VERIFIED if self.dataset else AttackMappingMode.UNVERIFIED

    @property
    def version(self) -> Optional[str]:
        return self.dataset.version if self.dataset else None

    def map_candidates(self, candidates: list[LLMTechnique], warnings: WarningCollector) -> list[AttackTechniqueMapping]:
        if self.dataset is None:
            warnings.info(
                "ATT&CK dataset not loaded: technique IDs are format-checked only and marked 'unverified'"
            )

        best: dict[str, LLMTechnique] = {}
        for candidate in candidates:
            technique_id = candidate.technique_id.strip().upper()
            if not _TECHNIQUE_ID_RE.match(technique_id):
                warnings.degrade(f"discarded malformed ATT&CK technique ID '{candidate.technique_id[:20]}'")
                continue
            if technique_id not in best or candidate.confidence > best[technique_id].confidence:
                best[technique_id] = candidate

        mappings = []
        for technique_id, candidate in best.items():
            justification = candidate.justification.strip() or None
            if self.dataset is None:
                mappings.append(
                    AttackTechniqueMapping(
                        technique_id=technique_id,
                        confidence=candidate.confidence,
                        justification=justification,
                        validation_status=AttackValidationStatus.UNVERIFIED,
                    )
                )
                continue

            record = self.dataset.techniques.get(technique_id)
            if record is None:
                warnings.degrade(
                    f"discarded ATT&CK technique '{technique_id}': not found in ATT&CK v{self.dataset.version}"
                )
                continue
            if record.revoked:
                status = AttackValidationStatus.REVOKED
                warnings.info(f"ATT&CK technique {technique_id} is revoked in ATT&CK v{self.dataset.version}")
            elif record.deprecated:
                status = AttackValidationStatus.DEPRECATED
                warnings.info(f"ATT&CK technique {technique_id} is deprecated in ATT&CK v{self.dataset.version}")
            else:
                status = AttackValidationStatus.VALIDATED
            mappings.append(
                AttackTechniqueMapping(
                    technique_id=technique_id,
                    name=record.name,
                    tactics=list(record.tactics),
                    confidence=candidate.confidence,
                    justification=justification,
                    validation_status=status,
                )
            )
        return sorted(mappings, key=lambda m: m.confidence, reverse=True)

    def status(self) -> dict:
        if self.dataset is None:
            return {
                "status": "unavailable",
                "mapping_mode": self.mode.value,
                "version": None,
                "technique_count": 0,
                "detail": self.unavailable_reason or "not loaded",
            }
        return {
            "status": "loaded",
            "mapping_mode": self.mode.value,
            "version": self.dataset.version,
            "technique_count": len(self.dataset.techniques),
            "detail": "technique IDs are validated against this dataset",
        }
