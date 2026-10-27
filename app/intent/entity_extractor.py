"""Entity extraction: deterministic pattern matching merged with grounded LLM suggestions.

Sources, in order of trust:

1. ``structured``: observables explicitly supplied with the incident;
2. ``pattern``: IPs, hashes, domains and ``.exe`` names found by regular expressions, plus hosts and
   users named by the fields of Windows event records (``Computer=``, ``User=``,
   ``TargetUserName=``, ``SubjectUserName=``; paper 968 presents incidents as such records);
3. ``llm``: values suggested by the model, kept only if they occur in the incident text.

In free text hosts and users have no reliable pattern, so there they come from (1) and (3).
Built-in service identities (SYSTEM, LOCAL SERVICE, ...) and machine accounts are not users.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.intent.grounding import GroundingCorpus, WarningCollector
from app.intent.schemas import LLMEntities
from app.schemas.incident import Entities, FileHash, Observables
from app.schemas.indicators import canonical_domain, canonical_hash, canonical_ip, refang

_IPV4_RE = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
_IPV6_RE = re.compile(r"(?<![0-9A-Za-z:])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![0-9A-Za-z:])")
_HASH_RE = re.compile(r"(?<![0-9A-Fa-f])(?:[0-9A-Fa-f]{64}|[0-9A-Fa-f]{40}|[0-9A-Fa-f]{32})(?![0-9A-Fa-f])")
_DOMAIN_RE = re.compile(
    # "(?![\w@-])" stops the local part of an e-mail address ("j.doe@...") being read as a domain.
    r"(?<![\w.-])(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z][a-z0-9-]{1,62}(?![\w@-])", re.IGNORECASE
)
_PROCESS_RE = re.compile(r"(?<![\w.-])[\w.-]+\.exe\b", re.IGNORECASE)
# Windows event-record fields (Splunk / Winlogbeat style "Field=value")
_HOST_FIELD_RE = re.compile(r"\b(?:Computer|ComputerName|Hostname|Host|WorkstationName)=\"?([A-Za-z0-9][A-Za-z0-9._-]{0,252})")
_USER_FIELD_RE = re.compile(
    r"\b(?:User|UserName|TargetUserName|SubjectUserName|AccountName)=\"?(?!NT AUTHORITY\\)([A-Za-z0-9][\w.\\@$-]{0,127})"
)
_SERVICE_IDENTITIES = {"system", "local service", "network service", "anonymous logon", "-"}

# Tokens ending in these look like domains but are almost always file names. Some (zip, py)
# are also real TLDs; this trade-off favours fewer false-positive domains.
_FILE_EXTENSIONS = {
    "exe", "dll", "sys", "ps1", "psm1", "bat", "cmd", "vbs", "vbe", "js", "jse", "hta", "scr", "msi", "lnk",
    "iso", "img", "zip", "rar", "7z", "gz", "tar", "doc", "docx", "docm", "xls", "xlsx", "xlsm", "ppt", "pptx",
    "pdf", "txt", "log", "csv", "tmp", "py", "sh", "jar", "dat", "bin", "ini", "cfg", "conf", "json", "xml",
    "html", "htm", "php", "asp", "aspx", "jsp", "png", "jpg", "jpeg", "gif", "eml", "msg", "evtx", "pcap",
}  # fmt: skip


@dataclass
class PatternEntities:
    ips: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    file_hashes: list[str] = field(default_factory=list)
    processes: list[str] = field(default_factory=list)
    hosts: list[str] = field(default_factory=list)
    users: list[str] = field(default_factory=list)


def _is_person(user: str) -> bool:
    name = user.rsplit("\\", 1)[-1]
    return name.lower() not in _SERVICE_IDENTITIES and not name.endswith("$")


def extract_pattern_entities(text: str) -> PatternEntities:
    text = refang(text)
    found = PatternEntities()
    for candidate in _IPV4_RE.findall(text) + _IPV6_RE.findall(text):
        try:
            found.ips.append(canonical_ip(candidate))
        except ValueError:
            continue  # e.g. version strings like 999.1.1.1 or timestamps like 10:15:00
    found.file_hashes = [h.lower() for h in _HASH_RE.findall(text)]
    for candidate in _DOMAIN_RE.findall(text):
        if candidate.rsplit(".", 1)[-1].lower() in _FILE_EXTENSIONS:
            continue
        try:
            found.domains.append(canonical_domain(candidate))
        except ValueError:
            continue
    found.processes = _PROCESS_RE.findall(text)
    found.hosts = _merge(_HOST_FIELD_RE.findall(text))
    found.users = _merge([u for u in _USER_FIELD_RE.findall(text) if _is_person(u)])
    return found


def build_entities(
    observables: Observables,
    llm_entities: LLMEntities,
    incident_text: str,
    warnings: WarningCollector,
) -> Entities:
    corpus = GroundingCorpus(incident_text)
    pattern = extract_pattern_entities(incident_text)

    known_ips = set(observables.ips) | set(pattern.ips)
    known_hashes = set(observables.file_hashes) | set(pattern.file_hashes)
    known_domains = set(observables.domains) | set(pattern.domains)

    def grounded(
        kind: str, values: list[str], canonicalise=None, known: set[str] | None = None, *, is_domain: bool = False
    ) -> list[str]:
        kept = []
        for value in values:
            value = value.strip()
            if not value:
                continue
            try:
                canonical = canonicalise(value) if canonicalise else " ".join(value.split())
            except ValueError:
                warnings.degrade(f"discarded LLM-proposed {kind} with invalid format")
                continue
            if (known is not None and canonical in known) or corpus.contains_token(
                canonical, allow_email_local_part=not is_domain
            ):
                kept.append(canonical)
            else:
                warnings.degrade(f"discarded LLM-proposed {kind} '{_clip(value)}' not found in the incident text")
        return kept

    def exact(kind: str, values: list[str], canonicalise, known: set[str]) -> list[str]:
        """IPs and hashes must match an indicator actually present (no partial matches)."""
        kept = []
        for value in values:
            try:
                canonical = canonicalise(value)
            except ValueError:
                warnings.degrade(f"discarded LLM-proposed {kind} with invalid format")
                continue
            if canonical in known:
                kept.append(canonical)
            else:
                warnings.degrade(f"discarded LLM-proposed {kind} '{_clip(value)}' not found in the incident text")
        return kept

    hashes = _merge(
        observables.file_hashes,
        pattern.file_hashes,
        exact("file hash", llm_entities.file_hashes, lambda v: canonical_hash(v)[1], known_hashes),
    )
    return Entities(
        hosts=_merge(observables.hosts, pattern.hosts, grounded("host", llm_entities.hosts)),
        users=_merge(observables.users, pattern.users, grounded("user", llm_entities.users)),
        ips=_merge(observables.ips, pattern.ips, exact("IP address", llm_entities.ips, canonical_ip, known_ips)),
        domains=_merge(
            observables.domains,
            pattern.domains,
            grounded("domain", llm_entities.domains, canonical_domain, known_domains, is_domain=True),
        ),
        file_hashes=[FileHash(algorithm=canonical_hash(h)[0], value=h) for h in hashes],
        processes=_merge(observables.processes, pattern.processes, grounded("process", llm_entities.processes)),
    )


def _merge(*groups: list[str]) -> list[str]:
    seen: set[str] = set()
    merged = []
    for group in groups:
        for value in group:
            if value.lower() not in seen:
                seen.add(value.lower())
                merged.append(value)
    return merged


def _clip(value: str, limit: int = 60) -> str:
    return value if len(value) <= limit else value[: limit - 3] + "..."
