"""Pure helpers for canonicalising indicators (IPs, hashes, domains).

Analysts and threat-intel reports often "defang" indicators so they cannot be clicked,
e.g. ``198.51.100[.]7`` or ``hxxp://evil[.]example``. ``refang`` reverses that for
*analysis only*; the raw incident text is always stored unchanged.
"""

from __future__ import annotations

import ipaddress
import re

from app.schemas.enums import HashAlgorithm

_REFANG_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\[\.\]|\(\.\)|\{\.\}|\[dot\]|\(dot\)", re.IGNORECASE), "."),
    (re.compile(r"\[:\]"), ":"),
    (re.compile(r"\[@\]|\[at\]", re.IGNORECASE), "@"),
    (re.compile(r"\bhxxp", re.IGNORECASE), "http"),
    (re.compile(r"\bfxp", re.IGNORECASE), "ftp"),
]

_HASH_LENGTHS = {32: HashAlgorithm.MD5, 40: HashAlgorithm.SHA1, 64: HashAlgorithm.SHA256}
_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")
_DOMAIN_RE = re.compile(r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z][a-z0-9-]{1,62}$")


def refang(text: str) -> str:
    for pattern, replacement in _REFANG_RULES:
        text = pattern.sub(replacement, text)
    return text


def canonical_ip(value: str) -> str:
    """Return the canonical string form of an IPv4/IPv6 address or raise ValueError."""
    return str(ipaddress.ip_address(refang(value.strip())))


def canonical_hash(value: str) -> tuple[HashAlgorithm, str]:
    """Return (algorithm, lowercase hex) for an MD5/SHA-1/SHA-256 digest or raise ValueError."""
    cleaned = value.strip()
    algorithm = _HASH_LENGTHS.get(len(cleaned))
    if algorithm is None or not _HEX_RE.match(cleaned):
        raise ValueError("expected an MD5 (32), SHA-1 (40) or SHA-256 (64) hex digest")
    return algorithm, cleaned.lower()


def canonical_domain(value: str) -> str:
    """Return a lowercase domain name (without trailing dot) or raise ValueError."""
    cleaned = refang(value.strip()).lower().rstrip(".")
    if not _DOMAIN_RE.match(cleaned):
        raise ValueError("not a valid domain name")
    return cleaned
