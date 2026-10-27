"""Checks that model output is supported by the incident input.

Prompting alone cannot stop a model from inventing an IP address or quoting text that was
never there, so every entity and every "observed" excerpt is checked against the incident
text here, in code.
"""

from __future__ import annotations

import re

from app.schemas.indicators import refang

# Models frequently JSON-escape Windows paths twice ("C:\\Users" for "C:\Users"), and the
# rendered incident shows structured fields via json.dumps (which doubles backslashes too).
# Collapsing runs of backslashes lets an exact path quote match without matching other paths.
_BACKSLASH_RUN_RE = re.compile(r"\\{2,}")


def canonical_text(text: str) -> str:
    """Case-, whitespace-, defang- and backslash-escaping-insensitive form used for grounding."""
    return " ".join(_BACKSLASH_RUN_RE.sub(r"\\", refang(text)).lower().split())


_canon = canonical_text


class GroundingCorpus:
    """Case-, whitespace-, defang- and backslash-escaping-insensitive view of the incident text."""

    MIN_LENGTH = 2

    def __init__(self, text: str) -> None:
        self._canonical = _canon(text)

    def contains(self, value: str) -> bool:
        needle = _canon(value).strip(" \"'`")
        return len(needle) >= self.MIN_LENGTH and needle in self._canonical

    def contains_token(self, value: str, *, allow_email_local_part: bool = True) -> bool:
        """True if ``value`` occurs as a whole token, so 'WS-01' does not match inside 'WS-012'.

        With ``allow_email_local_part=False`` an occurrence directly followed by '@' does not
        count (used for domains, so 'j.doe' in 'j.doe@corp.example' is not a domain).
        """
        needle = _canon(value).strip(" \"'`")
        if len(needle) < self.MIN_LENGTH:
            return False
        stop = r"(?![\w-]|\.\w)" if allow_email_local_part else r"(?![\w@-]|\.\w)"
        pattern = r"(?<![\w.-])" + re.escape(needle) + stop
        return re.search(pattern, self._canonical) is not None

    def contains_excerpt(self, excerpt: str) -> bool:
        """Like :meth:`contains`, tolerating leading/trailing ellipses the model may add."""
        cleaned = excerpt.strip().strip(".…").strip()
        return len(cleaned) >= 3 and self.contains(cleaned)


class WarningCollector:
    """Collects analysis warnings; ``degraded`` warnings make the extraction ``partial``."""

    def __init__(self) -> None:
        self.messages: list[str] = []
        self.degraded = False

    def info(self, message: str) -> None:
        if message not in self.messages:
            self.messages.append(message)

    def degrade(self, message: str) -> None:
        self.degraded = True
        self.info(message)
