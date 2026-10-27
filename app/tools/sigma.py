"""Sigma adapter: hunt for one ATT&CK technique with the Sigma rules tagged for it.

Rules come from a local directory (trusted, curated by the deployment). Each rule tagged
``attack.<technique>`` is converted to a Lucene query with pySigma's Elasticsearch backend and
run through the Elastic adapter, batched into ``_msearch`` calls. pySigma (>= 1.5) and PyYAML are
optional dependencies (``requirements-tools.txt``); without them, or without Elastic, the
operation is unavailable.

Sigma matches strings case-insensitively. Lucene does not, so rule values are lowercased before
conversion and the index must store strings lowercased (a ``keyword`` field with a lowercase
normalizer, as ``scripts/load_attack_data.py`` creates). Regular-expression values are left as written.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from app.core.errors import ToolUnavailableError
from app.schemas.enums import ToolStatus
from app.schemas.tools import OperationSpec, ToolObservation, ToolResult
from app.tools import artifacts as A
from app.tools.base import SecurityTool
from app.tools.elastic import ElasticTool

# The largest technique in SigmaHQ's Windows rules (T1059.001) has 181 rules; past the cap, rules are
# taken in path order.
_MAX_RULES = 250
_SEARCHES_PER_CALL = 50


def _default_converter(rule_text: str) -> list[str]:
    try:
        from sigma.backends.elasticsearch import LuceneBackend  # type: ignore[import-not-found]
        from sigma.collection import SigmaCollection  # type: ignore[import-not-found]
        from sigma.processing.pipeline import ProcessingItem, ProcessingPipeline  # type: ignore[import-not-found]
        from sigma.processing.transformations import CaseTransformation  # type: ignore[import-not-found]
    except ImportError:
        raise ToolUnavailableError("pySigma (>= 1.5) with the Elasticsearch backend is not installed") from None
    lowercase = ProcessingPipeline(items=[ProcessingItem(identifier="cade_lowercase_values",
                                                         transformation=CaseTransformation(method="lower"))])  # fmt: skip
    try:
        return [str(q) for q in LuceneBackend(processing_pipeline=lowercase).convert(SigmaCollection.from_yaml(rule_text))]
    except Exception:  # noqa: BLE001 - pySigma raises many types for unsupported rules
        return []


def _default_tags(rule_text: str) -> list[str]:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        raise ToolUnavailableError("PyYAML is not installed") from None
    try:
        document = yaml.safe_load(rule_text) or {}
    except yaml.YAMLError:
        return []
    return [str(t).lower() for t in (document.get("tags") or [])] if isinstance(document, dict) else []


class SigmaTool(SecurityTool):
    name = "sigma"

    def __init__(self, *, rules_dir: Path, elastic: Optional[ElasticTool], converter=None, tag_reader=None) -> None:
        super().__init__()
        self.rules_dir = rules_dir
        self.elastic = elastic
        self._convert = converter or _default_converter
        self._tags = tag_reader or _default_tags

    def _rules_for(self, technique: str) -> list[tuple[str, str]]:
        wanted = f"attack.{technique.lower()}"
        found = []
        for path in sorted(self.rules_dir.rglob("*.yml"))[:5000]:
            text = path.read_text(encoding="utf-8", errors="replace")
            if wanted in self._tags(text):
                found.append((path.stem, text))
            if len(found) >= _MAX_RULES:
                break
        return found

    async def run(
        self, spec: OperationSpec, target_type: str, target: str, context: dict[str, str], params: dict[str, Any]
    ) -> ToolResult:
        if self.elastic is None:
            raise ToolUnavailableError("Sigma hunting needs the Elastic adapter to be configured")
        if not self.rules_dir.is_dir():
            raise ToolUnavailableError("the Sigma rules directory does not exist")
        rules = self._rules_for(target)
        if not rules:
            return self.result(spec, target_type, status=ToolStatus.NO_DATA,
                               message="no Sigma rule is tagged with this technique")  # fmt: skip
        searches = [(rule_name, query) for rule_name, text in rules for query in self._convert(text)]
        unconverted = len(rules) - len({rule_name for rule_name, _ in searches})
        if not searches:
            return self.result(spec, target_type, status=ToolStatus.NO_DATA,
                               message=f"none of the {len(rules)} Sigma rule(s) for this technique could be converted")  # fmt: skip
        window = {"range": {"@timestamp": {"gte": f"now-{params['lookback_hours']}h"}}}
        hits, rejected = [], 0
        for start in range(0, len(searches), _SEARCHES_PER_CALL):
            batch = searches[start:start + _SEARCHES_PER_CALL]
            results = await self.elastic.search_many([[window, {"query_string": {"query": query}}] for _, query in batch],
                                                     10, ("@timestamp", "host.name", "event.category"))  # fmt: skip
            for (rule_name, _), rows in zip(batch, results):
                if rows is None:
                    rejected += 1
                    continue
                hits.extend({"rule": rule_name, **row} for row in rows)
        skipped = "".join([f"; {unconverted} could not be converted" if unconverted else "",
                           f"; Elastic rejected {rejected} quer{'y' if rejected == 1 else 'ies'}" if rejected else ""])  # fmt: skip
        if not hits:
            summary = f"Sigma: {len(rules)} rule(s) for {target} ran with no hits{skipped}"
            return self.result(spec, target_type, outputs={A.SIGMA_HITS: []}, findings=[summary],
                               observations=[ToolObservation(category=c, description=summary)
                                             for c in spec.categories])  # fmt: skip
        summary = (f"Sigma: {len(hits)} hit(s) from {len({h['rule'] for h in hits})} of {len(rules)} rule(s) "
                   f"for {target}{skipped}")  # fmt: skip
        return self.result(spec, target_type, outputs={A.SIGMA_HITS: hits[:100]}, findings=[summary],
                           observations=[ToolObservation(category=c, description=summary) for c in spec.categories])

    async def health(self) -> dict[str, Any]:
        if not self.rules_dir.is_dir():
            return {"status": "misconfigured", "detail": "rules directory missing"}
        return {"status": "configured", "elastic": self.elastic is not None}

