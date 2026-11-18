"""Sigma adapter: hunt for one ATT&CK technique with the Sigma rules tagged for it.

Rules come from a local directory (trusted, curated by the deployment). Each rule tagged
``attack.<technique>`` is converted to a Lucene query with pySigma's Elasticsearch backend and
run through the Elastic adapter, batched into ``_msearch`` calls. pySigma (>= 1.5) and PyYAML are
optional dependencies (``requirements-tools.txt``); without them, or without Elastic, the
operation is unavailable.

Sigma matches strings case-insensitively. Lucene does not, so rule values are lowercased before
conversion and the index must store strings lowercased (a ``keyword`` field with a lowercase
normalizer, as ``scripts/load_attack_data.py`` creates). Regular-expression values are left as written.

Field names. By default (``field_mapping="none"``) rule fields are queried as written, which suits the
paper's Windows recordings indexed with their raw field names. ``"auditbeat"`` (deployment option, not
part of paper 968) maps SigmaHQ's Linux rules onto Elastic Auditbeat's ECS events, where one event
carries a whole command (executable, full command line, arguments, user); see ``_AUDITBEAT``. Two
approximations: auditd ``a0``..``a9`` become "any argument" (``process.args``), losing their position,
and the auditd record ``type`` condition is dropped because Auditbeat merges a command's records.
Auditbeat does not lowercase strings, so values with capitals in the logs will not match.
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


FIELD_MAPPINGS = ("none", "auditbeat")

# Sigma (SigmaHQ Linux rules) -> Elastic Auditbeat 8.x auditd-module fields, per log source.
_AUDITBEAT = {
    "process_creation": {
        "Image": "process.executable", "CommandLine": "process.title", "User": "user.name",
        "CurrentDirectory": "process.working_directory", "LogonId": "auditd.session",
        "ParentImage": "process.parent.executable", "ParentCommandLine": "process.parent.title",
    },
    "file_event": {"TargetFilename": "file.path", "Image": "process.executable"},
    "network_connection": {
        "DestinationIp": "destination.ip", "DestinationPort": "destination.port",
        "DestinationHostname": "destination.domain", "Image": "process.executable",
    },
    # service: auditd. a0..a9 depend on the record type: EXECVE argv vs SYSCALL arguments (hex).
    "auditd": {
        "exe": "process.executable", "comm": "process.name", "SYSCALL": "auditd.data.syscall",
        "syscall": "auditd.data.syscall", "key": "tags", "name": "auditd.paths.name",
        "nametype": "auditd.paths.nametype", "euid": "user.effective.id", "uid": "user.id",
        "unit": "auditd.summary.object.primary", "proctitle": "process.title",
    },
}
_ARGS = [f"a{i}" for i in range(10)]

# Sigma "keywords" rules (no field) become unfielded Lucene terms, which query_string expands over every
# field of every index: with Auditbeat's thousands of mapped fields and dozens of leading wildcards that
# one query exhausted a 2 GB Elasticsearch heap. Under the Auditbeat mapping they search these instead:
# the raw log line (syslog/auth via Filebeat) and the full command line.
_UNFIELDED_TERMS_SEARCH = {"auditbeat": ["message", "process.title"]}


def _pipeline(field_mapping: str):
    from sigma.processing.conditions import (  # type: ignore[import-not-found]
        IncludeFieldCondition, LogsourceCondition, RuleContainsDetectionItemCondition)
    from sigma.processing.pipeline import ProcessingItem, ProcessingPipeline  # type: ignore[import-not-found]
    from sigma.processing.transformations import (  # type: ignore[import-not-found]
        AddConditionTransformation, CaseTransformation, DropDetectionItemTransformation, FieldMappingTransformation)

    items = []
    if field_mapping == "auditbeat":
        # pySigma binds each condition object to one processing item, so every item gets fresh ones.
        def auditd():
            return LogsourceCondition(product="linux", service="auditd")

        def execve():
            return RuleContainsDetectionItemCondition(field="type", value="EXECVE")

        for category in ("process_creation", "file_event", "network_connection"):
            items.append(ProcessingItem(identifier=f"cade_auditbeat_{category}",
                                        transformation=FieldMappingTransformation(_AUDITBEAT[category]),
                                        rule_conditions=[LogsourceCondition(product="linux", category=category)]))
        items += [
            ProcessingItem(identifier="cade_auditbeat_drop_initiated",  # no Auditbeat equivalent
                           transformation=DropDetectionItemTransformation(),
                           rule_conditions=[LogsourceCondition(product="linux", category="network_connection")],
                           field_name_conditions=[IncludeFieldCondition(fields=["Initiated"])]),
            ProcessingItem(identifier="cade_auditbeat_executed",
                           transformation=AddConditionTransformation({"event.action": "executed"}),
                           rule_conditions=[LogsourceCondition(product="linux", category="process_creation")]),
            # Before the type items are dropped: the EXECVE test needs them.
            ProcessingItem(identifier="cade_auditbeat_execve_args",
                           transformation=FieldMappingTransformation({a: "process.args" for a in _ARGS}),
                           rule_conditions=[auditd(), execve()], rule_condition_linking=all),
            ProcessingItem(identifier="cade_auditbeat_syscall_args",
                           transformation=FieldMappingTransformation({a: f"auditd.data.{a}" for a in _ARGS}),
                           rule_conditions=[auditd()]),
            ProcessingItem(identifier="cade_auditbeat_auditd_fields",
                           transformation=FieldMappingTransformation(_AUDITBEAT["auditd"]), rule_conditions=[auditd()]),
            ProcessingItem(identifier="cade_auditbeat_drop_record_type",
                           transformation=DropDetectionItemTransformation(), rule_conditions=[auditd()],
                           field_name_conditions=[IncludeFieldCondition(fields=["type"])]),
        ]
    items.append(ProcessingItem(identifier="cade_lowercase_values", transformation=CaseTransformation(method="lower")))
    return ProcessingPipeline(items=items)


def make_converter(field_mapping: str = "none"):
    """A rule-text -> Lucene queries converter; ``field_mapping`` is one of ``FIELD_MAPPINGS``."""
    if field_mapping not in FIELD_MAPPINGS:
        raise ValueError(f"unknown Sigma field mapping {field_mapping!r}; expected one of {FIELD_MAPPINGS}")

    def convert(rule_text: str) -> list[str]:
        try:
            from sigma.backends.elasticsearch import LuceneBackend  # type: ignore[import-not-found]
            from sigma.collection import SigmaCollection  # type: ignore[import-not-found]
            pipeline = _pipeline(field_mapping)
        except ImportError:
            raise ToolUnavailableError("pySigma (>= 1.5) with the Elasticsearch backend is not installed") from None
        try:
            return [str(q) for q in LuceneBackend(processing_pipeline=pipeline).convert(SigmaCollection.from_yaml(rule_text))]
        except Exception:  # noqa: BLE001 - pySigma raises many types for unsupported rules
            return []

    return convert


_default_converter = make_converter("none")


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

    def __init__(self, *, rules_dir: Path, elastic: Optional[ElasticTool], converter=None, tag_reader=None,
                 field_mapping: str = "none") -> None:  # fmt: skip
        super().__init__()
        self.rules_dir = rules_dir
        self.elastic = elastic
        self.field_mapping = field_mapping
        self._convert = converter or make_converter(field_mapping)
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
            options = {"fields": _UNFIELDED_TERMS_SEARCH[self.field_mapping]} if self.field_mapping in _UNFIELDED_TERMS_SEARCH else {}
            results = await self.elastic.search_many([[window, {"query_string": {"query": query, **options}}] for _, query in batch],
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
        return {"status": "configured", "elastic": self.elastic is not None, "field_mapping": self.field_mapping}

