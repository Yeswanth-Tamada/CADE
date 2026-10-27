"""The controlled action vocabulary and the static per-intent goal mapping.

Both are data files (``app/data/action_vocabulary.json``, ``app/data/intent_goals.json``) and are
validated thoroughly at load time -- tool bindings against the tool catalog, artifact types,
safety flags, goal satisfiability -- so a bad edit fails at startup, not mid-incident.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.core.errors import ConfigurationError
from app.schemas.enums import Criticality, EvidenceCategory
from app.schemas.playbook import ACTION_ID_PATTERN, Phase, Playbook, RequiredEvidence
from app.schemas.taxonomy import IntentTaxonomy
from app.schemas.tools import OperationSpec
from app.tools import artifacts as A
from app.tools.catalog import CATALOG, TOOLS

# CADE's own deterministic steps (implemented by the executor) and human tasks.
INTERNAL_OPERATIONS = frozenset({
    "extract_entities", "triage_alert", "analyze_process_tree", "analyze_network_flows", "analyze_auth_patterns",
    "analyze_email", "select_suspicious_files", "assess_scope", "verify_reputation", "generate_report",
})  # fmt: skip
INTERNAL_PREFIX = "cade"
ANALYST_PREFIX = "analyst"


class EvidenceUse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    category: EvidenceCategory
    critical: bool = False


class ActionDefinition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=ACTION_ID_PATTERN)
    name: str
    phase: Phase
    binding: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    consumes: tuple[EvidenceUse, ...] = ()
    techniques: tuple[str, ...] = ()
    destructive: bool = False
    verification: bool = False
    verifies: tuple[str, ...] = ()
    target: Optional[str] = None
    terminal: bool = False
    description: str = ""

    @property
    def binding_kind(self) -> str:
        """``tool``, ``internal`` or ``analyst``."""
        prefix = self.binding.split(".", 1)[0]
        return "tool" if prefix in TOOLS else "internal" if prefix == INTERNAL_PREFIX else "analyst"

    @property
    def operation(self) -> str:
        return self.binding.split(".", 1)[1]

    @property
    def operation_spec(self) -> Optional[OperationSpec]:
        return CATALOG.get(self.binding) if self.binding_kind == "tool" else None

    @property
    def tool_target_type(self) -> Optional[str]:
        spec = self.operation_spec
        return next((t for t in self.inputs if t in spec.accepts), None) if spec else None

    @property
    def params_spec(self) -> dict[str, tuple[Any, ...]]:
        spec = self.operation_spec
        return dict(spec.params) if spec else {}

    @property
    def categories(self) -> frozenset[EvidenceCategory]:
        return frozenset(use.category for use in self.consumes)


class IntentGoal(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=ACTION_ID_PATTERN)
    description: str
    actions: tuple[str, ...] = ()
    techniques: tuple[str, ...] = ()


class _VocabularyFile(BaseModel):
    version: str
    note: str = ""
    phases: list[Phase]
    actions: list[ActionDefinition] = Field(min_length=1)


class _GoalsFile(BaseModel):
    version: str
    note: str = ""
    intents: dict[str, list[IntentGoal]]


class Vocabulary:
    def __init__(self, version: str, actions: list[ActionDefinition], goals: dict[str, list[IntentGoal]],
                 goals_version: str = "") -> None:  # fmt: skip
        if len({a.id for a in actions}) != len(actions):
            raise ValueError("duplicate action ids")
        self.version = version
        self.goals_version = goals_version
        self.actions: dict[str, ActionDefinition] = {a.id: a for a in actions}
        self.goals = goals
        self._check()
        self.producers: dict[str, list[ActionDefinition]] = {}
        for action in actions:
            for artifact in action.outputs:
                self.producers.setdefault(artifact, []).append(action)

    # --- queries ---------------------------------------------------------------------------

    def get(self, action_id: str) -> Optional[ActionDefinition]:
        return self.actions.get(action_id)

    def goals_for(self, intent_label: str) -> list[IntentGoal]:
        return self.goals.get(intent_label, [])

    @staticmethod
    def addresses(action: ActionDefinition, goal: IntentGoal) -> bool:
        return action.id in goal.actions or bool(set(action.techniques) & set(goal.techniques))

    def actions_for_goal(self, goal: IntentGoal, *, include_destructive: bool = False) -> list[ActionDefinition]:
        return [a for a in self.actions.values()
                if self.addresses(a, goal) and (include_destructive or not a.destructive)]  # fmt: skip

    def actions_consuming(self, category: EvidenceCategory) -> list[ActionDefinition]:
        return [a for a in self.actions.values() if category in a.categories and not a.destructive]

    def verifications_for(self, target: str) -> list[ActionDefinition]:
        return [a for a in self.actions.values() if a.verification and target in a.verifies]

    def terminal_actions(self) -> list[ActionDefinition]:
        return [a for a in self.actions.values() if a.terminal]

    def resolve_params(self, action: ActionDefinition, params: Optional[dict[str, Any]]) -> dict[str, Any]:
        """Defaults for omitted params; raises ValueError for unknown names / disallowed values."""
        spec = action.operation_spec
        if spec is None:
            if params:
                raise ValueError(f"action '{action.id}' takes no parameters")
            return {}
        return spec.resolve_params(params)

    def required_evidence(self, playbook: Playbook, critical_categories: frozenset[EvidenceCategory] = frozenset()
                          ) -> RequiredEvidence:  # fmt: skip
        """R_c / R_s of a playbook: categories its actions consume; critical if any action marks it
        critical or the incident's intent marks it critical in the ontology (``critical_categories``)."""
        critical: set[EvidenceCategory] = set()
        supporting: set[EvidenceCategory] = set()
        for node in playbook.nodes:
            action = self.get(node.action)
            for use in action.consumes if action else ():
                (critical if use.critical or use.category in critical_categories else supporting).add(use.category)
        key = lambda c: c.value  # noqa: E731
        return RequiredEvidence(critical=sorted(critical, key=key), supporting=sorted(supporting - critical, key=key))

    # --- loading & validation -------------------------------------------------------------

    @classmethod
    def load(cls, actions_path: Path, goals_path: Path, taxonomy: Optional[IntentTaxonomy] = None) -> "Vocabulary":
        try:
            actions_file = _VocabularyFile.model_validate(json.loads(actions_path.read_text(encoding="utf-8")))
            goals_file = _GoalsFile.model_validate(json.loads(goals_path.read_text(encoding="utf-8")))
            vocabulary = cls(actions_file.version, actions_file.actions, goals_file.intents, goals_file.version)
        except FileNotFoundError as exc:
            raise ConfigurationError(f"playbook vocabulary file not found: {exc.filename}") from None
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            raise ConfigurationError(f"invalid playbook vocabulary or goal file: {exc}") from None
        if taxonomy is not None:
            missing = set(taxonomy.labels) - set(vocabulary.goals)
            if missing:
                raise ConfigurationError(f"intent goals missing for: {', '.join(sorted(missing))}")
        return vocabulary

    def _check(self) -> None:
        for action in self.actions.values():
            self._check_action(action)
        if not self.terminal_actions():
            raise ValueError("the vocabulary needs at least one terminal action")
        for intent, goals in self.goals.items():
            if not goals:
                raise ValueError(f"intent '{intent}' has no goals")
            for goal in goals:
                for action_id in goal.actions:
                    action = self.get(action_id)
                    if action is None or action.destructive:
                        raise ValueError(f"goal {intent}.{goal.id} names unknown or destructive action '{action_id}'")
                if not self.actions_for_goal(goal):
                    raise ValueError(f"goal {intent}.{goal.id} cannot be addressed by any non-destructive action")

    @staticmethod
    def _check_action(action: ActionDefinition) -> None:
        name = action.id
        for artifact in action.inputs + action.outputs + action.verifies:
            if artifact not in A.ARTIFACT_TYPES:
                raise ValueError(f"action '{name}' uses unknown artifact type '{artifact}'")
        if not action.inputs or not action.outputs:
            raise ValueError(f"action '{name}' needs inputs and outputs")
        if not action.terminal and A.FINDINGS not in action.outputs:
            raise ValueError(f"non-terminal action '{name}' must output '{A.FINDINGS}'")
        if action.terminal and (action.destructive or action.verification):
            raise ValueError(f"terminal action '{name}' cannot be destructive or a verification")
        if action.destructive:
            if action.verification or not action.target or action.target not in action.inputs:
                raise ValueError(f"destructive action '{name}' needs a target among its inputs")
            if A.VERIFICATION_RESULT not in action.inputs:
                raise ValueError(f"destructive action '{name}' must take a '{A.VERIFICATION_RESULT}' input")
        if action.verification and (A.VERIFICATION_RESULT not in action.outputs or not action.verifies):
            raise ValueError(f"verification action '{name}' must output a verification_result and say what it verifies")
        kind = action.binding_kind
        prefix, _, operation = action.binding.partition(".")
        if not operation:
            raise ValueError(f"action '{name}' has a malformed binding")
        if kind == "internal" and operation not in INTERNAL_OPERATIONS:
            raise ValueError(f"action '{name}' binds unknown internal operation '{operation}'")
        if kind == "analyst" and prefix != ANALYST_PREFIX:
            raise ValueError(f"action '{name}' binds unknown tool '{prefix}'")
        if kind == "tool":
            spec = CATALOG.get(action.binding)
            if spec is None:
                raise ValueError(f"action '{name}' binds unknown tool operation '{action.binding}'")
            targets = [t for t in action.inputs if t in spec.accepts]
            if len(targets) != 1:
                raise ValueError(f"action '{name}' must take exactly one target accepted by {spec.key}")
            if not set(spec.requires) <= set(action.inputs):
                raise ValueError(f"action '{name}' lacks inputs required by {spec.key}")
            if not set(action.outputs) - {A.FINDINGS} <= set(spec.produces):
                raise ValueError(f"action '{name}' declares outputs {spec.key} does not produce")
            if spec.destructive != action.destructive:
                raise ValueError(f"action '{name}' and {spec.key} disagree on being destructive")


def criticality_of(use: EvidenceUse) -> Criticality:
    return Criticality.CRITICAL if use.critical else Criticality.SUPPORTING
