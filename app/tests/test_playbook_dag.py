"""Typed playbook DAGs: vocabulary, validity rules, the destructive-action filter and repair (milestone M3)."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import pytest

from app.core.config import PROJECT_ROOT
from app.core.errors import ConfigurationError
from app.playbook.builder import Draft
from app.playbook.repair import PlaybookRepairer
from app.playbook.validator import PlaybookValidator, ViolationCode, initial_artifacts
from app.playbook.vocabulary import Vocabulary
from app.schemas.enums import EvidenceCategory
from app.schemas.playbook import Playbook, PlaybookOrigin
from app.tools import artifacts as A
from app.tools.catalog import CATALOG

DATA = PROJECT_ROOT / "app" / "data"


@pytest.fixture(scope="module")
def vocabulary(taxonomy) -> Vocabulary:
    return Vocabulary.load(DATA / "action_vocabulary.json", DATA / "intent_goals.json", taxonomy)


@pytest.fixture(scope="module")
def validator(vocabulary) -> PlaybookValidator:
    return PlaybookValidator(vocabulary)


@pytest.fixture(scope="module")
def repairer(validator) -> PlaybookRepairer:
    return PlaybookRepairer(validator)


def seeds() -> list[dict]:
    return json.loads((DATA / "seed_playbooks.json").read_text(encoding="utf-8"))["playbooks"]


def pb(nodes: str, edges: str = "", intent: str = "ransomware_investigation") -> Playbook:
    """Compact constructor: nodes 'n1:action n2:action', edges 'n1>n2:artifact'."""
    node_list = [dict(zip(("node_id", "action"), item.split(":"))) for item in nodes.split()]
    edge_list = []
    for item in edges.split():
        pair, artifact = item.split(":")
        source, target = pair.split(">")
        edge_list.append({"source": source, "target": target, "artifact": artifact})
    return Playbook(name="test", intent_label=intent, origin=PlaybookOrigin.LLM, nodes=node_list, edges=edge_list)


VALID = ("n1:extract_entities n2:query_process_events n3:verify_host_compromise n4:isolate_host "
         "n5:generate_incident_report",
         "n1>n2:host_id n1>n3:host_id n2>n3:findings n1>n4:host_id n3>n4:verification_result n4>n5:findings")  # fmt: skip


# --- vocabulary ----------------------------------------------------------------------------


def test_vocabulary_bindings_and_goals(vocabulary, taxonomy):
    assert set(taxonomy.labels) <= set(vocabulary.goals)
    for action in vocabulary.actions.values():
        if action.binding_kind == "tool":
            assert action.binding in CATALOG and action.tool_target_type is not None
        if action.destructive:
            assert A.VERIFICATION_RESULT in action.inputs and vocabulary.verifications_for(action.target)
    for goals in vocabulary.goals.values():
        for goal in goals:
            assert all(not a.destructive for a in vocabulary.actions_for_goal(goal))


def _broken_vocabulary(tmp_path: Path, mutate) -> tuple[Path, Path]:
    data = json.loads((DATA / "action_vocabulary.json").read_text(encoding="utf-8"))
    goals = json.loads((DATA / "intent_goals.json").read_text(encoding="utf-8"))
    mutate(data, goals)
    (tmp_path / "a.json").write_text(json.dumps(data))
    (tmp_path / "g.json").write_text(json.dumps(goals))
    return tmp_path / "a.json", tmp_path / "g.json"


def _action(data, action_id):
    return next(a for a in data["actions"] if a["id"] == action_id)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d, g: _action(d, "isolate_host")["inputs"].remove("verification_result"),
        lambda d, g: _action(d, "isolate_host").update(destructive=False),  # disagrees with the tool catalog
        lambda d, g: _action(d, "query_process_events").update(binding="elastic.no_such_op"),
        lambda d, g: _action(d, "query_process_events")["outputs"].append("mystery_artifact"),
        lambda d, g: _action(d, "triage_alert")["outputs"].remove("findings"),
        lambda d, g: _action(d, "triage_alert").update(binding="cade.unknown_step"),
        lambda d, g: g["intents"]["denial_of_service"].append({"id": "x", "description": "x", "techniques": ["T9999"]}),
        lambda d, g: g["intents"]["denial_of_service"].append({"id": "y", "description": "y", "actions": ["isolate_host"]}),
        lambda d, g: g["intents"].pop("phishing_investigation"),
        lambda d, g: d["actions"].append(copy.deepcopy(d["actions"][0])),  # duplicate id
    ],
)
def test_invalid_vocabulary_files_are_rejected(tmp_path, taxonomy, mutate):
    actions, goals = _broken_vocabulary(tmp_path, mutate)
    with pytest.raises(ConfigurationError):
        Vocabulary.load(actions, goals, taxonomy)


# --- validity --------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", seeds(), ids=lambda s: s["intent_label"])
def test_every_seed_playbook_is_valid_and_covers_its_goals(seed, vocabulary, validator):
    playbook = Playbook.model_validate(seed)
    assert validator.validate(playbook).valid
    goals = vocabulary.goals_for(playbook.intent_label)
    assert all(any(vocabulary.addresses(vocabulary.get(n.action), g) for n in playbook.nodes) for g in goals)


def test_reference_playbook_is_valid(validator):
    assert validator.validate(pb(*VALID)).valid


@pytest.mark.parametrize(
    "nodes, edges, code",
    [
        # unmet input: process events need a host_id nobody supplies
        ("n1:query_process_events n2:generate_incident_report", "n1>n2:findings", ViolationCode.UNMET_INPUT),
        # type mismatch: extract_entities does not output process_list
        ("n1:extract_entities n2:analyze_process_tree n3:generate_incident_report",
         "n1>n2:process_list n2>n3:findings", ViolationCode.TYPE_MISMATCH),
        # cycle
        ("n1:extract_entities n2:assess_scope n3:notify_stakeholders n4:generate_incident_report",
         "n1>n2:findings n2>n3:findings n3>n2:findings n2>n4:findings", ViolationCode.CYCLE),
        # no terminal action
        ("n1:extract_entities n2:query_process_events", "n1>n2:host_id", ViolationCode.NO_TERMINAL),
        # orphan: n3 does not lead to the report
        ("n1:extract_entities n2:triage_alert n3:query_dns_activity n4:generate_incident_report",
         "n1>n3:host_id n1>n4:findings n2>n4:findings", ViolationCode.ORPHAN),
        # terminal with a successor
        ("n1:extract_entities n2:generate_incident_report n3:generate_incident_report",
         "n1>n2:findings n2>n3:findings", ViolationCode.TYPE_MISMATCH),
        # destructive without verification: isolate_host fed by a reputation check's verification of an *IP*
        ("n1:extract_entities n2:ip_reputation n3:verify_malicious_indicator n4:isolate_host n5:generate_incident_report",
         "n1>n2:ip_address n2>n3:reputation_report n1>n4:host_id n3>n4:verification_result n4>n5:findings",
         ViolationCode.UNVERIFIED_DESTRUCTIVE),
        ("n1:extract_entities n2:no_such_action n3:generate_incident_report", "n1>n3:findings", ViolationCode.UNKNOWN_ACTION),
        ("n1:extract_entities n1:triage_alert n2:generate_incident_report", "n1>n2:findings", ViolationCode.DUPLICATE_NODE),
        ("n1:extract_entities n2:generate_incident_report", "n1>n2:findings n1>n2:findings", ViolationCode.DUPLICATE_EDGE),
        ("n1:extract_entities n2:generate_incident_report", "n1>n9:findings n1>n2:findings", ViolationCode.UNKNOWN_EDGE_NODE),
        ("n1:assess_scope n2:generate_incident_report", "n1>n1:findings n1>n2:findings", ViolationCode.SELF_LOOP),
    ],
)  # fmt: skip
def test_each_validity_rule(validator, nodes, edges, code):
    report = validator.validate(pb(nodes, edges))
    assert not report.valid
    assert code in report.codes()


def test_unverified_destructive_is_a_safety_violation(validator):
    nodes = "n1:extract_entities n2:isolate_host n3:generate_incident_report"
    report = validator.validate(pb(nodes, "n1>n2:host_id n2>n3:findings"))
    assert ViolationCode.UNVERIFIED_DESTRUCTIVE in report.codes() and not report.safe


def test_invalid_params_and_node_cap(vocabulary):
    playbook = pb(*VALID)
    playbook.nodes[1].params = {"lookback_hours": 12345}
    assert ViolationCode.INVALID_PARAMS in PlaybookValidator(vocabulary).validate(playbook).codes()
    assert ViolationCode.TOO_MANY_NODES in PlaybookValidator(vocabulary, node_cap=4).validate(pb(*VALID)).codes()


def test_initial_artifacts_never_include_gating_artifacts(validator):
    assert initial_artifacts([A.HOST_ID, A.VERIFICATION_RESULT, A.CONTAINMENT_STATUS]) == {A.INCIDENT_CONTEXT, A.HOST_ID}
    # Entity artifacts of the incident can satisfy inputs directly...
    direct = pb("n1:query_process_events n2:generate_incident_report", "n1>n2:findings")
    assert validator.validate(direct, available=[A.HOST_ID]).valid
    # ...but a verification result can never be assumed.
    unsafe = pb("n1:isolate_host n2:generate_incident_report", "n1>n2:findings")
    assert not validator.validate(unsafe, available=[A.HOST_ID, A.VERIFICATION_RESULT]).valid


def test_structural_hash_ignores_ids_and_order_but_not_wiring():
    a = pb(*VALID)
    renamed = Playbook.model_validate_json(
        a.model_dump_json().replace('"n1"', '"x1"').replace('"n2"', '"x2"').replace('"n3"', '"x3"')
    )
    shuffled = a.model_copy(update={"nodes": list(reversed(a.nodes)), "edges": list(reversed(a.edges))})
    assert a.structural_hash() == renamed.structural_hash() == shuffled.structural_hash()
    rewired = pb(VALID[0], VALID[1].replace("n2>n3:findings", "n1>n3:findings"))
    assert rewired.structural_hash() != a.structural_hash()


def test_required_evidence_partitions_critical_and_supporting(vocabulary):
    playbook = Playbook.model_validate(next(s for s in seeds() if s["intent_label"] == "suspicious_process_execution"))
    required = vocabulary.required_evidence(playbook)
    assert EvidenceCategory.PROCESS_TELEMETRY in required.critical
    assert EvidenceCategory.INDICATORS_OF_COMPROMISE in required.critical
    assert EvidenceCategory.COMMAND_LINE in required.supporting
    assert not set(required.critical) & set(required.supporting)


# --- repair ----------------------------------------------------------------------------------


def test_valid_playbook_is_returned_unchanged(repairer):
    playbook = pb(*VALID)
    assert repairer.repair(playbook) == playbook


def test_repair_retypes_breaks_cycles_and_wires_inputs(repairer, validator):
    broken = pb(
        "n1:extract_entities n2:query_process_events n3:analyze_process_tree n4:assess_scope n5:notify_stakeholders",
        "n1>n2:findings n2>n3:process_list n4>n5:findings n5>n4:findings",  # mistyped, cyclic, no report
    )
    repaired = repairer.repair(broken)
    assert repaired is not None and validator.validate(repaired).valid
    assert any(e.source == "n1" and e.target == "n2" and e.artifact == "host_id" for e in repaired.edges)
    assert "generate_incident_report" in repaired.action_multiset()


def test_repair_inserts_producers_for_unmet_inputs(repairer, validator):
    repaired = repairer.repair(pb("n1:analyze_process_tree"))
    assert repaired is not None and validator.validate(repaired).valid
    actions = repaired.action_multiset()
    assert "analyze_process_tree" in actions and any(a in actions for a in
                                                     ("query_process_events", "inspect_process_inventory", "live_process_listing"))  # fmt: skip


def test_repair_adds_the_right_verification_before_destructive_actions(repairer, validator, vocabulary):
    repaired = repairer.repair(pb("n1:extract_entities n2:isolate_host", "n1>n2:host_id"))
    assert repaired is not None and validator.validate(repaired).valid
    isolate = next(n for n in repaired.nodes if n.action == "isolate_host")
    verifiers = validator.verification_ancestors(repaired, isolate.node_id)
    assert verifiers and all("host_id" in vocabulary.get(repaired.node(v).action).verifies for v in verifiers)


def test_repair_drops_unknown_actions_and_discards_hopeless_candidates(repairer):
    repaired = repairer.repair(pb("n1:extract_entities n2:launch_missiles n3:generate_incident_report",
                                  "n1>n3:findings"))  # fmt: skip
    assert repaired is not None and "launch_missiles" not in repaired.action_multiset()
    assert repairer.repair(pb("n1:launch_missiles")) is None


def test_repair_respects_the_node_cap(vocabulary):
    validator = PlaybookValidator(vocabulary, node_cap=8)
    many = pb(" ".join(f"n{i}:triage_alert" for i in range(1, 15)) + " n15:generate_incident_report",
              " ".join(f"n{i}>n15:findings" for i in range(1, 15)))  # fmt: skip
    repaired = PlaybookRepairer(validator).repair(many)
    assert repaired is not None and len(repaired.nodes) <= 8 and validator.validate(repaired).valid


def test_repair_never_adds_destructive_actions_and_always_returns_valid(repairer, validator, vocabulary):
    rng = random.Random(7)
    action_ids = sorted(vocabulary.actions)
    artifacts = sorted(A.ARTIFACT_TYPES)
    for _ in range(150):
        count = rng.randint(1, 10)
        nodes = [{"node_id": f"n{i}", "action": rng.choice(action_ids)} for i in range(count)]
        edges = [{"source": f"n{rng.randrange(count)}", "target": f"n{rng.randrange(count)}", "artifact": rng.choice(artifacts)}
                 for _ in range(rng.randint(0, 12))]  # fmt: skip
        garbage = Playbook(name="g", intent_label="unknown", origin=PlaybookOrigin.LLM, nodes=nodes, edges=edges)
        repaired = repairer.repair(garbage, rng=random.Random(1))
        if repaired is None:
            continue
        assert validator.validate(repaired).valid
        destructive = lambda p: sum(1 for n in p.nodes if vocabulary.get(n.action) and vocabulary.get(n.action).destructive)  # noqa: E731
        assert destructive(repaired) <= destructive(garbage)


def test_draft_wiring_never_creates_cycles(vocabulary):
    draft = Draft.from_playbook(vocabulary, pb(*VALID))
    assert not draft.add_edge("n5", "n1", "findings")  # wrong types and would close a cycle
    assert not draft.add_edge("n4", "n2", "findings")  # findings is not an input of query_process_events
    assert not draft.add_edge("n2", "n2", "host_id")
