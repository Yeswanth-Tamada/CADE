"""The online routing protocol of paper 968 (Sections 4.2-4.3, 6.2 and 6.4), shared by every routing benchmark.

For each ordering (seeds 0-9) the scenarios are shuffled and a *fresh* learner -- the production
:class:`~app.routing.learning.RoutingLearner` with its LinUCB policy -- processes them online: the
hand-designed rule (Eq. 7) chooses until ``bootstrap`` incidents have been rewarded, every response
is credited with the Eq. 9 reward from the first incident on, and LinUCB (Eq. 8) chooses afterwards.
The rule alone is evaluated on the same scenarios, orderings and reward assumptions.

An environment supplies the scenarios and the *simulated* outcome of each response (IC_exec,
retention, latency); nothing here executes a playbook. Reward bookkeeping:

* L_max is the 90th-percentile latency of the ordering's first ``bootstrap`` responses. Both policies
  follow the rule there, so it is the same L_max for both (checked).
* The learner's training reward of every first decision equals the reported reward (checked).
* A response's reward is the Eq. 9 reward of its first decision: the terminal reward, less
  lambda_q * c_q when the first decision requested evidence.
* Latencies are the paper's Fig. 4b means (:class:`PathLatency`): assumed, not measured.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Protocol

from app.core.config import Settings
from app.evaluation import metrics as M
from app.routing.bandit import LinUCBPolicy
from app.routing.learning import RoutingLearner
from app.routing.reward import RewardWeights, percentile, terminal_reward
from app.schemas.routing import RoutingAction, RoutingDecision, StoredRoutingDecision
from app.storage.bandit_repository import BanditRepository
from app.storage.database import Database

REUSE, REQUEST, GENERATE = RoutingAction.REUSE, RoutingAction.REQUEST_EVIDENCE, RoutingAction.SYNTHESIZE
ACTIONS = (REUSE, REQUEST, GENERATE)
_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)
_TOLERANCE = 1e-9

SIMULATED_EXECUTION = "SIMULATED: a construction-based stand-in decides IC_exec and retention; no playbook was executed"


@dataclass(frozen=True)
class PathLatency:
    """Assumed end-to-end latency of each path: the paper's Fig. 4b means, not measurements of this code.

    Fig. 4b's request path (8.4 s) already contains retrieval, the evidence wait and executing the
    reused playbook, so a request followed by synthesis costs synthesis plus the 2.4 s evidence wait.
    ``additive_request`` reproduces the earlier evaluation, which charged 8.4 s on top of the terminal path.
    """

    reuse: float = 6.0
    request: float = 8.4
    synthesize: float = 238.0
    additive_request: bool = False

    def of(self, first: RoutingAction, terminal: RoutingAction) -> float:
        base = self.reuse if terminal == REUSE else self.synthesize
        if first != REQUEST:
            return base
        return base + (self.request if self.additive_request else self.request - self.reuse)

    def describe(self) -> str:
        request = (f"request adds {self.request} s to the terminal path (legacy accounting)" if self.additive_request
                   else f"request->reuse {self.request} s, request->synthesis {self.synthesize + self.request - self.reuse} s")  # fmt: skip
        return f"ASSUMED (paper Fig. 4b means, not measured): reuse {self.reuse} s, synthesis {self.synthesize} s, {request}"


@dataclass
class Outcome:
    """One simulated response: the first and terminal decisions and the execution telemetry of Eq. 9."""

    first: RoutingDecision
    terminal: RoutingDecision
    chain: list[RoutingDecision]
    coverage: float  # IC_exec
    retained: bool
    latency: float
    extra: dict[str, Any] = field(default_factory=dict)  # benchmark-specific diagnostics


class Scenario(Protocol):
    name: str  # unique within a benchmark
    cell: str  # Table-2 cell, e.g. "strong/full"
    expected: RoutingAction


class Environment(Protocol):
    latency: PathLatency

    async def respond(self, scenario: Any, learner: Optional[RoutingLearner]) -> Outcome:
        """Route (rule if ``learner`` is None or still bootstrapping), request at most once, then simulate execution."""

    async def alternatives(self, scenario: Any) -> dict[RoutingAction, list[tuple[float, bool, float]]]:
        """Every (IC_exec, retained, latency) a first action can lead to (after a request: each terminal)."""


@dataclass(frozen=True)
class Provenance:
    """What a run actually used, so a degraded or simulated run is never reported as a full one."""

    benchmark: str
    embedder: str
    index: str
    analysis: str
    data: str
    latencies: str
    fallbacks: tuple[str, ...] = ()
    execution: str = SIMULATED_EXECUTION
    evaluation: str = ""  # e.g. "primary reproduction" or "validation: held-out orderings"
    settings_audit: tuple[tuple[str, str, str, str], ...] = ()  # (setting, what the paper states, value used, status)

    @property
    def status(self) -> str:
        return "DEGRADED" if self.fallbacks else "FULL (real embedder and index; execution still simulated)"

    def as_dict(self) -> dict[str, Any]:
        return {"benchmark": self.benchmark, "evaluation": self.evaluation, "status": self.status,
                "embedder": self.embedder, "index": self.index, "fallbacks": list(self.fallbacks),
                "analysis": self.analysis, "data": self.data, "execution": self.execution, "latencies": self.latencies,
                "settings_audit": [{"setting": s, "paper": p, "value": v, "status": st} for s, p, v, st in self.settings_audit]}  # fmt: skip


def component_fallbacks(embedder_name: str, embedder_note: Optional[str], index_name: str,
                        index_note: Optional[str]) -> tuple[str, ...]:  # fmt: skip
    """Every way the retrieval stack differs from the paper's (sentence-transformer + FAISS)."""
    found = [n for n in (embedder_note, index_note) if n]
    if not embedder_name.startswith("st:"):
        found.append(f"embedder is {embedder_name}, not the paper's sentence-transformer")
    if index_name != "faiss":
        found.append(f"vector index is {index_name}, not FAISS")
    return tuple(dict.fromkeys(found))


def require_real_components(fallbacks: Sequence[str], allow_fallback: bool) -> None:
    if fallbacks and not allow_fallback:
        raise SystemExit("refusing to run with substituted components (use --allow-fallback to run a run marked "
                         "DEGRADED): " + "; ".join(fallbacks))  # fmt: skip


# --- rewards ------------------------------------------------------------------------------------


def reward_value(first: RoutingAction, coverage: float, retained: bool, latency: float, latency_max: float,
                 weights: RewardWeights) -> float:  # fmt: skip
    """Eq. 9 credited to the first decision (a request is charged lambda_q * c_q)."""
    value = terminal_reward(retained=retained, coverage=coverage, latency_seconds=latency,
                            latency_max_seconds=latency_max, weights=weights).value  # fmt: skip
    return value - weights.request * weights.request_cost if first == REQUEST else value


def best_first_rewards(alternatives: dict[RoutingAction, list[tuple[float, bool, float]]], latency_max: float,
                       weights: RewardWeights) -> dict[RoutingAction, float]:  # fmt: skip
    """The best reward each first action can earn (after a request, with the better terminal)."""
    return {action: max(reward_value(action, c, r, l, latency_max, weights) for c, r, l in options)
            for action, options in alternatives.items() if options}  # fmt: skip


def reward_preference(rewards: dict[RoutingAction, float], expected: RoutingAction) -> tuple[str, list[str]]:
    """Does Eq. 9 single out the expected action? Returns the class and the reward-optimal actions."""
    best = max(rewards.values())
    winners = [a for a in ACTIONS if a in rewards and rewards[a] >= best - _TOLERANCE]
    if winners == [expected]:
        kind = "unique_expected"
    elif len(winners) == 1:
        kind = "unique_other"
    else:
        kind = "tie_with_expected" if expected in winners else "tie_without_expected"
    return kind, [a.value for a in winners]


# --- the protocol -------------------------------------------------------------------------------


class LegacyBootstrapLearner(RoutingLearner):
    """DIAGNOSTIC ONLY -- the learner as it was before 2026-10-26: bootstrap rewards are applied at once,
    scored with the fallback L_max until the bootstrap latencies are complete (no deferral, no rescoring)."""

    @property
    def latency_max_final(self) -> bool:
        return True


def fresh_learner(settings: Settings, database_path: Path, *, legacy_bootstrap: bool = False) -> RoutingLearner:
    if database_path.exists():  # an existing file would carry another run's rewards into this ordering
        raise FileExistsError(f"{database_path} already exists: each ordering needs a fresh learner (use a new state directory)")
    database = Database(f"sqlite:///{database_path.as_posix()}")
    database.create_schema()
    return (LegacyBootstrapLearner if legacy_bootstrap else RoutingLearner)(
        BanditRepository(database), LinUCBPolicy(alpha=settings.bandit_alpha, ridge=settings.bandit_ridge_lambda),
        enabled=True, bootstrap=settings.bandit_bootstrap_incidents, size_scale=settings.bandit_library_size_scale,
        weights=weights_of(settings), latency_max_seconds=None,
        latency_fallback_seconds=settings.reward_latency_fallback_seconds, learn_from_simulated=True,
    )  # fmt: skip


def weights_of(settings: Settings) -> RewardWeights:
    return RewardWeights(settings.reward_lambda_coverage, settings.reward_lambda_latency,
                         settings.reward_lambda_request, settings.reward_request_cost)  # fmt: skip


async def run_ordering(env: Environment, scenarios: Sequence[Any], seed: int, *, settings: Settings, workdir: Path,
                       rule: dict[str, Outcome], alternatives: dict[str, dict],
                       legacy_bootstrap: bool = False) -> list[dict]:  # fmt: skip
    """One ordering: a fresh learner, the rule for the first ``bootstrap`` incidents, then LinUCB.

    ``workdir/bandit_<seed>.db`` keeps that ordering's learner state (arms and every reward it learned
    from). ``legacy_bootstrap`` is a diagnostic: the pre-fix learner, whose bootstrap training rewards
    differ from the reported ones (recorded in ``trained_reward`` instead of being an error).
    """
    order = list(scenarios)
    random.Random(seed).shuffle(order)
    learner = fresh_learner(settings, workdir / f"bandit_{seed}.db", legacy_bootstrap=legacy_bootstrap)
    weights, bootstrap = learner.weights, learner.bootstrap
    decision_id, rows, outcomes, first_ids = 0, [], [], []
    for position, scenario in enumerate(order):
        policy = "linucb" if learner.active else "rule"
        outcome = await env.respond(scenario, learner)
        stored = []
        for decision in outcome.chain:
            decision_id += 1
            stored.append(StoredRoutingDecision(**decision.model_dump(), decision_id=decision_id, created_at=_EPOCH))
        learner.credit(incident_id=scenario.name, run_id=f"{seed}-{position}", chain=stored,
                       terminal_arm=outcome.terminal.action, retained=outcome.retained,
                       execution_coverage=outcome.coverage, latency_seconds=outcome.latency, synthetic=True)  # fmt: skip
        outcomes.append(outcome)
        first_ids.append(stored[0].decision_id)
        rows.append({"ordering": seed, "position": position, "phase": "bootstrap" if position < bootstrap else "learned",
                     "policy": policy})  # fmt: skip

    if not learner.latency_max_final:
        raise RuntimeError(f"fewer than {bootstrap} scenarios: L_max is undefined")
    latency_max = learner.latency_max()
    rule_latency_max = max(percentile([rule[s.name].latency for s in order[:bootstrap]], 90), 1e-3)
    if abs(rule_latency_max - latency_max) > _TOLERANCE:
        raise RuntimeError("the bootstrap did not follow the rule: L_max differs between the policies")
    trained = {r.decision_id: r.reward for r in learner.repository.list(limit=10 * len(order) + 10)}
    for row, scenario, outcome, first_id in zip(rows, order, outcomes, first_ids):
        baseline = rule[scenario.name]
        reward = reward_value(outcome.first.action, outcome.coverage, outcome.retained, outcome.latency, latency_max, weights)
        if abs(trained[first_id] - reward) > _TOLERANCE and not legacy_bootstrap:
            raise RuntimeError(f"training reward {trained[first_id]} != reported reward {reward} ({scenario.name})")
        options = best_first_rewards(alternatives[scenario.name], latency_max, weights)
        kind, optimal = reward_preference(options, scenario.expected)
        rule_reward = reward_value(baseline.first.action, baseline.coverage, baseline.retained, baseline.latency,
                                   latency_max, weights)  # fmt: skip
        best = max(options.values())
        row.update({
            "scenario": scenario.name, "cell": scenario.cell, "expected": scenario.expected.value,
            "got": outcome.first.action.value, "terminal": outcome.terminal.action.value,
            "execution_coverage": round(outcome.coverage, 4), "retained": outcome.retained, "latency": outcome.latency,
            "reward": reward, "rule_got": baseline.first.action.value, "rule_terminal": baseline.terminal.action.value,
            "rule_retained": baseline.retained, "rule_latency": baseline.latency, "rule_reward": rule_reward,
            "latency_max": latency_max, "reward_preference": kind, "reward_optimal": optimal,
            "first_action_rewards": {a.value: v for a, v in options.items()}, "regret": best - reward,
            "rule_regret": best - rule_reward, "trained_reward": trained[first_id],
            "ucb": outcome.first.ucb, "features": outcome.first.features,
            "rule_terms": baseline.first.terms.model_dump(), "rule_scores": baseline.first.scores.model_dump(),
            **outcome.extra,
        })  # fmt: skip
    learner.repository.db.dispose()
    return rows


async def run(env: Environment, scenarios: Sequence[Any], *, settings: Settings, orderings: int = 10,
              workdir: Path, legacy_bootstrap: bool = False, seeds: Optional[Sequence[int]] = None) -> dict[str, Any]:  # fmt: skip
    """All orderings plus the summary; ``workdir`` holds one bandit database per ordering (a fresh learner each).

    The orderings use seeds ``0 .. orderings-1`` (paper Section 6.4) unless ``seeds`` lists others.
    """
    names = [s.name for s in scenarios]
    if len(set(names)) != len(names):
        raise ValueError("scenario names must be unique")
    workdir.mkdir(parents=True, exist_ok=True)
    rule = {s.name: await env.respond(s, None) for s in scenarios}  # the rule is deterministic per scenario
    alternatives = {s.name: await env.alternatives(s) for s in scenarios}
    rows = []
    for seed in (list(seeds) if seeds is not None else range(orderings)):
        rows += await run_ordering(env, scenarios, seed, settings=settings, workdir=workdir, rule=rule,
                                   alternatives=alternatives, legacy_bootstrap=legacy_bootstrap)  # fmt: skip
    return {"rows": rows, "summary": summarize(rows, bootstrap=settings.bandit_bootstrap_incidents)}


# --- summary ------------------------------------------------------------------------------------


def _policy_view(rows: Sequence[dict], policy: str) -> list[dict]:
    if policy == "linucb":
        return list(rows)
    return [{**r, "got": r["rule_got"], "reward": r["rule_reward"]} for r in rows]


def summarize(rows: Sequence[dict], *, bootstrap: int) -> dict[str, Any]:
    by_ordering: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_ordering[row["ordering"]].append(row)
    orderings = sorted(by_ordering)
    summary: dict[str, Any] = {"orderings": len(orderings), "seeds": orderings,
                               "scenarios": len(by_ordering[orderings[0]]) if orderings else 0,
                               "bootstrap": bootstrap, "ceilings": ceilings(rows, bootstrap=bootstrap)}  # fmt: skip
    totals = {}
    for policy in ("linucb", "rule"):
        per = {o: _policy_view(by_ordering[o], policy) for o in orderings}
        stats = defaultdict(list)
        for o in orderings:
            rs = per[o]
            c = M.classification(rs)
            stats["accuracy"].append(c["accuracy"])
            stats["macro_f1"].append(c["macro_f1"])
            for label in M.LABELS:
                for key in ("precision", "recall", "f1"):
                    stats[f"{key}_{label}"].append(c["per_class"][label][key])
            stats["accuracy_bootstrap"].append(M.accuracy(r for r in rs if r["phase"] == "bootstrap"))
            stats["accuracy_after_bootstrap"].append(M.accuracy(r for r in rs if r["phase"] == "learned"))
            stats["reward_total"].append(sum(r["reward"] for r in rs))
            stats["reward_bootstrap"].append(sum(r["reward"] for r in rs if r["phase"] == "bootstrap"))
            stats["reward_after_bootstrap"].append(sum(r["reward"] for r in rs if r["phase"] == "learned"))
            stats["reward_optimal_rate"].append(sum(r["got"] in r["reward_optimal"] for r in rs) / len(rs))
            stats["accuracy_reward_identifiable"].append(M.accuracy(r for r in rs if r["reward_preference"] == "unique_expected"))
        pooled = [r for o in orderings for r in per[o]]
        totals[policy] = per
        summary[policy] = {
            "per_ordering": {k: v for k, v in stats.items()},
            "mean_std": {k: M.mean_std(v) for k, v in stats.items()},
            "pooled": M.classification(pooled),
            "by_cell": M.grouped(pooled, key=lambda r: r["cell"]),
            "by_cell_per_ordering": {cell: M.mean_std([M.accuracy(r for r in per[o] if r["cell"] == cell) for o in orderings])
                                     for cell in sorted({r["cell"] for r in pooled})},
        }  # fmt: skip
    learned, baseline = summary["linucb"]["per_ordering"], summary["rule"]["per_ordering"]
    summary["relative_reward_gain"] = {
        "per_ordering": [M.relative_gain(a, b) for a, b in zip(learned["reward_total"], baseline["reward_total"])],
        "of_means": M.relative_gain(sum(learned["reward_total"]) / len(orderings), sum(baseline["reward_total"]) / len(orderings)),
    }  # fmt: skip
    summary["relative_reward_gain"]["mean_std"] = M.mean_std(summary["relative_reward_gain"]["per_ordering"])
    difference = sum(r["reward"] - r["rule_reward"] for r in rows)
    by_label = defaultdict(float)
    for r in rows:
        by_label[r["expected"]] += r["reward"] - r["rule_reward"]
    summary["reward_difference_share_by_expected"] = {k: (v / difference if difference else float("nan")) for k, v in by_label.items()}
    first = by_ordering[orderings[0]]
    summary["reward_preference"] = {
        "counts_per_ordering": {o: _count(r["reward_preference"] for r in by_ordering[o]) for o in orderings},
        "not_uniquely_expected_per_ordering": {o: sum(r["reward_preference"] != "unique_expected" for r in by_ordering[o])
                                               for o in orderings},
        "by_cell_ordering0": {cell: _count(r["reward_preference"] for r in first if r["cell"] == cell)
                              for cell in sorted({r["cell"] for r in first})},
    }  # fmt: skip
    n = summary["scenarios"]
    summary["cumulative_reward_curve"] = {policy: [M.mean_std([_cumulative(totals[policy][o])[i] for o in orderings])
                                                   for i in range(n)] for policy in ("linucb", "rule")}  # fmt: skip
    return summary


def ceilings(rows: Sequence[dict], *, bootstrap: int) -> dict[str, Any]:
    """Upper bounds on label accuracy under the protocol, per ordering and on average.

    For ordering o: B = correct decisions among the first ``bootstrap`` positions (the rule's, by the
    protocol). Each later scenario is classed by its reward-optimal first actions R* (``reward_optimal``):
    U -- R* = {label}; T -- label in R*, |R*| > 1; X -- label not in R*, |R*| = 1; W -- label not in R*,
    |R*| > 1. With n scenarios and m = n - bootstrap later positions:

    * any policy                                  (B + m) / n
    * reward-maximiser, uniform tie-breaking      (B + U + sum over T of 1/|R*|) / n
    * reward-maximiser, ties toward the label     (B + U + T) / n

    Learning cost is ignored, so these are upper bounds; they hold only for this benchmark instance,
    its settings and its (simulated) reward.
    """
    by_ordering: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_ordering[row["ordering"]].append(row)
    per = {}
    for o, rs in sorted(by_ordering.items()):
        rs = sorted(rs, key=lambda r: r["position"])
        boot, post = rs[:bootstrap], rs[bootstrap:]
        b = sum(r["got"] == r["expected"] for r in boot)
        u = sum(r["expected"] in r["reward_optimal"] and len(r["reward_optimal"]) == 1 for r in post)
        ties = [r for r in post if r["expected"] in r["reward_optimal"] and len(r["reward_optimal"]) > 1]
        x = sum(r["expected"] not in r["reward_optimal"] and len(r["reward_optimal"]) == 1 for r in post)
        w = sum(r["expected"] not in r["reward_optimal"] and len(r["reward_optimal"]) > 1 for r in post)
        credit = sum(1 / len(r["reward_optimal"]) for r in ties)
        n, m = len(rs), len(post)
        if u + len(ties) + x + w != m:
            raise RuntimeError("ceiling classes do not partition the post-bootstrap decisions")
        per[o] = {"n": n, "m": m, "B": b, "U": u, "T": len(ties), "T_uniform_credit": credit, "X": x, "W": w,
                  "any_policy": (b + m) / n, "reward_uniform_ties": (b + u + credit) / n,
                  "reward_favourable_ties": (b + u + len(ties)) / n}  # fmt: skip
    mean = lambda key: sum(p[key] for p in per.values()) / len(per) if per else float("nan")  # noqa: E731
    return {"per_ordering": per,
            "mean": {k: mean(k) for k in ("B", "U", "T", "T_uniform_credit", "X", "W", "any_policy",
                                          "reward_uniform_ties", "reward_favourable_ties")}}  # fmt: skip


def _count(values) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for v in values:
        counts[v] += 1
    return dict(sorted(counts.items()))


def _cumulative(rows: Sequence[dict]) -> list[float]:
    total, curve = 0.0, []
    for r in sorted(rows, key=lambda r: r["position"]):
        total += r["reward"]
        curve.append(total)
    return curve
