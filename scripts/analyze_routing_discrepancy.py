"""Discrepancy analysis of one routing benchmark run (rule vs LinUCB), from its results and kept learner state.

Reads ``results.json`` (with the per-decision fields written since 2026-10-26: features, per-action
rewards, regret, the rule's utility terms) and ``learner_state/bandit_<seed>.db`` (every reward each
ordering's learner was trained on). For every ordering it replays those rewards to reconstruct the
exact LinUCB state before each decision, so each choice can be split into estimate and confidence
bonus; the replay is checked against the UCB values recorded at decision time.

Writes ``analysis.json`` and ``ANALYSIS.md`` next to the results. Nothing here changes a run.

    python scripts/analyze_routing_discrepancy.py reports/.../rerun_paper968 [--reference reports/.../paper968]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.evaluation import metrics as M  # noqa: E402
from app.evaluation.report import SHORT, pct, table  # noqa: E402
from app.routing.bandit import ARMS, FEATURES, LinUCBPolicy  # noqa: E402
from app.schemas.routing import RoutingAction  # noqa: E402

LABELS = M.LABELS


def stats(values) -> dict:
    values = [v for v in values if v == v]
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    s = M.mean_std(values)
    return {"mean": s["mean"], "std": s["std"], "min": min(values), "max": max(values)}


def fmt(s: dict, percent: bool = True, digits: int = 1) -> str:
    if s["mean"] != s["mean"]:
        return "n/a"
    k = 100 if percent else 1
    unit = "%" if percent else ""
    return f"{k * s['mean']:.{digits}f} ± {k * s['std']:.{digits}f} [{k * s['min']:.{digits}f}, {k * s['max']:.{digits}f}]{unit}"


def pearson(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y) -> float:
    rank = lambda v: np.argsort(np.argsort(np.asarray(v, float), kind="stable"), kind="stable").astype(float)  # noqa: E731
    return pearson(rank(x), rank(y))


def stored_rewards(db: Path) -> list[dict]:
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    rows = [{"id": i, "decision_id": d, "incident": inc, "arm": arm, "terminal": bool(t), "reward": r,
             "features": json.loads(f), "components": json.loads(c)}
            for i, d, inc, arm, t, r, f, c in con.execute("select id, decision_id, incident_id, arm, terminal, reward, "
                                                          "features_json, components_json from routing_rewards order by id")]  # fmt: skip
    arms = {a: {"a": json.loads(x), "b": json.loads(y), "pulls": p} for a, x, y, p in con.execute("select arm, a_json, b_json, pulls from bandit_arms")}
    con.close()
    return rows, arms


def policy_view(rows, policy):
    return rows if policy == "linucb" else [{**r, "got": r["rule_got"], "reward": r["rule_reward"], "regret": r["rule_regret"]} for r in rows]


def per_ordering(rows: list[dict], bootstrap: int) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[r["ordering"]].append(r)
    out = {}
    for policy in ("linucb", "rule"):
        table_rows = {}
        for o, rs in sorted(by.items()):
            rs = policy_view(sorted(rs, key=lambda r: r["position"]), policy)
            c = M.classification(rs)
            boot, after = [r for r in rs if r["position"] < bootstrap], [r for r in rs if r["position"] >= bootstrap]
            table_rows[o] = {"accuracy": c["accuracy"], "macro_f1": c["macro_f1"],
                             **{f"{k}_{label}": c["per_class"][label][k] for label in LABELS for k in ("precision", "recall", "f1")},
                             "confusion": c["confusion"], "reward": sum(r["reward"] for r in rs),
                             "reward_bootstrap": sum(r["reward"] for r in boot), "reward_after": sum(r["reward"] for r in after),
                             "accuracy_bootstrap": M.accuracy(boot), "accuracy_after": M.accuracy(after),
                             "regret": sum(r["regret"] for r in rs),
                             "choices_after": dict(Counter(r["got"] for r in after))}  # fmt: skip
        out[policy] = table_rows
    out["gain"] = {o: M.relative_gain(out["linucb"][o]["reward"], out["rule"][o]["reward"]) for o in out["linucb"]}
    return out


def replay_scores(rows: list[dict], state: list[dict], alpha: float, ridge: float, bootstrap: int) -> tuple[list[dict], float]:
    """Reconstruct, before each decision, the LinUCB state from the stored rewards; split each choice into estimate and bonus."""
    position_of = {r["scenario"]: r["position"] for r in rows}
    by_position = defaultdict(list)
    for s in state:
        by_position[position_of[s["incident"]]].append(s)
    policy = LinUCBPolicy(alpha=alpha, ridge=ridge)
    decompositions, worst = [], 0.0
    for r in sorted(rows, key=lambda r: r["position"]):
        if r["position"] == bootstrap:  # the bootstrap rewards were applied (rescored) when the 20th arrived
            pass
        if r["position"] >= bootstrap and r["ucb"]:
            allowed = [RoutingAction(a) for a in r["ucb"]]
            scores = policy.scores(r["features"], allowed)
            for arm, sc in scores.items():
                worst = max(worst, abs(round(sc.ucb, 6) - r["ucb"][arm.value]))
            greedy = max(scores, key=lambda a: (scores[a].estimate, -ARMS.index(a)))
            decompositions.append({"position": r["position"], "scenario": r["scenario"], "cell": r["cell"],
                                   "expected": r["expected"], "got": r["got"], "greedy": greedy.value,
                                   "estimates": {a.value: s.estimate for a, s in scores.items()},
                                   "bonuses": {a.value: alpha * s.bonus for a, s in scores.items()}})  # fmt: skip
        for s in by_position[r["position"]]:
            policy.update(RoutingAction(s["arm"]), s["features"], s["reward"])
    return decompositions, worst


def state_at(rows, state, upto: int, alpha: float, ridge: float) -> LinUCBPolicy:
    position_of = {r["scenario"]: r["position"] for r in rows}
    policy = LinUCBPolicy(alpha=alpha, ridge=ridge)
    for s in state:
        if position_of[s["incident"]] < upto:
            policy.update(RoutingAction(s["arm"]), s["features"], s["reward"])
    return policy


def analyse(run: Path, reference: Path | None, alpha: float, ridge: float) -> tuple[dict, list[str]]:
    data = json.loads((run / "results.json").read_text(encoding="utf-8"))
    rows = data["rows"] if "rows" in data else data["learned"]
    summary = data["summary"]
    bootstrap = summary["bootstrap"]
    orderings = sorted({r["ordering"] for r in rows})
    out: dict = {"run": str(run), "provenance": data["provenance"]}
    md: list[str] = []

    # --- reproducibility -------------------------------------------------------------------------
    repro = {}
    if reference is not None:
        ref = json.loads((reference / "results.json").read_text(encoding="utf-8"))
        ref_rows = ref["rows"] if "rows" in ref else ref["learned"]
        key = lambda r: (r["ordering"], r["position"])  # noqa: E731
        mine = {key(r): r for r in rows}
        fields = ("scenario", "expected", "got", "terminal", "rule_got", "retained")
        mismatch = [k for k, r in ((key(r), r) for r in ref_rows) if any(r[f] != mine[k][f] for f in fields)]
        reward_diff = max(abs(r["reward"] - mine[key(r)]["reward"]) for r in ref_rows)
        repro = {"reference": str(reference), "decisions_compared": len(ref_rows), "decision_mismatches": len(mismatch),
                 "max_reward_difference": reward_diff}  # fmt: skip
        for name in ("scenarios.json",):
            if (reference / name).exists() and (run / name).exists():
                a, b = json.loads((reference / name).read_text(encoding="utf-8")), json.loads((run / name).read_text(encoding="utf-8"))
                repro["scenarios_identical"] = a["scenarios"] == b["scenarios"]
    catalog = ROOT / "app" / "data" / "paper_benchmark" / "catalog.json"
    repro["catalog_sha256"] = hashlib.sha256(catalog.read_bytes()).hexdigest()
    order = {o: [r["scenario"] for r in sorted((r for r in rows if r["ordering"] == o), key=lambda r: r["position"])] for o in orderings}
    repro["orderings"] = order
    repro["latency_max"] = {o: next(r["latency_max"] for r in rows if r["ordering"] == o) for o in orderings}
    switch = {o: min((r["position"] for r in rows if r["ordering"] == o and r["policy"] == "linucb"), default=None) for o in orderings}
    repro["first_linucb_position"] = switch

    # --- learner state per ordering --------------------------------------------------------------------
    state_dir = run / "learner_state"
    learner = {}
    decompositions = []
    worst_replay = 0.0
    mismatched_rewards: list[tuple[int, int]] = []
    for o in orderings:
        rs = [r for r in rows if r["ordering"] == o]
        state, arms = stored_rewards(state_dir / f"bandit_{o}.db")
        by_incident = defaultdict(list)
        for s in state:
            by_incident[s["incident"]].append(s)
        for r in rs:  # the reward of the first decision: the request credit if the first action was a request
            credits = by_incident[r["scenario"]]
            first = next((c for c in credits if not c["terminal"]), None) if r["got"] == "request_evidence" else \
                next(c for c in credits if c["terminal"])  # fmt: skip
            if first is None or abs(first["reward"] - r["reward"]) > 1e-9:
                mismatched_rewards.append((o, r["position"]))
        d, worst = replay_scores(rs, state, alpha, ridge, bootstrap)
        worst_replay = max(worst_replay, worst)
        for x in d:
            x["ordering"] = o
        decompositions += d
        boot_state = state_at(rs, state, bootstrap, alpha, ridge)
        thetas = {a.value: (np.linalg.inv(np.array(boot_state.arms[a].a)) @ np.array(boot_state.arms[a].b)).tolist() for a in ARMS}
        final_thetas = {a: (np.linalg.inv(np.array(v["a"])) @ np.array(v["b"])).tolist() for a, v in arms.items()}
        learner[o] = {"updates": len(state), "bootstrap_updates": sum(1 for s in state if {r["scenario"]: r["position"] for r in rs}[s["incident"]] < bootstrap),
                      "final_pulls": {a: v["pulls"] for a, v in arms.items()}, "theta_after_bootstrap": thetas,
                      "theta_final": final_thetas}  # fmt: skip
    out["reproducibility"] = {**repro, "replay_max_ucb_difference": worst_replay,
                              "decisions_trained_on_a_different_reward_than_reported": len(mismatched_rewards),
                              "mismatched_positions": sorted({p for _, p in mismatched_rewards})}
    out["learner"] = learner

    # --- per-ordering metrics --------------------------------------------------------------------------
    po = per_ordering(rows, bootstrap)
    out["per_ordering"] = po
    keys = ["accuracy", "macro_f1", "accuracy_bootstrap", "accuracy_after", "reward", "reward_bootstrap", "reward_after", "regret"]
    keys += [f"{k}_{label}" for label in LABELS for k in ("precision", "recall", "f1")]
    out["across_orderings"] = {p: {k: stats([po[p][o][k] for o in orderings]) for k in keys} for p in ("linucb", "rule")}
    out["across_orderings"]["gain"] = stats(list(po["gain"].values()))

    # --- REUSE-recall spread: what does it track? ---------------------------------------------------
    factors = {}
    for o in orderings:
        rs = sorted((r for r in rows if r["ordering"] == o), key=lambda r: r["position"])
        boot = rs[:bootstrap]
        reuse = [r for r in boot if r["got"] == "reuse"]
        reuse_none = [r for r in reuse if r["cell"].startswith("none")]
        factors[o] = {"reuse_recall_after": M.accuracy(r for r in rs[bootstrap:] if r["expected"] == "reuse"),
                      "bootstrap_reuse_pulls": len(reuse), "bootstrap_reuse_retained": sum(r["retained"] for r in reuse),
                      "bootstrap_reuse_success_rate": (sum(r["retained"] for r in reuse) / len(reuse)) if reuse else float("nan"),
                      "bootstrap_reuse_mean_reward": (sum(r["reward"] for r in reuse) / len(reuse)) if reuse else float("nan"),
                      "bootstrap_reuse_on_none_cells": len(reuse_none),
                      "bootstrap_reuse_mean_similarity": float(np.mean([r["top_similarity"] for r in reuse])) if reuse else float("nan"),
                      "bootstrap_synth_pulls": sum(r["got"] == "synthesize" for r in boot),
                      "bootstrap_strong_scenarios": sum(r["cell"].startswith("strong") for r in boot)}  # fmt: skip
    out["reuse_recall_factors"] = factors
    y = [factors[o]["reuse_recall_after"] for o in orderings]
    out["reuse_recall_correlations"] = {k: {"pearson": pearson([factors[o][k] for o in orderings], y),
                                            "spearman": spearman([factors[o][k] for o in orderings], y)}
                                        for k in factors[orderings[0]] if k != "reuse_recall_after"}  # fmt: skip

    # --- features, estimates and bonuses -------------------------------------------------------------
    first_ordering = [r for r in rows if r["ordering"] == orderings[0]]
    feats = np.array([r["features"] for r in first_ordering])
    out["feature_ranges"] = {name: {"min": float(feats[:, i].min()), "max": float(feats[:, i].max()), "std": float(feats[:, i].std())}
                             for i, name in enumerate(FEATURES)}  # fmt: skip
    cell_means = {}
    for cell in sorted({r["cell"] for r in first_ordering}):
        sel = np.array([r["features"] for r in first_ordering if r["cell"] == cell])
        cell_means[cell] = {name: float(sel[:, i].mean()) for i, name in enumerate(FEATURES)}
    out["feature_means_by_cell"] = cell_means
    flipped = [d for d in decompositions if d["greedy"] != d["got"]]
    out["exploration"] = {"post_bootstrap_decisions": len(decompositions), "bonus_flipped_choice": len(flipped),
                          "flipped_by_cell": dict(Counter(d["cell"] for d in flipped)),
                          "flipped_to": dict(Counter(f"{d['greedy']}->{d['got']}" for d in flipped)),
                          "greedy_accuracy": M.accuracy({**d, "got": d["greedy"]} for d in decompositions),
                          "chosen_accuracy": M.accuracy(decompositions)}  # fmt: skip
    # estimate gap reuse - synthesize at strong/full vs none/full contexts, right after the bootstrap
    gaps = {}
    for o in orderings:
        theta = learner[o]["theta_after_bootstrap"]
        g = {}
        for cell in ("strong/full", "strong/partial", "none/full", "none/partial"):
            if cell in cell_means:
                x = np.array([cell_means[cell][f] for f in FEATURES])
                g[cell] = float(np.dot(theta["reuse"], x) - np.dot(theta["synthesize"], x))
        g["theta_reuse_similarity"] = theta["reuse"][FEATURES.index("top_similarity")]
        gaps[o] = g
    out["estimate_gap_reuse_minus_synth_after_bootstrap"] = gaps

    # what the stored reuse observations would support without the prior (ordinary vs ridge least squares)
    fits = {}
    for o in orderings:
        state, _ = stored_rewards(state_dir / f"bandit_{o}.db")
        reuse = [(s["features"], s["reward"]) for s in state if s["arm"] == "reuse"]
        if len(reuse) < 3:
            continue
        X, yv = np.array([f for f, _ in reuse]), np.array([r for _, r in reuse])
        fit = {}
        for lam in (1.0, 0.01):
            theta = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ yv)
            fit[f"lambda_{lam}"] = {c: float(np.dot(theta, [cell_means[c][f] for f in FEATURES])) for c in ("strong/full", "none/full") if c in cell_means}
            fit[f"lambda_{lam}"]["theta_similarity"] = float(theta[FEATURES.index("top_similarity")])
        sims = X[:, FEATURES.index("top_similarity")]
        split = (cell_means["strong/full"]["top_similarity"] + cell_means["none/full"]["top_similarity"]) / 2
        fit["observations"] = len(reuse)
        fit["similarity_split"] = split
        fit["empirical_reward_high_similarity"] = float(yv[sims >= split].mean()) if (sims >= split).any() else float("nan")
        fit["empirical_reward_low_similarity"] = float(yv[sims < split].mean()) if (sims < split).any() else float("nan")
        fits[o] = fit
    out["reuse_arm_fits"] = fits

    # --- the rule's utilities -------------------------------------------------------------------------
    rule_rows = {r["scenario"]: r for r in first_ordering}
    entropies = [r["rule_terms"]["entropy"] for r in rule_rows.values()]
    none_wrong = [r for r in rule_rows.values() if r["cell"].startswith("none") and r["rule_got"] != "synthesize"]
    out["rule"] = {
        "entropy": {"min": min(entropies), "max": max(entropies), "at_least_0.995": sum(e >= 0.995 for e in entropies), "n": len(entropies)},
        "none_cells": sum(r["cell"].startswith("none") for r in rule_rows.values()),
        "none_cells_wrong": [{"scenario": r["scenario"], "chose": r["rule_got"], "top": r.get("top_candidate"),
                              "coverage": r["rule_terms"]["coverage"], "entropy": r["rule_terms"]["entropy"],
                              "scores": r["rule_scores"]} for r in sorted(none_wrong, key=lambda r: r["scenario"])],
        "by_cell": {cell: {"n": n, "rule_accuracy": M.accuracy({**r, "got": r["rule_got"]} for r in rule_rows.values() if r["cell"] == cell),
                           "coverage_1": sum(r["rule_terms"]["coverage"] >= 1 - 1e-9 for r in rule_rows.values() if r["cell"] == cell)}
                    for cell, n in Counter(r["cell"] for r in rule_rows.values()).items()},
    }  # fmt: skip
    # REQUEST: rule recall per tactic and cell, with the unmet fraction M of p*
    request = [r for r in rule_rows.values() if r["expected"] == "request_evidence"]
    group = lambda r: r.get("tactic") or r.get("intent") or "?"  # noqa: E731
    out["request"] = {
        "rule_recall_by_group": {g: M.accuracy({**r, "got": r["rule_got"]} for r in request if group(r) == g) for g in sorted({group(r) for r in request})},
        "rule_recall_by_cell": {c: M.accuracy({**r, "got": r["rule_got"]} for r in request if r["cell"] == c) for c in sorted({r["cell"] for r in request})},
        "rule_recall_by_unmet_fraction": {f"{m:.3f}": {"n": sum(round(r["rule_terms"]["unmet_fraction"], 3) == m for r in request),
                                                        "recall": M.accuracy({**r, "got": r["rule_got"]} for r in request if round(r["rule_terms"]["unmet_fraction"], 3) == m)}
                                          for m in sorted({round(r["rule_terms"]["unmet_fraction"], 3) for r in request})},
        "rule_misses": [{"scenario": r["scenario"], "chose": r["rule_got"], "M": r["rule_terms"]["unmet_fraction"],
                         "rho": r["rule_terms"]["requestability"], "coverage": r["rule_terms"]["coverage"],
                         "scores": r["rule_scores"]} for r in request if r["rule_got"] != "request_evidence"],
        "request_vs_generate_near_ties": sum(1 for r in rule_rows.values() if "request_evidence" in r["first_action_rewards"]
                                             and abs(r["first_action_rewards"]["request_evidence"] - r["first_action_rewards"]["synthesize"]) < 0.05),
        "request_offered_scenarios": sum(1 for r in rule_rows.values() if "request_evidence" in r["first_action_rewards"]),
    }  # fmt: skip

    # --- labels vs reward ------------------------------------------------------------------------------
    pref = defaultdict(list)
    for r in rows:
        pref[r["scenario"]].append(r)
    ambiguous = []
    for name, rs in sorted(pref.items()):
        if rs[0]["reward_preference"] == "unique_expected":
            continue
        ambiguous.append({"scenario": name, "cell": rs[0]["cell"], "expected": rs[0]["expected"],
                          "preference": rs[0]["reward_preference"], "reward_optimal": rs[0]["reward_optimal"],
                          "action_rewards": {k: round(v, 4) for k, v in rs[0]["first_action_rewards"].items()},
                          "rule": rs[0]["rule_got"], "rule_regret": round(rs[0]["rule_regret"], 4),
                          "linucb_choices": dict(Counter(r["got"] for r in rs)),
                          "linucb_mean_regret": round(float(np.mean([r["regret"] for r in rs])), 4)})  # fmt: skip
    out["label_reward_alignment"] = {
        "not_uniquely_expected_per_ordering": dict(Counter(r["ordering"] for r in rows if r["reward_preference"] != "unique_expected")),
        "scenarios": ambiguous,
        "mean_regret": {p: stats([po[p][o]["regret"] / len([r for r in rows if r["ordering"] == o]) for o in orderings]) for p in ("linucb", "rule")},
        "accuracy_identifiable": {p: stats([M.accuracy(policy_view([r for r in rows if r["ordering"] == o and r["reward_preference"] == "unique_expected"], p))
                                            for o in orderings]) for p in ("linucb", "rule")},
    }  # fmt: skip

    # --- error taxonomy: retrieval, reward model, policy --------------------------------------------------
    def kind(r, got):
        if r[got] == r["expected"]:
            return "correct"
        if r["reward_preference"] != "unique_expected":
            return "label not uniquely reward-optimal"
        own = r.get("p_star_is_own_technique", r.get("p_star_is_own_seed"))
        if r["cell"].startswith("strong") and own is False:
            return "retrieval: p* is not the scenario's own playbook"
        if r["cell"].startswith("weak") and r.get("p_star_same_tactic") is False:
            return "retrieval/construction: p* is from another tactic"
        return "policy (p* as intended, label reward-optimal)"

    out["error_taxonomy"] = {p: dict(Counter(kind(r, "got" if p == "linucb" else "rule_got") for r in rows)) for p in ("linucb", "rule")}

    # --- construction flags (paper benchmark only) -------------------------------------------------------
    if any("degenerate" in r for r in rows):
        flagged = lambda r: bool(r.get("degenerate"))  # noqa: E731
        scen = json.loads((run / "scenarios.json").read_text(encoding="utf-8"))["scenarios"] if (run / "scenarios.json").exists() else []
        relaxed = {s["name"] for s in scen if s.get("relaxed")}
        degenerate = {s["name"] for s in scen if s.get("degenerate")}
        excluded = relaxed | degenerate
        out["construction_flags"] = {
            "degenerate": sorted(degenerate), "relaxed": sorted(relaxed),
            "accuracy_all": {p: stats([M.accuracy(policy_view([r for r in rows if r["ordering"] == o], p)) for o in orderings]) for p in ("linucb", "rule")},
            "accuracy_without_flagged": {p: stats([M.accuracy(policy_view([r for r in rows if r["ordering"] == o and r["scenario"] not in excluded], p))
                                                   for o in orderings]) for p in ("linucb", "rule")},
            "flagged_accuracy": {p: stats([M.accuracy(policy_view([r for r in rows if r["ordering"] == o and r["scenario"] in excluded], p))
                                           for o in orderings]) for p in ("linucb", "rule")},
        }  # fmt: skip

    # --- markdown ---------------------------------------------------------------------------------------
    rp = out["reproducibility"]
    md += [f"# Discrepancy analysis: {data['provenance']['benchmark']}", ""]
    md += ["## Reproducibility", "",
           table(("Check", "Result"), [
               ("Decisions identical to the reference run", f"{rp.get('decision_mismatches', 'n/a')} mismatches of {rp.get('decisions_compared', 'n/a')}" if reference else "no reference"),
               ("Max reward difference vs reference", f"{rp.get('max_reward_difference', float('nan')):.2e}" if reference else "n/a"),
               ("Scenario definitions identical", rp.get("scenarios_identical", "n/a")),
               ("Catalog sha256", rp["catalog_sha256"][:16] + "…"),
               ("First LinUCB decision (position) per ordering", sorted(set(rp["first_linucb_position"].values()))),
               ("L_max per ordering (s)", sorted({round(v, 3) for v in rp["latency_max"].values()})),
               ("Decisions whose stored training reward differs from the reported reward",
                f"{rp['decisions_trained_on_a_different_reward_than_reported']}"
                + (f" (positions {rp['mismatched_positions'][0]}–{rp['mismatched_positions'][-1]})" if rp['mismatched_positions'] else "")),
               ("Replayed state reproduces recorded UCBs (max diff)", f"{rp['replay_max_ucb_difference']:.1e}"),
               ("Updates per ordering (bootstrap / total)", sorted({(v['bootstrap_updates'], v['updates']) for v in learner.values()})),
               ("Embedder / index / fallbacks", f"{data['provenance']['embedder']} / {data['provenance']['index']} / {data['provenance']['fallbacks'] or 'none'}"),
           ]), ""]  # fmt: skip
    md += ["## Per-ordering results (mean ± std [min, max] over orderings)", ""]
    rows_md = []
    for k in keys:
        percent = not k.startswith(("reward", "regret"))
        rows_md.append((k, fmt(out["across_orderings"]["linucb"][k], percent, 2 if not percent else 1),
                        fmt(out["across_orderings"]["rule"][k], percent, 2 if not percent else 1)))
    rows_md.append(("relative reward gain", fmt(out["across_orderings"]["gain"]), ""))
    md += [table(("Metric", "LinUCB", "Rule"), rows_md), ""]
    md += [table(("Ordering", "LinUCB acc", "Rule acc", "LinUCB reward", "Rule reward", "Gain", "REUSE recall (LinUCB)", "LinUCB choices after 20"),
                 [(o, pct(po["linucb"][o]["accuracy"]), pct(po["rule"][o]["accuracy"]), f"{po['linucb'][o]['reward']:.2f}",
                   f"{po['rule'][o]['reward']:.2f}", pct(po["gain"][o]), pct(po["linucb"][o]["recall_reuse"]),
                   ", ".join(f"{SHORT[a]} {n}" for a, n in sorted(po["linucb"][o]["choices_after"].items()))) for o in orderings]), ""]  # fmt: skip
    md += ["Per-ordering confusion matrices (LinUCB; rows expected REUSE/REQUEST/GENERATE, columns chosen):", "",
           table(("Ordering", "REUSE row", "REQUEST row", "GENERATE row"),
                 [(o, *[str(line) for line in po["linucb"][o]["confusion"]]) for o in orderings]), ""]  # fmt: skip
    md += ["## What the REUSE-recall spread tracks (10 orderings; correlation is not causation)", "",
           table(("Ordering", "REUSE recall after 20", "Bootstrap reuses", "…retained", "…on 'none' cells", "Reuse − synth estimate at strong/full after 20", "θ_reuse[similarity]"),
                 [(o, pct(factors[o]["reuse_recall_after"]), factors[o]["bootstrap_reuse_pulls"], factors[o]["bootstrap_reuse_retained"],
                   factors[o]["bootstrap_reuse_on_none_cells"], f"{gaps[o].get('strong/full', float('nan')):+.3f}",
                   f"{gaps[o]['theta_reuse_similarity']:+.3f}") for o in orderings]), "",
           table(("Factor", "Pearson r", "Spearman ρ"),
                 [(k, f"{v['pearson']:+.2f}", f"{v['spearman']:+.2f}") for k, v in out["reuse_recall_correlations"].items()]), ""]  # fmt: skip
    ex = out["exploration"]
    md += ["## Estimates, bonuses and features", "",
           f"After the bootstrap, the confidence bonus changed the choice away from the arm with the highest estimate in "
           f"{ex['bonus_flipped_choice']} of {ex['post_bootstrap_decisions']} decisions ({pct(ex['bonus_flipped_choice'] / max(ex['post_bootstrap_decisions'], 1))}); "
           f"flips: {ex['flipped_to']}. Accuracy of the chosen actions {pct(ex['chosen_accuracy'])}; of the greedy (estimate-only) "
           f"actions on the same states {pct(ex['greedy_accuracy'])}.", "",
           table(("Feature", "min", "max", "std"), [(k, f"{v['min']:.3f}", f"{v['max']:.3f}", f"{v['std']:.3f}") for k, v in out["feature_ranges"].items()]), "",
           table(("Cell",) + tuple(FEATURES[:-1]), [(c,) + tuple(f"{v[f]:.3f}" for f in FEATURES[:-1]) for c, v in cell_means.items()]), ""]  # fmt: skip
    if fits:
        md += ["Reuse-arm fits on each ordering's own reuse observations (predicted reward at the mean strong/full and none/full context):", "",
               table(("Ordering", "n", "ridge λ=1: strong / none / θ_sim", "λ=0.01: strong / none / θ_sim", "observed reward above / below the similarity midpoint"),
                     [(o, f["observations"],
                       f"{f['lambda_1.0'].get('strong/full', float('nan')):.2f} / {f['lambda_1.0'].get('none/full', float('nan')):.2f} / {f['lambda_1.0']['theta_similarity']:+.2f}",
                       f"{f['lambda_0.01'].get('strong/full', float('nan')):.2f} / {f['lambda_0.01'].get('none/full', float('nan')):.2f} / {f['lambda_0.01']['theta_similarity']:+.2f}",
                       f"{f['empirical_reward_high_similarity']:.2f} / {f['empirical_reward_low_similarity']:.2f}") for o, f in fits.items()]), ""]  # fmt: skip
    ru = out["rule"]
    md += ["## The rule (Eq. 7)", "",
           f"Entropy H of the first decisions: min {ru['entropy']['min']:.4f}, max {ru['entropy']['max']:.4f}; "
           f"{ru['entropy']['at_least_0.995']} of {ru['entropy']['n']} have H ≥ 0.995.", "",
           table(("Cell", "Scenarios", "Rule accuracy", "Cov(p*) = 1"), [(c, v["n"], pct(v["rule_accuracy"]), v["coverage_1"]) for c, v in sorted(ru["by_cell"].items())]), "",
           f"'No suitable playbook' scenarios where the rule does not synthesize: {len(ru['none_cells_wrong'])} of {ru['none_cells']}.", "",
           table(("Scenario", "Chose", "Cov", "H", "U_reuse", "U_request", "U_gen"),
                 [(w["scenario"], w["chose"], f"{w['coverage']:.3f}", f"{w['entropy']:.4f}", f"{w['scores']['reuse']:.5f}",
                   f"{w['scores']['request_evidence']:.5f}", f"{w['scores']['synthesize']:.5f}") for w in ru["none_cells_wrong"]]), ""]  # fmt: skip
    rq = out["request"]
    md += ["## REQUEST", "",
           table(("Group", "Rule REQUEST recall"), [(g, pct(v)) for g, v in rq["rule_recall_by_group"].items()]), "",
           table(("Unmet fraction M of p*", "Scenarios", "Rule REQUEST recall"), [(m, v["n"], pct(v["recall"])) for m, v in rq["rule_recall_by_unmet_fraction"].items()]), "",
           f"Scenarios where REQUEST is offered: {rq['request_offered_scenarios']}; REQUEST and GENERATE rewards within 0.05: {rq['request_vs_generate_near_ties']}.", ""]  # fmt: skip
    la = out["label_reward_alignment"]
    md += ["## Labels vs reward", "",
           f"Scenarios whose label is not the unique reward optimum, per ordering: {sorted(set(la['not_uniquely_expected_per_ordering'].values()))}. "
           f"Mean regret per decision: LinUCB {fmt(la['mean_regret']['linucb'], False, 3)}, rule {fmt(la['mean_regret']['rule'], False, 3)}. "
           f"Accuracy on the reward-identifiable subset (secondary): LinUCB {fmt(la['accuracy_identifiable']['linucb'])}, rule {fmt(la['accuracy_identifiable']['rule'])}.", "",
           table(("Scenario", "Expected", "Eq. 9 class", "Reward-optimal", "Best reward per first action", "Rule (regret)", "LinUCB choices (mean regret)"),
                 [(a["scenario"], SHORT[a["expected"]], a["preference"], ", ".join(SHORT[x] for x in a["reward_optimal"]),
                   ", ".join(f"{SHORT[k]} {v}" for k, v in a["action_rewards"].items()), f"{SHORT[a['rule']]} ({a['rule_regret']})",
                   ", ".join(f"{SHORT[k]} {v}" for k, v in a["linucb_choices"].items()) + f" ({a['linucb_mean_regret']})") for a in ambiguous]), ""]  # fmt: skip
    md += ["## Error taxonomy (all decisions, all orderings)", "",
           table(("Class", "LinUCB", "Rule"), [(k, out["error_taxonomy"]["linucb"].get(k, 0), out["error_taxonomy"]["rule"].get(k, 0))
                                               for k in sorted(set(out["error_taxonomy"]["linucb"]) | set(out["error_taxonomy"]["rule"]))]), ""]  # fmt: skip
    if "construction_flags" in out:
        cf = out["construction_flags"]
        md += ["## Construction flags", "",
               f"Degenerate (sparse evidence equal to another cell's): {len(cf['degenerate'])}; relaxed (recording reused): {len(cf['relaxed'])}.", "",
               table(("Accuracy", "LinUCB", "Rule"), [("all 96", fmt(cf["accuracy_all"]["linucb"]), fmt(cf["accuracy_all"]["rule"])),
                                                       ("without the flagged scenarios", fmt(cf["accuracy_without_flagged"]["linucb"]), fmt(cf["accuracy_without_flagged"]["rule"])),
                                                       ("flagged scenarios only", fmt(cf["flagged_accuracy"]["linucb"]), fmt(cf["flagged_accuracy"]["rule"]))]), ""]  # fmt: skip
    return out, md


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", type=Path, help="directory with results.json and learner_state/")
    parser.add_argument("--reference", type=Path, help="an earlier run of the same configuration to compare decisions with")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=1.0)
    args = parser.parse_args()
    out, md = analyse(args.run, args.reference, args.alpha, args.ridge)
    (args.run / "analysis.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    (args.run / "ANALYSIS.md").write_text("\n".join(md), encoding="utf-8")
    print(f"written {args.run / 'ANALYSIS.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
