"""Markdown tables for one routing benchmark run (one benchmark per report; results are never merged)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

from app.evaluation.metrics import LABELS

SHORT = {"reuse": "REUSE", "request_evidence": "REQUEST", "synthesize": "GENERATE"}
POLICIES = (("linucb", "LinUCB (rule for the first 20)"), ("rule", "Hand-designed rule (Eq. 7)"))


def pct(value: float) -> str:
    return "n/a" if value != value else f"{100 * value:.1f}%"


def ms(stat: dict, percent: bool = True, digits: int = 1) -> str:
    if stat["mean"] != stat["mean"]:
        return "n/a"
    if percent:
        return f"{100 * stat['mean']:.{digits}f} ± {100 * stat['std']:.{digits}f}%"
    return f"{stat['mean']:.{digits}f} ± {stat['std']:.{digits}f}"


def table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def provenance_block(provenance: dict) -> str:
    keys = ("benchmark", "evaluation", "status", "embedder", "index", "analysis", "data", "execution", "latencies")
    rows = [(k, provenance[k]) for k in keys if provenance.get(k)]
    rows.append(("fallbacks", "; ".join(provenance["fallbacks"]) or "none"))
    return table(("Component", "What this run used"), rows)


def seeds_text(seeds: Sequence[int]) -> str:
    seeds = list(seeds)
    if seeds and seeds == list(range(seeds[0], seeds[-1] + 1)):
        return f"seeds {seeds[0]}–{seeds[-1]}"
    return "seeds " + ", ".join(str(x) for x in seeds)


def ceilings_block(summary: dict) -> list[str]:
    """The accuracy ceilings of ``protocol.ceilings`` with their derivation (upper bounds; conditional)."""
    c = summary.get("ceilings")
    if not c or not c["per_ordering"]:
        return []
    per, mean = c["per_ordering"], c["mean"]
    p2 = lambda v: f"{100 * v:.2f}%"  # noqa: E731 -- two decimals: these bounds are compared closely
    first = next(iter(per.values()))
    n, m, bootstrap = first["n"], first["m"], summary["bootstrap"]
    out = ["## Accuracy ceilings (upper bounds; conditional on this benchmark instance)", "",
           f"Per ordering: B = correct decisions among the first {bootstrap} positions (the rule's, by the protocol). Each of "
           f"the m = {m} later scenarios is classed by its reward-optimal first actions R* (maximal Eq. 9 reward, best "
           "reachable terminal, this ordering's L_max, ties within 10⁻⁹): U — R* = {label}; T — label ∈ R*, |R*| > 1; "
           "X — label ∉ R*, |R*| = 1; W — label ∉ R*, |R*| > 1 (U + T + X + W = m, checked).", "",
           table(("Ordering", "B", "U", "T", "X", "W", f"Any policy (B + {m})/{n}", f"Reward-max., uniform ties (B + U + ΣT 1/|R*|)/{n}",
                  f"Reward-max., ties to label (B + U + T)/{n}"),
                 [(o, p["B"], p["U"], p["T"], p["X"], p["W"], p2(p["any_policy"]), p2(p["reward_uniform_ties"]),
                   p2(p["reward_favourable_ties"])) for o, p in per.items()]), "",
           f"Means: B̄ = {mean['B']:.1f}, Ū = {mean['U']:.1f}, T̄ = {mean['T']:.1f}, X̄ = {mean['X']:.1f}, W̄ = {mean['W']:.1f}; "
           f"ceilings {p2(mean['any_policy'])} (any policy), {p2(mean['reward_uniform_ties'])} (reward-maximiser, uniform "
           f"ties), {p2(mean['reward_favourable_ties'])} (ties toward the label).", ""]  # fmt: skip
    achieved = summary["linucb"]["mean_std"]["accuracy"]["mean"]
    out += [f"Achieved LinUCB accuracy {p2(achieved)} "
            f"{'≤' if achieved <= mean['any_policy'] + 1e-12 else '> (CHECK FAILED:'} the any-policy ceiling"
            f"{'' if achieved <= mean['any_policy'] + 1e-12 else ')'}. These bounds ignore learning cost and hold only for this "
            "benchmark instance, its settings and its simulated reward; they say nothing about whether results reported "
            "elsewhere were attainable on other benchmarks or settings.", ""]  # fmt: skip
    return out


def settings_block(provenance: dict) -> list[str]:
    audit = provenance.get("settings_audit") or []
    if not audit:
        return []
    return ["## Settings and their status", "",
            table(("Setting", "Paper 968", "Value used", "Status"), [(a["setting"], a["paper"], a["value"], a["status"]) for a in audit]), ""]


def render(title: str, summary: dict, provenance: dict, *, notes: Sequence[str] = (),
           paper: Optional[dict] = None) -> str:  # fmt: skip
    s, out = summary, [f"# {title}", "", provenance_block(provenance), ""]
    seeds = s.get("seeds") or list(range(s["orderings"]))
    out += [f"{s['scenarios']} scenarios × {s['orderings']} orderings ({seeds_text(seeds)}); a fresh LinUCB learner "
            f"per ordering; the rule chooses the first {s['bootstrap']} decisions of every ordering. Mean ± sample std over "
            f"orderings unless marked *pooled*.", ""]  # fmt: skip
    out += [f"- {n}" for n in notes] + ([""] if notes else [])
    out += settings_block(provenance)

    out += ["## Headline", ""]
    head = ("Policy", "Accuracy", "Macro-F1", "Acc. first 20", "Acc. after 20", "Cumulative reward", "Reward-optimal choice",
            "Acc. on reward-identifiable")  # fmt: skip
    rows = []
    for key, name in POLICIES:
        m = s[key]["mean_std"]
        rows.append((name, ms(m["accuracy"]), ms(m["macro_f1"], False, 3), ms(m["accuracy_bootstrap"]),
                     ms(m["accuracy_after_bootstrap"]), ms(m["reward_total"], False, 2), ms(m["reward_optimal_rate"]),
                     ms(m["accuracy_reward_identifiable"])))  # fmt: skip
    out += [table(head, rows), ""]
    gain = s["relative_reward_gain"]
    out += [f"Relative cumulative-reward gain of LinUCB over the rule: {ms(gain['mean_std'])} per ordering "
            f"({pct(gain['of_means'])} on the means).", ""]  # fmt: skip
    out += ["*Accuracy* is agreement with the constructed label; *reward-optimal choice* is the share of decisions whose "
            "first action earns the highest Eq. 9 reward available for that scenario (ties count as optimal); "
            "*reward-identifiable* restricts accuracy to scenarios where Eq. 9 uniquely favours the label.", ""]

    out += ["## Reward before and after the bootstrap", ""]
    rows = [(name, ms(s[k]["mean_std"]["reward_bootstrap"], False, 2), ms(s[k]["mean_std"]["reward_after_bootstrap"], False, 2),
             ms(s[k]["mean_std"]["reward_total"], False, 2)) for k, name in POLICIES]  # fmt: skip
    out += [table(("Policy", "Reward, first 20", "Reward, after 20", "Total"), rows), ""]
    share = s["reward_difference_share_by_expected"]
    out += ["Share of the LinUCB − rule reward difference by expected action (shares sum to 100%; a negative share means LinUCB earned less than the rule there): "
            + ", ".join(f"{SHORT[k]} {pct(share.get(k, float('nan')))}" for k in LABELS), ""]
    curve = s["cumulative_reward_curve"]
    marks = sorted({min(m, s["scenarios"]) for m in (10, 20, 40, 60, 80, s["scenarios"])})
    out += [table(("After N scenarios",) + tuple(str(m) for m in marks),
                  [(name,) + tuple(ms(curve[k][m - 1], False, 2) for m in marks) for k, name in POLICIES]), ""]  # fmt: skip

    out += ["## Per-action precision, recall and F1", ""]
    rows = []
    for key, name in POLICIES:
        m = s[key]["mean_std"]
        for label in LABELS:
            rows.append((name, SHORT[label], ms(m[f"precision_{label}"]), ms(m[f"recall_{label}"]), ms(m[f"f1_{label}"], False, 3)))
    out += [table(("Policy", "Action", "Precision", "Recall", "F1"), rows), ""]

    out += ["## Confusion matrices (pooled over orderings; rows = expected, columns = chosen)", ""]
    for key, name in POLICIES:
        p = s[key]["pooled"]
        rows = [(SHORT[label],) + tuple(f"{c} ({pct(r)})" for c, r in zip(p["confusion"][i], p["confusion_row_normalized"][i]))
                for i, label in enumerate(LABELS)]  # fmt: skip
        out += [f"**{name}** — pooled accuracy {pct(p['accuracy'])} ({p['correct']}/{p['n']})", "",
                table(("expected \\ chosen",) + tuple(SHORT[label] for label in LABELS), rows), ""]  # fmt: skip

    out += ["## By scenario cell", ""]
    cells = sorted(s["linucb"]["by_cell"])
    prefs = s["reward_preference"]["by_cell_ordering0"]
    rows = []
    for cell in cells:
        n = s["linucb"]["by_cell"][cell]["n"] // s["orderings"]
        rows.append((cell, n, ms(s["linucb"]["by_cell_per_ordering"][cell]), ms(s["rule"]["by_cell_per_ordering"][cell]),
                     ", ".join(f"{k} {v}" for k, v in prefs.get(cell, {}).items())))  # fmt: skip
    out += [table(("Cell", "Scenarios", "LinUCB", "Rule", "Eq. 9 preference (first ordering)"), rows), ""]

    out += ["## Does the reward single out the label?", ""]
    counts = s["reward_preference"]["counts_per_ordering"]
    not_unique = s["reward_preference"]["not_uniquely_expected_per_ordering"]
    values = sorted(set(not_unique.values()))
    out += [f"Scenarios where Eq. 9 does **not** uniquely favour the expected action: "
            f"{values[0] if len(values) == 1 else f'{values[0]}–{values[-1]}'} of {s['scenarios']} "
            f"(per ordering; L_max differs slightly between orderings).", ""]  # fmt: skip
    kinds = sorted({k for c in counts.values() for k in c})
    out += [table(("Ordering",) + tuple(kinds), [(o,) + tuple(c.get(k, 0) for k in kinds) for o, c in counts.items()]), ""]

    out += ceilings_block(s)
    if paper:
        out += ["## Against the paper's published values", "",
                "The paper evaluates on executed playbooks against indexed Splunk logs; this run simulates execution. "
                "The comparison is descriptive; nothing was tuned towards these values.", "",
                table(("Quantity", "Paper 968", "This run"), paper["rows"]), ""]  # fmt: skip
    return "\n".join(out)
