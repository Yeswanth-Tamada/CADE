"""Paper-matched routing benchmark (paper 968 Sections 6.1-6.4, Table 2): rule (Eq. 7) vs LinUCB (Eq. 8).

96 scenarios -- 8 ATT&CK tactics x the 12 Table-2 cells, 32 per expected action -- built from Splunk
Attack Data metadata (``app/data/paper_benchmark/catalog.json``) by :mod:`app.evaluation.paper968`,
run through the shared online protocol (:mod:`app.evaluation.protocol`): 10 orderings (seeds 0-9), a
fresh LinUCB learner per ordering, the rule for the first 20 decisions, LinUCB afterwards, both policies
scored with the same Eq. 9 reward.

What is real and what is not (also written into every report): retrieval uses the paper's embedder and
FAISS (the run refuses substitutes unless ``--allow-fallback``); the scenarios' techniques, tactics and
log sources come from real Splunk Attack Data YAML; the evidence categories are an ATT&CK-derived proxy;
execution (IC_exec, retention) is SIMULATED and latencies are ASSUMED (paper Fig. 4b). This does not
reproduce the paper's execution-backed evaluation.

    python scripts/eval_routing_paper.py --out-dir reports/<date>_routing_paper968
    python scripts/eval_routing_paper.py --goals all --out-dir ...      # sensitivity: every category is a goal
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import Settings  # noqa: E402
from app.evaluation import paper968 as B  # noqa: E402
from app.evaluation import protocol as P  # noqa: E402
from app.evaluation.metrics import accuracy  # noqa: E402
from app.evaluation.report import ms, pct, render, seeds_text  # noqa: E402
from app.library.embedder import build_embedder  # noqa: E402
from app.library.index import build_index  # noqa: E402
from app.playbook.vocabulary import Vocabulary  # noqa: E402
from app.routing.bandit import DIMENSION  # noqa: E402
from app.storage.database import Database  # noqa: E402


# The separate validation benchmark (V2): the five tactics with enough recordings unused by the 96 scenarios.
REPLICATION_TACTICS = ("execution", "persistence", "privilege_escalation", "defense_evasion", "discovery")
PAPER_SEEDS = list(range(10))  # paper Section 6.4


def parse_seeds(text: str) -> list[int]:
    """'10-19' or '0,3,7'."""
    if "-" in text:
        low, high = (int(x) for x in text.split("-", 1))
        return list(range(low, high + 1))
    return [int(x) for x in text.split(",") if x.strip()]


def settings_audit(settings, args, seeds, sharpness, embedder_name, index_name, fallbacks) -> tuple:
    """(setting, what Paper 968 states, value used, status) -- pre-declared in the reproduction's PREREGISTRATION.md."""
    documented = lambda ok: "documented" if ok else "DEPARTURE (declared sensitivity)"  # noqa: E731
    weights = (settings.reward_lambda_coverage, settings.reward_lambda_latency, settings.reward_lambda_request, settings.reward_request_cost)
    return (
        ("w, k, top-N", "Table 3: 0.7, 3, 10", f"{settings.routing_w:g}, {settings.routing_k:g}, {settings.routing_top_n}",
         documented((settings.routing_w, settings.routing_k, settings.routing_top_n) == (0.7, 3, 10))),
        ("LinUCB d, α", "Table 3: d = 9 (eight features + bias), α = 1.0", f"d = {DIMENSION}, α = {settings.bandit_alpha:g}",
         documented(DIMENSION == 9 and settings.bandit_alpha == 1.0)),
        ("Bootstrap", "Table 3, §7.3: the rule for the first 20", str(settings.bandit_bootstrap_incidents),
         documented(settings.bandit_bootstrap_incidents == 20)),
        ("Reward weights λ_c/λ_ℓ/λ_q; c_q", "Table 3: 0.6/0.3/0.1; §6.2: c_q = 1", "/".join(f"{w:g}" for w in weights[:3]) + f"; {weights[3]:g}",
         documented(weights == (0.6, 0.3, 0.1, 1.0))),
        ("L_max and bootstrap rewards", "§6.2: 90th-percentile latency of each ordering's first 20",
         "pre-fix: applied at once with the 600 s fallback" if args.diagnostic_legacy_bootstrap else "same; rewards held back until L_max is known",
         documented(not args.diagnostic_legacy_bootstrap)),
        ("Orderings", "§6.4: seeds 0–9", seeds_text(seeds), "documented" if seeds == PAPER_SEEDS else "validation (not the paper's seeds)"),
        ("Retrieval", "Table 3: all-MiniLM-L6-v2, FAISS inner product, top-10", f"{embedder_name}, {index_name}", documented(not fallbacks)),
        ("Entropy H", "Eq. 2: π_k = s_k / Σ s_i over the top-N", "exact, raw cosine" if sharpness == 1.0 else f"sharpened, p = {sharpness:g}",
         documented(sharpness == 1.0)),
        ("Critical / supporting per intent; intents", "§6.1: every category marked per intent; values not published",
         "≥ 50% of the tactic's candidate techniques" + (" (frozen from P1)" if args.benchmark == "replication" else "") + "; one intent per tactic",
         "inferred"),
        ("Requestability ρ", "§4.1 mean obtainability over missing critical; §6.1 'marked obtainable'",
         "deployment availability profile" if args.diagnostic_profile_requestability else "binary marks",
         "DEPARTURE (declared sensitivity)" if args.diagnostic_profile_requestability else "inferred"),
        ("Ridge λ", "§4.2 'refit by ridge regression'; disjoint LinUCB [10]", f"{settings.bandit_ridge_lambda:g}",
         "inferred (A_a = I_d, as in [10])" if settings.bandit_ridge_lambda == 1.0 else "DEPARTURE (declared sensitivity)"),
        ("Input scaling", "not stated", "none (as in [10])", "inferred"),
        ("Library-size feature", "'library size'", f"n / (n + {settings.bandit_library_size_scale:g})", "inferred"),
        ("Tie-break between equal scores", "not stated", "reuse, then request, then synthesize", "inferred"),
        ("Request arm offered", "≤ 1 request per incident", "only when an obtainable critical category of p* is missing",
         "inferred (implementation)"),
        ("IC_exec goals; attribution", "§6.2: ground-truth goals, attributable events (unspecified)",
         ("all recorded categories" if args.goals == "all" else "the recording's critical categories") + "; a category counts when present",
         "DEPARTURE (declared sensitivity)" if args.goals == "all" else "inferred"),
        ("Latencies", "Fig. 4b means", "reuse 6 s, request path 8.4 s, synthesis 238 s (+2.4 s after a request)", "assumed (simulated execution)"),
    )


def paper_rows(summary: dict, rows: list[dict]) -> list[tuple]:
    """Our numbers next to the paper's published ones (descriptive only)."""
    ref, lin, rule = B.PAPER, summary["linucb"], summary["rule"]
    f1 = lin["mean_std"]
    out = [
        ("LinUCB accuracy", pct(ref["linucb_accuracy"]), ms(f1["accuracy"])),
        ("LinUCB accuracy, first 20 / after", f"{pct(ref['linucb_accuracy_first20'])} / {pct(ref['linucb_accuracy_after20'])}",
         f"{ms(f1['accuracy_bootstrap'])} / {ms(f1['accuracy_after_bootstrap'])}"),
        ("LinUCB macro-F1", f"{ref['linucb_macro_f1']:.2f}", ms(f1["macro_f1"], False, 3)),
        ("LinUCB F1 REUSE / REQUEST / GENERATE", " / ".join(f"{ref['linucb_f1'][k]:.2f}" for k in ("reuse", "request_evidence", "synthesize")),
         " / ".join(f"{f1[f'f1_{k}']['mean']:.2f}" for k in ("reuse", "request_evidence", "synthesize"))),
        ("LinUCB REQUEST recall", pct(ref["linucb_request_recall"]), ms(f1["recall_request_evidence"])),
        ("Rule accuracy", pct(ref["rule_accuracy"]), ms(rule["mean_std"]["accuracy"])),
        ("Rule REQUEST recall", pct(ref["rule_request_recall"]), ms(rule["mean_std"]["recall_request_evidence"])),
        ("Cumulative reward LinUCB / rule", f"{ref['cumulative_reward']['linucb']:.0f} / {ref['cumulative_reward']['rule']:.0f}",
         f"{f1['reward_total']['mean']:.1f} / {rule['mean_std']['reward_total']['mean']:.1f}"),
        ("Relative reward gain", f"≈{pct(ref['relative_reward_gain'])}", ms(summary["relative_reward_gain"]["mean_std"])),
        ("REQUEST share of the reward difference", pct(ref["request_share_of_reward_difference"]),
         pct(summary["reward_difference_share_by_expected"].get("request_evidence", float("nan")))),
    ]  # fmt: skip
    matrix = lin["pooled"]["confusion_row_normalized"]
    for i, label in enumerate(("REUSE", "REQUEST", "GENERATE")):
        out.append((f"LinUCB confusion row {label} (reuse/request/generate)",
                    " / ".join(f"{100 * v:.0f}%" for v in ref["linucb_confusion_row_normalized"][i]),
                    " / ".join(f"{100 * v:.0f}%" for v in matrix[i])))  # fmt: skip
    for cell, value in ref["by_cell"].items():
        mine = accuracy(r for r in rows if (r["cell"].startswith("weak/sparse") if cell == "weak/sparse" else r["cell"] == cell))
        out.append((f"LinUCB accuracy, cell {cell}", pct(value), pct(mine)))
    evidence_of = {"full": ("full",), "partial": ("partial",), "sparse": ("sparse", "sparse_obtainable", "sparse_unobtainable")}
    for level, value in ref["by_evidence"].items():
        mine = accuracy(r for r in rows if r["cell"].split("/")[1] in evidence_of[level])
        out.append((f"LinUCB accuracy, {level} evidence", pct(value), pct(mine)))
    return out


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--orderings", type=int, default=10, help="random orderings, seeds 0..n-1 (paper: 10)")
    parser.add_argument("--seeds", help="explicit seeds instead, e.g. 10-19 (held-out orderings; a validation, not the paper's)")
    parser.add_argument("--benchmark", choices=["main", "replication"], default="main",
                        help="main: the 96-scenario Table-2 benchmark; replication: the separate 60-scenario validation "
                             "benchmark (five tactics, recordings unused by the 96, ontology frozen from them)")
    parser.add_argument("--goals", choices=["critical", "all"], default="critical",
                        help="IC_exec ground-truth goals: the recording's critical categories (primary) or all its categories")
    parser.add_argument("--sharpness", type=float, help="override CADE_ROUTING_SIMILARITY_SHARPNESS (1.0 = Eq. 2)")
    parser.add_argument("--embedder", choices=["sentence_transformers", "hashing"], help="override CADE_EMBEDDER")
    parser.add_argument("--allow-fallback", action="store_true", help="run with a substitute embedder/index (DEGRADED)")
    parser.add_argument("--out-dir", type=Path, help="write scenarios.json, results.json and REPORT.md here")
    parser.add_argument("--state-dir", type=Path, help="keep each ordering's learner database (bandit_<seed>.db) here")
    diagnostic = parser.add_argument_group("diagnostics (each departs from paper Table 3 and is labelled as such)")
    diagnostic.add_argument("--diagnostic-alpha", type=float, help="LinUCB alpha instead of 1.0")
    diagnostic.add_argument("--diagnostic-ridge", type=float, help="LinUCB ridge lambda instead of 1.0")
    diagnostic.add_argument("--diagnostic-legacy-bootstrap", action="store_true",
                            help="the pre-fix learner: bootstrap rewards applied at once with the 600 s fallback L_max")
    diagnostic.add_argument("--diagnostic-profile-requestability", action="store_true",
                            help="rho from the deployment availability profile instead of 1 for categories marked obtainable")
    args = parser.parse_args()

    settings = Settings()
    departures = []
    if args.diagnostic_alpha is not None:
        settings = settings.model_copy(update={"bandit_alpha": args.diagnostic_alpha})
        departures.append(f"alpha={args.diagnostic_alpha} (Table 3: 1.0)")
    if args.diagnostic_ridge is not None:
        settings = settings.model_copy(update={"bandit_ridge_lambda": args.diagnostic_ridge})
        departures.append(f"ridge lambda={args.diagnostic_ridge} (CADE default 1.0, i.e. A_a = I as in Li et al. 2010; paper 968 does not state lambda)")
    if args.diagnostic_legacy_bootstrap:
        departures.append("pre-fix bootstrap: rewards applied at once with the 600 s fallback L_max")
    if args.diagnostic_profile_requestability:
        departures.append("requestability from the deployment availability profile (primary: 1 for categories marked obtainable)")
    if args.goals == "all":
        departures.append("IC_exec goals = all recorded categories (primary: critical categories)")
    if args.sharpness is not None and args.sharpness != 1.0:
        departures.append(f"Eq. 2 sharpened, p = {args.sharpness:g} (paper: exact)")
    seeds = parse_seeds(args.seeds) if args.seeds else list(range(args.orderings))
    vocabulary = Vocabulary.load(settings.resolve_path(settings.action_vocabulary_path),
                                 settings.resolve_path(settings.intent_goals_path))  # fmt: skip
    embedder, embedder_note = build_embedder(args.embedder or settings.embedder, settings.embedding_model)
    index, index_note = build_index(settings.vector_index, embedder.dimension)
    fallbacks = P.component_fallbacks(embedder.name, embedder_note, index.name, index_note)
    P.require_real_components(fallbacks, args.allow_fallback)

    main_benchmark = B.build()
    overlap = None
    if args.benchmark == "replication":
        benchmark = B.build(exclude_recordings={sc.recording.id for sc in main_benchmark.scenarios}, tactics=REPLICATION_TACTICS,
                            tactic_of=main_benchmark.tactic_of, critical=main_benchmark.critical, name_prefix="v2:")  # fmt: skip
        mine, theirs = {sc.technique for sc in benchmark.scenarios}, {sc.technique for sc in main_benchmark.scenarios}
        overlap = (len(mine & theirs), len(mine))
    else:
        benchmark = main_benchmark
    if departures:
        kind = "SENSITIVITY (declared departure from Paper 968; not a reproduction result)"
    elif args.benchmark == "replication":
        kind = ("V2 validation benchmark: 60 scenarios, five tactics, recordings unused by P1, ontology and tactic assignment "
                "frozen from P1 (not directly comparable with P1 or the paper)")
    elif seeds == PAPER_SEEDS:
        kind = "P1 primary reproduction (Paper 968 construction and protocol; simulated execution)"
    else:
        kind = "V1 validation: held-out orderings of the same 96 scenarios (not independent of P1)"
    database = Database(f"sqlite:///{(Path(tempfile.mkdtemp(prefix='cade_paper_')) / 'library.db').as_posix()}")
    database.create_schema()
    library, entries = B.build_library(benchmark, settings, vocabulary, embedder, index, database)
    sharpness = args.sharpness if args.sharpness is not None else settings.routing_similarity_sharpness
    env = B.PaperEnv(benchmark, settings, vocabulary, library, entries, sharpness=sharpness, goals=args.goals,
                     requestability="profile" if args.diagnostic_profile_requestability else "marked")
    catalog = benchmark.catalog["provenance"]
    provenance = P.Provenance(
        benchmark=("V2: 60-scenario validation benchmark built by the Table-2 rules" if args.benchmark == "replication"
                   else "paper968: 96-scenario Table-2 benchmark (paper Section 6.1)")
                  + (f" -- DIAGNOSTIC, departs from the paper: {'; '.join(departures)}" if departures else ""),
        evaluation=kind,
        settings_audit=settings_audit(settings, args, seeds, sharpness, embedder.name, index.name, fallbacks),
        embedder=embedder.name, index=index.name,
        fallbacks=fallbacks,
        analysis="fixed ground-truth analysis per scenario (tactic intent, technique, observed and missing evidence); "
                 "no model call and no mock LLM",
        data=f"Splunk Attack Data YAML metadata at commit {catalog['splunk_attack_data']['commit'][:12]} "
             f"({catalog['splunk_attack_data']['commit_date'][:10]}) and ATT&CK v{catalog['attack']['version']}; "
             "evidence categories are an ATT&CK-derived proxy (no logs downloaded or indexed); "
             f"library = {len(entries)} generated technique playbooks; IC_exec goals = {args.goals} categories",
        latencies=env.latency.describe(),
    )  # fmt: skip
    composition = benchmark.composition()
    print(f"embedder={embedder.name} index={index.name} fallbacks={list(fallbacks) or 'none'} sharpness={sharpness} goals={args.goals}")
    print(f"composition: {composition}")
    print(f"ontology: {json.dumps({t: o['critical'] for t, o in benchmark.ontology().items()})}")

    workdir = args.state_dir or Path(tempfile.mkdtemp(prefix="cade_paper_bandit_"))
    result = await P.run(env, benchmark.scenarios, settings=settings, workdir=workdir, seeds=seeds,
                         legacy_bootstrap=args.diagnostic_legacy_bootstrap)  # fmt: skip
    s = result["summary"]
    for key in ("linucb", "rule"):
        m = s[key]["mean_std"]
        print(f"{key:>6}: accuracy {100 * m['accuracy']['mean']:.1f} +/- {100 * m['accuracy']['std']:.1f}%  "
              f"macro-F1 {m['macro_f1']['mean']:.3f}  reward {m['reward_total']['mean']:.2f} +/- {m['reward_total']['std']:.2f}  "
              f"first 20 {100 * m['accuracy_bootstrap']['mean']:.1f}%  after {100 * m['accuracy_after_bootstrap']['mean']:.1f}%")  # fmt: skip
    print(f"relative reward gain {100 * s['relative_reward_gain']['mean_std']['mean']:+.1f} +/- "
          f"{100 * s['relative_reward_gain']['mean_std']['std']:.1f}%; labels not uniquely reward-optimal: "
          f"{sorted(set(s['reward_preference']['not_uniquely_expected_per_ordering'].values()))} of {s['scenarios']}")  # fmt: skip

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "scenarios.json").write_text(json.dumps({
            "composition": composition, "ontology": benchmark.ontology(), "infeasible": benchmark.infeasible,
            "library": [{"technique": t, "tactic": benchmark.tactic_of[t], "footprint": [c.value for c in B.ordered(benchmark.footprint[t])]}
                        for t in benchmark.library_techniques],
            "scenarios": [sc.summary() for sc in benchmark.scenarios],
            **({"excluded_recordings_of": "the 96-scenario benchmark (P1)", "technique_overlap_with_P1": overlap}
               if overlap else {})}, indent=1), encoding="utf-8")  # fmt: skip
        (args.out_dir / "results.json").write_text(json.dumps({"provenance": provenance.as_dict(), "summary": s,
                                                                "rows": result["rows"]}, indent=1, default=str), encoding="utf-8")  # fmt: skip
        notes = [f"Evaluation: {kind}."]
        if overlap:
            notes.append(f"Recordings are disjoint from P1's; {overlap[0]} of this benchmark's {overlap[1]} techniques also occur in "
                         "P1, so it is not fully independent. It covers five tactics only (credential access, lateral movement "
                         "and impact have too few unused recordings).")
        notes += [f"Composition: {composition['by_expected']} over {composition['scenarios']} scenarios, 12 per tactic, "
                 f"{composition['distinct_techniques']} distinct techniques, {composition['library_playbooks']} library playbooks.",
                 f"Construction flags: {composition['degenerate']} scenarios whose sparse evidence coincides with another cell's "
                 f"(thin recordings); relaxed constructions: {composition['relaxed'] or 'none'} (listed in scenarios.json).",
                 "Derived ontology (critical categories per tactic): "
                 + "; ".join(f"{t}: {', '.join(o['critical'])}" for t, o in benchmark.ontology().items()) + "."]  # fmt: skip
        title = {"replication": "Routing: V2 validation benchmark (60 scenarios)"}.get(
            args.benchmark, "Routing: paper-matched 96-scenario benchmark")
        primary = kind.startswith("P1")
        report = render(f"{title} — {kind.split(' (')[0]}", s, provenance.as_dict(), notes=notes,
                        paper={"rows": paper_rows(s, result["rows"])} if primary else None)  # fmt: skip
        (args.out_dir / "REPORT.md").write_text(report, encoding="utf-8")
        print(f"written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
