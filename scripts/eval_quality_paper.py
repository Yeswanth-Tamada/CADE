"""Playbook quality on the 96-scenario benchmark: paper 968 Table 4, Fig. 2 and the Fig. 4b / Section 7.4 costs.

Runs the four baselines, the scorer ablation and (given a routing run) the full framework on the same
scenarios, and scores every returned playbook with IC, EA and the LLM judge's OS (Eq. 10). Results are cached
per playbook under ``--cache-dir`` so an interrupted run resumes where it stopped.

    # quick offline check of the wiring (mock generator, heuristic OS -- NOT a result)
    python scripts/eval_quality_paper.py --mock --limit 3 --out-dir reports/quality_smoke

    # real run with the configured Ollama models (hours on CPU; resumable)
    python scripts/eval_quality_paper.py --cache-dir reports/quality_cache --out-dir reports/<date>_quality \
        --routing-results reports/<date>_routing_paper968_simulated/results.json

What is real: the generator and judge are the configured models; IC/EA come from CADE's ontology; latencies
are measured on this machine. What is not the paper's: the tactic -> intent mapping, the templated incident
text (no events), cached synthesis (one playbook per scenario and evidence state), and -- unless Table 3's
models are configured -- the models.
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
from app.evaluation import quality as Q  # noqa: E402
from app.evaluation.protocol import component_fallbacks, require_real_components  # noqa: E402
from app.intent.llm_client import build_llm_client  # noqa: E402
from app.intent.mock_llm_client import MockLLMClient  # noqa: E402
from app.library.embedder import build_embedder  # noqa: E402
from app.library.index import build_index  # noqa: E402
from app.library.library import PlaybookLibrary  # noqa: E402
from app.playbook.validator import PlaybookValidator  # noqa: E402
from app.playbook.vocabulary import Vocabulary  # noqa: E402
from app.storage.database import Database  # noqa: E402
from app.storage.library_repository import LibraryRepository  # noqa: E402

TABLE3 = {"generator": "qwen3:14b", "judge": "gemma3:12b"}


def temp_database(name: str) -> Database:
    database = Database(f"sqlite:///{(Path(tempfile.mkdtemp(prefix='cade_quality_')) / name).as_posix()}")
    database.create_schema()
    return database


def fmt(row: dict, key: str) -> str:
    value = row.get(key)
    return "-" if value is None else (f"{value:.1f}" if key == "latency_s" else f"{value:.2f}")


def render(rows: dict, curves: dict, llm_only_fitness, costs: dict, provenance: dict) -> str:
    lines = ["# Playbook quality, cost and scorer ablation (paper 968 Table 4)", "",
             f"**Provenance:** {provenance['summary']}", ""]
    lines += ["| System | IC | EA | OS | f | Latency (s) | failures | paper IC / EA / OS / f / s |", "|---|---|---|---|---|---|---|---|"]
    for system in Q.SYSTEMS:
        if system not in rows:
            continue
        row, paper = rows[system], Q.PAPER_TABLE_4[system]
        lines.append(f"| {Q.LABELS[system]} | {fmt(row, 'ic')} | {fmt(row, 'ea')} | {fmt(row, 'os')} | {fmt(row, 'fitness')} | "
                     f"{fmt(row, 'latency_s')} | {row['failures']}/{row['n']} | {' / '.join(str(v) for v in paper)} |")  # fmt: skip
    if "cade_full" in rows:
        r = rows["cade_full"]
        lines += ["", f"CADE (full): mean over {r['orderings']} orderings (std of f {r['fitness_std']:.3f}); "
                      f"synthesis share {100 * r['synthesis_share']:.0f}% (paper: 35%)."]  # fmt: skip
    if curves:
        lines += ["", "## Fig. 2 — fitness per generation (generation-only runs)", "",
                  "| gen | best f | mean f | best IC |", "|---|---|---|---|"]
        for g, (b, m, ic) in enumerate(zip(curves["best_fitness"], curves["mean_fitness"], curves["best_ic"])):
            lines.append(f"| {g} | {b:.3f} | {m:.3f} | {ic:.3f} |")
        lines += ["", f"Mean generations to stop: {curves['generations_mean']:.1f} (paper: 13.2). "
                      f"LLM-only fitness (dashed line in Fig. 2): {llm_only_fitness:.3f} (paper: 0.69)."]  # fmt: skip
    if costs:
        lines += ["", "## Fig. 4b / Section 7.4 — synthesis cost (generation-only runs)", "",
                  f"- mean synthesis {costs['total_s']:.1f} s over {costs['runs']} runs (paper: 238 s): generation "
                  f"{costs['generation_s']:.1f} s, seeding {costs['seeding_s']:.1f} s, gap proposals {costs['gap_proposals_s']:.1f} s, "
                  f"evolution {costs['evolution_s']:.1f} s",
                  f"- judge calls per synthesis {costs['judge_calls']:.0f} (paper: about 110); judging "
                  f"{100 * costs['judge_share']:.1f}% of synthesis time (paper: 64.7%)"]  # fmt: skip
    lines += ["", "## How this run differs from the paper", ""] + [f"- {d}" for d in provenance["departures"]]
    return "\n".join(lines) + "\n"


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--systems", default="static_soar,llm_only,retrieval_only,generation_only,generation_only_osh",
                        help=f"comma-separated, from {', '.join(Q.SYSTEMS)} (cade_full needs --routing-results)")
    parser.add_argument("--routing-results", type=Path, help="results.json of eval_routing_paper.py (adds cade_full)")
    parser.add_argument("--policy", choices=["linucb", "rule"], default="linucb", help="whose decisions cade_full follows")
    parser.add_argument("--limit", type=int, help="first N scenarios only (smoke runs)")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "reports" / "quality_cache")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--mock", action="store_true", help="mock generator and heuristic OS: tests the wiring, NOT a result")
    parser.add_argument("--allow-fallback", action="store_true", help="run with a substitute embedder/index (DEGRADED)")
    args = parser.parse_args()
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    if args.routing_results and "cade_full" not in systems:
        systems.append("cade_full")
    unknown = set(systems) - set(Q.SYSTEMS)
    if unknown or ("cade_full" in systems and not args.routing_results):
        parser.error(f"unknown systems {sorted(unknown)}" if unknown else "cade_full needs --routing-results")

    settings = Settings()
    if args.mock:
        generator_llm, judge_llm = MockLLMClient(), None
    else:
        generator_llm, judge_llm = build_llm_client(settings, settings.generator_model), build_llm_client(settings, settings.judge_model)
        for role, client in (("generator", generator_llm), ("judge", judge_llm)):
            health = await client.health()
            if not health.reachable or not health.model_available:
                print(f"{role} model {client.model} is not available: {health.detail}")
                return 2
    kit = Q.build_kit(settings, generator_llm, judge_llm)

    vocabulary = Vocabulary.load(settings.resolve_path(settings.action_vocabulary_path),
                                 settings.resolve_path(settings.intent_goals_path))  # fmt: skip
    embedder, embedder_note = build_embedder(settings.embedder, settings.embedding_model)
    index, index_note = build_index(settings.vector_index, embedder.dimension)
    fallbacks = component_fallbacks(embedder.name, embedder_note, index.name, index_note)
    require_real_components(fallbacks, args.allow_fallback or args.mock)
    benchmark = B.build()
    library, entries = B.build_library(benchmark, settings, vocabulary, embedder, index, temp_database("library.db"))
    env = B.PaperEnv(benchmark, settings, vocabulary, library, entries)
    seed_index, _ = build_index(settings.vector_index, embedder.dimension)
    seeds = PlaybookLibrary(LibraryRepository(temp_database("seeds.db")), embedder, seed_index, kit.vocabulary,
                            PlaybookValidator(kit.vocabulary, node_cap=settings.node_cap), k=settings.routing_k)  # fmt: skip
    seeds.seed_if_empty(Q.seed_library_path(settings))
    seeds.load()
    cache_dir = args.cache_dir / ("mock" if args.mock else f"{kit.generator_model}__{kit.judge_model}".replace(":", "-"))
    run = Q.QualityRun(kit, env, seeds, Q.Cache(cache_dir))

    scenarios = benchmark.scenarios[: args.limit] if args.limit else benchmark.scenarios
    names = {sc.name for sc in scenarios}
    records: dict[str, list[Q.Record]] = {s: [] for s in systems if s != "cade_full"}
    total = len(scenarios) * len(records)
    done = 0
    for system in records:
        for scenario in scenarios:
            record = await run.produce(system, scenario, scenario.available)
            records[system].append(record)
            done += 1
            print(f"[{done}/{total}] {system:20s} {scenario.name:45s} f={record.fitness:.3f} "
                  f"{'FAILED ' if record.failed else ''}{record.latency_s:.1f}s", flush=True)

    cade: dict[int, list[Q.Record]] = {}
    if "cade_full" in systems:
        rows = json.loads(args.routing_results.read_text(encoding="utf-8"))["rows"]
        by_name = {sc.name: sc for sc in scenarios}
        for row in rows:
            if row["scenario"] not in names:
                continue
            terminal = row["terminal"] if args.policy == "linucb" else row["rule_terminal"]
            first = row["got"] if args.policy == "linucb" else row["rule_got"]
            record = await run.cade_full(by_name[row["scenario"]], terminal, first == "request_evidence")
            cade.setdefault(row["ordering"], []).append(record)
        print(f"cade_full: {sum(map(len, cade.values()))} routed decisions over {len(cade)} orderings")

    table = Q.table4(records, cade)
    synthesized = records.get("generation_only", [])
    curves = Q.generation_curves(synthesized)
    costs = Q.latency_breakdown(synthesized)
    llm_only_f = Q.mean(r.fitness for r in records.get("llm_only", []))
    departures = []
    if args.mock:
        departures.append("MOCK generator and heuristic OS: this checks the wiring and is NOT a result")
    for role, model in (("generator", kit.generator_model), ("judge", kit.judge_model)):
        if not args.mock and model != TABLE3[role]:
            departures.append(f"{role} model {model} (paper Table 3: {TABLE3[role]})")
    departures += [f"incidents are the templated one-line summary and evidence profile (no log events); "
                   f"intent per tactic mapped to CADE's ontology: {Q.TACTIC_INTENT}",
                   "synthesis cached: one playbook per scenario and evidence state, reused in every ordering",
                   "library = generated technique playbooks (paper: converted published playbooks)",
                   "latencies measured on this machine (CPU), not the paper's hardware"]  # fmt: skip
    if args.limit:
        departures.append(f"only the first {len(scenarios)} of 96 scenarios")
    provenance = {"summary": f"{len(scenarios)} scenarios; generator {kit.generator_model}; judge {kit.judge_model}; "
                             f"embedder {embedder.name}; index {index.name}; cade_full follows {args.policy if cade else 'n/a'}",
                  "departures": departures, "settings": {"pool": settings.generation_pool_size,
                  "population": settings.population_size, "max_generations": settings.max_generations,
                  "judge_samples": settings.os_judge_samples, "judge_top_k": settings.os_judge_top_k}}  # fmt: skip
    for system, row in table.items():
        print(f"{Q.LABELS[system]:45s} IC {row['ic']:.2f} EA {row['ea']:.2f} OS {row['os']:.2f} f {row['fitness']:.2f} "
              f"latency {row['latency_s']:.1f}s failures {row['failures']}/{row['n']}")
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "results.json").write_text(json.dumps({
            "provenance": provenance, "table4": table, "fig2": curves, "llm_only_fitness": llm_only_f, "costs": costs,
            "records": {s: [r.as_dict() for r in rs] for s, rs in records.items()},
            "cade_full": {str(o): [r.as_dict() for r in rs] for o, rs in cade.items()},
        }, indent=1, default=str), encoding="utf-8")  # fmt: skip
        (args.out_dir / "REPORT.md").write_text(render(table, curves, llm_only_f, costs, provenance), encoding="utf-8")
        print(f"written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
