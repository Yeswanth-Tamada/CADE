"""Option A: train and evaluate LinUCB on the 96-scenario benchmark with REAL synthesized playbooks.

Same scenarios and protocol as ``eval_routing_paper.py`` (10 orderings, fresh LinUCB per ordering, the rule for the
first 20 decisions), but every GENERATE is a real CADE synthesis with the configured Ollama models, and the reward
comes from the playbook actually returned. Execution is still SIMULATED (no logs); see app/evaluation/generated.py.

Synthesis is cached under ``--cache-dir`` (shared with eval_quality_paper.py), so the run can be stopped and
restarted and each needed playbook is written once. While it runs, Windows is asked not to idle-sleep.

    python scripts/eval_routing_generated.py --time-test                  # one synthesis, prints how long it took
    python scripts/eval_routing_generated.py --out-dir reports/<date>_routing_generated
    python scripts/eval_routing_generated.py --mock --out-dir reports/routing_generated_smoke   # wiring only
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import Settings  # noqa: E402
from app.evaluation import paper968 as B  # noqa: E402
from app.evaluation import protocol as P  # noqa: E402
from app.evaluation import quality as Q  # noqa: E402
from app.evaluation.generated import EXECUTION_NOTE, GeneratedPaperEnv  # noqa: E402
from app.evaluation.report import render  # noqa: E402
from app.intent.llm_client import build_llm_client  # noqa: E402
from app.intent.mock_llm_client import MockLLMClient  # noqa: E402
from app.library.embedder import build_embedder  # noqa: E402
from app.library.index import build_index  # noqa: E402
from app.library.library import PlaybookLibrary  # noqa: E402
from app.playbook.validator import PlaybookValidator  # noqa: E402
from app.playbook.vocabulary import Vocabulary  # noqa: E402
from app.storage.database import Database  # noqa: E402
from app.storage.library_repository import LibraryRepository  # noqa: E402

_ES_CONTINUOUS, _ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001


def keep_awake(on: bool) -> None:
    """Ask Windows not to idle-sleep while this process runs (like a media player); no setting is changed."""
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS | (_ES_SYSTEM_REQUIRED if on else 0))


def temp_database(name: str) -> Database:
    database = Database(f"sqlite:///{(Path(tempfile.mkdtemp(prefix='cade_generated_')) / name).as_posix()}")
    database.create_schema()
    return database


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--orderings", type=int, default=10)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "reports" / "quality_cache")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--time-test", action="store_true", help="synthesize one scenario and report the time")
    parser.add_argument("--eager-alternatives", action="store_true",
                        help="synthesize every state the reward diagnostics need (about twice the syntheses)")
    parser.add_argument("--mock", action="store_true", help="mock generator, heuristic OS: wiring only, NOT a result")
    parser.add_argument("--allow-fallback", action="store_true")
    args = parser.parse_args()

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
    fallbacks = P.component_fallbacks(embedder.name, embedder_note, index.name, index_note)
    P.require_real_components(fallbacks, args.allow_fallback or args.mock)

    benchmark = B.build()
    library, entries = B.build_library(benchmark, settings, vocabulary, embedder, index, temp_database("library.db"))
    env = GeneratedPaperEnv(benchmark, settings, vocabulary, library, entries, lazy_alternatives=not args.eager_alternatives)
    seed_index, _ = build_index(settings.vector_index, embedder.dimension)
    seeds = PlaybookLibrary(LibraryRepository(temp_database("seeds.db")), embedder, seed_index, kit.vocabulary,
                            PlaybookValidator(kit.vocabulary, node_cap=settings.node_cap), k=settings.routing_k)  # fmt: skip
    cache_dir = args.cache_dir / ("mock" if args.mock else f"{kit.generator_model}__{kit.judge_model}".replace(":", "-"))
    env.quality = Q.QualityRun(kit, env, seeds, Q.Cache(cache_dir))
    print(f"generator {kit.generator_model} | judge {kit.judge_model} | embedder {embedder.name} | cache {cache_dir}", flush=True)

    keep_awake(True)
    try:
        if args.time_test:
            scenario = next(s for s in benchmark.scenarios if s.cell == "none/full")
            started = time.perf_counter()
            record = await env.quality.produce("generation_only", scenario, scenario.available)
            timing = record.info.get("timing", {})
            print(f"one synthesis for {scenario.name}: {record.latency_s:.0f} s "
                  f"(generation {timing.get('generation_s', 0):.0f} s, evolution {timing.get('evolution_s', 0):.0f} s, "
                  f"{timing.get('judge_calls', 0)} judge calls taking {timing.get('judge_s', 0):.0f} s); "
                  f"wall clock incl. final scoring {time.perf_counter() - started:.0f} s; fitness {record.fitness:.3f}"
                  f"{' (FAILED: ' + str(record.info.get('error')) + ')' if record.failed else ''}")  # fmt: skip
            return 0

        workdir = Path(tempfile.mkdtemp(prefix="cade_generated_bandit_"))
        started = time.perf_counter()
        result = await P.run(env, benchmark.scenarios, settings=settings, orderings=args.orderings, workdir=workdir)
        hours = (time.perf_counter() - started) / 3600
    finally:
        keep_awake(False)

    s, stats = result["summary"], env.stats()
    for key in ("linucb", "rule"):
        m = s[key]["mean_std"]
        print(f"{key:>6}: accuracy {100 * m['accuracy']['mean']:.1f} +/- {100 * m['accuracy']['std']:.1f}%  "
              f"macro-F1 {m['macro_f1']['mean']:.3f}  reward {m['reward_total']['mean']:.2f} +/- {m['reward_total']['std']:.2f}  "
              f"first 20 {100 * m['accuracy_bootstrap']['mean']:.1f}%  after {100 * m['accuracy_after_bootstrap']['mean']:.1f}%")  # fmt: skip
    print(f"relative reward gain {100 * s['relative_reward_gain']['mean_std']['mean']:+.1f}%; syntheses {stats['syntheses']} "
          f"(mean {stats['synthesis_seconds_mean'] or 0:.0f} s); diagnostics using the stand-in: {stats['estimated_alternatives']}; "
          f"{hours:.1f} h")  # fmt: skip
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        provenance = P.Provenance(
            benchmark="paper968: 96-scenario Table-2 benchmark (paper Section 6.1) -- OPTION A: real synthesized playbooks",
            evaluation="MOCK wiring check, NOT a result" if args.mock else "option A: real local models, simulated execution",
            execution=EXECUTION_NOTE,
            settings_audit=(
                ("generator", "Qwen3 14B", str(kit.generator_model), "matches" if kit.generator_model == "qwen3:14b" else "differs"),
                ("judge", "Gemma 3 12B", str(kit.judge_model), "matches" if kit.judge_model == "gemma3:12b" else "differs"),
                ("pool B", "20", str(settings.generation_pool_size), "matches" if settings.generation_pool_size == 20 else "differs"),
                ("population", "30", str(settings.population_size), "matches" if settings.population_size == 30 else "differs"),
                ("generations", "<= 20", str(settings.max_generations), "matches" if settings.max_generations == 20 else "differs"),
                ("orderings", "10", str(args.orderings), "matches" if args.orderings == 10 else "differs"),
                ("synthesis", "per incident", "cached per scenario and evidence state", "choice"),
            ),
            embedder=embedder.name, index=index.name, fallbacks=fallbacks,
            analysis="fixed ground-truth analysis per scenario (tactic intent, technique, observed and missing evidence)",
            data="Splunk Attack Data YAML metadata (no logs); evidence categories are an ATT&CK-derived proxy; "
                 f"library = {len(entries)} generated technique playbooks",
            latencies=f"synthesis MEASURED on this machine (cached once per scenario and evidence state, mean "
                      f"{stats['synthesis_seconds_mean'] or 0:.0f} s); reuse/request {env.latency.describe()}",
        )  # fmt: skip
        (args.out_dir / "results.json").write_text(json.dumps({"provenance": provenance.as_dict(), "summary": s, "stats": stats,
                                                               "rows": result["rows"]}, indent=1, default=str), encoding="utf-8")  # fmt: skip
        notes = [f"{stats['syntheses']} real syntheses; reward diagnostics fell back to the stand-in "
                 f"{stats['estimated_alternatives']} times (training rewards never do)."]  # fmt: skip
        title = "Routing with real synthesized playbooks (option A) — 96 scenarios"
        (args.out_dir / "REPORT.md").write_text(render(title, s, provenance.as_dict(), notes=notes), encoding="utf-8")
        print(f"written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
