"""Library-growth experiment (paper 968 Fig. 4a): does reuse rise as retained playbooks are written back?

Starts from a small library (default 10 benchmark technique playbooks), streams the 96 scenarios pass after pass,
synthesizes when the router chooses GENERATE (configured Ollama models, cached per scenario and evidence state),
writes retained playbooks back with CADE's own write-back, and reports the reuse rate at library sizes
10, 20, 30, 40 and 50. See ``app/evaluation/growth.py`` for every choice the paper leaves open.

    python scripts/eval_library_growth.py --mock --out-dir reports/growth_smoke       # wiring only, NOT a result
    python scripts/eval_library_growth.py --cache-dir reports/quality_cache --out-dir reports/<date>_growth
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import Settings  # noqa: E402
from app.evaluation import growth as G  # noqa: E402
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


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--initial", type=int, default=10, help="technique playbooks in the starting library")
    parser.add_argument("--target", type=int, default=50, help="stop after the pass in which the library reaches this size")
    parser.add_argument("--max-passes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0, help="starting library and scenario order")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "reports" / "quality_cache")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--mock", action="store_true", help="mock generator and heuristic OS: tests the wiring, NOT a result")
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
    require_real_components(component_fallbacks(embedder.name, embedder_note, index.name, index_note), args.allow_fallback or args.mock)

    database = Database(f"sqlite:///{(Path(tempfile.mkdtemp(prefix='cade_growth_')) / 'library.db').as_posix()}")
    database.create_schema()
    benchmark = B.build()
    library = PlaybookLibrary(LibraryRepository(database), embedder, index, vocabulary,
                              PlaybookValidator(vocabulary, node_cap=settings.node_cap), k=settings.routing_k)  # fmt: skip
    initial = G.seed_initial(benchmark, library, vocabulary, args.initial, random.Random(args.seed))
    env = B.PaperEnv(benchmark, settings, vocabulary, library, {})
    cache_dir = args.cache_dir / ("mock" if args.mock else f"{kit.generator_model}__{kit.judge_model}".replace(":", "-"))
    quality = Q.QualityRun(kit, env, library, Q.Cache(cache_dir))
    rows = await G.GrowthRun(quality, benchmark, vocabulary).run(seed=args.seed, target=args.target, max_passes=args.max_passes)
    by_size = G.reuse_by_size(rows)

    print(f"start {len(initial)} playbooks -> end {len(library)} after {len(rows)} incidents "
          f"({sum(r['write_back'] == 'inserted' for r in rows)} written back, {sum(r['write_back'] == 'merged' for r in rows)} merged)")
    for size, row in by_size.items():
        rate = "n/a" if row["reuse_rate"] is None else f"{100 * row['reuse_rate']:.0f}%"
        print(f"  library ~{size:>2}: reuse {rate:>4} over {row['incidents']:>3} incidents (paper {100 * G.PAPER_FIG_4A[size]:.0f}%)")
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        departures = (["MOCK generator and heuristic OS: wiring check, NOT a result"] if args.mock else []) + [
            "routing by the hand-designed rule (Eq. 7)", "execution and retention SIMULATED on the returned playbook (no logs)",
            f"start: {len(initial)} randomly drawn technique playbooks (seed {args.seed}); up to {args.max_passes} passes",
            f"generator {kit.generator_model}, judge {kit.judge_model} (paper Table 3: qwen3:14b, gemma3:12b)"]  # fmt: skip
        (args.out_dir / "results.json").write_text(json.dumps({"initial": initial, "final_size": len(library),
                                                               "by_size": by_size, "departures": departures, "rows": rows},
                                                              indent=1, default=str), encoding="utf-8")  # fmt: skip
        lines = ["# Library growth (paper 968 Fig. 4a)", "", "| library size | incidents | reuse rate | paper |", "|---|---|---|---|"]
        for size, row in by_size.items():
            rate = "n/a" if row["reuse_rate"] is None else f"{100 * row['reuse_rate']:.0f}%"
            lines.append(f"| ~{size} | {row['incidents']} | {rate} | {100 * G.PAPER_FIG_4A[size]:.0f}% |")
        lines += ["", f"Library: {len(initial)} -> {len(library)} playbooks over {len(rows)} incidents.", "",
                  "## How this run differs from the paper", ""] + [f"- {d}" for d in departures]  # fmt: skip
        (args.out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
