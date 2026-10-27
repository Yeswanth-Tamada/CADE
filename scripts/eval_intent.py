"""Evaluate Part 1 (LLM intent extraction) on a labelled SYNTHETIC incident set.

Runs every case through the same IntentExtractor the API uses (configured model, temperature,
ATT&CK dataset and taxonomy) and checks the analysis against the expected result:

* intent      -- the label is one of the acceptable labels (first = preferred);
* technique   -- a validated ATT&CK technique (or its parent) from the expected list was proposed;
* tactic      -- the intent's ATT&CK tactics share a tactic with the proposed techniques (paper 968:
                 intents, evidence and actions are grounded in ATT&CK);
* entities    -- every expected entity occurs in an extracted entity;
* observed    -- every expected evidence category is observed;
* stated_gaps -- a missing item is attributed to the incident text only for categories the input
                 itself says are missing.

    python scripts/eval_intent.py                         # all cases, model from CADE_OLLAMA_MODEL
    python scripts/eval_intent.py --only C03 C04 --out report.json

Needs a reachable Ollama server (or CADE_LLM_PROVIDER=mock for a dry run of the harness).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import Settings  # noqa: E402
from app.core.errors import CADEError  # noqa: E402
from app.ingestion.normalizer import normalize_submission  # noqa: E402
from app.intent.attack_mapper import AttackMapper  # noqa: E402
from app.intent.llm_client import build_llm_client  # noqa: E402
from app.intent.llm_extractor import IntentExtractor  # noqa: E402
from app.schemas.enums import AttackValidationStatus, ProvenanceOrigin  # noqa: E402
from app.schemas.incident import CanonicalIncident, IncidentSubmission  # noqa: E402
from app.schemas.taxonomy import IntentTaxonomy  # noqa: E402

CHECKS = ("intent", "technique", "tactic", "entities", "observed", "stated_gaps")


def evaluate(result: CanonicalIncident, expected: dict, taxonomy: IntentTaxonomy) -> dict[str, tuple[bool, str]]:
    checks: dict[str, tuple[bool, str]] = {}
    label = result.intent.label
    checks["intent"] = (label in expected["intents"],
                        f"{label} ({result.intent.confidence:.2f}); expected {'/'.join(expected['intents'])}")

    valid = [t for t in result.intent.attack_techniques if t.validation_status == AttackValidationStatus.VALIDATED]
    proposed = {t.technique_id for t in valid} | {t.technique_id.split(".")[0] for t in valid}
    wanted = set(expected["techniques_any"])
    ids = ",".join(t.technique_id for t in result.intent.attack_techniques) or "none"
    checks["technique"] = ((bool(proposed & wanted) if wanted else True), ids)

    intent_tactics = set(taxonomy.get(label).tactics) if taxonomy.get(label) else set()
    technique_tactics = {tactic for t in valid for tactic in t.tactics}
    consistent = not intent_tactics or not technique_tactics or bool(intent_tactics & technique_tactics)
    checks["tactic"] = (consistent, f"intent {sorted(intent_tactics)} vs techniques {sorted(technique_tactics)}")

    entities = result.entities.model_dump()
    missing_entities = []
    for kind, values in expected["entities"].items():
        found = [str(v.get("value", v) if isinstance(v, dict) else v).lower() for v in entities.get(kind, [])]
        missing_entities += [f"{kind}:{v}" for v in values if not any(v.lower() in f for f in found)]
    checks["entities"] = (not missing_entities, "missing " + ", ".join(missing_entities) if missing_entities else "ok")

    observed = {e.category.value for e in result.evidence_profile.observed}
    lacking = sorted(set(expected["observed"]) - observed)
    checks["observed"] = (not lacking, f"lacking {lacking}" if lacking else f"ok {sorted(observed)}")

    allowed = set(expected["stated_gaps"])
    wrong = sorted({e.category.value for e in result.evidence_profile.missing
                    if e.provenance.origin == ProvenanceOrigin.INCIDENT_TEXT
                    and "*" not in allowed and e.category.value not in allowed})  # fmt: skip
    checks["stated_gaps"] = (not wrong, f"wrongly stated {wrong}" if wrong else "ok")
    return checks


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cases", type=Path, default=ROOT / "examples" / "intent_eval" / "cases.json")
    parser.add_argument("--only", nargs="*", help="case id prefixes to run (e.g. C03 C04)")
    parser.add_argument("--model", help="override CADE_OLLAMA_MODEL")
    parser.add_argument("--out", type=Path, help="write the full report (analyses + checks) as JSON")
    args = parser.parse_args()

    settings = Settings()
    taxonomy = IntentTaxonomy.load(settings.resolve_path(settings.intent_taxonomy_path))
    mapper = AttackMapper.from_path(settings.resolve_path(settings.attack_dataset_path))
    llm = build_llm_client(settings, args.model or settings.ollama_model)
    extractor = IntentExtractor(llm, taxonomy, mapper, max_attempts=settings.llm_max_attempts,
                                attack_prompt_hints=settings.attack_prompt_hints,
                                two_stage=settings.intent_two_stage)  # fmt: skip
    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    if args.only:
        cases = [c for c in cases if any(c["id"].startswith(p) for p in args.only)]
    print(f"model={llm.provider}/{llm.model} temperature={settings.llm_temperature} attack={mapper.mode.value} "
          f"taxonomy={taxonomy.version} two_stage={settings.intent_two_stage} artifact_hints={len(mapper.artifact_hints)} "
          f"cases={len(cases)}\n")  # fmt: skip

    report, totals = [], {c: 0 for c in CHECKS}
    for case in cases:
        incident = normalize_submission(IncidentSubmission(**case["submission"]),
                                        max_description_chars=settings.max_description_chars)  # fmt: skip
        started = time.perf_counter()
        try:
            result = await extractor.extract(incident)
        except CADEError as exc:
            print(f"{case['id']:<44} FAILED  {exc.category}: {exc.message}")
            report.append({"id": case["id"], "error": f"{exc.category}: {exc.message}"})
            continue
        checks = evaluate(result, case["expected"], taxonomy)
        for name, (ok, _) in checks.items():
            totals[name] += ok
        marks = " ".join(f"{name}={'ok' if ok else 'FAIL'}" for name, (ok, _) in checks.items())
        print(f"{case['id']:<44} {time.perf_counter() - started:5.1f}s  {marks}")
        for name, (ok, detail) in checks.items():
            if not ok or name in ("intent", "technique"):
                print(f"    {name:<11} {detail}")
        report.append({"id": case["id"], "checks": {k: {"ok": ok, "detail": d} for k, (ok, d) in checks.items()},
                       "analysis": json.loads(result.model_dump_json())})  # fmt: skip

    done = sum(1 for r in report if "checks" in r)
    print("\nPass rates over", done, "analysed case(s):")
    for name in CHECKS:
        print(f"  {name:<11} {totals[name]}/{done}")
    print(f"  extraction failures: {len(report) - done}")
    if args.out:
        args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"\nreport written to {args.out}")
    await llm.aclose()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
