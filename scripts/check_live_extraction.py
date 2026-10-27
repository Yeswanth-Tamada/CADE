"""Report how the configured *real* model handles the two manually tested Phase 1 scenarios.

This is a diagnostic report, not a test: live LLM output varies between runs and models, so
nothing here passes or fails. It prints, per scenario, what an analyst would reasonably expect
next to what the model produced after CADE's grounding and ATT&CK validation.

    python scripts/check_live_extraction.py                 # uses CADE_* settings / .env (Ollama)
    python scripts/check_live_extraction.py --runs 3        # repeat to see run-to-run variation

It runs the extraction pipeline in-process (no server, no database writes). All incidents are SYNTHETIC.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import LLMProvider, get_settings  # noqa: E402
from app.core.errors import CADEError  # noqa: E402
from app.ingestion.normalizer import normalize_submission  # noqa: E402
from app.intent.attack_mapper import AttackMapper  # noqa: E402
from app.intent.llm_client import build_llm_client  # noqa: E402
from app.intent.llm_extractor import IntentExtractor  # noqa: E402
from app.schemas.incident import IncidentSubmission  # noqa: E402
from app.schemas.taxonomy import IntentTaxonomy  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "synthetic_incidents"

# What an analyst would expect. Used only for side-by-side display, never to alter results.
SCENARIOS = {
    "encoded_powershell_edr.json": {
        "techniques": ["T1059.001"],
        "stated_gaps": [],
    },
    "scheduled_task_persistence.json": {
        "techniques": ["T1053.005"],
        "stated_gaps": ["file reputation", "file hash", "task XML", "security alert"],
    },
}


async def run(runs: int) -> int:
    settings = get_settings()
    if settings.llm_provider == LLMProvider.MOCK:
        print("CADE_LLM_PROVIDER is 'mock'; this report is only meaningful with a real model.")
        return 2
    taxonomy = IntentTaxonomy.load(settings.resolve_path(settings.intent_taxonomy_path))
    mapper = AttackMapper.from_path(settings.resolve_path(settings.attack_dataset_path))
    llm = build_llm_client(settings)
    extractor = IntentExtractor(
        llm, taxonomy, mapper, max_attempts=settings.llm_max_attempts, attack_prompt_hints=settings.attack_prompt_hints
    )
    health = await llm.health()
    print(f"model: {llm.provider}/{llm.model} ({health.status}); ATT&CK mapping: {mapper.mode.value}; "
          f"hints: {settings.attack_prompt_hints}")  # fmt: skip
    if health.status != "ok":
        print(f"LLM not ready: {health.detail}")
        await llm.aclose()
        return 2

    try:
        for name, expected in SCENARIOS.items():
            data = json.loads((EXAMPLES / name).read_text(encoding="utf-8"))
            data.pop("incident_id", None)
            incident = normalize_submission(IncidentSubmission(**data), max_description_chars=settings.max_description_chars)
            for attempt in range(1, runs + 1):
                print(f"\n=== {name} (run {attempt}/{runs}) ===")
                try:
                    result = await extractor.extract(incident)
                except CADEError as exc:
                    print(f"  extraction failed: {exc.category}: {exc.message}")
                    continue
                proposed = result.analysis.attack_candidates_proposed
                found = [t.technique_id for t in result.intent.attack_techniques]
                print(f"  status={result.analysis.extraction_status.value} intent={result.intent.label} "
                      f"({result.intent.confidence}) duration={result.analysis.duration_ms / 1000:.1f}s")  # fmt: skip
                print(f"  ATT&CK: proposed={proposed} kept={found} expected~{expected['techniques']}")
                for t in result.intent.attack_techniques:
                    print(f"    {t.technique_id} {t.name or ''} [{t.validation_status.value}, {t.confidence}] "
                          f"justification: {t.justification}")  # fmt: skip
                profile = result.evidence_profile
                print(f"  observed: {[(e.category.value, e.description) for e in profile.observed]}")
                print(f"  unconfirmed: {[(e.category.value, e.description) for e in profile.unconfirmed]}")
                for e in profile.missing:
                    print(f"  missing: {e.category.value} [{e.criticality.value}] {e.description} "
                          f"(provenance: {e.provenance.origin.value})")  # fmt: skip
                if expected["stated_gaps"]:
                    print(f"  stated gaps in the input: {expected['stated_gaps']}")
                print(f"  not_checked: {[c.value for c in profile.not_checked]}")
                for warning in result.analysis.warnings:
                    print(f"  warning: {warning}")
    finally:
        await llm.aclose()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Live-model report for the Phase 1 regression scenarios")
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()
    return asyncio.run(run(max(1, args.runs)))


if __name__ == "__main__":
    sys.exit(main())
