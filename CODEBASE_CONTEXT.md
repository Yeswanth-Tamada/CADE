# CADE Codebase Context

A high-signal orientation brief for an AI assistant working in this repository. Treat the source code as
authoritative if this brief and the implementation disagree. The README holds the user-facing documentation.

## Project at a glance

CADE (Coverage-Aware DAG Evolution) implements paper 968 (`968_revised_version (2).docx`) as a Python 3.10+
FastAPI service. Paper 968 is the only reference; section and equation numbers refer to it.

- Incident understanding (Phase 1), grounded in ATT&CK tactics and host log sources.
- Tool enrichment.
- Typed playbook DAGs.
- Fitness f = ∛(IC·EA·OS) (Eq. 10), with OS_H rule checks and a cross-family LLM judge on the top m (Eq. 11–12).
- LLM candidate generation with adaptive K (Jaccard clustering).
- The coverage-aware evolutionary search.
- Routing: the utility rule (Eq. 1–7) as bootstrap, then a LinUCB bandit (Eq. 8) trained on the Eq. 9 reward.
- A self-evolving playbook library.
- Controlled execution with approvals, verification and audit.
- A static analyst console.

**Evaluation:** only the routing part of the paper's evaluation exists, as three separately reported benchmarks
(`app/evaluation/`, README §7 item 17): the paper-matched 96-scenario Table-2 benchmark built from Splunk Attack Data
metadata, and the legacy and corrected 123/93-scenario intent-level benchmarks. All three simulate execution and assume
the Fig. 4b latencies. The quality evaluation (Table 4 baselines, ablations, judge) is not implemented. Never claim the
paper's experimental numbers for this code, and never merge results across benchmarks.

**Simulation is the default execution mode.** Simulated tool output is synthetic and must never be presented as real
evidence.

## Request flow

1. `POST /incidents`, then `POST /incidents/{id}/analyze`. This produces a `CanonicalIncident`
   (`app/schemas/incident.py`): intent, entities, evidence profile and ATT&CK candidates. Every claim is grounded in
   the input. With `intent_two_stage` (default) `IntentExtractor` (`app/intent/llm_extractor.py`) first decides the
   intent (activity → ATT&CK technique → tactic → label, steered by `AttackMapper.artifact_matches`), then writes
   the evidence profile with the label fixed. Measure with `scripts/eval_intent.py` (`examples/intent_eval/`).
2. `POST /incidents/{id}/respond` (`app/services/response_service.py`) does the following:
   - routes with `app/routing/router.py` (rule, or LinUCB via `app/routing/learning.py` after the bootstrap);
   - on `request_evidence`, enriches through `app/enrichment/enricher.py` and re-routes (at most
     `max_evidence_rounds` = 1 request; the final round cannot request);
   - on `reuse`, adapts the library playbook;
   - on `synthesize`, runs `generator` → `seeder` → `CADEEngine`.
   The result is a `ResponsePlan` (`app/schemas/response.py`). Long runs are background jobs (`app/services/jobs.py`).
3. `approve`, `verifications` and `execute` go through `app/services/plan_service.py`, which calls
   `app/execution/executor.py` and records IC_exec on the run.
4. `feedback` calls `PlaybookLibrary.record_outcome` (`app/library/library.py`) and credits the Eq. 9 reward to the
   routing decisions (`RoutingLearner.credit`; stored in `bandit_arms` / `routing_rewards`).

`app/bootstrap.py` builds all playbook components from `Settings`; `app/main.py` wires the services, routers and the
`/console` static mount.

## Invariants that must be preserved

- **Security:**
  - Every route except `/health` and `/console` uses `require(Permission.X)` (`app/security/auth.py`).
  - `test_every_route_except_health_requires_an_api_key` enumerates the OpenAPI paths, so new routes are covered
    automatically.
  - `X-CADE-Analyst` is an audit label only.
  - Approvals, verifications and destructive attempts are audited, and those audit writes are fatal.
  - The `audit_log` table is append-only (SQLite triggers).
- **Destructive actions:**
  - Rejected without a verification ancestor that verifies their target (validator, before scoring, before plan
    creation, before execution).
  - At run time they need per-node approval bound to the plan hash, and an approver different from the executor (by
    default).
  - A verification must have *confirmed* the concrete target.
  - In live mode they also need `allow_destructive_live` and `execute_live`.
  - Never relax these gates.
- **Simulated vs. real:**
  - The gateway (`app/tools/registry.py`) never substitutes simulation for a failing or unconfigured real tool.
  - Simulated observations live in `EffectiveEvidenceProfile.simulated`, never in `observed`.
  - Decisions and plans using them carry `synthetic_basis`.
  - Simulated runs do not train the library unless `library_learn_from_simulated`, nor the router unless
    `bandit_learn_from_simulated`.
- **LLM output is untrusted:**
  - Phase 1 output is schema-validated, grounded and ATT&CK-validated.
  - Generator output is parsed to steps, converted to typed edges from the vocabulary, then repaired or discarded.
  - Judge samples are schema-validated, and unusable samples are dropped.
- **Phase 1 rules:**
  - A failed analysis never carries a result.
  - Observed evidence needs grounded provenance (`incident_text`, `structured_input`, `pattern_extraction`, `tool`,
    `analyst_supplied`).
  - `SafetyFlags` stay conservative.
- **Logs:** no incident text, model output, tool output or secrets.

## Where things live

| Area | Files |
|---|---|
| Errors (category + HTTP status) | `app/core/errors.py` |
| Settings (`CADE_` env vars) | `app/core/config.py`, documented in `.env.example` |
| Ontology data | `app/data/action_vocabulary.json` (51 actions), `intent_goals.json`, `evidence_availability.json`, `seed_playbooks.json`, `intent_taxonomy.json` (tactics, critical evidence), `log_source_categories.json` (`app/tools/log_sources.py`) |
| Typed DAG | `app/schemas/playbook.py`, `app/playbook/{vocabulary,builder,validator,repair,graph}.py` |
| Tools | `app/tools/{catalog,base,simulated,registry,virustotal,wazuh,elastic,osquery,yara,sigma}.py` |
| Fitness | `app/fitness/{coverage,judge,fitness}.py` |
| Synthesis | `app/synthesis/{context,prompts,generator,constructive,seeding}.py` |
| Evolution | `app/evolution/{operators,selection,engine}.py` |
| Routing | `app/routing/{utility,router,bandit,reward,learning}.py`, `app/storage/bandit_repository.py` |
| Routing evaluation | `app/evaluation/{protocol,metrics,report,paper968}.py`, `scripts/eval_routing.py` (legacy/corrected), `scripts/eval_routing_paper.py`, `scripts/build_paper_catalog.py`, `app/data/paper_benchmark/catalog.json` |
| Learner state maintenance | `RoutingLearner.audit/rebuild` (`app/routing/learning.py`), `scripts/rebuild_bandit_state.py` (dry run unless `--apply`; backs up first), `scripts/analyze_routing_discrepancy.py` |
| Library | `app/library/{embedder,index,library}.py`, `app/storage/library_repository.py` |
| Execution | `app/execution/{executor,internal}.py`, `app/services/plan_service.py` |
| Persistence | `app/storage/` (one repository per area; all return Pydantic models; `run_sync` from `app/core/threads.py` for DB calls in async code) |

## Conventions

- Keep route handlers thin; use cases live in `app/services/`.
- Add errors as `CADEError` subclasses.
- Use `log_event` for structured logs.
- New tool operations go into `app/tools/catalog.py` and must be supported by both the real adapter and the simulator.
  Playbook actions binding them go into `action_vocabulary.json`, which is validated at load: unknown artifacts,
  bad bindings and destructive actions without a `verification_result` input are all rejected.
- New `cade.*` internal operations need a handler in `app/execution/internal.py`; a test checks they match.
- Paper hyperparameters (Table 3) are defaults: w = 0.7, k = 3, N = 10, population 30, ≤ 20 generations, 2 elites,
  20 nodes, B = 20, τ = 0.3 (Jaccard), m = 5, 3 judgments, plateau < 0.01 over 3 generations, LinUCB d = 9 and
  α = 1.0 with a 20-incident bootstrap, ≤ 1 request per incident, λc/λℓ/λq = 0.6/0.3/0.1, and the models qwen3:14b
  (generator) and gemma3:12b (judge). Implementation choices are listed in README §7. `routing_similarity_sharpness` =
  1.0 is exactly Eq. 2.

## Running

```powershell
python -m pytest -q                                   # offline test suite
python -m app.security.keys create --analyst alice --roles analyst
$env:CADE_LLM_PROVIDER="mock"; $env:CADE_EMBEDDER="hashing"; uvicorn app.main:app
python scripts/demo_full_pipeline.py --api-key <key> [--approver-key <key>] [--executor-key <key>]
```

Tests must use synthetic data, the mock LLM, the hashing embedder and NumPy index, and simulated or fake tools.
`make_client(...)` in `app/tests/conftest.py` issues an admin key by default, and accepts `roles`, `app_settings` and
`real_tools`.
