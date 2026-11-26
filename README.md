# CADE: Coverage-Aware DAG Evolution for Adaptive Incident Response

A reference implementation of *"CADE: Coverage-Aware DAG Evolution with Adaptive Routing and LLM-Judged Soundness for
Incident Response Playbook Synthesis"* (paper 968, `968_revised_version (2).docx`). Paper 968 is the only reference
for this implementation, and section numbers below refer to it.

> **Status: the system described in the paper is implemented.** It covers the following:
> - incident understanding, grounded in MITRE ATT&CK tactics and host log sources
> - evidence enrichment through security tools
> - typed playbook DAGs
> - the fitness function f = ∛(IC·EA·OS), with OS from a cross-family LLM judge on the top m candidates (Eq. 10–12)
> - LLM candidate generation with adaptive K (Jaccard clustering)
> - the coverage-aware evolutionary search (six mutations, coverage-directed crossover, stratified binary tournaments)
> - routing: the hand-designed rule (Eq. 1–7) as bootstrap, then a LinUCB contextual bandit (Eq. 8) trained on
>   execution telemetry (Eq. 9)
> - a self-evolving playbook library
> - controlled execution with approvals, verification and audit
> - an analyst console
>
> **Evaluation: built and run with small local models; execution still simulated.** The 96-scenario benchmark
> (Table 2), the routing protocol (10 orderings), the four baselines and the scorer ablation (Table 4), Fig. 2, the
> Fig. 4b costs and the library-growth experiment (Fig. 4a) are implemented (§9). All of them were run on
> 2026-11-22/23 with real local models (3B substitutes for the paper's models, on a 4 GB GPU) but with simulated
> execution; the results so far, and the live runs on the Google Cloud lab, are summarised in §9. No playbook is yet
> executed against indexed Splunk logs, so no experimental result from the paper is claimed for this
> implementation; see `docs/PAPER_TEXT_CORRECTIONS.md`.
>
> **Simulation is the default.** Every security-tool call is answered by a deterministic simulator and labelled
> SYNTHETIC unless you switch to live mode and configure real tools. This is a research/reference system, not a
> production SOAR platform.

---

## 1. How an incident flows through CADE

```
 SIEM alert / analyst note / log excerpt / threat-intel report            (POST /incidents)
                    │
     ┌──────────────▼───────────────┐  LLM → intent label, entities, evidence profile
     │ 1  Incident understanding    │  (observed / unconfirmed / missing[critical|supporting])
     │    + MITRE ATT&CK grounding  │  every claim grounded in the input text   (POST …/analyze)
     └──────────────┬───────────────┘
                    │ abstract representation: intent + evidence profile
     ┌──────────────▼───────────────┐  bootstrap rule: argmax of U_reuse, U_request, U_gen (Eq. 3–7)
     │ 6  Routing layer (Eq. 1–8)   │  then LinUCB: argmax θ̂ᵀx + α√(xᵀA⁻¹x) over the same terms (Eq. 8)
     │    top-N library retrieval   │  trained on r = 1_ret[λc·IC_exec + λℓ(1−ℓ/L_max)] − λq·c_q (Eq. 9)
     └───┬───────────┬──────────┬───┘                                     (POST …/respond)
   reuse │   request │ evidence │ synthesize
         │  ┌────────▼───────┐  │  ┌──────────────────────────────────────────────────────┐
         │  │ 2 Enrichment   │  │  │ 7 LLM candidate pool (B = 20, diversity prompts, prior │
         │  │ VirusTotal,    │  │  │   playbooks) → repair → dedup                          │
         │  │ Wazuh, Elastic,│  │  │ 7 adaptive K: Jaccard clusters cut at τ → K seeds      │
         │  │ osquery, YARA, │  │  │ 5 CADE evolution: coverage-aware init, gap-directed    │
         │  │ Sigma          │  │  │   variation, stratified selection, termination         │
         │  └───────┬────────┘  │  │ 4 fitness f = ∛(IC·EA·OS); 3 typed-DAG validity +     │
         │   re-route (≤1 request) │   destructive-needs-verification hard filter           │
         │                      │  └──────────────────────────┬───────────────────────────┘
   ┌─────▼──────────────────────▼─────────────────────────────▼──┐
   │ 9 Response plan → per-step approval → analyst verification   │  simulate (default) | live
   │   → controlled execution (topological, fail-closed) → audit  │  (POST /response-plans/…)
   └──────────────────────────────┬───────────────────────────────┘
                                  │ feedback: resolved? findings retained?   (POST /playbook-runs/…/feedback)
   ┌──────────────────────────────▼───────────────────────────────┐
   │ 8 Self-evolving library: success counts, ŝ = (s+k)/(n+2k),   │  embeddings + FAISS index
   │   near-duplicate merging, de-prioritisation                  │
   │ 6 Router training: Eq. 9 reward → LinUCB arm update          │
   └──────────────────────────────────────────────────────────────┘
```

### Paper → code map

| Paper | What | Code |
|---|---|---|
| §3 | Intent + evidence profile from text/structured input | `app/intent/` |
| §3, §6.1 | ATT&CK-tactic intents, critical/supporting evidence, host log sources (Sysmon, Security, PowerShell) | `app/data/`, `app/schemas/taxonomy.py`, `app/tools/log_sources.py` |
| §3 | Evidence from security tools | `app/tools/`, `app/enrichment/` |
| §5 | Typed DAG, validity rules, verification-before-destructive hard filter | `app/playbook/` |
| §5.1, §5.4, Eq. 10–12 | IC, EA, OS_H (rule checks), OS_J (3 judgments averaged, top m = 5), geometric mean | `app/fitness/` |
| §5.2 | LLM candidate pool, dedup, adaptive K (average linkage, Jaccard, τ = 0.3) | `app/synthesis/` |
| §5.3 | 6 mutations, coverage-directed typed crossover, stratified binary tournaments, generational replacement, stopping | `app/evolution/` |
| §4.1, Eq. 1–7 | Cov, H, ρ, M, Cov⁺, ŝ, the three utilities and the bootstrap rule | `app/routing/utility.py`, `app/routing/router.py` |
| §4.2, Eq. 8 | Disjoint LinUCB over 8 features + bias | `app/routing/bandit.py`, `app/routing/learning.py` |
| §4.3, Eq. 9 | Reward from execution telemetry (IC_exec, latency, retention, request cost) | `app/routing/reward.py`, `app/services/plan_service.py` |
| §3 | Library: retrieval, write-back, near-duplicates, de-prioritisation | `app/library/` |
| §3, §8 | Execution, safety, approvals, audit | `app/execution/`, `app/services/plan_service.py`, `app/security/` |
| Fig. 1 | Orchestration | `app/services/response_service.py`, `app/bootstrap.py` |

## 2. Quick start (Windows)

The easiest way to run the demo is with the included launcher. Install Python 3.10 or newer, then double-click
[`run_demo.bat`](./run_demo.bat) in File Explorer, or run it from PowerShell:

```powershell
.\run_demo.bat
```

The launcher creates a virtual environment, installs the demo dependencies, and starts CADE in portable simulation
mode. It opens the demo console in your browser and prints a temporary API key and a sample incident to paste there.
The first run may take several minutes while dependencies are installed and the MITRE ATT&CK dataset is downloaded.
Ollama is optional; without it, the demo uses the built-in mock model. Simulated security-tool results are labelled
**SYNTHETIC**.

See [DEMO.md](./DEMO.md) for launcher options, including `-Live`, `-Full`, and `-Port`.

### Manual setup (advanced)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt            # sentence-transformers/faiss are large; see §5 for an offline-only set
Copy-Item .env.example .env

# Issue API keys (each is printed once; only its hash is stored)
python -m app.security.keys create --analyst alice --roles analyst
python -m app.security.keys create --analyst bob   --roles approver
python -m app.security.keys create --analyst carol --roles executor

# Start offline: mock LLM (NOT a real model), hashing embedder, simulated tools
$env:CADE_LLM_PROVIDER="mock"; $env:CADE_EMBEDDER="hashing"
uvicorn app.main:app

# In another terminal: one synthetic incident through the whole pipeline
python scripts/demo_full_pipeline.py --api-key <alice> --approver-key <bob> --executor-key <carol> `
    --example ransomware_analyst_report.json --force-synthesis --verify-target FS-02
```

- Analyst console: <http://127.0.0.1:8000/console/>. Paste a key to connect.
- API docs: <http://127.0.0.1:8000/docs> (not served in production).

### Live conference demo (Google Cloud lab)

[`run_live_demo.bat`](./run_live_demo.bat) sets up the live lab demo from one window. It needs the two lab VMs,
`gcloud`, the SSH key `%USERPROFILE%\.ssh\google_compute_engine` and Ollama on this laptop; the lab itself is
described in [deploy/gcp/README.md](./deploy/gcp/README.md). The lab is not part of paper 968.

```powershell
.\run_live_demo.bat                          # start everything; press Enter in the window after the demo to stop it again
.\run_live_demo.bat start -Page story.html   # the same, with the six-stage screen instead of demo.html
.\run_live_demo.bat stop                     # close the tunnel and stop both VMs (e.g. after closing the window with X)
```

The launcher runs [`deploy/gcp/live_demo.ps1`](./deploy/gcp/live_demo.ps1), which checks each step and stops with
a message if one fails:

1. starts Ollama on the laptop if it is not running;
2. starts the `cade` and `victim` VMs and reads their current external IPs;
3. checks that the laptop's public IP is in the `cade-allow-ssh` firewall rule, and offers to add it (y/n);
4. waits for SSH and checks that the `cade` and `cade-forwarder` services are active;
5. copies the chosen screen to the VM (no restart needed);
6. opens `tunnel.ps1` in its own window and waits until CADE answers through it;
7. reads the `yeswanth` key from `%USERPROFILE%\.cade\cloud_api_keys.txt` and checks that CADE accepts it as an
   analyst key. If it does not, the launcher offers (y/n) to issue a new analyst key on the VM and save it in that
   file, keeping the old file as `.bak`. It then copies the key to the clipboard and opens the screen at
   <http://127.0.0.1:8800/console/demo.html>.

Then attack the victim from a terminal, for example with an Atomic Red Team test (see deploy/gcp/README.md). The
alert forwarder on the `cade` VM notices the attack in Elasticsearch and submits an incident. CADE analyses it,
routes it, builds or reuses a plan and runs it, and the screen follows each step. Both screens read only the live
API; there is no replay mode.

| Screen | What it shows | Use |
|---|---|---|
| `demo.html` (default) | Detailed wall: incident, evidence profile, routing terms (Eq. 3–5), fitness IC/EA/OS, plan DAG, step results | Paste the key, keep "latest incident", click Connect. Until the attack it shows the newest *old* incident. F11 for full screen. "Retain findings…" records analyst feedback, after a confirmation dialog |
| `story.html` | Six stages, one at a time: Alert → Understand → Decide → Build/Reuse/Ask → Run → Learn | Paste the key. It follows only incidents created after the page was opened (10 min grace). F full screen, T theme. "Confirm findings" on the Learn stage records analyst feedback |

- The launcher window prints the API key, so hide it before recording.
- The only changes outside the laptop are starting and stopping the VMs and, only when you answer "y", adding your
  IP to the firewall rule and issuing an analyst key. Nothing changes CADE's incidents or library until someone
  confirms findings on the screen.
- Closing the launcher window with X leaves the VMs running, and they keep billing. Run `run_live_demo.bat stop`.
- Retakes: the forwarder waits 600 s per rule before firing again (`FWD_COOLDOWN_SECONDS`), and `story.html` counts
  incidents created up to 10 min before it opened. Run the attack's cleanup, wait more than 10 minutes, and open
  the screen just before the next attack.
- Results of the rehearsals are in §9 ("Live runs").

## 3. Using real models (Ollama)

The paper uses three independently configured local models (Table 3): intent extraction, playbook generation and an
operational-soundness judge from a **different model family** than the generator.

```powershell
ollama pull llama3.2:3b       # intent extraction (CADE_OLLAMA_MODEL, temperature 0)
ollama pull qwen3:14b         # generator (CADE_GENERATOR_MODEL, temperature 0.8)
ollama pull gemma3:12b        # judge (CADE_JUDGE_MODEL, temperature 0.7); a different family than the generator
uvicorn app.main:app          # CADE_LLM_PROVIDER=ollama is the default
```

The 12–14B models need a GPU (or patience): on smaller hardware set `CADE_GENERATOR_MODEL` / `CADE_JUDGE_MODEL` to
smaller models of two different families (e.g. `llama3.2:3b` and `qwen2.5:3b`).

If the judge and the generator share a family, a warning is logged and shown in `/health`. On CPU, synthesis is slow
(dozens of LLM calls), so `respond`/`synthesize` run as background jobs; poll `GET /jobs/{id}`. To reduce cost, lower
`CADE_GENERATION_POOL_SIZE`, `CADE_OS_JUDGE_SAMPLES` and `CADE_OS_JUDGE_TOP_K`.

`python scripts/check_live_extraction.py` reports how the configured model handles the two manually tested Phase 1
scenarios. It is a report, not a test.

## 4. Security model

### Authentication and roles
Every endpoint except `/health` and the static console requires `Authorization: Bearer <API key>`. Keys are random
256-bit tokens; only their SHA-256 hash is stored. Manage them with `python -m app.security.keys create|list|revoke`.

| Role | Permissions |
|---|---|
| `viewer` | read incidents, plans, runs, library, vocabulary |
| `analyst` | + submit, analyse, enrich, add evidence, route/respond/synthesize, give feedback |
| `approver` | read + approve destructive steps + record verification verdicts |
| `executor` | read + execute plans in **simulate** mode |
| `live_executor` | read + execute in simulate **and live** mode |
| `library_admin` | read + import playbooks into the library |
| `admin` | everything (including `GET /audit`) |

The `X-CADE-Analyst` header is never used to identify or authorise anyone. It is only stored in the audit log as a
claimed label. `CADE_AUTH_ENABLED=false` (local development only) is refused in production, and even then approvals
and live execution are impossible.

### Safety invariants (enforced in code, covered by tests)
- **I1** An invalid DAG, or one with a destructive step lacking a verification ancestor that verifies its target, is
  rejected before scoring, before it becomes a plan, and again before execution.
- **I2** Each destructive step needs an explicit per-step approval by an `approver`. The approval is bound to the
  plan's structural hash.
- **I3** At run time a destructive step acts only on targets that a verification ancestor *confirmed*:
  - analyst verdicts name concrete entity values of the incident;
  - reputation-based verification needs a "malicious" verdict.
  Analyst verdicts are never assumed or simulated.
- **I4** Live destructive actions additionally need `CADE_ALLOW_DESTRUCTIVE_LIVE=true` **and** the `live_executor`
  role. The flag never replaces I2/I3.
- **I5** Missing inputs, approval or verification mean the step is skipped or blocked, with the reason recorded (fail
  closed). By default the approver of a destructive step cannot execute it, and at most one confirmed target is acted
  on per step.
- **I6** Phase 1 safety flags (`human_approval_required=true`) are copied into every plan and never relaxed.
- **I7** An append-only audit log (UPDATE/DELETE blocked by database triggers) records creations, analyses,
  enrichment, routing decisions, approvals, verifications, every execution start and finish, and every destructive
  skip, block, attempt and result. The attempt record is written *before* acting, and a failed write aborts the run.

### Simulated versus real data
- Simulated tool results carry `synthetic=true` and SYNTHETIC labels. They are stored apart from evidence (the
  `simulated` list) and are **never** counted as observed evidence.
- Routing may count simulated evidence only in simulate mode, and any decision or plan that did so is flagged
  `synthetic_basis`.
- When a real tool is configured, a failure or timeout is **never** replaced by a simulated result. It becomes an
  `error` result or a manual investigation task (`CADE_TOOL_FAILURE_POLICY`).
- Simulated runs do not train the library unless `CADE_LIBRARY_LEARN_FROM_SIMULATED=true`.

## 5. Connecting real tools (when you are ready)

Set `CADE_EXECUTION_MODE=live` and configure the adapters you have (see `.env.example`). `GET /tools` and `/health`
show which tools are `real`, `simulated` or `unavailable`.

| Tool | Settings | Scope / notes |
|---|---|---|
| VirusTotal v3 | `CADE_VIRUSTOTAL_API_KEY` | hash/IP/domain lookups only; client-side 4 req/min; 404 = no data, not "clean" |
| Wazuh | `CADE_WAZUH_API_URL`, `…_USERNAME`, `…_PASSWORD`; indexer `CADE_WAZUH_INDEXER_*` | processes, FIM, alerts (indexer); `block_ip` uses `!firewall-drop`; isolation only with your own `CADE_WAZUH_ISOLATION_COMMAND` |
| Elastic | `CADE_ELASTIC_URL`, `CADE_ELASTIC_API_KEY`, `CADE_ELASTIC_INDEX_PATTERN`; `CADE_KIBANA_URL` for isolation | ECS `term` filters only (no query-string interpolation) |
| osquery | `CADE_OSQUERY_BINARY` | the CADE host only; fixed SQL; other hosts are refused |
| YARA | `CADE_YARA_BINARY`, `CADE_YARA_RULES_PATH`, `CADE_YARA_SCAN_ROOTS` | only files resolving inside the scan roots |
| Sigma | `CADE_SIGMA_RULES_DIR` + Elastic; `pip install -r requirements-tools.txt` | every rule tagged `attack.<technique>` (up to 250) → pySigma (≥ 1.5) → Elastic `_msearch`. Rule values are lowercased, so the index must store strings lowercased (keyword + lowercase normalizer) for Sigma's case-insensitive matching. `CADE_SIGMA_FIELD_MAPPING=auditbeat` (lab option, not in the paper) maps SigmaHQ's Linux rules onto Elastic Auditbeat events (`process.executable`, `process.title`, `process.args`, …); auditd `a0..a9` lose their position and the record `type` is dropped |

Every adapter does the following:
- validates and canonicalises inputs before any call;
- uses timeouts, TLS verification on by default, and response-size caps;
- runs subprocesses with an argument list and no shell;
- never logs secrets.

Start with read-only tools. Keep `CADE_ALLOW_DESTRUCTIVE_LIVE=false` until you have tested approvals and
verification with your own (synthetic) incidents.

Paper 968 binds playbook nodes to Elastic, Sigma and VirusTotal, and indexes Splunk Attack Data recordings in
Elastic (§3, §6). To reproduce that locally:
- `python scripts/load_attack_data.py <folder>` indexes downloaded recordings (Sysmon / Security / PowerShell XML)
  into `cade-attack-data` with the ECS fields CADE queries, the raw Windows fields Sigma uses and the lowercase
  mapping above. Timestamps are shifted so the newest event of each file is one hour old (CADE looks back at most
  168 h); the original time is kept in `cade.original_timestamp`.
- `python scripts/check_tools.py --host <host.name>` makes one read-only live call per configured paper tool and
  prints what came back. `health()` alone only checks that settings are present.

## 6. API

| Method & path | Permission | Purpose |
|---|---|---|
| `GET /health` | public (details need a key) | API and dependency status, library, models, tool modes |
| `POST /incidents` · `GET /incidents` · `GET /incidents/{id}` | submit / read | ingest (JSON or text), list, read |
| `POST /incidents/{id}/analyze` | analyze | LLM intent extraction, evidence profile, ATT&CK grounding |
| `GET /incidents/{id}/evidence` | read | effective evidence profile (analysis + tool + analyst + simulated) |
| `POST /incidents/{id}/enrich` · `POST /incidents/{id}/evidence` | enrich | request evidence from tools · add analyst-supplied evidence |
| `POST /incidents/{id}/route` | respond | routing decision only (all Eq. 1–5 terms) |
| `POST /incidents/{id}/respond` · `…/synthesize` | respond | full pipeline / forced synthesis (`?wait=true` for synchronous) |
| `GET /incidents/{id}/response` · `GET /jobs/{id}` | read | decisions, plans, runs · background job status |
| `GET /response-plans/{id}` | read | plan with DAG, fitness, synthesis report, approvals, verifications |
| `POST /response-plans/{id}/approve` · `…/verifications` | approve | approve destructive steps · record a verification verdict |
| `POST /response-plans/{id}/execute` | execute (live: execute_live) | run the plan |
| `GET /playbook-runs/{id}` · `POST /playbook-runs/{id}/feedback` | read · feedback | run results · resolved / findings retained → library |
| `GET /library/playbooks[/{id}]` · `POST /library/playbooks` | read · manage_library | library entries · import (validated) |
| `GET /vocabulary` · `GET /tools` · `GET /audit` | read · read · read_audit | action vocabulary + goals · tools · audit log |

## 7. Implementation choices and differences from the paper

The paper leaves several details open. Each choice below is configurable unless stated, and none is presented as a
paper result.

1. **Metric names.** IC is Investigative Coverage, EA is **Evidence Alignment** and OS is **Operational Soundness**,
   as defined in paper §5.1.
2. **Ontologies.** The intent taxonomy, evidence categories, the 51-action vocabulary, the per-intent goals (IC
   mapping) and the availability profile (ρ) are implementation-defined data files in `app/data/`. They reference
   ATT&CK technique IDs. Each intent names the ATT&CK tactics it investigates (together they cover the paper's eight
   tactics; privilege escalation, defense evasion, discovery and destructive impact were added) and marks each
   relevant category critical or supporting (§6.1). The paper's seven host log-source categories map onto existing
   categories plus a new `script_execution` (PowerShell 4104); `app/data/log_source_categories.json` maps event ids to
   categories. Event records in the incident input (e.g. `EventCode=4104`, or `Sysmon … EventID: 13`) become observed
   evidence with `pattern_extraction` provenance, and three Elastic operations search by channel and event code
   (registry, script blocks, service/task creation).
3. **R_c / R_s of a playbook** are the evidence categories its actions consume; critical if any action marks the
   category critical **or the incident's intent marks it critical**. **A** is the set of categories observed so far.
4. **Cov⁺** adds the missing critical categories whose availability is ≥ τ (default 0.5). **ρ** is the mean
   obtainability over the missing critical categories, a category counting as obtainable (1) when its availability
   is ≥ τ and 0 otherwise — the same test, and what the paper benchmark's 'marked obtainable' does. Averaging the
   raw availability probabilities instead (`CADE_ROUTING_REQUESTABILITY=mean`, the behaviour before 2026-11-16)
   keeps ρ·M below 1, so (1 − ρM) in Eq. 5 never reaches 0 and, with H ≈ 1 (item 6), a request can never win:
   on the paper benchmark with the deployment profile the rule requested in 0 of 32 request scenarios, against 27
   with the thresholded ρ.
5. **Empty library.** With no candidates, Cov = 0 and H = 1, so the decision is synthesize.
6. **Routing entropy (important).** Eq. 2 normalises the *raw* cosine similarities of the top N = 10 candidates.
   Measured over the 11 seed playbooks:
   - With `all-MiniLM-L6-v2` + FAISS, retrieval put the correct playbook first for 9/11 intents (e.g. 0.80 vs 0.63),
     yet H was 0.985–0.998 for every intent.
   - With the hashing embedder, H was 0.91–0.96.

   (1 − H) is therefore about 0.01–0.09, which suppresses both reuse and evidence requests, so the router synthesizes
   unless a candidate covers all evidence. CADE implements Eq. 2 exactly by default. `CADE_ROUTING_SIMILARITY_SHARPNESS`
   > 1 (π ∝ sᵖ) is an opt-in deviation; p = 6 gave H ≈ 0.5–0.9 on the same data. Choosing it is a research decision,
   not a fix. Sharpening alone is not enough: Eq. 3 and Eq. 5 give U_reuse ≤ Cov·(1 − H)·ŝ with ŝ = 0.5 for an
   unused entry, against U_gen = (1 − Cov)(1 − ρM). In a live run (p = 6, credential-compromise incident), Cov = 0.70
   and H = 0.69 gave U_reuse = 0.11 against U_gen = 0.30, so reuse needs nearly complete evidence coverage *and* a
   confident retrieval (or a success history raising ŝ).
7. **Reuse vs. history.** Paper §8 conditions reuse on analyst-retained outcomes, so synthesized playbooks enter
   the library only after a resolved, retained run. Curated seed/imported playbooks are reuse-eligible as the
   hand-authored baseline (`CADE_LIBRARY_ALLOW_CURATED_REUSE`). Synthesized playbooks are generalised (entity inputs
   re-wired from `extract_entities`) before storage, so they apply to other incidents.
8. **Destructive actions never earn coverage.** Goals are investigative only, so the search is not rewarded for
   adding destructive steps. Operators never add them; they appear only if the generator proposes them, and repair
   keeps them verified or removes them.
9. **Orphan rule.** "No node is orphaned" is enforced as: every node must have a path to a terminal action.
10. **OS_H** (§5.4) is the fraction of rule checks passed, over every individual check: each node's tool fits its
    action type (containment binds a response operation, everything else a read-only one); each tool reads only
    categories its operation supports; each destructive step has an investigative ancestor (not the verification)
    taking one of its entity artifacts; no node repeats earlier work (same action and parameters, or the same tool
    operation on the same target type). Curated seed playbooks score 1.0. OS_H ranks candidates; the LLM judge
    scores the top m = 5 per generation plus the leader (cached per structure), and elites and the returned playbook
    are always judged. With the mock provider, OS_H is the OS.
11. **Evolution details.** The six mutations are the paper's; "rebind a node to another tool" substitutes an action
    of the same phase that reads the same kind of evidence through a different tool operation (no schema change).
    Gap-directed additions use the generator's proposal for the gap: one structured call per synthesis asks it to pick
    a closing action for every gap the seeds leave open (`CADE_DIRECTED_MUTATION_LLM`); answers outside the allowed
    closers are ignored. A mutation is chosen uniformly among those applicable to the parent, retried
    `CADE_MUTATION_RETRIES` times, then the parent is kept. Strata are candidates with the same number of covered
    items. `CADE_GENERATIONAL_REPLACEMENT=false` restores (μ + λ) survival.
12. **Thresholds.** τ = 0.3, B = 20, m = 5, plateau < 0.01 over 3 generations and the request cap of 1 are the paper's.
    Values the paper does not give (near-duplicate cosine 0.95 / structural Jaccard 0.8, de-prioritisation n ≥ 5 &
    ŝ < 0.35) are defaults, not tuned results. K is unclamped by default (`CADE_SEED_K_MIN/MAX` are optional guards),
    and `CADE_FITNESS_THRESHOLD` (default 0) can add a fitness requirement to the full-coverage stop.
13. **Learned routing.** The context x is Cov, Cov⁺, H, ρ, M, ŝ, the top candidate's similarity and the library size
    n/(n + 50), plus a bias (d = 9, all in [0, 1]). **Context scaling** (`CADE_BANDIT_FEATURE_SCALING`, default on;
    the paper states no input scaling): each feature is min-max scaled with its range over the decisions of the
    bootstrap incidents and clipped to [0, 1]; the range is fixed when the bootstrap ends, like L_max, and a
    feature that did not vary there is left as it is. Without it the top similarity, which moves within about
    0.15, cannot earn a useful weight under ridge λ = 1 (item 17). Decisions and the reward log keep the raw
    features, and bootstrap rewards are held back until the scale exists. Learner state trained before this change
    is reported as inconsistent at start-up and is only rebuilt by `scripts/rebuild_bandit_state.py`. The bandit
    only considers arms that can act (reuse needs a
    candidate, a request needs obtainable evidence and the per-incident cap not reached). Rewards are credited when
    the analyst submits feedback on an executed run; a run blocked before executing is not credited, and simulated runs
    do not train the router unless `CADE_BANDIT_LEARN_FROM_SIMULATED=true` (as for the library). The router's state
    and every reward are stored (`bandit_arms`, `routing_rewards`) and survive restarts. `CADE_ROUTING_POLICY=rule`
    keeps the hand-designed rule (the router ablation) while still logging rewards.
14. **Reward terms.** IC_exec is the fraction of the intent's goals answered by an executed, non-destructive step that
    returned data (`no_data` is not an answer). ℓ is machine time: routing to plan creation plus the run itself,
    excluding the time a plan waits for approvals. L_max is `CADE_REWARD_LATENCY_MAX_SECONDS`, else the 90th
    percentile of the first 20 rewarded incidents (paper §6.2). Until that percentile exists, rewards are
    *provisional*: they are stored (scored with the 600 s fallback, flagged `provisional`) but not applied to the
    bandit; when the 20th latency arrives they are rescored with the final L_max and applied in the order they were
    logged, so the bandit trains on exactly Eq. 9. The rule chooses throughout the bootstrap, so this changes no
    routing decision. (Before 2026-10-26 the bootstrap rewards were applied immediately with the 600 s fallback,
    which over-credited synthesis by ≈ 0.18 per bootstrap incident; rewards stored by that version carry no flag
    and are left as they are: start-up audits the stored state and reports such rewards in `/health` warnings, and
    `scripts/rebuild_bandit_state.py` shows — and only with `--apply`, after backing the database file up — rebuilds
    the arms from the rescored reward log.) Reward components are stored at full precision so they can be rescored
    exactly.
    Retention is the analyst's `findings_retained` (the paper's automatic retention rule is a benchmark device and
    is not used).
15. **Intent extraction (Part 1).** Paper §3 describes intent extraction into a canonical label, then evidence
    extraction; `CADE_INTENT_TWO_STAGE=true` (default) does exactly that with two calls to the local intent model:
    - **stage 1** writes the activity, then an ATT&CK technique, then its tactic, then the label. With a single
      combined call, llama3.2:3b labelled 10 of 13 incidents with the first allowed label;
    - **ATT&CK grounding**: technique names found in the text and well-known technique artifacts
      (`app/data/attack_artifact_hints.json`, Sigma-style patterns) are shown as hints with their dataset tactics.
      When they match, the tactic is chosen among theirs. A label that covers neither the chosen tactic nor a
      hinted one is re-chosen among the labels that do. Hints only steer the model, and every technique is
      validated against the dataset;
    - **event digest**: each Windows event record is restated as one line (`Sysmon event 13 (registry value set):
      TargetObject=...`) in the prompt only, never in the grounding text;
    - **stage 2** writes the evidence profile with the label fixed;
    - **grounding**: a missing item counts as stated by the incident only if its quote lies in an absence sentence
      that names that evidence (a bare "does not include" no longer qualifies). Hosts and users are extracted from
      event fields (`Computer=`, `User=`, `TargetUserName=`, ...). A model-server error (HTTP 5xx) gets one more
      attempt, and if stage 1 stays unusable the analysis falls back to one call;
    - **ATT&CK v20** split defense evasion into `stealth` and `defense-impairment`; the taxonomy and hints accept
      both numberings.

    `scripts/eval_intent.py` measures this on SYNTHETIC record-style incidents (`examples/intent_eval/`).
    llama3.2:3b, temperature 0, local:

    | Set | Intent | Technique | Entities | Observed | Stated gaps |
    |---|---|---|---|---|---|
    | tuned set (13; prompts and hints were developed on it), single call before the changes | 3/13 | 3/13 | 10/13 | 13/13 | 12/13 |
    | tuned set, two-stage | 12/13 | 13/13 | 13/13 | 13/13 | 13/13 |
    | **held-out set (10 unseen attacks, 9 without an artifact hint)**, two-stage | **3/10** | **3/10** | 10/10 | 10/10 | 10/10 |

    The deterministic parts generalise; the label for an attack without an artifact hint does not (see §11).
    Offering candidates retrieved from the ATT&CK dataset (BM25 over technique descriptions and analytics) put
    the right technique in the top 10 for all 10 held-out cases, but the 3B model still picked wrongly (3/10), so
    it is not used.
16. **Execution is stricter than the paper**, which describes automatic execution after selection. This
    implementation adds per-step approval, explicit verification, RBAC and fail-closed gates.
17. **Routing evaluation** (`app/evaluation/`, results in `reports/2026-10-26_routing_benchmarks/`). Three benchmarks,
    always reported separately, all run through one protocol (`app/evaluation/protocol.py`, paper §6.4): 10 orderings
    (seeds 0–9), a fresh production `RoutingLearner` per ordering, the rule for the first 20 decisions, LinUCB after,
    and the rule alone on the same orderings with the same Eq. 9 reward. L_max is the 90th percentile of each
    ordering's first 20 latencies for both policies, and the protocol checks that the bandit trained on exactly the
    reported reward. Reports give accuracy, per-action P/R/F1, confusion matrices, reward before/after the bootstrap,
    mean ± std over orderings, per-cell results and how many labels Eq. 9 does not uniquely favour.
    - **Common to all three:** execution is SIMULATED (a construction-based stand-in decides IC_exec and retention),
      latencies are ASSUMED (Fig. 4b: reuse 6 s, the whole request path 8.4 s, synthesis 238 s; a request followed by
      synthesis costs 240.4 s), the incident analysis is fixed ground truth, and a run refuses a substituted embedder
      or index unless `--allow-fallback` (then marked DEGRADED). Nothing here reproduces the paper's execution-backed
      evaluation on indexed Splunk logs.
    - **Paper-matched** (`scripts/eval_routing_paper.py`, `app/evaluation/paper968.py`): 96 scenarios, 8 tactics × the
      12 Table-2 cells (2-2-1-2-1-1-1-1-1), 32 per label. Built from Splunk Attack Data YAML metadata at a pinned commit
      (`app/data/paper_benchmark/catalog.json`, `scripts/build_paper_catalog.py`; no logs): a recording's evidence
      categories are the ATT&CK v20 Windows detection analytics' event codes (mapped with §6.1) that its declared log
      sources carry — a proxy. The ontology is derived (a category is critical for a tactic when ≥ half its
      techniques produce it), the library holds one generated playbook per technique (88), weak cells keep only one
      designated related technique of the tactic, and R_c / R_s come from each stored playbook's own intent
      (`Router(requirements=...)`). Execution goals are the recording's critical categories (`--goals all` as a
      sensitivity). 8 sparse cells coincide with another cell's evidence and 3 reuse a recording from another cell
      (thin recordings); all are flagged. Results with context scaling (2026-11-16): LinUCB 83.1 ± 2.6% (86.6%
      after the bootstrap; REUSE / REQUEST / GENERATE recall 86 / 86 / 78%), rule 65.6%, reward +19.1 ± 3.7%, 10/96
      labels not uniquely reward-optimal; held-out orderings (seeds 10–19) 82.1 ± 3.9%. Without scaling: LinUCB
      66.7 ± 8.5%, reward +9.1 ± 7.6%. The paper reports ≈ 90%, 73.5% and 14%.
    - *The two extended benchmarks' numbers below predate context scaling and the thresholded ρ and were not rerun.*
    - **Legacy extended** (`scripts/eval_routing.py --benchmark legacy`): the original 123 SYNTHETIC intent-level
      scenarios, unchanged. Known defect: only 7 of 40 "weak" scenarios retrieve a same-tactic candidate. With the
      corrected protocol: LinUCB 57.5 ± 4.9%, rule 54.5%, reward +49.9 ± 5.6% (before: 56.7%, +49.0%).
    - **Corrected extended** (`--benchmark corrected`): weak cells keep one designated same-tactic playbook; 30 weak
      scenarios that cannot be valid weak matches are excluded and listed. 93 scenarios (28/16/49): LinUCB
      53.4 ± 2.4%, rule 66.7%, reward +17.6 ± 6.4%.
    - **What limited LinUCB before context scaling:** what the rule experienced in the 20 bootstrap decisions
      decided each ordering (REUSE recall 9–91% on the paper benchmark): the only feature separating a suitable playbook from another
      tactic's is the top similarity, whose useful range (≈ 0.15) needs a coefficient that ridge λ = 1 on unscaled
      features suppresses at this sample size (λ is CADE's choice, A_a = I; the paper does not state it). A diagnostic
      run with λ = 0.01 gives 76.6 ± 2.8%; it is not adopted. Scaling the context instead (item 13) keeps λ = 1 and
      gives the 83.1% above; a full-information linear fit on the scaled features picks the label on 92 of 96
      scenarios, against 64 on the raw ones. Causes of the gap to the paper, ranked with evidence:
      `reports/2026-10-26_routing_benchmarks/discrepancy_analysis/REPORT.md`.

## 8. Phase 1 stabilisation (done before Phases 2–9)

Manual tests of Phase 1 were analysed against the stored records in `cade.db`. Each finding is classified below, and
every change below has a regression test in `app/tests/test_phase1_regressions.py`.

| Finding | Classification | Change |
|---|---|---|
| Scheduled-task analysis had no ATT&CK techniques | LLM limitation (the model proposed none; nothing was discarded). The stored record could not prove it | `attack_candidates_proposed` recorded. Optional prompt hints: technique names from the loaded dataset that occur in the text (`CADE_ATTACK_PROMPT_HINTS`); results still come only from the model and are validated |
| Exact Windows-path quotes downgraded | **Defect** (reproduced by a failing test first): grounding did not normalise JSON-doubled backslashes | backslash runs collapsed in grounding; different paths still never match |
| "taskeng.exe launching …from C:\\…" downgraded | Expected: a paraphrase, not a quote | locked in by a test |
| Stated gaps (reputation, hash, task XML, alert) inconsistently represented | Mostly LLM limitation, plus a design gap | missing items may quote the sentence stating the gap (provenance `incident_text` vs `llm_inference`). Info warning when a stated gap is not recorded. Prompt v2 category guide |
| Invalid-domain and duplicate-evidence warnings | Expected (safe discard / documented precedence) | none |
| No authentication | Security gap | API keys + RBAC + audit (§4) |

The live report after the change (one run, llama3.2:3b) showed the following:
- the scheduled-task scenario now proposed and validated **T1053.005**, and recorded three of four stated gaps with
  `incident_text` provenance;
- the PowerShell scenario kept T1059.001.

Remaining model errors (miscategorised evidence, over-listing missing items) are limitations of a 3B model, not of
the pipeline.

## 9. Tests

```powershell
python -m pytest -q          # offline: mock LLM, hashing embedder, NumPy index, simulated tools, synthetic data
python scripts/eval_intent.py                                          # live intent evaluation (needs Ollama)
python scripts/eval_intent.py --cases examples/intent_eval/heldout.json  # unseen attacks
python scripts/eval_routing_paper.py --out-dir reports/<dir>/paper968   # paper-matched 96-scenario routing benchmark
python scripts/eval_routing.py --benchmark legacy --learned            # legacy 123-scenario benchmark, rule + LinUCB
python scripts/eval_routing.py --benchmark corrected --learned         # weak-match semantics enforced (§7 item 17)
```

Experiments with the configured models (hours to days on CPU, much faster on a GPU; every synthesized playbook is
cached under `reports/quality_cache/<generator>__<judge>/`, shared by all three, so runs resume and reuse each other):

```powershell
python scripts/eval_routing_generated.py --time-test                    # one real synthesis: how long it takes here
python scripts/eval_routing_generated.py --out-dir reports/<dir>/routing_generated   # LinUCB with REAL synthesized playbooks
python scripts/eval_quality_paper.py --out-dir reports/<dir>/quality `
    --routing-results reports/<dir>/routing_generated/results.json      # Table 4, Fig. 2, Fig. 4b, scorer ablation
python scripts/eval_library_growth.py --out-dir reports/<dir>/growth     # Fig. 4a
# add --mock to any of them to check the wiring offline (NOT a result)
```

In all three, execution is still simulated (a goal counts as answered when the returned playbook reads that evidence
category and it is in hand; no logs are loaded), incidents are the templated summary plus the evidence profile, and
the library is generated (one playbook per technique). Each report lists every departure from the paper.

The suite covers the Phase 1 pipeline and regressions, security (401/403 on every route, audit, append-only log,
production checks), tool adapters (mocked HTTP and subprocess), enrichment honesty, DAG validity and repair, fitness,
generation and adaptive K, evolution operators, selection and termination, Eq. 1–9 (rule, LinUCB, reward and the
training loop), the ATT&CK / log-source ontology, the library, execution gates I1–I7, and the full HTTP pipeline. No
test contacts a real tool or performs a destructive action.

### Results so far

`reports/` is git-ignored, so the full reports stay on the machine that produced them. The numbers below are copied
from them, and each block names its report folder. Execution is simulated unless a block says *live*, the models
are 3B substitutes for the paper's (Table 3), and none of this is a reproduction of the paper's evaluation.

**Offline test suite** (2026-11-26): 577 passed, 1 failed (4 min). The failure,
`test_the_default_build_is_the_archived_96`, compares the benchmark with the local archive
`reports/2026-10-26_routing_benchmarks/paper968/scenarios.json`, whose first bytes are stray text ("how did "), so
it is not valid JSON. The archive is git-ignored and protected, so it was left unchanged. On a fresh clone there is
no archive, the test skips that comparison, and all 578 tests are expected to pass.

**Intent extraction** (Part 1, llama3.2:3b): 12/13 intents on the tuned set, 3/10 on the held-out set (§7 item 15).

**Routing with simulated synthesis** (`reports/2026-10-26_routing_benchmarks/`; paper-matched rerun with context
scaling and the thresholded ρ on 2026-11-16). Mean ± std over 10 orderings:

| Benchmark | LinUCB | Rule (Eq. 7) | Reward gain | Paper |
|---|---|---|---|---|
| Paper-matched, 96 scenarios | 83.1 ± 2.6% | 65.6% | +19.1 ± 3.7% | ≈ 90%, 73.5%, +14% |
| Paper-matched, held-out orderings (seeds 10–19) | 82.1 ± 3.9% | | | |
| Legacy, 123 (before scaling) | 57.5 ± 4.9% | 54.5% | +49.9 ± 5.6% | |
| Corrected, 93 (before scaling) | 53.4 ± 2.4% | 66.7% | +17.6 ± 6.4% | |

The pre-registered reproduction (`reports/2026-10-26_routing_reproduction_968/`) reruns the archived benchmark
exactly (0/960 mismatches). Details and the ranked causes of the gap are in §7 item 17.

**Experiments with real local models** (step 1, `reports/2026-11-22/`, commit c992e8e). Generator llama3.2:3b, judge
qwen2.5:3b, all-MiniLM-L6-v2 + FAISS, on an NVIDIA RTX 2050 (4 GB). Ollama ran with `OLLAMA_KV_CACHE_TYPE=q8_0` and
`OLLAMA_FLASH_ATTENTION=1` (the q8_0 cache fits all of llama3.2:3b on the GPU and cut one synthesis from 485 s to
357 s at the same fitness), and `CADE_LLM_REQUEST_TIMEOUT_SECONDS=300`. Every synthesis is cached, so the three
scripts share them.

- *Routing with real synthesized playbooks* (`eval_routing_generated.py`, 8.1 h, 99 syntheses, 0 failed, mean
  290 s each): LinUCB **82.4 ± 2.9%** (macro-F1 0.827), rule 65.6%, reward gain +19.2 ± 3.4%. This agrees with
  the simulated-synthesis result above.
- *Table 4* (`eval_quality_paper.py`):

  | System | IC | EA | OS | f | Latency (s) | Paper f |
  |---|---|---|---|---|---|---|
  | Static SOAR | 0.95 | 0.88 | 0.72 | 0.84 | 0.0 | 0.64 |
  | LLM-only | 0.86 | 0.83 | 0.60 | 0.74 | 22.3 | 0.69 |
  | Retrieval-only | 0.61 | 0.48 | 0.69 | 0.57 | 0.1 | 0.72 |
  | Generation-only | 1.00 | 1.00 | 0.67 | 0.88 | 291.4 | 0.81 |
  | Generation-only, OS_H only (scorer ablation) | 1.00 | 1.00 | 0.66 | 0.87 | 238.7 | 0.77 |
  | **CADE (full)** | 0.80 | 0.74 | 0.67 | **0.72** | 149.4 | **0.86** |

  CADE (full) synthesized for 51% of incidents (paper: 35%). The report's line "latencies measured on this
  machine (CPU)" is template text: the run used the GPU.
- *Fig. 2*: flat. Search stopped after 0.3 generations on average (paper: 13.2); 78 of 99 syntheses stopped at
  generation 0, because the first draft pool already had IC = EA = 1, so fitness varied only through OS.
- *Fig. 4b*: 291 s per synthesis (paper: 238 s), 51 judge calls (paper: about 110), judging 17% of synthesis time
  (paper: 65%).
- *Fig. 4a* (`eval_library_growth.py`, rule routing, one pass, 96 incidents): the library grew from 10 to 52
  playbooks.

  | Library size | ~10 | ~20 | ~30 | ~40 | ~50 |
  |---|---|---|---|---|---|
  | Reuse rate | 42% | 36% | 38% | 44% | 46% |
  | Paper | 28% | 41% | 50% | 57% | 61% |

  Each bin holds only 12–28 incidents.

Why Table 4, Fig. 2 and Fig. 4a differ from the paper (analysis of 2026-11-23; nothing has been changed for it):
- **Two ontologies.** Routing and the library use the benchmark's derived ontology, but Table 4 scores IC/EA with
  CADE's app intent ontology (`TACTIC_INTENT` in `app/evaluation/quality.py`). A correctly reused playbook can
  score low (credential-access reuse of the right T1003 playbook gets IC 0.20). 49% of CADE's decisions are
  reuses, which score f ≈ 0.55 against 0.87 for syntheses, so CADE (full) falls below generation-only.
- **Library and Static SOAR.** The library is minimal generated stand-ins (one per technique), not converted
  published playbooks. Static SOAR is CADE's own per-intent seeds, written for the same ontology that scores them.
- **Evolution stops at once.** The generation prompt lists the goals and categories that IC/EA count, so the 3B
  generator covers all of them in the first drafts and the paper's full-coverage stop ends the search. This also
  leaves the scorer ablation with almost no gap.
- **Synthesis after a request.** After a correct REQUEST, LinUCB synthesizes in 60% of cases instead of reusing,
  which raises the synthesis share and CADE's latency.

Open questions for the paper authors: the goals per intent, whether the generation prompt listed them, the sources
of the library and of Static SOAR, and whether Fig. 4a used LinUCB or the rule.

**Live runs** (real tools; functional tests on single attacks, not benchmarks):

| Date | Setup | Attack | Outcome |
|---|---|---|---|
| 2026-10-24 | Laptop; real Elastic, Sigma, VirusTotal (`reports/2026-10-24_full_table3_live_run/`) | Recorded encoded-PowerShell attack | T1059.001 found; rule chose synthesize; f 0.933 (IC 1, EA 1, OS 0.81), 18 nodes in 10 min; live run completed with issues, IC_exec 1.0 |
| 2026-10-26 | Google Cloud lab, attack from the laptop | VIC-SSH-0001: 7 invalid-user SSH logins and enumeration commands | T1087 + T1033; synthesize; f 0.917, 12 nodes; live run, execution coverage 0.4 |
| 2026-10-26 | Google Cloud lab | VIC-PERSIST-0001: new user, root cron job, systemd unit | T1543.002 + T1569.003; f 0.933; live run, execution coverage 0.833 |
| 2026-11-23 | Lab, first full rehearsal with `run_live_demo.bat` | Atomic Red Team T1136.001-1 (`useradd`) | Forwarder detected it after 35 s; intent after 34 s; rule chose build; 6.0 min build, f 0.90; 17-step run (6 done, 2 nothing found, 2 analyst tasks, 3 tool unavailable, 4 skipped), 33% of goals answered |
| 2026-11-24 | Lab, video dry run on c992e8e | Same test | Detected after about 35 s; understood in 38 s; 4.7 min build, f 0.92 (all 20 drafts valid); 12-step run (5 done, 1 nothing found, 1 analyst task, 2 tool unavailable, 3 skipped), 50% of goals answered; 5.3 min in total |

Problems seen in these runs and not yet fixed:
- the account-creation test is labelled T1087.001 (Account Discovery) instead of T1136.001, and "user accounts" is
  listed as missing evidence;
- osquery steps are "tool unavailable" on the victim, because osquery only covers the CADE host;
- Windows-only queries (registry, Windows event IDs) are planned against the Linux victim;
- only one of several techniques is captured when an attack combines them (VIC-PERSIST-0001 missed T1136 and
  T1053.003);
- IP reputation is looked up for internal addresses, and `CADE_MAX_TARGETS_PER_NODE=3` caps user lookups.

No feedback was recorded for any live run, so none of them trained the library or the router. A second attack
that reuses the confirmed playbook has not been tested yet.

## 10. Project structure

```
app/
  api/            HTTP routes (incidents, evidence, response, library) and error handling
  security/       API keys, roles/permissions, authentication dependency
  intent/         Phase 1: prompts, LLM clients (Ollama, mock), extraction, grounding, ATT&CK mapping
  tools/          tool catalog, simulators, VirusTotal/Wazuh/Elastic/osquery/YARA/Sigma adapters, gateway
  enrichment/     evidence enrichment and the availability profile
  playbook/       action vocabulary, typed DAG builder, validator, repair
  fitness/        coverage (IC, EA), OS judges, fitness evaluator
  synthesis/      LLM candidate generation, constructive fallback, adaptive-K seeding, the shared synthesis pipeline
  evolution/      operators, selection, the CADE engine
  routing/        Eq. 1–7 (rule), LinUCB bandit (Eq. 8), reward (Eq. 9), router and learner
  evaluation/     online protocol, metrics, reports, the paper-matched 96-scenario benchmark, real-synthesis routing
                  (generated.py), playbook quality and baselines (quality.py), library growth (growth.py)
  library/        embedder, vector index, self-evolving library
  execution/      internal steps and the gated executor
  services/       use cases: incidents, evidence, plans, response orchestration, jobs, audit
  storage/        SQLAlchemy models and repositories
  web/static/     analyst console (no external assets); demo.html and story.html, the live demo screens
  data/           taxonomy, vocabulary, goals, availability, seed playbooks, ATT&CK bundle (fetched)
scripts/          demo_workflow.py, demo_full_pipeline.py, check_live_extraction.py, fetch_attack_dataset.py,
                  check_tools.py, load_attack_data.py, eval_intent.py, eval_routing.py, eval_routing_paper.py,
                  eval_routing_generated.py, eval_quality_paper.py, eval_library_growth.py, build_paper_catalog.py,
                  alert_forwarder.py (lab VM detection bridge)
deploy/gcp/       lab VM tunnel, deploy script, live demo launcher (live_demo.ps1, run by run_live_demo.bat), systemd
                  units, victim Auditbeat config (see deploy/gcp/README.md)
docs/             PAPER_TEXT_CORRECTIONS.md, GPU_RUN_GUIDE.md (running the experiments on a GPU laptop)
examples/         SYNTHETIC incidents (10)
```

## 11. Known limitations

- Single process; background jobs are in-process asyncio tasks (no external queue). Evolution is CPU work on the
  event loop, so very large populations can delay other requests.
- SQLite with `create_all` and no migrations. Delete `cade.db` after schema changes in development.
- osquery covers only the CADE host. Wazuh isolation needs your own active-response script. Analyst-bound actions
  (memory acquisition, account disabling, …) become manual tasks.
- The LLM judge has not been validated against human ratings (paper §8 makes the same caveat).
- **Intent on unseen attacks is weak with the paper's 3B intent model**: 3/10 on the held-out set (§7 item 15).
  The label is reliable when an artifact hint matches, the entities and evidence profile are reliable
  regardless, and the analyst can correct a wrong label. Wider deterministic grounding (e.g. public Sigma rules)
  or a larger local model would help; neither is done.
- Grounding is lexical: a true but paraphrased excerpt is downgraded.
- The paper's evaluation is built (§9) but **execution is simulated** everywhere: no playbook runs against indexed
  Splunk logs, and incidents carry no events (the paper's execution-backed evaluation needs Elasticsearch with the
  recordings loaded). The library is generated, not converted from published playbooks. Statistical tests beyond
  mean ± std over orderings are not implemented.
- **Routing with the exact Eq. 2** gives H ≈ 1 (raw cosine similarities), so the rule reuses only when coverage is
  complete, and requests evidence only when all of the top playbook's critical evidence is missing and obtainable
  (§7 items 4, 6 and 17). LinUCB reaches 83% on the paper-matched benchmark against the paper's ≈ 90% with
  simulated execution and assumed latencies (§7 item 17), and 82.4% with real synthesized playbooks (§9).
- With the 3B models, Table 4 does not match the paper (CADE full f 0.72 against 0.86) and the evolutionary search
  usually stops at generation 0; the known causes are listed in §9.
- The learned router needs real, executed and reviewed incidents: in the default simulate mode it stays on the
  bootstrap rule. A low-volume deployment may never leave the bootstrap period, and the linear reward model misses
  feature interactions (paper §8).
- SQLite tables created by an older version are not altered; the new `bandit_arms` / `routing_rewards` tables are
  added automatically, everything else is stored as JSON.
