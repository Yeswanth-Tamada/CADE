# Paper 968: text that does not match the implementation

Drafted 2026-11-21 against `968_revised_version.docx`. Each item quotes the paper, says what the code does, and
proposes replacement text. Items marked **[after runs]** can only be finalised once the real experiments have run
(`scripts/eval_quality_paper.py`, `scripts/eval_library_growth.py`, and the execution-backed routing benchmark).

## Must change (the text describes something the project does not do)

1. **Library source** (Section 6.1)
   - Paper: *"The library is seeded from published open-source playbooks, converted to CADE's DAG schema, with at
     least one playbook per benchmark technique"*
   - Code: one playbook per technique is **generated** from CADE's action vocabulary (triage, one query per evidence
     category the technique produces, a Sigma hunt, a verification, a report). Published ATT&CK-mapped playbooks
     (Cortex XSOAR MITRE Courses of Action, austinsonger/Incident-Playbook) cover only 31 of the 82 techniques.
   - Proposed: *"The library holds one playbook per benchmark technique, built from CADE's action vocabulary: triage,
     one query per evidence category the technique produces in the recordings, a Sigma hunt for the technique, a
     verification step and a report. Published ATT&CK-mapped playbook collections covered too few of the benchmark's
     techniques to seed it."*

2. **Models** (Section 3 and Table 3) — only if the runs are not done with these models
   - Paper: Qwen3 14B generator, Gemma 3 12B judge.
   - Code: defaults are exactly these; the runs on the authors' laptop use `llama3.2:3b` (generator) and
     `qwen2.5:3b` (judge) because 12–14B models need a GPU.
   - Proposed (if so): replace the two Table 3 rows with the models actually used, and add to Section 8: *"Generation
     and judging used 3B models on CPU; larger models may produce better playbooks."*

3. **Tactic → intent for IC and EA** (Section 5.1 / 6.1) — the paper does not say how a benchmark tactic gets the
   goal list IC is computed over.
   - Code: each tactic is mapped to the CADE intent that investigates it (execution → suspicious process execution,
     persistence → C2 and persistence, credential access → credential compromise, impact → system impact; the other
     four keep their names).
   - Proposed addition to Section 6.1: *"IC and EA are computed against the intent ontology of the CADE intent that
     investigates each tactic."*

4. **Critical vs supporting categories** (Section 6.1: *"the ontology marks every category as either critical or
   supporting"*)
   - Code: a category is critical for a tactic when at least half of the tactic's candidate techniques produce it.
   - Proposed addition: *"…a category is critical for a tactic when at least half of the tactic's techniques produce
     it in the recordings, and supporting otherwise."*

5. **Incidents** (Section 6.1: *"An incident consists of the dataset's events, presented as structured records with
   a templated one-line summary"*)
   - Code today: the routing and quality benchmarks use the templated one-line summary and the evidence profile;
     no events are loaded. **[after runs]** becomes true once the execution-backed benchmark loads the recordings.

6. **Execution and IC_exec** (Sections 4.3, 6.1, 6.2: playbooks executed against the indexed logs)
   - Code today: execution is simulated (a goal counts as answered when the playbook queries that category and the
     evidence is in hand); latencies in the routing benchmark are the paper's own Fig. 4b values. **[after runs]**
     true only after the execution-backed benchmark. Until then, Section 6.2 must say execution was simulated.

## Results (Section 7 and the abstract) — **[after runs]**

None of these numbers is produced by the current code; each must be replaced with the measured value or removed:

| Paper | Produced by | Current status |
|---|---|---|
| Table 4 (IC, EA, OS, f, latency per system) | `eval_quality_paper.py` | code ready, not run with real models |
| Fig. 2 (fitness per generation, 13.2 generations) | `eval_quality_paper.py` (generation-only runs) | code ready, not run |
| ≈ 90% routing accuracy, macro-F1 0.90, Fig. 3a/b | `eval_routing_paper.py` | 83.1 ± 2.6% with **simulated** execution |
| 14% reward gain over the rule, Fig. 3c | `eval_routing_paper.py` | +19.1 ± 3.7% with **simulated** execution |
| 238 s synthesis, ~110 judge calls, 64.7% judging, 88 s per incident, Fig. 4b | `eval_quality_paper.py` (costs section) | code ready, not run |
| Reuse 28% → 61% as the library grows, Fig. 4a | `eval_library_growth.py` | code ready, not run |
| Abstract: +23.6% IC, +24.6% EA, ≈ 90% accuracy, 14% gain | all of the above | replace with measured values |

## Already correct (no change needed)

Sections 3–5 as written match the code: intent and evidence extraction grounded in ATT&CK; Eqs. 1–9; LinUCB with
d = 9, α = 1, rule for the first 20; fitness Eq. 10–12; B = 20 with adaptive K (average linkage, Jaccard, τ = 0.3);
population 30, 2 elites, ≤ 20 nodes, ≤ 20 generations; six mutations, typed crossover, coverage-stratified binary
tournaments, the three stopping rules; OS_H's four checks; a cross-family judge with three averaged judgments on the
top 5; the self-evolving library; Elastic, Sigma and VirusTotal bindings; 96 scenarios from Splunk Attack Data in the
Table 2 layout, 32 per decision, 10 orderings.

Two documented additions the paper does not mention (worth one sentence each in Section 4): requestability ρ counts a
missing category as obtainable when its availability is ≥ τ, and the LinUCB context is min-max scaled over the
bootstrap incidents.
