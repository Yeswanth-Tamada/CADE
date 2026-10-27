# `app/data` — reference data

| File | Purpose | In git? |
|---|---|---|
| `intent_taxonomy.json` | The configurable intent taxonomy (labels, descriptions, ATT&CK tactics, relevant evidence categories and which of them are critical). | Yes |
| `log_source_categories.json` | Host log sources (Sysmon, Windows Security, PowerShell): channel, event ids and the evidence category of each event (paper 968 §6.1). | Yes |
| `attack/enterprise-attack.json` | MITRE ATT&CK Enterprise STIX 2.1 bundle used to validate technique IDs. | **No** (large, fetched on demand) |
| `action_vocabulary.json` | The controlled playbook action vocabulary: typed inputs/outputs, tool binding, ATT&CK techniques investigated, evidence consumed, destructive / verification / terminal flags. | Yes |
| `intent_goals.json` | Static per-intent investigative goals; supplies Investigative Coverage (IC). | Yes |
| `evidence_availability.json` | Per-deployment availability of each evidence category (routing's rho; also the simulator's data probability). | Yes |
| `seed_playbooks.json` | One hand-authored, validated playbook per intent, loaded into an empty library. | Yes |
| `paper_benchmark/catalog.json` | Frozen metadata (no logs) of 401 Splunk Attack Data recordings at a pinned commit, with ATT&CK-v20-derived evidence categories; input of the paper-matched routing benchmark (`app/evaluation/paper968.py`). Rebuild with `scripts/build_paper_catalog.py` (needs network once). | Yes |

## Intent taxonomy

The labels in `intent_taxonomy.json` are an **implementation choice**, not a verbatim taxonomy from paper
968, which evaluates eight ATT&CK tactics (execution, persistence, privilege escalation, defense evasion,
credential access, discovery, lateral movement, impact). Each intent lists the `tactics` it investigates, so
together the intents cover all eight: incident-type intents (ransomware, phishing, credential compromise,
lateral movement, data exfiltration, DDoS/availability, web-application attack, malware C2 and persistence),
plus `privilege_escalation`, `defense_evasion`, `discovery` and `system_impact` for the tactics no other
intent covered. `suspicious_process_execution`, `malware_investigation` and the
required fallback `unknown` complete it.

To change the taxonomy, edit the file (or point `CADE_INTENT_TAXONOMY_PATH` to another file) and restart
the server. Rules (validated at startup):

* labels are lowercase `snake_case`, unique, and `unknown` must exist;
* `relevant_evidence` and `critical_evidence` use the evidence categories in `app/schemas/enums.py`
  (`EvidenceCategory`), and every critical category is also relevant;
* `tactics` are ATT&CK Enterprise tactic short names (e.g. `privilege-escalation`).

`relevant_evidence` determines which categories are reported as `not_checked` in the evidence profile and
defines the categories behind the Evidence Alignment (EA) metric. `critical_evidence` marks which of them are
critical for the intent (the rest are supporting); routing counts them in R_c (Eq. 1).

## Playbook ontology files

All four playbook files are **implementation choices** (the paper says the ontologies derive from MITRE ATT&CK
but publishes no vocabulary). They are validated at startup (`app/playbook/vocabulary.py`):

* every action binds a tool operation from `app/tools/catalog.py`, an internal step (`cade.*`) or an analyst
  task (`analyst.*`), with consistent input/output artifact types;
* destructive actions take a `verification_result` input and name a `target` that some verification action
  verifies; every non-terminal action outputs `findings`;
* every intent of the taxonomy has goals, and every goal can be addressed by a non-destructive action
  (destructive actions never count towards coverage).

`seed_playbooks.json` is only read when the library is empty (`CADE_LIBRARY_SEED_ON_STARTUP`), so an existing
database keeps its library: delete the database (development) or import new seeds through
`POST /library/playbooks` to get the seeds of newly added intents. Edit `evidence_availability.json` to describe
what your SOC can actually obtain.

## MITRE ATT&CK dataset

The application **never downloads data at startup**. Fetch the dataset explicitly:

```powershell
python scripts/fetch_attack_dataset.py            # pinned version (see the script), ~50 MB
python scripts/fetch_attack_dataset.py --version 19.2
```

This downloads the official STIX bundle from `github.com/mitre-attack/attack-stix-data` to
`app/data/attack/enterprise-attack.json`. You may instead place any ATT&CK Enterprise STIX 2.x bundle at
the path configured by `CADE_ATTACK_DATASET_PATH`.

Without the file, the service runs in **unverified mapping mode**: technique IDs suggested by the LLM are
only format-checked, are labelled `"validation_status": "unverified"`, carry no technique name, and every
analysis includes a warning. `GET /health` reports which mode is active and the dataset version.
