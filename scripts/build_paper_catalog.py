"""Build the frozen catalog behind the paper-matched routing benchmark (paper 968 Section 6.1).

Paper 968 builds its 96 scenarios from Splunk Attack Data: recordings of ATT&CK techniques whose
YAML lists the technique IDs and log-source types. This script snapshots exactly that metadata --
no logs -- at a pinned commit, and derives which of the paper's seven evidence categories each
recording can contain:

* a recording's *sources* are the Windows channels its YAML declares (Sysmon, Security, PowerShell);
* a technique's *attributable categories* come from the ATT&CK v20 detection analytics for Windows:
  every ``EventCode=`` of those channels is mapped to a category with the paper's Section 6.1 table
  (process creation = Sysmon 1 / Security 4688, network = Sysmon 3, file = Sysmon 11, registry =
  Sysmon 12-14, authentication = Security 4624/4625/4648, script = PowerShell 4104, service and
  scheduled task = Security 4697/4698);
* a recording's categories are the attributable categories its declared sources can carry.

This is a proxy for "the recording has attributable events in this category": the logs themselves
are not downloaded or indexed. Network access is needed once (GitHub); the output
``app/data/paper_benchmark/catalog.json`` is committed, so the benchmark runs offline afterwards.

    python scripts/build_paper_catalog.py                          # pinned commit, writes the catalog
    python scripts/build_paper_catalog.py --offline --cache-dir D  # rebuild from a cached download
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

REPOSITORY = "splunk/attack_data"
COMMIT = "115785491ca34054ac441a2ca9e87fba40b42a6b"  # master on 2026-10-23
OUT = ROOT / "app" / "data" / "paper_benchmark" / "catalog.json"
ATTACK = ROOT / "app" / "data" / "attack" / "enterprise-attack.json"

# The paper's eight tactics (Section 1) and their ATT&CK v20 short names (v20 split defense evasion).
TACTICS = {
    "execution": ["execution"], "persistence": ["persistence"], "privilege_escalation": ["privilege-escalation"],
    "defense_evasion": ["defense-evasion", "stealth", "defense-impairment"], "credential_access": ["credential-access"],
    "discovery": ["discovery"], "lateral_movement": ["lateral-movement"], "impact": ["impact"],
}  # fmt: skip
# Paper Section 6.1: event IDs of each evidence category.
EVENTS = {
    ("sysmon", 1): "process_telemetry", ("security", 4688): "process_telemetry", ("sysmon", 3): "network_traffic",
    ("sysmon", 11): "file_activity", ("sysmon", 12): "registry", ("sysmon", 13): "registry", ("sysmon", 14): "registry",
    ("security", 4624): "authentication", ("security", 4625): "authentication", ("security", 4648): "authentication",
    ("powershell", 4104): "script_execution", ("security", 4697): "scheduled_tasks_services",
    ("security", 4698): "scheduled_tasks_services",
}  # fmt: skip
SOURCE_LABEL = {"sysmon": "Sysmon", "security": "Security", "powershell": "PowerShell"}
ATTACK_CHANNELS = {"WinEventLog:Sysmon": "sysmon", "WinEventLog:Security": "security", "WinEventLog:PowerShell": "powershell"}
SPLUNK_SOURCES = {
    "XmlWinEventLog:Microsoft-Windows-Sysmon/Operational": "sysmon", "WinEventLog:Microsoft-Windows-Sysmon/Operational": "sysmon",
    "XmlWinEventLog:Security": "security", "WinEventLog:Security": "security",
    "XmlWinEventLog:Microsoft-Windows-PowerShell/Operational": "powershell",
    "WinEventLog:Microsoft-Windows-PowerShell/Operational": "powershell",
}  # fmt: skip
CARRIES = defaultdict(set)  # source -> categories it can carry
for (source, _), category in EVENTS.items():
    CARRIES[source].add(category)


def fetch(url: str, accept: str = "application/vnd.github+json") -> bytes:
    request = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": "cade-paper-benchmark"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def download(cache: Path, commit: str) -> dict:
    """The attack_techniques tree, the commit date and every dataset YAML, cached under ``cache``."""
    cache.mkdir(parents=True, exist_ok=True)
    meta = cache / "meta.json"
    if not meta.exists():
        api = f"https://api.github.com/repos/{REPOSITORY}"
        listing = json.loads(fetch(f"{api}/contents/datasets?ref={commit}"))
        tree_sha = next(d["sha"] for d in listing if d["name"] == "attack_techniques")
        tree = json.loads(fetch(f"{api}/git/trees/{tree_sha}?recursive=1"))
        if tree.get("truncated"):
            raise SystemExit("the GitHub tree listing was truncated")
        info = json.loads(fetch(f"{api}/commits/{commit}"))
        meta.write_text(json.dumps({"commit": commit, "commit_date": info["commit"]["committer"]["date"],
                                    "retrieved": datetime.now(timezone.utc).date().isoformat(),
                                    "paths": [e["path"] for e in tree["tree"] if e["type"] == "blob"]}), encoding="utf-8")  # fmt: skip
    data = json.loads(meta.read_text(encoding="utf-8"))
    ymls = [p for p in data["paths"] if p.endswith((".yml", ".yaml"))]
    (cache / "yml").mkdir(exist_ok=True)

    def one(path: str) -> tuple[str, bool]:
        target = cache / "yml" / (hashlib.sha1(path.encode()).hexdigest() + ".yml")
        if not target.exists():
            try:
                raw = f"https://raw.githubusercontent.com/{REPOSITORY}/{commit}/datasets/attack_techniques/{path}"
                target.write_bytes(fetch(raw, accept="text/plain"))
            except OSError:
                return path, False
        return path, True

    with futures.ThreadPoolExecutor(8) as pool:
        results = dict(pool.map(one, ymls))
    data["yml"] = {p: str(cache / "yml" / (hashlib.sha1(p.encode()).hexdigest() + ".yml")) for p, ok in results.items() if ok}
    data["yml_failed"] = sorted(p for p, ok in results.items() if not ok)
    return data


def attack_techniques(path: Path) -> tuple[str, dict]:
    """ATT&CK version and, per Windows technique: name, tactics, parent and attributable categories (with events)."""
    bundle = json.loads(path.read_text(encoding="utf-8"))
    objects = bundle["objects"]
    by_id = {o["id"]: o for o in objects}
    version = next((o.get("x_mitre_version") for o in objects if o["type"] == "x-mitre-collection"), "?")
    ext = lambda o: next((r["external_id"] for r in o.get("external_references", []) if r.get("source_name") == "mitre-attack"), None)  # noqa: E731
    techniques = {}
    for o in objects:
        if o["type"] != "attack-pattern" or o.get("revoked") or o.get("x_mitre_deprecated"):
            continue
        if "Windows" not in o.get("x_mitre_platforms", []):
            continue
        phases = {k["phase_name"] for k in o.get("kill_chain_phases", []) if k.get("kill_chain_name") == "mitre-attack"}
        tactics = [t for t, names in TACTICS.items() if phases & set(names)]
        if tactics:
            tid = ext(o)
            techniques[tid] = {"name": o["name"], "tactics": tactics, "parent": tid.split(".")[0] if "." in tid else None,
                               "attack_events": defaultdict(set)}  # fmt: skip
    for r in objects:
        if r["type"] != "relationship" or r["relationship_type"] != "detects":
            continue
        tid = ext(by_id[r["target_ref"]])
        if tid not in techniques:
            continue
        for analytic_ref in by_id[r["source_ref"]].get("x_mitre_analytic_refs", []):
            analytic = by_id.get(analytic_ref)
            if not analytic or "Windows" not in analytic.get("x_mitre_platforms", []):
                continue
            for ref in analytic.get("x_mitre_log_source_references", []):
                source = ATTACK_CHANNELS.get(ref.get("name"))
                match = re.fullmatch(r"\s*EventCode\s*=\s*([\d\s,]+)", ref.get("channel") or "")
                if not source or not match:
                    continue
                for code in (int(c) for c in re.findall(r"\d+", match.group(1))):
                    if (source, code) in EVENTS:
                        techniques[tid]["attack_events"][EVENTS[(source, code)]].add(f"{SOURCE_LABEL[source]} {code}")
    for t in techniques.values():
        t["attack_events"] = {c: sorted(e) for c, e in sorted(t["attack_events"].items())}
    return version, techniques


def build(data: dict, attack_version: str, techniques: dict) -> dict:
    recordings, excluded = [], Counter()
    for path, file in sorted(data["yml"].items()):
        try:
            document = yaml.safe_load(Path(file).read_text(encoding="utf-8"))
        except (yaml.YAMLError, UnicodeDecodeError):
            excluded["unreadable YAML"] += 1
            continue
        if not isinstance(document, dict):
            excluded["unreadable YAML"] += 1
            continue
        ids = [str(t) for t in document.get("mitre_technique") or []]
        sources = sorted({SPLUNK_SOURCES[d["source"]] for d in document.get("datasets") or []
                          if isinstance(d, dict) and d.get("source") in SPLUNK_SOURCES})  # fmt: skip
        if len(ids) != 1:
            excluded["not exactly one technique"] += 1
            continue
        if not sources:
            excluded["no Windows Sysmon/Security/PowerShell log"] += 1
            continue
        technique = techniques.get(ids[0])
        if technique is None:
            excluded["technique not a current Windows technique of the eight tactics"] += 1
            continue
        categories = sorted(c for c in technique["attack_events"] if any(c in CARRIES[s] for s in sources))
        if not categories:
            excluded["no attributable category in the declared sources"] += 1
            continue
        recordings.append({"id": str(document.get("id") or path), "path": path.rsplit("/", 1)[0], "yaml": path,
                           "technique": ids[0], "sources": sources, "categories": categories,
                           "atomic_red_team": "atomic_red_team" in path.split("/"), "date": str(document.get("date") or "")})  # fmt: skip
    used = {r["technique"] for r in recordings}
    return {
        "version": "cade-paper968-catalog-1",
        "note": "Metadata of Splunk Attack Data recordings (no logs) for the paper-matched routing benchmark; built by "
                "scripts/build_paper_catalog.py. 'categories' are a proxy: the ATT&CK v20 Windows detection analytics' "
                "event codes of the technique (mapped with paper 968 Section 6.1) that the recording's declared log "
                "sources can carry.",
        "provenance": {"splunk_attack_data": {"repository": f"https://github.com/{REPOSITORY}", "commit": data["commit"],
                                              "commit_date": data["commit_date"], "retrieved": data["retrieved"],
                                              "yaml_files": len(data["yml"]) + len(data["yml_failed"]),
                                              "yaml_unavailable": data["yml_failed"]},
                       "attack": {"version": attack_version, "source": "app/data/attack/enterprise-attack.json"}},
        "event_categories": {f"{SOURCE_LABEL[s]} {c}": cat for (s, c), cat in sorted(EVENTS.items())},
        "tactics": TACTICS,
        "techniques": {tid: techniques[tid] for tid in sorted(used)},
        "recordings": sorted(recordings, key=lambda r: (r["technique"], r["path"], r["id"])),
        "excluded": dict(sorted(excluded.items())),
    }  # fmt: skip


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--commit", default=COMMIT, help="Splunk attack_data commit (pinned by default)")
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "cade" / "splunk_attack_data")
    parser.add_argument("--attack", type=Path, default=ATTACK, help="ATT&CK Enterprise STIX bundle (v20)")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if not args.attack.exists():
        raise SystemExit(f"ATT&CK bundle not found at {args.attack}; run scripts/fetch_attack_dataset.py first")
    data = download(args.cache_dir / args.commit, args.commit)
    version, techniques = attack_techniques(args.attack)
    catalog = build(data, version, techniques)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(catalog, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    per_tactic = Counter(t for r in catalog["recordings"] for t in catalog["techniques"][r["technique"]]["tactics"])
    print(f"ATT&CK {version}; {len(catalog['recordings'])} recordings of {len(catalog['techniques'])} techniques "
          f"(excluded: {catalog['excluded']}); YAML unavailable: {len(data['yml_failed'])}")  # fmt: skip
    print("recordings listed under each tactic:", dict(per_tactic))
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
