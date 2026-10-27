"""Index Splunk Attack Data recordings into Elasticsearch for live tool checks (paper 968, Section 6).

Paper 968 builds its scenarios from Splunk Attack Data (Sysmon, Windows Security and PowerShell
logs recorded while ATT&CK techniques ran) and executes playbook query nodes against an Elastic
index of those logs. This script does the indexing step for local testing:

* each ``<Event>`` XML record becomes one document with the ECS fields CADE's Elastic adapter
  queries (``host.name``, ``event.code``, ``event.category``, ``winlog.channel``, ``process.*``,
  ``powershell.file.script_block_text`` ...) and, at the top level, the raw Windows field names
  (``Image``, ``CommandLine``, ``ScriptBlockText``, ``EventID`` ...) that Sigma rules use;
* timestamps are **shifted** so each file's newest event lands one hour ago, keeping the spacing
  between events. CADE's look-back windows are at most 168 hours and the recordings are years old.
  The original time is kept in ``cade.original_timestamp`` and ``cade.timestamp_shifted`` is true.

    python scripts/load_attack_data.py lab/attack_data                 # every *.log below the folder
    python scripts/load_attack_data.py lab/attack_data --recreate      # drop and rebuild the index

Credentials: the ``elastic`` superuser password from ``LAB_ELASTIC_PASSWORD`` (environment or .env).
The recordings are downloaded data and are parsed as untrusted input (no DTDs or entities).
"""

from __future__ import annotations

import argparse
import json
import ntpath
import os
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

NS = "{http://schemas.microsoft.com/win/2004/08/events/event}"
EVENT_RE = re.compile(r"<Event[ >].*?</Event>", re.S)
INDEX = "cade-attack-data"
# Strings are lowercase-normalized keywords: CADE's term filters and Sigma's wildcard queries (whose
# values the Sigma adapter lowercases) then match case-insensitively, as Sigma intends. Lucene indexes
# at most MAX_TERM_BYTES per value, so searchable copies are cut at that many UTF-8 bytes (the loader
# reports how many were); the full values stay in winlog.event_data, which is stored but not indexed.
MAX_TERM_BYTES = 32766
MAPPING = {
    "settings": {"analysis": {"normalizer": {"lowercase": {"type": "custom", "filter": ["lowercase"]}}}},
    "mappings": {
        "dynamic_templates": [{"strings": {"match_mapping_type": "string",
                                           "mapping": {"type": "keyword", "normalizer": "lowercase",
                                                       "ignore_above": MAX_TERM_BYTES}}}],
        "properties": {"@timestamp": {"type": "date"},
                       "cade": {"properties": {"original_timestamp": {"type": "date"}}},
                       "winlog": {"properties": {"event_data": {"type": "object", "enabled": False}}}},
    },
}  # fmt: skip
# (channel, event code) -> ECS event.category, after app/data/log_source_categories.json
CATEGORIES = {
    ("Microsoft-Windows-Sysmon/Operational", "1"): "process",
    ("Microsoft-Windows-Sysmon/Operational", "3"): "network",
    ("Microsoft-Windows-Sysmon/Operational", "11"): "file",
    ("Microsoft-Windows-Sysmon/Operational", "12"): "registry",
    ("Microsoft-Windows-Sysmon/Operational", "13"): "registry",
    ("Microsoft-Windows-Sysmon/Operational", "14"): "registry",
    ("Microsoft-Windows-Sysmon/Operational", "22"): "network",
    ("Security", "4688"): "process",
    ("Security", "4624"): "authentication",
    ("Security", "4625"): "authentication",
    ("Security", "4648"): "authentication",
}


def parse_event(xml_text: str) -> dict | None:
    if "<!DOCTYPE" in xml_text or "<!ENTITY" in xml_text:
        return None
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None
    system = root.find(f"{NS}System")
    if system is None:
        return None
    created = system.find(f"{NS}TimeCreated")
    code = (system.findtext(f"{NS}EventID") or "").strip()
    if created is None or not code:
        return None
    data = {}
    for item in root.iter(f"{NS}Data"):
        name = item.get("Name")
        if name and item.text is not None:
            data[name] = item.text
    provider = system.find(f"{NS}Provider")
    return {
        "time": datetime.fromisoformat(created.get("SystemTime", "").replace("Z", "+00:00")[:26] + "+00:00"),
        "code": code,
        "channel": (system.findtext(f"{NS}Channel") or "").strip(),
        "computer": (system.findtext(f"{NS}Computer") or "").strip(),
        "record_id": (system.findtext(f"{NS}EventRecordID") or "").strip(),
        "provider": provider.get("Name", "") if provider is not None else "",
        "data": data,
    }


def _basename(path: str | None) -> str | None:
    return ntpath.basename(path) if path else None


def _account(value: str | None) -> str | None:
    return value.split("\\")[-1] if value and value not in ("-", "") else None


def _searchable(value: str) -> str:
    encoded = value.encode("utf-8")
    return value if len(encoded) <= MAX_TERM_BYTES else encoded[:MAX_TERM_BYTES].decode("utf-8", "ignore")


def to_document(event: dict, timestamp: datetime, dataset: str) -> dict:
    channel, code = event["channel"], event["code"]
    data = {name: _searchable(value) for name, value in event["data"].items()}
    computer = event["computer"].lower()
    doc: dict = {
        **data,  # raw Windows field names, for Sigma
        "EventID": code,
        "Channel": channel,
        "Computer": event["computer"],
        "@timestamp": timestamp.isoformat(),
        "event": {"code": code, "provider": event["provider"]},
        "host": {"name": computer, "hostname": computer.split(".")[0]},
        "winlog": {"channel": channel, "computer_name": event["computer"], "record_id": event["record_id"],
                   "event_data": event["data"]},
        "cade": {"dataset": dataset, "original_timestamp": event["time"].isoformat(), "timestamp_shifted": True},
    }  # fmt: skip
    if (channel, code) in CATEGORIES:
        doc["event"]["category"] = CATEGORIES[(channel, code)]
    image = data.get("Image") or data.get("NewProcessName")
    parent = data.get("ParentImage") or data.get("ParentProcessName")
    process = {k: v for k, v in {"executable": image, "name": _basename(image), "command_line": data.get("CommandLine"),
                                 "parent": {"name": _basename(parent)} if parent else None}.items() if v}  # fmt: skip
    if process:
        doc["process"] = process
    user = _account(data.get("User")) or _account(data.get("TargetUserName")) or _account(data.get("SubjectUserName"))
    if user:
        doc["user"] = {"name": user}
    if data.get("SourceIp"):
        doc["source"] = {"ip": data["SourceIp"]}
    if data.get("DestinationIp"):
        doc["destination"] = {"ip": data["DestinationIp"]}
        if (data.get("DestinationPort") or "").isdigit():
            doc["destination"]["port"] = int(data["DestinationPort"])
    if data.get("QueryName"):
        doc["dns"] = {"question": {"name": data["QueryName"]}}
    if data.get("TargetObject"):
        doc["registry"] = {"path": data["TargetObject"], "data": {"strings": [data.get("Details", "")]}}
    if data.get("ScriptBlockText"):
        doc["powershell"] = {"file": {"script_block_text": data["ScriptBlockText"]}}
    return doc


def load_file(path: Path, root: Path, now: datetime) -> list[dict]:
    events = [e for e in (parse_event(m.group(0)) for m in EVENT_RE.finditer(path.read_text(encoding="utf-8",
                                                                                          errors="replace"))) if e]  # fmt: skip
    if not events:
        return []
    shift = (now - timedelta(hours=1)) - max(e["time"] for e in events)
    dataset = path.relative_to(root).as_posix()
    return [to_document(e, e["time"] + shift, dataset) for e in events]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path, help="folder holding downloaded Splunk Attack Data *.log files")
    parser.add_argument("--url", default="http://127.0.0.1:9200")
    parser.add_argument("--user", default="elastic")
    parser.add_argument("--recreate", action="store_true", help=f"delete and recreate the '{INDEX}' index first")
    args = parser.parse_args()

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    password = os.environ.get("LAB_ELASTIC_PASSWORD")
    if not password:
        print("LAB_ELASTIC_PASSWORD is not set (environment or .env).")
        return 2
    files = sorted(args.folder.rglob("*.log"))
    if not files:
        print(f"no *.log files under {args.folder}")
        return 2

    with httpx.Client(base_url=args.url, auth=(args.user, password), timeout=60) as client:
        if args.recreate:
            client.delete(f"/{INDEX}")
        if client.head(f"/{INDEX}").status_code == 404:
            client.put(f"/{INDEX}", json=MAPPING).raise_for_status()
        now = datetime.now(timezone.utc)
        hosts: dict[str, int] = {}
        for path in files:
            docs = load_file(path, args.folder, now)
            for start in range(0, len(docs), 500):
                body = "".join(f'{{"index":{{}}}}\n{json.dumps(d)}\n' for d in docs[start:start + 500])
                reply = client.post(f"/{INDEX}/_bulk", content=body,
                                    headers={"Content-Type": "application/x-ndjson"})  # fmt: skip
                reply.raise_for_status()
                if reply.json().get("errors"):
                    first = next(i for i in reply.json()["items"] if i["index"].get("error"))
                    print(f"{path.name}: bulk errors, first: {first['index']['error'].get('reason', '')[:200]}")
                    return 1
            codes = sorted({(d["winlog"]["channel"], d["event"]["code"]) for d in docs})
            for d in docs:
                hosts[d["host"]["name"]] = hosts.get(d["host"]["name"], 0) + 1
            cut = sum(any(len(v.encode("utf-8")) > MAX_TERM_BYTES for v in d["winlog"]["event_data"].values())
                      for d in docs)  # fmt: skip
            print(f"{path.relative_to(args.folder).as_posix()}: {len(docs)} events; "
                  + ", ".join(f"{ch.split('/')[0].split('-')[-1]} {c}" for ch, c in codes)
                  + (f" ({cut} with a field cut to {MAX_TERM_BYTES} bytes for search)" if cut else ""))  # fmt: skip
        client.post(f"/{INDEX}/_refresh").raise_for_status()
        total = client.get(f"/{INDEX}/_count").json().get("count")
    print(f"\nindex '{INDEX}' now holds {total} events. Hosts (use with --host):")
    for host, count in sorted(hosts.items(), key=lambda kv: -kv[1]):
        print(f"  {host}  ({count} events)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
