"""CADE detection-integration bridge (NOT part of paper 968's method).

Polls the Elasticsearch index that the victim ships to, applies a few simple
detection rules over its telemetry, and when one fires it builds a *grounded*
incident from the matched events and drives CADE: submit -> analyze -> respond
-> (optionally) execute. This is the missing "detection" layer the paper assumes
a SIEM provides; CADE itself remains the response engine.

Design notes:
- stdlib only (urllib); runs under the CADE venv but needs no extra packages.
- ES is read with CADE's read-only API key; CADE is driven with a dedicated,
  limited key (analyst[,live_executor]).
- Per-rule watermarks stop it re-firing on the same events: a rule only looks at
  events newer than the moment it last fired (or, for a rule that has never
  fired, than the moment the forwarder first knew about it), so historical
  data is ignored. A per-rule cooldown additionally rate-limits new bursts.
- It NEVER submits feedback (that trains the library/router -- left to a human).
- Destructive plan steps stay gated by CADE (approval + verification); AUTO_EXECUTE
  only ever runs read-only queries and raises manual tasks.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone

# ---- config (env-overridable) -------------------------------------------------
ES_URL = os.environ.get("FWD_ES_URL", "http://127.0.0.1:9200")
ES_INDEX = os.environ.get("FWD_ES_INDEX", "filebeat-*")
CADE_URL = os.environ.get("FWD_CADE_URL", "http://127.0.0.1:8000")
ES_KEY_FILE = os.environ.get("FWD_ES_KEY_FILE", "/root/cade_es_apikey.txt")
CADE_KEY_FILE = os.environ.get("FWD_CADE_KEY_FILE", "/home/yeswanth/forwarder_key.txt")
STATE_FILE = os.environ.get("FWD_STATE_FILE", "/home/yeswanth/forwarder_state.json")
POLL_SECONDS = int(os.environ.get("FWD_POLL_SECONDS", "15"))
COOLDOWN_SECONDS = int(os.environ.get("FWD_COOLDOWN_SECONDS", "600"))
AUTO_EXECUTE = os.environ.get("FWD_AUTO_EXECUTE", "true").lower() == "true"
LOGIN_FAIL_THRESHOLD = int(os.environ.get("FWD_LOGIN_FAIL_THRESHOLD", "5"))

ES_KEY = open(ES_KEY_FILE).read().strip() if os.path.exists(ES_KEY_FILE) else ""
CADE_KEY = open(CADE_KEY_FILE).read().strip() if os.path.exists(CADE_KEY_FILE) else ""


def _req(url: str, *, method: str = "GET", headers: dict | None = None, body: dict | None = None, timeout: int = 30):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        raw = e.read()
        return e.code, (json.loads(raw) if raw else None)


def es_search(query: dict):
    url = f"{ES_URL}/{ES_INDEX}/_search"
    h = {"Authorization": f"ApiKey {ES_KEY}", "Content-Type": "application/json"}
    code, body = _req(url, method="POST", headers=h, body=query)
    if code != 200:
        raise RuntimeError(f"ES {code}: {str(body)[:200]}")
    return body


def cade(path: str, *, method: str = "GET", body: dict | None = None):
    url = f"{CADE_URL}{path}"
    h = {"Authorization": f"Bearer {CADE_KEY}", "Content-Type": "application/json"}
    # analyze is synchronous and LLM-backed (~60-90s over the tunnel); execute runs the plan.
    return _req(url, method=method, headers=h, body=body, timeout=240)


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            return json.load(open(STATE_FILE))
        except Exception:
            pass
    return {"started_at": _now_iso(), "last_fired": {}, "watermarks": {}}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_watermarks(state: dict) -> None:
    """Give every rule a watermark; a rule without one starts at now, so it ignores historical data.

    State files written before per-rule watermarks existed only had ``started_at`` (which never
    moved), so their rules re-fired on the same old events after every cooldown; they start at now too.
    """
    marks = state.setdefault("watermarks", {})
    state.setdefault("last_fired", {})
    for rule in RULES:
        marks.setdefault(rule.__name__, _now_iso())


def save_state(state: dict):
    tmp = STATE_FILE + ".tmp"
    json.dump(state, open(tmp, "w"))
    os.replace(tmp, STATE_FILE)


def log(msg: str):
    print(f"{datetime.now(timezone.utc).isoformat()} {msg}", flush=True)


# ---- detection rules ----------------------------------------------------------
# Each returns (incident_dict | None). Only events newer than `since` are considered.

def rule_ssh_bruteforce(since: str):
    q = {
        "size": 0,
        "query": {"bool": {"must": [
            {"range": {"@timestamp": {"gte": since}}},
            {"term": {"event.dataset": "system.auth"}},
            {"term": {"event.outcome": "failure"}},
        ]}},
        "aggs": {"by_ip": {"terms": {"field": "source.ip", "size": 20},
                           "aggs": {"users": {"terms": {"field": "user.name", "size": 20}},
                                    "host": {"terms": {"field": "host.name", "size": 1}}}}},
    }
    for b in es_search(q)["aggregations"]["by_ip"]["buckets"]:
        if b["doc_count"] >= LOGIN_FAIL_THRESHOLD:
            ip = b["key"]
            users = [u["key"] for u in b["users"]["buckets"]]
            host = (b["host"]["buckets"] or [{"key": "unknown"}])[0]["key"]
            desc = (f"{b['doc_count']} failed SSH authentication attempts were observed on host {host} "
                    f"from source IP {ip} within a short window, targeting the accounts "
                    f"{', '.join(users)}. The pattern indicates SSH credential-access brute forcing "
                    f"against a Linux host.")
            return {"rule": "ssh_bruteforce", "incident_id": f"AUTO-SSH-{int(time.time())}",
                    "source": "siem_alert", "title": f"SSH brute force on {host} from {ip}",
                    "description": desc, "severity": "high",
                    "attributes": {"host": host, "source_ip": ip, "platform": "linux"}}
    return None


def rule_new_account(since: str):
    q = {
        "size": 10, "sort": [{"@timestamp": "desc"}],
        "query": {"bool": {"must": [
            {"range": {"@timestamp": {"gte": since}}},
            {"term": {"event.dataset": "auditd.log"}},
            {"term": {"event.action": "added-user-account"}},
        ]}},
        "_source": ["@timestamp", "host.name", "user.name", "message"],
    }
    hits = es_search(q)["hits"]["hits"]
    if hits:
        host = (hits[0]["_source"].get("host") or {}).get("name", "unknown")
        desc = (f"A new local user account was created on host {host} (auditd recorded an "
                f"added-user-account event). Unexpected account creation on a Linux host indicates "
                f"an attacker establishing persistence.")
        return {"rule": "new_account", "incident_id": f"AUTO-ACCT-{int(time.time())}",
                "source": "siem_alert", "title": f"New local account created on {host}",
                "description": desc, "severity": "high",
                "attributes": {"host": host, "platform": "linux"}}
    return None


def rule_sudoers_edit(since: str):
    q = {
        "size": 10, "sort": [{"@timestamp": "desc"}],
        "query": {"bool": {"must": [
            {"range": {"@timestamp": {"gte": since}}},
            {"term": {"event.dataset": "auditd.log"}},
            {"match_phrase": {"message": "/etc/sudoers"}},
        ]}},
        "_source": ["@timestamp", "host.name", "message"],
    }
    hits = es_search(q)["hits"]["hits"]
    if hits:
        host = (hits[0]["_source"].get("host") or {}).get("name", "unknown")
        desc = (f"A modification to /etc/sudoers was observed on host {host} (auditd privilege-change "
                f"event). Editing the sudoers policy on a Linux host indicates an attempt at privilege "
                f"escalation by granting elevated rights.")
        return {"rule": "sudoers_edit", "incident_id": f"AUTO-SUDO-{int(time.time())}",
                "source": "siem_alert", "title": f"sudoers policy modified on {host}",
                "description": desc, "severity": "high",
                "attributes": {"host": host, "platform": "linux"}}
    return None


RULES = [rule_ssh_bruteforce, rule_new_account, rule_sudoers_edit]


# ---- CADE orchestration -------------------------------------------------------

def drive_cade(inc: dict):
    code, body = cade("/incidents", method="POST", body={k: inc[k] for k in
                       ("incident_id", "source", "title", "description", "severity", "attributes")})
    if code not in (200, 201):
        log(f"  submit failed {code}: {str(body)[:160]}"); return
    iid = body["incident_id"]
    log(f"  submitted {iid}; analyzing...")
    code, body = cade(f"/incidents/{iid}/analyze", method="POST")
    if code != 200:
        log(f"  analyze failed {code}: {str(body)[:160]}"); return
    intent = ((body.get("result") or {}).get("intent") or {}).get("label")
    log(f"  analyzed: intent={intent}; responding (synthesis ~10 min)...")
    code, body = cade(f"/incidents/{iid}/respond", method="POST")
    if code not in (200, 202):
        log(f"  respond failed {code}: {str(body)[:160]}"); return
    job = body.get("job_id")
    log(f"  respond job {job} running; the demo wall will show it. Execute={AUTO_EXECUTE}")
    if not (AUTO_EXECUTE and job):
        return
    # poll the job, then execute the plan live
    for _ in range(160):
        time.sleep(6)
        c, j = cade(f"/jobs/{job}")
        if c == 200 and j.get("status") not in ("running", "pending"):
            break
    plan = None
    c, resp = cade(f"/incidents/{iid}/response")
    if c == 200 and (resp.get("plans") or []):
        plan = resp["plans"][-1].get("plan_id")
    if not plan:
        log("  no plan to execute"); return
    c, r = cade(f"/response-plans/{plan}/execute", method="POST")
    log(f"  executed {plan}: {c} status={ (r or {}).get('status') }")


def main():
    if not ES_KEY or not CADE_KEY:
        raise SystemExit("missing ES or CADE key file")
    state = load_state()
    init_watermarks(state)
    log(f"forwarder started; watermarks {state['watermarks']}; auto_execute={AUTO_EXECUTE}")
    save_state(state)
    while True:
        try:
            now = time.time()
            for rule in RULES:
                name = rule.__name__
                last = state["last_fired"].get(name, 0)
                if now - last < COOLDOWN_SECONDS:
                    continue
                checked_at = _now_iso()  # taken before the query, so no event can fall between two windows
                try:
                    inc = rule(state["watermarks"][name])
                except Exception as e:
                    log(f"rule {name} error: {e}"); continue
                if inc:
                    log(f"DETECTED [{inc['rule']}] -> {inc['title']}")
                    state["last_fired"][name] = now
                    state["watermarks"][name] = checked_at  # these events are handled; never match them again
                    save_state(state)
                    drive_cade(inc)
        except Exception as e:
            log(f"loop error: {e}")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
