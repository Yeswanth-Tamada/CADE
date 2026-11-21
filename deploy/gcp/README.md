# CADE lab on Google Cloud

The demo testbed (not part of paper 968): two VMs in project `cade-lab-512612`, zone `asia-south1-a`.

```
victim VM (10.160.0.3)                  cade VM (10.160.0.2)                        laptop
auditd -> Filebeat -------------------> Elasticsearch (Docker, :9200)   filebeat-*  (one doc per audit record)
kernel audit -> Auditbeat ------------->                                 auditbeat-* (one event per command)
                                          |  polled every 15 s            ^ Sigma hunts (CADE_SIGMA_FIELD_MAPPING=auditbeat)
                                        cade-forwarder  (scripts/alert_forwarder.py)
                                          |  rule fires -> submit -> analyze -> respond
                                        cade.service    (CADE API, 127.0.0.1:8000) <--- tunnel ---> Ollama :11434
                                                                                                    console :8800
```

- **victim**: auditd watches command execution, `/etc/passwd`, `/etc/shadow`, `/etc/sudoers`, cron and systemd
  units; Filebeat ships the events to Elasticsearch on the cade VM. No CADE code runs there.
  Auditbeat 8.17 ([victim/auditbeat.yml](victim/auditbeat.yml)) listens to the same kernel audit stream in
  multicast mode (auditd keeps the rules) and sends one correlated event per command (executable, full
  command line, arguments, user) to `auditbeat-*`. Sigma's Linux rules (`lab/sigma/rules/linux` on the cade
  VM) are mapped onto these events. Its Elasticsearch API key (`auditbeat-victim`, write to `auditbeat-*`
  only) is in the Auditbeat keystore, not the config file. CADE's read key also covers `auditbeat-*`.
- **VirusTotal**: an online lookup service; CADE on the cade VM calls it with the key in its `.env`.
  Nothing is installed for it.
- **cade**: runs CADE from `~/cade` (a copy of this repo, no git) with its own `.env`, `cade.db` and `.venv`,
  plus the detection forwarder. The forwarder has three rules (SSH brute force, new account, sudoers edit).
  It never approves destructive steps and never sends feedback.
- **LLM**: CADE on the VM has no model of its own. It calls the laptop's Ollama through the tunnel, so
  incidents are only analysed while the laptop is on and `tunnel.ps1` is running.
- **Mode**: `CADE_APP_ENV=production` (no interactive API docs, API keys required) and `CADE_EXECUTION_MODE=live`
  (playbook steps really query Elasticsearch, Sigma and VirusTotal). Destructive actions stay off
  (`CADE_ALLOW_DESTRUCTIVE_LIVE=false`) and always need an approver and a confirmed verification.

## Everyday use

```powershell
gcloud compute instances start cade victim --zone asia-south1-a      # starts billing
powershell -ExecutionPolicy Bypass -File deploy\gcp\tunnel.ps1       # keep this window open
# console: http://127.0.0.1:8800/console/  (API keys: %USERPROFILE%\.cade\cloud_api_keys.txt)
gcloud compute instances stop cade victim --zone asia-south1-a       # when done (disks still bill)
```

The VMs have no reserved address, so their external IPs change on every start; both scripts look the
current IP up with gcloud. SSH is only allowed from the IPs in the `cade-allow-ssh` firewall rule. If your
home IP changes, add the new one there (keep the others).

## Simulating attacks on the victim (Atomic Red Team)

The victim has Red Canary's [Atomic Red Team](https://github.com/redcanaryco/atomic-red-team): small, safe
scripts that each imitate one ATT&CK technique (PowerShell 7, the `invoke-atomicredteam` runner from the
PowerShell Gallery, test definitions in `/opt/AtomicRedTeam/atomics`). There is no real malware in the lab;
detections work on behaviour, so these tests produce realistic traces for Auditbeat, Sigma and CADE.

```bash
# on the victim (ssh yeswanth@<victim IP>)
sudo pwsh -c "Import-Module invoke-atomicredteam; Invoke-AtomicTest T1082 -PathToAtomicsFolder /opt/AtomicRedTeam/atomics -ShowDetailsBrief"
sudo pwsh -c "Import-Module invoke-atomicredteam; Invoke-AtomicTest T1082 -TestNumbers 3 -ShowDetails -PathToAtomicsFolder /opt/AtomicRedTeam/atomics"   # read it first
sudo pwsh -c "Import-Module invoke-atomicredteam; Invoke-AtomicTest T1082 -TestNumbers 3 -PathToAtomicsFolder /opt/AtomicRedTeam/atomics"
sudo pwsh -c "Import-Module invoke-atomicredteam; Invoke-AtomicTest T1082 -TestNumbers 3 -Cleanup -PathToAtomicsFolder /opt/AtomicRedTeam/atomics"
```

Always read a test with `-ShowDetails` before running it: some tests change the system (create users, edit
cron, stop services) and some download tools. Prefer discovery tests for demos and run `-Cleanup` afterwards.
Checked on 2026-11-21: T1082-3 ("List OS Information") was found by CADE's Sigma hunt
(`lnx_auditd_system_info_discovery`).

## Updating the code on the VM

The VM does not update itself. After you change code here:

```powershell
git add ... ; git commit -m "..."                                    # only committed code is deployed
powershell -ExecutionPolicy Bypass -File deploy\gcp\deploy.ps1 -DryRun   # see what would change
powershell -ExecutionPolicy Bypass -File deploy\gcp\deploy.ps1           # deploy + restart
```

`deploy.ps1` sends `git archive HEAD` and, on the VM (`remote_deploy.sh`):

1. backs up `cade.db` to `~/cade-backups/` (last 10 kept);
2. copies the committed files over `~/cade` with rsync. It never touches `.env`, `cade*.db`, `.venv/`,
   `lab/` or the key files, and never deletes anything (remove files that left the repo by hand);
3. reinstalls Python packages if a `requirements*.txt` changed;
4. installs `systemd/*.service` if they differ from the VM's units;
5. restarts `cade` and `cade-forwarder` and prints `/health` (`degraded` = tunnel not open).

The deployed commit is recorded in `~/cade/DEPLOYED_COMMIT`. To roll back, check out the older commit,
deploy it, and if needed restore a database backup.

Not in git (they live only on the VM): `lab/sigma` (a copy of the public SigmaHQ rules), `lab/attack_data`,
the VM's `.env`, and the key files `~/fwd_es_key.txt`, `~/forwarder_key.txt`.
