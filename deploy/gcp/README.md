# CADE lab on Google Cloud

The demo testbed (not part of paper 968): two VMs in project `cade-lab-512612`, zone `asia-south1-a`.

```
victim VM (10.160.0.3)                  cade VM (10.160.0.2)                        laptop
auditd -> Filebeat -------------------> Elasticsearch (Docker, :9200)
                                          |  polled every 15 s
                                        cade-forwarder  (scripts/alert_forwarder.py)
                                          |  rule fires -> submit -> analyze -> respond
                                        cade.service    (CADE API, 127.0.0.1:8000) <--- tunnel ---> Ollama :11434
                                                                                                    console :8800
```

- **victim**: auditd watches command execution, `/etc/passwd`, `/etc/shadow`, `/etc/sudoers`, cron and systemd
  units; Filebeat ships the events to Elasticsearch on the cade VM. No CADE code runs there.
- **cade**: runs CADE from `~/cade` (a copy of this repo, no git) with its own `.env`, `cade.db` and `.venv`,
  plus the detection forwarder. The forwarder has three rules (SSH brute force, new account, sudoers edit).
  It never approves destructive steps and never sends feedback.
- **LLM**: CADE on the VM has no model of its own. It calls the laptop's Ollama through the tunnel, so
  incidents are only analysed while the laptop is on and `tunnel.ps1` is running.

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
