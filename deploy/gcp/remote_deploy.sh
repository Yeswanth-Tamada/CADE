#!/usr/bin/env bash
# Runs ON the cade VM; started by deploy/gcp/deploy.ps1 (do not run it by hand).
#
#   remote_deploy.sh <archive.tar> <commit> <dry-run: 0|1> <restart: 0|1>
#
# Unpacks the git archive into a staging folder, then copies it over ~/cade with rsync.
# Only files that are in git are written. Everything the VM owns is left alone:
# .env, cade.db, .venv, lab/, the API key files, and any file not in the archive
# (nothing is deleted).
set -euo pipefail

ARCHIVE="$1"; COMMIT="$2"; DRY_RUN="${3:-0}"; RESTART="${4:-1}"
APP="$HOME/cade"
STAGE="$(mktemp -d /tmp/cade-deploy.XXXXXX)"
BACKUPS="$HOME/cade-backups"
trap 'rm -rf "$STAGE" "$ARCHIVE"' EXIT

tar -xf "$ARCHIVE" -C "$STAGE"

RSYNC_OPTS=(-rlt --checksum --itemize-changes
            --exclude=.env --exclude='cade*.db*' --exclude=.venv/ --exclude=lab/
            --exclude=__pycache__/ --exclude='*.pyc')

# rsync's itemized output -> "new  path" / "changed  path" (timestamp-only updates are not listed)
summarize() { awk '/^>f\+\+\+/ {print "  new      " $2; next} /^>f/ {print "  changed  " $2}'; }

if [ "$DRY_RUN" = "1" ]; then
    echo "== DRY RUN: files that would change in $APP (commit $COMMIT) =="
    changes="$(rsync -n "${RSYNC_OPTS[@]}" "$STAGE"/ "$APP"/ | summarize)"
    if [ -z "$changes" ]; then echo "(none: the VM already matches this commit)"; else echo "$changes"; echo "  ($(echo "$changes" | wc -l) files)"; fi
    echo "Currently deployed: $(cat "$APP/DEPLOYED_COMMIT" 2>/dev/null || echo 'unknown (deployed by hand)')"
    exit 0
fi

# 1. Back up the database (SQLite online backup: safe while CADE is running). Keep the last 10.
mkdir -p "$BACKUPS"
if [ -f "$APP/cade.db" ]; then
    backup="$BACKUPS/cade-$(date -u +%Y%m%d-%H%M%S).db"
    "$APP/.venv/bin/python" - "$APP/cade.db" "$backup" <<'PY'
import sqlite3, sys
src, dst = sqlite3.connect(sys.argv[1]), sqlite3.connect(sys.argv[2])
with dst:
    src.backup(dst)
PY
    echo "Database backed up to $backup"
    ls -1t "$BACKUPS"/cade-*.db | tail -n +11 | xargs -r rm -f
fi

# 2. Copy the code.
req_before="$(cat "$APP"/requirements*.txt 2>/dev/null | sha256sum)"
echo "== Changed files =="
changes="$(rsync "${RSYNC_OPTS[@]}" "$STAGE"/ "$APP"/ | summarize)"
if [ -z "$changes" ]; then echo "(none)"; else echo "$changes"; fi
req_after="$(cat "$APP"/requirements*.txt 2>/dev/null | sha256sum)"
printf '%s deployed %s\n' "$COMMIT" "$(date -u +%FT%TZ)" > "$APP/DEPLOYED_COMMIT"

# 3. Python packages, only when a requirements file changed.
if [ "$req_before" != "$req_after" ]; then
    echo "== requirements changed: installing =="
    "$APP/.venv/bin/pip" install --disable-pip-version-check -q -r "$APP/requirements.txt"
    [ -f "$APP/requirements-tools.txt" ] && "$APP/.venv/bin/pip" install --disable-pip-version-check -q -r "$APP/requirements-tools.txt"
fi

# 4. systemd units, only when the copies in the repo differ from the installed ones.
for unit in cade.service cade-forwarder.service; do
    src="$APP/deploy/gcp/systemd/$unit"
    if [ -f "$src" ] && ! cmp -s "$src" "/etc/systemd/system/$unit"; then
        echo "== updating /etc/systemd/system/$unit =="
        sudo install -m 0644 "$src" "/etc/systemd/system/$unit"
        sudo systemctl daemon-reload
    fi
done

# 5. Restart and wait for the API.
if [ "$RESTART" = "1" ]; then
    echo "== restarting cade and cade-forwarder =="
    sudo systemctl restart cade
    status=""
    for _ in $(seq 1 60); do
        status="$(curl -s -m 5 http://127.0.0.1:8000/health || true)"
        [ -n "$status" ] && break
        sleep 3
    done
    sudo systemctl restart cade-forwarder
    if [ -z "$status" ]; then
        echo "CADE did not answer within 3 minutes. Recent log:"
        sudo journalctl -u cade --no-pager -n 30
        exit 1
    fi
    echo "Health: $status"
    case "$status" in
        *'"degraded"'*) echo "(degraded usually just means the tunnel to your laptop's Ollama is not open)";;
    esac
fi
echo "Deployed $COMMIT"
