#!/usr/bin/env bash
# =====================================================================
#  CADE one-command demo launcher (macOS / Linux)
#  Same behaviour as run_demo.ps1. See DEMO.md.
#    ./run_demo.sh            portable, simulate mode (works on any laptop)
#    ./run_demo.sh --full     paper embeddings (large install)
#    ./run_demo.sh --live     use the real tools from your own .env
#    ./run_demo.sh --no-models do not pull Ollama models
#    ./run_demo.sh --port 8030
# =====================================================================
set -euo pipefail

PORT=8020; FULL=0; LIVE=0; NOMODELS=0
while [ $# -gt 0 ]; do
  case "$1" in
    --full) FULL=1;; --live) LIVE=1;; --no-models) NOMODELS=1;;
    --port) PORT="$2"; shift;; *) echo "unknown option: $1"; exit 1;;
  esac; shift
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
say()  { printf "\n==> %s\n" "$1"; }
ok()   { printf "    [ok] %s\n" "$1"; }
note() { printf "    %s\n" "$1"; }
warn() { printf "    [!] %s\n" "$1"; }
die()  { printf "\n[x] %s\n" "$1"; exit 1; }

echo "==================================================================="
echo "  CADE demo setup"
echo "==================================================================="

# -------------------------------------------------------------- Python
say "Looking for Python 3.10+"
PY=""
for cand in python3 python; do
  if command -v "$cand" >/dev/null 2>&1; then
    v="$($cand -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo 0.0)"
    maj="${v%%.*}"; min="${v##*.}"
    if [ "$maj" -gt 3 ] 2>/dev/null || { [ "$maj" -eq 3 ] && [ "$min" -ge 10 ]; } 2>/dev/null; then
      PY="$cand"; ok "Python $v ($cand)"; break
    fi
  fi
done
[ -n "$PY" ] || die "Python 3.10+ not found. Install it from https://www.python.org/downloads/ and try again."

# -------------------------------------------------------------- venv
VENVPY="$ROOT/.venv/bin/python"
if [ ! -x "$VENVPY" ]; then
  say "Creating the virtual environment (.venv)"
  "$PY" -m venv "$ROOT/.venv"
  [ -x "$VENVPY" ] || die "Could not create the virtual environment."
  ok "created"
else ok "virtual environment already present"; fi

# -------------------------------------------------------------- deps
say "Installing dependencies (first run can take a few minutes)"
"$VENVPY" -m pip install --quiet --disable-pip-version-check --upgrade pip
if [ "$FULL" = 1 ] || [ "$LIVE" = 1 ]; then
  "$VENVPY" -m pip install --disable-pip-version-check -r "$ROOT/requirements.txt"
else
  "$VENVPY" -m pip install --disable-pip-version-check \
    "fastapi>=0.115,<1" "uvicorn>=0.30,<1" "pydantic>=2.7,<3" "pydantic-settings>=2.3,<3" \
    "python-dotenv>=1.0,<2" "sqlalchemy>=2.0,<2.1" "httpx>=0.27,<1" "anyio>=4,<5" "numpy>=1.26,<3"
fi
ok "dependencies ready"

# -------------------------------------------------------------- ATT&CK data
if [ ! -f "$ROOT/app/data/attack/enterprise-attack.json" ]; then
  say "Fetching the MITRE ATT&CK dataset (~50 MB, one time)"
  "$VENVPY" "$ROOT/scripts/fetch_attack_dataset.py" && ok "downloaded" \
    || warn "could not download the ATT&CK data; the demo still runs, grounding is just weaker."
else ok "ATT&CK dataset already present"; fi

# -------------------------------------------------------------- Ollama (optional)
PROVIDER="mock"
if [ "$LIVE" != 1 ]; then export CADE_EMBEDDER=hashing CADE_VECTOR_INDEX=numpy CADE_EXECUTION_MODE=simulate
else export CADE_EXECUTION_MODE=live; fi   # --live forces real tools regardless of what .env says
if [ "$FULL" = 1 ]; then export CADE_EMBEDDER=sentence_transformers CADE_VECTOR_INDEX=faiss; fi

say "Checking for Ollama (real local models)"
if command -v ollama >/dev/null 2>&1; then
  installed="$(ollama list 2>/dev/null || true)"
  for m in llama3.2:3b qwen2.5:3b; do
    if ! printf '%s' "$installed" | grep -q "$m"; then
      if [ "$NOMODELS" != 1 ]; then warn "pulling $m (one time, a few GB)"; ollama pull "$m" || true; fi
    fi
  done
  installed="$(ollama list 2>/dev/null || true)"
  if printf '%s' "$installed" | grep -qE 'llama3.2:3b|qwen2.5:3b'; then
    PROVIDER="ollama"
    export CADE_OLLAMA_MODEL=llama3.2:3b CADE_GENERATOR_MODEL=llama3.2:3b CADE_JUDGE_MODEL=qwen2.5:3b
    ok "using real local models through Ollama"
  else warn "Ollama has no usable model; using the built-in mock model."; fi
else
  note "Ollama not installed - using the built-in mock model (pipeline still runs end to end)."
  note "For real local LLM reasoning, install Ollama from https://ollama.com/download and re-run."
fi
[ "$LIVE" = 1 ] || export CADE_LLM_PROVIDER="$PROVIDER"

# -------------------------------------------------------------- isolated DB + key
export CADE_DATABASE_URL="sqlite:///./cade_demo.db"
export CADE_LLM_REQUEST_TIMEOUT_SECONDS=300

say "Issuing a demo API key"
ROLES="analyst,executor"; [ "$LIVE" = 1 ] && ROLES="analyst,live_executor"
KEYOUT="$("$VENVPY" -m app.security.keys create --analyst demo --roles "$ROLES" --description run_demo.sh 2>&1 || true)"
KEY="$(printf '%s' "$KEYOUT" | grep -oE 'cade_[A-Za-z0-9_-]+' | head -1)"
[ -n "$KEY" ] || die "Could not create an API key.
$KEYOUT"
( command -v pbcopy >/dev/null 2>&1 && printf '%s' "$KEY" | pbcopy ) 2>/dev/null || \
( command -v xclip  >/dev/null 2>&1 && printf '%s' "$KEY" | xclip -selection clipboard ) 2>/dev/null || true
ok "key created"

# -------------------------------------------------------------- start server
say "Starting CADE on http://127.0.0.1:$PORT"
"$VENVPY" -m uvicorn app.main:app --host 127.0.0.1 --port "$PORT" >"$ROOT/cade_demo_server.log" 2>&1 &
SRV=$!
healthy=0
for _ in $(seq 1 60); do
  sleep 1
  if curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then healthy=1; break; fi
done
[ "$healthy" = 1 ] || { kill "$SRV" 2>/dev/null || true; die "Server did not come up on port $PORT (try --port 8030). See cade_demo_server.log"; }
ok "server is up (PID $SRV; log: cade_demo_server.log)"

# -------------------------------------------------------------- open the wall
URL="http://127.0.0.1:$PORT/console/demo.html"
say "Opening the demo wall"
if command -v google-chrome >/dev/null 2>&1;         then google-chrome "$URL" >/dev/null 2>&1 &
elif command -v google-chrome-stable >/dev/null 2>&1; then google-chrome-stable "$URL" >/dev/null 2>&1 &
elif [ -d "/Applications/Google Chrome.app" ];        then open -a "Google Chrome" "$URL"
elif command -v open >/dev/null 2>&1;                 then open "$URL"
elif command -v xdg-open >/dev/null 2>&1;             then xdg-open "$URL" >/dev/null 2>&1 &
else warn "open this in your browser: $URL"; fi

INCIDENT='Sysmon alert from host win-dc-397. Sysmon EventID: 1 (process created) ParentImage=C:\Windows\System32\cmd.exe Image=C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe CommandLine=powershell.exe -Exec bypass -enc VwByAGkAdABlAC0ASABvAHMAdAAgACIASABlAGwAbABvACAAVwBvAHIAbABkACIA Hashes=MD5=097CE5761C89434367598B34FE32893B,SHA256=BA4038FD20E474C047BE8AAD5BFACDB1BFC1DDBE12F803F473B7918D8D819436. Only this process-creation record is available; script-block logs, network activity and file reputation have not been reviewed.'

cat <<EOF

===================================================================
  READY. In the browser page:
===================================================================
  1. Click  Live  (top left).
  2. Paste the API key and click Connect.
  3. Click  + New incident , paste the incident text below, click Run.
  4. Watch understanding -> routing -> synthesis -> execution.

  API key : $KEY
  Page    : $URL
  Mode    : $([ "$LIVE" = 1 ] && echo "LIVE (real tools from your .env)" || echo "SIMULATE - tool results labelled SYNTHETIC; LLM=$PROVIDER")

  --- incident to paste -------------------------------------------
$INCIDENT
  -----------------------------------------------------------------

  To stop the server:  kill $SRV
EOF
