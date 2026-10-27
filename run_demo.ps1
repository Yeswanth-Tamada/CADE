# =====================================================================
#  CADE one-command demo launcher (Windows PowerShell)
# ---------------------------------------------------------------------
#  Sets everything up on a fresh laptop and opens the demo wall in Chrome.
#
#  Default (portable) mode works on ANY laptop with Python installed:
#    * creates a virtual environment and installs the dependencies
#    * fetches the MITRE ATT&CK data if it is missing
#    * uses local Ollama models if Ollama is installed, otherwise a built-in
#      mock model, so the pipeline always runs end to end
#    * runs CADE in SIMULATE mode: the real intent / routing / evolution /
#      fitness pipeline, with tool results clearly labelled SYNTHETIC
#    * starts the server, then opens app/web/static/demo.html in Chrome
#
#  It never edits your .env and never touches your other *.db files:
#  the demo uses its own database, cade_demo.db.
#
#  Usage (just double-click run_demo.bat, or):
#    powershell -ExecutionPolicy Bypass -File run_demo.ps1
#    powershell -ExecutionPolicy Bypass -File run_demo.ps1 -Full   # paper embeddings (large, slow install)
#    powershell -ExecutionPolicy Bypass -File run_demo.ps1 -Live   # use the real tools from YOUR .env
#    powershell -ExecutionPolicy Bypass -File run_demo.ps1 -NoModels  # skip pulling Ollama models
# =====================================================================

param(
    [int]$Port = 8020,
    [switch]$Full,       # install sentence-transformers + faiss (paper embeddings); slow, pulls PyTorch
    [switch]$Live,       # use the real tools configured in your own .env (not portable to other laptops)
    [switch]$NoModels    # do not download Ollama models even if Ollama is installed
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

function Say   ($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Ok    ($m) { Write-Host "    [ok] $m" -ForegroundColor Green }
function Note  ($m) { Write-Host "    $m" -ForegroundColor DarkGray }
function Warn  ($m) { Write-Host "    [!] $m" -ForegroundColor Yellow }
function Die   ($m) { Write-Host "`n[x] $m" -ForegroundColor Red; Read-Host "Press Enter to close"; exit 1 }

Write-Host "===================================================================" -ForegroundColor White
Write-Host "  CADE demo setup  ($(if($Live){'LIVE - your real tools'}elseif($Full){'portable, paper embeddings'}else{'portable, simulate'}))" -ForegroundColor White
Write-Host "===================================================================" -ForegroundColor White

# ---------------------------------------------------------------- Python
Say "Looking for Python 3.10+"
$pyLauncher = $null
foreach ($cand in @(@("py","-3"), @("python"), @("python3"))) {
    $exe = $cand[0]
    if (Get-Command $exe -ErrorAction SilentlyContinue) {
        $verText = & $exe @($cand[1..($cand.Count-1)] + "--version") 2>&1
        if ($verText -match "Python (\d+)\.(\d+)") {
            if ([int]$Matches[1] -gt 3 -or ([int]$Matches[1] -eq 3 -and [int]$Matches[2] -ge 10)) {
                $pyLauncher = $cand; Ok "$verText  ($exe)"; break
            }
        }
    }
}
if (-not $pyLauncher) { Die "Python 3.10 or newer was not found. Install it from https://www.python.org/downloads/ (tick 'Add python.exe to PATH'), then run this again." }

# ---------------------------------------------------------------- venv
$venvPy = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Say "Creating the virtual environment (.venv)"
    & $pyLauncher[0] @($pyLauncher[1..($pyLauncher.Count-1)] + @("-m","venv",(Join-Path $root ".venv")))
    if (-not (Test-Path $venvPy)) { Die "Could not create the virtual environment." }
    Ok "created"
} else { Ok "virtual environment already present" }

# ---------------------------------------------------------------- deps
Say "Installing dependencies (first run can take a few minutes)"
& $venvPy -m pip install --quiet --disable-pip-version-check --upgrade pip
if ($Full -or $Live) {
    & $venvPy -m pip install --disable-pip-version-check -r (Join-Path $root "requirements.txt")
} else {
    & $venvPy -m pip install --disable-pip-version-check `
        "fastapi>=0.115,<1" "uvicorn>=0.30,<1" "pydantic>=2.7,<3" "pydantic-settings>=2.3,<3" `
        "python-dotenv>=1.0,<2" "sqlalchemy>=2.0,<2.1" "httpx>=0.27,<1" "anyio>=4,<5" "numpy>=1.26,<3"
}
if ($LASTEXITCODE -ne 0) { Die "Dependency installation failed. Check your internet connection and try again." }
Ok "dependencies ready"

# ---------------------------------------------------------------- ATT&CK data
$attack = Join-Path $root "app\data\attack\enterprise-attack.json"
if (-not (Test-Path $attack)) {
    Say "Fetching the MITRE ATT&CK dataset (~50 MB, one time)"
    try { & $venvPy (Join-Path $root "scripts\fetch_attack_dataset.py"); Ok "downloaded" }
    catch { Warn "could not download the ATT&CK data; the demo still runs, technique grounding is just weaker." }
} else { Ok "ATT&CK dataset already present" }

# ---------------------------------------------------------------- Ollama (optional)
$provider = "mock"
if (-not $Live) { $env:CADE_EMBEDDER = "hashing"; $env:CADE_VECTOR_INDEX = "numpy"; $env:CADE_EXECUTION_MODE = "simulate" }
else            { $env:CADE_EXECUTION_MODE = "live" }   # -Live forces real tools regardless of what .env says
if ($Full)      { $env:CADE_EMBEDDER = "sentence_transformers"; $env:CADE_VECTOR_INDEX = "faiss" }

Say "Checking for Ollama (real local models)"
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    $installed = ""
    try { $installed = (& ollama list 2>$null | Out-String) } catch {}
    $want = @("llama3.2:3b","qwen2.5:3b")
    $missing = $want | Where-Object { $installed -notmatch [regex]::Escape($_) }
    if ($missing -and -not $NoModels) {
        Warn "pulling models the demo uses: $($missing -join ', ')  (a few GB; one time)"
        foreach ($m in $missing) { & ollama pull $m }
        try { $installed = (& ollama list 2>$null | Out-String) } catch {}
    }
    if (($want | Where-Object { $installed -match [regex]::Escape($_) }).Count -gt 0) {
        $provider = "ollama"
        $env:CADE_OLLAMA_MODEL   = "llama3.2:3b"
        $env:CADE_GENERATOR_MODEL = "llama3.2:3b"
        $env:CADE_JUDGE_MODEL     = "qwen2.5:3b"
        Ok "using real local models through Ollama"
    } else { Warn "Ollama has no usable model; using the built-in mock model instead." }
} else {
    Note "Ollama is not installed - using the built-in mock model (the pipeline still runs end to end)."
    Note "For real local LLM reasoning, install Ollama from https://ollama.com/download and run this again."
}
if (-not $Live) { $env:CADE_LLM_PROVIDER = $provider }

# ---------------------------------------------------------------- isolated database + key
# A dedicated database so your real cade.db / cade_live.db are never touched.
$env:CADE_DATABASE_URL = "sqlite:///./cade_demo.db"
$env:CADE_LLM_REQUEST_TIMEOUT_SECONDS = "300"

Say "Issuing a demo API key"
$roles = if ($Live) { "analyst,live_executor" } else { "analyst,executor" }
$keyOut = & $venvPy -m app.security.keys create --analyst demo --roles $roles --description "run_demo.ps1" 2>&1 | Out-String
$key = ([regex]::Match($keyOut, "cade_[A-Za-z0-9_\-]+")).Value
if (-not $key) { Die "Could not create an API key.`n$keyOut" }
try { Set-Clipboard -Value $key } catch {}
Ok "key created and copied to your clipboard"

# ---------------------------------------------------------------- start server
Say "Starting CADE on http://127.0.0.1:$Port"
$server = Start-Process -FilePath $venvPy `
    -ArgumentList @("-m","uvicorn","app.main:app","--host","127.0.0.1","--port","$Port") `
    -WorkingDirectory $root -PassThru -WindowStyle Minimized
$healthy = $false
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 1
    try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 "http://127.0.0.1:$Port/health" | Out-Null; $healthy = $true; break } catch {}
}
if (-not $healthy) { Die "The server did not come up on port $Port. Another program may be using it; try -Port 8030." }
Ok "server is up (its window is minimized; close it to stop the demo)"

# ---------------------------------------------------------------- open the wall in Chrome
$url = "http://127.0.0.1:$Port/console/demo.html"
Say "Opening the demo wall in Chrome"
$chrome = @(
    "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
    "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
    "$env:LOCALAPPDATA\Google\Chrome\Application\chrome.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($chrome) { Start-Process $chrome $url; Ok "opened in Chrome" }
else { Start-Process $url; Warn "Chrome was not found; opened in your default browser." }

# ---------------------------------------------------------------- what to do next
$incident = @'
Sysmon alert from host win-dc-397. Sysmon EventID: 1 (process created) ParentImage=C:\Windows\System32\cmd.exe Image=C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe CommandLine=powershell.exe -Exec bypass -enc VwByAGkAdABlAC0ASABvAHMAdAAgACIASABlAGwAbABvACAAVwBvAHIAbABkACIA Hashes=MD5=097CE5761C89434367598B34FE32893B,SHA256=BA4038FD20E474C047BE8AAD5BFACDB1BFC1DDBE12F803F473B7918D8D819436. Only this process-creation record is available; script-block logs, network activity and file reputation have not been reviewed.
'@

Write-Host "`n===================================================================" -ForegroundColor White
Write-Host "  READY. In the Chrome page:" -ForegroundColor Green
Write-Host "===================================================================" -ForegroundColor White
Write-Host "  1. Click  Live  (top left)."
Write-Host "  2. Paste the API key (it is already on your clipboard - Ctrl+V) and click Connect."
Write-Host "  3. Click  + New incident , paste the incident text below into 'What was observed', click Run."
Write-Host "  4. Watch understanding -> routing -> synthesis -> execution fill in live."
Write-Host ""
Write-Host "  API key : $key" -ForegroundColor Yellow
Write-Host "  Page    : $url"
Write-Host "  Mode    : $(if($Live){'LIVE (real tools from your .env)'}else{"SIMULATE - tool results are labelled SYNTHETIC; LLM=$provider"})"
Write-Host ""
Write-Host "  --- incident to paste -------------------------------------------" -ForegroundColor DarkGray
Write-Host $incident
Write-Host "  -----------------------------------------------------------------" -ForegroundColor DarkGray
if (-not $Live) {
    Write-Host "  Note: this is the portable demo. Tool steps return SIMULATED data on" -ForegroundColor DarkGray
    Write-Host "  purpose; the wall labels them synthetic. The intent, routing, evolution" -ForegroundColor DarkGray
    Write-Host "  and fitness are all real computation." -ForegroundColor DarkGray
}
Write-Host "`n  To stop: close the minimized 'python ... uvicorn' window (PID $($server.Id))." -ForegroundColor DarkGray
Write-Host ""
Read-Host "Press Enter to close this setup window (the server keeps running)"
