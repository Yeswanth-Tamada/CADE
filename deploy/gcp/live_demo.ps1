# One-click setup for the live conference demo (run through run_live_demo.bat).
#
#   start (default): Ollama on this laptop -> start both VMs -> check SSH access -> copy the live screen to the VM
#                    -> open the tunnel in its own window -> wait for CADE -> check the API key -> open the live screen
#                    -> wait until you press Enter after the demo -> close the tunnel and stop both VMs
#   stop:            close the tunnel and stop both VMs (stops compute billing; disks still bill)
#
# The live screen is demo.html (the detailed wall) unless you pass -Page story.html (the six-stage screen).
#
# Nothing here changes CADE's incidents or library. The only changes outside this laptop are starting/stopping the
# VMs and, each only if you answer "y": adding your current public IP to the cade-allow-ssh firewall rule, and
# issuing a new analyst API key on the cade VM when the one in the key file is not accepted or can only read.
# The live screen needs an analyst key so the analyst can confirm a run's findings on its last stage.

param(
    [ValidateSet("start", "stop")][string]$Action = "start",
    [string]$User = "yeswanth",
    [string]$Key = "$env:USERPROFILE\.ssh\google_compute_engine",
    [string]$Zone = "asia-south1-a",
    [string]$KeyFile = "$env:USERPROFILE\.cade\cloud_api_keys.txt",
    [string]$KeyName = "yeswanth",          # an analyst key: the live screen reads, and confirms findings at the end
    [ValidateSet("demo.html", "story.html")][string]$Page = "demo.html"
)
$ErrorActionPreference = "Continue"   # not Stop: ssh/gcloud write progress to stderr; failures are checked by exit code
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$screen = "http://127.0.0.1:8800/console/$Page"
$sshOpts = @("-i", $Key, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10")
$script:vmsStarted = $false

function Step($text) { Write-Host ""; Write-Host "== $text" -ForegroundColor Cyan }
function Ok($text) { Write-Host "   OK  $text" -ForegroundColor Green }
function Warn($text) { Write-Host "   !!  $text" -ForegroundColor Yellow }
function Fail($text) {
    Write-Host "   XX  $text" -ForegroundColor Red
    if ($script:vmsStarted) {
        $ans = Read-Host "   The VMs are still running (billing). Stop them now? (y = stop, Enter = leave them running)"
        if ($ans -match "^[yY]") { StopAll }
    }
    Read-Host "Press Enter to close"; exit 1
}
function Run($exe, [string[]]$argv, [int]$seconds) {
    # ssh/scp with a hard time limit. Once an ssh finished its remote command but never exited and the launcher
    # sat on step 4 for minutes; a call that overruns is now killed and reported as timed out (Code -1).
    $o = [IO.Path]::GetTempFileName(); $e = [IO.Path]::GetTempFileName()
    $quoted = $argv | ForEach-Object { if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ } }
    $p = Start-Process $exe -ArgumentList $quoted -NoNewWindow -PassThru -RedirectStandardOutput $o -RedirectStandardError $e
    $null = $p.Handle   # without this, ExitCode stays empty after the process exits
    if ($p.WaitForExit($seconds * 1000)) { $code = $p.ExitCode } else { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue; $code = -1 }
    $r = [pscustomobject]@{ Code = $code; Out = "$(Get-Content $o -Raw)".Trim(); Err = "$(Get-Content $e -Raw)".Trim() }
    Remove-Item $o, $e -Force -ErrorAction SilentlyContinue
    $r
}
function Remote($command, [int]$seconds = 20) { Run ssh (@("-n") + $sshOpts + @("$User@$cadeIp", $command)) $seconds }   # -n: never read the console
function Why($r) { if ($r.Code -eq -1) { "it timed out" } else { "ssh said: $($r.Err)" } }
function VmIp($name) {
    (gcloud compute instances describe $name --zone $Zone --format="value(networkInterfaces[0].accessConfigs[0].natIP)" 2>$null | Out-String).Trim()
}
function TunnelProcesses {
    Get-CimInstance Win32_Process -Filter "Name='ssh.exe'" | Where-Object { $_.CommandLine -like "*127.0.0.1:8800:127.0.0.1:8000*" }
}
function CloseTunnel($p) {
    Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    $win = Get-CimInstance Win32_Process -Filter "ProcessId=$($p.ParentProcessId)"   # the window tunnel.ps1 runs in
    if ($win -and $win.CommandLine -like "*tunnel.ps1*") { Stop-Process -Id $win.ProcessId -Force -ErrorAction SilentlyContinue }
}
function StopAll {
    Step "Closing the tunnel"
    $t = @(TunnelProcesses)
    $t | ForEach-Object { CloseTunnel $_ }
    if ($t.Count) { Ok "tunnel closed" } else { Ok "no tunnel was open" }
    Step "Stopping the VMs (cade, victim)"
    gcloud compute instances stop cade victim --zone $Zone
    if ($LASTEXITCODE) { Warn "gcloud could not stop the VMs. Stop them in the Cloud console, or run: run_live_demo.bat stop" }
    else { Ok "both VMs stopped (compute billing stopped; disks still bill a little)" }
}
function KeyAccepted($k) {
    if (-not $k) { return $false }
    try { Invoke-RestMethod -Uri "http://127.0.0.1:8800/incidents?limit=1" -Headers @{ Authorization = "Bearer $k" } -TimeoutSec 10 -ErrorAction Stop | Out-Null; $true } catch { $false }
}
function KeyCanConfirm($k) {
    # Confirming findings needs the analyst role. Send feedback for a run id that cannot exist: CADE checks the
    # role first, so an analyst key gets 404 (no such run) and a viewer key 403. Nothing is recorded either way.
    try {
        Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8800/playbook-runs/RUN-0000000000000000/feedback" `
            -Headers @{ Authorization = "Bearer $k" } -ContentType "application/json" `
            -Body '{"resolved": false, "findings_retained": false}' -TimeoutSec 10 -ErrorAction Stop | Out-Null
        $true
    } catch { $_.Exception.Response -and [int]$_.Exception.Response.StatusCode -eq 404 }
}
function IssueKey {
    # The key is printed once by the VM; it goes straight into the key file and is not kept anywhere else.
    $r = Remote "cd ~/cade && .venv/bin/python -m app.security.keys create --analyst $KeyName --roles analyst --description 'live screen'" 60
    $m = [regex]::Match($r.Out, '\): (cade_\S+)')
    if (-not $m.Success) { Warn "The VM did not issue a key ($(Why $r))."; return "" }
    $keep = @()
    if (Test-Path $KeyFile) {
        Copy-Item $KeyFile "$KeyFile.bak" -Force
        $keep = @(Get-Content $KeyFile | Where-Object { $_ -notmatch "^$([regex]::Escape($KeyName)) \(" })
    } else { New-Item -ItemType Directory -Force (Split-Path $KeyFile) | Out-Null }
    Set-Content -Path $KeyFile -Value ($keep + "$KeyName (analyst)=$($m.Groups[1].Value)") -Encoding ascii
    return $m.Groups[1].Value
}

if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) { Fail "gcloud is not installed or not on PATH." }

# ---------------------------------------------------------------- stop
if ($Action -eq "stop") {
    StopAll
    Read-Host "Press Enter to close"
    exit 0
}

# ---------------------------------------------------------------- start
Step "1/7  Ollama on this laptop"
function OllamaUp { try { Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 3 -ErrorAction Stop | Out-Null; $true } catch { $false } }
if (-not (OllamaUp)) {
    if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) { Fail "Ollama is not running and the 'ollama' command was not found." }
    Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
    for ($i = 0; $i -lt 20 -and -not (OllamaUp); $i++) { Start-Sleep -Seconds 1 }
}
if (OllamaUp) { Ok "Ollama is answering on 127.0.0.1:11434" } else { Fail "Ollama did not start." }

Step "2/7  Starting the VMs (cade, victim)"
$state = gcloud compute instances list --filter="name=(cade,victim)" --format="value(name,status)" 2>$null
$state | ForEach-Object { Write-Host "   $_" }
if (($state | Where-Object { $_ -notmatch "RUNNING" }).Count -gt 0 -or -not $state) {
    gcloud compute instances start cade victim --zone $Zone
    if ($LASTEXITCODE) { Fail "gcloud could not start the VMs." }
}
$script:vmsStarted = $true
$cadeIp = VmIp "cade"; $victimIp = VmIp "victim"
if (-not $cadeIp) { Fail "Could not read the cade VM's external IP." }
Ok "cade = $cadeIp   victim = $victimIp"

Step "3/7  Checking that this laptop may SSH in (firewall rule cade-allow-ssh)"
try { $myIp = (Invoke-RestMethod -Uri "https://api.ipify.org" -TimeoutSec 10 -ErrorAction Stop).ToString().Trim() } catch { $myIp = "" }
$ranges = (gcloud compute firewall-rules describe cade-allow-ssh --format="value(sourceRanges)" 2>$null | Out-String).Trim() -split "[;,\s]+" | Where-Object { $_ }
if (-not $myIp) {
    Warn "Could not look up this laptop's public IP; skipping the firewall check."
} elseif ($ranges -contains "$myIp/32") {
    Ok "$myIp is allowed"
} else {
    Warn "Your public IP $myIp is not in cade-allow-ssh (allowed now: $($ranges -join ', '))."
    $ans = Read-Host "   Add $myIp/32 to the rule, keeping the others? (y/n)"
    if ($ans -match "^[yY]") {
        $new = (@($ranges) + "$myIp/32") -join ","
        gcloud compute firewall-rules update cade-allow-ssh --source-ranges=$new
        if ($LASTEXITCODE) { Fail "Could not update the firewall rule." }
        Ok "added $myIp/32"
    } else { Warn "Left the rule unchanged; SSH will probably fail." }
}

Step "4/7  Waiting for SSH on the cade VM"
$sshOk = $false
for ($i = 0; $i -lt 18; $i++) {
    $r = Remote "true"
    if ($r.Code -eq 0) { $sshOk = $true; break }
    if ($r.Code -eq -1) { Warn "ssh did not finish within 20 s; trying again" }
    Start-Sleep -Seconds 5
}
if (-not $sshOk) { Fail "SSH to $cadeIp did not work after 18 tries ($(Why $r))." }   # e.g. a changed host key
$svc = ((Remote "systemctl is-active cade cade-forwarder").Out -split "\r?\n" | Where-Object { $_ }) -join " / "
Ok "SSH works. Services cade / cade-forwarder: $svc"
if ($svc -notmatch "^active / active") { Warn "Both services should say 'active'. CADE or the alert forwarder is not running." }

Step "5/7  Copying the live screen ($Page) to the VM"
for ($i = 0; $i -lt 3; $i++) {
    $r = Run scp (@("-q") + $sshOpts + @((Join-Path $repo "app\web\static\$Page"), "${User}@${cadeIp}:~/cade/app/web/static/$Page")) 60
    if ($r.Code -eq 0) { break }
}
if ($r.Code) { Fail "Could not copy $Page to the VM ($(Why $r))." } else { Ok "copied (no restart needed)" }

Step "6/7  Opening the tunnel in its own window (keep it open during the demo)"
$toCade = '@' + [regex]::Escape($cadeIp) + '("|\s|$)'
$stale = @(TunnelProcesses | Where-Object { $_.CommandLine -notmatch $toCade })   # left over from before the VM got a new IP
if ($stale.Count) { $stale | ForEach-Object { CloseTunnel $_ }; Warn "closed a tunnel to an old VM address" }
if (@(TunnelProcesses).Count) {
    Ok "a tunnel is already open"
} else {
    Start-Process powershell -ArgumentList @("-NoExit", "-ExecutionPolicy", "Bypass", "-File",
        "`"$(Join-Path $PSScriptRoot 'tunnel.ps1')`"", "-HostIp", $cadeIp) -WindowStyle Minimized
}
$health = $null
for ($i = 0; $i -lt 30 -and -not $health; $i++) {
    try { $health = Invoke-RestMethod -Uri "http://127.0.0.1:8800/health" -TimeoutSec 3 -ErrorAction Stop } catch { Start-Sleep -Seconds 2 }
}
if (-not $health) { Fail "CADE did not answer through the tunnel at http://127.0.0.1:8800 (check the tunnel window)." }
if ($health.status -eq "ok") { Ok "CADE is up (health: ok, LLM reachable)" }
else { Warn "CADE answered but health is '$($health.status)'. If the LLM is unreachable, check the tunnel window and Ollama." }

Step "7/7  API key and live screen"
# /health is public, so it says nothing about the key; the live screen needs a key CADE accepts for reading and
# for confirming findings (analyst role).
$apiKey = ""
if (Test-Path $KeyFile) {
    $line = Get-Content $KeyFile | Where-Object { $_ -match "^$([regex]::Escape($KeyName)) \(" } | Select-Object -First 1
    if ($line) { $apiKey = ($line -split "=", 2)[1].Trim() }
}
if ((KeyAccepted $apiKey) -and (KeyCanConfirm $apiKey)) {
    Ok "CADE accepts the '$KeyName' key as an analyst key"
} else {
    if (-not $apiKey) { Warn "There is no '$KeyName' key in $KeyFile." }
    elseif (KeyAccepted $apiKey) { Warn "The '$KeyName' key in $KeyFile can only watch; confirming findings needs an analyst key." }
    else { Warn "CADE rejects the '$KeyName' key in $KeyFile (it was probably issued for an older VM)." }
    $ans = Read-Host "   Issue a new analyst key on the cade VM and save it in that file? (y/n)"
    if ($ans -notmatch "^[yY]") { Fail "The live screen needs an analyst key CADE accepts." }
    $apiKey = IssueKey
    if (-not ((KeyAccepted $apiKey) -and (KeyCanConfirm $apiKey))) { Fail "Could not get a working analyst key from the VM." }
    Ok "new analyst key saved to $KeyFile (the previous file is kept as $KeyFile.bak)"
}
Set-Clipboard -Value $apiKey
Write-Host ""
Write-Host "   API key ($KeyName), already copied to the clipboard:" -ForegroundColor White
Write-Host ""
Write-Host "       $apiKey" -ForegroundColor Yellow
Write-Host ""
if ($Page -eq "demo.html") {
    Write-Host "   Paste the key into the key box (Ctrl+V), keep 'latest incident' and click Connect."
    Write-Host "   Until the attack it shows the newest OLD incident; it switches to the new one by itself once CADE opens it."
    Write-Host "   F11 for full screen. When a run has finished, 'Retain findings...' records the analyst's feedback."
} else {
    Write-Host "   Paste it into the 'Connect to CADE' box (Ctrl+V), then press F for full screen."
    Write-Host "   When a run has finished, the last stage (Learn) has a 'Confirm findings' button for the analyst."
}
Write-Host "   Hide this window before you start recording: it shows the key."
Start-Process $screen
Write-Host ""
Write-Host "   Live screen : $screen"
Write-Host "   Attack from : ssh -i `"$Key`" $User@$victimIp"
Write-Host ""
Write-Host "   Keep this window open during the demo. Closing it with X leaves the VMs running;" -ForegroundColor Yellow
Write-Host "   if that happens, run: run_live_demo.bat stop" -ForegroundColor Yellow
Write-Host ""
Read-Host "When the demo is over, press Enter here to close the tunnel and stop both VMs"
StopAll
Read-Host "Press Enter to close"
