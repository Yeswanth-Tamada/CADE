# Deploy the committed code (git HEAD) to the CADE lab VM on Google Cloud.
#
#   powershell -ExecutionPolicy Bypass -File deploy\gcp\deploy.ps1 -DryRun   # show what would change
#   powershell -ExecutionPolicy Bypass -File deploy\gcp\deploy.ps1           # deploy and restart
#
# Only committed files are sent, so the VM always runs a known commit (recorded in
# ~/cade/DEPLOYED_COMMIT). Uncommitted changes are listed and NOT deployed.
# The VM keeps its own .env, cade.db, .venv, lab/ and API key files; nothing on it is deleted.
# Before copying, the database is backed up to ~/cade-backups/ (last 10 kept).
# The VM must be running: gcloud compute instances start cade victim --zone asia-south1-a

param(
    [switch]$DryRun,
    [switch]$NoRestart,
    [string]$HostIp = "",               # empty = look up the VM's current external IP with gcloud
    [string]$User = "yeswanth",
    [string]$Key = "$env:USERPROFILE\.ssh\google_compute_engine",
    [string]$Instance = "cade",
    [string]$Zone = "asia-south1-a"
)
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $repo

# What is being deployed
$commit = (git rev-parse --short HEAD).Trim()
$subject = (git log -1 --format=%s).Trim()
Write-Host "Deploying commit $commit  ($subject)"
$dirty = git status --porcelain
if ($dirty) {
    Write-Warning "These uncommitted changes will NOT be deployed (commit them first if you want them on the VM):"
    $dirty | ForEach-Object { Write-Host "    $_" }
}

# Where to
if (-not $HostIp) {
    $HostIp = (gcloud compute instances describe $Instance --zone $Zone `
        --format="value(networkInterfaces[0].accessConfigs[0].natIP)" 2>$null | Out-String).Trim()
    if (-not $HostIp) {
        Write-Error "Could not get the external IP of '$Instance'. Is it running? (gcloud compute instances start $Instance --zone $Zone)"
        exit 1
    }
}
$target = "$User@$HostIp"
$sshOpts = @("-i", $Key, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=20")

# Package: git archive of HEAD with LF line endings (core.autocrlf is ignored on purpose)
$tmp = Join-Path $env:TEMP "cade-deploy-$commit"
New-Item -ItemType Directory -Force $tmp | Out-Null
$tar = Join-Path $tmp "cade.tar"
git -c core.autocrlf=false archive --format=tar -o $tar HEAD
if ($LASTEXITCODE) { Write-Error "git archive failed"; exit 1 }
# The remote script comes from the working tree; make sure it has LF endings
$remote = Join-Path $tmp "remote_deploy.sh"
$text = [IO.File]::ReadAllText((Join-Path $PSScriptRoot "remote_deploy.sh")) -replace "`r`n", "`n"
[IO.File]::WriteAllText($remote, $text, (New-Object Text.UTF8Encoding $false))

try {
    Write-Host "Uploading to $target ..."
    & scp @sshOpts -q $tar $remote "${target}:/tmp/"
    if ($LASTEXITCODE) { Write-Error "upload failed (is the VM running and is your IP allowed in the cade-allow-ssh firewall rule?)"; exit 1 }
    $dry = if ($DryRun) { "1" } else { "0" }
    $restart = if ($NoRestart) { "0" } else { "1" }
    & ssh @sshOpts $target "bash /tmp/remote_deploy.sh /tmp/cade.tar $commit $dry $restart; rc=`$?; rm -f /tmp/remote_deploy.sh; exit `$rc"
    $rc = $LASTEXITCODE
} finally {
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}
if ($rc) { Write-Error "deploy failed on the VM (exit $rc)"; exit $rc }
