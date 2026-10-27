# Connect this laptop to the CADE lab VM on Google Cloud.
#
# One SSH connection does two jobs:
#   -R 11434  the VM reaches this laptop's Ollama (GPU) at its own 127.0.0.1:11434
#   -L 8800   this laptop reaches the VM's CADE console at http://127.0.0.1:8800/console/
#             (8800, not 8000, so it never collides with a local CADE server on this laptop)
# Both ends listen on loopback only; nothing is exposed to the internet.
#
# Usage (Ollama must be running on this laptop):  powershell -File deploy\gcp\tunnel.ps1
# Leave the window open while you work; Ctrl+C closes the tunnel.

param(
    [string]$HostIp = "34.100.133.180",
    [string]$User = "yeswanth",
    [string]$Key = "$env:USERPROFILE\.ssh\google_compute_engine"
)

try {
    Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 5 | Out-Null
} catch {
    Write-Warning "Ollama is not answering on this laptop; CADE's LLM steps will fail until it does."
}

Write-Host "Tunnel open. Console: http://127.0.0.1:8800/console/  (Ctrl+C to close)"
ssh -N -i $Key `
    -o StrictHostKeyChecking=accept-new `
    -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 `
    -R 11434:127.0.0.1:11434 `
    -L 127.0.0.1:8800:127.0.0.1:8000 `
    "$User@$HostIp"
