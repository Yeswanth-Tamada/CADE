@echo off
REM Double-click to set up the live conference demo: starts Ollama and both GCP VMs, opens the tunnel,
REM checks the CADE API key and opens the live screen (demo.html). Keep the window open during the demo;
REM press Enter in it afterwards to close the tunnel and stop both VMs.
REM   run_live_demo.bat                          start everything (and stop it again when you press Enter at the end)
REM   run_live_demo.bat start -Page story.html   the same, but with the six-stage screen instead of demo.html
REM   run_live_demo.bat stop                     close the tunnel and stop both VMs (e.g. if the window was closed with X)
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0deploy\gcp\live_demo.ps1" %*
