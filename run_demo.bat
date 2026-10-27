@echo off
REM Double-click this to set up and launch the CADE demo on Windows.
REM It just runs run_demo.ps1 with the execution policy relaxed for this one run.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_demo.ps1" %*
