@echo off
cd /d "%~dp0"

if not exist .venv (
    echo Setting up environment for the first time, this may take a minute...
    py -3.13 -m venv .venv
    .venv\Scripts\pip install -r requirements.txt
)

.venv\Scripts\python.exe -m gui.main
