#!/bin/bash
# Double-click launcher for macOS (mirrors "Launch DLS Buddy (Windows).bat").
cd "$(dirname "$0")" || exit 1

if [ ! -d .venv ]; then
    echo "Setting up environment for the first time, this may take a minute..."
    PY="$(command -v python3.13 || command -v python3)"
    if [ -z "$PY" ]; then
        echo "Python 3 not found. Install Python 3.13 from https://www.python.org/downloads/ and re-run."
        read -n 1 -s -r -p "Press any key to close..."
        exit 1
    fi
    "$PY" -m venv .venv
    .venv/bin/pip install -r requirements.txt
fi

.venv/bin/python -m gui.main
