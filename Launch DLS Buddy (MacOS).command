#!/bin/bash
# Double-click launcher for macOS (mirrors "Launch DLS Buddy (Windows).bat").
cd "$(dirname "$0")" || exit 1

pause_and_exit() {
    echo
    echo "$1"
    read -n 1 -s -r -p "Press any key to close..."
    exit 1
}

# First run: create the virtual environment. A failed create must not leave a
# half-made .venv behind, or every later launch would skip this step and fail.
if [ ! -x .venv/bin/python ]; then
    echo "Setting up environment for the first time, this may take a minute..."
    PY="$(command -v python3.13 || command -v python3)"
    if [ -z "$PY" ]; then
        pause_and_exit "Python 3 not found. Install Python 3.13 from https://www.python.org/downloads/ and re-run."
    fi
    if ! "$PY" -m venv .venv; then
        rm -rf .venv
        pause_and_exit "Could not create the virtual environment (see the messages above)."
    fi
fi

# Install (or re-install) dependencies whenever requirements.txt has changed
# since the last successful install, so an update that adds a package works.
STAMP=.venv/requirements.installed
if [ ! -f "$STAMP" ] || ! cmp -s requirements.txt "$STAMP"; then
    echo "Installing dependencies..."
    if ! .venv/bin/python -m pip install -r requirements.txt; then
        pause_and_exit "Dependency installation failed. Check your network connection and run this launcher again."
    fi
    cp requirements.txt "$STAMP"
fi

.venv/bin/python -m gui.main || pause_and_exit "DLS Buddy exited with an error (see the messages above)."
