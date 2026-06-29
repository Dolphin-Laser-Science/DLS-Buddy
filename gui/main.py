"""
gui/main.py
===========

Application entry point for the DLS Buddy GUI.

Run it from the repository root so the top-level packages (core, analysis, …)
are importable:

    python -m gui.main

What this file does, in order
-----------------------------
1. Pin matplotlib to its Qt backend BEFORE anything imports pyplot. The plotting
   layer (`plotting/plots.py`) does `import matplotlib.pyplot` at import time, so
   the backend must be chosen first or matplotlib may bind to a non-Qt one.
2. Create the single `QApplication` -- the object that owns the GUI event loop.
3. Show the main window.
4. Call `app.exec()`, which starts the event loop: a blocking call that waits for
   user events (clicks, key presses, the matplotlib canvas redrawing) and
   dispatches each to the connected slot. It returns only when the last window
   closes; that return code is what we hand back to the OS.

The event loop is the heart of every desktop GUI: your code stops running "top to
bottom" and instead reacts to events. The widgets connected their signals to
controller-backed slots in main_window.py; the loop is what actually delivers
those signals.
"""

from __future__ import annotations

import os
import sys

# --- step 1: backend, before any pyplot import (directly or via plots.py) ---
os.environ.setdefault('QT_API', 'pyside6')
import matplotlib
matplotlib.use('QtAgg')

from PySide6 import QtWidgets

from gui.main_window import MainWindow


def main() -> int:
    # --- step 2: the one application object (owns the event loop) ---
    app = QtWidgets.QApplication(sys.argv)

    # --- step 3: build and show the window ---
    window = MainWindow()
    window.show()

    # --- step 4: run the event loop until the window closes ---
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
