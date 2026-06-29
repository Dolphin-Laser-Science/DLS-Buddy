"""
gui/export_helper.py
====================

A tiny shared helper for the analysis tabs' "Export CSV…" buttons. It opens a
save-file dialog, runs the supplied export callable, and reports success/failure
via a dialog — so each tab's export handler stays a one-liner and the widgets
never import the `exporting/` layer directly (they go through the controller).

    status = export_to_csv(self, 'cumulant_fit.csv',
                            lambda p: self.controller.export_correlogram_fit(iid, res, p))
    if status:
        self.status.setText(status)

`do_export(path)` should call the matching controller export method and return
the path written. Returns a short status string, or None if the user cancelled.
"""

from __future__ import annotations

from typing import Callable, Optional

from PySide6 import QtWidgets


def export_to_csv(parent, default_name: str,
                  do_export: Callable[[str], str]) -> Optional[str]:
    """Prompt for a path and run `do_export(path)`; report the outcome.

    Returns a status string on success/failure, or None if the dialog was
    cancelled (so the caller can leave its status line untouched).
    """
    path, _ = QtWidgets.QFileDialog.getSaveFileName(
        parent, 'Export to CSV', default_name,
        'CSV files (*.csv);;All files (*)')
    if not path:
        return None
    try:
        written = do_export(path)
    except Exception as exc:                       # noqa: BLE001 - surface to user
        QtWidgets.QMessageBox.critical(
            parent, 'Export failed', f'Could not export.\n\n{exc}')
        return 'Export failed — see dialog.'
    return f'Exported → {written}'
