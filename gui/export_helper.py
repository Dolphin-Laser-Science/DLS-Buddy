"""
gui/export_helper.py
====================

A tiny shared helper for the analysis tabs' "Export CSV…" buttons. It opens a
save-file dialog (seeded with the remembered export folder and a suggested
filename), runs the supplied export callable, and reports success/failure via a
dialog — so each tab's export handler stays a one-liner and widgets reach the
`exporting/` layer's *data* exporters only through the controller. The
suggested-filename scheme (`exporting/filenames.py`) is imported and re-exported
HERE, which is sound for the same reason the convention exists: it is a pure
string utility that touches no workspace data, so nothing data-shaped bypasses
the controller, and widgets keep importing only from this module.

    status = export_to_csv(self, self.controller, 'cumulant_fit.csv',
                            lambda p: self.controller.export_correlogram_fit(iid, res, p))
    if status:
        self.status.setText(status)

`do_export(path)` should call the matching controller export method and return
the path written — or return None to signal the user canceled from inside the
callable (e.g. declined a multi-file overwrite prompt), which is treated like a
dialog cancel. Returns a short status string, or None if canceled either way.

Directory memory: the dialog starts in `settings.last_export_dir` when that
folder still exists (falling back to the user's Documents folder), and a
successful export updates it. The open-dialog counterpart (`last_open_dir`,
via `open_start_dir`/`remember_open_dir`) is used by the shell's load buttons.
Both are app-managed settings fields (same pattern as `last_session_path`),
persisted best-effort — a failed settings write never blocks an export.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

from PySide6 import QtCore, QtWidgets

from exporting.filenames import (  # noqa: F401
    sample_token, sanitize_component, suggested_name)
#   (re-exported: widgets build suggested names through this module)


def sample_suggested_name(controller, sample_id: Optional[str],
                          analysis: str, ext: str = 'csv') -> str:
    """Suggested `{sample}_{analysis}.{ext}` name for a sample-scoped export.

    Uses the verbatim-cased committed fields; when they are unavailable or all
    blank (e.g. a manual `override-0007` sample with nothing committed), the
    sanitized sample id itself becomes the token — still unique, and routed
    through the same reserved-stem/length guards."""
    fields = (controller.sample_display_fields(sample_id)
              if sample_id is not None else None)
    if fields is None or not sample_token(*fields):
        fields = (sample_id or '', '', None)     # sanitized sid as the token
    return suggested_name(fields, analysis, ext)


def trace_suggested_name(controller, trace_id: str, ext: str = 'csv') -> str:
    """Suggested name for a raw count-rate trace export.

    An ATTACHED trace takes its measurement's sample token, so the trace CSV lands
    beside that sample's other exports; a standalone one falls back to its own
    label, then its id. The `count_rate_trace` stem matches the trace plot's image
    stem, so the CSV and the PNG of one trace sort together.
    """
    lt = controller.workspace.traces.get(trace_id)
    iid = getattr(lt, 'measurement_id', None)
    if iid is not None and iid in controller.workspace.measurements:
        # The sample token alone is NOT unique here: a trace belongs to one
        # measurement, and a multi-angle file gives a sample eight of them, which
        # would all propose the same name and quietly overwrite each other. The
        # angle is the within-sample axis the sidebar already distinguishes them by.
        lm = controller.workspace.measurements[iid]
        angle = lm.committed_params.get('angle_deg')
        if angle is None:
            angle = lm.working_params.get('angle_deg')
        analysis = ('count_rate_trace' if angle is None else
                    sanitize_component(f'count_rate_trace_{angle:g}deg'))
        return sample_suggested_name(
            controller, controller.sample_id_of(iid), analysis, ext)
    label = (getattr(lt, 'sample_label', None) or trace_id)
    return suggested_name((label, '', None), 'count_rate_trace', ext)


def _start_dir(stored: str) -> str:
    """A directory the dialog can start in: the remembered one if it still
    exists, else the user's Documents folder, else '' (Qt's own default)."""
    if stored and os.path.isdir(stored):
        return stored
    docs = QtCore.QStandardPaths.writableLocation(
        QtCore.QStandardPaths.StandardLocation.DocumentsLocation)
    return docs or ''


def export_start_dir(controller) -> str:
    """Where a save dialog (CSV or plot) should start."""
    return _start_dir(controller.settings.last_export_dir)


def open_start_dir(controller) -> str:
    """Where a data-file open dialog should start."""
    return _start_dir(controller.settings.last_open_dir)


def _remember(controller, field: str, path: str) -> None:
    """Persist the folder of a successful save/open; best-effort by design."""
    folder = os.path.dirname(os.path.abspath(path))
    if not folder or getattr(controller.settings, field) == folder:
        return
    setattr(controller.settings, field, folder)
    try:
        controller.settings.save()
    except OSError:
        pass    # a read-only install must not break exporting


def remember_export_dir(controller, path: str) -> None:
    _remember(controller, 'last_export_dir', path)


def remember_open_dir(controller, path: str) -> None:
    _remember(controller, 'last_open_dir', path)


def confirm_overwrite(parent, paths) -> bool:
    """Ask before overwriting files a save dialog never showed (multi-file
    exports write derived sibling paths). True = proceed. No existing files →
    no dialog, True."""
    existing = [p for p in paths if os.path.exists(p)]
    if not existing:
        return True
    names = [os.path.basename(p) for p in existing[:10]]
    if len(existing) > 10:
        names.append(f'…and {len(existing) - 10} more')
    listing = '\n'.join(names)
    resp = QtWidgets.QMessageBox.question(
        parent, 'Overwrite existing files?',
        'This export will overwrite:\n\n' + listing,
        QtWidgets.QMessageBox.StandardButton.Yes
        | QtWidgets.QMessageBox.StandardButton.Cancel,
        QtWidgets.QMessageBox.StandardButton.Cancel)
    return resp == QtWidgets.QMessageBox.StandardButton.Yes


def export_to_csv(parent, controller, default_name: str,
                  do_export: Callable[[str], Optional[str]]) -> Optional[str]:
    """Prompt for a path and run `do_export(path)`; report the outcome.

    Returns a status string on success/failure, or None if canceled — either
    the dialog itself, or `do_export` returning None (its cancel signal, e.g.
    a declined overwrite prompt)."""
    start = os.path.join(export_start_dir(controller), default_name)
    path, _ = QtWidgets.QFileDialog.getSaveFileName(
        parent, 'Export to CSV', start,
        'CSV files (*.csv);;All files (*)')
    if not path:
        return None
    try:
        written = do_export(path)
    except Exception as exc:                       # noqa: BLE001 - surface to user
        QtWidgets.QMessageBox.critical(
            parent, 'Export failed', f'Could not export.\n\n{exc}')
        return 'Export failed — see dialog.'
    if written is None:
        return None
    remember_export_dir(controller, path)
    return f'Exported → {written}'
