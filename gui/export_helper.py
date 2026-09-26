"""
gui/export_helper.py
====================

A tiny shared helper for the analysis tabs' "Export CSV…" buttons. It opens a
save-file dialog (seeded with the remembered export folder and a suggested
filename), runs the supplied export callable, and reports the outcome — success
as a status string the tab shows, failure in a dialog — so each tab's export
handler stays a one-liner and widgets reach the
`exporting/` layer's *data* exporters only through the controller. The
suggested-filename scheme (`exporting/filenames.py`) is imported and re-exported
HERE, which is sound for the same reason the convention exists: it is a pure
string utility that touches no workspace data, so nothing data-shaped bypasses
the controller, and widgets keep importing only from this module.

The scheme needs three facts about each measurement it names — the sample's
polymer/solvent, the run's name, and its committed angle — so the functions below
are where those are read off the workspace and handed to that pure layer as
strings. They read only what a name is made of; nothing analysable travels here.

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
from typing import Callable, List, Optional, Sequence, Tuple, Union

from PySide6 import QtCore, QtWidgets

from core.naming import measurement_name
from exporting.filenames import (  # noqa: F401
    ExportMember, batch_export_names, sample_token, sanitize_component,
    suggested_name)
#   (re-exported: widgets build suggested names through this module)


def sample_name_fields(controller, sample_id: Optional[str]):
    """The `(polymer, solvent)` pair behind a sample token, or None for a
    sample-independent export.

    Uses the verbatim-cased committed fields; when they are unavailable or all
    blank (e.g. a manual `override-0007` sample with nothing committed), the
    sanitized sample id itself becomes the token — still unique, and routed
    through the same reserved-stem/length guards.
    """
    if sample_id is None:
        return None
    fields = controller.sample_display_fields(sample_id)
    pair = (fields[0], fields[1]) if fields is not None else None
    if pair is None or not sample_token(*pair):
        pair = (sample_id, '')                   # sanitized sid as the token
    return pair


def sample_suggested_name(controller, sample_id: Optional[str],
                          analysis: str, ext: str = 'csv') -> str:
    """Suggested `{sample}_{analysis}.{ext}` name for a sample-scoped export."""
    return suggested_name(sample_name_fields(controller, sample_id), analysis, ext)


def measurement_export_names(controller, sample_id: Optional[str],
                             items: Sequence[Tuple[str, Optional[str]]],
                             analysis: str,
                             ext: str = 'csv') -> Tuple[str, List[str]]:
    """`(dialog name, per-file suffixes)` for a per-measurement export or batch.

    `items` is `(item_id, method-or-None)` in the order the files are written; the
    suffixes come back aligned with it. This is where the workspace is read — the
    name from the one builder, the angle from the COMMITTED parameters — so
    `exporting/filenames.py` stays pure over strings. An id the workspace no longer
    holds contributes no name token; the collision fallback still keeps its file
    apart from its neighbours.
    """
    ws = controller.workspace.measurements
    members = []
    for iid, method in items:
        lm = ws.get(iid)
        members.append(ExportMember(
            iid, measurement_name(lm) if lm is not None else '',
            lm.committed_params.get('angle_deg') if lm is not None else None,
            method))
    return batch_export_names(sample_name_fields(controller, sample_id), analysis,
                              members, ext)


def trace_suggested_name(controller, trace_id: str, ext: str = 'csv') -> str:
    """Suggested name for a raw count-rate trace export.

    An ATTACHED trace follows the one-measurement rule, so the trace CSV lands
    beside that measurement's other exports and carries the same name and angle
    they do (the sample token alone is not unique: a multi-angle file gives one
    sample eight traces, which would all propose the same name and quietly
    overwrite each other). A STANDALONE trace has no measurement and keeps its own
    label, then its id. The `count_rate_trace` stem matches the trace plot's image
    stem, so the CSV and the PNG of one trace sort together.
    """
    lt = controller.workspace.traces.get(trace_id)
    iid = getattr(lt, 'measurement_id', None)
    if iid is not None and iid in controller.workspace.measurements:
        return measurement_export_names(
            controller, controller.sample_id_of(iid), [(iid, None)],
            'count_rate_trace', ext)[0]
    label = (getattr(lt, 'sample_label', None) or trace_id)
    return suggested_name((label, ''), 'count_rate_trace', ext)


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


# How long an export's outcome stays on the window's status bar, where a tab with no
# status line of its own reports it (ms).
EXPORT_STATUS_MS = 6000


def written_summary(paths: Sequence[str]) -> str:
    """The status line for a multi-file export: `N files → <folder>`, or the one
    path. A count and a folder keep the control column's word-wrapping status label to
    one line however many measurements were ticked."""
    if len(paths) == 1:
        return f'→ {paths[0]}'
    folders = {os.path.dirname(p) for p in paths}
    where = folders.pop() if len(folders) == 1 else 'several folders'
    return f'{len(paths)} files → {where}'


def export_to_csv(parent, controller, default_name: str,
                  do_export: Callable[[str], Optional[Union[str, Sequence[str]]]]
                  ) -> Optional[str]:
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
    except Exception as exc:                       # noqa: BLE001 — surfaced in the dialog
        QtWidgets.QMessageBox.critical(
            parent, 'Export failed', f'Could not export.\n\n{exc}')
        return 'Export failed — see dialog.'
    if written is None:
        return None
    remember_export_dir(controller, path)
    if isinstance(written, (list, tuple)):          # a multi-file export's paths
        return f'Exported {written_summary(written)}'
    return f'Exported → {written}'
