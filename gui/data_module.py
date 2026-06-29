"""
gui/data_module.py
==================

The Data tab: confirm and manage the physical parameters of the selected
measurement. This is the home of the parser "confirmation step" (parse -> blanks
-> user fills -> commit) and the place where shared per-sample parameters will be
entered once (the propagation across a sample's measurements lands with the SLS /
multi-measurement work; for now it edits one measurement at a time).

It owns the editable parameter table, the Update (commit) / Undo (revert to
committed) buttons, the changed-since-commit highlighting, and a pending-update
indicator. It runs NO analysis -- the analysis modules (DLS, SLS) read the
committed parameters. All edits go through the controller (invariant: widgets
never touch the engine or the data model directly).

It exposes a `committed` signal so the shell can re-group and refresh the sidebar
after a commit (committing can change a measurement's sample identity).
"""

from __future__ import annotations

from typing import Optional, Tuple

from PySide6 import QtCore, QtWidgets

from core.workspace import _DLS_PARAM_KEYS, _SLS_PARAM_KEYS
from app.controller import _SHARED_PARAM_KEYS
from app import units as U
from gui.help import section_header


# Unit labels now live in the dedicated Unit column, so the parameter names here
# carry no unit suffix.
_PARAM_LABELS = {
    'label':       'Label',
    # User-facing label is "Solute name" (samples need not be polymers); the
    # internal field stays `polymer_name` to avoid a data-model/session rename.
    'polymer_name': 'Solute name',
    'solvent_name': 'Solvent name',
    'concentration_g_per_mL': 'Concentration',
    'temperature_K': 'Temperature',
    'angle_deg': 'Scattering angle',
    'wavelength_nm': 'Wavelength',
    'solvent_refractive_index': 'Solvent refractive index',
    'viscosity_Pa_s': 'Viscosity',
    'dn_dc_mL_per_g': 'dn/dc',
    'mw_fraction': 'Mw fraction',
    'analyzer_geometry': 'Polarization (DDLS)',
}
_STRING_KEYS = {'label', 'polymer_name', 'solvent_name', 'mw_fraction'}
# Polarisation/analyser geometry: a dropdown, not a typed value. The blank label
# means "unspecified" (None) -- ordinary polarised DLS, ignored by the depolarised
# analysis. Only VV + VH correlograms pair into D_r; VU is record-only.
_GEOMETRY_KEY = 'analyzer_geometry'
_GEOMETRY_BLANK = '—'
_GEOMETRY_OPTIONS = (_GEOMETRY_BLANK, 'VV', 'VH', 'VU')
# Numeric params the user may enter in a choice of units (canonical value stored).
# Maps the param key -> the units-module quantity name.
_QUANTITY = {
    'concentration_g_per_mL': 'concentration',
    'temperature_K': 'temperature',
    'viscosity_Pa_s': 'viscosity',
}
# Numeric params with a single fixed unit (already human-scale): shown, not chosen.
_FIXED_UNIT = {
    'angle_deg': 'deg',
    'wavelength_nm': 'nm',
    'dn_dc_mL_per_g': 'mL/g',
}


def _fmt_num(x: float) -> str:
    """Clean numeric display: no float-noise tail (0.00089, not 0.000890000…1)."""
    return f'{x:.10g}'
# A changed-but-uncommitted cell is flagged yellow. We also force a dark text
# colour: under the dark theme the default text is near-white, which is invisible
# on yellow (the value you just typed disappears). Black-on-yellow is readable in
# both themes.
_DIRTY_BG = QtCore.Qt.GlobalColor.yellow
_DIRTY_FG = QtCore.Qt.GlobalColor.black


class DataModule(QtWidgets.QWidget):
    """Parameter confirmation surface for the selected measurement."""

    committed = QtCore.Signal()   # emitted after a commit (grouping may change)

    def __init__(self, controller, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.item_id: Optional[str] = None
        # Every measurement highlighted in the sidebar (the focused one is item_id).
        # On commit, edits to the focused measurement are applied to all of these.
        self._selected_ids: Tuple[str, ...] = ()
        self.keys: Tuple[str, ...] = ()
        self._suppress = False
        # Current display unit per quantity (canonical value is always what's
        # stored). Defaults to the human-scale unit; the user's choice persists
        # across measurement switches within the session.
        self._units = {q: U.default_unit(q) for q in _QUANTITY.values()}
        self._build_ui()
        self.set_measurement(None)

    # ------------------------------------------------------------------ UI ---
    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        self.header = QtWidgets.QLabel()
        self.header.setWordWrap(True)
        layout.addWidget(self.header)

        layout.addWidget(section_header(
            'Physical parameters — edit, then Update to apply',
            'Which parameters each analysis needs:',
            bullets=[
                '<b>All:</b> solute &amp; solvent name, temperature, wavelength, '
                'refractive index.',
                '<b>DLS (size):</b> + viscosity, scattering angle.',
                '<b>SLS (Mw, A₂):</b> + dn/dc, concentration, and a calibration.',
                'Identity &amp; optics are shared across the sample (enter once); '
                'concentration and angle are per-measurement.',
            ]))
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(['Parameter', 'Value', 'Unit'])
        self.table.verticalHeader().setVisible(False)
        hdr = self.table.horizontalHeader()
        # All three columns are user-draggable (feedback A1/B3). The Value column is
        # no longer forced to Stretch — Stretch made it fill the whole window (the
        # "huge empty space"), and a stretched column cannot be dragged. Interactive
        # with a sensible default lets the user size it; trailing space is harmless.
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Interactive)
        self.table.setColumnWidth(0, 200)
        self.table.setColumnWidth(1, 260)
        self.table.setColumnWidth(2, 90)
        # Single click into a Value cell starts editing (feedback B2) — only the
        # Value column is editable, so navigating Parameter/Unit cells is harmless.
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.AllEditTriggers)
        self.table.itemChanged.connect(self._on_cell_changed)
        layout.addWidget(self.table, 1)
        # (The shared-vs-per-measurement note now lives in the section's "?" help,
        # so it is no longer repeated as an always-visible line — feedback B4.)

        row = QtWidgets.QHBoxLayout()
        self.update_button = QtWidgets.QPushButton('Update')
        self.update_button.setToolTip('Apply the edited parameters to the sample.')
        self.update_button.clicked.connect(self._on_update)
        self.undo_button = QtWidgets.QPushButton('Undo')
        self.undo_button.setToolTip(
            'Step back to the previously applied parameters. With un-applied edits '
            'showing, the first Undo discards those edits.')
        self.undo_button.clicked.connect(self._on_undo)
        self.reset_button = QtWidgets.QPushButton('Reset')
        self.reset_button.setToolTip(
            'Clear all entered parameters for this sample (the scattering angle is '
            'kept). Press Update to apply, or Undo to restore.')
        self.reset_button.clicked.connect(self._on_reset)
        row.addWidget(self.update_button)
        row.addWidget(self.undo_button)
        row.addWidget(self.reset_button)
        self.pending = QtWidgets.QLabel('')
        self.pending.setStyleSheet('color:#b06000;')
        row.addWidget(self.pending, 1)
        layout.addLayout(row)

    # ---------------------------------------------------------- selection ---
    def set_selected_ids(self, item_ids) -> None:
        """Record which measurements are highlighted in the sidebar (feedback A2).
        The shell calls this on every selection change; the focused measurement is
        still set via set_measurement."""
        self._selected_ids = tuple(item_ids)

    def set_measurement(self, item_id: Optional[str]) -> None:
        """Point the table at a measurement (or clear it when None)."""
        self.item_id = item_id
        if item_id is None:
            self.keys = ()
            self._suppress = True
            self.table.setRowCount(0)
            self._suppress = False
            # Don't repeat "load data" here (the load buttons + status bar already
            # say it, feedback B4) — just point at the next step.
            self.header.setText(
                'No measurement selected — pick one in the Workspace list.')
            self._set_enabled(False)
            self.pending.setText('')
            return

        lm = self.controller.workspace.measurements[item_id]
        self.keys = _DLS_PARAM_KEYS if lm.kind == 'dls' else _SLS_PARAM_KEYS
        self._build_rows()
        self._populate()
        self.header.setText(
            f'<b>{item_id}</b> &mdash; {lm.kind.upper()} measurement '
            f'({self._raw_summary(lm)})')
        self._set_enabled(True)
        self._refresh_pending()

    @staticmethod
    def _raw_summary(lm) -> str:
        if lm.kind == 'dls':
            return f'{len(lm.raw.get("delay_times_s", []))} delay points'
        return f'{len(lm.raw.get("angles_deg", []))} angles'

    # ------------------------------------------------------------- editing ---
    def _build_rows(self) -> None:
        self._suppress = True
        self.table.setRowCount(len(self.keys))
        for r, key in enumerate(self.keys):
            name = QtWidgets.QTableWidgetItem(_PARAM_LABELS.get(key, key))
            name.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
            name.setData(QtCore.Qt.ItemDataRole.UserRole, key)
            self.table.setItem(r, 0, name)
            # An empty, non-editable value item always exists (so _refresh_dirty can
            # address it); for the geometry row a combo widget is laid over it.
            value_item = QtWidgets.QTableWidgetItem('')
            if key == _GEOMETRY_KEY:
                value_item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(r, 1, value_item)
            if key == _GEOMETRY_KEY:
                self._build_geometry_cell(r)
            else:
                # Drop any cell widget left over from a previous measurement whose
                # key tuple put a combo at this row (setItem does NOT remove a cell
                # widget — it's a separate overlay layer). Without this, switching a
                # measurement type can leave a stale geometry combo here.
                self.table.removeCellWidget(r, 1)
            self._build_unit_cell(r, key)
        self._suppress = False

    def _build_geometry_cell(self, row: int) -> None:
        """Value column for the polarisation row: a VV/VH/VU/— dropdown."""
        combo = QtWidgets.QComboBox()
        combo.addItems(_GEOMETRY_OPTIONS)
        combo.setToolTip(
            'Polarisation/analyser geometry of this correlogram (incident, '
            'analyser): VV = polarised, VH = depolarised, VU = no analyser, '
            '— = unspecified. The DPLS sub-tab pairs a VV with a VH '
            'at each angle to extract rotational diffusion. Leave — for ordinary '
            'polarised DLS.')
        combo.currentTextChanged.connect(self._on_geometry_changed)
        self.table.setCellWidget(row, 1, combo)

    def _build_unit_cell(self, row: int, key: str) -> None:
        """Unit column: a unit-picker combo for convertible quantities, a static
        unit label for fixed-unit numbers, blank for names/dimensionless."""
        if key in _QUANTITY:
            quantity = _QUANTITY[key]
            combo = QtWidgets.QComboBox()
            combo.addItems(U.unit_options(quantity))
            combo.setCurrentText(self._units[quantity])     # before connecting
            combo.currentTextChanged.connect(
                lambda u, q=quantity: self._on_unit_changed(q, u))
            self.table.setCellWidget(row, 2, combo)
        else:
            # Remove any stale unit-picker combo left over from a prior measurement
            # whose key tuple placed a _QUANTITY combo at this row index (e.g. a DLS
            # viscosity combo leaking onto an SLS "Mw fraction" row — both land at
            # row 8). setItem does not clear a cell widget, so do it explicitly.
            self.table.removeCellWidget(row, 2)
            unit = _FIXED_UNIT.get(key, '')
            item = QtWidgets.QTableWidgetItem(unit)
            item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
            item.setForeground(QtCore.Qt.GlobalColor.gray)
            self.table.setItem(row, 2, item)

    def _display_text(self, key: str, v) -> str:
        """The value as shown in the current unit (canonical -> display)."""
        if v is None:
            return ''
        if key in _QUANTITY:
            return _fmt_num(U.from_canonical(_QUANTITY[key], float(v),
                                             self._units[_QUANTITY[key]]))
        if key in _FIXED_UNIT:
            return _fmt_num(float(v))
        return str(v)

    def _populate(self) -> None:
        self._suppress = True
        working = self.controller.workspace.measurements[self.item_id].working_params
        for r, key in enumerate(self.keys):
            if key == _GEOMETRY_KEY:
                combo = self.table.cellWidget(r, 1)
                v = working.get(key)
                combo.setCurrentText(v if v in _GEOMETRY_OPTIONS else _GEOMETRY_BLANK)
                continue
            self.table.item(r, 1).setText(self._display_text(key, working.get(key)))
        self._suppress = False
        self._refresh_dirty()

    def _on_unit_changed(self, quantity: str, unit: str) -> None:
        """User picked a different input unit: remember it and re-display the
        affected value(s) in the new unit (the stored canonical value is unchanged,
        so this never marks anything dirty)."""
        if self._suppress:
            return
        self._units[quantity] = unit
        if self.item_id is not None:
            self._populate()

    def _on_cell_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._suppress or item.column() != 1 or self.item_id is None:
            return
        key = self.table.item(item.row(), 0).data(QtCore.Qt.ItemDataRole.UserRole)
        text = item.text().strip()
        if key in _STRING_KEYS:
            value = text or None
        elif text == '':
            value = None
        else:
            try:
                entered = float(text)
            except ValueError:
                # Not a number: revert the cell to the stored value, no change.
                self._revert_cell(item.row(), key)
                return
            # Interpret the typed number in the row's selected unit, store canonical.
            if key in _QUANTITY:
                value = U.to_canonical(_QUANTITY[key], entered,
                                       self._units[_QUANTITY[key]])
            else:
                value = entered          # fixed-unit numbers are already canonical
        # Sample-shared params are entered once and propagate to every
        # measurement in the sample; per-measurement axes stay local.
        if key in _SHARED_PARAM_KEYS:
            self.controller.set_shared_param(self.item_id, key, value)
        else:
            # The Mw fraction is a per-measurement axis like angle/concentration:
            # set it only on the edited row (and any other highlighted rows), with
            # no propagate-and-commit popup — applying it is now consistent with
            # every other per-measurement parameter (feedback 2026-06-26 #13).
            self.controller.set_param(self.item_id, key, value)
        self._refresh_dirty()
        self._refresh_pending()

    def _on_geometry_changed(self, text: str) -> None:
        """Polarisation dropdown changed: store the per-measurement geometry."""
        if self._suppress or self.item_id is None:
            return
        value = None if text == _GEOMETRY_BLANK else text
        self.controller.set_param(self.item_id, _GEOMETRY_KEY, value)
        self._refresh_dirty()
        self._refresh_pending()

    def _revert_cell(self, row: int, key: str) -> None:
        self._suppress = True
        v = self.controller.workspace.measurements[self.item_id].working_params.get(key)
        self.table.item(row, 1).setText(self._display_text(key, v))
        self._suppress = False

    # ---------------------------------------------------------- commit/undo ---
    def _apply_edits_to_selection(self) -> None:
        """Copy the focused measurement's pending edits to the other highlighted
        measurements before committing (feedback A2). No-op unless several are
        highlighted."""
        if self.item_id is None:
            return
        others = [i for i in self._selected_ids if i != self.item_id]
        if not others:
            return
        working = self.controller.workspace.measurements[self.item_id].working_params
        for key in self.controller.dirty_keys(self.item_id):
            self.controller.apply_value_to_items(others, key, working.get(key))

    def _on_update(self) -> None:
        self._apply_edits_to_selection()
        self.controller.commit()
        self._populate()           # working == committed now -> highlights clear
        self._refresh_pending()
        self.committed.emit()      # shell re-groups + refreshes the sidebar

    def _on_undo(self) -> None:
        grouping_changed = self.controller.undo()
        self._populate()
        self._refresh_pending()
        if grouping_changed:
            # Stepping back to a previous applied state may change sample identity,
            # so re-group + refresh the sidebar exactly like a commit does.
            self.committed.emit()

    def _on_reset(self) -> None:
        if self.item_id is None:
            return
        ids = self._sample_scope_ids()
        n = len(ids)
        scope = 'this measurement' if n <= 1 else f'all {n} measurements in this sample'
        confirm = QtWidgets.QMessageBox.question(
            self, 'Reset parameters',
            f'Clear all entered parameters for {scope}? The scattering angle and the '
            'raw data are kept. Nothing is applied until you press Update, and Undo '
            'will restore the previous values.',
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No)
        if confirm != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.controller.reset_working_params(ids)
        self._populate()
        self._refresh_pending()

    def _sample_scope_ids(self) -> Tuple[str, ...]:
        """The measurements Reset/propagation act on: the highlighted set (a sidebar
        sample heading highlights all of its measurements) plus the focused one."""
        ids = list(self._selected_ids)
        if self.item_id is not None and self.item_id not in ids:
            ids.append(self.item_id)
        return tuple(ids)

    # ------------------------------------------------------------- helpers ---
    def _refresh_dirty(self) -> None:
        if self.item_id is None:
            return
        dirty = set(self.controller.dirty_keys(self.item_id))
        self._suppress = True
        for r, key in enumerate(self.keys):
            if key == _GEOMETRY_KEY:
                # The value cell holds a combo widget; tint it (not the hidden item).
                combo = self.table.cellWidget(r, 1)
                combo.setStyleSheet(
                    'background:yellow;color:black;' if key in dirty else '')
                continue
            cell = self.table.item(r, 1)
            if key in dirty:
                cell.setBackground(_DIRTY_BG)
                cell.setForeground(_DIRTY_FG)
            else:
                cell.setData(QtCore.Qt.ItemDataRole.BackgroundRole, None)
                cell.setData(QtCore.Qt.ItemDataRole.ForegroundRole, None)
        self._suppress = False

    def _refresh_pending(self) -> None:
        dirty = self.item_id is not None and self.controller.is_dirty()
        self.pending.setText('● changes pending — press Update' if dirty else '')
        # Undo is live whenever there are edits to discard OR a previously applied
        # state to step back to; otherwise there is nothing to undo.
        self.undo_button.setEnabled(
            self.item_id is not None and self.controller.can_undo())

    def _set_enabled(self, on: bool) -> None:
        self.table.setEnabled(on)
        self.update_button.setEnabled(on)
        self.undo_button.setEnabled(on)
        self.reset_button.setEnabled(on)
