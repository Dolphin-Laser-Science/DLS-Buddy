"""
gui/data_module.py
==================

The Data tab: confirm and manage the physical parameters of the selected
measurement. This is the home of the parser "confirmation step" (parse -> blanks
-> user fills -> commit) and the place where shared per-sample parameters are
entered once and propagated across the sample: `_on_cell_changed` routes shared
keys to `controller.set_shared_param`, and `_apply_edits_to_selection` fans an
edit across the highlighted measurements.

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

from PySide6 import QtCore, QtGui, QtWidgets

from core.workspace import _DLS_PARAM_KEYS, _SLS_PARAM_KEYS
from app.controller import _SHARED_PARAM_KEYS
from app import units as U
from analysis.uncertainty import format_value_at_uncertainty
from gui.help import section_header
from gui.theme import ThemedLabel, color as theme_color, token as theme_token
from gui.worker import busy_notice, runner


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
# Solvent name: an EDITABLE dropdown over the primary-tier library solvents (free
# text is preserved for non-library solvents + the existing alias/normalise path).
# Picking / editing it re-derives n and (for DLS) viscosity from the library.
_SOLVENT_KEY = 'solvent_name'
# The two auto-fillable, provenance-tagged fields, and their sidecar source keys
# (mirrors app.controller._SOLVENT_SOURCE_KEYS). A teal "library" dot is painted on
# these cells' DecorationRole; a direct edit claims the field as user-owned.
_SOLVENT_VALUE_KEYS = ('solvent_refractive_index', 'viscosity_Pa_s')
_SOLVENT_SOURCE_KEYS = {
    'solvent_refractive_index': 'solvent_refractive_index_source',
    'viscosity_Pa_s': 'viscosity_Pa_s_source',
}
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


class _EnterAdvanceDelegate(QtWidgets.QStyledItemDelegate):
    """Value-cell editor delegate that fires `enter_pressed` when the user commits a
    cell with Return/Enter, so the table can advance focus to the next field
    (feedback 2026-06-30 #6). The open editor is a child QLineEdit that consumes the
    key, so a table-level keyPressEvent wouldn't see it — hooking the editor's
    `returnPressed` is the reliable signal."""

    enter_pressed = QtCore.Signal()

    def createEditor(self, parent, option, index):
        editor = super().createEditor(parent, option, index)
        if isinstance(editor, QtWidgets.QLineEdit):
            editor.returnPressed.connect(self.enter_pressed)
        return editor


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
                '<b>Solvent library:</b> pick a solvent and the refractive index '
                '(and viscosity) auto-fill from temperature + wavelength, rounded '
                'at the last digit the library can stand behind.',
                'Library values re-derive when you change temperature or wavelength; '
                'a value you type by hand is never overwritten.',
                'A teal dot marks a library value; no dot means your own value. '
                'dn/dc is never proposed — always enter it yourself.',
                'Tip: press <b>Enter</b> in a value to jump to the next field.',
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
        # Enter in a Value cell commits and jumps to the next field (feedback #6). The
        # delegate (on the Value column only) reports the Return; we advance deferred so
        # the editor finishes committing through _on_cell_changed first.
        self._enter_delegate = _EnterAdvanceDelegate(self.table)
        self.table.setItemDelegateForColumn(1, self._enter_delegate)
        self._enter_delegate.enter_pressed.connect(
            lambda: QtCore.QTimer.singleShot(0, self._advance_to_next_field))
        self.table.itemChanged.connect(self._on_cell_changed)
        layout.addWidget(self.table, 1)
        # (The shared-vs-per-measurement note now lives in the section's "?" help,
        # so it is no longer repeated as an always-visible line — feedback B4.)

        # Provenance legend (the ONE place the dot's meaning + the never-overwrite
        # rule live as always-visible text; the per-cell tooltips carry only cell
        # facts). A plain QLabel renders the inline teal-dot HTML; it is re-rendered
        # on a theme switch by changeEvent (a stylesheet colour would freeze).
        self.legend = QtWidgets.QLabel()
        self.legend.setWordWrap(True)
        lf = self.legend.font()
        lf.setPointSize(max(1, lf.pointSize() - 1))       # smaller, no stylesheet
        self.legend.setFont(lf)
        layout.addWidget(self.legend)
        self._render_legend()
        # Non-blocking out-of-window note (engine declined to auto-fill / cleared a
        # stale library value): the user is never blocked — manual entry always works.
        self.autofill_note = ThemedLabel('', role='hint', size=11)
        self.autofill_note.setWordWrap(True)
        layout.addWidget(self.autofill_note)

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
        self.pending = ThemedLabel('', role='pending')
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
            self.autofill_note.setText('')
            return

        lm = self.controller.workspace.measurements[item_id]
        self.keys = _DLS_PARAM_KEYS if lm.kind == 'dls' else _SLS_PARAM_KEYS
        self._build_rows()
        self._populate()
        self.header.setText(
            f'<b>{item_id}</b> &mdash; {lm.kind.upper()} measurement '
            f'({self._raw_summary(lm)})')
        self._set_enabled(True)
        # A prior measurement's out-of-window hint must not linger on the next one
        # (autofill only re-runs on a solvent/T/λ edit, never on plain selection).
        self.autofill_note.setText('')
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
            # address it); for the geometry / solvent rows a combo widget is laid over
            # it (the underlying item is non-editable — the combo owns the value).
            value_item = QtWidgets.QTableWidgetItem('')
            if key in (_GEOMETRY_KEY, _SOLVENT_KEY):
                value_item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(r, 1, value_item)
            if key == _GEOMETRY_KEY:
                self._build_geometry_cell(r)
            elif key == _SOLVENT_KEY:
                self._build_solvent_cell(r)
            else:
                # Drop any cell widget left over from a previous measurement whose
                # key tuple put a combo at this row (setItem does NOT remove a cell
                # widget — it's a separate overlay layer). Without this, switching a
                # measurement type can leave a stale geometry/solvent combo here.
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
            '— = unspecified. The DDLS sub-tab pairs a VV with a VH '
            'at each angle to extract rotational diffusion. Leave — for ordinary '
            'polarised DLS.')
        combo.currentTextChanged.connect(self._on_geometry_changed)
        self.table.setCellWidget(row, 1, combo)

    def _build_solvent_cell(self, row: int) -> None:
        """Value column for the solvent row: an EDITABLE dropdown over the primary-tier
        library solvents. Editable keeps free-text entry for non-library solvents and
        the existing normalise/alias path; picking or typing one re-derives n/viscosity
        from the library (via the controller)."""
        combo = QtWidgets.QComboBox()
        combo.setEditable(True)
        combo.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        combo.addItem('')                                      # a blank/"none" option
        combo.addItems(self.controller.available_solvents('primary'))
        combo.setToolTip(
            'Solvent for this sample. Choosing a library solvent auto-fills the '
            'refractive index (and, for DLS, the viscosity) from the temperature and '
            'wavelength; type any name for a solvent not in the library.')
        # Dropdown pick fires textActivated; a hand-typed name fires editingFinished on
        # the line edit. Both route to the same handler; the handler no-ops when the
        # value is unchanged, so the (harmless) double-fire on a pick costs nothing.
        combo.textActivated.connect(self._on_solvent_changed)
        combo.lineEdit().editingFinished.connect(
            lambda c=combo: self._on_solvent_changed(c.currentText()))
        # Enter in the solvent field advances to the next field, matching every other
        # text value (the section help promises it). Deferred so _on_solvent_changed
        # commits + repopulates first (returnPressed fires before editingFinished).
        combo.lineEdit().returnPressed.connect(
            lambda: QtCore.QTimer.singleShot(0, self._advance_from_solvent))
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

    def _display_text(self, key: str, v, working: Optional[dict] = None) -> str:
        """The value as shown in the current unit (canonical -> display).

        A library-filled solvent cell (n / viscosity) is shown at the precision the
        library's per-condition σ supports, KEEPING the confident trailing zeros that
        rounding implies (1.3300, not 1.33); every other cell uses the plain clean
        format. Needs ``working`` to read the provenance tag + re-derive σ; without it
        the σ-aware branch is skipped (plain format)."""
        if v is None:
            return ''
        if (working is not None and key in _SOLVENT_VALUE_KEYS
                and working.get(_SOLVENT_SOURCE_KEYS[key]) == 'library:primary'
                and isinstance(v, (int, float))):
            sigma = self._solvent_sigma(key, working)
            if sigma is not None:
                if key in _QUANTITY:                    # viscosity: convert value + σ
                    # Converting the σ through from_canonical is valid only for a
                    # PURE-SCALE quantity (viscosity/concentration). The two library
                    # value keys are both scale-only; an affine quantity (temperature)
                    # must never join _SOLVENT_VALUE_KEYS or its offset would corrupt
                    # the chosen decimal place.
                    q = _QUANTITY[key]; unit = self._units[q]
                    return format_value_at_uncertainty(
                        U.from_canonical(q, float(v), unit),
                        U.from_canonical(q, float(sigma), unit))
                return format_value_at_uncertainty(float(v), sigma)   # n: dimensionless
        if key in _QUANTITY:
            return _fmt_num(U.from_canonical(_QUANTITY[key], float(v),
                                             self._units[_QUANTITY[key]]))
        if key in _FIXED_UNIT:
            return _fmt_num(float(v))
        # A leftover numeric (the dimensionless refractive index, now library-filled)
        # gets clean formatting too — otherwise str() shows 16-digit float noise.
        if isinstance(v, float):
            return _fmt_num(v)
        return str(v)

    def _solvent_sigma(self, key: str, working: dict) -> Optional[float]:
        """The canonical absolute per-condition σ for a library-filled n or viscosity
        cell (dimensionless for n, Pa·s for viscosity), or None if it can't be
        evaluated. Uses the SAME controller accessors the autofill rounded with
        (``solvent_value_n``/``solvent_value_eta`` — which turn the relative η σ into an
        absolute Pa·s ±), so the displayed precision matches the stored rounding exactly
        (Spec 3). The σ chooses the decimal place and travels no further (invariant #8)."""
        name = working.get(_SOLVENT_KEY)
        # A cleared solvent name can linger with a stale 'library:primary' tag (the
        # autofill early-returns without clearing it), so guard the falsy/non-string
        # case here rather than letting it reach the library (name.strip() would raise
        # AttributeError, which the except below does not catch) — return None so the
        # cell falls back to the plain format.
        if not isinstance(name, str) or not name.strip():
            return None
        temp_K = working.get('temperature_K')
        lam = working.get('wavelength_nm')
        try:
            if key == 'solvent_refractive_index':
                return self.controller.solvent_value_n(
                    name, float(lam), float(temp_K) - 273.15)[1]
            if key == 'viscosity_Pa_s':
                return self.controller.solvent_value_eta(
                    name, float(temp_K) - 273.15)[1]
        except (TypeError, ValueError, AttributeError):
            return None
        return None

    def _populate(self) -> None:
        self._suppress = True
        working = self.controller.workspace.measurements[self.item_id].working_params
        for r, key in enumerate(self.keys):
            if key == _GEOMETRY_KEY:
                combo = self.table.cellWidget(r, 1)
                v = working.get(key)
                combo.setCurrentText(v if v in _GEOMETRY_OPTIONS else _GEOMETRY_BLANK)
                continue
            if key == _SOLVENT_KEY:
                # setCurrentText on an editable combo does NOT emit textActivated /
                # editingFinished, so this never re-enters the handler.
                self.table.cellWidget(r, 1).setCurrentText(working.get(key) or '')
                continue
            self.table.item(r, 1).setText(
                self._display_text(key, working.get(key), working))
        self._suppress = False
        self._refresh_dirty()
        self._refresh_provenance()

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
        # A direct edit of n or viscosity claims that field as user-owned (sample-wide)
        # BEFORE the write, so library autofill never overwrites it. Only a real typed
        # value claims it; clearing the cell leaves the tag so a library value can still
        # re-derive on the next temperature/wavelength change.
        if key in _SOLVENT_VALUE_KEYS and value is not None:
            self.controller.mark_solvent_field_user(self.item_id, key)
        # Sample-shared params are entered once and propagate to every
        # measurement in the sample; per-measurement axes stay local.
        if key in _SHARED_PARAM_KEYS:
            self.controller.set_shared_param(self.item_id, key, value)
            # A temperature/wavelength change re-derives any library-sourced n/viscosity
            # at the new condition; repopulate so the freshly derived values show.
            # _populate already runs _refresh_dirty + _refresh_provenance, so return
            # after it rather than repeat both in the shared tail below.
            if key in ('temperature_K', 'wavelength_nm'):
                status = self.controller.reautofill_after(self.item_id, key)
                self._populate()
                self._apply_autofill_status(status)
                self._refresh_pending()
                return
        else:
            # The Mw fraction is a per-measurement axis like angle/concentration:
            # set it only on the edited row (and any other highlighted rows), with
            # no propagate-and-commit popup — applying it is now consistent with
            # every other per-measurement parameter (feedback 2026-06-26 #13).
            self.controller.set_param(self.item_id, key, value)
        self._refresh_dirty()
        self._refresh_provenance()
        self._refresh_pending()

    def _advance_to_next_field(self) -> None:
        """Move focus to the next editable Value cell after Enter (feedback #6). Skips to
        the next row whose Value is an editable item and opens its editor so the user can
        keep typing; if that row's Value is a widget (the geometry dropdown) it is focused
        instead. Stops at the last field (no wrap)."""
        start = self.table.currentRow()
        if start < 0:
            return
        for r in range(start + 1, self.table.rowCount()):
            item = self.table.item(r, 1)
            if item is not None and (item.flags() & QtCore.Qt.ItemFlag.ItemIsEditable):
                self.table.setCurrentCell(r, 1)
                self.table.editItem(item)
                return
            widget = self.table.cellWidget(r, 1)
            if widget is not None:                    # e.g. the geometry dropdown
                self.table.setCurrentCell(r, 1)
                widget.setFocus(QtCore.Qt.FocusReason.TabFocusReason)
                return

    def _advance_from_solvent(self) -> None:
        """Advance focus from the solvent combo to the next field (Enter-to-advance).
        The combo is a cell widget, so anchor the table's current cell on the solvent
        row first, then reuse the shared advance logic."""
        if _SOLVENT_KEY not in self.keys:
            return
        self.table.setCurrentCell(list(self.keys).index(_SOLVENT_KEY), 1)
        self._advance_to_next_field()

    def _on_geometry_changed(self, text: str) -> None:
        """Polarisation dropdown changed: store the per-measurement geometry."""
        if self._suppress or self.item_id is None:
            return
        value = None if text == _GEOMETRY_BLANK else text
        self.controller.set_param(self.item_id, _GEOMETRY_KEY, value)
        self._refresh_dirty()
        self._refresh_pending()

    def _on_solvent_changed(self, text: str) -> None:
        """Solvent combo picked / typed: store the shared solvent name, then re-derive
        n (and viscosity) from the library at the sample's temperature + wavelength."""
        if self._suppress or self.item_id is None:
            return
        value = text.strip() or None
        working = self.controller.workspace.measurements[self.item_id].working_params
        if working.get(_SOLVENT_KEY) == value:
            return                        # unchanged (e.g. editingFinished after a pick)
        self.controller.set_shared_param(self.item_id, _SOLVENT_KEY, value)
        status = self.controller.reautofill_after(self.item_id, _SOLVENT_KEY)
        self._populate()
        self._apply_autofill_status(status)
        self._refresh_pending()

    def _revert_cell(self, row: int, key: str) -> None:
        self._suppress = True
        working = self.controller.workspace.measurements[self.item_id].working_params
        self.table.item(row, 1).setText(
            self._display_text(key, working.get(key), working))
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
        # A background run reads COMMITTED params mid-flight — committing under
        # it would change the numbers it is computing from (invariant 4).
        if runner().is_busy:
            busy_notice(self.update_button)
            return
        self._apply_edits_to_selection()
        self.controller.commit()
        self._populate()           # working == committed now -> highlights clear
        self._refresh_pending()
        self.committed.emit()      # shell re-groups + refreshes the sidebar

    def _on_undo(self) -> None:
        if runner().is_busy:
            busy_notice(self.undo_button)
            return
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
            if key in (_GEOMETRY_KEY, _SOLVENT_KEY):
                # The value cell holds a combo widget; tint it (not the hidden item).
                combo = self.table.cellWidget(r, 1)
                if combo is not None:
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

    # ------------------------------------------------------ provenance dot ---
    def _refresh_provenance(self) -> None:
        """Paint the library-provenance dot on the n and viscosity cells.

        The dot rides the cell's DecorationRole, so it composes with the yellow
        BackgroundRole dirty tint (different item roles): a freshly auto-filled cell
        reads "teal dot + yellow bg = library value, pending commit". Only two states
        on the Data tab — teal (``library:primary``) or no dot (user / absent); the
        estimate-tier violet lives on the Solvent Explorer, not here. Every dot is
        backed by the cell tooltip (§2d) so colour is never the only signal
        (colourblind-safe). Bulk-grade dn/dT is deliberately NOT flagged here (Spec 1
        decision #5 — it is a docs-only caveat)."""
        if self.item_id is None:
            return
        working = self.controller.workspace.measurements[self.item_id].working_params
        self._suppress = True
        for r, key in enumerate(self.keys):
            if key not in _SOLVENT_VALUE_KEYS:
                continue
            cell = self.table.item(r, 1)
            if cell is None:
                continue
            src = working.get(_SOLVENT_SOURCE_KEYS[key])
            if src == 'library:primary':
                cell.setData(QtCore.Qt.ItemDataRole.DecorationRole,
                             self._dot_pixmap('lib_primary'))
                cell.setToolTip(self._lib_tooltip(key, working))
            else:
                cell.setData(QtCore.Qt.ItemDataRole.DecorationRole, None)
                cell.setToolTip('Your value — overrides the library.'
                                if working.get(key) is not None else '')
        self._suppress = False

    def _dot_pixmap(self, role: str) -> QtGui.QPixmap:
        """A small filled-circle pixmap in the theme colour for ``role`` (rebuilt on
        each paint so it re-themes on a light/dark switch)."""
        size = 10
        pm = QtGui.QPixmap(size, size)
        pm.fill(QtCore.Qt.GlobalColor.transparent)
        p = QtGui.QPainter(pm)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(theme_color(self, role))
        p.drawEllipse(1, 1, size - 2, size - 2)
        p.end()
        return pm

    def _lib_tooltip(self, key: str, working: dict) -> str:
        """Cell-specific facts for a library value: tier + this property's validity box
        + the PER-CONDITION uncertainty at this sample's committed T/λ (Spec 3 — the
        same σ the auto-fill rounding used, so the tooltip's ± always matches the
        cell's precision; falls back to the box-wide figure if the condition can't be
        evaluated). The shared behaviour (auto-fill / re-derive / never-overwrite) is
        stated once in the section ? help + the legend, not here. No citations
        (doc-rule #8) — 'sources: Advanced Guide'."""
        name = working.get(_SOLVENT_KEY)
        # A cleared solvent name can linger with a stale 'library:primary' tag, so a
        # falsy/non-string name would reach solvent_property_info -> normalize_solvent_name
        # -> name.strip() (AttributeError, NOT caught below). Guard it here.
        if not isinstance(name, str) or not name.strip():
            return 'Library value (primary).'
        try:
            info = self.controller.solvent_property_info(name)
        except (ValueError, TypeError):
            return 'Library value (primary).'
        temp_K = working.get('temperature_K')
        lam = working.get('wavelength_nm')
        if key == 'solvent_refractive_index':
            box = (f"valid {info['n_lambda_min_nm']:g}–{info['n_lambda_max_nm']:g} nm, "
                   f"{info['n_temp_min_C']:g}–{info['n_temp_max_C']:g} °C")
            unc = f"±{info['n_uncertainty']:g} (range max)"
            try:
                sig = self.controller.solvent_value_n(
                    name, float(lam), float(temp_K) - 273.15)[1]
                unc = f"±{sig:.2g} at this T/λ"
            except (TypeError, ValueError):
                pass
        else:
            box = (f"valid {info['eta_temp_min_C']:g}–{info['eta_temp_max_C']:g} °C")
            unc = f"±{info['eta_uncertainty_rel'] * 100:g}% (range max)"
            try:
                # solvent_value_eta returns the ABSOLUTE σ_η; the tooltip shows the
                # relative % (σ_abs / η), matching the box-wide eta_uncertainty_rel.
                eta_val, sig_abs = self.controller.solvent_value_eta(
                    name, float(temp_K) - 273.15)
                sig = sig_abs / eta_val
                unc = f"±{sig * 100:.2g}% at this T"
            except (TypeError, ValueError):
                pass
        return (f"Library value ({info['tier']}) · {box} · {unc} · "
                f"rounded at this ± (the uncertainty itself is never used in any "
                f"analysis) · sources: Advanced Guide.")

    def _render_legend(self) -> None:
        """(Re)render the provenance legend with the current theme's teal so the dot
        colour tracks a light/dark switch (a QLabel stylesheet colour would freeze)."""
        teal = theme_token(self, 'lib_primary')
        self.legend.setText(
            f'<span style="color:{teal}">●</span> library value (auto-filled)'
            ' &nbsp;·&nbsp; no dot = your value &nbsp;·&nbsp; '
            'a value you type is never overwritten')

    def _apply_autofill_status(self, status: Optional[dict]) -> None:
        """Show / clear the non-blocking out-of-window note from an autofill attempt.
        The engine has already declined to fill and cleared any stale library value;
        here we only surface the hint (manual entry always works)."""
        # Only surface fields that are actually parameters of THIS measurement kind:
        # viscosity is not an SLS field, so an out-of-range viscosity there must not
        # tell the user to hand-enter a value SLS never uses (the RI still filled).
        oor = [f for f in ((status or {}).get('out_of_range') or []) if f in self.keys]
        if not (oor and (status or {}).get('tier') == 'primary'):
            self.autofill_note.setText('')
            return
        solvent = self.controller.workspace.measurements[
            self.item_id].working_params.get(_SOLVENT_KEY)
        try:
            info = self.controller.solvent_property_info(solvent)
        except (ValueError, TypeError):
            self.autofill_note.setText('')
            return
        ranges = []
        if 'solvent_refractive_index' in oor:
            ranges.append(f"refractive index valid {info['n_lambda_min_nm']:g}–"
                          f"{info['n_lambda_max_nm']:g} nm, {info['n_temp_min_C']:g}–"
                          f"{info['n_temp_max_C']:g} °C")
        if 'viscosity_Pa_s' in oor and info.get('has_viscosity'):
            ranges.append(f"viscosity valid {info['eta_temp_min_C']:g}–"
                          f"{info['eta_temp_max_C']:g} °C")
        self.autofill_note.setText(
            f'Outside the library range for {solvent} '
            f'({"; ".join(ranges)}). Enter the value manually.')

    def changeEvent(self, ev: QtCore.QEvent) -> None:
        # A theme switch (app.setPalette) delivers PaletteChange: re-render the legend
        # dot and repaint the cell provenance dots in the new theme's colours (a
        # stylesheet colour would freeze — see gui.theme).
        # `hasattr` guards a PaletteChange that arrives before _build_ui finishes.
        if ev.type() == QtCore.QEvent.Type.PaletteChange and hasattr(self, 'legend'):
            self._render_legend()
            if self.item_id is not None:
                self._refresh_provenance()
        super().changeEvent(ev)

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
