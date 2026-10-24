"""
gui/data_module.py
==================

The Data tab: confirm and manage the physical parameters of the selected
measurement. This is the home of the parser "confirmation step" (parse -> blanks
-> user fills -> commit) and the place where shared per-sample parameters are
entered once and propagated across the sample: `_on_cell_changed` hands each edit
to `controller.edit_param`, and `_apply_edits_to_selection` carries the edits
across the highlighted measurements through `controller.apply_pending_edits`.

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

from core import units as U
from core.data_models import implausible_metadata_reasons
from core.naming import measurement_display_name, measurement_hover_card
from core.workspace import DLS_PARAM_KEYS, SLS_PARAM_KEYS, blank_labels
from core.formatting import format_fixed_sig, format_value_at_uncertainty
from gui.field_labels import CHAIN_TYPE_LABELS, PARAM_LABELS
from gui.help import section_header
from gui.solvent_picker import OTHER_DATA, index_of_solvent, populate_solvent_combo
from gui.theme import ThemedLabel, set_flag, color as theme_color, token as theme_token
from gui.widgets import (SourcePicker, confirm_regroup_losses,
                         sidebar_measurement_order)
from gui.worker import busy_notice, runner


# Internal field name -> user-facing label. Unit labels live in the dedicated Unit
# column, so no name carries a unit suffix. The map is shared (the DLS skipped-run
# warnings humanize field names from it too) — its single source of truth is
# gui/field_labels.py.
_PARAM_LABELS = PARAM_LABELS
_STRING_KEYS = {'label', 'polymer_name', 'solvent_name', 'mw_fraction'}
# Polarization/analyzer geometry: a dropdown, not a typed value. The blank label
# means "unspecified" (None) -- ordinary polarized DLS, ignored by the depolarized
# analysis. Only VV + VH correlograms pair into D_r; VU is record-only.
_GEOMETRY_KEY = 'analyzer_geometry'
_GEOMETRY_BLANK = '—'
_GEOMETRY_OPTIONS = (_GEOMETRY_BLANK, 'VV', 'VH', 'VU')
# Chain type: a sample-level architecture label (Linear reference / Branched target)
# for the Cross-Sample branching contraction factors g and h. A fixed-vocabulary combo
# like geometry, but SHARED across the sample's measurements like the solvent name
# (entered once, propagated) -- so it is committed via set_shared_param and rides the
# normal Update / dirty-highlight / Reset machinery.
_CHAIN_TYPE_KEY = 'chain_type'
_CHAIN_TYPE_BLANK = '—'
_CHAIN_TYPE_TO_DISPLAY = CHAIN_TYPE_LABELS  # stored -> display
_CHAIN_TYPE_OPTIONS = (_CHAIN_TYPE_BLANK, *CHAIN_TYPE_LABELS.values())
_CHAIN_TYPE_TO_STORED = {shown: stored for stored, shown in CHAIN_TYPE_LABELS.items()}
# Solvent blank: which c = 0 series THIS SLS series is referenced against. A dropdown
# over the sample's blanks (stored as the blank's item_id, never a filename) whose
# first entry is the automatic pairing -- the blank from this series' own source file,
# else the sample reference (Workspace.solvent_reference_for). Per-measurement like
# geometry; the row is hidden on a blank itself (and is an SLS key only). How the pairing was
# made is computed on demand (Controller.solvent_blank_status) and shown in the combo's tooltip.
_BLANK_KEY = 'solvent_blank_item_id'
_BLANK_AUTO = 'auto'
# The typed stand-in for a solvent blank that was never measured: the solvent's intensity
# AT 90 DEGREES, in the loaded file's own units. Greyed out whenever the sample holds a
# measured blank (which always wins), so the field can never look like it is in use when
# it is not. A sample-shared SLS key, entered once, like dn/dc.
_BLANK_I90_KEY = 'solvent_blank_intensity_90deg'
# Solvent name: a CLOSED dropdown over all library solvents (gui/solvent_picker.py)
# plus an "Other…" entry that reveals a text box for a solvent the library does not
# carry. Picking one re-derives n and (for DLS) viscosity from the library, as does a library
# name typed through Other… (matched ignoring case and surrounding spaces); any other typed
# name auto-fills nothing.
_SOLVENT_KEY = 'solvent_name'
# Every row whose value cell is a combo widget (tinted, not the hidden item).
_COMBO_KEYS = (_GEOMETRY_KEY, _CHAIN_TYPE_KEY, _SOLVENT_KEY, _BLANK_KEY)
# The two auto-fillable, provenance-tagged fields, and their sidecar source keys
# (mirrors app.controller._SOLVENT_SOURCE_KEYS). A teal "library" dot is painted on
# these cells' DecorationRole; a direct edit claims the field as user-owned.
_SOLVENT_VALUE_KEYS = ('solvent_refractive_index', 'viscosity_Pa_s')
_SOLVENT_SOURCE_KEYS = {
    'solvent_refractive_index': 'solvent_refractive_index_source',
    'viscosity_Pa_s': 'viscosity_Pa_s_source',
}
# The mean count rate and the sidecar recording where it came from (mirrors
# app.controller._MEAN_CR_KEY / _MEAN_CR_SOURCE_KEY): 'trace:*', 'vendor', 'generator'
# (a synthetic draw's own rate), 'replicate:mean' (a replicate average's) or 'user'.
_MEAN_CR_KEY = 'mean_count_rate_cps'
_MEAN_CR_SOURCE_KEY = 'mean_count_rate_cps_source'
# Numeric params the user may enter in a choice of units (canonical value stored).
# Maps the param key -> the units-module quantity name.
_QUANTITY = {
    'concentration_g_per_mL': 'concentration',
    'temperature_K': 'temperature',
    'viscosity_Pa_s': 'viscosity',
    # A unit picker rather than a fixed cps cell: the vendor scalar in an ALV file
    # is printed in kHz, and forcing a hand conversion is exactly the unit-slip
    # class the plausibility note exists to catch.
    'mean_count_rate_cps': 'intensity',
}

# Every cell that carries a provenance dot: param key -> the method that returns
# (dot token or None, tooltip). Data rather than a chain of branches, because the
# trace family will add more of these and a per-field `elif` inside the paint loop
# is how two of them end up disagreeing about the no-dot case.
_PROVENANCE_FIELDS = {
    'solvent_refractive_index': '_solvent_dot',
    'viscosity_Pa_s': '_solvent_dot',
    'mean_count_rate_cps': '_count_rate_dot',
}
# Numeric params with a single fixed unit (already human-scale): shown, not chosen.
_FIXED_UNIT = {
    'angle_deg': '°',
    'wavelength_nm': 'nm',
    'dn_dc_mL_per_g': 'mL/g',
}

# Passive tooltips on specific parameter-row LABELS (calculation nuance, honouring the
# global "Show tooltips" gate at render). Kept terse; how-to lives in the group ? badge.
_ROW_TOOLTIPS = {
    'label': (
        'A short name of your own, shown wherever the measurement appears and added to '
        'its sample and angle rather than replacing them, so “3” is enough. Left blank, '
        'the file name is used without its extension; the label changes no grouping '
        'and no analysis.'),
    'solvent_blank_item_id': (
        'The c = 0 solvent measurement subtracted from this series to form its excess '
        'Rayleigh ratio (Theory-and-Equations-Guide §1.7). “auto” takes the blank from '
        'this series\' own file, else the sample\'s reference blank (the first loaded); '
        'the dropdown\'s tooltip names the blank in use and why.'),
    'mean_count_rate_cps': (
        'Average photon count rate of this run, in the unit beside it. The nonlinear '
        'cumulant and distribution fits read it in their default residual weight '
        '(Theory-and-Equations-Guide §3.2.9), and Derived Intensity builds static '
        'points from it.'),
    'solvent_blank_intensity_90deg': (
        'The pure solvent\'s reading at 90°, in the loaded intensity column\'s units, '
        'before dark-count subtraction; used only when the sample has no measured '
        'c = 0 blank, and empty or zero subtracts no solvent. It is scaled by 1/sin θ '
        'at each angle, not subtracted uniformly, and every result on it is flagged '
        'unreliable, Rg included (Theory-and-Equations-Guide §1.7.2).'),
    'dn_dc_mL_per_g': (
        'Refractive-index increment of this solute in this solvent; the program never '
        'computes or looks one up (a file that records one fills it), since the optical '
        'constant goes as its square and an error enters Mw squared. Blank puts only Mw and the absolute A₂ on a flagged '
        'arbitrary scale; zero is refused, a negative value accepted '
        '(Theory-and-Equations-Guide §1.5).'),
    'solute_refractive_index': (
        'Refractive index of the particle or solute itself, not the solvent\'s and not '
        'dn/dc; optional, never proposed. It sets the contrast m = solute / solvent '
        'index for the DLS RDG and Mie weightings and the SLS Mie fit, and is the core '
        'index of the core–shell fit (Theory-and-Equations-Guide §3.7, §4.10 and §4.11).'),
    'shell_refractive_index': (
        'Refractive index of the shell of a core–shell (coated) particle; optional, never '
        'proposed. Only the SLS core–shell fit reads it, pairing it with the solute '
        '(core) index into the contrasts m1 = core / solvent and m2 = shell / solvent '
        '(Theory-and-Equations-Guide §4.11).'),
}


def _fmt_num(x: float) -> str:
    """Clean numeric display: no float-noise tail (0.00089, not 0.000890000…1)."""
    return f'{x:.10g}'
# A changed-but-uncommitted cell is flagged with a calm, theme-aware tint from the
# `edited_bg`/`edited_fg` tokens (never a raw `Qt.GlobalColor.yellow`, which freezes on a
# theme switch and fails contrast in dark mode — style guide §1 R1.4, §2 R2.4). The tint
# is resolved at paint time in `_refresh_dirty`; the meaning is ALSO carried by a
# non-colour cue (italic cell text) so it survives colour-blindness (§10 R10.1).


class _EnterAdvanceDelegate(QtWidgets.QStyledItemDelegate):
    """Value-cell editor delegate that fires `enter_pressed` when the user commits a
    cell with Return/Enter, so the table can advance focus to the next field.
    The open editor is a child QLineEdit that consumes the
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
        # The solvent row's (closed combo, Other text box) pair, and whether the user
        # has chosen Other but not yet typed a name. Rebuilt per measurement in
        # _build_solvent_cell; None while no solvent row exists (an SLS/DLS key set
        # always has one, but set_measurement(None) clears the table).
        self._solvent_widgets: Optional[Tuple[QtWidgets.QComboBox,
                                              QtWidgets.QLineEdit]] = None
        self._solvent_other_mode = False
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
            'Physical Parameters — edit, then Update to apply',
            'Which parameters each analysis needs:',
            bullets=[
                '<b>All:</b> solute and solvent name, temperature, wavelength, '
                'refractive index.',
                '<b>DLS:</b> + viscosity and scattering angle. <b>SLS:</b> + '
                'concentration, a calibration and dn/dc, which can stay blank '
                '(User-Manual §6.5).',
                'Identity and optics are shared across the sample; concentration and '
                'angle are per measurement.',
                'With several measurements highlighted, one typed <b>Label</b> is '
                'numbered in sidebar order (<b>batch 1</b>, <b>batch 2</b> …, '
                'zero-padded past nine); other edits apply to all.',
                '<b>Mean count rate</b> (DLS): applied at load when the file carries '
                'one; <b>From…</b> offers the other sources, for this run only.',
                '<b>Enter</b> in a value jumps to the next field.',
                'Each row, the solvent library and a missing blank: User-Manual §6.1–6.6.',
            ]))
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(['Parameter', 'Value', 'Unit'])
        self.table.verticalHeader().setVisible(False)
        hdr = self.table.horizontalHeader()
        # All three columns are user-draggable. The Value column is
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
        # Single click into a Value cell starts editing — only the
        # Value column is editable, so navigating Parameter/Unit cells is harmless.
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.AllEditTriggers)
        # Enter in a Value cell commits and jumps to the next field. The
        # delegate (on the Value column only) reports the Return; we advance deferred so
        # the editor finishes committing through _on_cell_changed first.
        self._enter_delegate = _EnterAdvanceDelegate(self.table)
        self.table.setItemDelegateForColumn(1, self._enter_delegate)
        self._enter_delegate.enter_pressed.connect(
            lambda: QtCore.QTimer.singleShot(0, self._advance_to_next_field))
        self.table.itemChanged.connect(self._on_cell_changed)
        layout.addWidget(self.table, 1)
        # (The shared-vs-per-measurement note now lives in the section's "?" help,
        # so it is no longer repeated as an always-visible line.)

        # Provenance legend (the ONE place the dot's meaning + the never-overwrite
        # rule live as always-visible text; the per-cell tooltips carry only cell
        # facts). A plain QLabel renders the inline teal-dot HTML; it is re-rendered
        # on a theme switch by changeEvent (a stylesheet color would freeze).
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
        # Passive plausibility note (a likely unit slip in the committed T / n): a
        # neutral steel ⓘ qualifier, not an alarm — the value is user-owned and used
        # as entered (invariant 3). Shares its band with the load-time warning via
        # core.data_models.implausible_metadata_reasons so the two never drift.
        self.plausibility_note = ThemedLabel('', size=11)
        self.plausibility_note.setWordWrap(True)
        layout.addWidget(self.plausibility_note)
        # Committed mean count rate vs the linked trace's own mean. Its own label
        # rather than a share of plausibility_note's: a unit slip in T/n is a
        # different, safety-relevant fact that must not be crowded out by this one.
        # Refreshed from _refresh_pending (not from an edit event), because it is a
        # standing fact about the measurement, not something that happened.
        self.trace_divergence_note = ThemedLabel('', size=11)
        self.trace_divergence_note.setWordWrap(True)
        layout.addWidget(self.trace_divergence_note)
        # The mean count rate is the one parameter more than one place can propose,
        # so it gets the shared offer affordance rather than a bespoke control. It
        # sits beside the divergence note because the two answer the same question
        # ("where did this number come from, and does it agree?"), and beneath the
        # table because it fills a cell rather than being one.
        cr_row = QtWidgets.QHBoxLayout()
        cr_row.setContentsMargins(0, 0, 0, 0)
        self.count_rate_source_label = ThemedLabel('', size=11)
        self.count_rate_picker = SourcePicker(
            'From…', title='Mean count rate',
            prompt='Fill the mean count rate from:')
        self.count_rate_picker.picked.connect(self._on_count_rate_source_picked)
        cr_row.addWidget(self.count_rate_source_label)
        cr_row.addWidget(self.count_rate_picker)
        cr_row.addStretch(1)
        self._count_rate_row = cr_row
        layout.addLayout(cr_row)
        row = QtWidgets.QHBoxLayout()
        self.update_button = QtWidgets.QPushButton('Update')
        self.update_button.setToolTip(
            'Update the edited parameters to the sample. If the change regroups '
            'samples, stored results follow; you are asked first if a merge '
            'would drop a hand-entered or picked value.')
        self.update_button.clicked.connect(self._on_update)
        self.undo_button = QtWidgets.QPushButton('Undo')
        self.undo_button.setToolTip(
            'Step back to the previously applied parameters. With un-applied edits '
            'showing, the first Undo discards those edits.')
        self.undo_button.clicked.connect(self._on_undo)
        self.reset_button = QtWidgets.QPushButton('Reset')
        self.reset_button.setToolTip(
            'Clear every entered parameter of the highlighted measurements (a sample '
            'heading highlights all of them), keeping the scattering angle; nothing is '
            'applied until Update, and Undo restores them.')
        self.reset_button.clicked.connect(self._on_reset)
        row.addWidget(self.update_button)
        row.addWidget(self.undo_button)
        row.addWidget(self.reset_button)
        self.pending = ThemedLabel('', role='pending')
        row.addWidget(self.pending, 1)
        layout.addLayout(row)

    # ---------------------------------------------------------- selection ---
    def set_selected_ids(self, item_ids) -> None:
        """Record which measurements are highlighted in the sidebar.
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
            # say it) — just point at the next step.
            self.header.setText(
                'No measurement selected — pick one in the Workspace list.')
            # The identity card goes with the measurement: a header that says nothing is
            # selected must not name a run on hover (it outlived a removal).
            self.header.setToolTip('')
            self._set_enabled(False)
            self.pending.setText('')
            self.autofill_note.setText('')
            self.plausibility_note.setText('')
            self.trace_divergence_note.setText('')
            # The provenance legend is data-dependent chrome (R8.2): hide it on an empty
            # table so an unpopulated tab doesn't explain a dot that isn't there.
            self.legend.hide()
            return

        lm = self.controller.workspace.measurements[item_id]
        self.keys = DLS_PARAM_KEYS if lm.kind == 'dls' else SLS_PARAM_KEYS
        self.legend.show()
        self._build_rows()
        self._populate()
        # Named by the one builder, not by the internal id. The tab's own table
        # shows every parameter, so the header's hover card is the identity card
        # every other surface shows -- the id among its lines.
        self.header.setText(
            f'<b>{measurement_display_name(lm)}</b> &mdash; '
            f'{lm.kind.upper()} measurement ({self._raw_summary(lm)})')
        self.header.setToolTip(measurement_hover_card(lm))
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
            if key in _ROW_TOOLTIPS:
                name.setToolTip(_ROW_TOOLTIPS[key])
            self.table.setItem(r, 0, name)
            # An empty, non-editable value item always exists (so _refresh_dirty can
            # address it); for the geometry / solvent rows a combo widget is laid over
            # it (the underlying item is non-editable — the combo owns the value).
            value_item = QtWidgets.QTableWidgetItem('')
            if key in _COMBO_KEYS:
                value_item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(r, 1, value_item)
            if key == _GEOMETRY_KEY:
                self._build_geometry_cell(r)
            elif key == _CHAIN_TYPE_KEY:
                self._build_chain_type_cell(r)
            elif key == _SOLVENT_KEY:
                self._build_solvent_cell(r)
            elif key == _BLANK_KEY:
                self._build_blank_cell(r)
            else:
                # Drop any cell widget left over from a previous measurement whose
                # key tuple put a combo at this row (setItem does NOT remove a cell
                # widget — it's a separate overlay layer). Without this, switching a
                # measurement type can leave a stale geometry/solvent combo here.
                self.table.removeCellWidget(r, 1)
            self._build_unit_cell(r, key)
        self._suppress = False

    def _build_geometry_cell(self, row: int) -> None:
        """Value column for the polarization row: a VV/VH/VU/— dropdown."""
        combo = QtWidgets.QComboBox()
        combo.addItems(_GEOMETRY_OPTIONS)
        combo.setToolTip(
            'Polarization geometry of this correlogram, as (incident, analyzer): '
            'VV = polarized, VH = depolarized, VU = no analyzer, — = unspecified, '
            'as for ordinary polarized DLS (Theory-and-Equations-Guide §1.6.1). The '
            'DDLS sub-tab pairs a VV with a VH at each angle to extract rotational '
            'diffusion.')
        combo.currentTextChanged.connect(self._on_geometry_changed)
        self.table.setCellWidget(row, 1, combo)

    def _build_chain_type_cell(self, row: int) -> None:
        """Value column for the chain-type row: a Linear / Branched / — dropdown. A
        sample-level architecture tag; propagated sample-wide (set_shared_param)."""
        combo = QtWidgets.QComboBox()
        combo.addItems(_CHAIN_TYPE_OPTIONS)
        combo.setToolTip(
            'Chain architecture of this sample, for the Cross-Sample branching '
            'contraction factors: Linear = a reference that defines the linear Rg–Mw '
            'and Rh–Mw laws; Branched = a target that gets g (from Rg) and/or h (from '
            'Rh); — = untagged. Shared across the sample’s measurements '
            '(Theory-and-Equations-Guide §5.3–5.4).')
        combo.currentTextChanged.connect(self._on_chain_type_changed)
        self.table.setCellWidget(row, 1, combo)

    def _build_solvent_cell(self, row: int) -> None:
        """Value column for the solvent row: a CLOSED dropdown over all 19 library
        solvents (grouped into measured/estimated sections by gui.solvent_picker), plus
        an ``Other…`` entry that reveals a text box beside it.

        Closed rather than editable, because that is what lets the rest of the program
        store a solvent name verbatim: a picked value is canonical by construction, so
        nothing downstream has to guess whether a typed "H2O" meant water. Free text is
        still available — it is just an explicit choice now, and a name entered that way
        is its own sample and gets no library n/η (the user supplies both by hand).

        The combo and the text box live in one container widget so the table cell can
        hold both; ``_solvent_widgets`` gives the refresh path a handle on the pair.
        """
        combo = QtWidgets.QComboBox()
        populate_solvent_combo(combo, include_blank=True, include_other=True)
        combo.setToolTip(
            'Solvent of this sample, part of its identity. A solvent under Measured '
            'properties fills the refractive index (and, for DLS, the viscosity) from '
            'the temperature and wavelength; one under Estimated properties, or a name '
            'typed through Other… that the library does not carry, fills nothing '
            '(User-Manual §6.3).')
        combo.activated.connect(self._on_solvent_picked)

        other = QtWidgets.QLineEdit()
        other.setPlaceholderText('Solvent name')
        other.setToolTip(
            'Name of a solvent the library does not carry, stored exactly as typed. '
            'Measurements match into one sample by it, ignoring only capitalization '
            'and surrounding spaces, so H2O and water are two samples.')
        other.hide()
        other.editingFinished.connect(self._on_solvent_typed)
        # Enter in the solvent field advances to the next field, matching every other
        # text value (the section help promises it). Deferred so _on_solvent_typed
        # commits + repopulates first (returnPressed fires before editingFinished).
        other.returnPressed.connect(
            lambda: QtCore.QTimer.singleShot(0, self._advance_from_solvent))

        holder = QtWidgets.QWidget()
        box = QtWidgets.QHBoxLayout(holder)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(4)
        box.addWidget(combo, 1)
        box.addWidget(other, 1)
        # Enter-to-advance walks cell widgets and focuses them; the proxy sends that
        # focus to the combo rather than the bare container.
        holder.setFocusProxy(combo)
        self._solvent_widgets = (combo, other)
        self._solvent_other_mode = False
        self.table.setCellWidget(row, 1, holder)

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
            # Muted secondary text from a theme token, NOT a raw Qt.GlobalColor.gray
            # (which freezes on a theme switch — see the module header + style guide §1
            # R1.1/R1.2). Re-resolved on a PaletteChange by _refresh_unit_colors.
            item.setForeground(theme_color(self, 'muted'))
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
        # A mean count rate the PROGRAM derived is a computed quantity with no ±, so
        # it follows the house rule for those: the Settings no-uncertainty
        # significant figures, not the raw 10-digit float. A value the user typed is
        # echoed exactly as entered, like every other hand-entered cell.
        if (key == _MEAN_CR_KEY and working is not None
                and isinstance(v, (int, float))
                and (working.get(_MEAN_CR_SOURCE_KEY) or '') not in ('', 'user')):
            q = _QUANTITY[key]
            return format_fixed_sig(
                U.from_canonical(q, float(v), self._units[q]),
                self.controller.settings.no_uncertainty_sig_figs)
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
        (this is the solvent library's per-condition display uncertainty). The σ chooses
        the decimal place and travels no further — it never enters the analysis."""
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
            if key == _CHAIN_TYPE_KEY:
                combo = self.table.cellWidget(r, 1)
                combo.setCurrentText(
                    _CHAIN_TYPE_TO_DISPLAY.get(working.get(key), _CHAIN_TYPE_BLANK))
                continue
            if key == _SOLVENT_KEY:
                self._show_solvent(working.get(key))
                continue
            if key == _BLANK_KEY:
                self._populate_blank_cell(r, working)
                continue
            self.table.item(r, 1).setText(
                self._display_text(key, working.get(key), working))
            if key == _BLANK_I90_KEY:
                self._populate_blank_i90_cell(r, working)
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
        # dn/dc = 0 is refused by SLSMeasurement (it collapses the Kc/ΔR ordinate), and
        # catching it here rather than at Run saves a commit-then-fail round trip. The
        # RULE lives on the model, which still enforces it for every other entry path;
        # this is only the early, friendlier surface for the one route a user types on.
        if key == 'dn_dc_mL_per_g' and value == 0:
            QtWidgets.QMessageBox.warning(
                self, 'dn/dc cannot be zero',
                'A dn/dc of zero would make the optical constant K zero and collapse '
                'the whole Kc/ΔR ordinate, so no molecular weight could come from it.'
                '\n\nIf you do not have a value, leave the cell BLANK instead. Every '
                'analysis still runs; Rg and the form-factor sizes are exact, and '
                'only Mw and the absolute A₂ are flagged as being on an arbitrary '
                'optical scale.')
            self._revert_cell(item.row(), key)
            return
        # The typed 90° solvent blank, judged by the controller's ONE predicate rather
        # than a second one here (rule 12: the boundary that refuses at commit and the
        # boundary that refuses at construction must be the same computation). This is a
        # courtesy only -- the binding guard runs where the blank is built, because the
        # dark count it is compared against is a calibration field the user can edit on
        # the SLS tab afterwards. Same shape as the dn/dc = 0 catch above.
        if key == _BLANK_I90_KEY and value is not None:
            sid = self.controller.workspace.sample_id_of(self.item_id)
            problem = self.controller.validate_typed_blank(sid, value) if sid else ''
            if problem:
                QtWidgets.QMessageBox.warning(
                    self, 'Solvent blank intensity cannot be used', problem)
                self._revert_cell(item.row(), key)
                return
        # The edit rules (claim a typed n or viscosity, clear a library one, write a
        # shared key sample-wide, re-derive after a condition change) are the
        # controller's: `Controller.edit_param`.
        status = self.controller.edit_param(self.item_id, key, value)
        if status is not None:
            # The library n and viscosity were re-derived at the new condition;
            # repopulate so the fresh values show. _populate already runs
            # _refresh_dirty + _refresh_provenance, so return after it.
            self._populate()
            self._apply_autofill_status(status)
            self._refresh_pending()
            return
        self._refresh_dirty()
        self._refresh_provenance()
        self._refresh_pending()

    def _advance_to_next_field(self) -> None:
        """Move focus to the next editable Value cell after Enter. Skips to
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

    def _build_blank_cell(self, row: int) -> None:
        """Value column for the solvent-blank row: an auto / <blank> dropdown.
        Its entries depend on the sample, so they are filled in _populate."""
        combo = QtWidgets.QComboBox()
        combo.currentIndexChanged.connect(self._on_blank_changed)
        self.table.setCellWidget(row, 1, combo)

    def _populate_blank_cell(self, row: int, working: dict) -> None:
        """Fill the solvent-blank dropdown from the sample's blanks and show which
        one is in use. Hidden on a blank itself (it references nothing) and when the
        sample has no blank at all (nothing to choose)."""
        combo = self.table.cellWidget(row, 1)
        if combo is None:
            return
        conc = working.get('concentration_g_per_mL')
        sid = self.controller.workspace.sample_id_of(self.item_id)
        blanks = self.controller.workspace.blank_candidates(sid) if sid else []
        hide = (conc is not None and conc == 0) or not blanks
        self.table.setRowHidden(row, hide)
        if hide:
            return
        combo.clear()
        combo.addItem(_BLANK_AUTO, None)
        labels = blank_labels(blanks)
        for b in blanks:
            # The same folder/file name the tooltip and the export use; a label that
            # already ends in the id (blanks only the id tells apart) is not suffixed twice.
            label, suffix = labels[b.item_id], f'[{b.item_id}]'
            combo.addItem(label if label.endswith(suffix) else f'{label}  {suffix}',
                          b.item_id)
        wanted = working.get(_BLANK_KEY)
        idx = combo.findData(wanted) if wanted else 0
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._refresh_blank_tooltip(combo)

    def _typed_blank_is_inert(self) -> bool:
        """Is the typed-90° blank field greyed out -- i.e. does this sample hold a
        MEASURED blank, which always wins? One predicate, read by the cell painter (for
        the edit flag and the tooltip) and by `_refresh_dirty` (for the muted brush), so
        a greyed cell can never be painted as live, or the reverse."""
        if self.item_id is None:
            return False
        sid = self.controller.workspace.sample_id_of(self.item_id)
        return bool(sid) and self.controller.sample_has_measured_blank(sid)

    def _populate_blank_i90_cell(self, row: int, working: dict) -> None:
        """Grey the typed-90°-blank cell whenever the sample holds a MEASURED blank.

        A measured blank always wins (`Workspace.solvent_reference_for` is the normative
        rule and is not edited by this feature), so an editable cell here would imply a
        choice the user does not have. A value typed earlier is KEPT and still shown --
        inert, and the tooltip says so -- because removing the sample's last blank makes
        it the blank in use again with no further gesture.
        """
        item = self.table.item(row, 1)
        if item is None:
            return
        has_blank = self._typed_blank_is_inert()
        sid = self.controller.workspace.sample_id_of(self.item_id)
        flags = item.flags()
        if has_blank:
            item.setFlags(flags & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        else:
            item.setFlags(flags | QtCore.Qt.ItemFlag.ItemIsEditable)
        # The muted BRUSH is not set here: `_refresh_dirty` clears ForegroundRole on
        # every clean cell and runs after this, so a brush painted here would be wiped.
        # It owns the non-dirty foreground, so it paints this one too -- one owner.
        if has_blank:
            stored = working.get(_BLANK_I90_KEY)
            tip = ('This sample has a measured c = 0 solvent blank, which always wins, '
                   'so this field is not in use.')
            if stored:
                tip += (' The value shown is kept but inert; it would become the blank '
                        "again only if the sample's last measured blank were removed.")
            item.setToolTip(tip)
        else:
            problem = self.controller.validate_typed_blank(sid) if sid else ''
            item.setToolTip(
                problem or 'No measured blank in this sample: this value will be used.')

    def _refresh_blank_tooltip(self, combo) -> None:
        """Describe the WORKING selection (what is picked, committed or not), so
        the tooltip can never contradict the index beside it."""
        status = self.controller.solvent_blank_status(self.item_id, use_working=True)
        how = {'user': 'your choice',
               'auto:same-file': 'automatic — the blank from this series\' own file',
               'auto:sample': 'automatic — the sample reference blank'}.get(
                   status['source'], status['source'])
        tip = (f'In use: {status["label"] or "none"} ({how}).'
               if status['item_id'] else 'No solvent blank in this sample.')
        if status['note']:
            tip += f' Note: {status["note"]}.'
        combo.setToolTip(tip)

    def _on_blank_changed(self, index: int) -> None:
        """Solvent-blank dropdown changed: store the chosen blank's item_id
        (None = automatic pairing) on this series only."""
        if self._suppress or self.item_id is None:
            return
        combo = self.sender()
        value = combo.itemData(index) if combo is not None else None
        self.controller.set_param(self.item_id, _BLANK_KEY, value)
        if combo is not None:
            self._refresh_blank_tooltip(combo)
        self._refresh_dirty()
        self._refresh_pending()

    def _on_geometry_changed(self, text: str) -> None:
        """Polarization dropdown changed: store the per-measurement geometry."""
        if self._suppress or self.item_id is None:
            return
        value = None if text == _GEOMETRY_BLANK else text
        self.controller.set_param(self.item_id, _GEOMETRY_KEY, value)
        self._refresh_dirty()
        self._refresh_pending()

    def _on_chain_type_changed(self, text: str) -> None:
        """Chain-type dropdown changed: store the sample-shared architecture tag
        (propagated across the sample's measurements, like the solvent name)."""
        if self._suppress or self.item_id is None:
            return
        value = _CHAIN_TYPE_TO_STORED.get(text)          # '—' -> None
        self.controller.set_shared_param(self.item_id, _CHAIN_TYPE_KEY, value)
        self._refresh_dirty()
        self._refresh_pending()

    def _show_solvent(self, name) -> None:
        """Display the stored solvent `name` in the closed combo + Other text box.

        A name the library carries selects its row (matched caselessly, so a Zetasizer
        file's ``Water`` finds ``water``). Anything else — a name typed here, or one an
        instrument recorded — shows as *Other* with the string in the text box, so a
        free-text solvent survives a round trip through the table instead of being
        silently dropped on the next refresh. ``_solvent_other_mode`` covers the moment
        between choosing *Other* and typing anything into it, when there is no stored
        name to infer the state from.
        """
        combo, other = self._solvent_widgets
        # Hiding a focused QLineEdit emits editingFinished, which would re-enter the
        # commit path; suppress for the duration. Saved/restored rather than set to
        # False, because _populate() calls this inside its own suppressed block.
        was_suppressed = self._suppress
        self._suppress = True
        try:
            self._show_solvent_unguarded(combo, other, name)
        finally:
            self._suppress = was_suppressed

    def _show_solvent_unguarded(self, combo, other, name) -> None:
        i = index_of_solvent(combo, name)
        free_text = isinstance(name, str) and name.strip() and i < 0
        if self._solvent_other_mode or free_text:
            combo.setCurrentIndex(combo.findData(OTHER_DATA))
            other.setText(name if free_text else '')
            other.show()
        else:
            combo.setCurrentIndex(i if i >= 0 else 0)      # 0 = the blank "not set" row
            other.clear()
            other.hide()

    def select_solvent(self, name) -> None:
        """Choose `name` as if the user had worked the solvent control.

        A library solvent goes through the dropdown; anything else goes through the
        Other text box; a blank clears the field. The public entry point for driving
        this control from outside the widget (the screenshot harness and the GUI
        tests), so they exercise the same routing a click does instead of a private
        handler that could drift away from it.
        """
        if self.item_id is None:
            return
        combo, _other = self._solvent_widgets
        i = index_of_solvent(combo, name)
        self._solvent_other_mode = False
        self._commit_solvent(combo.itemData(i) if i >= 0 else name)

    def _on_solvent_picked(self, index: int) -> None:
        """A row of the closed solvent dropdown was activated by the user.

        ``activated`` fires on user action only, never on a programmatic
        ``setCurrentIndex``, so the refresh path cannot re-enter this.
        """
        if self._suppress or self.item_id is None:
            return
        combo, other = self._solvent_widgets
        if combo.itemData(index) == OTHER_DATA:
            # Nothing to commit yet — the name is whatever they are about to type.
            # Deliberately does NOT clear the stored solvent: abandoning the text box
            # without typing leaves the previous value intact rather than wiping it.
            self._solvent_other_mode = True
            other.clear()
            other.show()
            other.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
            return
        self._solvent_other_mode = False
        self._commit_solvent(combo.itemData(index))

    def _on_solvent_typed(self) -> None:
        """The Other text box lost focus / was submitted: commit what was typed.

        An empty box has TWO meanings and only one of them clears the solvent:

        * **never typed** — the user chose Other… and then clicked away or tabbed off
          without entering anything. ``editingFinished`` fires on any focus-out, typed
          or not, so this arrives here looking identical to a deliberate clear. It must
          leave the stored name alone: committing '' would blank the solvent the user
          never asked to change, and take its library-derived n and viscosity with it.
          ``_solvent_other_mode`` is exactly the flag for "chose Other…, typed nothing
          yet", so it distinguishes the two.
        * **cleared** — the box was already showing a free-text name and the user
          emptied it. ``_solvent_other_mode`` is False there (the name came from the
          stored value, not from an unfinished pick), so the clear commits normally.
        """
        if self._suppress or self.item_id is None:
            return
        _combo, other = self._solvent_widgets
        if self._solvent_other_mode and not other.text().strip():
            # Abandoned without typing: restore the display to whatever is stored and
            # commit nothing. Leaving other_mode set would strand the empty box on the
            # next refresh, so it is dropped first.
            self._solvent_other_mode = False
            working = self.controller.workspace.measurements[
                self.item_id].working_params
            self._show_solvent(working.get(_SOLVENT_KEY))
            return
        self._solvent_other_mode = False
        self._commit_solvent(other.text())

    def _commit_solvent(self, name) -> None:
        """Store the shared solvent name, then re-derive n (and viscosity) from the
        library at the sample's temperature + wavelength. The name is stored VERBATIM;
        a name the library does not carry simply autofills nothing, and the user enters
        n and η themselves (invariant 3 — no guessed physics)."""
        value = (name or '').strip() or None
        working = self.controller.workspace.measurements[self.item_id].working_params
        if working.get(_SOLVENT_KEY) == value:
            self._show_solvent(value)     # unchanged, but the widget pair may not be
            return
        status = self.controller.edit_param(self.item_id, _SOLVENT_KEY, value)
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
        """Carry the focused measurement's pending edits to the other highlighted
        measurements before committing. No-op unless several are highlighted.

        What travels, and what each sample derives for itself, is the controller's
        rule (`Controller.apply_pending_edits`). The `label` is numbered here instead
        -- see `_fan_out_numbered_label` -- because it is the one key whose whole job
        is to tell two runs apart.
        """
        if self.item_id is None:
            return
        # The FOCUSED run joins the numbered set even when it is not in the
        # selection model: numbering only the others left it holding the bare typed
        # label beside `batch 2`..`batch 6`, which this method's own docstring rules
        # out and which a second identical gesture then duplicates. Same shape as
        # `_sample_scope_ids` four screens down.
        selected = list(self._selected_ids)
        if self.item_id not in selected:
            selected.append(self.item_id)
        others = [i for i in selected if i != self.item_id]
        if not others:
            return
        if 'label' in self.controller.dirty_keys(self.item_id):
            working = self.controller.workspace.measurements[self.item_id].working_params
            self._fan_out_numbered_label(selected, working.get('label'))
        self.controller.apply_pending_edits(self.item_id, others, skip=('label',))

    def _fan_out_numbered_label(self, selected, typed) -> None:
        """A label typed with SEVERAL measurements highlighted lands on each as that
        label plus a running number, in sidebar order, zero-padded to the width of the
        largest (`batch 02` before `batch 10`, so exported files sort in that order).

        Owner-ruled: a bulk-rename dialog was considered and dropped, and this is the
        gesture that replaces it -- so the numbering happens on the ordinary Update
        path and there is no second way to rename a selection.

        Verbatim fan-out was the defect: `label` is a per-measurement key, so one typed
        label landed identically on all six highlighted runs -- six measurements with
        one name, and under item 4's export scheme six files that collide on it, left
        to be told apart by the internal ID the collision net appends. The ID as a
        primary name is exactly what this sheet exists to remove.

        The FOCUSED measurement is numbered too. Numbering only the others would leave
        it holding the bare typed label beside `batch 2`..`batch 6`, which is both ugly
        and not unique if a later gesture types the same text again.

        An EMPTY label still fans out verbatim: clearing a selection's labels is a real
        gesture, and an empty label is not a name to number -- every run then falls back
        to its own file stem, which already differs.
        """
        text = (typed or '').strip()
        if not text:
            others = [i for i in selected if i != self.item_id]
            self.controller.apply_value_to_items(others, 'label', typed)
            return
        order = sidebar_measurement_order(self.controller)
        ids = sorted(selected, key=lambda i: order.get(i, len(order)))
        width = len(str(len(ids)))
        for n, iid in enumerate(ids, start=1):
            self.controller.apply_value_to_items(
                [iid], 'label', f'{text} {n:0{width}d}')

    def _on_update(self) -> None:
        # A background run reads COMMITTED params mid-flight — committing under
        # it would change the numbers it is computing from.
        if runner().is_busy:
            busy_notice(self.update_button)
            return
        # The fan-out cannot simply move after the ask: `commit_regroup_losses` plans
        # from the working params the commit is about to adopt, so a confirm that ran
        # first would see the focused measurement's edits alone and miss every loss
        # caused by fanning an identity key onto its siblings -- it would UNDER-report.

        # So the gesture mutates in order to decide, and puts it all back on a no.
        # Verbatim keys survived a declined confirm harmlessly (re-copying is a no-op);
        # the numbered label is a TRANSFORM, so a retry committed `batch 1 1`.
        snapshot = self.controller.snapshot_working_params()
        self._apply_edits_to_selection()
        # An identity edit can regroup samples and drop a stored hand-entered /
        # picked value with them — ask first, before anything commits (the one
        # shared confirm; Cancel leaves every edit pending, nothing applied).
        if not confirm_regroup_losses(self, self.controller,
                                      self.controller.commit_regroup_losses()):
            self.controller.restore_working_params(snapshot)
            return
        self.controller.commit()
        self._populate()           # working == committed now -> highlights clear
        self._refresh_pending()
        self.committed.emit()      # shell re-groups + refreshes the sidebar

    def _on_undo(self) -> None:
        if runner().is_busy:
            busy_notice(self.undo_button)
            return
        # Undo restores parameters, never results — so an Undo that re-scatters
        # a previously-merged sample destroys its carried values exactly like a
        # forward scatter. Same ask, same Cancel-is-inert (nothing is undone).
        if not confirm_regroup_losses(self, self.controller,
                                      self.controller.undo_regroup_losses()):
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
        if runner().is_busy:          # the Update/Undo siblings' guard, same reason
            busy_notice(self.reset_button)
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
        # Resolve the tint at paint time from the tokens (theme-aware; a raw color would
        # freeze on a theme switch). Italic is the colour-independent "edited" cue.
        bg = theme_color(self, 'edited_bg')
        fg = theme_color(self, 'edited_fg')
        bg_hex, fg_hex = bg.name(), fg.name()
        self._suppress = True
        for r, key in enumerate(self.keys):
            is_dirty = key in dirty
            if key in _COMBO_KEYS:
                # The value cell holds a combo widget; tint it (not the hidden item).
                combo = self.table.cellWidget(r, 1)
                if combo is not None:
                    combo.setStyleSheet(
                        f'background:{bg_hex};color:{fg_hex};font-style:italic;'
                        if is_dirty else '')
                continue
            cell = self.table.item(r, 1)
            font = cell.font()
            font.setItalic(is_dirty)
            cell.setFont(font)
            if is_dirty:
                cell.setBackground(bg)
                cell.setForeground(fg)
            elif key == _BLANK_I90_KEY and self._typed_blank_is_inert():
                # A greyed-out field must LOOK greyed: non-editable alone is invisible,
                # and the meaning may not ride on one channel (style guide 10 R10.1).
                # Muted from a theme token, never a raw grey, so it survives a theme
                # switch -- the unit-cell pattern. An edited cell still wins above.
                cell.setData(QtCore.Qt.ItemDataRole.BackgroundRole, None)
                cell.setForeground(theme_color(self, 'muted'))
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
        backed by the cell tooltip (§2d) so color is never the only signal
        (colorblind-safe). Bulk-grade dn/dT is deliberately NOT flagged here (it is a
        docs-only caveat)."""
        if self.item_id is None:
            return
        working = self.controller.workspace.measurements[self.item_id].working_params
        self._suppress = True
        for r, key in enumerate(self.keys):
            builder = _PROVENANCE_FIELDS.get(key)
            if builder is None:
                continue
            cell = self.table.item(r, 1)
            if cell is None:
                continue
            token, tip = getattr(self, builder)(key, working)
            cell.setData(QtCore.Qt.ItemDataRole.DecorationRole,
                         self._dot_pixmap(token) if token else None)
            cell.setToolTip(tip)
        self._suppress = False

    def _solvent_dot(self, key: str, working: dict):
        """Dot token + tooltip for a solvent-library field."""
        if working.get(_SOLVENT_SOURCE_KEYS[key]) == 'library:primary':
            return 'lib_primary', self._lib_tooltip(key, working)
        return None, ('Your value — overrides the library.'
                      if working.get(key) is not None else '')

    def _count_rate_dot(self, key: str, working: dict):
        """Dot token + tooltip for the mean count rate.

        Matches on the tag FAMILY, so the plain/outlier-clipped distinction can be
        refined inside the tag without this having to follow. Where the file recorded
        its own mean count rate, that number rides along in the tooltip: comparing it
        against what is in the cell is what catches a clipped trace section or a
        trace paired with the wrong angle.
        """
        src = working.get(_MEAN_CR_SOURCE_KEY)
        family = (src or '').split(':')[0]
        vendor = working.get('vendor_mean_count_rate_cps')
        extra = ''
        if vendor is not None:
            extra = f" The file itself recorded {vendor:.4g} cps."
        if family == 'trace':
            method = ('outlier-clipped mean' if src.endswith('sigma_clipped')
                      else 'mean')
            return 'lib_primary', (
                f'From the linked count-rate trace ({method}). Editable; a value '
                f'you type is never overwritten.{extra}')
        if family == 'vendor':
            return 'lib_primary', (
                "From the instrument file's own mean count rate. Editable; a "
                "value you type is never overwritten.")
        if family == 'generator':
            return 'lib_primary', (
                "From the synthetic generator: the mean count rate its noise model drew "
                "this correlogram at. Editable; a value you type is never overwritten.")
        if family == 'replicate':
            n = len(working.get('replicate_count_rates_cps') or ())
            spread = working.get('replicate_count_rate_spread')
            apart = ('' if spread is None else
                     f", which lie {format_fixed_sig(100.0 * float(spread), 2)} % apart")
            return 'lib_primary', (
                f"The mean of the {n} averaged runs' committed count rates{apart}. "
                f"Editable; a value you type is never overwritten.")
        return None, (f'Your value — used exactly as entered.{extra}'
                      if working.get(key) is not None else '')

    def _dot_pixmap(self, role: str) -> QtGui.QPixmap:
        """A small filled-circle pixmap in the theme color for ``role`` (rebuilt on
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
        + the PER-CONDITION uncertainty at this sample's committed T/λ (the
        same σ the auto-fill rounding used, so the tooltip's ± always matches the
        cell's precision; falls back to the box-wide figure if the condition can't be
        evaluated). The shared behavior (auto-fill / re-derive / never-overwrite) is
        stated once in the section ? help + the legend, not here. No citations in the
        tooltip — it points to the Theory-and-Equations-Guide instead."""
        name = working.get(_SOLVENT_KEY)
        # A cleared solvent name can linger with a stale 'library:primary' tag, so a
        # falsy/non-string name would reach solvent_property_info -> _lookup ->
        # casefold_identity_name -> name.strip() (AttributeError, NOT caught below).
        # Guard it here.
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
                f"analysis) · sources: Theory-and-Equations-Guide.")

    def _render_legend(self) -> None:
        """(Re)render the provenance legend with the current theme's teal so the dot
        color tracks a light/dark switch (a QLabel stylesheet color would freeze)."""
        teal = theme_token(self, 'lib_primary')
        self.legend.setText(
            f'<span style="color:{teal}">●</span> proposed value (the solvent '
            'library, or a mean count rate from a linked trace, the file, the '
            'synthetic generator, or an average\'s runs)'
            ' &nbsp;·&nbsp; no dot = your value &nbsp;·&nbsp; '
            'a value you type is never overwritten')

    def _apply_autofill_status(self, status: Optional[dict]) -> None:
        """Show / clear the non-blocking note from an autofill attempt.

        The engine has already declined to fill and cleared any stale library value;
        this only surfaces the hint (manual entry always works). It dispatches on
        ``status['reason']`` — the field that says WHY the engine declined — rather than
        on which fields it cleared. Those are different questions, and answering the
        second in place of the first is what made a missing temperature report itself as
        "outside the library range": every refusal branch clears stale values, so
        ``out_of_range`` alone cannot tell one refusal from another.
        """
        status = status or {}
        reason = status.get('reason')
        solvent = self.controller.workspace.measurements[
            self.item_id].working_params.get(_SOLVENT_KEY)

        # Only surface fields that are actually parameters of THIS measurement kind:
        # viscosity is not an SLS field, so an out-of-range viscosity there must not
        # tell the user to hand-enter a value SLS never uses (the RI still filled).
        oor = [f for f in (status.get('out_of_range') or []) if f in self.keys]
        # 'cleared' says whether the user just LOST values they could see. Every refusal
        # branch clears, so it qualifies a message rather than choosing one.
        cleared = ' The values that were there have been cleared.' if oor else ''

        if reason == 'not a primary-tier solvent':
            # Two situations share this reason and need different sentences.
            # status['tier'] tells them apart: 'estimate' is a library solvent whose data
            # is too weak to auto-fill, and the Explorer WILL show it; None is a name the
            # library does not carry at all, where pointing the user at the Explorer sends
            # them somewhere the name does not appear. Saying something is not optional
            # either way — silence looks identical to a successful fill, and the two blank
            # fields are the ones Rh and Mw are computed from. Deliberately NOT gated on
            # anything having been cleared: a fresh measurement has nothing stale to clear.
            if status.get('tier') == 'estimate':
                self.autofill_note.setText(
                    f'{solvent} has estimated properties, so nothing was filled in — '
                    f'enter the refractive index (and viscosity) yourself.{cleared} '
                    f'Utilities ▸ Solvent Explorer shows the library’s estimates.')
            else:
                self.autofill_note.setText(
                    f'{solvent} is not in the solvent library, so nothing was filled in — '
                    f'enter the refractive index (and viscosity) yourself.{cleared}')
            return

        if reason == 'need both temperature and wavelength':
            # A refusal for want of a CONDITION, not a solvent. Silent here would mean a
            # library value the user could see simply vanished when they blanked the
            # temperature, with nothing on screen connecting the two.
            self.autofill_note.setText(
                f'The library needs both a temperature and a wavelength to propose '
                f'values for {solvent}.{cleared}')
            return

        if reason == 'no solvent':
            # Same shape, the other missing input. Only worth a sentence when something
            # was actually lost — an empty measurement with no solvent yet is the normal
            # starting state, and nagging about it would be noise.
            self.autofill_note.setText(
                ('The library needs a solvent name to propose values.' + cleared)
                if oor else '')
            return

        # The range message belongs ONLY to the branch that actually evaluated the
        # library and found the condition outside its box — reason is None there.
        if not (oor and reason is None and status.get('tier') == 'primary'):
            self.autofill_note.setText('')
            return
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
        # dot and repaint the cell provenance dots in the new theme's colors (a
        # stylesheet color would freeze — see gui.theme).
        # `hasattr` guards a PaletteChange that arrives before _build_ui finishes.
        if ev.type() == QtCore.QEvent.Type.PaletteChange and hasattr(self, 'legend'):
            self._render_legend()
            if self.item_id is not None:
                self._refresh_provenance()
                self._refresh_dirty()      # re-resolve the edited-cell tint for the theme
                self._refresh_unit_colors()  # re-resolve the muted Units-column text
        super().changeEvent(ev)

    def _refresh_unit_colors(self) -> None:
        """Re-apply the muted theme token to the fixed-unit cells (column 2).

        Their foreground is set once at build time, so a raw colour would freeze on a
        light/dark switch (style guide §1 R1.1/R1.2). Only the fixed-unit rows carry a
        `QTableWidgetItem` here; the unit-*picker* rows hold a combo cell widget
        (`item(r, 2) is None`) and are skipped."""
        for r in range(len(self.keys)):
            cell = self.table.item(r, 2)
            if cell is not None:
                cell.setForeground(theme_color(self, 'muted'))

    def _refresh_pending(self) -> None:
        dirty = self.item_id is not None and self.controller.is_dirty()
        self.pending.setText('● changes pending — press Update' if dirty else '')
        # Undo is live whenever there are edits to discard OR a previously applied
        # state to step back to; otherwise there is nothing to undo.
        self.undo_button.setEnabled(
            self.item_id is not None and self.controller.can_undo())
        # Every mutation path (edit, commit, undo, reset, autofill) and selection change
        # routes through here, so refresh the passive notes from the same hook.
        self._refresh_plausibility()
        self._refresh_trace_divergence()
        self._refresh_count_rate_offers()

    def _refresh_plausibility(self) -> None:
        """Show a passive ⓘ note when the working temperature / refractive index is
        outside its physical plausibility band (a likely unit slip). Non-blocking —
        the value is used exactly as entered (invariant 3). The band is defined once,
        in core.data_models.implausible_metadata_reasons, shared with the load-time
        warning so the GUI note and the warning can never disagree."""
        if self.item_id is None:
            set_flag(self.plausibility_note, '', problem=False)
            return
        working = self.controller.workspace.measurements[self.item_id].working_params
        reasons = implausible_metadata_reasons(
            working.get('temperature_K'), working.get('solvent_refractive_index'))
        msg = ('Check for a unit slip — ' + '; '.join(reasons)
               + '. Used exactly as entered.') if reasons else ''
        set_flag(self.plausibility_note, msg, problem=False)

    def _refresh_trace_divergence(self) -> None:
        """Show a passive ⓘ note when the mean count rate being edited disagrees
        with the linked count-rate trace's own mean.

        A disclosure, never a refusal: the value is used exactly as entered. Reads
        the WORKING params so the note tracks the cell the user is editing, and
        renders both numbers in the row's currently selected display unit so the
        note and the cell can never appear to contradict each other.
        """
        if self.item_id is None:
            set_flag(self.trace_divergence_note, '', problem=False)
            return
        lm = self.controller.workspace.measurements.get(self.item_id)
        if lm is None or lm.kind != 'dls':
            set_flag(self.trace_divergence_note, '', problem=False)
            return
        st = self.controller.mean_count_rate_status(self.item_id, use_working=True)
        if st['usable'] is False:
            # A committed value the fit never sees (a legacy 0.0, a typed 0): say what the
            # fit does with it, rather than compare it against a trace as if it were used.
            set_flag(self.trace_divergence_note, st['note'], problem=True)
            return
        if not st['diverges']:
            set_flag(self.trace_divergence_note, '', problem=False)
            return
        unit = self._units.get('intensity', 'cps')
        shown = U.from_canonical('intensity', st['value'], unit)
        trace = U.from_canonical('intensity', st['trace_mean_cps'], unit)
        pct = st['rel_diff']
        pct_txt = f' ({pct * 100:+.0f} %)' if pct is not None else ''
        set_flag(self.trace_divergence_note,
                 f'Mean count rate differs from the linked trace: '
                 f'{_fmt_num(shown)} vs {_fmt_num(trace)} {unit}{pct_txt}. '
                 f'Used exactly as entered.',
                 problem=False)

    def _refresh_count_rate_offers(self) -> None:
        """Show which sources could fill this measurement's mean count rate.

        Reports what is on offer; writes nothing (invariant 3). Hidden entirely for
        a non-DLS item, where the parameter does not exist.
        """
        lm = (self.controller.workspace.measurements.get(self.item_id)
              if self.item_id is not None else None)
        if lm is None or lm.kind != 'dls':
            self.count_rate_source_label.setText('')
            self.count_rate_picker.set_offers([])
            self.count_rate_picker.setVisible(False)
            return
        self.count_rate_picker.setVisible(True)
        offers = self.controller.mean_count_rate_offers(self.item_id)
        unit = self._units.get('intensity', 'cps')
        self.count_rate_picker.set_offers(
            offers,
            value_format=lambda v: '{0} {1}'.format(
                _fmt_num(U.from_canonical('intensity', v, unit)), unit))
        source = lm.working_params.get(_MEAN_CR_SOURCE_KEY)
        self.count_rate_source_label.setText(
            'Mean count rate source: {0}'.format(source) if source
            else 'Mean count rate: not set')

    def _on_count_rate_source_picked(self, source: str) -> None:
        """Adopt an offered mean count rate into the working params.

        A deliberate user action, so it may replace a typed value -- they asked for
        it. It lands in the WORKING tier like any other edit and reaches the sample
        only through Update, so the commit path is unchanged.
        """
        if self.item_id is None:
            return
        if self.controller.apply_mean_count_rate_offer(self.item_id, source):
            # The same repaint every other edit path ends in: the table (cell,
            # dirty tint, provenance dot) and the pending indicator + notes.
            self._populate()
            self._refresh_pending()

    def _set_enabled(self, on: bool) -> None:
        self.table.setEnabled(on)
        self.update_button.setEnabled(on)
        self.undo_button.setEnabled(on)
        self.reset_button.setEnabled(on)
        self.count_rate_picker.setEnabled(on)
