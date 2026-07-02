"""
gui/sls_module.py
=================

The SLS tab: static light scattering analysis for the selected SAMPLE. Unlike the
DLS tab (one measurement), SLS is inherently per-sample — a Zimm set is a solvent
reference (c = 0) plus a concentration series, analysed together. Selecting any
measurement in the sidebar makes this tab operate on its whole sample.

It contains the visible **calibration panel** (manual entry of one calibrant point
→ k_c, geometry-aware toluene Rayleigh) and an analysis section (Zimm / Berry /
Debye / single-angle / calibration-free A₂ / excess Rayleigh ratio). Everything is
routed through the controller; no analysis or physics here.

Commit model: calibration and parameters are session/working state. "Apply
(commit)" commits them (and recomputes k_c); "Run analysis" uses the COMMITTED
state. Soft flags (uncalibrated, apparent) are GUI overlays, never on the figure.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtWidgets

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

from plotting.plots import (
    plot_zimm, plot_debye, plot_guinier, plot_rayleigh_ratio,
    plot_calibration_free_a2,
)
from gui.plot_controls import (
    AxisControlBar, make_split_panels, make_canvas_expanding, make_vertical_plot_stack)
from gui.export_helper import export_to_csv
from gui.help import add_help_to_groupbox
from gui.theme import ThemedLabel, span
from gui.worker import (
    BACKGROUND_RUN_TOOLTIP, BUSY_NOTICE, busy_notice, run_when_idle, runner)
from analysis.uncertainty import format_pm

# Reminder appended where a calibration-dependent quantity (Mw, absolute A₂) is
# shown with a ±: the SE is statistical (regression) only.
_STAT_CAVEAT = ' (± statistical; excludes calibration & dn/dc)'


def _adapt_form(form: QtWidgets.QFormLayout) -> None:
    """Let a form adapt to a narrow control panel instead of clipping (feedback #8):
    fields grow to the available width and a too-wide row wraps its field below the
    label rather than overflowing the panel."""
    form.setFieldGrowthPolicy(
        QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)


def _sample_label(sample) -> str:
    """Human-readable sample name for the SLS header — the same 'polymer / solvent @
    T' form the sidebar shows, instead of the raw 'poly|solv|temp' sample_id key whose
    unconfirmed parts render as '?' (feedback #8)."""
    poly, solv, temp = sample.polymer_name, sample.solvent_name, sample.temperature_K
    if poly and solv and temp is not None and not math.isnan(temp):
        return f'{poly} / {solv} @ {temp:g} K'
    return '(unconfirmed sample)'


_SLS_METHODS: List[Tuple[str, str]] = [
    ('Zimm', 'zimm'),
    ('Berry', 'berry'),
    ('Debye (single c, apparent)', 'debye'),
    ('Guinier (single c, apparent)', 'guinier'),
    ('Single-angle Mw (apparent)', 'single'),
    ('Calibration-free A₂', 'calfree'),
    ('Excess Rayleigh ratio', 'rayleigh'),
]

# Calibration line-edit fields: (attribute, label, allow_blank_as_None).
_CAL_FIELDS = [
    ('calibrant_intensity', 'Calibrant intensity', True),
    ('calibrant_angle_deg', 'Calibrant angle (deg)', False),
    ('standard_wavelength_nm', 'Standard wavelength (nm)', False),
    ('standard_temperature_C', 'Standard temperature (°C)', False),
    ('standard_refractive_index', 'Standard refractive index', True),
    ('dark_count_rate', 'Dark count rate', False),
]

# Intensity fields share the SLS file's own (instrument-specific) intensity unit;
# the program never converts them because only intensity RATIOS enter the analysis.
_CAL_TOOLTIPS = {
    'calibrant_intensity':
        'Enter in the SAME units as your SLS intensity file (cps, kcps, … — '
        'whatever the file uses). Only its ratio to the sample intensities enters '
        'the Rayleigh ratio, so the program does not convert it.',
    'dark_count_rate':
        'Detector dark count, in the SAME units as your SLS intensity file; it is '
        'subtracted from each intensity before the Rayleigh ratio.',
}


def _fmt(x: Optional[float], sig: int = 3) -> str:
    if x is None or not (isinstance(x, (int, float)) and math.isfinite(x)):
        return 'n/a'
    return f'{x:.{sig}g}'


class SLSModule(QtWidgets.QWidget):
    """Per-sample SLS analysis with a calibration panel."""

    committed = QtCore.Signal()   # emitted after Apply (grouping/k_c may change)

    def __init__(self, controller, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.sample_id: Optional[str] = None
        self._runnable = False
        self._suppress_cal = False
        self._suppress_mask = False
        self._ran = False               # a result is on screen (gate mask re-runs)
        # How to export the on-screen result: a (filename, path->path) pair set by
        # _run_method, or None when nothing exportable is shown. Wrapping it lets the
        # one export button serve every method without the handler re-deriving state.
        self._export: Optional[Tuple[str, object]] = None
        self._zimm_k = 1.0              # Zimm grid spacing, shared with the overlay
        # Unmasked Rayleigh series from the last run, keyed by (sample, fraction):
        # the masked-point overlay and click hit-testing read this instead of
        # recomputing on every draw (which would call the controller off the run
        # path and could race a background fit).
        self._full_rr: Dict[tuple, list] = {}
        self._run_epoch = 0            # async staleness token (bumped on sample switch)
        # The active molecular-weight fraction (None = unfractioned / whole sample).
        self._fraction: Optional[str] = None
        self._suppress_fraction = False
        # Last-run memory (within-session) keyed by (sample_id, fraction): which
        # method + axis points were last analysed, so switching samples/fractions
        # and back restores the view. SLS re-runs cheaply from committed state (+
        # persisted masks), so we replay the run rather than caching result objects.
        self._last_run_by_sample: Dict[tuple, Dict] = {}
        self.ax = None
        self._build_ui()
        self._populate_calibration()
        self._refresh_calibration_display()
        self.set_measurement(None)

    # ------------------------------------------------------------------ UI ---
    def _build_ui(self) -> None:
        # Control panel | plot split is draggable (feedback A3). The control column
        # has long calibration labels and a wide button, so it keeps a minimum width.
        _, left, right = make_split_panels(self, left_min_width=340)

        self.header = QtWidgets.QLabel()
        self.header.setWordWrap(True)
        left.addWidget(self.header)

        # The calibration / analysis / mask / depolarization groups stack in a vertically
        # resizable splitter (feedback #9): drag the grips to size them against each other
        # (e.g. enlarge the mask lists). Min heights keep each group from being clipped.
        cal = self._build_calibration_group()
        ana = self._build_analysis_group()
        mask = self._build_mask_group()
        depol = self._build_depolarization_group()
        vstack = make_vertical_plot_stack(
            [cal, ana, mask, depol], sizes=[210, 230, 200, 160],
            min_heights=[max(cal.sizeHint().height(), 120),
                         max(ana.sizeHint().height(), 120),
                         150,
                         max(depol.sizeHint().height(), 100)])
        left.addWidget(vstack, 1)

        # Analysis results as a table (feedback 2026-06-30 #14), rebuilt per method.
        left.addWidget(QtWidgets.QLabel('Results'))
        self.result_table = QtWidgets.QTableWidget(0, 2)
        self.result_table.horizontalHeader().setVisible(False)
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.result_table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.result_table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        left.addWidget(self.result_table)
        self.status = QtWidgets.QLabel('')        # one-line notes (e.g. Rayleigh series)
        self.status.setWordWrap(True)
        left.addWidget(self.status)
        self.flag_label = ThemedLabel('', role='error', bold=True)
        self.flag_label.setWordWrap(True)
        left.addWidget(self.flag_label)

        self.figure = Figure(figsize=(5.5, 4.6))
        self.canvas = make_canvas_expanding(FigureCanvas(self.figure))
        self.nav_toolbar = NavigationToolbar(self.canvas, self)
        right.addWidget(self.nav_toolbar)
        right.addWidget(self.canvas, 1)
        self.axis_bar = AxisControlBar(self.canvas)
        right.addWidget(self.axis_bar)
        # Click a plotted point to hide/show just that (concentration, angle).
        self.canvas.mpl_connect('button_press_event', self._on_canvas_click)

    def _build_calibration_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox('Calibration')
        add_help_to_groupbox(box, 'Turns raw intensities into absolute Rayleigh '
                             'ratios — required for Mw and A₂.', bullets=[
                                 'Enter <b>one calibrant point</b>: its intensity, '
                                 'angle, and the standard\'s wavelength / temperature '
                                 '/ refractive index / geometry.',
                                 'The program computes <b>k_c</b> from it — don\'t use '
                                 'a vendor constant (it goes stale after alignment).',
                                 'Calibrate once <b>session-wide</b>; tick '
                                 '"Per-sample calibration" only to override one sample.',
                                 'Rg and the calibration-free product survive without '
                                 'calibration; Mw and absolute A₂ do not.',
                             ])
        form = QtWidgets.QFormLayout(box)
        _adapt_form(form)      # wrap long rows + grow fields so nothing clips (#8)

        self.per_sample_check = QtWidgets.QCheckBox('Per-sample calibration')
        self.per_sample_check.setToolTip(
            'Give this sample its own calibration. When off, the session-wide '
            'calibration is used.')
        self.per_sample_check.toggled.connect(self._on_per_sample_toggled)
        form.addRow(self.per_sample_check)

        self.cal_edits: Dict[str, QtWidgets.QLineEdit] = {}
        for key, label, allow_blank in _CAL_FIELDS:
            edit = QtWidgets.QLineEdit()
            if allow_blank:
                edit.setPlaceholderText('(blank = none)')
            edit.editingFinished.connect(lambda k=key: self._on_cal_edit(k))
            self.cal_edits[key] = edit
            # Intensities (calibrant + dark) must be in WHATEVER unit the SLS file
            # uses: only their ratio to the file's sample intensities enters the
            # Rayleigh ratio, so the program can't convert them to a fixed unit.
            tip = _CAL_TOOLTIPS.get(key)
            if tip:
                edit.setToolTip(tip)
            form.addRow(label + ':', edit)

        self.geometry_combo = QtWidgets.QComboBox()
        self.geometry_combo.addItems(['VU', 'VV', 'VH'])
        self.geometry_combo.currentTextChanged.connect(self._on_geometry_changed)
        form.addRow('Standard geometry:', self.geometry_combo)

        self.kc_label = QtWidgets.QLabel('')
        self.kc_label.setWordWrap(True)
        # "Calibration k_c" — distinct from the Zimm spacing k (feedback A7): k_c is
        # the Rayleigh-ratio-per-intensity calibration constant, not the q²+k·c grid k.
        form.addRow('Calibration k_c:', self.kc_label)

        self.apply_button = QtWidgets.QPushButton('Apply parameters')
        self.apply_button.clicked.connect(self._on_apply)
        form.addRow(self.apply_button)
        return box

    def _build_analysis_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox('Analysis')
        add_help_to_groupbox(box, 'Get Mw, Rg and A₂ from the intensities.', bullets=[
            '<b>Thermodynamic</b> (Zimm/Berry): extrapolated to zero angle <i>and</i> '
            'zero concentration — the true Mw, Rg, A₂.',
            '<b>Apparent</b> (Debye/Guinier/single-angle): one concentration or angle '
            'only — a quick estimate, not extrapolated.',
            'Zimm fits the standard plot; <b>Berry</b> (√ axis) is better for larger '
            'or higher-Mw particles where Zimm curves.',
            'See the Advanced Guide for the equations.',
        ])
        form = QtWidgets.QFormLayout(box)
        _adapt_form(form)      # wrap long rows + grow fields so nothing clips (#8)

        # Molecular-weight fraction selector. A Mw series shares one solvent
        # reference but holds several fractions ("250k", "1M", ...); each is its own
        # Zimm fit. Hidden/disabled when a sample has just one (unlabelled) fraction.
        self.fraction_combo = QtWidgets.QComboBox()
        self.fraction_combo.currentIndexChanged.connect(self._on_fraction_changed)
        self.fraction_label = QtWidgets.QLabel('Mw fraction:')
        form.addRow(self.fraction_label, self.fraction_combo)

        self.method_combo = QtWidgets.QComboBox()
        for label, key in _SLS_METHODS:
            self.method_combo.addItem(label, key)
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        form.addRow('Method:', self.method_combo)

        self.conc_combo = QtWidgets.QComboBox()
        form.addRow('Concentration:', self.conc_combo)
        self.angle_combo = QtWidgets.QComboBox()
        form.addRow('Angle:', self.angle_combo)

        # Zimm/Berry aesthetic spacing constant k (x = q² + k·c). It only spreads
        # the drawn concentration curves; Mw/Rg/A₂ come from the real q²,c fit and
        # are unchanged. Blank/Suggest = auto (q²max / cmax).
        k_row = QtWidgets.QHBoxLayout()
        k_row.setContentsMargins(0, 0, 0, 0)
        self.zimm_k_edit = QtWidgets.QLineEdit()
        self.zimm_k_edit.setPlaceholderText('auto')
        self.zimm_k_edit.setToolTip(
            'Spacing constant k in the Zimm x-axis q² + k·c. Purely aesthetic — it '
            'spreads the concentration curves and does NOT change Mw, Rg or A₂. '
            'Blank = auto.')
        self.zimm_k_suggest = QtWidgets.QPushButton('Suggest')
        self.zimm_k_suggest.clicked.connect(self._on_suggest_k)
        k_row.addWidget(self.zimm_k_edit)
        k_row.addWidget(self.zimm_k_suggest)
        k_widget = QtWidgets.QWidget(); k_widget.setLayout(k_row)
        form.addRow('Zimm spacing k:', k_widget)

        mw_row = QtWidgets.QHBoxLayout()
        self.mw_edit = QtWidgets.QLineEdit()
        self.mw_edit.setPlaceholderText('Mw (g/mol)')
        self.mw_button = QtWidgets.QPushButton('Set manual Mw')
        self.mw_button.clicked.connect(self._on_set_mw)
        mw_row.addWidget(self.mw_edit)
        mw_row.addWidget(self.mw_button)
        mw_widget = QtWidgets.QWidget()
        mw_widget.setLayout(mw_row)
        form.addRow('Manual Mw:', mw_widget)
        self.mw_display = ThemedLabel('', role='muted')
        form.addRow('', self.mw_display)

        self.run_button = QtWidgets.QPushButton('Run analysis')
        self.run_button.setToolTip(BACKGROUND_RUN_TOOLTIP)
        self.run_button.clicked.connect(self._on_run)
        form.addRow(self.run_button)
        self.export_button = QtWidgets.QPushButton('Export CSV…')
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._on_export)
        form.addRow(self.export_button)

        self._on_method_changed()
        return box

    def _build_mask_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox('Data mask (hide / show)')
        v = QtWidgets.QVBoxLayout(box)
        lists = QtWidgets.QHBoxLayout()
        acol = QtWidgets.QVBoxLayout()
        acol.addWidget(QtWidgets.QLabel('Angles'))
        self.angle_list = QtWidgets.QListWidget()
        self.angle_list.setMinimumHeight(80)     # grows with its resizable pane (#9)
        self.angle_list.itemChanged.connect(self._on_angle_item_changed)
        acol.addWidget(self.angle_list)
        ccol = QtWidgets.QVBoxLayout()
        ccol.addWidget(QtWidgets.QLabel('Concentrations'))
        self.conc_list = QtWidgets.QListWidget()
        self.conc_list.setMinimumHeight(80)      # grows with its resizable pane (#9)
        self.conc_list.itemChanged.connect(self._on_conc_item_changed)
        ccol.addWidget(self.conc_list)
        lists.addLayout(acol)
        lists.addLayout(ccol)
        v.addLayout(lists)
        self.clear_mask_button = QtWidgets.QPushButton('Show all (clear mask)')
        self.clear_mask_button.clicked.connect(self._on_clear_masks)
        v.addWidget(self.clear_mask_button)
        hint = ThemedLabel('Untick an angle or concentration to hide it; the '
                           'analysis re-runs on the shown points (hidden ones '
                           'are greyed). Or click a point on the plot to '
                           'hide/show just that point.', role='hint', size=11)
        hint.setWordWrap(True)
        v.addWidget(hint)
        return box

    def _build_depolarization_group(self) -> QtWidgets.QGroupBox:
        # Static depolarized light scattering (DPLS Phase 1). A standalone calculator:
        # enter the VV and VH intensities (or rho_v directly) and read off the
        # depolarisation ratio, optical anisotropy, and the Cabannes correction that
        # strips anisotropy inflation from Mw. Assumes vertically polarised incident
        # light (the modern default). Not yet tied to loaded VV/VH series -- that
        # pairing waits for a real depolarised-acquisition path.
        box = QtWidgets.QGroupBox('Depolarization (anisotropy correction)')
        box.setToolTip(
            'Static depolarized light scattering. From the VV (polarized) and VH '
            '(depolarized) intensities — or the depolarization ratio ρv directly — '
            'compute ρv, the optical anisotropy δ², and the Cabannes factor that '
            'removes anisotropy inflation from Mw (R_iso = R_VV·(1 − 4ρv/3)). '
            'Assumes vertically polarized incident light.')
        form = QtWidgets.QFormLayout(box)
        _adapt_form(form)      # wrap long rows + grow fields so nothing clips (#8)

        self.depol_mode_combo = QtWidgets.QComboBox()
        self.depol_mode_combo.addItem('VV & VH intensities', 'intensities')
        self.depol_mode_combo.addItem('Depolarization ratio ρv', 'ratio')
        self.depol_mode_combo.currentIndexChanged.connect(self._on_depol_mode_changed)
        form.addRow('Input:', self.depol_mode_combo)

        # Two input pages in a stack, switched by the mode combo.
        self.depol_stack = QtWidgets.QStackedWidget()

        int_page = QtWidgets.QWidget()
        ip = QtWidgets.QFormLayout(int_page)
        ip.setContentsMargins(0, 0, 0, 0)
        _adapt_form(ip)
        self.depol_ivv_edit = QtWidgets.QLineEdit()
        self.depol_ivv_edit.setPlaceholderText('I_VV (file units)')
        self.depol_ivh_edit = QtWidgets.QLineEdit()
        self.depol_ivh_edit.setPlaceholderText('I_VH (file units)')
        self.depol_dark_edit = QtWidgets.QLineEdit()
        self.depol_dark_edit.setPlaceholderText('0 (optional)')
        _depol_tip = ('Same units as your intensities; subtracted from both before '
                      'the ratio. The depolarized channel is weak, so an '
                      'un-subtracted dark count biases ρv high.')
        self.depol_dark_edit.setToolTip(_depol_tip)
        ip.addRow('I_VV (polarized):', self.depol_ivv_edit)
        ip.addRow('I_VH (depolarized):', self.depol_ivh_edit)
        ip.addRow('Dark count:', self.depol_dark_edit)

        ratio_page = QtWidgets.QWidget()
        rp = QtWidgets.QFormLayout(ratio_page)
        rp.setContentsMargins(0, 0, 0, 0)
        _adapt_form(rp)
        self.depol_rhov_edit = QtWidgets.QLineEdit()
        self.depol_rhov_edit.setPlaceholderText('ρv = I_VH / I_VV')
        rp.addRow('ρv:', self.depol_rhov_edit)

        self.depol_stack.addWidget(int_page)     # index 0 -> 'intensities'
        self.depol_stack.addWidget(ratio_page)   # index 1 -> 'ratio'
        form.addRow(self.depol_stack)

        self.depol_compute_button = QtWidgets.QPushButton('Compute depolarization')
        self.depol_compute_button.clicked.connect(self._on_compute_depol)
        form.addRow(self.depol_compute_button)

        self.depol_table = QtWidgets.QTableWidget(0, 2)
        self.depol_table.horizontalHeader().setVisible(False)
        self.depol_table.verticalHeader().setVisible(False)
        self.depol_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.depol_table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.depol_table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        form.addRow(self.depol_table)
        self.depol_note = ThemedLabel('', role='hint', size=11)
        self.depol_note.setWordWrap(True)
        form.addRow(self.depol_note)
        return box

    def _on_depol_mode_changed(self) -> None:
        self.depol_stack.setCurrentIndex(self.depol_mode_combo.currentIndex())
        self.depol_table.setRowCount(0)
        self.depol_note.clear()

    def _on_compute_depol(self) -> None:
        mode = self.depol_mode_combo.currentData()
        try:
            if mode == 'ratio':
                rho_v = float(self.depol_rhov_edit.text().strip())
                res = self.controller.compute_depolarization(rho_v=rho_v)
            else:
                i_vv = float(self.depol_ivv_edit.text().strip())
                i_vh = float(self.depol_ivh_edit.text().strip())
                dark_txt = self.depol_dark_edit.text().strip()
                dark = float(dark_txt) if dark_txt else 0.0
                res = self.controller.compute_depolarization(
                    i_vv=i_vv, i_vh=i_vh, dark_count=dark)
        except ValueError as exc:
            self.depol_table.setRowCount(0)
            self.depol_note.setRole('error')
            self.depol_note.setText(f'Invalid input: {exc}')
            return
        self._show_depol_result(res)

    def _show_depol_result(self, res) -> None:
        rv = _fmt(res.rho_v, 4)
        if res.rho_v_se is not None:
            rv += f' ± {_fmt(res.rho_v_se, 2)}'
        rows = [
            ('ρ_v', rv),
            ('ρ_u', _fmt(res.rho_u, 4)),
            ('δ²', _fmt(res.optical_anisotropy_sq, 4)),
            ('Cabannes f', _fmt(res.cabannes_isotropic_factor, 4)),
            ('anisotropic %', _fmt(res.anisotropic_fraction * 100, 3)),
        ]
        if not res.physically_valid:
            note_role, note_text = 'error', f'⚠ {res.note}'
        else:
            # Show the Mw correction when the current sample has an analysed Mw.
            if self.sample_id is not None:
                corr = self.controller.cabannes_corrected_mw(
                    self.sample_id, self._fraction, res.cabannes_isotropic_factor)
                if corr is not None:
                    mw_app, mw_corr, src = corr
                    rows.append(('Isotropic-corrected Mw (g/mol)',
                                 f'{_fmt(mw_corr, 4)} (from {_fmt(mw_app, 4)}, {src})'))
            note_role, note_text = 'hint', (
                'R_iso = R_VV(1 − 4ρv/3) — apply to the Zimm/Debye Mw. '
                'Display only; not written to the sample.')
        t = self.depol_table
        t.setRowCount(len(rows))
        for r, (label, value) in enumerate(rows):
            t.setItem(r, 0, QtWidgets.QTableWidgetItem(label))
            t.setItem(r, 1, QtWidgets.QTableWidgetItem(value))
        t.setMaximumHeight(28 + 22 * max(len(rows), 1))
        t.resizeColumnsToContents()
        self.depol_note.setRole(note_role)
        self.depol_note.setText(note_text)

    def _zimm_spacing(self, rr) -> float:
        """The k in q^2 + k*c that plot_zimm uses (replicated so the greyed overlay
        lands at the same grid positions)."""
        nonzero = sorted((r for r in rr if r.concentration_g_per_mL != 0),
                         key=lambda r: r.concentration_g_per_mL)
        if not nonzero:
            return 1.0
        cmax = max(r.concentration_g_per_mL for r in nonzero)
        q2max = float(np.nanmax(nonzero[0].q2_nm2))
        return q2max / cmax if cmax > 0 else 1.0

    # ---------------------------------------------------------- selection ---
    def set_measurement(self, item_id: Optional[str]) -> None:
        """Resolve the measurement's SAMPLE and operate on it (SLS is per-sample)."""
        # Keep-last (#15): selecting a DLS-only measurement shouldn't blank the SLS plot.
        # If the new pick has no SLS data and an SLS sample is already shown, keep it.
        if item_id is not None and self.sample_id is not None:
            _sid = self.controller.sample_id_of(item_id)
            _sample = self.controller.workspace.samples.get(_sid) if _sid else None
            if _sample is None or not _sample.has_sls:
                return
        self.sample_id = None
        self._run_epoch += 1             # drop any in-flight fit for the old sample
        self._clear_result_table()       # stale result clears on a sample switch (#14)
        runnable = False
        if item_id is None:
            header = 'Select an SLS sample in the sidebar.'
        else:
            sid = self.controller.sample_id_of(item_id)
            sample = self.controller.workspace.samples.get(sid) if sid else None
            if sample is None:
                header = 'No sample for the selected measurement.'
            elif not sample.has_sls:
                header = (f'<b>{_sample_label(sample)}</b> has no SLS data yet — '
                          'load an SLS intensity file.')
            else:
                self.sample_id = sid
                runnable = True
                header = f'SLS sample: <b>{_sample_label(sample)}</b>'
        self._ran = False
        self._set_state(header, runnable)
        if runnable:
            self._populate_fraction_combo()   # sets self._fraction
            self._populate_axis_selectors()
            self._populate_mask_lists()
            self._refresh_mw_display()
            # Restore the last analysis run on this (sample, fraction) by replaying
            # it; otherwise show a blank plot.
            if (self.sample_id, self._fraction) in self._last_run_by_sample:
                self._restore_last_run(self.sample_id, self._fraction)
            else:
                self._clear_plot()
        else:
            self._populate_fraction_combo()   # clears it
            self._populate_mask_lists()       # clears the lists
        # Calibration scope follows the selected sample.
        self._sync_calibration_scope()
        self._populate_calibration()
        self._refresh_calibration_display()

    # ------------------------------------------------------------- mask ---
    def _populate_mask_lists(self) -> None:
        self._suppress_mask = True
        self.angle_list.clear()
        self.conc_list.clear()
        if self.sample_id is not None:
            frac = self._fraction
            mask = self.controller.sls_mask(self.sample_id, frac)
            for a in self.controller.sample_angles(self.sample_id, fraction=frac):
                it = QtWidgets.QListWidgetItem(f'{a:.0f}°')
                it.setData(QtCore.Qt.ItemDataRole.UserRole, a)
                it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(QtCore.Qt.CheckState.Unchecked
                                 if mask.is_angle_masked(a)
                                 else QtCore.Qt.CheckState.Checked)
                self.angle_list.addItem(it)
            for cc in self.controller.sample_concentrations(self.sample_id, fraction=frac):
                it = QtWidgets.QListWidgetItem(f'{cc * 1000:.4g} mg/mL')
                it.setData(QtCore.Qt.ItemDataRole.UserRole, cc)
                it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(QtCore.Qt.CheckState.Unchecked
                                 if mask.is_concentration_masked(cc)
                                 else QtCore.Qt.CheckState.Checked)
                self.conc_list.addItem(it)
        self._suppress_mask = False

    @QtCore.Slot('QListWidgetItem*')
    def _on_angle_item_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        if self._suppress_mask or self.sample_id is None:
            return
        if runner().is_busy:              # mask edits change what a running fit reads
            busy_notice(self)
            self._populate_mask_lists()   # revert the checkbox to committed state
            return
        a = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if item.checkState() == QtCore.Qt.CheckState.Checked:
            self.controller.unmask_angle(self.sample_id, a, self._fraction)
        else:
            self.controller.mask_angle(self.sample_id, a, self._fraction)
        self._rerun()

    @QtCore.Slot('QListWidgetItem*')
    def _on_conc_item_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        if self._suppress_mask or self.sample_id is None:
            return
        if runner().is_busy:
            busy_notice(self)
            self._populate_mask_lists()   # revert the checkbox to committed state
            return
        cc = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if item.checkState() == QtCore.Qt.CheckState.Checked:
            self.controller.unmask_concentration(self.sample_id, cc, self._fraction)
        else:
            self.controller.mask_concentration(self.sample_id, cc, self._fraction)
        self._rerun()

    @QtCore.Slot()
    def _on_clear_masks(self) -> None:
        if self.sample_id is None:
            return
        if runner().is_busy:
            busy_notice(self)
            return
        self.controller.clear_sls_mask(self.sample_id, self._fraction)
        self._populate_mask_lists()
        self._rerun()

    def _rerun(self) -> None:
        """Re-run the current analysis after a mask change (only once something
        has already been plotted, so toggling masks before a run is quiet)."""
        if self._ran and self._runnable and self.sample_id is not None:
            self._on_run()

    def _set_state(self, header_html: str, runnable: bool) -> None:
        self.header.setText(header_html)
        self._runnable = runnable
        self.run_button.setEnabled(runnable)
        if not runnable:
            self.export_button.setEnabled(False)
            self._export = None
            self.status.clear()
            self.flag_label.clear()
            self._clear_plot()

    # ------------------------------------------------------------ fraction ---
    def _populate_fraction_combo(self) -> None:
        """Fill the Mw-fraction selector from the sample's distinct labels and set
        the active fraction. Hidden when a sample has a single (unlabelled) one."""
        self._suppress_fraction = True
        self.fraction_combo.clear()
        fractions = ([None] if self.sample_id is None
                     else self.controller.sample_fractions(self.sample_id))
        for frac in fractions:
            self.fraction_combo.addItem('(unlabelled)' if frac is None else frac, frac)
        self._fraction = fractions[0]
        multi = len(fractions) > 1
        self.fraction_combo.setVisible(multi)
        self.fraction_label.setVisible(multi)
        self._suppress_fraction = False

    def _current_fraction(self) -> Optional[str]:
        return self._fraction

    @QtCore.Slot()
    def _on_fraction_changed(self) -> None:
        if self._suppress_fraction or self.sample_id is None:
            return
        self._fraction = self.fraction_combo.currentData()
        self._ran = False
        self._run_epoch += 1             # drop any in-flight fit for the old fraction
        self._populate_axis_selectors()
        self._populate_mask_lists()
        self._refresh_mw_display()
        if (self.sample_id, self._fraction) in self._last_run_by_sample:
            self._restore_last_run(self.sample_id, self._fraction)
        else:
            self._clear_plot()

    def _populate_axis_selectors(self) -> None:
        frac = self._fraction
        self.conc_combo.clear()
        for c in self.controller.sample_concentrations(self.sample_id, fraction=frac):
            self.conc_combo.addItem(f'{c * 1000:.4g} mg/mL', c)
        self.angle_combo.clear()
        for a in self.controller.sample_angles(self.sample_id, fraction=frac):
            self.angle_combo.addItem(f'{a:.0f}°', a)

    # ---------------------------------------------------------- calibration ---
    def _populate_calibration(self) -> None:
        cw = self.controller.calibration_fields(self.sample_id)
        self._suppress_cal = True
        for key, edit in self.cal_edits.items():
            v = cw.get(key)
            edit.setText('' if v is None else str(v))
        self.geometry_combo.setCurrentText(cw.get('standard_geometry', 'VU'))
        self._suppress_cal = False

    def _sync_calibration_scope(self) -> None:
        """Set the per-sample checkbox from the controller, without re-triggering it."""
        self._suppress_cal = True
        has = (self.sample_id is not None
               and self.controller.has_sample_calibration(self.sample_id))
        self.per_sample_check.setEnabled(self.sample_id is not None)
        self.per_sample_check.setChecked(has)
        self._suppress_cal = False

    @QtCore.Slot(bool)
    def _on_per_sample_toggled(self, checked: bool) -> None:
        if self._suppress_cal or self.sample_id is None:
            return
        if checked:
            self.controller.enable_sample_calibration(self.sample_id)
        else:
            self.controller.disable_sample_calibration(self.sample_id)
        self._populate_calibration()
        self._refresh_calibration_display()

    @QtCore.Slot()
    def _on_cal_edit(self, key: str) -> None:
        if self._suppress_cal:
            return
        allow_blank = dict((k, b) for k, _l, b in _CAL_FIELDS)[key]
        text = self.cal_edits[key].text().strip()
        if text == '':
            if not allow_blank:
                return
            value = None
        else:
            try:
                value = float(text)
            except ValueError:
                self.status.setText(f'{key}: "{text}" is not a number.')
                return
        self.controller.set_calibration_field(key, value, self.sample_id)
        self._refresh_calibration_display()

    @QtCore.Slot(str)
    def _on_geometry_changed(self, text: str) -> None:
        if self._suppress_cal:
            return
        self.controller.set_calibration_field('standard_geometry', text, self.sample_id)
        self._refresh_calibration_display()

    def _refresh_calibration_display(self) -> None:
        sid = self.sample_id
        scope = ('per-sample' if (sid is not None
                 and self.controller.has_sample_calibration(sid)) else 'session')
        try:
            preview = self.controller.preview_k_c(sid)
        except Exception as exc:
            self.kc_label.setText(span(self, 'error', f'cannot compute: {exc}'))
            return
        committed = self.controller.committed_k_c(sid)
        if preview is None:
            self.kc_label.setText(f'[{scope}] uncalibrated (enter a calibrant '
                                  'intensity) — Mw/A₂ will be flagged unreliable')
        else:
            pending = (committed is None or abs((preview or 0) - (committed or 0))
                       > 1e-30 * max(1.0, abs(preview)))
            note = '  ● pending — Apply to commit' if pending else ''
            self.kc_label.setText(
                f'[{scope}] preview = {preview:.4e}; committed = '
                f'{("none" if committed is None else f"{committed:.4e}")}{note}')

    @QtCore.Slot()
    def _on_apply(self) -> None:
        if runner().is_busy:              # commit changes committed calibration a
            busy_notice(self)             # background fit is reading (invariant 4)
            return
        self.controller.commit()
        self._populate_calibration()
        self._refresh_calibration_display()
        if self.sample_id is not None:
            self._refresh_mw_display()
        self.committed.emit()

    # --------------------------------------------------------- manual Mw ---
    @QtCore.Slot()
    def _on_set_mw(self) -> None:
        if self.sample_id is None:
            return
        if runner().is_busy:              # writes SampleResult.mw a fit may read
            busy_notice(self)
            return
        text = self.mw_edit.text().strip()
        if not text:
            return
        try:
            mw = float(text)
        except ValueError:
            QtWidgets.QMessageBox.warning(self, 'Invalid Mw',
                                          f'"{text}" is not a number.')
            return
        self.controller.set_manual_mw(self.sample_id, mw, self._fraction)
        self._refresh_mw_display()

    def _refresh_mw_display(self) -> None:
        if self.sample_id is None:
            self.mw_display.setText('')
            return
        r = self.controller.workspace.samples[self.sample_id].result_for(self._fraction)
        if r.mw_g_per_mol is None:
            self.mw_display.setText('Mw: not yet determined')
        else:
            self.mw_display.setText(
                f'Mw = {r.mw_g_per_mol:.3e} g/mol  ({r.mw_source})')

    # ------------------------------------------------------------- run ---
    @QtCore.Slot()
    def _on_method_changed(self) -> None:
        key = self.method_combo.currentData()
        self.conc_combo.setEnabled(key in ('debye', 'guinier', 'single'))
        self.angle_combo.setEnabled(key in ('single', 'calfree'))
        zimm = key in ('zimm', 'berry')
        self.zimm_k_edit.setEnabled(zimm)
        self.zimm_k_suggest.setEnabled(zimm)

    def _current_zimm_k(self, rr) -> float:
        """The spacing constant for the Zimm plot: the user's value if entered and
        valid, else the auto value (q²max / cmax)."""
        text = self.zimm_k_edit.text().strip()
        if text:
            try:
                return float(text)
            except ValueError:
                pass
        return self._zimm_spacing(rr)

    @QtCore.Slot()
    def _on_suggest_k(self) -> None:
        if self.sample_id is None:
            return
        if runner().is_busy:              # builds a Rayleigh series off the run path
            busy_notice(self)
            return
        try:
            rr = self.controller.masked_rayleigh(self.sample_id, self._fraction)
        except Exception:
            return
        self.zimm_k_edit.setText(f'{self._zimm_spacing(rr):.4g}')

    @QtCore.Slot()
    def _on_run(self) -> None:
        if not self._runnable or self.sample_id is None:
            return
        sid = self.sample_id
        samp = self.controller.workspace.samples[sid]
        if samp.solvent_reference_item_id is None:
            QtWidgets.QMessageBox.warning(
                self, 'No solvent reference',
                'This sample has no c = 0 solvent reference. SLS calibration and '
                'the excess Rayleigh ratio require one.')
            return
        # Read everything the fit needs on the main thread; the compute phase
        # runs on the worker (the SLS fits build + fit every angle/concentration),
        # then the present phase plots + fills tables back on the main thread.
        method = self.method_combo.currentData()
        method_text = self.method_combo.currentText()
        frac = self._fraction
        conc = self.conc_combo.currentData()
        angle = self.angle_combo.currentData()
        self._run_epoch += 1
        epoch = self._run_epoch

        def fail(exc: BaseException) -> None:
            self.export_button.setEnabled(False)
            QtWidgets.QMessageBox.critical(
                self, 'Analysis failed',
                f'Could not run {method_text!r}.\n\n{exc}\n\n'
                'Confirm parameters (Data tab) and the calibration, then Apply.')
            self.status.setText('Analysis failed — see dialog.')

        def done(payload) -> None:
            if epoch != self._run_epoch:
                return                       # sample/fraction changed — stale
            # Only the current view's series is ever read (by the overlay/click
            # hit-test), so keep a single entry rather than accumulating one per
            # (sample, fraction) for the life of the session.
            self._full_rr = {(sid, frac): payload['full_rr']}
            # The present phase (plot + tables) can also raise on a degenerate
            # result; route it through the same 'Analysis failed' path the old
            # single try/except gave, so a plotting edge case never escapes as an
            # uncaught slot exception with the UI half-updated.
            try:
                self._present_method(method, sid, conc, payload)
            except Exception as exc:         # noqa: BLE001
                fail(exc)
                return
            self._ran = True
            self.export_button.setEnabled(self._export is not None)
            self._last_run_by_sample[(sid, frac)] = {
                'method': method, 'conc': conc, 'angle': angle}

        if runner().try_submit(
                lambda: self._compute_method(method, sid, frac, conc, angle),
                done, fail, description=f'{method_text} fit',
                busy_widgets=(self.run_button,)):
            self.status.setText('Running in the background…')
        else:
            self.status.setText(BUSY_NOTICE)

    def _compute_method(self, method: str, sid: str, frac, conc, angle) -> Dict:
        """Worker phase: every controller call the chosen method needs, plus the
        unmasked Rayleigh series for the overlay/click cache. No Qt, no plotting."""
        c = self.controller
        payload: Dict = {'full_rr': c.run_rayleigh(sid, frac)}
        if method in ('zimm', 'berry'):
            payload['rr'] = c.masked_rayleigh(sid, frac)
            payload['res'] = c.run_zimm(sid, method, frac)
        elif method == 'debye':
            payload['res'] = c.run_debye(sid, conc, frac)
        elif method == 'guinier':
            payload['res'] = c.run_guinier(sid, conc, fraction=frac)
        elif method == 'single':
            payload['res'] = c.run_single_angle(sid, conc, angle, frac)
        elif method == 'calfree':
            payload['res'] = c.run_calibration_free_a2(sid, angle, fraction=frac)
        else:  # rayleigh
            payload['rr'] = c.masked_rayleigh(sid, frac)
        return payload

    @QtCore.Slot()
    def _on_export(self) -> None:
        if self._export is None:
            return
        default_name, do_export = self._export
        status = export_to_csv(self, default_name, do_export)
        if status:
            self.status.setText(status)

    @staticmethod
    def _set_combo_data(combo: QtWidgets.QComboBox, value) -> None:
        """Select the item whose stored data == value (no-op if not present)."""
        idx = combo.findData(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _restore_last_run(self, sid: str, fraction: Optional[str]) -> None:
        """Replay this (sample, fraction)'s last analysis (method + axis selection)."""
        last = self._last_run_by_sample[(sid, fraction)]
        self._set_combo_data(self.method_combo, last['method'])
        self._set_combo_data(self.conc_combo, last['conc'])
        self._set_combo_data(self.angle_combo, last['angle'])
        # Re-run from committed state + persisted masks. Deferred if a fit is in
        # flight (switching samples during a run replays once the worker frees).
        run_when_idle(self._on_run)

    def _present_method(self, method: str, sid: str, conc, payload: Dict) -> None:
        """Main-thread phase: plot + fill tables from the worker's payload. Reads
        precomputed results (payload['rr'] / ['res']); makes no controller fit
        calls itself (the masked-point overlay reads the cached full series)."""
        c = self.controller
        self._export = None
        if method in ('zimm', 'berry'):
            rr = payload['rr']
            res = payload['res']
            self._setup_axes()
            self._zimm_k = self._current_zimm_k(rr)
            plot_zimm(rr, res, ax=self.ax, spacing_constant=self._zimm_k)
            self.ax.set_title(f'{method.capitalize()} plot')
            self._refresh_mw_display()
            self._summarize_zimm(res)
            self._export = (f'{sid}_{method}.csv', lambda p: c.export_zimm(rr, res, p))
        elif method == 'debye':
            res = payload['res']
            self._export = (f'{sid}_debye.csv', lambda p: c.export_debye(res, p))
            self._setup_axes()
            plot_debye(res, ax=self.ax)
            self.ax.set_title('Debye plot (single concentration)')
            self._fill_result_table([
                ('Mw_app (g/mol)', format_pm(res.mw_apparent_g_per_mol,
                                             getattr(res, 'mw_apparent_se', None))),
                ('Rg_app (nm)', format_pm(res.rg_apparent_nm,
                                          getattr(res, 'rg_apparent_se', None))),
                ('R²', _fmt(res.r_squared)),
            ])
            self.flag_label.setText(self._apparent_flag(res.calibrated) + _STAT_CAVEAT)
        elif method == 'guinier':
            res = payload['res']
            self._export = (f'{sid}_guinier.csv', lambda p: c.export_guinier(res, p))
            self._setup_axes()
            plot_guinier(res, ax=self.ax)
            self.ax.set_title('Guinier plot (single concentration)')
            self._fill_result_table([
                ('Rg_app (nm)', format_pm(res.rg_nm, getattr(res, 'rg_se', None))),
                ('qRg(max)', _fmt(res.qrg_max, 2)),
                ('R²', _fmt(res.r_squared)),
            ])
            flag = ('⚠ apparent (single concentration): Rg still contains '
                    'concentration effects — extrapolate over c for the '
                    'thermodynamic Rg.')
            if not res.guinier_valid:
                flag += (f'  qRg(max) = {_fmt(res.qrg_max, 2)} > 1.3 is outside the '
                         'Guinier regime — treat Rg with caution (Berry/Zimm '
                         'linearise the high-qRg regime better).')
            self.flag_label.setText(flag)
        elif method == 'single':
            res = payload['res']
            self._export = (f'{sid}_single_angle.csv',
                            lambda p: c.export_single_angle(res, p))
            self._clear_plot()
            self._fill_result_table([
                ('Mw_app (g/mol)', _fmt(res.mw_apparent_g_per_mol)),
                ('Angle', f'{res.angle_deg:.0f}°'),
                ('c (mg/mL)', f'{res.concentration_g_per_mL * 1000:.3g}'),
            ])
            self.flag_label.setText(
                '⚠ apparent: single angle + single concentration (contains the '
                'form factor and the 2A₂c term).')
        elif method == 'calfree':
            res = payload['res']
            self._export = (f'{sid}_calibration_free_a2.csv',
                            lambda p: c.export_calibration_free_a2(res, p))
            self._setup_axes()
            plot_calibration_free_a2(res, ax=self.ax)
            self.ax.set_title('Calibration-free A₂')
            a2 = (format_pm(res.a2_mol_mL_per_g2, getattr(res, 'a2_se', None))
                  if res.a2_mol_mL_per_g2 is not None else 'set a manual Mw to get A₂')
            self._fill_result_table([
                ('2·A₂·Mw (scale-independent)',
                 format_pm(res.two_a2_mw, getattr(res, 'two_a2_mw_se', None))),
                ('A₂ (mol·mL/g²)', a2),
            ])
            self.flag_label.setText(
                '(± statistical only)'
                if getattr(res, 'two_a2_mw_se', None) is not None else '')
        else:  # rayleigh
            rr = payload['rr']
            self._export = (f'{sid}_rayleigh.csv',
                            lambda p: c.export_rayleigh_series(rr, p))
            self._setup_axes()
            for r in rr:
                if r.concentration_g_per_mL == 0:
                    continue
                plot_rayleigh_ratio(r, ax=self.ax,
                                    label=f'{r.concentration_g_per_mL*1000:.3g} mg/mL')
            self.ax.set_title('Excess Rayleigh ratio')
            calibrated = all(r.calibrated for r in rr)
            self._clear_result_table()           # a per-c series, not a scalar result
            self.status.setText(f'Excess Rayleigh ratio for {len(rr)} concentrations.')
            self.flag_label.setText(
                '' if calibrated else
                '⚠ uncalibrated: ΔR is on an arbitrary scale.')
        self._overlay_masked(method, sid)
        self.canvas.draw_idle()
        self.axis_bar.attach(self.ax)      # single-angle method leaves ax None

    def _overlay_masked(self, method: str, sid: str) -> None:
        """Draw the hidden points greyed (hollow) at their true plot positions, so
        you can see what is excluded and toggle it back. Uses the UNMASKED data
        (run_rayleigh) plus the sample's mask."""
        if self.ax is None:
            return
        mask = self.controller.sls_mask(sid, self._fraction)
        if mask.is_empty():
            return
        # Unmasked series from the last run (cached in _on_run.done), not a fresh
        # controller call — the draw path stays off the analysis engine.
        full = self._full_rr.get((sid, self._fraction))
        if full is None:
            return
        xs: List[float] = []
        ys: List[float] = []
        if method in ('zimm', 'berry'):
            berry = (method == 'berry')
            for r in full:
                cc = r.concentration_g_per_mL
                for i, ang in enumerate(r.angles_deg):
                    if not mask.is_masked(cc, float(ang)):
                        continue
                    y = r.kc_over_dR_mol_per_g[i]
                    if not np.isfinite(y) or (berry and y <= 0):
                        continue
                    ys.append(math.sqrt(y) if berry else float(y))
                    xs.append(float(r.q2_nm2[i]) + self._zimm_k * cc)
        elif method in ('debye', 'guinier'):
            cc = self.conc_combo.currentData()
            r = next((x for x in full if x.concentration_g_per_mL == cc), None)
            if r is None:
                return
            for i, ang in enumerate(r.angles_deg):
                if not mask.is_masked(cc, float(ang)):
                    continue
                if method == 'debye':
                    y = r.kc_over_dR_mol_per_g[i]
                else:
                    dR = r.excess_rayleigh_cm_inv[i]
                    if not (np.isfinite(dR) and dR > 0):
                        continue
                    y = math.log(dR)
                if not np.isfinite(y):
                    continue
                xs.append(float(r.q2_nm2[i]))
                ys.append(float(y))
        elif method == 'rayleigh':
            for r in full:
                cc = r.concentration_g_per_mL
                if cc == 0:
                    continue
                for i, ang in enumerate(r.angles_deg):
                    if mask.is_masked(cc, float(ang)) and np.isfinite(
                            r.excess_rayleigh_cm_inv[i]):
                        xs.append(float(r.q2_nm2[i]))
                        ys.append(float(r.excess_rayleigh_cm_inv[i]))
        else:
            return
        if xs:
            self.ax.scatter(xs, ys, s=42, facecolors='none', edgecolors='#b0b0b0',
                            linewidths=1.2, zorder=2)

    # ------------------------------------------------ click-to-mask points ---
    _CLICKABLE = ('zimm', 'berry', 'debye', 'guinier', 'rayleigh')
    _PICK_RADIUS_PX = 12.0

    def _point_coords(self, method: str, sid: str):
        """Every data point for the current method as (c, angle, x_data, y_data),
        for click hit-testing. Masked and unmasked points are both included. Reads
        the cached unmasked series (as of the last run) — no controller call."""
        full = self._full_rr.get((sid, self._fraction))
        if full is None:
            return []
        pts = []
        if method in ('zimm', 'berry'):
            berry = (method == 'berry')
            for r in full:
                cc = r.concentration_g_per_mL
                for i, ang in enumerate(r.angles_deg):
                    y = r.kc_over_dR_mol_per_g[i]
                    if not np.isfinite(y) or (berry and y <= 0):
                        continue
                    yy = math.sqrt(y) if berry else float(y)
                    pts.append((cc, float(ang),
                                float(r.q2_nm2[i]) + self._zimm_k * cc, yy))
        elif method in ('debye', 'guinier'):
            cc = self.conc_combo.currentData()
            r = next((x for x in full if x.concentration_g_per_mL == cc), None)
            if r is not None:
                for i, ang in enumerate(r.angles_deg):
                    if method == 'debye':
                        y = r.kc_over_dR_mol_per_g[i]
                    else:
                        dR = r.excess_rayleigh_cm_inv[i]
                        if not (np.isfinite(dR) and dR > 0):
                            continue
                        y = math.log(dR)
                    if not np.isfinite(y):
                        continue
                    pts.append((cc, float(ang), float(r.q2_nm2[i]), float(y)))
        elif method == 'rayleigh':
            for r in full:
                cc = r.concentration_g_per_mL
                if cc == 0:
                    continue
                for i, ang in enumerate(r.angles_deg):
                    y = r.excess_rayleigh_cm_inv[i]
                    if np.isfinite(y):
                        pts.append((cc, float(ang), float(r.q2_nm2[i]), float(y)))
        return pts

    def _on_canvas_click(self, event) -> None:
        """Toggle the point nearest the click (within a pixel radius)."""
        if (not self._ran or self.sample_id is None or self.ax is None
                or event.inaxes is not self.ax or event.button != 1):
            return
        if getattr(self.nav_toolbar, 'mode', ''):   # pan/zoom active -> ignore
            return
        if runner().is_busy:                 # click masks a point then re-runs
            busy_notice(self)
            return
        method = self.method_combo.currentData()
        if method not in self._CLICKABLE:
            return
        pts = self._point_coords(method, self.sample_id)
        if not pts:
            return
        disp = self.ax.transData.transform([(p[2], p[3]) for p in pts])
        d = np.hypot(disp[:, 0] - event.x, disp[:, 1] - event.y)
        j = int(np.argmin(d))
        if d[j] > self._PICK_RADIUS_PX:
            return
        cc, ang = pts[j][0], pts[j][1]
        frac = self._fraction
        mask = self.controller.sls_mask(self.sample_id, frac)
        if mask.is_point_masked(cc, ang):
            self.controller.unmask_point(self.sample_id, cc, ang, frac)
        elif mask.is_masked(cc, ang):
            return   # hidden by an angle/concentration mask -> use the lists
        else:
            self.controller.mask_point(self.sample_id, cc, ang, frac)
        self._on_run()

    def _fill_result_table(self, rows) -> None:
        """Rebuild the per-method results table from (label, value) pairs (#14)."""
        t = self.result_table
        t.setRowCount(len(rows))
        for r, (label, value) in enumerate(rows):
            t.setItem(r, 0, QtWidgets.QTableWidgetItem(label))
            t.setItem(r, 1, QtWidgets.QTableWidgetItem(value))
        t.setMaximumHeight(28 + 22 * max(len(rows), 1))
        t.resizeColumnsToContents()
        self.status.clear()

    def _clear_result_table(self) -> None:
        self.result_table.setRowCount(0)

    def _summarize_zimm(self, res) -> None:
        mw_mark = '' if res.mw_reliable else '  [unreliable — uncalibrated]'
        mw_se = getattr(res, 'mw_se', None)
        rg_se = getattr(res, 'rg_se', None)
        a2_se = getattr(res, 'a2_se', None)
        self._fill_result_table([
            ('Method', res.method.capitalize()),
            ('Mw (g/mol)', format_pm(res.mw_g_per_mol, mw_se) + mw_mark),
            ('Rg (nm)', format_pm(res.rg_nm, rg_se)),
            ('A₂ (mol·mL/g²)', format_pm(res.a2_mol_mL_per_g2, a2_se)),
            ('R²', _fmt(res.r_squared)),
        ])
        flag = (
            '' if res.calibrated else
            '⚠ uncalibrated: Mw and absolute A₂ are unreliable; Rg and the '
            'calibration-free 2·A₂·Mw remain valid (the scale cancels).')
        # Two-route consistency: Mw can be read off the c→0 line OR the q→0 line, and
        # the two should match; a large gap warns of curvature/extrapolation error
        # (no ± involved). `agree` is the RELATIVE DIFFERENCE between the two, so we
        # phrase it as "differ by X%" — "agreement X%" was ambiguous (feedback A7).
        agree = getattr(res, 'extrapolation_agreement_rel', None)
        mc0 = getattr(res, 'mw_from_c0_g_per_mol', None)
        mq0 = getattr(res, 'mw_from_q0_g_per_mol', None)
        if agree is not None and math.isfinite(agree) and mc0 is not None and mq0 is not None:
            note = (f'Mw from the two extrapolation routes — c→0: {_fmt(mc0)}, '
                    f'q→0: {_fmt(mq0)} g/mol — differ by {agree * 100:.0f}%')
            if agree > 0.10:
                note += ' ⚠ >10% — check curvature/extrapolation'
            flag = (flag + '\n' + note) if flag else note
        if mw_se is not None or a2_se is not None:
            flag = (flag + '\n' if flag else '') + _STAT_CAVEAT.strip()
        self.flag_label.setText(flag)

    @staticmethod
    def _apparent_flag(calibrated: bool) -> str:
        base = '⚠ apparent (single concentration): intercept is 1/Mw + 2A₂c.'
        if not calibrated:
            base += ' Also uncalibrated → Mw unreliable (Rg survives).'
        return base

    # ------------------------------------------------------------ figure ---
    def _setup_axes(self) -> None:
        if self.ax is not None:
            try:
                self.ax.set_xscale('linear')
            except Exception:
                pass
        self.figure.clf()
        self.ax = self.figure.add_subplot(1, 1, 1)

    def _clear_plot(self) -> None:
        if self.ax is not None:
            try:
                self.ax.set_xscale('linear')
            except Exception:
                pass
        self.figure.clf()
        self.ax = None
        self.canvas.draw_idle()
        if hasattr(self, 'axis_bar'):
            self.axis_bar.attach(None)
