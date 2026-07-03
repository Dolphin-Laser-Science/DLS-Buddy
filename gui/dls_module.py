"""
gui/dls_module.py
=================

The DLS tab: analysis only (reads COMMITTED parameters; the Data tab owns editing).
It is organised as an inner set of sub-tabs, each a persistent full-size view with
its own controls, so switching method or measurement never wipes another view:

  * Correlogram  — parametric fits (cumulant / single / double / KWW) shown on a
    four-scale correlogram (lin-log main + lin-lin / log-lin / log-log side stack,
    double-click a side view to promote it) with a residual panel, plus dedicated
    Cumulant and KWW result tables. The raw correlogram is shown as soon as a
    measurement is selected, before any fit.
  * Distribution — NNLS / CONTIN / Lognormal size (or Γ) distributions, with an
    optional NNLS+CONTIN overlay.
  * Γ vs q²       — multi-angle diffusive analysis over the sample's DLS angles.
  * D vs c        — concentration extrapolation over the sample's DLS concentrations.

All analysis goes through the controller (`run_*`); soft flags (PDI > 0.3,
non-convergence, non-diffusive) are GUI overlays, never drawn on the figure. Each
sub-tab caches its last result per measurement/sample (within-session) and restores
it on reselection (no recompute).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

from plotting.plots import (
    plot_correlogram_scaled, plot_distribution,
    plot_gamma_q2, plot_concentration_extrapolation, plot_ddls, _CYCLE,
    annotate_decollided,
    display_factor as _disp_factor, display_unit as _disp_unit,
)
from app import units as U
from gui.plot_controls import (
    AxisControlBar, make_split_panels, make_canvas_expanding,
    make_vertical_plot_stack, attach_residual_resizer,
)
from gui.export_helper import export_to_csv
from gui.help import add_help_to_groupbox, section_header
from gui.theme import ThemedLabel, color as theme_color
from gui.widgets import roomy_tabs, SelectionModel, MeasurementPicker
from gui.worker import BACKGROUND_RUN_TOOLTIP, BUSY_NOTICE, run_when_idle, runner
from analysis.uncertainty import format_pm


# Scale modes for the four correlogram views, named Yscale-Xscale -> (xscale, yscale).
_SCALE_MODES = ['lin-log', 'lin-lin', 'log-lin', 'log-log']
_SCALE_XY = {
    'lin-log': ('log', 'linear'),     # the conventional DLS view
    'lin-lin': ('linear', 'linear'),
    'log-lin': ('linear', 'log'),
    'log-log': ('log', 'log'),
}

class _AnalysisRegion:
    """The shared DLS analysis region (all in seconds), edited on the Correlogram
    tab (numeric fields + draggable markers) and used by every per-measurement DLS
    run: parametric fits use the fit WINDOW [tau_min, tau_max]; distribution fits
    use the window plus a BASELINE estimated as mean(g2-1) over [base_lo, base_hi]."""

    def __init__(self) -> None:
        self.tau_min_s: Optional[float] = None
        self.tau_max_s: Optional[float] = None
        self.base_lo_s: Optional[float] = None
        self.base_hi_s: Optional[float] = None
        self._initialised = False

    def init_from_data(self, tau: np.ndarray) -> None:
        """Seed sensible defaults once: window = full range, baseline = last 25 %
        (matching the engine's default tail estimate)."""
        if self._initialised or tau.size == 0:
            return
        lo, hi = float(tau.min()), float(tau.max())
        self.tau_min_s, self.tau_max_s = lo, hi
        self.base_lo_s = float(np.quantile(tau, 0.75))
        self.base_hi_s = hi
        self._initialised = True

    def force_reseed(self) -> None:
        """Drop the seeded defaults so the next init_from_data re-seeds from data
        (used by the Correlogram tab's 'Reset window + baseline')."""
        self._initialised = False

    def window_kwargs(self) -> Dict[str, float]:
        kw: Dict[str, float] = {}
        if self.tau_min_s is not None:
            kw['tau_min_s'] = self.tau_min_s
        if self.tau_max_s is not None:
            kw['tau_max_s'] = self.tau_max_s
        return kw

    def baseline_value(self, tau: np.ndarray, g2m1: np.ndarray) -> Optional[float]:
        if self.base_lo_s is None or self.base_hi_s is None:
            return None
        lo, hi = sorted((self.base_lo_s, self.base_hi_s))
        m = (tau >= lo) & (tau <= hi)
        return float(np.mean(g2m1[m])) if np.any(m) else None


_CORR_METHODS: List[Tuple[str, str]] = [
    ('Cumulant', 'cumulant'),
    ('Single exponential', 'single'),
    ('Double exponential', 'double'),
    ('Stretched Exponential (KWW)', 'kww'),
]
_DIST_METHODS: List[Tuple[str, str]] = [
    ('Lognormal', 'lognormal'),
    ('NNLS', 'nnls'),
    ('CONTIN', 'contin'),
]


def _fmt(x: Optional[float], sig: int = 3) -> str:
    if x is None or not (isinstance(x, (int, float)) and math.isfinite(x)):
        return 'n/a'
    return f'{x:.{sig}g}'


def _ordinal_peak(i: int) -> str:
    """'1st peak', '2nd peak', … for a 0-based index (positional-peak row labels,
    matching the replicate-averaging peak language)."""
    n = i + 1
    suffix = ('th' if 11 <= (n % 100) <= 13
              else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th'))
    return f'{n}{suffix} peak'


def _widen_row_gap(figure, hspace: float = 0.06) -> None:
    """Add a little extra vertical space between stacked subplots so the draggable
    residual grip line (feedback #5) sits in clear space and doesn't touch a plot.
    Tolerant of matplotlib layout-engine API differences."""
    try:
        engine = figure.get_layout_engine()
        if engine is not None and hasattr(engine, 'set'):
            engine.set(hspace=hspace)
    except Exception:
        pass


def _tau_window_kwargs(min_edit: QtWidgets.QLineEdit, max_edit: QtWidgets.QLineEdit,
                       unit_combo: QtWidgets.QComboBox) -> Dict[str, float]:
    """Read a min/max delay window in the selected unit, converted to seconds.
    Raises ValueError on a non-number or min >= max."""
    unit = unit_combo.currentText()
    kw: Dict[str, float] = {}
    for edit, name in ((min_edit, 'tau_min_s'), (max_edit, 'tau_max_s')):
        text = edit.text().strip()
        if not text:
            continue
        try:
            kw[name] = U.to_canonical('time', float(text), unit)
        except ValueError:
            raise ValueError(f'"{text}" is not a valid delay time.') from None
    if ('tau_min_s' in kw and 'tau_max_s' in kw
            and kw['tau_min_s'] >= kw['tau_max_s']):
        raise ValueError('Delay-window min must be less than max.')
    return kw


def _delay_window_row() -> Tuple[QtWidgets.QWidget, QtWidgets.QLineEdit,
                                 QtWidgets.QLineEdit, QtWidgets.QComboBox]:
    """Build the [min][max][unit] delay-window control (unit defaults to µs)."""
    row = QtWidgets.QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    mn = QtWidgets.QLineEdit(); mn.setPlaceholderText('min')
    mx = QtWidgets.QLineEdit(); mx.setPlaceholderText('max')
    unit = QtWidgets.QComboBox(); unit.addItems(U.unit_options('time'))
    for w in (mn, mx, unit):
        row.addWidget(w)
    holder = QtWidgets.QWidget(); holder.setLayout(row)
    return holder, mn, mx, unit


def _vtable(rows: List[str]) -> QtWidgets.QTableWidget:
    """A compact 2-column (parameter | value) table seeded with '—' values."""
    t = QtWidgets.QTableWidget(len(rows), 2)
    t.horizontalHeader().setVisible(False)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
    t.horizontalHeader().setSectionResizeMode(
        0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
    t.horizontalHeader().setSectionResizeMode(
        1, QtWidgets.QHeaderView.ResizeMode.Stretch)
    for r, name in enumerate(rows):
        t.setItem(r, 0, QtWidgets.QTableWidgetItem(name))
        t.setItem(r, 1, QtWidgets.QTableWidgetItem('—'))
    t.setMaximumHeight(28 + 22 * len(rows))
    return t


def _fill_vtable(table, values) -> None:
    """Set the value column (col 1) of a `_vtable` from a list of strings."""
    for r, v in enumerate(values):
        it = table.item(r, 1)
        if it is not None:
            it.setText(v)


def _reset_vtable(table) -> None:
    """Reset a `_vtable`'s value column to '—'."""
    for r in range(table.rowCount()):
        it = table.item(r, 1)
        if it is not None:
            it.setText('—')


# ===========================================================================
# Multi-measurement co-plotting (shared by the Correlogram & Distribution tabs)
# ===========================================================================

def _meas_label(lm) -> str:
    """A short label for a DLS measurement in the selection list / legend / table."""
    p = lm.committed_params
    user_label = p.get('label') or ''
    name = user_label if user_label else lm.item_id
    parts = [] if user_label else [f'{p.get("polymer_name") or "?"}/{p.get("solvent_name") or "?"}']
    ang = p.get('angle_deg')
    if ang is not None:
        parts.append(f'{ang:g}°')
    conc = p.get('concentration_g_per_mL')
    if conc:
        parts.append(f'{conc * 1000:.3g} mg/mL')
    frac = p.get('mw_fraction')
    if frac:
        parts.append(f'[{frac}]')
    if getattr(lm, 'derived_kind', None) == 'replicate_average':
        parts.append(f'(avg of {len(lm.derived_from or [])})')
    suffix = ' '.join(parts)
    return f'{name}: {suffix}' if suffix else name


def _sample_header(sample) -> str:
    """Bold group header text for a sample in the DLS selection list."""
    poly = sample.polymer_name or '?'
    solv = sample.solvent_name or '?'
    T = sample.temperature_K
    if T and T == T:   # not NaN
        return f'{poly} / {solv} @ {T:g} K'
    return f'{poly} / {solv}'


def _meas_short(lm) -> str:
    """A very short label for a table column header (item id + angle/fraction)."""
    p = lm.committed_params
    bits = [lm.item_id]
    ang = p.get('angle_deg')
    if ang is not None:
        bits.append(f'{ang:g}°')
    frac = p.get('mw_fraction')
    if frac:
        bits.append(f'[{frac}]')
    return ' '.join(bits)


# ===========================================================================
# Replicate-averaging dialogs (driven from the sidebar right-click)
# ===========================================================================

# Display name -> controller method key, in menu order. The distribution methods
# (NNLS/CONTIN/lognormal) yield peaks, averaged positionally (Rh-ascending).
_AVERAGE_METHODS = [
    ('Cumulant', 'cumulant'),
    ('Single exponential', 'single'),
    ('Double exponential', 'double'),
    ('Stretched Exponential (KWW)', 'kww'),
    ('NNLS distribution', 'nnls'),
    ('CONTIN distribution', 'contin'),
    ('Lognormal distribution', 'lognormal'),
]


def ask_average_method(parent) -> Optional[str]:
    """Ask which dynamic fit to run on each replicate. Returns the method key, or
    None if the user cancelled. Cumulant is the default (the ISO 22412 z-average
    sizing basis)."""
    labels = [lbl for lbl, _ in _AVERAGE_METHODS]
    choice, ok = QtWidgets.QInputDialog.getItem(
        parent, 'Average derived results',
        'Fit each replicate with which method?\n'
        '(the mean ± SD/√N of its parameters is reported).\n'
        'Distribution methods report per-peak averages in Rh order; CONTIN over '
        'many replicates is slow.',
        labels, 0, False)
    if not ok:
        return None
    for lbl, key in _AVERAGE_METHODS:
        if lbl == choice:
            return key
    return None


def format_average_summary(summary: dict) -> str:
    """Render the controller's average_dls_results summary as display text.

    Kept Qt-free so it can be unit-tested headlessly; the GUI wraps it in a
    QMessageBox. Handles both the parametric summary (a `parameters` list) and the
    distribution summary (a `peaks` list with a possible count-disagreement warning)."""
    method = summary.get('method', '?')
    n_ok = summary.get('n_fit_ok', 0)
    n_rep = summary.get('n_replicates', 0)
    lines = [f'DLS replicate average — {method}',
             f'{n_ok} of {n_rep} replicate(s) fit successfully.', '']

    peaks = summary.get('peaks')
    if peaks is not None:                          # distribution method
        if not peaks:
            lines.append('No peaks resolved in the replicates.')
        for pk in peaks:
            seen = f"seen in {pk['n_resolved']}/{pk['n_total']} runs"
            rh = format_pm(pk['rh_mean'], pk['rh_sem'], 'nm').strip()
            w = pk['weight_mean']
            wtxt = '' if (w is None or not math.isfinite(w)) else \
                   f", weight ≈ {w * 100:.0f}%"
            lines.append(f"Peak {pk['position']} ({seen}): Rh = {rh}{wtxt}")
        if summary.get('peak_count_warning'):
            lines += ['', f"⚠ {summary['peak_count_warning']}"]
        lines += ['',
                  'Distribution peaks are reported only (not written to the sample); '
                  'their positions are regularization-dependent, so treat the ± as a '
                  'reproducibility spread, not a calibrated error.']
        return '\n'.join(lines)

    for p in summary.get('parameters', []):        # parametric method
        # ± is the SD/√N across replicates (None when < 2 usable fits).
        lines.append(f"{p['name']:<10} = "
                     f"{format_pm(p['mean'], p['sem'], p['unit']).strip()}")
    lines.append('')
    if summary.get('rh_written'):
        lines.append('Rh ± SE written to the sample '
                     '(Cross-Sample / ρ = Rg/Rh will use it).')
    elif summary.get('rh_skip_reason'):
        lines.append(f"Rh not written: {summary['rh_skip_reason']}")
    return '\n'.join(lines)


def show_average_summary(parent, summary: dict) -> None:
    """Show the derived-results averaging summary in a modal box."""
    QtWidgets.QMessageBox.information(
        parent, 'DLS replicate average', format_average_summary(summary))


# The DLS overlay selection + checklist were promoted to the shared, framework-side
# `SelectionModel` + `MeasurementPicker` in `gui/widgets.py` (used by every analysis
# tab now, with real checkboxes and one visual idiom). `DLSModule` builds one shared
# `SelectionModel(colour_cycle=_CYCLE)` so overlay colours stay stable, and the
# Correlogram / Distribution / Summary tabs each embed a `MeasurementPicker` bound to it.
# `_meas_label` / `_sample_header` (above) are injected as the picker's label callables.

# Help shown on the "?" badge of the DLS measurement pickers (how-to-use, doc-rule #8).
_PICKER_HELP = 'Tick the measurements to analyse and overlay.'
_PICKER_BULLETS = [
    'Ticked measurements are fit and co-plotted together.',
    'Grouped by sample — you can tick across samples to compare.',
    '<b>Select all / none</b> toggles the whole list at once.',
    'Selecting in the Workspace sidebar only navigates; the tick boxes here '
    'decide what is analysed.',
]
_SUMMARY_PICKER_BULLETS = [
    'Ticks pick which measurements the table shows when '
    '<b>“Ticked only”</b> is on.',
    'With “Ticked only” off, the table lists every DLS result.',
]


# ===========================================================================
# Correlogram sub-tab (parametric fits + 4-scale views + residuals + tables)
# ===========================================================================

# One results-row set per parametric method. A single results table (feedback
# 2026-06-29 #12) is rebuilt with the selected method's rows, replacing the old
# always-visible Cumulant + KWW tables and the single/double-into-status line.
_RESULT_ROWS = {
    'cumulant': ['Γ (s⁻¹)', 'Rh (nm)', 'PDI', 'μ₂ (s⁻²)', 'order', 'method', 'baseline B'],
    'single':   ['Γ (s⁻¹)', 'Rh (nm)', 'β', 'converged'],
    'double':   ['fast Rh (nm)', 'fast frac', 'slow Rh (nm)', 'slow frac', 'converged'],
    'kww':      ['stretch s', 'τc (s)', '⟨τ⟩ (s)', 'Rh(τc) (nm)', 'Rh(⟨τ⟩) (nm)', 'converged'],
}


class _CorrelogramTab(QtWidgets.QWidget):
    """Parametric-fit view: four correlogram scales + residuals + result tables,
    over ONE or MANY co-plotted measurements (ticked in the checklist). All share
    one τ-window / baseline; each measurement gets a stable colour and a results
    column."""

    def __init__(self, controller, region, selection, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.region = region                    # shared _AnalysisRegion (seconds)
        self.selection = selection              # shared _OverlaySelection
        self.item_id: Optional[str] = None
        self._runnable = False
        self._raw: Dict[str, Tuple] = {}        # iid -> (tau, g2m1) for checked DLS
        self._union_tau: Optional[Tuple[float, float]] = None
        self._cache: Dict[str, Dict] = {}       # iid -> {'key','result'}
        self._run_failures: Dict[str, str] = {} # iid -> reason (last run)
        self._scales = list(_SCALE_MODES)       # [main, side0, side1, side2]
        self._markers: Dict[str, object] = {}   # handle name -> Line2D on main_ax
        self._handle_glyphs: Dict[str, object] = {}  # handle name -> caret Line2D
        self._base_span = None                  # baseline shaded region (Polygon)
        self._drag: Optional[str] = None        # handle being dragged, or None
        self._suppress_fields = False
        self._build_ui()

    def _build_ui(self) -> None:
        _, left, right = make_split_panels(self)

        self.checklist = MeasurementPicker(
            self.controller, self.selection, kinds=('dls',),
            label_fn=_meas_label, header_fn=_sample_header,
            help_text=_PICKER_HELP, help_bullets=_PICKER_BULLETS)
        self.checklist.selectionChanged.connect(self._on_selection_changed)

        box = QtWidgets.QGroupBox('Parametric fit')
        add_help_to_groupbox(box, 'Fit the correlogram to get a size (Rh).', bullets=[
            'Pick a <b>method</b>, set the <b>delay window</b> (the τ range fitted), '
            'then <b>Run fit</b>.',
            'Drag the markers on the plot, or type values. The <b>window</b> handles '
            '(green carets) sit at the <b>top</b>; the <b>baseline</b> handles (grey '
            'carets) at the <b>bottom</b> — so grab the top or bottom half to pick one '
            'when they overlap.',
            'The <b>baseline region</b> sets where g₂−1 → 0 is estimated (used by the '
            'distribution methods).',
        ])
        form = QtWidgets.QFormLayout(box)
        self.method_combo = QtWidgets.QComboBox()
        for label, key in _CORR_METHODS:
            self.method_combo.addItem(label, key)
        self.method_combo.setToolTip(
            'Cumulant: one average size + spread (PDI) — best for narrow, single '
            'populations.\n'
            'Single/Double exponential: one or two distinct decay modes.\n'
            'Stretched Exponential (KWW): a broadened single mode.\n'
            'For full size distributions use the Distribution sub-tab '
            '(NNLS / CONTIN).')
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        form.addRow('Method:', self.method_combo)
        self.order_spin = QtWidgets.QSpinBox()
        self.order_spin.setRange(1, 3)
        self.order_spin.setValue(self.controller.settings.cumulant_order)
        form.addRow('Cumulant order:', self.order_spin)
        win, self.tau_min, self.tau_max, self.tau_unit = _delay_window_row()
        form.addRow('Delay window:', win)
        # Baseline region (used by distribution fits): mean g2-1 over this τ range.
        brow = QtWidgets.QHBoxLayout(); brow.setContentsMargins(0, 0, 0, 0)
        self.base_lo = QtWidgets.QLineEdit(); self.base_lo.setPlaceholderText('low')
        self.base_hi = QtWidgets.QLineEdit(); self.base_hi.setPlaceholderText('high')
        brow.addWidget(self.base_lo); brow.addWidget(self.base_hi)
        bwidget = QtWidgets.QWidget(); bwidget.setLayout(brow)
        form.addRow('Baseline region:', bwidget)
        hint = ThemedLabel('Drag the markers on the plot (window carets on top, '
                           'baseline carets on the bottom), or type values; same unit '
                           'as the delay window. One window applies to every ticked '
                           'measurement.', role='hint', size=10)
        hint.setWordWrap(True)
        form.addRow(hint)
        # Reset the window + baseline back to the defaults (full lag range; baseline
        # = last 25 %) — handy after dragging/zooming (feedback 2026-06-26 #7).
        self.reset_region_button = QtWidgets.QPushButton('Reset window + baseline')
        self.reset_region_button.clicked.connect(self._on_reset_region)
        form.addRow(self.reset_region_button)
        # Two-way sync: editing a field moves its marker.
        for edit in (self.tau_min, self.tau_max, self.base_lo, self.base_hi):
            edit.editingFinished.connect(self._on_field_edit)
        self.tau_unit.currentTextChanged.connect(lambda _t: self._fields_from_region())
        self.run_button = QtWidgets.QPushButton('Run fit')
        self.run_button.clicked.connect(self._on_run)
        form.addRow(self.run_button)

        # One results area for the selected method (feedback 2026-06-29 #12): a single
        # table rebuilt with that method's rows, with Export directly beneath it.
        results_section = QtWidgets.QWidget()
        rlay = QtWidgets.QVBoxLayout(results_section)
        rlay.setContentsMargins(0, 0, 0, 0)
        self.results_label = QtWidgets.QLabel('Results')
        rlay.addWidget(self.results_label)
        self.result_table = _vtable(_RESULT_ROWS['cumulant'])
        rlay.addWidget(self.result_table)
        self.export_button = QtWidgets.QPushButton('Export CSV…')
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._on_export)
        rlay.addWidget(self.export_button)
        rlay.addStretch(1)
        # Vertically resizable control column (feedback #9): drag the grips to size the
        # checklist / fit controls / results panes against each other.
        vstack = make_vertical_plot_stack(
            [self.checklist, box, results_section], sizes=[160, 360, 150],
            min_heights=[80, max(box.sizeHint().height(), 200), 60])
        left.addWidget(vstack, 1)

        self.status = QtWidgets.QLabel('')
        self.status.setWordWrap(True)
        left.addWidget(self.status)
        self.flag_label = ThemedLabel('', role='error', bold=True)
        self.flag_label.setWordWrap(True)
        left.addWidget(self.flag_label)

        self.figure = Figure(figsize=(5.5, 4.6), constrained_layout=True)
        _widen_row_gap(self.figure)              # room for the residual grip line (#5)
        self.canvas = make_canvas_expanding(FigureCanvas(self.figure))
        # Flat 4×2 gridspec (constrained_layout handles a flat grid robustly; a nested
        # subgridspec trips its tick-bbox pass on degenerate log axes). The fit spans
        # the top 3 left rows, the residual the bottom-left row, the 3 side views the
        # top 3 right rows. The fit:residual split is adjustable (draggable residual,
        # feedback 2026-06-29 #9): the first three row heights stay equal (= a) so the
        # main + side views scale together, the 4th carries the residual's share. The
        # residual lives on this one canvas so it stays aligned under the fit.
        self._resid_ratio = 0.22                 # residual share of the left column
        a = (1.0 - self._resid_ratio) / 3.0
        self._gs = self.figure.add_gridspec(
            4, 2, width_ratios=[3, 1],
            height_ratios=[a, a, a, self._resid_ratio])
        self.main_ax = self.figure.add_subplot(self._gs[0:3, 0])
        self.resid_ax = self.figure.add_subplot(self._gs[3, 0])
        self.side_axes = [self.figure.add_subplot(self._gs[i, 1]) for i in range(3)]
        self.nav_toolbar = NavigationToolbar(self.canvas, self)
        self.canvas.mpl_connect('button_press_event', self._on_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('button_release_event', self._on_release)
        # Drag the gap between fit and residual to resize the residual (stays aligned).
        self._resid_resizer = attach_residual_resizer(
            self.canvas, self.figure, self.main_ax, self.resid_ax,
            self._apply_resid_ratio)
        right.addWidget(self.nav_toolbar)
        right.addWidget(self.canvas, 1)

    def _apply_resid_ratio(self, frac: float) -> None:
        """Resize the residual vs the fit (draggable handle). The first three row
        heights stay equal so the fit + side views scale together; the 4th row takes
        the residual's share. Same canvas → the residual stays aligned under the fit;
        constrained_layout re-flows on the redraw."""
        self._resid_ratio = frac
        a = (1.0 - frac) / 3.0
        self._gs.set_height_ratios([a, a, a, frac])
        self.canvas.draw_idle()

    def reseed_from_settings(self) -> None:
        self.order_spin.setValue(self.controller.settings.cumulant_order)

    def showEvent(self, event) -> None:                      # keep selection in sync
        super().showEvent(event)
        self.checklist.refresh()
        self._reload_raw()
        self._redraw()
        self._refresh_tables()
        self._update_status()

    # ---- selection ----
    def set_measurement(self, item_id: Optional[str], runnable: bool) -> None:
        self.item_id = item_id
        self._runnable = runnable
        # Sidebar selection only focuses/navigates; it no longer auto-ticks the
        # measurement for analysis. The "Measurements to plot" checklist is the
        # sole driver of what gets fit/overlaid (feedback 2026-06-26 #A).
        self.checklist.refresh()
        self._reload_raw()
        self.run_button.setEnabled(bool(self._raw))
        self.export_button.setEnabled(self._any_cached())
        self._redraw()
        self._refresh_tables()
        self._update_status()

    @QtCore.Slot()
    def _on_selection_changed(self) -> None:
        self._reload_raw()
        self.run_button.setEnabled(bool(self._raw))
        self.export_button.setEnabled(self._any_cached())
        self._redraw()
        self._refresh_tables()
        self._update_status()

    def _reload_raw(self) -> None:
        """Load the raw correlogram of every ticked DLS measurement and seed/extend
        the shared window from their UNION lag range."""
        self._raw = {}
        taus = []
        ws = self.controller.workspace.measurements
        for iid in self.selection.ids():
            m = ws.get(iid)
            if m is None or m.kind != 'dls':
                continue
            tau = np.asarray(m.raw['delay_times_s'], dtype=float)
            g2 = np.asarray(m.raw['correlogram'], dtype=float)
            self._raw[iid] = (tau, g2)
            taus.append(tau)
        if taus:
            allt = np.concatenate(taus)
            self._union_tau = (float(allt.min()), float(allt.max()))
            self.region.init_from_data(allt)      # seed once (full union range)
        else:
            self._union_tau = None
        self._fields_from_region()

    def _any_cached(self) -> bool:
        key = self.method_combo.currentData()
        return any(self._cache.get(i, {}).get('key') == key
                   for i in self.selection.ids())

    def _fit_for(self, iid: str):
        """The cached result for `iid` IF it was fit with the current method, else
        None (so switching method without re-running drops stale fits)."""
        v = self._cache.get(iid)
        if v and v['key'] == self.method_combo.currentData():
            return v['result']
        return None

    # ---- actions ----
    @QtCore.Slot()
    def _on_method_changed(self) -> None:
        self.order_spin.setEnabled(self.method_combo.currentData() == 'cumulant')
        self.export_button.setEnabled(self._any_cached())
        self._redraw()
        self._refresh_tables()
        self._update_status()

    @QtCore.Slot()
    def _on_run(self) -> None:
        ids = list(self._raw.keys())
        if not ids:
            return
        # Cumulant/exp/KWW fits are fast enough to stay synchronous, but they
        # still write the shared controller.results dict — so they must not run
        # while a background fit is writing it too (invariant 4). Refuse rather
        # than race; the fit is a click away once the worker frees.
        if runner().is_busy:
            self.status.setText(BUSY_NOTICE)
            return
        kw = self.region.window_kwargs()        # the shared window (seconds)
        key = self.method_combo.currentData()
        c = self.controller
        self._run_failures = {}
        for iid in ids:
            try:
                if key == 'cumulant':
                    res = c.run_cumulants(iid, order=self.order_spin.value(), **kw)
                elif key == 'single':
                    res = c.run_single_exponential(iid, **kw)
                elif key == 'double':
                    res = c.run_double_exponential(iid, **kw)
                else:
                    res = c.run_kww(iid, **kw)
                self._cache[iid] = {'key': key, 'result': res}
            except Exception as exc:               # per-measurement; never abort all
                self._run_failures[iid] = str(exc)
                self._cache.pop(iid, None)
        self.export_button.setEnabled(self._any_cached())
        self._redraw()
        self._refresh_tables()
        self._update_status()

    @QtCore.Slot()
    def _on_export(self) -> None:
        key = self.method_combo.currentData()
        cached = [(iid, self._cache[iid]['result']) for iid in self.selection.ids()
                  if self._cache.get(iid, {}).get('key') == key]
        if not cached:
            self.status.setText('Run a fit first.')
            return

        def do_export(path: str) -> str:
            import os
            stem, ext = os.path.splitext(path)
            written = []
            for iid, res in cached:
                p = path if len(cached) == 1 else f'{stem}_{iid}{ext}'
                written.append(self.controller.export_correlogram_fit(iid, res, p))
            return '; '.join(written)

        status = export_to_csv(self, f'correlogram_{key}.csv', do_export)
        if status:
            self.status.setText(status)

    # ---- region fields <-> shared region (seconds) ----
    _HANDLES = [('tau_min', 'tau_min_s'), ('tau_max', 'tau_max_s'),
                ('base_lo', 'base_lo_s'), ('base_hi', 'base_hi_s')]

    def _fields_from_region(self) -> None:
        unit = self.tau_unit.currentText()
        self._suppress_fields = True
        for edit, (_, attr) in zip(
                (self.tau_min, self.tau_max, self.base_lo, self.base_hi),
                self._HANDLES, strict=True):
            x = getattr(self.region, attr)
            edit.setText('' if x is None
                         else f'{U.from_canonical("time", x, unit):.4g}')
        self._suppress_fields = False

    def _on_field_edit(self) -> None:
        if self._suppress_fields:
            return
        unit = self.tau_unit.currentText()

        def parse(edit):
            t = edit.text().strip()
            if not t:
                return None
            try:
                return U.to_canonical('time', float(t), unit)
            except ValueError:
                return 'ERR'

        vals = {attr: parse(edit) for edit, (_, attr) in zip(
            (self.tau_min, self.tau_max, self.base_lo, self.base_hi), self._HANDLES, strict=True)}
        if 'ERR' in vals.values():
            self._fields_from_region()           # revert bad input
            return
        if (vals['tau_min_s'] is not None and vals['tau_max_s'] is not None
                and vals['tau_min_s'] >= vals['tau_max_s']):
            self._fields_from_region()
            return
        if (vals['base_lo_s'] is not None and vals['base_hi_s'] is not None
                and vals['base_lo_s'] >= vals['base_hi_s']):
            self._fields_from_region()
            return
        for attr, v in vals.items():
            setattr(self.region, attr, v)
        self._redraw()

    def _on_reset_region(self) -> None:
        """Restore the window + baseline to the from-data defaults (feedback #7)."""
        self.region.force_reseed()
        self._reload_raw()           # re-seeds the region from the union lag range
        self._redraw()

    # ---- drag the markers on the main correlogram ----
    def _on_press(self, event) -> None:
        # The residual-resize handle gets first refusal on a press in its gap band,
        # so dragging the fit/residual divider never starts a marker drag.
        if self._resid_resizer.consumes_press(event):
            return
        # Double-click a side view → promote it to the main slot.
        if getattr(event, 'dblclick', False):
            for k, sax in enumerate(self.side_axes):
                if event.inaxes is sax:
                    self._scales[0], self._scales[k + 1] = (
                        self._scales[k + 1], self._scales[0])
                    self._redraw()
                    return
            return
        # Otherwise grab the nearest marker handle on the main axes.
        if (event.inaxes is not self.main_ax or event.xdata is None
                or getattr(self.nav_toolbar, 'mode', '')):
            return
        self._drag = self._nearest_handle(event)

    # Which handles live in which half of the plot (feedback #7): window markers carry
    # their carets at the top, baseline markers at the bottom, so a press is disambiguated
    # by y-band even when two markers share an x.
    _TOP_HANDLES = ('tau_min', 'tau_max')
    _BOTTOM_HANDLES = ('base_lo', 'base_hi')

    def _nearest_handle(self, event) -> Optional[str]:
        # The region stores τ in seconds, but the axis is drawn in display units
        # (µs by default), so handle positions are scaled by _disp_factor('time')
        # before being mapped to pixels.
        tfac = _disp_factor('time')
        if event.x is None or event.y is None:
            return None
        # Pick the candidate kind by which half of the axes the press is in
        # (top -> window, bottom -> baseline; see _TOP_HANDLES above).
        try:
            yf = self.main_ax.transAxes.inverted().transform((event.x, event.y))[1]
        except Exception:
            return None
        wanted = self._TOP_HANDLES if yf >= 0.5 else self._BOTTOM_HANDLES
        y_mid = sum(self.main_ax.get_ylim()) / 2.0
        best, best_px = None, 8.0                 # 8-pixel grab tolerance (within the band)
        for name, attr in self._HANDLES:
            if name not in wanted:
                continue
            x = getattr(self.region, attr)
            if x is None:
                continue
            try:
                px = self.main_ax.transData.transform((x * tfac, y_mid))[0]
            except Exception:
                continue
            if abs(px - event.x) < best_px:
                best, best_px = name, abs(px - event.x)
        return best

    def _on_motion(self, event) -> None:
        # (No hover-cursor cue here: the residual resizer manages the canvas cursor on
        # the same motion signal and would override it. The offset carets are the
        # affordance instead.)
        if (self._drag is None or event.inaxes is not self.main_ax
                or event.xdata is None):
            return
        # event.xdata is in display units (µs); the region + clamp work in seconds.
        tfac = _disp_factor('time')
        x = self._clamp_handle(self._drag, float(event.xdata) / tfac)
        attr = dict(self._HANDLES)[self._drag]
        setattr(self.region, attr, x)
        line = self._markers.get(self._drag)
        if line is not None:
            line.set_xdata([x * tfac, x * tfac])   # draw in display units
        glyph = self._handle_glyphs.get(self._drag)
        if glyph is not None:
            glyph.set_xdata([x * tfac])            # caret tracks its line
        self.canvas.draw_idle()                    # move only the line + caret (cheap)

    def _clamp_handle(self, name: str, x: float) -> float:
        if self._union_tau is not None:           # clamp to the union of all ticked
            lo, hi = self._union_tau
            x = min(max(x, lo), hi)
        r = self.region
        if name == 'tau_min' and r.tau_max_s is not None:
            x = min(x, r.tau_max_s * (1 - 1e-9))
        elif name == 'tau_max' and r.tau_min_s is not None:
            x = max(x, r.tau_min_s * (1 + 1e-9))
        elif name == 'base_lo' and r.base_hi_s is not None:
            x = min(x, r.base_hi_s * (1 - 1e-9))
        elif name == 'base_hi' and r.base_lo_s is not None:
            x = max(x, r.base_lo_s * (1 + 1e-9))
        return x

    def _on_release(self, event) -> None:
        if self._drag is None:
            return
        self._drag = None
        self._fields_from_region()
        self._redraw()                           # recreate markers + baseline span

    # Marker styling + offset-handle geometry (feedback #7): window carets at the
    # top, baseline carets at the bottom (the disambiguation described above).
    # _BASE_HANDLE_Y is kept a touch above the bottom so the caret never lands in
    # the residual-resize gap band below main_ax.
    _WIN_COLOUR, _BASE_COLOUR = '#2ca02c', '#888'
    _WIN_HANDLE_Y, _BASE_HANDLE_Y = 0.96, 0.05

    def _draw_markers(self) -> None:
        """Draw the window (dotted, top carets) + baseline (dashed + shaded, bottom
        carets) markers on main_ax, each a full-height line plus an offset grab handle."""
        self._markers = {}
        self._handle_glyphs = {}
        self._base_span = None
        r = self.region
        # The region stores τ in canonical seconds, but the correlogram x-axis is
        # drawn in display units (µs by default, via plot_correlogram_scaled), so
        # every marker position is scaled by _disp_factor('time'). Without this the
        # markers land at the raw-seconds value on a µs axis (e.g. τ_max ≈ 1 s draws
        # at the "1 µs" tick, the "capped at 1 µs" bug).
        tfac = _disp_factor('time')
        # Carets use a blended transform: x in data units, y in axes fraction.
        htrans = self.main_ax.get_xaxis_transform()

        def _caret(x, marker, colour, y):
            return self.main_ax.plot(
                [x], [y], marker=marker, color=colour, markersize=10,
                markeredgewidth=0, transform=htrans, clip_on=False, zorder=6)[0]

        # Only the first line of each kind carries a legend label, so the legend
        # shows one "τ window" and one "baseline region" entry (feedback B4).
        if r.tau_min_s is not None:
            self._markers['tau_min'] = self.main_ax.axvline(
                r.tau_min_s * tfac, color=self._WIN_COLOUR, ls=':', lw=1.4,
                label='τ window (fit range)')
            self._handle_glyphs['tau_min'] = _caret(
                r.tau_min_s * tfac, 'v', self._WIN_COLOUR, self._WIN_HANDLE_Y)
        if r.tau_max_s is not None:
            self._markers['tau_max'] = self.main_ax.axvline(
                r.tau_max_s * tfac, color=self._WIN_COLOUR, ls=':', lw=1.4)
            self._handle_glyphs['tau_max'] = _caret(
                r.tau_max_s * tfac, 'v', self._WIN_COLOUR, self._WIN_HANDLE_Y)
        if r.base_lo_s is not None and r.base_hi_s is not None:
            lo, hi = sorted((r.base_lo_s * tfac, r.base_hi_s * tfac))
            self._base_span = self.main_ax.axvspan(lo, hi, color='#999', alpha=0.12,
                                                   label='baseline region')
            self._markers['base_lo'] = self.main_ax.axvline(
                r.base_lo_s * tfac, color=self._BASE_COLOUR, ls='--', lw=1.2)
            self._handle_glyphs['base_lo'] = _caret(
                r.base_lo_s * tfac, '^', self._BASE_COLOUR, self._BASE_HANDLE_Y)
            self._markers['base_hi'] = self.main_ax.axvline(
                r.base_hi_s * tfac, color=self._BASE_COLOUR, ls='--', lw=1.2)
            self._handle_glyphs['base_hi'] = _caret(
                r.base_hi_s * tfac, '^', self._BASE_COLOUR, self._BASE_HANDLE_Y)

    def _redraw(self) -> None:
        for ax in (self.main_ax, self.resid_ax, *self.side_axes):
            ax.clear()
        if not self._raw:
            self.main_ax.set_title('Tick a measurement to plot')
            self.canvas.draw_idle()
            return
        ws = self.controller.workspace.measurements
        xs, ys = _SCALE_XY[self._scales[0]]
        # main view: overlay every ticked measurement (data + its fit, same colour)
        for iid, (tau, g2) in self._raw.items():
            col = self.selection.colour_for(iid)
            res = self._fit_for(iid)
            ft, fg = ((np.asarray(res.fit_tau_s, dtype=float),
                       np.asarray(res.fitted_g2m1, dtype=float)) if res else (None, None))
            plot_correlogram_scaled(self.main_ax, tau, g2, ft, fg, xscale=xs,
                                    yscale=ys, colour=col, label=_meas_label(ws[iid]))
        self.main_ax.set_title(f'Correlogram — {self._scales[0]}'
                               '   (double-click a side view to promote)')
        self._draw_markers()                     # window + baseline handles
        handles, labels = self.main_ax.get_legend_handles_labels()
        if labels:
            self.main_ax.legend(frameon=False, fontsize=7, loc='best')
        # residual panel: one residual per fit (matching colour). Plot in the same
        # DISPLAY units as the main axis (× display factor) so it lines up under the
        # fit — previously it drew raw seconds on a µs axis and squished to the left.
        tfac = _disp_factor('time')
        for iid in self._raw:
            res = self._fit_for(iid)
            if res is not None:
                self.resid_ax.plot(np.asarray(res.fit_tau_s, dtype=float) * tfac,
                                   np.asarray(res.residuals, dtype=float), '-',
                                   color=self.selection.colour_for(iid), lw=1.0)
        self.resid_ax.axhline(0.0, color='#444', lw=0.6, ls=':')
        self.resid_ax.set_xscale(xs)
        self.resid_ax.set_xlim(self.main_ax.get_xlim())
        self.resid_ax.set_xlabel(rf'Delay time $\tau$ ({_disp_unit("time")})')
        self.resid_ax.set_ylabel('resid.')
        # side views: overlay all measurements at each scale
        for k, sax in enumerate(self.side_axes):
            name = self._scales[k + 1]
            sxs, sys = _SCALE_XY[name]
            for iid, (tau, g2) in self._raw.items():
                col = self.selection.colour_for(iid)
                res = self._fit_for(iid)
                ft, fg = ((np.asarray(res.fit_tau_s, dtype=float),
                           np.asarray(res.fitted_g2m1, dtype=float)) if res
                          else (None, None))
                plot_correlogram_scaled(sax, tau, g2, ft, fg, xscale=sxs, yscale=sys,
                                        compact=True, colour=col)
            sax.set_title(name, fontsize=8)
        self.canvas.draw_idle()

    # ---- tables (one value column per ticked measurement) ----
    def _refresh_tables(self) -> None:
        self._rebuild_tables()
        self._fill_tables()

    def _rebuild_tables(self) -> None:
        ws = self.controller.workspace.measurements
        ids = [i for i in self.selection.ids() if i in ws]
        key = self.method_combo.currentData()
        rows = _RESULT_ROWS[key]
        method_label = {k: l for l, k in _CORR_METHODS}.get(key, 'Method')
        self.results_label.setText(f'{method_label} results')
        table = self.result_table
        table.clear()
        table.setColumnCount(1 + len(ids))
        table.setRowCount(len(rows))
        table.horizontalHeader().setVisible(True)
        table.verticalHeader().setVisible(False)
        table.setHorizontalHeaderLabels(['', *[_meas_short(ws[i]) for i in ids]])
        for col, iid in enumerate(ids, start=1):
            hi = table.horizontalHeaderItem(col)
            if hi is not None:
                hi.setForeground(QtGui.QColor(self.selection.colour_for(iid)))
        table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        for col in range(1, 1 + len(ids)):
            table.horizontalHeader().setSectionResizeMode(
                col, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for r, name in enumerate(rows):
            table.setItem(r, 0, QtWidgets.QTableWidgetItem(name))
            for col in range(1, 1 + len(ids)):
                table.setItem(r, col, QtWidgets.QTableWidgetItem('—'))
        table.setMaximumHeight(34 + 22 * len(rows))

    def _fill_tables(self) -> None:
        key = self.method_combo.currentData()
        ids = [i for i in self.selection.ids()
               if i in self.controller.workspace.measurements]
        for col, iid in enumerate(ids, start=1):
            res = self._fit_for(iid)
            if res is None:
                continue
            self._fill_col(self.result_table, col, self._result_values(key, res))

    @staticmethod
    def _result_values(key: str, res) -> List[str]:
        """The result-table column values, in `_RESULT_ROWS[key]` order."""
        if key == 'cumulant':
            method_lbl = res.method + ('' if res.success else ' (linear fallback)')
            baseline_lbl = _fmt(res.baseline, 4) if res.method == 'nonlinear' else '—'
            return [_fmt(res.gamma_s_inv, 4), _fmt(res.rh_nm),
                    f'{_fmt(res.pdi)} {"✓" if res.pdi_valid else "⚠>0.3"}',
                    _fmt(res.mu2_s_inv2, 3), str(res.order), method_lbl, baseline_lbl]
        if key == 'single':
            return [_fmt(res.mode.gamma_s_inv, 4), _fmt(res.mode.rh_nm),
                    _fmt(res.beta), 'yes' if res.success else 'no']
        if key == 'double':
            return [_fmt(res.mode1.rh_nm), _fmt(res.mode1.amplitude_fraction, 2),
                    _fmt(res.mode2.rh_nm), _fmt(res.mode2.amplitude_fraction, 2),
                    'yes' if res.success else 'no']
        return [_fmt(res.stretch), _fmt(res.tau_c_s, 3), _fmt(res.mean_tau_s, 3),
                _fmt(res.rh_from_tau_c_nm), _fmt(res.rh_from_mean_tau_nm),
                'yes' if res.success else 'no']

    @staticmethod
    def _fill_col(table, col: int, values: List[str]) -> None:
        for r, v in enumerate(values):
            it = table.item(r, col)
            if it is not None:
                it.setText(v)

    # ---- status / flags ----
    def _update_status(self) -> None:
        key = self.method_combo.currentData()
        ws = self.controller.workspace.measurements
        flags = []
        for iid in self.selection.ids():
            lm = ws.get(iid)
            if lm is None:
                continue
            if iid in self._run_failures:
                flags.append(f'⚠ {_meas_short(lm)}: skipped — {self._run_failures[iid]}')
                continue
            res = self._fit_for(iid)
            if res is None:
                continue
            # Every method's numbers now live in the single results table (#12); the
            # status line carries only warnings + the committed-values note.
            _text, flag = self._summary(key, res)
            if flag:
                flags.append(f'{_meas_short(lm)}: {flag}')
        note = ''
        if self.controller.is_dirty() and self._any_cached():
            note = '(ran on last committed values)'
        self.status.setText(note)
        self.flag_label.setText('\n'.join(flags))

    def _summary(self, key: str, r) -> Tuple[str, str]:
        if key == 'cumulant':
            flag = ('' if r.pdi_valid else
                    f'⚠ PDI = {_fmt(r.pdi, 2)} > 0.3: cumulant size unreliable for this '
                    'polydispersity — prefer a distribution method.')
            return 'Cumulant fit (see table).', flag
        if key == 'single':
            return (f'Single exp: Γ = {_fmt(r.mode.gamma_s_inv, 4)} s⁻¹   '
                    f'Rh = {_fmt(r.mode.rh_nm)} nm'), _conv(r.success)
        if key == 'double':
            return (f'Double exp: fast Rh = {_fmt(r.mode1.rh_nm)} nm '
                    f'(f = {_fmt(r.mode1.amplitude_fraction, 2)}),   '
                    f'slow Rh = {_fmt(r.mode2.rh_nm)} nm '
                    f'(f = {_fmt(r.mode2.amplitude_fraction, 2)})'), _conv(r.success)
        return 'KWW fit (see table).', _conv(r.success)


def _conv(success: bool) -> str:
    return ('' if success else
            '⚠ the nonlinear fit did not converge; values are seed estimates.')


# ===========================================================================
# Distribution sub-tab (NNLS / CONTIN / Lognormal)
# ===========================================================================

class _DistributionTab(QtWidgets.QWidget):
    """Size / decay-rate distribution view over ONE or MANY ticked measurements,
    each computed by ONE or MANY chosen methods (Lognormal / NNLS / CONTIN). Every
    (measurement × method) curve is overlaid; the τ-window + baseline are shared with
    the Correlogram tab."""

    def __init__(self, controller, region, selection, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.region = region                    # shared _AnalysisRegion
        self.selection = selection              # shared _OverlaySelection
        self.item_id: Optional[str] = None
        self._runnable = False
        self._results: List[Tuple] = []         # [(iid, method, result), ...] last run
        self._failures: List[str] = []
        # Staleness token for async runs: bumped when the ticked set changes, so
        # a background fit whose inputs are outdated is dropped, not drawn.
        self._run_epoch = 0
        self._build_ui()

    def _build_ui(self) -> None:
        _, left, right = make_split_panels(self)

        self.checklist = MeasurementPicker(
            self.controller, self.selection, kinds=('dls',),
            label_fn=_meas_label, header_fn=_sample_header,
            help_text=_PICKER_HELP, help_bullets=_PICKER_BULLETS)
        self.checklist.selectionChanged.connect(self._on_selection_changed)

        box = QtWidgets.QGroupBox('Distribution')
        add_help_to_groupbox(box, 'Recover a full size distribution (not just one '
                             'average) from the correlogram.', bullets=[
                                 '<b>CONTIN</b> (default): smooth, regularised — the '
                                 'robust general choice.',
                                 '<b>NNLS</b>: sharper but noisier; can split modes.',
                                 '<b>Lognormal</b>: assumes a single skewed peak — '
                                 'least free, most stable.',
                                 'These are distribution-weighted, not z-average — '
                                 'compare against the cumulant Rh.',
                             ])
        form = QtWidgets.QFormLayout(box)
        # Per-method checkboxes (replacing the old single "Overlay NNLS+CONTIN").
        # Adding a method later = one more entry in _DIST_METHODS.
        self.method_checks: Dict[str, QtWidgets.QCheckBox] = {}
        mrow = QtWidgets.QVBoxLayout(); mrow.setContentsMargins(0, 0, 0, 0)
        for label, key in _DIST_METHODS:
            cb = QtWidgets.QCheckBox(label)
            cb.setChecked(key == 'contin')        # CONTIN default (feedback 2026-06-26 #5)
            self.method_checks[key] = cb
            mrow.addWidget(cb)
        mwidget = QtWidgets.QWidget(); mwidget.setLayout(mrow)
        form.addRow('Methods:', mwidget)
        self.axis_combo = QtWidgets.QComboBox()
        self.axis_combo.addItem('Rh (nm)', 'rh')
        self.axis_combo.addItem('Γ (decay rate, 1/s)', 'gamma')
        self.axis_combo.currentIndexChanged.connect(self._on_axis_changed)
        form.addRow('Distribution axis:', self.axis_combo)
        # Rh grid + CONTIN L-curve α: seeded from Settings, overridable per run
        # (these moved here from the Settings tab — they belong next to the fit).
        self.rh_min = QtWidgets.QDoubleSpinBox()
        self.rh_min.setRange(0.01, 1.0e5); self.rh_min.setDecimals(2)
        self.rh_max = QtWidgets.QDoubleSpinBox()
        self.rh_max.setRange(0.01, 1.0e5); self.rh_max.setDecimals(1)
        self.rh_points = QtWidgets.QSpinBox(); self.rh_points.setRange(10, 1000)
        grow = QtWidgets.QHBoxLayout(); grow.setContentsMargins(0, 0, 0, 0)
        for w in (self.rh_min, self.rh_max, self.rh_points):
            grow.addWidget(w)
        gholder = QtWidgets.QWidget(); gholder.setLayout(grow)
        form.addRow('Rh grid (min / max / pts):', gholder)
        self.alpha_min = QtWidgets.QLineEdit()
        self.alpha_max = QtWidgets.QLineEdit()
        arow = QtWidgets.QHBoxLayout(); arow.setContentsMargins(0, 0, 0, 0)
        for w in (self.alpha_min, self.alpha_max):
            arow.addWidget(w)
        aholder = QtWidgets.QWidget(); aholder.setLayout(arow)
        form.addRow('CONTIN α (min / max):', aholder)
        self.reseed_from_settings()
        note = ThemedLabel('Delay window + baseline region are set on the '
                           'Correlogram tab (shared). Each ticked method runs on '
                           'every ticked measurement.', role='hint', size=10)
        note.setWordWrap(True)
        form.addRow(note)
        self.run_button = QtWidgets.QPushButton('Run')
        self.run_button.setToolTip(BACKGROUND_RUN_TOOLTIP)
        self.run_button.clicked.connect(self._on_run)
        form.addRow(self.run_button)
        self.export_button = QtWidgets.QPushButton('Export CSV…')
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._on_export)
        form.addRow(self.export_button)
        # Per-measurement peak results for the ticked curves (feedback #10), mirroring
        # the Correlogram tab's results panel: one coloured column per (measurement ·
        # method), positional-peak rows. Populated in _draw, cleared on selection change.
        results_section = QtWidgets.QWidget()
        rlay = QtWidgets.QVBoxLayout(results_section)
        rlay.setContentsMargins(0, 0, 0, 0)
        self.results_label = QtWidgets.QLabel('Peak results')
        rlay.addWidget(self.results_label)
        self.result_table = QtWidgets.QTableWidget(0, 0)
        self.result_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.result_table.verticalHeader().setVisible(False)
        rlay.addWidget(self.result_table, 1)
        # Vertically resizable control column (feedback #9): drag the grips to size the
        # checklist / controls / results panes against each other.
        vstack = make_vertical_plot_stack(
            [self.checklist, box, results_section], sizes=[170, 300, 150],
            min_heights=[90, max(box.sizeHint().height(), 160), 70])
        left.addWidget(vstack, 1)
        self.status = QtWidgets.QLabel('')
        self.status.setWordWrap(True)
        left.addWidget(self.status)

        self.figure = Figure(figsize=(5.3, 4.4), constrained_layout=True)
        # The distribution + residual do NOT share x (both carry their own x-label), so a
        # generous gap is needed for the grip line to clear the main plot's "Rh (nm)"
        # label (feedback #5: the indicator must not overlap the plots).
        _widen_row_gap(self.figure, hspace=0.20)
        self.canvas = make_canvas_expanding(FigureCanvas(self.figure))
        # Main distribution panel + a residual panel below (like the Correlogram
        # tab). The residual x-axis is the delay time τ, NOT the Rh/Γ axis, so the
        # two do not share x. The residual is height-adjustable via a draggable handle
        # (feedback 2026-06-29 #9); a flat gridspec keeps constrained_layout happy.
        self._resid_ratio = 0.22
        a = (1.0 - self._resid_ratio) / 3.0
        self._gs = self.figure.add_gridspec(4, 1, height_ratios=[a, a, a, self._resid_ratio])
        self.ax = self.figure.add_subplot(self._gs[0:3, 0])
        self.resid_ax = self.figure.add_subplot(self._gs[3, 0])
        right.addWidget(NavigationToolbar(self.canvas, self))
        right.addWidget(self.canvas, 1)
        self._resid_resizer = attach_residual_resizer(
            self.canvas, self.figure, self.ax, self.resid_ax, self._apply_resid_ratio)
        self.axis_bar = AxisControlBar(self.canvas)
        right.addWidget(self.axis_bar)
        self._clear('Tick measurements + methods, then Run.')

    def _apply_resid_ratio(self, frac: float) -> None:
        """Resize the residual vs the distribution panel (draggable handle)."""
        self._resid_ratio = frac
        a = (1.0 - frac) / 3.0
        self._gs.set_height_ratios([a, a, a, frac])
        self.canvas.draw_idle()

    def reseed_from_settings(self) -> None:
        """(Re)seed the Rh grid + α fields from the global Settings defaults. The
        per-run values the user types here always win (run_distribution honours
        explicit kwargs over its settings setdefault)."""
        s = self.controller.settings
        self.rh_min.setValue(s.rh_grid_min_nm)
        self.rh_max.setValue(s.rh_grid_max_nm)
        self.rh_points.setValue(s.rh_grid_points)
        self.alpha_min.setText(f'{s.lcurve_alpha_min:g}')
        self.alpha_max.setText(f'{s.lcurve_alpha_max:g}')

    @staticmethod
    def _as_float(text: str, fallback: float) -> float:
        try:
            return float(text)
        except (TypeError, ValueError):
            return fallback

    def _grid_kwargs(self, method: str) -> Dict:
        """Per-run Rh grid (and, for CONTIN, the L-curve α range) from the fields."""
        s = self.controller.settings
        kw = dict(rh_min_nm=self.rh_min.value(), rh_max_nm=self.rh_max.value(),
                  n_grid=self.rh_points.value())
        if method == 'contin':
            kw['alpha_min'] = self._as_float(self.alpha_min.text().strip(),
                                             s.lcurve_alpha_min)
            kw['alpha_max'] = self._as_float(self.alpha_max.text().strip(),
                                             s.lcurve_alpha_max)
        return kw

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.checklist.refresh()
        self.run_button.setEnabled(self._has_checked())

    def _has_checked(self) -> bool:
        ws = self.controller.workspace.measurements
        return any(i in ws and ws[i].kind == 'dls' for i in self.selection.ids())

    # ---- selection ----
    def set_measurement(self, item_id: Optional[str], runnable: bool) -> None:
        self.item_id = item_id
        self._runnable = runnable
        # Decoupled from sidebar selection (feedback 2026-06-26 #A): the checklist
        # is the only thing that picks measurements for analysis.
        self.checklist.refresh()
        self.run_button.setEnabled(self._has_checked())
        if not self._results:
            self._clear('Tick measurements + methods, then Run.')

    @QtCore.Slot()
    def _on_selection_changed(self) -> None:
        # The checked set changed → previous curves are stale; require a re-Run.
        # Any in-flight background fit is also stale: bump the epoch so its
        # result is discarded on arrival.
        self._run_epoch += 1
        self._results = []
        self._failures = []
        self.export_button.setEnabled(False)
        self.run_button.setEnabled(self._has_checked())
        self._clear('Selection changed — press Run.')

    @QtCore.Slot()
    def _on_axis_changed(self) -> None:
        if self._results:
            self._draw()                          # re-plot cached results on new axis

    def _ticked_methods(self) -> List[Tuple[str, str]]:
        return [(label, key) for label, key in _DIST_METHODS
                if self.method_checks[key].isChecked()]

    def _baseline_kwargs(self, iid: str) -> Dict:
        """Shared window + this measurement's own baseline over the shared region."""
        kw = self.region.window_kwargs()
        m = self.controller.workspace.measurements[iid]
        baseline = self.region.baseline_value(
            np.asarray(m.raw['delay_times_s'], dtype=float),
            np.asarray(m.raw['correlogram'], dtype=float))
        if baseline is not None:
            kw['baseline'] = baseline
        return kw

    @QtCore.Slot()
    def _on_run(self) -> None:
        ws = self.controller.workspace.measurements
        ids = [i for i in self.selection.ids() if i in ws and ws[i].kind == 'dls']
        methods = self._ticked_methods()
        if not ids:
            self.status.setText('Tick at least one measurement.'); return
        if not methods:
            self.status.setText('Tick at least one distribution method.'); return
        # Everything the fit needs is read from the widgets HERE, on the main
        # thread; the thunk below runs on the worker and touches only the
        # controller (the whole method is dispatched regardless of `key`, so a
        # future distribution method inherits background execution for free).
        jobs = [(iid, key, {**self._baseline_kwargs(iid), **self._grid_kwargs(key)})
                for iid in ids for _label, key in methods]
        labels = {iid: _meas_short(ws[iid]) for iid in ids}
        controller = self.controller
        epoch = self._run_epoch

        def work():
            results, failures = [], []
            for iid, key, kw in jobs:
                try:
                    res = controller.run_distribution(iid, key, **kw)
                    results.append((iid, key, res))
                except Exception as exc:           # per (measurement, method)
                    failures.append(f'{labels[iid]}·{key.upper()}: {exc}')
            return results, failures

        def done(payload) -> None:
            if epoch != self._run_epoch:
                return          # selection changed while the fit ran — stale
            self._results, self._failures = payload
            self.export_button.setEnabled(bool(self._results))
            self._draw()

        if runner().try_submit(work, done, description='distribution fit',
                               busy_widgets=(self.run_button,)):
            self.status.setText('Fitting in the background…')
        else:
            self.status.setText(BUSY_NOTICE)

    @staticmethod
    def _as_distribution(result):
        return getattr(result, 'distribution', result)

    @QtCore.Slot()
    def _on_export(self) -> None:
        if not self._results:
            return
        axis = self.axis_combo.currentData()
        items = list(self._results)

        def do_export(path: str) -> str:
            import os
            stem, ext = os.path.splitext(path)
            written = []
            for iid, key, res in items:
                p = path if len(items) == 1 else f'{stem}_{iid}_{key}{ext}'
                written.append(self.controller.export_distribution(res, p, axis))
            return '; '.join(written)

        status = export_to_csv(self, 'distribution.csv', do_export)
        if status:
            self.status.setText(status)

    def _draw(self) -> None:
        self.ax.clear()
        self.resid_ax.clear()
        axis = self.axis_combo.currentData()
        ws = self.controller.workspace.measurements
        single = len(self._results) == 1
        peak_items = []          # (x, y, text, colour) collected across all curves
        for iid, _key, res in self._results:
            d = self._as_distribution(res)
            colour = self.selection.colour_for(iid)
            label = f'{_meas_short(ws[iid])} · {d.method.upper()}'
            plot_distribution(d, ax=self.ax, axis=axis, label=label,
                              colour=colour, fill=single)
            peak_items.extend(self._collect_peak_labels(d, res, axis, colour))
            # Residuals (data - reconstructed g2-1) vs delay time, per curve. Plot in
            # display units (× factor) so the τ axis matches the rest of the app.
            tau = np.asarray(d.fit_tau_s, dtype=float) * _disp_factor('time')
            self.resid_ax.plot(tau, np.asarray(d.residuals, dtype=float),
                               '-', color=colour, lw=1.0)
        # One de-collided pass so peaks that coincide across curves stagger (capped at
        # 6 + "+N more") instead of piling into an unreadable blob (feedback #8).
        annotate_decollided(self.ax, peak_items)
        self.ax.set_title('DLS size / rate distribution')
        self.resid_ax.axhline(0.0, color='0.6', lw=0.8)
        self.resid_ax.set_xscale('log')
        self.resid_ax.set_xlabel(rf'Delay time $\tau$ ({_disp_unit("time")})')
        self.resid_ax.set_ylabel('resid.')
        self.canvas.draw_idle()
        self.axis_bar.attach(self.ax)
        # Status now carries only operational notes; per-result peaks live in the
        # Summary tab (which is also where they persist).
        notes = []
        if sum(1 for _i, k, _r in self._results if k == 'contin') > 3:
            notes.append('CONTIN runs an L-curve per measurement — this can be slow.')
        if getattr(self, '_failures', None):
            notes.append('⚠ skipped: ' + '; '.join(self._failures))
        if self.controller.is_dirty() and self._results:
            notes.append('(ran on last committed values)')
        notes.append('Peak values are also in the Summary tab.')
        self.status.setText('   '.join(notes))
        self._refresh_results()

    def _refresh_results(self) -> None:
        """Fill the peak-results panel from the ticked results — one coloured column per
        (measurement · method), positional-peak rows (feedback #10). Mirrors the
        Correlogram results panel; reuses the same peak source as the plot labels."""
        ws = self.controller.workspace.measurements
        cols = [(iid, key, res) for iid, key, res in self._results if iid in ws]
        peaks_per = [self.controller.distribution_peaks(res) for _i, _k, res in cols]
        max_peaks = max((len(p) for p in peaks_per), default=0)
        n_rows = max(max_peaks, 1)
        table = self.result_table
        table.clear()
        table.setColumnCount(1 + len(cols))
        table.setRowCount(n_rows)
        table.horizontalHeader().setVisible(True)
        table.setHorizontalHeaderLabels(
            ['', *[f'{_meas_short(ws[iid])} · {key.upper()}' for iid, key, _r in cols]])
        for col, (iid, _k, _r) in enumerate(cols, start=1):
            hi = table.horizontalHeaderItem(col)
            if hi is not None:
                hi.setForeground(QtGui.QColor(self.selection.colour_for(iid)))
        table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        for col in range(1, 1 + len(cols)):
            table.horizontalHeader().setSectionResizeMode(
                col, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for r in range(n_rows):
            table.setItem(r, 0, QtWidgets.QTableWidgetItem(_ordinal_peak(r)))
            for col, peaks in enumerate(peaks_per, start=1):
                if r < len(peaks):
                    rh_nm, _gamma, weight = peaks[r]
                    txt = f'{_fmt(rh_nm)} nm · {weight * 100:.0f}%'
                else:
                    txt = '—'
                table.setItem(r, col, QtWidgets.QTableWidgetItem(txt))
        table.setMaximumHeight(34 + 22 * n_rows)

    def _clear_results(self) -> None:
        self.result_table.clear()
        self.result_table.setRowCount(0)
        self.result_table.setColumnCount(0)

    def _collect_peak_labels(self, dist, res, axis: str, colour: str) -> list:
        """Collect this curve's peak labels as (x, y, text, colour) tuples for the
        shared de-collision pass (GUI overlay; kept on saved images too, like the
        Correlogram τ markers). Uses the controller's peak finder so the analysis layer
        is not imported here. y is the curve's height at the peak, from the grid."""
        grid = np.asarray(dist.rh_grid_nm if axis == 'rh'
                          else dist.gamma_grid_s_inv, dtype=float)
        w = np.asarray(dist.weights, dtype=float)
        items = []
        for rh_nm, gamma_s_inv, _weight in self.controller.distribution_peaks(res):
            x = rh_nm if axis == 'rh' else gamma_s_inv
            if x is None or not math.isfinite(x) or grid.size == 0:
                continue
            y = float(w[int(np.argmin(np.abs(grid - x)))])
            txt = f'{_fmt(rh_nm)} nm' if axis == 'rh' else f'{_fmt(gamma_s_inv)} s⁻¹'
            items.append((x, y, txt, colour))
        return items

    def _clear(self, message: str) -> None:
        self.ax.clear()
        self.resid_ax.clear()
        self.ax.set_title(message)
        self.canvas.draw_idle()
        self.axis_bar.attach(self.ax)
        self._clear_results()


# ===========================================================================
# Sample-level sub-tabs (Γ vs q²  and  D vs c)
# ===========================================================================

class _SampleAnalysisTab(QtWidgets.QWidget):
    """A single full-size sample-level plot: Gamma vs q^2, or D vs c."""

    selectionChanged = QtCore.Signal()           # ticked include-set changed (for the mirror)

    def __init__(self, controller, kind: str, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.kind = kind                       # 'gamma_q2' or 'conc_extrap'
        self.item_id: Optional[str] = None
        self._runnable = False
        self._cache: Dict[str, object] = {}      # sample_id -> fitted result (subset)
        self._points: Dict[str, list] = {}       # sample_id -> dls_sample_points rows
        self._included: Dict[str, set] = {}      # sample_id -> ticked item_ids (#11/#12)
        self._suppress_table = False             # re-entrancy guard for checkbox edits
        self._last_sid: Optional[str] = None     # last DLS sample shown (keep-last, #15)
        self._run_epoch = 0                      # async staleness token (sample/tick set)
        self._recompute_pending = False          # a refit is queued for when idle
        self._build_ui()

    def _build_ui(self) -> None:
        _, left, right = make_split_panels(self)

        # ---- controls section ----
        controls = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(controls); cl.setContentsMargins(0, 0, 0, 0)
        note = ('Fits Γ = D q² across the sample\'s DLS angles (current Mw '
                'fraction).' if self.kind == 'gamma_q2' else
                'Extrapolates D(c) → c→0 across the sample\'s DLS concentrations '
                '(current Mw fraction).')
        lbl = ThemedLabel(note, role='hint', size=11); lbl.setWordWrap(True)
        cl.addWidget(lbl)
        # Item 13: make the data source explicit.
        src = ThemedLabel(
            'Γ at each angle (and D = Γ/q²) comes from an internal 2nd-order cumulant '
            'fit of each correlogram — independent of any saved Correlogram/Distribution '
            'result — using the global skip-channels + cumulant method (Settings).',
            role='hint', size=10)
        src.setWordWrap(True); cl.addWidget(src)
        self.run_button = QtWidgets.QPushButton('Run')
        self.run_button.setToolTip(BACKGROUND_RUN_TOOLTIP)
        self.run_button.clicked.connect(self._on_run)
        cl.addWidget(self.run_button)
        self.export_button = QtWidgets.QPushButton('Export CSV…')
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._on_export)
        cl.addWidget(self.export_button)
        self.status = QtWidgets.QLabel('')
        self.status.setWordWrap(True)
        cl.addWidget(self.status)

        # ---- per-measurement table section (tick to include — #11/#12) ----
        tsec = QtWidgets.QWidget()
        tl = QtWidgets.QVBoxLayout(tsec); tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(section_header(
            'Measurements (tick to include in the fit)',
            'Tick which of this sample\'s measurements enter the fit.',
            bullets=[
                'Only this sample\'s DLS measurements appear — it is a single-sample fit.',
                'The fit needs ≥ 2 distinct '
                + ('angles.' if self.kind == 'gamma_q2' else 'concentrations.'),
                'Rows that can\'t be built are greyed and can\'t be ticked.',
                'Ticked rows read blue and light the matching sidebar leaves.']))
        self.table = QtWidgets.QTableWidget(0, 0)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.itemChanged.connect(self._on_table_changed)
        tl.addWidget(self.table, 1)

        # ---- results section (#14) ----
        rsec = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(rsec); rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(QtWidgets.QLabel('Results'))
        self.result_table = _vtable([lbl for lbl, _ in self._result_spec()])
        rl.addWidget(self.result_table)
        self.flag_label = ThemedLabel('', role='error', bold=True)
        self.flag_label.setWordWrap(True)
        rl.addWidget(self.flag_label)

        vstack = make_vertical_plot_stack(
            [controls, tsec, rsec], sizes=[200, 220, 160],
            min_heights=[max(controls.sizeHint().height(), 150), 90, 90])
        left.addWidget(vstack, 1)

        self.figure = Figure(figsize=(5.2, 4.3), constrained_layout=True)
        self.canvas = make_canvas_expanding(FigureCanvas(self.figure))
        self.ax = self.figure.add_subplot(111)
        self._nav = NavigationToolbar(self.canvas, self)
        right.addWidget(self._nav)
        right.addWidget(self.canvas, 1)
        self.axis_bar = AxisControlBar(self.canvas)
        right.addWidget(self.axis_bar)

    def _result_spec(self):
        """(label, value-fn) per result row, for the results _vtable (#14)."""
        df, du = _disp_factor('diffusion'), _disp_unit('diffusion')

        def pm(v, se):
            return format_pm(v * df, (se * df) if se is not None else None)
        if self.kind == 'gamma_q2':
            return [
                (f'D ({du})', lambda r: pm(r.d_m2_s, r.d_se)),
                ('Rh (nm)', lambda r: format_pm(r.rh_nm, r.rh_se)),
                ('R²', lambda r: _fmt(r.r_squared)),
                ('Diffusive?', lambda r: '✓' if r.is_diffusive else '⚠ no'),
                ('angles fitted', lambda r: str(r.q2_m2.size)),
            ]
        return [
            (f'D₀ ({du})', lambda r: pm(r.d0_m2_s, r.d0_se)),
            ('Rh₀ (nm)', lambda r: format_pm(r.rh0_nm, r.rh0_se)),
            ('k_D (mL/g)', lambda r: format_pm(r.kd_mL_per_g, r.kd_se)),
            ('R²', lambda r: _fmt(r.r_squared)),
            ('concs fitted', lambda r: str(r.n_concentrations)),
        ]

    def _sample_id(self) -> Optional[str]:
        if self.item_id is None:
            return None
        return self.controller.sample_id_of(self.item_id)

    def selected_item_ids(self) -> list:
        """The ticked measurements for the current sample (sidebar-mirror contract)."""
        sid = self._sample_id()
        return sorted(self._included.get(sid, set())) if sid else []

    def _fraction(self) -> Optional[str]:
        return self.controller.workspace.measurements[self.item_id].committed_params.get(
            'mw_fraction')

    def set_measurement(self, item_id: Optional[str], runnable: bool) -> None:
        # Single-sample tab following the sidebar focus: focusing a non-DLS measurement
        # keeps the current sample (display/table/plot untouched) rather than blanking
        # it. This is the intended navigation for a single-sample view — you stay on
        # your sample until you focus another DLS one — not a stopgap guard.
        if item_id is not None and not runnable and self._last_sid is not None:
            return
        self.item_id = item_id
        self._runnable = runnable
        self.flag_label.clear()
        sid = self._sample_id()
        if sid != self._last_sid:
            self._run_epoch += 1     # re-pointed: drop any in-flight run's result
        if not (runnable and sid is not None):
            self.table.setRowCount(0)
            self.run_button.setEnabled(False)
            self.export_button.setEnabled(False)
            self._clear_results()
            self._clear('Select a DLS measurement.')
            self.status.clear()
            return
        # Enumerate the sample's per-measurement points (cumulant Γ/D) for the table.
        self._points[sid] = self.controller.dls_sample_points(
            sid, self.kind, self._fraction())
        fresh = {r['item_id'] for r in self._points[sid] if r['ok']}
        prev = self._included.get(sid)
        self._included[sid] = (prev & fresh) if prev is not None else set(fresh)
        self._refresh_table(sid)
        self._update_run_enabled(sid)
        self.export_button.setEnabled(sid in self._cache)
        self._last_sid = sid                      # remember for keep-last (#15)
        if sid in self._cache:
            self._fill_results(self._cache[sid])
            self._draw(self._cache[sid])
        else:
            self._clear_results()
            self._clear('Tick measurements, then Run.')

    # ---- per-measurement table (tick to include — #11/#12) ----
    def _refresh_table(self, sid: str) -> None:
        pts = self._points.get(sid, [])
        inc = self._included.get(sid, set())
        cu = _disp_unit('concentration')
        if self.kind == 'gamma_q2':
            headers = ['', 'Angle (°)', f'c ({cu})',
                       f'Γ ({_disp_unit("decay_rate")})',
                       f'D_app ({_disp_unit("diffusion")})']
        else:
            headers = ['', f'c ({cu})', 'Angle (°)',
                       f'D_app ({_disp_unit("diffusion")})']
        dfc, dfg, dfd = (_disp_factor('concentration'), _disp_factor('decay_rate'),
                         _disp_factor('diffusion'))
        Flag = QtCore.Qt.ItemFlag
        self._suppress_table = True
        try:
            t = self.table
            t.setColumnCount(len(headers))
            t.setHorizontalHeaderLabels(headers)
            t.setRowCount(len(pts))
            for r, row in enumerate(pts):
                chk = QtWidgets.QTableWidgetItem()
                chk.setData(QtCore.Qt.ItemDataRole.UserRole, row['item_id'])
                chk.setFlags(Flag.ItemIsUserCheckable | Flag.ItemIsEnabled
                             if row['ok'] else Flag.ItemIsUserCheckable)
                chk.setCheckState(QtCore.Qt.CheckState.Checked
                                  if (row['ok'] and row['item_id'] in inc)
                                  else QtCore.Qt.CheckState.Unchecked)
                t.setItem(r, 0, chk)
                ang = '—' if row['angle_deg'] is None else f"{row['angle_deg']:g}"
                conc = ('—' if row['concentration_g_per_mL'] is None
                        else f"{row['concentration_g_per_mL'] * dfc:g}")
                gtxt = _fmt(row['gamma_s_inv'] * dfg) if row['ok'] else '—'
                dtxt = _fmt(row['d_app_m2_s'] * dfd) if row['ok'] else '—'
                vals = ([ang, conc, gtxt, dtxt] if self.kind == 'gamma_q2'
                        else [conc, ang, dtxt])
                for c, v in enumerate(vals, start=1):
                    it = QtWidgets.QTableWidgetItem(v)
                    it.setFlags(Flag.ItemIsEnabled)
                    t.setItem(r, c, it)
                self._tint_row(r, row['ok'] and row['item_id'] in inc)
            t.resizeColumnsToContents()
        finally:
            self._suppress_table = False

    def _tint_row(self, r: int, on: bool) -> None:
        """Colour a table row to match the shared idiom: ticked rows read
        `marker_selected` blue, like the sidebar mirror and the DLS picker."""
        col = theme_color(self.table, 'marker_selected') if on else QtGui.QBrush()
        for c in range(self.table.columnCount()):
            it = self.table.item(r, c)
            if it is not None:
                it.setForeground(col)

    def _on_table_changed(self, item) -> None:
        if self._suppress_table or item.column() != 0:
            return
        sid = self._sample_id()
        if sid is None:
            return
        iid = item.data(QtCore.Qt.ItemDataRole.UserRole)
        inc = self._included.setdefault(sid, set())
        checked = item.checkState() == QtCore.Qt.CheckState.Checked
        if checked:
            inc.add(iid)
        else:
            inc.discard(iid)
        self._suppress_table = True    # setForeground re-emits itemChanged — guard it
        self._tint_row(item.row(), checked)
        self._suppress_table = False
        self.selectionChanged.emit()  # repaint the sidebar mirror for the new subset
        self._run_epoch += 1         # ticked set changed: in-flight run is stale
        enough = self._update_run_enabled(sid)
        if sid in self._cache:
            if enough:
                self._recompute(sid)            # live refit on the new subset
            else:
                self._cache.pop(sid, None)
                self.export_button.setEnabled(False)
                self._clear_results()
                self._clear('Tick at least two — then Run.')

    def _distinct_keys(self, sid: str) -> int:
        """Distinct angles (Γq²) / concentrations (D-vs-c) among the ticked ok points
        (the engine needs ≥2 unique to fit)."""
        inc = self._included.get(sid, set())
        by_id = {r['item_id']: r for r in self._points.get(sid, [])}
        keys = set()
        for iid in inc:
            r = by_id.get(iid)
            if not (r and r['ok']):
                continue
            k = (r['angle_deg'] if self.kind == 'gamma_q2'
                 else r['concentration_g_per_mL'])
            if k is not None:
                keys.add(round(float(k), 9))
        return len(keys)

    def _update_run_enabled(self, sid: str) -> bool:
        enough = self._distinct_keys(sid) >= 2
        self.run_button.setEnabled(bool(self._runnable) and enough)
        if not enough:
            word = 'angles' if self.kind == 'gamma_q2' else 'concentrations'
            self.status.setText(f'Tick at least two distinct {word} to run.')
        else:
            self.status.clear()
        return enough

    def _run_engine(self, sid: str, frac, include_ids):
        if self.kind == 'gamma_q2':
            return self.controller.run_gamma_q2(
                sid, fraction=frac, include_ids=include_ids)
        return self.controller.run_concentration_extrapolation(
            sid, fraction=frac, include_ids=include_ids)

    @QtCore.Slot()
    def _on_run(self) -> None:
        if not self._runnable or self.item_id is None:
            return
        sid = self._sample_id()
        if sid is None:
            QtWidgets.QMessageBox.warning(self, 'No sample',
                                          'This measurement is not assigned to a sample.')
            return
        if self._update_run_enabled(sid):
            self._recompute(sid)

    def _recompute(self, sid: str) -> None:
        """Fit over the ticked subset (on the worker thread — it builds + fits
        every included measurement) and refresh the plot + results table."""
        if runner().is_busy:
            # Can't dispatch now. Do NOT bump the epoch (that would drop the
            # in-flight fit with nothing to replace it); instead retry once the
            # worker frees, with whatever the ticked set is by then.
            if not self._recompute_pending:
                self._recompute_pending = True
                run_when_idle(self._flush_recompute)
            self.status.setText(BUSY_NOTICE)
            return
        frac = self._fraction()
        include_ids = set(self._included.get(sid, set()))
        self._run_epoch += 1                # this run supersedes any in flight
        epoch = self._run_epoch

        def done(res) -> None:
            if epoch != self._run_epoch:
                return                      # sample/tick set changed — stale
            self._cache[sid] = res
            self.export_button.setEnabled(True)
            self._fill_results(res)
            self._draw(res)
            self.status.clear()

        def fail(exc: BaseException) -> None:
            QtWidgets.QMessageBox.critical(
                self, 'Analysis failed', f'Could not run this analysis.\n\n{exc}')
            self.status.setText('Analysis failed — see dialog.')

        description = ('Γ vs q² fit' if self.kind == 'gamma_q2'
                       else 'D vs c extrapolation')
        runner().try_submit(
            lambda: self._run_engine(sid, frac, include_ids),
            done, fail, description=description,
            busy_widgets=(self.run_button,))
        self.status.setText('Fitting in the background…')

    def _flush_recompute(self) -> None:
        """Deferred retry of a recompute that was refused while the worker was
        busy — re-runs with the current sample + ticked set (if still valid)."""
        self._recompute_pending = False
        sid = self._sample_id()
        if sid is not None and self._runnable and self._distinct_keys(sid) >= 2:
            self._recompute(sid)

    def _fill_results(self, res) -> None:
        for r, (_label, fn) in enumerate(self._result_spec()):
            try:
                txt = fn(res)
            except Exception:
                txt = '—'
            it = self.result_table.item(r, 1)
            if it is not None:
                it.setText(txt)

    def _clear_results(self) -> None:
        for r in range(self.result_table.rowCount()):
            it = self.result_table.item(r, 1)
            if it is not None:
                it.setText('—')
        self.flag_label.clear()

    @QtCore.Slot()
    def _on_export(self) -> None:
        sid = self._sample_id()
        if sid is None or sid not in self._cache:
            return
        res = self._cache[sid]
        if self.kind == 'gamma_q2':
            name = f'{sid}_gamma_q2.csv'
            fn = self.controller.export_gamma_q2
        else:
            name = f'{sid}_D_vs_c.csv'
            fn = self.controller.export_concentration_extrapolation
        status = export_to_csv(self, name, lambda p: fn(res, p))
        if status:
            self.status.setText(status)

    def _draw(self, res) -> None:
        self.ax.clear()
        sid = self._sample_id()
        if self.kind == 'gamma_q2':
            plot_gamma_q2(res, ax=self.ax)
            self.ax.set_title('Γ vs q² (multi-angle)')
            flag = ('' if res.is_diffusive else
                    '⚠ not purely diffusive: Γ is not linear through the origin in q² '
                    '— Rh is apparent; check for a slow mode or internal motion (qRg>1).')
            if res.d_se is not None:
                flag = (flag + '\n' if flag else '') + '(± statistical only)'
        else:
            plot_concentration_extrapolation(res, ax=self.ax)
            self.ax.set_title('D vs c → infinite dilution')
            flag = ('(± statistical only)'
                    if (res.d0_se is not None or res.kd_se is not None) else '')
        if sid is not None:
            self._grey_points(sid)        # grey the unticked (excluded) ok points
        self.canvas.draw_idle()
        self.axis_bar.attach(self.ax)
        self.flag_label.setText(flag)

    def _grey_points(self, sid: str) -> None:
        """Grey × the ok points that are NOT ticked, from the per-measurement
        enumeration (by item_id, so replicates are disambiguated)."""
        inc = self._included.get(sid, set())
        if self.kind == 'gamma_q2':
            xf, yf, xk, yk = (_disp_factor('scattering_q2'),
                              _disp_factor('decay_rate'), 'q2_m2', 'gamma_s_inv')
        else:
            xf, yf, xk, yk = (_disp_factor('concentration'),
                              _disp_factor('diffusion'),
                              'concentration_g_per_mL', 'd_app_m2_s')
        for row in self._points.get(sid, []):
            if not row['ok'] or row['item_id'] in inc or row[xk] is None:
                continue
            self.ax.plot([row[xk] * xf], [row[yk] * yf], 'x', color='#999999',
                         ms=9, mew=2, zorder=4)

    def _clear(self, message: str) -> None:
        self.ax.clear()
        self.ax.set_title(message)
        self.canvas.draw_idle()
        self.axis_bar.attach(self.ax)


class _DDLSTab(QtWidgets.QWidget):
    """Depolarized DLS (DDLS): rotational diffusion D_r from a sample's VV/VH
    correlogram pairs. Sample-scoped, like the Γ vs q² tab. Tag each correlogram's
    polarisation (VV/VH) in the Data tab; this pairs them by angle and extracts
    D_r = (Γ_VH − Γ_VV)/6, with D_t from the VV channel."""

    _RESULT_ROWS = ['D_r (rad²/s)', 'τ_rot (µs)', 'D_t (m²/s)', 'R_h,t (nm)',
                    'paired angles', 'single-exp (qL)']

    selectionChanged = QtCore.Signal()           # included angle set changed (mirror)

    def __init__(self, controller, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.item_id: Optional[str] = None
        self._runnable = False
        self._cache: Dict[str, tuple] = {}      # sample_id -> (result, info) fitted
        self._full: Dict[str, tuple] = {}       # sample_id -> (result, info) all angles
        # sample_id -> the INCLUDED paired angles (tick to include, like Γ vs q²);
        # the excluded set passed to the engine is derived as paired − included.
        self._included: Dict[str, set] = {}
        self._suppress_ddls = False             # re-entrancy guard for checkbox edits
        self._last_sid: Optional[str] = None    # last DLS sample shown (keep-last, #15)
        self._run_epoch = 0                     # async staleness token
        self._recompute_pending = False         # a refit is queued for when idle
        self._build_ui()

    def _build_ui(self) -> None:
        _, left, right = make_split_panels(self)
        note = ThemedLabel(
            'Pairs the sample\'s VV (polarized) and VH (depolarized) correlograms '
            'by angle and extracts the rotational diffusion coefficient '
            'D_r = (Γ_VH − Γ_VV)/6 (D_t from the VV channel). Tag each correlogram\'s '
            'polarization in the Data tab.', role='hint', size=11)
        note.setWordWrap(True)
        left.addWidget(note)

        # Paired angles with an include checkbox (tick to include in the D_r/D_t fit —
        # the same idiom as Γ vs q²), plus each correlogram's polarisation + pairing.
        left.addWidget(section_header(
            'Paired angles (tick to include in the fit)',
            'Tick which paired angles enter the D_r/D_t fit.',
            bullets=['Only angles with BOTH a VV and a VH correlogram can be paired.',
                     'Untick an angle to drop it (e.g. an outlier); the fit re-runs on '
                     'the ticked angles.',
                     'Ticked angles read blue and light the matching sidebar leaves.']))
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(['', 'Angle (°)', 'Polarization', 'Paired'])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setMaximumHeight(170)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemChanged.connect(self._on_angle_toggled)
        left.addWidget(self.table)

        rod_row = QtWidgets.QHBoxLayout()
        rod_row.setContentsMargins(0, 0, 0, 0)
        rod_row.addWidget(QtWidgets.QLabel('Rod length L (nm):'))
        self.rod_edit = QtWidgets.QLineEdit()
        self.rod_edit.setPlaceholderText('optional — for the qL < 3 check')
        self.rod_edit.setToolTip(
            'Rod length, used only to evaluate qL (the single-exponential validity '
            'guard). Leave blank if unknown.')
        rod_row.addWidget(self.rod_edit)
        rod_w = QtWidgets.QWidget(); rod_w.setLayout(rod_row)
        left.addWidget(rod_w)

        self.run_button = QtWidgets.QPushButton('Run DDLS')
        self.run_button.setToolTip(BACKGROUND_RUN_TOOLTIP)
        self.run_button.clicked.connect(self._on_run)
        left.addWidget(self.run_button)
        self.export_button = QtWidgets.QPushButton('Export CSV…')
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._on_export)
        left.addWidget(self.export_button)
        # Outlier removal (#9): untick an angle above to drop it; unticked angles are
        # greyed on the plot. "Include all" re-ticks every paired angle.
        self.reset_excl_button = QtWidgets.QPushButton('Include all angles')
        self.reset_excl_button.clicked.connect(self._on_include_all)
        left.addWidget(self.reset_excl_button)

        left.addWidget(QtWidgets.QLabel('Results'))
        self.results = _vtable(self._RESULT_ROWS)
        left.addWidget(self.results)
        self.status = ThemedLabel('', role='muted', size=11)
        self.status.setWordWrap(True)
        left.addWidget(self.status)
        self.flag_label = ThemedLabel('', role='error', bold=True)
        self.flag_label.setWordWrap(True)
        left.addWidget(self.flag_label)

        # Shape models (model-dependent dimensions). Framed + captioned so the
        # "assumed shape, not measured" caveat reads clearly vs the D_r/D_t observables.
        self.shape_box = QtWidgets.QGroupBox(
            'Shape models (assumed geometry — not measured)')
        shape_layout = QtWidgets.QVBoxLayout(self.shape_box)
        self.shape_table = _readonly_table(
            ['Model', 'Dimensions', 'Aspect / ratio', 'OK?'], [])
        shape_layout.addWidget(self.shape_table)
        self.shape_caveat = ThemedLabel('', role='hint', size=11)
        self.shape_caveat.setWordWrap(True)
        shape_layout.addWidget(self.shape_caveat)
        left.addWidget(self.shape_box)
        self.shape_box.setVisible(False)
        left.addStretch(1)

        self.figure = Figure(figsize=(5.2, 4.3), constrained_layout=True)
        self.canvas = make_canvas_expanding(FigureCanvas(self.figure))
        self.ax = self.figure.add_subplot(111)
        self._nav = NavigationToolbar(self.canvas, self)
        right.addWidget(self._nav)
        right.addWidget(self.canvas, 1)
        self.axis_bar = AxisControlBar(self.canvas)
        right.addWidget(self.axis_bar)

    def _sample_id(self) -> Optional[str]:
        if self.item_id is None:
            return None
        return self.controller.sample_id_of(self.item_id)

    def set_measurement(self, item_id: Optional[str], runnable: bool) -> None:
        # DDLS is a single-sample tab that follows the sidebar focus. Focusing a
        # non-DLS measurement keeps the current sample rather than blanking it — the
        # intended navigation for a single-sample view (not a stopgap): you stay on
        # your sample until you focus another DLS one.
        if item_id is not None and not runnable and self._last_sid is not None:
            return
        self.item_id = item_id
        self._runnable = runnable
        self.run_button.setEnabled(runnable)
        self.flag_label.clear()
        sid = self._sample_id()
        if sid != self._last_sid:
            self._run_epoch += 1     # re-pointed: drop any in-flight run's result
        self.export_button.setEnabled(runnable and sid is not None and sid in self._cache)
        if runnable and sid is not None:
            self._refresh_table(sid)
            self._last_sid = sid                  # remember for keep-last (#15)
            if sid in self._cache:
                self._draw(*self._cache[sid])
            else:
                self._clear_results()
                self._clear("Run DDLS to pair this sample's VV/VH correlograms.")
        else:
            self.table.setRowCount(0)
            self._clear_results()
            self._clear('Select a DLS measurement.' if not runnable
                        else 'No sample for this measurement.')

    def _clear_results(self) -> None:
        _reset_vtable(self.results)
        self.status.clear()
        self.flag_label.clear()
        self.shape_table.setRowCount(0)
        self.shape_caveat.clear()
        self.shape_box.setVisible(False)

    def _paired_angles(self, sid: str) -> set:
        """The angles that have BOTH a VV and a VH correlogram (fittable)."""
        return {float(r['angle_deg'])
                for r in self.controller.ddls_correlogram_summary(sid) if r['paired']}

    def _excluded_angles(self, sid: str) -> set:
        """Angles to exclude from the fit = paired − included (engine takes excludes)."""
        inc = self._included.get(sid)
        if inc is None:
            return set()                          # not initialised yet → nothing excluded
        return self._paired_angles(sid) - inc

    def _refresh_table(self, sid: str) -> None:
        rows = self.controller.ddls_correlogram_summary(sid)
        paired = {float(r['angle_deg']) for r in rows if r['paired']}
        # Default: every paired angle included. Preserve prior ticks across refreshes
        # (intersect with the still-paired set), like the Γ vs q² tab.
        prev = self._included.get(sid)
        self._included[sid] = (prev & paired) if prev is not None else set(paired)
        inc = self._included[sid]
        self._suppress_ddls = True
        try:
            self.table.setRowCount(len(rows))
            for r, row in enumerate(rows):
                ang = float(row['angle_deg'])
                chk = QtWidgets.QTableWidgetItem()
                chk.setData(QtCore.Qt.ItemDataRole.UserRole, ang)
                # Only paired angles are fittable → only they are checkable.
                if row['paired']:
                    chk.setFlags(QtCore.Qt.ItemFlag.ItemIsUserCheckable
                                 | QtCore.Qt.ItemFlag.ItemIsEnabled)
                    chk.setCheckState(QtCore.Qt.CheckState.Checked if ang in inc
                                      else QtCore.Qt.CheckState.Unchecked)
                else:
                    chk.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
                self.table.setItem(r, 0, chk)
                self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(f"{ang:g}"))
                self.table.setItem(r, 2, QtWidgets.QTableWidgetItem(row['geometry'] or '—'))
                self.table.setItem(r, 3, QtWidgets.QTableWidgetItem('✓' if row['paired'] else ''))
                self._tint_ddls_row(r, row['paired'] and ang in inc)
        finally:
            self._suppress_ddls = False

    def _tint_ddls_row(self, r: int, on: bool) -> None:
        col = theme_color(self.table, 'marker_selected') if on else QtGui.QBrush()
        for c in range(self.table.columnCount()):
            it = self.table.item(r, c)
            if it is not None:
                it.setForeground(col)

    @QtCore.Slot('QTableWidgetItem*')
    def _on_angle_toggled(self, item) -> None:
        if self._suppress_ddls or item.column() != 0:
            return
        sid = self._sample_id()
        if sid is None:
            return
        ang = item.data(QtCore.Qt.ItemDataRole.UserRole)
        inc = self._included.setdefault(sid, set())
        checked = item.checkState() == QtCore.Qt.CheckState.Checked
        if checked:
            inc.add(float(ang))
        else:
            inc.discard(float(ang))
        # An angle has two rows (VV + VH); keep both checkboxes + tints in sync.
        # (setCheckState/setForeground re-emit itemChanged, so guard the sync.)
        self._suppress_ddls = True
        for r in range(self.table.rowCount()):
            it0 = self.table.item(r, 0)
            if (it0 is not None
                    and it0.data(QtCore.Qt.ItemDataRole.UserRole) == ang
                    and it0.flags() & QtCore.Qt.ItemFlag.ItemIsUserCheckable):
                it0.setCheckState(QtCore.Qt.CheckState.Checked if checked
                                  else QtCore.Qt.CheckState.Unchecked)
                self._tint_ddls_row(r, checked)
        self._suppress_ddls = False
        self.selectionChanged.emit()          # repaint the sidebar mirror
        self._run_epoch += 1                  # ticked set changed: in-flight run is stale
        if sid in self._cache:
            self._recompute(sid)              # live refit on the new subset

    @QtCore.Slot()
    def _on_run(self) -> None:
        if not self._runnable or self.item_id is None:
            return
        sid = self._sample_id()
        if sid is None:
            QtWidgets.QMessageBox.warning(self, 'No sample',
                                          'This measurement is not assigned to a sample.')
            return
        rod = None
        txt = self.rod_edit.text().strip()
        if txt:
            try:
                rod = float(txt)
            except ValueError:
                QtWidgets.QMessageBox.warning(
                    self, 'Invalid rod length',
                    'Rod length must be a number in nm, or blank.')
                return
        self._rod_nm = rod
        self._run_epoch += 1     # rod length is a fit input: drop any in-flight fit
        self._recompute(sid)

    def _recompute(self, sid: str) -> None:
        if runner().is_busy:
            # Can't dispatch now; retry when the worker frees (do NOT bump the
            # epoch here — that would drop the in-flight fit with no replacement).
            if not self._recompute_pending:
                self._recompute_pending = True
                run_when_idle(self._flush_recompute)
            self.status.setText(BUSY_NOTICE)
            return
        rod = getattr(self, '_rod_nm', None)
        excl = self._excluded_angles(sid)         # paired − included
        controller = self.controller
        self._run_epoch += 1                # this run supersedes any in flight
        epoch = self._run_epoch

        def work():
            # ONE thunk for both calls, in this order: the second (full) call
            # overwrites results[('ddls', sid)], which ddls_shape reads lazily
            # on the main thread when drawing — running full last keeps that
            # cache warm so no Monte-Carlo re-run happens on the GUI thread.
            res, info = controller.run_ddls(
                sid, rod_length_nm=rod, exclude_angles=excl)
            full = (controller.run_ddls(sid, rod_length_nm=rod)
                    if excl else (res, info))
            return (res, info), full

        def done(payload) -> None:
            if epoch != self._run_epoch:
                return                      # sample/exclusions changed — stale
            (res, info), full = payload
            self._cache[sid] = (res, info)
            self._full[sid] = full
            self.export_button.setEnabled(True)
            self._refresh_table(sid)
            self._draw(res, info)

        def fail(exc: BaseException) -> None:
            QtWidgets.QMessageBox.critical(
                self, 'DDLS failed',
                f'Could not run DDLS.\n\n{exc}\n\nTag at least one angle with BOTH a '
                'VV and a VH correlogram (Data tab → Polarization), then confirm '
                'parameters.')
            self.status.setText('DDLS failed — see dialog.')

        runner().try_submit(work, done, fail,
                            description='DDLS rotational-diffusion fit',
                            busy_widgets=(self.run_button,))
        self.status.setText('Fitting in the background…')

    def _flush_recompute(self) -> None:
        """Deferred retry of a recompute refused while the worker was busy."""
        self._recompute_pending = False
        sid = self._sample_id()
        if sid is not None and self._runnable:
            self._recompute(sid)

    def selected_item_ids(self) -> list:
        """The DLS measurements at the currently-included angles (sidebar-mirror
        contract): the mirror lights the VV/VH correlograms feeding the fit."""
        sid = self._sample_id()
        if sid is None:
            return []
        inc = self._included.get(sid, set())
        out = []
        for lm in self.controller.workspace.sample_measurements(sid, 'dls'):
            ang = lm.committed_params.get('angle_deg')
            if ang is not None and float(ang) in inc:
                out.append(lm.item_id)
        return out

    @QtCore.Slot()
    def _on_include_all(self) -> None:
        sid = self._sample_id()
        if sid is None:
            return
        paired = self._paired_angles(sid)
        if self._included.get(sid) == paired:
            return                                # already all-included
        self._included[sid] = set(paired)
        self._refresh_table(sid)                 # re-tick every row
        self.selectionChanged.emit()
        self._run_epoch += 1     # included set changed: drop any in-flight fit
        if sid in self._cache:
            self._recompute(sid)

    @QtCore.Slot()
    def _on_export(self) -> None:
        sid = self._sample_id()
        if sid is None or sid not in self._cache:
            return
        res, _info = self._cache[sid]
        # Include the shape models when they are available (viscosity present,
        # physical D_r); export the observables alone otherwise.
        try:
            shapes = self.controller.ddls_shape(sid, model='both')
        except Exception:
            shapes = None
        # export_to_csv shows the save dialog and writes the file; its returned
        # status is shown in the (non-error) results label, not the red flag line.
        status = export_to_csv(
            self, f'{sid}_ddls.csv',
            lambda p: self.controller.export_ddls(res, p, shapes=shapes))
        if status:
            self.status.setText(status)

    def _draw(self, res, info) -> None:
        self.ax.clear()
        plot_ddls(res, ax=self.ax)
        sid = self._sample_id()
        excl = self._excluded_angles(sid) if sid else None
        if excl and sid in self._full:
            full = self._full[sid][0]
            qf, gf = _disp_factor('scattering_q2'), _disp_factor('decay_rate')
            q2 = np.asarray(full.q2_m2, float) * qf
            ang = np.asarray(full.angles_deg, float)
            for arr in (full.gamma_vv_s_inv, full.gamma_vh_s_inv):
                ys = np.asarray(arr, float) * gf
                for x, y, a in zip(q2, ys, ang, strict=True):
                    if any(np.isclose(a, e) for e in excl):
                        self.ax.plot([x], [y], 'x', color='#999999', ms=9, mew=2,
                                     zorder=5)
        self.canvas.draw_idle()
        self.axis_bar.attach(self.ax)

        rh = (_fmt(res.rh_t_nm) if math.isfinite(res.rh_t_nm)
              else 'n/a (needs viscosity)')
        paired = f'{len(info["paired_angles"])} · {res.method}'
        if info.get('n_replicate_angles'):
            paired += f' · {info["n_replicate_angles"]} replicate-avg'
        if res.single_exponential_valid is True:
            ql = 'qL < 3 ✓'
        elif res.single_exponential_valid is False:
            ql = 'qL ≥ 3 ⚠ (see shape models)'
        else:
            ql = '—'
        _fill_vtable(self.results, [
            format_pm(res.d_r_rad2_s, res.d_r_se),
            _fmt(res.rotational_time_s * 1e6),
            format_pm(res.d_t_m2_s, res.d_t_se),
            rh, paired, ql])
        self.status.clear()

        bits = []
        if res.notes:
            bits.append(res.notes)
        skipped = []
        if info['vv_only']:
            skipped.append(f"VV-only at {info['vv_only']}°")
        if info['vh_only']:
            skipped.append(f"VH-only at {info['vh_only']}°")
        if info['untagged']:
            skipped.append(f"{len(info['untagged'])} untagged")
        if skipped:
            bits.append('Skipped: ' + '; '.join(skipped) + '.')
        if res.d_r_se is not None or res.d_t_se is not None:
            bits.append('(± statistical: D_t from the multi-angle VV fit, '
                        'D_r from the per-angle spread; excludes calibration.)')
        self.flag_label.setText('  '.join(bits))

        self._draw_shape()

    def _draw_shape(self) -> None:
        """Compute and show both shape models for the current sample's DDLS result."""
        sid = self._sample_id()
        try:
            shapes = self.controller.ddls_shape(sid, model='both')
        except Exception as exc:
            self.shape_table.setRowCount(0)
            self.shape_caveat.setText(f'Shape models unavailable: {exc}')
            self.shape_box.setVisible(True)
            return
        rod, sph = shapes['rod'], shapes['sphere']

        if not rod.converged:
            rod_cells = ['Rod', 'no rigid rod reproduces both D_t and D_r', '', '✗']
        else:
            ok = '✓ in range' if rod.in_valid_range else '⚠ p outside 2–30'
            rod_cells = [
                'Rod',
                f'L = {format_pm(rod.length_nm, rod.length_se)} nm,  '
                f'd = {format_pm(rod.diameter_nm, rod.diameter_se)} nm',
                f'p = {rod.aspect_ratio_p:.2f}', ok]
        sph_cells = [
            'Sphere',
            f'R(D_r) = {format_pm(sph.radius_rot_nm, sph.radius_rot_se)} nm  '
            f'vs  R(D_t)=Rh = {_fmt(sph.radius_trans_nm)} nm',
            f'ratio {sph.sphericity_ratio:.2f}',
            '✓' if sph.is_consistent else '✗']
        t = self.shape_table
        t.setRowCount(2)
        for r, cells in enumerate((rod_cells, sph_cells)):
            for c, txt in enumerate(cells):
                t.setItem(r, c, QtWidgets.QTableWidgetItem(txt))
        t.resizeColumnsToContents()

        verdict = ('Sphere consistent — a near-spherical particle.'
                   if sph.is_consistent else
                   'Sphere inconsistent (ratio ≠ 1) → the rod model is the relevant one.')
        self.shape_caveat.setText(
            verdict + ' Dimensions assume the stated shape — not a direct measurement.')
        self.shape_box.setVisible(True)

    def _clear(self, message: str) -> None:
        self.ax.clear()
        self.ax.set_title(message)
        self.canvas.draw_idle()
        self.axis_bar.attach(self.ax)
        self.shape_box.setVisible(False)


# ===========================================================================
# Container: the DLS tab = header + inner sub-tabs
# ===========================================================================

# ===========================================================================
# Summary sub-tab (durable results table over the snapshot store)
# ===========================================================================

# How each sample-level source_kind is labelled in the table's "Source" column.
_SOURCE_LABELS = {
    'replicate_avg': 'Replicate avg',
    'gamma_q2': 'Γ vs q²',
    'conc_extrap': 'D vs c',
}


def _peak_cell(peaks: List[Tuple[float, float]]) -> str:
    """Render a distribution method's peaks as 'Rh (Int %); Rh (Int %)'."""
    if not peaks:
        return '—'
    return '; '.join(
        f'{_fmt(rh)} ({frac * 100:.0f}%)' if frac is not None else _fmt(rh)
        for rh, frac in peaks)


def _readonly_table(headers: List[str], rows: List[List[str]]) -> QtWidgets.QTableWidget:
    """A read-only QTableWidget filled from a header list + a list of string rows."""
    t = QtWidgets.QTableWidget(len(rows), len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
    for r, cells in enumerate(rows):
        for c, text in enumerate(cells):
            t.setItem(r, c, QtWidgets.QTableWidgetItem(text))
    t.horizontalHeader().setSectionResizeMode(
        QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
    return t


class _SummaryTab(QtWidgets.QWidget):
    """A workspace-wide, durable results table (the DLS Summary view).

    Two stacked tables read the controller's snapshot store: a wide per-measurement
    table (one row per measurement, methods side-by-side) and a compact sample-level
    Rh table (replicate averages + Γ-q² + D-c, with the apparent/thermodynamic
    distinction). Both persist across save/reload because the store does. The left
    panel embeds the SAME shared measurement checklist as the Correlogram and
    Distribution tabs, which here doubles as the table's 'Ticked only' filter."""

    def __init__(self, controller, selection, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.selection = selection              # shared _OverlaySelection
        self._build_ui()

    def _build_ui(self) -> None:
        _, left, right = make_split_panels(self)

        self.checklist = MeasurementPicker(
            self.controller, self.selection, kinds=('dls',),
            label_fn=_meas_label, header_fn=_sample_header, title='Measurements',
            help_text='Tick measurements to filter the Summary table.',
            help_bullets=_SUMMARY_PICKER_BULLETS)
        self.checklist.selectionChanged.connect(self._refresh_tables)
        left.addWidget(self.checklist)

        self.ticked_only = QtWidgets.QCheckBox('Ticked only')
        self.ticked_only.setToolTip('Show only the measurements ticked above '
                                    '(otherwise every measurement that has results).')
        self.ticked_only.toggled.connect(self._refresh_tables)
        left.addWidget(self.ticked_only)

        self.export_button = QtWidgets.QPushButton('Export CSV…')
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._on_export)
        left.addWidget(self.export_button)

        self.status = QtWidgets.QLabel('')
        self.status.setWordWrap(True)
        left.addWidget(self.status)
        left.addStretch(1)

        right.addWidget(QtWidgets.QLabel('<b>Per-measurement results</b>'))
        self.per_meas_table = _readonly_table(['Sample', 'Measurement'], [])
        right.addWidget(self.per_meas_table, 1)
        right.addWidget(QtWidgets.QLabel('<b>Sample-level Rh</b> '
                                         '(averages, Γ–q², D–c)'))
        self.sample_table = _readonly_table(
            ['Sample', 'Source', 'Rh (nm)', '±SE', 'Rh Type', 'From'], [])
        right.addWidget(self.sample_table, 1)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.checklist.refresh()
        self._refresh_tables()

    def set_measurement(self, item_id: Optional[str], runnable: bool) -> None:
        # Workspace-wide tab: it doesn't depend on the sidebar pick, but the
        # DLSModule fans set_measurement out to every sub-tab, so refresh here too.
        self.checklist.refresh()
        self._refresh_tables()

    @QtCore.Slot()
    def _refresh_tables(self) -> None:
        ids = self.selection.ids() if self.ticked_only.isChecked() else None
        meas = self.controller.dls_summary_measurement_rows(ids)
        self._fill_per_meas(meas)
        sample = self.controller.dls_summary_sample_rows()
        self._fill_sample(sample)
        self.export_button.setEnabled(bool(meas) or bool(sample))

    def _fill_per_meas(self, recs: List[Dict]) -> None:
        # Optional columns only when some row populates them, to keep it tidy.
        show_logn = any(r['peaks'].get('lognormal') for r in recs)
        show_dbl = any(r['double_fast'] is not None for r in recs)
        headers = ['Sample', 'Measurement', 'Cumulant Rh', 'PDI', 'Single Rh',
                   'KWW Rh', 'NNLS (Rh / Int %)', 'CONTIN (Rh / Int %)']
        if show_logn:
            headers.append('Lognormal (Rh / Int %)')
        if show_dbl:
            headers.append('Double-exp (Rh fast / slow)')
        rows = []
        for r in recs:
            cells = [
                r['sample_label'], r['measurement_label'],
                _fmt(r['cumulant_rh']), _fmt(r['pdi']), _fmt(r['single_rh']),
                _fmt(r['kww_rh']),
                _peak_cell(r['peaks'].get('nnls', [])),
                _peak_cell(r['peaks'].get('contin', [])),
            ]
            if show_logn:
                cells.append(_peak_cell(r['peaks'].get('lognormal', [])))
            if show_dbl:
                cells.append(
                    f"{_fmt(r['double_fast'])} / {_fmt(r['double_slow'])}"
                    if r['double_fast'] is not None else '—')
            rows.append(cells)
        self._replace_table(self.per_meas_table, headers, rows)

    def _fill_sample(self, recs: List[Dict]) -> None:
        headers = ['Sample', 'Source', 'Rh (nm)', '±SE', 'Rh Type', 'From']
        rows = []
        for r in recs:
            se = r['rh_se']
            rows.append([
                r['sample_label'],
                _SOURCE_LABELS.get(r['source_kind'], r['source_kind']),
                _fmt(r['rh_nm']),
                _fmt(se) if se is not None else '—',
                r['rh_type_label'], r['from_label'],
            ])
        self._replace_table(self.sample_table, headers, rows)

    @staticmethod
    def _replace_table(table: QtWidgets.QTableWidget, headers: List[str],
                       rows: List[List[str]]) -> None:
        table.clear()
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(rows))
        for r, cells in enumerate(rows):
            for c, text in enumerate(cells):
                table.setItem(r, c, QtWidgets.QTableWidgetItem(text))
        table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents)

    @QtCore.Slot()
    def _on_export(self) -> None:
        ids = self.selection.ids() if self.ticked_only.isChecked() else None

        def do_export(path: str) -> str:
            return self.controller.export_dls_summary(path, ids)

        status = export_to_csv(self, 'dls_summary.csv', do_export)
        if status:
            self.status.setText(status)


class DLSModule(QtWidgets.QWidget):
    """DLS analysis tab: a header + persistent sub-tabs. The Correlogram and
    Distribution tabs co-plot any ticked measurements (the former Overlay tab is
    folded in), sharing one analysis region and one measurement selection. A
    Summary tab reads the durable snapshot store written by every DLS run."""

    # Fires whenever the measurements this module has SELECTED for analysis change
    # (the shared overlay set, a sample-tab include change, or a sub-tab switch). The
    # shell connects it to repaint the sidebar's selection mirror.
    selectionChanged = QtCore.Signal()

    def __init__(self, controller, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.item_id: Optional[str] = None
        layout = QtWidgets.QVBoxLayout(self)
        self.header = QtWidgets.QLabel()
        self.header.setWordWrap(True)
        layout.addWidget(self.header)

        self.tabs = roomy_tabs(QtWidgets.QTabWidget())   # roomier tabs so labels don't clip (#3)
        # Shared DLS analysis region (fit window + baseline) AND the shared set of
        # measurements ticked for co-plotting — both used by the Correlogram and
        # Distribution tabs so the two stay consistent.
        self.region = _AnalysisRegion()
        # One shared selection model for the co-plotting tabs (Correlogram / Distribution
        # / Summary). The colour cycle keeps each measurement a stable overlay colour.
        self.selection = SelectionModel(colour_cycle=_CYCLE)
        self.correlogram_tab = _CorrelogramTab(controller, self.region, self.selection)
        self.distribution_tab = _DistributionTab(controller, self.region, self.selection)
        self.gamma_q2_tab = _SampleAnalysisTab(controller, 'gamma_q2')
        self.conc_tab = _SampleAnalysisTab(controller, 'conc_extrap')
        self.ddls_tab = _DDLSTab(controller)
        self.summary_tab = _SummaryTab(controller, self.selection)
        self._subtabs = [self.correlogram_tab, self.distribution_tab,
                         self.gamma_q2_tab, self.conc_tab, self.ddls_tab,
                         self.summary_tab]
        self.tabs.addTab(self.correlogram_tab, 'Correlogram')
        self.tabs.addTab(self.distribution_tab, 'Distribution')
        self.tabs.addTab(self.gamma_q2_tab, 'Γ vs q²')
        self.tabs.addTab(self.conc_tab, 'D vs c')
        self.tabs.addTab(self.ddls_tab, 'DDLS')
        self.tabs.addTab(self.summary_tab, 'Summary')
        self.tabs.setMovable(True)               # drag to reorder sub-tabs (A4)
        layout.addWidget(self.tabs, 1)
        # Re-emit selection changes for the sidebar mirror: the shared overlay model,
        # each sample-scoped sub-tab's own include set, and a sub-tab switch (which
        # changes WHICH selection is active) all bubble up as selectionChanged.
        self.selection.changed.connect(self.selectionChanged)
        self.tabs.currentChanged.connect(lambda _i: self.selectionChanged.emit())
        for tab in (self.gamma_q2_tab, self.conc_tab, self.ddls_tab):
            tab.selectionChanged.connect(self.selectionChanged)
        self.set_measurement(None)

    def selected_item_ids(self) -> list:
        """The measurements the ACTIVE sub-tab currently has selected (for the sidebar
        mirror). Co-plot tabs use the shared overlay set; the Γ-q²/D-vs-c tabs use their
        own per-sample include set; DDLS has none yet."""
        w = self.tabs.currentWidget()
        if w in (self.correlogram_tab, self.distribution_tab, self.summary_tab):
            ws = self.controller.workspace.measurements
            return [i for i in self.selection.ids() if i in ws and ws[i].kind == 'dls']
        if w in (self.gamma_q2_tab, self.conc_tab, self.ddls_tab):
            return w.selected_item_ids()
        return []

    def reseed_from_settings(self) -> None:
        self.correlogram_tab.reseed_from_settings()
        self.distribution_tab.reseed_from_settings()

    def set_measurement(self, item_id: Optional[str]) -> None:
        self.item_id = item_id
        if item_id is None:
            self.header.setText('Pick a measurement in the Workspace list; '
                                'tick measurements in the sub-tabs to analyse.')
            runnable = False
        else:
            kind = self.controller.workspace.measurements[item_id].kind
            runnable = (kind == 'dls')
            if runnable:
                self.header.setText(f'DLS measurement: <b>{item_id}</b>')
            else:
                self.header.setText(
                    f'<b>{item_id}</b> is a {kind.upper()} measurement — '
                    'open it in the SLS tab.')
        for tab in self._subtabs:
            tab.set_measurement(item_id, runnable)
