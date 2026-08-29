"""
gui/dls_module.py
=================

The DLS tab: analysis only (reads COMMITTED parameters; the Data tab owns editing).
It is organized as an inner set of sub-tabs, each a persistent full-size view with
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
non-diffusive, the reliability notes) are GUI overlays, never drawn on the figure.
Exception, owner-ruled: a failed distribution fit's "(failed)" legend suffix IS drawn
-- it states what the curve is (a seed estimate, not a fit), the same
analysis-determined tier as plotting's dropped-point and ceiling-halo markers, so a
saved image cannot present a seed curve as fitted. Each
sub-tab caches its last result per measurement/sample (within-session) and restores
it on reselection (no recompute).
"""

from __future__ import annotations

import math
import re
from collections import namedtuple
from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from plotting.plots import (
    plot_correlogram_scaled, plot_distribution, plot_lcurve,
    plot_weighted_distribution,
    plot_gamma_q2, plot_concentration_extrapolation, plot_ddls,
    plot_mode_map, plot_mode_family_fit, plot_amplitude_vs_q,
    cycle as _plot_cycle, marker_cycle as _marker_cycle,
    linestyle_cycle as _linestyle_cycle,
    annotate_decollided,
    display_factor as _disp_factor, display_unit as _disp_unit,
)
from analysis.optical_model import (
    WEIGHTINGS, WEIGHTING_LABELS, OPTICAL_MODELS, OPTICAL_MODEL_LABELS,
    INTENSITY, RDG, MIE, OpticalModelError,
)
from app import units as U
from app.controller import FIT_ELIGIBLE_QUALITIES
from core.data_models import IDENTITY_FORM_BY_FIELD
from gui.plot_controls import (
    AxisControlBar, make_split_panels, make_canvas_expanding,
    make_vertical_plot_stack, attach_residual_resizer, set_plot_export_name,
    themed_navtoolbar,
)
from gui.export_helper import (
    confirm_overwrite, export_to_csv, sample_suggested_name, suggested_name)
from exporting.export import alpha_selection_note
from gui.field_labels import humanize_field_names
from gui.help import add_help_to_groupbox, section_header, HelpBadge
from gui.plot_overlays import (draw_excluded_markers, reissue_legend_preserving_title,
                               show_plot_placeholder, PLACEHOLDER_GRAY)
from gui.theme import ThemedLabel, color as theme_color, set_flag
from gui.widgets import (roomy_tabs, SelectionModel, MeasurementPicker, GroupTickBar,
                         RichTextDelegate, RichTextHeaderView)
from gui.worker import BACKGROUND_RUN_TOOLTIP, BUSY_NOTICE, run_when_idle, runner
from analysis.uncertainty import (format_fixed_sig, format_pm, pdg_erases_value,
                                  format_value_at_uncertainty)
from analysis import dls as dls_engine
from analysis.dls import CUMULANT_PDI_VALIDITY_LIMIT


# Scale modes for the four correlogram views, named Yscale-Xscale -> (xscale, yscale).
_SCALE_MODES = ['lin-log', 'lin-lin', 'log-lin', 'log-log']
_SCALE_XY = {
    'lin-log': ('log', 'linear'),     # the conventional DLS view
    'lin-lin': ('linear', 'linear'),
    'log-lin': ('linear', 'log'),
    'log-log': ('log', 'log'),
}

# Characteristic decay scales of a ticked correlogram set, in canonical SECONDS, used to
# frame the linear-τ views (and, later, to place a 1/e / inflection guide line — see
# _decay_frame). All taken over the SLOWEST ticked curve.
_DecayFrame = namedtuple('_DecayFrame', 'plateau tau_1e_s tau_end_s data_max_s')


def _decay_frame(curves) -> Optional['_DecayFrame']:
    """Characteristic decay scales, in canonical SECONDS, of an iterable of
    ``(delay_times_s, g2m1)`` correlograms, taken over the SLOWEST curve (largest 1/e
    time — the smallest angle → longest decay, so a frame built from it keeps every faster
    curve visible). Returns None when there is no usable curve. Qt-free so it is unit
    tested directly.

    Fields (see :class:`_DecayFrame`): ``plateau`` (peak g₂−1 of that curve, i.e. β),
    ``tau_1e_s`` (first lag where g₂−1 falls to plateau/e — the robust steep-region
    characteristic time, and exactly where a future 1/e guide line would sit),
    ``tau_end_s`` (last lag still above ~5 % of the plateau — the "mostly decayed" point),
    ``data_max_s`` (largest lag across ALL curves)."""
    best = None                    # (plateau, tau_1e, tau_end) of the slowest curve so far
    data_max = 0.0
    for tau, g2 in curves:
        tau = np.asarray(tau, dtype=float)
        g2 = np.asarray(g2, dtype=float)
        if tau.size == 0 or not np.isfinite(g2).any():
            continue
        data_max = max(data_max, float(np.nanmax(tau)))
        g0 = float(np.nanmax(g2))
        if not (math.isfinite(g0) and g0 > 0):
            continue
        # 1/e crossing sits in the STEEP part of the decay → robust to the noisy tail
        # (first downward crossing); the last 5 %-crossing marks where it is gone.
        below_1e = np.where(g2 <= g0 / math.e)[0]
        tau_1e = float(tau[below_1e[0]]) if below_1e.size else float(tau[-1])
        above_5 = np.where(g2 > 0.05 * g0)[0]
        tau_end = float(tau[above_5[-1]]) if above_5.size else tau_1e
        if best is None or tau_1e > best[1]:      # keep the slowest (largest τ_1e)
            best = (g0, tau_1e, tau_end)
    if best is None or not (math.isfinite(data_max) and data_max > 0):
        return None
    return _DecayFrame(best[0], best[1], best[2], data_max)

class _AnalysisRegion:
    """The shared DLS analysis region (all in seconds), edited on the Correlogram
    tab (numeric fields + draggable markers) and used by every per-measurement DLS
    run: parametric fits use the fit WINDOW [tau_min, tau_max]; distribution fits
    use the window plus a BASELINE measured over [base_lo, base_hi].

    It deliberately does NOT carry the two engine inputs Settings seeds (the skipped
    channels, the cumulant method). Making them per-run controls here was tried and
    reverted: the Cross-Sample candidate builders re-derive the same fit from
    `controller.settings`, so a panel-local value promoted a DIFFERENT Rh than the
    panel showed (30.0 nm shown, 23.73 nm promoted, measured). One source of truth the
    candidate builders read too is the real fix, and it must also settle the cumulant
    ORDER, which diverges the same way today (tier 1 hardcodes `order=2`).
    """

    def __init__(self) -> None:
        self.tau_min_s: Optional[float] = None
        self.tau_max_s: Optional[float] = None
        self.base_lo_s: Optional[float] = None
        self.base_hi_s: Optional[float] = None
        self._initialized = False

    def init_from_data(self, tau: np.ndarray) -> None:
        """Seed sensible defaults once: window = full range, baseline = last 25 %
        (matching the engine's default tail estimate)."""
        if self._initialized or tau.size == 0:
            return
        lo, hi = float(tau.min()), float(tau.max())
        self.tau_min_s, self.tau_max_s = lo, hi
        self.base_lo_s = float(np.quantile(tau, 0.75))
        self.base_hi_s = hi
        self._initialized = True

    def force_reseed(self) -> None:
        """Drop the seeded defaults so the next init_from_data re-seeds from data
        (used by the Correlogram tab's 'Reset window + baseline')."""
        self._initialized = False

    def window_kwargs(self) -> Dict[str, float]:
        kw: Dict[str, float] = {}
        if self.tau_min_s is not None:
            kw['tau_min_s'] = self.tau_min_s
        if self.tau_max_s is not None:
            kw['tau_max_s'] = self.tau_max_s
        return kw

    def baseline_value(self, tau: np.ndarray, g2m1: np.ndarray) -> Optional[float]:
        """The baseline the engine measures over this region's window.

        The arithmetic belongs to the engine (invariant 5): this widget owns the
        handles, `dls_engine.baseline_over_window` owns the number. It used to be
        spelled again here, which made a physics-affecting value a second definition
        that could drift from the engine's.
        """
        if self.base_lo_s is None or self.base_hi_s is None:
            return None
        return dls_engine.baseline_over_window(tau, g2m1,
                                               self.base_lo_s, self.base_hi_s)


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


# Shown when a displayed fit was computed from inputs that have since changed: a
# committed parameter, a tab-owned input (the delay window / baseline / cumulant
# order, the ticked set) or a settings-seeded one (skip_initial_channels, the cumulant
# method, the SE estimator). Each tab's `_run_signature` names the inputs its fit
# reads and is stored with the cached fit; a redisplay compares the two -- the SLS
# tab's `sls_run_signature` shape. Complements the pre-commit '(ran on last
# committed values)' note: this covers the post-commit window, where `is_dirty()` has
# already gone false but the plot still reflects the old inputs.
_STALE_HINT = 'Inputs changed since this fit — press Run to refresh.'

#: `_recompute(sid, fraction=_FOCUSED)`: fit the focused fraction, unless a deferred
#: Run re-enters with the (sample, fraction) it was pressed on.
_FOCUSED = object()


def _ordinal_peak(i: int) -> str:
    """'1st peak', '2nd peak', … for a 0-based index (positional-peak row labels,
    matching the replicate-averaging peak language)."""
    n = i + 1
    suffix = ('th' if 11 <= (n % 100) <= 13
              else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th'))
    return f'{n}{suffix} peak'


def _widen_row_gap(figure, hspace: float = 0.06) -> None:
    """Add a little extra vertical space between stacked subplots so the draggable
    residual grip line sits in clear space and doesn't touch a plot.
    Tolerant of matplotlib layout-engine API differences."""
    try:
        engine = figure.get_layout_engine()
        if engine is not None and hasattr(engine, 'set'):
            engine.set(hspace=hspace)
    except Exception:
        pass


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
    # Col 0 holds quantity names (R_h, D_r, k_D …); render it as rich text so their
    # subscripts show (a plain cell can't — no Unicode subscript for h/w/g/D).
    t.setItemDelegateForColumn(0, RichTextDelegate(t))
    for r, name in enumerate(rows):
        t.setItem(r, 0, QtWidgets.QTableWidgetItem(name))
        t.setItem(r, 1, QtWidgets.QTableWidgetItem('—'))
    # Cap at the REAL content height (the SLS `_fill_result_table` sizing), never a
    # 22-px-per-row literal: rows paint ~30 px at the shipped font, so the literal
    # clipped the last row of every 5-row table at any pane floor.
    row_h = t.verticalHeader().defaultSectionSize()
    t.setMaximumHeight(row_h * max(len(rows), 1) + 2 * t.frameWidth() + 2)
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

def _selection_scoped_name(controller, iids, analysis: str, ext: str = 'csv') -> str:
    """Suggested export name for a ticked-measurement view: carries the sample
    token when every measurement belongs to ONE sample, bare otherwise. The
    Correlogram/Distribution tick set can span samples, where a token would be
    ill-defined — but the common single-sample case must not propose the same
    filename for every sample (review finding)."""
    sids = {controller.sample_id_of(iid) for iid in iids}
    sid = next(iter(sids)) if len(sids) == 1 else None
    if sid is None:
        return suggested_name(None, analysis, ext)
    return sample_suggested_name(controller, sid, analysis, ext)


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
    None if the user canceled. Cumulant is the default (the ISO 22412 z-average
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
                  'Peaks are reported only (not written); positions are '
                  'regularization-dependent, so ± is a reproducibility spread.']
        return '\n'.join(lines)

    for p in summary.get('parameters', []):        # parametric method
        # ± is the SD/√N across replicates (None when < 2 usable fits, and also when the
        # replicates agree exactly -- a zero spread is withheld, not shown as ± 0).
        # `format_pm` renders None and 0.0 identically, so this shows the bare value.
        lines.append(f"{p['name']:<10} = "
                     f"{format_pm(p['mean'], p['sem'], p['unit']).strip()}")
    lines.append('')
    if summary.get('rh_written'):
        lines.append('Rh ± SE written to the sample '
                     '(Cross-Sample / ρ = Rg/Rh will use it).')
        # The replacement disclosure is unconditional -- it never turns off; the
        # optional ask-BEFORE confirm is the Settings checkbox (tier rule:
        # SampleResult docstring).
        if summary.get('rh_replaced_pick'):
            lines.append(f"This run replaced your picked Rh "
                         f"({summary['rh_replaced_pick']}).")
    elif summary.get('rh_skip_reason'):
        lines.append(f"Rh not written: {summary['rh_skip_reason']}")
    else:
        # Belt and braces. The controller's chain is exhaustive, so this line should be
        # unreachable -- but a missing reason must never render as SILENCE about the
        # sample Rh.
        lines.append('Rh not written (no reason recorded).')
    return '\n'.join(lines)


def show_average_summary(parent, summary: dict) -> None:
    """Show the derived-results averaging summary in a modal box."""
    QtWidgets.QMessageBox.information(
        parent, 'DLS replicate average', format_average_summary(summary))


# The DLS overlay selection + checklist were promoted to the shared, framework-side
# `SelectionModel` + `MeasurementPicker` in `gui/widgets.py` (used by every analysis
# tab now, with real checkboxes and one visual idiom). `DLSModule` builds one shared
# `SelectionModel(color_cycle=cycle)` so overlay colors stay stable, and the
# Correlogram / Distribution / Summary tabs each embed a `MeasurementPicker` bound to it.
# `_meas_label` / `_sample_header` (above) are injected as the picker's label callables.

# Help shown on the "?" badge of the DLS measurement pickers (how-to-use).
_PICKER_HELP = 'Tick the measurements to analyze and overlay.'
_PICKER_BULLETS = [
    'Ticked measurements are fit and co-plotted together.',
    'Grouped by sample — you can tick across samples to compare.',
    '<b>Select all / none</b> toggles the whole list at once.',
    'Selecting in the Workspace sidebar only navigates; the tick boxes here '
    'decide what is analyzed.',
]
_SUMMARY_PICKER_BULLETS = [
    'Ticks pick which measurements the table shows when '
    '<b>“Ticked only”</b> is on.',
    'With “Ticked only” off, the table lists every DLS result.',
]

# Bulk "tick all at X" group fields for the DLS measurement pickers:
# concentration shown in mg/mL (stored g/mL), angle in degrees. Shared by
# the Distribution picker (both fields) and the Γ-q²/D-c tabs (one field each). Each is
# (committed-param key, noun, item-formatter-with-unit).
_GROUP_FIELD_CONC = ('concentration_g_per_mL', 'concentration',
                     lambda c: f'{c * 1000:g} mg/mL')
_GROUP_FIELD_ANGLE = ('angle_deg', 'angle', lambda a: f'{a:g}°')
# "Same value?" for a group tick goes through THE shared field->form registry
# (core/data_models.IDENTITY_FORM_BY_FIELD), so the GUI's matching can never
# disagree with the distinctness gates for the same physical quantity.
_IDENTITY_FORMS = IDENTITY_FORM_BY_FIELD
# The cross-sample pickers (Correlogram + Distribution) offer both group-tick fields, so
# their help gains one extra bullet over the bare picker.
_GROUP_PICKER_BULLETS = _PICKER_BULLETS + [
    '<b>Tick all at concentration / angle</b> ticks every listed measurement '
    'sharing the chosen value (concentration in mg/mL, angle in °).',
]


# ===========================================================================
# Correlogram sub-tab (parametric fits + 4-scale views + residuals + tables)
# ===========================================================================

# One results-row set per parametric method. A single results table is rebuilt
# with the selected method's rows, replacing the old
# always-visible Cumulant + KWW tables and the single/double-into-status line.
#
# Every Rh row names its kind in the label: one angle at one concentration is APPARENT
# (invariant 7), and the cumulant Rh is additionally a z-average -- the Distribution tab
# says "distribution-weighted, not z-average" of its peaks, so this table must say the
# other half where the number is read (ui_style_guide R5.3: never hidden).
_RESULT_ROWS = {
    'cumulant': ['Γ (s⁻¹)', 'R<sub>h</sub> (nm, z-average, apparent)', 'PDI', 'μ₂ (s⁻²)',
                 'order', 'method', 'baseline B'],
    'single':   ['Γ (s⁻¹)', 'R<sub>h</sub> (nm, apparent)', 'β', 'converged'],
    'double':   ['fast R<sub>h</sub> (nm, apparent)', 'fast frac',
                 'slow R<sub>h</sub> (nm, apparent)', 'slow frac', 'converged'],
    'kww':      ['stretch s', 'τ<sub>c</sub> (s)', '⟨τ⟩ (s)',
                 'R<sub>h</sub>(τ<sub>c</sub>) (nm, apparent)',
                 'R<sub>h</sub>(⟨τ⟩) (nm, apparent)', 'converged'],
}


class _CorrelogramTab(QtWidgets.QWidget):
    """Parametric-fit view: four correlogram scales + residuals + result tables,
    over ONE or MANY co-plotted measurements (ticked in the checklist). All share
    one τ-window / baseline; each measurement gets a stable color and a results
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
        self._cache: Dict[str, Dict] = {}       # iid -> {'key','result','sig'}
        self._run_failures: Dict[str, str] = {} # iid -> reason (last run)
        self._scales = list(_SCALE_MODES)       # [main, side0, side1, side2]
        self._markers: Dict[str, object] = {}   # handle name -> Line2D on main_ax
        self._handle_glyphs: Dict[str, object] = {}  # handle name -> caret Line2D
        self._placeholder = None                # empty-state Text artist (nothing ticked)
        self._base_span = None                  # baseline shaded region (Polygon)
        self._drag: Optional[str] = None        # handle being dragged, or None
        self._suppress_fields = False
        self._build_ui()

    def _build_ui(self) -> None:
        splitter, left, right = make_split_panels(self, left_min_width=300,
                                                  sizes=(360, 740))

        self.checklist = MeasurementPicker(
            self.controller, self.selection, kinds=('dls',),
            label_fn=_meas_label, header_fn=_sample_header,
            help_text=_PICKER_HELP, help_bullets=_GROUP_PICKER_BULLETS,
            group_fields=(_GROUP_FIELD_CONC, _GROUP_FIELD_ANGLE))
        self.checklist.selectionChanged.connect(self._on_selection_changed)

        box = QtWidgets.QGroupBox('Parametric Fit')
        add_help_to_groupbox(box, 'Fit the correlogram to get a size (Rh).', bullets=[
            'Pick a <b>method</b>, set the <b>delay window</b> (the τ range fitted), '
            'then <b>Run</b>.',
            'Drag the markers on the plot, or type values (same unit as the delay '
            'window; one window applies to every ticked measurement). The '
            '<b>window</b> handles (green carets) sit at the <b>top</b>; the '
            '<b>baseline</b> handles (purple carets) at the <b>bottom</b> — so grab '
            'the top or bottom half to pick one when they overlap.',
            'Moving the window or baseline, or changing the method or cumulant '
            'order, after a Run flags the shown fit stale — press <b>Run</b> again to '
            'refresh it.',
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
            '(NNLS / CONTIN).\n'
            'A decay that does not finish inside the delay window is flagged (⚠ beside '
            'the Rh), because the rate — and the size from it — is then an extrapolation '
            'past the end of the measurement: measure a longer delay range. '
            'Single/Double exponential ask it of each mode\'s rate, '
            'so the flag lands on THAT mode only; Cumulant and KWW ask it of the '
            'correlogram itself.\n'
            'Every method also flags an Rh outside the range DLS can establish — above '
            '~5 µm (10 µm diameter) sedimentation outweighs Brownian motion, below 0.2 nm '
            'the fit chased a short-lag artifact — and the cumulant flags a PDI above '
            '0.3. See the Theory-and-Equations-Guide §3.1.')
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        form.addRow('Method:', self.method_combo)
        self.order_spin = QtWidgets.QSpinBox()
        self.order_spin.setRange(1, 3)
        self.order_spin.setValue(self.controller.settings.cumulant_order)
        self.order_spin.setToolTip(
            'Degree of the cumulant polynomial fit of ln(g₂−1): 1 = size (D) only, '
            '2 adds PDI (polydispersity), 3 adds skew. Applies to the cumulant method '
            'only — greyed out for the other methods.')
        form.addRow('Cumulant order:', self.order_spin)
        # The order is a fit input: changing it beside a shown fit flags that fit stale.
        self.order_spin.valueChanged.connect(lambda _v: self._update_status())
        win, self.tau_min, self.tau_max, self.tau_unit = _delay_window_row()
        form.addRow('Delay window:', win)
        # Baseline region (used by distribution fits): mean g2-1 over this τ range.
        brow = QtWidgets.QHBoxLayout(); brow.setContentsMargins(0, 0, 0, 0)
        self.base_lo = QtWidgets.QLineEdit(); self.base_lo.setPlaceholderText('low')
        self.base_hi = QtWidgets.QLineEdit(); self.base_hi.setPlaceholderText('high')
        brow.addWidget(self.base_lo); brow.addWidget(self.base_hi)
        bwidget = QtWidgets.QWidget(); bwidget.setLayout(brow)
        form.addRow('Baseline region:', bwidget)
        # Reset the window + baseline back to the defaults (full lag range; baseline
        # = last 25 %) — handy after dragging/zooming.
        self.reset_region_button = QtWidgets.QPushButton('Reset window + baseline')
        self.reset_region_button.clicked.connect(self._on_reset_region)
        form.addRow(self.reset_region_button)
        # Two-way sync: editing a field moves its marker.
        for edit in (self.tau_min, self.tau_max, self.base_lo, self.base_hi):
            edit.editingFinished.connect(self._on_field_edit)
        self.tau_unit.currentTextChanged.connect(lambda _t: self._fields_from_region())
        self.run_button = QtWidgets.QPushButton('Run')
        self.run_button.clicked.connect(self._on_run)
        form.addRow(self.run_button)

        # One results area for the selected method: a single
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
        # Vertically resizable control column: drag the grips to size the
        # checklist / fit controls / results panes against each other.
        vstack = make_vertical_plot_stack(
            [self.checklist, box, results_section], sizes=[160, 240, 160],
            min_heights=[80, 180, 60])
        left.addWidget(vstack, 1)

        # The status line and the reliability flag sit UNDER the scrolled control
        # column, not inside it: a flag is an always-visible label (style guide §5),
        # and at the shipped window size the column was taller than its viewport,
        # which put the results table and the ⚠ below the fold. Pinning the two here
        # also takes the word-wrapping labels out of the scrolled layout, so the
        # scroll area sizes the column to its minimums (which fit) rather than to
        # the widgets' natural heights (which did not).
        self.status = QtWidgets.QLabel('')
        self.status.setWordWrap(True)
        self.flag_label = ThemedLabel('', role='error', bold=True)
        self.flag_label.setWordWrap(True)
        pane = QtWidgets.QWidget()
        pane_layout = QtWidgets.QVBoxLayout(pane)
        pane_layout.setContentsMargins(0, 0, 0, 0)
        pane_layout.addWidget(splitter.widget(0), 1)   # re-parents the scroll area
        pane_layout.addWidget(self.status)
        pane_layout.addWidget(self.flag_label)
        splitter.insertWidget(0, pane)
        splitter.setStretchFactor(0, 0)          # the control pane keeps its width
        splitter.setSizes([360, 740])

        self.figure = Figure(figsize=(5.5, 4.6), constrained_layout=True)
        _widen_row_gap(self.figure)              # room for the residual grip line
        self.canvas = make_canvas_expanding(FigureCanvas(self.figure))
        # Flat 4×2 gridspec (constrained_layout handles a flat grid robustly; a nested
        # subgridspec trips its tick-bbox pass on degenerate log axes). The fit spans
        # the top 3 left rows, the residual the bottom-left row, the 3 side views the
        # top 3 right rows. The fit:residual split is adjustable (draggable residual):
        # the first three row heights stay equal (= a) so the
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
        self.nav_toolbar = themed_navtoolbar(self.canvas, self, self.controller)
        self.canvas.mpl_connect('button_press_event', self._on_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('button_release_event', self._on_release)
        # Drag the gap between fit and residual to resize the residual (stays aligned).
        self._resid_resizer = attach_residual_resizer(
            self.canvas, self.figure, self.main_ax, self.resid_ax,
            self._apply_resid_ratio)
        right.addWidget(self.nav_toolbar)
        right.addWidget(self.canvas, 1)
        # The "promote a side view" instruction lives here, not in the axis title: the
        # main axis is narrow (3/4 width minus the side stack), so a long title overflowed
        # into the y-label gutter and clipped its leading "C".
        promote_hint = ThemedLabel(
            'Double-click a side view to promote it to the main plot.',
            role='hint', size=10)
        promote_hint.setWordWrap(True)
        right.addWidget(promote_hint)

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
        # sole driver of what gets fit/overlaid.
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

    def _any_stale(self) -> bool:
        """True if any co-plotted, method-matching cached fit was computed from
        committed params that have since changed."""
        key = self.method_combo.currentData()
        for iid in self.selection.ids():
            v = self._cache.get(iid)
            if v and v.get('key') == key and v.get('sig') != self._run_signature(iid, key):
                return True
        return False

    def _run_signature(self, iid: str, key: str) -> tuple:
        """Everything a parametric fit of `iid` reads beyond its raw correlogram: the
        measurement's committed params, the method, the cumulant order (cumulant only),
        the shared delay window, and the settings the engine seeds
        (skip_initial_channels; the cumulant method, cumulant only). Stored with each
        cached fit and compared on every redisplay and region edit, so a shown fit is
        flagged stale exactly when a re-Run would read different inputs."""
        s = self.controller.settings
        cumulant = key == 'cumulant'
        return (self.controller.committed_signature([iid]), key,
                self.order_spin.value() if cumulant else None,
                tuple(sorted(self.region.window_kwargs().items())),
                s.skip_initial_channels, s.cumulant_method if cumulant else None)

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
        # while a background fit is writing it too. Refuse rather
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
                self._cache[iid] = {'key': key, 'result': res,
                                    'sig': self._run_signature(iid, key)}
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

        def do_export(path: str):
            import os
            stem, ext = os.path.splitext(path)
            siblings = ([] if len(cached) == 1
                        else [f'{stem}_{iid}{ext}' for iid, _res in cached])
            if not confirm_overwrite(self, siblings):
                return None
            written = []
            for iid, res in cached:
                p = path if len(cached) == 1 else f'{stem}_{iid}{ext}'
                written.append(self.controller.export_correlogram_fit(iid, res, p))
            return '; '.join(written)

        status = export_to_csv(
            self, self.controller,
            _selection_scoped_name(self.controller, [iid for iid, _res in cached],
                                   f'correlogram_{key}'),
            do_export)
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
        self._update_status()        # the window is a fit input: a shown fit may be stale

    def _on_reset_region(self) -> None:
        """Restore the window + baseline to the from-data defaults."""
        self.region.force_reseed()
        self._reload_raw()           # re-seeds the region from the union lag range
        self._redraw()
        self._update_status()

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

    # Which handles live in which half of the plot: window markers carry
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
        self._update_status()                    # a dragged marker moved a fit input

    # Marker styling + offset-handle geometry: window carets at the
    # top, baseline carets at the bottom (the disambiguation described above).
    # _BASE_HANDLE_Y is kept a touch above the bottom so the caret never lands in
    # the residual-resize gap band below main_ax.
    _WIN_COLOR, _BASE_COLOR = '#2ca02c', '#7e57c2'   # window green, baseline purple
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

        def _caret(x, marker, color, y):
            return self.main_ax.plot(
                [x], [y], marker=marker, color=color, markersize=10,
                markeredgewidth=0, transform=htrans, clip_on=False, zorder=6)[0]

        # Only the first line of each kind carries a legend label, so the legend
        # shows one "τ window" and one "baseline region" entry.
        if r.tau_min_s is not None:
            self._markers['tau_min'] = self.main_ax.axvline(
                r.tau_min_s * tfac, color=self._WIN_COLOR, ls=':', lw=1.4,
                label='τ window (fit range)')
            self._handle_glyphs['tau_min'] = _caret(
                r.tau_min_s * tfac, 'v', self._WIN_COLOR, self._WIN_HANDLE_Y)
        if r.tau_max_s is not None:
            self._markers['tau_max'] = self.main_ax.axvline(
                r.tau_max_s * tfac, color=self._WIN_COLOR, ls=':', lw=1.4)
            self._handle_glyphs['tau_max'] = _caret(
                r.tau_max_s * tfac, 'v', self._WIN_COLOR, self._WIN_HANDLE_Y)
        if r.base_lo_s is not None and r.base_hi_s is not None:
            lo, hi = sorted((r.base_lo_s * tfac, r.base_hi_s * tfac))
            self._base_span = self.main_ax.axvspan(lo, hi, color=self._BASE_COLOR,
                                                   alpha=0.20, label='baseline region')
            self._markers['base_lo'] = self.main_ax.axvline(
                r.base_lo_s * tfac, color=self._BASE_COLOR, ls='--', lw=1.2)
            self._handle_glyphs['base_lo'] = _caret(
                r.base_lo_s * tfac, '^', self._BASE_COLOR, self._BASE_HANDLE_Y)
            self._markers['base_hi'] = self.main_ax.axvline(
                r.base_hi_s * tfac, color=self._BASE_COLOR, ls='--', lw=1.2)
            self._handle_glyphs['base_hi'] = _caret(
                r.base_hi_s * tfac, '^', self._BASE_COLOR, self._BASE_HANDLE_Y)

    # A linear-τ view is framed to this multiple of the slowest curve's 1/e decay time,
    # so the decay sits roughly centered rather than crushed at the origin. Set
    # empirically (a single exp reaches ~5 % of its plateau at 3·τ_1e, so ×5 leaves a
    # little tail room); tuned on the real multi-angle set + a synthetic clean set.
    _LINEAR_FRAME_K = 5.0

    def _decay_characteristics(self) -> Optional[_DecayFrame]:
        """Decay scales of the currently-ticked correlograms (seconds), via the Qt-free
        :func:`_decay_frame`. Kept display-agnostic so both the linear-τ framing and any
        later 1/e guide-line overlay can reuse it."""
        return _decay_frame(self._raw.values())

    def _linear_x_clamp_display(self) -> Optional[float]:
        """Upper x-limit (DISPLAY units) for a LINEAR-τ correlogram view (lin-lin /
        log-lin), chosen so the decay sits roughly centered instead of crushed at the
        origin — the default full lag range (out to ~1e6 µs) buries a decay that
        completes decades earlier. Frame to ``_LINEAR_FRAME_K × τ_1e`` of the *slowest*
        ticked curve (its 1/e time sits at ~1/K of the axis, the mostly-decayed 5 % point
        near the middle), capped at the measured range. We deliberately key off τ_1e — the
        robust steep-region characteristic time — and NOT the 5 %-decayed point: one curve
        with a long shallow tail (a noisy aggregate mode) would otherwise blow the frame
        out and re-crush every other curve. That tail stays readable in the log views.
        Returns None when nothing is ticked, or when the ticked data has no usable
        positive-plateau decay to frame on (both fall through to full-range autoscale)."""
        dc = self._decay_characteristics()
        if dc is None:
            return None
        upper = min(self._LINEAR_FRAME_K * dc.tau_1e_s, dc.data_max_s)
        if not (math.isfinite(upper) and upper > 0):
            return None
        return upper * _disp_factor('time')

    def _redraw(self) -> None:
        for ax in (self.main_ax, self.resid_ax, *self.side_axes):
            ax.clear()
        if self._placeholder is not None:       # ax.clear() doesn't drop a figure.text
            self._placeholder.remove()
            self._placeholder = None
        if not self._raw:
            # Nothing ticked: hide EVERY axis (a blank main box in a sea of hidden
            # satellite/residual boxes reads as broken) and show one centred grey message
            # via the shared helper — the same clean empty-state look as the SLS tab.
            # Restored once a curve exists.
            self._placeholder = show_plot_placeholder(
                self.figure, self.canvas, 'Tick a measurement to plot',
                axes=(self.main_ax, self.resid_ax, *self.side_axes))
            return
        for ax in (self.main_ax, self.resid_ax, *self.side_axes):
            ax.set_visible(True)
        # Plot saves suggest the same stem as the CSV export for the active method.
        set_plot_export_name(
            self.canvas,
            _selection_scoped_name(self.controller, list(self._raw),
                                   f'correlogram_{self.method_combo.currentData()}'))
        ws = self.controller.workspace.measurements
        xs, ys = _SCALE_XY[self._scales[0]]
        x_clamp = self._linear_x_clamp_display()   # frames linear-τ views on the decay
        # main view: overlay every ticked measurement (data + its fit, same color)
        for iid, (tau, g2) in self._raw.items():
            col = self.selection.color_for(iid)
            res = self._fit_for(iid)
            ft, fg = ((np.asarray(res.fit_tau_s, dtype=float),
                       np.asarray(res.fitted_g2m1, dtype=float)) if res else (None, None))
            plot_correlogram_scaled(tau, g2, ft, fg, ax=self.main_ax, xscale=xs,
                                    yscale=ys, color=col, label=_meas_label(ws[iid]),
                                    marker=self.selection.marker_for(iid))
        self.main_ax.set_title(f'Correlogram — {self._scales[0]}')
        self._draw_markers()                     # window + baseline handles
        if xs == 'linear':
            # Pin a linear delay axis to 0 (negative τ is meaningless) and frame the right
            # to the decay region so it sits centered, not crushed at the origin. Done
            # before the residual copies the x-lim below.
            right = x_clamp if x_clamp is not None else self.main_ax.get_xlim()[1]
            self.main_ax.set_xlim(0.0, right)
        handles, labels = self.main_ax.get_legend_handles_labels()
        if labels:
            self.main_ax.legend(frameon=False, fontsize=7, loc='best')
        # residual panel: one residual per fit (matching color). Plot in the same
        # DISPLAY units as the main axis (× display factor) so it lines up under the
        # fit — previously it drew raw seconds on a µs axis and squished to the left.
        tfac = _disp_factor('time')
        for iid in self._raw:
            res = self._fit_for(iid)
            if res is not None:
                self.resid_ax.plot(np.asarray(res.fit_tau_s, dtype=float) * tfac,
                                   np.asarray(res.residuals, dtype=float), '-',
                                   color=self.selection.color_for(iid), lw=1.0)
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
                col = self.selection.color_for(iid)
                res = self._fit_for(iid)
                ft, fg = ((np.asarray(res.fit_tau_s, dtype=float),
                           np.asarray(res.fitted_g2m1, dtype=float)) if res
                          else (None, None))
                plot_correlogram_scaled(tau, g2, ft, fg, ax=sax, xscale=sxs, yscale=sys,
                                        compact=True, color=col,
                                        marker=self.selection.marker_for(iid))
            if sxs == 'linear':                   # pin to 0; frame on the decay
                right = x_clamp if x_clamp is not None else sax.get_xlim()[1]
                sax.set_xlim(0.0, right)
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
                hi.setForeground(QtGui.QColor(self.selection.color_for(iid)))
                hi.setToolTip(hi.text())     # a narrow column elides the header
        table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        for col in range(1, 1 + len(ids)):
            table.horizontalHeader().setSectionResizeMode(
                col, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for r, name in enumerate(rows):
            label = QtWidgets.QTableWidgetItem(name)
            label.setToolTip(name)
            table.setItem(r, 0, label)
            for col in range(1, 1 + len(ids)):
                table.setItem(r, col, QtWidgets.QTableWidgetItem('—'))
        row_h = table.verticalHeader().defaultSectionSize()   # real row height
        table.setMaximumHeight(table.horizontalHeader().height()
                               + row_h * len(rows) + 2 * table.frameWidth() + 2)

    def _fill_tables(self) -> None:
        key = self.method_combo.currentData()
        ids = [i for i in self.selection.ids()
               if i in self.controller.workspace.measurements]
        for col, iid in enumerate(ids, start=1):
            res = self._fit_for(iid)
            if res is None:
                continue
            self._fill_col(self.result_table, col,
                           self._result_values(key, res, self._no_unc_sig()))

    def _no_unc_sig(self) -> int:
        """Sig figs for the ±-less result values (Settings → no-uncertainty precision).

        A single-correlogram fit has no honest ±, so ALL of its reported
        numbers — Rh, Γ, μ₂, β, τ, stretch, amplitude fractions — use this one fixed
        precision. Standing policy: a value with a ± is set by its ± and never comes
        here; a value without one is set by this knob (default 3)."""
        return self.controller.settings.no_uncertainty_sig_figs

    @staticmethod
    def _result_values(key: str, res, sig: int = 3) -> List[str]:
        """The result-table column values, in `_RESULT_ROWS[key]` order. Every reported
        number here is from a single correlogram (no honest ±), so all use
        the `sig` no-uncertainty precision; `order`/method/convergence are labels."""
        def q(v):
            return format_fixed_sig(v, sig)
        if key == 'cumulant':
            # A failed nonlinear fit now reports NaN outputs, not a linear-fit
            # substitute — label it '(failed)', and the Rh/Gamma cells read 'n/a'.
            method_lbl = res.method + ('' if res.success else ' (failed)')
            baseline_lbl = q(res.baseline) if res.method == 'nonlinear' else '—'
            # PDI cell: value + reliability glyph when measured; an explicit
            # 'unmeasured' for an order-1 fit (no mu2 term) so a blank PDI is never
            # read as a confirmed monodisperse PDI = 0.
            if math.isfinite(res.pdi):
                # The printed ceiling is the engine's own constant, never a literal, so
                # the glyph cannot go stale against the flag it explains.
                pdi_glyph = ('✓' if res.pdi_valid else
                             ('⚠<0' if res.pdi < 0
                              else f'⚠>{CUMULANT_PDI_VALIDITY_LIMIT:g}'))
                pdi_cell = f'{q(res.pdi)} {pdi_glyph}'
            elif res.order < 2:
                pdi_cell = 'unmeasured (order 1)'
            else:
                pdi_cell = q(res.pdi)   # 'n/a' — a failed/degenerate fit (row flagged)
            # Rh cell: the same glyph convention as its neighbour, for the same reason.
            # The PDI cell carried ✓/⚠ for a long time while the Rh cell beside it stayed
            # bare, so the one judgement that is ABOUT the size was the one the size column
            # did not show. Same tab, same result object, same tier (ui_style_guide R5.2:
            # one condition, one tier, across surfaces); the full sentence is already in
            # flag_label two inches below, so no tooltip is owed here.
            rh_cell = q(res.rh_nm)
            if not getattr(res, 'rh_reliable', True):
                rh_cell = f'{rh_cell} ⚠'
            return [q(res.gamma_s_inv), rh_cell, pdi_cell,
                    q(res.mu2_s_inv2), str(res.order), method_lbl, baseline_lbl]
        # Parametric-fit Rh cells: the same glyph convention as the cumulant Rh cell
        # above — one condition, one tier, across the whole parametric family. A NaN Rh
        # renders 'n/a' via q(); the glyph rides alongside so the blank still says why.
        # PER MODE for single/double: the result-level flag is the roll-up over the modes,
        # so reading it here would stamp ⚠ on a fast mode that is perfectly measured
        # because its slow neighbour never decayed. KWW has no modes, so it keeps the
        # result flag.
        def mode_rh(mode):
            cell = q(mode.rh_nm)
            return cell if getattr(mode, 'rh_reliable', True) else f'{cell} ⚠'

        def rh(v):
            cell = q(v)
            return cell if getattr(res, 'rh_reliable', True) else f'{cell} ⚠'
        if key == 'single':
            return [q(res.mode.gamma_s_inv), mode_rh(res.mode),
                    q(res.beta), 'yes' if res.success else 'no']
        if key == 'double':
            return [mode_rh(res.mode1), q(res.mode1.amplitude_fraction),
                    mode_rh(res.mode2), q(res.mode2.amplitude_fraction),
                    'yes' if res.success else 'no']
        return [q(res.stretch), q(res.tau_c_s), q(res.mean_tau_s),
                rh(res.rh_from_tau_c_nm), rh(res.rh_from_mean_tau_nm),
                'yes' if res.success else 'no']

    @staticmethod
    def _fill_col(table, col: int, values: List[str]) -> None:
        for r, v in enumerate(values):
            it = table.item(r, col)
            if it is not None:
                it.setText(v)
                # With several measurements ticked a column is ~30 px wide and a
                # value paints as '0.7…' -- the reliability glyph is the elided tail;
                # the full text rides the tooltip so number and glyph stay recoverable.
                it.setToolTip(v)

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
                flags.append(f'⚠ {_meas_short(lm)}: skipped — '
                             f'{humanize_field_names(self._run_failures[iid])}')
                continue
            res = self._fit_for(iid)
            if res is None:
                continue
            # Every method's numbers now live in the single results table; the
            # status line carries only warnings + the committed-values note.
            _text, flag = self._summary(key, res)
            # `_summary` can return SEVERAL lines -- a cumulant can be both high-PDI and
            # outside the DLS size range, two independent judgements -- and flag_label
            # renders one per '\n'. Prefixing the joined STRING would put the measurement on
            # the first line only, so in a multi-measurement run the second would read as if
            # it belonged to whichever measurement came next. Done at the join rather than
            # by changing _summary's return type, which the caller would still need this
            # loop after. splitlines() makes '' -> [], so a trailing newline cannot produce
            # a bare-prefix line.
            for line in flag.splitlines():
                if line.strip():
                    flags.append(f'{_meas_short(lm)}: {line}')
        notes = []
        if self.controller.is_dirty() and self._any_cached():
            notes.append('(ran on last committed values)')
        if self._any_stale():
            notes.append(_STALE_HINT)
        self.status.setText('  '.join(notes))
        self.flag_label.setText('\n'.join(flags))

    def _summary(self, key: str, r) -> Tuple[str, str]:
        # Status-line echoes of the same ±-less numbers the table shows — keep them on
        # the one no-uncertainty knob so a value never reads at two precisions in one view.
        q = lambda v: format_fixed_sig(v, self._no_unc_sig())
        if key == 'cumulant':
            # Two judgements, both shown. The "PDI > 0.3" verdict is worded by the fit
            # itself (a precondition of the size, it rides `reliability_note`, on the
            # line below), so this line keeps the one PDI failure the fit does not word:
            # a negative second cumulant. Only a MEASURED PDI warns -- an order-1 fit has
            # none (unmeasured, NaN) but a perfectly reliable Rh.
            flags = []
            if math.isfinite(r.pdi) and not r.pdi_valid and r.pdi < 0:
                flags.append(f'⚠ PDI = {q(r.pdi)} is negative (μ₂ < 0): the fit\'s '
                             'curvature term absorbed a baseline offset or a rising tail '
                             'rather than a spread of sizes — check the baseline and the '
                             'delay window.')
            if not getattr(r, 'rh_reliable', True):
                flags.append(f'⚠ {r.reliability_note}')
            return 'Cumulant fit (see table).', '\n'.join(flags)
        # Parametric branches: convergence and Rh reliability are independent
        # judgements (same reasoning as the cumulant flags above), so both can show.
        if key == 'single':
            return (f'Single exp: Γ = {q(r.mode.gamma_s_inv)} s⁻¹   '
                    f'Rh = {q(r.mode.rh_nm)} nm'), _param_flags(r)
        if key == 'double':
            return (f'Double exp: fast Rh = {q(r.mode1.rh_nm)} nm '
                    f'(f = {q(r.mode1.amplitude_fraction)}),   '
                    f'slow Rh = {q(r.mode2.rh_nm)} nm '
                    f'(f = {q(r.mode2.amplitude_fraction)})'), _param_flags(r)
        return 'KWW fit (see table).', _param_flags(r)


def _conv(success: bool) -> str:
    # The parametric fitters NaN their outputs on failure (owner decision D2), so the
    # honest sentence is "withheld", never "seed estimates" -- only the lognormal
    # distribution keeps a seed, and it does not report through this helper.
    return ('' if success else
            '⚠ the nonlinear fit did not converge; its values are withheld (n/a).')


def _param_flags(r) -> str:
    """Flag lines for a single/double/KWW fit: non-convergence plus the Rh
    reliability note (e.g. no viscosity → no Rh, or a mode that never finished
    decaying); one line per judgement.

    A fit with MODES gets one line per flagged mode, named fast/slow, because its modes
    are judged independently and a joined sentence would leave the reader guessing which
    size it condemns. `_update_status` already renders one line per newline.

    A cause shared by BOTH modes says its sentence once, labelled `both modes:` (owner
    ruling). The earlier dedup kept whichever label was
    seen first and silently dropped the other, so a missing viscosity -- which blanks both
    Rh values for one reason -- read as `fast mode:` and implied the slow one was fine.
    """
    flags = [ln for ln in (_conv(r.success),) if ln]
    labelled = [(lbl, m) for lbl, m in (('fast', getattr(r, 'mode1', None)),
                                        ('slow', getattr(r, 'mode2', None)),
                                        ('', getattr(r, 'mode', None)))
                if m is not None]
    if labelled:
        notes = [('' if getattr(m, 'rh_reliable', True) else m.reliability_note)
                 for _lbl, m in labelled]
        live = [n for n in notes if n]
        if len(labelled) > 1 and len(live) == len(labelled) and len(set(live)) == 1:
            flags.append(f'⚠ both modes: {live[0]}')
        else:
            for (lbl, _m), note in zip(labelled, notes, strict=True):
                if note:
                    flags.append(f'⚠ {lbl + " mode: " if lbl else ""}{note}')
    elif not getattr(r, 'rh_reliable', True):
        flags.append(f'⚠ {r.reliability_note}')
    return '\n'.join(flags)


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
        self._results_sig: tuple = ()           # committed-param signature of the last run's inputs
        self._failures: List[str] = []
        # Per-draw reweighting cache: [(iid, method, WeightedDistributionResult or None)],
        # parallel to self._results. None means intensity (no reweighting) or a reweight
        # error (the flag names it). Rebuilt every _draw; used by export + the peak table.
        self._weighted: List[Tuple] = []
        self._reweight_errors: List[str] = []
        self._placeholder = None                # empty-state Text artist (no distribution)
        # Staleness token for async runs: bumped when the ticked set changes, so
        # a background fit whose inputs are outdated is dropped, not drawn.
        self._run_epoch = 0
        self._build_ui()

    def _build_ui(self) -> None:
        # Distribution has the longest form labels ('CONTIN α selection:' etc.), so it
        # gets a slightly wider default + minimum than the other DLS tabs (labels
        # clipped at the old 340 px default). The Rh-grid triple-spinbox row measures
        # 384 px at the shipped size, so 385 keeps its two '?' badges inside the column
        # (they overhung by 10 and 6 px at 360).
        _, left, right = make_split_panels(self, left_min_width=385,
                                           sizes=(403, 717))

        self.checklist = MeasurementPicker(
            self.controller, self.selection, kinds=('dls',),
            label_fn=_meas_label, header_fn=_sample_header,
            help_text=_PICKER_HELP, help_bullets=_GROUP_PICKER_BULLETS,
            group_fields=(_GROUP_FIELD_CONC, _GROUP_FIELD_ANGLE))
        self.checklist.selectionChanged.connect(self._on_selection_changed)

        box = QtWidgets.QGroupBox('Distribution')
        add_help_to_groupbox(box, 'Recover a full size distribution (not just one '
                             'average) from the correlogram.', bullets=[
                                 '<b>CONTIN</b> (default): smooth, regularized — the '
                                 'robust general choice.',
                                 '<b>NNLS</b>: sharper but noisier; can split modes.',
                                 '<b>Lognormal</b>: assumes a single skewed peak — '
                                 'least free, most stable. If it cannot converge, the '
                                 'curve is labeled “(failed)”, shows the starting '
                                 'estimate, and offers no peaks.',
                                 'These are distribution-weighted, not z-average — '
                                 'compare against the cumulant Rh.',
                                 '<b>CONTIN α selection</b>: GCV (default, '
                                 'self-calibrating), the L-curve corner, or the F-test '
                                 '(see the Theory-and-Equations-Guide). '
                                 'For the F-test, a higher “probability to reject” gives '
                                 'a smoother fit, lower gives more detail.',
                             ])
        form = QtWidgets.QFormLayout(box)
        # Per-method checkboxes (replacing the old single "Overlay NNLS+CONTIN").
        # Adding a method later = one more entry in _DIST_METHODS.
        self.method_checks: Dict[str, QtWidgets.QCheckBox] = {}
        mrow = QtWidgets.QVBoxLayout(); mrow.setContentsMargins(0, 0, 0, 0)
        for label, key in _DIST_METHODS:
            cb = QtWidgets.QCheckBox(label)
            cb.setChecked(key == 'contin')        # CONTIN default
            self.method_checks[key] = cb
            mrow.addWidget(cb)
        mwidget = QtWidgets.QWidget(); mwidget.setLayout(mrow)
        form.addRow('Methods:', mwidget)
        self.axis_combo = QtWidgets.QComboBox()
        self.axis_combo.addItem('Rh (nm)', 'rh')
        self.axis_combo.addItem('Γ (decay rate, 1/s)', 'gamma')
        self.axis_combo.setToolTip('Plot the distribution against the size (Rh, via Stokes–Einstein — needs the committed viscosity and temperature) or against the raw decay rate Γ, which is what the correlogram measures; the Γ view lets you compare angles before the size conversion is trusted.')
        self.axis_combo.currentIndexChanged.connect(self._on_axis_changed)
        form.addRow('Distribution axis:', self.axis_combo)

        # Weighting selector (Phase 1b): re-express the SAME intensity distribution as
        # volume/surface/number. A re-presentation, not a re-fit — switching weighting
        # redraws from the cached intensity result (no recompute).
        self.weighting_combo = QtWidgets.QComboBox()
        for key in WEIGHTINGS:
            self.weighting_combo.addItem(WEIGHTING_LABELS[key], key)
        self.weighting_combo.currentIndexChanged.connect(self._on_weighting_changed)
        wrow = QtWidgets.QHBoxLayout(); wrow.setContentsMargins(0, 0, 0, 0)
        wrow.addWidget(self.weighting_combo, 1)
        wrow.addWidget(HelpBadge(
            'Volume / surface / number views re-weight the same intensity fit; they '
            'don’t improve or re-measure it.',
            bullets=[
                'The number view divides out how strongly each size scatters, so tiny '
                'particles (which scatter very weakly) get hugely magnified — it '
                '<b>amplifies small-size noise</b>.',
                'Treat a large small-particle population in the number view as '
                '<b>suspect</b> unless the intensity data clearly supports it.',
                'Reweighting <b>cannot fix</b> a noisy or ill-conditioned distribution — '
                'it re-presents the same ill-posed inversion, so it carries no ±.',
                'Well-posed only for <b>spheres</b> (Rh = geometric radius); it refuses '
                'non-spherical models. Number distributions from DLS are deprecated by '
                'the DLS standard — use them with care (Theory-and-Equations-Guide §3.6).',
                'Pick the sphere scattering model to match the sample: <b>Rayleigh</b> '
                '(very small particles), <b>RDG</b> (low contrast), or <b>Mie</b> (the '
                'general high-contrast/large case, e.g. latex in water). RDG and Mie need '
                'the hand-entered <b>solute refractive index</b> (Data tab).',
            ]))
        wholder = QtWidgets.QWidget(); wholder.setLayout(wrow)
        form.addRow('Weighting:', wholder)
        # Sphere scattering model (revealed only for a non-intensity weighting).
        self.model_combo = QtWidgets.QComboBox()
        for key in OPTICAL_MODELS:
            self.model_combo.addItem(OPTICAL_MODEL_LABELS[key], key)
        self.model_combo.setToolTip(
            'How per-particle scattering is modelled to invert intensity → number.\n'
            '• Rayleigh point: I ∝ R⁶, valid for very small particles (size parameter '
            'x = 2πR/λ ≲ 0.1); needs no refractive-index contrast.\n'
            '• RDG sphere: I ∝ V²·P(qR), valid for low contrast (|m−1| ≲ 0.1, '
            '2x|m−1| ≲ 0.1); needs the solute refractive index (Data tab).\n'
            '• Mie sphere: the full Mie theory — the general case that works where RDG and '
            'Rayleigh fail (high contrast, larger particles, e.g. latex in water); also needs '
            'the solute refractive index. See the Theory-and-Equations-Guide.')
        self.model_combo.currentIndexChanged.connect(self._on_weighting_changed)
        form.addRow('Sphere model:', self.model_combo)
        self._model_label = form.labelForField(self.model_combo)
        # Read-only solute-RI reminder (RDG only): editing happens on the Data tab
        # (invariant 5 — the Data tab owns parameter editing).
        self.solute_ri_hint = ThemedLabel('', role='hint', size=10)
        self.solute_ri_hint.setWordWrap(True)
        form.addRow('', self.solute_ri_hint)
        self._solute_ri_hint_label = form.labelForField(self.solute_ri_hint)
        # Rh grid + CONTIN L-curve α: seeded from Settings, overridable per run
        # (these moved here from the Settings tab — they belong next to the fit).
        self.rh_min = QtWidgets.QDoubleSpinBox()
        self.rh_min.setRange(0.01, 1.0e5); self.rh_min.setDecimals(2)
        self.rh_max = QtWidgets.QDoubleSpinBox()
        self.rh_max.setRange(0.01, 1.0e5); self.rh_max.setDecimals(1)
        self.rh_points = QtWidgets.QSpinBox(); self.rh_points.setRange(10, 1000)
        # A cost hint, not a cap: the range stays 10-1000 (owner-ruled 2026-08-26).
        self.rh_points.setToolTip(
            'Grid resolution of the inversion. CONTIN’s automatic α selection solves a '
            'dense system of this size about 40 times, so the cost climbs steeply with '
            'the point count — roughly half a second at 100 points, ~5 s at 300 and '
            '~30 s at 1000 with the default GCV selector on a typical laptop (the '
            'L-curve corner is cheaper but climbs the same way). The fit runs in the '
            'background, so the window stays usable.')
        grow = QtWidgets.QHBoxLayout(); grow.setContentsMargins(0, 0, 0, 0)
        for w in (self.rh_min, self.rh_max, self.rh_points):
            grow.addWidget(w)
        gholder = QtWidgets.QWidget(); gholder.setLayout(grow)
        form.addRow('R<sub>h</sub> grid (min / max / pts):', gholder)
        # CONTIN α selection: L-curve corner (default) or Provencher's F-test. The two
        # methods need different controls, so only the selected method's control is
        # shown (owner: show/hide, not gray-out — no confusing inert fields).
        self.alpha_method = QtWidgets.QComboBox()
        self.alpha_method.addItem('GCV (generalized cross-validation)', 'gcv')
        self.alpha_method.addItem('L-curve corner', 'lcurve')
        self.alpha_method.addItem('F-test (probability to reject)', 'ftest')
        self.alpha_method.setToolTip(
            'How CONTIN chooses its regularization α automatically.\n'
            '• GCV (default): generalized cross-validation — self-calibrating and '
            'robust for this kernel; the recommended choice.\n'
            '• L-curve corner: the maximum-curvature elbow of the fit-vs-smoothness '
            'trade-off (tends to smooth more heavily).\n'
            '• F-test: the smoothest solution whose fit is not significantly worse than '
            'the best — for comparison against legacy CONTIN output.\n'
            'See the Theory-and-Equations-Guide (CONTIN).')
        form.addRow('CONTIN α selection:', self.alpha_method)
        self.alpha_min = QtWidgets.QLineEdit()
        self.alpha_max = QtWidgets.QLineEdit()
        arow = QtWidgets.QHBoxLayout(); arow.setContentsMargins(0, 0, 0, 0)
        for w in (self.alpha_min, self.alpha_max):
            arow.addWidget(w)
        self._alpha_range_holder = QtWidgets.QWidget()
        self._alpha_range_holder.setLayout(arow)
        form.addRow('CONTIN α (min / max):', self._alpha_range_holder)
        self._alpha_range_label = form.labelForField(self._alpha_range_holder)
        # F-test level (only shown when the F-test method is selected).
        self.ftest_prob = QtWidgets.QDoubleSpinBox()
        self.ftest_prob.setRange(0.01, 0.99)
        self.ftest_prob.setSingleStep(0.05)
        self.ftest_prob.setDecimals(2)
        self.ftest_prob.setToolTip(
            'F-test significance level (default 0.50). A HIGHER value '
            'accepts more fit degradation and so selects a SMOOTHER, more parsimonious '
            'distribution; a LOWER value keeps a rougher, more-detailed one.')
        form.addRow('F-test probability to reject:', self.ftest_prob)
        self._ftest_prob_label = form.labelForField(self.ftest_prob)
        self.alpha_method.currentIndexChanged.connect(
            self._update_alpha_controls_visibility)
        self.reseed_from_settings()
        self._update_alpha_controls_visibility()
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
        # Per-measurement peak results for the ticked curves, mirroring
        # the Correlogram tab's results panel: one colored column per (measurement ·
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
        # Vertically resizable control column: drag the grips to size the
        # checklist / controls / results panes against each other.
        vstack = make_vertical_plot_stack(
            [self.checklist, box, results_section], sizes=[170, 300, 150],
            min_heights=[90, max(box.sizeHint().height(), 160), 70])
        left.addWidget(vstack, 1)
        self.status = QtWidgets.QLabel('')
        self.status.setWordWrap(True)
        left.addWidget(self.status)
        # A red flag label for run-skips, mirroring the Correlogram tab so a skipped
        # measurement uses the same channel everywhere (style guide §5 R5.5) rather than
        # hiding on the muted status line. Set via the shared tiered renderer.
        self.flag_label = ThemedLabel('', role='error', bold=True)
        self.flag_label.setWordWrap(True)
        left.addWidget(self.flag_label)
        # Neutral ⓘ note (steel-blue calmer-warnings tier) for the model-dependence of a
        # reweighted view — distinct from the red flag_label, which is for problems.
        self.info_label = ThemedLabel('', size=11)
        self.info_label.setWordWrap(True)
        left.addWidget(self.info_label)

        self.figure = Figure(figsize=(5.3, 4.4), constrained_layout=True)
        # The distribution + residual do NOT share x (both carry their own x-label), so a
        # generous gap is needed for the grip line to clear the main plot's "Rh (nm)"
        # label (the indicator must not overlap the plots).
        _widen_row_gap(self.figure, hspace=0.20)
        self.canvas = make_canvas_expanding(FigureCanvas(self.figure))
        # Main distribution panel + a residual panel below (like the Correlogram
        # tab). The residual x-axis is the delay time τ, NOT the Rh/Γ axis, so the
        # two do not share x. The residual is height-adjustable via a draggable handle
        # via that handle; a flat gridspec keeps constrained_layout happy.
        self._resid_ratio = 0.22
        a = (1.0 - self._resid_ratio) / 3.0
        self._gs = self.figure.add_gridspec(4, 1, height_ratios=[a, a, a, self._resid_ratio])
        self.ax = self.figure.add_subplot(self._gs[0:3, 0])
        self.resid_ax = self.figure.add_subplot(self._gs[3, 0])
        right.addWidget(themed_navtoolbar(self.canvas, self, self.controller))
        # Bottom-panel selector: swap the lower panel between the fit residuals and
        # the CONTIN α-selection sweep (the diagnostic behind the automatic α). The
        # α-sweep is CONTIN-only; it draws nothing for NNLS/lognormal runs. Mirrors the
        # Utilities Diagnostic combo. The "Export α-sweep…" button rides alongside it so
        # the export sits next to the view it writes.
        brow = QtWidgets.QHBoxLayout(); brow.setContentsMargins(0, 0, 0, 0)
        brow.addWidget(QtWidgets.QLabel('Bottom panel:'))
        self.bottom_combo = QtWidgets.QComboBox()
        self.bottom_combo.addItem('Residuals', 'residuals')
        self.bottom_combo.addItem('α-sweep (CONTIN)', 'alpha')
        self.bottom_combo.setToolTip(
            'What the lower panel shows.\n'
            '• Residuals: data − reconstructed g₂−1 vs delay time, per curve.\n'
            '• α-sweep: how CONTIN chose its regularization α — the L-curve corner, '
            'the GCV minimum, or the F-test curve (see the '
            'Theory-and-Equations-Guide). CONTIN results only.')
        self.bottom_combo.currentIndexChanged.connect(self._on_bottom_changed)
        brow.addWidget(self.bottom_combo)
        brow.addWidget(HelpBadge(
            'Read the α-sweep to judge whether to trust CONTIN’s automatic α.',
            bullets=[
                '<b>L-curve</b>: a sharp <b>corner</b> is a clean trade-off between fit '
                'and smoothness; a flat/round corner means α is poorly determined.',
                '<b>GCV</b>: a distinct <b>minimum</b> is a confident pick; a flat valley '
                'is ambiguous.',
                '<b>F-test</b>: the curve climbs with α — the pick is the smoothest fit '
                'not significantly worse than the best. On fine Rh grids its reference '
                '<i>is</i> the floor, so it can pin there — the halo says when it did.',
                'A red halo marks an α pinned to a <b>sweep edge</b>: at the ceiling, '
                'possible over-regularization; at the floor, the selector wanted a smaller '
                'α than the range allows — at a low floor (the default 1e-4) that is '
                'essentially the unregularized NNLS fit, whose extra peaks may be inversion '
                'noise. Widen the α range.',
                'One sweep per CONTIN curve, colored to match the distribution above.',
            ]))
        self.export_sweep_button = QtWidgets.QPushButton('Export α-sweep…')
        self.export_sweep_button.setEnabled(False)
        self.export_sweep_button.setToolTip(
            'Write the α-sweep (alphas, norms, and the GCV/F-test score curve) to an '
            'Origin CSV — one file per CONTIN curve.')
        self.export_sweep_button.clicked.connect(self._on_export_sweep)
        brow.addWidget(self.export_sweep_button)
        brow.addStretch(1)
        bholder = QtWidgets.QWidget(); bholder.setLayout(brow)
        right.addWidget(bholder)
        right.addWidget(self.canvas, 1)
        self._resid_resizer = attach_residual_resizer(
            self.canvas, self.figure, self.ax, self.resid_ax, self._apply_resid_ratio)
        self.axis_bar = AxisControlBar(self.canvas)
        right.addWidget(self.axis_bar)
        self._update_weighting_controls_visibility()
        self._clear('Tick measurements + methods, then Run.')

    def _apply_resid_ratio(self, frac: float) -> None:
        """Resize the residual vs the distribution panel (draggable handle)."""
        self._resid_ratio = frac
        a = (1.0 - frac) / 3.0
        self._gs.set_height_ratios([a, a, a, frac])
        self.canvas.draw_idle()

    def _update_alpha_controls_visibility(self) -> None:
        """Show only the selected α-selection method's control (L-curve α range vs the
        F-test level) — the owner asked for show/hide, not grayed-out inert fields."""
        is_ftest = self.alpha_method.currentData() == 'ftest'
        self._alpha_range_holder.setVisible(not is_ftest)
        if self._alpha_range_label is not None:
            self._alpha_range_label.setVisible(not is_ftest)
        self.ftest_prob.setVisible(is_ftest)
        if self._ftest_prob_label is not None:
            self._ftest_prob_label.setVisible(is_ftest)

    def reseed_from_settings(self) -> None:
        """(Re)seed the Rh grid + α fields from the global Settings defaults. The
        per-run values the user types here always win (run_distribution honors
        explicit kwargs over its settings setdefault)."""
        s = self.controller.settings
        self.rh_min.setValue(s.rh_grid_min_nm)
        self.rh_max.setValue(s.rh_grid_max_nm)
        self.rh_points.setValue(s.rh_grid_points)
        self.alpha_min.setText(f'{s.lcurve_alpha_min:g}')
        self.alpha_max.setText(f'{s.lcurve_alpha_max:g}')
        i = self.alpha_method.findData(s.contin_alpha_method)
        self.alpha_method.setCurrentIndex(i if i >= 0 else 0)
        self.ftest_prob.setValue(s.contin_ftest_prob_reject)
        self._update_alpha_controls_visibility()

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
            kw['alpha_method'] = self.alpha_method.currentData()
            kw['ftest_prob_reject'] = self.ftest_prob.value()
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
        # Decoupled from sidebar selection: the checklist
        # is the only thing that picks measurements for analysis.
        self.checklist.refresh()
        self.run_button.setEnabled(self._has_checked())
        if not self._results:
            self._clear('Tick measurements + methods, then Run.')
        elif self._results_stale():
            # A displayed result whose inputs changed (a commit, or the shared
            # window / baseline / grid moved since) — flag it without redrawing.
            self.status.setText(_STALE_HINT)

    def _run_signature(self, pairs) -> tuple:
        """Everything the distributions for the (measurement, method) `pairs` read beyond
        their raw correlograms: committed params, per pair the kwargs the Run hands the
        engine (the shared window + this measurement's baseline over it, the Rh grid
        and, for CONTIN, the α range and selector), and the settings seed the engine
        fills in itself (skip_initial_channels). The pairs ride in the signature, so
        `_results_stale` needs no side table. Compared on redisplay: a distribution
        fitted under a window or grid since moved is flagged stale (the two tabs share
        one region, so a Correlogram-tab edit reaches here)."""
        ids = sorted({iid for iid, _key in pairs})
        per = tuple(sorted(
            (iid, key, tuple(sorted({**self._baseline_kwargs(iid),
                                     **self._grid_kwargs(key)}.items())))
            for iid, key in pairs))
        return (self.controller.committed_signature(ids), per,
                self.controller.settings.skip_initial_channels)

    def _results_stale(self) -> bool:
        sig = self._results_sig
        if not sig:
            return False
        return self._run_signature([(iid, key) for iid, key, _kw in sig[1]]) != sig

    @QtCore.Slot()
    def _on_selection_changed(self) -> None:
        # The checked set changed → previous curves are stale; require a re-Run.
        # Any in-flight background fit is also stale: bump the epoch so its
        # result is discarded on arrival.
        self._run_epoch += 1
        self._results = []
        self._results_sig = ()
        self._failures = []
        self.export_button.setEnabled(False)
        self.run_button.setEnabled(self._has_checked())
        self._update_weighting_controls_visibility()   # refresh the RDG solute-RI hint
        self._clear('Selection changed — press Run.')

    @QtCore.Slot()
    def _on_axis_changed(self) -> None:
        if self._results:
            self._draw()                          # re-plot cached results on new axis

    @QtCore.Slot()
    def _on_weighting_changed(self) -> None:
        """Weighting or sphere-model changed: update the controls, then re-draw the CACHED
        intensity results under the new weighting (a re-presentation — never a re-fit)."""
        self._update_weighting_controls_visibility()
        if self._results:
            self._draw()

    def _active_weighting(self) -> Tuple[str, Optional[str]]:
        """(weighting, model) currently selected; model is None for the intensity view."""
        weighting = self.weighting_combo.currentData()
        if weighting == INTENSITY:
            return INTENSITY, None
        return weighting, self.model_combo.currentData()

    def _update_weighting_controls_visibility(self) -> None:
        """Reveal the sphere-model row only for a non-intensity weighting, and the
        solute-RI reminder for the RDG and Mie models (both form m = n_solute/n_medium)."""
        weighting = self.weighting_combo.currentData()
        non_intensity = weighting != INTENSITY
        self.model_combo.setVisible(non_intensity)
        if self._model_label is not None:
            self._model_label.setVisible(non_intensity)
        needs_ri = non_intensity and self.model_combo.currentData() in (RDG, MIE)
        self.solute_ri_hint.setVisible(needs_ri)
        if self._solute_ri_hint_label is not None:
            self._solute_ri_hint_label.setVisible(needs_ri)
        if needs_ri:
            self.solute_ri_hint.setText(self._solute_ri_summary())

    def _solute_ri_summary(self) -> str:
        """A read-only reminder of the ticked measurements' committed solute RI (edited on
        the Data tab, invariant 5). Names the value when all agree, or prompts when unset."""
        ws = self.controller.workspace.measurements
        ids = [i for i in self.selection.ids() if i in ws and ws[i].kind == 'dls']
        vals = []
        for iid in ids:
            v = ws[iid].committed_params.get('solute_refractive_index')
            vals.append(v)
        present = [v for v in vals if v is not None]
        if not ids:
            return 'RDG / Mie need the solute refractive index — set it on the Data tab.'
        if not present:
            return ('Solute refractive index not set — enter it on the Data tab '
                    '(the RDG and Mie models need it; it is never guessed).')
        uniq = sorted({round(float(v), 6) for v in present})
        shown = ', '.join(f'{v:g}' for v in uniq)
        tail = '' if len(present) == len(ids) else ' (some measurements unset)'
        return f'Solute refractive index: {shown} (from the Data tab){tail}.'

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
        sig = self._run_signature([(iid, key) for iid, key, _kw in jobs])

        def work():
            results, failures = [], []
            for iid, key, kw in jobs:
                try:
                    res = controller.run_distribution(iid, key, **kw)
                    results.append((iid, key, res))
                except Exception as exc:           # per (measurement, method)
                    failures.append(
                        f'{labels[iid]}·{key.upper()}: {humanize_field_names(str(exc))}')
            return results, failures

        def done(payload) -> None:
            if epoch != self._run_epoch:
                return          # selection changed while the fit ran — stale
            self._results, self._failures = payload
            self._results_sig = sig
            self.export_button.setEnabled(bool(self._results))
            self._draw()

        def fail(exc: BaseException) -> None:
            # A failure landing after a selection change must not write onto the
            # view now shown -- the same epoch guard `done` applies.
            QtWidgets.QMessageBox.critical(
                self, 'Analysis failed',
                f'Could not run the distribution fit.\n\n'
                f'{humanize_field_names(str(exc))}')
            if epoch == self._run_epoch:
                self.status.setText('Analysis failed — see dialog.')

        if runner().try_submit(work, done, fail, description='distribution fit',
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
        weighting, _model = self._active_weighting()
        weighted = weighting != INTENSITY
        # In a weighted view the plot forces the Rh axis (the axis combo is disabled but may
        # still hold a stale 'gamma' selection). Match it for the intensity FALLBACK export
        # too, so a reweight-errored curve never writes a Γ-axis (decay-rate) CSV into a file
        # offered as a weighted size distribution.
        axis = 'rh' if weighted else self.axis_combo.currentData()
        items = list(self._results)
        wd_by = {(iid, key): wd for iid, key, wd in self._weighted}

        def do_export(path: str):
            import os
            stem, ext = os.path.splitext(path)
            siblings = ([] if len(items) == 1
                        else [f'{stem}_{iid}_{key}{ext}' for iid, key, _res in items])
            if not confirm_overwrite(self, siblings):
                return None
            written = []
            for iid, key, res in items:
                p = path if len(items) == 1 else f'{stem}_{iid}_{key}{ext}'
                wd = wd_by.get((iid, key)) if weighted else None
                if wd is not None:
                    written.append(self.controller.export_weighted_distribution(wd, p))
                else:
                    # Intensity view, or a curve whose reweight errored: export the
                    # intensity distribution (never silently write nothing).
                    written.append(self.controller.export_distribution(res, p, axis))
            return '; '.join(written)

        name = _selection_scoped_name(
            self.controller, [iid for iid, _key, _res in items],
            'weighted_distribution' if weighted else 'distribution')
        status = export_to_csv(self, self.controller, name, do_export)
        if status:
            self.status.setText(status)

    def _draw(self) -> None:
        if self._placeholder is not None:    # leaving the empty-state: drop the message
            self._placeholder.remove()
            self._placeholder = None
        self.ax.clear()
        self.resid_ax.clear()
        self.ax.set_visible(True)            # restore after an empty _clear() hid both
        self.resid_ax.set_visible(True)
        self.axis_bar.attach(self.ax)        # re-bind the axis controls (detached while empty)
        weighting, model = self._active_weighting()
        weighted = weighting != INTENSITY
        # A weighted view is a size re-expression → always plotted vs Rh; the Γ axis stays
        # meaningful only for the intensity view (so disable it while reweighted).
        axis = 'rh' if weighted else self.axis_combo.currentData()
        self.axis_combo.setEnabled(not weighted)
        # Plot saves suggest the same stem as the CSV export for the active view.
        set_plot_export_name(
            self.canvas,
            _selection_scoped_name(
                self.controller, [iid for iid, _key, _res in self._results],
                'weighted_distribution' if weighted else 'distribution'))
        ws = self.controller.workspace.measurements
        single = len(self._results) == 1
        peak_items = []          # (x, y, text, color) collected across all curves
        self._weighted = []
        reweight_errors: List[str] = []
        labelled_invalid = False        # label the shared 'outside model validity' entry once
        for iid, _key, res in self._results:
            d = self._as_distribution(res)
            color = self.selection.color_for(iid)
            label = f'{_meas_short(ws[iid])} · {d.method.upper()}'
            # A failed lognormal fit draws the seed estimate, not a fit — say so in
            # the legend, matching the cumulant table's '(failed)' cell.
            if not getattr(d, 'success', True):
                label += ' (failed)'
            wd = None
            if weighted:
                try:
                    wd = self.controller.reweight_distribution(iid, d, weighting, model)
                except OpticalModelError as exc:
                    reweight_errors.append(f'{_meas_short(ws[iid])}: {exc}')
                except Exception as exc:                       # defensive
                    reweight_errors.append(
                        f'{_meas_short(ws[iid])}: {humanize_field_names(str(exc))}')
            self._weighted.append((iid, _key, wd))
            if wd is not None:
                # Label the shared out-of-range marker only on the FIRST flagged curve, so
                # an overlay's legend never repeats it -- without a second ax.legend() dedup
                # pass (avoiding extra Agg/Qt artist churn).
                inv_lbl = ('outside model validity'
                           if (not wd.all_valid and not labelled_invalid) else None)
                if not wd.all_valid:
                    labelled_invalid = True
                plot_weighted_distribution(
                    wd, ax=self.ax, label=label, color=color, fill=single,
                    linestyle=self.selection.linestyle_for(iid) or '-',
                    invalid_label=inv_lbl)
                peak_items.extend(self._collect_weighted_peak_labels(wd, color))
            elif not weighted:
                plot_distribution(d, ax=self.ax, axis=axis, label=label,
                                  color=color, fill=single,
                                  linestyle=self.selection.linestyle_for(iid) or '-')
                peak_items.extend(self._collect_peak_labels(d, res, axis, color))
            # weighted-but-wd-None (a reweight error): skip the curve; the flag explains.
        self._reweight_errors = reweight_errors
        # One de-collided pass so peaks that coincide across curves stagger (capped at
        # 6 + "+N more") instead of piling into an unreadable blob.
        annotate_decollided(self.ax, peak_items)
        self.ax.set_title(
            f'DLS {WEIGHTING_LABELS[weighting].lower()}-weighted size distribution'
            if weighted else 'DLS size / rate distribution')
        self._draw_bottom()                  # residuals OR the CONTIN α-sweep
        self.canvas.draw_idle()
        self.axis_bar.attach(self.ax)
        # Status now carries only operational notes; per-result peaks live in the
        # Summary tab (which is also where they persist).
        notes = []
        n_contin = sum(1 for _i, k, _r in self._results if k == 'contin')
        if n_contin > 3:
            notes.append('CONTIN runs an α sweep per measurement — this can be slow.')
        # α-selection provenance: every CONTIN run this pass shares the same selector
        # (seeded from one control), so one note names it (method disclosure —
        # a distribution is never ambiguous about how α was chosen). The wording comes
        # from the shared `alpha_selection_note` helper the CSV export also uses, so the
        # two surfaces can't drift as selectors are added. A user-supplied
        # α returns '' here (the α input control already shows it).
        contin_res = next((r for _i, k, r in self._results if k == 'contin'), None)
        alpha_note = alpha_selection_note(
            getattr(contin_res, 'alpha_selection_method', None),
            getattr(contin_res, 'ftest_prob_reject', None), style='gui')
        if alpha_note:
            notes.append(alpha_note)
        if self.controller.is_dirty() and self._results:
            notes.append('(ran on last committed values)')
        if self._results_stale():
            notes.append(_STALE_HINT)
        notes.append('Peak values are also in the Summary tab.')
        self.status.setText('   '.join(notes))
        # A skipped run is a data/operational problem, not a muted status note: render it
        # on the red flag label via the shared renderer, matching the Correlogram tab
        # (style guide §5 R5.5). A CONTIN α pinned to a sweep edge is a data-conditional
        # caution; the wording is the result's own reliability_note (one home, set in
        # fit_contin) and only the how-to advice is added here -- kept on the same visible
        # flag, same tier (R5.2: one condition class). Empty text clears it.
        flag_msgs = []
        failures = getattr(self, '_failures', None)
        if failures:
            flag_msgs.append('skipped: ' + '; '.join(failures))
        if getattr(self, '_reweight_errors', None):
            flag_msgs.append('reweighting skipped — ' + '; '.join(self._reweight_errors))
        # The sweep-edge caution is the result's own reliability_note (one wording on
        # every surface); the GUI adds only the how-to advice. Deduped, order-preserving.
        edge_notes = list(dict.fromkeys(
            getattr(r, 'reliability_note', '') for _i, k, r in self._results
            if k == 'contin' and getattr(r, 'reliability_note', '')))
        for note in edge_notes:
            flag_msgs.append(f'{note} — inspect the α-sweep or widen the α range')
        for f_iid, _k, r in self._results:
            if not getattr(self._as_distribution(r), 'success', True):
                flag_msgs.append(
                    f'{_meas_short(ws[f_iid])}: lognormal fit did not converge; '
                    'the curve shows the seed estimate, not a fit')
        set_flag(self.flag_label, '; '.join(flag_msgs), problem=True)
        # Neutral ⓘ model-dependence note for the drawn reweighted view. Aggregate UNIQUE
        # note lines across ALL drawn curves (deduped, order-preserving), so a later curve's
        # out-of-validity caveat is shown even when the first curve is fully in-range — not
        # just the first curve's notes.
        info_lines: List[str] = []
        for _i, _k, wd in self._weighted:
            if wd is not None:
                for n in wd.notes:
                    if n not in info_lines:
                        info_lines.append(n)
        set_flag(self.info_label, '\n'.join(info_lines), problem=False)
        self._refresh_results()

    # ---- bottom panel: residuals <-> CONTIN α-sweep ----
    def _contin_results(self) -> list:
        """The ticked CONTIN results (in run order) — the only ones with an α-sweep."""
        return [(iid, res) for iid, key, res in self._results if key == 'contin']

    def _draw_bottom(self) -> None:
        """Draw the lower panel per the combo: fit residuals, or the CONTIN α-sweep.
        Called from `_draw` on a cleared `resid_ax`; also updates the α-sweep export
        button's enabled state (α-sweep view AND ≥1 CONTIN result)."""
        view = self.bottom_combo.currentData()
        contin = self._contin_results()
        self.export_sweep_button.setEnabled(view == 'alpha' and bool(contin))
        if view == 'alpha':
            self._draw_alpha_sweep(contin)
        else:
            self._draw_residuals()

    def _draw_residuals(self) -> None:
        """Per-curve fit residuals (data − reconstructed g₂−1) vs delay time, in the
        active display units so the τ axis matches the rest of the app."""
        for iid, _key, res in self._results:
            d = self._as_distribution(res)
            tau = np.asarray(d.fit_tau_s, dtype=float) * _disp_factor('time')
            self.resid_ax.plot(tau, np.asarray(d.residuals, dtype=float),
                               '-', color=self.selection.color_for(iid), lw=1.0)
        self.resid_ax.axhline(0.0, color='0.6', lw=0.8)
        self.resid_ax.set_xscale('log')
        self.resid_ax.set_xlabel(rf'Delay time $\tau$ ({_disp_unit("time")})')
        self.resid_ax.set_ylabel('resid.')

    def _draw_alpha_sweep(self, contin: list) -> None:
        """Overlay each ticked CONTIN result's α-selection sweep in its curve color,
        method-aware (L-curve corner / GCV minimum / F-test curve). NNLS/lognormal have
        no sweep; if nothing to draw, show the CONTIN-only empty-state message instead.

        Every CONTIN result here comes from ONE Run pass (which replaces `_results`
        wholesale and reads a single α-selection control), so they all share one
        `alpha_selection_method` and therefore one ordinate — the overlay never mixes an
        L-curve (residual-vs-solution) with a GCV/F-test (score-vs-α) axis on this shared
        panel. If runs ever become incremental, guard for a mixed-method set here."""
        ws = self.controller.workspace.measurements
        drawn = 0
        for iid, res in contin:
            lcurve = getattr(res, 'lcurve', None)
            if lcurve is None:
                continue
            try:
                plot_lcurve(
                    lcurve, ax=self.resid_ax,
                    method=getattr(res, 'alpha_selection_method', 'lcurve'),
                    color=self.selection.color_for(iid),
                    label=_meas_short(ws[iid]) if iid in ws else None,
                    mark_ceiling=getattr(res, 'alpha_at_ceiling', False),
                    mark_floor=getattr(res, 'alpha_at_floor', False))
                drawn += 1
            except ValueError:
                # No sweep to show for this result: a user-fixed α (method 'user' —
                # no sweep, only a degenerate single point) or a requested ordinate
                # whose score curve is absent (e.g. 'gcv' with gcv is None). Neither is
                # reachable from this tab's sweep controls today, but skip rather than
                # draw a misleading one-point "corner"; drawn==0 → the empty-state.
                continue
        if drawn == 0:
            self.resid_ax.set_axis_off()
            self.resid_ax.text(
                0.5, 0.5, 'No α-sweep — CONTIN only',
                transform=self.resid_ax.transAxes, ha='center', va='center',
                color=PLACEHOLDER_GRAY, fontsize=9)

    @QtCore.Slot()
    def _on_bottom_changed(self) -> None:
        """Redraw only the lower panel when the Residuals ↔ α-sweep combo changes."""
        if not self._results:
            return
        self.resid_ax.set_axis_on()          # a prior empty-state may have hidden it
        self.resid_ax.clear()
        self._draw_bottom()
        self.canvas.draw_idle()

    @QtCore.Slot()
    def _on_export_sweep(self) -> None:
        contin = self._contin_results()
        if not contin:
            return

        def do_export(path: str):
            import os
            stem, ext = os.path.splitext(path)
            siblings = ([] if len(contin) == 1
                        else [f'{stem}_{iid}{ext}' for iid, _res in contin])
            if not confirm_overwrite(self, siblings):
                return None
            written = []
            for iid, res in contin:
                p = path if len(contin) == 1 else f'{stem}_{iid}{ext}'
                written.append(self.controller.export_lcurve(res, p))
            return '; '.join(written)

        status = export_to_csv(
            self, self.controller,
            _selection_scoped_name(self.controller, [iid for iid, _res in contin],
                                   'alpha_sweep'),
            do_export)
        if status:
            self.status.setText(status)

    def _refresh_results(self) -> None:
        """Fill the peak-results panel from the ticked results — one colored column per
        (measurement · method), positional-peak rows. Mirrors the
        Correlogram results panel; reuses the same peak source as the plot labels."""
        ws = self.controller.workspace.measurements
        cols = [(iid, key, res) for iid, key, res in self._results if iid in ws]
        # When a reweighted view is active, the peak table shows the SAME weighting as the
        # plot (rh, fraction-in-weighting); otherwise the intensity peaks. A curve whose
        # reweight errored (wd is None) shows no peaks.
        wd_by = {(iid, key): wd for iid, key, wd in self._weighted}
        weighting, _model = self._active_weighting()
        weighted = weighting != INTENSITY

        def _peaks_for(iid, key, res):
            if weighted:
                wd = wd_by.get((iid, key))
                return ([(rh, frac) for rh, frac in
                         self.controller.weighted_distribution_peaks(wd)] if wd else [])
            return [(rh, frac) for rh, _g, frac in self.controller.distribution_peaks(res)]

        peaks_per = [_peaks_for(iid, key, res) for iid, key, res in cols]
        max_peaks = max((len(p) for p in peaks_per), default=0)
        n_rows = max(max_peaks, 1)
        table = self.result_table
        table.clear()
        table.setColumnCount(1 + len(cols))
        table.setRowCount(n_rows)
        table.horizontalHeader().setVisible(True)
        suffix = f' · {WEIGHTING_LABELS[weighting]}' if weighted else ''
        table.setHorizontalHeaderLabels(
            ['', *[f'{_meas_short(ws[iid])} · {key.upper()}{suffix}'
                   for iid, key, _r in cols]])
        for col, (iid, _k, _r) in enumerate(cols, start=1):
            hi = table.horizontalHeaderItem(col)
            if hi is not None:
                hi.setForeground(QtGui.QColor(self.selection.color_for(iid)))
        table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        for col in range(1, 1 + len(cols)):
            table.horizontalHeader().setSectionResizeMode(
                col, QtWidgets.QHeaderView.ResizeMode.Stretch)
        # Distribution peaks are from an ill-posed inversion — no honest ±
        # — so the Rh uses the documented fixed-sig fallback, never a fabricated ±.
        sig = self.controller.settings.no_uncertainty_sig_figs
        for r in range(n_rows):
            table.setItem(r, 0, QtWidgets.QTableWidgetItem(_ordinal_peak(r)))
            for col, peaks in enumerate(peaks_per, start=1):
                if r < len(peaks):
                    rh_nm, weight = peaks[r]
                    txt = (f'{format_fixed_sig(rh_nm, sig)} nm · '
                           f'{format_fixed_sig(weight * 100, sig)}%')
                else:
                    txt = '—'
                table.setItem(r, col, QtWidgets.QTableWidgetItem(txt))
        # No max-height cap: the peak table sits in a stretch slot of the vertical
        # splitter, so it should fill (and be resizable into) that slot rather than
        # stay pinned to a row-count height (which stopped the results section from
        # expanding vertically). The 22 px/row estimate also clipped rows on
        # platforms with taller default rows.

    def _clear_results(self) -> None:
        self.result_table.clear()
        self.result_table.setRowCount(0)
        self.result_table.setColumnCount(0)

    def _collect_weighted_peak_labels(self, wd, color: str) -> list:
        """Peak labels (x, y, text, color) for a reweighted view — always vs Rh, fraction
        in the chosen weighting. Uses the controller's weighted peak finder (invariant 5).
        The annotation x is scaled to the active RADIUS display unit to match the plotted
        curve (drawn at rh*display_factor), so labels don't land off-axis when the unit ≠ nm."""
        rh = np.asarray(wd.rh_grid_nm, dtype=float)
        w = np.asarray(wd.weights, dtype=float)
        sig = self.controller.settings.no_uncertainty_sig_figs   # reweighted: ±-less
        rfac = _disp_factor('radius')
        items = []
        for rh_nm, _frac in self.controller.weighted_distribution_peaks(wd):
            if rh_nm is None or not math.isfinite(rh_nm) or rh.size == 0:
                continue
            y = float(w[int(np.argmin(np.abs(rh - rh_nm)))])       # y from the RAW grid
            items.append((rh_nm * rfac, y, f'{format_fixed_sig(rh_nm, sig)} nm', color))
        return items

    def _collect_peak_labels(self, dist, res, axis: str, color: str) -> list:
        """Collect this curve's peak labels as (x, y, text, color) tuples for the
        shared de-collision pass (GUI overlay; kept on saved images too, like the
        Correlogram τ markers). Uses the controller's peak finder so the analysis layer
        is not imported here. y is the curve's height at the peak, from the grid."""
        grid = np.asarray(dist.rh_grid_nm if axis == 'rh'
                          else dist.gamma_grid_s_inv, dtype=float)
        w = np.asarray(dist.weights, dtype=float)
        sig = self.controller.settings.no_uncertainty_sig_figs   # distribution peaks: ±-less
        # Scale the annotation x to the active display unit to match the plotted curve
        # (plot_distribution draws at x*display_factor); the y lookup stays on the RAW grid.
        fac = _disp_factor('radius' if axis == 'rh' else 'decay_rate')
        items = []
        for rh_nm, gamma_s_inv, _weight in self.controller.distribution_peaks(res):
            x = rh_nm if axis == 'rh' else gamma_s_inv
            if x is None or not math.isfinite(x) or grid.size == 0:
                continue
            y = float(w[int(np.argmin(np.abs(grid - x)))])
            txt = (f'{format_fixed_sig(rh_nm, sig)} nm' if axis == 'rh'
                   else f'{format_fixed_sig(gamma_s_inv, sig)} s⁻¹')
            items.append((x * fac, y, txt, color))
        return items

    def _clear(self, message: str) -> None:
        if self._placeholder is not None:       # ax.clear() doesn't drop a figure.text
            self._placeholder.remove()
            self._placeholder = None
        self.ax.clear()
        self.resid_ax.clear()
        # Hide BOTH axes and show one centred grey message via the shared helper — the
        # clean empty-state look (a blank 0–1 box reads as broken). _draw restores them.
        self._placeholder = show_plot_placeholder(
            self.figure, self.canvas, message, axes=(self.ax, self.resid_ax))
        self.axis_bar.attach(None)              # nothing to drive while empty
        self.export_sweep_button.setEnabled(False)
        self._clear_results()
        self._weighted = []
        self._reweight_errors = []
        self.axis_combo.setEnabled(True)        # the weighted view disables it; restore
        # Drop any stale skipped-run flag: this is the reset choke point (selection
        # change etc.), and _draw — the only place the flag is set — does not run on a
        # cleared/empty-results state, so the red alarm would otherwise linger.
        self.flag_label.clear()
        self.info_label.clear()


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
        # The fit stores are keyed (sample_id, fraction), NOT sample_id alone:
        # a fit is strictly per-fraction, so a sid-keyed cache would redisplay
        # one fraction's Rh while the table grays those very rows under a
        # different focus -- the mis-attribution the literal-fraction partition
        # exists to prevent, resurfacing through the display.
        self._cache: Dict[tuple, object] = {}      # (sid, fraction) -> fitted result
        self._cache_sig: Dict[tuple, tuple] = {}   # (sid, fraction) -> input signature
        self._points: Dict[str, list] = {}       # sample_id -> dls_sample_rows metadata
        self._run_points: Dict[tuple, list] = {}   # (sid, fraction) -> last run's all_points
        self._included: Dict[tuple, set] = {}    # (sid, fraction) -> ticked item_ids
        self._suppress_table = False             # re-entrancy guard for checkbox edits
        self._last_sid: Optional[str] = None     # last DLS sample shown (keep-last)
        self._last_ckey: Optional[tuple] = None  # (sample, fraction) shown: the epoch's key
        self._run_epoch = 0                      # async staleness token (view/tick set)
        self._recompute_pending: object = False  # the (sid, fraction) of a Run queued for idle
        self._placeholder = None                 # empty-state Text artist (no fit shown)
        self._build_ui()

    def _build_ui(self) -> None:
        _, left, right = make_split_panels(self, left_min_width=300,
                                           sizes=(360, 740))

        # ---- controls section ----
        controls = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(controls); cl.setContentsMargins(0, 0, 0, 0)
        note = ('Fits Γ = D q² across the sample\'s DLS angles (current Mw '
                'fraction).' if self.kind == 'gamma_q2' else
                'Extrapolates D(c) → c→0 across the sample\'s DLS concentrations '
                '(current Mw fraction).')
        lbl = ThemedLabel(note, role='hint', size=11); lbl.setWordWrap(True)
        cl.addWidget(lbl)
        # Make the data source explicit AND that nothing is computed until Run.
        src = ThemedLabel(
            'Nothing is fitted until you press Run: the table below lists this sample\'s '
            'measurements, and Run computes Γ (and D = Γ/q²) for each from an internal '
            '2nd-order cumulant fit — independent of any saved Correlogram/Distribution '
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

        # ---- per-measurement table section (tick to include) ----
        tsec = QtWidgets.QWidget()
        tl = QtWidgets.QVBoxLayout(tsec); tl.setContentsMargins(0, 0, 0, 0)
        # The Γ-q² fit varies ANGLE at a fixed concentration → group by concentration;
        # the D-c fit varies CONCENTRATION at a fixed angle → group by angle.
        # The group-by field is set with the GroupTickBar below.
        tl.addWidget(section_header(
            'Measurements (tick to include in the fit)',
            'Tick which of this sample\'s measurements enter the fit, then press Run.',
            bullets=[
                'Only this sample\'s DLS measurements appear — it is a single-sample fit.',
                'The fit needs ≥ 2 distinct '
                + ('angles.' if self.kind == 'gamma_q2' else 'concentrations.'),
                'Grayed rows can\'t be ticked — hover a row to see why '
                '(unconfirmed parameters, or a different Mw fraction).',
                'Γ and D fill in only after you Run; ticking never refits on its own.',
                'Ticked rows read blue and light the matching sidebar leaves.',
                'Unticked points still plot as a gray × (labeled "excluded" in the '
                'legend) so you can see what the fit left out.',
                'Ticking both a polarized (VV) and a depolarized (VH) run is refused: '
                'Γ_VH = Γ_VV + 6·D_r is a rotational term, not scatter, so one line '
                'through both fits neither. Tick a single geometry, or use the DDLS '
                'sub-tab, which wants exactly that pair.',
                # A conditional ELEMENT, never a conditional `bullets=` argument:
                # `extract_help.py::_bullets` matches on `ast.List`, so wrapping the list
                # in a concatenation makes it an `ast.BinOp` and the extractor silently
                # returns NOTHING -- which dropped all six of these bullets from the
                # generated companion help doc, with the sync test green because both
                # sides regenerated together. The per-variant expansion below is the
                # established idiom (see the "≥ 2 distinct angles/concentrations" bullet).
                ('A ticked point whose fit gives no usable decay rate (a non-positive '
                 'or failed Γ) is dropped by the fit itself — its angle is marked with a '
                 'dotted line on the plot and named in the notice under the results; '
                 'untick a point you do not trust.'
                 if self.kind == 'gamma_q2' else
                 'A ticked point whose fit gives no usable D (a non-positive decay rate) '
                 'is dropped by the fit itself — its concentration is marked with a '
                 'dotted line on the plot and named in the notice under the results.')]))
        # Empty/ineligible-state hint + NaN-temperature sibling note.
        self.table_note = ThemedLabel('', role='hint', size=10)
        self.table_note.setWordWrap(True); tl.addWidget(self.table_note)
        self.temp_note = ThemedLabel('', role='hint', size=10)
        self.temp_note.setWordWrap(True); tl.addWidget(self.temp_note)
        # Selection helpers: Select all / none + "tick all at X".
        helper = QtWidgets.QWidget()
        hl = QtWidgets.QHBoxLayout(helper); hl.setContentsMargins(0, 0, 0, 0)
        self.select_all_btn = QtWidgets.QPushButton('Select all')
        self.select_all_btn.setToolTip('Tick every eligible measurement of this sample.')
        self.select_all_btn.clicked.connect(self._select_all)
        self.select_none_btn = QtWidgets.QPushButton('Select none')
        self.select_none_btn.setToolTip('Untick every measurement.')
        self.select_none_btn.clicked.connect(self._select_none)
        hl.addWidget(self.select_all_btn)
        hl.addWidget(self.select_none_btn)
        hl.addStretch(1)
        tl.addWidget(helper)
        # "Tick all at X" bulk selector via the shared GroupTickBar, on its own row so it
        # fits the narrow column: concentration in mg/mL for Γ-q² (angle varies), angle in
        # ° for D-c (concentration varies). The bar emits (key, value); we resolve this
        # sample's matching rows and tick them.
        self._group_field_spec = (_GROUP_FIELD_CONC if self.kind == 'gamma_q2'
                                  else _GROUP_FIELD_ANGLE)
        self.group_bar = GroupTickBar((self._group_field_spec,))
        self.group_bar.tickRequested.connect(self._on_group_tick)
        tl.addWidget(self.group_bar)
        self.table = QtWidgets.QTableWidget(0, 0)
        self.table.setHorizontalHeader(RichTextHeaderView(self.table))   # D_app header → rich text
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.itemChanged.connect(self._on_table_changed)
        tl.addWidget(self.table, 1)

        # ---- results section ----
        rsec = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(rsec); rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(QtWidgets.QLabel('Results'))
        self.result_table = _vtable([lbl for lbl, _ in self._result_spec()])
        # A fixed row count, so the table's floor IS its content height (R3.2): the
        # splitter then cannot squeeze the last rows under the pane edge, whatever
        # the two wrap labels below it take (measured at 760 px: a 90-px pane floor
        # left the 4th and 5th rows cut).
        self.result_table.setMinimumHeight(self.result_table.maximumHeight())
        rl.addWidget(self.result_table)
        self.flag_label = ThemedLabel('', role='error', bold=True)
        self.flag_label.setWordWrap(True)
        rl.addWidget(self.flag_label)
        # Passive ⓘ notices carried on the result (e.g. a mixed sample identity) — a
        # neutral qualifier tier, separate from the severity-tracking flag_label above.
        # The analysis emits the same text via warnings.warn (keep-both).
        self.notes_label = ThemedLabel('', size=11)
        self.notes_label.setWordWrap(True)
        rl.addWidget(self.notes_label)

        vstack = make_vertical_plot_stack(
            [controls, tsec, rsec], sizes=[200, 220, 160],
            min_heights=[max(controls.sizeHint().height(), 150), 90, 150])  # rsec: the 5-row results table (R3.2)
        left.addWidget(vstack, 1)

        self.figure = Figure(figsize=(5.2, 4.3), constrained_layout=True)
        self.canvas = make_canvas_expanding(FigureCanvas(self.figure))
        self.ax = self.figure.add_subplot(111)
        self._nav = themed_navtoolbar(self.canvas, self, self.controller)
        right.addWidget(self._nav)
        right.addWidget(self.canvas, 1)
        self.axis_bar = AxisControlBar(self.canvas)
        right.addWidget(self.axis_bar)

    def _result_spec(self):
        """(label, value-fn) per result row, for the results _vtable."""
        df, du = _disp_factor('diffusion'), _disp_unit('diffusion')

        def pm(v, se):
            return format_pm(v * df, (se * df) if se is not None else None)
        if self.kind == 'gamma_q2':
            return [
                (f'D ({du})', lambda r: pm(r.d_m2_s, r.d_se)),
                ('R<sub>h</sub> (nm)', lambda r: format_pm(r.rh_nm, r.rh_se)),
                ('R²', lambda r: format_fixed_sig(
                    r.r_squared, self.controller.settings.no_uncertainty_sig_figs)),
                ('Diffusive?', lambda r: '✓' if r.is_diffusive else '⚠ no'),
                ('angles fitted', lambda r: str(int(np.isfinite(
                    np.asarray(r.gamma_s_inv, dtype=float)).sum()))),
            ]
        return [
            (f'D₀ ({du})', lambda r: pm(r.d0_m2_s, r.d0_se)),
            ('R<sub>h,0</sub> (nm)', lambda r: format_pm(r.rh0_nm, r.rh0_se)),
            ('k<sub>D</sub> (mL/g)', lambda r: format_pm(r.kd_mL_per_g, r.kd_se)),
            ('R²', lambda r: format_fixed_sig(
                r.r_squared, self.controller.settings.no_uncertainty_sig_figs)),
            ('concs fitted', lambda r: str(r.n_concentrations)),
        ]

    def _sample_id(self) -> Optional[str]:
        if self.item_id is None:
            return None
        return self.controller.sample_id_of(self.item_id)

    def selected_item_ids(self) -> list:
        """The ticked measurements for the current sample (sidebar-mirror contract)."""
        sid = self._sample_id()
        return sorted(self._included.get(self._ckey(sid), set())) if sid else []

    def _fraction(self) -> Optional[str]:
        lm = self.controller.workspace.measurements.get(self.item_id)
        return lm.committed_params.get('mw_fraction') if lm is not None else None

    def _ckey(self, sid: str) -> tuple:
        """Fit-store key: a fit is strictly per (sample, focused fraction)."""
        return (sid, self._fraction())

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
        set_flag(self.notes_label, '', problem=False)   # drop stale result notes
        sid = self._sample_id()
        ckey = self._ckey(sid) if sid is not None else None
        # The epoch is keyed on (sample, fraction), never the sample alone: a
        # fraction switch within one sample is a re-point too, and a fit landing
        # after it must not paint the fraction no longer focused (the SLS tab's
        # fraction-change bump is the pattern).
        if ckey != self._last_ckey:
            self._run_epoch += 1     # re-pointed: drop any in-flight run's display
        self._last_ckey = ckey
        if not (runnable and sid is not None):
            self.table.setRowCount(0)
            self.run_button.setEnabled(False)
            self.export_button.setEnabled(False)
            self._set_helpers_enabled(False)
            self.temp_note.clear()
            self.table_note.setText(
                'Select a DLS measurement in the sidebar to see this sample\'s '
                'measurements here.')
            self._clear_results()
            self._clear('Select a DLS measurement.')
            self.status.clear()
            return
        # Enumerate the sample's measurements as METADATA ONLY — no fitting on focus.
        # Γ/D appear only after an explicit Run.
        self._points[sid] = self.controller.dls_sample_rows(
            sid, self.kind, self._fraction())
        fresh = {r['item_id'] for r in self._points[sid] if r['ok']}
        # The ticked set is per (sample, fraction) like the fit stores: the rows are
        # per-fraction, so a sid-keyed set intersected with another fraction's rows
        # would empty on every fraction switch and never come back.
        prev = self._included.get(ckey)
        # Auto-tick all eligible rows on first focus (ticking now only selects — it
        # computes nothing), so the common "use every point" case is one Run away.
        self._included[ckey] = (prev & fresh) if prev is not None else set(fresh)
        # Restore this (sample, fraction)'s last-run per-point values (table Γ/D +
        # graying) — a run of another fraction stays in its own slot, unseen here.
        cached = self._cache.get(ckey)
        self._run_points[ckey] = list(getattr(cached, 'all_points', None) or [])
        self._refresh_table(sid)
        self._populate_group_combo(sid)
        self._update_run_enabled(sid)
        self._set_helpers_enabled(True)
        self.export_button.setEnabled(ckey in self._cache)
        self._last_sid = sid                      # remember for keep-last
        # Empty-state / eligibility guidance.
        if not self._points[sid]:
            self.table_note.setText('This sample has no DLS measurements.')
        elif not fresh:
            self.table_note.setText(
                'No measurements have confirmed parameters yet — set them in the Data '
                'tab and press Update, then return here.')
        else:
            self.table_note.clear()
        n_pending = self.controller.dls_pending_temperature_siblings(sid)
        if n_pending:
            self.temp_note.setText(
                f'{n_pending} measurement(s) of this polymer/solvent have no confirmed '
                'temperature and are grouped separately — confirm it in the Data tab to '
                'include them here.')
        else:
            self.temp_note.clear()
        if ckey in self._cache:
            self._fill_results(self._cache[ckey])
            self._draw(self._cache[ckey])
            if self._cache_sig.get(ckey) != self._run_signature(ckey):
                self.status.setText(_STALE_HINT)
        else:
            self._clear_results()
            self._clear('Tick measurements, then Run.')

    def _run_signature(self, ckey: tuple) -> tuple:
        """Everything a Γ-q² / D-vs-c fit for (sample, fraction) reads beyond its raw
        data: the committed params of the TICKED measurements (their ids ride in the
        signature, so a tick change moves it) and the settings the engine seeds --
        skip_initial_channels, the cumulant method, the SE estimator (the ± is
        estimator-routed; the confirm in Settings promises a recompute). Stored with
        the cached fit at dispatch and compared on redisplay, so a cached fit is flagged
        stale exactly when a re-Run would read different inputs."""
        s = self.controller.settings
        return (self.controller.committed_signature(self._included.get(ckey, set())),
                s.skip_initial_channels, s.cumulant_method, s.se_estimator)

    # ---- per-measurement table (tick to include) ----
    def _refresh_table(self, sid: str) -> None:
        pts = self._points.get(sid, [])
        inc = self._included.get(self._ckey(sid), set())
        cu = _disp_unit('concentration')
        if self.kind == 'gamma_q2':
            headers = ['', 'Angle (°)', f'c ({cu})',
                       f'Γ ({_disp_unit("decay_rate")})',
                       f'D<sub>app</sub> ({_disp_unit("diffusion")})']
        else:
            headers = ['', f'c ({cu})', 'Angle (°)',
                       f'D<sub>app</sub> ({_disp_unit("diffusion")})']
        dfc, dfg, dfd = (_disp_factor('concentration'), _disp_factor('decay_rate'),
                         _disp_factor('diffusion'))
        # Per-point Γ/D + quality come from the last Run — '—'
        # until then, so nothing is computed on focus.
        run_by_id = {p['item_id']: p
                     for p in self._run_points.get(self._ckey(sid), [])}
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
                chk.setToolTip(self._row_tooltip(row, run_by_id.get(row['item_id'])))
                t.setItem(r, 0, chk)
                ang = '—' if row['angle_deg'] is None else f"{row['angle_deg']:g}"
                conc = ('—' if row['concentration_g_per_mL'] is None
                        else f"{row['concentration_g_per_mL'] * dfc:g}")
                pt = run_by_id.get(row['item_id'])
                has_val = pt is not None and pt['quality'] in FIT_ELIGIBLE_QUALITIES
                sig = self.controller.settings.no_uncertainty_sig_figs   # per-angle Γ/D: ±-less
                gtxt = format_fixed_sig(pt['gamma_s_inv'] * dfg, sig) if has_val else '—'
                dtxt = format_fixed_sig(pt['d_app_m2_s'] * dfd, sig) if has_val else '—'
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

    def _row_tooltip(self, row: dict, pt: Optional[dict]) -> str:
        """Explain a row's eligibility / quality so the user knows why it is grayed or
        excluded."""
        reason = row.get('reason')
        if reason == 'unbuilt':
            return ('Parameters not confirmed yet — set them in the Data tab and press '
                    'Update to make this measurement eligible.')
        if reason == 'other_fraction':
            frac = row.get('mw_fraction')
            return (f'A different Mw fraction ({frac!r}) than the focused measurement — '
                    'a single fit must not mix fractions. Focus a measurement of this '
                    'fraction to include it.')
        if pt is not None:
            q = pt['quality']
            if q == 'fit_failed':
                return 'Cumulant fit failed for this correlogram — excluded from the fit.'
            if q == 'nonpositive_gamma':
                return ('The cumulant fit converged on a non-positive decay rate '
                        '(Γ ≤ 0) — non-physical, so no D can be formed. The point is '
                        'still passed to the fit, which drops it from the line (Γ vs q² '
                        'and D vs c alike) and says so in the notes; untick it to leave '
                        'it out explicitly. Usually a weak scatterer on a noisy '
                        'baseline — check the correlogram and the delay window.')
            if q == 'nonpositive_q':
                return 'Non-physical scattering vector (q ≤ 0) — excluded from the fit.'
            if q == 'high_pdi':
                return (f'PDI exceeds the cumulant validity limit '
                        f'({CUMULANT_PDI_VALIDITY_LIMIT:g}) — the point is included but '
                        'its Γ is unreliable; consider unticking it.')
        return 'Eligible — ticked rows enter the fit when you press Run.'

    def _tint_row(self, r: int, on: bool) -> None:
        """Color a table row to match the shared idiom: ticked rows read
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
        inc = self._included.setdefault(self._ckey(sid), set())
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
        # Ticking only selects — it never refits or clears the plot.
        # A shown result just gets marked stale until the next Run.
        enough = self._update_run_enabled(sid)
        if self._ckey(sid) in self._cache and enough:
            self.status.setText('Selection changed — press Run to refresh.')

    # ---- selection helpers (Select all/none + tick-all-at-X) ----
    def _group_field(self) -> str:
        """The metadata key the 'tick all at X' selector groups by: concentration for
        the Γ-q² tab (angle varies), angle for the D-c tab (concentration varies)."""
        return self._group_field_spec[0]

    def _set_helpers_enabled(self, on: bool) -> None:
        self.select_all_btn.setEnabled(on)
        self.select_none_btn.setEnabled(on)
        self.group_bar.setEnabled(on)

    def _populate_group_combo(self, sid: str) -> None:
        """Fill the 'tick all at X' combo with the distinct values among ELIGIBLE rows
        (formatted with units by the GroupTickBar — mg/mL for concentration, ° for angle)."""
        field = self._group_field()
        vals = sorted({r[field] for r in self._points.get(sid, [])
                       if r['ok'] and r[field] is not None})
        self.group_bar.set_values(field, vals)

    def _apply_included(self, sid: str, ids: set) -> None:
        """Set the ticked include-set for a sample and refresh the dependent UI once
        (table, run-enabled, sidebar mirror) — the shared path for the bulk helpers."""
        self._included[self._ckey(sid)] = set(ids)
        self._refresh_table(sid)
        self.selectionChanged.emit()
        self._run_epoch += 1
        enough = self._update_run_enabled(sid)
        if self._ckey(sid) in self._cache and enough:
            self.status.setText('Selection changed — press Run to refresh.')

    @QtCore.Slot()
    def _select_all(self) -> None:
        sid = self._sample_id()
        if sid is None:
            return
        self._apply_included(sid, {r['item_id'] for r in self._points.get(sid, [])
                                   if r['ok']})

    @QtCore.Slot()
    def _select_none(self) -> None:
        sid = self._sample_id()
        if sid is None:
            return
        self._apply_included(sid, set())

    def _on_group_tick(self, field: str, target) -> None:
        """Add every eligible measurement sharing the chosen value to the ticked set
        (GroupTickBar.tickRequested handler)."""
        sid = self._sample_id()
        if sid is None or target is None:
            return
        form = _IDENTITY_FORMS[field]
        add = {r['item_id'] for r in self._points.get(sid, [])
               if r['ok'] and r[field] is not None
               and form(r[field]) == form(target)}
        self._apply_included(sid, self._included.get(self._ckey(sid), set()) | add)

    def showEvent(self, event) -> None:
        """Re-populate from the current focus when the tab becomes visible, so
        switching to it shows content without a sidebar re-click."""
        super().showEvent(event)
        if self.item_id is not None:
            self.set_measurement(self.item_id, self._runnable)

    def _distinct_keys(self, sid: str) -> int:
        """Distinct angles (Γq²) / concentrations (D-vs-c) among the ticked ok points
        (the engine needs ≥2 unique to fit)."""
        inc = self._included.get(self._ckey(sid), set())
        by_id = {r['item_id']: r for r in self._points.get(sid, [])}
        key_field = ('angle_deg' if self.kind == 'gamma_q2'
                     else 'concentration_g_per_mL')
        form = _IDENTITY_FORMS[key_field]
        keys = set()
        for iid in inc:
            r = by_id.get(iid)
            if not (r and r['ok']):
                continue
            k = r[key_field]
            if k is not None:
                keys.add(form(k))
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

    def _recompute(self, sid: str, fraction=_FOCUSED) -> None:
        """Fit over the ticked subset (on the worker thread — it builds + fits
        every included measurement) and refresh the plot + results table.
        `fraction` is the focused one, unless a deferred Run re-enters with the
        (sample, fraction) it was pressed on."""
        frac = self._fraction() if fraction is _FOCUSED else fraction
        # Key the stores by the RUN's (sample, fraction), captured now, never by
        # the focus at completion time — the focus may have moved while the
        # worker fitted this one.
        ckey = (sid, frac)
        if runner().is_busy:
            # Can't dispatch now. Do NOT bump the epoch (that would drop the
            # in-flight fit with nothing to replace it); instead retry once the
            # worker frees -- against the (sample, fraction) Run was pressed on,
            # whatever the sidebar points at by then.
            first = not self._recompute_pending
            self._recompute_pending = ckey   # the LATEST press wins the slot
            if first:
                run_when_idle(self._flush_recompute)
            self.status.setText(BUSY_NOTICE)
            return
        include_ids = set(self._included.get(ckey, set()))
        sig = self._run_signature(ckey)
        if ckey == self._last_ckey:
            self._run_epoch += 1            # this run supersedes any in flight
            epoch = self._run_epoch
            self.status.setText('Fitting in the background…')
        else:
            # A deferred Run whose target is no longer on screen: fit it (what
            # the user asked for), but never paint it over the view now shown.
            epoch = None

        def done(res) -> None:
            # The fit lands in ITS OWN slot whatever is focused now (a refocus
            # shows it); only the display is gated on the view being the one this
            # run started from.
            self._cache[ckey] = res
            self._cache_sig[ckey] = sig
            self._run_points[ckey] = list(getattr(res, 'all_points', None) or [])
            if epoch != self._run_epoch:
                return                      # view / tick set changed — stale VIEW
            # Fold the run's per-point values into the table + graying:
            # Γ/D now appear, and quality flags become hoverable.
            self._refresh_table(sid)
            self.export_button.setEnabled(True)
            self._fill_results(res)
            self._draw(res)
            self.status.clear()

        def fail(exc: BaseException) -> None:
            # The dialog names the run's own sample; the status line is written only
            # when the view is still the one this run started from (the `done` guard).
            QtWidgets.QMessageBox.critical(
                self, 'Analysis failed',
                f'Could not run this analysis on sample {sid}.\n\n'
                f'{humanize_field_names(str(exc))}')
            if epoch == self._run_epoch:
                self.status.setText('Analysis failed — see dialog.')

        description = ('Γ vs q² fit' if self.kind == 'gamma_q2'
                       else 'D vs c extrapolation')
        runner().try_submit(
            lambda: self._run_engine(sid, frac, include_ids),
            done, fail, description=description,
            busy_widgets=(self.run_button,))

    def _flush_recompute(self) -> None:
        """Deferred retry of a recompute refused while the worker was busy — against
        the (sample, fraction) Run was pressed on, never whatever is focused now
        (a Run the user never pressed on the focused sample must not fit it)."""
        target, self._recompute_pending = self._recompute_pending, False
        if not target or not self._runnable:
            return
        sid, frac = target
        # `_distinct_keys(sid) >= 2` is the same precondition `_recompute`'s live path
        # checks before dispatching: the deferred retry re-reads the CURRENT ticked set,
        # so a tick change while the worker was busy can leave fewer than two usable
        # points and the engine's ValueError would reach the user as a modal error.
        if (sid in self.controller.workspace.samples
                and self._distinct_keys(sid) >= 2):
            self._recompute(sid, frac)

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
        set_flag(self.notes_label, '', problem=False)   # drop stale result notes

    @QtCore.Slot()
    def _on_export(self) -> None:
        sid = self._sample_id()
        if sid is None or self._ckey(sid) not in self._cache:
            return
        res = self._cache[self._ckey(sid)]
        if self.kind == 'gamma_q2':
            # 'D_vs_c' below: kept over the exporter's longer 'concentration_
            # extrapolation' stem — it names the plotted axes (deliberate).
            name = sample_suggested_name(self.controller, sid, 'gamma_q2')
            fn = self.controller.export_gamma_q2
        else:
            name = sample_suggested_name(self.controller, sid, 'D_vs_c')
            fn = self.controller.export_concentration_extrapolation
        status = export_to_csv(self, self.controller, name, lambda p: fn(res, p))
        if status:
            self.status.setText(status)

    def _draw(self, res) -> None:
        if self._placeholder is not None:        # leaving the empty-state: drop the message
            self._placeholder.remove()
            self._placeholder = None
        self.ax.clear()
        self.ax.set_visible(True)                # restore after an empty _clear() hid it
        self.axis_bar.attach(self.ax)            # re-bind the axis controls (detached while empty)
        sid = self._sample_id()
        # Plot saves suggest the same stem as the CSV export for this analysis.
        set_plot_export_name(self.canvas, sample_suggested_name(
            self.controller, sid,
            'gamma_q2' if self.kind == 'gamma_q2' else 'D_vs_c'))
        # Glyph-less flag text + a `problem` tier; set_flag prepends ⚠/ⓘ to match the
        # color (style guide §5, R5.1). The non-diffusive note is the genuine problem;
        # the ± caveat is a neutral qualifier. A single label carries one tier, so — as
        # on the SLS tab — a real problem tints the whole (possibly multi-line) label red.
        # Both branches ALSO read the pooled fit's own `rh_reliable` / `reliability_note`
        # (a flagged per-point fit inside the line, a pooled Rh outside the size range,
        # no viscosity) -- the same pair the per-measurement fits show, so the verdict
        # that reaches the Cross-Sample promotion is the one shown here.
        rh_flag = ('' if getattr(res, 'rh_reliable', True)
                   else getattr(res, 'reliability_note', '') or 'Rh is unreliable.')
        if self.kind == 'gamma_q2':
            plot_gamma_q2(res, ax=self.ax)
            self.ax.set_title('Γ vs q² (multi-angle)')
            problem = (not res.is_diffusive) or bool(rh_flag)
            # The apparent qualifier is said on BOTH branches: a diffusive q→0 Rh at one
            # concentration is still apparent (invariant 7), and this panel is the first
            # place the number is read -- the mode-family sibling says it unconditionally.
            if res.is_diffusive:
                flag = ('Rh is apparent (one concentration'
                        + ('; z-average from the cumulant Γ' if res.gamma_source == 'cumulant'
                           else '') + ').')
            elif not math.isfinite(res.r_squared):
                # No R² means the linearity criteria were never evaluated (the
                # `exporting/export.py` sibling's `not math.isfinite(result.r_squared)`
                # arm): fewer than two usable angles -- where `rh_flag` below carries the
                # result's own reliability note -- or a degenerate R² denominator on a
                # normal run, where nothing else names the reason. A red label always
                # says why (round 1 f12).
                flag = ('' if rh_flag else
                        'no Γ-vs-q² line could be judged (R² is undefined), so the '
                        'linearity criteria were not evaluated — Rh is apparent.')
            else:
                flag = ('not purely diffusive: Γ is not linear through the origin in q² '
                        '— Rh is apparent; check for a slow mode or internal motion (qRg>1).')
            if rh_flag:
                flag = (flag + '\n' if flag else '') + rh_flag
            if res.d_se is not None:
                # The estimator travels with the ± it qualifies (invariant 8 clause A),
                # in the DDLS tab's words.
                flag = (flag + '\n' if flag else '') + '(± statistical only)' + _ols_marker(getattr(res, 'se_estimator', 'hc3'))
        else:
            plot_concentration_extrapolation(res, ax=self.ax)
            self.ax.set_title('D vs c → infinite dilution')
            # Style guide R5.2 — the same condition takes the same tier across tabs. An
            # unreliable value is red `error` on the SLS tab (`_mw_reliability_flag`, keyed
            # on the `mw_reliable` CONJUNCTION and preferring the result's own
            # `reliability_note`), and non-diffusive is red on the Γ-q² branch just above.
            # This branch once had `problem = False` hard-coded, so a D0 ≤ 0 sign flip
            # reached the user as nothing at all; `kd_reliable` is the DLS sibling of
            # `mw_reliable` (rule 10). The result keeps ONE note for its kD and Rh0 flags.
            problem = not (res.kd_reliable and getattr(res, 'rh_reliable', True))
            flag = ('' if not problem else
                    (res.reliability_note
                     or 'kD is unreliable (non-physical or degenerate fit).'))
            if res.d0_se is not None or res.kd_se is not None:
                flag = (flag + '\n' if flag else '') + '(± statistical only)' + _ols_marker(getattr(res, 'se_estimator', 'hc3'))
        drew_excluded = False
        if sid is not None:
            drew_excluded = self._gray_points(sid)   # gray the unticked (excluded) ok points
        if drew_excluded:
            # The gray ×'s are added AFTER the analysis layer built the legend, so
            # re-issue it (preserving any title, e.g. the D-vs-c k_D title) to give the
            # 'excluded' marker an entry.
            reissue_legend_preserving_title(self.ax, fontsize=9)
        self.canvas.draw_idle()
        self.axis_bar.attach(self.ax)
        set_flag(self.flag_label, flag, problem=problem)
        # Passive ⓘ result notes (e.g. a mixed sample identity), neutral tier — the
        # same text the analysis also emits via warnings.warn (keep-both).
        set_flag(self.notes_label, '\n'.join(getattr(res, 'notes', ())), problem=False)

    def _gray_points(self, sid: str) -> bool:
        """Gray × the computed points that are NOT ticked, from the run's per-point
        values (by item_id, so replicates are disambiguated). Returns True if any
        excluded point was drawn (so the caller can add a legend entry). Only the
        first marker is labeled so the legend shows one 'excluded' entry, not one
        per point."""
        inc = self._included.get(self._ckey(sid), set())
        if self.kind == 'gamma_q2':
            xf, yf, xk, yk = (_disp_factor('scattering_q2'),
                              _disp_factor('decay_rate'), 'q2_m2', 'gamma_s_inv')
        else:
            xf, yf, xk, yk = (_disp_factor('concentration'),
                              _disp_factor('diffusion'),
                              'concentration_g_per_mL', 'd_app_m2_s')
        xy = [(pt[xk] * xf, pt[yk] * yf)
              for pt in self._run_points.get(self._ckey(sid), [])
              if not (pt['quality'] not in FIT_ELIGIBLE_QUALITIES or pt['item_id'] in inc
                      or pt[xk] is None or not math.isfinite(pt[xk]))]
        return draw_excluded_markers(self.ax, xy, zorder=4)

    def _clear(self, message: str) -> None:
        if self._placeholder is not None:        # ax.clear() doesn't drop a figure.text
            self._placeholder.remove()
            self._placeholder = None
        self.ax.clear()
        # Hide the axis and show one centred grey message via the shared helper — the
        # clean empty-state look (a blank plot box reads as broken). _draw restores it.
        self._placeholder = show_plot_placeholder(
            self.figure, self.canvas, message, axes=(self.ax,))
        self.axis_bar.attach(None)               # nothing to drive while empty


# Why 'Record mode set' cannot act yet, keyed by the controller's reason code
# (the _INELIGIBLE_REASON pattern: the wording lives at the display surface).
_RECORD_NOT_READY = {
    'no_map': 'Run the mode map first — a mode set records what the map '
              'currently shows.',
    'no_families': 'Add a family and pick map points into it first; the '
                   'recorded set snapshots the picked families.',
    'no_matches': 'No picked family currently matches a map point (the '
                  'extraction changed since the picks were made) — re-pick, '
                  'or re-run with the original source.',
}


class _ModeMapTab(QtWidgets.QWidget):
    """The mode map: every extracted mode of every run in (q², Γ) space, with
    user-picked mode families and per-family Γ-vs-q² fits.

    No association machinery: which points form a family is the USER's pick,
    made by clicking map points into the active family (the SLS mask
    click-to-toggle pattern; picks are value-keyed, so a re-extraction orphans
    a pick rather than re-aiming it). The through-origin verdict is a
    tolerance-carrying LABEL — a slow mode failing it is the interesting
    finding, not an error — and every family's Rh is apparent + mode-resolved.
    The map facets by polarization state, one state shown at a time, so a
    family can never span states.
    """

    selectionChanged = QtCore.Signal()           # ticked include-set changed
    _PICK_RADIUS_PX = 12.0

    _SOURCES = [('Double exponential', 'double'), ('CONTIN', 'contin'),
                ('NNLS', 'nnls')]
    _STATE_LABELS = {'VV': 'VV (polarized)', 'VH': 'VH (depolarized)',
                     None: 'Unpolarized (VU / untagged)'}

    def __init__(self, controller, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.item_id: Optional[str] = None
        self._runnable = False
        self._points: Dict[str, list] = {}       # sid -> metadata rows
        self._included: Dict[tuple, set] = {}    # (sid, fraction) -> ticked item_ids
        self._cache_sig: Dict[tuple, tuple] = {}  # (sid, frac) -> input signature
        self._suppress_table = False
        self._suppress_family = False
        self._last_sid: Optional[str] = None
        self._last_ckey: Optional[tuple] = None  # (sample, fraction) shown: the epoch's key
        self._run_epoch = 0
        self._recompute_pending: object = False  # the (sid, fraction) of a Run queued for idle
        self._placeholder = None
        self._build_ui()

    def _build_ui(self) -> None:
        _, left, right = make_split_panels(self, left_min_width=300,
                                           sizes=(360, 740))

        # ---- controls section ----
        controls = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(controls); cl.setContentsMargins(0, 0, 0, 0)
        cl.addWidget(section_header(
            'Mode Map', 'Extract every relaxation mode of every run, then click '
            'points into families to fit each mode separately.',
            bullets=[
                'Run extracts the modes of each ticked run with the chosen '
                'source and plots them all in (q², Γ) space.',
                'Add a family, select it, then click map points to assign them '
                '— clicking again removes; clicking a point of another family '
                'moves it (the status line says so).',
                'Each family with points at ≥ 2 angles is fitted: Γ = D·q² '
                'through the origin, plus the a·q² + b·q⁴ deviation form.',
                'Every family Rh is APPARENT and MODE-RESOLVED; whether the '
                'family passes through the origin is a label, never a filter.',
                'Marker shape and fill each carry one warning: a HOLLOW point '
                'sits below the amplitude floor (imprecise), a SQUARE point '
                'never finished decaying inside its delay window (its rate is '
                'extrapolated). A hollow square is both.',
                'The map shows one polarization at a time — a family can never '
                'mix VV with VH (Γ_VH carries a rotational term).',
                'Designate one family "fast" and one "slow" to see the '
                'amplitude report (A_f/A_s vs sin²(θ/2)).',
                'Both distribution sources invert on the Rh grid set in the '
                'Settings tab; the α selection there reaches CONTIN only — '
                'NNLS is the unregularized case and has no α. A denser Rh '
                'grid resolves more closely-spaced modes and costs sharply '
                'more time per run.',
                'The Distribution tab starts from those same Settings but has '
                'its own per-run fields, and it applies a τ window and baseline '
                'the map does not — so a run fitted there can resolve '
                'differently from the same run on this map.']))
        srow = QtWidgets.QWidget()
        sl = QtWidgets.QFormLayout(srow); sl.setContentsMargins(0, 0, 0, 0)
        self.source_combo = QtWidgets.QComboBox()
        for label, code in self._SOURCES:
            self.source_combo.addItem(label, code)
        self.source_combo.setToolTip(
            'Which fit supplies each run\'s modes: the two-mode double-'
            'exponential (fast), or a CONTIN/NNLS distribution\'s resolved '
            'peaks (slower per run). Both distribution sources invert on the Rh '
            'grid from the Settings tab, and CONTIN also on the α selection '
            'there (NNLS has none); the inversion cost rises steeply with grid '
            'resolution. The Distribution tab seeds its '
            'per-run fields from the same Settings but can be changed there, '
            'and applies its own τ window and baseline. See the '
            'Theory-and-Equations-Guide.')
        sl.addRow('Source:', self.source_combo)
        self.facet_combo = QtWidgets.QComboBox()
        self.facet_combo.setToolTip(
            'The polarization state shown (and pickable). Γ_VH = Γ_VV + 6·D_r '
            'is a physical difference, so states are never pooled on one map.')
        self.facet_combo.currentIndexChanged.connect(self._on_facet_changed)
        sl.addRow('Polarization:', self.facet_combo)
        cl.addWidget(srow)
        self.run_button = QtWidgets.QPushButton('Run')
        self.run_button.setToolTip(BACKGROUND_RUN_TOOLTIP)
        self.run_button.clicked.connect(self._on_run)
        cl.addWidget(self.run_button)
        self.export_button = QtWidgets.QPushButton('Export CSV…')
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._on_export)
        cl.addWidget(self.export_button)
        self.bounds_button = QtWidgets.QPushButton('Integration bounds…')
        self.bounds_button.setEnabled(False)
        self.bounds_button.setToolTip(
            'Set manual Rh integration intervals per run for the CONTIN/NNLS '
            'sources, replacing the automatic lobe split where peaks overlap. '
            'Overlapping peaks violate the wide-separation condition of the '
            'mode decomposition — for genuine overlap the double-exponential\'s '
            'model amplitudes are the better route. Bounds apply at the next '
            'Run and travel as provenance; the parametric source ignores them. '
            'See the Theory-and-Equations-Guide.')
        self.bounds_button.clicked.connect(self._on_bounds)
        cl.addWidget(self.bounds_button)
        self.record_set_button = QtWidgets.QPushButton('Record mode set…')
        self.record_set_button.setEnabled(False)
        self.record_set_button.setToolTip(
            'Snapshot the picked families — their amplitudes, decay rates, '
            'regularization and integration bounds — into a recorded mode '
            'set. Recorded sets appear on the SLS tab\'s Mode-Resolved '
            'sub-tab, where any SLS intensity series of this sample can be '
            'partitioned by them. A recorded set never follows later re-fits '
            '— record again for a fresh one.')
        self.record_set_button.clicked.connect(self._on_record_set)
        cl.addWidget(self.record_set_button)
        self.status = QtWidgets.QLabel('')
        self.status.setWordWrap(True)
        cl.addWidget(self.status)

        # ---- measurements table (tick to include) ----
        tsec = QtWidgets.QWidget()
        tl = QtWidgets.QVBoxLayout(tsec); tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(section_header(
            'Measurements (tick to include in the map)',
            'Tick which runs are extracted when you press Run.',
            bullets=[
                'Only this sample\'s DLS measurements appear.',
                'Grayed rows can\'t be ticked — hover a row to see why.',
                'A run whose extraction fails contributes no points; the '
                'reason appears under the results.']))
        self.table_note = ThemedLabel('', role='hint', size=10)
        self.table_note.setWordWrap(True); tl.addWidget(self.table_note)
        helper = QtWidgets.QWidget()
        hl = QtWidgets.QHBoxLayout(helper); hl.setContentsMargins(0, 0, 0, 0)
        self.select_all_btn = QtWidgets.QPushButton('Select all')
        self.select_all_btn.clicked.connect(self._select_all)
        self.select_none_btn = QtWidgets.QPushButton('Select none')
        self.select_none_btn.clicked.connect(self._select_none)
        hl.addWidget(self.select_all_btn); hl.addWidget(self.select_none_btn)
        hl.addStretch(1)
        tl.addWidget(helper)
        self.table = QtWidgets.QTableWidget(0, 0)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(120)  # ~4 rows + header at the shipped 760 px height (R3.2)
        self.table.itemChanged.connect(self._on_table_changed)
        tl.addWidget(self.table, 1)

        # ---- families + results section ----
        fsec = QtWidgets.QWidget()
        fl = QtWidgets.QVBoxLayout(fsec); fl.setContentsMargins(0, 0, 0, 0)
        fl.addWidget(section_header(
            'Families', 'The selected family is the one your clicks assign '
            'points to.',
            bullets=[
                'Add creates a family; double-click renames it.',
                'The colour/marker chip matches the family\'s map points.',
                'Role (fast/slow) feeds the amplitude report; at most one '
                'family holds each role.',
                'Delete removes the family and its results.']))
        frow = QtWidgets.QWidget()
        fr = QtWidgets.QHBoxLayout(frow); fr.setContentsMargins(0, 0, 0, 0)
        self.add_family_btn = QtWidgets.QPushButton('Add family')
        self.add_family_btn.clicked.connect(self._on_add_family)
        self.del_family_btn = QtWidgets.QPushButton('Delete')
        self.del_family_btn.clicked.connect(self._on_del_family)
        self.role_combo = QtWidgets.QComboBox()
        self.role_combo.addItem('—', '')
        self.role_combo.addItem('fast', 'fast')
        self.role_combo.addItem('slow', 'slow')
        self.role_combo.setToolTip('Designate the selected family for the '
                                   'amplitude report.')
        self.role_combo.currentIndexChanged.connect(self._on_role_changed)
        fr.addWidget(self.add_family_btn); fr.addWidget(self.del_family_btn)
        fr.addWidget(QtWidgets.QLabel('Role:')); fr.addWidget(self.role_combo)
        fr.addStretch(1)
        fl.addWidget(frow)
        self.family_list = QtWidgets.QListWidget()
        self.family_list.setMinimumHeight(80)
        self.family_list.currentRowChanged.connect(self._on_family_selected)
        self.family_list.itemChanged.connect(self._on_family_renamed)
        fl.addWidget(self.family_list)
        self.result_table = _vtable([lbl for lbl, _ in self._result_spec()])
        fl.addWidget(self.result_table)
        self.flag_label = ThemedLabel('', size=11)
        self.flag_label.setWordWrap(True)
        fl.addWidget(self.flag_label)
        self.notes_label = ThemedLabel('', size=11)
        self.notes_label.setWordWrap(True)
        fl.addWidget(self.notes_label)

        vstack = make_vertical_plot_stack(
            [controls, tsec, fsec], sizes=[210, 170, 220],
            min_heights=[max(controls.sizeHint().height(), 150), 130, 120])
        left.addWidget(vstack, 1)

        # ---- right: view combo + canvas ----
        rw = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(rw); rl.setContentsMargins(0, 0, 0, 0)
        vrow = QtWidgets.QWidget()
        vl = QtWidgets.QHBoxLayout(vrow); vl.setContentsMargins(0, 0, 0, 0)
        vl.addWidget(QtWidgets.QLabel('View:'))
        self.view_combo = QtWidgets.QComboBox()
        self.view_combo.addItem('Mode map', 'map')
        self.view_combo.addItem('Amplitude vs q', 'amplitude')
        self.view_combo.setToolTip(
            'Amplitude vs q needs a "fast" and a "slow" family (Role above): '
            'it plots A_f/A_s against sin²(θ/2). No ± anywhere — single-run '
            'amplitudes carry no uncertainty.')
        self.view_combo.currentIndexChanged.connect(lambda _i: self._draw())
        vl.addWidget(self.view_combo); vl.addStretch(1)
        rl.addWidget(vrow)
        self.figure = Figure(figsize=(5.2, 4.3), constrained_layout=True)
        self.canvas = make_canvas_expanding(FigureCanvas(self.figure))
        self.ax = self.figure.add_subplot(111)
        self._nav = themed_navtoolbar(self.canvas, self, self.controller)
        rl.addWidget(self._nav)
        rl.addWidget(self.canvas, 1)
        self.axis_bar = AxisControlBar(self.canvas)
        rl.addWidget(self.axis_bar)
        right.addWidget(rw, 1)
        self.canvas.mpl_connect('button_press_event', self._on_canvas_click)

    def _result_spec(self):
        df, du = _disp_factor('diffusion'), _disp_unit('diffusion')

        def pm(v, se):
            return format_pm(v * df, (se * df) if se is not None else None)
        sig = lambda x: format_fixed_sig(  # noqa: E731 — tiny local formatter
            x, self.controller.settings.no_uncertainty_sig_figs)
        return [
            (f'D ({du})', lambda r: pm(r.d_m2_s, r.d_se)),
            ('R<sub>h</sub> (nm)', lambda r: format_pm(r.rh_nm, r.rh_se)),
            ('R²', lambda r: sig(r.r_squared)),
            ('Through origin?', lambda r: '✓' if r.is_diffusive else '⚠ no'),
            (f'a ({du})', lambda r: pm(r.a_m2_s, r.a_se)),
            ('b (m⁴/s)', lambda r: format_pm(r.b_m4_s, r.b_se)),
            ('modes fitted', lambda r: str(r.angles_deg.size)),
        ]

    # ---- identity / selection plumbing (the _SampleAnalysisTab shapes) ----
    def _sample_id(self) -> Optional[str]:
        if self.item_id is None:
            return None
        return self.controller.sample_id_of(self.item_id)

    def selected_item_ids(self) -> list:
        sid = self._sample_id()
        return sorted(self._included.get(self._ckey(sid), set())) if sid else []

    def _fraction(self) -> Optional[str]:
        lm = self.controller.workspace.measurements.get(self.item_id)
        return lm.committed_params.get('mw_fraction') if lm is not None else None

    def _ckey(self, sid: str) -> tuple:
        return (sid, self._fraction())

    def _map_result(self, sid: str):
        return self.controller.results.get(('mode_map',) + self._ckey(sid))

    def _family_fits(self, sid: str) -> dict:
        return self.controller.results.get(
            ('mode_family',) + self._ckey(sid)) or {}

    def set_measurement(self, item_id: Optional[str], runnable: bool) -> None:
        if item_id is not None and not runnable and self._last_sid is not None:
            return                                    # keep-last navigation
        self.item_id = item_id
        self._runnable = runnable
        sid = self._sample_id()
        ckey = self._ckey(sid) if sid is not None else None
        if ckey != self._last_ckey:      # (sample, fraction), like the Γ-q² tab
            self._run_epoch += 1
        self._last_ckey = ckey
        if not (runnable and sid is not None):
            self.table.setRowCount(0)
            self.family_list.clear()
            self.run_button.setEnabled(False)
            self.export_button.setEnabled(False)
            self.bounds_button.setEnabled(False)
            self.record_set_button.setEnabled(False)
            self.table_note.setText(
                'Select a DLS measurement in the sidebar to build this '
                'sample\'s mode map.')
            self._clear_results()
            self._clear('Select a DLS measurement.')
            self.status.clear()
            return
        self._points[sid] = self.controller.dls_sample_rows(
            sid, 'gamma_q2', self._fraction())
        fresh = {r['item_id'] for r in self._points[sid] if r['ok']}
        prev = self._included.get(ckey)       # per (sample, fraction), like the rows
        self._included[ckey] = (prev & fresh) if prev is not None else set(fresh)
        self._refresh_table(sid)
        self._refresh_facets(sid)
        self._refresh_families(sid)
        self._update_run_enabled(sid)
        has_map = self._map_result(sid) is not None
        self.export_button.setEnabled(has_map)
        self.record_set_button.setEnabled(has_map)
        self.bounds_button.setEnabled(bool(fresh))
        self._last_sid = sid
        if not self._points[sid]:
            self.table_note.setText('This sample has no DLS measurements.')
        elif not fresh:
            self.table_note.setText(
                'No measurements have confirmed parameters yet — set them in '
                'the Data tab and press Update, then return here.')
        else:
            self.table_note.clear()
        if has_map:
            self._fill_results(self._selected_family_fit(sid))
            self._draw()
            if self._cache_sig.get(ckey) != self._run_signature(ckey):
                self.status.setText(_STALE_HINT)
        else:
            self._clear_results()
            self._clear('Tick measurements, then Run to extract this '
                        'sample\'s modes.')

    def _run_signature(self, ckey: tuple) -> tuple:
        """Everything a mode-map extraction for (sample, fraction) reads beyond its raw
        data: the committed params of the ticked measurements, the source, and the
        settings the engine seeds -- skip_initial_channels for every source, plus the
        Rh grid and (CONTIN) the α range/selector for a distribution source. Stored at
        dispatch, compared on redisplay (the Γ-q² tab's shape)."""
        s = self.controller.settings
        source = self.source_combo.currentData()
        seeds: tuple = (s.skip_initial_channels,)
        if source != 'double':
            # The controller's ONE seeding home, asked read-only (an empty kwargs
            # dict) rather than mirrored: a seed added there reaches this key too.
            seeds += tuple(sorted(
                self.controller._seed_distribution_kwargs({}, source).items()))
        return (self.controller.committed_signature(self._included.get(ckey, set())),
                source, seeds)

    # ---- measurements table ----
    def _refresh_table(self, sid: str) -> None:
        pts = self._points.get(sid, [])
        inc = self._included.get(self._ckey(sid), set())
        cu = _disp_unit('concentration')
        dfc = _disp_factor('concentration')
        Flag = QtCore.Qt.ItemFlag
        self._suppress_table = True
        try:
            t = self.table
            t.setColumnCount(3)
            t.setHorizontalHeaderLabels(['', 'Angle (°)', f'c ({cu})'])
            t.setRowCount(len(pts))
            for r, row in enumerate(pts):
                chk = QtWidgets.QTableWidgetItem()
                chk.setData(QtCore.Qt.ItemDataRole.UserRole, row['item_id'])
                chk.setFlags(Flag.ItemIsUserCheckable | Flag.ItemIsEnabled
                             if row['ok'] else Flag.ItemIsUserCheckable)
                chk.setCheckState(QtCore.Qt.CheckState.Checked
                                  if (row['ok'] and row['item_id'] in inc)
                                  else QtCore.Qt.CheckState.Unchecked)
                if row.get('reason') == 'unbuilt':
                    chk.setToolTip('Parameters not confirmed yet — set them '
                                   'in the Data tab and press Update.')
                elif row.get('reason') == 'other_fraction':
                    chk.setToolTip('A different Mw fraction than the focused '
                                   'measurement — one map never mixes '
                                   'fractions.')
                else:
                    chk.setToolTip('Ticked runs are extracted when you Run.')
                t.setItem(r, 0, chk)
                ang = '—' if row['angle_deg'] is None else f"{row['angle_deg']:g}"
                conc = ('—' if row['concentration_g_per_mL'] is None
                        else f"{row['concentration_g_per_mL'] * dfc:g}")
                for c, v in enumerate([ang, conc], start=1):
                    it = QtWidgets.QTableWidgetItem(v)
                    it.setFlags(Flag.ItemIsEnabled)
                    t.setItem(r, c, it)
            t.resizeColumnsToContents()
        finally:
            self._suppress_table = False

    def _on_table_changed(self, item) -> None:
        if self._suppress_table or item.column() != 0:
            return
        sid = self._sample_id()
        if sid is None:
            return
        iid = item.data(QtCore.Qt.ItemDataRole.UserRole)
        inc = self._included.setdefault(self._ckey(sid), set())
        if item.checkState() == QtCore.Qt.CheckState.Checked:
            inc.add(iid)
        else:
            inc.discard(iid)
        self.selectionChanged.emit()
        self._run_epoch += 1
        if self._map_result(sid) is not None:
            self.status.setText('Selection changed — press Run to refresh.')
        self._update_run_enabled(sid)

    @QtCore.Slot()
    def _select_all(self) -> None:
        sid = self._sample_id()
        if sid is None:
            return
        self._included[self._ckey(sid)] = {r['item_id'] for r in self._points.get(sid, [])
                               if r['ok']}
        self._refresh_table(sid)
        self.selectionChanged.emit()
        self._run_epoch += 1
        self._update_run_enabled(sid)

    @QtCore.Slot()
    def _select_none(self) -> None:
        sid = self._sample_id()
        if sid is None:
            return
        self._included[self._ckey(sid)] = set()
        self._refresh_table(sid)
        self.selectionChanged.emit()
        self._run_epoch += 1
        self._update_run_enabled(sid)

    def _update_run_enabled(self, sid: str) -> bool:
        enough = bool(self._included.get(self._ckey(sid)))
        self.run_button.setEnabled(bool(self._runnable) and enough)
        if not enough:
            self.status.setText('Tick at least one measurement to run.')
        return enough

    # ---- polarization facet ----
    def _refresh_facets(self, sid: str) -> None:
        """One state at a time, no 'all' — the facet IS the partition. The
        state set comes from the controller's public query (invariant 5:
        the GUI never re-derives the classification itself)."""
        ordered = (self.controller.mode_map_states(sid, self._fraction())
                   or [None])
        current = self.facet_combo.currentData()
        self.facet_combo.blockSignals(True)
        self.facet_combo.clear()
        for s in ordered:
            self.facet_combo.addItem(self._STATE_LABELS[s], s)
        if current in ordered:
            self.facet_combo.setCurrentIndex(ordered.index(current))
        self.facet_combo.setEnabled(len(ordered) > 1)
        self.facet_combo.blockSignals(False)

    def _facet_state(self):
        return self.facet_combo.currentData()

    def _on_facet_changed(self, _i) -> None:
        self._draw()

    # ---- families panel ----
    def _store(self, sid: str):
        return self.controller.mode_picks_for(sid, self._fraction())

    def _family_colour(self, index: int):
        colours = _plot_cycle()
        return colours[index % len(colours)]

    def _family_marker(self, index: int):
        markers = _marker_cycle()
        return markers[index % len(markers)]

    def _refresh_families(self, sid: str) -> None:
        store = self._store(sid)
        status = self.controller.mode_family_status(sid, self._fraction())
        sel_label = self._active_family_label()
        self._suppress_family = True
        try:
            self.family_list.clear()
            for i, fam in enumerate(store.families):
                n_picks, n_matched = status.get(fam.label, (0, 0))
                role = f'  [{fam.role}]' if fam.role else ''
                # A fully-orphaned family (picks, zero matches, a map present)
                # says so HERE -- with no fit there is no result to note on,
                # so the panel line is the disclosure surface.
                orphan = (' — matches nothing (re-run or re-pick)'
                          if (n_picks and not n_matched and sid
                              and self._map_result(sid) is not None) else '')
                it = QtWidgets.QListWidgetItem(
                    f'{fam.label}{role}  — {n_matched}/{n_picks} pick(s)'
                    f'{orphan}')
                it.setData(QtCore.Qt.ItemDataRole.UserRole, fam.label)
                it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
                it.setForeground(QtGui.QColor(self._family_colour(i)))
                self.family_list.addItem(it)
            if store.families:
                # By label, never by row: a rebuild after a sample / fraction switch
                # would otherwise select whichever family sits at the old ordinal
                # (the `_ModeResolvedTab._refresh` rule: preserve by id, never row).
                rows = [i for i, fam in enumerate(store.families) if fam.label == sel_label]
                self.family_list.setCurrentRow(rows[0] if rows else 0)
        finally:
            self._suppress_family = False
        self._sync_role_combo(sid)

    def _active_family_label(self) -> Optional[str]:
        it = self.family_list.currentItem()
        return it.data(QtCore.Qt.ItemDataRole.UserRole) if it else None

    def _sync_role_combo(self, sid: str) -> None:
        label = self._active_family_label()
        fam = self._store(sid).family(label) if label else None
        self.role_combo.blockSignals(True)
        self.role_combo.setCurrentIndex(
            {'': 0, 'fast': 1, 'slow': 2}.get(fam.role if fam else '', 0))
        self.role_combo.setEnabled(fam is not None)
        self.role_combo.blockSignals(False)

    @QtCore.Slot()
    def _on_add_family(self) -> None:
        sid = self._sample_id()
        if sid is None:
            return
        self.controller.add_mode_family(sid, self._fraction())
        self._refresh_families(sid)
        self.family_list.setCurrentRow(self.family_list.count() - 1)

    @QtCore.Slot()
    def _on_del_family(self) -> None:
        sid = self._sample_id()
        label = self._active_family_label()
        if sid is None or label is None:
            return
        self.controller.remove_mode_family(sid, self._fraction(), label)
        self._refresh_families(sid)
        self._fill_results(self._selected_family_fit(sid))
        self._draw()

    def _on_family_selected(self, _row: int) -> None:
        if self._suppress_family:
            return
        sid = self._sample_id()
        if sid is None:
            return
        self._sync_role_combo(sid)
        self._fill_results(self._selected_family_fit(sid))
        self._draw()

    def _on_family_renamed(self, item) -> None:
        if self._suppress_family:
            return
        sid = self._sample_id()
        old = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if sid is None or old is None:
            return
        # The edited text is the whole display line; strip the FIXED suffix
        # (role chip + pick counts) with a regex anchored at the END, so a
        # label that itself contains an em dash or brackets survives the
        # round-trip intact (never split on a delimiter the user's own text can contain).
        new = re.sub(r'(\s*\[(?:fast|slow)\])?\s+—\s+\d+/\d+ pick\(s\)'
                     r'( — matches nothing \(re-run or re-pick\))?$', '',
                     item.text()).strip()
        if not new or new == old:
            self._refresh_families(sid)
            return
        try:
            self.controller.rename_mode_family(sid, self._fraction(), old, new)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, 'Rename', str(exc))
        else:
            # The item still keys the OLD label; re-key it before `_refresh_families`
            # reads the selection back by label, or the rebuild lands on row 0 (round 1
            # f2). Suppressed: setData fires itemChanged into this very handler.
            self._suppress_family = True
            try:
                item.setData(QtCore.Qt.ItemDataRole.UserRole, new)
            finally:
                self._suppress_family = False
        self.controller.run_mode_family_fits(sid, self._fraction())
        self._refresh_families(sid)
        self._draw()

    def _on_role_changed(self, _i) -> None:
        sid = self._sample_id()
        label = self._active_family_label()
        if sid is None or label is None:
            return
        self.controller.set_mode_family_role(
            sid, self._fraction(), label, self.role_combo.currentData())
        self._refresh_families(sid)
        self._draw()

    def _selected_family_fit(self, sid: str):
        label = self._active_family_label()
        return self._family_fits(sid).get(label) if label else None

    # ---- run (worker-dispatched; the _recompute shape) ----
    @QtCore.Slot()
    def _on_run(self) -> None:
        if not self._runnable or self.item_id is None:
            return
        sid = self._sample_id()
        if sid is None:
            QtWidgets.QMessageBox.warning(
                self, 'No sample',
                'This measurement is not assigned to a sample.')
            return
        if self._update_run_enabled(sid):
            self._recompute(sid)

    def _recompute(self, sid: str, fraction=_FOCUSED) -> None:
        frac = self._fraction() if fraction is _FOCUSED else fraction
        ckey = (sid, frac)
        if runner().is_busy:
            first = not self._recompute_pending
            self._recompute_pending = ckey   # the LATEST press wins the slot
            if first:
                run_when_idle(self._flush_recompute)
            self.status.setText(BUSY_NOTICE)
            return
        include_ids = set(self._included.get(ckey, set()))
        source = self.source_combo.currentData()
        sig = self._run_signature(ckey)
        if ckey == self._last_ckey:
            self._run_epoch += 1
            epoch = self._run_epoch
            self.status.setText('Extracting modes in the background…')
        else:
            epoch = None      # deferred Run of a target no longer shown: never paint it here

        def done(_res) -> None:
            self._cache_sig[ckey] = sig       # the map itself lives in the controller
            if epoch != self._run_epoch:
                return
            self.export_button.setEnabled(True)
            self.record_set_button.setEnabled(True)
            self._refresh_facets(sid)
            self._refresh_families(sid)
            self._fill_results(self._selected_family_fit(sid))
            self._draw()
            self.status.clear()

        def fail(exc: BaseException) -> None:
            QtWidgets.QMessageBox.critical(
                self, 'Analysis failed',
                f'Could not build the mode map for sample {sid}.\n\n'
                f'{humanize_field_names(str(exc))}')
            if epoch == self._run_epoch:       # the `done` guard: never paint a late
                self.status.setText('Analysis failed — see dialog.')   # failure elsewhere

        runner().try_submit(
            lambda: self.controller.run_mode_map(
                sid, fraction=frac, source=source, include_ids=include_ids),
            done, fail, description='Mode map extraction',
            busy_widgets=(self.run_button,))

    def _flush_recompute(self) -> None:
        """Deferred retry against the (sample, fraction) Run was pressed on (see the
        Γ-q² tab's `_flush_recompute`)."""
        target, self._recompute_pending = self._recompute_pending, False
        if not target or not self._runnable:
            return
        sid, frac = target
        if sid in self.controller.workspace.samples and self._included.get(target):
            self._recompute(sid, frac)

    # ---- click-to-pick (the SLS mask-editor pattern) ----
    def _point_coords(self, sid: str):
        """(angle, gamma, polarization_state, x_disp, y_disp) for every point
        of the CURRENT facet — the single source of both the click hit-test
        targets and the overlay positions (the D5 discipline)."""
        res = self._map_result(sid)
        if res is None:
            return []
        state = self._facet_state()
        xf = _disp_factor('scattering_q2')
        yf = _disp_factor('decay_rate')
        return [(p.angle_deg, p.gamma_s_inv, p.polarization_state,
                 p.q2_m2 * xf, p.gamma_s_inv * yf)
                for p in res.points if p.polarization_state == state]

    def _on_canvas_click(self, event) -> None:
        sid = self._sample_id()
        if (sid is None or self._map_result(sid) is None
                or self.view_combo.currentData() != 'map'):
            return
        if event.inaxes is not self.ax or event.button != 1:
            return
        if getattr(self._nav, 'mode', ''):
            return                                    # pan/zoom active
        if runner().is_busy:
            self.status.setText(BUSY_NOTICE)
            return
        label = self._active_family_label()
        if label is None:
            self.status.setText('Add (or select) a family first — clicks '
                                'assign points to the selected family.')
            return
        coords = self._point_coords(sid)
        if not coords:
            return
        pts_disp = self.ax.transData.transform(
            [(x, y) for (_a, _g, _s, x, y) in coords])
        d = np.hypot(pts_disp[:, 0] - event.x, pts_disp[:, 1] - event.y)
        i = int(np.argmin(d))
        if d[i] > self._PICK_RADIUS_PX:
            return
        angle, gamma, state = coords[i][0], coords[i][1], coords[i][2]
        try:
            what = self.controller.toggle_mode_pick(
                sid, self._fraction(), label, angle, gamma, state)
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        try:
            self.controller.run_mode_family_fits(sid, self._fraction())
        except ValueError as exc:
            self.status.setText(str(exc))
        verb = {'added': 'added to', 'removed': 'removed from',
                'moved': 'moved to'}[what]
        self.status.setText(f'Point at {angle:g}° {verb} “{label}”.')
        self._refresh_families(sid)
        self._fill_results(self._selected_family_fit(sid))
        self._draw()

    # ---- results + drawing ----
    def _fill_results(self, fit) -> None:
        for r, (_label, fn) in enumerate(self._result_spec()):
            try:
                txt = fn(fit) if fit is not None else '—'
            except Exception:
                txt = '—'
            it = self.result_table.item(r, 1)
            if it is not None:
                it.setText(txt)
        sid = self._sample_id()
        if fit is None:
            self.flag_label.clear()
            res = self._map_result(sid) if sid else None
            set_flag(self.notes_label,
                     '\n'.join(getattr(res, 'notes', ())), problem=False)
            return
        # The origin verdict is a LABEL on a mode — a slow mode failing it is
        # the expected, informative case — so it renders as a neutral ⓘ
        # qualifier here, unlike the Γ-vs-q² tab's red flag, where the same
        # words indict the whole sample's translational fit (§5 R5.2 keys the
        # tier to the CONDITION; the condition differs: per-mode vs sample).
        bits = []
        if not fit.is_diffusive:
            bits.append('mode map: does not pass through the origin (within '
                        'the stated tolerance) — this mode is not purely '
                        'diffusive; its Rh is apparent.')
        bits.append('Rh is apparent and mode-resolved.')
        # A VH family's caveat prints WHERE the value prints (rule 10's
        # surfacing sweep), in the
        # DDLS tab's own words.
        label = self._active_family_label()
        fam = self._store(sid).family(label) if (sid and label) else None
        if fam is not None and fam.polarization_state == 'VH':
            bits.append('VH (depolarized) family: Γ_VH = Γ_VV + 6·D_r, so '
                        'the fitted slope is not a translational D and this '
                        'Rh is not a hydrodynamic radius — use the DDLS '
                        'sub-tab to separate D_r.')
        if fit.d_se is not None:
            bits.append('(± statistical only)' + _ols_marker(getattr(fit, 'se_estimator', 'hc3')))
        set_flag(self.flag_label, '\n'.join(bits), problem=False)
        set_flag(self.notes_label, '\n'.join(fit.notes), problem=False)

    def _clear_results(self) -> None:
        for r in range(self.result_table.rowCount()):
            it = self.result_table.item(r, 1)
            if it is not None:
                it.setText('—')
        self.flag_label.clear()
        set_flag(self.notes_label, '', problem=False)

    def _draw(self) -> None:
        sid = self._sample_id()
        if sid is None or self._map_result(sid) is None:
            return
        if self._placeholder is not None:
            self._placeholder.remove()
            self._placeholder = None
        self.ax.clear()
        self.ax.set_visible(True)
        set_plot_export_name(self.canvas, sample_suggested_name(
            self.controller, sid, 'mode_map'))
        if self.view_combo.currentData() == 'amplitude':
            self._draw_amplitude(sid)
        else:
            self._draw_map(sid)
        self.canvas.draw_idle()
        self.axis_bar.attach(self.ax)

    def _draw_map(self, sid: str) -> None:
        res = self._map_result(sid)
        state = self._facet_state()
        facet_pts = [p for p in res.points if p.polarization_state == state]
        plot_mode_map(facet_pts, ax=self.ax)
        store = self._store(sid)
        fits = self._family_fits(sid)
        xf = _disp_factor('scattering_q2')
        yf = _disp_factor('decay_rate')
        for i, fam in enumerate(store.families):
            # State EQUALITY on a BOUND family, exactly as the matcher and the
            # export label it (the `is not None` wildcard drew an unpolarized
            # family's fit on the VV/VH facets -- the display sibling of the
            # matcher's own wildcard defect).
            if not fam.bound or fam.polarization_state != state:
                continue
            fit = fits.get(fam.label)
            colour = self._family_colour(i)
            marker = self._family_marker(i)
            if fit is not None:
                plot_mode_family_fit(fit, ax=self.ax, color=colour,
                                     marker=marker, label=fam.label)
            else:
                picked = [(p.q2_m2 * xf, p.gamma_s_inv * yf)
                          for p in facet_pts
                          if fam.has_pick(p.angle_deg, p.gamma_s_inv)]
                if picked:
                    xs, ys = zip(*picked, strict=True)
                    self.ax.scatter(xs, ys, s=44, color=colour, marker=marker,
                                    zorder=3, label=f'{fam.label} (unfitted)')
        reissue_legend_preserving_title(self.ax, fontsize=9)
        self.ax.set_title(f'Mode map — {self._STATE_LABELS[state]}')

    def _draw_amplitude(self, sid: str) -> None:
        try:
            rep = self.controller.mode_amplitude_report(sid, self._fraction())
        except ValueError as exc:
            self._clear(str(exc))
            return
        plot_amplitude_vs_q(rep, ax=self.ax)
        self.ax.set_title('Amplitude report — A_f/A_s vs sin²(θ/2)')
        set_flag(self.notes_label, '\n'.join(rep.notes), problem=False)

    @QtCore.Slot()
    def _on_export(self) -> None:
        sid = self._sample_id()
        if sid is None or self._map_result(sid) is None:
            return
        frac = self._fraction()
        name = sample_suggested_name(self.controller, sid, 'mode_map')
        status = export_to_csv(
            self, self.controller, name,
            lambda p: self.controller.export_mode_map(sid, frac, p))
        if status:
            self.status.setText(status)

    # ---- recorded mode sets + manual integration bounds ----
    def _on_record_set(self) -> None:
        sid = self._sample_id()
        if sid is None:
            return
        frac = self._fraction()
        ready = self.controller.mode_set_readiness(sid, frac)
        if not ready['ok']:
            self.status.setText(_RECORD_NOT_READY.get(
                ready['reason_code'], 'The mode set cannot be recorded yet.'))
            return
        n = len(self.controller.mode_sets_for(sid, frac))
        label, ok = QtWidgets.QInputDialog.getText(
            self, 'Record mode set',
            'Label for the recorded set (a snapshot — later re-fits never '
            'change it):', text=f'Mode set {n + 1}')
        if not ok:
            return
        try:
            rset = self.controller.record_mode_set(
                sid, frac, label=label.strip() or None)
        except ValueError as exc:
            self.status.setText(humanize_field_names(str(exc)))
            return
        n_pts = sum(len(m.points) for m in rset.modes)
        msg = ('Recorded mode set “{0}” ({1} families, {2} points). '
               'Partition it on the SLS tab\'s Mode-Resolved sub-tab.'.format(
                   rset.label, len(rset.modes), n_pts))
        if rset.notes:
            msg += ' ' + ' '.join(rset.notes)
        self.status.setText(msg)

    def _on_bounds(self) -> None:
        sid = self._sample_id()
        if sid is None:
            return
        frac = self._fraction()
        rows = [r for r in self._points.get(sid, []) if r['ok']]
        if not rows:
            return
        stored = self.controller.mode_bounds_for(sid, frac)
        auto = self.controller.mode_set_bounds_defaults(sid, frac)
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle('Manual integration bounds')
        lay = QtWidgets.QVBoxLayout(dlg)
        intro = QtWidgets.QLabel(
            'Rh intervals per run, as lo:hi pairs separated by “;” (e.g. '
            '5:60; 61:1000). Blank = the automatic lobe split. Bounds apply '
            'to the CONTIN/NNLS sources at the next Run.\n'
            'A manual split of overlapping peaks violates the '
            'wide-separation validity condition — the results will say so. '
            'For genuinely overlapping modes the double-exponential\'s model '
            'amplitudes are the better route.')
        intro.setWordWrap(True)
        lay.addWidget(intro)
        table = QtWidgets.QTableWidget(len(rows), 2)
        table.setHorizontalHeaderLabels(['Angle (°)', 'Bounds (nm)'])
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(True)
        for i, row in enumerate(rows):
            ang = ('—' if row['angle_deg'] is None
                   else f"{row['angle_deg']:g}")
            it = QtWidgets.QTableWidgetItem(ang)
            it.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
            table.setItem(i, 0, it)
            cur = stored.get(row['item_id'], [])
            text = '; '.join(f'{lo:g}:{hi:g}' for lo, hi in cur)
            cell = QtWidgets.QTableWidgetItem(text)
            lobes = auto.get(row['item_id'], [])
            if lobes:
                cell.setToolTip('Automatic lobes of the last run: '
                                + '; '.join(f'{lo:g}:{hi:g}'
                                            for lo, hi in lobes))
            table.setItem(i, 1, cell)
        table.resizeColumnsToContents()
        lay.addWidget(table, 1)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        # Phase 1: parse and validate EVERY row before anything is stored —
        # a later row's error must never leave earlier rows silently
        # committed.
        ops: list = []                       # (item_id, pairs-or-None)
        for i, row in enumerate(rows):
            text = (table.item(i, 1).text() or '').strip()
            if not text:
                ops.append((row['item_id'], None))
                continue
            try:
                pairs = [tuple(float(x) for x in seg.split(':'))
                         for seg in text.split(';') if seg.strip()]
                if any(len(p) != 2 for p in pairs):
                    raise ValueError(
                        'each interval must be one lo:hi pair')
                self.controller.validate_mode_bounds(pairs)
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(
                    self, 'Integration bounds',
                    'Row {0}: {1}. Nothing was changed.'.format(i + 1, exc))
                return
            ops.append((row['item_id'], pairs or None))
        # Phase 2: apply the fully validated set.
        n_set = n_cleared = 0
        for item_id, pairs in ops:
            if pairs is None:
                if item_id in stored:
                    self.controller.clear_mode_bounds(sid, frac, item_id)
                    n_cleared += 1
            else:
                self.controller.set_mode_bounds(sid, frac, item_id, pairs)
                n_set += 1
        if n_set or n_cleared:
            self.status.setText(
                'Manual bounds set for {0} run(s), cleared for {1}. '
                'Run the map again (CONTIN/NNLS source) to apply '
                'them.'.format(n_set, n_cleared))

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self.item_id is not None:
            self.set_measurement(self.item_id, self._runnable)

    def _clear(self, message: str) -> None:
        if self._placeholder is not None:
            self._placeholder.remove()
            self._placeholder = None
        self.ax.clear()
        self._placeholder = show_plot_placeholder(
            self.figure, self.canvas, message, axes=(self.ax,))
        self.axis_bar.attach(None)


_ROD_FROM_BOX = object()   # `_DDLSTab._run_signature`: read the rod length from the box


class _DDLSTab(QtWidgets.QWidget):
    """Depolarized DLS (DDLS): rotational diffusion D_r from a sample's VV/VH
    correlogram pairs. Sample-scoped, like the Γ vs q² tab. Tag each correlogram's
    polarization (VV/VH) in the Data tab; this pairs them by angle and extracts
    D_r = (Γ_VH − Γ_VV)/6, with D_t from the VV channel."""

    _RESULT_ROWS = ['D<sub>r</sub> (rad²/s)', 'τ<sub>rot</sub> (µs)', 'D<sub>t</sub> (m²/s)',
                    'R<sub>h,t</sub> (nm)', 'paired angles', 'single-exp (qL)']

    selectionChanged = QtCore.Signal()           # included angle set changed (mirror)

    def __init__(self, controller, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.item_id: Optional[str] = None
        self._runnable = False
        self._cache: Dict[str, tuple] = {}      # sample_id -> (result, info) fitted
        self._cache_sig: Dict[str, tuple] = {}  # sample_id -> committed-param signature of the fit's inputs
        # sample_id -> (ddls_result, shapes dict): memoizes the shape-model inverse
        # (incl. the ellipsoid Monte-Carlo) so re-selecting a sample does not re-run
        # the MC on the GUI thread; keyed by result identity, so a re-run invalidates it.
        self._shape_cache: Dict[str, tuple] = {}
        self._full: Dict[str, tuple] = {}       # sample_id -> (result, info) all angles
        # sample_id -> the INCLUDED paired angles (tick to include, like Γ vs q²);
        # the excluded set passed to the engine is derived as paired − included.
        self._included: Dict[str, set] = {}
        self._suppress_ddls = False             # re-entrancy guard for checkbox edits
        self._last_sid: Optional[str] = None    # last DLS sample shown (keep-last)
        self._run_epoch = 0                     # async staleness token
        self._recompute_pending: object = False  # the sample id of a Run queued for idle
        self._rod_by_sid: Dict[str, Optional[float]] = {}   # rod length each sample's last fit used
        self._placeholder = None                # empty-state Text artist (no fit shown)
        self._build_ui()

    def _build_ui(self) -> None:
        _, left, right = make_split_panels(self, left_min_width=300,
                                           sizes=(360, 740))
        note = ThemedLabel(
            'Pairs the sample\'s VV (polarized) and VH (depolarized) correlograms '
            'by angle and extracts the rotational diffusion coefficient '
            'D_r = (Γ_VH − Γ_VV)/6 (D_t from the VV channel). Tag each correlogram\'s '
            'polarization in the Data tab.', role='hint', size=11)
        note.setWordWrap(True)
        left.addWidget(note)

        # Paired angles with an include checkbox (tick to include in the D_r/D_t fit —
        # the same idiom as Γ vs q²), plus each correlogram's polarization + pairing.
        left.addWidget(section_header(
            'Paired Angles (tick to include in the fit)',
            'Tick which paired angles enter the D_r/D_t fit.',
            bullets=['Only angles with BOTH a VV and a VH correlogram can be paired.',
                     'Untick an angle to drop it (e.g. an outlier); the fit re-runs on '
                     'the ticked angles.',
                     'Dropped angles still plot as a gray × (labeled "excluded" in the '
                     'legend) so you can see what was left out.',
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
            'guard). Leave blank if unknown. Kept per sample: switching samples shows '
            'the rod length that sample\'s last fit used.')
        rod_row.addWidget(self.rod_edit)
        rod_w = QtWidgets.QWidget(); rod_w.setLayout(rod_row)
        left.addWidget(rod_w)

        self.run_button = QtWidgets.QPushButton('Run')
        self.run_button.setToolTip(BACKGROUND_RUN_TOOLTIP)
        self.run_button.clicked.connect(self._on_run)
        left.addWidget(self.run_button)
        self.export_button = QtWidgets.QPushButton('Export CSV…')
        self.export_button.setEnabled(False)
        self.export_button.setToolTip(
            'Writes the paired angles, D_t / D_r and any shape models to an '
            'Origin-format CSV. A D_r flagged unphysical here is repeated in the '
            'Comments cell of every column that presents it, so the caveat travels '
            'with the file. See the Theory-and-Equations-Guide §3.4.')
        self.export_button.clicked.connect(self._on_export)
        left.addWidget(self.export_button)
        # Outlier removal: untick an angle above to drop it; unticked angles are
        # grayed on the plot. "Include all" re-ticks every paired angle.
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
            'Shape Models (assumed geometry — not measured)')
        shape_layout = QtWidgets.QVBoxLayout(self.shape_box)
        self.shape_table = _readonly_table(
            ['Model', 'Dimensions', 'Aspect / ratio', 'OK?'], [])
        # The Dimensions cell (col 1) names quantities (R(D_r), R(D_t)=R_h); render it
        # rich. Its values never contain a literal '<', so the delegate's HTML path is safe.
        self.shape_table.setItemDelegateForColumn(1, RichTextDelegate(self.shape_table))
        shape_layout.addWidget(self.shape_table)
        self.shape_caveat = ThemedLabel('', role='hint', size=11)
        self.shape_caveat.setWordWrap(True)
        shape_layout.addWidget(self.shape_caveat)
        add_help_to_groupbox(
            self.shape_box,
            'Particle dimensions inverted from the DDLS translational (D_t) and '
            'rotational (D_r) diffusion coefficients, assuming a shape — '
            'model-dependent, not a direct measurement.',
            bullets=[
                'Sphere: one radius from D_r, cross-checked against Rh (from D_t); '
                'the ratio flags whether a sphere is plausible.',
                'Rod: length and diameter of a rigid cylinder (trusted for aspect '
                'ratio 2–30).',
                'Ellipsoid: polar (a) and equatorial (b) semi-axes, prolate or '
                'oblate (trusted for 0.1–10). A given D_t/D_r can be reproduced by '
                'more than one ellipsoid — every one that fits is listed, none is '
                'preferred.',
                'p is the axial ratio; ✓/⚠ marks whether it is inside the model’s '
                'validated range. See the Theory & Equations Guide for the formulae.',
            ])
        left.addWidget(self.shape_box)
        self.shape_box.setVisible(False)
        left.addStretch(1)

        self.figure = Figure(figsize=(5.2, 4.3), constrained_layout=True)
        self.canvas = make_canvas_expanding(FigureCanvas(self.figure))
        self.ax = self.figure.add_subplot(111)
        self._nav = themed_navtoolbar(self.canvas, self, self.controller)
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
        self.flag_label.clear()
        sid = self._sample_id()
        # Run needs a runnable (DLS) measurement AND an assigned sample: _on_run only
        # warns when sid is None, so an enabled button over an ungrouped measurement is
        # a false affordance. Mirror the Γ-q²/D-c tabs this one's docstring follows.
        self.run_button.setEnabled(runnable and sid is not None)
        if sid != self._last_sid:
            self._run_epoch += 1     # re-pointed: drop any in-flight run's result
            # A rod length is a per-particle input: typed for one sample it must
            # not become the next sample's fit input (the manual-box family). The box
            # shows the rod THIS sample's last fit used, so an unchanged cached fit
            # does not read stale and a re-Run keeps its qL verdict.
            stored = self._rod_by_sid.get(sid) if sid is not None else None
            self.rod_edit.setText('' if stored is None else f'{stored:g}')
        self.export_button.setEnabled(runnable and sid is not None and sid in self._cache)
        if runnable and sid is not None:
            self._refresh_table(sid)
            self._last_sid = sid                  # remember for keep-last
            if sid in self._cache:
                self._draw(*self._cache[sid])
                if self._cache_sig.get(sid) != self._run_signature(sid):
                    self.status.setText(_STALE_HINT)
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
            return set()                          # not initialized yet → nothing excluded
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
        try:
            rod = self._rod_input()
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self, 'Invalid rod length',
                'Rod length must be a number in nm, or blank.')
            return
        self._rod_by_sid[sid] = rod
        self._run_epoch += 1     # rod length is a fit input: drop any in-flight fit
        self._recompute(sid)

    def _rod_input(self) -> Optional[float]:
        """The rod length typed in the box (nm), None when blank; ValueError on text
        that is not a number."""
        txt = self.rod_edit.text().strip()
        return float(txt) if txt else None

    def _run_signature(self, sid: str, rod=_ROD_FROM_BOX) -> tuple:
        """Everything a DDLS fit of `sid` reads beyond its raw correlograms: the
        committed params of the VV/VH pairs feeding it, the rod length, and the
        settings the engine seeds (skip_initial_channels, the cumulant method, the
        SE estimator behind the D_r/D_t ±). Stored at dispatch with the rod the fit
        reads (`rod=`), compared on redisplay against the box (the Γ-q² tab's shape)."""
        s = self.controller.settings
        if rod is _ROD_FROM_BOX:
            try:
                rod = self._rod_input()
            except ValueError:
                rod = self.rod_edit.text().strip()
        return (self.controller.committed_signature(self._input_ids(sid)), rod,
                s.skip_initial_channels, s.cumulant_method, s.se_estimator)

    def _recompute(self, sid: str) -> None:
        if runner().is_busy:
            # Can't dispatch now; retry when the worker frees (do NOT bump the
            # epoch here — that would drop the in-flight fit with no replacement)
            # -- against the sample Run was pressed on, whatever is focused by then.
            first = not self._recompute_pending
            self._recompute_pending = sid   # the LATEST press wins the slot
            if first:
                run_when_idle(self._flush_recompute)
            self.status.setText(BUSY_NOTICE)
            return
        rod = self._rod_by_sid.get(sid)
        excl = self._excluded_angles(sid)         # paired − included
        controller = self.controller
        sig = self._run_signature(sid, rod=rod)   # names the VV/VH ids feeding this fit
        if sid == self._last_sid:
            self._run_epoch += 1            # this run supersedes any in flight
            epoch = self._run_epoch
            self.status.setText('Fitting in the background…')
        else:
            epoch = None      # deferred Run of a sample no longer shown: never paint it here

        def work():
            # ONE thunk for both calls. Run order is load-bearing: run_ddls writes
            # results[('ddls', sid)] each call (last-write-wins), and ddls_shape reads
            # THAT cache lazily on the main thread when drawing/exporting the shape
            # models. Run the EXCLUDED fit LAST so the cache — hence the shape models —
            # matches the D_t/D_r shown in the table, not the un-excluded angles.
            # The un-excluded `full` (run first, only when excluding) is kept solely to
            # draw the excluded-point markers; a cached result also keeps ddls_shape from
            # re-running the Monte-Carlo fit on the GUI thread.
            full = (controller.run_ddls(sid, rod_length_nm=rod) if excl else None)
            res, info = controller.run_ddls(
                sid, rod_length_nm=rod, exclude_angles=excl)
            if full is None:
                full = (res, info)
            return (res, info), full

        def done(payload) -> None:
            # The fit lands in its own sample's slot whatever is focused now; only
            # the display is gated on the view being the one this run started from.
            (res, info), full = payload
            self._cache[sid] = (res, info)
            self._cache_sig[sid] = sig
            self._full[sid] = full
            if epoch != self._run_epoch:
                return                      # sample/exclusions changed — stale VIEW
            self.export_button.setEnabled(True)
            self._refresh_table(sid)
            self._draw(res, info)

        def fail(exc: BaseException) -> None:
            QtWidgets.QMessageBox.critical(
                self, 'DDLS failed',
                f'Could not run DDLS.\n\n{humanize_field_names(str(exc))}\n\nTag at least '
                'one angle with BOTH a VV and a VH correlogram (Data tab → Polarization), '
                'then confirm parameters.')
            if epoch == self._run_epoch:       # the `done` guard: never paint a late
                self.status.setText('DDLS failed — see dialog.')       # failure elsewhere

        runner().try_submit(work, done, fail,
                            description='DDLS rotational-diffusion fit',
                            busy_widgets=(self.run_button,))

    def _flush_recompute(self) -> None:
        """Deferred retry of a recompute refused while the worker was busy — against
        the sample Run was pressed on (see the Γ-q² tab's `_flush_recompute`)."""
        target, self._recompute_pending = self._recompute_pending, False
        if target and self._runnable and target in self.controller.workspace.samples:
            self._recompute(target)

    def selected_item_ids(self) -> list:
        """The DLS measurements at the currently-included angles (sidebar-mirror
        contract): the mirror lights the VV/VH correlograms feeding the fit."""
        sid = self._sample_id()
        return [] if sid is None else self._input_ids(sid)

    def _input_ids(self, sid: str) -> list:
        """The VV/VH ids at `sid`'s included angles — the fit's inputs, for the
        sample named rather than the focused one (a deferred Run needs its own)."""
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
            self, self.controller,
            sample_suggested_name(self.controller, sid, 'ddls'),
            lambda p: self.controller.export_ddls(res, p, shapes=shapes))
        if status:
            self.status.setText(status)

    def _draw(self, res, info) -> None:
        if self._placeholder is not None:       # leaving the empty-state: drop the message
            self._placeholder.remove()
            self._placeholder = None
        self.ax.clear()
        self.ax.set_visible(True)               # restore after an empty _clear() hid it
        self.axis_bar.attach(self.ax)           # re-bind the axis controls (detached while empty)
        plot_ddls(res, ax=self.ax)
        sid = self._sample_id()
        # Plot saves suggest the same stem as the CSV export for this analysis.
        set_plot_export_name(self.canvas,
                             sample_suggested_name(self.controller, sid, 'ddls'))
        excl = self._excluded_angles(sid) if sid else None
        drew_excluded = False
        if excl and sid in self._full:
            full = self._full[sid][0]
            qf, gf = _disp_factor('scattering_q2'), _disp_factor('decay_rate')
            q2 = np.asarray(full.q2_m2, float) * qf
            ang = np.asarray(full.angles_deg, float)
            xy = [(x, y)
                  for arr in (full.gamma_vv_s_inv, full.gamma_vh_s_inv)
                  for x, y, a in zip(q2, np.asarray(arr, float) * gf, ang, strict=True)
                  if any(np.isclose(a, e) for e in excl)]
            drew_excluded = draw_excluded_markers(self.ax, xy, zorder=5)
        if drew_excluded:
            # Re-issue the legend so the gray ×'s (added after plot_ddls built it) show.
            # plot_ddls sets a titleless, framed legend at fontsize 8; match that.
            reissue_legend_preserving_title(self.ax, fontsize=8, frameon=True)
        self.canvas.draw_idle()
        self.axis_bar.attach(self.ax)

        sig = self.controller.settings.no_uncertainty_sig_figs
        rh = (format_fixed_sig(res.rh_t_nm, sig) if math.isfinite(res.rh_t_nm)
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
            format_fixed_sig(res.rotational_time_s * 1e6, sig),
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
                        'D_r from the per-angle spread; excludes calibration.)'
                        + _ols_marker(getattr(res, 'se_estimator', 'hc3')))
        # Tier the flag structurally (style guide §5) off `d_r_reliable`, never by
        # re-spelling the condition: BOTH its causes ship red ⚠ — an unphysical D_r ≤ 0
        # (Γ_VH ≤ Γ_VV), and a failed VH fit, which leaves D_r resting on fewer angles
        # than `n_angles` states. It is the PER-ANGLE verdict (a noisy set can have one
        # negative angle yet a positive mean, which an aggregate test demotes to calm) and
        # carries the `isinf` term. The qL caveat, the single-angle "no ±" note, the ±
        # caveat and the "Skipped: VH-only at N°" inventory are neutral → calm ⓘ (R5.5:
        # an angle with no VH partner never entered the fit, so nothing overstates it).
        problem = not res.d_r_reliable
        set_flag(self.flag_label, '  '.join(bits), problem=problem)

        self._draw_shape()

    def _draw_shape(self) -> None:
        """Compute and show both shape models for the current sample's DDLS result."""
        sid = self._sample_id()
        try:
            # Memoize by DDLS-result identity: the ellipsoid inverse runs a Monte-Carlo
            # (heavy), and _draw_shape fires on every sample selection, not just Run.
            res = self.controller.results.get(('ddls', sid))
            cached = self._shape_cache.get(sid)
            if res is not None and cached is not None and cached[0] is res:
                shapes = cached[1]
            else:
                shapes = self.controller.ddls_shape(sid, model='both')
                self._shape_cache[sid] = (
                    self.controller.results.get(('ddls', sid)), shapes)
        except Exception as exc:
            self.shape_table.setRowCount(0)
            self.shape_table.setToolTip('')      # drop any stale caveat hover
            self.shape_caveat.setText(f'Shape models unavailable: {exc}')
            self.shape_box.setVisible(True)
            return
        rod, sph = shapes['rod'], shapes['sphere']
        ellipsoids = shapes.get('ellipsoid') or []   # [] if the ellipsoid solve failed

        sig = self.controller.settings.no_uncertainty_sig_figs

        def pm_or_knob(v, se):
            # L/d/p/a/b/R(D_r) carry a Monte-Carlo/delta-method SE — WHEN it exists (both
            # D_t and D_r have one), show σ-driven so the ± sets the precision. When it
            # is None (a single-correlogram D_t, or <2 replicate angles), fall back to
            # the one no-uncertainty knob, uniformly with the ±-less cells below — not
            # format_pm's hard-coded no-SE default.

            # A sigma decades larger rounds the value itself to zero, which is how the
            # de-censored aspect-ratio MC erased what it measured. Both facts are real:
            # show them side by side (owner-ruled 2026-08-28). Display only -- the SE on
            # the result is untouched and the export writes both in full.
            if pdg_erases_value(v, se):
                # TWO sig figs on the spread, not one: `_pdg_decimals` keeps two whenever
                # sigma's leading digit is 1, so that a leading-1 sigma is not coarsened
                # (1.4e6 -> 1e6 understates it by 29 %). The displayed spread is subject
                # to the same must-not-under-report rule as the estimator (invariant 8).
                return f'{format_fixed_sig(v, sig)} ± {format_fixed_sig(se, 2)}'
            return format_pm(v, se) if se is not None else format_fixed_sig(v, sig)

        if not rod.converged:
            rod_cells = ['Rod', 'no rigid rod reproduces both D_t and D_r', '', '✗']
        else:
            ok = '✓ in range' if rod.in_valid_range else '⚠ p outside 2–30'
            rod_cells = [
                'Rod',
                f'L = {pm_or_knob(rod.length_nm, rod.length_se)} nm,  '
                f'd = {pm_or_knob(rod.diameter_nm, rod.diameter_se)} nm',
                f'p = {pm_or_knob(rod.aspect_ratio_p, rod.aspect_ratio_se)}', ok]
        # R(D_t) = Rh DOES carry an SE — propagated from d_t_se, the same ± the export
        # labels with its estimator — so it shows σ-driven like its neighbour; only the
        # sphericity ratio, which has none, falls back to the knob.
        sph_cells = [
            'Sphere',
            f'R(D<sub>r</sub>) = {pm_or_knob(sph.radius_rot_nm, sph.radius_rot_se)} nm  '
            f'vs  R(D<sub>t</sub>)=R<sub>h</sub> = '
            f'{pm_or_knob(sph.radius_trans_nm, sph.radius_trans_se)} nm',
            f'ratio {format_fixed_sig(sph.sphericity_ratio, sig)}',
            '✓' if sph.is_consistent else '✗']
        # Ellipsoid: one row per fitting solution (0-3; a given D_t/D_r admits up to
        # three -- 1 prolate + 2 oblate). The no-fit sentinel renders its own row.
        ell_rows = []
        for e in ellipsoids:
            if not e.converged:
                ell_rows.append(
                    ['Ellipsoid', 'no ellipsoid reproduces both D_t and D_r', '', '✗'])
                continue
            kind = 'prolate' if e.is_prolate else 'oblate'
            ok = '✓ in range' if e.in_valid_range else '⚠ p outside 0.1–10'
            ell_rows.append([
                f'Ellipsoid ({kind})',
                f'a = {pm_or_knob(e.semi_axis_polar_nm, e.semi_axis_polar_se)} nm,  '
                f'b = {pm_or_knob(e.equatorial_radius_nm, e.equatorial_radius_se)} nm',
                f'p = {pm_or_knob(e.axial_ratio_p, e.axial_ratio_se)}', ok])

        t = self.shape_table
        rows = [rod_cells, sph_cells, *ell_rows]
        t.setRowCount(len(rows))
        for r, cells in enumerate(rows):
            for c, txt in enumerate(cells):
                t.setItem(r, c, QtWidgets.QTableWidgetItem(txt))
        t.resizeColumnsToContents()

        n_ell = sum(1 for e in ellipsoids if e.converged)
        if sph.is_consistent:
            verdict = 'Sphere consistent — a near-spherical particle.'
        else:
            verdict = ('Sphere inconsistent (ratio ≠ 1) → an anisotropic model '
                       '(rod or ellipsoid) is the relevant one.')
        if n_ell == 1:
            verdict += ' One ellipsoid of revolution reproduces these D_t, D_r.'
        elif n_ell >= 2:
            verdict += (f' The ellipsoid fit is degenerate — {n_ell} shapes '
                        f'(prolate/oblate) reproduce these D_t, D_r equally.')
        # This verdict + the "assumed shape" caveat fire on every DDLS run, so they live
        # in a passive hover tooltip on the table (auto-gated by Settings → Show tooltips),
        # not an always-on line — the ✓/✗ and ratio cells already carry the visible verdict.
        # The Monte-Carlo census behind the rod/ellipsoid ± rides the same hover: the ± is
        # the spread over the draws a shape reproduced, so the count is part of its meaning.
        census = []
        for label, s in (('rod', rod), *[(f'ellipsoid {i + 1}', e)
                                          for i, e in enumerate(ellipsoids)]):
            if getattr(s, 'mc_n_draws', 0):
                census.append(f'{label} ± from {s.mc_n_used} of {s.mc_n_draws} '
                              f'Monte-Carlo draws that a {label.split()[0]} reproduced '
                              f'(seed {s.mc_seed})')
        census_text = ('\n' + '; '.join(census) + '.') if census else ''
        est_text = ('\nThe ± derive from the D_t SE, which was computed with the '
                    'classical-OLS estimator [SE: classical OLS].'
                    if getattr(rod, 'se_estimator', 'hc3') == 'ols' else '')
        self.shape_table.setToolTip(
            verdict + ' Dimensions assume the stated shape — not a direct measurement.'
            + census_text + est_text)
        self.shape_caveat.setText('')     # standalone caveat demoted to the tooltip above
        self.shape_box.setVisible(True)

    def _clear(self, message: str) -> None:
        if self._placeholder is not None:       # ax.clear() doesn't drop a figure.text
            self._placeholder.remove()
            self._placeholder = None
        self.ax.clear()
        # Hide the axis and show one centred grey message via the shared helper — the
        # clean empty-state look (a blank plot box reads as broken). _draw restores it.
        self._placeholder = show_plot_placeholder(
            self.figure, self.canvas, message, axes=(self.ax,))
        self.axis_bar.attach(None)              # nothing to drive while empty
        self.shape_box.setVisible(False)


# ===========================================================================
# Container: the DLS tab = header + inner sub-tabs
# ===========================================================================

# ===========================================================================
# Summary sub-tab (durable results table over the snapshot store)
# ===========================================================================

# How each sample-level source_kind is labeled in the table's "Source" column.
_SOURCE_LABELS = {
    'replicate_avg': 'Replicate avg',
    'gamma_q2': 'Γ vs q²',
    'conc_extrap': 'D vs c',
}
# The ± column says what kind of ± it holds where it is read (ui_style_guide R5.3: the
# "statistical only" qualifier is never hidden), and a cell adds the estimator marker
# when the classical-OLS opt-in produced it -- the one wording every surface shares.
_SE_HEADER = '±SE (statistical)'
_OLS_MARKER = ' [SE: classical OLS]'


def _ols_marker(se_estimator) -> str:
    """The estimator marker a ± cell appends: the OLS opt-in names itself, HC3 (the
    default) and a non-regression SE (`None`) stay silent -- the export's convention."""
    return _OLS_MARKER if se_estimator == 'ols' else ''


def _peak_cell(peaks: List[Tuple[float, float]], sig: int = 3) -> str:
    """Render a distribution method's peaks as 'Rh (Int %); Rh (Int %)'.

    NNLS/CONTIN/lognormal peaks are from an ill-posed inversion — no honest ±
    — so BOTH the Rh and its intensity weight use the one
    no-uncertainty knob (`sig`), never a separate fixed convention."""
    if not peaks:
        return '—'
    return '; '.join(
        f'{format_fixed_sig(rh, sig)} ({format_fixed_sig(frac * 100, sig)}%)'
        if frac is not None else format_fixed_sig(rh, sig)
        for rh, frac in peaks)


def _readonly_table(headers: List[str], rows: List[List[str]]) -> QtWidgets.QTableWidget:
    """A read-only QTableWidget filled from a header list + a list of string rows."""
    t = QtWidgets.QTableWidget(len(rows), len(headers))
    # Headers may name quantities (R_h …); render them as rich text (a plain header can't).
    t.setHorizontalHeader(RichTextHeaderView(t))
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
        _, left, right = make_split_panels(self, left_min_width=300,
                                           sizes=(360, 740))

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
        right.addWidget(QtWidgets.QLabel('<b>Sample-level R<sub>h</sub></b> '
                                         '(averages, Γ–q², D–c)'))
        self.sample_table = _readonly_table(
            ['Sample', 'Source', 'R<sub>h</sub> (nm)', _SE_HEADER, 'R<sub>h</sub> Type', 'From'], [])
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
        _rh = 'R<sub>h</sub>'
        headers = ['Sample', 'Measurement', f'Cumulant {_rh}', 'PDI', f'Single {_rh}',
                   f'KWW {_rh}', f'NNLS ({_rh} / Int %)', f'CONTIN ({_rh} / Int %)']
        if show_logn:
            headers.append(f'Lognormal ({_rh} / Int %)')
        if show_dbl:
            headers.append(f'Double-exp ({_rh} fast / slow)')
        # Every number here is from a single correlogram or an ill-posed inversion — no
        # honest ± — so all use the one no-uncertainty knob, never a
        # fabricated uncertainty.
        sig = self.controller.settings.no_uncertainty_sig_figs
        rows = []
        for r in recs:
            # An unreliable Rh is marked here and explains itself on hover. ⚠ rather
            # than ⓘ: a data-quality problem, and R5.2 requires one condition to take
            # one tier across surfaces. The `is not None` guard mirrors the house rule
            # that a missing value takes no mark.
            #
            # EVERY fit column, not just the cumulant: the other three were exempt only
            # because an exponential's unreliable Rh used to be NaN, with no shown number
            # to qualify. The undecayed-mode criterion leaves it finite.
            def flagged(text, value, note):
                return (f'⚠ {text}', note) if note and value is not None else text

            cells = [
                r['sample_label'], r['measurement_label'],
                flagged(format_fixed_sig(r['cumulant_rh'], sig),
                        r['cumulant_rh'], r.get('cumulant_rh_note', '')),
                format_fixed_sig(r['pdi'], sig),
                flagged(format_fixed_sig(r['single_rh'], sig),
                        r['single_rh'], r.get('single_rh_note', '')),
                flagged(format_fixed_sig(r['kww_rh'], sig),
                        r['kww_rh'], r.get('kww_rh_note', '')),
                _peak_cell(r['peaks'].get('nnls', []), sig),
                flagged(_peak_cell(r['peaks'].get('contin', []), sig),
                        r['peaks'].get('contin') or None,
                        r.get('peak_notes', {}).get('contin', '')),
            ]
            if show_logn:
                cells.append(_peak_cell(r['peaks'].get('lognormal', []), sig))
            if show_dbl:
                cells.append(flagged(
                    f"{format_fixed_sig(r['double_fast'], sig)} / "
                    f"{format_fixed_sig(r['double_slow'], sig)}"
                    if r['double_fast'] is not None else '—',
                    r['double_fast'], r.get('double_note', '')))
            rows.append(cells)
        self._replace_table(self.per_meas_table, headers, rows)

    def _fill_sample(self, recs: List[Dict]) -> None:
        headers = ['Sample', 'Source', 'Rh (nm)', _SE_HEADER, 'Rh Type', 'From']
        sig = self.controller.settings.no_uncertainty_sig_figs
        rows = []
        for r in recs:
            se = r['rh_se']
            # With an honest SE, round Rh to the place its σ supports and
            # show the SE at that same place, so value and ± agree (45 ± 1, not 45.2 ± 1.03).
            # Without one, fall back to the documented fixed-sig precision — never a ±.
            # A classical-OLS ± (the Settings opt-in) says so IN the cell: the estimator
            # travels with the number it qualifies (invariant 8 clause A), in the same
            # words the Cross-Sample ρ cell and the export use.
            if se is not None and math.isfinite(se) and se > 0:
                rh_cell = format_value_at_uncertainty(r['rh_nm'], se)
                se_cell = format_value_at_uncertainty(se, se) + _ols_marker(r.get('se_estimator'))
            else:
                rh_cell = format_fixed_sig(r['rh_nm'], sig)
                se_cell = '—'
            rows.append([
                r['sample_label'],
                _SOURCE_LABELS.get(r['source_kind'], r['source_kind']),
                rh_cell, se_cell,
                r['rh_type_label'], r['from_label'],
            ])
        self._replace_table(self.sample_table, headers, rows)

    @staticmethod
    def _replace_table(table: QtWidgets.QTableWidget, headers: List[str],
                       rows: List[List[str]]) -> None:
        """Rebuild `table`. A cell is a plain string, or `(text, tooltip)`.

        The tuple form exists so a marked cell can explain itself where the mark appears
        (ui_style_guide §5: a soft, always-on interpretive caveat belongs in a passive
        tooltip). Plain strings still work unchanged -- `_fill_sample` passes only those --
        and that back-compatibility is asserted, not assumed.
        """
        table.clear()
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(rows))
        for r, cells in enumerate(rows):
            for c, cell in enumerate(cells):
                text, tip = cell if isinstance(cell, tuple) else (cell, '')
                item = QtWidgets.QTableWidgetItem(text)
                if tip:
                    item.setToolTip(tip)
                table.setItem(r, c, item)
        table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents)

    @QtCore.Slot()
    def _on_export(self) -> None:
        ids = self.selection.ids() if self.ticked_only.isChecked() else None

        def do_export(path: str) -> str:
            return self.controller.export_dls_summary(path, ids)

        status = export_to_csv(self, self.controller, 'dls_summary.csv', do_export)
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

        self.tabs = roomy_tabs(QtWidgets.QTabWidget())   # roomier tabs so labels don't clip
        # Shared DLS analysis region (fit window + baseline) AND the shared set of
        # measurements ticked for co-plotting — both used by the Correlogram and
        # Distribution tabs so the two stay consistent.
        self.region = _AnalysisRegion()
        # One shared selection model for the co-plotting tabs (Correlogram / Distribution
        # / Summary). The color cycle keeps each measurement a stable overlay color.
        # Pass the live `cycle` FUNCTION (not a snapshot) so a plot-palette switch
        # reaches the DLS overlay colours; markers/linestyles are palette-independent.
        self.selection = SelectionModel(color_cycle=_plot_cycle,
                                        marker_cycle=_marker_cycle(),
                                        linestyle_cycle=_linestyle_cycle())
        self.correlogram_tab = _CorrelogramTab(controller, self.region, self.selection)
        self.distribution_tab = _DistributionTab(controller, self.region, self.selection)
        self.gamma_q2_tab = _SampleAnalysisTab(controller, 'gamma_q2')
        self.mode_map_tab = _ModeMapTab(controller)
        self.conc_tab = _SampleAnalysisTab(controller, 'conc_extrap')
        self.ddls_tab = _DDLSTab(controller)
        self.summary_tab = _SummaryTab(controller, self.selection)
        self._subtabs = [self.correlogram_tab, self.distribution_tab,
                         self.gamma_q2_tab, self.mode_map_tab, self.conc_tab,
                         self.ddls_tab, self.summary_tab]
        self.tabs.addTab(self.correlogram_tab, 'Correlogram')
        self.tabs.addTab(self.distribution_tab, 'Distribution')
        self.tabs.addTab(self.gamma_q2_tab, 'Γ vs q²')
        self.tabs.addTab(self.mode_map_tab, 'Mode Map')   # generalizes Γ vs q² per mode
        self.tabs.addTab(self.conc_tab, 'D vs c')
        self.tabs.addTab(self.ddls_tab, 'DDLS')
        self.tabs.addTab(self.summary_tab, 'Summary')
        self.tabs.setMovable(True)               # drag to reorder sub-tabs
        layout.addWidget(self.tabs, 1)
        # Re-emit selection changes for the sidebar mirror: the shared overlay model,
        # each sample-scoped sub-tab's own include set, and a sub-tab switch (which
        # changes WHICH selection is active) all bubble up as selectionChanged.
        self.selection.changed.connect(self.selectionChanged)
        self.tabs.currentChanged.connect(lambda _i: self.selectionChanged.emit())
        for tab in (self.gamma_q2_tab, self.mode_map_tab, self.conc_tab,
                    self.ddls_tab):
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
        if w in (self.gamma_q2_tab, self.mode_map_tab, self.conc_tab,
                 self.ddls_tab):
            return w.selected_item_ids()
        return []

    def reseed_from_settings(self) -> None:
        self.correlogram_tab.reseed_from_settings()
        self.distribution_tab.reseed_from_settings()

    def focus_mode_map(self) -> None:
        """Bring the Mode Map sub-tab to the front (cross-tab navigation from
        the SLS tab's Mode-Resolved list — navigation, never duplication)."""
        self.tabs.setCurrentWidget(self.mode_map_tab)

    def set_measurement(self, item_id: Optional[str]) -> None:
        self.item_id = item_id
        if item_id is None:
            self.header.setText('Pick a measurement in the Workspace list; '
                                'tick measurements in the sub-tabs to analyze.')
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
