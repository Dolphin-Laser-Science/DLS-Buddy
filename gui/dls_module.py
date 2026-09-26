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
Exception, owner-ruled: a failed distribution fit's "failed" note IS drawn on the
legend -- it states what the curve is (a seed estimate, not a fit), the same
analysis-determined tier as plotting's dropped-point and ceiling-halo markers, so a
saved image cannot present a seed curve as fitted. It names the METHOD that failed,
since a legend entry stands for a measurement and may cover several of its curves. Each
sub-tab caches its last result per measurement/sample (within-session) and restores
it on reselection (no recompute).
"""

from __future__ import annotations

import html
import math
import itertools
import re
from collections import namedtuple
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from plotting.plots import (
    plot_correlogram_fit_line, plot_correlogram_scaled, plot_distribution, plot_lcurve,
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
from app.analysis_region import RegionState, region_summary
from app.controller import _DLS_DETAIL_SPECS, FIT_ELIGIBLE_QUALITIES
from core.data_models import IDENTITY_FORM_BY_FIELD
from core.naming import (UNCONFIRMED_SAMPLE, measurement_display_name,
                         measurement_hover_card, sample_display_name)
from gui.plot_controls import (
    AxisControlBar, make_split_panels, make_canvas_expanding,
    make_vertical_plot_stack, pin_under_column, scrolling_notes, COLUMN_STACK_STRETCH,
    attach_residual_resizer, set_plot_export_dressing, set_plot_export_name,
    themed_navtoolbar,
)
from gui.export_helper import (
    confirm_overwrite, export_to_csv, measurement_export_names,
    sample_suggested_name)
from exporting.export import alpha_selection_note
from gui.field_labels import humanize_field_names
from gui.help import add_help_to_groupbox, section_header, HelpBadge
from gui.plot_overlays import (draw_excluded_markers, reissue_legend_preserving_title,
                               show_plot_placeholder, PLACEHOLDER_GRAY)
from gui.settings_module import DistributionFitFields
from gui.theme import ThemedLabel, color as theme_color, set_flag
from gui.widgets import (roomy_tabs, SelectionModel, MeasurementPicker, GroupTickBar,
                         marker_swatch, NAME_COL_MAX_W, RichTextDelegate,
                         RichTextHeaderView, sidebar_measurement_order)
from gui.worker import BACKGROUND_RUN_TOOLTIP, BUSY_NOTICE, run_when_idle, runner
from analysis.uncertainty import (format_fixed_sig, format_pm, pdg_erases_value,
                                  format_value_at_uncertainty)
from analysis.dls import CUMULANT_PDI_VALIDITY_LIMIT
from analysis.dls.distributions import (ALPHA_NO_CORNER_NOTE, FTEST_BAND_NOTE_TEMPLATE,
                                        FTEST_REFERENCE_NOTE)


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
    tab (numeric fields + draggable markers): the fit WINDOW [tau_min, tau_max] plus a
    BASELINE span [base_lo, base_hi]. This object holds the handles and whether the
    user has moved each kind; what a fit reads from them is decided in ONE place,
    `app/analysis_region.py::region_fit_kwargs` (through `Controller.
    analysis_region_kwargs`), which the Correlogram and Distribution sub-tabs and every
    checked "use the Correlogram tab's delay window and baseline" path all call. That
    module's docstring states the rule; `state()` is the snapshot handed to it.

    NOTHING IS APPLIED UNTIL THE USER SETS IT (owner-ruled 2026-09-03 for the span,
    2026-09-22 for the window). `window_touched` / `baseline_touched` are False while the
    handles still hold the values `init_from_data` proposed from the ticked data, and an
    untouched part reaches no fit: each run keeps its own full lag range and its fit's
    own baseline estimate. The moment a handle of a kind is dragged or its field edited,
    that part applies, and `Reset window + baseline` clears both flags again. This is NOT
    policing: a region the user set is never refused, second-guessed or overridden
    (invariant 4 clause 2); only a region the user never set stops being applied. The
    measured hazard it closes is a region seeded from one run -- a baseline span, or a
    window on one lag axis -- reaching a run it was never placed on.

    It deliberately does NOT carry the engine inputs Settings seeds (the skipped
    channels, the cumulant method and order). Making them per-run controls here was tried
    and reverted: the Cross-Sample candidate builders re-derive the same fit from
    `controller.settings`, so a panel-local value promoted a DIFFERENT Rh than the
    panel showed (30.0 nm shown, 23.73 nm promoted, measured). The controller's seed
    homes are that one source of truth; the Correlogram tab's order box is the one
    per-run value, and the tier-1 candidate promotes the tab's own fit while it is fresh.
    """

    def __init__(self) -> None:
        self.tau_min_s: Optional[float] = None
        self.tau_max_s: Optional[float] = None
        self.base_lo_s: Optional[float] = None
        self.base_hi_s: Optional[float] = None
        # Has the USER placed each part, as opposed to `init_from_data` proposing it? The
        # Data tab's changed-since-commit highlight is the house pattern: a value the
        # program proposed and a value the user asserted are different states, and only
        # the second may steer a number.
        self.window_touched = False
        self.baseline_touched = False

    def init_from_data(self, tau: np.ndarray) -> None:
        """Propose the defaults from the ticked data: window = full range, baseline =
        last 25 % of the lags. An UNTOUCHED part follows the data on every call, so its
        handles draw the current selection's range; a part the user moved is theirs and
        stays put. Neither proposal reaches a fit (see the class docstring)."""
        if tau.size == 0:
            return
        lo, hi = float(tau.min()), float(tau.max())
        proposal = {'tau_min_s': lo, 'tau_max_s': hi,
                    'base_lo_s': float(np.quantile(tau, 0.75)), 'base_hi_s': hi}
        for attrs, touched in ((self._WINDOW_ATTRS, self.window_touched),
                               (self._BASELINE_ATTRS, self.baseline_touched)):
            for attr in attrs:
                # A part the user placed keeps its values even when typed before any data
                # was ticked (the flag, not the first call, decides); an end still empty
                # takes the proposal.
                if not touched or getattr(self, attr) is None:
                    setattr(self, attr, proposal[attr])

    def force_reseed(self) -> None:
        """Drop the placed values so the next init_from_data re-proposes both parts from
        the data (the Correlogram tab's 'Reset window + baseline'), and forget that the
        user ever moved either -- Reset is the one control that undoes a placement, so it
        is the one that clears the flags."""
        self.window_touched = False
        self.baseline_touched = False

    _WINDOW_ATTRS = ('tau_min_s', 'tau_max_s')
    _BASELINE_ATTRS = ('base_lo_s', 'base_hi_s')

    def set_handle(self, attr: str, value) -> None:
        """Move one handle, marking its kind touched when the handle actually moves.

        The one write path for both editors (drag and numeric field), so the two cannot
        disagree about what counts as a placement. The value comparison matters: the
        numeric-field editor rewrites all four handles on every edit, so a plain
        `setattr` there would mark the baseline touched when the user typed a
        delay-window bound and never went near the purple span -- and the window touched
        when they only moved the span.
        """
        if value != getattr(self, attr):
            if attr in self._WINDOW_ATTRS:
                self.window_touched = True
            elif attr in self._BASELINE_ATTRS:
                self.baseline_touched = True
        setattr(self, attr, value)

    def state(self) -> RegionState:
        """The Qt-free snapshot the one reading helper takes -- read on the main thread
        at dispatch, so a background fit never sees a handle move under it."""
        return RegionState(self.tau_min_s, self.tau_max_s, self.base_lo_s,
                           self.base_hi_s, window_touched=self.window_touched,
                           baseline_touched=self.baseline_touched)


def _region_fit_kwargs(controller, region: _AnalysisRegion, iid: str, method: str) -> Dict:
    """The GUI's thin call into the one reading (`Controller.analysis_region_kwargs`):
    what a `method` fit of measurement `iid` reads from the shared region. Every tab
    that fits under the region builds its Run and its staleness key from this, so a
    region cannot mean one thing on one tab and another elsewhere."""
    return controller.analysis_region_kwargs(iid, method, region.state())


class RegionCheckBox(QtWidgets.QWidget):
    """The opt-in "use the Correlogram tab's delay window and baseline" box -- ONE
    widget, one behaviour, on every surface that re-runs a fit behind the scenes: the
    Γ vs q², D vs c, Mode Map and DDLS sub-tabs and the "Average derived results…"
    prompt.

    - its label line says what the region holds now, or that nothing is set;
    - unticked (the default, held per (sample, fraction) by each tab and never saved)
      the run uses each measurement's own defaults, as the paths did before the box;
    - ticked, the run reads the region through the one reading every tab shares
      (`Controller.analysis_region_kwargs`), and the result's notes say so;
    - its `key()` joins the owning tab's staleness key, so toggling it -- or moving a
      handle while it is ticked -- flags a shown result stale, and leaves an unticked
      one alone.

    Opt-in, never automatic, because the region is ONE per DLS tab, not one per sample:
    a span placed while looking at one sample would otherwise reach another sample's
    runs unasked. A LIVE link to the other tab's region while ticked, unlike the Mode
    Map's one-time copy of the Distribution tab's fields (`ui_style_guide.md` §4 states
    the difference)."""

    toggled = QtCore.Signal(bool)

    LABEL = "Use the Correlogram tab's delay window and baseline"

    def __init__(self, region: _AnalysisRegion, parent=None) -> None:
        super().__init__(parent)
        self._region = region
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self.box = QtWidgets.QCheckBox(self.LABEL)
        self.box.setToolTip(
            'Ticked, each run is fitted as the Correlogram tab fits it: over the delay '
            'window set there, with a baseline measured over the part of the purple span '
            'inside that window. A window or span never moved there is not applied, so '
            'ticking changes nothing until you set one. Unticked, each run keeps its own '
            'full lag range and its fit\'s own baseline estimate. The result\'s notes say '
            'which it used.')
        self.box.toggled.connect(self.toggled)
        row.addWidget(self.box)
        row.addWidget(HelpBadge(
            'Fit these runs over the region set on the Correlogram tab.', bullets=[
                'Set the delay window or the purple baseline span on the '
                '<b>Correlogram</b> sub-tab, then tick this box and press <b>Run</b>.',
                'Only a window or span you have moved is used; the line under the box '
                'says what is set now.',
                'The baseline is measured over the part of the span inside the delay '
                'window; the greyed part outside it is not used.',
                'Held per sample and not saved with the session; each new sample starts '
                'unticked.',
                'Ticking it, or moving the region while it is ticked, marks a shown '
                'result out of date — press <b>Run</b> again.',
                'The result’s notes, its export and any Summary row it writes say it '
                'used the Correlogram tab’s region.',
            ]))
        row.addStretch(1)
        lay.addLayout(row)
        self.summary = ThemedLabel('', role='hint', size=10)
        self.summary.setWordWrap(True)
        lay.addWidget(self.summary)
        self.refresh()

    def refresh(self) -> None:
        """Re-read the region into the label line (the owning module calls this when the
        Correlogram tab reports a move)."""
        self.summary.setText(region_summary(self._region.state()))

    def isChecked(self) -> bool:
        return self.box.isChecked()

    def set_checked_silently(self, on: bool) -> None:
        """Show a held per-(sample, fraction) state without firing `toggled`."""
        was = self.box.blockSignals(True)
        try:
            self.box.setChecked(bool(on))
        finally:
            self.box.blockSignals(was)

    # The tabs hold the choice per (sample, fraction) -- the box shows only the one on
    # screen -- so both readers take the held choice rather than the box's own state; a
    # deferred Run of another (sample, fraction) reads its own.
    def region_for_run(self, checked: bool) -> Optional[RegionState]:
        """The snapshot a Run hands the controller: the region when `checked`, else None.
        Read on the main thread at dispatch."""
        return self._region.state() if checked else None

    def key(self, checked: bool) -> tuple:
        """The choice's part of a staleness key: the choice and, when `checked`, the
        region's APPLIED parts (`RegionState.applied`), so re-seeding an untouched handle
        never flags a result stale and every move of a set one does."""
        return (True, self._region.state().applied()) if checked else (False,)


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

# Every fit that can be drawn over a correlogram, in the order the Correlogram tab's
# "Fits to show" list presents them: the four that tab runs, then the three the
# Distribution tab runs. DERIVED from the two rosters rather than typed again, so a
# method added to either is offered here without a second edit and cannot be half-added
# (rule 11: drive a family from a scan, never a hand list). The gate that keeps the
# maps below total over it is
# `tests/test_gui_smoke.py::test_every_drawable_fit_method_has_a_distinct_line_style_and_a_short_name`.
_FIT_METHODS: List[Tuple[str, str]] = _CORR_METHODS + _DIST_METHODS
_CORR_METHOD_KEYS = frozenset(key for _label, key in _CORR_METHODS)

# Short names for the "Fits to show" list and the plot legend. The roster's own labels
# are form-row labels ('Stretched Exponential (KWW)') and do not fit a two-column
# checkbox grid in a 360 px control column, nor a legend entry beside a full
# `sample . axes . name`.
_FIT_SHORT_LABELS: Dict[str, str] = {
    'cumulant': 'Cumulant', 'single': 'Single exp', 'double': 'Double exp',
    'kww': 'KWW', 'lognormal': 'Lognormal', 'nnls': 'NNLS', 'contin': 'CONTIN',
}

# LINE STYLE IS THE METHOD, on both DLS co-plotting tabs and in both of their residual
# panels, so a dash pattern means the same thing wherever it is drawn. Fixed per method
# -- never an index-keyed cycle, which would re-assign a style as the ticked set moved.
# It retires line style as the Distribution tab's per-MEASUREMENT channel; the marker
# takes that channel there (see `_DistributionTab._draw`). Spacings separate WITHIN each
# family: one parametric against one distribution fit here, NNLS against CONTIN there.
_METHOD_LINESTYLE: Dict[str, object] = {
    'cumulant':  '-',                        # solid
    'single':    (0, (6, 2)),                # long dash
    'double':    (0, (2, 2)),                # short dash
    'kww':       (0, (6, 2, 1, 2)),          # dash-dot
    'lognormal': (0, (1, 2)),                # dotted
    'nnls':      (0, (4, 1, 1, 1, 1, 1)),    # dash-dot-dot
    'contin':    (0, (8, 2, 2, 2)),          # long dash + dash
}


def _fit_label(key: str) -> str:
    """The short name of a fit method, for the layers list, the legend and the residual
    panel. Falls back to the roster's own form-row label so a method added to
    `_CORR_METHODS` / `_DIST_METHODS` and not to `_FIT_SHORT_LABELS` still names
    itself -- the map is a shortening, not the roster."""
    if key in _FIT_SHORT_LABELS:
        return _FIT_SHORT_LABELS[key]
    return next((lbl for lbl, k in _FIT_METHODS if k == key), key)


def _neutral_ink() -> str:
    """The figure's own text colour, for chrome that must not read as a data series.

    Taken from the live rcParams rather than hard-coded, so it follows the on-screen
    plot theme (`plotting.plots.set_onscreen_plot_theme`) into dark mode and back --
    a hard-coded grey is unreadable on one of the two."""
    import matplotlib
    return matplotlib.rcParams.get('text.color', '0.25')


def draw_fit_residuals(ax, entries, *, xscale: str = 'log') -> None:
    """Draw ONE residual panel: the residuals of exactly the fits drawn above it, each
    in its measurement's colour and its method's line style, with the methods on
    display named on the panel.

    `entries` is ``[(colour, method_key, result), ...]`` in draw order; each result
    carries `fit_tau_s` and `residuals`. BOTH DLS residual panels -- the Correlogram
    tab's and the Distribution tab's -- go through this one body, so the two cannot
    disagree about what a residual panel shows or how it is coded, and the drawing
    exists once.

    All seven methods define `residuals` identically -- (g2-1) data minus fitted,
    unweighted, on `fit_tau_s` (`analysis/dls/cumulants.py`, `exponentials.py`,
    `distributions.py`) -- which is what lets them share one axis honestly. A method
    added later whose residual is a different quantity does not join a panel until it
    is.

    The caller owns the axes and passes it CLEARED; no axes is created or recreated here
    (the serial gui tier's heap-fault history -- reuse a persistent cleared axis).
    """
    tfac = _disp_factor('time')
    for color, key, res in entries:
        ax.plot(np.asarray(res.fit_tau_s, dtype=float) * tfac,
                np.asarray(res.residuals, dtype=float),
                color=color, ls=_METHOD_LINESTYLE.get(key, '-'), lw=1.0)
    ax.axhline(0.0, color='#444', lw=0.6, ls=':')
    ax.set_xscale(xscale)
    ax.set_xlabel(rf'Delay time $\tau$ ({_disp_unit("time")})')
    ax.set_ylabel('resid.')
    # The panel names what it is showing, in the GAP above it rather than inside it or
    # in the y-label: the label is rotated into a gutter too narrow for a method list,
    # a title would take height from a panel that is only 22 % of the column, and text
    # placed inside overlapped the residuals as soon as the panel got short (measured
    # 2026-09-18 on the nine-run harness frame, where it sat across the data). The gap
    # is already widened for the resizer grip and carries the main axis's centred
    # x-label, which a LEFT-aligned note sits beside rather than over. No colour is
    # passed, so the text inherits the figure's own (themed) text colour.
    shown = list(dict.fromkeys(key for _c, key, _r in entries))
    if shown:
        ax.text(0.0, 1.06, 'Residuals: ' + ', '.join(_fit_label(k) for k in shown),
                transform=ax.transAxes, ha='left', va='bottom', fontsize=7, zorder=5)


# Shown when a displayed fit was computed from inputs that have since changed: a
# committed parameter, a tab-owned input (the delay window / baseline / cumulant
# order, the ticked set) or a settings-seeded one (skip_initial_channels, the cumulant
# method, the cumulant fit cutoff, the SE estimator). Each tab's `_run_signature` names
# the inputs its fit reads and is stored with the cached fit; a redisplay compares the
# two -- the SLS tab's `sls_run_signature` shape. Complements the pre-commit '(ran on
# last committed values)' note: this covers the post-commit window, where `is_dirty()`
# has already gone false but the plot still reflects the old inputs.
_STALE_HINT = 'Inputs changed since this fit — press Run to refresh.'

#: `_recompute(sid, fraction=_FOCUSED)`: fit the focused fraction, unless a deferred
#: Run re-enters with the (sample, fraction) it was pressed on.
_FOCUSED = object()


# Every numeral in a status message, for grouping several measurements' copies of one
# message into a single sentence (item 7). Matches an optionally signed integer or
# decimal with an optional exponent -- the forms `format_fixed_sig` emits.
_STATUS_NUMERAL = re.compile(r'-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?')

# What a masked numeral becomes while grouping. It must be a string no MESSAGE can
# contain, because the grouping rests on "same mask => same numeral count in the same
# positions": a subject carrying the mask token literally decomposes differently from
# its group-mates, and indexing one member's numerals by another's range raises
# IndexError in a Qt slot. NUL cannot appear in a Qt label, and `chr(0)` is built
# rather than typed -- a literal NUL in a source file makes the file binary to grep
# and to every byte-level tool.
_STATUS_MASK = chr(0)


# How many runs a VALUE-BEARING grouped message names before it says "and N more".
# Names are long and this label sits under the scrolled control column, where an
# uncapped listing of nine wraps to three lines and takes the height out of the
# results tables above it.

# A per-call argument, not a rule, because the cap is only honest where the elided runs
# are recoverable: a value-bearing flag is also on its own glance cell. A passive note
# is rendered nowhere else, and a message with no varying numeral degenerates to a bare
# name list -- both are passed no cap.
_STATUS_LIST_MAX = 4


def _elide_listing(items, cap) -> str:
    """Join a grouped message's run list, capped at `cap` (None = name them all)."""
    if cap is None or len(items) <= cap:
        return '   '.join(items)
    return '   '.join(items[:cap]) + f'   … and {len(items) - cap} more'


def group_status_lines(entries, *, value_cap=None):
    """Render [(measurement_name, message)] so one KIND of message is explained once.

    The commonest DLS warning carries its own value ("PDI = 0.328 is above the 0.3
    validity limit ..."), so no two rendered lines were ever identical and a
    three-replicate run printed the same paragraph three times -- the collaborator's
    "errors and warnings spam the bottom of the column".

    A message's KIND is its text with every numeral masked, which needs no cooperation
    from the engine: the notes keep their wording and nothing about them is
    restructured for this. Two messages of one kind then differ only in their numbers.

    Within a kind, a numeral position that DIFFERS across the measurements is a value
    and a position that agrees is part of the sentence -- a printed threshold, a limit,
    an order. So the shared sentence keeps its constants and blanks only what varies,
    and each measurement is named beside its own value beneath it. A kind with one
    member renders exactly as before, prefixed with its measurement.

    Pure: no Qt, no widgets, so the grouping is unit-testable on strings alone.
    """
    order, groups = [], {}
    for name, line in entries:
        key = _STATUS_NUMERAL.sub(_STATUS_MASK, line)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append((name, line))

    out = []
    for key in order:
        members = groups[key]
        if len(members) == 1:
            name, line = members[0]
            out.append(f'{name}: {line}')
            continue
        # Same mask => same numeral count in the same positions, so these line up.
        numerals = [_STATUS_NUMERAL.findall(line) for _n, line in members]
        varying = [i for i in range(len(numerals[0]))
                   if len({nums[i] for nums in numerals}) > 1]
        counter = itertools.count()
        blank = set(varying)

        def _mask(m, _c=counter, _b=blank):
            # Bound as defaults: the lambda is consumed by this one `sub` call, but a
            # closure over the loop variables is the shape that bites the NEXT edit.
            return '…' if next(_c) in _b else m.group(0)

        out.append(_STATUS_NUMERAL.sub(_mask, members[0][1]))
        if varying:
            listed = [f'{name}: {" / ".join(nums[i] for i in varying)}'
                      for (name, _l), nums in zip(members, numerals, strict=True)]
            cap = value_cap
        else:
            # Identical messages with nothing to distinguish: name the measurements,
            # all of them -- there is no per-run value and so no cell carrying it.
            listed = [name for name, _l in members]
            cap = None
        out.append('    ' + _elide_listing(listed, cap))
    return out


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
    except Exception:  # noqa: BLE001 — layout-engine API differences
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


def _repoint_vtable(table, rows) -> None:
    """Re-point an existing `_vtable` at a different row set, in place.

    `_vtable` BUILDS a table; this reuses one when the method box changes the rows it
    should show. In place rather than swapping a fresh widget into the layout, so the
    pane's splitter share, its scroll position and every connection survive a method
    change. Re-applies the row-label tooltips and the content height cap exactly as
    `_vtable` does, because both are keyed to the row set that just changed.
    """
    table.setRowCount(len(rows))
    for r, name in enumerate(rows):
        label = QtWidgets.QTableWidgetItem(name)
        # The μ₃/μ₄ rows carry the in-program caution against over-reading a bare higher
        # moment; every other row's tooltip is its own label, for narrow-column elision.
        label.setToolTip(_RESULT_ROW_TOOLTIPS.get(name, name))
        table.setItem(r, 0, label)
        table.setItem(r, 1, QtWidgets.QTableWidgetItem('—'))
    row_h = table.verticalHeader().defaultSectionSize()
    table.setMaximumHeight(row_h * max(len(rows), 1) + 2 * table.frameWidth() + 2)
    # It may also SHRINK to three rows and scroll. A `_vtable` otherwise reports its
    # whole row count as its minimum, which in a short section is taken out of whatever
    # shares the space with it.
    table.setMinimumHeight(row_h * min(max(len(rows), 1), 3) + 2 * table.frameWidth() + 2)


def _reset_vtable(table) -> None:
    """Reset a `_vtable`'s value column to '—'."""
    for r in range(table.rowCount()):
        it = table.item(r, 1)
        if it is not None:
            it.setText('—')


# ===========================================================================
# Multi-measurement co-plotting (shared by the Correlogram & Distribution tabs)
# ===========================================================================

def _selection_sample_id(controller, iids):
    """The one sample every ticked id belongs to, or None when the set spans samples.

    ONE definition of "does this selection span samples?", shared by the export-name
    helper below and by the legends' sample prefix (`_sample_prefixes`).
    """
    sids = {controller.sample_id_of(iid) for iid in iids}
    return next(iter(sids)) if len(sids) == 1 else None


def _selection_export_names(controller, items, analysis: str, ext: str = 'csv'):
    """`(dialog name, per-file suffixes)` for a ticked-measurement export.

    `items` is `(item_id, method-or-None)` in fan-out order. The sample token rides
    the name when every ticked measurement belongs to ONE sample and is dropped
    otherwise, where a token would be ill-defined — but the common single-sample
    case must not propose the same filename for every sample (review finding).
    The suffixes are what the fan-out appends to whatever stem the user types; the
    dialog shows the whole name when exactly one measurement is ticked.
    """
    return measurement_export_names(
        controller, _selection_sample_id(controller, [iid for iid, _m in items]),
        items, analysis, ext)


def _sample_label_of(controller, iid) -> str:
    """This measurement's sample name, or `(unconfirmed sample)` when it is in none.

    `Controller.sample_id_of` is declared `Optional[str]` and a measurement in no
    sample is a listable, tickable state -- `MeasurementPicker.refresh` puts one under
    exactly that header. Both DLS legend builders subscripted `workspace.samples[...]`
    with its result, which is `samples[None]` -> KeyError inside a draw, swallowed by
    Qt at the slot boundary and leaving a half-drawn figure. `_sample_prefixes` two
    functions down already guards the same lookup; this is that guard, shared, so the
    two consumers cannot drift from it again.
    """
    sid = controller.sample_id_of(iid)
    sample = controller.workspace.samples.get(sid) if sid else None
    return sample_display_name(sample) if sample is not None else UNCONFIRMED_SAMPLE


def _sample_prefixes(controller, iids) -> dict:
    """Per-id sample prefix for a co-plotted set: the sample's name when the set SPANS
    samples, None when it does not.

    Builds no name of its own -- it decides ONE thing, whether the set spans samples,
    and hands `core.naming` the answer. The predicate is the one
    `_selection_export_names` uses for export names, so a legend and a filename can
    never disagree about it. A single-sample view names its sample once, above.
    """
    iids = list(iids)
    if _selection_sample_id(controller, iids) is not None:
        return {iid: None for iid in iids}
    samples = controller.workspace.samples
    out = {}
    for iid in iids:
        sid = controller.sample_id_of(iid)
        s = samples.get(sid) if sid else None
        out[iid] = sample_display_name(s) if s is not None else None
    return out


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


#: What the averaging prompt can read or copy from the DLS tab, handed in by the module
#: that owns the tabs: the shared region, and zero-argument / per-method accessors for the
#: Correlogram tab's order box and the Distribution tab's fields, read at confirmation.
AverageSources = namedtuple('AverageSources',
                            'region cumulant_order distribution_kwargs')


class AverageRequest(namedtuple('AverageRequest',
                                'method region cumulant_order distribution_kwargs',
                                defaults=(None, None, None))):
    """What the averaging prompt asked for: the method, the region when its box was
    ticked, and the one-time copy of the Correlogram tab's order or the Distribution
    tab's fields when the copy box was ticked (None: the Settings values)."""

    def run_kwargs(self) -> dict:
        """The keyword arguments `Controller.average_dls_results` takes for them."""
        return dict(region=self.region, cumulant_order=self.cumulant_order,
                    distribution_kwargs=self.distribution_kwargs)


class AverageDialog(QtWidgets.QDialog):
    """The "Average derived results…" prompt: which fit each replicate gets, whether it
    reads the Correlogram tab's region (the shared `RegionCheckBox`), and whether it
    copies the matching DLS sub-tab's per-run settings -- the Correlogram tab's cumulant
    order for a cumulant average, the Distribution tab's fields for CONTIN, NNLS or
    lognormal. The Settings values an unticked copy runs on are shown either way, from
    the same controller computation that writes the summary's note, so what the prompt
    showed and what the average recorded cannot disagree. Averaging runs once, so a copy
    taken here is one-time by construction -- the Mode Map's copy button, in the one
    place averaging is started."""

    def __init__(self, parent, controller, sources: Optional[AverageSources] = None) -> None:
        super().__init__(parent)
        self.controller = controller
        self._sources = sources
        self.setWindowTitle('Average derived results')
        lay = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            'Fit each replicate with which method? The mean ± SD/√N of its parameters is '
            'reported. Distribution methods report per-peak averages in Rh order; CONTIN '
            'over many replicates is slow.')
        intro.setWordWrap(True)
        lay.addWidget(intro)
        self.method_combo = QtWidgets.QComboBox()
        for label, key in _AVERAGE_METHODS:
            self.method_combo.addItem(label, key)
        lay.addWidget(self.method_combo)
        self.region_box = RegionCheckBox(
            sources.region if sources is not None else _AnalysisRegion())
        lay.addWidget(self.region_box)
        self.copy_box = QtWidgets.QCheckBox()
        self.copy_box.setToolTip(
            'A one-time copy of that sub-tab\'s current per-run value(s), taken when you '
            'press OK; unticked, the average runs on the Settings values shown below. '
            'The average\'s summary records which, and a written Summary row names the '
            'cumulant order it ran at.')
        lay.addWidget(self.copy_box)
        self.settings_label = ThemedLabel('', role='hint', size=10)
        self.settings_label.setWordWrap(True)
        lay.addWidget(self.settings_label)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self.method_combo.currentIndexChanged.connect(self._sync_method)
        self._sync_method()

    def method(self) -> str:
        return self.method_combo.currentData()

    def _sync_method(self, *_args) -> None:
        """Show the copy choice the method has -- the order for a cumulant average, the
        Distribution tab's fields for an inversion, none for the other fits -- and the
        Settings values it would otherwise run on."""
        method = self.method()
        s = self._sources
        if method == 'cumulant':
            now = f' (now {s.cumulant_order()})' if s is not None else ''
            self.copy_box.setText(f"Use the Correlogram tab's cumulant order{now}")
        else:
            self.copy_box.setText("Use the Distribution tab's settings")
        copyable = method == 'cumulant' or method in ('contin', 'nnls', 'lognormal')
        self.copy_box.setVisible(copyable and s is not None)
        try:
            text = self.controller.average_inputs_text(method)
            text = f'Unticked: {text}' if text else ''
        except ValueError as exc:
            # A hand-edited Settings token the seed home refuses: say so here, where the
            # average would otherwise fail on every replicate with the same refusal.
            text = f'Settings cannot seed this fit: {exc}'
        self.settings_label.setText(text)
        self.settings_label.setVisible(bool(text))

    def request(self) -> AverageRequest:
        """The request for what the dialog shows now. The copy is taken HERE, once."""
        method = self.method()
        s = self._sources
        copy = (not self.copy_box.isHidden()) and self.copy_box.isChecked() and s is not None
        order = s.cumulant_order() if copy and method == 'cumulant' else None
        dist = (s.distribution_kwargs(method)
                if copy and method in ('contin', 'nnls', 'lognormal') else None)
        return AverageRequest(method, self.region_box.region_for_run(
            self.region_box.isChecked()), order, dist)


def ask_average_method(parent, controller,
                       sources: Optional[AverageSources] = None) -> Optional[AverageRequest]:
    """Ask which dynamic fit to run on each replicate, and whether it reads the
    Correlogram tab's region and copies the matching sub-tab's per-run settings.
    Returns the request, or None if the user canceled. Cumulant is the default (ISO
    22412's sizing basis, whose size is the scattered-intensity-weighted harmonic mean)."""
    dlg = AverageDialog(parent, controller, sources)
    if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return None
    try:
        return dlg.request()
    except ValueError as exc:       # a copied CONTIN α range the engine would refuse
        QtWidgets.QMessageBox.warning(parent, 'Average derived results', str(exc))
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
             f'{n_ok} of {n_rep} replicate(s) fit successfully.']
    # What the fits read -- the settings (Settings or a copied tab's) and, when its box was
    # ticked, the Correlogram tab's region -- in the controller's own words, one sentence
    # each (`Controller.average_inputs_text` / `region_note`).
    lines += [f'ⓘ {note}' for note in summary.get('notes') or ()]
    lines.append('')

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
# `core.naming`'s `measurement_display_name` / `sample_display_name` are injected as the
# picker's label callables -- the one builder every surface in the program shares.

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
    '<b>“Show fit details”</b> adds every other quantity each parametric fit reports '
    '(decay rate, moments, fitted order and baseline, fractions, stretch, convergence). '
    'Only the methods you actually ran add columns, each column names its fit, and the '
    'table scrolls sideways when they do not fit.',
    'ⓘ on a cell marks a note about how that number was obtained — hover to read it. '
    'It is a remark, not a warning; ⚠ is the warning.',
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
# (invariant 7), said where the number is read (ui_style_guide R5.3), and it is the whole
# qualifier -- WHICH average a cumulant fit reports (the scattered-intensity-weighted
# harmonic mean) is the Theory-and-Equations Guide's to define (CLAUDE.md invariant 7).

# Passive (nuance) tooltips for the cumulant order control and the μ₃ / μ₄ result rows.
# They carry the WHOLE caution: the higher moments ship bare, with no per-value validity
# glyph, because their unreliability is a constant property of the method rather than a
# per-run precondition (owner-ruled 2026-09-05; the ruling's home, its literature basis and
# the measured magnitudes are on `CumulantResult`). No citation in the text (ui_style_guide
# R6.1) -- the guide section is the pointer. Module constants so the GUI tier can assert them.
_CUMULANT_ORDER_TOOLTIP = (
    'Degree of the cumulant expansion: 1 = size (D) only, 2 adds PDI (polydispersity), '
    '3 adds μ₃ (the decay-rate distribution\'s asymmetry), 4 adds μ₄; cumulant method '
    'only, greyed out for the others. Orders '
    '3–4 need broad distributions — on this program\'s synthetic benchmarks order 4 '
    'improves on order 2 in Γ at PDI ≈ 0.6, and in both Γ and the PDI only on clean data '
    'at PDI ≈ 0.2–0.3; everywhere else — lower PDIs, noisier data up to PDI ≈ 0.3, and '
    'the PDI at PDI ≈ 0.6 — order 2 is at least as precise. Order 3 is offered for its '
    'diagnostic μ₃: the literature does not recommend it, and on those benchmarks it is the '
    'one order that can report a badly wrong PDI as reliable on a broad sample — read the '
    'size at order 2 and broad distributions at order 4; see the '
    'Theory-and-Equations-Guide §3.2.6.')
_HIGHER_MOMENT_TOOLTIP = (
    'Fitted parameter, not a measured moment: at best (nonlinear method, broad sample) its '
    'uncertainty is comparable to its own value, and on the linear method or below '
    'PDI ≈ 0.3 it is noise — a 0 is the fit\'s bound, not a measurement. Use it to compare '
    'fits, not to characterize the sample; see the Theory-and-Equations-Guide §3.2.6.')
# Row-LABEL tooltips of the results table, keyed by the row's label so a reordering of
# `_RESULT_ROWS` cannot move a caution onto the wrong quantity. On the label cell, not the
# value cell: a value cell's tooltip carries that value's own reliability REASON (or, with
# nothing to explain, its full text for the elided-column case), and the two must not
# fight. Applied where the labels are BUILT (`_repoint_vtable`, which runs on every method
# change and would otherwise overwrite them); a row not listed here keeps its own label as
# the tooltip, the elided-header convention.
_RESULT_ROW_TOOLTIPS = {'μ₃ (s⁻³)': _HIGHER_MOMENT_TOOLTIP, 'μ₄ (s⁻⁴)': _HIGHER_MOMENT_TOOLTIP}

_RESULT_ROWS = {
    'cumulant': ['Γ (s⁻¹)', 'R<sub>h</sub> (nm, apparent)', 'PDI', 'μ₂ (s⁻²)',
                 'μ₃ (s⁻³)', 'μ₄ (s⁻⁴)', 'order', 'method', 'baseline B'],
    'single':   ['Γ (s⁻¹)', 'R<sub>h</sub> (nm, apparent)', 'β', 'baseline', 'converged'],
    'double':   ['fast R<sub>h</sub> (nm, apparent)', 'fast frac',
                 'slow R<sub>h</sub> (nm, apparent)', 'slow frac', 'baseline', 'converged'],
    'kww':      ['stretch s', 'τ<sub>c</sub> (s)', '⟨τ⟩ (s)',
                 'R<sub>h</sub>(τ<sub>c</sub>) (nm, apparent)',
                 'R<sub>h</sub>(⟨τ⟩) (nm, apparent)', 'baseline', 'converged'],
}

# The HEADLINE quantities of each method -- what the glance table shows, one row per
# ticked run. Every other quantity stays in the focused-run panel (`_RESULT_ROWS`).
#
# A glance column is named by the `_RESULT_ROWS` row it SHOWS, never by an index, for
# the reason `_RESULT_ROW_TOOLTIPS` is keyed by label. The header is the short form;
# "apparent" is said once on the table's title line.

# That resolution proves the LABEL exists and nothing more -- the index then subscripts
# `_result_values`, an independent hand-written list, and `_result_notes` hard-codes the
# same integers a third time. What forbids a drift between them is a pair of tests, not
# this resolution: `test_every_result_row_renders_the_quantity_its_label_names` and
# `test_a_reliability_note_lands_on_the_size_it_judges`. An earlier version of this
# comment claimed their safety before they existed.
_GLANCE_ROWS = {
    'cumulant': [('R<sub>h</sub> (nm)', 'R<sub>h</sub> (nm, apparent)'),
                 ('PDI', 'PDI')],
    'single':   [('R<sub>h</sub> (nm)', 'R<sub>h</sub> (nm, apparent)')],
    'double':   [('fast R<sub>h</sub> (nm)', 'fast R<sub>h</sub> (nm, apparent)'),
                 ('slow R<sub>h</sub> (nm)', 'slow R<sub>h</sub> (nm, apparent)')],
    'kww':      [('R<sub>h</sub>(τ<sub>c</sub>) (nm)',
                  'R<sub>h</sub>(τ<sub>c</sub>) (nm, apparent)'),
                 ('R<sub>h</sub>(⟨τ⟩) (nm)',
                  'R<sub>h</sub>(⟨τ⟩) (nm, apparent)')],
}

# Resolved to (header, index-into-`_result_values`) at import, so a headline naming a
# row that does not exist is an ImportError here and not a wrong number on screen.
# `.index` rather than `assert`: an assert disappears under -O, this cannot.
# How many glance rows are visible before the table scrolls. Six, because that is the
# replicate-set size the readability criterion is written against -- six ticked runs
# must be readable without scrolling, and a seventh must not cost the panel below.
_GLANCE_VISIBLE_ROWS = 6

# ... and the fewest it keeps when the section is squeezed. Three is enough to read a
# value, see that a second run differs, and know to scroll; below that the table stops
# being a summary at all.
_GLANCE_FLOOR_ROWS = 3

# THE LEGEND IS SPLIT BY DESTINATION (ui_style_guide R10.6). The SCREEN gets a
# compact key -- one entry per METHOD drawn plus the window/baseline markers, constant
# in the selection size -- and the SAVED file gets that plus one entry per MEASUREMENT,
# grouped under its sample, in a figure grown to hold it. `_draw_key` and
# `export_legend_dressing` carry the measurements behind each half.

#: Point size of the on-screen key, and of the export legend.
_KEY_FONT_PT = 7.0
_EXPORT_FONT_PT = 7.0

#: Legend `labelspacing`, in units of the font size.
_LEGEND_LABELSPACING = 0.3

# The on-screen key shortens two marker labels. `tau window (fit range)` and
# `baseline region` render 229 px wide together in the 160 px corner cell the key
# lives in (measured); at these they measure 93 px and fit with room. The full
# wording goes to the export legend, which has the width.
_KEY_SHORT_LABELS = {'τ window (fit range)': 'τ window',
                     'baseline region': 'baseline'}

#: A saved figure grows by its legend's own height, capped so a large selection
#: cannot emit a strip. 2.0 in is about 40 rows at 7 pt -- past any real selection,
#: and short of an image whose legend dwarfs its plot.
_EXPORT_GROW_CAP_IN = 2.0

#: The residual panel's floor, in pixels of drawn axes. Below this the panel stops
#: showing a residual at all. It does NOT fire at the shipped window size once the
#: legend is a key (measured: 59.8 px); it is insurance for a smaller window, where
#: the fraction-of-the-stack split under-serves a panel whose own chrome is fixed.
_RESID_FLOOR_PX = 45.0


#: The labels `_draw_markers` puts on the correlogram's own chrome artists -- DERIVED
#: from the shortening map above rather than typed again, so the two cannot drift.
#: Matched EXACTLY: a substring test on 'window'/'baseline' also caught any
#: measurement whose name contains either word -- `baseline run`, `baseline_90.asc`
#: -- pulling its data scatter into a key that is supposed to be constant in the
#: selection size, and into the exported legend a second time.
_MARKER_LABELS = tuple(_KEY_SHORT_LABELS)


def _marker_legend_entries(ax):
    """The window/baseline marker entries a correlogram axes has labelled itself."""
    handles, labels = ax.get_legend_handles_labels()
    return [(h, lab) for h, lab in zip(handles, labels, strict=True)
            if lab in _MARKER_LABELS]


def _method_legend_entries(methods):
    """One neutral-ink entry per method drawn, carrying that method's line style."""
    ink = _neutral_ink()
    return [(Line2D([], [], color=ink, lw=1.4, ls=_METHOD_LINESTYLE.get(k, '-')),
             _fit_label(k)) for k in methods]


def _measurement_legend_entries(selection, names_by_sample):
    """One entry per measurement, grouped under a bold sample row.

    `names_by_sample` is `{sample label: [(iid, unprefixed name)]}` in draw order.
    The sample is a GROUP ROW rather than a prefix on every entry -- the shape the
    measurement picker and the DLS Summary table already use, and what takes the
    longest entry from 76 characters to 44. Returns `(entries, group_labels)`; the
    caller bolds the group rows.
    """
    out, groups = [], []
    for sample, rows in names_by_sample.items():
        out.append((Line2D([], [], ls='none'), sample))
        groups.append(sample)
        for iid, name in rows:
            out.append((Line2D([], [], color=selection.color_for(iid),
                               marker=selection.marker_for(iid) or 'o',
                               markerfacecolor='none', ls='none'),
                        '   ' + name))
    return out, groups


def _draw_key(figure, entries, *, bbox=None, ncol=None):
    """Draw the on-screen KEY and return it, or None when there is nothing to show.

    `bbox` places it inside a region of the figure (the Correlogram tab's empty
    bottom-right gridspec cell); without one it sits below the panel, which is what
    the Distribution tab does -- that figure is a plain stack with no free cell.
    Clears any previous legend first: a figure legend outlives `ax.clear()`.
    """
    figure.legends.clear()
    if not entries:
        return None
    handles = [h for h, _lab in entries]
    labels = [_KEY_SHORT_LABELS.get(lab, lab) for _h, lab in entries]
    if bbox is not None:
        # BOTTOM-aligned inside the cell: the cell is empty by construction, but the
        # side view above spills its x tick labels down into it, so a top-aligned key
        # prints over them (measured).
        return figure.legend(handles, labels, loc='lower left',
                             bbox_to_anchor=(bbox.x0, bbox.y0, bbox.width,
                                             bbox.height),
                             frameon=False, fontsize=_KEY_FONT_PT, handlelength=2.0,
                             handletextpad=0.5, borderaxespad=0.0,
                             labelspacing=_LEGEND_LABELSPACING)
    return figure.legend(handles, labels, loc='outside lower center', frameon=False,
                         fontsize=_KEY_FONT_PT, handlelength=2.0,
                         ncol=ncol or len(labels), columnspacing=1.6,
                         labelspacing=_LEGEND_LABELSPACING)


def export_legend_dressing(figure, entries, groups):
    """A `save_figure_via` dressing: put the FULL legend on the written figure.

    Returns the zero-argument callable that hook wants, which itself returns an undo.
    The figure grows by the legend's own height (capped at `_EXPORT_GROW_CAP_IN`) so
    nothing is taken from either panel -- the saved plot is LARGER than the on-screen
    one, not rationed against its legend.

    ONE column, deliberately: matplotlib fills columns top-to-bottom, so a sample
    group splits across a column boundary and the right column clips the margin
    (both measured). One column needs no straddling rule at all.
    """
    def dress():
        keep = list(figure.legends)
        figure.legends.clear()
        size_in = figure.get_size_inches().copy()
        grow = min(len(entries) * _EXPORT_FONT_PT * 1.45 / figure.dpi + 0.12,
                   _EXPORT_GROW_CAP_IN)
        figure.set_size_inches(size_in[0], size_in[1] + grow)
        leg = figure.legend([h for h, _lab in entries],
                            [lab for _h, lab in entries],
                            loc='outside lower center', frameon=False,
                            fontsize=_EXPORT_FONT_PT, handlelength=2.0,
                            handletextpad=0.6, ncol=1,
                            labelspacing=_LEGEND_LABELSPACING)
        for text in leg.get_texts():
            if text.get_text() in groups:
                text.set_fontweight('bold')

        def undo():
            leg.remove()
            figure.legends.extend(keep)
            figure.set_size_inches(*size_in)
        return undo
    return dress


def _hide_shared_x(upper_ax):
    """Drop the x tick labels and x label from the UPPER axes of a fit+residual
    stack: the two share an x range and the residual carries the only axis, which is
    the standard convention. Both were printed twice until this (measured: removing
    the duplicate gives the residual ~6 px and the fit ~20 px).

    ONLY where the two panels genuinely share x. Applied to the DLS Distribution tab,
    whose panels are Rh (or Γ) over delay time τ, it strips the distribution's own
    axis and leaves it above an axis for a different quantity -- which is what it did
    until round `8fb4` caught it. The caller asserts the shared range; this helper
    cannot check it, since an axes carries no record of what its neighbour plots.
    """
    upper_ax.set_xlabel('')
    upper_ax.tick_params(labelbottom=False)


# Columns in the "Fits to show" grid -- see the measurement at its construction.
_LAYER_COLS = 4

_GLANCE_COLS = {
    key: [(header, _RESULT_ROWS[key].index(row)) for header, row in pairs]
    for key, pairs in _GLANCE_ROWS.items()
}


class _CorrelogramTab(QtWidgets.QWidget):
    """Parametric-fit view: four correlogram scales + residuals + result tables,
    over ONE or MANY co-plotted measurements (ticked in the checklist). All share
    one τ-window / baseline; each measurement gets a stable colour and a glance row.

    ANY stored fit can be drawn over the data, not only the one the Method box is
    showing: the "Fits to show" list offers all seven -- the four parametric fits this
    tab runs and the three distribution fits the Distribution tab runs -- so a cumulant
    and a CONTIN reconstruction can be compared on one correlogram. Colour and marker
    are the MEASUREMENT, line style is the METHOD, and the residual panel beneath shows
    the residuals of exactly the fits drawn above it, in the same coding."""

    #: The user moved the shared region -- a drag released, a field edited, or Reset.
    #: The owning `DLSModule` routes it to every "use the Correlogram tab's delay window
    #: and baseline" checkbox, which relabels itself and flags a checked result stale.
    regionChanged = QtCore.Signal()

    def __init__(self, controller, region, selection, parent=None, *,
                 distribution_fits=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.region = region                    # shared _AnalysisRegion (seconds)
        self.selection = selection              # shared _OverlaySelection
        self.item_id: Optional[str] = None
        self._runnable = False
        self._raw: Dict[str, Tuple] = {}        # iid -> (tau, g2m1) for checked DLS
        self._union_tau: Optional[Tuple[float, float]] = None
        # (iid, method) -> {'result','sig'}. Keyed by the PAIR since B8: a
        # correlogram can carry several fits at once, so storing one fit per
        # measurement made comparing two impossible and dropped the previous method's
        # result on every Run. The Distribution tab's results already had this shape.
        self._cache: Dict[Tuple[str, str], Dict] = {}
        # Zero-arg accessor for the fits the DISTRIBUTION tab owns, injected by the
        # owning DLSModule -- `{(iid, method): {'result','stale'}}`. A default that
        # answers nothing keeps this tab constructible on its own (its unit tests do),
        # and the indirection is why no widget here reaches into its sibling.
        self._distribution_fits = distribution_fits or (lambda: {})
        # (iid, method) -> reason, the SAME key as `_cache`, and cleared with each Run.
        # Keyed by iid alone it was READ against whatever method the box happened to
        # show, so one method's Run failure printed "skipped" beside another method's
        # displayed numbers AND suppressed that number's own reliability note
        # (invariant 4 inverted). It became a sibling of `_cache` the moment the cache
        # took the pair key, and did not move with it.
        self._run_failures: Dict[Tuple[str, str], str] = {}
        self._scales = list(_SCALE_MODES)       # [main, side0, side1, side2]
        self._markers: Dict[str, object] = {}   # handle name -> Line2D on main_ax
        self._handle_glyphs: Dict[str, object] = {}  # handle name -> caret Line2D
        self._placeholder = None                # empty-state Text artist (nothing ticked)
        self._base_span = None                  # baseline shaded region inside the window
        self._base_span_outside: list = []      # ... and its greyed parts outside it
        self._drag: Optional[str] = None        # handle being dragged, or None
        self._suppress_fields = False
        self._build_ui()

    def _build_ui(self) -> None:
        splitter, left, right = make_split_panels(self, left_min_width=300,
                                                  sizes=(360, 740))

        self.checklist = MeasurementPicker(
            self.controller, self.selection, kinds=('dls',),
            label_fn=measurement_display_name, header_fn=sample_display_name,
            tooltip_fn=measurement_hover_card,
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
            'Moving the window or baseline, or changing the method, cumulant order '
            'or fit cutoff, after a Run flags the shown fit stale — press <b>Run</b> '
            'again to refresh it.',
            '<b>Cumulant order</b>: 2 suits most samples. Order 4 helps mainly on broad '
            'distributions (PDI around 0.6), and there in Γ rather than in the PDI; below '
            'PDI ≈ 0.3, order 2 is at least as precise except on clean data from about '
            'PDI 0.2. μ₃ '
            'and μ₄ are diagnostics for comparing fits, not measurements of the sample.',
            'The <b>baseline region</b> sets where g₂−1 → 0 is measured, and each '
            'method reads it its own way: the single/double/KWW fits and the '
            '<i>linear</i> cumulant <b>subtract</b> it (shown in the <b>baseline</b> '
            'row), the <i>nonlinear</i> cumulant measures its own floating baseline '
            '<b>over</b> it, and the distribution methods <b>normalize</b> by it.',
            'The window and the baseline region are each applied only <b>once you move '
            'them</b>. Until then each run keeps its own full lag range, and each fit '
            'runs as if no baseline had been given: the single/double/KWW fits and the '
            'linear cumulant subtract <b>nothing</b>, the nonlinear cumulant measures its '
            'floating baseline over the last quarter of the lags that collected '
            'coincidence, and the distribution methods estimate their own. So a region '
            'left over from another measurement cannot quietly change a number. '
            '<b>Reset window + baseline</b> returns to that state.',
            'Only the part of the baseline region <b>inside the delay window</b> is read; '
            'the part outside it is drawn grey and not used.',
            'The <i>linear</i> cumulant regresses only the run of points around the peak '
            'that stays above a fraction of it — a tenth by default, set in '
            '<b>Settings ▸ Cumulant fit cutoff</b> — so its drawn <b>fit</b> line and the '
            'residual panel under it stop where that run does. Every method regresses '
            'only the channels in the delay window that collected any coincidence — a '
            'channel at g₂ = 0 is dropped by every fit and counted in a note — and the '
            'other methods\' line spans the window.',
            'A <i>linear</i> cumulant fit that bends back <b>upward</b> at its right-hand '
            'end is telling you the sample is too polydisperse for the method — the same '
            'thing the PDI warning says. Prefer a distribution method (NNLS / CONTIN) '
            'when you see it.',
            'The Γ vs q², D vs c, Mode Map and DDLS sub-tabs and <b>Average derived '
            'results…</b> read this region only when you tick their <b>Use the '
            'Correlogram tab\'s delay window and baseline</b> box; otherwise each run '
            'uses its own defaults.',
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
            'The linear cumulant fits a quadratic in the exponent, so its curve has a '
            'minimum at τ = Γ/μ₂ = 1/(PDI·Γ) and rises after it. At a high PDI that '
            'point falls inside the measured lags and the drawn fit visibly turns back '
            'up — the expansion past its range of validity, not a fitting failure. The '
            'nonlinear form can only do this above PDI 2. See the '
            'Theory-and-Equations-Guide.\n'
            'Every method also flags an Rh outside the range DLS can establish — above '
            '~5 µm (10 µm diameter) sedimentation outweighs Brownian motion, below 0.2 nm '
            'the fit chased a short-lag artifact — and the cumulant flags a PDI above '
            '0.3.\n'
            'Once you move the purple baseline region, the baseline measured over its part '
            'inside the delay window is subtracted before the fit as a pinned offset '
            '(Single/Double/KWW, linear cumulant), and the nonlinear cumulant measures its '
            'floating baseline over that part instead of the last quarter of the live lags. '
            'Until you move it, the first four subtract nothing and the nonlinear cumulant '
            'measures its own baseline over the last quarter of the lags that collected '
            'coincidence. An offset left in the data is otherwise absorbed as a spurious '
            'slow mode.\n'
            'A nonlinear cumulant whose µ₂ comes to rest on its zero bound adds a passive '
            'note: the PDI shown is that bound, not a measurement. At order 4 a µ₄ resting '
            'on its bound gets its own note. The Theory-and-Equations-Guide §3.2.7 explains '
            'what can cause either. '
            'Which residual weighting built the fit, chosen in Settings, is recorded on the '
            'result; the export '
            'names it only when it is not the program\'s default, the sourced '
            'Biganzoli–Ferri variance (with the run\'s committed mean count rate when one '
            'exists, its classical limit when none does). That variance is evaluated on '
            'the correlator\'s own per-channel sample times when the file states them; '
            'otherwise on a stand-in derived from the delay axis, which the export names. '
            'It also uses the run\'s duration where the file states it, and a long-run '
            'form where it does not. See the Theory-and-Equations-Guide §3.2.9.')
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        form.addRow('Method:', self.method_combo)
        self.order_spin = QtWidgets.QSpinBox()
        # 1-4, the engine's own range (`fit_cumulants`): order 4 is where the literature
        # stops, and the regime/noise caveats on orders 3-4 are this tooltip's job, since
        # the higher moments ship with no per-value flag (`CumulantResult.mu4_s_inv4`).
        self.order_spin.setRange(1, 4)
        self.order_spin.setValue(self.controller.settings.cumulant_order)
        self.order_spin.setToolTip(_CUMULANT_ORDER_TOOLTIP)
        form.addRow('Cumulant order:', self.order_spin)
        # The order is a fit input: changing it beside a shown fit flags that fit stale.
        self.order_spin.valueChanged.connect(lambda _v: self._on_fit_input_moved())
        win, self.tau_min, self.tau_max, self.tau_unit = _delay_window_row()
        form.addRow('Delay window:', win)
        # Baseline region (every DLS fit reads it): mean g2-1 over this τ range.
        brow = QtWidgets.QHBoxLayout(); brow.setContentsMargins(0, 0, 0, 0)
        self.base_lo = QtWidgets.QLineEdit(); self.base_lo.setPlaceholderText('low')
        self.base_hi = QtWidgets.QLineEdit(); self.base_hi.setPlaceholderText('high')
        brow.addWidget(self.base_lo); brow.addWidget(self.base_hi)
        bwidget = QtWidgets.QWidget(); bwidget.setLayout(brow)
        form.addRow('Baseline region:', bwidget)
        # Reset the window + baseline back to the defaults (full lag range; baseline
        # = last 25 %) — handy after dragging/zooming, and the one control that clears
        # both touched flags, so the fits go back to their own per-file estimates.
        self.reset_region_button = QtWidgets.QPushButton('Reset window + baseline')
        self.reset_region_button.setToolTip(
            'Put the delay window back to the full lag range and the baseline region '
            'back to the last quarter of the lags, and stop applying either: until you '
            'move a handle again each run keeps its own full lag range, the '
            'single/double/KWW and linear-cumulant fits subtract nothing, the nonlinear '
            'cumulant measures its own baseline over the last quarter of the lags that '
            'collected coincidence, and the distribution methods estimate their own.')
        self.reset_region_button.clicked.connect(self._on_reset_region)
        form.addRow(self.reset_region_button)
        # Two-way sync: editing a field moves its marker.
        for edit in (self.tau_min, self.tau_max, self.base_lo, self.base_hi):
            edit.editingFinished.connect(self._on_field_edit)
        self.tau_unit.currentTextChanged.connect(lambda _t: self._fields_from_region())
        self.run_button = QtWidgets.QPushButton('Run')
        self.run_button.clicked.connect(self._on_run)
        # NOT added to the form: it is pinned below the scroll area instead -- see
        # `fit_pane`. In the form it sat at content y=182 against a viewport measured
        # at 120/155/172 px (1280x760 / 1600x1020 / 1920x1080), i.e. below the fold at
        # every shipped window size, so the tab's primary action had to be scrolled to.

        # ---- "Fits to show": this tab's first LAYERS list (the locked GUI decision's
        # checkbox-layers pattern; the Distribution tab's `method_checks` is the nearest
        # existing one). One entry per method that can reconstruct a correlogram,
        # enabled once that fit exists for a ticked run.
        # It rides with the PLOT, not the left control column, and that is MEASURED: at
        # 1280x760 that column's three panes all sit at their floors (picker 200 px, its
        # own minimum; fit 120; results 300; of 636), so a fourth would overlap its
        # children (R3.2), and in the scrolled fit pane the box sat below the fold.
        layers_box = QtWidgets.QGroupBox('Fits to show')
        add_help_to_groupbox(
            layers_box,
            'Draw any stored fit over the data — not just the one in the Method box.',
            bullets=[
                'An entry turns on once that fit <b>exists</b> for a ticked '
                'measurement. The four parametric fits come from this tab’s <b>Run</b>; '
                '<b>Lognormal / NNLS / CONTIN</b> come from the '
                '<b>Distribution</b> sub-tab’s own Run.',
                'The <b>colour and point shape</b> are the measurement; the '
                '<b>line style</b> is the method, and means the same thing on the '
                'Distribution sub-tab.',
                'The <b>residual panel</b> beneath the plot shows the residuals of '
                'exactly the fits ticked here, in the same colours and line styles.',
                'A fit whose inputs have changed since it ran is marked '
                '<b>(stale)</b> and is not drawn — press <b>Run</b> on the tab that '
                'owns it to refresh it.',
                'The <b>results table</b> follows the <b>Method</b> box, not '
                'this list: it shows one method’s numbers at a time.',
                'The <b>key</b> in the plot’s corner names the methods drawn. The '
                '<b>measurements</b> are named in the figure <b>Save</b> writes — '
                'the checklist and results table beside the plot already name them '
                'here, and a saved image has neither.',
            ])
        # Two columns: seven checkboxes in one column take ~140 px of a control column
        # whose results section is already floored at 300 px.
        # Two rows of FOUR: measured on the real platform, the widest row is 434 px
        # against a 550 px canvas with every entry carrying its "(stale)" suffix, where
        # one row of seven needs 742 px and would be cut.
        lgrid = QtWidgets.QGridLayout(layers_box)
        lgrid.setContentsMargins(8, 2, 8, 2)
        lgrid.setVerticalSpacing(1)
        self.layer_checks: Dict[str, QtWidgets.QCheckBox] = {}
        for i, (label, key) in enumerate(_FIT_METHODS):
            cb = QtWidgets.QCheckBox(_fit_label(key))
            cb.setToolTip(
                f'Draw the stored {label} fit of every ticked measurement over the '
                'data, and its residuals in the panel beneath. Enabled once that fit '
                'exists; the line style shown is this method’s, on both DLS '
                'plot sub-tabs.')
            cb.setEnabled(False)               # until a fit of this method exists
            cb.stateChanged.connect(self._on_layer_toggled)
            self.layer_checks[key] = cb
            lgrid.addWidget(cb, i // _LAYER_COLS, i % _LAYER_COLS)
        lgrid.setColumnStretch(_LAYER_COLS, 1)     # keep the row left-packed
        # Guards the programmatic tick/enable pass below against re-entering `_redraw`
        # through `stateChanged` -- a sync that redraws that syncs.
        self._syncing_layers = False

        # One results area for the selected method: a single
        # table rebuilt with that method's rows, with Export directly beneath it.
        results_section = QtWidgets.QWidget()
        rlay = QtWidgets.QVBoxLayout(results_section)
        rlay.setContentsMargins(0, 0, 0, 0)
        self.results_label = QtWidgets.QLabel('Results')
        rlay.addWidget(self.results_label)
        # GLANCE table -- one ROW per ticked run: its plot swatch, its name, and the
        # method's headline values (`_GLANCE_COLS`). One row per run rather than one
        # COLUMN per run, which is what this table used to be and why it failed by
        # construction: the control column is 360 px wide, so three ticked runs left
        # each value 70-80 px and every cell painted as '0…'. Measured once the one
        # name builder landed: a one-column-per-measurement header overran its column by
        # 228 px, i.e. a name cannot be held at any width in that shape.
        self.glance_table = QtWidgets.QTableWidget(0, 1)
        self.glance_table.setHorizontalHeader(RichTextHeaderView(self.glance_table))
        self.glance_table.verticalHeader().setVisible(False)
        self.glance_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        # Left-elided like the derived-run table: a file stem's TAIL is what tells six
        # replicates of one set apart, so the head is the part that may be dropped.
        self.glance_table.setTextElideMode(QtCore.Qt.TextElideMode.ElideLeft)
        # Stretch 1 against the detail panel's 0: when the section is short -- which it
        # is at the shipped window size, where all three panes sit at their minimums --
        # the SUMMARY must survive and the details must scroll, not the reverse. Without
        # this the panel held its full height and the glance table collapsed to its
        # header row, so the tab showed column titles and no values.
        rlay.addWidget(self.glance_table, 1)
        # FOCUSED-RUN panel -- every quantity of the fit that is not a headline, for the
        # one run the sidebar has focused. This is the widget this section held before
        # `_fill_tables` began rebuilding it as one column per measurement, so pointing
        # it back at a single run is partly a revert. A planned change moves these
        # quantities onto the Summary tab and takes this panel with them; until then
        # nothing loses its home.
        self.focus_label = QtWidgets.QLabel('')
        self.focus_label.setWordWrap(True)
        rlay.addWidget(self.focus_label)
        self.result_table = _vtable(_RESULT_ROWS['cumulant'])
        rlay.addWidget(self.result_table)
        # Export is pinned under the column with the status line (`pin_under_column`
        # below), not in this pane: when the column is taller than its viewport, the
        # last pane is the one that scrolls out of view.
        self.export_button = QtWidgets.QPushButton('Export CSV…')
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._on_export)
        rlay.addStretch(1)
        # Vertically resizable control column: drag the grips to size the
        # checklist / fit controls / results panes against each other.

        # The fit controls ride in a scroll area so their MINIMUM stops dictating the
        # column. `make_vertical_plot_stack` floors each pane at its own
        # `minimumSizeHint`; this box's form rows put that near 300 px, which the
        # splitter honours by starving the pane below -- leaving the results section
        # about 60 px, its title and a header row and no values.
        fit_scroll = QtWidgets.QScrollArea()
        fit_scroll.setWidgetResizable(True)
        fit_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        fit_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        fit_scroll.setWidget(box)
        # Run is pinned OUTSIDE the scrolled form (the §3 column template): the fit
        # controls may scroll, the action that runs them may not.
        fit_pane = QtWidgets.QWidget()
        fit_layout = QtWidgets.QVBoxLayout(fit_pane)
        fit_layout.setContentsMargins(0, 0, 0, 0)
        fit_layout.setSpacing(4)
        fit_layout.addWidget(fit_scroll, 1)
        fit_layout.addWidget(self.run_button)
        # The floor is passed to the stack, NOT set on the widget: Qt's `qSmartMinSize`
        # lets an explicit `setMinimumHeight` REPLACE the layout-derived minimum rather
        # than floor it, so a literal clearing the content at the shipped font stops
        # clearing it at a larger one (measured: 272 px at 9 pt, 320 px at 18 pt, which
        # 150 % text scaling reaches). `make_vertical_plot_stack` takes the max of the
        # two. A splitter handed less than a pane needs OVERLAPS its children (R3.2).
        vstack = make_vertical_plot_stack(
            [self.checklist, fit_pane, results_section], sizes=[200, 250, 300],
            min_heights=[80, 120, 300], stretch=COLUMN_STACK_STRETCH)
        left.addWidget(vstack, 1)

        # Export, the status line and the reliability flag sit UNDER the scrolled
        # control column, not inside it (style guide R8.6 and the §3 column template):
        # a flag is an always-visible label (§5), and at the shipped window size the
        # column is taller than its viewport, so whatever sits last scrolls out of view.
        # Pinning also takes the word-wrapping labels out of the scrolled layout, so the
        # scroll area sizes the column to its minimums rather than to the widgets'
        # natural heights.
        self.status = QtWidgets.QLabel('')
        self.status.setWordWrap(True)
        self.flag_label = ThemedLabel('', role='error', bold=True)
        self.flag_label.setWordWrap(True)
        # Passive notices the fit carries about HOW it was obtained (dead channels, a
        # floating baseline resting on its bound) -- the neutral ⓘ qualifier tier, kept
        # in its own label so it can never be painted at the flag_label's alarm level
        # (style guide §5/R5.1). It sits under the results table, not pinned, and
        # scrolls: it names every run it applies to, so with several ticked, pinned, it
        # would run past the window's bottom edge.
        self.notes_label = ThemedLabel('', size=11)
        self.notes_label.setWordWrap(True)
        rlay.insertWidget(rlay.count() - 1, scrolling_notes(self.notes_label))
        pin_under_column(splitter, self.export_button, self.status, self.flag_label)

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
        self._resid_dragged = False              # set by the drag handle; see the floor
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
            self._on_resid_dragged)
        right.addWidget(self.nav_toolbar)
        right.addWidget(layers_box)
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

    def _floor_resid_ratio(self) -> None:
        """Raise the residual's share until its drawn box clears `_RESID_FLOOR_PX`.

        The share is a FRACTION of a varying stack while the panel's own chrome is a
        fixed ~29 px, so the fraction under-serves the panel exactly when the stack is
        short. It does not fire at the shipped window size (measured 59.8 px there);
        it is insurance for a smaller one.

        Only while the user has not dragged the handle: an explicit drag is a decision
        and outranks the floor (`_ResidualResizer` owns `_resid_ratio` after that).
        """
        if self._resid_dragged:
            return
        for _ in range(12):
            try:
                box = self.resid_ax.get_window_extent(self.canvas.get_renderer())
            except Exception:  # noqa: BLE001 — not laid out yet; nothing to floor
                return
            if box.height >= _RESID_FLOOR_PX or self._resid_ratio >= 0.40:
                return
            self._apply_resid_ratio(min(self._resid_ratio + 0.02, 0.40))
            self.canvas.draw()

    def _on_resid_dragged(self, frac: float) -> None:
        """The drag handle moved: apply it, and stop the floor second-guessing it.
        An explicit drag is a decision and outranks `_floor_resid_ratio`."""
        self._resid_dragged = True
        self._apply_resid_ratio(frac)

    def reseed_from_settings(self) -> None:
        self.order_spin.setValue(self.controller.settings.cumulant_order)

    def showEvent(self, event) -> None:                      # keep selection in sync
        super().showEvent(event)
        self.checklist.refresh()
        self._reload_raw()
        self._sync_layers()
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
        self._sync_layers()
        self._redraw()
        self._refresh_tables()
        self._update_status()

    @QtCore.Slot()
    def _on_selection_changed(self) -> None:
        self._reload_raw()
        self.run_button.setEnabled(bool(self._raw))
        self.export_button.setEnabled(self._any_cached())
        self._sync_layers()
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
            self.region.init_from_data(allt)      # untouched parts follow the union range
        else:
            self._union_tau = None
        self._fields_from_region()

    def _any_cached(self) -> bool:
        """Whether the METHOD BOX's fit exists for any ticked run. The Export button and
        the status line follow the Method box, not the layers list: an export writes one
        method's numbers."""
        key = self.method_combo.currentData()
        return any((i, key) in self._cache for i in self.selection.ids())

    def _any_stale(self) -> bool:
        """True if any co-plotted, method-matching cached fit was computed from
        committed params that have since changed."""
        key = self.method_combo.currentData()
        return any(self._cache_is_stale(iid, key) for iid in self.selection.ids())

    def _cache_is_stale(self, iid: str, key: str) -> bool:
        """Whether THIS tab's stored fit of `(iid, key)` was computed from inputs that
        have since moved. False when there is no such fit -- absent is not stale."""
        v = self._cache.get((iid, key))
        return v is not None and v['sig'] != self._run_signature(iid, key)

    # ---- the drawable fits: this tab's, plus the ones the Distribution tab owns ----
    def _stored_fits(self) -> Dict[Tuple[str, str], Dict]:
        """Every stored fit available for the TICKED set, keyed `(iid, method)`, each
        carrying the staleness verdict of the tab that RAN it -- per item 8's rule that
        staleness is judged by its owner.

        For the four parametric methods that is this tab's per-fit signature comparison.
        For the three distribution methods it is the Distribution tab's own verdict,
        which is per fit too: that tab keeps each fit in its own (measurement, method)
        slot with its own signature (this tab's `_cache` shape), so one run's fit can be
        fresh beside another's stale one. Its ticks and method boxes do not filter what
        it hands over; this tab filters by its own ticked set, here."""
        ticked = set(self.selection.ids())
        out: Dict[Tuple[str, str], Dict] = {}
        for (iid, key), v in self._cache.items():
            if iid in ticked:
                out[(iid, key)] = {'result': v['result'],
                                   'stale': self._cache_is_stale(iid, key)}
        for (iid, key), v in self._distribution_fits().items():
            if iid in ticked:
                out[(iid, key)] = v
        return out

    def _shown_fits(self, stored: Optional[Dict] = None) -> List[Tuple[str, str, object]]:
        """`[(iid, method, result)]` for every fit the layers list has ticked, in
        measurement order then method order, skipping the stale ones (item 8: a fit its
        own tab marks stale is not drawn). The one ordering both the plot and the
        residual panel beneath it iterate, so the two cannot disagree about which curve
        is which."""
        stored = self._stored_fits() if stored is None else stored
        out = []
        for iid in self._raw:                    # ticked, in selection order
            for _label, key in _FIT_METHODS:
                cb = self.layer_checks.get(key)
                if cb is None or not cb.isChecked():
                    continue
                entry = stored.get((iid, key))
                if entry is None or entry['stale']:
                    continue
                out.append((iid, key, entry['result']))
        return out

    def _sync_layers(self, stored: Optional[Dict] = None, tick: Iterable[str] = ()) -> None:
        """Bring the layers list level with what is stored: enable a method once a fit
        of it exists for a ticked run, say so when the fits it would draw are stale, and
        tick the methods in `tick` (the ones a Run just produced).

        Signals are blocked over the pass: each programmatic `setChecked` would
        otherwise re-enter `_redraw` through `stateChanged`, once per box."""
        stored = self._stored_fits() if stored is None else stored
        self._syncing_layers = True
        try:
            for _label, key in _FIT_METHODS:
                cb = self.layer_checks[key]
                have = [v for (_i, k), v in stored.items() if k == key]
                cb.setEnabled(bool(have))
                # "(stale)" when EVERY stored fit of this method is stale: with one
                # fresh fit among them the tick still draws something, and the label
                # would be claiming otherwise.
                stale = bool(have) and all(v['stale'] for v in have)
                cb.setText(_fit_label(key) + (' (stale)' if stale else ''))
                if key in tick and have:
                    cb.setChecked(True)
        finally:
            self._syncing_layers = False

    @QtCore.Slot()
    def _on_layer_toggled(self) -> None:
        if self._syncing_layers:
            return
        self._redraw()

    @QtCore.Slot()
    def on_fits_changed(self, methods) -> None:
        """The Distribution tab finished a Run, or its ticks changed (it keeps every
        fit, so `stored_fits` is re-read either way): re-offer its methods here and tick
        the ones a Run just produced.

        Routed by the owning `DLSModule`, never by that widget reaching into this one
        (invariant 5's spirit at the widget layer): it owns both tabs and connects the
        signal to this slot.

        `_reload_raw` FIRST, as `set_measurement` and `_on_selection_changed` both do:
        removing a ticked run prunes the shared selection, and the Distribution tab --
        built first, so connected to `SelectionModel.changed` first -- emits from inside
        that prune, while `self._raw` still holds the deleted id. Redrawing then raised
        `KeyError` on `ws[iid]`; Qt swallowed it at the emit boundary, so the draw
        aborted half-done and only a console traceback said so. `_reload_raw` skips ids
        the workspace no longer has.
        """
        self._reload_raw()
        self._sync_layers(tick=set(methods or ()))
        self._redraw()
        # ... and the status, which is where a DRAWN-but-failed fit says so. Without
        # this the note only appeared after some other gesture refreshed the status,
        # i.e. never, for the one path that brings a failed lognormal to this tab.
        self._update_status()

    def _fit_kwargs(self, iid: str, key: Optional[str] = None) -> Dict:
        """What this method's fit of `iid` reads from the shared region -- the one
        reading (`_region_fit_kwargs`): the window and the baseline only once the user has
        set each, measured inside the window, as the one baseline input this method
        reads. `key` defaults to the method the combo is showing, so the staleness checks
        and the Run ask the same question.
        """
        return _region_fit_kwargs(self.controller, self.region, iid,
                                  key or self.method_combo.currentData())

    def _run_signature(self, iid: str, key: str) -> tuple:
        """Everything a parametric fit of `iid` reads beyond its raw correlogram: the
        measurement's committed params, the method, the cumulant order (cumulant only),
        the shared delay window and this measurement's baseline over the shared
        region, and the settings the engine seeds (skip_initial_channels; the cumulant
        method, fit cutoff and weighting, cumulant only). Stored with each cached fit and compared on every
        redisplay and region edit, so a shown fit is flagged stale exactly when a
        re-Run would read different inputs -- a dragged baseline handle included.

        The region's APPLIED parts ride here in their own slot as well as through the
        kwargs they gate, so that pressing Reset beside a shown fit flags it stale exactly
        as a drag does even in the corner where the kwargs happen not to move (a span that
        selects no channel on this correlogram emits no `baseline` either way)."""
        s = self.controller.settings
        cumulant = key == 'cumulant'
        return (self.controller.committed_signature([iid]), key,
                self.order_spin.value() if cumulant else None,
                tuple(sorted(self._fit_kwargs(iid, key).items())),
                self.region.state().applied(),
                s.skip_initial_channels, s.cumulant_method if cumulant else None,
                # Settings-seeded, so these ride here rather than beside the plot -- the
                # reason such a value needs no per-run widget to flag a shown fit stale.
                # `if cumulant` for the same reason `cumulant_method` is: the exponential
                # arms take the ENGINE default cutoff and weighting for their seeds, and
                # do not move when these settings do.
                s.cumulant_fit_cutoff if cumulant else None,
                s.cumulant_weighting if cumulant else None)

    def _fit_for(self, iid: str, key: Optional[str] = None):
        """This tab's stored result for `(iid, key)`, `key` defaulting to the method the
        combo is showing -- the tables and the status line, which follow the Method box.
        None when there is none. What the PLOT draws is `_shown_fits`, which follows the
        layers list and also reaches the Distribution tab's results."""
        v = self._cache.get((iid, key or self.method_combo.currentData()))
        return v['result'] if v else None

    # ---- actions ----
    def _on_fit_input_moved(self) -> None:
        """A fit INPUT moved beside a shown fit: re-label the layers list (which marks
        the now-stale entries) and refresh the status.

        One spelling of the response, because it was previously open-coded and four of
        the nine handlers that move an input had only half of it -- the curve was
        dropped from the plot while its checkbox still read as live."""
        self._sync_layers()
        self._redraw()
        self._update_status()

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
        key = self.method_combo.currentData()
        c = self.controller
        # Cleared WHOLESALE, as before: a failure record carries no signature, so a
        # kept one cannot say whether the input that caused it has since been fixed.
        # The defect this round fixed was the READ key, not the lifetime -- keeping
        # per-method records would need per-method staleness to stay honest.
        self._run_failures = {}
        for iid in ids:
            try:
                kw = self._fit_kwargs(iid, key)   # window + region, per method
                if key == 'cumulant':
                    res = c.run_cumulants(iid, order=self.order_spin.value(), **kw)
                elif key == 'single':
                    res = c.run_single_exponential(iid, **kw)
                elif key == 'double':
                    res = c.run_double_exponential(iid, **kw)
                else:
                    res = c.run_kww(iid, **kw)
                self._cache[(iid, key)] = {'result': res,
                                           'sig': self._run_signature(iid, key)}
            except Exception as exc:  # noqa: BLE001 — per-measurement, failure recorded; per-measurement; never abort all
                self._run_failures[(iid, key)] = str(exc)
                # Only THIS method's entry: a failed cumulant Run must not silently
                # drop the CONTIN reconstruction the user has ticked beside it.
                self._cache.pop((iid, key), None)
        self.export_button.setEnabled(self._any_cached())
        # A fresh Run ticks its own method, so the fit just computed is on screen
        # without a second gesture.
        self._sync_layers(tick={key})
        self._redraw()
        self._refresh_tables()
        self._update_status()

    @QtCore.Slot()
    def _on_export(self) -> None:
        key = self.method_combo.currentData()
        cached = [(iid, self._cache[(iid, key)]['result'])
                  for iid in self.selection.ids() if (iid, key) in self._cache]
        if not cached:
            self.status.setText('Run a fit first.')
            return

        name, suffixes = _selection_export_names(
            self.controller, [(iid, key) for iid, _res in cached], 'correlogram')

        def do_export(path: str):
            import os
            stem, ext = os.path.splitext(path)
            paths = [f'{stem}{sfx}{ext}' for sfx in suffixes]
            siblings = [] if len(cached) == 1 else paths
            if not confirm_overwrite(self, siblings):
                return None
            written = []
            for (iid, res), p in zip(cached, paths, strict=True):
                written.append(self.controller.export_correlogram_fit(iid, res, p))
            return written                     # summarised by `export_to_csv`

        status = export_to_csv(self, self.controller, name, do_export)
        if status:
            self.status.setText(status)

    # ---- region fields <-> shared region (seconds) ----
    _HANDLES = [('tau_min', 'tau_min_s'), ('tau_max', 'tau_max_s'),
                ('base_lo', 'base_lo_s'), ('base_hi', 'base_hi_s')]

    # Significant figures the four handle fields display. Named because the round-trip
    # BACK through it decides whether a field was edited (see `_displayed_handle`), so the
    # writer and the comparison must use one number.
    _HANDLE_SIG = 4

    def _fields_from_region(self) -> None:
        unit = self.tau_unit.currentText()
        self._suppress_fields = True
        for edit, (_, attr) in zip(
                (self.tau_min, self.tau_max, self.base_lo, self.base_hi),
                self._HANDLES, strict=True):
            x = getattr(self.region, attr)
            edit.setText('' if x is None
                         else f'{U.from_canonical("time", x, unit):.{self._HANDLE_SIG}g}')
        self._suppress_fields = False

    def _field_was_edited(self, edit, attr: str, unit: str, value) -> bool:
        """Did the user actually edit THIS field, or is it just being read back?

        `editingFinished` fires on one widget but the handler parses all four, and the
        fields render to `_HANDLE_SIG` significant figures, so parsing them all back does
        not round-trip: a stored 1.234567e-4 comes back as 1.235e-4. Writing that back
        silently truncated three handles the user never touched -- and, once the baseline
        region gained a touched state, marked it PLACED when the user had typed a
        delay-window bound.

        TWO independent tests, OR-ed, because neither alone covers the field:

        - Qt's own `isModified()`, which is True exactly when a human changed the text
          and False after any programmatic `setText`. This is the one signal that can
          tell "the user retyped the digits already shown" -- a real confirm-this-region
          gesture -- from "this box was never touched"; a value comparison cannot, since
          the two produce identical text.
        - the value comparison, for the programmatic path: a caller that sets the text
          and calls the handler (the headless tests, and any future code that drives the
          fields) leaves `isModified()` False, and would otherwise write nothing at all.
        """
        if edit is not None and edit.isModified():
            return True
        return value != self._displayed_handle(attr, unit)

    def _displayed_handle(self, attr: str, unit: str):
        """What `attr`'s field is currently SHOWING, read back in canonical seconds."""
        x = getattr(self.region, attr)
        if x is None:
            return None
        shown = f'{U.from_canonical("time", x, unit):.{self._HANDLE_SIG}g}'
        try:
            return U.to_canonical('time', float(shown), unit)
        except ValueError:
            return None

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
        for edit, (_name, attr) in zip(
                (self.tau_min, self.tau_max, self.base_lo, self.base_hi),
                self._HANDLES, strict=True):
            v = vals[attr]
            if self._field_was_edited(edit, attr, unit, v):
                self.region.set_handle(attr, v)  # marks a MOVED handle's kind touched
            # Consumed either way: a modified flag left standing would make the NEXT
            # field's edit rewrite this handle at display precision, which is the
            # truncation the comparison above exists to prevent.
            edit.setModified(False)
        self._redraw()
        self._sync_layers()          # ... and the layers list says which are stale
        self._update_status()        # the window is a fit input: a shown fit may be stale
        self.regionChanged.emit()

    def _on_reset_region(self) -> None:
        """Restore the window + baseline to the from-data defaults."""
        self.region.force_reseed()
        self._reload_raw()           # re-seeds the region from the union lag range
        self._sync_layers()          # the reset staled every shown fit; say so
        self._redraw()
        self._update_status()
        self.regionChanged.emit()

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
        except Exception:  # noqa: BLE001 — transform guard, no handle
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
            except Exception:  # noqa: BLE001 — transform guard, handle skipped
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
        self.region.set_handle(attr, x)         # marks a MOVED handle's kind touched
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
        self._sync_layers()                      # ... and the layers list says so
        self._redraw()                           # recreate markers + baseline span
        self._update_status()                    # a dragged marker moved a fit input
        self.regionChanged.emit()

    # Marker styling + offset-handle geometry: window carets at the
    # top, baseline carets at the bottom (the disambiguation described above).
    # _BASE_HANDLE_Y is kept a touch above the bottom so the caret never lands in
    # the residual-resize gap band below main_ax.
    _WIN_COLOR, _BASE_COLOR = '#2ca02c', '#7e57c2'   # window green, baseline purple
    # The part of the baseline span outside the delay window, which no fit reads: a
    # neutral grey, hatched so it stays distinct from the purple fill in both themes.
    _OUTSIDE_COLOR = PLACEHOLDER_GRAY
    _WIN_HANDLE_Y, _BASE_HANDLE_Y = 0.96, 0.05

    def _draw_markers(self) -> None:
        """Draw the window (dotted, top carets) + baseline (dashed + shaded, bottom
        carets) markers on main_ax, each a full-height line plus an offset grab handle."""
        self._markers = {}
        self._handle_glyphs = {}
        self._base_span = None
        self._base_span_outside = []
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
            # NOTHING OUTSIDE THE DELAY WINDOW COUNTS: a fit measures the span only over
            # its part inside the window (`app/analysis_region.py`), so the part outside
            # is drawn grey and the purple fill stops at the window edge. The handles
            # still move freely; this only says what the fits will read. The window drawn
            # is the handles' (an untouched window follows the ticked data's range).
            wlo = r.tau_min_s * tfac if r.tau_min_s is not None else lo
            whi = r.tau_max_s * tfac if r.tau_max_s is not None else hi
            in_lo, in_hi = max(lo, wlo), min(hi, whi)
            self._base_span = None
            if in_lo < in_hi:
                self._base_span = self.main_ax.axvspan(
                    in_lo, in_hi, color=self._BASE_COLOR, alpha=0.20,
                    label='baseline region')
            self._base_span_outside = [
                self.main_ax.axvspan(a, b, color=self._OUTSIDE_COLOR, alpha=0.25,
                                     hatch='//', linewidth=0, zorder=0.5)
                for a, b in ((lo, min(hi, wlo)), (max(lo, whi), hi)) if a < b]
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
            self.figure.legends.clear()   # outlives ax.clear(); would float alone
            set_plot_export_dressing(self.canvas, None)   # ... and so does this
            self._placeholder = show_plot_placeholder(
                self.figure, self.canvas, 'Tick a measurement to plot',
                axes=(self.main_ax, self.resid_ax, *self.side_axes))
            return
        for ax in (self.main_ax, self.resid_ax, *self.side_axes):
            ax.set_visible(True)
        # Plot saves suggest the same stem as the CSV export for the active method.
        set_plot_export_name(
            self.canvas,
            _selection_export_names(
                self.controller,
                [(iid, self.method_combo.currentData()) for iid in self._raw],
                'correlogram')[0])
        ws = self.controller.workspace.measurements
        xs, ys = _SCALE_XY[self._scales[0]]
        x_clamp = self._linear_x_clamp_display()   # frames linear-τ views on the decay
        # A legend has no sample header above it, so it is one of the two surfaces that
        # prefixes the sample -- and only when the ticked set spans samples.
        prefixes = _sample_prefixes(self.controller, self._raw)
        # Everything drawn on this pass: one entry per (ticked measurement x ticked,
        # non-stale method). Computed ONCE and grouped by measurement, so the main view,
        # the three side views and the residual panel below all iterate the same list
        # and cannot disagree about which curve belongs to which fit.
        shown = self._shown_fits()
        fits_by_meas: Dict[str, List[Tuple[str, object]]] = {}
        for iid, key, res in shown:
            fits_by_meas.setdefault(iid, []).append((key, res))
        # main view: every ticked measurement's data ONCE, then each of its ticked fits.
        # Colour and marker are the MEASUREMENT (item 3); line style is the METHOD.
        for iid, (tau, g2) in self._raw.items():
            col = self.selection.color_for(iid)
            plot_correlogram_scaled(tau, g2, ax=self.main_ax, xscale=xs,
                                    yscale=ys, color=col,
                                    label=measurement_display_name(
                                        ws[iid], sample=prefixes[iid]),
                                    marker=self.selection.marker_for(iid))
            for key, res in fits_by_meas.get(iid, ()):
                plot_correlogram_fit_line(res.fit_tau_s, res.fitted_g2m1, ax=self.main_ax,
                                     xscale=xs, yscale=ys, color=col,
                                     linestyle=_METHOD_LINESTYLE[key])
        self.main_ax.set_title(f'Correlogram — {self._scales[0]}')
        self._draw_markers()                     # window + baseline handles
        if xs == 'linear':
            # Pin a linear delay axis to 0 (negative tau is meaningless) and frame the
            # right to the decay region so it sits centered, not crushed at the origin.
            # Done before the residual copies the x-lim below.
            right = x_clamp if x_clamp is not None else self.main_ax.get_xlim()[1]
            self.main_ax.set_xlim(0.0, right)
        self._draw_legend(shown)
        # residual panel: the residuals of exactly the fits drawn above, in the same
        # colour and line style, through the one helper the Distribution tab's panel
        # also uses. Drawn in the same DISPLAY units as the main axis so it lines up
        # under the fit -- it once drew raw seconds on a microsecond axis.
        draw_fit_residuals(
            self.resid_ax,
            [(self.selection.color_for(iid), key, res) for iid, key, res in shown],
            xscale=xs)
        self.resid_ax.set_xlim(self.main_ax.get_xlim())
        _hide_shared_x(self.main_ax)         # the residual carries the only x axis
        # side views: overlay all measurements at each scale
        for k, sax in enumerate(self.side_axes):
            name = self._scales[k + 1]
            sxs, sys = _SCALE_XY[name]
            for iid, (tau, g2) in self._raw.items():
                col = self.selection.color_for(iid)
                plot_correlogram_scaled(tau, g2, ax=sax, xscale=sxs, yscale=sys,
                                        compact=True, color=col,
                                        marker=self.selection.marker_for(iid))
                for key, res in fits_by_meas.get(iid, ()):
                    plot_correlogram_fit_line(res.fit_tau_s, res.fitted_g2m1, ax=sax,
                                         xscale=sxs, yscale=sys, color=col,
                                         linestyle=_METHOD_LINESTYLE[key],
                                         compact=True)
            if sxs == 'linear':                   # pin to 0; frame on the decay
                right = x_clamp if x_clamp is not None else sax.get_xlim()[1]
                sax.set_xlim(0.0, right)
            sax.set_title(name, fontsize=8)
        self.canvas.draw_idle()
        self._floor_resid_ratio()            # after the layout is real, not before

    def _draw_legend(self, shown) -> None:
        """The on-screen KEY -- one entry per METHOD drawn plus the window/baseline
        markers -- and the dressing that gives the SAVED figure the full legend.

        Not one entry per curve, and no longer one per measurement either. Before B8
        the data scatter and its fit line were each labelled, so nine ticked runs drew
        eighteen entries, clipped at eleven, covering the plot area. Fixing the count
        left thirteen full-length entries that no placement could carry: below the
        panel they took 154 px and left the residual 19 px of a 477 px canvas.

        The measurement names now go to the file, where a reader has no picker or
        glance table beside them; on screen those two already name every ticked run
        against its swatch. What stays here is `_measurement_legend_entries`' worth of
        nothing and a key that does not grow with the selection.

        The key sits in the figure's EMPTY bottom-right gridspec cell -- `[3, 1]`, the
        square under the side-view stack and beside the residual, which nothing has
        ever drawn in -- so it costs the plot no height at all. Measured on the nine-run
        frame: residual 19.0 -> 59.8 px, fit 154.6 -> 299.6 px. When the user drags the
        residual small enough that the cell cannot hold the key, it falls back to
        sitting below the panel.
        """
        ws = self.controller.workspace.measurements
        methods = list(dict.fromkeys(key for _i, key, _r in shown))
        key_entries = _method_legend_entries(methods) + _marker_legend_entries(
            self.main_ax)
        # A fit whose own `success` is False drew a SEED estimate, not a fit, and the
        # saved image must not present it as fitted (module docstring, owner-ruled).
        # This tab draws the Distribution tab's stored fits now, so it inherits that
        # duty; `_shown_fits` hands back the result objects, which carry `success`.
        failed = {}
        for iid, key, res in shown:
            if not getattr(res, 'success', True):
                failed.setdefault(iid, []).append(_fit_label(key))
        leg = _draw_key(self.figure, key_entries, bbox=self._key_cell())
        if leg is not None and not self._key_fits(leg):
            leg = _draw_key(self.figure, key_entries)      # below the panel instead

        # The saved figure additionally names every measurement, grouped by sample.
        prefixes = _sample_prefixes(self.controller, self._raw)
        by_sample = {}
        for iid in self._raw:
            sample = prefixes[iid] or _sample_label_of(self.controller, iid)
            note = failed.get(iid)
            by_sample.setdefault(sample, []).append(
                (iid, measurement_display_name(ws[iid])
                 + (f' ({", ".join(note)} failed)' if note else '')))
        meas_entries, groups = _measurement_legend_entries(self.selection, by_sample)
        set_plot_export_dressing(
            self.canvas,
            export_legend_dressing(self.figure, meas_entries + key_entries, groups)
            if meas_entries else None)

    def _key_cell(self):
        """The region the on-screen key sits in: the SIDE-VIEW column's x-range, at the
        `[3, 1]` cell's height -- or None before the first draw has positioned it.

        The x-range comes from the drawn side views, not from the gridspec cell. The
        cell is empty by construction, but with nothing to its right the residual axes
        SPILLS into it: measured 1015 px against a cell starting at 1003 px at
        1600x1020, and 1372 against 1351 at 1920x1080, so a key anchored at the cell's
        left edge printed over the residual plot (owner report, 2026-09-19). It did not
        at 1280x760, which is why the shipped screenshots did not show it. The side
        views never spill -- they have the residual's own row beneath them -- and they
        are the column the key visually belongs to.
        """
        try:
            cell = self._gs[3, 1].get_position(self.figure)
            side = self.side_axes[-1].get_position()
        except Exception:  # noqa: BLE001 — no layout yet: fall back to below the panel
            return None
        from matplotlib.transforms import Bbox
        return Bbox([[side.x0, cell.y0], [side.x1, cell.y1]])

    def _key_fits(self, leg) -> bool:
        """Whether `leg` is inside its cell. The cell shrinks with the residual's
        share, which the user can drag to a minimum, so this is a real branch."""
        cell = self._key_cell()
        if cell is None:
            return False
        try:
            ext = leg.get_window_extent(self.canvas.get_renderer())
        except Exception:  # noqa: BLE001 — unmeasurable: take the safe placement
            return False
        w = self.figure.get_figwidth() * self.figure.dpi
        h = self.figure.get_figheight() * self.figure.dpi
        return ext.width <= cell.width * w + 1 and ext.height <= cell.height * h + 1


    # ---- tables (a GLANCE row per ticked run + a panel for the focused one) ----
    def _refresh_tables(self) -> None:
        self._rebuild_tables()
        self._fill_tables()

    def _focused_id(self):
        """The run the detail panel describes: the sidebar's focused measurement when it
        is ticked, else the first ticked run.

        A panel that blanked whenever focus sat on an unticked run would withhold the
        detail quantities exactly when several runs are co-plotted -- which is when they
        are wanted -- so it falls back rather than emptying."""
        ids = [i for i in self.selection.ids()
               if i in self.controller.workspace.measurements]
        if self.item_id in ids:
            return self.item_id
        return ids[0] if ids else None

    def _rebuild_tables(self) -> None:
        ws = self.controller.workspace.measurements
        ids = [i for i in self.selection.ids() if i in ws]
        key = self.method_combo.currentData()
        method_label = {k: l for l, k in _CORR_METHODS}.get(key, 'Method')
        # "apparent" is said ONCE here instead of on every headline column: the glance
        # table is only as readable as its narrowest column, and the qualifier is a
        # property of the whole table (invariant 7 -- single-angle results are apparent).
        self.results_label.setText(f'{method_label} results — apparent sizes')

        # `.get`, not a subscript: a method listed in `_CORR_METHODS` with no entry
        # here is a KeyError on every draw of the tab. The roster gate below makes that
        # state a red test instead, and this keeps it from being a crash meanwhile.
        cols = _GLANCE_COLS.get(key, [])
        g = self.glance_table
        g.clear()
        g.setColumnCount(1 + len(cols))
        g.setRowCount(len(ids))
        g.setHorizontalHeaderLabels(['Measurement', *[h for h, _idx in cols]])
        for r, iid in enumerate(ids):
            lm = ws[iid]
            name = QtWidgets.QTableWidgetItem(measurement_display_name(lm))
            # Colour AND marker shape, so a row means the same thing as its curve for a
            # reader who cannot use the colour (`marker_swatch`, WCAG 1.4.1).
            name.setIcon(marker_swatch(self.selection.color_for(iid),
                                       self.selection.marker_for(iid)))
            name.setToolTip(measurement_hover_card(lm))
            g.setItem(r, 0, name)
            for c in range(1, 1 + len(cols)):
                g.setItem(r, c, QtWidgets.QTableWidgetItem('—'))
        g.resizeColumnsToContents()
        # Cap the NAME column and give the rest to the values. Uncapped, one long file
        # stem takes the whole 360 px control column and pushes the headline values out
        # -- the defect this table exists to fix. The widget elides left, so the capped
        # name keeps the tail that tells replicates of one set apart.
        if g.columnWidth(0) > NAME_COL_MAX_W:
            g.setColumnWidth(0, NAME_COL_MAX_W)
        for c in range(1, 1 + len(cols)):
            g.horizontalHeader().setSectionResizeMode(
                c, QtWidgets.QHeaderView.ResizeMode.Stretch)
        # Height capped at `_GLANCE_VISIBLE_ROWS`, NOT grown to the row count. Grown,
        # nine ticked runs asked for ~320 px of a 330 px section and pushed the
        # focused-run panel, the export button and its own rows out of the pane -- the
        # user saw a header row and nothing under it. Beyond the cap it scrolls.
        row_h = g.verticalHeader().defaultSectionSize()
        chrome = g.horizontalHeader().height() + 2 * g.frameWidth() + 2
        shown = min(max(len(ids), 1), _GLANCE_VISIBLE_ROWS)
        g.setMaximumHeight(chrome + row_h * shown)
        # ... and a floor, so a squeezed section cannot shrink it to its header alone.
        g.setMinimumHeight(chrome + row_h * min(max(len(ids), 1), _GLANCE_FLOOR_ROWS))

        # Everything that is not a headline stays on this tab, for the focused run.
        _repoint_vtable(self.result_table, _RESULT_ROWS.get(key, []))

    def _fill_tables(self) -> None:
        key = self.method_combo.currentData()
        ws = self.controller.workspace.measurements
        ids = [i for i in self.selection.ids() if i in ws]
        sig = self._no_unc_sig()
        cols = _GLANCE_COLS.get(key, [])
        for r, iid in enumerate(ids):
            res = self._fit_for(iid)
            if res is None:
                continue
            # ONE computation feeds both views: the glance cells are indices into the
            # same `_result_values` list the panel below renders in full, so the two can
            # never print different numbers for one run.
            values = self._result_values(key, res, sig)
            notes = self._result_notes(key, res)
            for c, (_hdr, idx) in enumerate(cols, start=1):
                it = self.glance_table.item(r, c)
                if it is None:
                    continue
                it.setText(values[idx])
                # A flagged value carries its ⚠ on its OWN cell -- the mark sits on the
                # value it judges, never on the name -- and the reason on hover. With
                # nothing to explain the tooltip is the cell's own text, which is what
                # keeps a value recoverable when a narrow column elides it.
                it.setToolTip(notes[idx] or values[idx])

        focus = self._focused_id()
        if focus is None:
            self.focus_label.setText('')
            self.focus_label.setToolTip('')
        else:
            self.focus_label.setText(
                f'Details — {measurement_display_name(ws[focus])}')
            self.focus_label.setToolTip(measurement_hover_card(ws[focus]))
        res = self._fit_for(focus) if focus is not None else None
        if res is None:
            _reset_vtable(self.result_table)
            return
        _fill_vtable(self.result_table, self._result_values(key, res, sig))
        for r, note in enumerate(self._result_notes(key, res)):
            it = self.result_table.item(r, 1)
            if it is not None:
                it.setToolTip(note or it.text())

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
            baseline_lbl = q(res.baseline)   # fitted B (nonlinear) or the pinned offset

            # One rule for every order-gated moment cell (PDI/μ₂ at order 2, μ₃ at 3, μ₄
            # at 4): below the order that fits the quantity it is UNMEASURED and says so,
            # so a blank is never read as a fitted zero; at or above it the value renders
            # through q(), which already turns a failed fit's None or NaN into 'n/a'.
            def moment_cell(value, needed_order):
                if res.order < needed_order:
                    return f'unmeasured (order {res.order})'
                return q(value)

            # PDI cell: value + reliability glyph when measured; 'unmeasured (order 1)'
            # for an order-1 fit (no mu2 term), 'n/a' for a failed one.
            if math.isfinite(res.pdi):
                # The printed ceiling is the engine's own constant, never a literal, so
                # the glyph cannot go stale against the flag it explains.
                pdi_glyph = ('✓' if res.pdi_valid else
                             ('⚠<0' if res.pdi < 0
                              else f'⚠>{CUMULANT_PDI_VALIDITY_LIMIT:g}'))
                pdi_cell = f'{q(res.pdi)} {pdi_glyph}'
            else:
                pdi_cell = moment_cell(res.pdi, 2)
            # Rh cell: the same glyph convention as its neighbour, for the same reason.
            # The PDI cell carried ✓/⚠ for a long time while the Rh cell beside it stayed
            # bare, so the one judgement that is ABOUT the size was the one the size column
            # did not show. Same tab, same result object, same tier (ui_style_guide R5.2:
            # one condition, one tier, across surfaces); the full sentence is already in
            # flag_label two inches below, so no tooltip is owed here.
            rh_cell = q(res.rh_nm)
            if not getattr(res, 'rh_reliable', True):
                rh_cell = f'{rh_cell} ⚠'

            # μ₃ / μ₄ cells: the BARE value, no validity glyph -- the ruling and its reason
            # are on `CumulantResult`; the caution is the row label's tooltip.
            mu3_cell = moment_cell(res.mu3_s_inv3, 3)
            mu4_cell = moment_cell(res.mu4_s_inv4, 4)
            # mu2 through the SAME order gate as its siblings. It was the one moment
            # rendered raw, and below order 2 the engine returns 0.0 rather than None, so
            # the panel printed a bare '0' for a moment it never estimated -- while the
            # PDI cell directly above said "unmeasured (order 1)" about the same quantity.
            return [q(res.gamma_s_inv), rh_cell, pdi_cell,
                    moment_cell(res.mu2_s_inv2, 2), mu3_cell, mu4_cell,
                    str(res.order), method_lbl, baseline_lbl]
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
                    q(res.beta), q(res.baseline), 'yes' if res.success else 'no']
        if key == 'double':
            return [mode_rh(res.mode1), q(res.mode1.amplitude_fraction),
                    mode_rh(res.mode2), q(res.mode2.amplitude_fraction),
                    q(res.baseline), 'yes' if res.success else 'no']
        return [q(res.stretch), q(res.tau_c_s), q(res.mean_tau_s),
                rh(res.rh_from_tau_c_nm), rh(res.rh_from_mean_tau_nm),
                q(res.baseline), 'yes' if res.success else 'no']

    @staticmethod
    def _result_notes(key: str, res) -> List[str]:
        """Why each flagged row of `_result_values` is flagged, '' where it is not.

        Index-aligned with `_result_values` and so with `_RESULT_ROWS[key]`, because the
        glance table addresses both by the same index -- the mark and its reason must
        land on one cell. The text is the ENGINE's own `reliability_note`, never a
        restatement: the sentence has one home (`analysis/dls/`), and a static copy in
        the GUI is the drift this project has been bitten by before.

        A flag answers 'did this method's preconditions hold for this fit' (invariant 4),
        so the note is attached to the value the judgement is ABOUT -- per MODE for
        single/double, whose modes are judged independently, and to the result for the
        cumulant and KWW, which have one verdict. Rows with an inline, self-describing
        glyph (the PDI cell prints its own '>0.3' ceiling) take no note: there is nothing
        a tooltip would add that the cell does not already say.
        """
        notes = [''] * len(_RESULT_ROWS[key])

        def mode_note(mode) -> str:
            return ('' if getattr(mode, 'rh_reliable', True)
                    else getattr(mode, 'reliability_note', '') or '')

        def result_note() -> str:
            return ('' if getattr(res, 'rh_reliable', True)
                    else getattr(res, 'reliability_note', '') or '')

        if key == 'cumulant':
            notes[1] = result_note()
        elif key == 'single':
            notes[1] = mode_note(res.mode)
        elif key == 'double':
            notes[0] = mode_note(res.mode1)
            notes[2] = mode_note(res.mode2)
        else:                                    # kww: one verdict, two derived sizes
            notes[3] = notes[4] = result_note()
        return notes

    # ---- status / flags ----
    def _update_status(self) -> None:
        key = self.method_combo.currentData()
        ws = self.controller.workspace.measurements
        # (measurement name, message) PAIRS, not pre-joined strings: the grouping needs
        # the two apart so one KIND of message can be explained once with each run named
        # beside its own value. That is item 7's fix -- the commonest message carries its
        # own number, so three replicates printed the same paragraph three times.
        flags = []
        skipped = []          # capped differently -- see the join below
        passive = []
        for iid in self.selection.ids():
            lm = ws.get(iid)
            if lm is None:
                continue
            # The DISPLAYED method's failure, never another's: the status area speaks
            # for the numbers the tables are showing.
            failure = self._run_failures.get((iid, key))
            if failure is not None:
                skipped.append((measurement_display_name(lm),
                                f'⚠ skipped — {humanize_field_names(failure)}'))
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
                    flags.append((measurement_display_name(lm), line))
            # The result's own passive `notes` tier, measurement-prefixed for the same
            # reason the flags are: several are co-plotted and an unlabelled second line
            # would read as the next measurement's.
            for line in getattr(res, 'notes', ()) or ():
                if str(line).strip():
                    passive.append((measurement_display_name(lm), str(line)))
        # Grouped WITHIN a tier, never across one: a flag and a passive note are
        # different claims about the fit and must not share a sentence (R5.1).
        # No cap on the passive tier: its notes are rendered nowhere else.
        set_flag(self.notes_label, '\n'.join(group_status_lines(passive)),
                 problem=False)
        notes = []
        if self.controller.is_dirty() and self._any_cached():
            notes.append('(ran on last committed values)')
        if self._any_stale():
            notes.append(_STALE_HINT)
        self.status.setText('  '.join(notes))
        # A DRAWN fit that did not converge is a flag of its own, and on this tab it
        # is the only surface saying so: the layers list draws the Distribution tab's
        # fits too, and a failed lognormal's curve is its seed estimate. No cap -- the
        # message has no varying numeral, so a capped listing would degenerate to names.
        drawn_failures = [
            (measurement_display_name(ws[iid]),
             f'⚠ {_fit_label(key)} fit did not converge; the curve drawn is the '
             'seed estimate, not a fit')
            for iid, key, res in self._shown_fits()
            if iid in ws and not getattr(res, 'success', True)]
        flags.extend(drawn_failures)
        # The cap applies only to the flags whose premise it rests on: a value-bearing
        # flag is also on its own glance-table cell, one hover from its reason, so an
        # elided run stays recoverable. A SKIPPED run has no stored fit, so its glance
        # row is '—' with no tooltip -- eliding it names it nowhere, the half of round
        # 2's finding 5 the cap was meant to close. Those go out uncapped.
        self.flag_label.setText('\n'.join(
            group_status_lines(skipped)
            + group_status_lines(flags, value_cap=_STATUS_LIST_MAX)))

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
    the Correlogram tab.

    Colour and point MARKER are the measurement, line STYLE is the method -- the same
    coding as the Correlogram tab, and the reason NNLS and CONTIN can now be told apart
    here (they were drawn in one colour and one solid line). This tab OWNS its three
    fits; the Correlogram tab reads them through `stored_fits`, via the module that owns
    both, and is told they moved by `fitsChanged`.

    Each fit is kept in its own (measurement, method) slot with its own signature, the
    Correlogram tab's `_cache` shape: a fit of one measurement reads nothing of the
    ticked set, so the ticks and the method boxes choose which stored fits are SHOWN
    and never discard one. A fit whose inputs have moved stays in its slot, is not
    drawn, and its run reads "press Run" (`_shown`)."""

    #: Emitted whenever this tab's stored fits change, carrying the method keys a Run
    #: just produced (empty on a tick change, so the layers list re-reads which fits the
    #: new ticked set can draw). The owning `DLSModule` routes it to the Correlogram
    #: tab's layers list, so a CONTIN Run here offers -- and ticks -- CONTIN over there
    #: without one widget reaching into the other.
    fitsChanged = QtCore.Signal(list)

    def __init__(self, controller, region, selection, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.region = region                    # shared _AnalysisRegion
        self.selection = selection              # shared _OverlaySelection
        self.item_id: Optional[str] = None
        self._runnable = False
        # (iid, method) -> {'result', 'sig'}: every fit this tab has run, each stamped
        # with `_fit_signature` at dispatch. What is drawn is `_shown()`, never this.
        self._fits: Dict[Tuple[str, str], Dict] = {}
        # (iid, method) -> the message naming why that pair failed in the LAST Run.
        # Replaced wholesale by each Run, as the Correlogram tab's `_run_failures`: a
        # failure carries no signature, so a kept one could not say whether its cause
        # has since been fixed. Only the ticked pairs' failures are shown.
        self._failures: Dict[Tuple[str, str], str] = {}
        # Per-draw reweighting cache: [(iid, method, WeightedDistributionResult or None)],
        # parallel to the fits the last _draw showed. None means intensity (no
        # reweighting) or a reweight error (the flag names it). Rebuilt every _draw;
        # used by export + the peak table.
        self._weighted: List[Tuple] = []
        self._reweight_errors: List[str] = []
        self._placeholder = None                # empty-state Text artist (no distribution)
        # Bumped by a tick change and read by a Run's `fail` ONLY, so a whole-Run
        # failure landing after the ticks moved never paints the status of the view now
        # shown. `done` needs no such guard: a finished fit lands in its own slot.
        self._run_epoch = 0
        self._build_ui()

    def _build_ui(self) -> None:
        # Distribution has the longest form labels ('CONTIN α selection:' etc.), so it
        # gets a slightly wider default + minimum than the other DLS tabs (labels
        # clipped at the old 340 px default). The Rh-grid triple-spinbox row measures
        # 384 px at the shipped size, so 385 keeps its two '?' badges inside the column
        # (they overhung by 10 and 6 px at 360).
        splitter, left, right = make_split_panels(self, left_min_width=385,
                                                  sizes=(403, 717))

        self.checklist = MeasurementPicker(
            self.controller, self.selection, kinds=('dls',),
            label_fn=measurement_display_name, header_fn=sample_display_name,
            tooltip_fn=measurement_hover_card,
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
                                 'These are distribution-weighted, not the cumulant '
                                 'average — compare against the cumulant Rh.',
                                 '<b>Colour and point shape</b> are the measurement; '
                                 '<b>line style</b> is the method, and means the same '
                                 'thing on the Correlogram sub-tab — so one run’s '
                                 'NNLS and CONTIN curves are told apart. The legend '
                                 'shows a method key; the figure <b>Save</b> writes '
                                 'names every measurement, grouped by sample.',
                                 '<b>Keeping fits</b>: each run’s fit is kept, per '
                                 'method. Unticking a run or a method hides its curves; '
                                 'ticking it again shows them without a new Run while '
                                 'nothing it was fitted with has changed. A ticked run '
                                 'whose fit is missing or out of date asks you to press '
                                 'Run.',
                                 '<b>Residual weighting</b>: which noise model the fit '
                                 'measures its error against — the sourced variance '
                                 '(default), or the older g₁² form kept to reproduce '
                                 'distributions reported before this version. The '
                                 'sourced weight fits twice, so a Run takes about twice '
                                 'as long; changing it marks a fit already on screen as '
                                 'out of date.',
                                 '<b>CONTIN α selection</b>: robust GCV (default), the '
                                 'L-curve corner, or the F-test (see the '
                                 'Theory-and-Equations-Guide). '
                                 'For the F-test, a higher “probability to reject” gives '
                                 'a smoother fit, lower gives more detail.',
                                 '<b>GCV robustness γ</b> (shown for GCV): 0.1 is the '
                                 'default, robust GCV; 1 is classical GCV. Robust GCV '
                                 'guards against under-smoothing (extra peaks that come '
                                 'and go between repeats), which classical GCV is prone '
                                 'to because the correlator’s channel noise is '
                                 'correlated. Set 1 to reproduce a classical-GCV result.',
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
                'the DLS standard — use them with care (Theory-and-Equations-Guide §3.7).',
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
        # The fit inputs: the Rh grid, the residual weighting and CONTIN's α selector,
        # range, γ and F-test level -- the shared group the Settings tab and the Mode
        # Map hold too, seeded from Settings and overridable per run. Not
        # `weighting_combo` above, which re-expresses a finished intensity distribution
        # as volume/surface/number: that is a presentation of the answer, this group
        # holds inputs to it.
        self.fit_fields = DistributionFitFields(self)
        self.fit_fields.add_rows(form)
        self.reseed_from_settings()
        note = ThemedLabel('Delay window + baseline region are set on the '
                           'Correlogram tab (shared). Each ticked method runs on '
                           'every ticked measurement.', role='hint', size=10)
        note.setWordWrap(True)
        form.addRow(note)
        # Run is pinned under the scrolled form and Export under the column (the §3
        # column template), so the form's length can never carry them out of view.
        self.run_button = QtWidgets.QPushButton('Run')
        self.run_button.setToolTip(BACKGROUND_RUN_TOOLTIP)
        self.run_button.clicked.connect(self._on_run)
        self.export_button = QtWidgets.QPushButton('Export CSV…')
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._on_export)
        # Peak results for the ticked curves, mirroring the Correlogram tab's glance
        # table: one ROW per (measurement · method), one column per positional peak.
        # Populated in _draw from the shown fits, emptied with the plot.
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

        # Scrolled controls and a stated results floor, for the reason the Correlogram
        # tab's are: this tab passed the box's own `sizeHint` as the pane minimum
        # OUTRIGHT, so the splitter starved the pane below it and the peak table fell
        # off the bottom of the column entirely.
        dist_scroll = QtWidgets.QScrollArea()
        dist_scroll.setWidgetResizable(True)
        dist_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        dist_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        dist_scroll.setWidget(box)
        dist_pane = QtWidgets.QWidget()
        dist_layout = QtWidgets.QVBoxLayout(dist_pane)
        dist_layout.setContentsMargins(0, 0, 0, 0)
        dist_layout.setSpacing(4)
        dist_layout.addWidget(dist_scroll, 1)
        dist_layout.addWidget(self.run_button)
        # Neutral ⓘ note (steel-blue calmer-warnings tier) for the model-dependence of a
        # reweighted view — distinct from the red flag_label, which is for problems.
        # Under the peak table, as on the Correlogram tab.
        self.info_label = ThemedLabel('', size=11)
        self.info_label.setWordWrap(True)
        rlay.addWidget(scrolling_notes(self.info_label))
        vstack = make_vertical_plot_stack(
            [self.checklist, dist_pane, results_section], sizes=[200, 380, 250],
            min_heights=[90, 150, 200], stretch=COLUMN_STACK_STRETCH)
        left.addWidget(vstack, 1)
        self.status = QtWidgets.QLabel('')
        self.status.setWordWrap(True)
        # A red flag label for run-skips, mirroring the Correlogram tab so a skipped
        # measurement uses the same channel everywhere (style guide §5 R5.5) rather than
        # hiding on the muted status line. Set via the shared tiered renderer.
        self.flag_label = ThemedLabel('', role='error', bold=True)
        self.flag_label.setWordWrap(True)
        pin_under_column(splitter, self.export_button, self.status, self.flag_label)

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
        self._resid_dragged = False              # set by the drag handle; see the floor
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
            '• Residuals: data − reconstructed g₂−1 vs delay time, for the '
            'methods ticked on this sub-tab, each in its own line style. Parametric '
            'residuals stay on the Correlogram sub-tab, whose panel follows the same rule.\n'
            '• α-sweep: how CONTIN chose its regularization α — the L-curve corner, '
            'the GCV minimum, or the F-test curve (see the '
            'Theory-and-Equations-Guide). CONTIN results only.')
        self.bottom_combo.currentIndexChanged.connect(self._on_bottom_changed)
        brow.addWidget(self.bottom_combo)
        brow.addWidget(HelpBadge(
            'Read the α-sweep to judge whether to trust CONTIN’s automatic α.',
            bullets=[
                '<b>L-curve</b>: fit residual against the <b>size</b> of the distribution; '
                'a sharp <b>corner</b> is a clean trade-off, a rounded one means α is '
                'poorly determined. Under it the halo also marks <b>no corner</b> (the '
                'sweep held no L-shaped bend; the least-regularized α is reported) and a '
                'one-point sweep.',
                '<b>GCV</b>: a distinct <b>minimum</b> is a confident pick; a flat valley '
                'is ambiguous.',
                '<b>F-test</b>: the curve climbs with α — the pick is the smoothest fit '
                'not significantly worse than the best. On fine Rh grids its reference '
                '<i>is</i> the floor, so it can pin there — the halo says when it did. '
                'Two more notes come from CONTIN’s own checks on the F-test: <b>floor not '
                'small enough</b> (the reference solution is already regularized — lower '
                'the α floor) and <b>transition not bracketed</b> (the chosen solution’s '
                'probability to reject lies outside 0.05–0.95, checked when the requested '
                'level lies inside that band — widen the α range or add sweep points).',
                'A red halo marks an α pinned to a <b>sweep edge</b>: the selector wanted '
                'an α beyond the range. At the ceiling it wanted a larger one — the fit '
                'shown is the smoothest the range holds; raise the α ceiling. At the floor '
                'it wanted a smaller one — at a low floor (the default 1e-4) that is '
                'essentially the unregularized NNLS fit, whose extra peaks may be inversion '
                'noise; lower the floor only if you mean to. The F-test’s two precondition '
                'notes are about the sweep, not about where the pick sits, so they draw no '
                'halo — read them on the flag line beside the plot.',
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
            self.canvas, self.figure, self.ax, self.resid_ax, self._on_resid_dragged)
        self.axis_bar = AxisControlBar(self.canvas)
        right.addWidget(self.axis_bar)
        self._update_weighting_controls_visibility()
        self._clear('Tick measurements + methods, then Run.')
        # Connected last, once the axes they redraw exist: a method box and every fit
        # field choose which stored fits are shown, so each moves the plot at once.
        for cb in self.method_checks.values():
            cb.toggled.connect(self._on_fit_input_moved)
        self.fit_fields.changed.connect(self._on_fit_input_moved)

    def _apply_resid_ratio(self, frac: float) -> None:
        """Resize the residual vs the distribution panel (draggable handle)."""
        self._resid_ratio = frac
        a = (1.0 - frac) / 3.0
        self._gs.set_height_ratios([a, a, a, frac])
        self.canvas.draw_idle()

    def _on_resid_dragged(self, frac: float) -> None:
        """The drag handle moved: apply it, and stop the floor second-guessing it.
        An explicit drag is a decision and outranks `_floor_resid_ratio`."""
        self._resid_dragged = True
        self._apply_resid_ratio(frac)

    def _floor_resid_ratio(self) -> None:
        """Raise the residual's share until its drawn box clears `_RESID_FLOOR_PX`.

        The Correlogram tab's twin, for the same reason and with the same
        drag-outranks-the-floor rule -- both panels are a fixed FRACTION of a varying
        stack while their own chrome is fixed, so both under-serve a short stack.
        """
        if self._resid_dragged:
            return
        for _ in range(12):
            try:
                box = self.resid_ax.get_window_extent(self.canvas.get_renderer())
            except Exception:  # noqa: BLE001 — not laid out yet; nothing to floor
                return
            if box.height >= _RESID_FLOOR_PX or self._resid_ratio >= 0.40:
                return
            self._apply_resid_ratio(min(self._resid_ratio + 0.02, 0.40))
            self.canvas.draw()

    def reseed_from_settings(self) -> None:
        """(Re)seed the fit fields from the global Settings defaults. The per-run values
        the user sets here always win (run_distribution honors explicit kwargs over its
        settings setdefault)."""
        self.fit_fields.seed_from_settings(self.controller.settings)

    def _grid_kwargs(self, method: str) -> Dict:
        """This tab's per-run fit kwargs for `method` -- the shared group's
        `fit_kwargs`. Riding here also puts them in `_fit_signature`, so moving a field
        marks a shown fit out of date."""
        return self.fit_fields.fit_kwargs(method)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.checklist.refresh()
        self.run_button.setEnabled(self._has_checked())
        # The shared region moves on the Correlogram tab with no signal to this one, so
        # the shown subset is re-read here: a fit the move made stale leaves the plot.
        self._draw()

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
        # The shell re-points here after a commit, so the shown subset is re-read: a fit
        # whose committed params moved leaves the plot and its run reads "press Run".
        self._draw()

    def _fit_signature(self, iid: str, key: str) -> tuple:
        """Everything a `key` distribution fit of `iid` reads beyond its raw correlogram:
        the measurement's committed params, the method, the kwargs the Run hands the
        engine (the shared window + this measurement's baseline over it, the Rh grid and,
        for CONTIN, the α range, the selector and its level or γ), and the settings seed
        the engine fills in itself (skip_initial_channels). Stamped on each stored fit at
        dispatch and compared on every redraw, so a fit fitted under a window, grid or
        commit since moved is flagged stale on its own (the two tabs share one region,
        so a Correlogram-tab edit reaches here). The region's applied parts ride in their
        own slot too, as on the Correlogram tab, so Reset flags a shown fit stale even
        where the kwargs happen not to move.

        No term for the ticked set: a fit of one measurement does not read it, and
        `RegionState.applied()` reads only the handles the user has set, so a tick
        change moves no fit's signature."""
        return (self.controller.committed_signature([iid]), key,
                tuple(sorted({**self._baseline_kwargs(iid, key),
                              **self._grid_kwargs(key)}.items())),
                self.controller.settings.skip_initial_channels,
                self.region.state().applied())

    def _fit_is_stale(self, iid: str, key: str) -> bool:
        """Whether the stored fit of `(iid, key)` was computed from inputs that have
        since moved. False when there is no such fit -- absent is not stale."""
        v = self._fits.get((iid, key))
        return v is not None and v['sig'] != self._fit_signature(iid, key)

    def _ticked_pairs(self) -> List[Tuple[str, str]]:
        """Every (measurement, method) pair the ticks ask to see: the ticked DLS
        measurements still in the workspace, in selection order, by the ticked method
        boxes, in `_DIST_METHODS` order."""
        ws = self.controller.workspace.measurements
        ids = [i for i in self.selection.ids() if i in ws and ws[i].kind == 'dls']
        return [(iid, key) for iid in ids for _label, key in self._ticked_methods()]

    def _shown(self) -> List[Tuple[str, str, object]]:
        """`[(iid, method, result)]` for every FRESH stored fit of a ticked pair -- the
        one subset the plot, the peak table, the α-sweep panel, both exports and the
        status and flag lines read, so none of them can speak for a fit the others do
        not show. A stale fit keeps its slot and is not shown."""
        out = []
        for iid, key in self._ticked_pairs():
            v = self._fits.get((iid, key))
            if v is not None and v['sig'] == self._fit_signature(iid, key):
                out.append((iid, key, v['result']))
        return out

    def _unshown_note(self) -> str:
        """The status sentence naming each ticked pair that is not drawn: a stale fit
        takes the stale hint, a missing one asks for a Run. A pair that failed in the
        last Run is left to the red flag, which names it with its reason (the
        Correlogram tab's rule). '' when every ticked pair is shown."""
        ws = self.controller.workspace.measurements
        stale: Dict[str, List[str]] = {}
        missing: Dict[str, List[str]] = {}
        for iid, key in self._ticked_pairs():
            if (iid, key) in self._failures:
                continue
            v = self._fits.get((iid, key))
            if v is not None and not self._fit_is_stale(iid, key):
                continue
            (stale if v is not None else missing).setdefault(
                measurement_display_name(ws[iid]), []).append(_fit_label(key))

        def named(runs: Dict[str, List[str]]) -> str:
            return '; '.join(f'{name} ({", ".join(keys)})' for name, keys in runs.items())

        parts = []
        if stale:
            parts.append(f'{named(stale)}: {_STALE_HINT}')
        if missing:
            parts.append(f'No fit yet for {named(missing)} — press Run.')
        return '   '.join(parts)

    def _ticked_failures(self) -> List[str]:
        """The last Run's failure messages for the pairs ticked now, in tick order."""
        return [self._failures[p] for p in self._ticked_pairs() if p in self._failures]

    @QtCore.Slot()
    def _on_selection_changed(self) -> None:
        """The ticked set changed. Every stored fit is kept -- each is a fit of one
        measurement and reads nothing of the ticked set -- so the plot is redrawn from
        the store: unticking hides a run's curves and ticking it again shows them while
        they are fresh. A redraw only, on the axes built once (`ax.clear()`, never a new
        axes), and the Correlogram tab's layers list is told to re-read `stored_fits`."""
        self._run_epoch += 1
        # Housekeeping, not correctness: an id is never reused within a workspace, so a
        # slot whose measurement was removed can never be shown again.
        ws = self.controller.workspace.measurements
        self._fits = {p: v for p, v in self._fits.items() if p[0] in ws}
        self._failures = {p: m for p, m in self._failures.items() if p[0] in ws}
        self.run_button.setEnabled(self._has_checked())
        self._update_weighting_controls_visibility()   # refresh the RDG solute-RI hint
        self._draw()
        self.fitsChanged.emit([])

    def _on_fit_input_moved(self, *_args) -> None:
        """A method box or a fit field moved: redraw from the store, so a fit whose
        input moved -- or whose method box is unticked -- leaves the plot and each
        ticked run without a current fit reads "press Run". Nothing leaves the store:
        ticking the method again, or moving the field back, shows those fits without a
        Run. The Correlogram tab's `_on_fit_input_moved` shape."""
        self._draw()

    @QtCore.Slot()
    def _on_axis_changed(self) -> None:
        if self._shown():
            self._draw()                          # re-plot the shown fits on the new axis

    @QtCore.Slot()
    def _on_weighting_changed(self) -> None:
        """Weighting or sphere-model changed: update the controls, then re-draw the SHOWN
        intensity fits under the new weighting (a re-presentation — never a re-fit)."""
        self._update_weighting_controls_visibility()
        if self._shown():
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

    def _baseline_kwargs(self, iid: str, method: str) -> Dict:
        """What a `method` inversion of `iid` reads from the shared region -- the one
        reading (`_region_fit_kwargs`): the window once set, and a baseline value only
        once the span is placed, measured over its part inside the window. An untouched
        span sends none, and the engine estimates its own over the window's tail."""
        return _region_fit_kwargs(self.controller, self.region, iid, method)

    @QtCore.Slot()
    def _on_run(self) -> None:
        ws = self.controller.workspace.measurements
        ids = [i for i in self.selection.ids() if i in ws and ws[i].kind == 'dls']
        methods = self._ticked_methods()
        if not ids:
            self.status.setText('Tick at least one measurement.'); return
        if not methods:
            self.status.setText('Tick at least one distribution method.'); return
        if any(key == 'contin' for _label, key in methods):
            problem = self.fit_fields.alpha_problem()
            if problem:                       # never run an α the engine refuses
                self.status.setText(problem); return
        # Everything the fit needs is read from the widgets HERE, on the main
        # thread; the thunk below runs on the worker and touches only the
        # controller (the whole method is dispatched regardless of `key`, so a
        # future distribution method inherits background execution for free).
        jobs = [(iid, key, {**self._baseline_kwargs(iid, key), **self._grid_kwargs(key)})
                for iid in ids for _label, key in methods]
        # Each fit's signature is taken HERE too, beside the kwargs it describes: a field
        # moved while the fit runs then leaves it stale on arrival, never fresh.
        sigs = {(iid, key): self._fit_signature(iid, key) for iid, key, _kw in jobs}
        labels = {iid: measurement_display_name(ws[iid]) for iid in ids}
        controller = self.controller
        epoch = self._run_epoch

        def work():
            results, failures = [], []
            for iid, key, kw in jobs:
                try:
                    res = controller.run_distribution(iid, key, **kw)
                    results.append((iid, key, res))
                except Exception as exc:  # noqa: BLE001 — per (measurement, method), recorded; per (measurement, method)
                    failures.append((iid, key, f'{labels[iid]}·{key.upper()}: '
                                               f'{humanize_field_names(str(exc))}'))
            return results, failures

        def done(payload) -> None:
            # Every finished fit lands in its own slot whatever is ticked now: it is a
            # valid fit of that (measurement, method), and only its display is gated
            # (`_shown`, ui_style_guide.md §4.1 item 2).
            results, failures = payload
            for iid, key, res in results:
                self._fits[(iid, key)] = {'result': res, 'sig': sigs[(iid, key)]}
            for iid, key, _msg in failures:
                # Only THIS pair's slot: a failed NNLS must not drop the run's CONTIN.
                self._fits.pop((iid, key), None)
            self._failures = {(iid, key): msg for iid, key, msg in failures}
            self._draw()
            # The Correlogram tab can draw these too: tell it they exist, and which
            # methods to tick (a fresh Run ticks its own method).
            self.fitsChanged.emit(
                list(dict.fromkeys(key for _iid, key, _res in results)))

        def fail(exc: BaseException) -> None:
            # A whole-Run failure landing after a tick change must not write onto the
            # view now shown: the epoch a tick change bumps, read here only.
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

    def stored_fits(self) -> Dict[Tuple[str, str], Dict]:
        """Every fit this tab has stored, keyed `(iid, method)`, each with its OWN
        staleness verdict (`_fit_is_stale`) -- the accessor the Correlogram tab reads
        through its owning module, which filters by its own ticks.

        Per fit, not per Run: each fit keeps its own slot and signature, so one run's
        fit can be fresh while another's is stale. Neither this tab's ticks nor its
        method boxes filter the answer; they choose only what THIS tab shows.
        """
        return {(iid, key): {'result': self._as_distribution(v['result']),
                             'stale': self._fit_is_stale(iid, key)}
                for (iid, key), v in self._fits.items()}

    @QtCore.Slot()
    def _on_export(self) -> None:
        items = self._shown()
        if not items:
            return
        weighting, _model = self._active_weighting()
        weighted = weighting != INTENSITY
        # In a weighted view the plot forces the Rh axis (the axis combo is disabled but may
        # still hold a stale 'gamma' selection). Match it for the intensity FALLBACK export
        # too, so a reweight-errored curve never writes a Γ-axis (decay-rate) CSV into a file
        # offered as a weighted size distribution.
        axis = 'rh' if weighted else self.axis_combo.currentData()
        wd_by = {(iid, key): wd for iid, key, wd in self._weighted}

        name, suffixes = _selection_export_names(
            self.controller, [(iid, key) for iid, key, _res in items],
            'weighted_distribution' if weighted else 'distribution')

        def do_export(path: str):
            import os
            stem, ext = os.path.splitext(path)
            paths = [f'{stem}{sfx}{ext}' for sfx in suffixes]
            siblings = [] if len(items) == 1 else paths
            if not confirm_overwrite(self, siblings):
                return None
            written = []
            for (iid, key, res), p in zip(items, paths, strict=True):
                wd = wd_by.get((iid, key)) if weighted else None
                if wd is not None:
                    written.append(
                        self.controller.export_weighted_distribution(wd, p, item_id=iid))
                else:
                    # Intensity view, or a curve whose reweight errored: export the
                    # intensity distribution (never silently write nothing).
                    written.append(
                        self.controller.export_distribution(res, p, axis, item_id=iid))
            return written                     # summarised by `export_to_csv`

        status = export_to_csv(self, self.controller, name, do_export)
        if status:
            self.status.setText(status)

    def _draw(self) -> None:
        """Redraw the plot, the peak table, the bottom panel and the status and flag
        lines from the store's SHOWN subset (`_shown`) -- the one redraw every gesture
        ends in: a Run landing, a tick, a method box, a fit field, a view control, a
        re-point. On the axes built once; nothing shown empties them through `_clear`."""
        shown = self._shown()
        self.export_button.setEnabled(bool(shown))
        if not shown:
            self._clear('Tick measurements + methods, then Run.'
                        if not self._ticked_pairs()
                        else 'No current fit for the ticked runs — press Run.')
            self.status.setText(self._unshown_note())
            # `_clear` drops the flag; a ticked pair's failure in the last Run is still
            # news, and with nothing drawn the flag is where it can still be read.
            skipped = self._ticked_failures()
            set_flag(self.flag_label, 'skipped: ' + '; '.join(skipped) if skipped else '',
                     problem=True)
            return
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
            _selection_export_names(
                self.controller, [(iid, key) for iid, key, _res in shown],
                'weighted_distribution' if weighted else 'distribution')[0])
        ws = self.controller.workspace.measurements
        single = len(shown) == 1
        peak_items = []          # (x, y, text, color) collected across all curves
        # Legend material, in draw order: the name each ticked measurement is listed
        # under, the methods actually drawn, and which of a run's methods failed.
        legend_names: Dict[str, str] = {}
        legend_samples: Dict[str, str] = {}
        methods_drawn: List[str] = []
        failed_methods: Dict[str, List[str]] = {}
        drawn: List[Tuple] = []      # (iid, method, distribution) actually plotted
        self._weighted = []
        reweight_errors: List[str] = []
        labelled_invalid = False        # label the shared 'outside model validity' entry once
        # As on the Correlogram tab: a legend carries the sample only when the ticked
        # set spans samples.
        prefixes = _sample_prefixes(self.controller,
                                    [iid for iid, _k, _r in shown])
        for iid, key, res in shown:
            d = self._as_distribution(res)
            color = self.selection.color_for(iid)
            # Colour + MARKER are the measurement, line STYLE is the method -- the
            # Correlogram tab's coding, so a dash pattern means the same thing on both.
            # The marker is what carries "which measurement" for a reader who cannot use
            # the colour (R10.1), now that line style has been taken by the method; it
            # is placed sparsely, since a marker per inversion-grid point is a band.
            marker = self.selection.marker_for(iid)
            ls = _METHOD_LINESTYLE.get(key, '-')
            name = measurement_display_name(ws[iid], sample=prefixes[iid])
            # The legend bookkeeping is at the BOTTOM of this loop, where the curve
            # is actually drawn: up here, a skipped curve still got an entry.
            # A failed lognormal fit draws the seed estimate, not a fit -- say so in
            # the legend (module docstring, owner-ruled: a SAVED image must not present
            # a seed curve as fitted). With one entry per measurement rather than one
            # per curve, the note has to name WHICH method failed, since the entry now
            # stands for every method drawn for that run.
            if not getattr(d, 'success', True):
                failed_methods.setdefault(iid, []).append(_fit_label(key))
            # Every curve is drawn but none is LISTED: the legend below carries one
            # entry per measurement plus one per method, not one per curve. Nine ticked
            # runs drew nine `name . METHOD` entries, which overran the axes on both
            # sides and covered the y-label (measured 2026-09-18).
            label = '_nolegend_'
            wd = None
            if weighted:
                try:
                    wd = self.controller.reweight_distribution(iid, d, weighting, model)
                except OpticalModelError as exc:
                    reweight_errors.append(f'{measurement_display_name(ws[iid])}: {exc}')
                except Exception as exc:  # noqa: BLE001 — surfaced in reweight_errors; defensive
                    reweight_errors.append(
                        f'{measurement_display_name(ws[iid])}: {humanize_field_names(str(exc))}')
            self._weighted.append((iid, key, wd))
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
                    linestyle=ls, marker=marker, invalid_label=inv_lbl)
                peak_items.extend(self._collect_weighted_peak_labels(wd, color))
            elif not weighted:
                plot_distribution(d, ax=self.ax, axis=axis, label=label,
                                  color=color, fill=single,
                                  linestyle=ls, marker=marker)
                peak_items.extend(self._collect_peak_labels(d, res, axis, color))
            else:
                # weighted-but-wd-None (a reweight error): the curve is skipped and the
                # flag explains. Nothing about it is listed, and it gets no residual.
                continue
            drawn.append((iid, key, d))
            legend_names.setdefault(iid, name)
            legend_samples.setdefault(
                iid, prefixes[iid] or _sample_label_of(self.controller, iid))
            methods_drawn.append(key)
        self._drawn = drawn
        self._reweight_errors = reweight_errors
        # One de-collided pass so peaks that coincide across curves stagger (capped at
        # 6 + "+N more") instead of piling into an unreadable blob.
        annotate_decollided(self.ax, peak_items)
        self._draw_legend(legend_names, methods_drawn, failed_methods,
                          groups=legend_samples)
        self.ax.set_title(
            f'DLS {WEIGHTING_LABELS[weighting].lower()}-weighted size distribution'
            if weighted else 'DLS size / rate distribution')
        self._draw_bottom()                  # residuals OR the CONTIN α-sweep
        self.canvas.draw_idle()
        self._floor_resid_ratio()            # after the layout is real, not before
        self.axis_bar.attach(self.ax)
        # Status now carries only operational notes; per-result peaks live in the
        # Summary tab (which is also where they persist).
        notes = []
        n_contin = sum(1 for _i, k, _r in shown if k == 'contin')
        if n_contin > 3:
            notes.append('CONTIN runs an α sweep per measurement — this can be slow.')
        # α-selection provenance: every CONTIN fit shown is fresh, and a fresh fit's
        # signature carries the α-selection fields as they stand now, so every one ran
        # under the one selector the control shows and one note names it (method
        # disclosure — a distribution is never ambiguous about how α was chosen). The
        # wording comes from the shared `alpha_selection_note` helper the CSV export also
        # uses, so the two surfaces can't drift as selectors are added. A user-supplied
        # α returns '' here (the α input control already shows it).
        contin_res = next((r for _i, k, r in shown if k == 'contin'), None)
        alpha_note = alpha_selection_note(
            getattr(contin_res, 'alpha_selection_method', None),
            getattr(contin_res, 'ftest_prob_reject', None),
            gcv_gamma=getattr(contin_res, 'gcv_gamma', None), style='gui')
        if alpha_note:
            notes.append(alpha_note)
        if self.controller.is_dirty():
            notes.append('(ran on last committed values)')
        unshown = self._unshown_note()       # the ticked runs this plot is NOT showing
        if unshown:
            notes.append(unshown)
        notes.append('Peak values are also in the Summary tab.')
        self.status.setText('   '.join(notes))
        # A skipped run is a data/operational problem, not a muted status note: render it
        # on the red flag label via the shared renderer, matching the Correlogram tab
        # (style guide §5 R5.5). A CONTIN α pinned to a sweep edge is a data-conditional
        # caution; the wording is the result's own reliability_note (one home, set in
        # fit_contin) and only the how-to advice is added here -- kept on the same visible
        # flag, same tier (R5.2: one condition class). Empty text clears it.
        flag_msgs = []
        failures = self._ticked_failures()     # the last Run's, for the pairs ticked now
        if failures:
            flag_msgs.append('skipped: ' + '; '.join(failures))
        if getattr(self, '_reweight_errors', None):
            flag_msgs.append('reweighting skipped — ' + '; '.join(self._reweight_errors))
        # The sweep-edge caution is the result's own reliability_note (one wording on
        # every surface); the GUI adds only the how-to advice. Deduped, order-preserving.
        edge_notes = list(dict.fromkeys(
            getattr(r, 'reliability_note', '') for _i, k, r in shown
            if k == 'contin' and getattr(r, 'reliability_note', '')))
        # The advice follows which precondition the note names -- the F-test's two manual
        # checks each have their own remedy, matched on the engine's own constants (the
        # band note's stem is the template up to its first format field), never on
        # retyped wording; every other state keeps the generic sweep advice.
        band_stem = FTEST_BAND_NOTE_TEMPLATE.split('{', 1)[0]
        for note in edge_notes:
            advice = []
            if FTEST_REFERENCE_NOTE in note:
                advice.append('lower the α floor')
            if band_stem in note:
                advice.append('widen the α range or add sweep points')
            if not advice:
                advice.append('inspect the α-sweep or widen the α range')
            flag_msgs.append(f'{note} — ' + '; '.join(advice))
        # One sentence however many curves carry it, each run named beside it -- the
        # same grouping the Correlogram tab's status area uses, on the same helper.
        # Before this, six failed lognormal curves printed the same sentence six times,
        # which is the "warnings spam the bottom of the column" this batch set out to
        # fix on one tab and left standing on its neighbour. No cap: this message has
        # no per-run value, so the listing degenerates to names and eliding them would
        # leave the affected runs identified nowhere.
        failed = [(measurement_display_name(ws[f_iid]),
                   'lognormal fit did not converge; the curve shows the seed estimate, '
                   'not a fit')
                  for f_iid, _k, r in shown
                  if not getattr(self._as_distribution(r), 'success', True)]
        flag_msgs.extend(group_status_lines(failed))
        set_flag(self.flag_label, '; '.join(flag_msgs), problem=True)
        # The drawn fits' own passive `notes` -- the weight's missing count rate, a weight
        # that could not be formed, a dead channel -- on the neutral tier beside the
        # reweighted view's. Never on `flag_label`: they qualify a reported fit, they are
        # not failed preconditions (§5). Deduped by (run, sentence) so a run ticked for two
        # methods is named once, then grouped so one sentence appears once however many
        # curves carry it, with the runs named beside it -- the helper the flag label uses
        # a few lines above.
        fit_notes = list(dict.fromkeys(
            (measurement_display_name(ws[iid]), str(line))
            for iid, _key, d in drawn
            for line in (getattr(d, 'notes', ()) or ()) if str(line).strip()))
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
        set_flag(self.info_label,
                 '\n'.join(group_status_lines(fit_notes) + info_lines), problem=False)
        self._refresh_results(shown)

    def _draw_legend(self, names, methods_drawn, failed, groups=None) -> None:
        """The on-screen KEY, and the dressing that gives the SAVED figure the full
        legend -- the Correlogram tab's split, with one difference forced by layout.

        THIS figure is a plain 4x1 stack, so it has no empty corner cell to put the key
        in; it goes below the panel, where four short entries cost about 30 px. The
        CONTENT and the coding are identical to the other tab's: measurement names on
        the written file, a method key on screen.

        Any artist the plotters labelled themselves -- the shared "outside model
        validity" marker -- is kept, ahead of the entries built here. A failed
        lognormal's note rides its measurement entry into the export legend, which is
        the surface the owner's ruling is about (a saved image must not present a seed
        curve as fitted); on screen the same fact is on the status flag.
        """
        if self.ax.get_legend() is not None:
            self.ax.get_legend().remove()
        plotter = list(zip(*self.ax.get_legend_handles_labels(), strict=True))             if self.ax.get_legend_handles_labels()[1] else []
        key_entries = plotter + _method_legend_entries(dict.fromkeys(methods_drawn))
        _draw_key(self.figure, key_entries)

        by_sample = {}
        for iid, name in names.items():
            note = failed.get(iid)
            by_sample.setdefault((groups or {}).get(iid) or '', []).append(
                (iid, name + (f' ({", ".join(note)} failed)' if note else '')))
        meas_entries, group_rows = _measurement_legend_entries(
            self.selection, by_sample)
        set_plot_export_dressing(
            self.canvas,
            export_legend_dressing(self.figure, meas_entries + key_entries,
                                   group_rows)
            if names else None)

    # ---- bottom panel: residuals <-> CONTIN α-sweep ----
    def _contin_results(self) -> list:
        """The shown CONTIN fits (in tick order) — the only ones with an α-sweep."""
        return [(iid, res) for iid, key, res in self._shown() if key == 'contin']

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
        """Per-curve fit residuals (data − reconstructed g₂−1) vs delay time, through
        the one helper the Correlogram tab's residual panel also uses -- so the two
        panels cannot disagree about what a residual panel shows or how it is coded.

        THIS tab's ticked methods only: a parametric residual never appears here, and a
        distribution residual can appear here and on the Correlogram tab alike. Before
        the shared helper this drew every method of one measurement in one colour and
        one solid line, so NNLS and CONTIN could not be told apart; the method line
        style is what fixes that.
        """
        draw_fit_residuals(
            self.resid_ax,
            [(self.selection.color_for(iid), key, d)
             for iid, key, d in getattr(self, '_drawn', [])])
        # NO `_hide_shared_x` here. This tab's two panels do NOT share an x axis --
        # the distribution is drawn against Rh (or Γ) and the residual against delay
        # time τ, which this file's own construction comment states twice and
        # `_widen_row_gap(hspace=0.20)` exists for. Stripping the upper panel's axis
        # left a size distribution with no axis of its own sitting over a delay-time
        # axis on an unrelated range -- a peak position readable off the wrong
        # quantity, on screen and in the saved file. The helper is the CORRELOGRAM
        # tab's, where both panels really are τ.

    def _draw_alpha_sweep(self, contin: list) -> None:
        """Overlay each ticked CONTIN result's α-selection sweep in its curve color,
        method-aware (L-curve corner / GCV minimum / F-test curve). NNLS/lognormal have
        no sweep; if nothing to draw, show the CONTIN-only empty-state message instead.

        Every CONTIN result here is FRESH (`_shown`), and a fresh fit's signature
        carries the α-selection fields as they stand now, so every one ran under the one
        selector the control shows: they share one `alpha_selection_method` and
        therefore one ordinate — the overlay never mixes an L-curve
        (residual-vs-solution) with a GCV/F-test (score-vs-α) axis on this shared panel.
        A fit run under another selector is stale: it keeps its slot and is not drawn,
        here or above. That freshness rule is what holds this premise now that fits
        persist across Runs; a shown set that could mix selectors would need a guard
        here."""
        ws = self.controller.workspace.measurements
        drawn = 0
        for iid, res in contin:
            lcurve = getattr(res, 'lcurve', None)
            if lcurve is None:
                continue
            # The halo is a POSITIONAL mark, keyed on the states that say where the pick
            # sits -- the two edge flags and the L-curve's no-corner note (its one-point
            # sweep included), which reports the least-regularized end -- never on the
            # `alpha_reliable` roll-up above them: the F-test's two precondition notes
            # are about the sweep, not the pick's position, and draw nothing here (the
            # flag line beside the plot carries them). `plot_lcurve` keys the glyph by
            # the two edge names, so no-corner rides the floor mark; the wording each
            # state gets is the result's own reliability_note.
            ceiling = bool(getattr(res, 'alpha_at_ceiling', False))
            positional = (ceiling or bool(getattr(res, 'alpha_at_floor', False))
                          or ALPHA_NO_CORNER_NOTE in getattr(res, 'reliability_note', ''))
            try:
                plot_lcurve(
                    lcurve, ax=self.resid_ax,
                    method=getattr(res, 'alpha_selection_method', 'lcurve'),
                    color=self.selection.color_for(iid),
                    label=measurement_display_name(ws[iid]) if iid in ws else None,
                    mark_ceiling=ceiling,
                    mark_floor=positional and not ceiling,
                    gcv_gamma=getattr(res, 'gcv_gamma', None))
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
        if not self._shown():
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

        # No method token: the alpha sweep is CONTIN's own diagnostic, so the analysis
        # offers no choice to name.
        name, suffixes = _selection_export_names(
            self.controller, [(iid, None) for iid, _res in contin], 'alpha_sweep')

        def do_export(path: str):
            import os
            stem, ext = os.path.splitext(path)
            paths = [f'{stem}{sfx}{ext}' for sfx in suffixes]
            siblings = [] if len(contin) == 1 else paths
            if not confirm_overwrite(self, siblings):
                return None
            written = []
            for (iid, res), p in zip(contin, paths, strict=True):
                written.append(self.controller.export_lcurve(res, p, item_id=iid))
            return written                     # summarised by `export_to_csv`

        status = export_to_csv(self, self.controller, name, do_export)
        if status:
            self.status.setText(status)

    def _refresh_results(self, shown: List[Tuple[str, str, object]]) -> None:
        """Fill the peak-results panel from the `shown` fits — one colored row per
        (measurement · method), positional-peak columns. Mirrors the
        Correlogram results panel; reuses the same peak source as the plot labels."""
        ws = self.controller.workspace.measurements
        cols = [(iid, key, res) for iid, key, res in shown if iid in ws]
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
        # One ROW per (run · method), one COLUMN per positional peak -- transposed from
        # the column-per-run shape this table had, for the reason the Correlogram glance
        # table was rebuilt: this one grows a column per measurement AND per method, so
        # it failed by construction faster than that one did. The peak count is bounded
        # (three fit across the control column); the ticked set is not.
        table.setColumnCount(1 + n_rows)
        table.setRowCount(len(cols))
        table.horizontalHeader().setVisible(True)
        # Units live in the HEADER so a cell is '29.8 · 85%' and three peaks fit.
        table.setHorizontalHeaderLabels(
            ['Measurement', *[f'{_ordinal_peak(r)} (nm · %)' for r in range(n_rows)]])
        # The weighting names the whole table, not each column: it is one control's
        # value, and repeating it per column was spending the width the peaks need.
        self.results_label.setText(
            f'Peak results — {WEIGHTING_LABELS[weighting]}' if weighted
            else 'Peak results')
        # Left-elided and capped like the Correlogram glance table, same reason: a file
        # stem's TAIL is what tells replicates apart.
        table.setTextElideMode(QtCore.Qt.TextElideMode.ElideLeft)
        # Distribution peaks are from an ill-posed inversion — no honest ±
        # — so the Rh uses the documented fixed-sig fallback, never a fabricated ±.
        sig = self.controller.settings.no_uncertainty_sig_figs
        for r, (iid, key, _res) in enumerate(cols):
            name = QtWidgets.QTableWidgetItem(
                f'{measurement_display_name(ws[iid])} · {key.upper()}'
                if iid in ws else key.upper())
            name.setForeground(QtGui.QColor(self.selection.color_for(iid)))
            name.setIcon(marker_swatch(self.selection.color_for(iid),
                                       self.selection.marker_for(iid)))
            if iid in ws:
                name.setToolTip(measurement_hover_card(ws[iid]))
            table.setItem(r, 0, name)
            peaks = peaks_per[r]
            for c in range(1, 1 + n_rows):
                if c - 1 < len(peaks):
                    rh_nm, weight = peaks[c - 1]
                    txt = (f'{format_fixed_sig(rh_nm, sig)} · '
                           f'{format_fixed_sig(weight * 100, sig)}%')
                else:
                    txt = '—'
                it = QtWidgets.QTableWidgetItem(txt)
                it.setToolTip(txt)
                table.setItem(r, c, it)
        table.resizeColumnsToContents()
        if table.columnWidth(0) > NAME_COL_MAX_W:
            table.setColumnWidth(0, NAME_COL_MAX_W)
        for c in range(1, 1 + n_rows):
            table.horizontalHeader().setSectionResizeMode(
                c, QtWidgets.QHeaderView.ResizeMode.Stretch)
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
        # A figure legend is NOT dropped by `ax.clear()` and `show_plot_placeholder`
        # only HIDES the axes, so without this the empty-state message renders over the
        # last run's legend, naming curves that are not drawn -- and a save of that
        # state writes it. The export dressing goes with it, for the same reason.
        self.figure.legends.clear()
        set_plot_export_dressing(self.canvas, None)
        self._drawn = []
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
        # Drop any stale skipped-run flag: this is the reset choke point, and only
        # `_draw` sets the flag -- after this call when nothing is shown, re-stating just
        # the ticked pairs' failures -- so the red alarm cannot linger.
        self.flag_label.clear()
        self.info_label.clear()


# ===========================================================================
# Sample-level sub-tabs (Γ vs q²  and  D vs c)
# ===========================================================================

class _SampleAnalysisTab(QtWidgets.QWidget):
    """A single full-size sample-level plot: Gamma vs q^2, or D vs c."""

    selectionChanged = QtCore.Signal()           # ticked include-set changed (for the mirror)

    def __init__(self, controller, kind: str, parent=None, *, region=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.kind = kind                       # 'gamma_q2' or 'conc_extrap'
        # The DLS tab's shared region (owned by DLSModule), read only when this tab's
        # "use the Correlogram tab's delay window and baseline" box is ticked; a
        # standalone tab gets one of its own that nothing ever sets.
        self._region = region if region is not None else _AnalysisRegion()
        # (sid, fraction) -> is that box ticked. Per (sample, fraction), like the ticked
        # set, and never saved: each new (sample, fraction) starts unticked.
        self._use_region: Dict[tuple, bool] = {}
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
        splitter, left, right = make_split_panels(self, left_min_width=300,
                                                  sizes=(360, 740))

        # ---- controls section (the §3 column template: picker, then the region box
        # and Run, then results; Export and the status pinned under the column) ----
        controls = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(controls); cl.setContentsMargins(0, 0, 0, 0)
        self.region_box = RegionCheckBox(self._region)
        self.region_box.toggled.connect(self._on_region_toggled)
        cl.addWidget(self.region_box)
        self.run_button = QtWidgets.QPushButton('Run')
        self.run_button.setToolTip(BACKGROUND_RUN_TOOLTIP)
        self.run_button.clicked.connect(self._on_run)
        cl.addWidget(self.run_button)
        self.export_button = QtWidgets.QPushButton('Export CSV…')
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._on_export)
        self.status = QtWidgets.QLabel('')
        self.status.setWordWrap(True)

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
                ('Fits Γ = D q² across the sample\'s DLS angles (current Mw fraction).'
                 if self.kind == 'gamma_q2' else
                 'Extrapolates D(c) → c→0 across the sample\'s DLS concentrations '
                 '(current Mw fraction).'),
                'Nothing is fitted until you press <b>Run</b>. Run computes Γ (and '
                'D = Γ/q²) for each ticked measurement from its own cumulant fit — '
                'independent of any saved Correlogram/Distribution result — using the '
                'Settings skipped channels, cumulant method and cumulant order.',
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
        self.table_note = ThemedLabel('', role='hint', size=10, hide_when_empty=True)
        self.table_note.setWordWrap(True); tl.addWidget(self.table_note)
        self.temp_note = ThemedLabel('', role='hint', size=10, hide_when_empty=True)
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
        # Passive ⓘ notices carried on the result (e.g. a mixed sample identity) — a
        # neutral qualifier tier, separate from the severity-tracking flag_label above.
        # The analysis emits the same text via warnings.warn (keep-both).
        self.notes_label = ThemedLabel('', size=11)
        self.notes_label.setWordWrap(True)
        # The notes SCROLL: their length follows the result -- each run's relayed
        # notes, a region note -- and a pane floor sized when the tab was built
        # overlapped them onto the table once they grew (measured at the shipped window
        # size: 237 px of content in a 221 px pane). The table keeps its fixed floor
        # above them. The flag is pinned under the column with the status (R8.6).
        rl.addWidget(scrolling_notes(self.notes_label), 1)

        # Picker first, then the region box and Run, then results (the §3 column
        # template). The controls pane is two rows, so it takes no share of extra height.
        vstack = make_vertical_plot_stack(
            [tsec, controls, rsec], sizes=[260, 70, 250],
            min_heights=[90, controls.sizeHint().height(), 150],  # rsec: the 5-row results table (R3.2)
            stretch=(1, 0, 1))
        left.addWidget(vstack, 1)
        pin_under_column(splitter, self.export_button, self.status, self.flag_label)

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
        self.region_box.set_checked_silently(self._use_region.get(ckey, False))
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
        skip_initial_channels, the cumulant method, the cumulant fit cutoff and the
        cumulant residual weighting (the Gamma-vs-q2 fit reads the user's values), the SE
        estimator (the ± is
        estimator-routed; the confirm in Settings promises a recompute). Stored with
        the cached fit at dispatch and compared on redisplay, so a cached fit is flagged
        stale exactly when a re-Run would read different inputs. The cumulant order joins
        them (every point fits at the Settings order), and so does the region choice --
        this (sample, fraction)'s box and, when ticked, the region's applied parts -- so
        toggling the box, or moving the region while it is ticked, flags a shown fit
        stale and an unticked one is left alone."""
        s = self.controller.settings
        return (self.controller.committed_signature(self._included.get(ckey, set())),
                s.skip_initial_channels, s.cumulant_method, s.cumulant_fit_cutoff,
                s.cumulant_weighting, s.cumulant_order, s.se_estimator,
                self.region_box.key(self._use_region.get(ckey, False)))

    # ---- the region choice ----
    def _on_region_toggled(self, on: bool) -> None:
        sid = self._sample_id()
        if sid is None:
            return
        self._use_region[self._ckey(sid)] = bool(on)
        self._on_region_input_moved()

    def on_region_changed(self) -> None:
        """The Correlogram tab moved the shared region: relabel the box, and flag a
        shown result stale if this (sample, fraction)'s box is ticked."""
        self.region_box.refresh()
        self._on_region_input_moved()

    def _on_region_input_moved(self) -> None:
        """Compare the shown fit's stored key against the live one -- stale when they
        differ, and the hint cleared when they agree again (a box ticked and unticked)."""
        sid = self._sample_id()
        if sid is None:
            return
        ckey = self._ckey(sid)
        if ckey not in self._cache:
            return
        if self._cache_sig.get(ckey) != self._run_signature(ckey):
            self.status.setText(_STALE_HINT)
        elif self.status.text() == _STALE_HINT:
            self.status.clear()

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

    def _run_engine(self, sid: str, frac, include_ids, region=None):
        if self.kind == 'gamma_q2':
            return self.controller.run_gamma_q2(
                sid, fraction=frac, include_ids=include_ids, region=region)
        return self.controller.run_concentration_extrapolation(
            sid, fraction=frac, include_ids=include_ids, region=region)

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
        # Read here, on the main thread: the region is a widget-owned value and a handle
        # may move while the worker fits.
        region = self.region_box.region_for_run(self._use_region.get(ckey, False))
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
            # The box and the region stay live while the run is in flight: compare the key
            # it ran under with the live one, as the Mode Map does on landing.
            self._on_region_input_moved()

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
            lambda: self._run_engine(sid, frac, include_ids, region),
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
            except Exception:  # noqa: BLE001 — cell shows a dash
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
                        + ('; from the cumulant Γ' if res.gamma_source == 'cumulant'
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

    def __init__(self, controller, parent=None, *, distribution_fit_values=None,
                 region=None) -> None:
        super().__init__(parent)
        self.controller = controller
        # The shared region and this tab's per-(sample, fraction) choice to read it --
        # the Γ-vs-q² tab's shape (`_SampleAnalysisTab.__init__`).
        self._region = region if region is not None else _AnalysisRegion()
        self._use_region: Dict[tuple, bool] = {}
        # The Distribution sub-tab's current field values, read once per press of the
        # copy button -- an accessor handed in by the owning module, so neither tab
        # reaches into the other (the Correlogram tab's `distribution_fits` shape).
        self._distribution_fit_values = distribution_fit_values
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
        # The Distribution tab's floor and default, for the reason stated there: the
        # Inversion Settings group carries the same Rh-grid triple-spinbox row.
        splitter, left, right = make_split_panels(self, left_min_width=385,
                                                  sizes=(403, 717))

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
                'The CONTIN and NNLS sources invert on this tab\'s own '
                '<b>Inversion Settings</b>, which start from the Settings tab: '
                'the Rh grid and residual weighting and, for CONTIN, the α '
                'selection (NNLS is the unregularized case and has no α). A '
                'denser Rh grid resolves more closely-spaced modes and costs '
                'sharply more time per run.',
                '<b>Copy from the Distribution tab</b> fills those fields once '
                'with the Distribution sub-tab\'s current values — tune CONTIN on '
                'one run there, then map the sample with the same settings. It is '
                'a one-time copy, not a link: a later change there does not follow.',
                'Tick <b>Use the Correlogram tab\'s delay window and baseline</b> to '
                'extract each run over the region set there, as the Correlogram and '
                'Distribution tabs fit it; unticked, each run keeps its own lag range '
                'and baseline estimate.']))
        srow = QtWidgets.QWidget()
        sl = QtWidgets.QFormLayout(srow); sl.setContentsMargins(0, 0, 0, 0)
        self.source_combo = QtWidgets.QComboBox()
        for label, code in self._SOURCES:
            self.source_combo.addItem(label, code)
        self.source_combo.setToolTip(
            'Which fit supplies each run\'s modes: the two-mode double-'
            'exponential (fast), or a CONTIN/NNLS distribution\'s resolved '
            'peaks (slower per run). Both distribution sources invert on the '
            'Inversion Settings below — the Rh grid and residual weighting, and '
            'for CONTIN the α selection (NNLS has none); the inversion cost rises '
            'steeply with grid resolution. The map reads the Correlogram tab\'s τ '
            'window and baseline only when its box below the source is ticked. See '
            'the Theory-and-Equations-Guide.')
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        sl.addRow('Source:', self.source_combo)
        self.facet_combo = QtWidgets.QComboBox()
        self.facet_combo.setToolTip(
            'The polarization state shown (and pickable). Γ_VH = Γ_VV + 6·D_r '
            'is a physical difference, so states are never pooled on one map.')
        self.facet_combo.currentIndexChanged.connect(self._on_facet_changed)
        sl.addRow('Polarization:', self.facet_combo)
        cl.addWidget(srow)
        self.region_box = RegionCheckBox(self._region)
        self.region_box.toggled.connect(self._on_region_toggled)
        cl.addWidget(self.region_box)
        # The inversion's own fields -- the shared group the Distribution sub-tab and the
        # Settings tab hold -- shown for the distribution sources only: the parametric
        # double exponential reads none of them.
        self.inversion_box = QtWidgets.QGroupBox('Inversion Settings')
        add_help_to_groupbox(
            self.inversion_box,
            'What the CONTIN and NNLS sources invert each ticked run on.',
            bullets=[
                'They start from the Settings tab’s DLS Distribution Defaults and '
                'can be changed here for this map; a change marks a map already '
                'shown as out of date. A Settings-tab Update that changes any setting, '
                'or Restore defaults, sets them back to the Settings values, replacing '
                'an edit or a copy made here — as it does the Distribution sub-tab’s '
                'fields.',
                'The Rh grid and residual weighting apply to both sources; the α rows '
                'are CONTIN’s alone, and hide for NNLS.',
                '<b>Copy from the Distribution tab</b> fills every field once with the '
                'Distribution sub-tab’s current values. It is a one-time copy, not a '
                'link: a later change there does not follow, and the map runs on the '
                'values shown here.',
                'The CONTIN α range and, for the F-test selector, its level go into the '
                'map’s notes; each point records its own α, selector, γ and grid.',
            ])
        ibox_form = QtWidgets.QFormLayout(self.inversion_box)
        self.fit_fields = DistributionFitFields(self)
        self.fit_fields.add_rows(ibox_form)
        self.fit_fields.seed_from_settings(self.controller.settings)
        self.fit_fields.changed.connect(self._on_fit_input_moved)
        self.copy_fields_button = QtWidgets.QPushButton('Copy from the Distribution tab')
        self.copy_fields_button.setToolTip(
            'Fill every field above once with the Distribution sub-tab’s current '
            'values. A one-time copy: a later change on the Distribution sub-tab does '
            'not follow, and a Settings-tab Update that changes any setting sets the '
            'fields back to the Settings values.')
        self.copy_fields_button.clicked.connect(self._on_copy_fit_fields)
        self.copy_fields_button.setVisible(self._distribution_fit_values is not None)
        ibox_form.addRow(self.copy_fields_button)
        cl.addWidget(self.inversion_box)
        self._sync_source_controls()
        self.run_button = QtWidgets.QPushButton('Run')
        self.run_button.setToolTip(BACKGROUND_RUN_TOOLTIP)
        self.run_button.clicked.connect(self._on_run)
        # Run is pinned under the scrolled controls and Export under the column (the
        # §3 column template), so the Inversion Settings group a distribution source
        # shows can never carry them out of view.
        self.export_button = QtWidgets.QPushButton('Export CSV…')
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._on_export)
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
        # Record mode set… acts on the families, so it sits in their pane below.
        self.status = QtWidgets.QLabel('')
        self.status.setWordWrap(True)

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
        self.table_note = ThemedLabel('', role='hint', size=10, hide_when_empty=True)
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
        fl.addWidget(self.record_set_button)
        self.flag_label = ThemedLabel('', size=11)
        self.flag_label.setWordWrap(True)
        self.notes_label = ThemedLabel('', size=11)
        self.notes_label.setWordWrap(True)
        fl.addWidget(scrolling_notes(self.notes_label))

        # Scrolled, as the Distribution tab's controls are: the Inversion Settings group
        # makes the section taller than a short window can give it, and a floor at its
        # full height would starve the two panes beside it. The floor is the section's
        # height WITHOUT that group (the default double-exponential source hides it, and
        # a layout skips a hidden widget); Run is pinned under the scroll area, so it
        # stays in view whichever source is chosen.
        controls_scroll = QtWidgets.QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        controls_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        controls_scroll.setWidget(controls)
        floor = max(controls.sizeHint().height(), 120)
        ctrl_pane = QtWidgets.QWidget()
        ctrl_layout = QtWidgets.QVBoxLayout(ctrl_pane)
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        ctrl_layout.setSpacing(4)
        ctrl_layout.addWidget(controls_scroll, 1)
        ctrl_layout.addWidget(self.run_button)
        # Picker first, then the controls and Run, then the families and results (the
        # §3 column template).
        vstack = make_vertical_plot_stack(
            [tsec, ctrl_pane, fsec], sizes=[150, floor + 40, 200],
            min_heights=[130, floor, 120], stretch=COLUMN_STACK_STRETCH)
        left.addWidget(vstack, 1)
        pin_under_column(splitter, self.export_button, self.status, self.flag_label)

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
        self.region_box.set_checked_silently(self._use_region.get(ckey, False))
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
        data: the committed params of the ticked measurements, the source, the settings
        the engine seeds (skip_initial_channels, for every source) and, for a
        distribution source, the Inversion Settings fields the Run hands it. Stored at
        dispatch, compared on redisplay and on every edit of those fields (the Γ-q²
        tab's shape)."""
        s = self.controller.settings
        source = self.source_combo.currentData()
        # `se_estimator` too: the Run ends by refitting the families, whose ± it routes,
        # and a switch clears those fits -- so a map shown across one needs a re-Run.
        seeds: tuple = (s.skip_initial_channels, s.se_estimator)
        if source != 'double':
            seeds += tuple(sorted(self._inversion_kwargs(source).items()))
        # The region choice (the Γ-q² tab's reasoning): toggling the box, or moving the
        # region while it is ticked, flags a shown map stale; an unticked one is left be.
        return (self.controller.committed_signature(self._included.get(ckey, set())),
                source, seeds, self.region_box.key(self._use_region.get(ckey, False)))

    # ---- the region choice ----
    def _on_region_toggled(self, on: bool) -> None:
        sid = self._sample_id()
        if sid is None:
            return
        self._use_region[self._ckey(sid)] = bool(on)
        self._on_fit_input_moved()

    def on_region_changed(self) -> None:
        """The Correlogram tab moved the shared region: relabel the box and re-check the
        shown map's key (stale only while this (sample, fraction)'s box is ticked)."""
        self.region_box.refresh()
        self._on_fit_input_moved()

    # ---- inversion settings ----
    def _inversion_kwargs(self, source: str) -> Dict:
        """The kwargs a Run hands `run_mode_map` from this tab's own fields: none for the
        parametric source, which would refuse them."""
        return {} if source == 'double' else self.fit_fields.fit_kwargs(source)

    def _sync_source_controls(self) -> None:
        """Show the inversion fields for a distribution source only, and CONTIN's own
        rows for CONTIN only -- show/hide, never an inert field."""
        source = self.source_combo.currentData()
        self.inversion_box.setVisible(source != 'double')
        self.fit_fields.set_contin_rows_visible(source == 'contin')

    def _on_source_changed(self, _i) -> None:
        self._sync_source_controls()
        self._on_fit_input_moved()

    def _on_fit_input_moved(self) -> None:
        """An input the next Run reads moved: say so beside a map drawn under the old
        inputs. Compared rather than assumed, so moving a field back clears the hint."""
        sid = self._sample_id()
        if sid is None or self._map_result(sid) is None:
            return
        ckey = self._ckey(sid)
        if self._cache_sig.get(ckey) != self._run_signature(ckey):
            self.status.setText(_STALE_HINT)
        elif self.status.text() == _STALE_HINT:
            self.status.clear()

    def _on_copy_fit_fields(self) -> None:
        """Fill the fields ONCE from the Distribution sub-tab's current values. A copy,
        never a link: after it the map runs on what these fields show, so no value
        changes behind the Mode Map's back."""
        if self._distribution_fit_values is None:
            return
        self.fit_fields.seed(self._distribution_fit_values())
        copied = 'Copied the Distribution tab’s fit settings.'
        stale = self.status.text() == _STALE_HINT
        self.status.setText(f'{copied} {_STALE_HINT}' if stale else copied)

    def reseed_from_settings(self) -> None:
        """(Re)seed the Inversion Settings from the Settings defaults -- the cascade the
        shell runs after a Settings Update, as for the Distribution sub-tab."""
        self.fit_fields.seed_from_settings(self.controller.settings)

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
        if self.source_combo.currentData() == 'contin':
            problem = self.fit_fields.alpha_problem()
            if problem:                       # never run an α the engine refuses
                self.status.setText(problem)
                return
        if runner().is_busy:
            first = not self._recompute_pending
            self._recompute_pending = ckey   # the LATEST press wins the slot
            if first:
                run_when_idle(self._flush_recompute)
            self.status.setText(BUSY_NOTICE)
            return
        include_ids = set(self._included.get(ckey, set()))
        source = self.source_combo.currentData()
        # Read from the widgets HERE, on the main thread; the thunk below runs on the
        # worker and touches only the controller.
        inversion_kw = self._inversion_kwargs(source)
        region = self.region_box.region_for_run(self._use_region.get(ckey, False))
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
            # The fields stay editable while the run is in flight, so compare the key the
            # run was dispatched under against the live one rather than assume a match.
            self._on_fit_input_moved()

        def fail(exc: BaseException) -> None:
            QtWidgets.QMessageBox.critical(
                self, 'Analysis failed',
                f'Could not build the mode map for sample {sid}.\n\n'
                f'{humanize_field_names(str(exc))}')
            if epoch == self._run_epoch:       # the `done` guard: never paint a late
                self.status.setText('Analysis failed — see dialog.')   # failure elsewhere

        runner().try_submit(
            lambda: self.controller.run_mode_map(
                sid, fraction=frac, source=source, include_ids=include_ids,
                region=region, **inversion_kw),
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
            except Exception:  # noqa: BLE001 — cell shows a dash
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

    def __init__(self, controller, parent=None, *, region=None) -> None:
        super().__init__(parent)
        self.controller = controller
        # The shared region and this tab's choice to read it -- held per SAMPLE, since
        # DDLS has no fraction (`run_ddls` takes none and this tab caches by sample).
        self._region = region if region is not None else _AnalysisRegion()
        self._use_region: Dict[str, bool] = {}
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
        splitter, left, right = make_split_panels(self, left_min_width=300,
                                                  sizes=(360, 740))

        # Paired angles with an include checkbox (tick to include in the D_r/D_t fit —
        # the same idiom as Γ vs q²), plus each correlogram's polarization + pairing.
        left.addWidget(section_header(
            'Paired Angles (tick to include in the fit)',
            'Tick which paired angles enter the D_r/D_t fit.',
            bullets=['DDLS pairs the sample\'s VV (polarized) and VH (depolarized) '
                     'correlograms by angle and extracts the rotational diffusion '
                     'coefficient D_r = (Γ_VH − Γ_VV)/6 (D_t from the VV channel).',
                     'Tag each correlogram\'s polarization in the <b>Data</b> tab.',
                     'Only angles with BOTH a VV and a VH correlogram can be paired.',
                     'Untick an angle to drop it (e.g. an outlier); the fit re-runs on '
                     'the ticked angles.',
                     'Dropped angles still plot as a gray × (labeled "excluded" in the '
                     'legend) so you can see what was left out.',
                     'Ticked angles read blue and light the matching sidebar leaves.']))
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(['', 'Angle (°)', 'Polarization', 'Paired'])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setMaximumHeight(170)
        # No row numbers and a content-sized check column, so the four columns fit the
        # 300 px column without a sideways scroll bar.
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemChanged.connect(self._on_angle_toggled)
        left.addWidget(self.table)
        # Outlier removal: untick an angle above to drop it; unticked angles are
        # grayed on the plot. "Include all" re-ticks every paired angle -- the table's
        # bulk-select, so it sits under the table.
        self.reset_excl_button = QtWidgets.QPushButton('Include all angles')
        self.reset_excl_button.clicked.connect(self._on_include_all)
        incl_row = QtWidgets.QHBoxLayout()
        incl_row.setContentsMargins(0, 0, 0, 0)
        incl_row.addWidget(self.reset_excl_button)
        incl_row.addStretch(1)
        left.addLayout(incl_row)

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
        self.region_box = RegionCheckBox(self._region)
        self.region_box.toggled.connect(self._on_region_toggled)
        left.addWidget(self.region_box)

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
            'with the file. See the Theory-and-Equations-Guide §3.6.')
        self.export_button.clicked.connect(self._on_export)

        left.addWidget(QtWidgets.QLabel('Results'))
        self.results = _vtable(self._RESULT_ROWS)
        left.addWidget(self.results)
        self.status = ThemedLabel('', role='muted', size=11)
        self.status.setWordWrap(True)
        self.flag_label = ThemedLabel('', role='error', bold=True)
        self.flag_label.setWordWrap(True)

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
        # Export, the status and the flag pinned under the column (§3 column template).
        pin_under_column(splitter, self.export_button, self.status, self.flag_label)

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
        self.region_box.set_checked_silently(self._use_region.get(sid, False))
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
        cumulant fit cutoff and weighting, the SE estimator behind the D_r/D_t ±). Stored at dispatch
        with the rod the fit reads (`rod=`), compared on redisplay against the box (the
        Γ-q² tab's shape).

        The settings half is not maintained by hand: every field
        `Controller.run_ddls` reads must appear here, and
        `test_every_run_signature_carries_the_settings_its_engine_path_reads` derives
        that set by scanning the controller rather than trusting this list."""
        s = self.controller.settings
        if rod is _ROD_FROM_BOX:
            try:
                rod = self._rod_input()
            except ValueError:
                rod = self.rod_edit.text().strip()
        return (self.controller.committed_signature(self._input_ids(sid)), rod,
                s.skip_initial_channels, s.cumulant_method, s.cumulant_fit_cutoff,
                s.cumulant_weighting, s.cumulant_order, s.se_estimator,
                self.region_box.key(self._use_region.get(sid, False)))

    # ---- the region choice (the Γ-q² tab's shape, held per sample) ----
    def _on_region_toggled(self, on: bool) -> None:
        sid = self._sample_id()
        if sid is None:
            return
        self._use_region[sid] = bool(on)
        self._on_region_input_moved()

    def on_region_changed(self) -> None:
        self.region_box.refresh()
        self._on_region_input_moved()

    def _on_region_input_moved(self) -> None:
        """A region input moved: flag the shown fit stale when its stored key no longer
        matches, and clear the hint when it matches again. Never an automatic refit --
        unlike an angle tick, which refits the ticked subset -- because the region is
        another tab's input and the user decides when to re-read it."""
        sid = self._sample_id()
        if sid is None or sid not in self._cache:
            return
        if self._cache_sig.get(sid) != self._run_signature(sid):    # the rod box, as
            self.status.setText(_STALE_HINT)                        # set_measurement
        elif self.status.text() == _STALE_HINT:
            self.status.clear()

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
        region = self.region_box.region_for_run(self._use_region.get(sid, False))
        sig = self._run_signature(sid, rod=rod)   # names the VV/VH ids feeding this fit
        if sid == self._last_sid:
            self._run_epoch += 1            # this run supersedes any in flight
            epoch = self._run_epoch
            self.status.setText('Fitting in the background…')
        else:
            epoch = None      # deferred Run of a sample no longer shown: never paint it here

        def work():
            # ONE thunk for both calls. Run order is load-bearing: run_ddls writes
            # results[('ddls', sid)] each call (last-write-wins), so the EXCLUDED fit runs
            # LAST and the controller's store ends up holding the fit shown in the table --
            # the one a caller of ddls_shape with no fit of its own reads. This tab's shape
            # table and export pass the shown fit itself. The un-excluded `full` (run
            # first, only when excluding) is kept solely to draw the excluded-point markers.
            full = (controller.run_ddls(sid, rod_length_nm=rod, region=region)
                    if excl else None)
            res, info = controller.run_ddls(
                sid, rod_length_nm=rod, exclude_angles=excl, region=region)
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
            # The box and the region stay live while the run is in flight: compare the key
            # it ran under with the live one, as the Mode Map does on landing.
            self._on_region_input_moved()

        def fail(exc: BaseException) -> None:
            # A fit that cannot run is flagged in the panel, not a modal (style guide
            # R8.4): an angle untick re-runs this live, and a modal would come up per
            # click. The controller's message is relayed as written. The previous fit's
            # numbers go with its flag, so none stays on screen under the failure. A
            # failure for a sample no longer shown is told in a dialog naming it, never
            # painted over the view now shown; one superseded by a newer run of this
            # same sample is dropped, because that run reports its own outcome.
            msg = humanize_field_names(str(exc))
            if sid != self._last_sid:
                who = self.controller.workspace.samples.get(sid)
                QtWidgets.QMessageBox.critical(
                    self, 'DDLS failed',
                    f'Could not run DDLS on {sample_display_name(who) if who else sid}.'
                    f'\n\n{msg}')
                return
            if epoch != self._run_epoch:
                return
            self._clear_results()
            self.export_button.setEnabled(False)
            set_flag(self.flag_label, f'DDLS could not run: {msg}')
            self.status.setText('DDLS failed — see the warning below.')

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
        # physical D_r); export the observables alone otherwise. From the shown fit,
        # as the shape table is.
        try:
            shapes = self.controller.ddls_shape(sid, model='both', result=res)
        except Exception:  # noqa: BLE001 — shape models optional in export
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
        """Compute and show the shape models of the DDLS fit this tab shows for the
        current sample -- its own cached fit, never the controller's store, which a
        Settings clear can empty while the tab keeps showing the fit, marked out of date."""
        sid = self._sample_id()
        try:
            # Memoize by the shown fit's identity: the ellipsoid inverse runs a Monte-Carlo
            # (heavy), and _draw_shape fires on every sample selection, not just Run.
            res, _info = self._cache[sid]
            cached = self._shape_cache.get(sid)
            if cached is not None and cached[0] is res:
                shapes = cached[1]
            else:
                shapes = self.controller.ddls_shape(sid, model='both', result=res)
                self._shape_cache[sid] = (res, shapes)
        except Exception as exc:  # noqa: BLE001 — surfaced in the caveat
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


# ONE mark for a cell whose run simply has no such fit. Before this the same state
# rendered two ways in one table -- 'n/a' from `format_fixed_sig(None)` for the
# parametric columns, '—' from `_peak_cell([])` for the distribution ones.
#
# It is NOT a synonym for 'n/a': a stored NaN means the fit RAN and withheld its value
# (owner decision D2), and that still reads 'n/a'. `format_fixed_sig` maps missing and
# non-finite alike to 'n/a', so routing an absent value through it is what destroyed the
# distinction; `_num` keeps it.
_ABSENT = '—'


class _GroupRow(str):
    """A table row that is one SPANNING header, not a row of cells.

    A `str` subclass so the row still carries its text with no wrapper to unpack, and
    so `_replace_table` can tell it from a list of cells by type alone. Used for the
    Summary per-measurement table's sample bands -- the shape the measurement picker
    already uses -- in place of a Sample column repeated down every run.
    """


def _num(value, sig: int) -> str:
    """A Summary number cell: `_ABSENT` when the run has no such fit, else the value
    (which is 'n/a' when the fit ran and withheld it)."""
    return _ABSENT if value is None else format_fixed_sig(value, sig)


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


#: How deep a cumulant fit must go before a moment is a measurement rather than a blank,
#: keyed by the detail spec's STABLE STORAGE KEY. `_detail_cell` below is the home of what
#: the gate does and why it runs before the value is inspected at all.
_MOMENT_MIN_ORDER = {'mu2': 2, 'mu3': 3, 'mu4': 4}


#: How a detail column names the method it belongs to. Short, because it rides on every
#: detail header and the table is already wide.
_METHOD_SHORT = {'cumulant': 'cumulant', 'single': 'single exp',
                 'double': 'dbl exp', 'kww': 'KWW'}


def _detail_quantity(name: str, unit: str) -> str:
    """The quantity half of a detail header: the name, then its unit in parentheses.

    Deliberately the same spelling `_RESULT_ROWS` uses for the Correlogram panel's row
    labels, so `_RESULT_ROW_TOOLTIPS` keys both surfaces -- the owner-ruled μ₃/μ₄ caution
    is then one string with one home, copied onto this table rather than restated in it.
    `test_every_detail_quantity_is_spelled_as_the_correlogram_panel_spells_it` pins the
    parity for every quantity, not just the two that carry a caution today.
    """
    return f'{name} ({unit})' if unit else name


def _detail_header(name: str, unit: str, method: str) -> str:
    """A detail column's full header: the quantity, then the method that produced it.

    The method is named on EVERY detail column, never only where a name would collide.
    Several parametric arms reuse quantity names -- `baseline` and `converged` on three
    of them, `Γ` on two -- and the table is built to scroll sideways, so the headline
    column that would disambiguate is routinely off-screen when the duplicate is read.
    Measured at 1280 x 760 with all four arms run: 26 columns wanting 1548 px in a 602 px
    viewport, with two `baseline` columns visible at once and neither headline in view.

    Unconditionally rather than on collision, for the reason the measurement names
    already follow: a name that changes according to what else is on screen is a name a
    reader cannot trust or search for. A column means the same thing whatever is beside it.
    """
    return f'{_detail_quantity(name, unit)} · {_METHOD_SHORT.get(method, method)}'


def _detail_cell(value, key: str, details: Dict, sig: int) -> str:
    """One detail value, rendered. The stored value is the fit's own; painting is here.

    Three kinds arrive and each is shown as itself: a bool is the convergence answer
    ('yes'/'no', never 'True'), an int or str is printed as it stands (the order, the
    cumulant method), and every number goes through the no-uncertainty significant-figures
    knob -- these come from a single correlogram, so none carries a ± (invariant 8).

    A missing moment is the interesting case. Below the order that fits it there IS no
    value, and a bare blank -- or worse, a bare 0 -- would read as a fitted result, so the
    cell says which order the fit actually ran at.

    THE ORDER GATE RUNS FIRST, on the order rather than on the value's shape, and this is
    the normative home of that rule. The two absences do not look alike: µ₃ and µ₄ come
    back None below their order, but µ₂ comes back `0.0` -- a real float that is
    structurally absent rather than fitted -- so a gate that only asked "is it None?"
    printed a confident zero for the one moment a reader is most likely to act on, while
    the Correlogram panel's own cell said "unmeasured (order 1)" for the same fit. Asking
    the fit what order it ran at makes the value's shape stop mattering.

    The gate is keyed by storage key (`_MOMENT_MIN_ORDER`), so retitling a column cannot
    quietly disarm it.
    """
    need = _MOMENT_MIN_ORDER.get(key)
    order = details.get('order')
    if need is not None and isinstance(order, int) and order < need:
        return f'unmeasured (order {order})'
    if value is None:
        return _ABSENT
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, (int, str)):
        return str(value)
    return format_fixed_sig(value, sig)


def _with_detail_columns(specs: List, sig: int) -> List:
    """Expand each parametric method's headline into its full set of detail columns.

    The detail columns are read from the controller's one spec, so this function never
    names a quantity: it only decides WHERE the expansion goes -- directly after the
    method's existing headline columns, so a method's quantities stay together under one
    heading rather than being interleaved with another's.
    """
    out = []
    for spec in specs:
        out.append(spec)
        header, _populated, _cell, method, target, _tip = spec
        details = _DLS_DETAIL_SPECS.get(method)
        # Only after the method's LAST headline column, so the details follow the whole
        # headline rather than splitting it (the cumulant has two: Rh and PDI).
        if not details or spec is not _last_headline_of(specs, method):
            continue
        for skey, name, _getter, unit, target in details:
            out.append((
                _detail_header(name, unit, method),
                (lambda m: lambda r: bool(r['details'].get(m)))(method),
                (lambda m, k: lambda r: _detail_cell(
                    r['details'].get(m, {}).get(k), k, r['details'].get(m, {}), sig)
                 )(method, skey),
                method, target,
                # The caution is COPIED, keyed by the QUANTITY -- one string, one home
                # (`_RESULT_ROW_TOOLTIPS`), serving this table and the Correlogram panel.
                _RESULT_ROW_TOOLTIPS.get(_detail_quantity(name, unit), '')))
    return out


def _last_headline_of(specs: List, method: str):
    """The final headline column belonging to `method`, or None."""
    owned = [s for s in specs if s[3] == method]
    return owned[-1] if owned else None


def _with_passive_notes(cell, method: str, target, is_headline: bool, shown):
    """Wrap a cell builder so a passive notice reaches the cell it is about.

    The ⓘ is the neutral qualifier tier (ui_style_guide section 5): a passive note says
    how the fit was obtained, never that a precondition failed, so it must not borrow the
    ⚠ channel. Where a cell carries BOTH -- a flagged value that also rests on a bound --
    the ⚠ keeps the glyph, because a data-quality problem outranks a neutral remark, and
    the tooltip carries the flag's sentence first and the notices after it. One glyph per
    cell: two would say the cell has two kinds of problem when it has one.
    """
    def build(r):
        base = cell(r)
        notes = (r.get('notes', {}) or {}).get(method, {}) or {}
        mine = list(notes.get(target, ())) if target else []
        if is_headline:
            # Everything with nowhere else to go: whole-fit notices, and any note whose
            # quantity has no visible column.
            for key, sentences in notes.items():
                if not key or key not in shown:
                    mine.extend(sentences)
        if not mine:
            return base
        # The cell may already be (text, tooltip) or (text, tooltip, ICON) -- the
        # measurement column carries its plot swatch in a third element. Rebuilding it as
        # a bare pair dropped that icon, so a run that acquired a passive note silently
        # lost its colour swatch: an identity mark removed by an unrelated disclosure.
        # The tail is preserved rather than assumed empty.
        if isinstance(base, str):
            text, tip, tail = base, '', ()
        else:
            text, tip, tail = base[0], base[1], tuple(base[2:])
        joined = '\n'.join(([tip] if tip else []) + mine)
        return (text if text.startswith('⚠') else f'ⓘ {text}', joined, *tail)
    return build


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
        splitter, left, right = make_split_panels(self, left_min_width=300,
                                                  sizes=(360, 740))

        self.checklist = MeasurementPicker(
            self.controller, self.selection, kinds=('dls',),
            label_fn=measurement_display_name, header_fn=sample_display_name,
            tooltip_fn=measurement_hover_card, title='Measurements',
            help_text='Tick measurements to filter the Summary table.',
            help_bullets=_SUMMARY_PICKER_BULLETS)
        self.checklist.selectionChanged.connect(self._refresh_tables)
        left.addWidget(self.checklist, 1)      # the picker takes the column's spare height

        self.ticked_only = QtWidgets.QCheckBox('Ticked only')
        self.ticked_only.setToolTip('Show only the measurements ticked above '
                                    '(otherwise every measurement that has results).')
        self.ticked_only.toggled.connect(self._refresh_tables)
        left.addWidget(self.ticked_only)

        # Off by default: the headline columns are what most readers want, and the full
        # set is wide. The tooltip is the passive nuance tier (how the numbers behave),
        # while the how-to-use sits in the picker's "?" badge above.
        self.show_details = QtWidgets.QCheckBox('Show fit details')
        self.show_details.setToolTip(
            'Add every other quantity each parametric fit reports — the decay rate, the '
            'moments, the fitted order and baseline, the amplitude fractions and the '
            'stretch. These come from a single correlogram, so none carries an '
            'uncertainty; see the Theory-and-Equations-Guide §3.')
        self.show_details.toggled.connect(self._refresh_tables)
        left.addWidget(self.show_details)

        self.export_button = QtWidgets.QPushButton('Export CSV…')
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._on_export)

        self.status = QtWidgets.QLabel('')
        self.status.setWordWrap(True)
        # Export and the status pinned at the column's foot (§3 column template).
        pin_under_column(splitter, self.export_button, self.status)

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

    @staticmethod
    def _per_meas_columns(recs: List[Dict], sig: int, show_details: bool = False):
        """[(header, cell_fn, header_tooltip)] for the methods any LISTED run actually has.

        One spec per column, header and cell builder together, so a method cannot show
        up in the headers and not in the cells. Every method is on demand: before this,
        the double-exponential and lognormal columns appeared when populated and the
        other five were always there, so a cumulant-only workspace still paid for four
        columns of `_ABSENT` -- the asymmetry, not the on-demand rule, was the defect.

        With `show_details`, each parametric method that ran expands from its headline to
        the full set its detail spec names. The detail columns are built from that spec
        (`app.controller._DLS_DETAIL_SPECS`), never listed again here, so a quantity
        added there appears without a second edit -- the same one-spec rule that governs
        what the durable row saves.

        Every column carries the METHOD it belongs to and the result ATTRIBUTE it shows,
        which is how a passive note reaches the cell it explains rather than the fit as a
        whole. The headline column of a method is simply its first, and it is where a
        fit-level note -- or one whose target no visible column shows -- lands.
        """
        rh = 'R<sub>h</sub>'

        def flagged(text, value, note):
            # The mark sits on the value it judges and explains itself there (⚠, not ℹ:
            # a data-quality problem, and R5.2 gives one condition one tier across
            # surfaces). A missing value takes no mark -- the house rule.
            return (f'⚠ {text}', note) if note and value is not None else text

        def peaks(method: str):
            return lambda r: flagged(_peak_cell(r['peaks'].get(method, []), sig),
                                     r['peaks'].get(method) or None,
                                     r.get('peak_notes', {}).get(method, ''))

        def scalar(key: str, note_key: str):
            return lambda r: flagged(_num(r[key], sig), r[key], r.get(note_key, ''))

        # (header, populated, cell, method, note-target attribute, header tooltip).
        # The tooltip is CARRIED rather than looked up from the finished header: a detail
        # header now names its method too, so the caution's key is the quantity half and
        # splitting the header back apart would be a second spelling to keep in step.
        specs = [
            (f'Cumulant {rh}', lambda r: r['cumulant_rh'] is not None,
             scalar('cumulant_rh', 'cumulant_rh_note'), 'cumulant', 'rh_nm', ''),
            ('PDI', lambda r: r['pdi'] is not None,
             lambda r: _num(r['pdi'], sig), 'cumulant', 'pdi', ''),
            (f'Single {rh}', lambda r: r['single_rh'] is not None,
             scalar('single_rh', 'single_rh_note'), 'single', 'rh_nm', ''),
            (f'Double-exp ({rh} fast / slow)', lambda r: r['double_fast'] is not None,
             lambda r: flagged(
                 f"{_num(r['double_fast'], sig)} / {_num(r['double_slow'], sig)}"
                 if r['double_fast'] is not None else _ABSENT,
                 r['double_fast'], r.get('double_note', '')), 'double', None, ''),
            (f'KWW {rh}', lambda r: r['kww_rh'] is not None,
             scalar('kww_rh', 'kww_rh_note'), 'kww', 'rh_nm', ''),
            (f'NNLS ({rh} / Int %)', lambda r: bool(r['peaks'].get('nnls')),
             peaks('nnls'), 'nnls', None, ''),
            (f'CONTIN ({rh} / Int %)', lambda r: bool(r['peaks'].get('contin')),
             peaks('contin'), 'contin', None, ''),
            (f'Lognormal ({rh} / Int %)', lambda r: bool(r['peaks'].get('lognormal')),
             peaks('lognormal'), 'lognormal', None, ''),
        ]
        if show_details:
            specs = _with_detail_columns(specs, sig)

        live = [s for s in specs if any(s[1](r) for r in recs)]
        # Which targets a VISIBLE column shows, per method. A note tagged anything else
        # -- '' for a whole-fit notice, or a quantity whose column is not on screen --
        # falls back to that method's headline, so no note is ever dropped for want of a
        # column, and none is guessed onto a neighbour that is not about it.
        shown = {}
        first_col_of = {}
        for idx, (_h, _p, _c, method, target, _tip) in enumerate(live):
            if target:
                shown.setdefault(method, set()).add(target)
            first_col_of.setdefault(method, idx)

        out = []
        for idx, (header, _populated, cell, method, target, tip) in enumerate(live):
            is_headline = first_col_of.get(method) == idx
            out.append((header,
                        _with_passive_notes(cell, method, target, is_headline,
                                            shown.get(method, frozenset())),
                        tip))
        return out

    def _fill_per_meas(self, recs: List[Dict]) -> None:
        # Every number here is from a single correlogram or an ill-posed inversion — no
        # honest ± — so all use the one no-uncertainty knob, never a
        # fabricated uncertainty.
        sig = self.controller.settings.no_uncertainty_sig_figs
        ws = self.controller.workspace.measurements
        order = sidebar_measurement_order(self.controller)
        recs = sorted(recs, key=lambda r: order.get(r.get('item_id') or '', len(order)))
        cols = self._per_meas_columns(recs, sig, self.show_details.isChecked())
        headers = ['Measurement', *[h for h, _cell, _tip in cols]]
        # Header tooltips are index-aligned with `headers`, so the Measurement column
        # takes the leading '' rather than the first data column's caution.
        header_tips = ['', *[tip for _h, _cell, tip in cols]]

        rows: list = []
        current_sample = None
        for r in recs:
            # The sample is a GROUP HEADER row rather than a column repeated down every
            # run, the shape the measurement picker already uses. It was the widest
            # column in the table and said the same thing N times.
            if r['sample_label'] != current_sample:
                current_sample = r['sample_label']
                rows.append(_GroupRow(current_sample))
            iid = r.get('item_id') or ''
            lm = ws.get(iid)
            # The Measurement cell carries item 1's hover card, like every other
            # surface that names a run, and the run's plot swatch when it is TICKED.

            # Gated on the ticked set EXPLICITLY, the way `MeasurementPicker`'s painter
            # is. `color_for` does NOT answer None for an unticked id -- `_index_of`
            # returns `len(ids)` for a miss, so every unticked run drew the same real
            # colour, the one the NEXT ticked run will take. A swatch is an identity
            # claim, and with "Ticked only" off this table lists unticked runs.
            ticked = iid in set(self.selection.ids())
            name = (r['measurement_label'],
                    measurement_hover_card(lm) if lm is not None else '',
                    marker_swatch(self.selection.color_for(iid),
                                  self.selection.marker_for(iid))
                    if ticked else QtGui.QIcon())
            rows.append([name, *[cell(r) for _h, cell, _tip in cols]])
        self._replace_table(self.per_meas_table, headers, rows, header_tips)

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
                       rows: List[List[str]],
                       header_tips: Optional[List[str]] = None) -> None:
        """Rebuild `table`. A cell is a plain string, `(text, tooltip)` or
        `(text, tooltip, icon)`; a ROW may instead be a `_GroupRow`, painted as one
        spanning band.

        The tuple forms exist so a marked cell can explain itself where the mark appears
        (ui_style_guide §5: a soft, always-on interpretive caveat belongs in a passive
        tooltip) and so a run can carry its plot swatch beside its name. Plain strings
        still work unchanged -- `_fill_sample` passes only those -- and that
        back-compatibility is asserted, not assumed.

        `header_tips`, when given, is index-aligned with `headers` and carries a caution
        that belongs to the QUANTITY rather than to any one run's value -- the μ₃/μ₄
        sentence. On the header, not the cells, for the reason `_RESULT_ROW_TOOLTIPS`
        gives: a value cell's tooltip is reserved for that value's own reason, and the
        two must not fight.
        """
        table.clear()
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        for c, tip in enumerate(header_tips or ()):
            item = table.horizontalHeaderItem(c)
            if item is not None and tip:
                item.setToolTip(tip)
        table.setRowCount(len(rows))
        for r, cells in enumerate(rows):
            if isinstance(cells, _GroupRow):
                item = QtWidgets.QTableWidgetItem(str(cells))
                font = item.font()
                font.setBold(True)
                item.setFont(font)
                table.setItem(r, 0, item)
                # One band across the whole width. Re-set every rebuild because
                # `clear()` drops the items but not the spans.
                table.setSpan(r, 0, 1, max(len(headers), 1))
                continue
            table.setSpan(r, 0, 1, 1)          # undo any span this row held before
            for c, cell in enumerate(cells):
                if isinstance(cell, tuple):
                    text = cell[0]
                    tip = cell[1] if len(cell) > 1 else ''
                    icon = cell[2] if len(cell) > 2 else None
                else:
                    text, tip, icon = cell, '', None
                item = QtWidgets.QTableWidgetItem(text)
                if tip:
                    item.setToolTip(tip)
                if icon is not None and not icon.isNull():
                    item.setIcon(icon)
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
        # The Distribution tab is built FIRST so the Correlogram tab can be handed its
        # accessor at construction. Only the construction order moves; the sub-tab order
        # on screen is set by `addTab` below and is unchanged.
        self.distribution_tab = _DistributionTab(controller, self.region, self.selection)
        self.correlogram_tab = _CorrelogramTab(
            controller, self.region, self.selection,
            distribution_fits=self.distribution_fits)
        # The four tabs that re-run a fit behind the scenes read the region only through
        # their own "use the Correlogram tab's delay window and baseline" box.
        self.gamma_q2_tab = _SampleAnalysisTab(controller, 'gamma_q2', region=self.region)
        self.mode_map_tab = _ModeMapTab(
            controller, distribution_fit_values=self.distribution_fit_values,
            region=self.region)
        self.conc_tab = _SampleAnalysisTab(controller, 'conc_extrap', region=self.region)
        self.ddls_tab = _DDLSTab(controller, region=self.region)
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
        # The module owns both co-plotting tabs, so IT carries the news that the
        # Distribution tab's stored fits moved -- neither widget reaches into the other.
        self.distribution_tab.fitsChanged.connect(
            self.correlogram_tab.on_fits_changed)
        for tab in (self.gamma_q2_tab, self.mode_map_tab, self.conc_tab,
                    self.ddls_tab):
            tab.selectionChanged.connect(self.selectionChanged)
            # ... and the news that the region moved, the same routing: each box
            # relabels itself and flags a checked result stale.
            self.correlogram_tab.regionChanged.connect(tab.on_region_changed)
        self.set_measurement(None)

    def distribution_fits(self) -> Dict[Tuple[str, str], Dict]:
        """The Distribution sub-tab's stored fits, keyed `(iid, method)`, each with that
        tab's own staleness verdict.

        The Correlogram tab's "Fits to show" list reads its three distribution entries
        through here, so the two sibling widgets never touch each other's state: the
        module that owns both is the only thing that knows they are siblings.
        """
        return self.distribution_tab.stored_fits()

    def distribution_fit_values(self) -> Dict:
        """The Distribution sub-tab's current fit-field values, for the Mode Map's
        one-time copy -- read through the module that owns both, like
        `distribution_fits`."""
        return self.distribution_tab.fit_fields.values()

    def average_prompt_sources(self) -> 'AverageSources':
        """What the "Average derived results…" prompt can copy or read, handed to it by
        the module that owns the tabs (the `distribution_fits` accessor shape): the
        shared region, the Correlogram tab's order box and the Distribution tab's fields.
        Read at the moment the user confirms, so the copy is of the values shown then."""
        fields = self.distribution_tab.fit_fields

        def distribution_kwargs(method: str) -> dict:
            # The copy reads the fields a CONTIN Run reads, so it refuses what that Run
            # refuses: `alpha_problem` is the one predicate for every consumer of the group.
            problem = fields.alpha_problem() if method == 'contin' else ''
            if problem:
                raise ValueError(problem)
            return fields.fit_kwargs(method)

        return AverageSources(
            region=self.region,
            cumulant_order=lambda: int(self.correlogram_tab.order_spin.value()),
            distribution_kwargs=distribution_kwargs)

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
        self.mode_map_tab.reseed_from_settings()

    def focus_mode_map(self) -> None:
        """Bring the Mode Map sub-tab to the front (cross-tab navigation from
        the SLS tab's Mode-Resolved list — navigation, never duplication)."""
        self.tabs.setCurrentWidget(self.mode_map_tab)

    def set_measurement(self, item_id: Optional[str]) -> None:
        self.item_id = item_id
        if item_id is None:
            self.header.setText('Pick a measurement in the Workspace list; '
                                'tick measurements in the sub-tabs to analyze.')
            self.header.setToolTip('')        # the card goes with the measurement
            runnable = False
        else:
            kind = self.controller.workspace.measurements[item_id].kind
            runnable = (kind == 'dls')
            # Named, not id'd: the one builder, and its hover card for the rest
            # (the id included, so a bug report still has it).
            lm = self.controller.workspace.measurements[item_id]
            name = measurement_display_name(lm)
            self.header.setToolTip(measurement_hover_card(lm))
            if runnable:
                self.header.setText(
                    f'DLS measurement: <b>{html.escape(name)}</b>')
            else:
                self.header.setText(
                    f'<b>{html.escape(name)}</b> is a {kind.upper()} '
                    'measurement — '
                    'open it in the SLS tab.')
        for tab in self._subtabs:
            tab.set_measurement(item_id, runnable)
