"""
The DLS fit methods the Correlogram and Distribution sub-tabs offer, their labels and line
styles, the run key of a correlogram fit, and the shared residual-panel drawing.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from plotting.plots import display_factor as _disp_factor, display_unit as _disp_unit
from core.note_registry import REGISTRY
from core.notes import find_notes


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


# The result kind the controller stores each Correlogram method's fit under, keyed
# (kind, item_id): the tab draws the run from there (`Controller.shown_run`).
_CORR_RESULT_KINDS = {'cumulant': 'cumulants', 'single': 'single_exp',
                      'double': 'double_exp', 'kww': 'kww'}


def _corr_run_key(iid: str, method: str) -> tuple:
    """The controller key of the Correlogram tab's `method` fit of `iid`."""
    return (_CORR_RESULT_KINDS[method], iid)


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


def _failed_without_cause(res, fed) -> bool:
    """A fit that did not succeed, where no line of the surface's notes block says why:
    its failure fails no precondition note (a silent flag), or the block was not fed it
    (`fed`, the ids of the results it was). Such a failure is an operational fact, one
    short flag line (ui_style_guide §5). A CONTIN result's success is its distribution's."""
    fit = getattr(res, 'distribution', None) or res
    if getattr(fit, 'success', True) is not False:
        return False
    return id(res) not in fed or not any(
        'success' in REGISTRY[n.code].fails
        for n in find_notes(res, kind='precondition'))


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
