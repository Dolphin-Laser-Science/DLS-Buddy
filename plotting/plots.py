"""
plotting/plots.py
=================

Matplotlib plotting for the DLS and SLS result objects.

This is the visualization layer. It is built before, and independently of, any
GUI, and is designed to drop straight into one later. Every function follows the
same contract:

  - It accepts an optional `ax`. If none is given, it creates a figure and axes;
    if one is given, it draws onto it (so several analyses can share axes and be
    overlaid -- NNLS vs CONTIN, Zimm vs Berry).
  - It NEVER calls plt.show() or saves a file. The caller (a script, or the GUI)
    decides what to do with the figure.
  - It returns a `PlotHandles` carrying the figure, the axes, and a dict of the
    individual artists, so a GUI can toggle their visibility, restyle them, or
    remove them without re-running the analysis.

Interactivity (re-fitting on a slider move, etc.) is NOT here: the GUI reads its
controls, re-calls the analysis function with new arguments, and calls the
matching plot function again. Plotting stays static and stateless.

Styling uses the Okabe-Ito colorblind-safe palette, applied consistently across
every plot so overlays remain distinguishable.

Change history
--------------
2026-06-13  plotting/plots.py v1: DLS (correlogram fit, distribution, L-curve,
            Gamma-q^2, concentration extrapolation) and SLS (Rayleigh ratio,
            Debye, Zimm/Berry grid, calibration-free A2) plotters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from analysis.dls import distribution_axis


# ===========================================================================
# Shared style and helpers
# ===========================================================================

# Okabe-Ito colorblind-safe palette (the default). Plots reference colors by
# SEMANTIC ROLE ('blue', 'vermilion', ...), not literal color, so a palette swap
# remaps every plot consistently. The role keys are shared across all palettes.
OKABE_ITO = {
    'black': '#000000', 'orange': '#E69F00', 'sky': '#56B4E9',
    'green': '#009E73', 'yellow': '#F0E442', 'blue': '#0072B2',
    'vermilion': '#D55E00', 'purple': '#CC79A7',
}
_PALETTES = {
    'okabe_ito': dict(OKABE_ITO),
    'tab10': {
        'black': '#000000', 'orange': '#ff7f0e', 'sky': '#17becf',
        'green': '#2ca02c', 'yellow': '#bcbd22', 'blue': '#1f77b4',
        'vermilion': '#d62728', 'purple': '#9467bd',
    },
    'grayscale': {
        'black': '#000000', 'orange': '#808080', 'sky': '#b3b3b3',
        'green': '#666666', 'yellow': '#cccccc', 'blue': '#1a1a1a',
        'vermilion': '#4d4d4d', 'purple': '#999999',
    },
}

# The ACTIVE palette — a module-level dict mutated IN PLACE by set_palette, so every
# `PALETTE['blue']` reference in the plot functions follows a swap without re-import.
PALETTE = dict(OKABE_ITO)
_CYCLE_KEYS = ('blue', 'vermilion', 'green', 'orange', 'purple', 'sky', 'black')
_CYCLE = [PALETTE[k] for k in _CYCLE_KEYS]


def set_palette(name: str) -> None:
    """Switch the active plot palette ('okabe_ito' / 'tab10' / 'grayscale').

    Applies to plots drawn afterwards; figures already on screen keep their colors
    until redrawn. Unknown names fall back to the Okabe-Ito default."""
    global _CYCLE
    chosen = _PALETTES.get(name, _PALETTES['okabe_ito'])
    PALETTE.clear()
    PALETTE.update(chosen)
    _CYCLE = [PALETTE[k] for k in _CYCLE_KEYS]


def cycle() -> list:
    """The CURRENT overlay color cycle (reflects the active `set_palette`). Pass this
    function itself to a live consumer (e.g. `SelectionModel(color_cycle=cycle)`) so a
    palette switch reaches it — `_CYCLE` is rebound on each swap, so a captured copy or
    a stale `import _CYCLE` binding would freeze on the startup palette."""
    return list(_CYCLE)


# ---------------------------------------------------------------------------
# Per-series marker + linestyle cycles (colour-independence — WCAG 1.4.1)
# ---------------------------------------------------------------------------
# Overlaid series (co-plotted correlograms, Zimm concentrations, scaling series) must be
# distinguishable WITHOUT colour — for a colour-vision-deficient reader and in greyscale.
# These run in lock-step with `_CYCLE` (same length, index-keyed), so series N gets a
# stable (colour, marker, linestyle) triple. The GUI's SelectionModel exposes matching
# marker_for/linestyle_for keyed the same way.
_MARKER_CYCLE = ['o', 's', '^', 'D', 'v', 'P', 'X']          # circle, square, triangle...
_LINESTYLE_CYCLE = ['-', '--', ':', '-.', (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1))]


def marker_for(i: int) -> str:
    """Stable marker for series index `i` (cycles)."""
    return _MARKER_CYCLE[i % len(_MARKER_CYCLE)]


def linestyle_for(i: int):
    """Stable linestyle for series index `i` (cycles)."""
    return _LINESTYLE_CYCLE[i % len(_LINESTYLE_CYCLE)]


# ---------------------------------------------------------------------------
# On-screen figure theming — white default + opt-in dark
# ---------------------------------------------------------------------------
# The on-screen matplotlib figure defaults to WHITE (clean exports/print). The GUI may
# opt in to theming the *on-screen* figure to a dark palette when the app theme is dark
# (Settings "Match plot to app theme"). EXPORT ALWAYS STAYS WHITE (locked, style guide
# R10.3): `save_figure_white` recolours to white, saves, and restores. Only the figure
# CHROME (facecolor, spines, ticks, labels, title, grid, legend frame) is themed — the
# data SERIES keep their PALETTE colours so they stay legible in both.
import matplotlib as _mpl

_LIGHT_FIG = {'fig': 'white', 'ax': 'white', 'chrome': 'black', 'grid': '#b0b0b0'}
_DARK_FIG = {'fig': '#2b2b2b', 'ax': '#2b2b2b', 'chrome': '#d8d8d8', 'grid': '#555555'}
_ONSCREEN_DARK = False                    # current on-screen theme (module state)


def onscreen_plot_dark() -> bool:
    """Whether the on-screen plot theme is currently dark."""
    return _ONSCREEN_DARK


def set_onscreen_plot_theme(dark: bool) -> None:
    """Set the on-screen plot chrome theme via rcParams, so figures drawn/redrawn
    afterwards inherit it with NO change to the plot functions (they clear + recreate
    axes, which pick up the rcParams). Export stays white regardless (savefig.facecolor).
    Call `apply_figure_theme` on already-drawn canvases for an immediate refresh."""
    global _ONSCREEN_DARK
    _ONSCREEN_DARK = bool(dark)
    p = _DARK_FIG if dark else _LIGHT_FIG
    _mpl.rcParams.update({
        'figure.facecolor': p['fig'], 'axes.facecolor': p['ax'],
        'axes.edgecolor': p['chrome'], 'axes.labelcolor': p['chrome'],
        'axes.titlecolor': p['chrome'], 'xtick.color': p['chrome'],
        'ytick.color': p['chrome'], 'xtick.labelcolor': p['chrome'],
        'ytick.labelcolor': p['chrome'], 'text.color': p['chrome'],
        'grid.color': p['grid'], 'legend.facecolor': p['fig'],
        'legend.edgecolor': p['chrome'],
        'savefig.facecolor': 'white', 'savefig.edgecolor': 'white',
    })


def apply_figure_theme(fig, dark: bool) -> None:
    """Force the chrome colours of an ALREADY-DRAWN figure to the light/dark theme
    (rcParams only affect NEW artists). Used for an immediate on-screen refresh and,
    with dark=False, by `save_figure_white`. Recolours facecolor, spines, ticks, axis
    labels, titles, grid lines and any legend frame/text; leaves data series alone."""
    p = _DARK_FIG if dark else _LIGHT_FIG
    fig.set_facecolor(p['fig'])
    fig.set_edgecolor(p['fig'])
    for ax in fig.axes:
        ax.set_facecolor(p['ax'])
        for spine in ax.spines.values():
            spine.set_edgecolor(p['chrome'])
        ax.tick_params(colors=p['chrome'], which='both')
        ax.xaxis.label.set_color(p['chrome'])
        ax.yaxis.label.set_color(p['chrome'])
        ax.title.set_color(p['chrome'])
        for lbl in (*ax.get_xticklabels(), *ax.get_yticklabels()):
            lbl.set_color(p['chrome'])
        for gl in (*ax.get_xgridlines(), *ax.get_ygridlines()):
            gl.set_color(p['grid'])
        leg = ax.get_legend()
        if leg is not None:
            leg.get_frame().set_facecolor(p['fig'])
            leg.get_frame().set_edgecolor(p['chrome'])
            for t in leg.get_texts():
                t.set_color(p['chrome'])


def save_figure_white(fig, path: str) -> str:
    """Save `fig` to `path` on a WHITE background regardless of the on-screen theme
    (locked policy — exports must be clean/print-parity). Recolours to white, saves,
    then restores the current on-screen theme."""
    apply_figure_theme(fig, dark=False)
    try:
        fig.savefig(path, facecolor='white', edgecolor='white')
    finally:
        apply_figure_theme(fig, dark=_ONSCREEN_DARK)
        if fig.canvas is not None:
            fig.canvas.draw_idle()
    return path


# ---------------------------------------------------------------------------
# Plot-axis display units
# ---------------------------------------------------------------------------
# Plots store/compute in canonical units but DISPLAY in human-scale units (µs, kcps,
# nm⁻², µm²/s, mg/mL, ...). The active choice is a module-level dict mutated by
# set_plot_units (mirroring set_palette); each plot scales its canonical arrays and
# labels its axes via _disp(). app.units is a pure (GUI-free) boundary helper.
from app import units as _U

_PLOT_QUANTITIES = ('time', 'intensity', 'scattering_q2', 'diffusion',
                    'decay_rate', 'radius', 'molar_mass', 'concentration')
PLOT_UNITS: Dict[str, str] = {q: _U.default_unit(q) for q in _PLOT_QUANTITIES}


def set_plot_units(mapping: Optional[Dict[str, str]]) -> None:
    """Set the per-quantity plot display units. `mapping` keys are plot quantities
    ('time', 'intensity', 'scattering_q2', 'diffusion', 'decay_rate', 'radius',
    'molar_mass', 'concentration'); unknown/missing keys keep the human-scale
    default. Applies to plots drawn afterwards."""
    for q in _PLOT_QUANTITIES:
        PLOT_UNITS[q] = _U.default_unit(q)            # reset to defaults first
    for q, u in (mapping or {}).items():
        if q in _PLOT_QUANTITIES and u in _U.unit_options(q):
            PLOT_UNITS[q] = u


def _disp(quantity: str, canonical) -> tuple:
    """(scaled_array, unit_label) for a canonical value/array in the active unit."""
    unit = PLOT_UNITS.get(quantity, _U.canonical_unit(quantity))
    return _U.from_canonical(quantity, np.asarray(canonical, float), unit), unit


def display_factor(quantity: str) -> float:
    """Multiplier converting a canonical scalar to the active display unit (for GUI
    overlays that must line up with a plot's converted axis)."""
    unit = PLOT_UNITS.get(quantity, _U.canonical_unit(quantity))
    return float(_U.from_canonical(quantity, 1.0, unit))


def display_unit(quantity: str) -> str:
    """The active display unit label for a plot quantity."""
    return PLOT_UNITS.get(quantity, _U.canonical_unit(quantity))


@dataclass
class PlotHandles:
    """What a plot function returns.

    figure, axes : the matplotlib Figure and Axes drawn on.
    artists      : named artists (lines, collections, ...) for a GUI to toggle,
                   restyle, or remove. Always includes any secondary axes
                   (e.g. a residual panel) under a descriptive key.
    """
    figure: Any
    axes: Any
    artists: Dict[str, Any] = field(default_factory=dict)


def annotate_decollided(ax, items, *, max_stack: int = 6, fontsize: int = 8,
                        gap_pts: float = 11.0, base_offset_pts: float = 4.0,
                        cluster_px: float = 20.0, overflow_color: str = '#888'):
    """Annotate data points with text, staggering labels that would otherwise overlap.

    `items` is an iterable of ``(x, y, text, color)`` in DATA coordinates. Labels whose
    anchor points land within `cluster_px` pixels of one another are treated as a single
    cluster and **stacked vertically** (upward, `gap_pts` apart) above the cluster's
    highest point, so they don't pile into an unreadable blob (usability feedback
    2026-06-30 item 8). At most `max_stack` labels are drawn per cluster; any beyond that
    are replaced by one muted **"+N more"** marker so a dense cluster stays readable
    instead of either disappearing or becoming a mess. Non-overlapping labels are drawn
    singly in place, so the common single-label case is unchanged.

    De-collision is computed in DISPLAY (pixel) space, so it is independent of the axis
    scale (callers use log axes). Call this AFTER the data and axis limits are set on
    `ax`, since it reads ``ax.transData``. Returns the list of Text artists.
    """
    pts_data = [it for it in items
                if it[0] is not None and it[1] is not None
                and np.isfinite(it[0]) and np.isfinite(it[1])]
    if not pts_data:
        return []
    trans = ax.transData
    disp = [trans.transform((x, y)) for (x, y, _t, _c) in pts_data]  # display px
    n = len(pts_data)

    # Greedy single-link clustering by display-space proximity.
    clusters: list = []
    thresh_sq = cluster_px * cluster_px
    for i in range(n):
        joined = None
        for members in clusters:
            if any((disp[i][0] - disp[j][0]) ** 2 + (disp[i][1] - disp[j][1]) ** 2
                   <= thresh_sq for j in members):
                joined = members
                break
        if joined is None:
            clusters.append([i])
        else:
            joined.append(i)

    texts = []
    for members in clusters:
        # Stack from the cluster's highest point (largest display y) upward; center the
        # column on the members' mean x so it reads as one tidy stack.
        order = sorted(members, key=lambda j: disp[j][1], reverse=True)
        anchor_x = float(np.mean([pts_data[j][0] for j in members]))
        anchor_y = pts_data[order[0]][1]
        shown = order[:max_stack]
        for rank, j in enumerate(shown):
            _x, _y, txt, col = pts_data[j]
            texts.append(ax.annotate(
                txt, xy=(anchor_x, anchor_y),
                xytext=(0, base_offset_pts + rank * gap_pts),
                textcoords='offset points', ha='center', fontsize=fontsize,
                color=col, clip_on=False))
        extra = len(order) - len(shown)
        if extra > 0:
            texts.append(ax.annotate(
                f'+{extra} more', xy=(anchor_x, anchor_y),
                xytext=(0, base_offset_pts + max_stack * gap_pts),
                textcoords='offset points', ha='center',
                fontsize=max(fontsize - 1, 6), color=overflow_color, clip_on=False))
    return texts


def _new_figure(figsize) -> Figure:
    """Create a Figure on an Agg canvas, OFF pyplot's global ``Gcf`` registry.

    ``plt.subplots`` retains every figure it makes in ``Gcf`` until an explicit
    ``plt.close(fig)``, so a long-running Qt GUI that re-plots on each
    interaction would leak figures (steady memory growth + matplotlib's
    ">20 figures" warning) unless every caller remembered to close them. A
    GUI-embedded plotting layer instead builds a bare ``Figure`` and attaches a
    canvas directly: the caller owns it, nothing is parked in the registry.
    (The GUI tabs pass their own ``ax`` and never hit this path; scripts and
    tests that let a helper create the axes get a registry-free figure.)"""
    fig = Figure(figsize=figsize)
    FigureCanvasAgg(fig)         # sets fig.canvas; enables tight_layout/savefig
    return fig


def _get_ax(ax: Optional[Any], figsize=(6.0, 4.5)):
    """Return (figure, axes, created). Create a figure only if ax is None."""
    if ax is None:
        fig = _new_figure(figsize)
        ax = fig.subplots()
        return fig, ax, True
    return ax.figure, ax, False


def _next_color(ax) -> str:
    """Pick a palette color not obviously already in use on this axes."""
    used = len(ax.lines) + len(ax.collections)
    return _CYCLE[used % len(_CYCLE)]


# ===========================================================================
# DLS plots
# ===========================================================================

def plot_correlogram_fit(
    result,
    ax: Optional[Any] = None,
    residual_ax: Optional[Any] = None,
    color: Optional[str] = None,
    label: Optional[str] = None,
    show_residuals: bool = True,
) -> PlotHandles:
    """Plot a parametric DLS fit: g2-1 data with the model, and residuals.

    Works for any result carrying fit_tau_s / fitted_g2m1 / residuals (cumulant,
    single, double, KWW). The measured g2-1 is reconstructed as fit + residual.

    If `ax` is None and show_residuals is True, a two-panel figure is created
    (correlogram above, residuals below, shared x). If `ax` is given, the
    correlogram is drawn on it; residuals go on `residual_ax` if provided.

    Parameters
    ----------
    result : a parametric DLS result object
    ax, residual_ax : matplotlib Axes, optional
    color : str, optional
        Overrides the automatic palette color (useful when overlaying).
    label : str, optional
        Legend label for the fit line.
    show_residuals : bool

    Returns
    -------
    PlotHandles
    """
    tau = np.asarray(result.fit_tau_s, dtype=float)
    fit = np.asarray(result.fitted_g2m1, dtype=float)
    resid = np.asarray(result.residuals, dtype=float)
    data = fit + resid

    created = False
    if ax is None and show_residuals:
        fig = _new_figure((6.0, 5.4))
        ax, residual_ax = fig.subplots(
            2, 1, sharex=True,
            gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.08})
        created = True
    else:
        fig, ax, created = _get_ax(ax)

    c = color or _next_color(ax)
    tau_d, t_unit = _disp('time', tau)
    artists = {}
    artists['data'] = ax.scatter(tau_d, data, s=18, facecolors='none',
                                 edgecolors=c, alpha=0.7, label='data')
    artists['fit'] = ax.plot(tau_d, fit, '-', color=c, lw=1.8,
                             label=label or 'fit')[0]
    ax.set_xscale('log')
    ax.set_ylabel(r'$g_2(\tau) - 1$')
    ax.legend(frameon=False, fontsize=9)
    if residual_ax is None:
        ax.set_xlabel(rf'Delay time $\tau$ ({t_unit})')

    if residual_ax is not None:
        artists['residuals'] = residual_ax.plot(tau_d, resid, '-', color=c,
                                                lw=1.0)[0]
        residual_ax.axhline(0.0, color=PALETTE['black'], lw=0.6, ls=':')
        residual_ax.set_xscale('log')
        residual_ax.set_xlabel(rf'Delay time $\tau$ ({t_unit})')
        residual_ax.set_ylabel('resid.')
        artists['residual_ax'] = residual_ax

    return PlotHandles(fig, ax, artists)


# Smallest value any log axis will plot. A computed curve (e.g. a correlogram fit
# evaluated over the full tau-range) can underflow toward 0, leaving a tail of
# sub-normal positives spanning ~300 decades; on a log axis that sends autoscale
# and the tick formatter (round(inf)) haywire under constrained_layout. This floor
# trims such a tail. It sits far below any physically meaningful plotted quantity
# (a normalized g2-1 has ~1e-3-1e-4 precision), so it never clips real data.
_LOG_FLOOR = 1e-12


def mask_log_axes(x, y, *, xlog: bool, ylog: bool, floor: float = _LOG_FLOOR):
    """Drop points a log axis cannot render, returning filtered float arrays.

    Removes any non-finite point, and on a log axis any value below `floor` --
    which discards both non-positive values and an underflowing computed-curve
    tail (see `_LOG_FLOOR`). Shared sanitizer for log-scale plots; use it wherever
    a *computed* curve is drawn on a log axis and could underflow. NOTE it filters
    x and y together, so it is not suitable where a separate index must stay
    aligned to the unfiltered arrays (e.g. the L-curve's optimal-point marker),
    nor where a non-positive point is itself meaningful and should surface rather
    than vanish (e.g. a negative A2 on the scaling plot).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if xlog:
        m &= x >= floor
    if ylog:
        m &= y >= floor
    return x[m], y[m]


def plot_correlogram_scaled(
    ax: Any,
    tau,
    g2m1,
    fit_tau=None,
    fit_g2m1=None,
    *,
    xscale: str = 'log',
    yscale: str = 'linear',
    color: Optional[str] = None,
    label: Optional[str] = None,
    compact: bool = False,
    marker: str = 'o',
) -> PlotHandles:
    """Draw a correlogram (+ optional fit overlay) on one axes at given scales.

    `xscale` / `yscale` are 'linear' or 'log'. On a log axis, non-positive values
    are masked (g2-1 dips just below 0 in the noise tail). `compact` shrinks the
    ticks and drops axis labels for the small side views of the multi-scale panel.
    `color`/`label` let several correlograms be overlaid on one axes with a legend;
    `marker` varies the point SHAPE per overlaid series so they stay distinct without
    colour (WCAG 1.4.1 / greyscale).
    Used by the DLS tab's 4-view correlogram and the multi-measurement overlay.
    """
    fig, ax, _ = _get_ax(ax)
    tfac = display_factor('time')
    tau = np.asarray(tau, dtype=float) * tfac
    g2 = np.asarray(g2m1, dtype=float)
    c = color or PALETTE['blue']

    xlog, ylog = xscale == 'log', yscale == 'log'

    artists = {}
    dx, dy = mask_log_axes(tau, g2, xlog=xlog, ylog=ylog)
    artists['data'] = ax.scatter(dx, dy, s=(7 if compact else 18), marker=marker,
                                 facecolors='none', edgecolors=c, alpha=0.7,
                                 label=(label or 'data'))
    if fit_tau is not None and fit_g2m1 is not None:
        fx, fy = mask_log_axes(np.asarray(fit_tau, dtype=float) * tfac,
                               np.asarray(fit_g2m1, dtype=float),
                               xlog=xlog, ylog=ylog)
        # When an explicit color is given (multi-measurement overlay) the fit line
        # matches its data color so each measurement reads as one color; with no
        # color (single-measurement view) the fit keeps the vermilion default.
        fit_c = color or PALETTE['vermilion']
        artists['fit'] = ax.plot(fx, fy, '-', color=fit_c,
                                 lw=(1.2 if compact else 1.8), label='fit')[0]
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if compact:
        ax.tick_params(labelsize=6)
    else:
        ax.set_xlabel(rf'Delay time $\tau$ ({display_unit("time")})')
        ax.set_ylabel(r'$g_2(\tau) - 1$')
    return PlotHandles(fig, ax, artists)


def plot_distribution(
    result,
    ax: Optional[Any] = None,
    axis: str = 'rh',
    color: Optional[str] = None,
    label: Optional[str] = None,
    fill: bool = True,
    linestyle: str = '-',
) -> PlotHandles:
    """Plot an NNLS or CONTIN size/rate distribution.

    Overlay-capable: call twice on the same `ax` (e.g. NNLS and CONTIN) and each
    gets its own palette color and label.

    Parameters
    ----------
    result : DistributionResult
    ax : matplotlib Axes, optional
    axis : str
        'rh' (hydrodynamic radius, nm) or 'gamma' (decay rate, 1/s).
    color, label : optional styling.
    fill : bool
        Lightly fill under the curve.

    Returns
    -------
    PlotHandles
    """
    x, weights, xlabel = distribution_axis(result, axis)
    x = np.asarray(x, dtype=float)
    # Display-unit scaling: Rh in the active radius unit (default nm),
    # Γ in the active decay-rate unit (default s⁻¹).
    if axis == 'gamma':
        x = x * display_factor('decay_rate')
        xlabel = rf'$\Gamma$ ({display_unit("decay_rate")})'
    else:
        x = x * display_factor('radius')
        xlabel = rf'$R_h$ ({display_unit("radius")})'
    fig, ax, created = _get_ax(ax)
    c = color or _next_color(ax)
    lbl = label or (result.method.upper() if hasattr(result, 'method') else None)

    artists = {}
    artists['line'] = ax.plot(x, weights, color=c, ls=linestyle, lw=1.8, label=lbl)[0]
    if fill:
        artists['fill'] = ax.fill_between(x, weights, color=c, alpha=0.15)
    ax.set_xscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Weight (intensity fraction)')
    if lbl:
        ax.legend(frameon=False, fontsize=9)
    return PlotHandles(fig, ax, artists)


def plot_lcurve(lcurve, ax: Optional[Any] = None) -> PlotHandles:
    """Plot a CONTIN L-curve (solution norm vs residual norm), marking the corner."""
    fig, ax, created = _get_ax(ax)
    res = np.asarray(lcurve.residual_norms, dtype=float)
    sol = np.asarray(lcurve.solution_norms, dtype=float)
    artists = {}
    artists['curve'] = ax.plot(res, sol, '-o', color=PALETTE['blue'],
                               ms=4, lw=1.2)[0]
    i = lcurve.optimal_index
    artists['optimal'] = ax.plot(res[i], sol[i], 'o', color=PALETTE['vermilion'],
                                 ms=11, mfc='none', mew=2,
                                 label=f'optimal alpha = {lcurve.optimal_alpha:.2e}')[0]
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Residual norm $\|A x - y\|^2$')
    ax.set_ylabel(r'Solution seminorm $\|L x\|^2$')
    ax.legend(frameon=False, fontsize=9)
    return PlotHandles(fig, ax, artists)


def plot_gamma_q2(result, ax: Optional[Any] = None) -> PlotHandles:
    """Plot Gamma vs q^2 with the through-origin diffusion fit.

    The data points, the through-origin line (slope = D), and the unconstrained
    intercept are shown. A non-zero intercept flags non-diffusive behavior.
    """
    fig, ax, created = _get_ax(ax)
    qfac, q_unit = display_factor('scattering_q2'), display_unit('scattering_q2')
    gfac, g_unit = display_factor('decay_rate'), display_unit('decay_rate')
    q2 = np.asarray(result.q2_m2, dtype=float)
    gamma = np.asarray(result.gamma_s_inv, dtype=float)
    artists = {}
    artists['data'] = ax.scatter(q2 * qfac, gamma * gfac, s=36, color=PALETTE['blue'],
                                 zorder=3, label='data')
    # through-origin fit line (computed in canonical units, then scaled to display)
    xline = np.linspace(0.0, q2.max() * 1.05, 50)
    artists['fit'] = ax.plot(xline * qfac, result.d_m2_s * xline * gfac, '-',
                             color=PALETTE['vermilion'], lw=1.6,
                             label=f'D = {result.d_m2_s:.3e} m$^2$/s')[0]
    flag = 'diffusive' if result.is_diffusive else 'NON-diffusive'
    artists['intercept'] = ax.axhline(
        result.intercept_s_inv * gfac, color=PALETTE['green'], lw=1.0, ls='--',
        label=f'intercept ({flag}, rel={result.intercept_relative:.2f})')
    ax.set_xlabel(rf'$q^2$ ({q_unit})')
    ax.set_ylabel(rf'$\Gamma$ ({g_unit})')
    ax.legend(frameon=False, fontsize=9)
    return PlotHandles(fig, ax, artists)


def plot_concentration_extrapolation(result, ax: Optional[Any] = None) -> PlotHandles:
    """Plot apparent D vs concentration with the extrapolation to infinite dilution."""
    fig, ax, created = _get_ax(ax)
    cfac, c_unit = display_factor('concentration'), display_unit('concentration')
    dfac, d_unit = display_factor('diffusion'), display_unit('diffusion')
    c = np.asarray(result.concentrations_g_per_mL, dtype=float)
    d = np.asarray(result.d_values_m2_s, dtype=float)
    artists = {}
    artists['data'] = ax.scatter(c * cfac, d * dfac, s=36, color=PALETTE['blue'],
                                 zorder=3, label='apparent D')
    xline = np.linspace(0.0, c.max() * 1.05, 50)
    artists['fit'] = ax.plot(xline * cfac, (result.slope * xline + result.d0_m2_s) * dfac,
                             '-', color=PALETTE['vermilion'], lw=1.6,
                             label=f'$D_0$ = {result.d0_m2_s:.3e} m$^2$/s')[0]
    artists['d0'] = ax.plot(0.0, result.d0_m2_s * dfac, 'o', color=PALETTE['vermilion'],
                            ms=10, mfc='none', mew=2)[0]
    ax.set_xlabel(rf'Concentration ({c_unit})')
    ax.set_ylabel(rf'$D_{{app}}$ ({d_unit})')
    ax.legend(frameon=False, fontsize=9,
              title=f'$k_D$ = {result.kd_mL_per_g:.1f} mL/g')
    return PlotHandles(fig, ax, artists)


# ===========================================================================
# SLS plots
# ===========================================================================

def plot_rayleigh_ratio(result, ax: Optional[Any] = None,
                        color: Optional[str] = None,
                        label: Optional[str] = None) -> PlotHandles:
    """Plot the excess Rayleigh ratio dR vs q^2 for one concentration."""
    fig, ax, created = _get_ax(ax)
    c = color or _next_color(ax)
    q2 = np.asarray(result.q2_nm2, dtype=float)
    dR = np.asarray(result.excess_rayleigh_cm_inv, dtype=float)
    lbl = label or f'c = {result.concentration_g_per_mL*1000:.3g} mg/mL'
    artists = {'data': ax.plot(q2, dR, 'o-', color=c, ms=4, lw=1.2, label=lbl)[0]}
    ax.set_xlabel(r'$q^2$ (nm$^{-2}$)')
    ax.set_ylabel(r'$\Delta R$ (cm$^{-1}$)')
    ax.legend(frameon=False, fontsize=9)
    return PlotHandles(fig, ax, artists)


def plot_debye(result, ax: Optional[Any] = None) -> PlotHandles:
    """Plot a single-concentration Debye analysis: Kc/dR vs q^2 (apparent)."""
    fig, ax, created = _get_ax(ax)
    q2 = np.asarray(result.q2_nm2, dtype=float)
    y = np.asarray(result.kc_over_dR, dtype=float)
    artists = {}
    artists['data'] = ax.scatter(q2, y, s=36, color=PALETTE['blue'],
                                 zorder=3, label='data')
    xline = np.linspace(0.0, q2.max() * 1.05, 50)
    artists['fit'] = ax.plot(xline, result.slope * xline + result.intercept_mol_per_g,
                             '-', color=PALETTE['vermilion'], lw=1.6)[0]
    mw_txt = f'{result.mw_apparent_g_per_mol:.3e}'
    ax.set_xlabel(r'$q^2$ (nm$^{-2}$)')
    ax.set_ylabel(r'$Kc/\Delta R$ (mol/g)')
    ax.legend(frameon=False, fontsize=9,
              title=f'apparent $M_w$ = {mw_txt}\napparent $R_g$ = {result.rg_apparent_nm:.1f} nm')
    return PlotHandles(fig, ax, artists)


def plot_guinier(result, ax: Optional[Any] = None) -> PlotHandles:
    """Plot a single-concentration Guinier analysis: ln(dR) vs q^2 (apparent)."""
    fig, ax, created = _get_ax(ax)
    q2 = np.asarray(result.q2_nm2, dtype=float)
    y = np.asarray(result.ln_excess_rayleigh, dtype=float)
    artists = {}
    artists['data'] = ax.scatter(q2, y, s=36, color=PALETTE['blue'],
                                 zorder=3, label='data')
    xline = np.linspace(0.0, q2.max() * 1.05, 50)
    artists['fit'] = ax.plot(xline, result.slope * xline + result.intercept,
                             '-', color=PALETTE['vermilion'], lw=1.6)[0]
    ax.set_xlabel(r'$q^2$ (nm$^{-2}$)')
    ax.set_ylabel(r'$\ln(\Delta R)$')
    ax.legend(frameon=False, fontsize=9,
              title=(f'apparent $R_g$ = {result.rg_nm:.1f} nm   '
                     f'($qR_g^{{\\max}}$ = {result.qrg_max:.2f})'))
    return PlotHandles(fig, ax, artists)


def plot_zimm(
    rayleigh_results: Sequence,
    zimm_result,
    ax: Optional[Any] = None,
    spacing_constant: Optional[float] = None,
) -> PlotHandles:
    """Draw the classic Zimm (or Berry) grid with both extrapolations.

    The ordinate is Kc/dR (Zimm) or its square root (Berry), matching
    `zimm_result.method`. Each concentration's points are spread along the x-axis
    by q^2 + k*c, where k (the spacing constant) separates the curves. The two
    extrapolated series -- to c -> 0 (against q^2) and to q -> 0 (against k*c) --
    are overlaid, meeting near the common intercept (1/Mw, or its root for Berry).

    Parameters
    ----------
    rayleigh_results : sequence of RayleighRatioResult
        One per concentration (the c = 0 reference is skipped).
    zimm_result : ZimmBerryResult
    ax : matplotlib Axes, optional
    spacing_constant : float, optional
        k in q^2 + k*c. If None, chosen so the concentration spread roughly
        matches the q^2 spread.

    Returns
    -------
    PlotHandles
    """
    samples = [r for r in rayleigh_results if r.concentration_g_per_mL != 0]
    samples.sort(key=lambda r: r.concentration_g_per_mL)
    berry = (zimm_result.method == 'berry')

    def ordinate(kc):
        return np.sqrt(kc) if berry else kc

    q2_all = samples[0].q2_nm2
    c_max = max(r.concentration_g_per_mL for r in samples)
    if spacing_constant is None:
        spacing_constant = float(q2_all.max() / c_max) if c_max > 0 else 1.0
    k = spacing_constant

    fig, ax, created = _get_ax(ax, figsize=(7.0, 5.0))
    artists = {'concentrations': [], 'extrapolation_c': None, 'extrapolation_q': None}

    # Per-concentration data, spread along x by k*c.
    for i, r in enumerate(samples):
        col = _CYCLE[i % len(_CYCLE)]
        q2 = np.asarray(r.q2_nm2, dtype=float)
        y = ordinate(np.asarray(r.kc_over_dR_mol_per_g, dtype=float))
        x = q2 + k * r.concentration_g_per_mL
        # Vary the marker SHAPE per concentration so the series stay distinct without
        # colour (WCAG 1.4.1 / greyscale), not hue alone.
        pts = ax.plot(x, y, marker=marker_for(i), ls='none', color=col, ms=5,
                      label=f'{r.concentration_g_per_mL*1000:.3g} mg/mL')[0]
        artists['concentrations'].append(pts)

    # Both extrapolations come from the global plane fit  ord = a + b q^2 + d c, so
    # the c->0 line (vs q^2) and the q->0 line (vs k*c) are straight and meet at the
    # common intercept a (= 1/Mw, or its root for Berry) at x = 0. Continue each line
    # all the way to x = 0 so the extrapolation visibly reaches the axis;
    # the route-specific intercepts stay as markers on the real data range.
    a = float(zimm_result.coef_intercept)
    b = float(zimm_result.coef_q2)
    d = float(zimm_result.coef_c)

    # c -> 0 extrapolated series (intercept per angle), plotted against q^2.
    q2_levels = np.asarray(zimm_result.q2_nm2, dtype=float)
    inter_q = np.asarray(zimm_result.intercept_per_angle, dtype=float)
    xc = np.array([0.0, float(q2_levels.max())])
    ax.plot(xc, a + b * xc, '--', color=PALETTE['black'], lw=1.0,
            label=r'$c \to 0$')
    artists['extrapolation_c'] = ax.plot(
        q2_levels, inter_q, 's', color=PALETTE['black'], ms=5)[0]

    # q -> 0 extrapolated series (intercept per concentration), against k*c.
    conc = np.asarray(zimm_result.concentrations_g_per_mL, dtype=float)
    inter_c = np.asarray(zimm_result.intercept_per_concentration, dtype=float)
    xq = np.array([0.0, float((k * conc).max())])
    ax.plot(xq, a + d * (xq / k), '--', color=PALETTE['vermilion'], lw=1.0,
            label=r'$q \to 0$')   # ord = a + d*c, with x = k*c  ->  c = x/k
    artists['extrapolation_q'] = ax.plot(
        k * conc, inter_c, '^', color=PALETTE['vermilion'], ms=6)[0]

    # The common intercept at q^2 -> 0, c -> 0 (the Mw point) on the y-axis.
    artists['intercept'] = ax.plot([0.0], [a], 'o', color=PALETTE['black'],
                                   ms=7, mfc='none', zorder=5,
                                   label='intercept (1/$M_w$)')[0]

    ax.set_xlabel(rf'$q^2 + k\,c$ (nm$^{{-2}}$;  $k$ = {k:.3g})')
    ylab = r'$\sqrt{Kc/\Delta R}$' if berry else r'$Kc/\Delta R$ (mol/g)'
    ax.set_ylabel(ylab)

    mw_txt = f'{zimm_result.mw_g_per_mol:.3e}'
    title = (f'{zimm_result.method.upper()}: '
             f'$M_w$={mw_txt} g/mol, '
             f'$R_g$={zimm_result.rg_nm:.1f} nm, '
             f'$A_2$={zimm_result.a2_mol_mL_per_g2:.2e}')
    ax.set_title(title, fontsize=9)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    return PlotHandles(fig, ax, artists)


def plot_calibration_free_a2(result, ax: Optional[Any] = None) -> PlotHandles:
    """Plot the calibration-free A2 construction: Y(c) vs c."""
    fig, ax, created = _get_ax(ax)
    c = np.asarray(result.concentrations_g_per_mL, dtype=float)
    Y = np.asarray(result.Y, dtype=float)
    artists = {}
    artists['data'] = ax.scatter(c, Y, s=36, color=PALETTE['blue'],
                                 zorder=3, label='data')
    xline = np.linspace(0.0, c.max() * 1.05, 50)
    artists['fit'] = ax.plot(xline, result.slope * xline + result.intercept, '-',
                             color=PALETTE['vermilion'], lw=1.6)[0]
    ax.set_xlabel(r'Concentration (g/mL)')
    ax.set_ylabel(r'$Y(c) = [I(c_{ref})/c_{ref}]\,/\,[I(c)/c]$')
    ax.legend(frameon=False, fontsize=9,
              title=f'$2A_2M_w$ = {result.two_a2_mw:.3e}')
    return PlotHandles(fig, ax, artists)


# ===========================================================================
# Utilities plots
# ===========================================================================

def plot_i_sin_theta(result, ax: Optional[Any] = None) -> PlotHandles:
    """Plot I·sin(θ) vs angle for one or more SLS measurements (overlaid).

    `result` is a utilities.ISinThetaResult. For an ideal isotropic, dust-free
    scattering volume the curve is flat across angle; curvature or asymmetry about
    90° flags an alignment/stray-light/dust problem. `result.mode` ('absolute' or
    'normalized') sets the y-axis label — the normalized mode (each curve over its
    own mean) overlays curves of different scale for shape comparison.
    """
    fig, ax, created = _get_ax(ax)
    artists = {}
    for curve in result.curves:
        color = _next_color(ax)
        artists[curve.label] = ax.plot(
            curve.angles_deg, curve.i_sin_theta, 'o-', color=color,
            ms=4, lw=1.2, label=curve.label)[0]
    ax.set_xlabel(r'Scattering angle $\theta$ (°)')
    ax.set_ylabel(r'$I\,\sin\theta$ (normalized)' if result.mode == 'normalized'
                  else r'$I\,\sin\theta$ (a.u.)')
    if result.curves:
        ax.legend(frameon=False, fontsize=8)
    return PlotHandles(fig, ax, artists)


def plot_intensity_trace(trace, mode: str = 'absolute',
                         baseline_cps: Optional[float] = None,
                         ax: Optional[Any] = None) -> PlotHandles:
    """Plot a count-rate trace (utilities/IntensityTrace) vs time.

    `mode` 'absolute' shows the raw count rate (cps), with the baseline drawn as a
    dashed line if `baseline_cps` is given; 'relative' shows the count rate divided
    by the baseline (fluctuating about 1.0), which needs `baseline_cps`. The
    relative view makes drift and dust spikes easy to compare across traces of
    different intensity.
    """
    fig, ax, created = _get_ax(ax)
    ifac, i_unit = display_factor('intensity'), display_unit('intensity')
    t = np.asarray(trace.times_s, dtype=float)
    cr = np.asarray(trace.count_rates_cps, dtype=float)
    artists = {}
    if mode == 'relative' and baseline_cps:
        artists['trace'] = ax.plot(t, cr / baseline_cps, '-',
                                   color=PALETTE['blue'], lw=0.8)[0]
        artists['baseline'] = ax.axhline(
            1.0, color=PALETTE['vermilion'], lw=1.0, ls='--')
        ax.set_ylabel('Count rate / baseline')
    else:
        artists['trace'] = ax.plot(t, cr * ifac, '-', color=PALETTE['blue'], lw=0.8)[0]
        if baseline_cps:
            artists['baseline'] = ax.axhline(
                baseline_cps * ifac, color=PALETTE['vermilion'], lw=1.0, ls='--',
                label='baseline')
            ax.legend(frameon=False, fontsize=8)
        ax.set_ylabel(f'Count rate ({i_unit})')
    ax.set_xlabel('Time (s)')
    return PlotHandles(fig, ax, artists)


def plot_count_rate_histogram(result, ax: Optional[Any] = None) -> PlotHandles:
    """Count-rate histogram with optional Gaussian/Poisson overlays (a
    trace_analysis.HistogramFitResult). The legend title reports the Fano factor
    (variance/mean in count space): ~1 for ideal shot noise, >1 for excess
    fluctuations (slow modes, dust). Each overlay's legend entry carries its
    reduced chi-square (chi^2_r ~ 1 = a good fit) when the fit was computed."""
    fig, ax, created = _get_ax(ax)
    ifac, i_unit = display_factor('intensity'), display_unit('intensity')
    centers = np.asarray(result.bin_centers, dtype=float) * ifac
    counts = np.asarray(result.counts, dtype=float)
    width = (float(result.bin_edges[1] - result.bin_edges[0]) * ifac
             if np.asarray(result.bin_edges).size > 1 else None)
    artists = {'hist': ax.bar(centers, counts, width=width, align='center',
                              color=PALETTE['sky'], alpha=0.7)}
    if result.gaussian_curve is not None:
        g_chi2 = result.gaussian_chi2_reduced
        g_label = ('Gaussian' if g_chi2 is None
                   else f'Gaussian ($\\chi^2_r$ = {g_chi2:.2f})')
        artists['gaussian'] = ax.plot(centers, result.gaussian_curve, '-',
                                      color=PALETTE['vermilion'], lw=1.4,
                                      label=g_label)[0]
    if result.poisson_curve is not None:
        p_chi2 = result.poisson_chi2_reduced
        p_label = ('Poisson' if p_chi2 is None
                   else f'Poisson ($\\chi^2_r$ = {p_chi2:.2f})')
        artists['poisson'] = ax.plot(centers, result.poisson_curve, '--',
                                     color=PALETTE['green'], lw=1.4,
                                     label=p_label)[0]
    ax.set_xlabel(f'Count rate ({i_unit})')
    ax.set_ylabel('Frequency')
    ax.legend(frameon=False, fontsize=8,
              title=f'Fano = {result.fano_factor:.2f}')
    return PlotHandles(fig, ax, artists)


def plot_block_variance(result, ax: Optional[Any] = None) -> PlotHandles:
    """Standard error of the mean vs block size (a trace_analysis.BlockVarianceResult).
    A flat curve = uncorrelated (white) noise; a rise that plateaus signals
    positive correlations (slow modes). The legend reports the SE ratio and the
    verdict."""
    fig, ax, created = _get_ax(ax)
    artists = {'se': ax.plot(result.block_sizes, result.standard_errors, 'o-',
                             color=PALETTE['blue'], ms=4, lw=1.2)[0]}
    ax.set_xscale('log')
    ax.set_xlabel('Block size (points)')
    ax.set_ylabel('Std error of the mean')
    verdict = ('correlations detected' if result.correlations_detected
               else 'no correlations')
    ax.legend([artists['se']],
              [f'SE ratio {result.se_ratio:.2f} — {verdict}'],
              frameon=False, fontsize=8)
    return PlotHandles(fig, ax, artists)


def plot_synthetic_correlogram(result, ax: Optional[Any] = None) -> PlotHandles:
    """Plot a generated synthetic correlogram (utilities.SyntheticCorrelogramResult).

    The curve is shown vs log delay time; the legend reports the ground-truth
    effective Rh and PDI an ideal cumulant fit should recover.
    """
    fig, ax, created = _get_ax(ax)
    tau_d, t_unit = _disp('time', result.delay_times_s)
    artists = {'curve': ax.plot(tau_d, result.signal, '-',
                                color=PALETTE['blue'], lw=1.4)[0]}
    ax.set_xscale('log')
    ax.set_xlabel(rf'Delay time $\tau$ ({t_unit})')
    ylabel = {'g2m1': r'$g_2(\tau) - 1$', 'g2': r'$g_2(\tau)$',
              'g1': r'$g_1(\tau)$'}.get(result.output_form, 'signal')
    ax.set_ylabel(ylabel)
    ax.legend([artists['curve']],
              [f'$R_h^{{eff}}$ = {result.rh_eff_nm:.1f} nm,  PDI = {result.pdi:.3f}'],
              frameon=False, fontsize=9)
    return PlotHandles(fig, ax, artists)


def plot_synthetic_multi_dls(dls, ax: Optional[Any] = None) -> PlotHandles:
    """Overlay the generated correlograms of a multi-angle DLS set, one per angle.

    `dls` is a synthetic_dataset.MultiAngleDLS (raw generated data). This is the
    raw-data preview — the angular Γ-vs-q² analysis is what you get by loading the
    saved .ASC into the DLS tab, not something drawn here.
    """
    fig, ax, created = _get_ax(ax)
    artists = {}
    delay, t_unit = _disp('time', dls.delay_times_s)
    for a in dls.angles_deg:
        col = _next_color(ax)
        artists[a] = ax.plot(delay, np.asarray(dls.signals[a], dtype=float), '-',
                             color=col, lw=1.0, label=f'{a:g}°')[0]
    ax.set_xscale('log')
    ax.set_xlabel(rf'Delay time $\tau$ ({t_unit})')
    ax.set_ylabel(r'$g_2(\tau) - 1$')
    ax.legend(frameon=False, fontsize=7, ncol=2,
              title=f'{len(dls.angles_deg)} angles')
    return PlotHandles(fig, ax, artists)


def plot_synthetic_sls_set(sls, ax: Optional[Any] = None) -> PlotHandles:
    """Plot a generated SLS set as intensity vs angle, one curve per concentration.

    `sls` is a synthetic_dataset.SyntheticSLSSet (raw generated intensities,
    including the c = 0 solvent reference). The Zimm/Debye/Guinier analysis comes
    from loading the saved Brookhaven file into the SLS tab; this is just the data.
    """
    fig, ax, created = _get_ax(ax)
    artists = {}
    angles = np.asarray(sls.angles_deg, dtype=float)
    for c in sls.concentrations_g_per_mL:
        col = _next_color(ax)
        label = 'solvent (c = 0)' if c == 0 else f'{c * 1e3:g} mg/mL'
        marker = 'o-' if angles.size > 1 else 'o'
        artists[c] = ax.plot(angles, np.asarray(sls.intensities[c], dtype=float),
                             marker, color=col, ms=4, lw=1.0, label=label)[0]
    ax.set_xlabel(r'Scattering angle $\theta$ (°)')
    ax.set_ylabel('Intensity (a.u.)')
    cal = 'calibrated' if sls.calibrated else 'uncalibrated'
    ax.legend(frameon=False, fontsize=7,
              title=f'{len(sls.concentrations_g_per_mL)} conc · {cal}')
    return PlotHandles(fig, ax, artists)


# ===========================================================================
# Cross-sample plots
# ===========================================================================

def plot_scaling(result, quantity: str = 'rg', labels: Optional[Sequence[str]] = None,
                 ax: Optional[Any] = None) -> PlotHandles:
    """Log-log scaling plot from a utilities.ScalingResult (Rg-Mw or A2-Mw).

    `result` is a ScalingResult (its x = Mw, y = Rg or A2 are the positive points
    actually fitted). `quantity` ('rg'|'a2') only sets the y-axis label and the
    exponent symbol. `labels` (optional, aligned with result.x/result.y) annotate
    the points with their sample names. The power-law fit line is drawn only when
    the fit is valid (>= 2 points). No show()/save -- the caller owns the figure.
    """
    fig, ax, created = _get_ax(ax)
    mfac = display_factor('molar_mass')
    yfac = display_factor('radius') if quantity in ('rg', 'rh') else 1.0
    x = np.asarray(result.x, dtype=float)
    y = np.asarray(result.y, dtype=float)
    artists = {}
    artists['data'] = ax.scatter(x * mfac, y * yfac, s=42, color=PALETTE['blue'],
                                 zorder=3)
    if result.fit_valid and x.size:
        xline = np.geomspace(x.min(), x.max(), 50)
        sym = r'\nu' if quantity in ('rg', 'rh') else '-a'
        artists['fit'] = ax.plot(
            xline * mfac, (result.prefactor * xline ** result.exponent) * yfac, '-',
            color=PALETTE['vermilion'], lw=1.6,
            label=f'${sym}$ = {result.exponent:.3f}  ($R^2$ = {result.r_squared:.3f})')[0]
        ax.legend(frameon=False, fontsize=9)
    # Scaling plots are ALWAYS log-log (power laws plot as straight lines). Log
    # axes need positive limits, so when there are no points yet (e.g. a sample
    # without a computed Mw) we set representative positive limits to avoid the
    # "non-positive limits on a log axis" warning while still showing log-log.
    ax.set_xscale('log')
    ax.set_yscale('log')
    if not x.size:
        ax.set_xlim(1e4 * mfac, 1e7 * mfac)              # representative Mw range
        ax.set_ylim((1.0 * yfac, 1e3 * yfac) if quantity in ('rg', 'rh')  # Rg/Rh ~ nm
                    else (1e-5, 1e-3))                  # A2 ~ mol*mL/g^2
    mw_unit, r_unit = display_unit('molar_mass'), display_unit('radius')
    ax.set_xlabel(rf'$M_w$ ({mw_unit})')
    ax.set_ylabel({'rg': rf'$R_g$ ({r_unit})', 'rh': rf'$R_h$ ({r_unit})'}.get(
        quantity, r'$A_2$ (mol·mL/g$^2$)'))
    if labels is not None:
        # Stagger sample-name labels so close points don't collide; a
        # crowded corner caps at "+N more" rather than piling up. Done LAST, after the
        # log scales + limits are set, so the de-collision sees the final transData.
        annotate_decollided(
            ax, [(xi * mfac, yi * yfac, lb, '#555') for xi, yi, lb in zip(x, y, labels, strict=True)],
            fontsize=7)
    return PlotHandles(fig, ax, artists)


def plot_ddls(result, ax: Optional[Any] = None) -> PlotHandles:
    """Depolarized DLS: field decay rate Gamma vs q^2 for the VV and VH channels.

    The polarized (VV) points fall on a line through the origin with slope D_t
    (Gamma_VV = q^2 D_t); the depolarized (VH) points lie on a parallel line offset
    up by the rotational intercept 6 D_r (Gamma_VH = q^2 D_t + 6 D_r). The fitted
    lines (from the result's D_t and D_r) and the 6 D_r intercept are drawn, so the
    translational/rotational separation is visible at a glance.

    Parameters
    ----------
    result : DDLSResult (analysis.depolarization.analyze_ddls)
    ax : matplotlib Axes, optional

    Returns
    -------
    PlotHandles
    """
    fig, ax, _ = _get_ax(ax)
    qfac, q_unit = display_factor('scattering_q2'), display_unit('scattering_q2')
    gfac, g_unit = display_factor('decay_rate'), display_unit('decay_rate')
    q2 = np.asarray(result.q2_m2, dtype=float)
    artists: Dict[str, Any] = {}

    # measured points
    artists['vv_points'] = ax.scatter(
        q2 * qfac, np.asarray(result.gamma_vv_s_inv, float) * gfac,
        color=PALETTE['blue'], zorder=3, label=r'$\Gamma_{VV}$ (polarized)')
    artists['vh_points'] = ax.scatter(
        q2 * qfac, np.asarray(result.gamma_vh_s_inv, float) * gfac,
        marker='s', color=PALETTE['vermilion'], zorder=3,
        label=r'$\Gamma_{VH}$ (depolarized)')

    # fitted lines from D_t (slope) and 6 D_r (VH intercept), spanning 0..max q^2
    q2_line = np.linspace(0.0, float(q2.max()) * 1.05, 50)
    d_t, d_r = result.d_t_m2_s, result.d_r_rad2_s
    artists['vv_fit'] = ax.plot(
        q2_line * qfac, q2_line * d_t * gfac, '-', color=PALETTE['blue'], lw=1.3,
        label=r'$q^2 D_t$')[0]
    artists['vh_fit'] = ax.plot(
        q2_line * qfac, (q2_line * d_t + 6.0 * d_r) * gfac, '-',
        color=PALETTE['vermilion'], lw=1.3, label=r'$q^2 D_t + 6 D_r$')[0]
    artists['intercept'] = ax.axhline(
        6.0 * d_r * gfac, ls='--', color=PALETTE['black'], lw=0.8,
        label=fr'$6 D_r = {6*d_r:.2e}\ \mathrm{{s^{{-1}}}}$')

    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(rf'$q^2$ ({q_unit})')
    ax.set_ylabel(rf'$\Gamma$ ({g_unit})')
    ax.set_title('Depolarized DLS: decay rate vs $q^2$')
    ax.legend(fontsize=8)
    return PlotHandles(fig, ax, artists)
