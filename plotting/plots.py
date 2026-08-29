"""
plotting/plots.py
=================

Matplotlib plotting for the DLS and SLS result objects.

This is the visualization layer. It is built before, and independently of, any
GUI, and is designed to drop straight into one later. Every function follows the
same contract:

  - It accepts an optional `ax`. If none is given, it creates a figure and axes;
    if one is given, it draws onto it (so several analyses can share axes and be
    overlaid -- NNLS vs CONTIN, Zimm vs Berry). Overlaying: the SINGLE-series
    plotters take `series_index`/`color` to key each call; the MULTI-series
    plotters (form factors, Zimm, the synthetic-set previews, I·sinθ) key their
    per-series colours from index 0 per call — pass `series_offset` (e.g. the
    previous call's series count) on the second result so its series CONTINUE
    the cycle instead of repeating the first result's colours.
  - It NEVER calls plt.show() or saves a file. The caller (a script, or the GUI)
    decides what to do with the figure.
  - It returns a `PlotHandles` carrying the figure, the axes, and a dict of the
    individual artists. The dict is a **scripting/test affordance**: the shipped
    GUI redraws by re-calling the plot function on its cached result (analyses
    are never re-run for a redraw), so nothing in `gui/` consumes the artists —
    the regression suite and scripts do.

Artist-key convention (uniform since spec 07-23-engineering-03):

  - Fixed, documented string keys name each artist's role ('data', 'fit',
    'curve', 'optimal', ...). Never dynamic/format-string keys — a consumer
    must not have to reconstruct a format string to index the dict.
  - Overlay families (one artist per concentration / angle / labelled curve)
    use ``artists['series']``: a list in draw order, one dict per series,
    carrying the identifying value ('concentration_g_per_mL', 'angle_deg',
    'label') plus the artist roles ('data'; 'fit', or 'fits' as a per-solution
    list for the degenerate-solution overlays).

Display-unit boundary (policy as of spec engineering-04 stage 2; the split is
pinned per plot function by
``tests/test_plots.py::test_display_unit_split_is_pinned``):

  - DLS, trace, cross-sample AND SLS analysis plots scale their drawn arrays
    through ``PLOT_UNITS`` (`set_plot_units` / `_disp` / `display_factor` /
    `display_factor_from`): time, intensity, scattering_q2 (Γ-q², DDLS, and
    the SLS q² abscissas — Rayleigh ratio, Debye, Guinier, the form factors,
    Zimm), decay_rate, radius, diffusion, concentration (D-vs-c and the
    calibration-free A₂ abscissa), molar_mass.
  - The SLS result fields keep nm-based storage (``q2_nm2``, ``q_nm_inv``) BY
    DECISION (stage 2's canonical-unit call, owner-signed 2026-07-24): the
    plotters convert from that declared source unit via `display_factor_from`
    (→ ``app.units.convert``), whose same-unit factor is exactly 1.0 — so on
    the default nm⁻² choice the drawn q² arrays are bit-identical to the
    unscaled draw. A full-SI field migration remains open as its own spec.
  - `plot_zimm`'s abscissa q² + k·c is a q²-unit quantity: it is computed
    canonically and the composite scaled by the q² factor; the printed spacing
    constant k is shown in (active q² unit)·mL/g and the concentration unit
    choice never touches the Zimm plot.
  - Fixed-canonical by policy: the SLS ordinates (Kc/ΔR in mol/g, ΔR in cm⁻¹,
    ln ΔR — no display quantity exists for them), `plot_i_sin_theta`, and the
    synthetic-set previews (generator previews, not analysis reads).
  - Legend concentration labels always read mg/mL (``conc_mg_ml_label``),
    independent of the axis-unit choice.

Deliberate signature exceptions (everything else is `plot_*(result, ax=None)`):

  - `plot_correlogram_scaled` takes raw (tau, g2m1[, fit]) arrays, not a result
    object — it is the DLS multi-view/overlay path drawing arbitrary series.
  - `plot_zimm` takes two inputs (the per-concentration Rayleigh results AND
    the Zimm/Berry fit) because the grid is drawn from both.

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
2026-07-24  spec engineering-02/03: shared form-factor overlay machinery; one
            artist-key convention ('series' lists, no dynamic keys); _get_ax
            returns (fig, ax); plot_correlogram_scaled ax keyword-normalized;
            export-white policy collapsed into save_figure_via.
2026-07-24  spec engineering-04 stage 2: the SLS q² abscissas + the
            calibration-free A₂ concentration axis honour PLOT_UNITS
            (display_factor_from / zimm_spacing); boundary section above
            rewritten to the signed-off policy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from analysis import form_factor as ff
from analysis.dls import distribution_axis
from physics import constants as phys


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
    until redrawn. Unknown names fall back to the Okabe-Ito default. Both PALETTE
    and _CYCLE are mutated IN PLACE (one discipline, spec engineering-04), so every
    module-internal reference follows the swap."""
    chosen = _PALETTES.get(name, _PALETTES['okabe_ito'])
    PALETTE.clear()
    PALETTE.update(chosen)
    _CYCLE[:] = [PALETTE[k] for k in _CYCLE_KEYS]


def cycle() -> list:
    """A snapshot of the CURRENT overlay color cycle (reflects the active
    `set_palette`). The returned list is a copy — a live consumer should take this
    function itself (e.g. `SelectionModel(color_cycle=cycle)`) and call it per
    redraw, so a palette switch reaches it."""
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


def marker_cycle() -> list:
    """Snapshot of the per-series marker cycle (index-aligned with `cycle()`).

    Accessor mirroring `cycle()` so consumers (e.g. the GUI's SelectionModel)
    never import the private constants (spec engineering-04)."""
    return list(_MARKER_CYCLE)


def linestyle_cycle() -> list:
    """Snapshot of the per-series linestyle cycle (index-aligned with `cycle()`)."""
    return list(_LINESTYLE_CYCLE)


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


def _theme_rcparams(p) -> dict:
    """The process-global matplotlib rcParams this module writes, valued from
    figure palette `p` — THE declaration of plotting's rcParams blast radius.
    `THEME_RCPARAMS_KEYS` (below) publishes the key set so tests snapshot and
    restore exactly this state and nothing can silently join it."""
    return {
        'figure.facecolor': p['fig'], 'axes.facecolor': p['ax'],
        'axes.edgecolor': p['chrome'], 'axes.labelcolor': p['chrome'],
        'axes.titlecolor': p['chrome'], 'xtick.color': p['chrome'],
        'ytick.color': p['chrome'], 'xtick.labelcolor': p['chrome'],
        'ytick.labelcolor': p['chrome'], 'text.color': p['chrome'],
        'grid.color': p['grid'], 'legend.facecolor': p['fig'],
        'legend.edgecolor': p['chrome'],
        'savefig.facecolor': 'white', 'savefig.edgecolor': 'white',
    }


# The ONLY rcParams keys plotting owns (consumed by tests/conftest.py's autouse
# style-state fixture). Derived from the writer itself, so the two cannot drift.
THEME_RCPARAMS_KEYS = tuple(_theme_rcparams(_LIGHT_FIG))


def set_onscreen_plot_theme(dark: bool) -> None:
    """Set the on-screen plot chrome theme via rcParams, so figures drawn/redrawn
    afterwards inherit it with NO change to the plot functions (they clear + recreate
    axes, which pick up the rcParams). Export stays white regardless (savefig.facecolor).
    Call `apply_figure_theme` on already-drawn canvases for an immediate refresh."""
    global _ONSCREEN_DARK
    _ONSCREEN_DARK = bool(dark)
    _mpl.rcParams.update(_theme_rcparams(_DARK_FIG if dark else _LIGHT_FIG))


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


def save_figure_via(fig, do_save) -> None:
    """Run `do_save()` with `fig` recoloured WHITE, then restore the on-screen theme.

    THE single implementation of the locked export policy (style guide R10.3:
    exports must be clean/print-parity regardless of the on-screen theme). Both
    consumers delegate here so the policy cannot fork again:
    `save_figure_white` (the scripting API) and the GUI's Save-button toolbar
    (`gui.plot_controls._WhiteSaveToolbar` — the path a user actually hits).
    `do_save` is any callable that writes the figure (a savefig, or matplotlib's
    own save dialog)."""
    apply_figure_theme(fig, dark=False)
    try:
        do_save()
    finally:
        apply_figure_theme(fig, dark=_ONSCREEN_DARK)
        if fig.canvas is not None:
            fig.canvas.draw_idle()


def save_figure_white(fig, path: str) -> str:
    """Save `fig` to `path` on a WHITE background regardless of the on-screen theme
    (thin wrapper over `save_figure_via`, the single policy implementation)."""
    save_figure_via(
        fig, lambda: fig.savefig(path, facecolor='white', edgecolor='white'))
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


def display_factor_from(quantity: str, source_unit: str) -> float:
    """Multiplier converting a value stored in `source_unit` to the active
    display unit — for arrays whose declared storage unit is NOT the canonical
    one: the SLS result fields keep q² in nm⁻² by decision (spec
    engineering-04 stage 2), so their axes route through here instead of
    `display_factor`. When the active unit IS the source unit the factor is
    exactly 1.0 (`app.units.convert`'s identity guarantee), which is what
    makes default-unit SLS rendering bit-identical to the unscaled draw."""
    unit = PLOT_UNITS.get(quantity, _U.canonical_unit(quantity))
    return float(_U.convert(quantity, 1.0, source_unit, unit))


def zimm_spacing(rayleigh_results) -> float:
    """The default Zimm spacing constant k in q² + k·c, in canonical
    nm⁻²·mL/g: the lowest concentration's max measured q² over the max
    concentration, so the concentration spread roughly matches the q² spread.

    The single home for the constant (spec engineering-04 stage 2):
    `plot_zimm` and the GUI's masked-point overlay / click hit-testing all
    read it from here, so the plotted and hit-tested coordinates cannot
    drift apart. The c = 0 reference is skipped, like everywhere else; a set
    with no c > 0 sample (only a blank loaded) has no spread to set, so k
    falls back to 1.0 — this guard lives HERE so every caller shares it."""
    samples = [r for r in rayleigh_results if r.concentration_g_per_mL != 0]
    if not samples:
        return 1.0
    lowest = min(samples, key=lambda r: r.concentration_g_per_mL)
    c_max = max(r.concentration_g_per_mL for r in samples)
    q2_max = float(np.asarray(lowest.q2_nm2, dtype=float).max())
    return q2_max / c_max if (math.isfinite(c_max) and c_max > 0) else 1.0


def conc_mg_ml_label(c_g_per_mL: float) -> str:
    """Legend text for a concentration: canonical g/mL rendered as mg/mL.

    Display-only convenience, named once (spec engineering-04; previously seven
    inline `c*1000` copies — six here, one in `gui/sls_module.py`, which imports
    this PUBLIC helper): legends read a FIXED mg/mL regardless of PLOT_UNITS's
    'concentration' choice (which governs AXES) — stated policy as of stage 2,
    so a legend stays comparable across sessions whatever the axis units."""
    return f'{c_g_per_mL * 1000:.3g} mg/mL'


@dataclass
class PlotHandles:
    """What a plot function returns.

    figure, axes : the matplotlib Figure and Axes drawn on.
    artists      : named artists (lines, collections, ...) under the module's
                   uniform key convention (fixed role names + a 'series' list
                   for overlay families — see the module docstring). A
                   scripting/test affordance; the GUI redraws from cached
                   results instead of retaining these. Includes any secondary
                   axes (e.g. a residual panel) under a descriptive key.
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
    """Return (figure, axes). Create a registry-free figure only if ax is None.

    (A third `created` flag was returned until spec engineering-03 removed it:
    it was unpacked at every call site and read at none of them.)"""
    if ax is None:
        fig = _new_figure(figsize)
        ax = fig.subplots()
        return fig, ax
    return ax.figure, ax


def _next_color(ax) -> str:
    """Pick a palette color not obviously already in use on this axes.

    Heuristic: it counts EVERY line and collection as a "series", so fit lines,
    `fill_between` fills and `axhline`s advance the cycle too — adding a
    decorative artist to a plot function shifts every later overlay's colour.
    Overlay call sites should prefer an explicit `series_index`
    (`_series_color`), which is immune to that; this stays as the
    no-index fallback."""
    used = len(ax.lines) + len(ax.collections)
    return _CYCLE[used % len(_CYCLE)]


def _series_color(ax, series_index: Optional[int]) -> str:
    """Index-keyed overlay colour when the caller supplies `series_index` —
    stable against decorative-artist churn, and aligned with
    `marker_for`/`linestyle_for` so series N keeps one (colour, marker,
    linestyle) triple (WCAG 1.4.1). Falls back to `_next_color`'s
    artist-count heuristic when no index is given."""
    if series_index is None:
        return _next_color(ax)
    return _CYCLE[series_index % len(_CYCLE)]


# Shared fit-overlay geometry. Every straight fit line spans the data with the
# same padding and sampling, and every curved P(q) model overlay shares one grid
# density — named once so a styling change reaches the whole family instead of
# whichever plotter happened to be edited (engineering review 2026-07-23).
_FIT_XMAX_PAD = 1.05        # fit line / model grid extends to x_max * this
_FIT_LINE_POINTS = 50       # samples along a straight fit line
_MODEL_CURVE_POINTS = 200   # samples along a curved P(q) model overlay


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
    series_index: Optional[int] = None,
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
    series_index : int, optional
        Overlay slot for a stable, index-keyed colour (see `_series_color`).

    Returns
    -------
    PlotHandles
    """
    tau = np.asarray(result.fit_tau_s, dtype=float)
    fit = np.asarray(result.fitted_g2m1, dtype=float)
    resid = np.asarray(result.residuals, dtype=float)
    data = fit + resid

    if ax is None and show_residuals:
        fig = _new_figure((6.0, 5.4))
        ax, residual_ax = fig.subplots(
            2, 1, sharex=True,
            gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.08})
    else:
        fig, ax = _get_ax(ax)

    c = color or _series_color(ax, series_index)
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
    tail (see `_LOG_FLOOR`). Its only caller today is `plot_correlogram_scaled`
    (both its data and fit paths); six other plotters draw computed curves on
    log axes UNGUARDED, and whether they should adopt this mask changes which
    points get drawn -- an open behaviour question recorded for the owner (spec
    engineering-03), deliberately not prescribed here. NOTE it filters
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
    tau,
    g2m1,
    fit_tau=None,
    fit_g2m1=None,
    ax: Optional[Any] = None,
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
    fig, ax = _get_ax(ax)
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
    series_index: Optional[int] = None,
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
    fig, ax = _get_ax(ax)
    c = color or _series_color(ax, series_index)
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


def plot_weighted_distribution(
    result,
    ax: Optional[Any] = None,
    color: Optional[str] = None,
    label: Optional[str] = None,
    fill: bool = True,
    linestyle: str = '-',
    invalid_label: Optional[str] = 'outside model validity',
    series_index: Optional[int] = None,
) -> PlotHandles:
    """Plot a reweighted (volume/surface/number) distribution vs Rh, marking the size
    bins that fall OUTSIDE the chosen sphere model's validity range.

    Always plotted against Rh (the weighting is a size re-expression). Out-of-range bins
    are overlaid: a light red band spans their Rh region and red markers sit on the curve
    there -- the GUI-owned validity overlay (never baked into export data; the flag lives
    on the figure only, per the locked GUI decision). The y-label names the weighting so
    the derived view is never mistaken for the measured intensity distribution.

    Parameters
    ----------
    result : WeightedDistributionResult
    ax, color, label, fill, linestyle : styling (overlay-capable, like plot_distribution).
    invalid_label : str or None
        Legend label for the out-of-range markers. Pass None on all but the FIRST overlaid
        curve so the shared "outside model validity" entry isn't repeated in the legend
        (the caller labels it once); avoids a second ``ax.legend()`` dedup pass.
    """
    rh = np.asarray(result.rh_grid_nm, dtype=float)
    w = np.asarray(result.weights, dtype=float)
    valid = np.asarray(result.valid_mask, dtype=bool)
    order = np.argsort(rh)
    rh, w, valid = rh[order], w[order], valid[order]
    x = rh * display_factor('radius')
    fig, ax = _get_ax(ax)
    c = color or _series_color(ax, series_index)
    lbl = label or str(getattr(result, 'weighting', '')).capitalize()

    artists = {}
    artists['line'] = ax.plot(x, w, color=c, ls=linestyle, lw=1.8, label=lbl)[0]
    if fill:
        artists['fill'] = ax.fill_between(x, w, color=c, alpha=0.15)
    # Validity overlay: shade the out-of-range Rh region and dot the invalid bins.
    if valid.size and not valid.all():
        inv = ~valid
        artists['invalid'] = ax.plot(
            x[inv], w[inv], linestyle='none', marker='x', ms=5,
            color='#c0392b', label=invalid_label)[0]
        # A faint red band over the contiguous invalid Rh span (usually the large-R tail).
        xi = x[inv]
        if xi.size:
            artists['band'] = ax.axvspan(float(xi.min()), float(xi.max()),
                                         color='#c0392b', alpha=0.07)
    ax.set_xscale('log')
    ax.set_xlabel(rf'$R_h$ ({display_unit("radius")})')
    ax.set_ylabel(f'Weight ({str(getattr(result, "weighting", "")).lower()} fraction)')
    if lbl:
        ax.legend(frameon=False, fontsize=9)
    return PlotHandles(fig, ax, artists)


def plot_lcurve(lcurve, ax: Optional[Any] = None, method: str = 'lcurve',
                color: Optional[str] = None, label: Optional[str] = None,
                mark_ceiling: bool = False, mark_floor: bool = False) -> PlotHandles:
    """Plot a CONTIN alpha-selection sweep, marking the chosen alpha.

    Method-aware ordinate (the abscissa/ordinate that made the pick visible):

      * ``'lcurve'`` — the classic L-curve: solution seminorm vs residual norm
        (log-log), the corner ringed.
      * ``'gcv'``    — the GCV score V(alpha) vs alpha (log-x), the minimum ringed.
      * ``'ftest'``  — the cumulative F ("probability to reject") vs alpha (log-x),
        the chosen alpha ringed.

    Overlay-capable: pass an explicit ``color`` and ``label`` and call once per
    CONTIN result on a shared ``ax`` (the chosen-alpha marker takes the same color,
    so several sweeps stay legible). ``mark_ceiling=True`` / ``mark_floor=True`` ring
    the pick with the same edge-class warning halo (the alpha landed on a sweep edge;
    the wording for each edge is the result's own ``reliability_note``, not repeated
    here). Flags stay GUI-owned overlays: nothing is baked into a saved figure.

    Raises ``ValueError`` when ``method`` is not one of ``{'lcurve','gcv','ftest'}``
    (e.g. ``'user'`` — a fixed/user-supplied alpha has no sweep, only a degenerate
    single-point ``LCurveResult``), or when the requested method's score curve is
    absent (e.g. ``method='gcv'`` on a sweep with ``gcv is None``); the GUI turns
    either into an empty-state rather than a broken/misleading plot.
    """
    if method not in ('lcurve', 'gcv', 'ftest'):
        raise ValueError(f'no alpha-sweep to plot for method {method!r}')
    fig, ax = _get_ax(ax)
    c = color or PALETTE['blue']
    i = lcurve.optimal_index
    artists = {}
    if method == 'gcv':
        score = getattr(lcurve, 'gcv', None)
        if score is None:
            raise ValueError('no GCV score curve on this sweep')
        x = np.asarray(lcurve.alphas, dtype=float)
        y = np.asarray(score, dtype=float)
        xlabel, ylabel = r'Regularization $\alpha$', r'GCV score $V(\alpha)$'
    elif method == 'ftest':
        fc = getattr(lcurve, 'ftest_fc', None)
        if fc is None:
            raise ValueError('no F-test curve on this sweep')
        x = np.asarray(lcurve.alphas, dtype=float)
        y = np.asarray(fc, dtype=float)
        xlabel, ylabel = r'Regularization $\alpha$', 'Cumulative F'
    else:                                             # 'lcurve' (default)
        x = np.asarray(lcurve.residual_norms, dtype=float)
        y = np.asarray(lcurve.solution_norms, dtype=float)
        xlabel = r'Residual norm $\|A x - y\|^2$'
        ylabel = r'Solution seminorm $\|L x\|^2$'
        ax.set_yscale('log')
    artists['curve'] = ax.plot(x, y, '-o', color=c, ms=4, lw=1.2, label=label)[0]
    artists['optimal'] = ax.plot(x[i], y[i], 'o', color=c, ms=11, mfc='none', mew=2,
                                 label=f'optimal alpha = {lcurve.optimal_alpha:.2e}')[0]
    if mark_ceiling:                                  # over-regularization halo
        artists['ceiling'] = ax.plot(x[i], y[i], 'o', color=PALETTE['vermilion'],
                                     ms=17, mfc='none', mew=1.6, ls='none')[0]
    if mark_floor:                                    # floor-edge halo (same glyph:
        artists['floor'] = ax.plot(x[i], y[i], 'o', color=PALETTE['vermilion'],
                                   ms=17, mfc='none', mew=1.6, ls='none')[0]   # one edge class)
    ax.set_xscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=8)
    return PlotHandles(fig, ax, artists)


def plot_gamma_q2(result, ax: Optional[Any] = None) -> PlotHandles:
    """Plot Gamma vs q^2 with the through-origin diffusion fit.

    The data points, the through-origin line (slope = D), and the unconstrained
    intercept are shown. A non-zero intercept flags non-diffusive behavior.
    """
    fig, ax = _get_ax(ax)
    qfac, q_unit = display_factor('scattering_q2'), display_unit('scattering_q2')
    gfac, g_unit = display_factor('decay_rate'), display_unit('decay_rate')
    q2 = np.asarray(result.q2_m2, dtype=float)
    gamma = np.asarray(result.gamma_s_inv, dtype=float)
    # An angle the engine DROPPED (no usable Gamma, so `gamma_s_inv` carries a NaN) is
    # marked with a dotted vertical line at its q^2 -- the sibling
    # `plot_concentration_extrapolation`'s policy: a NaN in a scatter draws nothing, and
    # an omitted point must not look like an angle that was never measured.
    dropped = ~np.isfinite(gamma)
    artists = {}
    artists['data'] = ax.scatter(q2[~dropped] * qfac, gamma[~dropped] * gfac, s=36,
                                 color=PALETTE['blue'], zorder=3, label='data')
    if dropped.any():
        lines = []
        for i, q2_drop in enumerate(q2[dropped]):
            lines.append(ax.axvline(
                q2_drop * qfac, color=PALETTE['black'], ls=':', lw=1.3, alpha=0.55,
                zorder=1, label='no usable Γ (dropped)' if i == 0 else None))
        artists['dropped'] = lines
    # A withheld result (fewer than two usable angles) has no line to draw and no D to
    # name -- printing `D = nan` would be a value where there is none.
    if math.isfinite(result.d_m2_s) and (~dropped).any():
        # through-origin fit line (computed in canonical units, then scaled to display)
        xline = np.linspace(0.0, q2[~dropped].max() * _FIT_XMAX_PAD, _FIT_LINE_POINTS)
        artists['fit'] = ax.plot(xline * qfac, result.d_m2_s * xline * gfac, '-',
                                 color=PALETTE['vermilion'], lw=1.6,
                                 label=f'D = {result.d_m2_s:.3e} m$^2$/s')[0]
    if math.isfinite(result.intercept_s_inv):
        flag = 'diffusive' if result.is_diffusive else 'NON-diffusive'
        # `+ 0.0` after rounding turns a negative zero (-0.004 -> "-0.00") into "0.00".
        rel = round(result.intercept_relative, 2) + 0.0
        artists['intercept'] = ax.axhline(
            result.intercept_s_inv * gfac, color=PALETTE['green'], lw=1.0, ls='--',
            label=f'intercept ({flag}, rel={rel:.2f})')
    ax.set_xlabel(rf'$q^2$ ({q_unit})')
    ax.set_ylabel(rf'$\Gamma$ ({g_unit})')
    ax.legend(frameon=False, fontsize=9)
    return PlotHandles(fig, ax, artists)


def plot_concentration_extrapolation(result, ax: Optional[Any] = None) -> PlotHandles:
    """Plot apparent D vs concentration with the extrapolation to infinite dilution.

    A concentration the engine DROPPED (no usable D, so `d_values_m2_s` carries a NaN)
    is marked with a dotted vertical line at its c, because a NaN in a scatter draws
    nothing at all -- an omitted point is indistinguishable from a concentration that
    was never measured, which is exactly what the drop policy must not look like. This
    is an analysis-determined exclusion, so it belongs in the plotting layer beside the
    data, like `plot_scaling`'s "branched (excluded from fit)" markers -- NOT the
    GUI-owned overlay tier, which is for the user's own tick/untick choices."""
    fig, ax = _get_ax(ax)
    cfac, c_unit = display_factor('concentration'), display_unit('concentration')
    dfac, d_unit = display_factor('diffusion'), display_unit('diffusion')
    c = np.asarray(result.concentrations_g_per_mL, dtype=float)
    d = np.asarray(result.d_values_m2_s, dtype=float)
    dropped = ~np.isfinite(d)
    artists = {}
    artists['data'] = ax.scatter(c[~dropped] * cfac, d[~dropped] * dfac, s=36,
                                 color=PALETTE['blue'], zorder=3, label='apparent D')
    if dropped.any():
        lines = []
        for i, c_drop in enumerate(c[dropped]):
            lines.append(ax.axvline(
                c_drop * cfac, color=PALETTE['black'], ls=':', lw=1.3, alpha=0.55,
                zorder=1, label='no usable D (dropped)' if i == 0 else None))
        artists['dropped'] = lines
    # A withheld result (fewer than two usable concentrations) has no line to draw and
    # no D0 to name -- printing `D_0 = nan` would be a value where there is none.
    if math.isfinite(result.d0_m2_s):
        # The line spans the FITTED concentrations, not every supplied one: drawn out to
        # `c.max()` it would run straight through a dropped concentration and read as
        # though the fit covered it. Identical for a result with no drops.
        xline = np.linspace(0.0, c[~dropped].max() * _FIT_XMAX_PAD, _FIT_LINE_POINTS)
        artists['fit'] = ax.plot(
            xline * cfac, (result.slope * xline + result.d0_m2_s) * dfac,
            '-', color=PALETTE['vermilion'], lw=1.6,
            label=f'$D_0$ = {result.d0_m2_s:.3e} m$^2$/s')[0]
        artists['d0'] = ax.plot(0.0, result.d0_m2_s * dfac, 'o',
                                color=PALETTE['vermilion'], ms=10, mfc='none', mew=2)[0]
    ax.set_xlabel(rf'Concentration ({c_unit})')
    ax.set_ylabel(rf'$D_{{app}}$ ({d_unit})')
    ax.legend(frameon=False, fontsize=9,
              title=(f'$k_D$ = {result.kd_mL_per_g:.1f} mL/g'
                     if math.isfinite(result.kd_mL_per_g) else None))
    return PlotHandles(fig, ax, artists)


def plot_mode_map(points, ax: Optional[Any] = None) -> PlotHandles:
    """The mode map: every extracted mode of every run in (q^2, Gamma) space.

    Base scatter only, in a neutral gray -- which points form a family is the
    USER's pick, and family colours/emphasis are the GUI's pick-state overlay
    tier, never baked into the figure (saved plots stay clean). The y axis
    defaults to log: a fast and a slow mode are typically decades apart, and a
    linear axis would squash the slow family onto the floor of the picking
    surface.

    Two INDEPENDENT judgements ride on the markers, each with its own channel so
    they compose instead of masking one another (ui_style_guide R10.1 -- never a
    colour alone, and here never one channel for two meanings):

      * FILL for amplitude -- hollow means below the ~2 % floor, so the rate and
        amplitude are imprecise;
      * SHAPE for the lag window -- a square means the mode never finished
        decaying inside the measured lags, so its rate is an extrapolation.

    A hollow square is both. Only the combinations actually present are drawn.
    """
    fig, ax = _get_ax(ax)
    qfac, q_unit = display_factor('scattering_q2'), display_unit('scattering_q2')
    gfac, g_unit = display_factor('decay_rate'), display_unit('decay_rate')
    q2 = np.array([p.q2_m2 for p in points], dtype=float)
    gamma = np.array([p.gamma_s_inv for p in points], dtype=float)
    low = np.array([p.low_amplitude for p in points], dtype=bool)
    # getattr, not attribute access: the map is also drawn from RECORDED snapshots and
    # from test doubles built before this flag existed; an absent flag means "no verdict",
    # which draws as an ordinary point.
    undecayed = np.array([getattr(p, 'decay_incomplete', False) for p in points],
                         dtype=bool)
    artists = {}
    if q2.size:
        for key, sel, marker, hollow, label in (
            ('modes', ~low & ~undecayed, 'o', False, 'extracted modes'),
            ('low', low & ~undecayed, 'o', True, 'low-amplitude mode (imprecise)'),
            ('undecayed', ~low & undecayed, 's', False,
             'undecayed mode (rate extrapolated)'),
            ('low_undecayed', low & undecayed, 's', True,
             'low-amplitude AND undecayed'),
        ):
            if not sel.any():
                continue
            style = (dict(facecolors='none', edgecolors='#999999') if hollow
                     else dict(color='#999999'))
            artists[key] = ax.scatter(
                q2[sel] * qfac, gamma[sel] * gfac, s=34, marker=marker,
                zorder=2, label=label, **style)
        ax.set_yscale('log')
    ax.set_xlabel(rf'$q^2$ ({q_unit})')
    ax.set_ylabel(rf'$\Gamma$ ({g_unit})')
    if q2.size:
        ax.legend(frameon=False, fontsize=9)
    return PlotHandles(fig, ax, artists)


def plot_mode_family_fit(result, ax: Optional[Any] = None, *,
                         color=None, marker='o', label='') -> PlotHandles:
    """One picked mode family's points + its through-origin line on the map.

    Draws the family's snapshot points in its colour/marker, the through-origin
    line Gamma = D q^2, and -- when it was fitted -- the q^2+q^4 deviation
    curve dashed. Meant to be layered onto :func:`plot_mode_map`'s axes; the
    caller supplies the family colour/marker (one per family, colour never the
    sole carrier)."""
    fig, ax = _get_ax(ax)
    qfac = display_factor('scattering_q2')
    gfac = display_factor('decay_rate')
    color = color if color is not None else PALETTE['blue']
    q2 = np.asarray(result.q2_m2, dtype=float)
    gamma = np.asarray(result.gamma_s_inv, dtype=float)
    artists = {}
    artists['data'] = ax.scatter(q2 * qfac, gamma * gfac, s=44, color=color,
                                 marker=marker, zorder=3, label=label or None)
    xline = np.linspace(q2.min() * 0.5, q2.max() * _FIT_XMAX_PAD,
                        _FIT_LINE_POINTS)
    artists['fit'] = ax.plot(
        xline * qfac, result.d_m2_s * xline * gfac, '-', color=color, lw=1.4,
        label=(f'{label}: D = {result.d_m2_s:.3e} m$^2$/s' if label else None))[0]
    if math.isfinite(result.a_m2_s) and math.isfinite(result.b_m4_s):
        artists['q4'] = ax.plot(
            xline * qfac,
            (result.a_m2_s * xline + result.b_m4_s * xline * xline) * gfac,
            '--', color=color, lw=1.1, alpha=0.8,
            label=(f'{label}: a·q²+b·q⁴' if label else None))[0]
    ax.legend(frameon=False, fontsize=9)
    return PlotHandles(fig, ax, artists)


def plot_amplitude_vs_q(result, ax: Optional[Any] = None) -> PlotHandles:
    """The fast/slow amplitude report: A_f/A_s vs sin^2(theta/2).

    The literature's own axis for the two-mode amplitude structure. The
    per-angle fractions themselves live in the results table/export; this
    plots the ratio, with the flatness verdict in the legend title. No error
    bars anywhere: single-run amplitudes carry no +- (invariant 8)."""
    fig, ax = _get_ax(ax)
    x = np.asarray(result.sin2_half_theta, dtype=float)
    r = np.asarray(result.ratio_f_over_s, dtype=float)
    artists = {}
    ok = np.isfinite(r)
    artists['ratio'] = ax.plot(x[ok], r[ok], 'o-', color=PALETTE['blue'],
                               ms=7, lw=1.2, label='$A_f/A_s$')[0]
    ax.set_xlabel(r'$\sin^2(\theta/2)$')
    ax.set_ylabel(r'$A_f\,/\,A_s$')
    flat_txt = ('fast amplitude q-flat'
                if result.fast_flat else 'fast amplitude varies with q')
    ax.legend(frameon=False, fontsize=9, title=flat_txt)
    return PlotHandles(fig, ax, artists)


# ===========================================================================
# SLS plots
# ===========================================================================

# An absolute Mw or A2 whose scale is unknown is not a meaningful number, and a figure
# travels further than the tab it was drawn in — a saved PNG carries no results table and no
# below-fold flag with it. So a plotter that prints such a value withholds it and names the
# cause, exactly as the SLS tab dashes the same number beside the plot.
#
# THE one body of that convention for both surfaces (`gui.sls_module` imports these rather
# than keeping a second spelling), so a figure and the readout under it can never disagree
# about the same number. The dependency runs GUI -> plotting, never the reverse.

def scale_causes(res) -> list:
    """Which of a result's two scales are unknown, as user-facing nouns (empty = both
    known). Tolerant getattr so a partially-built stand-in degrades to "scale known"
    rather than raising inside a paint path."""
    causes = []
    if not getattr(res, 'calibrated', True):
        causes.append('uncalibrated')
    if not getattr(res, 'dn_dc_supplied', True):
        causes.append('no usable dn/dc')
    return causes


def scaled_value(res, rendered: str, value: Optional[float] = None) -> str:
    """`rendered` when the value may be printed, else a dash naming what is missing.

    TWO questions, and a caller that asks only the first ships the second's failure. The scale
    one is `scale_causes`. The other is whether there is a number at all: a non-positive
    intercept NaNs a fitted Mw independently of calibration, so a fully calibrated result can
    carry a value that is not a number. Pass `value` and that state dashes as
    `— (no usable value)` — the same words the stored-Mw readout uses for it, so the figure and
    the readout beside it cannot disagree.

    The GUI does not pass `value`: its call sites render through `format_pm` first, which
    already answers the finiteness question (`n/a`). A figure has no such stage, and no
    neighbouring cell to interpret a bare `n/a` once it is saved.

    PLAIN text: it is read by a matplotlib title (no markup) and by the SLS results table,
    whose `RichTextDelegate` covers column 0 only — markup in a VALUE cell would render as
    literal tags on both surfaces.
    """
    causes = scale_causes(res)
    if value is not None and not math.isfinite(value):
        causes.append('no usable value')
    return f"— ({', '.join(causes)})" if causes else rendered


def plot_rayleigh_ratio(result, ax: Optional[Any] = None,
                        color: Optional[str] = None,
                        label: Optional[str] = None,
                        series_index: Optional[int] = None) -> PlotHandles:
    """Plot the excess Rayleigh ratio dR vs q^2 for one concentration."""
    fig, ax = _get_ax(ax)
    c = color or _series_color(ax, series_index)
    qfac = display_factor_from('scattering_q2', 'nm⁻²')
    q2 = np.asarray(result.q2_nm2, dtype=float)
    dR = np.asarray(result.excess_rayleigh_cm_inv, dtype=float)
    lbl = label or f'c = {conc_mg_ml_label(result.concentration_g_per_mL)}'
    artists = {'data': ax.plot(q2 * qfac, dR, 'o-', color=c, ms=4, lw=1.2,
                               label=lbl)[0]}
    ax.set_xlabel(rf'$q^2$ ({display_unit("scattering_q2")})')
    ax.set_ylabel(r'$\Delta R$ (cm$^{-1}$)')
    ax.legend(frameon=False, fontsize=9)
    return PlotHandles(fig, ax, artists)


def plot_debye(result, ax: Optional[Any] = None) -> PlotHandles:
    """Plot a single-concentration Debye analysis: Kc/dR vs q^2 (apparent)."""
    fig, ax = _get_ax(ax)
    qfac = display_factor_from('scattering_q2', 'nm⁻²')
    q2 = np.asarray(result.q2_nm2, dtype=float)
    y = np.asarray(result.kc_over_dR, dtype=float)
    artists = {}
    artists['data'] = ax.scatter(q2 * qfac, y, s=36, color=PALETTE['blue'],
                                 zorder=3, label='data')
    # Fit y is computed on the canonical (nm⁻²) grid — the slope is per nm⁻² —
    # and only the drawn x is scaled to the display unit.
    xline = np.linspace(0.0, q2.max() * _FIT_XMAX_PAD, _FIT_LINE_POINTS)
    artists['fit'] = ax.plot(xline * qfac,
                             result.slope * xline + result.intercept_mol_per_g,
                             '-', color=PALETTE['vermilion'], lw=1.6)[0]
    mw_txt = scaled_value(result, f'{result.mw_apparent_g_per_mol:.3e}',
                          result.mw_apparent_g_per_mol)
    ax.set_xlabel(rf'$q^2$ ({display_unit("scattering_q2")})')
    ax.set_ylabel(r'$Kc/\Delta R$ (mol/g)')
    ax.legend(frameon=False, fontsize=9,
              title=f'apparent $M_w$ = {mw_txt}\napparent $R_g$ = {result.rg_apparent_nm:.1f} nm')
    return PlotHandles(fig, ax, artists)


def plot_guinier(result, ax: Optional[Any] = None) -> PlotHandles:
    """Plot a single-concentration Guinier analysis: ln(dR) vs q^2 (apparent)."""
    fig, ax = _get_ax(ax)
    qfac = display_factor_from('scattering_q2', 'nm⁻²')
    q2 = np.asarray(result.q2_nm2, dtype=float)
    y = np.asarray(result.ln_excess_rayleigh, dtype=float)
    artists = {}
    artists['data'] = ax.scatter(q2 * qfac, y, s=36, color=PALETTE['blue'],
                                 zorder=3, label='data')
    # Fit y on the canonical grid (slope is per nm⁻²); only drawn x is scaled.
    xline = np.linspace(0.0, q2.max() * _FIT_XMAX_PAD, _FIT_LINE_POINTS)
    artists['fit'] = ax.plot(xline * qfac,
                             result.slope * xline + result.intercept,
                             '-', color=PALETTE['vermilion'], lw=1.6)[0]
    ax.set_xlabel(rf'$q^2$ ({display_unit("scattering_q2")})')
    ax.set_ylabel(r'$\ln(\Delta R)$')
    ax.legend(frameon=False, fontsize=9,
              title=(f'apparent $R_g$ = {result.rg_nm:.1f} nm   '
                     f'($qR_g^{{\\max}}$ = {result.qrg_max:.2f})'))
    return PlotHandles(fig, ax, artists)


# ---------------------------------------------------------------------------
# Shared form-factor overlay machinery
# ---------------------------------------------------------------------------

# The drawn model equation  Kc/ΔR = 1/(Mw·P(q)) + 2·A₂·c  is assembled by
# ``analysis.form_factor.model_ordinate`` — the SAME function the fits minimize — and the
# P(q) kernel dispatch by ``analysis.form_factor.model_P``, so the overlay evaluates the
# *fitted* model. Two DISPLAY-ONLY guards live here, once, for every form-factor plotter:
#   * P-floor: the ordinate 1/(Mw·P) diverges where P(q) touches zero, past the first
#     form-factor minimum; P below the floor is NaN so the curve BREAKS there.
#   * y-cap: any finite model value beyond a modest multiple of the DATA range is masked
#     to NaN, so a near-divergence never squashes the scatter.
_FORM_FACTOR_P_FLOOR = 1e-4
_FORM_FACTOR_Y_CAP_MULT = 3.0


def _model_q_grid(q, *, k_nm_inv: Optional[float] = None):
    """The q grid (nm^-1) a model overlay is drawn on: 0 .. q_max * pad.

    Pass ``k_nm_inv`` (= 2πn/λ from ``physics.constants.optical_wavevector_k``)
    to additionally cap the grid at the EXACT back-scatter bound q = 2k — the
    largest physically reachable q (θ = 180°), past which every P(q) model is
    meaningless and the Mie/coated kernels merely clamp (the drawn curve would
    flatline). All three form-factor plotters share the cap (spec
    engineering-05 settled the spec-02 drift note the ellipsoid's way out of).
    Because measured q ≤ 2k by construction, the capped grid always reaches at
    least the outermost measured point: the cap bounds only the ×pad extension.
    (Its predecessor ``2k·0.999`` could truncate below a 175° datum —
    sin(87.5°) ≈ 0.99905 > 0.999; the margin guarded no arcsin domain, the
    kernels clamp via ``np.clip``, so it was dropped rather than re-justified.
    Since the spec-05 review response, every fit echoes its optics, so every
    overlay — the analytic sphere/coil included — takes the cap; only optics-less
    hand-built stand-ins fall back to the uncapped padded grid.)
    """
    q_max = q.max() * _FIT_XMAX_PAD
    if k_nm_inv is not None:
        q_max = min(q_max, 2.0 * k_nm_inv)
    return np.linspace(0.0, q_max, _MODEL_CURVE_POINTS)


def _form_factor_y_cap(y) -> float:
    """The display cap for a drawn model curve: a multiple of the data range
    (see the block comment above), or +inf when the data give no finite max."""
    return (float(np.nanmax(y)) * _FORM_FACTOR_Y_CAP_MULT
            if y.size and np.isfinite(np.nanmax(y)) else np.inf)


def _model_overlay_params(r, P):
    """(floor-masked P, 1/Mw, A2) for one fitted solution's drawn model curve —
    the view-only prep around the analysis-owned ordinate.

    The near-zero-P region is broken to NaN (P-floor), a missing/zero Mw maps to
    NaN (so the curve simply doesn't draw), and A2 falls back to 0 when the fit
    reported none. ``r`` is any form-factor result carrying ``mw_g_per_mol`` /
    ``a2_mol_mL_per_g2``.
    """
    inv_mw = 1.0 / r.mw_g_per_mol if r.mw_g_per_mol else float('nan')
    a2 = r.a2_mol_mL_per_g2 if r.a2_mol_mL_per_g2 is not None else 0.0
    return np.where(P > _FORM_FACTOR_P_FLOOR, P, np.nan), inv_mw, a2


def _model_curve(ax, qq, P_masked, inv_mw, a2, ci, y_cap, *, color, style='-',
                 lw=1.4, x_scale=1.0):
    """Draw ONE concentration's capped model curve (vs q²) and return the line.

    The ordinate comes from ``analysis.form_factor.model_ordinate`` — the SAME
    function the fits minimize (spec engineering-05) — so the view layer no
    longer assembles ``1/(Mw·P) + 2·A₂·c`` itself; it only applies the display
    guards (the caller's P-floor mask inside ``P_masked``, and the y-cap here:
    break the line, don't spike). Takes the caller's colour — it never calls
    ``_next_color``, so the caller owns the colour-assignment order (the
    stateful artist count; see E1-O-7 / spec 04). ``x_scale`` is the caller's
    q² display factor: the model is computed on the canonical nm grid and only
    the drawn abscissa is scaled (spec 04 stage 2).
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        model = ff.model_ordinate(P_masked, inv_mw, a2, ci)
    model = np.where(model <= y_cap, model, np.nan)
    return ax.plot(qq ** 2 * x_scale, model, style, color=color, lw=lw)[0]


# Per-solution line styles for the degenerate-solution overlays (ellipsoid,
# core-shell): nearly-overlapping solutions stay distinguishable by dash
# pattern, since they share each concentration's colour.
_SOLUTION_STYLES = ('-', '--', ':', '-.')


def plot_form_factor(result, ax: Optional[Any] = None,
                     series_offset: int = 0) -> PlotHandles:
    """Plot a full-angular form-factor P(q) fit: Kc/dR vs q^2 with the fitted model
    curve (1/(Mw P(q)) + 2 A2 c) overlaid per concentration.

    Unlike the Debye/Guinier straight-line fits, the model line is the CURVED full form
    factor, so at larger qRg it tracks the data where a q^2 line would not. Data points
    are grouped and colored by concentration; each concentration gets its own smooth model
    curve (drawn from the fitted parameters via analysis.form_factor.model_P)."""
    fig, ax = _get_ax(ax)
    q = np.asarray(result.q_nm_inv, dtype=float)
    y = np.asarray(result.kc_over_dR, dtype=float)
    c = np.asarray(result.concentrations_g_per_mL, dtype=float)

    artists: Dict[str, Any] = {}
    # This plotter renders the SINGLE-solution models; a mis-routed multi-solution
    # result must fail loudly, as the pre-spec-05 form_factor_P call did (review
    # 2026-07-24) — the list plotters carry the 2k cap wiring for their own results
    # and the degenerate-solution styling this one lacks.
    if result.model in (ff.ELLIPSOID, ff.CORE_SHELL):
        raise ValueError(
            f"plot_form_factor renders the single-solution sphere/coil/Mie fits; a "
            f"{result.model!r} result list has its own plotter "
            f"(plot_{result.model}_form_factor).")
    # Every P(q) model is meaningless past back-scatter q = 2k — cap the grid
    # whenever the result carries its optics (every fit echoes them since the
    # spec-05 review response; the guard covers hand-built stand-ins, which fall
    # back to the uncapped grid).
    k = (phys.optical_wavevector_k(result.wavelength_nm, result.n_medium)
         if result.wavelength_nm and result.n_medium else None)
    qq = _model_q_grid(q, k_nm_inv=k)
    y_cap = _form_factor_y_cap(y)
    Pm, inv_mw, a2 = _model_overlay_params(result, ff.model_P(result, qq))
    qfac = display_factor_from('scattering_q2', 'nm⁻²')
    # Concentration colours are INDEX-keyed (spec engineering-04, owner-approved
    # visible change): previously the interleaved scatter+curve draw advanced
    # _next_color's artist count by two per group, so concentrations skipped
    # every other cycle slot.
    artists['series'] = []
    for i, ci in enumerate(np.unique(c)):
        col = _series_color(ax, series_offset + i)
        m = c == ci
        pts = ax.scatter(
            q[m] ** 2 * qfac, y[m], s=36, color=col, zorder=3,
            label=f'c = {conc_mg_ml_label(ci)}')
        line = _model_curve(ax, qq, Pm, inv_mw, a2, ci, y_cap, color=col,
                            x_scale=qfac)
        artists['series'].append(
            {'concentration_g_per_mL': float(ci), 'data': pts, 'fit': line})

    label = ff.MODEL_LABELS.get(result.model, result.model)
    regime = 'apparent' if result.is_apparent else 'thermodynamic'
    size_txt = (f'R = {result.r_nm:.1f} nm' if result.model in (ff.SPHERE, ff.MIE)
                else f'$R_g$ = {result.rg_nm:.1f} nm')
    ax.set_xlabel(rf'$q^2$ ({display_unit("scattering_q2")})')
    ax.set_ylabel(r'$Kc/\Delta R$ (mol/g)')
    ax.legend(frameon=False, fontsize=9,
              title=(f'{label} P(q), {regime}\n{size_txt}   '
                     f'($qR_g^{{\\max}}$ = {result.qrg_max:.2f})'))
    return PlotHandles(fig, ax, artists)


def _qrg_max_span(results) -> str:
    """qRg(max) legend fragment for a SOLUTION LIST: each solution has its own Rg (so its
    own qRg(max)); with several solutions show the min-max span rather than silently the
    first-listed solution's value (the list order is a presentation convention, not fit
    preference -- focused-review 2026-07-23). One solution shows its value plainly."""
    lo = min(r.qrg_max for r in results)
    hi = max(r.qrg_max for r in results)
    return f'{lo:.2f}' if f'{lo:.2f}' == f'{hi:.2f}' else f'{lo:.2f}–{hi:.2f}'


def plot_ellipsoid_form_factor(results, ax: Optional[Any] = None,
                               series_offset: int = 0) -> PlotHandles:
    """Plot an ellipsoid-of-revolution P(q) fit: Kc/dR vs q^2 with EACH solution's model
    curve overlaid (a prolate and an oblate ellipsoid can fit the same data -- `results` is
    the list from ellipsoid_form_factor_analysis). Data points are shared and colored by
    concentration; each solution's curves use a distinct line style, so degenerate solutions
    (which nearly overlap -- that is what makes them degenerate) are still distinguishable."""
    fig, ax = _get_ax(ax)
    first = results[0]
    q = np.asarray(first.q_nm_inv, dtype=float)
    y = np.asarray(first.kc_over_dR, dtype=float)
    c = np.asarray(first.concentrations_g_per_mL, dtype=float)
    artists: Dict[str, Any] = {}
    # The Pedersen P(q) is analytic at any q, but q > 2k is physically unreachable
    # for every model — cap like the Mie/coated siblings (spec engineering-05).
    # The optics fields are always fit-populated; the guard covers hand-built
    # stand-ins, which fall back to the uncapped grid.
    k = (phys.optical_wavevector_k(first.wavelength_nm, first.n_medium)
         if first.wavelength_nm and first.n_medium else None)
    qq = _model_q_grid(q, k_nm_inv=k)
    y_cap = _form_factor_y_cap(y)
    qfac = display_factor_from('scattering_q2', 'nm⁻²')
    uniq_c = np.unique(c)

    # Data once, colored by concentration (fix the color per concentration so every
    # solution's overlay shares it).
    # Concentration colours are index-keyed (spec engineering-04); on a fresh
    # axes this matches the old artist-count order exactly, but it can no longer
    # shift when a decorative artist is added upstream.
    conc_color = {}
    artists['series'] = []
    for i, ci in enumerate(uniq_c):
        col = _series_color(ax, series_offset + i)
        conc_color[ci] = col
        m = c == ci
        pts = ax.scatter(
            q[m] ** 2 * qfac, y[m], s=36, color=col, zorder=3,
            label=f'c = {conc_mg_ml_label(ci)}')
        artists['series'].append(
            {'concentration_g_per_mL': float(ci), 'data': pts, 'fits': []})

    sol_labels = []
    for si, r in enumerate(results):
        Pm, inv_mw, a2 = _model_overlay_params(r, ff.model_P(r, qq))
        style = _SOLUTION_STYLES[si % len(_SOLUTION_STYLES)]
        # One fit line per concentration for THIS solution: entry['fits'][si].
        for entry, ci in zip(artists['series'], uniq_c, strict=True):
            entry['fits'].append(_model_curve(
                ax, qq, Pm, inv_mw, a2, ci, y_cap, color=conc_color[ci],
                style=style, x_scale=qfac))
        kind = 'prolate' if r.is_prolate else 'oblate'
        sol_labels.append(f'{style}  {kind} p={r.axial_ratio:.2f}, '
                          f'$R_g$={r.rg_nm:.1f} nm')

    regime = 'apparent' if first.is_apparent else 'thermodynamic'
    degen = f' — {len(results)} degenerate solutions' if first.degenerate else ''
    ax.set_xlabel(rf'$q^2$ ({display_unit("scattering_q2")})')
    ax.set_ylabel(r'$Kc/\Delta R$ (mol/g)')
    ax.legend(frameon=False, fontsize=9,
              title=(f'Ellipsoid P(q), {regime}{degen}\n' + '\n'.join(sol_labels)
                     + f'\n($qR_g^{{\\max}}$ = {_qrg_max_span(results)})'))
    return PlotHandles(fig, ax, artists)


def plot_core_shell_form_factor(results, ax: Optional[Any] = None,
                                series_offset: int = 0) -> PlotHandles:
    """Plot a core-shell (coated-sphere) P(q) fit: Kc/dR vs q^2 with EACH solution's model
    curve overlaid (several core/outer radius pairs can fit the same data -- `results` is the
    list from core_shell_form_factor_analysis). Data points are shared and colored by
    concentration; each solution uses a distinct line style, so degenerate solutions stay
    distinguishable. Like the Mie overlay, the model grid is capped at the back-scatter
    bound q = 2k (k = 2*pi*n/lambda) where the coated P(q) stops being defined, so the
    drawn curve tracks the physical range instead of flat-lining past back-scatter."""
    fig, ax = _get_ax(ax)
    first = results[0]
    q = np.asarray(first.q_nm_inv, dtype=float)
    y = np.asarray(first.kc_over_dR, dtype=float)
    c = np.asarray(first.concentrations_g_per_mL, dtype=float)
    artists: Dict[str, Any] = {}
    k = phys.optical_wavevector_k(first.wavelength_nm, first.n_medium)
    qq = _model_q_grid(q, k_nm_inv=k)
    y_cap = _form_factor_y_cap(y)
    qfac = display_factor_from('scattering_q2', 'nm⁻²')
    uniq_c = np.unique(c)

    # Concentration colours are index-keyed (spec engineering-04); on a fresh
    # axes this matches the old artist-count order exactly, but it can no longer
    # shift when a decorative artist is added upstream.
    conc_color = {}
    artists['series'] = []
    for i, ci in enumerate(uniq_c):
        col = _series_color(ax, series_offset + i)
        conc_color[ci] = col
        m = c == ci
        pts = ax.scatter(
            q[m] ** 2 * qfac, y[m], s=36, color=col, zorder=3,
            label=f'c = {conc_mg_ml_label(ci)}')
        artists['series'].append(
            {'concentration_g_per_mL': float(ci), 'data': pts, 'fits': []})

    sol_labels = []
    for si, r in enumerate(results):
        Pm, inv_mw, a2 = _model_overlay_params(r, ff.model_P(r, qq))
        style = _SOLUTION_STYLES[si % len(_SOLUTION_STYLES)]
        # One fit line per concentration for THIS solution: entry['fits'][si].
        for entry, ci in zip(artists['series'], uniq_c, strict=True):
            entry['fits'].append(_model_curve(
                ax, qq, Pm, inv_mw, a2, ci, y_cap, color=conc_color[ci],
                style=style, x_scale=qfac))
        sol_labels.append(f'{style}  $R_{{core}}$={r.r_core_nm:.1f}, '
                          f'$R_{{outer}}$={r.r_outer_nm:.1f} nm')

    regime = 'apparent' if first.is_apparent else 'thermodynamic'
    degen = f' — {len(results)} degenerate solutions' if first.degenerate else ''
    ax.set_xlabel(rf'$q^2$ ({display_unit("scattering_q2")})')
    ax.set_ylabel(r'$Kc/\Delta R$ (mol/g)')
    ax.legend(frameon=False, fontsize=9,
              title=(f'Core–shell P(q), {regime}{degen}\n' + '\n'.join(sol_labels)
                     + f'\n($qR_g^{{\\max}}$ = {_qrg_max_span(results)})'))
    return PlotHandles(fig, ax, artists)


def plot_zimm(
    rayleigh_results: Sequence,
    zimm_result,
    ax: Optional[Any] = None,
    spacing_constant: Optional[float] = None,
    series_offset: int = 0,
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
        k in q^2 + k*c, in CANONICAL nm^-2 per (g/mL). If None, `zimm_spacing`
        picks it so the concentration spread roughly matches the q^2 spread.
        The k printed in the x-label is converted to the active q^2 display
        unit (per mL/g); the abscissa is a q^2-unit quantity, so the
        'concentration' display choice never touches this plot (spec 04
        stage 2).
    series_offset : int, optional
        Shifts the per-concentration colour/marker index — pass the previous
        result's series count when overlaying two grids on one ax.

    Returns
    -------
    PlotHandles
    """
    samples = [r for r in rayleigh_results if r.concentration_g_per_mL != 0]
    samples.sort(key=lambda r: r.concentration_g_per_mL)
    berry = (zimm_result.method == 'berry')

    def ordinate(kc):
        return np.sqrt(kc) if berry else kc

    if spacing_constant is None:
        spacing_constant = zimm_spacing(samples)
    k = spacing_constant
    # The abscissa q^2 + k*c is a q^2-unit quantity: computed canonically,
    # then the COMPOSITE is scaled to the display unit (spec 04 stage 2).
    qfac = display_factor_from('scattering_q2', 'nm⁻²')

    fig, ax = _get_ax(ax, figsize=(7.0, 5.0))
    artists = {'series': [], 'extrapolation_c': None, 'extrapolation_q': None}

    # Per-concentration data, spread along x by k*c.
    for i, r in enumerate(samples):
        col = _series_color(ax, series_offset + i)
        q2 = np.asarray(r.q2_nm2, dtype=float)
        y = ordinate(np.asarray(r.kc_over_dR_mol_per_g, dtype=float))
        x = (q2 + k * r.concentration_g_per_mL) * qfac
        # Vary the marker SHAPE per concentration so the series stay distinct without
        # colour (WCAG 1.4.1 / greyscale), not hue alone. Marker and colour share
        # the offset index, keeping the (colour, marker) pair stable per series.
        pts = ax.plot(x, y, marker=marker_for(series_offset + i), ls='none',
                      color=col, ms=5,
                      label=conc_mg_ml_label(r.concentration_g_per_mL))[0]
        artists['series'].append(
            {'concentration_g_per_mL': r.concentration_g_per_mL, 'data': pts})

    # Both extrapolations come from the global plane fit  ord = a + b q^2 + d c, so
    # the c->0 line (vs q^2) and the q->0 line (vs k*c) are straight and meet at the
    # common intercept a (= 1/Mw, or its root for Berry) at x = 0. Continue each line
    # all the way to x = 0 so the extrapolation visibly reaches the axis;
    # the route-specific intercepts stay as markers on the real data range.
    a = float(zimm_result.coef_intercept)
    b = float(zimm_result.coef_q2)
    d = float(zimm_result.coef_c)

    # c -> 0 extrapolated series (intercept per angle), plotted against q^2.
    # All extrapolation y's are computed on the CANONICAL abscissa; only the
    # drawn x is scaled to the display unit.
    q2_levels = np.asarray(zimm_result.q2_nm2, dtype=float)
    inter_q = np.asarray(zimm_result.intercept_per_angle, dtype=float)
    xc = np.array([0.0, float(q2_levels.max())])
    ax.plot(xc * qfac, a + b * xc, '--', color=PALETTE['black'], lw=1.0,
            label=r'$c \to 0$')
    artists['extrapolation_c'] = ax.plot(
        q2_levels * qfac, inter_q, 's', color=PALETTE['black'], ms=5)[0]

    # q -> 0 extrapolated series (intercept per concentration), against k*c.
    conc = np.asarray(zimm_result.concentrations_g_per_mL, dtype=float)
    inter_c = np.asarray(zimm_result.intercept_per_concentration, dtype=float)
    xq = np.array([0.0, float((k * conc).max())])
    ax.plot(xq * qfac, a + d * (xq / k), '--', color=PALETTE['vermilion'], lw=1.0,
            label=r'$q \to 0$')   # ord = a + d*c, with x = k*c  ->  c = x/k
    artists['extrapolation_q'] = ax.plot(
        k * conc * qfac, inter_c, '^', color=PALETTE['vermilion'], ms=6)[0]

    # The common intercept at q^2 -> 0, c -> 0 (the Mw point) on the y-axis.
    artists['intercept'] = ax.plot([0.0], [a], 'o', color=PALETTE['black'],
                                   ms=7, mfc='none', zorder=5,
                                   label='intercept (1/$M_w$)')[0]

    # The printed k follows the active q^2 unit (its implied unit is
    # q^2-unit·mL/g; at the default nm⁻², qfac is exactly 1.0 and the number
    # is unchanged).
    ax.set_xlabel(rf'$q^2 + k\,c$ ({display_unit("scattering_q2")};'
                  rf'  $k$ = {k * qfac:.3g})')
    ylab = r'$\sqrt{Kc/\Delta R}$' if berry else r'$Kc/\Delta R$ (mol/g)'
    ax.set_ylabel(ylab)

    # The unit rides INSIDE the rendered value, so a withheld Mw reads "— (uncalibrated)"
    # rather than leaving a stranded "g/mol" behind the dash.
    mw_txt = scaled_value(zimm_result, f'{zimm_result.mw_g_per_mol:.3e} g/mol',
                          zimm_result.mw_g_per_mol)
    a2_txt = scaled_value(zimm_result, f'{zimm_result.a2_mol_mL_per_g2:.2e}',
                          zimm_result.a2_mol_mL_per_g2)
    title = (f'{zimm_result.method.upper()}: '
             f'$M_w$={mw_txt}, '
             f'$R_g$={zimm_result.rg_nm:.1f} nm, '
             f'$A_2$={a2_txt}')
    ax.set_title(title, fontsize=9)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    return PlotHandles(fig, ax, artists)


def plot_calibration_free_a2(result, ax: Optional[Any] = None) -> PlotHandles:
    """Plot the calibration-free A2 construction: Y(c) vs c.

    The concentration axis honours PLOT_UNITS — the one SLS plotter with a
    concentration abscissa. Its DEFAULT display is therefore mg/mL (the global
    concentration default, matching the D-vs-c plot and every legend), an
    owner-accepted visible change from the fixed g/mL axis it had before spec
    04 stage 2. The fields are canonical g/mL, so the standard `_disp` route
    applies; the fitted slope stays per (g/mL) and only drawn x is scaled."""
    fig, ax = _get_ax(ax)
    c = np.asarray(result.concentrations_g_per_mL, dtype=float)
    Y = np.asarray(result.Y, dtype=float)
    cfac = display_factor('concentration')
    artists = {}
    artists['data'] = ax.scatter(c * cfac, Y, s=36, color=PALETTE['blue'],
                                 zorder=3, label='data')
    xline = np.linspace(0.0, c.max() * _FIT_XMAX_PAD, _FIT_LINE_POINTS)
    artists['fit'] = ax.plot(xline * cfac,
                             result.slope * xline + result.intercept, '-',
                             color=PALETTE['vermilion'], lw=1.6)[0]
    ax.set_xlabel(rf'Concentration ({display_unit("concentration")})')
    ax.set_ylabel(r'$Y(c) = [I(c_{ref})/c_{ref}]\,/\,[I(c)/c]$')
    ax.legend(frameon=False, fontsize=9,
              title=f'$2A_2M_w$ = {result.two_a2_mw:.3e}')
    return PlotHandles(fig, ax, artists)


# ===========================================================================
# Utilities plots
# ===========================================================================

def plot_i_sin_theta(result, ax: Optional[Any] = None,
                     series_offset: int = 0) -> PlotHandles:
    """Plot I·sin(θ) vs angle for one or more SLS measurements (overlaid).

    `result` is a utilities.ISinThetaResult. For an ideal isotropic, dust-free
    scattering volume the curve is flat across angle; curvature or asymmetry about
    90° flags an alignment/stray-light/dust problem. `result.mode` ('absolute' or
    'normalized') sets the y-axis label — the normalized mode (each curve over its
    own mean) overlays curves of different scale for shape comparison.
    """
    fig, ax = _get_ax(ax)
    artists = {'series': []}
    for i, curve in enumerate(result.curves):
        color = _series_color(ax, series_offset + i)          # index-keyed (spec 04)
        line = ax.plot(
            curve.angles_deg, curve.i_sin_theta, 'o-', color=color,
            ms=4, lw=1.2, label=curve.label)[0]
        artists['series'].append({'label': curve.label, 'data': line})
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
    fig, ax = _get_ax(ax)
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


# One threshold for every width-unbounded number in the histogram legend (the
# chi^2_r labels and the Fano title blow up together on a non-Poisson trace).
_LEGEND_MAGNITUDE_CAP = 1e3


def _chi2r_label(name: str, chi2: Optional[float]) -> str:
    """One legend label for a histogram overlay's reduced chi-square.

    The magnitude is capped: a badly non-Poisson trace (a dust spike) can push
    chi^2_r to 1e26-class values, and ``:.2f`` prints every integer digit, so the
    legend grew wider than the figure (seen on a real ALV trace, Fano ~5446).
    The number's job here is "is this Poisson?", which the cap answers; the
    healthy range keeps its exact former rendering. Both overlays MUST come
    through this one helper so their formats cannot diverge.
    """
    if chi2 is None:
        return name
    if chi2 > _LEGEND_MAGNITUDE_CAP:
        return f'{name} ($\\chi^2_r > 10^3$)'
    return f'{name} ($\\chi^2_r$ = {chi2:.2f})'


def plot_count_rate_histogram(result, ax: Optional[Any] = None) -> PlotHandles:
    """Count-rate histogram with optional Gaussian/Poisson overlays (a
    trace_analysis.HistogramFitResult). The legend title reports the Fano factor
    (variance/mean in count space): ~1 for ideal shot noise, >1 for excess
    fluctuations (slow modes, dust). Each overlay's legend entry carries its
    reduced chi-square (chi^2_r ~ 1 = a good fit) when the fit was computed."""
    fig, ax = _get_ax(ax)
    ifac, i_unit = display_factor('intensity'), display_unit('intensity')
    centers = np.asarray(result.bin_centers, dtype=float) * ifac
    counts = np.asarray(result.counts, dtype=float)
    width = (float(result.bin_edges[1] - result.bin_edges[0]) * ifac
             if np.asarray(result.bin_edges).size > 1 else None)
    artists = {'hist': ax.bar(centers, counts, width=width, align='center',
                              color=PALETTE['sky'], alpha=0.7)}
    if result.gaussian_curve is not None:
        artists['gaussian'] = ax.plot(centers, result.gaussian_curve, '-',
                                      color=PALETTE['vermilion'], lw=1.4,
                                      label=_chi2r_label('Gaussian',
                                                         result.gaussian_chi2_reduced))[0]
    if result.poisson_curve is not None:
        artists['poisson'] = ax.plot(centers, result.poisson_curve, '--',
                                     color=PALETTE['green'], lw=1.4,
                                     label=_chi2r_label('Poisson',
                                                        result.poisson_chi2_reduced))[0]
    ax.set_xlabel(f'Count rate ({i_unit})')
    ax.set_ylabel('Frequency')
    ax.legend(frameon=False, fontsize=8,
              title=('Fano > $10^3$'
                     if result.fano_factor > _LEGEND_MAGNITUDE_CAP
                     else f'Fano = {result.fano_factor:.2f}'))
    return PlotHandles(fig, ax, artists)


def plot_block_variance(result, ax: Optional[Any] = None) -> PlotHandles:
    """Standard error of the mean vs block size (a trace_analysis.BlockVarianceResult).
    A flat curve = uncorrelated (white) noise; a rise that plateaus signals
    positive correlations (slow modes). The legend reports the SE ratio and the
    verdict."""
    fig, ax = _get_ax(ax)
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


def _thin_log_ticks(ax) -> None:
    """Drop minor tick LABELS on log axes spanning less than a decade.

    Matplotlib labels the minor ticks when a log axis covers under one decade,
    which on the Traces tab's short sub-plot renders as a run of overlapping
    exponents. The ticks themselves stay; only their labels go, so the reader
    loses nothing but the collision. Applied to the three frequency/stability
    views, which are the only log-log plots small enough to hit it.
    """
    from matplotlib.ticker import NullFormatter
    for axis in (ax.xaxis, ax.yaxis):
        if axis.get_scale() == 'log':
            axis.set_minor_formatter(NullFormatter())


def plot_allan_deviation(result, ax: Optional[Any] = None) -> PlotHandles:
    """Allan deviation vs averaging time (a trace_analysis.BlockVarianceResult).

    The stability read of the same blocking analysis `plot_block_variance` shows.
    Slope -1/2 is white noise, where averaging pays in full; where the curve
    flattens, averaging has stopped paying and that tau is the longest useful
    averaging window. The two reference slopes are drawn through the leftmost
    point so they can be compared by eye without reading coordinates.
    """
    fig, ax = _get_ax(ax)
    tau = np.asarray(result.tau_s, dtype=float)
    dev = np.asarray(result.allan_deviation, dtype=float)
    artists = {'allan': ax.plot(tau, dev, 'o-', color=PALETTE['blue'],
                                ms=4, lw=1.2, label=r'$\sigma_y(\tau)$')[0]}
    usable = np.isfinite(tau) & (tau > 0) & np.isfinite(dev) & (dev > 0)
    if usable.any():
        t0 = float(tau[usable][0])
        d0 = float(dev[usable][0])
        guide = tau[usable]
        artists['white'] = ax.plot(
            guide, d0 * (guide / t0) ** -0.5, ':', color=PALETTE['vermilion'],
            lw=1.1, label=r'$-1/2$ (white)')[0]
        artists['floor'] = ax.plot(
            guide, np.full(guide.size, d0), '--', color=PALETTE['green'],
            lw=1.1, label='0 (floor)')[0]
        ax.set_xscale('log')
        ax.set_yscale('log')
        _thin_log_ticks(ax)
    ax.set_xlabel(r'Averaging time $\tau$ (s)')
    ax.set_ylabel(r'$\sigma_y(\tau)$')
    ax.legend(frameon=False, fontsize=8)
    return PlotHandles(fig, ax, artists)


def plot_welch_psd(result, ax: Optional[Any] = None) -> PlotHandles:
    """Welch power spectral density of a trace (a trace_analysis.WelchPSDResult).

    Log-log, with the flat shot-noise floor drawn as a reference: broadband power
    above it is excess fluctuation, a narrow peak above it is a candidate
    periodicity. The legend title reports the equivalent degrees of freedom,
    which is how much averaging the estimate received.
    """
    fig, ax = _get_ax(ax)
    freq = np.asarray(result.frequencies_hz, dtype=float)
    psd = np.asarray(result.psd, dtype=float)
    # Zero frequency cannot be shown on a log axis, and after per-segment mean
    # removal it carries nothing anyway.
    shown = freq > 0
    edf_note = ('E.D.F. —' if not np.isfinite(result.edf)
                else f'E.D.F. {result.edf:.0f}')
    artists = {'psd': ax.plot(freq[shown], psd[shown], '-', color=PALETTE['blue'],
                              lw=1.1,
                              label=f'Welch PSD ({result.n_segments} seg, '
                                    f'{edf_note})')[0]}
    if np.isfinite(result.shot_floor_psd) and result.shot_floor_psd > 0:
        artists['shot'] = ax.axhline(
            result.shot_floor_psd, ls='--', color=PALETTE['vermilion'], lw=1.2,
            label=r'shot noise $2\bar{r}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    _thin_log_ticks(ax)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel(r'PSD (cps$^2$/Hz)')
    ax.legend(frameon=False, fontsize=8)
    return PlotHandles(fig, ax, artists)


def plot_periodogram(result, ax: Optional[Any] = None) -> PlotHandles:
    """Classical periodogram with Scargle's false-alarm level (a
    trace_analysis.PeriodogramResult).

    Power is in units of the data variance, which is the normalization Scargle's
    exponential null is stated for. A peak above the drawn level is unlikely to
    be noise; the legend title carries the verdict, and says so plainly when the
    trace's own correlations make that null the wrong one to test against.
    """
    fig, ax = _get_ax(ax)
    freq = np.asarray(result.frequencies_hz, dtype=float)
    power = np.asarray(result.power, dtype=float)
    artists = {'periodogram': ax.plot(freq, power, '-', color=PALETTE['blue'],
                                      lw=1.0, label='periodogram')[0]}
    if np.isfinite(result.significance_level):
        pct = f'{result.requested_false_alarm:.3g}'
        artists['level'] = ax.axhline(
            result.significance_level, ls='--', color=PALETTE['vermilion'],
            lw=1.2, label=f'p = {pct} level')
    ax.set_xscale('log')
    _thin_log_ticks(ax)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel(r'Power / $\sigma_0^2$')
    # The verdict leads, because it is the answer the view exists to give. The
    # unreliable case REPLACES the peak claim rather than qualifying it: a
    # correlated trace breaks the null the peak's probability was computed under,
    # so printing that probability beside the warning would invite reading it
    # anyway.
    if result.null_assumption_violated:
        verdict = 'unreliable — trace is correlated'
    elif result.peak_is_significant:
        verdict = (f'{result.peak_frequency_hz:.4g} Hz significant '
                   f'(p = {result.false_alarm_probability:.1g})')
    else:
        verdict = 'no significant peak'
    ax.legend(frameon=False, fontsize=8, title=verdict)
    return PlotHandles(fig, ax, artists)


def plot_synthetic_correlogram(result, ax: Optional[Any] = None) -> PlotHandles:
    """Plot a generated synthetic correlogram (utilities.SyntheticCorrelogramResult).

    The curve is shown vs log delay time; the legend reports the ground-truth
    effective Rh and PDI an ideal cumulant fit should recover.
    """
    fig, ax = _get_ax(ax)
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


def plot_synthetic_multi_dls(dls, ax: Optional[Any] = None,
                             series_offset: int = 0) -> PlotHandles:
    """Overlay the generated correlograms of a multi-angle DLS set, one per angle.

    `dls` is a synthetic_dataset.MultiAngleDLS (raw generated data). This is the
    raw-data preview — the angular Γ-vs-q² analysis is what you get by loading the
    saved .ASC into the DLS tab, not something drawn here.
    """
    fig, ax = _get_ax(ax)
    artists = {'series': []}
    delay, t_unit = _disp('time', dls.delay_times_s)
    for i, a in enumerate(dls.angles_deg):
        col = _series_color(ax, series_offset + i)               # index-keyed (spec 04)
        line = ax.plot(delay, np.asarray(dls.signals[a], dtype=float), '-',
                       color=col, lw=1.0, label=f'{a:g}°')[0]
        artists['series'].append({'angle_deg': float(a), 'data': line})
    ax.set_xscale('log')
    ax.set_xlabel(rf'Delay time $\tau$ ({t_unit})')
    ax.set_ylabel(r'$g_2(\tau) - 1$')
    ax.legend(frameon=False, fontsize=7, ncol=2,
              title=f'{len(dls.angles_deg)} angles')
    return PlotHandles(fig, ax, artists)


def plot_synthetic_sls_set(sls, ax: Optional[Any] = None,
                           series_offset: int = 0) -> PlotHandles:
    """Plot a generated SLS set as intensity vs angle, one curve per concentration.

    `sls` is a synthetic_dataset.SyntheticSLSSet (raw generated intensities,
    including the c = 0 solvent reference). The Zimm/Debye/Guinier analysis comes
    from loading the saved Brookhaven file into the SLS tab; this is just the data.
    """
    fig, ax = _get_ax(ax)
    artists = {'series': []}
    angles = np.asarray(sls.angles_deg, dtype=float)
    for i, c in enumerate(sls.concentrations_g_per_mL):
        col = _series_color(ax, series_offset + i)               # index-keyed (spec 04)
        # (label formerly used a bare ':g' spec — unified on conc_mg_ml_label's
        # ':.3g'; identical for the generator's round-number concentrations)
        label = 'solvent (c = 0)' if c == 0 else conc_mg_ml_label(c)
        marker = 'o-' if angles.size > 1 else 'o'
        line = ax.plot(angles, np.asarray(sls.intensities[c], dtype=float),
                       marker, color=col, ms=4, lw=1.0, label=label)[0]
        artists['series'].append({'concentration_g_per_mL': float(c), 'data': line})
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
                 ax: Optional[Any] = None, *,
                 branched_x: Optional[Sequence[float]] = None,
                 branched_y: Optional[Sequence[float]] = None,
                 branched_labels: Optional[Sequence[str]] = None) -> PlotHandles:
    """Log-log scaling plot from a utilities.ScalingResult (Rg-Mw or A2-Mw).

    `result` is a ScalingResult (its x = Mw, y = Rg or A2 are the positive points
    actually fitted). `quantity` ('rg'|'a2') only sets the y-axis label and the
    exponent symbol. `labels` (optional, aligned with result.x/result.y) annotate
    the points with their sample names. The power-law fit line is drawn only when
    the fit is valid (>= 2 points). `branched_x/y/labels` are Branched-tagged points
    drawn HOLLOW and held OUT of the fit (so the architectural contraction is visible
    without biasing the linear scaling law). No show()/save -- the caller owns the figure.
    """
    fig, ax = _get_ax(ax)
    mfac = display_factor('molar_mass')
    yfac = display_factor('radius') if quantity in ('rg', 'rh') else 1.0
    x = np.asarray(result.x, dtype=float)
    y = np.asarray(result.y, dtype=float)
    artists = {}
    artists['data'] = ax.scatter(x * mfac, y * yfac, s=42, color=PALETTE['blue'],
                                 zorder=3)
    bx = np.asarray(branched_x if branched_x is not None else [], dtype=float)
    byv = np.asarray(branched_y if branched_y is not None else [], dtype=float)
    if bx.size:
        # Hollow markers: plotted but NOT fitted (a branched chain would bias the law).
        artists['branched'] = ax.scatter(
            bx * mfac, byv * yfac, s=46, facecolors='none',
            edgecolors=PALETTE['blue'], linewidths=1.3, zorder=3,
            label='branched (excluded from fit)')
    if result.fit_valid and x.size:
        xline = np.geomspace(x.min(), x.max(), _FIT_LINE_POINTS)
        sym = r'\nu' if quantity in ('rg', 'rh') else '-a'
        exponent = round(result.exponent, 3) + 0.0      # fold a negative zero ("-0.000")
        artists['fit'] = ax.plot(
            xline * mfac, (result.prefactor * xline ** result.exponent) * yfac, '-',
            color=PALETTE['vermilion'], lw=1.6,
            label=f'${sym}$ = {exponent:.3f}  ($R^2$ = {result.r_squared:.3f})')[0]
        ax.legend(frameon=False, fontsize=9)
    elif bx.size:
        ax.legend(frameon=False, fontsize=9)         # show the branched key with no fit
    # Scaling plots are ALWAYS log-log (power laws plot as straight lines). Log
    # axes need positive limits, so when there are no points yet (e.g. a sample
    # without a computed Mw) we set representative positive limits to avoid the
    # "non-positive limits on a log axis" warning while still showing log-log.
    ax.set_xscale('log')
    ax.set_yscale('log')
    if not x.size and not bx.size:
        # Only fall back to a representative window when there are NO points of either
        # kind; with branched-only points (no valid linear fit) let matplotlib autoscale
        # over them, else the fixed window would scatter the hollow markers off-screen.
        ax.set_xlim(1e4 * mfac, 1e7 * mfac)              # representative Mw range
        ax.set_ylim((1.0 * yfac, 1e3 * yfac) if quantity in ('rg', 'rh')  # Rg/Rh ~ nm
                    else (1e-5, 1e-3))                  # A2 ~ mol*mL/g^2
    mw_unit, r_unit = display_unit('molar_mass'), display_unit('radius')
    ax.set_xlabel(rf'$M_w$ ({mw_unit})')
    ax.set_ylabel({'rg': rf'$R_g$ ({r_unit})', 'rh': rf'$R_h$ ({r_unit})'}.get(
        quantity, r'$A_2$ (mol·mL/g$^2$)'))
    if labels is not None or (bx.size and branched_labels is not None):
        # Stagger sample-name labels so close points don't collide; a
        # crowded corner caps at "+N more" rather than piling up. Done LAST, after the
        # log scales + limits are set, so the de-collision sees the final transData.
        pts = []
        if labels is not None:
            pts += [(xi * mfac, yi * yfac, lb, '#555')
                    for xi, yi, lb in zip(x, y, labels, strict=True)]
        if bx.size and branched_labels is not None:
            pts += [(xi * mfac, yi * yfac, lb, '#777')
                    for xi, yi, lb in zip(bx, byv, branched_labels, strict=True)]
        annotate_decollided(ax, pts, fontsize=7)
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
    fig, ax = _get_ax(ax)
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
    q2_line = np.linspace(0.0, float(q2.max()) * _FIT_XMAX_PAD, _FIT_LINE_POINTS)
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
