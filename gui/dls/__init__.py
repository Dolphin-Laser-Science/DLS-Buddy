"""
gui/dls/
========

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
  * Mode Map      — every extracted mode of every run in (q², Γ) space, with user-picked
    mode families and a Γ-vs-q² fit per family.
  * D vs c        — concentration extrapolation over the sample's DLS concentrations.
  * DDLS          — depolarized DLS: the VV/VH pairs' rotational diffusion and the
    shape models.
  * Summary       — the workspace-wide results tables: one row per measurement, methods
    side by side, and the sample-level Rh table.

All analysis goes through the controller (`run_*`); soft flags (PDI > 0.3,
non-diffusive, the reliability notes) are GUI overlays, never drawn on the figure.
Exception, owner-ruled: a failed distribution fit's "failed" note IS drawn on the
legend -- it states what the curve is (a seed estimate, not a fit), the same
analysis-determined tier as plotting's dropped-point and ceiling-halo markers, so a
saved image cannot present a seed curve as fitted. It names the METHOD that failed,
since a legend entry stands for a measurement and may cover several of its curves. Each
sub-tab draws the run it last landed for a measurement or sample from the controller's
result store (within-session), so a reselection redraws it with no recompute.

The package holds one module per sub-tab (`correlogram.py`, `distribution.py`,
`sample_analysis.py` for Γ vs q² and D vs c, `mode_map.py`, `ddls.py`, `summary.py`), the
replicate-averaging dialog (`averaging.py`), what the sub-tabs share (`region.py`, `fits.py`,
`legend.py`, `tables.py`, `common.py`, `bases.py`) and the tab itself (`module.py`).
`gui/dls_module.py` re-exports every name under one import path.
"""
