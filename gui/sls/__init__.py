"""
gui/sls/
========

The SLS tab: static light scattering analysis for the selected SAMPLE. Unlike the
DLS tab (one measurement), SLS is inherently per-sample — a Zimm set is a solvent
reference (c = 0) plus a concentration series, analyzed together. Selecting any
measurement in the sidebar makes this tab operate on its whole sample.

It contains the visible **calibration panel** (manual entry of one calibrant point
→ k_c, geometry-aware toluene Rayleigh) and an analysis section (Zimm / Berry /
Debye / Guinier / single-angle ladder / calibration-free A₂ / excess Rayleigh ratio,
plus the form-factor P(q) fits). Everything is routed through the controller; no
analysis or physics here.

Commit model: calibration and parameters are session/working state. "Update"
commits them (and recomputes k_c); "Run" uses the COMMITTED state. A result's notes
(each failed precondition, one line in a `NotesBlock` under the result) and its
qualifiers (words on the row labels: apparent, the ± words) are GUI overlays, never on
the figure (ui_style_guide §5).

The package holds one module per sub-tab (`analysis_tab.py`, `derived_intensity.py`,
`mode_resolved.py`), the result-table rows (`rows.py`), what the sub-tabs share
(`common.py`) and the tab itself (`module.py`). `gui/sls_module.py` re-exports every name
under one import path.
"""
