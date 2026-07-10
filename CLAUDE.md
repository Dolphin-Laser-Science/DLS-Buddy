# CLAUDE.md — Light Scattering Analysis Platform ("DLS Buddy")

Orientation for working on this project. This is the high-signal subset needed to be
productive without re-deriving things or breaking an invariant. For a human/forker map
of the code see `docs/4. Code-Map.md`; for the science and the numbered equations see
`docs/3. Theory-and-Equations-Guide.pdf`, and for usage see `docs/1. Quickstart-Guide.pdf`
and `docs/2. User-Manual.pdf`.

---

## What this is

A general-purpose Python platform for analyzing **static and dynamic light scattering
(SLS/DLS)** data from polymer solutions. Deliberately **instrument-agnostic**. Built
module by module, correctness-first, each module validated numerically against synthetic
ground truth and real datasets before moving on.

The analysis engine, Origin-compatible export, matplotlib plotting layer, and the
framework-agnostic controller/workspace are complete and validated. The **PySide6 GUI**
is built out: the application shell (sidebar navigator + six module tabs) and all six
module tabs (Data, DLS, SLS, Cross-Sample, Utilities, Settings) are implemented and
validated headless.

## How to work on this code

This project is **correctness-first**, and the domain is physical:

- **Correctness over cleverness or brevity.** Track units explicitly. Validate
  numerically against a known answer; don't claim something works without it.
- Ground physical claims in **primary literature** and cite it (authors + year). Never
  invent a citation. The user-facing citation index is `docs/5. Code-References.md`.
- When a result is corrected, acknowledge the specific error and re-derive from scratch;
  don't defend a wrong result for consistency.
- **Flag hard-to-reverse choices before acting on them** and give the reasoning.

## Working rhythm (followed every module)

1. Implement a focused unit of work.
2. **Validate** against synthetic ground truth and/or real test data.
3. Keep modules independently testable.
4. When user-facing behavior changes, keep the in-program help and the guide PDFs current.

**In-program help rule (not optional):** whenever a change adds, renames, moves, or
alters the behaviour of a user-facing control, update its help in the same change — the
section's `?` `HelpBadge` (`gui/help.py`: `section_header` / `add_help_to_groupbox`)
and/or its passive `setToolTip`. *How-to-use* help is the visible `?` badge (concise,
lists over paragraphs, no citations); *calculation nuance* is a brief tooltip that points
to the Theory-and-Equations-Guide. Tooltips honour the global Settings "Show tooltips" gate; `?`
badges always work on click.

---

## Architecture invariants — NEVER violate these

1. **Pure analysis functions.** No GUI dependencies, no file I/O, no plotting, no
   mutation of inputs. They take data + parameters and return result objects.
2. **Two-layer parsers.** Instrument-specific parsers translate native files into the
   common internal data model; all analysis and GUI code operate ONLY on that common
   model. Adding a new instrument format = one new parser file; analysis and GUI are
   never touched.
3. **Nothing hard-coded** that is instrument-, solvent-, or polymer-specific. All
   physical parameters (refractive index, wavelength, temperature, viscosity, dn/dc,
   angle) are always user-supplied.
4. **Diagnostic flags come from general light-scattering criteria**, never system-specific
   thresholds. Where a threshold is needed it is a function parameter with a sensible,
   documented default — not a magic number for one solvent.
5. **Controller sits between the GUI and the engine.** No analysis logic in the GUI. The
   controller is pure Python (no Qt imports) so the GUI framework can be swapped by
   rewriting only the widget shell.
6. **The data model supports associating DLS and SLS datasets by sample identity**
   (polymer, solvent, concentration, temperature) to enable ρ = Rg/Rh.
7. **Apparent vs thermodynamic** results are always distinguished and never conflated
   (single-angle/single-concentration = apparent; fully extrapolated = thermodynamic).
   Likewise DLS **z-average** (cumulants) vs **distribution-weighted** (CONTIN/NNLS) is
   always flagged.
8. **Uncertainty is part of the analysis — neither over- nor under-reported, and omitted
   when none is honest.** Report the **statistical** (regression) SE only where the fitted
   points are genuinely independent, and label it as **excluding** calibration/dn-dc
   systematics. Use the shared `analysis/uncertainty.py` toolkit (HC3
   heteroscedasticity-consistent covariance; delta-method propagation; `format_pm`
   display). Do **NOT** report a ± from a **single correlogram** (its lag channels are
   correlated, Schätzel 1990 → OLS under-reports; ISO 22412 uses repeats). No SE for
   NNLS/CONTIN (ill-posed) or single-angle Mw (one datum). Every new SE must be
   **Monte-Carlo-validated** against the sampling SD before it ships. When adding or
   modifying ANY analysis, state how it treats uncertainty and document the formula in the
   Theory-and-Equations-Guide (numbered LaTeX), exactly as the analysis itself is documented.
9. **No owner-machine specifics in the product or docs.** The program (code, config,
   defaults) must never hard-code or assume a particular computer setup — file paths,
   usernames, drive layout, or sync mechanism; anything machine-specific stays
   user-supplied or generalized.

## Physics that must be enforced in code

- Scattering vector `q = (4π n / λ) sin(θ/2)`; n and λ always explicit.
- Stokes–Einstein `Rh = kT / (6π η D)`; T and η always required, never assumed.
- `(n_solvent / n_standard)²` correction is **mandatory** on excess Rayleigh ratios
  whenever the solvent differs from the calibration standard.
- Toluene Rayleigh factor is **always computed from Sivokhin & Kazantsev (2021)** via
  `physics/constants.py`, with an explicit **geometry** (VU / VV / VH). Never use a
  vendor/instrument-stored Rayleigh value. Key facts: `R_VV(532 nm, 25 °C) = 2.34e-5
  cm⁻¹` (Takahashi et al. 2019); `R_VU = R_VV·(1 + ρv)`, `R_VH = R_VV·ρv`; the
  temperature coefficient is **positive** (~+0.43 %/°C).
- Cumulant PDI > 0.3 → flag as unreliable.
- **Mw and absolute A₂ require calibration.** If a run is uncalibrated, flag those two as
  unreliable — but **Rg survives** (slope/intercept, the unknown scale cancels) and so
  does the **calibration-free 2·A₂·Mw product**. Don't flag Rg or the product.

## Domain gotchas that drive code decisions

- **dn/dc is the central vulnerability** of low-contrast systems: a small value means weak
  scattering, and the error enters Mw squared. Treat it as user-supplied and important;
  never guess it.
- **Vendor calibration constants are display-only.** They go stale after a re-alignment
  and often carry outdated Rayleigh factors. The program computes its own k_c from one
  calibrant point (intensity, angle, standard + geometry).
- Some systems have **Mw biased by preferential co-solvent adsorption**, so a trustworthy
  Mw may be characterized in a different solvent and **entered by hand**. `SampleResult`
  tracks Mw provenance ('computed' vs 'user'); a user Mw must never be overwritten by a
  re-analysis, and scaling plots should prefer it.
- Dust in high-viscosity media sits still and causes persistent intensity offsets, not
  transient spikes; relevant if cleanliness diagnostics are added.

---

## Package layout

Top-level packages (each with an `__init__.py`) using top-level package imports
(`from core.data_models import …`, `from analysis.dls import …`). Default branch `Main`.

```
<repo root>/
├── CLAUDE.md                 # this file — agent/contributor orientation + invariants
├── README.md                 # front door (license pointer)
├── PATCH_NOTES.md            # per-release changes + known issues
├── LICENSE                   # GPLv3
├── requirements.txt          # pinned deps (Python 3.13 venv)
├── docs/                     # user-facing docs: 1. Quickstart-Guide.pdf, 2. User-Manual.pdf,
│                             #   3. Theory-and-Equations-Guide.pdf, 4. Code-Map.md,
│                             #   5. Code-References.md (generated citation index)
├── test-data/                # datasets used by validation
├── core/
│   ├── data_models.py        # SampleKey, DLSMeasurement, SLSMeasurement, solvent vocab
│   └── workspace.py          # LoadedMeasurement/Trace, SampleResult, Sample, Workspace, sessions
├── parsers/                  # base_parser + per-instrument (brookhaven/generic/zetasizer/alv) + trace
├── physics/
│   └── constants.py          # kB, NA, q, Stokes-Einstein, optical constant K, toluene Rayleigh
├── analysis/
│   ├── dls/                  # cumulants, exponentials, distributions, angular, replicate (+ _common)
│   ├── sls.py                # unified calibration, Debye, Zimm/Berry, calibration-free A₂
│   ├── depolarization.py     # static DPLS + DDLS rotational diffusion + rod/sphere shape models
│   ├── uncertainty.py        # HC3 covariance, delta-method, replicate mean/SE, format_pm
│   ├── synthetic_dataset.py  # reusable synthetic-data engine (forward models + writers)
│   ├── trace_analysis.py     # intensity-trace diagnostics
│   └── utilities.py          # i·sinθ, ρ=Rg/Rh, scaling power-law, candidate picker, synth correlogram
├── exporting/
│   └── export.py             # Origin-compatible CSV  (NOT "io/": a top-level io/ shadows stdlib io)
├── plotting/
│   └── plots.py              # matplotlib; ax-accepting, handle-returning, no show()
├── app/
│   ├── controller.py         # framework-agnostic controller (commit/working state, sessions)
│   ├── settings.py           # SettingsState: global seed defaults + appearance
│   ├── units.py              # pure boundary layer: human-scale entry/read, canonical storage
│   └── version.py            # single source of truth for __version__
└── gui/                      # PySide6 widgets only — no analysis/physics
    ├── main.py               # entry point: python -m gui.main
    ├── main_window.py        # shell: sidebar navigator + six module tabs
    ├── data_module.py / dls_module.py / sls_module.py / cross_module.py
    ├── utilities_module.py / settings_module.py
    ├── plot_controls.py / export_helper.py / help.py
```

**Gotcha:** do not name the export package `io/`. A top-level `io` package shadows
Python's standard-library `io` and breaks imports. Use `exporting/`.

## Environment / dependencies

- Python 3.10+ source (uses `from __future__ import annotations`). The working venv is
  **Python 3.13** — PySide6 has no 3.14 wheels yet, so avoid 3.14 for now.
- Runtime: `numpy`, `scipy` (optimize/nnls/curve_fit), `statsmodels` (ADF test in
  utilities), `matplotlib`, `PySide6`. Pinned in `requirements.txt`.
- Recreate the venv with `py -3.13 -m venv .venv` then `pip install -r requirements.txt`;
  point your interpreter at `.venv\Scripts\python.exe` (Windows) or `.venv/bin/python`
  (macOS). (`.venv/` is gitignored.) The double-click launchers bootstrap the venv on
  first run and launch the app: `Launch DLS Buddy (Windows).bat` and
  `Launch DLS Buddy (MacOS).command`.

## How to run / validate

- The GUI runs with `python -m gui.main` from the repo root.
- The program is developed against a pytest regression suite (maintained in the
  development repository, not part of this distribution — nothing in it is needed to
  run the app). Everything is checked against **ground truth**: the program's own
  forward model (a closed round-trip) and the committed real datasets — the DLS engine
  against a real correlogram + the ALV replicate set, the SLS engine + controller
  against the real polystyrene/toluene Zimm set (reproduces Mw ≈ 1.01e6 / Rg ≈ 40.5 nm);
  physics constants asserted analytically. Sessions round-trip to 1e-9. The GUI is
  validated headless (real widgets under Qt's offscreen platform) plus real-platform
  screenshot review.
- Plotting is validated by rendering to PNG and inspecting, never by a blocking `show()`.

---

## Current state

**Complete and validated:** data model; all parsers (Brookhaven DLS/SLS/trace, generic
DLS/SLS/trace, Zetasizer clipboard, ALV .ASC multi-angle + single-angle); physics constants
(geometry-aware Rayleigh); utilities + synthetic generator; DLS engine; SLS engine
(unified calibration, Debye, Zimm/Berry, calibration-free A₂); Origin export; matplotlib
plotting; workspace + controller (grouping, commit/working state, sessions).

**GUI:** all six tabs built and validated headless. **Data** (parameter confirmation,
commit/undo/highlight, shared-param propagation, Unit column). **DLS** (cumulant / single
/ double / KWW / NNLS / CONTIN / lognormal, shared τ window + baseline, multi-measurement
co-plotting, DDLS sub-tab). **SLS** (per-sample calibration panel → k_c; Zimm / Berry /
Debye / Guinier / single-angle / calibration-free A₂ / Rayleigh; manual-Mw override; data
masking; a Depolarization calculator). **Cross-Sample** (ρ = Rg/Rh table + Rg–Mw / A₂–Mw
log–log scaling; provenance-aware source pickers). **Utilities** (Traces store with
diagnostics; I·sin θ check; synthetic generator). **Settings** (seed defaults +
light/dark theme + plot palette + input/display units via `app/units.py`).

**Depolarized scattering (DPLS/DDLS)** is built and feature-complete, validated headless
against synthetic VV/VH ground truth and published tables: static depolarization
(depolarization ratios, Cabannes correction, optical anisotropy δ²), dynamic DDLS
(rotational diffusion D_r = (Γ_VH − Γ_VV)/6, Zero & Pécora 1982 / Pecora 1964), and shape
models (rod: Tirado 1984; sphere: Balog 2015) with Monte-Carlo / delta-method SEs. A real
VH (analyzer) acquisition path remains hardware-gated.

## Locked GUI decisions

- **PySide6** (Qt). The controller pattern keeps everything below the widgets
  framework-agnostic, so the GUI could be reimplemented without touching analysis/physics.
- **Shell: sidebar navigator + module tabs.** A left sidebar lists measurements
  auto-grouped into samples; six tabs (Data, DLS, SLS, Cross-Sample, Utilities, Settings).
  Tabs have scopes and the sidebar adapts: sample-scoped → navigator; aggregate
  (Cross-Sample) → include/exclude; global (Settings) → disabled.
- **Data tab owns parameter editing + commit; the analysis modules are read-only** on
  committed params. Explicit **Update** button with changed-since-commit highlighting,
  pending-update indicator, and undo-to-committed.
- **Settings seed, never override.** Any Settings default that affects a number seeds that
  module's per-run control (still overridable, recorded per result) — never a hidden
  global multiplier. Appearance settings may be purely global.
- **Hybrid workspace grouping**: auto-propose by sample key (polymer, solvent,
  rounded-temperature) + manual override (sidebar right-click).
- **Calibration is a visible panel in the SLS section.**
- **Threading: designed-for, not enabled.** Every expensive call is a single controller
  method so it can move to a worker thread later without touching widgets.
- **Flags are GUI-owned overlays**, never baked into matplotlib figures or export data.
  Saved plot images must be clean. In exports, an uncalibrated result writes "uncalibrated,
  arbitrary scale" into the **Comments** cell of the affected columns only.
- **Sessions: self-contained JSON** (embeds the data; also stores source paths for
  optional reload-from-source). Not pickle.

## Planned (recorded, not yet built)

- Threading enablement; a solvent property library and solute/polymer library (intensity →
  volume/number with an explicit scattering model + validity limits); reload-from-source;
  session schema versioning once the data model stabilizes; caching the per-sample SLS fits
  the Cross-Sample tab recomputes on each refresh; an A₂ source picker and a visual peak
  picker in the DLS distribution view; packaging.
- A real VH (analyzer) depolarized acquisition path (hardware-gated).
