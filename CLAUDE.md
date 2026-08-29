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
  invent a citation. The user-facing citation index is `docs/5. Citation-Index.md`.
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
3. **No hidden, hard-coded, or unoverridable physics.** Pure analysis functions contain
   nothing instrument-, solvent-, or polymer-specific; they operate only on the explicit
   physical parameters handed to them (refractive index, wavelength, temperature,
   viscosity, dn/dc, angle), and every such parameter is user-owned. The program may
   *offer* a physical value as a convenience — the solvent-property library proposes
   refractive index and viscosity from a chosen solvent, temperature, and wavelength,
   for solvents whose sources meet its dense-data grade (lower-grade solvents surface
   their values for reference without proposing them) —
   but only as a transparent, provenance-tagged, freely-overridable default that reaches
   the analysis solely through the user's explicit commit. **dn/dc is never proposed,
   defaulted, or guessed** — it is strictly hand-entered. It may be left absent: the SLS
   engine then computes on a disclosed arbitrary optical scale (K = 1) with every
   scale-dependent output flagged unreliable — absence is honest, a guessed value never is.
4. **Diagnostic flags come from general light-scattering criteria**, never system-specific
   thresholds. Where a threshold is needed it is a function parameter with a sensible,
   documented default — not a magic number for one solvent.
   **What a reliability flag promises:** a flag
   answers **"did this method's preconditions hold for this fit?"** — never "is this a good
   answer?"
   1. A flag is **False when a precondition of the method failed on this run, and its note
      names that precondition — never a proxy.** Preconditions include: the measurement
      contains what the method fits (e.g. a mode's decay completes inside the lag window —
      the `Γ·Δτ` criterion, `analysis/dls/_common.py`); required physical inputs are present
      (viscosity, calibration); and the reported value is one the method can meaningfully
      produce (finite, within the physics the method describes) — value-range is one
      precondition among several, not the definition. Independently failed preconditions are
      each stated; neither implies the other.
   2. A flag is **not a fit-quality score, and does not police the user's point selection.**
      A user who masks, windows or sub-selects owns that choice; the program discloses what
      it did and does not second-guess the intent. A flag turning green under such a
      selection is not a defect — this clause is citable to **decline** a finding.
   **Flag-not-refuse:** a failed precondition flags the result, it
   does not withhold it — a computed value is always reported, carrying its flag, because
   the number is real output the user may legitimately need. NaN is reserved for the cases
   where no honest number exists, each a documented exception (e.g. the sub-floor cumulant
   Rh, an afterpulsing artifact rather than a measurement). Presentation (tier, glyph,
   colour) is the GUI style's business; this invariant governs meaning and where a
   threshold may come from.
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
   display). The default reporting is **conservative** — HC3 never anti-conservative under
   non-uniform precision; a less-robust classical **OLS** SE is offered only as a labelled,
   opt-in choice for literature comparability, and the chosen estimator is recorded on the
   result and emitted in exports. Do **NOT** report a ± from a **single correlogram** (its lag
   channels are correlated, Schätzel 1990 → OLS under-reports; ISO 22412 uses repeats). No SE
   for NNLS/CONTIN (ill-posed) or single-angle Mw (one datum). Every SE that propagates through
   a **nonlinear** transform must be **Monte-Carlo-validated** against the sampling SD before it
   ships (a linear pass-through of an already-validated covariance inherits that validation).
   Where a single-shot SE rests on **few, high-leverage points** (a short concentration
   ladder, a ratio of two fitted coefficients), no residual-based estimator is unbiased: the
   honest target is HC3's **conservative** SE, disclosed as such, with more points or
   replicates as the route to an unbiased ±. When adding or
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
- Toluene Rayleigh ratio is **always computed from Sivokhin & Kazantsev (2021)** via
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
  and often carry outdated Rayleigh ratios. The program computes its own k_c from one
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
│                             #   5. Citation-Index.md (generated citation index),
│                             #   6. Acknowledgements.md, 7. AI-Use-Statement.txt
├── test-data/                # datasets used by validation
│
└── (product code)            # top-level packages: core/ parsers/ physics/ analysis/
                              #   exporting/ plotting/ app/ gui/. Per-package purpose
                              #   and per-file detail: docs/4. Code-Map.md.
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

**Complete and validated:** data model; all parsers (Brookhaven DLS/SLS/trace + raw `.dat`, generic
DLS/SLS/trace, Zetasizer clipboard + export, ALV .ASC multi-angle + single-angle, NanoTemper Panta `.xlsx`); physics
constants (geometry-aware Rayleigh) + solvent property library; utilities + synthetic generator; DLS engine; SLS engine
(unified calibration, Debye, Zimm/Berry, calibration-free A₂); Origin export; matplotlib
plotting; workspace + controller (grouping, commit/working state, sessions). The
**form-factor / optical scattering model** is a pluggable abstraction — Rayleigh, RDG, full
Mie, core–shell, with no hard-coded R⁶ shortcut; its Rg (and calibrated Mw) feed the
Cross-Sample ρ = Rg/Rh and scaling pickers, flagged model-dependent, and a degenerate
ellipsoid / core–shell fit stays selectable rather than being silently excluded.

**GUI:** all six tabs built and validated headless. **Data** (parameter confirmation,
commit/undo/highlight, shared-param propagation, Unit column, and a solvent property library that *proposes*
transparent, provenance-tagged, freely-overridable refractive-index / viscosity defaults —
never dn/dc). **DLS** (cumulant / single
/ double / KWW / NNLS / CONTIN / lognormal, weighted volume/surface/number size
distributions, shared τ window + baseline, multi-measurement co-plotting,
replicate averaging with the platform's first single-sample dynamic ±, and the
Correlogram / Distribution / Γ vs q² / D vs c / Mode Map / DDLS / Summary sub-tabs — the Γ vs q²
and D vs c views being the extrapolated, thermodynamic-tier paths). **SLS** (per-sample
calibration panel → k_c; Zimm / Berry /
Debye / Guinier / single-angle / calibration-free A₂ / Rayleigh; manual-Mw override; data
masking; a Depolarization calculator). **Cross-Sample** (ρ = Rg/Rh table + an
**Architecture** sub-tab — branching contraction factors g and h, chain type, and the
random-branching index — + Rg–Mw / A₂–Mw
log–log scaling; provenance-aware source pickers). **Utilities** (Traces store with
diagnostics; I·sin θ check; synthetic generator; Solvent Explorer sub-tab). **Settings** (seed defaults +
light/dark theme + plot palette + input/display units via `app/units.py`).

**Depolarized scattering (DPLS/DDLS)** is built and feature-complete, validated headless
against synthetic VV/VH ground truth and published tables: static depolarization
(depolarization ratios, Cabannes correction, optical anisotropy δ²), dynamic DDLS
(rotational diffusion D_r = (Γ_VH − Γ_VV)/6, Zero & Pécora 1982 / Pecora 1964), and shape
models (rod: Tirado 1984; sphere: Balog 2015; ellipsoid: Perrin/Tanford, with
prolate/oblate degeneracy handling) with Monte-Carlo / delta-method SEs. A real
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
- **Threading: enabled for the heavy calls.** Each expensive analysis call is dispatched to
  a single worker thread (one job in flight app-wide); the controller stays Qt-free. The GUI
  shows a busy cursor + status message while a fit runs and refuses workspace mutations
  (commit/load/mask/settings) during it. Fast ops (cumulant/exp fit, preview, commits) stay
  synchronous. There is no Cancel (a running scipy call can't be safely interrupted).
- **Flags are GUI-owned overlays**, never baked into matplotlib figures or export data.
  Saved plot images must be clean. In exports, an uncalibrated result writes "uncalibrated,
  arbitrary scale" into the **Comments** cell of the affected columns only.
- **Sessions: self-contained JSON** (embeds the data; also stores source paths for
  optional reload-from-source). Not pickle.

## Planned (recorded, not yet built)

- Reload-from-source; a visual
  peak picker in the DLS distribution view (the backend peak-averaging exists and is
  report-only; the click-to-pick UI is not built); packaging. (Session files are
  **versioned**: a session saved by an older schema is refused with a clear message,
  deliberately without a migration path — re-analysis from source data is the recovery
  route.)
- A real VH (analyzer) depolarized acquisition path (hardware-gated — the development
  instrument is VU geometry, with no analyzer).
