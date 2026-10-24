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
4. When user-facing behavior changes, keep the in-program help current and describe the
   change for the guides (their sources are maintained outside this distribution; `docs/`
   ships the built PDFs).

**In-program help rule (not optional):** whenever a change adds, renames, moves, or
alters the behaviour of a user-facing control, update its help in the same change — the
section's `?` `HelpBadge` (`gui/help.py`: `section_header` / `add_help_to_groupbox`)
and/or its passive `setToolTip`. *How-to-use* help is the visible `?` badge (concise,
lists over paragraphs, no citations); *calculation nuance* is a brief tooltip that points
to the Theory-and-Equations-Guide. Tooltips honour the global Settings "Show tooltips on
hover" gate; `?` badges always work on click.

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
   for solvents in its primary tier (lower-tier solvents surface their values for
   reference without proposing them) —
   but only as a transparent, provenance-tagged, freely-overridable default that reaches
   the analysis solely through the user's explicit commit; the engine never receives a
   value the user has not confirmed, and no solvent constant is baked into an analysis
   function. **dn/dc follows the same mechanism:** no library exists yet, and when one does
   its value is keyed by polymer, solvent, wavelength and temperature, provenance-tagged,
   freely overridable, reaches the engine only through the user's commit, and travels to
   the export beside Mw with its source named. dn/dc may be left absent: the SLS engine
   then computes on a disclosed arbitrary optical scale (K = 1) with every scale-dependent
   output flagged unreliable — absence is honest, an unsourced value never is. Any offered
   default is science-based, never tuned to a particular instrument or lab (see #9).
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
   Likewise DLS **cumulant** results and **distribution-weighted** (CONTIN/NNLS) results
   are never conflated: each is named by its method on screen and in exports, and the
   average each method reports is defined in the Theory-and-Equations Guide. The cumulant
   size is ISO 22412's intensity-weighted *harmonic* mean, so the product never labels it a
   "z-average" (the guide quotes ISO's name once).
8. **Uncertainty is part of the analysis — neither over- nor under-reported, and omitted
   when none is honest.** Report the **statistical** (regression) SE only where the fitted
   points are genuinely independent, and label it as **excluding** calibration/dn-dc
   systematics. Use the shared `analysis/uncertainty.py` toolkit (HC3
   heteroscedasticity-consistent covariance; delta-method propagation) and
   `core/formatting.py`'s `format_pm` display. The default reporting is **conservative** —
   HC3 never anti-conservative under non-uniform precision; a less-robust classical **OLS** SE is offered only as a labelled,
   opt-in choice for literature comparability, and the chosen estimator is recorded on the
   result and emitted in exports. Do **NOT** report a ± from a **single correlogram** (its lag
   channels are correlated, Schätzel 1990 → OLS under-reports; ISO 22412 uses repeats); that
   uncertainty ships only through the replicate-averaging path. No SE
   for NNLS/CONTIN (ill-posed) or single-angle Mw (one datum). Every SE that propagates through
   a **nonlinear** transform must be **Monte-Carlo-validated** against the sampling SD, drawn
   in a regime that stresses the approximation, before it ships; a linear pass-through of an
   already-validated covariance inherits that validation (say so; do not re-run it).
   Where a single-shot SE rests on **few, high-leverage points** (a short concentration
   ladder, a ratio of two fitted coefficients), no residual-based estimator is unbiased: the
   honest target is HC3's **conservative** SE (meet or exceed the sampling spread, never
   under-report), disclosed as such, its direction and rough magnitude noted in the guide and
   a passive tooltip, with more points or replicates as the route to an unbiased ±; its
   Monte-Carlo gate asserts the conservative-safe property (at or above the sampling SD, not
   grossly over), not a symmetric tolerance — the calibration-free 2·A₂·Mw is the current
   instance. When adding or
   modifying ANY analysis, state how it treats uncertainty and document the formula with the
   analysis itself; the Theory-and-Equations-Guide (numbered LaTeX) is where the shipped
   formulas are documented, and its sources are maintained outside this distribution, so a
   contributor proposes the equation text alongside the change.
9. **No owner-machine specifics in the product or docs.** The program (code, config,
   defaults) must never hard-code or assume a particular computer setup — file paths,
   usernames, drive layout, or sync mechanism; anything machine-specific stays
   user-supplied or generalized. Nothing about the maintainer personally or their setup
   appears in user-facing documentation, the author's name and professional contact the
   sole exception.

## Physics that must be enforced in code

- Scattering vector `q = (4π n / λ) sin(θ/2)`; n and λ always explicit.
- Stokes–Einstein `Rh = kT / (6π η D)`; T and η always required, never assumed.
- The refractive-index correction `(n_solvent / n_standard)^x` is **mandatory** on excess
  Rayleigh ratios whenever the solvent differs from the calibration standard. The exponent
  depends on the detection optics (1 ≤ x ≤ 2, Chu 1991 Eq. 8.4.19; x = 1 for slit
  detection). The program ships x = 2 (Coumou 1960); when x becomes settable it is
  user-owned under invariant 3 and recorded on the result.
- Toluene Rayleigh ratio is **always computed from Sivokhin & Kazantsev (2021)** via
  `physics/constants.py`, with an explicit **geometry** (VU / VV / VH). Never use a
  vendor/instrument-stored Rayleigh value. Key facts: `R_VV(532 nm, 25 °C) = 2.34e-5
  cm⁻¹` (Takahashi et al. 2019); `R_VU = R_VV·(1 + ρv)`, `R_VH = R_VV·ρv`; the
  temperature coefficient is **positive** (+0.421 %/°C, Sivokhin & Kazantsev 2021, Eq. 9).
- Cumulant PDI > 0.3 → flag as unreliable.
- **Mw and absolute A₂ require calibration.** If a run is uncalibrated, flag those two as
  unreliable — but **Rg survives** (slope/intercept, the unknown scale cancels) and so
  does the **calibration-free 2·A₂·Mw product**. Don't flag Rg or the product.

## Domain gotchas that drive code decisions

- **dn/dc is the central vulnerability** of low-contrast systems: a small value means weak
  scattering, and the error enters Mw squared, exactly as the solvent refractive index does.
  Its data is what differs: it belongs to the polymer and solvent pair at a wavelength and
  temperature, tabulated values for one pair scatter by several percent between sources, and
  a co-solvent system's measured value can differ from any table. So a value always carries
  its source, wavelength and temperature; it is never unsourced.
- **Vendor calibration constants are display-only.** They go stale after a re-alignment
  and often carry outdated Rayleigh ratios. The program computes its own k_c from one
  calibrant point (intensity, angle, standard + geometry).
- Some systems have **Mw biased by preferential co-solvent adsorption**, so a trustworthy
  Mw may be characterized in a different solvent and **entered by hand**. `SampleResult`
  tracks Mw provenance in three tiers, 'computed' | 'picked' | 'user' (the rule's home is its
  docstring in `core/workspace.py`); a user Mw must never be overwritten by a re-analysis, and
  scaling plots should prefer it.
- Dust in high-viscosity media sits still and causes persistent intensity offsets, not
  transient spikes; relevant if cleanliness diagnostics are added.

---

## Package layout

Top-level packages (each with an `__init__.py`) using top-level package imports
(`from core.data_models import …`, `from analysis.dls import …`).

```
<repo root>/
├── CLAUDE.md                 # this file — agent/contributor orientation + invariants
├── README.md                 # front door (features, how to run, docs index, license)
├── PATCH_NOTES.md            # per-release changes + known issues
├── LICENSE                   # GPLv3
├── requirements.txt          # pinned deps (Python 3.13 venv)
├── Launch DLS Buddy (Windows).bat      # double-click launchers: create the venv on first
├── Launch DLS Buddy (MacOS).command    #   run, install requirements.txt, start the app
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

- The source is Python 3.10-compatible (uses `from __future__ import annotations`), but the
  pinned dependencies need 3.11 or later. The tested interpreter is **Python 3.13**; 3.14 is
  untested.
- Runtime: `numpy`, `scipy` (optimize/nnls/curve_fit), `statsmodels` (ADF test in
  utilities), `matplotlib`, `PySide6`, `openpyxl` (the Panta `.xlsx` loader — a hard
  import-time dependency). Pinned in `requirements.txt`.
- Recreate the venv with `py -3.13 -m venv .venv` (Windows) or `python3.13 -m venv .venv`
  (macOS) then `pip install -r requirements.txt`;
  point your interpreter at `.venv\Scripts\python.exe` (Windows) or `.venv/bin/python`
  (macOS). (`.venv/` is gitignored.) The double-click launchers bootstrap the venv on
  first run and launch the app: `Launch DLS Buddy (Windows).bat` and
  `Launch DLS Buddy (MacOS).command`.

## How to run / validate

- The GUI runs with `python -m gui.main` from the repo root.
- The program is developed against a pytest regression suite (kept outside this
  distribution — nothing in it is needed to
  run the app). Everything is checked against **ground truth**: the program's own
  forward model (a closed round-trip) and the committed real datasets — the DLS engine
  against a real correlogram + the ALV replicate set, the SLS engine + controller
  against the real polystyrene/toluene Zimm set (reproduces Mw ≈ 1.01e6 / Rg ≈ 40.5 nm);
  physics constants asserted analytically. Sessions round-trip to 1e-9; a manually
  entered Mw survives re-analysis; the replicate-averaging ± is Monte-Carlo-validated
  (ratio 1.006 against the sampling SD). The GUI is
  validated headless (real widgets under Qt's offscreen platform) plus real-platform
  screenshot review.
- Plotting is validated by rendering to PNG and inspecting, never by a blocking `show()`.

---

## Current state

**Complete and validated:** data model; all parsers (Brookhaven DLS/SLS/trace + raw `.dat` + the
single-angular-scan SLS export, generic
DLS/SLS/trace, Zetasizer clipboard + export, ALV .ASC multi-angle + single-angle + trace, NanoTemper Panta `.xlsx`); physics
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
Debye / Guinier / single-angle Mw + A₂ ladder (apparent) / calibration-free A₂ / Rayleigh / Form factor P(q);
manual-Mw override; data
masking; a Depolarization calculator; Derived Intensity and Mode-Resolved sub-tabs —
SLS points built from a sample's DLS runs, whole-run or per-mode). **Cross-Sample** (an
**Architecture** sub-tab — the ρ = Rg/Rh table with branching contraction factors g and h,
chain type, and the random-branching index — and a **Scaling** sub-tab — Rg or Rh vs Mw and
A₂ vs Mw log–log scaling; provenance-aware source pickers). **Utilities** (Traces store with
diagnostics; I·sin θ check; synthetic generator; Solvent Explorer sub-tab). **Settings** (seed defaults +
light/dark theme + plot palette + input/display units via `core/units.py`).

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
- **Loading is instrument-agnostic:** the load buttons carry no vendor name.
  `parsers/autodetect.py` tries each strict instrument parser (the roster is
  `parsers/roster.py`'s `DLS_PARSERS` / `SLS_PARSERS`) until exactly one recognises the
  file, then falls back to the generic plain-text parsers, which stay out of auto-detect
  because they match anything (the generic DLS path prompts for the delay unit and data
  form; generic SLS loads directly), and only then reports an unrecognised format.
- **Threading: enabled for the heavy calls** (`gui/worker.py`). Each expensive analysis call
  is dispatched to a single worker thread (one job in flight app-wide); the controller stays
  Qt-free. The GUI shows a busy cursor + status message while a fit runs and refuses
  workspace mutations (commit/load/mask/settings) during it; a superseded result is dropped
  by a per-tab epoch guard. Fast ops (cumulant/exp fit, the k_c preview, commits) stay
  synchronous. There is no Cancel (a running scipy call can't be safely interrupted).
- **Flags are GUI-owned overlays**, never baked into matplotlib figures or export data.
  Saved plot images must be clean. In exports, an uncalibrated result writes "uncalibrated,
  arbitrary scale" into the **Comments** cell of the affected columns only, with no change
  to the row count, so the Origin import stays intact; "calibrated" is the silent default.
- **Sessions: self-contained JSON** (embeds the data; also stores source paths for
  optional reload-from-source), never pickle. A schema change may refuse an older session
  with a message rather than migrate it (version 1 is refused; version 2 loads with its
  notes carried as legacy text); a save never overwrites or deletes a file the program did
  not create; a load never silently misreads a file. One known earlier exception: v1.20.0's
  marker for non-finite values, which a build older than v1.20.0 loads as an object where a
  number belongs.

## Planned (recorded, not yet built)

- Reload-from-source; a visual
  peak picker in the DLS distribution view (the backend peak-averaging exists and is
  report-only; the click-to-pick UI is not built); packaging. (Session files are
  **versioned**: a schema the program no longer reads is refused with a clear message
  rather than migrated — re-analysis from source data is the recovery route.)
- A real VH (analyzer) depolarized acquisition path (hardware-gated: VU-only instruments
  have no analyzer).
