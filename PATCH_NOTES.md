# Patch Notes

User-facing summary of what each release changed, plus a running list of known
issues.

Versioning follows `MAJOR.MINOR.PATCH`: MINOR for new user-facing capability,
PATCH for fixes/polish. The version is shown in the GUI
window title.

> **Scroll to the bottom** for the running list of **known issues / outstanding**.

---

## 1.22.0 — Replicate averaging, analyses that say what they need, click-to-mask, and help rewritten (2026-10-23)

> **Important.** Some numbers move in this release, so re-run a stored analysis rather than
> compare across versions. Every **Berry** *M*<sub>w</sub>, *R*<sub>g</sub> and *A*₂ moves
> (the fit gains its cross term; Zimm is unchanged). A **distribution peak** is reported at
> its geometric mean rather than at the grid point with the most weight. The **Guinier qRg
> validity limit** defaults to 1.0 (was 1.3) and now judges Zimm, Debye and Guinier fits
> alike. Cross-Sample defaults to Zimm before Berry and never to a flagged value. Sessions
> saved by this release do not open in an earlier build; earlier sessions still open here.

### DLS

- **Replicate averaging does more and says more.** Replicates whose temperatures differ
  slightly can be averaged; an averaged correlogram records its runs' amplitude and
  count-rate spread; an average is no longer pooled beside its own source runs by default;
  and an average of flagged fits says so, with each cause once.
- **Distribution peaks are reported at their geometric mean** on every surface, with a ±
  from replicates where the runs agree on the grid, and every peak and mode point carries a
  size verdict (grid end, size range, incomplete decay).
- **An analysis the ticked measurements cannot run is grayed and says what is missing**
  under "Needs:", before Run, on every DLS sub-tab.
- **Results match their inputs.** A refused Run, a regroup or a removal takes the stale
  result off the screen instead of leaving it drawn.

### SLS

- **The Berry fit gains its *q*²*c* cross term**, and every Berry result moves (on the
  PS/toluene set, *M*<sub>w</sub> 9.28e5 → 9.83e5 g/mol).
- **One qRg limit judges Zimm, Debye and Guinier**, flagging *M*<sub>w</sub> as well as
  *R*<sub>g</sub> past it; a Berry fit past it is labeled *past the Zimm limit* rather than
  flagged.
- **A fit the sample cannot run is grayed with its reason**, and Run follows the selection.
- **The Calibration panel reads as your reading, then the standard**, refuses impossible
  entries as you type them, and names a blank standard refractive index as such instead of
  "uncalibrated".
- **Click-to-mask follows what the fit uses**, and calibration-free *A*₂ runs with its lowest
  concentration hidden.

### Everywhere

- **Every note is one clause, the same wherever it appears**, with a letter at the value and
  one line per cause; what a number is reads on its label (*apparent*, *± statistical only*);
  exports carry each caveat in its column's Comments cell.
- **Cross-Sample labels say what a value is and, when flagged, why**, and every Source
  Selection row marks a flagged choice.
- **Help is shorter and checked against the code**: every badge and tooltip fits a word
  budget, and load notices point at the User Manual section for the file format.
- **Faster.** The sidebar, the trace view and hidden plots no longer stall the window, and
  the DDLS shape models arrive with their fit.
- **Guides.** New equations for the replicate amplitude spread, the grid-resolution term and
  the peak geometric mean; attributions narrowed to what each cited page prints; the Manual
  gains tables of what every analysis needs to run.

---

## 1.21.0 — Sourced noise weights for every DLS fit, the single-angle ladder, and rewritten guides (2026-09-25)

> **Important.** DLS numbers move in this release — re-run a stored analysis rather than
> compare across versions. Distribution and mode-resolved results move under the new
> default residual weight, and nonlinear cumulant results under theirs (typically a couple
> of per cent in size; most on long correlograms whose file states no sampling times;
> order-3 fits by more). An L-curve result from an earlier version cannot be reproduced
> bit for bit. To reproduce an earlier distribution, set **Residual weighting** to
> *g₁² (older form)* and, for a GCV result, **GCV robustness γ** to 1. The earlier nonlinear
> cumulant weight is not kept.

### DLS — how the fits weigh the data

- **Distribution fits (CONTIN, NNLS, lognormal) weight by a sourced variance.** The default
  residual weight is *g*₁²/σ², with the per-channel variance σ² of Biganzoli and Ferri (2018)
  evaluated on a first-pass fit. A fit takes about twice as long. A committed mean count rate
  is recommended, never required; without one the fit uses the infinite-count-rate limit
  and says so. The older weight stays selectable per run (**DLS ▸ Distribution ▸ Residual
  weighting**, seeded from Settings), and every export names the weight that built it.
- **CONTIN's default α selector is robust GCV** (Lukas, γ = 0.1), built for the correlated
  noise between correlogram channels, where classical GCV tends to under-smooth. A **GCV
  robustness γ** control sets it per run; γ = 1 is classical GCV. On a noisy sample expected
  to be multimodal, compare against γ = 1: robust GCV can merge two modes classical GCV
  separates.
- **The L-curve is rebuilt on Hansen's method**: residual against the distribution's norm,
  corner found by adaptive pruning. Widening the α range no longer merges two separated
  populations. New notes cover a sweep with no corner or a single point, drawn with the same
  halo on the α-sweep panel as the floor and ceiling cautions; an inverted α range is refused.
- **GCV and the F-test read their degrees of freedom from one decomposition**, exact to
  rounding and several times faster. The F-test gains the two checks CONTIN's manual
  prescribes, each shown as a reliability note.
- **The nonlinear cumulant weights by the same sourced variance**, evaluated on a first-pass
  single exponential (Koppel 1972), with the photon-counting terms when a count rate is
  committed. A Brookhaven `.dat` supplies its correlator's own per-section sampling times;
  other formats use a stand-in read off the delay axis, and the result records which. Every
  cumulant the program fits — the DLS tab, Γ vs q², D vs c, replicate averaging, DDLS and
  the Cross-Sample candidates — uses the same weight and count rate for a run.
- **New: an unweighted cumulant**, under **Settings ▸ Cumulant residual weighting**, for
  comparison with fits made without a noise model. It is not the more accurate answer.
- **The floating baseline is bounded to the data's own scale** rather than a fixed ±0.05, so
  low-amplitude correlograms can move (on the worked reline case, *R*<sub>h</sub> by 16 %).

### DLS — cumulant order, moments and flags

- **New: cumulant order 4**, with **µ₃** and **µ₄** rows in the results and the export
  (*unmeasured* below the order that fits them, never 0). They are diagnostics for comparing
  fits of one sample, not a skewness and kurtosis; read µ₄ from the nonlinear method only.
  Order 4 is for broad samples; order 2 stays the default, and the order control's help says
  why order 3 is not recommended.
- **Fixed: an order-3 third moment could be just its starting value.** It now fits;
  order-3 sizes move, some by a lot. Order 2 is unaffected by this fix.
- **New setting: Cumulant fit cutoff** (*Settings ▸ DLS Analysis Defaults*), default 0.1.
  ISO 22412 prescribes 0.01. A lower cutoff lowers the PDI and can unflag a run whose size has
  moved further from the truth. The linear cumulant now fits the contiguous stretch above the
  cutoff, which moves a few runs with a large residual offset slightly.
- **A correlogram with no decay gets no size.** The cumulant returns a blank, flagged size
  naming why; Γ vs q² and D vs c drop that point and name it.
- **Every fit drops dead channels** (g₂ = 0) through one shared rule and counts them in a note,
  and the "did not finish decaying" check reads only the fitted channels.
- **Passive notes (ⓘ)** when µ₂ or µ₄ rests on its bound, when the floating baseline rests on
  its limit, and when channels recorded nothing. Notes never change the reliability flag, and
  they now sit on the cell they concern, in the Summary table and the saved session.

### DLS — delay window, baseline and the tab

- **The baseline region applies to every method** on the Correlogram sub-tab (the exponentials,
  KWW and both cumulants), with a **baseline** row and export note. **The delay window and
  baseline apply only once you move them**: until then each run is fitted over its own full
  lag range, and the Distribution sub-tab uses the fit's own baseline estimate. A baseline
  region outside the delay window is ignored by every method.
- **Background fits can use the tab's region.** Γ vs q², D vs c, the Mode Map, DDLS and
  **Average derived results…** gain a **Use the Correlogram tab's delay window and baseline**
  box, and every fit without an order control of its own runs at the Settings cumulant order.
- **Results tables are one row per run** with the plot swatch beside the name; details for the
  selected run sit in a panel below, and ⚠ sits on the unreliable cell.
- **Fits to show** draws any fit already run over the correlogram — all seven methods,
  distributions included — and marks a fit whose inputs changed *(stale)*.
- **Plots**: color and marker are the measurement, line style is the method, on both plot
  sub-tabs and their residual panels. The on-screen legend is a compact key; **Save** writes
  the full legend into the file.
- **DLS Summary**: **Show fit details** expands each method to every quantity it produced, and
  these are now saved with the session.
- **The Distribution sub-tab keeps each fit** across tick changes, as the Correlogram sub-tab does.
- **Settings ▸ DLS Distribution Defaults** seeds the *R*<sub>h</sub> grid, α selection and range,
  GCV γ and F-test level, and **Cumulant order** is back in Settings. The **Mode Map** gains its
  own **Inversion Settings** with **Copy from the Distribution tab**.
- **Changing a setting a fit reads clears what depends on it, after asking** — Skip initial lag
  channels now clears every DLS result (it used to clear the cumulant ones alone).
- **DDLS and replicate averaging**: a run or angle that cannot be fitted, or decays at a rate at
  or below zero, is left out and named instead of failing the set or skewing the mean; their
  fits' notes are repeated once in the result. An averaged correlogram cannot be averaged again.

### SLS & Cross-Sample

- **New method: Single-angle *M*<sub>w</sub> + *A*₂ (ladder, apparent)** — the Debye plot at one
  angle over the concentration series, with HC3 ± and the calibration-free 2*A*₂*M*<sub>w</sub>.
  Its results are offered (never chosen for you) in the Cross-Sample source picker, one entry per
  angle. The one-datum *Single-angle M*<sub>w</sub> *(apparent)* method is removed.
- **New: SLS without a measured solvent blank**, after a confirmation, with an optional
  **Solvent blank intensity at 90°** on the Data tab (scaled by 1/sin θ). Every result from such
  a run is flagged unreliable, *R*<sub>g</sub> included, and is not offered to Cross-Sample.
- **Derived Intensity**: a blank recorded only as a count-rate trace can be derived; same-angle
  repeats more than 5 % from their mean are reported; re-deriving replaces only that fraction's
  points; runs are named.
- **The scaling CSV has a Source column**, and generic SLS files with a single angle load.

### Data, names and exports

- **Measurements are named, not numbered**: your **Label**, else the file name, with angle,
  concentration and fraction beside it, everywhere a measurement appears, with a hover card.
  Labeling several runs at once numbers them (`PVP-A 1` … `PVP-A 6`). An average reads
  `avg of 6`; an unconfirmed sample reads *(unconfirmed sample)*.
- **Exported files are named for their contents** (sample, analysis, method, measurement,
  angle), and a single-measurement export carries a **Source file** column.
- **Malvern Zetasizer**: add **Wavelength**, **Scattering Angle** and **Mean Count Rate** to the
  export template and they arrive pre-filled. A column **renamed** in the Zetasizer software now
  arrives blank; User-Manual §5.5 lists the template.
- **The Mode Map export gains *Alpha*, *Alpha selection* and *GCV gamma* columns**; the columns
  after them move three places right, so an import template that reads them by position needs
  updating.

### Utilities — Synthetic data

- **Noise that looks like an instrument's**: a **Per-channel + shot noise** model driven by a
  mean count rate, and a **Channel noise correlation** control, both off by default. The per-channel
  models now draw the exact Biganzoli–Ferri variance, so **the same seed regenerates a different
  per-channel set than before**.
- **Baseline offset (g₂−1)** plants a flat offset on the correlogram and the VV/VH artifact;
  **sphere** joins the VV/VH shapes; a multi-angle set's count rate follows the 1/sin θ law.

### Layout and help

- **Run and Export stay in view**: every control column puts **Run** under its settings and
  **Export** at the foot. Fewer pop-ups — a fit that cannot run shows ⚠ in the panel — and each
  **?** badge sits beside its title.

### Fixes

- An invalid CONTIN α range is refused at entry and reverted when a settings file holds one; a
  mean count rate of zero is no longer accepted; a hand-edited `.dat` sampling time no longer
  crashes the nonlinear cumulant; the Correlogram sub-tab's **Run** no longer scrolls out of view,
  its delay axis is labeled once, and a repeated warning is explained once. Plus
  **miscellaneous bug fixes and polish** across the DLS tab.

### Documentation

- **Quickstart Guide**: a tutorial on the PEG 300k example set that ships in
  `test-data/Synthetic/Clean/`.
- **User Manual**: one chapter per tab, with new Data and Settings chapters and new figures.
- **Theory-and-Equations Guide**: rebuilt as one explanation in the order the physics is used,
  with a notation table and thirteen new figures. Both guides were read page by page and the
  statements that did not match the program were corrected.

---

## 1.20.0 — Consolidated Feature List (2026-08-29)

### Static light scattering (SLS)

- **Form-factor P(q) fitting — a whole model family.** A new *Form factor P(q)* method
  fits the full angular form factor (valid to larger *qR*<sub>g</sub> than a slope fit),
  with models for **sphere, Gaussian coil, ellipsoid, full Mie sphere, and core–shell**.
  The ellipsoid returns shape (axial ratio, prolate/oblate); Mie handles large,
  high-contrast particles (e.g. latex in water) where Rayleigh/RDG refuse; core–shell
  returns two radii. Each carries HC3 uncertainties (Monte-Carlo-validated), honours the
  apparent-vs-thermodynamic distinction, and survives an uncalibrated run for
  *R*<sub>g</sub>/shape. Non-spherical models need a hand-entered solute (and shell)
  refractive index — never guessed.
- **Ambiguous fits are made selectable, never chosen for you.** A degenerate ellipsoid or
  core–shell fit lists every admissible solution as an explicit pick (with a consensus
  *R*<sub>g</sub> where solutions agree), flagged *degenerate* through ρ, scaling, and
  exports. A form-factor *R*<sub>g</sub>/*M*<sub>w</sub> can feed the Cross-Sample tab,
  labelled model-dependent.
- **SLS without a d*n*/d*c*.** The increment is no longer required. Ratio-based results
  (every *R*<sub>g</sub>, form-factor sizes, the calibration-free 2·*A*₂·*M*<sub>w</sub>)
  are bit-for-bit unaffected; absolute *M*<sub>w</sub>/*A*₂ are dashed out on a disclosed
  **arbitrary optical scale**, reported separately from an uncalibrated run. A d*n*/d*c*
  of **zero is now refused** (it previously fabricated a reliable-looking size); extreme
  values take the same disclosed path instead of crashing or blanking *R*<sub>g</sub>.
- **Derived Intensity — SLS from DLS runs you already have.** A new sub-tab turns a
  sample's DLS runs into an ordinary SLS measurement (one intensity per angle); Brookhaven
  `.dat` headers supply a whole-run mean count rate even with no trace attached.
- **Mode-Resolved SLS (Sedlák partition).** Split a static series by its dynamic modes
  using the DLS decomposition; run Guinier/Debye/Zimm on a single mode's share
  (**Amplitude Guinier** for apparent per-mode sizes). Mode sets are snapshotted from the
  DLS Mode Map with full provenance.
- SLS plot axes now honour **Settings ▸ Display Units**; each series is referenced against
  its own solvent blank.

### Dynamic light scattering (DLS)

- **Weighted size distributions.** A CONTIN/NNLS/lognormal distribution can be shown
  Intensity (default), Volume, Surface, or Number weighted, with an explicit scattering
  model (Rayleigh point / RDG / Mie) and honest validity gating. Reweighted views carry no
  ± and are labelled model-dependent (ISO 22412 deprecates number distributions from DLS).
- **Mode Map — mode-resolved dynamics.** Plots every extracted relaxation mode of every run
  in (*q*², Γ) space; you click points into mode families (never auto-paired), each getting
  its own *D*/*R*<sub>h</sub> fit plus a slow-mode τ⁻¹ = *aq*² + *bq*⁴ form and a fast/slow
  amplitude report. Picks survive save/reload, matched by value.
- **CONTIN smoothing you can see.** An α-sweep diagnostic panel shows how α was chosen
  (L-curve corner / GCV minimum / F-test), with the pick ringed and over-smoothing haloed;
  exportable.
- **Ellipsoid shape model for DDLS** (prolate/oblate, alongside sphere and rod), with
  per-solution Monte-Carlo ± and degeneracy handling.
- **Uniform reliability preconditions.** Every DLS method now applies the same checks before
  calling a size reliable — including flagging a correlogram that **never decayed** inside
  the delay window — and a cumulant **PDI > 0.3** now flags the *R*<sub>h</sub> everywhere
  it travels (Summary, exports, Cross-Sample candidates, replicate averages). A flag warns
  rather than withholds: the number is still shown.

### Cross-Sample

- **Architecture tab** with branching factors: **g** (from *R*<sub>g</sub>, Zimm–Stockmayer),
  **h** (from *R*<sub>h</sub>, Burchard–Schmidt–Stockmayer), and a **branching index n**
  (random tri-/tetra-functional). Tag samples **Linear**/**Branched** (persisted); branched
  samples are held out of the scaling power-law fits but still plotted.
- **Richer, better-cited ρ = *R*<sub>g</sub>/*R*<sub>h</sub> interpretation** (Burchard 1999),
  correcting the old microgel reading and adding stars, dendrimers, and cyclic chains.

### Utilities ▸ Traces

- **Windowed Hampel overlay** distinguishes a dust spike from a slow drift (rolling median
  ± band); the summary counts **events**, not samples, and abstains visibly where it cannot
  judge.
- **Periodicity and averaging diagnostics** — power spectral density, periodogram (with a
  probability on the tallest peak, marked unreliable on a drifting trace), and **Allan
  deviation** ("how long may I average?"), all against a shot-noise floor and all exportable.
- **Synthetic Generator greatly expanded** — DPLS (VV/VH), branching architecture,
  form-factor series, mode-resolved linked sets, two-mode crossing correlograms, and trace
  parameters (drift, dust, long-residence offset, planted periodicity) — so every analysis
  is drivable from known ground truth.

### Data, loading & the workspace

- **New loaders:** NanoTemper Prometheus **Panta `.xlsx`** (spreadsheet loading adds the
  `openpyxl` dependency — `pip install -r requirements.txt`), Brookhaven raw ASCII **`.dat`**
  correlograms (with their settings), and Brookhaven **single angular-scan SLS** exports.
- **Count-rate traces are workspace objects** — attach/detach/export via right-click; an ALV
  `.ASC` keeps its trace; a per-measurement **Mean count rate** parameter with provenance
  (linked trace / file record), never overwriting a value you typed.
- **Result custody.** Stored results — including a hand-entered *M*<sub>w</sub>, dust masks,
  and per-sample calibrations — now **follow a sample** through regroup/merge/move/undo, with
  a confirm naming anything dropped. Removing a run clears the sample *R*<sub>h</sub> built
  from it. Sessions are written **atomically**, keep non-finite values, and persist pending
  calibration/parameter edits across a quit.
- **Solvent & solute names are stored exactly as entered** (the alias list is gone; `H2O` no
  longer becomes `water`); the library offers 19 solvents split **Measured**/**Estimated**.
- **Solvent library refinements:** four viscosity bands narrowed on measured sources; water's
  refractive index fitted to IAPWS across **15–60 °C**; confidence bands now state accuracy
  (inter-laboratory allowance), display-only.

### Honesty & uncertainty

- **Every ± now says what produced it** (HC3 vs classical OLS) in tables and exports.
- **Arbitrary-scale and uncalibrated are reported separately**, so a dashed value tells you
  whether to calibrate, enter a d*n*/d*c*, or both. Warnings now travel into the exports'
  Comments cells across seven export paths.

### Fixes & hardening

- **Numerical hardening at the extremes** of the number range; near-degenerate concentration
  ladders now complete with trustworthy-or-withheld ±; results that could return a plausible
  zero now return "n/a".
- **Stale-fit hints now cover every input a fit reads**, and a fit never lands on a
  *M*<sub>w</sub> fraction you moved away from; a deferred Run retries on the sample you
  pressed it on.
- Faster form-factor fits (~10×) with no change to results; several startup and mid-fit crash
  fixes; **destructive dialogs now default to No**.
- Plus **miscellaneous bug fixes and polish** across every tab.

---

## 1.0.5 — DLS correlogram polish (2026-07-14)

- **DLS → Correlogram: "tick all at concentration / angle".** The correlogram picker gains the two bulk selectors the Distribution sub-tab already had — tick every listed measurement sharing a concentration (mg/mL) or an angle (°) in one click.
- **Baseline markers are easier to see** — the correlogram's baseline carets and shaded region changed from gray to purple, distinct from the green fit-window markers.
- **Linear delay-time views frame the decay.** On the linear-τ views (lin-lin and log-lin, including the side thumbnails), the delay-time axis auto-frames to the decay of the slowest ticked curve instead of the full lag range. The log-τ main view is unchanged.

---

## 1.0.4 — Documentation refresh (2026-07-14)

- **Docs:** a clarity/consistency pass over all three guides (Quickstart, User-Manual, Theory-and-Equations) — tighter prose, consistent terminology ("Rayleigh ratio", "lognormal"), correct subscripts, and citations collected in one numbered reference list at the end of the Theory guide. User-Manual file-format sections reordered (generic first) and renumbered; the solvent-library chapter trimmed. No change to how the program computes anything.
- **Repo:** the bundled `test-data/` is regrouped into `Real/` and `Synthetic/` folders.

---

## 1.0.3 — Smaller saved sessions (2026-07-13)

- **Polish:** saved session files (`.lsjson`) are now written compactly instead of pretty-printed, roughly 1.5× smaller and a touch faster to load. Sessions saved by older versions still open unchanged, and the file is still standard JSON (`python -m json.tool your-session.lsjson` to read one by eye).

---

## 1.0.2 — Correct uncertainty-estimator label after a mid-session switch (2026-07-13)

- **Fix:** the Cross-Sample ρ = Rg/Rh table's uncertainty-estimator note (`[SE: classical OLS]`) now reflects the estimator that produced the stored Rg/Rh ±, not whichever is selected when the table is drawn. Previously, switching the regression-SE estimator (Settings → *Regression SE estimator*) after a value was computed could label an existing ± with the wrong estimator until re-run. Point estimates and ± values were always correct; only the label could be stale. (Re-run under the chosen estimator to refresh the ± itself.)

---

## 1.0.1 — Backlog polish (2026-07-13)

- The count-rate histogram diagnostic (Utilities → Traces) now reports each fitted overlay's **reduced chi-square** in its legend (χ²ᵣ ≈ 1 is a good fit), alongside the Gaussian and Poisson curves — a quick goodness-of-fit read for the shot-noise check.
- Internal cleanup: removed several built-but-unused functions from the shipped source (no behaviour change).

---

## 1.0.0 — First stable release (2026-07-13)

First stable release — a general-purpose, instrument-agnostic platform for static and dynamic light scattering (SLS/DLS) from polymer solutions.

- **Loading** from Brookhaven, Malvern Zetasizer, and ALV files, plus a generic plain-text fallback; load buttons carry no vendor name and the format is auto-detected.
- **DLS engine** — cumulants, single / double / stretched-exponential fits, NNLS and CONTIN size distributions, Γ–*q*² and concentration extrapolation, and true-replicate averaging.
- **SLS engine** — calibration, Zimm and Berry, Debye and Guinier, and the calibration-free 2·A₂·*M*<sub>w</sub> product; apparent vs thermodynamic results always distinguished.
- **Depolarized DLS** (VU/VV/VH), the ρ = *R*<sub>g</sub>/*R*<sub>h</sub> and power-law scaling cross-sample tools, the solvent-property library, and Origin-compatible CSV export.
- **Desktop application** — six tabs (Data, DLS, SLS, Cross-Sample, Utilities, Settings) with a sidebar workspace navigator, light/dark theming, and two-tier in-program help.
- **Uncertainty** is reported only where fitted points are independent, labelled as excluding calibration/dn-dc systematics, and omitted where none is honest. *M*<sub>w</sub> and absolute A₂ are flagged when uncalibrated; *R*<sub>g</sub> and the calibration-free product survive.
- Incorporates the pre-1.0 hardening pass (numerical/uncertainty rigor, fail-loud input validation, export provenance, GUI stale-state fixes, 0.23.x help/consistency polish). Start with the guides under `docs/`.

---

## 0.23.5 — In-program help & label polish (2026-07-12)

*No change to any calculation or result value. Part of the pre-1.0 audit.*

- **DLS Cumulant order spin gains a tooltip** explaining the order (1 = size, 2 adds PDI, 3 adds skew) and why it greys out for other methods.
- **SLS method-list tooltip now describes the Excess Rayleigh ratio option** it previously skipped.
- **Empty SLS plot shows a centered prompt** ("Load SLS intensities…" / "Press Run…") instead of a blank rectangle, matching the depolarized-DLS plot.
- **Consistent capitalization** — section titles use Title Case ("Parametric Fit", "Traces to Plot"); User-Manual matches in-app names for Solvent Library, Display Units, Paired Angles.
- **Theming fix** — the parameter table's fixed-unit cells ("°", "nm") now follow the light/dark theme.

---

## 0.23.4 — On-screen flag for hot-calibration extrapolation (2026-07-12)

*No change to any calculation or result value. Part of the pre-1.0 audit.*

- **Calibrating above 50 °C or below 10 °C now shows an on-screen ⓘ note** next to k_c in the SLS Calibration panel. The toluene Rayleigh-ratio temperature correction is validated only over 10–50 °C; outside it, k_c is extrapolated (never blocked). The note clears inside 10–50 °C.

---

## 0.23.3 — GUI plumbing fixes (2026-07-12)

*Fixes only — no change to any calculation or result value. Part of the pre-1.0 audit.*

- **Removing a sample with a second solvent blank now removes it fully** — the extra c = 0 blank no longer survives as a member for grouping, removal, and shared-parameter edits.
- **A trace load cancelled mid-batch no longer hides already-loaded traces** — the sidebar updates immediately and still reports the ones that didn't load.
- **A rejected log-scale axis change now explains why** ("log min must be > 0") instead of silently reverting.
- **Smaller robustness guards** for unchecking a per-sample calibration mid-fit, the Depolarization panel Mw read, an out-of-list Settings dropdown, a partial axis-limit edit, a multi-sample manual-entry queue, and a synthetic-generator concentration edge case.

---

## 0.23.2 — Deliberate picks preserved + clearer uncalibrated exports (2026-07-12)

*Fixes only — no change to any calculation or result value. Part of the pre-1.0 audit.*

- **A value you explicitly picked in Cross-Sample is no longer overwritten by a re-run** — a chosen *M*<sub>w</sub>, *R*<sub>g</sub>, A₂ or *R*<sub>h</sub> survives a re-run, like a hand-entered value. Pick again to adopt a fresh fit.
- **Uncalibrated Zimm exports now flag every arbitrary-scale column** — per-concentration Kc/dR columns and extrapolated intercepts carry the "uncalibrated, arbitrary scale" note. *R*<sub>g</sub> stays valid and unmarked.

---

## 0.23.1 — Sturdier settings + stricter file loading (2026-07-12)

*Robustness only — no change to any calculation or result value. Part of the pre-1.0 fail-loud audit.*

- **Invalid saved settings are repaired or reported, never silently wrong** — a wrong-typed value is coerced when unambiguous (`"2"` → 2), else reset to default with a startup notice naming which. A non-object settings file no longer blocks startup.
- **Foreign files are rejected more reliably on load** — a count-rate trace, a Zetasizer clipboard export with unreadable columns, or an intensity list containing `NaN`/`inf` is refused with a clear message instead of partly accepted.

---

## 0.23.0 — Analysis warnings now visible in the app (2026-07-10)

*No change to any calculation or result value — signals previously console-only now surface as passive ⓘ notes.*

- **Unrecognized solvent name** (Data tab) — a note below the parameter table says the name is used as-entered for sample matching.
- **Dropped non-physical points** (SLS tab) — a Berry or calibration-free A₂ fit reports the count and reason of points dropped before its square root / ratio.
- **Negative intensities on load** (generic SLS files) — reported in a "loaded with notes" dialog; values are still stored and used as-entered.
- Console messages are unchanged for scripted/headless use — nothing removed, only made visible.

---

## 0.22.0 — Documentation refresh + Theory-Guide renumbering (2026-07-10)

*Documentation only — no change to any calculation. Also brings the public build up to date with the 0.21.0 ill-conditioned-fit flag and 0.20.3 stress-corpus hardening (below).*

- **Theory-and-Equations-Guide renumbered sequentially** — chapters now run 1–6, equations 1–55 in reading order; every cross-reference (guide, User-Manual, Citation-Index) updated, and a mis-referenced replicate-average SE equation corrected in passing.
- **Quickstart + User-Manual prose pass** — trimmed text, American spelling, expanded Data-tab parameter table, and a decoupled PDF version marker ("Applies to vX.Y.Z and later") so a patch bump no longer forces a doc rebuild.
- **Numbering integrity is checked automatically** at PDF build so chapters/equations/index can't drift out of order again.

---

## 0.21.0 — Ill-conditioned-fit reliability flag (2026-07-10)

A new numerical-health flag for SLS, plus two latent-crash fixes. No change to any correct calculation.

- **A numerically degenerate SLS fit is flagged unreliable** — when the extrapolation design is near-collinear (clustered angles, or two nearly-equal concentrations), the Zimm/Berry, Debye and Guinier *M*<sub>w</sub> (and absolute A₂) and the calibration-free 2·A₂·*M*<sub>w</sub> product carry a red SLS-tab note and export Comments note advising more/wider-spread concentrations or angles. The value is still shown; *R*<sub>g</sub> is unaffected. The test is the design's condition number (scale-invariant), never an "*M*<sub>w</sub> too large" cutoff.
- **A single-angle *M*<sub>w</sub> from a vanishing scattering signal is flagged** — when *Kc*/Δ*R* is so small its reciprocal loses all precision.
- **Two edge-case crashes fixed** — a Guinier plot with an overflowing dR(0) intercept and a near-singular linear fit (tiny negative variance) now degrade to a flagged/uncertainty-free result.
- **A degenerate fit never reports a ± of exactly zero** — a non-positive propagated variance omits the uncertainty rather than showing 0.

---

## 0.20.3 — Robustness hardening (2026-07-10)

Guards and reliability flags for physically impossible input and fits, driven by an adversarial stress-test corpus. No change to any correct calculation.

- **Impossible files are refused at load** — a correlogram with `nan`/`inf`, a non-strictly-increasing delay axis, or a trace with a negative count rate is rejected with a clear message.
- **An implausible temperature or refractive index is flagged** — a likely unit slip (2980 K for 298 K, *n* = 13.3 for 1.33) shows a Data-tab note but the value is still used as entered.
- **Impossible cumulant fits are no longer marked reliable** — a negative PDI (over-subtracted baseline) or negative decay rate (a rising correlogram) is flagged unreliable.
- **A non-physical Zimm/Berry *M*<sub>w</sub> is caught** — a non-positive extrapolated intercept reports NaN and is flagged, matching Debye.
- **Extreme intensities no longer crash the SLS math** — near-ceiling values degrade to a finite result.
- **A constant (all-zero) intensity trace** reports a coefficient of variation of 0, not NaN.

---

## 0.20.2 — Documentation overhaul (2026-07-09)

A top-to-bottom guide review, science and citations checked first. No change to the analysis.

- **The Quickstart is now two documents** — a short *Quickstart-Guide* (load → confirm → run → export, with annotated screenshots) and a new comprehensive **User-Manual** (every module, format, parameter) that adds a **solvent table** of the Solvent Explorer's built-in solvents and their n/η validity ranges.
- **The Advanced Guide is renamed the *Theory-and-Equations-Guide***.
- **The `docs/` folder is numbered** — 1. Quickstart-Guide, 2. User-Manual, 3. Theory-and-Equations-Guide, 4. Code-Map, 5. Code-References, plus new **6. Acknowledgements** and **7. AI-Use-Statement**.
- **Citations audited** against primary sources; each confirmed present in the library and supporting its claim.
- **In-source comments cleaned up** — private-note shorthand rewritten to stand alone; every literature citation kept.

---

## 0.20.1 — Cross-Sample crash fix (2026-07-09)

- **Fixed a crash opening Cross-Sample for a sample whose DLS measurements have no concentration.** The ρ table's *R*<sub>h</sub> picker now handles a missing concentration: a multi-angle set still yields its Γ-vs-*q*² (*q*→0) *R*<sub>h</sub>, labelled *"at c = ?"*, while the D-vs-c (*c*→0) extrapolation is offered only for measurements that have a concentration. No numbers change when concentration is present.

---

## 0.20.0 — Uncertainty-driven result precision (2026-07-09)

Results are shown to the precision their uncertainty supports. The numbers are unchanged; only displayed digits.

- **A value with a ± is rounded to the place its ± supports** (e.g. a *R*<sub>g</sub> with ±2 nm SE shows as `42 nm`), across the Cross-Sample ρ table, the DLS Summary *R*<sub>h</sub> table, and SLS results.
- **The ± is shown at the same place** so value and uncertainty agree (`42 ± 2`, never `41.7 ± 1.9`).
- **Results with no honest uncertainty keep a fixed significant-figure precision** — single-correlogram sizes, NNLS/CONTIN peaks, single-angle *M*<sub>w</sub>, and every ±-less number (Γ, PDI, R², qRg, distribution weights). **3 by default**, set under **Settings → Result Formatting → No-uncertainty precision**; it never affects a value that has a ±.
- **Consistent scientific notation** everywhere (e.g. `1.23e6`).

---

## 0.19.0 — Unified "current sample" selection (2026-07-09)

One sample is active at a time and every tab follows it. No analysis or numbers change.

- **The Workspace tree is a navigator for every tab** — clicking a sample/measurement/heading switches Data, DLS, SLS, Cross-Sample, and I·sin θ. Previously SLS / I·sin θ kept their own sample and Cross-Sample ignored the tree.
- **In-tab dropdowns and the sidebar drive one choice** — either updates the other. One source of truth.
- **A sample a tab can't show reads cleanly** — e.g. "No SLS data for <sample>" instead of silently staying on another sample; calibration and last result are kept on return.
- **Per-measurement ticking** (DLS overlay/fit checklists) is unchanged — separate from which sample is active.

---

## 0.18.0 — Accessibility floor (2026-07-08)

Meets the accessibility floor: colour is never the only signal, text/UI contrast meets WCAG AA in both themes, plus new appearance settings. No analysis or numbers change.

- **Colorblind-safe by default, colour never the sole cue** — overlaid series differ by marker shape / line style as well as colour; the default palette is Okabe–Ito (the palette control notes this).
- **Contrast pass** — theme colors nudged so every text/label/marker clears WCAG AA in both themes.
- **Match plot to app theme (opt-in)** — a Settings option themes on-screen plots dark when the app is dark. **Saved/exported images always stay white.**
- **UI density** — a new Compact / Comfortable / Large setting scales app-wide text size.
- **Reopen last session on startup (off by default)** — the workspace is auto-saved on close and restored next launch; a missing/unreadable session starts empty.

---

## 0.17.10 — Empty-state onboarding + layout polish (2026-07-08)

Onboards a newcomer on the first screen and fixes layout/polish defects. No analysis or numbers change.

- **Empty workspace onboards** — Data / DLS / SLS / Cross-Sample show a centred "← Load a DLS correlogram or SLS intensities to begin" instead of chrome over blank tables. The correlogram draws no empty satellite/residual boxes before a curve exists; the Data provenance legend is hidden until a measurement loads.
- **Distribution picker fixed** — the bulk-select rows no longer overlap the measurement checklist.
- **Calmer "changed field" highlight** — an edited-but-unsaved cell is a theme-aware tint with italic value, replacing the hard yellow fill.
- **Correlogram title no longer clipped** — clears the y-axis gutter; the promote hint moved below the plot; linear side views are pinned to a sensible range.
- **Uncalibrated Mw reads honestly** — shown as "— (uncalibrated)". *R*<sub>g</sub> and the calibration-free 2·A₂·*M*<sub>w</sub> product are unaffected.

---

## 0.17.9 — In-program help: consistent tiers, fuller coverage (2026-07-08)

A help-only pass; no analysis or numbers change. Brings `?` badges and tooltips into one system.

- **No literature citations in the UI** — the CONTIN α-selection help drops author-year cites and points to the Advanced Guide.
- **New `?` help buttons** on the Traces and Synthetic Generator tabs.
- **Clearer SLS help** — the Data Mask and Depolarization boxes move their how-to onto a `?` button with a short hover note.
- **More tooltips where missing** — the VU/VV/VH geometry selector (SLS calibration + Settings), calibration fields, Guinier qRg limit, SLS method selector, Traces diagnostic selector.
- **Trimmed the longest tooltips** to a sentence or two pointing to the Advanced Guide.

---

## 0.17.8 — Terminology & casing normalized (2026-07-08)

A text-only consistency pass — one verb per action, one spelling, one casing, one unit rendering. No logic, analysis, or exported numbers change.

- **One word for "commit edits": Update** — the SLS "Apply parameters" and Settings "Apply" buttons are now **Update**; the SLS pending note reads "press Update".
- **One word for "run an analysis": Run** — "Run fit", "Run DDLS", "Run analysis", "Compute depolarization" are all **Run**. ("Generate" for synthetic data keeps its name.)
- **American spelling throughout** — analyse→analyze, normalised→normalized, grey→gray, labelled→labeled.
- **Consistent casing** — Title Case for tab/section titles; Sentence case for buttons and form labels, each ending in a colon.
- **One degree rendering** — "(deg)" → "(°)".
- **"Show all (clear mask)" is now Clear mask.**

---

## 0.17.7 — Consistent warning tiers across DLS + Cross-Sample (2026-07-08)

A visual-only pass making the two-tier warning system (from 0.11.1) consistent everywhere. No analysis, thresholds, or wording changed — only which tier a flag is painted in.

- **Neutral DLS notes are now calm, not alarm-red** — on Γ vs *q*² / D vs c and DDLS, a qualifier like "± statistical only" renders as a steel-blue ⓘ note. A genuine problem (a non-diffusive Γ-q², or unphysical *D*<sub>r</sub> ≤ 0) still shows red ⚠.
- **Cross-Sample "uncalibrated" badges are now red, not amber** — the *M*<sub>w</sub> / A₂ badges and panel banner; amber reverts to meaning only "pending update".
- **A skipped run reads the same everywhere** — a size-distribution (NNLS/CONTIN) measurement that failed to fit now shows on the red flag line.
- Under the hood, all flags route through one shared renderer so glyph and colour tier can't disagree.

---

## 0.17.6 — Internal cleanup + minor performance (2026-07-07)

The last internal code-review batch — naming/doc/dead-code tidy and redundant-work speedups. Almost nothing user-visible; no scientific number changes.

- **Session files are now strict, portable JSON** — non-finite values are written as `null` (no `NaN`/`Infinity` tokens) and reload correctly.
- **Boolean columns export as `yes`/blank**, not `1`/`0`.
- **Scattering angle is validated** — an out-of-range angle (≤ 0 or ≥ 180°) raises a clear error.
- Under the hood: de-duplicated masked-point overlays and the SLS draw-vs-click transform, routed Data-tab solvent readouts through the controller, and vectorised two count-rate trace hot loops — all output-preserving.

---

## 0.17.5 — Edge-case hardening: robust parsers, honest failed fits, portable sessions (2026-07-07)

Robustness fixes (fourth review batch). No new features.

- **Zetasizer files are parsed more robustly** — a comma inside a quoted name in an export no longer shifts columns (a misaligned row is rejected); a mostly-non-numeric clipboard file is rejected rather than loaded as blank data.
- **A failed fit no longer looks like a good one** — a non-converged DLS fit reads **n/a** instead of the starting estimate; a failed nonlinear cumulant is labelled **(failed)** and doesn't fall back to the linear fit under the same row.
- **No impossible negative molecular weight** — a noisy single-concentration Debye fit with a non-positive intercept reports *M*<sub>w</sub> as **n/a**. *R*<sub>g</sub> unaffected.
- **Cumulant fit-quality (RMS) is comparable across methods** — linear and nonlinear report RMS over the same points.
- **Older builds can open newer session files** — an unknown extra field is ignored.
- **Assorted robustness guards** — a flat-topped distribution peak counts as one population; the trace running-average window is exactly the requested width; degenerate edge inputs (zero-length trace, zero/negative ALV viscosity, DLS/SLS temperature round-off, baseline-less relative view) are handled cleanly.

---

## 0.17.4 — Performance: snappier Cross-Sample refresh, responsive Traces tab (2026-07-07)

Speed and responsiveness (third review batch). No change to any result.

- **Faster Cross-Sample refresh** — DLS *R*<sub>h</sub> fits are computed once and reused until an input changes, instead of re-run for the automatic pick and again per source dropdown.
- **Responsive Traces tab** — a trace is drawn and its histogram / block-variance diagnostics computed once per click; for a long trace the diagnostics run in the background so the window no longer freezes.
- **No slow memory growth from plotting** — figures no longer accumulate in matplotlib's global registry.

---

## 0.17.3 — Reliability fixes: no silently-dropped blanks, fractioned series plottable (2026-07-07)

Bug fixes (second review batch). No new features.

- **Fixed: a second solvent blank (c = 0) could silently disappear** — the first-loaded blank stays the active reference; any extra is kept and shown in the Workspace tree as *"extra blank (unused)"*, and loading more than one blank raises a notice naming the one in use.
- **Fixed: an *M*<sub>w</sub> series stored entirely in named fractions was missing from the scaling plots** — eligibility and plotting now use the same per-fraction test, so such a series appears in the Cross-Sample *R*<sub>g</sub>–*M*<sub>w</sub> / A₂–*M*<sub>w</sub> views and is fitted.
- **Clearer error when dn/dc hasn't been entered** — the intended *"dn/dc must be a finite number"* message instead of an internal `TypeError`. (dn/dc is still never guessed or defaulted.)
- **Internal:** regression-uncertainty routines return "no ±" instead of crashing on a rank-deficient fit (e.g. a single-angle, single-concentration Zimm/Berry).

---

## 0.17.2 — Correctness fixes: A₂ provenance + calibration-free A₂ guard (2026-07-07)

Bug fixes (first review batch). No new features.

- **Fixed: A₂ uncertainty / calibration label could be left stale after a Zimm/Berry run** — the ±, calibration flag and description are now refreshed together with the value. (A hand-entered A₂ is still never overwritten.)
- **Fixed: the calibration-free 2·A₂·*M*<sub>w</sub> could be corrupted by a near-zero data point** — points whose solvent-subtracted scattering falls to ≤ 0 are dropped with a warning (as Berry does); a non-positive reference concentration or too few remaining points stops the analysis with a clear message.
- **Internal:** ticking a Cross-Sample sample while a background fit runs now waits for it to finish, closing a rare race.

---

## 0.17.1 — Cross-Sample source panel: explicit selection + A₂ picker (2026-07-07)

A redesign of how you pick which fitted values represent each sample. Analysis numbers unchanged.

- **The focused sample is chosen explicitly, in the tab** — the source panel has its own **Sample** and **Fraction** selectors; the ρ table is a read-only results view and the Workspace tree merely highlights the focused sample.
- **New A₂ source picker** — the A₂–*M*<sub>w</sub> plot has a visible A₂ row to choose which fit supplies A₂ (Zimm or Berry). A₂ has **no hand-entry box** (unlike *M*<sub>w</sub>) — only fit-derived values are offered.
- **Uncalibrated *M*<sub>w</sub> / A₂ are flagged in the panel** — each row shows a **⚠ uncalibrated** badge with a one-line banner. *R*<sub>g</sub> is scale-independent and never flagged.
- **The source dropdowns are less overwhelming** — candidates group by result type (extrapolated / distribution / averaged / single-condition), hiding per-angle / per-concentration apparent values behind a **"Show all single-condition results"** checkbox.
- **A source you pick now sticks** — a non-default *R*<sub>g</sub> / *R*<sub>h</sub> / *M*<sub>w</sub> / A₂ survives switching tabs or committing an edit.

---

## 0.17.0 — UI/UX pass: consistent explicit-Run, clearer selection, tidier plots (2026-07-07)

Usability fixes and one consistency change from an owner feedback round. Analysis results unchanged.

- **SLS now works like the DLS tabs: nothing recomputes until you press Run** — switching sample/fraction or committing redisplays the last fit and, if a feeding input changed, shows *"Inputs changed since this fit — press Run to refresh"*. A mask edit still refits immediately.
- **"Tick all at concentration / angle" shows units, and the Distribution tab gets both** — the Γ vs *q*² / D vs c dropdowns read **mg/mL** and **°**; the Distribution tab gains **tick all at concentration** and **tick all at angle**.
- **Excluded points are labelled on the plot** — grey ×'s (unticked Γ vs *q*², D vs c, DDLS) and hollow grey SLS points appear in the legend ("excluded" / "masked").
- **Data tab keeps confident trailing zeros** — a library-filled *n* or viscosity shows to the precision its uncertainty supports (e.g. `1.3300`). Hand-entered values show as typed.
- **Distribution-tab layout fixes** — control labels no longer clip at default width; the Peak results table grows to fill; the distribution/residual divider no longer overlaps the axis label.
- **Faster Cross-Sample refresh** (internal) — SLS fits cached and reused until an input changes.

---

## 0.16.0 — Γ vs q² / D vs c: explicit-Run workflow + easier point selection (2026-07-06)

The multi-angle **Γ vs *q*²** and concentration-series **D vs c** tabs were reworked to be predictable about when they compute and easier to load.

- **Nothing is fitted or plotted until you press Run** — the points table lists measurements as metadata only; Γ and *D*<sub>app</sub> read "—" until Run.
- **The table refreshes when you commit** — confirming Data-tab parameters repopulates the Γ-q²/D-c tables immediately.
- **Ticking only selects; it never refits or clears the plot** — a changed selection marks a shown result *"selection changed — press Run to refresh"*.
- **New selection helpers** — **Select all**, **Select none**, and **Tick all at** a chosen value (concentration on Γ vs *q*², angle on D vs c).
- **Clearer eligibility** — un-tickable rows are greyed with a hover tooltip saying why (unconfirmed parameters, or a different *M*<sub>w</sub> fraction); after a Run, a point whose cumulant fit failed or whose PDI exceeds the limit is flagged.
- **"Inputs changed — press Run to refresh" hint across every DLS analysis tab** (Correlogram, Distribution, Γ vs *q*², D vs c, DDLS) — the plot stays but a status line marks it out of date, and only for results whose own inputs changed.

> **Note.** Starting with this release the developer test suite is no longer included in the download — it is development tooling, not needed to run the program.

---

## 0.15.1 — Water refractive-index band: honest widening at warm temperatures (2026-07-06)

- **Water's dn/dT slope allowance is widened** so the confidence band bounds the measured drift across 15–45 °C: the box-wide *n* uncertainty becomes **9×10⁻⁴** (was 6×10⁻⁴ pre-0.15), correcting an under-statement above ~38 °C where water's dn/dT steepens (IAPWS, Harvey *et al.* 1998). Near room temperature nothing changes — the band still narrows to ~3×10⁻⁴ and auto-filled values are identical.
- **No other solvent is affected** — water was the only record whose served range outruns its source's measured temperatures.

---

## 0.15.0 — Solvent library: per-condition confidence band (2026-07-06)

The library's display uncertainty is now evaluated at your wavelength and temperature, not one flat figure for the whole validity range.

- **The Explorer band grows and narrows honestly** — narrowest near each source's best-measured conditions, wider toward the validity-range edges. Its shape derives from real quantities (source-stated accuracy, the library's fit-residual envelopes, standard dn/dT-slope propagation — Advanced Guide §12, Eqs. 52–53). A bulk/handbook source with no temperature shape stays **flat**.
- **Numeric readouts and auto-fill follow** — the per-condition value is used, so mid-range look-ups often gain a digit (water at 532 nm, 25 °C now reads *n* = 1.3349 ± 0.0003, was ± 0.0006). Box-wide tooltip figures remain the band's maximum.
- **Honesty corrections both ways** — most box-wide *n* uncertainties tightened; a few viscosity figures widened slightly (water 1.0 → 1.1 %, *n*-hexane/methanol 2.0 → 2.1 %, acetone 5.5 → 5.8 %, THF 6.0 → 6.1 %, ethylene glycol 4.9 → 5.0 %). Three viscosity validity ranges were clipped to their source's stated range: toluene 260–370 K, benzene 288–340 K, methanol 273–343 K.
- **The uncertainty is still displayed, never folded into a reported ±** — the underlying n/η fits are identical; an auto-filled value may keep one more digit where the per-condition uncertainty is genuinely tighter.

---

## 0.14.1 — Solvent Explorer relocation + layout; auto-fill rounding (2026-07-06)

- **Solvent Explorer moved to a Utilities sub-tab** (fourth, after Traces, I·sin θ, Synthetic generator), restoring the six-tab shell; behaviour unchanged (needs no loaded sample, never writes into a measurement).
- **Two Explorer plots now share one canvas** side by side (single toolbar), resizing together; freed space goes to the condition/readout area.
- **Temperature-plot axes swapped**: viscosity on the **left** (log-capable, log by default), refractive index on the right.
- **Auto-filled values rounded at the last confident digit** — the place its library uncertainty can stand behind (e.g. water at 25 °C, 633 nm → *n* = 1.3315, η = 0.891 mPa·s). Uncertainty picks only the rounding place, never entering analysis; hand-typed values untouched. Re-running auto-fill on a pre-0.14.1 session updates fields once to the rounded form.

---

## 0.14.0 — Solvent property library: auto-fill + Solvent Explorer tab (2026-07-06)

- **Data tab auto-fill.** Solvent name is a dropdown of the library's primary solvents (water, toluene, ethanol, methanol, benzene, cyclohexane, *n*-hexane, acetone, glycerol, ethylene glycol; free text still works). Picking one fills **refractive index** and — for DLS — **viscosity** from temperature and wavelength, marked with a **teal dot**; they re-derive on temperature/wavelength change, a hand-typed value is never overwritten (no dot), and outside a solvent's validity range the library value is cleared for manual entry. **dn/dc is never proposed.**
- **New Solvent Explorer tab.** Global look-up + visual calculator over the whole library (both confidence tiers; estimate-tier labelled). Enter solvent, temperature, wavelength to read *n* and η with uncertainty and a tier badge; two plots (*n* and η vs temperature on twin axes, *n* vs wavelength), each drawn only across the covered range with a **confidence band** and a condition marker. The band is for judgement only — never folded into any reported ± (Advanced Guide §12).
- **Settings → Solvent library → Default solvent** sets the Explorer's opening solvent.

The library only *offers* values; nothing enters analysis without your commit, and no solvent constant is baked in.

---

## 0.13.1 — CONTIN F-test: exact Provencher degrees-of-freedom (2026-07-03)

- **F-test α-selection corrected** to use the effective degrees of freedom **at the least-squares reference**, held fixed across the α sweep (per Provencher 1982), where 0.13.0 used the per-α value. It now selects a **smoother** α, in the same range as the L-curve. Only the F-test option is affected; default L-curve and all else unchanged.

> **Re-run** any distributions fitted with the F-test in 0.13.0.

---

## 0.13.0 — CONTIN α selection: L-curve (default) or Provencher's F-test (2026-07-03)

- **Choose how CONTIN picks α** on the DLS Distribution tab:
  - **L-curve corner** (default, unchanged): elbow of the fit-vs-smoothness trade-off.
  - **F-test (probability to reject)**: Provencher's 1982 criterion — smoothest solution whose fit is not significantly worse than the best. A **probability-to-reject** field (default 0.50) tunes smoothness (higher = smoother).
- The selector shows only the chosen method's controls; the method (and F-test level) is noted under the plot and written into the CONTIN export. Distributions are unchanged when L-curve is left selected.

> **Note.** A single correlogram's lag channels are correlated, so the F-test level is a guide, not an exact test — hence L-curve remains the default (Advanced Guide, CONTIN §).

---

## 0.12.0 — Choose the uncertainty estimator: HC3 (default) or classical OLS (2026-07-03)

- **New Settings → Uncertainty → Regression SE estimator** switches the ± for every straight-line/multilinear fit (SLS Zimm/Berry, Debye, Guinier, calibration-free A₂; DLS Γ vs *q*² and *k*<sub>D</sub>; the *R*<sub>g</sub>/A₂–*M*<sub>w</sub> scaling exponent):
  - **HC3 (robust)** — default; heteroscedasticity-consistent SE that never under-reports under varying point precision.
  - **Classical OLS** — textbook *s*²(XᵀX)⁻¹; choose only to reproduce a value like-for-like from classical software/tables. Can under-report (~10 % low on a short, high-leverage ladder).
- The choice is global, persists across restart, and is **recorded on each result and in the export** (an OLS ± column's *Comments* cell reads `SE: classical OLS`; HC3 silent). Point estimates are identical either way. Switching asks to confirm, then refreshes affected results.
- **SLS and DLS CSV exports now include the previously-omitted ± columns** (*M*<sub>w</sub> SE, *R*<sub>g</sub> SE, A₂ SE, D SE, …). See Advanced Guide §15.1.1.

---

## 0.11.1 — Calmer warnings: neutral qualifiers no longer look like errors (2026-07-03)

- **"Apparent" and "± statistical" notes now read as a neutral steel-blue ⓘ line**, distinct from the bold red **⚠** reserved for a genuine data-quality issue (uncalibrated, or Zimm/Berry routes disagreeing >10%). Nothing hidden; a qRg out of the Guinier regime still escalates to red.
- **DDLS shape verdict moved to a hover tooltip** — "Sphere consistent / inconsistent" and the assumed-geometry caveat show on hovering the shape table; ✓/✗ and ratio cells plus the panel title *"assumed geometry — not measured"* keep the gist at a glance.
- **Trimmed the distribution replicate-average peak caveat** to one sentence.
- **An uncalibrated single-angle *M*<sub>w</sub> is now flagged** `[unreliable — uncalibrated]`, shows the red **⚠** tier, and writes "uncalibrated, arbitrary scale" into the export — matching Debye/Guinier/Zimm.

Data-conditional warnings (PDI>0.3, non-diffusive γ-*q*², uncalibrated, >10% disagreement, unphysical ρv) are unchanged.

---

## 0.11.0 — One consistent way to choose measurements (2026-07-02)

- **One picker everywhere — tick to include.** Every analysis tab selects with **checkboxes** (tick = use), grouped by sample, with **Select all / none** and a **?** help button; ticked rows read blue. Replaces the older highlight-to-select lists across DLS Correlogram/Distribution/Summary, Γ vs *q*², D vs c, and DDLS.
- **The Workspace list mirrors the active tab's selection** (bold blue on matching measurements); it navigates while the tabs decide what is analysed. Unconfirmed samples read **amber**.
- **Clicking a measurement of a different kind no longer blanks the tab** — each tab keeps its own selection.
- **SLS and I·sin θ now have their own Sample dropdown** (only samples with the needed data appear); I·sin θ previously had no picker.
- **DDLS: tick angles to include** in the D_r/D_t fit (all ticked by default; "Include all angles" re-ticks) instead of clicking plot points.
- **Cross-Sample: replicate-averaged *R*<sub>h</sub> is now a selectable source**, preferred over a single-correlogram value, so refreshing no longer overwrites an averaged *R*<sub>h</sub>.
- **Cross-Sample result tables follow the display-unit setting** — ρ table's *R*<sub>g</sub>/*R*<sub>h</sub> columns and manual-entry fields honor **Settings → Display units** (renamed from "Plot axis units"); values stored canonically, converted for display, and a typed value is read in the shown unit.
- **Depolarized dynamic sub-tab labelled "DDLS" everywhere** (was "DPLS" on the tab). No calculation change.

No analysis numbers changed.

---

## 0.10.0 — Background analysis (2026-07-02)

- **Heavy analyses now run in the background — the window no longer freezes.** A slow fit (CONTIN distribution, DDLS fit, replicate averaging, Γ-vs-*q*² / D-vs-c, SLS Zimm/Berry/Debye/Guinier, trace ADF check, synthetic-data generation) keeps the window responsive; the **Run** button shows a busy state and cursor. **No number changed** — results identical to the synchronous version.
- **One analysis at a time.** While a fit runs, starting another or editing parameters/calibration/masks/settings is held off with a wait note. There is **no Cancel** — a running computation can't be safely interrupted; change inputs mid-run and the stale result is discarded on arrival.

---

## 0.9.0 — Usability batch (2026-06-30)

Readability and theme polish. **No change to the analysis math.**

- **Dark theme is easier to read** — **"?"** help buttons and several greyed notes/labels now have proper contrast on dark.
- **Settings section headers readable on the light theme** (were near-invisible grey) and theme-aware.
- **Colours follow a theme switch immediately** — headers, notes, flags, and "?" badges retint live.
- **Tab labels no longer clip.**
- **Overlapping plot labels are readable** — peak labels stack neatly, cap at six with "+N more"; same for sample-name labels on Cross-Sample scaling plots.
- **Correlogram baseline-region boxes read "low"/"high"** (was "lo"/"hi").
- **macOS launcher** — start via **`Launch DLS Buddy (MacOS).command`** (Windows launcher renamed **`Launch DLS Buddy (Windows).bat`**); first run sets up the environment automatically.
- **Data tab: Enter jumps to the next field.**
- **Distribution tab: a peak-results panel** — per-method peak size + weight for ticked measurements, colour-matched to the curves.
- **Draggable boundaries are now visible** — splitter dividers and the fit/residual boundary show grips/handles.
- **Resizable control columns** on the DLS Correlogram/Distribution and SLS tabs (drag grips to resize list, controls, results, masks).
- **Correlogram markers easier to grab** — window carets at top, baseline carets at bottom, so overlapping markers are separable.
- **Γ vs *q*² and D vs c tabs reworked** — each shows a tick-box table of the sample's measurements (choose any subset), per-point Γ/D and fit results (D, *R*<sub>h</sub>, R², …) in tables; Γ/D come from an internal cumulant fit of each correlogram. Run enabled once ≥2 distinct angles/concentrations are ticked.
- **DDLS and SLS results shown as tables** (Zimm/Berry, Debye, Guinier, single-angle, calibration-free A₂, depolarization).
- **Plots no longer disappear when switching measurements** — each tab keeps its last view.
- **Quickstart: a "workflow at a glance" overview** (Load → confirm parameters → run a method → export), and **Chapter 10 + launch/Visualization sections synced** to the reworked tabs.

---

## 0.8.2 — In-program help system + tooltip toggle (2026-06-29)

**No change to the analysis math.**

- **"?" help buttons** on key sections (Workspace, Data parameters, DLS fit/distribution methods, SLS calibration and analysis, Cross-Sample) give concise how-to-use help and point to the Advanced Guide.
- **Show/hide tooltips** — **Settings → Tooltips → "Show tooltips on hover"** toggles passive tooltips ("?" buttons still work on click when off).
- **Clearer method guidance** — hovering a method selector compares options (Cumulant vs distribution; Zimm vs Berry; CONTIN vs NNLS vs Lognormal) and spells out thermodynamic vs apparent.
- **Less repetition** — a fresh app no longer says "load data" in three places.
- **Fixed: the SLS control panel could clip on the right** — checkbox renamed **"Per-sample calibration"** (explanation in its tooltip); any over-wide control panel now gets a horizontal scrollbar instead of clipping.

---

## 0.8.1 — UI/UX batch: bugfixes, Undo/Reset, results panel, resizable plots (2026-06-29)

**No change to the analysis math.**

- **Fixed: Correlogram τ-window/baseline markers** now sit at correct positions (were drawn in seconds on a microsecond axis); dragging works. Fit window and results were always correct.
- **Fixed: an SLS sample's "Mw fraction" row showed a stray "mPa·s" unit** — now correctly unitless.
- **Undo steps back to the previously applied parameters** — first Undo discards un-applied edits, again steps through earlier Updates; disabled when nothing to undo.
- **New "Reset" button** clears all entered parameters for the sample (keeps the instrument scattering angle), asks confirmation, and is undoable.
- **DLS results in one place** — the Correlogram tab has a single results table for the selected method (Cumulant / Single / Double / Stretched Exponential) with **Export CSV** beneath it.
- **Plots fit the screen and resize** — the control panel scrolls on its own; drag the gap between a fit and its residual to resize; the Utilities Traces tab gets a draggable vertical splitter.
- **Renames/cleanup** — "Depolarized (DDLS)" → **DPLS**; "Update (commit)" → **Update**; "Undo to committed" → **Undo**; "KWW (stretched)" → **Stretched Exponential (KWW)**; in-program citations removed (they live in the guides).
- **SLS tab polish** — clearer sample names in the header, shorter "no SLS data yet" message, wrapping calibration labels.

---

## 0.8.0 — UI polish batch: workspace, traces, plot units (2026-06-26)

Broad usability pass (no change to the analysis math).

- **Workspace management** — right-click a **header** to remove in bulk (sample header removes the sample, DLS/SLS headers remove only their measurements); **left-click a header** selects every measurement under it.
- **Selecting ≠ analysing** — picking a measurement no longer auto-adds it to analysis; the **"Measurements to plot"** checklist alone decides what is fit/overlaid.
- **Intensity traces are first-class** — traces appear under a **Traces** node (load/remove there), the Traces tab has the same multi-select checklist for overlays, and Brookhaven count-rate exports load via auto-detection.
- **Mw fraction applied like any per-measurement parameter** (set on the row(s), press **Update**); the apply-to-all pop-up is gone.
- **Distributions default to CONTIN** (was NNLS).
- **Reset the delay window + baseline** to defaults (full lag range; last 25 %) with one Correlogram-tab button.
- **Remove outlier points and recompute** on **Γ vs *q*²**, **D vs c**, and **DDLS** — click a point to exclude it; **Reset** restores all.
- **Human-centric plot units, configurable** — defaults: delay time **µs**, count rate **kcps**, *q*² **nm⁻²**, D **µm²/s**, concentration **mg/mL**; **Settings → Plot axis units** sets each axis default (analysis stays canonical).
- **Resizable plots** — grow with the window; intensity-trace plots taller by default.
- **Define '*k*'** (shot-noise band multiplier) on the trace tab, plus a new **running-average window** control (points; 0 = auto).
- **Synthetic generator** accepts temperature and viscosity in your own units (defaults **°C**, **mPa·s**); β/noise/points defaults live on the tab.
- **Settings de-cluttered** — synthetic-generator and trace defaults moved to their own tabs; **cP** dropped (identical to mPa·s).
- **Docs** — square-root symbols render correctly in the PDFs; vendor names generalised out except where a specific parser/format is described.

---

## 0.7.0 — Nonlinear (Frisken) cumulant, now the default (2026-06-25)

- **New default cumulant method: nonlinear (Frisken 2001).** Fits `g₂−1` directly with a **floating baseline**, rather than the classic linear fit to `ln(g₂−1)` (Koppel 1972, still available). More robust to baseline drift and noisy/low-count data, and uses the whole fit window. The table shows which method produced each result and its fitted baseline.
- **Choose in Settings → Cumulant method** (Nonlinear / Linear) — a **global** choice applied to every cumulant analysis (per-measurement, Γ vs *q*², replicate averaging).
- **Switching the method clears existing cumulant-based results** (asks to confirm, only if such results exist); distributions, SLS, and any hand-entered *R*<sub>h</sub> are always kept.
- **Robustness** — the nonlinear baseline is constrained to genuine drift and **falls back to the linear fit (flagged) if it fails to converge**. Like linear, it reports **no ± from a single correlogram**.
- **Known limitation** — a corrupted *first* channel (afterpulsing) is still best removed with *Skip initial lag channels* (v0.6.0); the nonlinear method handles drift and tail noise, not a short-lag spike.

---

## 0.6.0 — DLS cumulant/distribution: skip leading lag channels (2026-06-25)

- **New Settings option: "Skip initial lag channels".** Drops the first *N* correlator channels from **every** DLS fit (cumulant, single/double/KWW, NNLS/CONTIN/lognormal). The leading multi-τ channels carry afterpulsing and dead-time artefacts that can drag a fit to a falsely *small* size. Default **0 (no skip)**, so existing results are unchanged.
  - **Composes with the delay-window minimum** — the first fitted point is the later of "channel *N*" and "first lag ≥ the window minimum".
  - Validated against SMALS: on a noisy latex dataset a skip of 9 recovered detectors the un-skipped cumulant misread (e.g. 5.9 nm → 22 nm at one angle), matching the clean angles.
  - **Use sparingly with distribution methods (NNLS/CONTIN)** — short lags constrain the small/fast end, so an over-aggressive skip can erase a genuine small-particle population. Keep to the genuinely artefact-contaminated channels.

---

## 0.5.1 — UI polish: selection UX, plot sizing, scroll behaviour, Label field (2026-06-23)

- **Selection lists use click-to-highlight** (Shift / Ctrl) instead of checkboxes in the DLS overlay checklist and Cross-Sample include/exclude list; the DLS checklist groups measurements under bold **sample headers**.
- **Select all / Select none buttons** on both lists.
- **Label field** on the Data tab (top row) — a cosmetic per-measurement name that replaces the source filename in the sidebar tree and DLS list once committed; does not affect analysis or grouping.
- **All plot areas ~15% smaller by default** across DLS, SLS, Cross-Sample, Utilities; the Utilities **Traces** tab gains a draggable divider.
- **Scroll wheel no longer changes dropdown values or cycles tabs** (suppressed on every QComboBox and tab bar).
- **"Load data to begin"** replaces "Load a correlogram to begin".
- **"Solute name"** replaces "Sample name" in the Data tab table.
- **Right-click → Remove sample** on the sidebar tree (removes the entire sample with confirmation; disk files untouched).

---

## 0.5.0 — DLS Summary tab + averaged-results persistence (2026-06-22)

- **New DLS "Summary" sub-tab** — a workspace-wide, exportable results table that persists across save/reload, replacing the one-time averaged-results pop-up and the sequential peak list. Two stacked tables:
  - **Per-measurement results** — cumulant *R*<sub>h</sub> + PDI, single/KWW *R*<sub>h</sub>, and NNLS/CONTIN peaks as `Rh (Int %)`, where **Int %** is intensity-weighted peak area (NOT mass/weight percent).
  - **Sample-level *R*<sub>h</sub>** — replicate averages, Γ–*q*² (q→0, *apparent*) and D–c (c→0, *thermodynamic*), with an **Rh Type** column and a **From** column listing contributing measurements.
  - The left panel shares the Correlogram/Distribution measurement checklist and doubles as a **"Ticked only"** filter.
- **Averaged derived results now persist** — *Average derived results* records durably in the Summary table and session file, distinct from an averaged-correlogram measurement. Per-measurement fit results populate the table as run.
- **Distribution tab: peak labels + a residuals panel** (data − reconstructed g₂−1 vs delay time), like the Correlogram tab.
- **Controls moved next to the fit they drive** — **cumulant order** on the Correlogram tab; **Rh grid (min/max/points)** and **CONTIN L-curve α** range on the Distribution tab (still seeded from Settings, per-run value wins).
- **CSV exports written as UTF-8** (so labels like "Γ vs *q*²" export cleanly); pure-ASCII exports unchanged.

---

## 0.4.0 — Malvern multi-measurement parsers + repo tidy-up (2026-06-21)

- **Malvern Zetasizer clipboard** now accepts files with **many records** — each record column loads as its own DLS measurement (single-record copy still loads as one). Backward compatible.
- **New Malvern Zetasizer *export* parser** for the structured export (comma-separated, one measurement per row). Pre-fills parameters the file carries — refractive index, temperature, viscosity, sample/material/dispersant names — which you confirm/edit at the Data tab. Columns matched by header name (any column order works); template documented in Quickstart §2.2.
- **Fixed a crash** when loading a correlogram with no scattering angle yet (e.g. a fresh Zetasizer file).
- **Repository tidy-up** (no effect on use) — `code_map.md`/`code_references.md` moved to `docs/`, version file to `app/`, maintainer-only setup script no longer shipped.

---

## 0.3.0 — Documentation overhaul (2026-06-21)

- Split the user guide into a **Quickstart** (how to use) and an **Advanced Guide** (theory, numbered equations, bibliography), both shipped as PDFs in `docs/`.
- Added **`code_map.md`** (directory + per-file tour) and renamed the citation index to **`code_references.md`** (source → where used).
- Added this **`PATCH_NOTES.md`** and a project **version** (shown in the window title).
- Reworked **`README.md`** into a concise front door.
- Renamed the `Test Data` folder to `test-data`.
- Internal docs moved out of the tracked repo; the repo now ships only user-facing artifacts.

---

## 0.2 and earlier — Engine + GUI build-out (pre-versioning)

The analysis engine and application were built and validated module by module before formal version numbers existed. Capabilities as of this release:

- **Data model + parsers** — Brookhaven (DLS/SLS/trace), generic (DLS/SLS/trace), Malvern Zetasizer clipboard, ALV `.ASC` multi-angle; instrument-agnostic loading with auto-detection and a generic fallback.
- **DLS** — cumulants, single/double/KWW exponentials, NNLS, CONTIN, lognormal; Gamma-q² and concentration extrapolation; multi-measurement co-plotting; replicate averaging (correlogram average + mean ± SD/√N, ISO 22412).
- **SLS** — unified calibration from a single calibrant point, Debye, Zimm/Berry, Guinier, single-angle, calibration-free A₂, with data masking and a manual-*M*<sub>w</sub> override.
- **Cross-Sample** — ρ = *R*<sub>g</sub>/*R*<sub>h</sub> table and *R*<sub>g</sub>–*M*<sub>w</sub> / A₂–*M*<sub>w</sub> scaling plots with provenance-aware source pickers.
- **Utilities** — count-rate trace diagnostics, I·sin(θ) check, synthetic data generator.
- **Settings** — seed defaults for every module, light/dark theme, plot palette, input/display units.
- **DPLS/DDLS** — static depolarization geometry plumbing and a depolarized-DLS path for rotational diffusion (see Known issues).
- Origin-compatible CSV export; matplotlib plotting layer; self-contained JSON sessions.

---

## Known issues / outstanding

- **The DLS analysis region reaches a background path only when its box is ticked.** Γ vs
  q², D vs c, the Mode Map, DDLS and replicate averaging fit each run on its own defaults
  unless you tick **Use the Correlogram tab's delay window and baseline**, and the passive
  Cross-Sample re-derivation of a single-angle *R*<sub>h</sub> never reads it (it offers the
  Correlogram tab's own fit while that fit is current). So with the box unticked, a value you
  placed a region for on the Correlogram tab and the value pooled, averaged or promoted for the
  same run can still differ by that region — now visible beside the box rather than hidden.
- **Ethanol's displayed viscosity uncertainty below ≈ −8 °C may be optimistic.** Two published reference correlations diverge toward the cold edge of ethanol's supported range, reaching ≈5 % against the displayed 3 % band at −20 °C; the comparison correlation is extrapolating below its own fitted data range there, so which side errs is unsettled. Affects the displayed uncertainty (and the suggested rounding of a proposed viscosity) only — no fitted result.
- **DPLS/DDLS not yet tested on real depolarized data** — validated only against synthetic ground truth so far.
- **A DDLS D_t does not yet check that the VV rates lie on a line through the origin.** Its verdict reads the VV fits' decay checks, but not the linearity a Γ-vs-q² line is judged on. An optically anisotropic particle adds a rotational term to the VV decay, which gives Γ_VV vs q² an intercept; D_t then reads high, and Rh_t and D_r low, with the verdict still clean.
- **Visual peak picker** in the DLS distribution view is planned (peaks are already offered as *R*<sub>h</sub> sources elsewhere, but not click-selectable in the plot).
- **Session files carry a schema version but are not migrated** — a session written by a schema this build no longer reads is refused with a message rather than converted (a version-2 session still loads, its notes carried as plain text); re-analyse from the source files.
- A few library PDFs are **citation "promotion candidates"** not yet formally cited.
- **A general lack of polish** — rough edges in wording, layout, and visual finish remain throughout; the science is the priority, and the surface is still catching up.
