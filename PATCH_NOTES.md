# Patch Notes

User-facing summary of what each release changed, plus a running list of known
issues.

Versioning follows `MAJOR.MINOR.PATCH`: MINOR for new user-facing capability,
PATCH for fixes/polish. The version is set in `version.py` and shown in the GUI
window title.

---

## 1.0.3 — Smaller saved sessions (2026-07-13)

- **Polish:** saved session files (`.lsjson`) are now written compactly instead of
  pretty-printed. Because a session embeds the raw correlogram and trace data (thousands
  of numbers), the old one-number-per-line layout inflated the file roughly 1.5×; the new
  format is about that much smaller and loads a touch faster. Sessions saved by older
  versions still open unchanged, and the file is still standard JSON — to read one by eye,
  run `python -m json.tool your-session.lsjson`.

---

## 1.0.2 — Correct uncertainty-estimator label after a mid-session switch (2026-07-13)

- **Fix:** the Cross-Sample ρ = Rg/Rh table's uncertainty-estimator note (`[SE: classical
  OLS]`) now reflects the estimator that actually produced the stored Rg/Rh ±, not whichever
  estimator happens to be selected when the table is drawn. Previously, switching the
  regression-SE estimator (Settings → *Regression SE estimator*) after a value was computed
  could label an existing ± with the wrong estimator until the analysis was re-run. The point
  estimates and ± values themselves were always correct; only the estimator *label* could be
  stale. (The trustworthy fix is still to re-run the analysis under the chosen estimator, which
  refreshes the ± itself.)

---

## 1.0.1 — Backlog polish (2026-07-13)

- The count-rate histogram diagnostic (Utilities → Traces) now reports each fitted
  overlay's **reduced chi-square** in its legend (χ²ᵣ ≈ 1 is a good fit), alongside the
  Gaussian and Poisson curves it already draws — a quick goodness-of-fit read for the
  shot-noise check.
- Internal cleanup: removed several built-but-unused functions from the shipped source
  (no behaviour change).

---

## 1.0.0 — First stable release (2026-07-13)

The first stable release of DLS Buddy — a general-purpose, instrument-agnostic platform for
analyzing static and dynamic light scattering (SLS/DLS) from polymer solutions.

Everything is in place and validated:

- **Loading** from Brookhaven, Malvern Zetasizer, and ALV files, with a generic plain-text
  fallback — the load buttons carry no vendor name; the format is auto-detected.
- **DLS engine** — cumulants, single / double / stretched-exponential fits, NNLS and CONTIN
  size distributions, Γ–q² and concentration extrapolation, and true-replicate averaging.
- **SLS engine** — calibration, Zimm and Berry, Debye and Guinier, and the calibration-free
  2·A₂·Mw product. Apparent vs thermodynamic results are always distinguished.
- **Depolarized DLS** (VU/VV/VH), the ρ = Rg/Rh and power-law scaling cross-sample tools,
  the solvent-property library, and Origin-compatible CSV export.
- **The desktop application** — six tabs (Data, DLS, SLS, Cross-Sample, Utilities, Settings)
  with a sidebar workspace navigator, light/dark theming, and two-tier in-program help.

Uncertainty is reported where the fitted points are genuinely independent and labelled as
excluding calibration/dn-dc systematics; it is omitted where none is honest. Mw and absolute
A₂ are flagged when a run is uncalibrated, while Rg and the calibration-free product survive.

This release also incorporates the pre-1.0 hardening pass: numerical and uncertainty rigor,
fail-loud input validation, provenance carried through exports, GUI stale-state and plumbing
fixes, and the help / consistency polish from the 0.23.x line.

Start with the Quickstart, User Manual, and Theory-and-Equations guides under `docs/`.

---

## 0.23.5 — In-program help & label polish (2026-07-12)

*No change to any calculation or result value. Part of the pre-1.0 audit.*

- **More controls now explain themselves.** The DLS **Cumulant order** spin has a tooltip
  (what the order means — 1 = size only, 2 adds PDI, 3 adds skew — and why it greys out for
  the other methods), and the SLS method list's tooltip now describes the **Excess Rayleigh
  ratio** option it previously skipped.
- **The empty SLS plot now says what to do.** Before a fit (or for a sample with no SLS data)
  the Zimm/Debye plot area showed a blank white rectangle; it now shows a short centered note
  ("Load SLS intensities…" / "Press Run…"), matching the depolarized-DLS plot.
- **Consistent capitalization.** A few section titles now use Title Case like their neighbours
  ("Parametric Fit", "Traces to Plot"), and the User-Manual matches the in-app names for the
  Solvent Library, Display Units, and Paired Angles controls.
- **Theming fix.** The parameter table's fixed-unit text (e.g. the "°" and "nm" cells) now
  follows the light/dark theme instead of staying a fixed grey.

---

## 0.23.4 — On-screen flag for hot-calibration extrapolation (2026-07-12)

*No change to any calculation or result value. Part of the pre-1.0 audit.*

- **Calibrating above 50 °C (or below 10 °C) now shows an on-screen note.** The toluene
  Rayleigh-ratio temperature correction is validated only over 10–50 °C; outside that range the
  calibration constant k_c is extrapolated. That extrapolation was already computed (never
  blocked, so elevated-temperature reline work keeps running) but was flagged only to the
  terminal. The SLS Calibration panel now shows a neutral ⓘ note next to k_c whenever the standard
  temperature is outside the table, so the extrapolation is visible in the app. It clears when the
  temperature returns to the 10–50 °C range.

---

## 0.23.3 — GUI plumbing fixes (2026-07-12)

*Fixes only — no change to any calculation or result value. Part of the pre-1.0 audit.*

- **Removing a sample with a second solvent blank now removes it completely.** If a sample
  had two c = 0 SLS blanks (the second parked as a retrievable "extra" reference), "Remove
  sample" could previously leave the extra blank behind — it would reappear after the next
  regroup — or, if you selected only the extra blank, silently do nothing at all. Both are
  fixed; the extra blank is now a full member of the sample for grouping, removal, and
  shared-parameter edits.
- **A trace load cancelled partway through a multi-file batch no longer hides already-loaded
  traces.** Loading several trace files where a later one falls back to the plain-text format
  and you cancel its units prompt used to leave the earlier files loaded into the workspace
  but invisible in the sidebar until an unrelated refresh. The sidebar now updates
  immediately, and you still get the "some traces not loaded" notice for the ones that
  didn't load.
- **A rejected log-scale axis change now explains why.** Picking "log" for a plot axis whose
  limits include zero or a negative value now shows the reason ("log min must be > 0")
  instead of silently reverting with no message.
- Smaller robustness fixes: unchecking a sample's per-sample calibration mid-fit, the
  Depolarization panel's Mw read, a Settings dropdown holding a value outside its list, a
  partial axis-limit edit, a multi-sample manual-entry queue, and a synthetic-generator
  concentration edge case are all now guarded consistently with the rest of the app.

---

## 0.23.2 — Deliberate picks preserved + clearer uncalibrated exports (2026-07-12)

*Fixes only — no change to any calculation or result value. Part of the pre-1.0 audit.*

- **A value you explicitly picked in Cross-Sample is no longer overwritten by a re-run.**
  If you chose a specific Mw, Rg, A₂ or Rh for a sample from the source picker, re-running a
  Zimm/Berry fit (or a replicate average) now leaves your choice in place — exactly as a
  hand-entered value already was. Previously the re-run silently reset it to the auto-selected
  default. To adopt a fresh fit, pick it again.
- **Uncalibrated Zimm exports now flag every arbitrary-scale column.** When you export an
  uncalibrated Zimm/Berry dataset, the per-concentration Kc/dR columns and the extrapolated
  intercepts now carry the same "uncalibrated, arbitrary scale" note their Mw/A₂ already did,
  so a re-import into Origin can't be mistaken for absolute scale. (Rg stays valid — it doesn't
  depend on the calibration — and is left unmarked, as before.)

---

## 0.23.1 — Sturdier settings + stricter file loading (2026-07-12)

*Robustness only — no change to any calculation or result value. Part of the pre-1.0
fail-loud audit: close a set of gaps where a bad input file or a hand-edited settings file
could be silently misread instead of caught.*

- **Invalid saved settings are now repaired or reported, never silently wrong.** If your
  `settings.json` has been hand-edited (or left over from a different build) and a value has
  the wrong type, the app now fixes it when the intent is unambiguous (e.g. `"2"` → 2) and,
  when it can't, resets just that setting to its default and tells you at startup which ones
  and what they are now — instead of either crashing or quietly running with a value you
  didn't intend. A settings file that isn't a valid settings object no longer prevents the
  app from starting.
- **Foreign files are rejected more reliably on load.** A count-rate trace, a Zetasizer
  clipboard export with unreadable record columns, or a static-scattering intensity list
  containing non-numeric (`NaN`/`inf`) values is now refused at load with a clear message,
  rather than being partly accepted and silently corrupting the numbers downstream. Every
  supported instrument file still loads exactly as before.

---

## 0.23.0 — Analysis warnings now visible in the app (2026-07-10)

*No change to any calculation or result value — several signals the analysis already
produced were only ever written to a console the packaged app doesn't show, and are now
surfaced in the window as passive **ⓘ** notes.*

- **Unrecognized solvent name** (Data tab). Typing a solvent that isn't in the recognized
  vocabulary now shows a note below the parameter table: the name is used as-entered for
  sample matching. (Previously silent in the app.)
- **Dropped non-physical points** (SLS tab). When a Berry or calibration-free A₂ fit drops
  points with a non-positive value before its square root / ratio, the count and reason now
  appear as a note under the result — so a quietly shrunk fit is no longer invisible.
- **Negative intensities on load** (generic SLS files). Loading a plain-text SLS file that
  contains negative intensities (possible after background subtraction) now reports it in a
  short "loaded with notes" dialog; the values are still stored and used as-entered.
- Under the hood, the same messages continue to be emitted on the console for scripted/headless
  use — nothing was removed, only made visible.

---

## 0.22.0 — Documentation refresh + Theory-Guide renumbering (2026-07-10)

*Documentation only — no change to any calculation. This release also brings the public
build up to date with the **0.21.0** ill-conditioned-fit reliability flag and the **0.20.3**
synthetic-stress-corpus hardening (both detailed in their own sections below).*

- **Theory-and-Equations-Guide renumbered to be sequential.** Chapters now run **1–6** (were
  7, 9, 10, 11, 12, 15, with gaps) and equations **1–55** in reading order (were scrambled
  1–53 with 7a/7b/30a suffixes). Every cross-reference — inside the guide, in the User-Manual,
  and in the generated Citation-Index — was updated to match. A pre-existing mis-reference (the
  replicate-average standard error pointed at the wrong equation) was corrected in passing.
- **Quickstart + User-Manual prose pass.** Trimmed and clarified text, American spelling, an
  expanded Data-tab parameter table, and a decoupled PDF version marker ("Applies to vX.Y.Z and
  later") so a patch bump no longer forces a doc rebuild.
- **Numbering integrity is now checked automatically** when the documentation PDFs are
  built, so the guide's chapters, equations, and index can't silently drift out of order again.

---

## 0.21.0 — Ill-conditioned-fit reliability flag (2026-07-10)

A new numerical-health flag for static light scattering, plus two latent-crash fixes it
uncovered. No change to any correct calculation — a well-posed fit behaves exactly as
before; this only adds a warning (and closes two edge-case crashes) for degenerate fits.

- **A numerically degenerate SLS fit is now flagged unreliable.** When the extrapolation
  design is near-collinear — clustered angles, or two nearly-equal concentrations, so the
  fit cannot honestly resolve *M*<sub>w</sub> — the Zimm/Berry, Debye and Guinier
  *M*<sub>w</sub> (and absolute *A*₂), and the calibration-free 2·*A*₂·*M*<sub>w</sub>
  product, are marked unreliable, with a red note in the SLS tab and in the export Comments
  cell advising to add concentrations/angles or widen their spread. The value is still
  shown; *R*<sub>g</sub> (from the slope) is unaffected. The
  test is the design's condition number — a scale-invariant numerical criterion, **not** an
  "*M*<sub>w</sub> too large" cutoff, so a legitimate ultra-high-*M*<sub>w</sub> sample is
  never flagged for its size alone.
- **A single-angle *M*<sub>w</sub> from a vanishing scattering signal is flagged.** When
  *Kc*/Δ*R* is so small that its reciprocal loses all precision, the apparent
  *M*<sub>w</sub> is marked unreliable instead of reported as a confident number.
- **Two edge-case crashes fixed.** A Guinier plot with a very large intercept (an
  overflowing dR(0)) and a near-singular linear fit (a tiny negative variance) previously
  raised an error; both now degrade to a flagged/uncertainty-free result.
- **A degenerate fit never reports a ± of exactly zero.** Where error propagation cancels
  to a non-positive variance, the uncertainty is now omitted rather than shown as 0.

## 0.20.3 — Robustness hardening (2026-07-10)

Guards and reliability flags for physically impossible input and fits, driven by an
adversarial "stress" test corpus. No change to any correct calculation; these only
change how the program responds to bad or extreme data.

- **Impossible files are now refused at load.** A correlogram containing `nan`/`inf`,
  a delay (lag) axis that is not strictly increasing, or an intensity trace with a
  negative count rate is rejected with a clear message instead of being silently
  analyzed.
- **A physically implausible temperature or refractive index is flagged.** A likely
  unit slip — for example 2980 K for 298 K, or *n* = 13.3 for 1.33 — now shows a calm
  note under the Data-tab parameter table (and raises a warning), but the value is
  still used exactly as entered (these are yours to set and override).
- **Impossible cumulant fits are no longer marked reliable.** A fit that returns a
  negative polydispersity index (from an over-subtracted baseline) or a negative decay
  rate (from a correlogram that rises instead of decays) is now flagged unreliable,
  where before it could slip through as a plausible-looking size.
- **A non-physical Zimm/Berry molecular weight is caught.** When the extrapolated
  intercept is non-positive — which would imply a negative or infinite *M*<sub>w</sub>
  — the result is reported as not-a-number and flagged unreliable, matching the
  existing Debye behaviour.
- **Extreme intensities no longer crash the SLS math.** Values near the floating-point
  ceiling degrade to a finite result instead of raising an error.
- **A constant (all-zero) intensity trace** now reports a coefficient of variation of
  0 rather than a bare not-a-number.

## 0.20.2 — Documentation overhaul (2026-07-09)

A top-to-bottom review of the guides, with the science and citations checked first. No
change to the analysis; this is documentation and in-source comments only.

- **The Quickstart is now two documents.** *Quickstart-Guide* is short again — the load →
  confirm parameters → run → export path through the most-used modules, with annotated
  screenshots. Everything it used to carry moved into a new comprehensive **User-Manual**
  (every module, file format, and parameter), which also gains a **solvent table** of the
  Solvent Explorer's built-in solvents and the temperature/wavelength ranges over which each
  one predicts refractive index and viscosity.
- **The Advanced Guide is now the *Theory-and-Equations-Guide*** — a name that says what it
  holds (the physics, numbered equations, and the literature behind each method).
- **The `docs/` folder is numbered** so it reads in order: 1. Quickstart-Guide, 2. User-Manual,
  3. Theory-and-Equations-Guide, 4. Code-Map, 5. Code-References, plus two new files —
  **6. Acknowledgements** (beta testers and test-data contributors) and **7. AI-Use-Statement**
  (how the project was built, in the author's words).
- **Citations audited.** Every scientific claim was cross-checked against its primary source,
  and each citation confirmed to exist in the reference library and to support the statement it
  is attached to.
- **In-source comments cleaned up** for anyone reading or forking the code: internal shorthand
  that pointed at private notes was rewritten to stand on its own, while every literature
  citation was kept.

## 0.20.1 — Cross-Sample crash fix (2026-07-09)

- **Fixed a crash when opening the Cross-Sample tab for a sample whose DLS measurements
  have no concentration.** Concentration is optional for DLS (only the D-vs-c extrapolation
  needs it), but leaving it blank made the ρ table's *R*<sub>h</sub> picker fail and took down
  the whole tab. The picker now handles a missing concentration: a multi-angle set still yields
  its Γ-vs-q² (*q*→0) *R*<sub>h</sub>, labelled *"at c = ?"* since the concentration is unknown,
  while the D-vs-c (*c*→0) extrapolation is offered only for the measurements that do have a
  concentration. No numbers change when a concentration is present.

## 0.20.0 — Uncertainty-driven result precision (2026-07-09)

Results are now shown to the precision their uncertainty actually supports — no more digits
than the ± justifies. The numbers themselves are unchanged; only how many are displayed.

- **A value with a ± is rounded to the place its ± supports.** If an Rh, Rg, or Mw carries a
  standard error, the displayed value is rounded to the last digit that error can stand
  behind (e.g. a Rg with a ±2 nm SE shows as `42 nm`, not `41.7 nm` — reporting the tenths
  would imply confidence the data doesn't support). This now applies across the Cross-Sample
  ρ table (Rg/Rh), the DLS Summary sample-level Rh table, and the SLS results.
- **The ± is shown at the same place**, so value and uncertainty always agree (`42 ± 2`, never
  `41.7 ± 1.9`).
- **Results with no honest uncertainty keep an honest, modest precision.** Any result without a
  defensible ± — a size from a single correlogram, an NNLS/CONTIN distribution peak, a
  single-angle Mw, and every derived number that carries no ± (Γ, PDI, R², qRg, distribution
  weights, …) — is shown at a fixed number of significant figures instead (none is invented).
  **3 by default**, adjustable under **Settings → Result Formatting → No-uncertainty precision**;
  this one setting controls all of them uniformly. It *only* affects results with no ± — it never
  changes the precision of a value that has a ± (that stays driven by the uncertainty). Note that
  lowering it also coarsens R²/qRg.
- **Consistent scientific notation.** Large/small numbers now render the same way everywhere
  (e.g. `1.23e6`), whether or not the value carries a ±.

## 0.19.0 — Unified "current sample" selection (2026-07-09)

One sample is "active" at a time, and **every tab follows it** — no more tabs disagreeing
about which sample you're looking at. No analysis or numbers change.

- **The Workspace tree is now a real navigator for every tab.** Click a sample (or a
  measurement, or a DLS/SLS heading) and Data, DLS, SLS, Cross-Sample, and I·sin θ all
  switch to it. Previously the SLS / I·sin θ tabs kept their own last sample and
  Cross-Sample ignored the tree entirely.
- **The in-tab sample dropdowns and the sidebar are two ways to drive one choice.** Picking
  a sample in a tab's dropdown moves the sidebar and the other tabs too; clicking in the
  sidebar updates the dropdowns. One source of truth.
- **A sample a tab can't show reads cleanly.** Focusing, say, a DLS-only sample while on the
  SLS tab now shows "No SLS data for <sample>" (with the dropdown noting the same), instead
  of silently staying on a different sample. Nothing is lost — your calibration and last
  result are kept, so returning to the sample restores them.
- Per-measurement ticking (the DLS overlay/fit checklists) is unchanged — that's a separate
  choice from "which sample is active".

## 0.18.0 — Accessibility floor (2026-07-08)

Meets the accessibility floor: colour is never the only signal, text/UI contrast meets
WCAG AA in both themes, and there are new appearance/interface settings. No analysis or
numbers change.

- **Colorblind-safe by default, and colour is never the sole cue.** Overlaid plot series
  (co-plotted correlograms, Zimm concentrations) now differ by **marker shape / line
  style** as well as colour, so they stay distinct for colorblind readers and in grayscale.
  The default plot palette (Okabe–Ito) is colorblind-safe; the palette control notes this.
- **Contrast pass.** A few theme colors were nudged so every text/label/marker color clears
  WCAG AA against its background in both light and dark themes.
- **Match plot to app theme (opt-in).** A new Settings option themes the on-screen plots to
  a dark background when the app theme is dark. **Saved/exported images always stay white**
  for clean, print-parity figures — this only changes what you see on screen.
- **UI density.** A new Compact / Comfortable / Large setting scales the application text
  size app-wide — larger for readability, compact for screen real-estate.
- **Reopen last session on startup (off by default).** When enabled, the workspace is
  auto-saved when you close the program and restored on the next launch; if the saved
  session is missing or unreadable, the program starts empty.

## 0.17.10 — Empty-state onboarding + layout polish (2026-07-08)

Onboards a newcomer on the first screen and fixes the layout/polish defects that made the
app look unfinished. No analysis or numbers change.

- **Empty workspace onboards.** With nothing loaded, the Data / DLS / SLS / Cross-Sample
  tabs now show a single centred call-to-action — "← Load a DLS correlogram or SLS
  intensities to begin" — instead of full chrome over blank tables and plots. The DLS
  correlogram no longer draws empty satellite/residual boxes before a curve exists, and the
  Data provenance legend is hidden until a measurement is loaded.
- **Distribution picker fixed.** The "Select all/none" and "Tick all at concentration/angle"
  rows no longer overlap the measurement checklist — the list keeps a sensible minimum
  height so the bulk-select rows always sit below it.
- **Calmer "changed field" highlight.** An edited-but-unsaved parameter cell is now a calm,
  theme-aware tint (readable in light and dark) with the value shown in italic — replacing
  the hard full-yellow fill that was hard to read in dark mode.
- **Correlogram title no longer clipped.** The "Correlogram — lin-log" title now clears the
  y-axis gutter; the "double-click a side view to promote" hint moved below the plot. The
  linear-scale side views are pinned to a sensible range so a very long delay window no
  longer crushes the decay against the origin.
- **Uncalibrated Mw reads honestly.** When there is no calibration, the absolute Mw is shown
  as "— (uncalibrated)" at the value (not a meaningless large number). Rg and the
  calibration-free 2·A₂·Mw product are unaffected — they remain valid without calibration.

## 0.17.9 — In-program help: consistent tiers, fuller coverage (2026-07-08)

A help-only pass; no analysis or numbers change. Brings the `?` badges and hover tooltips
into one consistent system.

- **No literature citations in the UI.** The CONTIN α-selection help no longer prints
  author-year citations — the plain-language description stays and points to the Advanced
  Guide, where the references live.
- **New `?` help buttons** on the **Traces** and **Synthetic Generator** tabs (previously
  the only tabs without one).
- **Clearer SLS help.** The Data Mask and Depolarization boxes now put their how-to on a
  `?` button (with a short hover note), instead of a long always-on paragraph.
- **More tooltips where they were missing:** the VU/VV/VH standard-geometry selector (both
  the SLS calibration panel and Settings), the calibration fields, the Guinier qRg limit,
  the SLS method selector, and the Traces diagnostic selector.
- **Trimmed the longest tooltips** to a sentence or two that point to the Advanced Guide.

## 0.17.8 — Terminology & casing normalized (2026-07-08)

A text-only consistency pass so the app reads as one voice — one verb per action, one
spelling, one casing, one unit rendering. No logic, analysis, or exported numbers change;
only user-facing labels.

- **One word for "commit edits": Update.** The SLS "Apply parameters" button and the
  Settings "Apply" button are now **Update** (matching the Data tab); the SLS pending note
  reads "press Update".
- **One word for "run an analysis": Run.** "Run fit", "Run DDLS", "Run analysis", and
  "Compute depolarization" are all just **Run** now. ("Generate" for synthetic data is a
  different action and keeps its name.)
- **American spelling throughout** — analyse→analyze, normalised→normalized, grey→gray,
  labelled→labeled, and the polarization/analyzer geometry tooltip.
- **Consistent casing** — Title Case for tab and section titles ("Synthetic Generator",
  "Physical Parameters", "Data Mask (Hide / Show)"); Sentence case for buttons and form
  labels, each ending in a colon.
- **One degree rendering** — "(deg)" in labels is now "(°)"; the Data table's angle unit
  shows "°".
- "Show all (clear mask)" on the SLS tab is now **Clear mask**.

## 0.17.7 — Consistent warning tiers across DLS + Cross-Sample (2026-07-08)

A visual-only pass that makes the two-tier warning system (settled in 0.11.1 "Calmer
warnings") consistent everywhere. No analysis, thresholds, or wording changed — only which
tier a flag is painted in, so the red alarm channel is spent only on genuine problems.

- **Neutral DLS notes are now calm, not alarm-red.** On the Γ vs q² / D vs c and DDLS
  views, an expected qualifier like "± statistical only" now renders as a steel-blue **ⓘ**
  note instead of a bold red **⚠** — matching how the SLS tab already treats "apparent" /
  "± statistical". A genuine problem (a non-diffusive Γ-q², or an unphysical rotational
  diffusion D_r ≤ 0) still shows the red **⚠**.
- **The Cross-Sample "uncalibrated" badges are now red, not amber.** The Mw / A₂ badges and
  the panel banner flagged an uncalibrated (arbitrary-scale) value in amber; that condition
  is a genuine data-quality problem and now uses the same red **⚠** as the SLS tab and the
  amber tier goes back to meaning only "pending update".
- **A skipped run reads the same everywhere.** In the size-distribution (NNLS/CONTIN) view a
  measurement that failed to fit is now shown on the red flag line, the same channel the
  cumulant/exponential view already uses, instead of being tucked into the muted status line.

Under the hood: all these flags now route through one shared renderer that picks the colour
tier *and* the matching ⚠/ⓘ glyph in a single place, so a flag's glyph and colour can never
disagree.

## 0.17.6 — Internal cleanup + minor performance (2026-07-07)

The last of the internal code-review batches: naming/doc/dead-code tidy-up and a few
redundant-work speedups. **Almost nothing is user-visible** — no scientific number changes.
The only observable bits:

- **Session files are now strict, portable JSON.** A saved session no longer writes the
  non-standard `NaN`/`Infinity` tokens (which some external JSON tools reject); non-finite
  values are written as `null` and reload correctly. Sessions saved by DLS Buddy itself
  behave exactly as before.
- **Boolean columns export as `yes`/blank** (consistent with the rest of the export), not
  `1`/`0`.
- **Scattering angle is validated.** Passing an out-of-range angle (≤ 0 or ≥ 180°) to the
  scattering-vector calculation now raises a clear error instead of returning a nonsense
  value, matching the other physics functions.

Under the hood: de-duplicated the excluded/masked-point plot overlays and the SLS
draw-vs-click transform (so they can't drift apart), routed the Data-tab solvent library
readouts through the controller, and vectorised two count-rate trace hot loops
(running-average, Poisson histogram overlay) — all output-preserving.

## 0.17.5 — Edge-case hardening: robust parsers, honest failed fits, portable sessions (2026-07-07)

Robustness fixes from the internal code-review pass (fourth of several batches). No new
features; each closes a way an unusual-but-legitimate input could be mis-read or a failed
result mistaken for a good one. (Follows 0.17.4; versioned to stack on that batch.)

- **Zetasizer files are parsed more robustly.** A comma inside a quoted sample/material name
  in a Zetasizer *export* no longer shifts that row's columns (wrong correlogram / RI /
  temperature); a misaligned row is now rejected with a clear message instead of silently
  misread. A Zetasizer *clipboard* file that is mostly non-numeric — or whose header only
  loosely resembles one — is now rejected rather than loaded as blank (NaN) data.
- **A failed fit no longer looks like a good one.** When a DLS fit (single/double/stretched
  exponential, or the nonlinear cumulant) fails to converge, its size/rate outputs now read
  **n/a** instead of quietly showing the starting estimate. The nonlinear cumulant no longer
  substitutes the simpler linear fit under the same row — a failed nonlinear fit is labelled
  **(failed)**. Check the success/label before trusting a fit.
- **No impossible negative molecular weight.** A noisy single-concentration Debye fit whose
  extrapolated intercept goes non-positive now reports Mw as **n/a** (a negative Mw is
  physically impossible) instead of a negative number. Rg is unaffected.
- **Cumulant fit-quality (RMS) is comparable across methods.** The linear and nonlinear
  cumulant now report their RMS residual over the same set of points, so comparing their
  fit quality is meaningful.
- **Older builds can open newer session files.** A session saved by a newer version (with an
  extra field) now loads in an older version instead of failing — the unknown field is
  ignored. The "portable session" promise holds across versions.
- **Assorted robustness guards** (none reached in normal use): a size distribution's flat-topped
  peak counts as one population, not two; the trace running-average window is exactly the
  requested width; degenerate edge inputs (a zero-length trace, a zero/negative viscosity in an
  ALV header, a temperature round-off between the DLS and SLS paths, a relative trace view with
  no baseline) are handled or rejected cleanly rather than producing a wrong or mislabelled
  result.

## 0.17.4 — Performance: snappier Cross-Sample refresh, responsive Traces tab (2026-07-07)

Speed and responsiveness from the internal code-review pass (third of several batches). No
new features and **no change to any result** — these are pure performance fixes.

- **Faster Cross-Sample refresh.** Refreshing the Cross-Sample tab used to re-run every DLS
  Rh fit (cumulant per measurement, Γ-vs-q² per concentration, D-vs-c per angle) several times
  over — once for the automatic pick and again for each source dropdown. The results are now
  computed once and reused until an input actually changes, so the tab updates noticeably
  quicker on samples with many angles/concentrations. (The SLS side already worked this way.)
- **Responsive Traces tab, no repeated work.** Selecting a trace used to redraw the view two
  or three times per click and compute the histogram / block-variance diagnostics twice. Each
  is now computed once and shared, and for a **long trace** the diagnostics run in the
  background so the window no longer briefly freezes.
- **No slow memory growth from plotting.** Figures created by the plotting layer no longer
  accumulate in matplotlib's global registry during a long session.

## 0.17.3 — Reliability fixes: no silently-dropped blanks, fractioned series plottable (2026-07-07)

Bug fixes from the internal code-review pass (second of several batches). No new features.

- **Fixed: a second solvent blank (c = 0) could silently disappear.** If a sample had two
  solvent-blank series loaded (for example a re-measured blank), the second one used to
  overwrite the first in the single solvent-reference slot, and the first became invisible —
  present in the session but unreachable from the SLS tab or analysis, with no warning. Now the
  **first-loaded blank stays the active reference**, any extra blank is **kept and shown** in the
  Workspace tree marked *"extra blank (unused)"*, and loading SLS data with more than one blank
  raises a short notice telling you which blank is in use. Nothing is dropped; remove or re-assign
  a blank if a different one should be the reference.
- **Fixed: a molecular-weight series stored entirely in named fractions was missing from the
  scaling plots.** If a sample's Mw/Rg lived only in named Mw fractions (e.g. "250k", "1M") with
  no unfractioned result, it was judged ineligible and never offered or plotted in the
  Cross-Sample Rg–Mw / A₂–Mw scaling views — even though its per-fraction points are perfectly
  usable. Eligibility and plotting now use the same per-fraction test, so such a series appears
  and is fitted.
- **Clearer error when dn/dc hasn't been entered yet.** Running an SLS build before entering
  dn/dc now gives the intended *"dn/dc must be a finite number"* message instead of a confusing
  internal `TypeError`. (dn/dc is still never guessed or defaulted — you always enter it.)
- **Internal:** the regression-uncertainty routines now return "no ±" (rather than crashing) on a
  rank-deficient fit — e.g. a Zimm/Berry extrapolation attempted from a single angle *and* a
  single concentration. Such a fit is unidentifiable, so no honest uncertainty exists; the point
  estimate is still returned and the ± reads as unavailable.

## 0.17.2 — Correctness fixes: A₂ provenance + calibration-free A₂ guard (2026-07-07)

Bug fixes from an internal code-review pass (first of several batches). No new features.

- **Fixed: A₂ uncertainty / calibration label could be left stale after a Zimm/Berry run.**
  When you ran a Zimm or Berry fit in the SLS tab, the sample's A₂ **value** was updated but its
  **± uncertainty, calibration flag and description** could keep the values from a previously chosen
  A₂ — so the Cross-Sample table could show an A₂ paired with the wrong ± and the wrong "uncalibrated"
  label. The run now refreshes all of them together. (A hand-entered A₂ is still never overwritten.)
- **Fixed: the calibration-free 2·A₂·Mw could be silently corrupted by a near-zero data point.**
  In low-contrast systems, a concentration whose solvent-subtracted scattering falls to (or just below)
  zero — from an over-subtracted solvent reference or the noise floor — used to feed a meaningless value
  into the calculation with no warning. Such points are now dropped with a warning (as the Berry method
  already does), and if the *reference* concentration itself is non-positive, or too few points remain,
  the analysis stops with a clear message instead of returning a wrong number.
- **Internal:** ticking a sample in the Cross-Sample list while a background fit is running now waits
  for the fit to finish before updating, closing a rare race that could disturb results mid-analysis.

## 0.17.1 — Cross-Sample source panel: explicit selection + A₂ picker (2026-07-07)

A redesign of how you pick which fitted values represent each sample in the Cross-Sample
tab, from the same owner feedback round. The analysis numbers are unchanged — this is about
making the selection explicit and consistent with the rest of the program.

- **The focused sample is now chosen explicitly, in the tab.** The source panel has its own
  **Sample** and **Fraction** selectors, so it always says which sample and which Mw fraction
  you are editing. Previously the sample was re-pointed by a hidden side effect of clicking a
  ρ-table row, and the fraction was only shown in the panel title. The ρ table is now a
  read-only results view, and the Workspace tree on the left simply *highlights* the focused
  sample (it no longer needs to be used to drive this tab).
- **New A₂ source picker.** The A₂–Mw scaling plot now has a visible A₂ row, so you can see
  and choose which fit supplies A₂ (Zimm or Berry) instead of it being picked silently. A₂ has
  **no hand-entry box** (unlike Mw): it is a solvent/temperature-specific interaction
  coefficient with no external "standard", and it is the very quantity the A₂–Mw plot fits, so
  only fit-derived values are offered.
- **Uncalibrated Mw / A₂ are now flagged in the panel.** Because Mw and A₂ are on an arbitrary
  scale until the run is calibrated, each row shows a **⚠ uncalibrated** badge and the panel
  shows a one-line banner when the chosen value is uncalibrated. Rg is scale-independent, so it
  is never flagged.
- **The source dropdowns are no longer overwhelming.** They now group candidates by result
  type (extrapolated / distribution / averaged / single-condition) and hide the long tail of
  per-angle / per-concentration apparent values behind a **"Show all single-condition results"**
  checkbox. The best value is still selected by default either way.
- **A source you pick now sticks.** Choosing a non-default Rg / Rh / Mw / A₂ in the source panel
  is remembered — switching tabs or committing an edit no longer silently reverts it to the
  auto-default (it previously did).

## 0.17.0 — UI/UX pass: consistent explicit-Run, clearer selection, tidier plots (2026-07-07)

A batch of usability fixes and one consistency change, from an owner feedback round. The
analysis results are unchanged throughout — this release is about *how* you drive the
program and read its plots, not the numbers.

- **SLS now works like the DLS tabs: nothing recomputes until you press Run.** Previously
  the SLS tab silently re-ran its fit every time you switched sample or committed a
  parameter. Now switching sample/fraction or committing **redisplays the last fit** and,
  if an input that fed it changed, shows *"Inputs changed since this fit — press Run to
  refresh"* — instead of quietly recomputing. Hiding/showing a point (mask) still refits
  immediately, as before. This makes every analysis tab behave the same way.
- **"Tick all at concentration / angle" now shows units, and the Distribution tab gets
  both.** The bulk-selection dropdowns on Γ vs q² / D vs c read **mg/mL** and **°** (they
  showed a bare number before), and the Distribution tab — which only had Select all/none —
  now has the same **tick all at concentration** and **tick all at angle** helpers over its
  measurement list.
- **Excluded points are now labelled on the plot.** The grey ×'s for measurements you've
  unticked (Γ vs q², D vs c, DDLS) and the hollow grey markers for masked SLS points now
  appear in the plot legend ("excluded" / "masked"), so it's clear what the fit left out.
- **Data tab keeps confident trailing zeros.** A library-filled refractive index or
  viscosity is now shown to the precision its uncertainty supports — e.g. `1.3300` instead
  of `1.33` — so the displayed value matches the precision the program actually stands
  behind. Hand-entered values are shown as you typed them.
- **Distribution-tab layout fixes:** the left-hand control labels no longer clip at the
  default window width; the "Peak results" table now grows to fill its space instead of
  staying pinned small; and the draggable divider between the distribution and its residual
  no longer overlaps the axis label.
- **Faster Cross-Sample refresh** (internal): the tab no longer re-runs the same SLS fits
  several times per refresh — they are cached and reused until an input changes. No visible
  change, just less waiting on samples with many concentrations.

---

## 0.16.0 — Γ vs q² / D vs c: explicit-Run workflow + easier point selection (2026-07-06)

The multi-angle **Γ vs q²** and concentration-series **D vs c** tabs were reworked so
they are predictable about *when* they compute, and easier to load with data. Based on
owner feedback that it was unclear how measurements became eligible and that "something
got plotted before any analysis was run."

- **Nothing is fitted or plotted until you press Run.** On selecting a sample the
  points table now lists its measurements as *metadata only* — Γ and *D*<sub>app</sub>
  read "—" until you Run. This removes the earlier behaviour where focusing a sample
  silently computed a cumulant fit per measurement.
- **The table refreshes when you commit.** After confirming parameters in the Data tab,
  the Γ-q²/D-c tables repopulate immediately — no more switching away and re-clicking
  the sample in the sidebar to make measurements appear.
- **Ticking only selects; it never refits or clears the plot.** Changing which points
  are included marks a shown result *"selection changed — press Run to refresh"* instead
  of silently refitting or wiping the plot.
- **New selection helpers:** **Select all**, **Select none**, and **Tick all at** a
  chosen value — concentration on Γ vs q², angle on D vs c.
- **Clearer eligibility.** Rows that can't be ticked are greyed with a hover tooltip
  saying why (parameters not confirmed, or a different Mw fraction — which is now shown
  greyed rather than hidden). A note flags same-polymer/solvent measurements that are
  grouped separately because their temperature isn't confirmed yet. After a Run, a point
  whose cumulant fit failed or whose PDI exceeds the validity limit is flagged in its
  tooltip so you can untick it.
- **"Inputs changed — press Run to refresh" hint** across every DLS analysis tab
  (Correlogram, Distribution, Γ vs q², D vs c, DDLS). If you commit a change to a
  parameter that fed a displayed fit, the plot stays but a status line tells you it is
  now out of date — so a fit is never silently shown against parameters it wasn't
  computed with. The hint is precise: it appears only for the results whose *own* inputs
  changed, not for an unrelated edit elsewhere. (SLS already re-runs its fit on commit,
  so it is always current.)

Analysis results are unchanged — the fit, its Γ source (internal 2nd-order cumulant),
and the reported uncertainties are identical; only *when* and *how* points are selected
and displayed changed.

**Distribution note:** starting with this release the automated developer test suite is
no longer included in the download — it is development tooling; nothing in it is needed
to run the program.

---

## 0.15.1 — Water refractive-index band: honest widening at warm temperatures (2026-07-06)

A focused correctness review found that water's refractive-index confidence band
**under-stated the true model error above ~38 °C**: water's d*n*/d*T* steepens with
temperature (per the IAPWS standard formulation, Harvey *et al.* 1998), and the
library's constant slope — fitted near 20 °C — accumulates ~7×10⁻⁴ of error by 45 °C,
where the band claimed ~3.5×10⁻⁴. (The pre-0.15 flat figure, ±6×10⁻⁴, under-stated it
too.)

- Water's d*n*/d*T* slope allowance is widened so the band bounds the measured drift
  across the whole 15–45 °C range: the box-wide *n* uncertainty becomes **9×10⁻⁴**
  (was 6×10⁻⁴ pre-0.15). Near room temperature nothing changes — the band still
  narrows to ~3×10⁻⁴ there, and auto-filled values are identical.
- No other solvent is affected (water was the only record whose served range outruns
  its source's measured temperatures).

---

## 0.15.0 — Solvent library: per-condition confidence band (2026-07-06)

The library's display uncertainty is now evaluated **at your wavelength and
temperature**, not as one flat figure for the whole validity range.

- **The Explorer band now grows and narrows honestly.** It is narrowest near the
  conditions each source measures best (e.g. water *n* around 20 °C) and widens toward
  the edges of the validity range. Every part of its shape derives from real
  quantities — the source's stated accuracy, the library's own fit-residual envelopes,
  and standard d*n*/d*T*-slope propagation (Advanced Guide §12, Eqs. 52–53). Where a
  bulk/handbook source states no temperature shape, the band stays **flat** — nothing
  is ever invented.
- **Numeric readouts and auto-fill follow.** The Explorer's ± and the Data-tab
  auto-fill rounding now use the per-condition value, so mid-range look-ups often gain
  a digit (water at 532 nm, 25 °C now reads *n* = 1.3349 ± 0.0003 instead of ± 0.0006).
  The box-wide figures in tooltips remain the band's maximum — the per-condition value
  never exceeds them.
- **Honesty corrections both ways.** Most box-wide *n* uncertainties tightened
  (removing box-edge pessimism); a few viscosity figures widened slightly where the
  fit deviation now counts on top of the source's stated uncertainty instead of beside
  it (water 1.0 → 1.1 %, *n*-hexane/methanol 2.0 → 2.1 %, acetone 5.5 → 5.8 %,
  THF 6.0 → 6.1 %, ethylene glycol 4.9 → 5.0 %). Three viscosity validity ranges were
  **clipped to the range their source states its uncertainty for**: toluene 260–370 K,
  benzene 288–340 K, methanol 273–343 K (values outside those ranges were previously
  served with an uncertainty the source does not support).
- Nothing changes in how analyses treat the uncertainty: it is still **displayed,
  never folded into a reported ±**. The underlying n/η fits are identical; an
  auto-filled value may simply keep **one more digit** where the per-condition
  uncertainty is genuinely tighter than the old box-wide figure (re-running
  auto-fill on an old session updates the fields once, as in 0.14.1).

---

## 0.14.1 — Solvent Explorer relocation + layout; auto-fill rounding (2026-07-06)

Five owner-requested refinements to the 0.14.0 solvent-library surface.

- **The Solvent Explorer moved under Utilities.** It is now the fourth **Utilities
  sub-tab** (after Traces, I·sin θ, Synthetic generator), restoring the six-tab shell.
  Everything about it works as before — it still needs no loaded sample and never
  writes into a measurement.
- **One shared figure, two plots side by side.** The Explorer's two stacked plots are
  now two smaller panels on a single canvas (one toolbar), resizing together; the
  freed vertical space goes to the condition/readout area above.
- **Temperature plot axes swapped.** Viscosity — the quantity that actually varies
  strongly with temperature — is now the **left** axis (log-capable, log by default);
  refractive index rides the right axis. Colours, bands, and markers are unchanged.
- **Auto-filled values are rounded at the last confident digit.** The Data-tab
  auto-fill used to write full-precision floats (e.g. *n* = 1.3325541…); it now rounds
  each proposed value at the decimal place its library uncertainty can stand behind
  (water at 25 °C, 633 nm → *n* = 1.3315; η = 0.891 mPa·s). The uncertainty only picks
  the rounding place — it still never enters any analysis. Values you type yourself
  are, as always, never touched. (If you re-run auto-fill on a pre-0.14.1 session,
  the fields will update once to the rounded form.)

---

## 0.14.0 — Solvent property library: auto-fill + Solvent Explorer tab (2026-07-06)

The solvent-property library (added headless in 0.13.x) now has its front-ends.

- **Data tab — auto-fill.** The solvent name is now a dropdown of the library’s primary
  solvents (water, toluene, ethanol, methanol, benzene, cyclohexane, *n*-hexane, acetone,
  glycerol, ethylene glycol; free text still works for anything else). Pick one and the
  **refractive index** and — for DLS — the **viscosity** fill in from the temperature and
  wavelength, marked with a **teal dot**. They **re-derive** whenever you change the
  temperature or wavelength; a value you **type by hand is never overwritten** (and shows
  no dot). Outside a solvent’s validity range the program does not guess — it clears the
  library value and tells you to enter it manually. **dn/dc is never proposed** — it stays
  hand-entered.
- **New Solvent Explorer tab.** A global look-up + visual calculator over the whole library
  (both confidence tiers; estimate-tier solvents are labelled). Enter a solvent, temperature,
  and wavelength to read *n* and η with their uncertainty and a tier badge, and see two
  plots: *n* and η vs temperature (twin axes, each drawn only across the range the library
  covers) and *n* vs wavelength, each with a **confidence band** and a marker at your chosen
  condition. The band is a conservative bound shown for judgement — it is **never** folded
  into any reported ± (Advanced Guide §12).
- **Settings → Solvent library → Default solvent** sets which solvent the Explorer opens on.

The library only ever *offers* a value; nothing enters an analysis without your commit, and
no solvent constant is baked into the maths. Sources and the per-solvent validity/uncertainty
detail are in the Advanced Guide (§12) — never shown as citations in the program.

---

## 0.13.1 — CONTIN F-test: exact Provencher degrees-of-freedom (2026-07-03)

Correctness fix to the new CONTIN F-test α-selection (0.13.0), validated against Provencher's
original 1982 theory paper. The F-test now uses the effective degrees of freedom **at the
least-squares reference**, held fixed across the α sweep, exactly as Provencher defines it — where
0.13.0 mistakenly used the per-α value. In practice the F-test now selects a **smoother** α, in the
same range as the L-curve (both defensible), rather than a systematically rougher one. Only the F-test
option is affected; the default L-curve and every other analysis are unchanged. If you used the F-test
in 0.13.0, re-run those distributions.

---

## 0.13.0 — CONTIN α selection: L-curve (default) or Provencher's F-test (2026-07-03)

The DLS Distribution tab now lets you choose how CONTIN picks its regularisation parameter α.

- **L-curve corner** (default, unchanged): the elbow of the fit-vs-smoothness trade-off — modern
  and robust.
- **F-test (probability to reject)**: Provencher's original 1982 criterion — the smoothest solution
  whose fit is not statistically significantly worse than the best. Use it to compare against legacy
  CONTIN output. A **probability-to-reject** field (default 0.50) tunes the balance: a higher value
  gives a smoother, more parsimonious distribution; a lower value keeps more detail.

The selector lives on the Distribution tab next to the α controls; picking a method shows only that
method's controls. The chosen method (and, for the F-test, the level) is noted under the plot and
written into the CONTIN export. Distributions are unchanged when the default L-curve is left selected.
Note (Advanced Guide, CONTIN §): the F-test assumes independent residuals, but a single correlogram's
lag channels are correlated, so its level is a guide rather than an exact test — which is why the
L-curve stays the default.

---

## 0.12.0 — Choose the uncertainty estimator: HC3 (default) or classical OLS (2026-07-03)

A new **Settings → Uncertainty → Regression SE estimator** control lets you switch the ± reported
for every straight-line and multilinear fit (SLS Zimm/Berry, Debye, Guinier, calibration-free A₂;
DLS Γ vs q² and kD; the Rg/A₂–Mw scaling exponent) between two estimators.

- **HC3 (robust)** — the default, unchanged. A heteroscedasticity-consistent standard error that
  never under-reports when point precision varies across angles or concentrations.
- **Classical OLS** — the textbook *s*²(XᵀX)⁻¹ standard error. Choose it only to reproduce a value
  from classical software, a published table, or a spreadsheet **like-for-like**. It can
  under-report (~10 % low on a short, high-leverage concentration ladder), which is why it is not
  the default.

The choice is global, persists across restart, and is **recorded on each result and written into
the export**: an OLS ± column's *Comments* cell reads `SE: classical OLS` (HC3 stays silent). Point
estimates (Mw, Rg, D, Rh) are identical either way — only the ± changes. Switching the estimator
asks to confirm, then refreshes the affected results. SLS and DLS CSV exports now also include the
previously-omitted ± columns (Mw SE, Rg SE, A₂ SE, D SE, …). See Advanced Guide §15.1.1.

---

## 0.11.1 — Calmer warnings: neutral qualifiers no longer look like errors (2026-07-03)

Some SLS results printed a **bold red** line on every single-concentration or single-angle
run — "apparent…", "± statistical…" — in the same alarm-red used for a genuine problem. Those
are neutral, expected facts about that kind of result, not failures; too many always-on red
lines train the eye to ignore all of them.

- **"Apparent" and "± statistical" notes now read as a calm, neutral qualifier** — a
  steel-blue **ⓘ** line, distinct from the bold red **⚠** reserved for a genuine data-quality
  issue (uncalibrated, or the two Zimm/Berry extrapolation routes disagreeing by >10%). Nothing
  is hidden — apparent-vs-thermodynamic is still always shown — only the colour tier changed.
  A qRg out of the Guinier regime still escalates to the red tier.
- **The DDLS shape verdict moved to a hover tooltip.** "Sphere consistent / inconsistent" and
  the "dimensions assume the stated shape" caveat now show on hovering the shape table — the
  ✓/✗ and ratio cells, plus the panel title *"assumed geometry — not measured"*, still carry
  the gist at a glance — instead of an always-on line.
- **Trimmed the distribution replicate-average peak caveat** to one concise sentence.
- **An uncalibrated single-angle Mw is now flagged.** Previously a single-angle Mw computed
  without calibration was shown (and exported) with no "uncalibrated" note; it now marks the value
  `[unreliable — uncalibrated]`, shows the red **⚠** tier, and writes "uncalibrated, arbitrary
  scale" into the export — matching Debye/Guinier/Zimm.

Data-conditional warnings (PDI>0.3, non-diffusive γ-q², uncalibrated, >10% disagreement,
unphysical ρv) are unchanged.

---

## 0.11.0 — One consistent way to choose measurements (2026-07-02)

A deliberate pass to make selecting measurements clean, consistent, and obvious across
every tab. Before, each tab picked measurements differently; now there is **one idiom**,
and the Workspace list on the left reflects it.

- **One picker everywhere — tick to include.** Every analysis tab now selects with real
  **checkboxes** (tick = use it), grouped by sample, with **Select all / none** and a
  **?** help button. Ticked rows read **blue**. This replaces the older mix of
  highlight-to-select lists, so "what is selected" looks the same in the DLS
  Correlogram/Distribution/Summary, Γ vs q², D vs c, and DDLS tabs.
- **The Workspace list mirrors what the active tab has selected.** Whatever you tick in a
  tab is shown **bold blue** on the matching measurements in the left-hand Workspace
  list — so you can see at a glance what each tab is analysing. The Workspace list
  **navigates**; the tabs decide what gets analysed. Samples whose identity isn't
  confirmed yet read **amber**.
- **Clicking around no longer wipes a tab.** Selecting a measurement of a different kind
  (e.g. an SLS row while looking at a DLS fit) no longer blanks the tab you were on —
  each tab keeps its own selection until you change it there.
- **SLS and I·sin θ now have their own Sample dropdown.** Pick the sample to analyse right
  in the tab (only samples with the needed data appear), instead of relying on the
  sidebar. The I·sin θ tab previously had no picker at all.
- **DDLS: tick angles to include.** Choosing which paired angles enter the D_r/D_t fit is
  now a checkbox list (all ticked by default; untick an outlier) instead of clicking
  points on the plot. "Include all angles" re-ticks everything.
- **Cross-Sample: replicate-averaged Rh is now a selectable Rh source** — and it is
  preferred over a single-correlogram value, so refreshing the Cross-Sample tab no longer
  overwrites an Rh you deliberately obtained by averaging replicates.
- **Cross-Sample result tables now follow the display-unit setting.** The ρ table's Rg/Rh
  columns and the manual-entry fields honor **Settings → Display units** (renamed from
  "Plot axis units", since it now governs both plot axes and result tables) — e.g. show Rg/Rh
  in µm or Mw in kg/mol. Values are stored canonically and only converted for display, and a
  value you type is interpreted in the shown unit.
- **Consistent name for the depolarized dynamic sub-tab.** The DLS sub-tab for depolarized
  dynamic light scattering is now labelled **"DDLS"** everywhere (it was shown as "DPLS" on
  the tab while called DDLS in its controls and help). "DDLS" is depolarized *dynamic*
  scattering (rotational diffusion); the static depolarization calculator in the SLS tab is
  unchanged. No change to any calculation.

No analysis numbers changed. (Fit results are identical; this release is about how you
choose what to analyse and how the selection is shown.)

## 0.10.0 — Background analysis (2026-07-02)

- **Heavy analyses now run in the background — the window no longer freezes.** When a
  slow fit is running — a CONTIN size distribution, a DDLS rotational-diffusion fit,
  replicate averaging, a Γ-vs-q² / D-vs-c fit, an SLS Zimm/Berry/Debye/Guinier fit, the
  trace stationarity (ADF) check, or synthetic-data generation — the window stays movable
  and responsive. The **Run** button shows a busy state and a busy cursor appears; the
  result and plot update exactly as before once the fit finishes. **No change to any
  number** — results are identical to the previous (synchronous) version.
- **One analysis at a time.** While a fit is running, starting another, or editing
  parameters/calibration/masks/settings, is briefly held off with a "wait for the running
  analysis to finish" note (this keeps a background fit from computing on half-changed
  inputs). There is **no Cancel** — a running scientific computation can't be safely
  interrupted; if you change the inputs mid-run, the now-stale result is simply discarded
  when it arrives.

## 0.9.0 — Usability batch (2026-06-30)

Readability and theme polish from user feedback. **No change to the analysis math.**
First of several themed batches from this feedback round.

- **The dark theme is easier to read.** The small **"?"** help buttons are no longer
  faint on the dark theme, and several greyed notes/labels that were tuned for the light
  theme now have proper contrast on dark.
- **Settings section headers are readable on the light theme.** They previously rendered
  in a near-invisible light grey; they now use a clear, theme-aware colour (and switch
  correctly whenever you change the theme).
- **Colours now follow a theme switch immediately.** Changing **Settings → Theme** retints
  the headers, notes, flags, and "?" badges live, instead of leaving some stuck in the
  previous theme's colours.
- Tidy-up: the Settings intro no longer SHOUTS the word "seed" in all caps.
- **Tab labels no longer clip.** Tabs are sized with a little more room, so labels like
  "Settings" and the DLS sub-tabs aren't cut off on the right.
- **Overlapping plot labels are readable.** When several curves peak at nearly the same
  size, the peak labels (e.g. "0.33 nm") now stack neatly instead of piling into a blob;
  a crowded stack caps at six labels and shows "+N more" rather than becoming a mess. The
  same applies to sample-name labels on the Cross-Sample scaling plots.
- Tidy-up: the Correlogram baseline-region boxes now read "low"/"high" instead of
  "lo"/"hi".
- **macOS launcher.** The app can now be started on a Mac by double-clicking
  **`Launch DLS Buddy (MacOS).command`** (the Windows launcher is renamed to
  **`Launch DLS Buddy (Windows).bat`**). Like the Windows one, the first run sets up the
  environment automatically.
- **Data tab: Enter jumps to the next field.** After typing a value, pressing Enter now
  moves to the next parameter so you can keep entering values without reaching for the
  mouse.
- **Distribution tab: a peak-results panel.** The Distribution sub-tab now shows a results
  table for the ticked measurements (peak size + weight per method), like the Correlogram
  tab — colour-matched to the plotted curves.
- **Draggable boundaries are now visible.** Splitter dividers (control panel ↔ plot, and
  between stacked plots) show a small grip so you can tell they're draggable, and the
  fit/residual boundary now has a visible handle line in a clearer gap.
- **Resizable control columns.** On the DLS Correlogram/Distribution and SLS tabs you can
  drag the grips in the left column to resize the measurement list, controls, results
  table, and mask lists against each other.
- **Correlogram markers are easier to grab.** The delay-window and baseline-region markers
  now have offset handles — window carets at the top of the plot, baseline carets at the
  bottom — so you can pick and drag the one you want even when they sit at the same delay
  time.
- **Γ vs q² and D vs c tabs reworked.** Each now shows a table of the sample's
  measurements with **tick boxes** — you can see exactly which points are in scope and
  choose/exclude any subset (e.g. one concentration's angles), instead of every point
  being mixed into one line. The per-point Γ / D and the fit results (D, Rh, R², …) are
  shown in tables, and a note explains that Γ/D come from an internal cumulant fit of each
  correlogram (so you don't need to run the Correlogram tab first). Run is enabled only
  once at least two distinct angles/concentrations are ticked.
- **DPLS and SLS results shown as tables.** The depolarized (DDLS) results, shape models,
  and every SLS analysis (Zimm/Berry, Debye, Guinier, single-angle, calibration-free A₂,
  depolarization) now appear in tidy tables instead of run-together text.
- **Plots no longer disappear when switching measurements.** Selecting a measurement that
  belongs to a different analysis (e.g. clicking an SLS measurement while viewing a DPLS
  plot) no longer blanks the other tab's plot — each tab keeps its last view.
- **Quickstart: a "workflow at a glance" overview.** The Quickstart guide now opens with a
  short, step-by-step overview of a basic DLS/SLS analysis (Load → confirm parameters →
  run a method → export) so new users can see the whole flow before the details.
- **Quickstart synced to the reworked tabs.** Chapter 10 (Running an Analysis) and the
  launch/Visualization sections now describe the current UI — the Γ vs q² / D vs c points
  tables with tick-to-include, the Distribution peak-results panel, the offset
  correlogram-marker carets, the SLS/DDLS results tables and keep-last plots, the resizable
  plot/control splitter grips, and the per-OS double-click launchers.

---

## 0.8.2 — In-program help system + tooltip toggle (2026-06-29)

Makes the program more self-explanatory (user feedback). **No change to the
analysis math.**

- **"?" help buttons.** Key sections now have a small circular **?** you can click (or
  hover) for concise, plain-language help — Workspace, Data parameters, the DLS fit /
  distribution methods, SLS calibration and analysis, and Cross-Sample. They explain
  *how to use* the section and point to the Advanced Guide for the underlying maths.
- **Show/hide tooltips.** **Settings → Tooltips → "Show tooltips on hover"** turns all
  passive hover tooltips on or off (the "?" buttons still work on click when off).
- **Clearer method guidance.** Hovering a method selector now briefly explains and
  compares the options (Cumulant vs distribution; Zimm vs Berry; CONTIN vs NNLS vs
  Lognormal), and the help spells out **thermodynamic vs apparent** results.
- **Less repetition.** A freshly launched app no longer tells you to "load data" in
  three different places.
- **Fixed: the SLS control panel could clip on the right.** A long checkbox label was
  forcing the panel too wide, so part of the calibration controls were cut off and
  unreachable on a narrow panel. The panel now fits (the checkbox is now
  **"Per-sample calibration"** with the explanation moved to its tooltip), and any
  control panel that is genuinely too wide now gets a horizontal scrollbar instead of
  clipping.

---

## 0.8.1 — UI/UX batch: bugfixes, Undo/Reset, results panel, resizable plots (2026-06-29)

Fixes and usability polish from user feedback. **No change to the analysis math.**

- **Fixed: the delay-window upper marker looked "capped at 1 µs."** The green τ-window
  and baseline markers on the Correlogram now sit at the correct positions (they were
  being drawn in seconds on a microsecond axis); dragging them works correctly too.
  The actual fit window and results were always correct — only the on-plot markers were
  misplaced.
- **Fixed: an SLS sample's "Mw fraction" row showed a stray "mPa·s" unit.** That row now
  correctly has no unit (it's a text label like "1M").
- **Undo now does what you'd expect.** "Undo" steps back to the **previously applied**
  parameters. If you have un-applied edits showing, the first Undo discards those; press
  it again to step back through earlier Updates. It's disabled when there's nothing to undo.
- **New "Reset" button** (next to Update/Undo) clears all entered parameters for the
  sample so you can start fresh. It keeps the instrument-supplied scattering angle, asks
  for confirmation, and is **undoable** (nothing is applied until you press Update).
- **DLS results are in one place.** The Correlogram tab now has a single results table that
  shows the selected method's results (Cumulant / Single / Double / Stretched Exponential),
  with **Export CSV** directly beneath it — replacing the two stacked tables.
- **Plots fit the screen and resize.** Plots no longer overflow so the whole page scrolls;
  the control panel beside a plot scrolls on its own instead. **Drag the gap between a fit
  and its residual** to resize the residual — it stays perfectly aligned under the fit. On
  the Utilities Traces tab, the trace and its diagnostic plot sit in a **draggable vertical
  splitter**.
- **Renames/cleanup.** "Depolarized (DDLS)" → **DPLS**; "Update (commit)" → **Update**;
  "Undo to committed" → **Undo**; "KWW (stretched)" → **Stretched Exponential (KWW)**.
  Removed in-program literature citations (they live in the guides instead).
- **SLS tab polish.** Clearer sample names in the header (no more "?|?|…"), a shorter
  "no SLS data yet" message, and calibration labels that wrap instead of getting cut off.

---

## 0.8.0 — UI polish batch: workspace, traces, plot units (2026-06-26)

A broad usability pass across the GUI (no change to the analysis math).

- **Workspace management.** Right-click a **header** to remove in bulk: the sample
  header removes the whole sample, the **DLS** header removes only its DLS
  measurements, the **SLS** header only its SLS. **Left-click a header** selects
  every measurement under it (for a one-shot parameter edit + commit).
- **Selecting ≠ analysing.** Picking a measurement in the workspace no longer
  auto-adds it to the DLS analysis. The **“Measurements to plot”** checklist is now
  the only thing that decides what is fit/overlaid — tick what you want.
- **Intensity traces are first-class.** Traces now appear under a **Traces** node in
  the main workspace (load / remove there), and the Traces tab has the same
  multi-select checklist as the correlogram tab, so you can **overlay several
  traces**. Traces from the **Brookhaven** count-rate export now load via
  auto-detection (drop the file on *Load trace…*, no format picking).
- **Mw fraction** is applied like any other per-measurement parameter — set it on the
  row (or several highlighted rows) and press **Update**. The old
  apply-to-all-and-commit pop-up is gone.
- **Distributions** default to **CONTIN** (was NNLS).
- **Reset the delay window + baseline** to the defaults (full lag range; last 25 %)
  with one button on the Correlogram tab.
- **Remove outlier points and recompute** on **Γ vs q²**, **D vs c**, and **DDLS**:
  click a point to grey it out and exclude it from the fit; **Reset** brings them all
  back.
- **Human-centric plot units, configurable.** Plots now default to readable units —
  delay time in **µs**, count rate in **kcps**, q² in **nm⁻²**, D in **µm²/s**,
  concentration in **mg/mL** — and **Settings → Plot axis units** lets you choose the
  default unit for each axis. (Analysis still runs in canonical units; only the
  display converts.)
- **Resizable plots.** Plots grow with the window instead of being pinned to a fixed
  size, and the intensity-trace plots are taller by default.
- **Define ‘k’** (the shot-noise band multiplier) on the trace tab, and a new
  **running-average window** control (points; 0 = auto).
- **Synthetic generator** now lets you enter temperature and viscosity in your own
  units (defaults **°C** and **mPa·s**), and its β / noise / points defaults live on
  the tab itself.
- **Settings de-cluttered:** the synthetic-generator and intensity-trace defaults
  moved out of Settings into their own tabs; **cP** was dropped as a viscosity unit
  (identical to mPa·s).
- **Docs:** square-root symbols now render correctly in the PDFs, and vendor names
  were generalised out of the guides except where a specific parser/format is being
  described.

## 0.7.0 — Nonlinear (Frisken) cumulant, now the default (2026-06-25)

- **New cumulant fitting method: nonlinear (Frisken 2001), and it is now the
  default.** It fits the correlation function `g₂−1` *directly* with a **floating
  baseline**, rather than the classic linear fit to `ln(g₂−1)` (Koppel 1972, still
  available). Because the baseline and coherence are fitted instead of assumed, it
  is more robust to baseline drift and noisy/low-count data, and it uses the whole
  fit window (no amplitude cutoff). On a clean sample the two methods agree; the
  table shows which method produced each result and its fitted baseline.
- **Choose the method in Settings → Cumulant method** (Nonlinear / Linear). It is a
  **global** choice that applies to every cumulant-based analysis — the per-measurement
  cumulant, Γ vs q², and replicate averaging — so your whole DLS pipeline uses one
  method consistently.
- **Switching the method clears existing cumulant-based results** (cumulant fits,
  Γ vs q², replicate averages) so the workspace can't show a mix of methods. You are
  **asked to confirm first**, and only if such results exist; **distributions, SLS,
  and any hand-entered Rh are always kept**.
- Robustness: the nonlinear fit constrains the baseline to genuine drift (it cannot
  absorb the decay), and **falls back to the linear fit (flagged) if it fails to
  converge**, so you always get a result. Like the linear method it reports **no ±
  from a single correlogram** (uncertainty still comes from replicate averaging).
- **Known limitation:** a corrupted *first* channel (afterpulsing) is still best
  removed with *Skip initial lag channels* (v0.6.0); the nonlinear method handles
  drift and tail noise, not a short-lag spike — on such a detector it falls back to
  linear and flags it.

## 0.6.0 — DLS cumulant/distribution: skip leading lag channels (2026-06-25)

- **New Settings option: "Skip initial lag channels".** Drops the first *N*
  correlator channels (the shortest lags) from **every** DLS fit — cumulant,
  single/double/KWW, and NNLS/CONTIN/lognormal alike. The first few channels of a
  multi-τ correlator carry detector **afterpulsing** and dead-time artefacts that
  are not diffusion; a noisy first channel can drag a fit to a falsely *small*
  size. Setting the skip to your correlator's artefact count removes that bias.
  Default is **0 (no skip)**, so existing results are unchanged.
  - It **composes with the delay-window minimum**: the first fitted point is the
    later of "channel *N*" and "first lag ≥ the window minimum". The skip (by
    channel index) and the window (by time) are complementary.
  - Validated against the SMALS reference platform: on a deliberately noisy latex
    dataset a skip of 9 recovered detectors that the un-skipped cumulant misread
    (e.g. one angle 5.9 nm → 22 nm), matching the size the clean angles give.
  - **Use it sparingly with distribution methods (NNLS/CONTIN):** the short lags
    constrain the *small/fast* end of the size distribution, so an over-aggressive
    skip can erase a genuine small-particle population (a cumulant *mean* would only
    shift slightly). Keep it to the genuinely artefact-contaminated leading channels.

## 0.5.1 — UI polish: selection UX, plot sizing, scroll behaviour, Label field (2026-06-23)

Nine UI improvements from user feedback:

- **Measurement selection lists now use click-to-highlight** (Shift / Ctrl supported)
  instead of checkboxes in the DLS overlay checklist and the Cross-Sample
  include/exclude list. The DLS checklist also groups measurements under bold
  **sample headers** so you can tell at a glance which sample each measurement
  belongs to.
- **Select all / Select none buttons** added to both selection lists.
- **Label field** added to the Data tab (top row of the parameter table). A
  cosmetic per-measurement name: once committed it replaces the source filename
  everywhere the measurement is displayed — sidebar tree, DLS selection list.
  Does not affect analysis or sample grouping.
- **All plot areas are ~15% smaller by default** across the DLS, SLS,
  Cross-Sample, and Utilities modules. DLS and SLS already had draggable
  splitters; the Utilities **Traces** tab now has one too (drag the divider
  between the trace list and the plots).
- **Scroll wheel no longer changes dropdown values or cycles tabs.** An
  application-level event filter now suppresses wheel events on every
  QComboBox and tab bar — scroll wheel scrolls only.
- **"Load data to begin"** replaces the old "Load a correlogram to begin"
  status-bar message (the app handles more than correlograms).
- **"Solute name"** replaces "Sample name" in the Data tab parameter table
  (the field describes the polymer/molecule, not the sample group).
- **Right-click → Remove sample** added to the sidebar workspace tree.
  Removes the entire sample (all DLS, SLS, and solvent-reference measurements)
  with a confirmation dialog; source files on disk are untouched.

## 0.5.0 — DLS Summary tab + averaged-results persistence (2026-06-22)

- **New DLS "Summary" sub-tab.** A workspace-wide, exportable results table that
  persists across save/reload, replacing the one-time pop-up for averaged results
  and the hard-to-read sequential peak list on the Distribution tab. Two stacked
  tables:
  - **Per-measurement results** (one row per measurement): cumulant Rh + PDI,
    single/KWW Rh, and NNLS/CONTIN peaks shown as `Rh (Int %)` — where **Int %** is
    the intensity-weighted peak area (NOT a mass/weight percent).
  - **Sample-level Rh**: replicate averages, Γ–q² (q→0, *apparent*) and D–c (c→0,
    *thermodynamic*), with an **Rh Type** column making that distinction explicit and
    a **From** column listing the contributing measurements.
  - The left panel shares the same measurement checklist as the Correlogram and
    Distribution tabs and doubles as a **"Ticked only"** filter.
- **Averaged derived results now persist.** Running *Average derived results* still
  shows its summary, but the outcome is also recorded durably in the Summary table
  (and in the session file), clearly distinguished from an averaged-correlogram
  measurement. Per-measurement fit results populate the table as you run them.
- **Distribution tab: peak labels + a residuals panel.** Resolved peaks are now
  labelled on the size/rate distribution plot, and a residuals panel (data −
  reconstructed g₂−1 vs delay time) sits beneath it, like the Correlogram tab.
- **Controls moved next to the fit they drive.** The **cumulant order** lives on the
  Correlogram tab and the **Rh grid (min/max/points)** and **CONTIN L-curve α** range
  on the Distribution tab — both still seeded from Settings (the per-run value wins).
  These no longer appear under the Settings tab.
- CSV exports are now written as **UTF-8** (so labels like "Γ vs q²" export cleanly);
  pure-ASCII exports are unchanged.

## 0.4.0 — Malvern multi-measurement parsers + repo tidy-up (2026-06-21)

- **Malvern Zetasizer clipboard** loading now accepts files with **many records** —
  every record column in the pasted text loads as its own DLS measurement (a
  single-record copy still loads as one). Backward compatible.
- **New Malvern Zetasizer *export* parser** for the software's structured export
  (comma-separated, one measurement per row). Unlike the clipboard format it also
  pre-fills the parameters the file carries — refractive index, temperature,
  viscosity, and the sample/material/dispersant names — which you confirm or edit at
  the Data tab. Columns are matched by header name, so any column order from the
  configurable exporter works. The expected export template is documented in the
  Quickstart (§2.2).
- **Fixed a crash** when loading a correlogram that has no scattering angle yet
  (e.g. any freshly loaded Zetasizer file): selecting it no longer errors in the
  depolarized-DLS summary.
- **Repository tidy-up** (no effect on using the program): `code_map.md` and
  `code_references.md` now live in `docs/`; the version file moved to `app/`; a
  maintainer-only setup script is no longer shipped in the repo (forkers never
  needed it).

## 0.3.0 — Documentation overhaul (2026-06-21)

- Split the user guide into a **Quickstart** (how to use the program) and an
  **Advanced Guide** (theory, numbered equations, bibliography), both shipped as
  PDFs in `docs/`.
- Added **`code_map.md`** (directory + per-file tour for anyone reading/forking the
  code) and renamed the citation index to **`code_references.md`** (each source now
  maps to where it is used in the code and guide).
- Added this **`PATCH_NOTES.md`** and a project **version** (`version.py`, shown in
  the window title).
- Reworked **`README.md`** into a concise front door.
- Renamed the `Test Data` folder to `test-data`.
- Internal docs (the long development log, editable markdown sources) moved out of
  the tracked repo; the repo now ships only the user-facing artifacts.

## 0.2 and earlier — Engine + GUI build-out (pre-versioning)

The analysis engine and application were built and validated module by module
before formal version numbers existed. Capabilities as of this release:

- **Data model + parsers**: Brookhaven (DLS/SLS/trace), generic (DLS/SLS/trace),
  Malvern Zetasizer clipboard, ALV `.ASC` multi-angle. Loading is
  instrument-agnostic with auto-detection and a generic fallback.
- **DLS**: cumulants, single/double/KWW exponentials, NNLS, CONTIN, lognormal;
  Gamma-q^2 and concentration extrapolation; multi-measurement co-plotting;
  replicate averaging (correlogram average + mean +/- SD/sqrt(N), ISO 22412).
- **SLS**: unified calibration from a single calibrant point, Debye, Zimm/Berry,
  Guinier, single-angle, calibration-free A2, with data masking and a manual-Mw
  override.
- **Cross-Sample**: rho = Rg/Rh table and Rg-Mw / A2-Mw scaling plots with
  provenance-aware source pickers.
- **Utilities**: count-rate trace diagnostics, I*sin(theta) check, synthetic data
  generator.
- **Settings**: seed defaults for every module, light/dark theme, plot palette,
  input/display units.
- **DPLS/DDLS**: static depolarization geometry plumbing and a depolarized-DLS
  analysis path for rotational diffusion (see Known issues).
- Origin-compatible CSV export; matplotlib plotting layer; self-contained JSON
  sessions.

---

## Known issues / outstanding

- **DPLS/DDLS not yet tested on real depolarized data** — validated only against
  synthetic ground truth so far.
- **Visual peak picker** in the DLS distribution view is planned (peaks are already
  offered as Rh sources elsewhere, but not click-selectable in the plot).
- **Session JSON is not yet schema-versioned** — old sessions may not load after a
  data-model change.
- A few library PDFs are **citation "promotion candidates"** not yet formally cited.
