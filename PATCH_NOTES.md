# Patch Notes

User-facing summary of what each release changed, plus a running list of known
issues.

Versioning is pre-1.0 `0.MINOR.PATCH`: MINOR for new user-facing capability,
PATCH for fixes/polish. The version is set in `version.py` and shown in the GUI
window title.

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

## Unreleased

- **Consistent name for the depolarized dynamic sub-tab.** The DLS sub-tab for
  depolarized dynamic light scattering is now labelled **"DDLS"** everywhere (it was
  shown as "DPLS" on the tab while called DDLS in its controls and help). "DDLS" is
  depolarized *dynamic* scattering (rotational diffusion); the static depolarization
  calculator in the SLS tab is unchanged. No change to any calculation.

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
- **`divide by zero` RuntimeWarning** from `analysis/utilities.fit_count_rate_histogram`
  when a histogram bin is empty (chi-squared term). Harmless but noisy; guard pending.
- **Visual peak picker** in the DLS distribution view is planned (peaks are already
  offered as Rh sources elsewhere, but not click-selectable in the plot).
- **A2 source picker** in the Cross-Sample tab is planned.
- **Session JSON is not yet schema-versioned** — old sessions may not load after a
  data-model change.
- A few library PDFs are **citation "promotion candidates"** not yet formally cited.
