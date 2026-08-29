"""
gui/sls_module.py
=================

The SLS tab: static light scattering analysis for the selected SAMPLE. Unlike the
DLS tab (one measurement), SLS is inherently per-sample — a Zimm set is a solvent
reference (c = 0) plus a concentration series, analyzed together. Selecting any
measurement in the sidebar makes this tab operate on its whole sample.

It contains the visible **calibration panel** (manual entry of one calibrant point
→ k_c, geometry-aware toluene Rayleigh) and an analysis section (Zimm / Berry /
Debye / single-angle / calibration-free A₂ / excess Rayleigh ratio). Everything is
routed through the controller; no analysis or physics here.

Commit model: calibration and parameters are session/working state. "Update"
commits them (and recomputes k_c); "Run" uses the COMMITTED state. Soft flags
(uncalibrated, apparent) are GUI overlays, never on the figure.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtWidgets

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from plotting.plots import (
    plot_zimm, plot_debye, plot_guinier, plot_rayleigh_ratio,
    scale_causes, scaled_value,
    plot_calibration_free_a2, plot_form_factor, plot_ellipsoid_form_factor,
    plot_core_shell_form_factor, conc_mg_ml_label, display_factor_from,
    zimm_spacing,
)
from analysis import form_factor as ff
from gui.plot_controls import (
    AxisControlBar, make_split_panels, make_canvas_expanding, make_vertical_plot_stack,
    set_plot_export_name, themed_navtoolbar)
from gui.export_helper import export_to_csv, sample_suggested_name
from gui.field_labels import humanize_field_names
from gui.help import add_help_to_groupbox, section_header
from gui.plot_overlays import reissue_legend_preserving_title, show_plot_placeholder
from gui.widgets import (GripSplitter, RichTextDelegate, SampleSelector,
                         confirm_regroup_losses, roomy_tabs)
from gui.theme import ThemedLabel, set_flag, span, token as theme_token
from gui.worker import (
    BACKGROUND_RUN_TOOLTIP, BUSY_NOTICE, busy_notice, runner)
from gui.dls_module import _ols_marker
from analysis.uncertainty import format_fixed_sig, format_pm

# Single not-applicable placeholder for a structurally missing table cell, with a
# reason tooltip beside it (ui_style_guide S9; the cross_module precedent).
_EMPTY = '—'

# Reminder appended where a calibration-dependent quantity (Mw, absolute A₂) is
# shown with a ±: the SE is statistical (regression) only.
_STAT_CAVEAT = ' (± statistical; excludes calibration & dn/dc)'

# An absolute Mw whose scale is unknown is not a meaningful number. Rendered AT the value
# (not just in a below-fold flag) so the caveat travels with it; Rg and the
# calibration-free 2·A₂·Mw product survive and stay shown. TWO independent SCALE causes, both
# able to hold at once: no calibration constant (ΔR arbitrary) and no dn/dc (K = 1, so
# Kc/ΔR arbitrary). The dash names which, because the remedies differ — measure a
# calibrant, versus supply an increment — and a user told only “uncalibrated” would go
# calibrate and still see a dash. For a STORED Mw `_mw_scale_dash` adds a third, non-scale
# cause (a non-finite value); `_UNCAL_MW` stays the no-cause-available spelling.
_UNCAL_MW = '— (uncalibrated)'


# The convention's one body now lives in `plotting.plots`: the Debye/Zimm figure titles
# honour it too, a plotter cannot import the GUI, and a second spelling here is exactly the
# family defect it exists to prevent. Aliased under the private names the call sites use.
_scale_causes = scale_causes


def _scale_phrase(res) -> str:
    """The below-fold sentence's name for the unknown scale(s); '' when both are known.

    Prose spelling of `_scale_causes`, kept beside it so the dash at the value and the
    flag under the plot can never name different causes for the same result.
    """
    words = {'uncalibrated': 'uncalibrated',
             'no usable dn/dc': 'no usable dn/dc supplied'}
    return ' and '.join(words[c] for c in _scale_causes(res))


# The "dash the value, name the cause" convention, so the ten Mw/A₂ cells on this tab cannot
# disagree about which causes they honour (ui_style_guide §5 R5.2's carve-out: the dash IS the
# disclosure, so the surrounding readout stays calm) — and, since the move, cannot disagree
# with the figure drawn beside them either.
_scaled_value = scaled_value

# The calibration-free A₂ divides the (genuinely calibration-free) product 2·A₂·Mw by the
# sample's stored Mw, so the absolute A₂ inherits that Mw's scale. When the controller
# refuses an arbitrary-scale Mw the value is withheld — dashed AT the value, same
# convention as _UNCAL_MW, because 'set a manual Mw to get A₂' would be a LIE here: there
# IS an Mw, the program declined it, and a user who reads that types nothing and concludes
# the feature is broken. PLAIN text, unlike the flag and the row LABEL beside it: the
# results table applies `RichTextDelegate` to column 0 only, so markup in a VALUE cell
# renders as literal '<sub>' tags. `_UNCAL_MW` is plain for the same reason.
_UNCAL_A2 = '— (Mw on an arbitrary scale)'


# Calculation-nuance hover for the ill-conditioned flag (the always-visible flag says
# WHAT and the remedy; this tooltip carries the WHY/threshold, pointing to the guide —
# the how-to-use vs calculation-nuance tier split, CLAUDE.md rule 8).
_ILL_COND_TOOLTIP = (
    'Numerical-health flag: the extrapolation design (1, q², c) is near-singular, so the '
    '1/coefficient Mw/A₂ loses precision independent of magnitude. Threshold: condition '
    'number > 1×10¹² (a healthy design is ~10²–10⁴; float64 loses all precision near '
    '~10¹⁶). This is scale-invariant, not an “Mw too large” cutoff. The covariance behind '
    'the ± stops being computable at that same condition number, so a fit flagged here '
    'carries no ± either — the values are still shown. See the '
    'Theory-and-Equations-Guide, numerical-conditioning note.')

# Shown when the calibration-free A₂ withholds its absolute A₂ because the sample's stored
# Mw is on an arbitrary scale. `_set_flag(problem=True)` supplies the ⚠, so this text does
# not carry one. Deliberately names BOTH the cause and the two remedies, and says what
# survives — the product above the withheld cell is unaffected, and a user who reads only
# a warning would not know that.
_WITHHELD_A2_FLAG = (
    'A₂ withheld: this sample’s M<sub>w</sub> is on an arbitrary scale — the run was '
    'uncalibrated, or had no usable dn/dc, or both — and A₂ = 2·A₂·M<sub>w</sub> / (2·M<sub>w</sub>) '
    'would inherit it. The 2·A₂·M<sub>w</sub> above is unaffected. Supply what the M<sub>w</sub> '
    'readout’s dash names, or enter a manual M<sub>w</sub>, to get A₂.')

_WITHHELD_A2_TOOLTIP = (
    'The product 2·A₂·Mw is the calibration-free quantity: it is a ratio of excess '
    'intensities, so the unknown scale cancels and it is reported normally. The ABSOLUTE '
    'A₂ is that product divided by Mw, which puts the scale back in — an Mw on an '
    'arbitrary scale (no calibration, or no usable dn/dc) is not a molar mass, and '
    'dividing by '
    'it gives a number that is not an A₂. A hand-entered Mw (Data tab) is treated as a '
    'trusted external value and does yield A₂. See the Theory-and-Equations-Guide §4.4.')


def _mw_unreliable_caution(res) -> str:
    """Below-fold caution when a CALIBRATED SLS fit's Mw is unreliable for a NUMERICAL
    reason — an ill-conditioned design, a non-finite overflow, or a non-physical
    (sign-guarded) intercept/divisor.

    Empty when Mw is reliable, or when EITHER scale is unknown -- uncalibrated or no
    dn/dc (each has its own louder wording and dashes the value out). The exclusion is
    on `_scale_causes`, not on `calibrated` alone: `mw_reliable` is a conjunction, and
    when it gained the dn/dc term this clause silently began claiming a NUMERICAL fault
    on every calibrated dn/dc-less fit -- converged, well-conditioned, exact Rg, and
    told it was a "non-physical or degenerate fit". Unlike those cases the NUMBER is
    still shown — finite, or rendered 'n/a' when non-finite — so the reason rides in the
    flag. This is a data-conditional problem, so it is the red ⚠ `error` tier, matching
    "uncalibrated" (ui_style_guide §5 R5.2), not a neutral ⓘ qualifier. It keys on
    `mw_reliable` (a conjunction of calibration, sign, conditioning and finiteness gates),
    not on `well_conditioned` alone, so a value unreliable for a non-conditioning reason
    is never left with a neutral "apparent" flag next to an 'n/a'. Prefers the result's
    specific `reliability_note`; the condition-number tooltip is attached separately, only
    for the genuine ill-conditioning case (`not well_conditioned`)."""
    if not _scale_causes(res) and not getattr(res, 'mw_reliable', True):
        return (getattr(res, 'reliability_note', '')
                or 'Mw is unreliable (non-physical or degenerate fit).')
    return ''


def _rg_unreliable_caution(res) -> str:
    """Below-fold caution when a form-factor fit's SIZE is unreliable -- today, a fit that
    collapsed below `analysis.sls.FORM_FACTOR_MIN_SIZE_NM`.

    Deliberately NOT gated on `_scale_causes`, unlike its Mw sibling above. That exclusion
    is right for Mw, whose unreliability an unknown scale already explains more loudly; it
    is wrong here, because the size floor has nothing to do with scale -- Rg survives an
    uncalibrated run, so on such a run the scale wording is the ONLY thing shown and the
    sub-molecular reason would reach the user nowhere at all (round 943d f1: the tab
    asserted "Mw unreliable (Rg/R survive)" beside a row reading Rg = 0.0200 nm).

    Red ⚠ `error` tier, matching the Mw caution and the DLS `rh_reliable` glyph
    (ui_style_guide §5 R5.2: one condition, one tier, across surfaces).
    """
    if getattr(res, 'rg_reliable', True):
        return ''
    return (getattr(res, 'reliability_note', '')
            or 'The fitted size is not physically meaningful.')


def _rg_cell(cell: str, res) -> str:
    """Mark a size cell whose own reliability flag is False -- the DLS convention
    (`gui/dls_module.py`'s `rh` cells), so the judgement ABOUT the size is shown BY the
    size. The sentence itself rides on the flag label below."""
    return cell if getattr(res, 'rg_reliable', True) else f'{cell} ⚠'

# Shown when a redisplayed (cached) SLS fit no longer matches its committed params /
# mask / calibration / estimator — the same explicit-Run + staleness-hint contract the
# DLS tabs use. Matches gui.dls_module._STALE_HINT wording.
_SLS_STALE_HINT = 'Inputs changed since this fit — press Run to refresh.'

# Display names for the quantities a Zimm/Berry run writes back (§9 casing).
_QTY_DISPLAY = {'mw': 'Mw', 'rg': 'Rg', 'a2': 'A₂', 'rh': 'Rh'}


def _adapt_form(form: QtWidgets.QFormLayout) -> None:
    """Let a form adapt to a narrow control panel instead of clipping:
    fields grow to the available width and a too-wide row wraps its field below the
    label rather than overflowing the panel."""
    form.setFieldGrowthPolicy(
        QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)


def _sample_label(sample) -> str:
    """Human-readable sample name for the SLS header — the same 'polymer / solvent @
    T' form the sidebar shows, instead of the raw 'poly|solv|temp' sample_id key whose
    unconfirmed parts render as '?'."""
    poly, solv, temp = sample.polymer_name, sample.solvent_name, sample.temperature_K
    if poly and solv and temp is not None and not math.isnan(temp):
        return f'{poly} / {solv} @ {temp:g} K'
    return '(unconfirmed sample)'


_SLS_METHODS: List[Tuple[str, str]] = [
    ('Zimm', 'zimm'),
    ('Berry', 'berry'),
    ('Form factor P(q)', 'form_factor'),
    ('Debye (single c, apparent)', 'debye'),
    ('Guinier (single c, apparent)', 'guinier'),
    ('Single-angle Mw (apparent)', 'single'),
    ('Calibration-free A₂', 'calfree'),
    ('Excess Rayleigh ratio', 'rayleigh'),
]

# Form-factor model dropdown entries: (label, engine key). The ellipsoid (Phase 2) and the
# core-shell sphere (v1.10.0) return MULTIPLE solutions -- a prolate and an oblate ellipsoid,
# or several radius pairs, can fit the same P(q) -- so their display path differs from the
# single-result sphere / coil / Mie (see _present_method, which routes a list by model). The
# Mie sphere (Phase 3) is single-valued in R given the contrast m and needs the hand-entered
# solute refractive index; the core-shell sphere needs BOTH the core and shell indices.
# Labels come from ff.MODEL_LABELS so the engine owns the canonical wording.
_FORM_FACTOR_MODELS: List[Tuple[str, str]] = [
    (ff.MODEL_LABELS[ff.SPHERE], ff.SPHERE),
    (ff.MODEL_LABELS[ff.DEBYE], ff.DEBYE),
    (ff.MODEL_LABELS[ff.ELLIPSOID], ff.ELLIPSOID),
    (ff.MODEL_LABELS[ff.MIE], ff.MIE),
    (ff.MODEL_LABELS[ff.CORE_SHELL], ff.CORE_SHELL),
]

# Calibration line-edit fields: (attribute, label, allow_blank_as_None).
_CAL_FIELDS = [
    ('calibrant_intensity', 'Calibrant intensity', True),
    ('calibrant_angle_deg', 'Calibrant angle (°)', False),
    ('standard_wavelength_nm', 'Standard wavelength (nm)', False),
    ('standard_temperature_C', 'Standard temperature (°C)', False),
    ('standard_refractive_index', 'Standard refractive index', True),
    ('dark_count_rate', 'Dark count rate', False),
]

# Intensity fields share the SLS file's own (instrument-specific) intensity unit;
# the program never converts them because only intensity RATIOS enter the analysis.
_CAL_TOOLTIPS = {
    'calibrant_intensity':
        'Enter in the SAME units as your SLS intensity file (cps, kcps, … — '
        'whatever the file uses). Only its ratio to the sample intensities enters '
        'the Rayleigh ratio, so the program does not convert it.',
    'dark_count_rate':
        'Detector dark count, in the SAME units as your SLS intensity file; it is '
        'subtracted from each intensity before the Rayleigh ratio.',
    'calibrant_angle_deg':
        'Scattering angle at which the calibrant intensity was measured.',
    'standard_wavelength_nm':
        'Laser wavelength used for the calibration standard (normally the same as '
        'the sample measurement). The toluene Rayleigh ratio is tabulated at 532 and '
        '660 nm; a wavelength within 2 nm of one is served that entry and the note '
        'beside k_c says so, and a mismatch with the loaded SLS data is noted there too.',
    'standard_temperature_C':
        'Temperature of the calibration standard; it sets the standard’s Rayleigh '
        'ratio (toluene’s is temperature-dependent). See the Theory-and-Equations-Guide.',
    'standard_refractive_index':
        'Refractive index of the calibration standard at this wavelength and '
        'temperature; enters the (n_solvent/n_standard)² correction. A teal dot marks '
        'a value proposed from the solvent library (toluene), re-proposed when you '
        'change the wavelength or temperature. A value you type (no dot) is never '
        'overwritten; clear the field to return to the library value. Where the '
        'library has no value the field stays blank and the correction cannot be '
        'applied: the run is then flagged uncalibrated (Mw and absolute A₂ '
        'unreliable; Rg unaffected).',
}


class _AnalysisTab(QtWidgets.QWidget):
    """Per-sample SLS analysis with a calibration panel.

    The SLS tab's first sub-tab ("Analysis"). It was the whole tab until the
    trace-derived surface was added beside it; :class:`SLSModule` is now the
    shell that holds both.
    """

    committed = QtCore.Signal()   # emitted after Apply (grouping/k_c may change)
    selectionChanged = QtCore.Signal()   # emitted when the analyzed sample changes (mirror)
    # Emitted when the USER picks a sample in this tab's dropdown, so the shell makes it
    # the single active sample and fans it out to every tab + the sidebar.
    sampleFocusRequested = QtCore.Signal(str)

    def __init__(self, controller, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.sample_id: Optional[str] = None
        self._runnable = False
        self._suppress_cal = False
        self._suppress_mask = False
        self._ran = False               # a result is on screen (gate mask re-runs)
        # How to export the on-screen result: a (filename, path->path) pair set by
        # _run_method, or None when nothing exportable is shown. Wrapping it lets the
        # one export button serve every method without the handler re-deriving state.
        self._export: Optional[Tuple[str, object]] = None
        # The single concentration the current debye/guinier plot was drawn for. The
        # click-to-mask hit-test must use THIS, not a fresh conc_combo read: the combo
        # stays enabled for debye/guinier and changing it doesn't redisplay, so a live
        # read would mask a point on a concentration that isn't on screen.
        self._plotted_conc: object = None
        # ... and the method it was drawn with, for the same reason: the Method
        # combo stays live and does not redisplay, so a click after browsing it
        # would be hit-tested under another method's transform.
        self._plotted_method: Optional[str] = None
        self._zimm_k = 1.0              # Zimm grid spacing, shared with the overlay
        # The q² display factor the Zimm-k FIELD text is currently written in. A
        # typed k is an axis-unit number (matching the k printed in the x-label),
        # so when Settings changes the q² display unit the stale text must be
        # rewritten into the new unit — else it would be reinterpreted against the
        # new factor and silently rescale (collapsing the grid). This records the
        # field's current unit so `_reconcile_k_field_units` can convert it; 1.0 is
        # the default nm⁻² factor.
        self._k_field_qfac = 1.0
        # Unmasked Rayleigh series from the last run, keyed by (sample, fraction):
        # the masked-point overlay and click hit-testing read this instead of
        # recomputing on every draw (which would call the controller off the run
        # path and could race a background fit).
        self._full_rr: Dict[tuple, list] = {}
        self._run_epoch = 0            # async staleness token (bumped on sample switch)
        # The active molecular-weight fraction (None = unfractioned / whole sample).
        self._fraction: Optional[str] = None
        self._suppress_fraction = False
        # Last-run memory (within-session) keyed by (sample_id, fraction): which
        # method + axis points were last analyzed, so switching samples/fractions
        # and back restores the view. SLS re-runs cheaply from committed state (+
        # persisted masks), so we replay the run rather than caching result objects.
        self._last_run_by_sample: Dict[tuple, Dict] = {}
        self.ax = None
        self._build_ui()
        self._populate_calibration()
        self._refresh_calibration_display()
        self.set_measurement(None)

    # ------------------------------------------------------------------ UI ---
    def _build_ui(self) -> None:
        # Control panel | plot split is draggable. The control column
        # has long calibration labels and a wide button, so it keeps a minimum width.
        _, left, right = make_split_panels(self, left_min_width=340)

        # The SLS dropdown is a mirror/override of the shell's single active sample
        # picking here sets the active sample everywhere; clicking a sample
        # in the sidebar drives this too. A whole Zimm set (solvent ref + concentration
        # series) is analyzed per sample.
        self.sls_selector = SampleSelector(
            self.controller, predicate=lambda s: s.has_sls,
            label_fn=_sample_label, title='SLS Sample',
            help_text='Choose which sample to analyze (its whole Zimm set).',
            help_bullets=['Only samples that have SLS intensity data appear.',
                          'Picking here — or clicking a sample in the Workspace — sets '
                          'the active sample for every tab.'])
        self.sls_selector.sampleChanged.connect(self._on_sls_sample)
        left.addWidget(self.sls_selector)

        self.header = QtWidgets.QLabel()
        self.header.setWordWrap(True)
        left.addWidget(self.header)

        # Calibration + Analysis stack at their NATURAL height in the left scroll area
        # (no inner splitter). A fixed-height splitter pane clipped the word-wrapped
        # "uncalibrated" k_c note top and bottom; letting the groups size to content and
        # leaning on the scroll area for overflow keeps that message fully readable. The
        # Mask panel + Results moved under the plot (right column); the anisotropy
        # (Cabannes) correction now lives inside the Calibration group.
        cal = self._build_calibration_group()
        ana = self._build_analysis_group()
        mask = self._build_mask_group()
        left.addWidget(cal)
        left.addWidget(ana)

        # Warnings stay at the bottom of the control column — the shared house convention
        # across every tab (results → red ⚠ flag → neutral ⓘ notes). With Results + Mask
        # now under the plot, they sit above the fold anyway.
        self.status = QtWidgets.QLabel('')        # one-line notes (e.g. Rayleigh series)
        self.status.setWordWrap(True)
        left.addWidget(self.status)
        self.flag_label = ThemedLabel('', role='error', bold=True)
        self.flag_label.setWordWrap(True)
        left.addWidget(self.flag_label)
        # Passive ⓘ notices carried on the result (e.g. dropped non-positive points) —
        # a neutral qualifier tier, separate from the severity-tracking flag_label above.
        # The analysis emits the same text via warnings.warn (keep-both).
        self.notes_label = ThemedLabel('', size=11)
        self.notes_label.setWordWrap(True)
        left.addWidget(self.notes_label)
        left.addStretch(1)      # keep the controls top-aligned; scroll area handles overflow

        # Right column: the plot over a draggable Results | Mask strip. Keeping the
        # results table next to the plot it describes (and the mask near the points you
        # click) declutters the left column; the Cross-Sample tab sets the same precedent
        # of content in the right column.
        self.figure = Figure(figsize=(5.5, 4.6))
        self.canvas = make_canvas_expanding(FigureCanvas(self.figure))
        self.nav_toolbar = themed_navtoolbar(self.canvas, self, self.controller)
        self.axis_bar = AxisControlBar(self.canvas)
        plot_holder = QtWidgets.QWidget()
        pv = QtWidgets.QVBoxLayout(plot_holder)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.addWidget(self.nav_toolbar)
        pv.addWidget(self.canvas, 1)
        pv.addWidget(self.axis_bar)

        # Analysis results as a table, rebuilt per method. A titled group box (rather
        # than a bare label) so it aligns with the neighbouring Data Mask box's title.
        results_col = QtWidgets.QGroupBox('Results')
        rc = QtWidgets.QVBoxLayout(results_col)
        self.result_table = QtWidgets.QTableWidget(0, 2)
        self.result_table.horizontalHeader().setVisible(False)
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.result_table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.result_table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        # Column 0 holds quantity names (M_w, R_g, k_c …); render it as rich text so their
        # subscripts show — a plain table item can't (no Unicode subscript for w/g/c).
        self.result_table.setItemDelegateForColumn(0, RichTextDelegate(self.result_table))
        rc.addWidget(self.result_table, 1)
        results_col.setMinimumWidth(240)     # keep the value column readable

        # Results | Mask side by side under the plot, in a draggable horizontal splitter
        # so the user can rebalance them (the mask's two lists would otherwise crowd out
        # the results table). Initial split favours the wider mask.
        self.bottom_holder = GripSplitter(QtCore.Qt.Orientation.Horizontal)
        self.bottom_holder.addWidget(results_col)
        self.bottom_holder.addWidget(mask)
        self.bottom_holder.setStretchFactor(0, 0)
        self.bottom_holder.setStretchFactor(1, 1)
        self.bottom_holder.setSizes([320, 560])

        rstack = make_vertical_plot_stack(
            [plot_holder, self.bottom_holder], sizes=[500, 220],
            min_heights=[220, max(mask.sizeHint().height(), 150)])
        right.addWidget(rstack, 1)
        # Click a plotted point to hide/show just that (concentration, angle).
        self.canvas.mpl_connect('button_press_event', self._on_canvas_click)

    def _build_calibration_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox('Calibration')
        add_help_to_groupbox(box, 'Turns raw intensities into absolute Rayleigh '
                             'ratios — required for Mw and A₂.', bullets=[
                                 'Enter <b>one calibrant point</b>: its intensity, '
                                 'angle, and the standard\'s wavelength / temperature '
                                 '/ refractive index / geometry.',
                                 'The program computes <b>k_c</b> from it — don\'t use '
                                 'a vendor constant (it goes stale after alignment).',
                                 'Calibrate once <b>session-wide</b>; tick '
                                 '"Per-sample calibration" only to override one sample.',
                                 'Rg and the calibration-free product survive without '
                                 'calibration; Mw and absolute A₂ do not.',
                                 'A standard temperature <b>outside 10–50 °C</b> is '
                                 'flagged: the toluene Rayleigh-ratio correction is '
                                 'extrapolated beyond its validated table there.',
                                 'The note beside k_c also says when the standard '
                                 'wavelength is served from the nearest table entry '
                                 '(532 / 660 nm), when it differs from the loaded SLS '
                                 'data\'s wavelength, and when the standard refractive '
                                 'index is blank (the run is then flagged uncalibrated) '
                                 'or differs from the solvent library.',
                                 'A <b>teal dot</b> beside the standard refractive '
                                 'index marks the solvent library\'s proposal, which '
                                 'follows the wavelength and temperature; a value you '
                                 'type has no dot and is never overwritten. Clear the '
                                 'field to return to the library value.',
                             ])
        form = QtWidgets.QFormLayout(box)
        _adapt_form(form)      # wrap long rows + grow fields so nothing clips

        self.per_sample_check = QtWidgets.QCheckBox('Per-sample calibration')
        self.per_sample_check.setToolTip(
            'Give this sample its own calibration. When off, the session-wide '
            'calibration is used.')
        self.per_sample_check.toggled.connect(self._on_per_sample_toggled)
        form.addRow(self.per_sample_check)

        self.cal_edits: Dict[str, QtWidgets.QLineEdit] = {}
        for key, label, allow_blank in _CAL_FIELDS:
            edit = QtWidgets.QLineEdit()
            if allow_blank:
                edit.setPlaceholderText('(blank = none)')
            edit.editingFinished.connect(lambda k=key: self._on_cal_edit(k))
            self.cal_edits[key] = edit
            # Intensities (calibrant + dark) must be in WHATEVER unit the SLS file
            # uses: only their ratio to the file's sample intensities enters the
            # Rayleigh ratio, so the program can't convert them to a fixed unit.
            tip = _CAL_TOOLTIPS.get(key)
            if tip:
                edit.setToolTip(tip)
            if key == 'standard_refractive_index':
                # The provenance dot the Data tab paints on a library-proposed cell,
                # on a form line-edit: a leading label beside the field, shown while
                # the value is the library's (`_populate_calibration`), hidden for a
                # typed one -- one hue for "proposed", the tooltip carrying the source
                # (ui_style_guide, provenance dot).
                self.std_index_dot = QtWidgets.QLabel()
                self.std_index_dot.setToolTip(
                    'Proposed from the solvent library (toluene) at the standard\'s '
                    'wavelength and temperature; re-proposed when either changes. '
                    'No dot = your value, never overwritten.')
                row = QtWidgets.QWidget()
                hb = QtWidgets.QHBoxLayout(row)
                hb.setContentsMargins(0, 0, 0, 0)
                hb.addWidget(self.std_index_dot)
                hb.addWidget(edit, 1)
                form.addRow(label + ':', row)
                continue
            form.addRow(label + ':', edit)

        self.geometry_combo = QtWidgets.QComboBox()
        self.geometry_combo.addItems(['VU', 'VV', 'VH'])
        self.geometry_combo.setToolTip(
            'Scattering geometry of the CALIBRANT (the toluene standard) — it sets which '
            'toluene Rayleigh ratio is used: VV = polarized, VH = depolarized, VU = no '
            'analyzer. Distinct from the sample anisotropy correction below. '
            'See the Theory-and-Equations-Guide.')
        self.geometry_combo.currentTextChanged.connect(self._on_geometry_changed)
        # "Calibrant (toluene) geometry", not "Standard geometry", so it is unmistakably
        # about the STANDARD — distinct from the sample's optical-anisotropy correction.
        form.addRow('Calibrant (toluene) geometry:', self.geometry_combo)

        self.kc_label = QtWidgets.QLabel('')
        self.kc_label.setWordWrap(True)
        # "Calibration k_c" — distinct from the Zimm spacing k: k_c is
        # the Rayleigh-ratio-per-intensity calibration constant, not the q²+k·c grid k.
        # (Subscript via HTML — a QFormLayout label is a QLabel and renders rich text.)
        form.addRow('Calibration k<sub>c</sub>:', self.kc_label)

        # Neutral ⓘ notes next to k_c, one per line, composed by the controller
        # (`preview_calibration_note`): a standard temperature outside the 10–50 °C
        # toluene Rayleigh table or a wavelength served from its nearest entry (both
        # extrapolated k_c; the physics layer emits the same text via warnings.warn,
        # keep-both), a blank or off-library standard refractive index, and a standard
        # wavelength that is not the wavelength the loaded SLS data carry.
        self.cal_note = ThemedLabel('', size=11)
        self.cal_note.setWordWrap(True)
        self.cal_note.setToolTip(
            'Shown when the calibration needs a word before you run: the standard '
            'temperature or wavelength is off the toluene Rayleigh-ratio table (the '
            'value is extrapolated), the standard refractive index is blank or differs '
            'from the solvent library, or the standard wavelength is not the wavelength '
            'your SLS data were measured at. See the Theory-and-Equations-Guide §1.')
        form.addRow('', self.cal_note)

        self._build_anisotropy_subsection(form)

        self.apply_button = QtWidgets.QPushButton('Update')
        self.apply_button.setToolTip(
            'Apply the calibration (and any other pending edits) to the sample. '
            'If a pending change regroups samples, stored results follow; you '
            'are asked first if a merge would drop a hand-entered or picked '
            'value or a per-sample calibration.')
        self.apply_button.clicked.connect(self._on_apply)
        form.addRow(self.apply_button)
        return box

    def _build_anisotropy_subsection(self, form: QtWidgets.QFormLayout) -> None:
        """Optical-anisotropy (Cabannes) check, folded into the Calibration group — a
        DISPLAY-ONLY calculator (not applied to the sample or exported).

        Tick it and give the depolarization ratio ρv (directly, or as VV/VH intensities);
        it shows ρv, the optical anisotropy δ², the Cabannes factor f = 1 − 4ρv/3, and the
        isotropic-corrected weight f·Mw for the current sample's analyzed Mw. The first three
        are ratios and survive an uncalibrated run; f·Mw does not, and is dashed out there
        like every other absolute Mw. To *use* the corrected value, enter it by hand as a
        Manual Mw (Analysis) — the program never writes it back, so it can never silently
        propagate into Cross-Sample/exports.
        Placed with the calibration machinery it relates to; its VV/VH is the *sample's*
        anisotropy, distinct from the 'Calibrant (toluene) geometry' above."""
        self.aniso_check = QtWidgets.QCheckBox('Anisotropy correction')
        self.aniso_check.setToolTip(
            'Optional, display-only. Shows the isotropic-corrected Mw (Mw × f, '
            'f = 1 − 4ρv/3) for a depolarizing sample from its depolarization ratio ρv. '
            'Not applied or exported — enter it as a Manual Mw to use it. This is the '
            'sample’s own anisotropy, not the calibrant geometry above. ρv, δ² and f are '
            'ratios, so they are shown whether or not the M<sub>w</sub> scale is known; '
            'the corrected M<sub>w</sub> is not (it would inherit that scale).')
        self.aniso_check.toggled.connect(self._on_anisotropy_toggled)
        form.addRow(self.aniso_check)

        # The inputs + readout live in a body widget shown only when the box is ticked,
        # so an unused correction adds just one checkbox row to the panel.
        self.aniso_body = QtWidgets.QWidget()
        bform = QtWidgets.QFormLayout(self.aniso_body)
        bform.setContentsMargins(16, 0, 0, 0)      # indent under the checkbox
        _adapt_form(bform)

        self.aniso_mode_combo = QtWidgets.QComboBox()
        self.aniso_mode_combo.addItem('VV & VH intensities', 'intensities')
        self.aniso_mode_combo.addItem('Depolarization ratio ρv', 'ratio')
        self.aniso_mode_combo.currentIndexChanged.connect(self._on_anisotropy_mode_changed)
        bform.addRow('Input:', self.aniso_mode_combo)

        self.aniso_stack = QtWidgets.QStackedWidget()
        int_page = QtWidgets.QWidget()
        ip = QtWidgets.QFormLayout(int_page)
        ip.setContentsMargins(0, 0, 0, 0)
        _adapt_form(ip)
        self.aniso_ivv_edit = QtWidgets.QLineEdit()
        self.aniso_ivv_edit.setPlaceholderText('in file units')
        self.aniso_ivh_edit = QtWidgets.QLineEdit()
        self.aniso_ivh_edit.setPlaceholderText('in file units')
        self.aniso_dark_edit = QtWidgets.QLineEdit()
        self.aniso_dark_edit.setPlaceholderText('0 (optional)')
        _atip = ('Same units as your intensities; subtracted from both before the ratio. '
                 'The depolarized VH channel is weak, so an un-subtracted dark count '
                 'biases ρv high.')
        self.aniso_dark_edit.setToolTip(_atip)
        for w in (self.aniso_ivv_edit, self.aniso_ivh_edit, self.aniso_dark_edit):
            w.editingFinished.connect(self._on_anisotropy_input_changed)
        ip.addRow('I<sub>VV</sub> (polarized):', self.aniso_ivv_edit)
        ip.addRow('I<sub>VH</sub> (depolarized):', self.aniso_ivh_edit)
        ip.addRow('Dark count:', self.aniso_dark_edit)

        ratio_page = QtWidgets.QWidget()
        rp = QtWidgets.QFormLayout(ratio_page)
        rp.setContentsMargins(0, 0, 0, 0)
        _adapt_form(rp)
        self.aniso_rhov_edit = QtWidgets.QLineEdit()
        self.aniso_rhov_edit.setPlaceholderText('e.g. 0.05')
        self.aniso_rhov_edit.editingFinished.connect(self._on_anisotropy_input_changed)
        rp.addRow('ρ<sub>v</sub>:', self.aniso_rhov_edit)

        self.aniso_stack.addWidget(int_page)     # index 0 -> 'intensities'
        self.aniso_stack.addWidget(ratio_page)   # index 1 -> 'ratio'
        bform.addRow(self.aniso_stack)

        # Compact read-only readout: ρv, δ², the Cabannes factor f, and f·Mw (display only).
        self.aniso_readout = ThemedLabel('', role='hint', size=11)
        self.aniso_readout.setWordWrap(True)
        bform.addRow('', self.aniso_readout)

        form.addRow(self.aniso_body)
        self.aniso_body.setVisible(False)

    def refresh_anisotropy_check(self) -> None:
        """Re-evaluate the display-only readout — call after a run/sample change so the
        f·Mw line tracks the current analyzed Mw. Public so the run/redisplay paths can
        refresh it without reaching into the private handler."""
        self._refresh_anisotropy_display()

    def _build_analysis_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox('Analysis')
        add_help_to_groupbox(box, 'Get Mw, Rg and A₂ from the intensities.', bullets=[
            '<b>Thermodynamic</b> (Zimm/Berry): extrapolated to zero angle <i>and</i> '
            'zero concentration — the true Mw, Rg, A₂.',
            '<b>Apparent</b> (Debye/Guinier/single-angle): one concentration or angle '
            'only — a quick estimate, not extrapolated.',
            'Zimm fits the standard plot; <b>Berry</b> (square-root axis) is better for '
            'larger or higher-Mw particles where Zimm curves.',
            '<b>Form factor P(q)</b>: fits the full angular form factor (choose '
            '<b>Sphere</b>, <b>Gaussian coil</b>, <b>Ellipsoid</b>, <b>Mie sphere</b>, or '
            '<b>Core–shell sphere</b>) instead of the small-qRg slope, so it stays valid to '
            'larger qRg and returns Mw + Rg (and, for a sphere/Mie, the geometric radius R). '
            'Apparent at one concentration; thermodynamic (A₂ resolved) across several. The '
            '<b>Ellipsoid</b> also fits a shape (axial ratio); because a prolate and an oblate '
            'ellipsoid can fit the same P(q), it shows <i>every</i> matching solution and flags '
            'the fit <b>degenerate</b> — pick the shape from independent knowledge, not from the '
            'fit alone. <b>Mie sphere</b> is the general case for high contrast / larger '
            'particles (e.g. latex in water) where the analytic sphere fails; it needs the '
            'hand-entered <b>solute refractive index</b> (Data tab) to set the contrast. '
            '<b>Core–shell sphere</b> fits a coated particle (core of one index in a shell of '
            'another) — it needs the core (solute) <i>and</i> shell refractive indices (Data '
            'tab) and fits two radii; two radii from one curve are often under-determined, so '
            'like the ellipsoid it shows every matching solution and flags the fit '
            '<b>degenerate</b>. Its reported Rg is the contrast-weighted scattering Rg. '
            'Degenerate solutions are listed smallest shape first (ellipsoid: by axial '
            'ratio; core–shell: by outer radius) — a fixed listing convention, not a '
            'ranking: the solutions fit equally.',
            'Flags under the plot come in two tiers: a calm <b>ⓘ</b> note is a neutral, '
            'expected qualifier (e.g. “apparent”, “± statistical only”) — not a problem; '
            'a bold red <b>⚠</b> is a genuine data-quality issue (uncalibrated; an '
            '<b>ill-conditioned design</b> — clustered angles or near-equal '
            'concentrations make Mw/A₂ numerically unreliable; or the two extrapolation '
            'routes disagree by &gt;10%).',
            '<b>Run</b> is the only thing that computes a fit. Switching sample '
            'or fraction, or committing a parameter, redisplays the last fit and — if the '
            'inputs changed — flags it “press Run to refresh” rather than silently '
            'recomputing. Hiding/showing a point (mask) refits immediately.',
            'See the Theory-and-Equations-Guide for the equations.',
        ])
        form = QtWidgets.QFormLayout(box)
        _adapt_form(form)      # wrap long rows + grow fields so nothing clips

        # Molecular-weight fraction selector. A Mw series shares one solvent
        # reference but holds several fractions ("250k", "1M", ...); each is its own
        # Zimm fit. Hidden/disabled when a sample has just one (unlabeled) fraction.
        self.fraction_combo = QtWidgets.QComboBox()
        self.fraction_combo.currentIndexChanged.connect(self._on_fraction_changed)
        self.fraction_label = QtWidgets.QLabel('M<sub>w</sub> fraction:')
        form.addRow(self.fraction_label, self.fraction_combo)

        self.method_combo = QtWidgets.QComboBox()
        for label, key in _SLS_METHODS:
            self.method_combo.addItem(label, key)
        self.method_combo.setToolTip(
            'Zimm / Berry: thermodynamic Mw, Rg and A₂ (extrapolated to zero angle and '
            'zero concentration). Form factor P(q): fits the full angular form factor '
            '(sphere / Gaussian coil / ellipsoid), valid to larger qRg than the Zimm/Guinier '
            'expansion. Debye / Guinier / single-angle: apparent (one '
            'condition). Calibration-free 2·A₂·Mw needs no calibration; the absolute A₂ '
            'beneath it divides by Mw, so it is shown only when that Mw is on an absolute '
            'scale (calibrated, or hand-entered) and withheld otherwise. '
            'Excess Rayleigh ratio: ΔR vs angle for each concentration (arbitrary '
            'scale if uncalibrated) — the measured scattering itself, not a fitted '
            'Mw/Rg.')
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        form.addRow('Method:', self.method_combo)

        # Form-factor model chooser: shown only when the Form factor P(q) method is
        # selected (see _on_method_changed). Sphere returns a geometric radius R (+ Rg);
        # the Gaussian coil returns Rg; the ellipsoid returns Rg + an axial ratio and the
        # core-shell sphere returns two radii — both can yield MULTIPLE (degenerate) solutions.
        self.model_combo = QtWidgets.QComboBox()
        for label, key in _FORM_FACTOR_MODELS:
            self.model_combo.addItem(label, key)
        self.model_combo.setToolTip(
            'Sphere: solid homogeneous sphere P(q) — returns the geometric radius R and '
            'Rg. Gaussian coil (Debye): flexible-chain P(q) — returns Rg. Ellipsoid of '
            'revolution: fits Rg + an axial ratio (polar/equatorial); a prolate and an '
            'oblate ellipsoid can fit the same P(q), so it may return several solutions, '
            'all flagged degenerate, and the axial ratio is unresolved at low qRg. '
            'Mie sphere: the full Mie theory — the general case that works where the sphere '
            'form factor fails (high contrast, larger particles, e.g. latex in water); needs '
            'the solute refractive index (Data tab) and returns R and Rg. Core–shell sphere: '
            'a coated particle (core of one index inside a shell of another); needs the core '
            '(solute) AND shell refractive indices (Data tab) and fits two radii — two radii '
            'from one curve are often under-determined, so it may return several solutions, all '
            'flagged degenerate. The full form factor extends the valid qRg range beyond the '
            'Guinier/Zimm slope. See the Theory-and-Equations-Guide.')
        self.model_label = QtWidgets.QLabel('P(q) model:')
        form.addRow(self.model_label, self.model_combo)
        self.model_combo.currentIndexChanged.connect(self._on_method_changed)
        # Read-only solute-RI reminder, shown only for the Mie model (which forms the contrast
        # m = n_solute/n_solvent). Editing happens on the Data tab (invariant 5).
        self.mie_solute_ri_hint = ThemedLabel('', role='hint', size=10)
        self.mie_solute_ri_hint.setWordWrap(True)
        form.addRow('', self.mie_solute_ri_hint)
        self._mie_solute_ri_hint_label = form.labelForField(self.mie_solute_ri_hint)

        self.conc_combo = QtWidgets.QComboBox()
        form.addRow('Concentration:', self.conc_combo)
        self.angle_combo = QtWidgets.QComboBox()
        form.addRow('Angle:', self.angle_combo)

        # Zimm/Berry aesthetic spacing constant k (x = q² + k·c). It only spreads
        # the drawn concentration curves; Mw/Rg/A₂ come from the real q²,c fit and
        # are unchanged. Blank/Suggest = auto (q²max / cmax).
        k_row = QtWidgets.QHBoxLayout()
        k_row.setContentsMargins(0, 0, 0, 0)
        self.zimm_k_edit = QtWidgets.QLineEdit()
        self.zimm_k_edit.setPlaceholderText('auto')
        self.zimm_k_edit.setToolTip(
            'Spacing constant k in the Zimm x-axis q² + k·c, entered in the axis '
            'units (the active q² display unit, per mL/g — matching the k printed '
            'in the x-label). Purely aesthetic — it spreads the concentration '
            'curves and does NOT change Mw, Rg or A₂. Blank = auto.')
        self.zimm_k_suggest = QtWidgets.QPushButton('Suggest')
        self.zimm_k_suggest.clicked.connect(self._on_suggest_k)
        k_row.addWidget(self.zimm_k_edit)
        k_row.addWidget(self.zimm_k_suggest)
        k_widget = QtWidgets.QWidget(); k_widget.setLayout(k_row)
        form.addRow('Zimm spacing k:', k_widget)

        mw_row = QtWidgets.QHBoxLayout()
        self.mw_edit = QtWidgets.QLineEdit()
        self.mw_edit.setPlaceholderText('Mw (g/mol)')
        self.mw_edit.setToolTip('A manual Mw for the sample shown; the box clears when you '
                                'switch sample or fraction, so a value typed for one never '
                                'lands on another.')
        self.mw_button = QtWidgets.QPushButton('Set manual Mw')
        self.mw_button.clicked.connect(self._on_set_mw)
        mw_row.addWidget(self.mw_edit)
        mw_row.addWidget(self.mw_button)
        mw_widget = QtWidgets.QWidget()
        mw_widget.setLayout(mw_row)
        form.addRow('Manual M<sub>w</sub>:', mw_widget)
        self.mw_display = ThemedLabel('', role='muted')
        form.addRow('', self.mw_display)

        self.run_button = QtWidgets.QPushButton('Run')
        self.run_button.setToolTip(BACKGROUND_RUN_TOOLTIP)
        self.run_button.clicked.connect(self._on_run)
        form.addRow(self.run_button)
        self.export_button = QtWidgets.QPushButton('Export CSV…')
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._on_export)
        form.addRow(self.export_button)

        self._on_method_changed()
        return box

    def _build_mask_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox('Data Mask (Hide / Show)')
        add_help_to_groupbox(box, 'Hide points from the fit.', bullets=[
            'Untick an angle or concentration to hide all of its points.',
            'Or click a single point on the plot to hide/show just it.',
            'Hidden points show as hollow gray “masked” markers; the fit re-runs '
            'on the shown points.',
        ])
        v = QtWidgets.QVBoxLayout(box)
        lists = QtWidgets.QHBoxLayout()
        acol = QtWidgets.QVBoxLayout()
        acol.addWidget(QtWidgets.QLabel('Angles'))
        self.angle_list = QtWidgets.QListWidget()
        self.angle_list.setMinimumHeight(80)     # grows with its resizable pane
        self.angle_list.itemChanged.connect(self._on_angle_item_changed)
        acol.addWidget(self.angle_list)
        ccol = QtWidgets.QVBoxLayout()
        ccol.addWidget(QtWidgets.QLabel('Concentrations'))
        self.conc_list = QtWidgets.QListWidget()
        self.conc_list.setMinimumHeight(80)      # grows with its resizable pane
        self.conc_list.itemChanged.connect(self._on_conc_item_changed)
        ccol.addWidget(self.conc_list)
        lists.addLayout(acol)
        lists.addLayout(ccol)
        v.addLayout(lists)
        self.clear_mask_button = QtWidgets.QPushButton('Clear mask')
        self.clear_mask_button.clicked.connect(self._on_clear_masks)
        v.addWidget(self.clear_mask_button)
        hint = ThemedLabel('Untick to hide, or click a point on the plot.',
                           role='hint', size=11)
        hint.setWordWrap(True)
        v.addWidget(hint)
        return box

    # ------------------------------------------- anisotropy (Cabannes) ---
    @QtCore.Slot(bool)
    def _on_anisotropy_toggled(self, checked: bool) -> None:
        self.aniso_body.setVisible(checked)
        if not self._suppress_cal:
            self._refresh_anisotropy_display()

    @QtCore.Slot()
    def _on_anisotropy_mode_changed(self) -> None:
        self.aniso_stack.setCurrentIndex(self.aniso_mode_combo.currentIndex())
        if not self._suppress_cal:
            self._refresh_anisotropy_display()

    @QtCore.Slot()
    def _on_anisotropy_input_changed(self) -> None:
        if not self._suppress_cal:
            self._refresh_anisotropy_display()

    def _read_anisotropy_rho_v(self) -> Optional[float]:
        """The depolarization ratio ρv from the current inputs, or None if blank/invalid.
        The intensities path reuses the engine (dark-subtracted VH/VV ratio) so ρv is
        derived the one validated way; a non-positive VV−dark raises and reads as None."""
        mode = self.aniso_mode_combo.currentData()
        try:
            if mode == 'ratio':
                txt = self.aniso_rhov_edit.text().strip()
                return float(txt) if txt else None
            i_vv = float(self.aniso_ivv_edit.text().strip())
            i_vh = float(self.aniso_ivh_edit.text().strip())
            dark_txt = self.aniso_dark_edit.text().strip()
            dark = float(dark_txt) if dark_txt else 0.0
            res = self.controller.compute_depolarization(
                i_vv=i_vv, i_vh=i_vh, dark_count=dark)
            return res.rho_v
        except (ValueError, ZeroDivisionError):
            return None

    def _refresh_anisotropy_display(self, *, reset_inputs: bool = False) -> None:
        """Recompute the DISPLAY-ONLY readout from the entered ρv: ρv, δ², the Cabannes
        factor f, and the isotropic-corrected f·Mw for the current sample's analyzed Mw.
        Nothing is written to the sample or exported (physics via the controller; the GUI
        just shows it). ``reset_inputs=True`` (a sample switch) collapses + clears the
        calculator so a previous sample's entry can't linger."""
        has_sample = self.sample_id is not None
        self.aniso_check.setEnabled(has_sample)
        if reset_inputs:
            self._suppress_cal = True
            self.aniso_check.setChecked(False)
            self.aniso_body.setVisible(False)
            for e in (self.aniso_ivv_edit, self.aniso_ivh_edit,
                      self.aniso_dark_edit, self.aniso_rhov_edit):
                e.clear()
            self.aniso_mode_combo.setCurrentIndex(0)
            self.aniso_stack.setCurrentIndex(0)
            self._suppress_cal = False
        if not (has_sample and self.aniso_check.isChecked()):
            self.aniso_readout.clear()
            return
        rho_v = self._read_anisotropy_rho_v()
        sig = self.controller.settings.no_uncertainty_sig_figs
        if rho_v is None:
            self.aniso_readout.setRole('hint')
            self.aniso_readout.setText(
                'Enter ρ<sub>v</sub> (or VV/VH) to see the corrected M<sub>w</sub>.')
            return
        try:
            dep = self.controller.compute_depolarization(rho_v=rho_v)
        except ValueError:
            dep = None
        if dep is None or not dep.physically_valid:
            self.aniso_readout.setRole('error')
            self.aniso_readout.setText(
                f'⚠ ρ<sub>v</sub> = {rho_v:.3g} is outside the physical range [0, 0.75).')
            return
        parts = (f'ρ<sub>v</sub> = {format_fixed_sig(dep.rho_v, sig)},  δ² = '
                 f'{format_fixed_sig(dep.optical_anisotropy_sq, sig)},  Cabannes f = '
                 f'{format_fixed_sig(dep.cabannes_isotropic_factor, sig)}.  ')
        # cabannes_corrected_mw reads the sample's analyzed Mw (display-only; None until a
        # fit has run). Nothing is written back — the user enters it as a Manual Mw to use.
        corr = self.controller.cabannes_corrected_mw(
            self.sample_id, self._fraction, dep.cabannes_isotropic_factor)
        self.aniso_readout.setRole('hint')
        if corr is None:
            self.aniso_readout.setText(
                parts + 'Run an analysis to see the corrected M<sub>w</sub>.')
            return
        mw_app, mw_corr, mw_source = corr
        if self._mw_is_arbitrary_scale():
            # ρv, δ² and f are ratios and survive an uncalibrated run, so they stay; only
            # f·Mw inherits the arbitrary scale, and it is dashed at the value exactly as
            # _refresh_mw_display dashes the Mw it multiplies (rule 10 — same sample field,
            # same gate). Tier stays `hint`: the dash IS the disclosure, and escalating one
            # of the two Mw surfaces but not the other is the inconsistency being closed.
            self.aniso_readout.setText(
                parts + f'Corrected M<sub>w</sub> = f·M<sub>w</sub>: '
                f'{self._mw_scale_dash()}. This sample’s M<sub>w</sub> is on an '
                'arbitrary scale; supply what the dash names to see the corrected '
                'value.')
        else:
            self.aniso_readout.setText(
                parts + f'Corrected M<sub>w</sub> = f·M<sub>w</sub> = '
                f'{format_fixed_sig(mw_corr, sig)} g/mol '
                f'(from {format_fixed_sig(mw_app, sig)}) — display only; enter as a '
                'Manual M<sub>w</sub> to use it.')

    # ---------------------------------------------------------- selection ---
    def set_measurement(self, item_id: Optional[str]) -> None:
        """Follow the shell's active sample: always adopt the focused
        measurement's sample. If that sample has no SLS data, show a clean named
        empty-state (not the old keep-last) — no work is lost, since calibration/masks
        are controller-backed and the last run is cached per (sample, fraction).

        The clicked leaf's Mw fraction is honoured: focus lands on THAT fraction (not the
        first-sorted one), so clicking a "[1M]" measurement shows the 1M Zimm set, mirroring
        how the DLS tab derives its fraction from the focused item."""
        sid = self.controller.sample_id_of(item_id) if item_id is not None else None
        prefer = None
        if item_id is not None:
            lm = self.controller.workspace.measurements.get(item_id)
            if lm is not None:
                prefer = lm.committed_params.get('mw_fraction')
        self.show_active_sample(sid, prefer_fraction=prefer)

    def show_active_sample(self, sid: Optional[str], *,
                           prefer_fraction: Optional[str] = None) -> None:
        """Point the tab at the active sample `sid`. Mirrors the dropdown to it when the
        sample has SLS data, else shows a neutral placeholder there; `_load_sample`
        renders the analysis or the named "no SLS data" empty-state. `prefer_fraction`
        (the focused leaf's Mw fraction, when a specific measurement was clicked) selects
        that fraction if the sample carries it; otherwise the first-sorted one."""
        self.sls_selector.refresh()
        sample = self.controller.workspace.samples.get(sid) if sid else None
        if sample is not None and sample.has_sls:
            self.sls_selector.set_current_sample_id(sid)
        elif sid is not None:
            self.sls_selector.show_incompatible('(active sample has no SLS data)')
        self._load_sample(sid, prefer_fraction=prefer_fraction)

    @QtCore.Slot(str)
    def _on_sls_sample(self, sid: str) -> None:
        """The user picked a sample in the SLS dropdown → ask the shell to make it the
        single active sample (it fans back to `show_active_sample` + the sidebar)."""
        self.sampleFocusRequested.emit(sid or '')

    def _load_sample(self, sid: Optional[str], *,
                     prefer_fraction: Optional[str] = None) -> None:
        """Point the tab at `sid` (a sample id, or None) and rebuild its per-sample
        state (fraction/axis/mask/calibration + the restored last run). `prefer_fraction`
        seeds the fraction selector to the focused leaf's Mw fraction when the sample
        carries it (else the first-sorted one)."""
        self.sample_id = None
        self._run_epoch += 1             # drop any in-flight fit for the old sample
        self._clear_result_table()       # stale result clears on a sample switch
        self.mw_edit.clear()             # an unsubmitted manual Mw was typed for the
                                         # OLD sample (the Cross-Sample box's rule)
        runnable = False
        if sid is None:
            header = 'Pick a sample in the Workspace list (or the dropdown above).'
        else:
            sample = self.controller.workspace.samples.get(sid)
            if sample is None:
                header = 'Pick a sample in the Workspace list (or the dropdown above).'
            elif not sample.has_sls:
                # Named empty-state for an active sample with no SLS data (replaces the
                # old keep-last; nothing is lost, state is controller-backed).
                header = (f'No SLS data for <b>{_sample_label(sample)}</b> — load an SLS '
                          'intensity file, or pick another sample.')
            else:
                self.sample_id = sid
                runnable = True
                header = f'SLS sample: <b>{_sample_label(sample)}</b>'
        self._ran = False
        self._set_state(header, runnable)
        if runnable:
            # Keep a typed Zimm k in the active q² unit across a Settings unit
            # change (this path is the unit-change redraw). No-op when unchanged.
            self._reconcile_k_field_units()
            self._populate_fraction_combo(prefer=prefer_fraction)   # sets self._fraction
            self._populate_axis_selectors()
            self._populate_mask_lists()
            self._refresh_mw_display()
            self._on_method_changed()   # refresh the Mie solute-RI hint for the new sample
            # Redisplay the last analysis run on this (sample, fraction) from its cache
            # (no recompute); otherwise show the pre-run placeholder.
            if (self.sample_id, self._fraction) in self._last_run_by_sample:
                self._redisplay_cached(self.sample_id, self._fraction)
            else:
                self._clear_plot('Press Run to see the Zimm/Debye plot.')
        else:
            self._populate_fraction_combo()   # clears it
            self._populate_mask_lists()       # clears the lists
        # Calibration scope follows the selected sample.
        self._sync_calibration_scope()
        self._populate_calibration()
        self._refresh_calibration_display()
        self._refresh_anisotropy_display(reset_inputs=True)   # collapse the display-only check
        self.selectionChanged.emit()          # repaint the sidebar mirror

    def selected_item_ids(self) -> list:
        """The SLS measurements of the analyzed sample (sidebar-mirror contract): SLS
        works on a whole sample, so the mirror lights that sample's SLS rows + its
        solvent reference."""
        if self.sample_id is None:
            return []
        s = self.controller.workspace.samples.get(self.sample_id)
        if s is None:
            return []
        ids = list(s.sls_item_ids)
        if s.solvent_reference_item_id:
            ids.append(s.solvent_reference_item_id)
        return ids

    # ------------------------------------------------------------- mask ---
    def _populate_mask_lists(self) -> None:
        self._suppress_mask = True
        self.angle_list.clear()
        self.conc_list.clear()
        if self.sample_id is not None:
            frac = self._fraction
            mask = self.controller.sls_mask(self.sample_id, frac)
            for a in self.controller.sample_angles(self.sample_id, fraction=frac):
                it = QtWidgets.QListWidgetItem(f'{a:.0f}°')
                it.setData(QtCore.Qt.ItemDataRole.UserRole, a)
                it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(QtCore.Qt.CheckState.Unchecked
                                 if mask.is_angle_masked(a)
                                 else QtCore.Qt.CheckState.Checked)
                self.angle_list.addItem(it)
            for cc in self.controller.sample_concentrations(self.sample_id, fraction=frac):
                it = QtWidgets.QListWidgetItem(f'{cc * 1000:.4g} mg/mL')
                it.setData(QtCore.Qt.ItemDataRole.UserRole, cc)
                it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(QtCore.Qt.CheckState.Unchecked
                                 if mask.is_concentration_masked(cc)
                                 else QtCore.Qt.CheckState.Checked)
                self.conc_list.addItem(it)
        self._suppress_mask = False

    @QtCore.Slot('QListWidgetItem*')
    def _on_angle_item_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        if self._suppress_mask or self.sample_id is None:
            return
        if runner().is_busy:              # mask edits change what a running fit reads
            busy_notice(self)
            self._populate_mask_lists()   # revert the checkbox to committed state
            return
        a = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if item.checkState() == QtCore.Qt.CheckState.Checked:
            self.controller.unmask_angle(self.sample_id, a, self._fraction)
        else:
            self.controller.mask_angle(self.sample_id, a, self._fraction)
        self._rerun()

    @QtCore.Slot('QListWidgetItem*')
    def _on_conc_item_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        if self._suppress_mask or self.sample_id is None:
            return
        if runner().is_busy:
            busy_notice(self)
            self._populate_mask_lists()   # revert the checkbox to committed state
            return
        cc = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if item.checkState() == QtCore.Qt.CheckState.Checked:
            self.controller.unmask_concentration(self.sample_id, cc, self._fraction)
        else:
            self.controller.mask_concentration(self.sample_id, cc, self._fraction)
        self._rerun()

    @QtCore.Slot()
    def _on_clear_masks(self) -> None:
        if self.sample_id is None:
            return
        if runner().is_busy:
            busy_notice(self)
            return
        self.controller.clear_sls_mask(self.sample_id, self._fraction)
        self._populate_mask_lists()
        self._rerun()

    def _rerun(self) -> None:
        """Re-run the current analysis after a mask change (only once something
        has already been plotted, so toggling masks before a run is quiet)."""
        if self._ran and self._runnable and self.sample_id is not None:
            self._on_run()

    def _set_state(self, header_html: str, runnable: bool) -> None:
        self.header.setText(header_html)
        self._runnable = runnable
        self.run_button.setEnabled(runnable)
        self.bottom_holder.setVisible(runnable)   # R8.2: the results / mask chrome only with data to fill it
        # A stale soft-flag must never survive a sample/state switch (like the result
        # table). QLabel.clear() drops only the text, so clear the tooltip too —
        # otherwise the calibration-free conservative-SE tooltip could linger on an
        # unrelated sample. A subsequent run re-sets both in _present_method.
        self.flag_label.clear()
        self.flag_label.setToolTip('')
        self.flag_label.setRole('error')   # reset the tier so a stale qualifier
        self.flag_label.setBold(True)      # color never survives a state switch
        set_flag(self.notes_label, '', problem=False)   # drop stale result notes
        # The Export button + closure are bound to a specific (sample, fraction)'s
        # result. Reset them on EVERY state switch, not only the not-runnable branch:
        # switching to a runnable-but-never-run sample/fraction must not leave the
        # button enabled over a blank plot, firing the PREVIOUS sample's export.
        # A cache redisplay or a fresh run re-enables it (_redisplay_cached / done).
        self._disable_export()
        if not runnable:
            self.status.clear()
            self._clear_plot('Load SLS intensities to see the Zimm/Debye plot.')

    def _disable_export(self) -> None:
        """Unbind the Export button from any prior result (no result is on screen).
        The redisplay/run paths re-enable it once a result is drawn."""
        self.export_button.setEnabled(False)
        self._export = None

    # ------------------------------------------------------------ fraction ---
    def _populate_fraction_combo(self, prefer: Optional[str] = None) -> None:
        """Fill the Mw-fraction selector from the sample's distinct labels and set
        the active fraction. Hidden when a sample has a single (unlabeled) one. `prefer`
        (the focused leaf's Mw fraction) is selected when the sample carries it, so
        clicking a specific fraction's measurement focuses THAT fraction; otherwise the
        first-sorted label."""
        self._suppress_fraction = True
        self.fraction_combo.clear()
        fractions = ([None] if self.sample_id is None
                     else self.controller.sample_fractions(self.sample_id))
        for frac in fractions:
            self.fraction_combo.addItem('(unlabeled)' if frac is None else frac, frac)
        idx = fractions.index(prefer) if prefer in fractions else 0
        self._fraction = fractions[idx]
        self.fraction_combo.setCurrentIndex(idx)     # keep the visible combo in sync
        multi = len(fractions) > 1
        self.fraction_combo.setVisible(multi)
        self.fraction_label.setVisible(multi)
        self._suppress_fraction = False

    def _current_fraction(self) -> Optional[str]:
        return self._fraction

    @QtCore.Slot()
    def _on_fraction_changed(self) -> None:
        if self._suppress_fraction or self.sample_id is None:
            return
        self._fraction = self.fraction_combo.currentData()
        self._ran = False
        self._run_epoch += 1             # drop any in-flight fit for the old fraction
        self._disable_export()           # unbind the prior fraction's export;
                                         # a cache redisplay below re-enables it
        self.mw_edit.clear()             # an unsubmitted manual Mw was for the OLD fraction
        self._populate_axis_selectors()
        self._populate_mask_lists()
        self._refresh_mw_display()
        self._on_method_changed()   # refresh the Mie solute-RI hint for the new fraction
        if (self.sample_id, self._fraction) in self._last_run_by_sample:
            self._redisplay_cached(self.sample_id, self._fraction)
        else:
            self._clear_plot('Press Run to see the Zimm/Debye plot.')

    def _populate_axis_selectors(self) -> None:
        frac = self._fraction
        self.conc_combo.clear()
        for c in self.controller.sample_concentrations(self.sample_id, fraction=frac):
            self.conc_combo.addItem(f'{c * 1000:.4g} mg/mL', c)
        self.angle_combo.clear()
        for a in self.controller.sample_angles(self.sample_id, fraction=frac):
            self.angle_combo.addItem(f'{a:.0f}°', a)

    # ---------------------------------------------------------- calibration ---
    def _populate_calibration(self) -> None:
        cw = self.controller.calibration_fields(self.sample_id)
        self._suppress_cal = True
        for key, edit in self.cal_edits.items():
            v = cw.get(key)
            edit.setText('' if v is None else str(v))
        self.geometry_combo.setCurrentText(cw.get('standard_geometry', 'VU'))
        # The dot rides the stored provenance tag, never the value (a typed value that
        # equals the library's is still the user's). Re-rendered here so its colour
        # tracks a theme switch, as the Data tab's legend does.
        proposed = (cw.get('standard_refractive_index_source') == 'library'
                    and cw.get('standard_refractive_index') is not None)
        teal = theme_token(self, 'lib_primary')
        self.std_index_dot.setText(f'<span style="color:{teal}">●</span>')
        self.std_index_dot.setVisible(proposed)
        self._suppress_cal = False

    def _sync_calibration_scope(self) -> None:
        """Set the per-sample checkbox from the controller, without re-triggering it."""
        self._suppress_cal = True
        has = (self.sample_id is not None
               and self.controller.has_sample_calibration(self.sample_id))
        self.per_sample_check.setEnabled(self.sample_id is not None)
        self.per_sample_check.setChecked(has)
        self._suppress_cal = False

    @QtCore.Slot(bool)
    def _on_per_sample_toggled(self, checked: bool) -> None:
        if self._suppress_cal or self.sample_id is None:
            return
        if runner().is_busy:              # disable_sample_calibration pops the
            busy_notice(self)             # COMMITTED dict a running fit reads
            self._sync_calibration_scope()   # revert the checkbox to committed state
            return
        if checked:
            self.controller.enable_sample_calibration(self.sample_id)
        else:
            self.controller.disable_sample_calibration(self.sample_id)
        self._populate_calibration()
        self._refresh_calibration_display()

    @QtCore.Slot()
    def _on_cal_edit(self, key: str) -> None:
        if self._suppress_cal:
            return
        edit = self.cal_edits[key]
        if not edit.isModified():
            # `editingFinished` fires on a bare focus-out as well as on a typed edit.
            # For the standard refractive index that no-op write would claim the field
            # as the user's and freeze the library re-proposal (invariant 3), so a Tab
            # through the form would silently change what the program does; for every
            # other field it is a redundant write. Qt's own modified flag is the
            # distinction the Data tab draws too -- only a real typed value claims a
            # field -- and `_populate_calibration`'s setText clears it, so a value the
            # PROGRAM put there never reads as typed.
            return
        edit.setModified(False)
        allow_blank = dict((k, b) for k, _l, b in _CAL_FIELDS)[key]
        text = edit.text().strip()
        if text == '':
            if not allow_blank:
                return
            value = None
        else:
            try:
                value = float(text)
            except ValueError:
                self.status.setText(f'{key}: "{text}" is not a number.')
                return
        self.controller.set_calibration_field(key, value, self.sample_id)
        if key in ('standard_wavelength_nm', 'standard_temperature_C',
                   'standard_refractive_index'):
            # The controller re-proposes the standard's refractive index from the
            # solvent library at the new condition (unless the user typed their own),
            # and refills a cleared index from the library; repopulate so the proposed
            # value and its provenance dot show in the field. Safe here: this slot
            # fires on editingFinished, not per keystroke.
            self._populate_calibration()
        self._refresh_calibration_display()

    @QtCore.Slot(str)
    def _on_geometry_changed(self, text: str) -> None:
        if self._suppress_cal:
            return
        self.controller.set_calibration_field('standard_geometry', text, self.sample_id)
        self._refresh_calibration_display()

    def _refresh_calibration_display(self) -> None:
        sid = self.sample_id
        scope = ('per-sample' if (sid is not None
                 and self.controller.has_sample_calibration(sid)) else 'session')
        set_flag(self.cal_note,
                 self.controller.preview_calibration_note(sid) or '', problem=False)
        try:
            preview = self.controller.preview_k_c(sid)
        except Exception as exc:
            self.kc_label.setText(span(self, 'error', f'cannot compute: {exc}'))
            return
        committed = self.controller.committed_k_c(sid)
        if preview is None:
            self.kc_label.setText(f'[{scope}] uncalibrated (enter a calibrant '
                                  'intensity) — Mw/A₂ will be flagged unreliable')
        else:
            pending = (committed is None or abs((preview or 0) - (committed or 0))
                       > 1e-30 * max(1.0, abs(preview)))
            note = '  ● pending — press Update' if pending else ''
            self.kc_label.setText(
                f'[{scope}] preview = {preview:.4e}; committed = '
                f'{("none" if committed is None else f"{committed:.4e}")}{note}')

    @QtCore.Slot()
    def _on_apply(self) -> None:
        if runner().is_busy:              # commit changes committed calibration a
            busy_notice(self)             # background fit is reading
            return
        # A commit adopts EVERY pending edit, including a Data-tab identity edit
        # left uncommitted -- so this path can regroup samples too, and it owes
        # the same ask every committing surface gives (one shared confirm;
        # Cancel applies nothing).
        if not confirm_regroup_losses(self, self.controller,
                                      self.controller.commit_regroup_losses()):
            return
        sample = (self.controller.workspace.samples.get(self.sample_id)
                  if self.sample_id is not None else None)
        item_ids = list(sample.sls_item_ids) if sample is not None else []
        self.controller.commit()
        if self.sample_id is not None and \
                self.sample_id not in self.controller.workspace.samples:
            # The commit adopted an identity edit that re-keyed THIS sample, so
            # its id is dead: re-point at the sample now holding the same SLS
            # measurements before anything dereferences it. The shell re-fans
            # the focus on `committed` as well; this keeps the tab consistent
            # on its own (the refreshes below all read `self.sample_id`).
            successor = next((s for s in map(self.controller.sample_id_of, item_ids)
                              if s is not None), None)
            self.show_active_sample(successor, prefer_fraction=self._fraction)
            self.committed.emit()
            return
        self.sls_selector.refresh()       # a commit can rename/regroup the sample
        self._populate_calibration()
        self._refresh_calibration_display()
        if self.sample_id is not None:
            self._refresh_mw_display()
        self.committed.emit()

    # --------------------------------------------------------- manual Mw ---
    @QtCore.Slot()
    def _on_set_mw(self) -> None:
        if self.sample_id is None:
            return
        if runner().is_busy:              # writes SampleResult.mw a fit may read
            busy_notice(self)
            return
        text = self.mw_edit.text().strip()
        if not text:
            return
        try:
            mw = float(text)
        except ValueError:
            QtWidgets.QMessageBox.warning(self, 'Invalid Mw',
                                          f'"{text}" is not a number.')
            return
        # The setter validates too (positive AND finite), and its ValueError has to be
        # caught HERE: the try above covers only float(text), which succeeds for '0',
        # '-1' and '1e400'. Without this the refusal escapes into the Qt slot as an
        # unhandled exception -- a crash on a plain typo.
        try:
            self.controller.set_manual_mw(self.sample_id, mw, self._fraction)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, 'Invalid Mw', str(exc))
            return
        self._refresh_mw_display()

    def _mw_is_arbitrary_scale(self) -> bool:
        """Is the active sample's stored M<sub>w</sub> on an arbitrary (uncalibrated) scale?

        The one gate behind every place this tab prints — or declines to print, or divides
        by — an absolute M<sub>w</sub>: the M<sub>w</sub> readout, the Cabannes
        f·M<sub>w</sub> line and the calibration-free A₂ all ask here, so they can never
        disagree about the same number (rule 10). Thin wrapper over
        `SampleResult.mw_is_arbitrary_scale` — the one predicate body, shared with the
        controller and Cross-Sample — adding only the no-sample
        guard this tab needs."""
        if self.sample_id is None:
            return False
        r = self.controller.workspace.samples[self.sample_id].result_for(self._fraction)
        return r.mw_is_arbitrary_scale()

    def _mw_scale_dash(self) -> str:
        """The dash for the STORED M<sub>w</sub>, naming why it is not printed as a number.

        `_scale_causes`' stored-value counterpart: `SampleResult` carries the same two
        flags under its own spellings. Named causes matter more here than on a fit
        readout, because both surfaces that use this also print a REMEDY, and
        “calibrate” is the wrong instruction when the missing thing is a dn/dc.

        THREE causes, not two. The caller dashes on a broader question than "is the scale
        unknown?" — `_refresh_mw_display` also dashes a non-finite Mw, which
        `is_arbitrary_scale` deliberately calls NOT arbitrary (there is no number there to be
        on a scale). A Zimm fit whose intercept goes non-positive stores exactly that: NaN,
        on a run whose calibration is fine. Without the third term this fell through to
        `_UNCAL_MW` and blamed calibration, sending the user to redo work that would not
        change the reading.
        """
        if self.sample_id is None:
            return _UNCAL_MW
        r = self.controller.workspace.samples[self.sample_id].result_for(self._fraction)
        causes = []
        if r.calibrated is False:
            causes.append('uncalibrated')
        if r.dn_dc_supplied is False:
            causes.append('no usable dn/dc')
        if r.mw_g_per_mol is not None and not math.isfinite(r.mw_g_per_mol):
            causes.append('no usable value')
        return f"— ({', '.join(causes)})" if causes else _UNCAL_MW

    def _refresh_mw_display(self) -> None:
        if self.sample_id is None:
            self.mw_display.setText('')
            return
        r = self.controller.workspace.samples[self.sample_id].result_for(self._fraction)
        if r.mw_g_per_mol is None:
            self.mw_display.setText('M<sub>w</sub>: not yet determined')
        elif self._mw_is_arbitrary_scale() or not math.isfinite(r.mw_g_per_mol):
            # The question here is "may I print this as an absolute molar mass?", which is
            # broader than "is the scale unknown?". `is_arbitrary_scale` answers False for a
            # non-finite value — there is no number there to be on a scale — so a nan/inf Mw
            # needs its own term or it prints as `nan g/mol`, losing the very verdict this
            # readout exists to give.
            # A computed/picked Mw from an uncalibrated run is on an arbitrary scale —
            # don't present it as a real number. A hand-entered ('user')
            # Mw is a trusted value and is always shown.
            self.mw_display.setText(f'M<sub>w</sub>: {self._mw_scale_dash()}')
        else:
            self.mw_display.setText(
                f'M<sub>w</sub> = {r.mw_g_per_mol:.3e} g/mol  ({r.mw_source})')

    # ------------------------------------------------------------- run ---
    @QtCore.Slot()
    def _on_method_changed(self) -> None:
        key = self.method_combo.currentData()
        self.conc_combo.setEnabled(key in ('debye', 'guinier', 'single'))
        self.angle_combo.setEnabled(key in ('single', 'calfree'))
        zimm = key in ('zimm', 'berry')
        self.zimm_k_edit.setEnabled(zimm)
        self.zimm_k_suggest.setEnabled(zimm)
        # The P(q) model chooser is only meaningful for the form-factor fit.
        form_factor = key == 'form_factor'
        self.model_label.setVisible(form_factor)
        self.model_combo.setVisible(form_factor)
        # The refractive-index reminder is shown for the models that need a hand-entered
        # contrast: Mie (one solute RI) and core-shell (core + shell RIs). Both refresh here,
        # which is called on sample AND fraction switch (rule 10 -- no stale hint).
        model_data = self.model_combo.currentData()
        is_mie = form_factor and model_data == ff.MIE
        is_core_shell = form_factor and model_data == ff.CORE_SHELL
        needs_ri = is_mie or is_core_shell
        self.mie_solute_ri_hint.setVisible(needs_ri)
        if self._mie_solute_ri_hint_label is not None:
            self._mie_solute_ri_hint_label.setVisible(needs_ri)
        if is_mie:
            self.mie_solute_ri_hint.setText(self._mie_solute_ri_summary())
        elif is_core_shell:
            self.mie_solute_ri_hint.setText(self._core_shell_ri_summary())

    def _mie_solute_ri_summary(self) -> str:
        """Read-only reminder of the current sample's committed solute RI (edited on the
        Data tab, invariant 5) — the Mie fit forms m = n_solute/n_solvent from it. Names the
        value when set, or prompts when unset (it is never guessed — invariant 3)."""
        sid = self.sample_id
        if sid is None:
            return 'Mie needs the solute refractive index — set it on the Data tab.'
        n = self.controller.sls_solute_refractive_index(sid, self._current_fraction())
        if n is None:
            return ('Solute refractive index not set — enter it on the Data tab '
                    '(the Mie model needs it to form the contrast m; it is never guessed).')
        return f'Solute refractive index: {n:g} (from the Data tab).'

    def _core_shell_ri_summary(self) -> str:
        """Read-only reminder of the current sample's committed CORE and SHELL refractive
        indices (edited on the Data tab, invariant 5) — the core-shell fit forms the two
        contrasts m1 = n_core/n_solvent, m2 = n_shell/n_solvent from them. Names both when set,
        or prompts for the missing one (neither is ever guessed — invariant 3)."""
        sid = self.sample_id
        if sid is None:
            return ('Core-shell needs the core and shell refractive indices — set them on '
                    'the Data tab.')
        frac = self._current_fraction()
        n_core = self.controller.sls_solute_refractive_index(sid, frac)
        n_shell = self.controller.sls_shell_refractive_index(sid, frac)
        missing = [name for name, v in (('core', n_core), ('shell', n_shell)) if v is None]
        if missing:
            return (f'{" and ".join(missing).capitalize()} refractive index not set — enter '
                    'it on the Data tab (the core-shell model needs both the core '
                    '[solute RI] and shell indices; neither is ever guessed).')
        return (f'Core refractive index: {n_core:g}, shell: {n_shell:g} (from the Data tab).')

    def _reconcile_k_field_units(self) -> None:
        """Rewrite the Zimm-k field text into the ACTIVE q² display unit when it
        changed (a Settings ▸ Display units switch), mirroring how the axis-limit
        bar's typed limits re-render — so a user-typed k stays the same physical
        constant across a unit change instead of being reinterpreted against the
        new factor (which would rescale the grid). A no-op when the unit is
        unchanged, so it is safe to call on every (re)display. Converts relative
        to the field's PREVIOUS factor (`_k_field_qfac`), since by redisplay time
        `set_plot_units` has already swapped the global factor and the old one
        can no longer be recovered from `app.units`."""
        new_qfac = display_factor_from('scattering_q2', 'nm⁻²')
        old_qfac = self._k_field_qfac
        if new_qfac == old_qfac:
            return
        text = self.zimm_k_edit.text().strip()
        if text:
            try:
                shown = float(text)
            except ValueError:
                shown = None
            if shown is not None:
                # value is in old axis units; canonical = shown/old, new = canonical*new.
                with QtCore.QSignalBlocker(self.zimm_k_edit):
                    self.zimm_k_edit.setText(f'{shown / old_qfac * new_qfac:.4g}')
        self._k_field_qfac = new_qfac

    def _current_zimm_k(self, rr) -> float:
        """The spacing constant for the Zimm plot, returned CANONICAL
        (nm⁻²·mL/g): the user's typed value if entered and valid — typed in
        the AXIS units (the active q² display unit per mL/g, matching the k
        printed in the x-label, like the axis-limit bar's typed limits) —
        else the auto value (q²max / cmax). On the default nm⁻² choice the
        conversion factor is exactly 1.0. The field text is kept in the active
        unit by `_reconcile_k_field_units` across a Settings unit change, so the
        divide here always uses the factor the text was written in."""
        text = self.zimm_k_edit.text().strip()
        if text:
            try:
                return float(text) / display_factor_from('scattering_q2', 'nm⁻²')
            except ValueError:
                pass
        return zimm_spacing(rr)

    @QtCore.Slot()
    def _on_suggest_k(self) -> None:
        if self.sample_id is None:
            return
        if runner().is_busy:              # builds a Rayleigh series off the run path
            busy_notice(self)
            return
        try:
            rr = self.controller.masked_rayleigh(self.sample_id, self._fraction)
        except Exception:
            return
        # Suggest fills the edit in the AXIS units (matching how a typed k is
        # read back); canonical at the default nm⁻² choice, factor exactly 1.0.
        # Record the unit the field is now written in, so a later unit change
        # reconciles from the right factor.
        qfac = display_factor_from('scattering_q2', 'nm⁻²')
        self.zimm_k_edit.setText(f'{zimm_spacing(rr) * qfac:.4g}')
        self._k_field_qfac = qfac

    @QtCore.Slot()
    def _on_run(self) -> None:
        if not self._runnable or self.sample_id is None:
            return
        sid = self.sample_id
        samp = self.controller.workspace.samples[sid]
        if samp.solvent_reference_item_id is None:
            QtWidgets.QMessageBox.warning(
                self, 'No solvent reference',
                'This sample has no c = 0 solvent reference. SLS calibration and '
                'the excess Rayleigh ratio require one.')
            return
        # Read everything the fit needs on the main thread; the compute phase
        # runs on the worker (the SLS fits build + fit every angle/concentration),
        # then the present phase plots + fills tables back on the main thread.
        method = self.method_combo.currentData()
        method_text = self.method_combo.currentText()
        frac = self._fraction
        conc = self.conc_combo.currentData()
        angle = self.angle_combo.currentData()
        model = self.model_combo.currentData()   # form-factor P(q) model (read on main thread)

        # An explicit Run replaces an explicitly picked value (tier rule:
        # SampleResult docstring). Resolve which picks are at stake BEFORE the
        # dispatch — one query behind the optional ask-first confirm and the
        # always-on disclose-after line, so they cannot disagree.
        replaced_picks = []
        if method in ('zimm', 'berry'):     # the one SLS path that writes sample values
            replaced_picks = self.controller.picks_a_run_would_replace(
                sid, frac, ('mw', 'rg', 'a2'))
        if replaced_picks and self.controller.settings.confirm_run_replaces_pick:
            details = '\n'.join(f'  • {_QTY_DISPLAY[q]}: {lbl}'
                                for q, lbl in replaced_picks)
            resp = QtWidgets.QMessageBox.question(
                self, 'Replace picked value(s)?',
                f'Running {method_text} will replace the value(s) you picked in '
                f'Cross-Sample:\n{details}\n\nCancel keeps the pick(s) and runs '
                f'nothing.\n\nContinue?',
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Cancel)
            if resp != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        self._run_epoch += 1
        epoch = self._run_epoch

        def _picks_replaced_by_this_run():
            """Truth from state, not from intent: a pick counts as replaced only if
            its tier actually moved — so a compute-phase failure (nothing written)
            discloses nothing, while a present-phase failure after a successful
            write still does."""
            r = self.controller.workspace.samples[sid].result_for(frac)
            return [(q, lbl) for q, lbl in replaced_picks
                    if getattr(r, f'{q}_source') != 'picked']

        def fail(exc: BaseException) -> None:
            # A failure landing after a sample/fraction switch must not act on
            # the view now shown (its Export belongs to the fit redisplayed
            # there) -- the same epoch guard `done` applies. The dialog names
            # the sample either way, so a late one cannot be misread.
            fresh = epoch == self._run_epoch
            who = _sample_label(self.controller.workspace.samples[sid])
            if fresh:
                self.export_button.setEnabled(False)
            QtWidgets.QMessageBox.critical(
                self, 'Analysis failed',
                f'Could not run {method_text!r} on {who}.\n\n'
                f'{humanize_field_names(str(exc))}\n\n'
                'Confirm parameters (Data tab) and the calibration, then Update.')
            gone = _picks_replaced_by_this_run()
            if gone:
                names = ', '.join(_QTY_DISPLAY[q] for q, _ in gone)
                self.status.setText(f'Analysis failed — see dialog. The run had '
                                    f'already replaced your picked {names} on {who}.')
            elif fresh:
                self.status.setText('Analysis failed — see dialog.')

        def done(payload) -> None:
            # Disclose-after is NOT view-gated: the worker's write happened
            # whether or not the user switched samples mid-fit, and a disclosure
            # dropped by the stale-view guard would be the silent replacement
            # the tier rule forbids (SampleResult docstring). Composed once;
            # written on BOTH arms — the stale arm immediately, the fresh arm
            # after presentation (`_fill_result_table` ends with
            # `status.clear()`, wiping an early write). It names the sample, so
            # a line landing after a switch cannot be misread on the new view.
            gone = _picks_replaced_by_this_run()
            disclosure = ''
            if gone:
                names = ', '.join(_QTY_DISPLAY[q] for q, _ in gone)
                who = _sample_label(self.controller.workspace.samples[sid])
                disclosure = (f'{method_text} on {who} replaced your picked '
                              f'{names} (re-pick in Cross-Sample if needed).')
            if epoch != self._run_epoch:
                if disclosure:
                    self.status.setText(disclosure)
                return          # sample/fraction changed — stale VIEW: skip only
                                # the presentation and the redisplay cache below
            # Only the current view's series is ever read (by the overlay/click
            # hit-test), so keep a single entry rather than accumulating one per
            # (sample, fraction) for the life of the session.
            self._full_rr = {(sid, frac): payload['full_rr']}
            # The present phase (plot + tables) can also raise on a degenerate
            # result; route it through the same 'Analysis failed' path the old
            # single try/except gave, so a plotting edge case never escapes as an
            # uncaught slot exception with the UI half-updated.
            try:
                self._present_method(method, sid, conc, payload)
            except Exception as exc:         # noqa: BLE001
                fail(exc)
                return
            self._ran = True
            self.export_button.setEnabled(self._export is not None)
            # Cache the full payload + a signature so a sample/fraction switch or a
            # commit can REDISPLAY this fit without recomputing (explicit-Run parity
            # with the DLS tabs); the signature drives the
            # staleness hint on redisplay.
            self._last_run_by_sample[(sid, frac)] = {
                'method': method, 'conc': conc, 'angle': angle, 'model': model,
                'payload': payload,
                'sig': self.controller.sls_run_signature(sid, frac)}
            # The fresh-arm write, after presentation's own status.clear(). Never
            # in _present_method or the cached payload: a redisplay replays the
            # payload verbatim, and a replayed "replaced your pick" would be a
            # stale claim about a past event.
            if disclosure:
                self.status.setText(disclosure)

        if runner().try_submit(
                lambda: self._compute_method(method, sid, frac, conc, angle, model),
                done, fail, description=f'{method_text} fit',
                busy_widgets=(self.run_button,)):
            self.status.setText('Running in the background…')
        else:
            self.status.setText(BUSY_NOTICE)

    def _compute_method(self, method: str, sid: str, frac, conc, angle, model=None) -> Dict:
        """Worker phase: every controller call the chosen method needs, plus the
        unmasked Rayleigh series for the overlay/click cache. No Qt, no plotting."""
        c = self.controller
        payload: Dict = {'full_rr': c.run_rayleigh(sid, frac)}
        if method in ('zimm', 'berry'):
            payload['rr'] = c.masked_rayleigh(sid, frac)
            payload['res'] = c.run_zimm(sid, method, frac)
        elif method == 'form_factor':
            payload['res'] = c.run_form_factor(sid, model, frac)
        elif method == 'debye':
            payload['res'] = c.run_debye(sid, conc, frac)
        elif method == 'guinier':
            payload['res'] = c.run_guinier(sid, conc, fraction=frac)
        elif method == 'single':
            payload['res'] = c.run_single_angle(sid, conc, angle, frac)
        elif method == 'calfree':
            payload['res'] = c.run_calibration_free_a2(sid, angle, fraction=frac)
        else:  # rayleigh
            payload['rr'] = c.masked_rayleigh(sid, frac)
        return payload

    def _stash_export(self, sid: str, analysis: str, do_export) -> None:
        """Bind the Export button to the result on screen. One choke point, so
        the suggested filename (house `{sample}_{analysis}.csv` scheme) and the
        plot-save stem can never drift apart."""
        name = sample_suggested_name(self.controller, sid, analysis)
        self._export = (name, do_export)
        set_plot_export_name(self.canvas, name)

    @QtCore.Slot()
    def _on_export(self) -> None:
        if self._export is None:
            return
        default_name, do_export = self._export
        status = export_to_csv(self, self.controller, default_name, do_export)
        if status:
            self.status.setText(status)

    @staticmethod
    def _set_combo_data(combo: QtWidgets.QComboBox, value) -> None:
        """Select the item whose stored data == value (no-op if not present)."""
        idx = combo.findData(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _redisplay_cached(self, sid: str, fraction: Optional[str]) -> None:
        """REDISPLAY this (sample, fraction)'s last fit from its cached payload — no
        recompute (explicit-Run parity with the DLS tabs). Restores
        the method/axis combos and replots, then flags the result stale if the committed
        params / mask / calibration / estimator have changed since it ran. The only
        recompute paths are the Run button and a mask edit (cheap; via `_rerun`)."""
        cached = self._last_run_by_sample.get((sid, fraction))
        if cached is None:
            self._clear_plot()
            return
        method, conc = cached['method'], cached['conc']
        self._set_combo_data(self.method_combo, method)
        self._set_combo_data(self.conc_combo, conc)
        self._set_combo_data(self.angle_combo, cached['angle'])
        if cached.get('model') is not None:
            self._set_combo_data(self.model_combo, cached['model'])
        payload = cached['payload']
        self._full_rr = {(sid, fraction): payload['full_rr']}
        # Clear the status FIRST so the stale hint can't accumulate across repeated
        # redisplays (most methods leave self.status untouched in _present_method, so a
        # leftover "Running…"/prior-hint string would otherwise be the append base).
        self.status.clear()
        try:
            self._present_method(method, sid, conc, payload)
        except Exception:                    # noqa: BLE001
            # A cached payload that no longer plots (rare) — clear rather than crash.
            self._clear_plot()
            self.export_button.setEnabled(False)
            self.status.setText('Could not redisplay the last fit — press Run.')
            return
        self._ran = True
        self.export_button.setEnabled(self._export is not None)
        # Staleness: append the hint (on the now-clean status base, or the one method —
        # rayleigh — that sets its own note) when a re-run would differ.
        if self.controller.sls_run_signature(sid, fraction) != cached['sig']:
            base = self.status.text()
            self.status.setText(f'{base}  —  {_SLS_STALE_HINT}' if base
                                else _SLS_STALE_HINT)

    def _present_ellipsoid(self, sid: str, results, sig: int) -> None:
        """Display the ellipsoid P(q) fit -- a LIST of solutions (a prolate and an oblate
        ellipsoid can reproduce the same P(q)). One results-table block per solution plus a
        degeneracy verdict; every solution is shown and none is silently preferred (the same
        structural-degeneracy contract as the DPLS shape display)."""
        c = self.controller
        first = results[0]
        self._stash_export(sid, 'form_factor_ellipsoid',
                           lambda p: c.export_form_factor(results, p))
        self._setup_axes()
        regime = 'apparent' if first.is_apparent else 'thermodynamic'
        self.ax.set_title(f'Form factor P(q): Ellipsoid ({regime})')

        if not first.converged:
            self._fill_result_table([('Ellipsoid', 'no fit')])
            self._set_flag('No ellipsoid of revolution reproduces the angular P(q) — try '
                           'the sphere or Gaussian-coil model.', problem=True)
            return

        plot_ellipsoid_form_factor(results, ax=self.ax)
        mw_label = 'M<sub>w,app</sub>' if first.is_apparent else 'M<sub>w</sub>'
        rows = []
        for i, r in enumerate(results, 1):
            kind = 'prolate' if r.is_prolate else 'oblate'
            tag = f'S{i} ({kind})' if len(results) > 1 else kind.capitalize()
            if r.axial_ratio_reliable and r.axial_ratio_se is not None:
                p_val = format_pm(r.axial_ratio, r.axial_ratio_se)
            else:
                p_val = f'{format_fixed_sig(r.axial_ratio, sig)} (unresolved)'
            mw_val = _scaled_value(r, format_pm(r.mw_g_per_mol, r.mw_se))
            rows.append((f'{tag}: axial ratio p', p_val))
            rows.append((f'{tag}: R<sub>g</sub> (nm)', format_pm(r.rg_nm, r.rg_se)))
            rows.append((f'{tag}: {mw_label} (g/mol)', mw_val))
            if not first.is_apparent:
                a2_val = _scaled_value(r, format_pm(r.a2_mol_mL_per_g2, r.a2_se))
                rows.append((f'{tag}: A₂ (mol·mL/g²)', a2_val))
            # qRg(max) and R² are PER-SOLUTION values (each solution has its own Rg and
            # its own fit quality), so with several solutions they live in each S{i}
            # block -- an unlabeled trailing pair would silently show whichever solution
            # happens to list first (focused-review 2026-07-23; the list order is a
            # presentation convention, not fit preference).
            if len(results) > 1:
                rows.append((f'{tag}: qR<sub>g</sub>(max)',
                             format_fixed_sig(r.qrg_max, sig)))
                rows.append((f'{tag}: R²', format_fixed_sig(r.r_squared, sig)))
        if len(results) == 1:
            rows.append(('qR<sub>g</sub>(max)', format_fixed_sig(first.qrg_max, sig)))
            rows.append(('R²', format_fixed_sig(first.r_squared, sig)))
        self._fill_result_table(rows)

        # Degeneracy verdict leads (the headline caveat of a shape fit); calibration and
        # Mw reliability are the genuine red problems. Unresolved axial ratio + degeneracy
        # are neutral structural notes (calmer-warnings tier).
        if first.degenerate:
            flag = (f'The shape fit is degenerate — {len(results)} ellipsoids '
                    '(prolate/oblate) reproduce this P(q) equally; all shown, none preferred.')
        else:
            flag = 'One ellipsoid of revolution reproduces this P(q).'
        flag += ('  Apparent (single concentration): the 2A₂c term folds into Mw.'
                 if first.is_apparent else
                 '  Thermodynamic across angle and concentration.')
        # the estimator travels with the ± it qualifies (invariant 8 clause A),
        # in the DDLS tab's words
        flag += _STAT_CAVEAT + _ols_marker(getattr(first, 'se_estimator', 'hc3'))
        rg_cautions = [c for c in (_rg_unreliable_caution(r) for r in results) if c]
        if _scale_causes(first):
            # "(Rg/shape survive)" is an upper-tail claim -- see the sphere presenter.
            flag += ('  ' + _scale_phrase(first).capitalize() + ' → Mw'
                     + ('' if first.is_apparent else '/A₂')
                     + (' unreliable (Rg/shape survive).' if not rg_cautions
                        else ' unreliable.'))
        # Surface the REASON a calibrated fit's Mw is unreliable (e.g. past the sphere
        # first-minimum proxy) — the same below-fold caution every sibling SLS method
        # appends, so a red flag never reads as "problem" with no explanation.
        cautions = [c for c in (_mw_unreliable_caution(r) for r in results) if c]
        if cautions:
            flag += '  ' + cautions[0]
        if rg_cautions:
            flag += '  ' + rg_cautions[0]
        if any(not r.axial_ratio_reliable for r in results):
            flag += ('  Axial ratio unresolved at this qRg (all shapes share the Guinier '
                     'limit) — Rg is still reliable.')
        problem = (bool(_scale_causes(first))
                   or any(not r.mw_reliable for r in results))
        self._set_flag(flag, problem=problem)
        if any(not getattr(r, 'well_conditioned', True) for r in results):
            self.flag_label.setToolTip(_ILL_COND_TOOLTIP)

    def _present_core_shell(self, sid: str, results, sig: int) -> None:
        """Display the core-shell P(q) fit -- a LIST of solutions (several core/outer radius
        pairs can reproduce the same P(q)). One results-table block per solution plus a
        degeneracy verdict; every solution is shown and none is silently preferred (the same
        structural-degeneracy contract as the ellipsoid / DPLS shape display)."""
        c = self.controller
        first = results[0]
        self._stash_export(sid, 'form_factor_core_shell',
                           lambda p: c.export_form_factor(results, p))
        self._setup_axes()
        regime = 'apparent' if first.is_apparent else 'thermodynamic'
        self.ax.set_title(f'Form factor P(q): Core–shell sphere ({regime})')

        if not first.converged:
            self._fill_result_table([('Core–shell', 'no fit')])
            self._set_flag('No core-shell sphere reproduces the angular P(q) — try the '
                           'sphere, Mie, or Gaussian-coil model.', problem=True)
            return

        plot_core_shell_form_factor(results, ax=self.ax)
        mw_label = 'M<sub>w,app</sub>' if first.is_apparent else 'M<sub>w</sub>'
        rows = []
        for i, r in enumerate(results, 1):
            tag = f'S{i}' if len(results) > 1 else 'Fit'
            # `rg_reliable` now has TWO causes, so the cell must read the one that
            # actually fired: a nan Rg IS the contrast profile cancelling, while a finite
            # Rg under a False flag is the size floor on r_outer (round 943d f2 -- this
            # branch previously blamed the contrast for both and hid the value).
            if r.rg_reliable:
                rg_val = f'{format_fixed_sig(r.rg_nm, sig)} (contrast-weighted)'
            elif r.rg_undefined:
                rg_val = 'undefined (contrast profile)'
            else:
                rg_val = _rg_cell(f'{format_fixed_sig(r.rg_nm, sig)} (contrast-weighted)', r)
            mw_val = _scaled_value(r, format_pm(r.mw_g_per_mol, r.mw_se))
            rows.append((f'{tag}: R<sub>core</sub> (nm)',
                         format_pm(r.r_core_nm, r.r_core_se)))
            rows.append((f'{tag}: R<sub>outer</sub> (nm)',
                         format_pm(r.r_outer_nm, r.r_outer_se)))
            rows.append((f'{tag}: shell t (nm)', format_fixed_sig(r.shell_thickness_nm, sig)))
            rows.append((f'{tag}: R<sub>g</sub> (nm)', rg_val))
            rows.append((f'{tag}: {mw_label} (g/mol)', mw_val))
            if not first.is_apparent:
                a2_val = _scaled_value(r, format_pm(r.a2_mol_mL_per_g2, r.a2_se))
                rows.append((f'{tag}: A₂ (mol·mL/g²)', a2_val))
            # Per-solution qRg(max)/R² when plural -- the ellipsoid sibling's rule
            # (focused-review 2026-07-23): an unlabeled trailing pair would show
            # whichever solution lists first under the presentation-order convention.
            if len(results) > 1:
                rows.append((f'{tag}: qR<sub>g</sub>(max)',
                             format_fixed_sig(r.qrg_max, sig)))
                rows.append((f'{tag}: R²', format_fixed_sig(r.r_squared, sig)))
        if len(results) == 1:
            rows.append(('qR<sub>g</sub>(max)', format_fixed_sig(first.qrg_max, sig)))
            rows.append(('R²', format_fixed_sig(first.r_squared, sig)))
        self._fill_result_table(rows)

        # Degeneracy verdict leads (the headline caveat of a two-radius fit); calibration and
        # Mw reliability are the genuine red problems. Degeneracy + contrast-matched Rg are
        # neutral structural notes (calmer-warnings tier).
        if first.degenerate:
            flag = (f'The two-radius fit is degenerate — {len(results)} core-shell geometries '
                    '(different radius pairs) reproduce this P(q) equally; all shown, none '
                    'preferred.')
        else:
            flag = 'One core-shell geometry reproduces this P(q).'
        flag += ('  Apparent (single concentration): the 2A₂c term folds into Mw.'
                 if first.is_apparent else
                 '  Thermodynamic across angle and concentration.')
        flag += _STAT_CAVEAT + _ols_marker(getattr(first, 'se_estimator', 'hc3'))
        if _scale_causes(first):
            flag += ('  ' + _scale_phrase(first).capitalize() + ' → Mw'
                     + ('' if first.is_apparent else '/A₂')
                     + ' unreliable (radii/Rg survive).')
        cautions = [c for c in (_mw_unreliable_caution(r) for r in results) if c]
        if cautions:
            flag += '  ' + cautions[0]
        cs_rg_cautions = [c for c in (_rg_unreliable_caution(r) for r in results) if c]
        if cs_rg_cautions:
            flag += '  ' + cs_rg_cautions[0]
        if any(r.rg_undefined for r in results):
            flag += ('  The core/shell contrast profile has no defined scattering Rg (the '
                     'contrasts nearly cancel, or their radial moments oppose) — Rg is not '
                     'reported; the radii still are.')
        problem = (bool(_scale_causes(first))
                   or any(not r.mw_reliable for r in results))
        self._set_flag(flag, problem=problem)
        if any(not getattr(r, 'well_conditioned', True) for r in results):
            self.flag_label.setToolTip(_ILL_COND_TOOLTIP)

    def _present_method(self, method: str, sid: str, conc, payload: Dict) -> None:
        """Main-thread phase: plot + fill tables from the worker's payload. Reads
        precomputed results (payload['rr'] / ['res']); makes no controller fit
        calls itself (the masked-point overlay reads the cached full series)."""
        c = self.controller
        sig = c.settings.no_uncertainty_sig_figs    # ±-less results (R², qRg, single-angle Mw)
        self._export = None
        self._plotted_conc = conc           # the conc this plot is drawn for
        self._plotted_method = method       # ... and the method (its transform)
        self.flag_label.setToolTip('')      # cleared per display; set per-method below
        if method in ('zimm', 'berry'):
            rr = payload['rr']
            res = payload['res']
            self._setup_axes()
            self._zimm_k = self._current_zimm_k(rr)
            plot_zimm(rr, res, ax=self.ax, spacing_constant=self._zimm_k)
            self.ax.set_title(f'{method.capitalize()} plot')
            self._refresh_mw_display()
            self._summarize_zimm(res)
            self._stash_export(sid, method, lambda p: c.export_zimm(rr, res, p))
        elif method == 'form_factor' and isinstance(payload['res'], list):
            # Both the ellipsoid and the core-shell sphere return a LIST of solutions; route
            # by the model so each gets its own display (rule 10 -- the whole display surface).
            if payload['res'] and payload['res'][0].model == ff.CORE_SHELL:
                self._present_core_shell(sid, payload['res'], sig)
            else:
                self._present_ellipsoid(sid, payload['res'], sig)
        elif method == 'form_factor':
            res = payload['res']
            model_label = ff.MODEL_LABELS.get(res.model, res.model)
            self._stash_export(sid, f'form_factor_{res.model}',
                               lambda p: c.export_form_factor(res, p))
            self._setup_axes()
            plot_form_factor(res, ax=self.ax)
            regime = 'apparent' if res.is_apparent else 'thermodynamic'
            self.ax.set_title(f'Form factor P(q): {model_label} ({regime})')
            # Mw is apparent at one concentration, thermodynamic across concentrations;
            # dashed when uncalibrated (arbitrary scale). Rg / R come from the angular
            # shape and survive an uncalibrated run.
            mw_label = 'M<sub>w,app</sub>' if res.is_apparent else 'M<sub>w</sub>'
            mw_val = _scaled_value(res, format_pm(res.mw_g_per_mol, res.mw_se))
            rows = [
                (f'{mw_label} (g/mol)', mw_val),
                ('R<sub>g</sub> (nm)', _rg_cell(format_pm(res.rg_nm, res.rg_se), res)),
            ]
            # Sphere and Mie are both solid spheres, so both report the geometric radius R
            # (the coil has none) -- matching the plot legend and the export column. R is a
            # fixed multiple of Rg, so it carries the same size verdict.
            if res.model in (ff.SPHERE, ff.MIE):
                rows.append(('R (nm)', _rg_cell(format_pm(res.r_nm, res.r_se), res)))
            if not res.is_apparent:
                a2_val = _scaled_value(res,
                                       format_pm(res.a2_mol_mL_per_g2, res.a2_se))
                rows.append(('A₂ (mol·mL/g²)', a2_val))
            rows.append(('qR<sub>g</sub>(max)', format_fixed_sig(res.qrg_max, sig)))
            rows.append(('R²', format_fixed_sig(res.r_squared, sig)))
            self._fill_result_table(rows)
            caution = _mw_unreliable_caution(res)
            if res.is_apparent:
                flag = ('apparent (single concentration): the 2A₂c term folds into Mw — '
                        'extrapolate across concentration for the thermodynamic Mw and A₂.')
            else:
                flag = ('thermodynamic: Mw, Rg and A₂ from the full P(q) fit across '
                        'angle and concentration.')
            flag += _STAT_CAVEAT + _ols_marker(getattr(res, 'se_estimator', 'hc3'))
            rg_caution = _rg_unreliable_caution(res)
            if _scale_causes(res):
                # "(Rg/R survive)" is an UPPER-tail claim: the scale cancels out of the
                # angular shape. It inverts below the size floor, where the size is the
                # defect -- so it is stated only while the size verdict allows it.
                flag += ('  ' + _scale_phrase(res).capitalize() + ' → Mw'
                         + ('' if res.is_apparent else '/A₂')
                         + (' unreliable (Rg/R survive).' if not rg_caution
                            else ' unreliable.'))
            if caution:
                flag += '  ' + caution
            if rg_caution:
                flag += '  ' + rg_caution
            self._set_flag(flag, problem=(bool(_scale_causes(res)) or bool(caution)
                                          or bool(rg_caution)))
            if not getattr(res, 'well_conditioned', True):
                self.flag_label.setToolTip(_ILL_COND_TOOLTIP)
        elif method == 'debye':
            res = payload['res']
            self._stash_export(sid, 'debye', lambda p: c.export_debye(res, p))
            self._setup_axes()
            plot_debye(res, ax=self.ax)
            self.ax.set_title('Debye plot (single concentration)')
            mw_app = _scaled_value(res, format_pm(
                res.mw_apparent_g_per_mol, getattr(res, 'mw_apparent_se', None)))
            self._fill_result_table([
                ('M<sub>w,app</sub> (g/mol)', mw_app),
                ('R<sub>g,app</sub> (nm)',
                 _rg_cell(format_pm(res.rg_apparent_nm,
                                    getattr(res, 'rg_apparent_se', None)), res)),
                ('R²', format_fixed_sig(res.r_squared, sig)),
            ])
            caution = _mw_unreliable_caution(res)
            rg_caution = _rg_unreliable_caution(res)
            flag = (self._apparent_flag(res) + _STAT_CAVEAT
                    + _ols_marker(getattr(res, 'se_estimator', 'hc3')))
            if caution:
                flag += '  ' + caution
            if rg_caution:
                flag += '  Rg: ' + rg_caution
            self._set_flag(flag,
                           problem=bool(_scale_causes(res)) or bool(caution)
                           or bool(rg_caution))
            if not getattr(res, 'well_conditioned', True):
                self.flag_label.setToolTip(_ILL_COND_TOOLTIP)
        elif method == 'guinier':
            res = payload['res']
            self._stash_export(sid, 'guinier', lambda p: c.export_guinier(res, p))
            self._setup_axes()
            plot_guinier(res, ax=self.ax)
            self.ax.set_title('Guinier plot (single concentration)')
            self._fill_result_table([
                ('R<sub>g,app</sub> (nm)',
                 _rg_cell(format_pm(res.rg_nm, getattr(res, 'rg_se', None)), res)),
                ('qR<sub>g</sub>(max)', format_fixed_sig(res.qrg_max, sig)),
                ('R²', format_fixed_sig(res.r_squared, sig)),
            ])
            # Tier tracks severity: the plain "apparent" note is a neutral qualifier;
            # an out-of-regime qRg(max) or an ill-conditioned design is a genuine caution
            # (red). _set_flag adds the glyph. The regime sentence is the ENGINE's
            # `reliability_note` (one wording, one home -- it names the limit the verdict
            # actually read, where this tab used to print a literal 1.3 whatever the
            # Settings value was); the ill-conditioned note rides the same field.
            note = getattr(res, 'reliability_note', '') or ''
            caution = _mw_unreliable_caution(res)
            rg_caution = '' if note else _rg_unreliable_caution(res)
            flag = ('apparent (single concentration): Rg still contains '
                    'concentration effects — extrapolate over c for the '
                    'thermodynamic Rg.')
            if note:
                flag += '  ' + note + ('' if note.endswith('.') else '.')
            elif caution:
                flag += '  Mw (from the intercept): ' + caution
            if rg_caution:
                flag += '  Rg: ' + rg_caution
            self._set_flag(flag, problem=bool(note) or bool(caution) or bool(rg_caution))
            if not getattr(res, 'well_conditioned', True):
                self.flag_label.setToolTip(_ILL_COND_TOOLTIP)
        elif method == 'single':
            res = payload['res']
            self._stash_export(sid, 'single_angle',
                               lambda p: c.export_single_angle(res, p))
            self._clear_plot()
            # Single angle + single concentration = one datum, so Mw_app has no honest ±
            # — the documented fixed-sig fallback, never a fabricated ±. Dash the value
            # only when uncalibrated (arbitrary scale); an ill-conditioned (subnormal
            # divisor) Mw is finite, so it is still shown, with a red caution below.
            mw_app = _scaled_value(res, format_fixed_sig(
                res.mw_apparent_g_per_mol,
                self.controller.settings.no_uncertainty_sig_figs))
            self._fill_result_table([
                ('M<sub>w,app</sub> (g/mol)', mw_app),
                ('Angle', f'{res.angle_deg:.0f}°'),
                ('c (mg/mL)', f'{res.concentration_g_per_mL * 1000:.3g}'),
            ])
            caution = _mw_unreliable_caution(res)
            if _scale_causes(res):
                self._set_flag(
                    f'apparent + {_scale_phrase(res)}: single angle + single '
                    'concentration (contains the form factor and the 2A₂c term), and '
                    'Mw_app is on an arbitrary scale.', problem=True)
            elif caution:
                self._set_flag(
                    'apparent: single angle + single concentration (contains the '
                    'form factor and the 2A₂c term).  ' + caution, problem=True)
            else:
                self._set_flag(
                    'apparent: single angle + single concentration (contains the '
                    'form factor and the 2A₂c term).', problem=False)
        elif method == 'calfree':
            res = payload['res']
            self._stash_export(sid, 'calibration_free_a2',
                               lambda p: c.export_calibration_free_a2(res, p))
            self._setup_axes()
            plot_calibration_free_a2(res, ax=self.ax)
            self.ax.set_title('Calibration-free A₂')
            # Three states, and the middle one is why this is not a two-way branch: an
            # absent A₂ can mean "no Mw stored" (type one in) or "the stored Mw is on an
            # arbitrary scale and was refused" (calibrate, or type a TRUSTED one in). The
            # remedies differ, so the messages must.
            withheld = res.a2_mol_mL_per_g2 is None and self._mw_is_arbitrary_scale()
            if res.a2_mol_mL_per_g2 is not None:
                a2 = format_pm(res.a2_mol_mL_per_g2, getattr(res, 'a2_se', None))
            elif withheld:
                a2 = _UNCAL_A2
            else:
                a2 = 'set a manual Mw to get A₂'
            self._fill_result_table([
                ('2·A₂·M<sub>w</sub> (scale-independent)',
                 format_pm(res.two_a2_mw, getattr(res, 'two_a2_mw_se', None))),
                ('A₂ (mol·mL/g²)', a2),
            ])
            # An ill-conditioned Y-vs-c design (near-equal concentrations) makes
            # 2·A₂·Mw = slope/intercept numerically unreliable — red tier, value still
            # shown. The neutral ± caveat is secondary and yields to it.
            ill_cond = '' if getattr(res, 'well_conditioned', True) else (
                getattr(res, 'reliability_note', '')
                or 'ill-conditioned design (near-equal concentrations); '
                   '2·A₂·Mw numerically unreliable.')
            if ill_cond:
                # Ill-conditioning corrupts the PRODUCT and so everything below it, and
                # leads. The withheld clause is appended rather than dropped (the pattern
                # the 'single' branch above uses for its caution) — two independent facts,
                # both true, and the user needs both.
                self._set_flag(ill_cond + (f' {_WITHHELD_A2_FLAG}' if withheld else ''),
                               problem=True)
                self.flag_label.setToolTip(_ILL_COND_TOOLTIP)
            elif withheld:
                self._set_flag(_WITHHELD_A2_FLAG, problem=True)
                self.flag_label.setToolTip(_WITHHELD_A2_TOOLTIP)
            elif getattr(res, 'two_a2_mw_se', None) is not None:
                self._set_flag('± statistical only'
                               + _ols_marker(getattr(res, 'se_estimator', 'hc3')),
                               problem=False)
                self.flag_label.setToolTip(
                    'No calibration or dn/dc enters 2·A₂·Mw (they cancel in the intensity '
                    'ratio); the ± is a conservative statistical SE that never under-reports '
                    'and tightens with more concentration points. The A₂ ± beneath it is that '
                    'SE divided by 2·Mw with the Mw treated as EXACT, so it excludes the '
                    'error on the Mw you supplied — which enters A₂ linearly. '
                    'See the Theory-and-Equations-Guide §6.1.')
            else:
                self._set_flag('', problem=False)
                self.flag_label.setToolTip('')   # no SE → drop the §15.1 hover
        else:  # rayleigh
            rr = payload['rr']
            self._stash_export(sid, 'rayleigh',
                               lambda p: c.export_rayleigh_series(rr, p))
            self._setup_axes()
            for r in rr:
                if r.concentration_g_per_mL == 0:
                    continue
                # Deliberately no 'c = ' prefix (the plotter's own default label
                # carries one): the multi-series legend here reads tighter bare.
                plot_rayleigh_ratio(r, ax=self.ax,
                                    label=conc_mg_ml_label(r.concentration_g_per_mL))
            self.ax.set_title('Excess Rayleigh ratio')
            calibrated = all(r.calibrated for r in rr)
            self._clear_result_table()           # a per-c series, not a scalar result
            self.status.setText(f'Excess Rayleigh ratio for {len(rr)} concentrations.')
            self._set_flag(
                '' if calibrated else
                'uncalibrated: ΔR is on an arbitrary scale.',
                problem=not calibrated)
        # Passive ⓘ result notes (uniform across every method that returns a result —
        # rayleigh carries a per-c series, not a scalar result, so it has none). One
        # render site, tolerant getattr, so no method branch can silently drop them.
        res_for_notes = payload.get('res')
        notes = list(getattr(res_for_notes, 'notes', ())
                     if res_for_notes is not None else ())
        # Any fit that consumes a fraction's concentration series can silently mix >1 Mw
        # series of one polymer (untagged fractions) -- a ladder regression pools them
        # (zimm/berry/calfree), a single-concentration pick is ambiguous (debye/guinier/
        # single). Append the pooling hint for that whole family, not just Zimm/Berry/Debye
        # (rule 10; the pure engine can't see mw_fraction, so the controller supplies it).
        if method in ('zimm', 'berry', 'form_factor', 'debye', 'guinier',
                      'single', 'calfree'):
            hint = self.controller.sls_pooling_hint(sid, self._fraction)
            if hint:
                notes.append(hint)
        set_flag(self.notes_label, '\n'.join(notes), problem=False)
        self._overlay_masked(method, sid)
        self.canvas.draw_idle()
        self.axis_bar.attach(self.ax)      # single-angle method leaves ax None
        self.refresh_anisotropy_check()    # a fresh Mw updates the display-only f·Mw line

    def _overlay_masked(self, method: str, sid: str) -> None:
        """Draw the hidden points grayed (hollow) at their true plot positions, so
        you can see what is excluded and toggle it back. Derives the points from the
        SAME per-method transform the click hit-test uses (``_point_coords``),
        filtered to the masked ones — so the gray markers and the clickable targets
        can never land at different positions (code-review D5)."""
        if self.ax is None:
            return
        mask = self.controller.sls_mask(sid, self._fraction)
        if mask.is_empty():
            return
        # _point_coords returns (c, angle, x, y) for every point (masked + unmasked)
        # from the cached unmasked series — no fresh controller call. Keep only the
        # masked ones for the gray overlay.
        masked = [(x, y) for (c, ang, x, y) in self._point_coords(method, sid)
                  if mask.is_masked(c, float(ang))]
        if masked:
            xs, ys = zip(*masked, strict=True)
            self.ax.scatter(list(xs), list(ys), s=42, facecolors='none',
                            edgecolors='#b0b0b0', linewidths=1.2, zorder=2,
                            label='masked (excluded)')
            # The hollow markers are added after the method plot built its legend, so
            # re-issue it (preserving any title) to give 'masked' an entry — but only
            # when a legend already exists (unchanged from the hand-written block).
            if self.ax.get_legend() is not None:
                reissue_legend_preserving_title(self.ax, fontsize=9)

    # ------------------------------------------------ click-to-mask points ---
    _CLICKABLE = ('zimm', 'berry', 'form_factor', 'debye', 'guinier', 'rayleigh')
    _PICK_RADIUS_PX = 12.0

    def _point_coords(self, method: str, sid: str):
        """Every data point for the current method as (c, angle, x_data, y_data),
        for click hit-testing. Masked and unmasked points are both included. Reads
        the cached unmasked series (as of the last run) — no controller call.
        x_data is in the DRAWN axis units: the canonical q² (or q²+k·c composite)
        scaled by the active q² display factor, exactly as the plotters draw it
        (spec 04 stage 2; factor exactly 1.0 on the default nm⁻² choice)."""
        full = self._full_rr.get((sid, self._fraction))
        if full is None:
            return []
        qfac = display_factor_from('scattering_q2', 'nm⁻²')
        pts = []
        if method in ('zimm', 'berry'):
            berry = (method == 'berry')
            for r in full:
                cc = r.concentration_g_per_mL
                for i, ang in enumerate(r.angles_deg):
                    y = r.kc_over_dR_mol_per_g[i]
                    if not np.isfinite(y) or (berry and y <= 0):
                        continue
                    yy = math.sqrt(y) if berry else float(y)
                    pts.append((cc, float(ang),
                                (float(r.q2_nm2[i]) + self._zimm_k * cc) * qfac,
                                yy))
        elif method in ('debye', 'guinier'):
            cc = self._plotted_conc          # the conc actually plotted, not a live
                                             # conc_combo read
            r = next((x for x in full if x.concentration_g_per_mL == cc), None)
            if r is not None:
                for i, ang in enumerate(r.angles_deg):
                    if method == 'debye':
                        y = r.kc_over_dR_mol_per_g[i]
                    else:
                        dR = r.excess_rayleigh_cm_inv[i]
                        if not (np.isfinite(dR) and dR > 0):
                            continue
                        y = math.log(dR)
                    if not np.isfinite(y):
                        continue
                    pts.append((cc, float(ang), float(r.q2_nm2[i]) * qfac,
                                float(y)))
        elif method == 'form_factor':
            # Kc/dR vs q^2 for every concentration (matches plot_form_factor's x = q^2,
            # y = Kc/dR — no Zimm spacing offset, no Berry sqrt).
            for r in full:
                cc = r.concentration_g_per_mL
                if cc == 0:
                    continue
                for i, ang in enumerate(r.angles_deg):
                    y = r.kc_over_dR_mol_per_g[i]
                    if np.isfinite(y):
                        pts.append((cc, float(ang), float(r.q2_nm2[i]) * qfac,
                                    float(y)))
        elif method == 'rayleigh':
            for r in full:
                cc = r.concentration_g_per_mL
                if cc == 0:
                    continue
                for i, ang in enumerate(r.angles_deg):
                    y = r.excess_rayleigh_cm_inv[i]
                    if np.isfinite(y):
                        pts.append((cc, float(ang), float(r.q2_nm2[i]) * qfac,
                                    float(y)))
        return pts

    def _on_canvas_click(self, event) -> None:
        """Toggle the point nearest the click (within a pixel radius)."""
        if (not self._ran or self.sample_id is None or self.ax is None
                or event.inaxes is not self.ax or event.button != 1):
            return
        if getattr(self.nav_toolbar, 'mode', ''):   # pan/zoom active -> ignore
            return
        if runner().is_busy:                 # click masks a point then re-runs
            busy_notice(self)
            return
        method = self._plotted_method        # what is DRAWN, never the live combo
        if method not in self._CLICKABLE:
            return
        pts = self._point_coords(method, self.sample_id)
        if not pts:
            return
        disp = self.ax.transData.transform([(p[2], p[3]) for p in pts])
        d = np.hypot(disp[:, 0] - event.x, disp[:, 1] - event.y)
        j = int(np.argmin(d))
        if d[j] > self._PICK_RADIUS_PX:
            return
        cc, ang = pts[j][0], pts[j][1]
        frac = self._fraction
        mask = self.controller.sls_mask(self.sample_id, frac)
        if mask.is_point_masked(cc, ang):
            self.controller.unmask_point(self.sample_id, cc, ang, frac)
        elif mask.is_masked(cc, ang):
            return   # hidden by an angle/concentration mask -> use the lists
        else:
            self.controller.mask_point(self.sample_id, cc, ang, frac)
        self._on_run()

    def _fill_result_table(self, rows) -> None:
        """Rebuild the per-method results table from (label, value) pairs."""
        t = self.result_table
        t.setRowCount(len(rows))
        for r, (label, value) in enumerate(rows):
            t.setItem(r, 0, QtWidgets.QTableWidgetItem(label))
            t.setItem(r, 1, QtWidgets.QTableWidgetItem(value))
        # Cap the table at exactly its content height (real row height, not a per-row
        # guess) so the last row never clips into a scrollbar while it sits in the
        # roomy under-plot strip.
        row_h = t.verticalHeader().defaultSectionSize()
        t.setMaximumHeight(row_h * max(len(rows), 1) + 2 * t.frameWidth() + 2)
        t.resizeColumnsToContents()
        self.status.clear()

    def _clear_result_table(self) -> None:
        self.result_table.setRowCount(0)

    def _summarize_zimm(self, res) -> None:
        sig = self.controller.settings.no_uncertainty_sig_figs   # ±-less: R², two-route Mw
        mw_se = getattr(res, 'mw_se', None)
        rg_se = getattr(res, 'rg_se', None)
        a2_se = getattr(res, 'a2_se', None)
        # Uncalibrated Mw AND absolute A₂ are both on an arbitrary scale (Mw_app = Mw/f,
        # A₂_app = f·A₂) — dash BOTH at the value rather than show a meaningless number.
        # Only Rg (scale-free) survives here; the calibration-free 2·A₂·Mw product is a
        # separate method/table. An ill-conditioned (but calibrated) fit is NOT dashed —
        # the value is finite, so we keep it (format_pm renders a non-physical NaN as
        # 'n/a') and warn below the fold; so key the dash on `calibrated`, not on the
        # reliability flag (which now also folds in conditioning).
        mw_value = _scaled_value(res, format_pm(res.mw_g_per_mol, mw_se))
        a2_value = _scaled_value(res, format_pm(res.a2_mol_mL_per_g2, a2_se))
        self._fill_result_table([
            ('Method', res.method.capitalize()),
            ('M<sub>w</sub> (g/mol)', mw_value),
            ('R<sub>g</sub> (nm)', _rg_cell(format_pm(res.rg_nm, rg_se), res)),
            ('A₂ (mol·mL/g²)', a2_value),
            ('R²', format_fixed_sig(res.r_squared, sig)),
        ])
        # The "Rg survives" clause is gated on the size's own verdict -- beside a
        # flagged Rg cell it must not assert validity (the form-factor arm's rule).
        rg_caution = _rg_unreliable_caution(res)
        flag = (
            f'{_scale_phrase(res)}: Mw and absolute A₂ are unreliable; the '
            'calibration-free 2·A₂·Mw remains valid (the scale cancels)'
            + ('; Rg: ' + rg_caution if rg_caution
               else '; Rg remains valid (the scale cancels).')
            if _scale_causes(res) else (('Rg: ' + rg_caution) if rg_caution else ''))
        # Calibrated but Mw/A₂ unreliable for a numerical reason (ill-conditioned design,
        # non-finite overflow, or a non-physical intercept): value shown, red tier
        # (see _mw_unreliable_caution).
        cond_caution = _mw_unreliable_caution(res)
        if cond_caution:
            flag = (flag + '\n' + cond_caution) if flag else cond_caution
        # Two-route consistency: Mw can be read off the c→0 line OR the q→0 line, and the
        # two should match; a large gap warns of curvature/extrapolation error (no ±
        # involved). `agree` is the RELATIVE DIFFERENCE between the two, so it is phrased as
        # "differ by X%" — "agreement X%" was ambiguous. The two absolute Mw print only when
        # when BOTH SCALES ARE KNOWN: otherwise they carry the same arbitrary scale the
        # table has just dashed out, and would read as a molar mass many orders off. The
        # RATIO survives on both branches, being scale-independent. The Cabannes readout
        # in `_refresh_anisotropy_display` withholds its f·Mw on the same condition.
        agree = getattr(res, 'extrapolation_agreement_rel', None)
        mc0 = getattr(res, 'mw_from_c0_g_per_mol', None)
        mq0 = getattr(res, 'mw_from_q0_g_per_mol', None)
        route_disagrees = False
        if agree is not None and math.isfinite(agree) and mc0 is not None and mq0 is not None:
            # The leading clause of `flag` has already named the missing scale, so the
            # withheld branch states only why the two values are absent.
            routes = ('(values withheld — arbitrary scale)' if _scale_causes(res) else
                      f'— c→0: {format_fixed_sig(mc0, sig)}, '
                      f'q→0: {format_fixed_sig(mq0, sig)} g/mol —')
            note = (f'Mw from the two extrapolation routes {routes} '
                    f'differ by {agree * 100:.0f}%')
            if agree > 0.10:
                route_disagrees = True
                note += (' >10% — the angle × concentration grid is incomplete (a level with one'
                         ' usable angle, or a missing concentration) and the two routes weight it'
                         ' differently; on a complete grid they agree by construction')
            flag = (flag + '\n' + note) if flag else note
        if mw_se is not None or a2_se is not None:
            flag = ((flag + '\n' if flag else '') + _STAT_CAVEAT.strip()
                    + _ols_marker(getattr(res, 'se_estimator', 'hc3')))
        # Red alarm only for a genuine problem (uncalibrated, ill-conditioned design, or
        # the two Mw routes disagree by >10 %); the plain two-route note and the ± caveat
        # are neutral qualifiers. _set_flag adds the ⚠/ⓘ glyph to match the tier.
        problem = (bool(_scale_causes(res)) or route_disagrees or bool(cond_caution)
                   or bool(rg_caution))
        self._set_flag(flag, problem=problem)
        if not getattr(res, 'well_conditioned', True):
            self.flag_label.setToolTip(_ILL_COND_TOOLTIP)

    def _set_flag(self, text: str, *, problem: bool = True) -> None:
        """Set this tab's flag label via the shared two-tier renderer.

        ``problem=True`` → a genuine data-quality issue (uncalibrated, >10 % route
        disagreement …): bold red ``error``, message led with ⚠.
        ``problem=False`` → a neutral, expected result-type qualifier (apparent /
        ± statistical): a calm non-bold ``qualifier`` accent, message led with ⓘ.
        See :func:`gui.theme.set_flag` — the one place glyph + color are chosen so
        they can never disagree (style guide §5, R5.1)."""
        set_flag(self.flag_label, text, problem=problem)

    @staticmethod
    def _apparent_flag(res) -> str:
        # Glyph-less: _set_flag prepends the tier glyph. The unknown-scale clause is the
        # genuine problem the caller routes on (problem=bool(_scale_causes(res))).
        base = 'apparent (single concentration): intercept is 1/Mw + 2A₂c.'
        phrase = _scale_phrase(res)
        if phrase:
            base += f' Also {phrase} → Mw unreliable (Rg survives).'
        return base

    # ------------------------------------------------------------ figure ---
    def _setup_axes(self) -> None:
        if self.ax is not None:
            try:
                self.ax.set_xscale('linear')
            except Exception:
                pass
        self.figure.clf()
        self.ax = self.figure.add_subplot(1, 1, 1)

    def _clear_plot(self, message: Optional[str] = None) -> None:
        if self.ax is not None:
            try:
                self.ax.set_xscale('linear')
            except Exception:
                pass
        if message:
            # Empty-state message via the shared helper (`clear_figure=True`): a `clf()`
            # then a single FIGURE-level `figure.text`, NO new Axes — the same clean look
            # every DLS/SLS plot tab now shows. The no-Axes-churn path is deliberate:
            # recreating an Axes on each empty-state refresh once churned matplotlib/Agg
            # teardown enough to trip a Qt heap fault (0xc0000374) across the many windows
            # the gui-smoke suite builds. SLS holds `self.ax = None` while empty (the
            # canvas click-guard `_on_canvas_click` relies on that).
            show_plot_placeholder(self.figure, self.canvas, message, clear_figure=True)
        else:
            self.figure.clf()
            self.canvas.draw_idle()
        self.ax = None
        if hasattr(self, 'axis_bar'):
            self.axis_bar.attach(None)


# ===========================================================================
# Derived Intensity -- static points from DLS runs' mean count rates
# ===========================================================================
# Why a DLS run cannot supply a static point, keyed by the controller's reason code.
# The wording lives here rather than in the controller: the controller is framework-
# agnostic and carries no user-facing prose, and a literal table is resolvable by the
# help-inventory scan in a way a composed string is not.
_INELIGIBLE_REASON = {
    'no_count_rate': ('No mean count rate committed. Set one on the Data tab -- a '
                      'linked count-rate trace, or the instrument file\'s own '
                      'value, can fill it.'),
    'bad_count_rate': ('The committed mean count rate is not a positive number, so '
                       'it cannot be a scattered intensity.'),
    'no_concentration': ('No concentration committed. Set one on the Data tab; a '
                         'static measurement needs one.'),
}


class _DerivedIntensityTab(QtWidgets.QWidget):
    """Turn the selected sample's DLS runs into an SLS intensity series.

    A DLS run's average scattered intensity IS the static observable, so a
    goniometer that recorded only correlograms still yields a full static
    analysis -- the Sedlak & Amis workflow on the BI-200SM.

    The tab shows every DLS run in the sample with the mean count rate it carries
    and where that number came from. The SOURCE COLUMN IS READ-ONLY here: the mean
    count rate is chosen once, on the Data tab, so there is exactly one
    authoritative home for it and no second copy of the choice to drift.
    """

    workspaceChanged = QtCore.Signal()   # measurements were created

    # Short headers on purpose: spelled out, six columns overflowed the control
    # panel and pushed Status off the edge. Units follow ui_style_guide S9's
    # table convention (the degree sign, not the word).
    _COLS = ('Use', 'Angle (°)', 'Count rate (cps)', 'Source',
             'c (mg/mL)', 'Status')

    def __init__(self, controller, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.sample_id: Optional[str] = None
        self._rows: List[Dict[str, Any]] = []
        self._series: List[Any] = []
        # Runs the user has deliberately UNTICKED, by item id. _refresh rebuilds the
        # table from scratch on every sidebar change or Settings commit, so without
        # this the exclusions are silently re-ticked between choosing them and
        # pressing Derive. Keyed by id, not row, because the row order moves.
        self._excluded: set = set()
        self.ax = None
        self._build_ui()
        self._refresh()

    # ---------------------------------------------------------------- UI ---
    def _build_ui(self) -> None:
        _, left, right = make_split_panels(self, left_min_width=420,
                                           sizes=(470, 700))
        left.addWidget(section_header(
            'Derived Intensity',
            'Build an SLS intensity series from this sample\'s DLS runs.',
            bullets=[
                'Each DLS run contributes its mean count rate as the intensity '
                'at that angle.',
                'The mean count rate and its source are set on the Data tab; '
                'they are shown here read-only.',
                'Runs at the same angle and concentration are averaged; one '
                'measurement is created per concentration.',
                'Derived points are ordinary SLS measurements -- calibrate and '
                'analyze them on the Analysis sub-tab.',
                'Their intensities are count rates in cps. The calibrant '
                'intensity and any solvent blank analyzed with them must be in '
                'cps as well; only the ratio matters, but the units must match.',
                'A derived point is a snapshot: re-running a fit or editing a '
                'source run never changes it. Derive again to refresh.',
            ]))

        self.header = QtWidgets.QLabel()
        self.header.setWordWrap(True)
        left.addWidget(self.header)

        self.table = QtWidgets.QTableWidget(0, len(self._COLS))
        self.table.setHorizontalHeaderLabels(list(self._COLS))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.table.horizontalHeader().setStretchLastSection(True)
        left.addWidget(self.table, 1)

        tick_row = QtWidgets.QHBoxLayout()
        self.all_button = QtWidgets.QPushButton('Select all')
        self.none_button = QtWidgets.QPushButton('Select none')
        self.all_button.clicked.connect(lambda: self._set_all(True))
        self.none_button.clicked.connect(lambda: self._set_all(False))
        tick_row.addWidget(self.all_button)
        tick_row.addWidget(self.none_button)
        tick_row.addStretch(1)
        left.addLayout(tick_row)

        form = QtWidgets.QFormLayout()
        self.dn_dc_edit = QtWidgets.QLineEdit()
        self.dn_dc_edit.setPlaceholderText('optional')
        self.dn_dc_edit.setToolTip(
            'Refractive-index increment for the derived measurements, in mL/g. '
            'Never proposed or guessed -- enter it yourself. Left blank, the '
            'derived points carry no dn/dc and every result computed from them '
            'is disclosed as being on an arbitrary optical scale.')
        form.addRow('dn/dc (mL/g):', self.dn_dc_edit)
        left.addLayout(form)

        self.derive_button = QtWidgets.QPushButton('Derive')
        self.derive_button.setToolTip(
            'Create one SLS measurement per concentration from the ticked runs. '
            'The intensities are count rates in cps, so the calibrant intensity '
            'and any solvent blank you use with them must be in cps too -- the '
            'program cannot detect a unit mismatch, and entering a calibrant in '
            'kcps would scale Mw by 1000. '
            'Intensities are comparable across runs only if the attenuation and '
            'laser power were the same for all of them -- the program cannot '
            'check this. At high count rates detector dead time makes the '
            'response non-linear; see your instrument manual for its limit.')
        self.derive_button.clicked.connect(self._on_derive)
        left.addWidget(self.derive_button)

        self.status = QtWidgets.QLabel('')
        self.status.setWordWrap(True)
        left.addWidget(self.status)
        self.flag_label = ThemedLabel('', role='error', bold=True)
        self.flag_label.setWordWrap(True)
        left.addWidget(self.flag_label)
        self.notes_label = ThemedLabel('', size=11)
        self.notes_label.setWordWrap(True)
        left.addWidget(self.notes_label)

        self.figure = Figure(figsize=(5.5, 4.6))
        self.canvas = make_canvas_expanding(FigureCanvas(self.figure))
        self.nav_toolbar = themed_navtoolbar(self.canvas, self, self.controller)
        self.axis_bar = AxisControlBar(self.canvas)
        holder = QtWidgets.QWidget()
        pv = QtWidgets.QVBoxLayout(holder)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.addWidget(self.nav_toolbar)
        pv.addWidget(self.canvas, 1)
        pv.addWidget(self.axis_bar)
        right.addWidget(holder, 1)

    # ----------------------------------------------------------- state ---
    def set_measurement(self, item_id: Optional[str]) -> None:
        """Follow the shell's active sample."""
        sid = (self.controller.sample_id_of(item_id)
               if item_id is not None else None)
        self.show_active_sample(sid)

    def show_active_sample(self, sid: Optional[str], **_kw) -> None:
        # A derived series, its status line and its notes all describe the sample
        # they were derived FROM, so they are cleared when the sample changes rather
        # than left to be read against the new one. Only the run table is rebuilt by
        # _refresh; everything computed on demand has to be dropped explicitly.
        if sid != self.sample_id:
            self._series = []
            self._excluded.clear()
            self.status.clear()
            self.notes_label.clear()
            self.flag_label.clear()
        self.sample_id = sid
        self._refresh()

    def _ticked_item_ids(self) -> List[str]:
        out = []
        for row, info in enumerate(self._rows):
            widget = self.table.cellWidget(row, 0)
            if widget is not None and widget.isChecked():
                out.append(info['item_id'])
        return out

    def _on_row_toggled(self, item_id: str, checked: bool) -> None:
        """Remember an exclusion so the next refresh does not undo it."""
        if checked:
            self._excluded.discard(item_id)
        else:
            self._excluded.add(item_id)

    def _set_all(self, checked: bool) -> None:
        for row, info in enumerate(self._rows):
            widget = self.table.cellWidget(row, 0)
            if widget is not None and widget.isEnabled():
                widget.setChecked(checked)          # drives _on_row_toggled
            elif not checked:
                self._excluded.add(info['item_id'])

    # --------------------------------------------------------- display ---
    def _refresh(self) -> None:
        self._rows = (self.controller.derived_intensity_candidates(self.sample_id)
                      if self.sample_id else [])
        self.table.setRowCount(len(self._rows))
        for row, info in enumerate(self._rows):
            box = QtWidgets.QCheckBox()
            box.setEnabled(bool(info['eligible']))
            box.setChecked(bool(info['eligible'])
                           and info['item_id'] not in self._excluded)
            box.toggled.connect(
                lambda on, iid=info['item_id']: self._on_row_toggled(iid, on))
            reason = _INELIGIBLE_REASON.get(info['reason_code'], '')
            if not info['eligible']:
                box.setToolTip(reason)
            self.table.setCellWidget(row, 0, box)

            conc = info['concentration_g_per_mL']
            cells = [
                _EMPTY if info['angle_deg'] is None
                else format_fixed_sig(info['angle_deg'], 4),
                _EMPTY if info['mean_count_rate_cps'] is None
                # Rounded to whole counts per second before display: the
                # sub-cps tail is an artifact of dividing counts by a time,
                # not resolution the detector has. Same .10g rendering the
                # Data tab gives this quantity, so the two cannot disagree.
                else '{0:.10g}'.format(round(info['mean_count_rate_cps'])),
                info['source'] or _EMPTY,
                _EMPTY if conc is None else format_fixed_sig(conc * 1000.0, 4),
                'ready' if info['eligible'] else 'not usable',
            ]
            for col, text in enumerate(cells, start=1):
                item = QtWidgets.QTableWidgetItem(text)
                # The reason rides as a tooltip on every cell of the row rather
                # than as column text: spelled out in the cell it widened the
                # table past the panel and pushed Status off the edge entirely.
                if reason:
                    item.setToolTip(reason)
                self.table.setItem(row, col, item)
        self.table.resizeColumnsToContents()

        if not self.sample_id:
            self.header.setText('No sample selected.')
        elif not self._rows:
            self.header.setText(
                'This sample has no DLS runs, so there is nothing to derive from.')
        else:
            n_ok = sum(1 for r in self._rows if r['eligible'])
            text = ('{0} DLS run(s) in this sample; {1} can supply a static '
                    'point.'.format(len(self._rows), n_ok))
            if n_ok < len(self._rows):
                text += (' A row marked "not usable" says what it needs in its '
                         'tooltip (turn on "Show tooltips on hover" under Settings '
                         'if none appears).')
            self.header.setText(text)
        self.derive_button.setEnabled(
            any(r['eligible'] for r in self._rows))
        self._draw()

    def _draw(self) -> None:
        if not self._series:
            show_plot_placeholder(
                self.figure, self.canvas,
                'Tick the runs to use, then press Derive.', clear_figure=True)
            self.ax = None
            self.axis_bar.attach(None)
            return
        self.figure.clf()
        self.ax = self.figure.add_subplot(111)
        for series in self._series:
            label = ('c = {0} mg/mL'.format(
                format_fixed_sig((series.concentration_g_per_mL or 0.0) * 1000.0, 3)))
            self.ax.plot(series.angles_deg, series.intensities, 'o-', label=label)
        self.ax.set_xlabel(r'Scattering angle $\theta$ (°)')
        self.ax.set_ylabel('Mean count rate (cps)')
        self.ax.legend(fontsize=8)
        self.figure.tight_layout()
        self.canvas.draw_idle()
        self.axis_bar.attach(self.ax)

    # ---------------------------------------------------------- action ---
    def _on_derive(self) -> None:
        # Creating measurements mutates the workspace and regroups, so it is
        # refused while a fit is in flight -- the same guard every commit path uses.
        if runner().is_busy:
            busy_notice(self)
            return
        item_ids = self._ticked_item_ids()
        if not item_ids:
            self.status.setText('Tick at least one run to derive from.')
            return
        text = self.dn_dc_edit.text().strip()
        dn_dc = None
        if text:
            try:
                dn_dc = float(text)
            except ValueError:
                QtWidgets.QMessageBox.warning(
                    self, 'dn/dc', 'dn/dc must be a number, or blank.')
                return
        try:
            out = self.controller.derive_static_points(
                item_ids, dn_dc_mL_per_g=dn_dc)
        except ValueError as exc:
            set_flag(self.flag_label, humanize_field_names(str(exc)))
            self.status.clear()
            return
        self.flag_label.clear()
        self._series = out['series']
        self.status.setText(
            'Created {0} SLS measurement(s) from {1} run(s). Calibrate and '
            'analyze them on the Analysis sub-tab.'.format(
                len(out['item_ids']), len(item_ids)))
        set_flag(self.notes_label, ' '.join(out['notes']), problem=False)
        self._draw()
        self.workspaceChanged.emit()


# ===========================================================================
# Mode-Resolved -- the Sedlak partition of an SLS series by DLS mode amplitudes
# ===========================================================================
class _ModeResolvedTab(QtWidgets.QWidget):
    """Partition this sample's excess intensity by its recorded mode sets.

    The DLS side OWNS mode definition: sets are recorded on the DLS tab's Mode
    Map ("Record mode set") and appear here as a READ-ONLY list of snapshots
    with their provenance -- re-fitting over there never changes a recorded
    set, and "Open in Mode Map" is navigation, never a duplicated view.

    Each mode's share dR_i = A_i * dR is an ordinary Rayleigh series computed
    LIVE (per-mode results are never workspace measurements and never touch
    the sample's own results); every per-mode result is apparent +
    mode-resolved, with the partition's provenance in its export Comments.
    """

    modeMapFocusRequested = QtCore.Signal(str)   # sample_id -> DLS ▸ Mode Map

    _SET_COLS = ('Label', 'Method', 'Families', 'Points', 'Recorded',
                 'Sources')
    _METHODS = [('Amplitude Guinier (per-mode size)', 'amplitude_guinier'),
                ('Guinier', 'guinier'), ('Debye', 'debye'),
                ('Zimm (≥ 2 concentrations)', 'zimm'),
                ('Berry (≥ 2 concentrations)', 'berry')]

    def __init__(self, controller, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.sample_id: Optional[str] = None
        self._sets: List[Tuple[Optional[str], Any]] = []   # (fraction, set)
        self._parts: Dict[float, Any] = {}                 # conc -> partition
        self._guinier: Optional[Any] = None                # last size fit
        self._guinier_conc: Optional[float] = None          # ... and the c it ran at
        self.ax = None
        self._build_ui()
        self._refresh()

    # ---------------------------------------------------------------- UI ---
    def _build_ui(self) -> None:
        _, left, right = make_split_panels(self, left_min_width=420,
                                           sizes=(470, 700))
        left.addWidget(section_header(
            'Mode-Resolved',
            'Split an SLS intensity series by the DLS modes\' amplitudes — '
            'the Sedlák partition — and analyze each mode\'s share; see the '
            'Theory-and-Equations-Guide §4.1.2.',
            bullets=[
                'Record a mode set on the DLS tab\'s Mode Map; it appears '
                'here with its method, regularization and date.',
                'A mode set is a snapshot — re-fitting on the DLS side never '
                'changes it. Record again for a fresh one.',
                'Partition splits every SLS series of this sample (loaded or '
                'derived) that shares a concentration with the set: '
                'dR_mode = amplitude fraction × dR.',
                'Amplitude Guinier fits one mode\'s series alone and gives '
                'that mode\'s apparent size.',
                'Guinier/Debye/Zimm/Berry run on a mode\'s series like any '
                'SLS analysis; every result stays flagged apparent and '
                'mode-resolved, with the partition in its export Comments.',
                'Open in Mode Map jumps to where the set was defined.']))

        self.header = QtWidgets.QLabel()
        self.header.setWordWrap(True)
        left.addWidget(self.header)

        self.sets_table = QtWidgets.QTableWidget(0, len(self._SET_COLS))
        self.sets_table.setHorizontalHeaderLabels(list(self._SET_COLS))
        self.sets_table.verticalHeader().setVisible(False)
        self.sets_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.sets_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.sets_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.sets_table.horizontalHeader().setStretchLastSection(True)
        self.sets_table.itemSelectionChanged.connect(self._on_set_selected)
        self.sets_table.setContextMenuPolicy(
            QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.sets_table.customContextMenuRequested.connect(self._on_set_menu)
        left.addWidget(self.sets_table, 1)

        nav_row = QtWidgets.QHBoxLayout()
        self.open_map_button = QtWidgets.QPushButton('Open in Mode Map')
        self.open_map_button.setToolTip(
            'Jump to the DLS tab\'s Mode Map, where mode sets are defined '
            'and recorded (right-click a set here to rename or remove it).')
        self.open_map_button.clicked.connect(self._on_open_map)
        nav_row.addWidget(self.open_map_button)
        nav_row.addStretch(1)
        left.addLayout(nav_row)

        self.partition_button = QtWidgets.QPushButton('Partition')
        self.partition_button.setToolTip(
            'Split every matching SLS series by the selected set\'s '
            'amplitudes. Amplitudes below ~2 % of a run\'s total are '
            'imprecise — the practical floor of the decomposition. The '
            'partition assumes well-separated modes; the results say when '
            'that is strained. See the Theory-and-Equations-Guide.')
        self.partition_button.clicked.connect(self._on_partition)
        left.addWidget(self.partition_button)

        conc_row = QtWidgets.QFormLayout()
        self.conc_combo = QtWidgets.QComboBox()
        self.conc_combo.currentIndexChanged.connect(
            lambda _i: self._show_partition())
        conc_row.addRow('Concentration:', self.conc_combo)
        left.addLayout(conc_row)

        self.part_table = QtWidgets.QTableWidget(0, 0)
        self.part_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.part_table.verticalHeader().setVisible(False)
        self.part_table.setMinimumHeight(110)
        left.addWidget(self.part_table, 1)

        fit_form = QtWidgets.QFormLayout()
        self.mode_combo = QtWidgets.QComboBox()
        fit_form.addRow('Mode:', self.mode_combo)
        self.method_combo = QtWidgets.QComboBox()
        for label, code in self._METHODS:
            self.method_combo.addItem(label, code)
        self.method_combo.setToolTip(
            'Every result here is computed from ONE mode\'s share of the '
            'excess intensity and stays apparent + mode-resolved. Its Kc/dR '
            'ordinate uses the TOTAL concentration and the chain dn/dc — for '
            'a mode that is not the chains themselves (solvent structure, '
            'domains), that contrast is an unverifiable assumption, and an '
            'Mw computed with it is unanchored, not just noisy. Charged '
            'systems additionally need dialysis-equilibrium (Donnan) '
            'conditions for absolute Mw/A₂. See the '
            'Theory-and-Equations-Guide.')
        fit_form.addRow('Analysis:', self.method_combo)
        left.addLayout(fit_form)
        self.fit_button = QtWidgets.QPushButton('Run')
        self.fit_button.clicked.connect(self._on_fit)
        left.addWidget(self.fit_button)

        self.result_label = QtWidgets.QLabel('')
        self.result_label.setWordWrap(True)
        self.result_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        left.addWidget(self.result_label)

        self.export_button = QtWidgets.QPushButton('Export CSV…')
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._on_export)
        left.addWidget(self.export_button)

        self.status = QtWidgets.QLabel('')
        self.status.setWordWrap(True)
        left.addWidget(self.status)
        self.flag_label = ThemedLabel('', role='error', bold=True)
        self.flag_label.setWordWrap(True)
        left.addWidget(self.flag_label)
        self.notes_label = ThemedLabel('', size=11)
        self.notes_label.setWordWrap(True)
        left.addWidget(self.notes_label)

        rw = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(rw)
        rl.setContentsMargins(0, 0, 0, 0)
        vrow = QtWidgets.QHBoxLayout()
        vrow.addWidget(QtWidgets.QLabel('View:'))
        self.view_combo = QtWidgets.QComboBox()
        self.view_combo.addItem('Partitioned intensities', 'partition')
        self.view_combo.addItem('Amplitude Guinier', 'guinier')
        self.view_combo.currentIndexChanged.connect(lambda _i: self._draw())
        vrow.addWidget(self.view_combo)
        vrow.addStretch(1)
        rl.addLayout(vrow)
        self.figure = Figure(figsize=(5.5, 4.6))
        self.canvas = make_canvas_expanding(FigureCanvas(self.figure))
        self.nav_toolbar = themed_navtoolbar(self.canvas, self, self.controller)
        self.axis_bar = AxisControlBar(self.canvas)
        rl.addWidget(self.nav_toolbar)
        rl.addWidget(self.canvas, 1)
        rl.addWidget(self.axis_bar)
        right.addWidget(rw, 1)

    # ----------------------------------------------------------- state ---
    def set_measurement(self, item_id: Optional[str]) -> None:
        sid = (self.controller.sample_id_of(item_id)
               if item_id is not None else None)
        self.show_active_sample(sid)

    def _clear_computed(self) -> None:
        """Drop everything computed on demand — the partition, the fit
        readout, and every label describing them. ONE body called from both
        the sample switch and the set switch (the two hand-maintained blocks had
        already drifted, leaving a previous set's fitted Rg and notes on screen)."""
        self._parts = {}
        self._guinier = None
        self._guinier_conc = None
        self.conc_combo.clear()
        self.part_table.setRowCount(0)
        self.part_table.setColumnCount(0)
        self.result_label.clear()
        self.status.clear()
        self.flag_label.clear()
        self.notes_label.clear()
        self.export_button.setEnabled(False)

    def show_active_sample(self, sid: Optional[str], **_kw) -> None:
        # Everything computed on demand describes the sample it came from, so
        # a sample change drops it (the _DerivedIntensityTab discipline).
        if sid != self.sample_id:
            self._clear_computed()
        self.sample_id = sid
        self._refresh()

    def showEvent(self, event) -> None:
        """Re-read the recorded-set list whenever the tab becomes visible:
        'Record mode set' on the DLS side emits no signal, so without this a
        freshly recorded set stays invisible until an unrelated sample
        re-selection."""
        super().showEvent(event)
        self._refresh()

    def _selected(self) -> Optional[Tuple[Optional[str], Any]]:
        row = self.sets_table.currentRow()
        if 0 <= row < len(self._sets):
            return self._sets[row]
        return None

    # --------------------------------------------------------- display ---
    def _refresh(self) -> None:
        # Preserve the selected SET across the rebuild (by set_id, never row):
        # a refresh must not fire the set-switch clear on the unchanged set,
        # or every tab switch would silently drop the computed partition.
        prev = self._selected()
        prev_id = prev[1].set_id if prev is not None else None
        self._sets = (self.controller.mode_sets_index(self.sample_id)
                      if self.sample_id else [])
        self.sets_table.blockSignals(True)
        self.sets_table.setRowCount(len(self._sets))
        for row, (frac, s) in enumerate(self._sets):
            dangling = self.controller.mode_set_dangling_sources(
                self.sample_id, frac, s.set_id)
            ids = {p.item_id for m in s.modes for p in m.points if p.item_id}
            # The inversion summary comes from the controller ALREADY JOINED —
            # THE one computation the exported provenance also uses, so the two
            # surfaces cannot drift. Joining the components here instead is how
            # the size grid reached the export and not this cell.
            inv = self.controller.mode_set_inversion_summary(
                self.sample_id, frac, s.set_id)
            alpha_txt = f' ({inv})' if inv else ''
            src_cell = '{0} runs'.format(len(ids))
            if dangling:
                src_cell = '⚠ {0} ({1} removed)'.format(src_cell,
                                                        len(dangling))
            cells = [
                s.label + (f'  [{frac}]' if frac else ''),
                s.source_method + alpha_txt,   # controller-derived, joined
                ', '.join(m.label + (f' ({m.role})' if m.role else '')
                          for m in s.modes),
                str(sum(len(m.points) for m in s.modes)),
                s.recorded[:10] if s.recorded else _EMPTY,
                src_cell,
            ]
            for col, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                if dangling and col == len(cells) - 1:
                    # A literal, statically-resolvable tooltip (the help
                    # inventory scan); the COUNT is already in the cell.
                    item.setToolTip(
                        'Source run(s) were removed since this set was '
                        'recorded. The snapshot survives — a recorded set is '
                        'never deleted with its sources — and its exported '
                        'provenance says how many are gone.')
                self.sets_table.setItem(row, col, item)
        self.sets_table.resizeColumnsToContents()
        # Re-select the SAME set silently; a vanished or never-made selection
        # falls through to the ordinary (signal-firing) first-row select.
        same_row = next((i for i, (_f, s) in enumerate(self._sets)
                         if s.set_id == prev_id), None)
        if same_row is not None:
            self.sets_table.selectRow(same_row)
        self.sets_table.blockSignals(False)

        if not self.sample_id:
            self.header.setText('No sample selected.')
        elif not self._sets:
            self.header.setText(
                'No recorded mode sets on this sample yet. Record one on the '
                'DLS tab\'s Mode Map (run the map, pick families, then '
                '"Record mode set").')
        else:
            self.header.setText(
                '{0} recorded mode set(s). Select one, then Partition.'
                .format(len(self._sets)))
        has = bool(self._sets)
        self.partition_button.setEnabled(has)
        self.fit_button.setEnabled(has)
        if not has:
            # Nothing computed can describe a sample with no recorded set
            # (removing the last one lands here): the one clear body.
            self._clear_computed()
        elif same_row is None and self.sets_table.currentRow() < 0:
            self.sets_table.selectRow(0)
        elif same_row is None and self.sets_table.currentRow() >= 0:
            # the rebuild moved the selection off the remembered set (it was
            # removed): treat it as a set switch
            self._on_set_selected()
        self._draw()

    def _on_set_selected(self) -> None:
        sel = self._selected()
        self.mode_combo.clear()
        if sel is not None:
            for m in sel[1].modes:
                self.mode_combo.addItem(
                    m.label + (f' ({m.role})' if m.role else ''), m.label)
        # a different set's partition — and every readout of it — is stale by
        # definition (one clear body, finding 6)
        self._clear_computed()
        self._draw()

    def _on_set_menu(self, pos) -> None:
        # Rename/Remove mutate the controller's store — refused while a fit
        # is in flight, like every mutating sibling handler (finding 4).
        if runner().is_busy:
            busy_notice(self)
            return
        sel = self._selected()
        if sel is None:
            return
        frac, s = sel
        menu = QtWidgets.QMenu(self)
        act_rename = menu.addAction('Rename…')
        act_remove = menu.addAction('Remove…')
        chosen = menu.exec(self.sets_table.viewport().mapToGlobal(pos))
        if chosen is act_rename:
            new, ok = QtWidgets.QInputDialog.getText(
                self, 'Rename mode set', 'New label:', text=s.label)
            if ok and new.strip():
                self.controller.rename_mode_set(self.sample_id, frac,
                                                s.set_id, new.strip())
                self._refresh()
        elif chosen is act_remove:
            # Explicit button set + a No default (R7.4): Enter must not
            # destroy a recorded set.
            if QtWidgets.QMessageBox.question(
                    self, 'Remove mode set',
                    'Remove the recorded mode set “{0}” and its computed '
                    'partition results? The picks on the Mode Map are not '
                    'touched.'.format(s.label),
                    QtWidgets.QMessageBox.StandardButton.Yes
                    | QtWidgets.QMessageBox.StandardButton.No,
                    QtWidgets.QMessageBox.StandardButton.No) \
                    == QtWidgets.QMessageBox.StandardButton.Yes:
                self.controller.remove_mode_set(self.sample_id, frac,
                                                s.set_id)
                # No hand-clear here: `_refresh` drops the computed state
                # through the one clear body when the removed set was the shown
                # one (a reselect) or the last one (nothing left to describe).
                self._refresh()

    def _on_open_map(self) -> None:
        if self.sample_id:
            self.modeMapFocusRequested.emit(self.sample_id)

    # ---------------------------------------------------------- actions ---
    def _on_partition(self) -> None:
        # Writes controller.results and reads workspace rows the worker may
        # be mutating — refused while a job is in flight (finding 4). The
        # partition itself stays synchronous: it is the fast-ops class (the
        # same cost as a live mask-edit refit).
        if runner().is_busy:
            busy_notice(self)
            return
        sel = self._selected()
        if sel is None or not self.sample_id:
            return
        frac, s = sel
        try:
            self._parts = self.controller.run_mode_partition(
                self.sample_id, s.set_id, fraction=frac)
        except (ValueError, KeyError) as exc:
            set_flag(self.flag_label, humanize_field_names(str(exc)))
            self._parts = {}
            self.export_button.setEnabled(False)
            self.status.clear()
            return
        self.flag_label.clear()
        self.conc_combo.blockSignals(True)
        self.conc_combo.clear()
        for c in sorted(self._parts):
            self.conc_combo.addItem(conc_mg_ml_label(c), c)
        self.conc_combo.blockSignals(False)
        self.export_button.setEnabled(True)
        self.status.setText(
            'Partitioned {0} concentration(s) by “{1}”.'.format(
                len(self._parts), s.label))
        self._show_partition()

    def _show_partition(self) -> None:
        c = self.conc_combo.currentData()
        part = self._parts.get(c) if c is not None else None
        if part is None:
            self.part_table.setRowCount(0)
            self.part_table.setColumnCount(0)
            self._draw()
            return
        labels = [pm.label for pm in part.per_mode]
        cols = (['θ (°)', 'Σ fractions']
                + ['dR [{0}] (cm⁻¹)'.format(x) for x in labels])
        self.part_table.setColumnCount(len(cols))
        self.part_table.setHorizontalHeaderLabels(cols)
        n = len(part.angles_deg)
        self.part_table.setRowCount(n)
        sig = self.controller.settings.no_uncertainty_sig_figs
        for i in range(n):
            vals = [format_fixed_sig(float(part.angles_deg[i]), 4),
                    (_EMPTY if not np.isfinite(part.closure_fraction[i])
                     else format_fixed_sig(float(part.closure_fraction[i]),
                                           sig))]
            for pm in part.per_mode:
                v = float(pm.rayleigh.excess_rayleigh_cm_inv[i])
                vals.append(_EMPTY if not np.isfinite(v)
                            else format_fixed_sig(v, sig))
            for col, text in enumerate(vals):
                self.part_table.setItem(
                    i, col, QtWidgets.QTableWidgetItem(text))
        self.part_table.resizeColumnsToContents()
        set_flag(self.notes_label, ' '.join(part.notes), problem=False)
        self._draw()

    def _on_fit(self) -> None:
        # Worker-dispatched: the zimm/berry branch is the same fit the
        # Analysis sub-tab classifies as worker work, and the run writes
        # controller.results — never concurrent with another job.
        sel = self._selected()
        if sel is None or not self.sample_id:
            return
        frac, s = sel
        mode = self.mode_combo.currentData()
        method = self.method_combo.currentData()
        if mode is None:
            return
        c = self.conc_combo.currentData()
        if method in ('amplitude_guinier', 'guinier', 'debye') and c is None:
            # Pre-empted HERE with the user's actual remedy — never the
            # controller's API prose (finding 12).
            self.status.setText(
                'Press Partition first, then choose a concentration in the '
                'Concentration box.')
            return
        kw = {}
        if method in ('amplitude_guinier', 'guinier', 'debye'):
            kw['concentration_g_per_mL'] = c
        sid = self.sample_id
        set_id = s.set_id

        def work():
            return self.controller.run_mode_static(
                sid, set_id, mode, method, fraction=frac, **kw)

        def done(res) -> None:
            cur = self._selected()
            if (self.sample_id != sid or cur is None
                    or cur[1].set_id != set_id):
                return                       # superseded by a switch mid-run
            self._present_fit(res, mode, method, frac, set_id, conc=c)

        def fail(exc: BaseException) -> None:
            cur = self._selected()
            if (self.sample_id != sid or cur is None
                    or cur[1].set_id != set_id):
                # Superseded by a switch mid-run (the `done` guard): the readout now
                # shows another set, so the failure is told, never painted over it.
                who = self.controller.workspace.samples.get(sid)
                QtWidgets.QMessageBox.warning(
                    self, 'Mode-resolved analysis failed',
                    f'Could not run {method!r} on {_sample_label(who) if who else sid} '
                    f'(set {set_id}).\n\n{humanize_field_names(str(exc))}')
                return
            set_flag(self.flag_label, humanize_field_names(str(exc)))
            self.result_label.clear()

        if not runner().try_submit(work, done, fail,
                                   description='mode-resolved analysis',
                                   busy_widgets=(self.fit_button,)):
            busy_notice(self)

    def _present_fit(self, res, mode: str, method: str,
                     frac: Optional[str], set_id: str,
                     conc: Optional[float] = None) -> None:
        """Show a completed mode-resolved fit. `conc` is the concentration a
        per-concentration method (amplitude Guinier, Guinier, Debye) ran at: the
        readout names it, and `_draw` flags the fit when the Concentration
        selector -- which re-fills the partition table beside it -- moves off it."""
        self._parts = self.controller.results.get(
            ('mode_partition', self.sample_id, frac, set_id), self._parts)
        sig = self.controller.settings.no_uncertainty_sig_figs
        flag = 'apparent, mode-resolved'
        caution = ''
        self._guinier_conc = conc          # every per-concentration method, not one
        if method == 'amplitude_guinier':
            self._guinier = res
            rows = [
                'R<sub>g,app</sub> ({0}): <b>{1} nm</b>'.format(
                    mode, format_pm(res.rg_app_nm, res.rg_se)),
                'R² = {0}; q·R<sub>g</sub> max = {1} (limit {2:g})'.format(
                    format_fixed_sig(res.r_squared, sig),
                    format_fixed_sig(res.qrg_max, sig), res.qrg_max_valid),
                'resolution floor ≈ {0} nm (from this q-range)'.format(
                    format_fixed_sig(res.rg_resolution_floor_nm, 2)),
            ]
            if conc is not None:
                rows.append('fitted at c = {0}'.format(conc_mg_ml_label(conc)))
            self.view_combo.setCurrentIndex(1)
        else:
            # Mw/A₂ go through the tab's ONE dash-the-value body and the
            # numerical-reliability caution, exactly as the Analysis sub-tab's
            # cells do — WITH their ± via format_pm (the PDG ±-sets-sig-figs
            # policy). A ±-less value
            # falls back to the no-uncertainty sig-figs knob, never
            # format_pm's own 4-sig-fig default. Rg is slope/intercept,
            # where the scale cancels — never dashed.
            def _pm(v: float, se) -> str:
                if se is not None and np.isfinite(se) and se > 0:
                    return format_pm(v, se)
                return format_fixed_sig(v, sig)

            rows = []
            mw = getattr(res, 'mw_g_per_mol',
                         getattr(res, 'mw_apparent_g_per_mol', float('nan')))
            mw_se = getattr(res, 'mw_se',
                            getattr(res, 'mw_apparent_se', None))
            mw_txt = _scaled_value(
                res, ('{0} g/mol'.format(_pm(mw, mw_se))
                      if np.isfinite(mw) else 'n/a'))
            rows.append('M<sub>w</sub> (mode, apparent-class): ' + mw_txt)
            rg = getattr(res, 'rg_nm', getattr(res, 'rg_apparent_nm',
                                               float('nan')))
            if np.isfinite(rg):
                rg_se = getattr(res, 'rg_se',
                                getattr(res, 'rg_apparent_se', None))
                rows.append('R<sub>g</sub> (mode): {0} nm'.format(
                    _pm(rg, rg_se)))
            a2 = getattr(res, 'a2_mol_mL_per_g2', None)
            if a2 is not None:
                a2_se = getattr(res, 'a2_se', None)
                rows.append('A₂ (mode, apparent-class): ' + _scaled_value(
                    res, ('{0} mol·mL/g²'.format(_pm(a2, a2_se))
                          if np.isfinite(a2) else 'n/a')))
            if conc is not None:
                rows.append('fitted at c = {0}'.format(conc_mg_ml_label(conc)))
            caution = _mw_unreliable_caution(res)
            if caution:
                flag += ' — ' + caution
        # The tier ORs every condition the Analysis sub-tab ORs into its own `problem`
        # (R5.2: one condition, one tier, across surfaces): an unknown scale (which
        # `_mw_unreliable_caution` deliberately yields to), the Mw caution, and the size
        # caution -- the amplitude-Guinier fit carries `rg_reliable` too (round 1 f1).
        rg_caution = _rg_unreliable_caution(res)
        if rg_caution:
            flag += ' — ' + rg_caution
        if _scale_causes(res):
            flag += ('  ' + _scale_phrase(res).capitalize()
                     + ' → Mw unreliable (Rg survives).')
        problem = bool(_scale_causes(res)) or bool(caution) or bool(rg_caution)
        # The estimator travels with the ± it qualifies (invariant 8 clause A), in the
        # DDLS tab's words -- and only when a ± is rendered: a withheld SE leaves no ±
        # for the marker to describe (round 1 f6).
        if any(getattr(res, k, None) is not None
               for k in ('rg_se', 'mw_se', 'mw_apparent_se', 'a2_se', 'rg_apparent_se')):
            flag += _ols_marker(getattr(res, 'se_estimator', 'hc3'))
        # the neutral 'apparent, mode-resolved' qualifier is ⓘ (R5.3); only a
        # genuine caution makes the label red -- and the tier + glyph come
        # from the one helper (R5.1)
        set_flag(self.flag_label, flag, problem=problem)
        self.result_label.setText('<br>'.join(rows) if rows else
                                  'No extractable value — see the notes.')
        notes = list(getattr(res, 'notes', ()) or ())
        rel = getattr(res, 'reliability_note', '')
        if rel:
            notes.append(rel)
        set_flag(self.notes_label, ' '.join(str(x) for x in notes), problem=False)
        self._draw()

    _GUINIER_CONC_HINT = 'The mode-resolved fit shown is for c = {0} — press ' \
                         'Run to fit c = {1}.'

    def _flag_guinier_conc(self, conc) -> None:
        """The Guinier view beside a partition table that follows the Concentration
        selector: when the selector moves off the fit's own concentration the plot
        stays and the status line says which c it is for (R8.5); back on it, the
        hint goes."""
        stale = (self._guinier_conc is not None and conc is not None
                 and conc != self._guinier_conc)
        hint_shown = self.status.text().startswith(
            self._GUINIER_CONC_HINT.split('{0}')[0])
        if stale:
            self.status.setText(self._GUINIER_CONC_HINT.format(
                conc_mg_ml_label(self._guinier_conc), conc_mg_ml_label(conc)))
        elif hint_shown:
            self.status.clear()

    def _on_export(self) -> None:
        # Recomputes the partition + fits at export time — refused while a
        # job is in flight, like every computing sibling (finding 4).
        if runner().is_busy:
            busy_notice(self)
            return
        sel = self._selected()
        c = self.conc_combo.currentData()
        if sel is None or c is None or not self.sample_id:
            return
        frac, s = sel
        name = sample_suggested_name(self.controller, self.sample_id,
                                     'mode_partition')
        status = export_to_csv(
            self, self.controller, name,
            lambda p: self.controller.export_mode_partition_csv(
                self.sample_id, frac, s.set_id, p,
                concentration_g_per_mL=c))
        if status:
            self.status.setText(status)

    # ------------------------------------------------------------ plot ---
    def _draw(self) -> None:
        view = self.view_combo.currentData()
        c = self.conc_combo.currentData()
        part = self._parts.get(c) if c is not None else None
        self._flag_guinier_conc(c)        # both views: the readout beside the table
        if view == 'guinier':
            g = self._guinier
            if g is None:
                show_plot_placeholder(
                    self.figure, self.canvas,
                    'Run the Amplitude Guinier analysis on a mode.',
                    clear_figure=True)
                self.ax = None
                self.axis_bar.attach(None)
                return
            self.figure.clf()
            self.ax = self.figure.add_subplot(111)
            self.ax.plot(g.q2_nm2, g.inv_amplitude, 'o', label='1/dR (mode)')
            xs = np.linspace(float(g.q2_nm2.min()), float(g.q2_nm2.max()), 50)
            self.ax.plot(xs, g.intercept + g.slope * xs, '-', label='fit')
            self.ax.set_xlabel(r'$q^2$ (nm$^{-2}$)')
            self.ax.set_ylabel('1 / dR (mode)')
            self.ax.legend(fontsize=8)
        else:
            if part is None:
                show_plot_placeholder(
                    self.figure, self.canvas,
                    'Select a recorded mode set, then press Partition.',
                    clear_figure=True)
                self.ax = None
                self.axis_bar.attach(None)
                return
            self.figure.clf()
            self.ax = self.figure.add_subplot(111)
            self.ax.plot(part.angles_deg, part.excess_total_cm_inv, 'k--',
                         label='total dR')
            for pm in part.per_mode:
                self.ax.plot(part.angles_deg,
                             pm.rayleigh.excess_rayleigh_cm_inv, 'o-',
                             label='dR [{0}]'.format(pm.label))
            self.ax.set_xlabel(r'Scattering angle $\theta$ (°)')
            self.ax.set_ylabel(r'Excess Rayleigh ratio dR (cm$^{-1}$)')
            self.ax.legend(fontsize=8)
        self.figure.tight_layout()
        self.canvas.draw_idle()
        self.axis_bar.attach(self.ax)


# ===========================================================================
# The SLS tab shell
# ===========================================================================
class SLSModule(QtWidgets.QWidget):
    """The SLS tab: analysis beside the surface that derives static points.

    A thin shell, like :class:`gui.dls_module.DLSModule` -- it owns the sub-tabs
    and forwards the small API the application shell drives them through. All the
    analysis lives in :class:`_AnalysisTab`.
    """

    committed = QtCore.Signal()
    selectionChanged = QtCore.Signal()
    sampleFocusRequested = QtCore.Signal(str)
    workspaceChanged = QtCore.Signal()
    modeMapFocusRequested = QtCore.Signal(str)   # -> DLS tab's Mode Map

    def __init__(self, controller, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = roomy_tabs(QtWidgets.QTabWidget())
        self.analysis = _AnalysisTab(controller)
        self.derived = _DerivedIntensityTab(controller)
        self.mode_resolved = _ModeResolvedTab(controller)
        self.tabs.addTab(self.analysis, 'Analysis')
        self.tabs.addTab(self.derived, 'Derived Intensity')
        self.tabs.addTab(self.mode_resolved, 'Mode-Resolved')
        self.tabs.setMovable(True)
        layout.addWidget(self.tabs, 1)

        self.analysis.committed.connect(self.committed)
        self.analysis.selectionChanged.connect(self.selectionChanged)
        self.analysis.sampleFocusRequested.connect(self.sampleFocusRequested)
        # A derivation creates measurements, so the shell must rebuild the sidebar
        # and the Data table -- and this tab's own candidate list is now stale too
        # (the new SLS point does not appear in it, but a re-derive should re-read
        # the runs it came from).
        self.derived.workspaceChanged.connect(self.workspaceChanged)
        self.derived.workspaceChanged.connect(self._on_derived)
        # Cross-tab navigation from the Mode-Resolved list to where mode sets
        # are DEFINED (never a duplicated view) -- the main window routes it.
        self.mode_resolved.modeMapFocusRequested.connect(
            self.modeMapFocusRequested)

    # -- the API the application shell drives --------------------------------
    @property
    def sample_id(self) -> Optional[str]:
        """The analyzed sample (the analysis sub-tab owns it)."""
        return self.analysis.sample_id

    def set_measurement(self, item_id: Optional[str]) -> None:
        for tab in (self.analysis, self.derived, self.mode_resolved):
            tab.set_measurement(item_id)

    def show_active_sample(self, sid: Optional[str], **kw) -> None:
        self.analysis.show_active_sample(sid, **kw)
        self.derived.show_active_sample(sid)
        self.mode_resolved.show_active_sample(sid)

    def selected_item_ids(self) -> list:
        return self.analysis.selected_item_ids()

    def _on_derived(self) -> None:
        """After a derivation, re-point the analysis sub-tab at the same sample so
        the new measurements are picked up without the user re-selecting."""
        self.analysis.show_active_sample(self.analysis.sample_id)
