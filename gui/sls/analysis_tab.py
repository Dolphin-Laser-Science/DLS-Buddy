"""
The SLS Analysis sub-tab (`_AnalysisTab`): the calibration panel, the per-sample static
scattering analyses and their result table, and data masking.
"""
from __future__ import annotations

import html
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PySide6 import QtCore, QtWidgets

from core.notes import find_notes, notes_in
from core.naming import sample_display_name
from plotting.plots import (plot_zimm, plot_debye, plot_guinier, plot_rayleigh_ratio,
                            plot_calibration_free_a2, plot_single_angle_ladder, plot_form_factor,
                            plot_ellipsoid_form_factor, plot_core_shell_form_factor,
                            conc_mg_ml_label, display_factor_from, zimm_spacing)
from analysis import form_factor as ff
from gui.plot_controls import (make_split_panels, make_vertical_plot_stack, pin_under_column,
                               set_plot_export_name)
from gui.export_helper import export_to_csv, sample_suggested_name
from gui.field_labels import humanize_field_names, missing_input_lines
from gui.help import add_help_to_groupbox
from gui.notes_block import MARKER_GLYPH, NotesBlock
from gui.plot_overlays import reissue_legend_preserving_title, show_plot_placeholder
from gui.widgets import (STALE_HINT, STALE_UNRUNNABLE, GripSplitter, RichTextDelegate,
                         SampleSelector, confirm_regroup_losses, make_plot_pane)
from gui.theme import ThemedLabel, set_flag, span, token as theme_token
from gui.worker import BACKGROUND_RUN_TOOLTIP, BUSY_NOTICE, RunSlot, busy_notice, runner
from core.formatting import format_fixed_sig, format_pm

from gui.sls.common import (_scale_causes, _scaled_value, _RG_MW_QUALIFIERS,
                            _confirm_unmeasured_blank)
from gui.sls.rows import _Row, _value_row, _append_label_tip, _diagnostic_row, _shape_solution_tail


# An absolute Mw whose scale is unknown is not a meaningful number. Rendered AT the value
# so the caveat travels with it; Rg and the calibration-free 2·A₂·Mw product survive and
# stay shown. TWO independent SCALE causes, both able to hold at once: no calibration
# constant (ΔR arbitrary) and no dn/dc (K = 1, so Kc/ΔR arbitrary). The dash names which,
# because the remedies differ — measure a calibrant, versus supply an increment. For a
# STORED Mw `_mw_scale_dash` adds a third, non-scale cause (a non-finite value);
# `_UNCAL_MW` stays the no-cause-available spelling.
_UNCAL_MW = '— (uncalibrated)'


# The calibration-free A₂ divides the (genuinely calibration-free) product 2·A₂·Mw by the
# sample's stored Mw, so the absolute A₂ inherits that Mw's scale. When the controller
# refuses an arbitrary-scale Mw the value is withheld — dashed AT the value, same
# convention as _UNCAL_MW, because 'set a manual Mw to get A₂' would be a LIE here: there
# IS an Mw, the program declined it. PLAIN text, unlike the row LABEL beside it: the
# results table applies `RichTextDelegate` to column 0 only, so markup in a VALUE cell
# renders as literal '<sub>' tags. `_UNCAL_MW` is plain for the same reason.
_UNCAL_A2 = '— (Mw on an arbitrary scale)'


# Calculation nuance on the withheld A₂ cell, a passive tooltip; the scale cause itself is
# the notes block's line, read off the stored Mw (`SampleResult.scale_notes`).
_WITHHELD_A2_TOOLTIP = (
    'The absolute A₂ is 2·A₂·Mw divided by Mw. The product is a ratio of excess '
    'intensities, so the unknown scale cancels in it; an Mw on an arbitrary scale (no '
    'calibration, or no usable dn/dc) puts the scale back, so A₂ is withheld. A '
    'hand-entered Mw (Manual Mw) is trusted and yields A₂. '
    'See the Theory-and-Equations-Guide §4.5.')


# Calculation nuance on the ellipsoid's axial-ratio row label, a passive tooltip after the
# clauses its qualifier words carry: when the ± is shown, and that a marker says the rest.
_AXIAL_RATIO_TIP = (
    'The axial ratio carries its ± whenever the angles resolve it, a wide one included, and '
    'none only where they cannot tell the shapes apart; a marker names why it is flagged. '
    'See the Theory-and-Equations-Guide §4.9.3.')


# Shown when a redisplayed (cached) SLS fit no longer matches its committed params /
# mask / calibration / estimator — the same explicit-Run + staleness-hint contract the
# DLS tabs use, in the same wording.
_SLS_STALE_HINT = STALE_HINT


# The same hint for a fit kept on screen while the selected fit cannot run: Run is off
# then, so the hint must not ask for it.
_SLS_STALE_UNRUNNABLE = STALE_UNRUNNABLE


# Display names for the quantities a Zimm/Berry run writes back (§9 casing).
_QTY_DISPLAY = {'mw': 'Mw', 'rg': 'Rg', 'a2': 'A₂', 'rh': 'Rh'}


def _adapt_form(form: QtWidgets.QFormLayout) -> None:
    """Let a form adapt to a narrow control panel instead of clipping:
    fields grow to the available width and a too-wide row wraps its field below the
    label rather than overflowing the panel."""
    form.setFieldGrowthPolicy(
        QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)


# Method dropdown entries: (label, engine key), in menu order. The single-angle Mw at ONE
# concentration ('single') is deliberately UNLISTED rather than removed: the engine and
# controller paths behind it are kept, so re-listing it is one line here plus 'single'
# back in tests/test_gui_smoke.py's _SWEEP_METHODS, whose equality gate reads this list. One angle
# at one concentration is a single datum -- no Rg, and no honest uncertainty (CLAUDE.md
# invariant 8) -- while the single-angle LADDER below covers the realistic case, a
# concentration series read at one angle.
_SLS_METHODS: List[Tuple[str, str]] = [
    ('Zimm', 'zimm'),
    ('Berry', 'berry'),
    ('Form factor P(q)', 'form_factor'),
    ('Debye (single c, apparent)', 'debye'),
    ('Guinier (single c, apparent)', 'guinier'),
    ('Single-angle Mw + A₂ (ladder, apparent)', 'ladder'),
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


# Calibration line-edit fields, in two captioned blocks that mirror
# k_c = R_std / ((I - dark) * sin(theta)): the user's own reading, and the standard whose
# Rayleigh ratio and refractive index the program looks up. The geometry combo closes the
# standard block. Each field: (attribute, label, allow_blank_as_None).
_CAL_BLOCKS: List[Tuple[str, List[Tuple[str, str, bool]]]] = [
    ('Your Calibrant Reading', [
        ('calibrant_intensity', 'Calibrant intensity', True),
        ('dark_count_rate', 'Dark count rate (0 if already net)', True),
        ('calibrant_angle_deg', 'Calibrant angle (°)', False),
    ]),
    ('The Standard', [
        ('standard_wavelength_nm', 'Standard wavelength (nm)', False),
        ('standard_temperature_C', 'Standard temperature (°C)', False),
        ('standard_refractive_index', 'Standard refractive index', True),
    ]),
]


_CAL_FIELDS = [f for _caption, block in _CAL_BLOCKS for f in block]


# What a blank means, for each field that may be left blank.
_CAL_PLACEHOLDERS = {
    'calibrant_intensity': '(blank = uncalibrated)',
    'dark_count_rate': '(blank = 0)',
    'standard_refractive_index': '(blank = none)',
}


# Intensity fields share the SLS file's own (instrument-specific) intensity unit;
# the program never converts them because only intensity RATIOS enter the analysis.
_CAL_TOOLTIPS = {
    'calibrant_intensity':
        'Only its ratio to the sample intensities enters the Rayleigh ratio, so it '
        'is in your SLS file\'s own units (cps, kcps, …) and the program does not '
        'convert it.',
    'dark_count_rate':
        'Subtracted from every intensity, in your SLS file\'s units. With a measured '
        'solvent blank it cancels in the subtraction, so it changes only '
        'k<sub>c</sub>, and with it Mw and absolute A₂; without one (a typed 90° '
        'blank, or none) it also changes the excess Rayleigh ratio, differently at '
        'each angle.',
    'calibrant_angle_deg':
        'Scattering angle at which the calibrant intensity was measured.',
    'standard_wavelength_nm':
        'Laser wavelength used for the calibration standard (normally the same as '
        'the sample measurement). The toluene Rayleigh ratio is tabulated at 532 and '
        '660 nm; a wavelength within 2 nm of one is served that entry and the note '
        'beside k_c says so, and a mismatch with the loaded SLS data is noted there too.',
    'standard_temperature_C':
        'Temperature of the calibration standard; it sets the standard’s Rayleigh '
        'ratio (toluene’s is temperature-dependent). See the '
        'Theory-and-Equations-Guide §1.6.2.',
    'standard_refractive_index':
        'The standard\'s refractive index at this wavelength and temperature; it enters '
        'the (n_solvent/n_standard)² correction. Blank where the library has no value, '
        'and then the correction cannot be applied: Mw and absolute A₂ are flagged "no '
        'standard refractive index" (or "uncalibrated" without a k<sub>c</sub>), and Rg '
        'is unaffected (Theory-and-Equations-Guide §1.7.1).',
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
        # The Run's slot: a busy press is refused; its epoch is the async staleness
        # token, bumped by each Run and by a sample or fraction switch.
        self._run_slot = RunSlot()
        # The active molecular-weight fraction (None = unfractioned / whole sample).
        self._fraction: Optional[str] = None
        self._suppress_fraction = False
        # Last-run memory (within-session) keyed by (sample_id, fraction): which
        # method + axis points were last analyzed, so switching samples/fractions
        # and back restores the view. SLS re-runs cheaply from committed state (+
        # persisted masks), so we replay the run rather than caching result objects.
        self._last_run_by_sample: Dict[tuple, Dict] = {}
        # Which fits the shown sample can run (`Controller.sls_fit_availability`, keyed
        # as `SLS_FIT_REQUIREMENTS`), the status line the Run gate last wrote, and each
        # dropdown entry's own tooltip, restored when the entry is available again.
        self._availability: Dict[Any, Any] = {}
        self._gate_status = ''
        self._entry_tips: Dict[tuple, Any] = {}
        self.ax = None
        self._build_ui()
        self._populate_calibration()
        self._refresh_calibration_display()
        self.set_measurement(None)

    # ------------------------------------------------------------------ UI ---
    def _build_ui(self) -> None:
        # Control panel | plot split is draggable. The control column
        # has long calibration labels and a wide button, so it keeps a minimum width.
        splitter, left, right = make_split_panels(self, left_min_width=340)

        # The SLS dropdown is a mirror/override of the shell's single active sample
        # picking here sets the active sample everywhere; clicking a sample
        # in the sidebar drives this too. A whole Zimm set (solvent ref + concentration
        # series) is analyzed per sample.
        self.sls_selector = SampleSelector(
            self.controller, predicate=lambda s: s.has_sls,
            label_fn=sample_display_name, title='SLS Sample',
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
        left.addStretch(1)      # keep the controls top-aligned; scroll area handles overflow

        # Run, Export, the status line and the ⚠ flag are pinned under the scroll area
        # (the §3 column template), so the Calibration group above the Analysis group
        # can never carry them out of view.
        self.status = QtWidgets.QLabel('')        # one-line notes (e.g. Rayleigh series)
        self.status.setWordWrap(True)
        self.flag_label = ThemedLabel('', role='error', bold=True)
        self.flag_label.setWordWrap(True)
        pin_under_column(splitter, self.run_button, self.export_button, self.status,
                         self.flag_label)

        # Right column: the plot over a draggable Results | Mask strip. Keeping the
        # results table next to the plot it describes (and the mask near the points you
        # click) declutters the left column; the Cross-Sample tab sets the same precedent
        # of content in the right column.
        self.figure, self.canvas, self.nav_toolbar, self.axis_bar = make_plot_pane(
            self, self.controller, (5.5, 4.6))
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
        # The ± clause the table's ± rows share, said once above them (invariant 8's label).
        self.pm_caption = ThemedLabel('', 'qualifier', hide_when_empty=True)
        self.pm_caption.setWordWrap(True)
        rc.addWidget(self.pm_caption)
        rc.addWidget(self.result_table, 1)
        # One line per failed precondition of the fit the table shows, pinned under it
        # (R8.6); a marked cell opens the same popup on click.
        self.notes_block = NotesBlock(what='solutions')
        self.notes_block.watch(self.result_table)
        rc.addWidget(self.notes_block)
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
                             'ratios, which Mw and A₂ need.', bullets=[
                                 '<b>Your calibrant reading</b>: its intensity and dark '
                                 'count rate in your SLS file\'s units (dark 0 if net), '
                                 'and its angle. A blank intensity runs uncalibrated.',
                                 '<b>The standard</b>: <b>k<sub>c</sub></b> comes from '
                                 'your reading and its wavelength, temperature and '
                                 'geometry; its refractive index enters the '
                                 'solvent-to-standard correction. Don\'t use a vendor '
                                 'constant (it goes stale after alignment).',
                                 'A <b>teal dot</b> marks the solvent library\'s '
                                 'proposed refractive index; a value you type is never '
                                 'overwritten, and clearing the field restores the '
                                 'proposal.',
                                 'Calibrate once <b>session-wide</b>; tick '
                                 '"Per-sample calibration" to override one sample.',
                                 'Uncalibrated, Rg and the 2·A₂·Mw product survive; Mw '
                                 'and absolute A₂ are flagged.',
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
        for caption, block in _CAL_BLOCKS:
            # A block's caption is a header row spanning the form, the way the
            # Settings form captions its sections (`SettingsModule._header`).
            form.addRow(ThemedLabel(caption, role='header', bold=True,
                                    extra='margin-top:8px;'))
            for key, label, allow_blank in block:
                edit = QtWidgets.QLineEdit()
                if allow_blank:
                    edit.setPlaceholderText(_CAL_PLACEHOLDERS[key])
                edit.editingFinished.connect(lambda k=key: self._on_cal_edit(k))
                self.cal_edits[key] = edit
                # Intensities (calibrant + dark) must be in WHATEVER unit the SLS file
                # uses: only their ratio to the file's sample intensities enters the
                # Rayleigh ratio, so the program can't convert them to a fixed unit.
                tip = _CAL_TOOLTIPS.get(key)
                if tip:
                    edit.setToolTip(tip)
                if key == 'standard_refractive_index':
                    # The provenance dot the Data tab paints on a library-proposed
                    # cell, on a form line-edit: a leading label beside the field,
                    # shown while the value is the library's (`_populate_calibration`),
                    # hidden for a typed one -- one hue for "proposed", the tooltip
                    # carrying the source (ui_style_guide, provenance dot).
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

        # The standard block's last row, after its refractive index.
        self.geometry_combo = QtWidgets.QComboBox()
        self.geometry_combo.addItems(['VU', 'VV', 'VH'])
        self.geometry_combo.setToolTip(
            'Scattering geometry of the CALIBRANT (the toluene standard) — it sets which '
            'toluene Rayleigh ratio is used: VV = polarized, VH = depolarized, VU = no '
            'analyzer. Distinct from the sample anisotropy correction below. '
            'See the Theory-and-Equations-Guide §1.6.')
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
            'Display-only: shows the isotropic-corrected Mw (f·Mw, f = 1 − 4ρv/3) from '
            'the sample’s own depolarization ratio ρv, not the calibrant geometry above. '
            'Not applied or exported; enter it as a Manual Mw to use it. ρv, δ² and f '
            'always show, f·Mw only on an absolute M<sub>w</sub> scale '
            '(Theory-and-Equations-Guide §4.7).')
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
        # The depolarization result's own notes (a ρv outside the physical range), with
        # the readout they mark.
        self.aniso_notes = NotesBlock()
        bform.addRow('', self.aniso_notes)

        form.addRow(self.aniso_body)
        self.aniso_body.setVisible(False)

    def refresh_anisotropy_check(self) -> None:
        """Re-evaluate the display-only readout — call after a run/sample change so the
        f·Mw line tracks the current analyzed Mw. Public so the run/redisplay paths can
        refresh it without reaching into the private handler."""
        self._refresh_anisotropy_display()

    def _build_analysis_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox('Analysis')
        add_help_to_groupbox(box, 'Get Mw, Rg and A₂ from the intensities (User-Manual '
                             '§9.1).', bullets=[
            '<b>Zimm</b>: extrapolates to zero angle and concentration; thermodynamic '
            'Mw, Rg and A₂ at small qR<sub>g</sub>.',
            '<b>Berry</b>: the same on a square-root axis, placed at qR<sub>g</sub> near '
            '1.',
            '<b>Form factor P(q)</b>: fits the chosen <b>P(q) model</b> to the whole '
            'curve; apparent at one concentration, thermodynamic over several.',
            '<b>Debye</b>: the chosen <b>Concentration</b> across angle; apparent Mw and '
            'Rg.',
            '<b>Guinier</b>: that concentration as log ΔR against q²; Rg only, '
            'surviving an uncalibrated run.',
            '<b>Single-angle Mw + A₂ (ladder)</b>: the concentrations at the chosen '
            '<b>Angle</b>; apparent Mw and A₂.',
            '<b>Calibration-free A₂</b>: 2·A₂·Mw at the chosen <b>Angle</b>, no '
            'calibration needed; A₂ itself needs an absolute Mw.',
            '<b>Excess Rayleigh ratio</b>: ΔR against angle for each concentration, '
            'nothing fitted.',
            'Only <b>Run</b> computes a fit; a mask edit refits. A grayed entry\'s '
            'tooltip names what it lacks.',
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
            'Only a double extrapolation (zero angle and concentration) is '
            'thermodynamic. Rg and 2·A₂·Mw survive an uncalibrated run; absolute Mw and '
            'A₂ do not. Past the Settings qRg limit, Zimm, Debye and Guinier Rg and Mw '
            'are flagged, Berry\'s labeled. On two concentrations the Berry ± reads high: '
            'about 1.1× at twelve angles, 2× at three. '
            'Theory-and-Equations-Guide §4.3, §1.7.1, §4.4, §6.1.')
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
            'Sphere: a solid homogeneous sphere; returns the geometric radius R and Rg. '
            'Gaussian coil: a flexible chain; returns Rg. '
            'Ellipsoid: Rg and an axial ratio; a prolate and an oblate shape can fit '
            'equally, and then every solution is listed, flagged degenerate. '
            'Mie sphere: the full Mie theory, for high contrast or larger particles where '
            'the sphere fails; needs the solute refractive index (Data tab). '
            'Core–shell sphere: a coated particle with two radii; needs the solute (core) '
            'and shell refractive indices (Data tab); several pairs can fit, each listed, '
            'flagged degenerate. Several solutions are listed smallest first (axial ratio, '
            'or outer radius), a fixed convention, not a ranking. '
            'See the Theory-and-Equations-Guide §4.8–4.11.')
        self.model_label = QtWidgets.QLabel('P(q) model:')
        form.addRow(self.model_label, self.model_combo)
        self.model_combo.currentIndexChanged.connect(self._on_method_changed)
        # Read-only solute-RI reminder, shown only for the Mie model (which forms the contrast
        # m = n_solute/n_solvent). Editing happens on the Data tab (invariant 5).
        self.mie_solute_ri_hint = ThemedLabel('', role='hint', size=10)
        self.mie_solute_ri_hint.setWordWrap(True)
        form.addRow('', self.mie_solute_ri_hint)
        self._mie_solute_ri_hint_label = form.labelForField(self.mie_solute_ri_hint)

        # Debye and Guinier fit the chosen concentration's angles, the ladder and the
        # calibration-free A₂ the concentrations at the chosen angle, so a choice here
        # can change which fits the sample can run.
        self.conc_combo = QtWidgets.QComboBox()
        self.conc_combo.currentIndexChanged.connect(self._refresh_fit_availability)
        form.addRow('Concentration:', self.conc_combo)
        self.angle_combo = QtWidgets.QComboBox()
        self.angle_combo.currentIndexChanged.connect(self._refresh_fit_availability)
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
        # Run and Export are pinned under the column in `_build_ui`, not form rows.
        self.export_button = QtWidgets.QPushButton('Export CSV…')
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._on_export)

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
            self.aniso_notes.show_notes(None)
            return
        rho_v = self._read_anisotropy_rho_v()
        sig = self.controller.settings.no_uncertainty_sig_figs
        if rho_v is None:
            self.aniso_notes.show_notes(None)
            self.aniso_readout.setRole('hint')
            self.aniso_readout.setText(
                'Enter ρ<sub>v</sub> (or VV/VH) to see the corrected M<sub>w</sub>.')
            return
        try:
            dep = self.controller.compute_depolarization(rho_v=rho_v)
        except ValueError:
            dep = None
        self.aniso_notes.show_notes(dep)
        if dep is None or not dep.physically_valid:
            # Out of range, the closed forms do not apply: ρv alone, with its marker.
            mark = self.aniso_notes.marker(dep, ('rho_v',)) if dep is not None else ''
            self.aniso_readout.setRole('hint')
            self.aniso_readout.setText(
                f'ρ<sub>v</sub> = {rho_v:.3g}' + (f' {mark}' if mark else ''))
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
            # same gate). Tier stays `hint`: the dash IS the disclosure; the stored Mw's
            # scale cause is the readout's notes line, its remedy on click.
            r = self.controller.workspace.samples[self.sample_id].result_for(self._fraction)
            self.aniso_notes.show_notes(tuple(notes_in(dep)) + tuple(r.scale_notes()))
            self.aniso_readout.setText(
                parts + f'Corrected M<sub>w</sub> = f·M<sub>w</sub>: '
                f'{self._mw_scale_dash()}.')
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
                header = (f'No SLS data for '
                          f'<b>{html.escape(sample_display_name(sample))}</b> — '
                          'load an SLS intensity file, or pick another sample.')
            else:
                self.sample_id = sid
                runnable = True
                header = (f'SLS sample: '
                          f'<b>{html.escape(sample_display_name(sample))}</b>')
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
        # Last, after any redisplay: it rewrites the status line the Run gate writes to.
        self._refresh_fit_availability()
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
        has already been plotted, so toggling masks before a run is quiet). Either way
        the fits the new mask leaves runnable are read again, `_on_run` reading them
        first."""
        if self._ran and self._runnable and self.sample_id is not None:
            self._on_run(reuse_blank_answer=True)
        else:
            self._refresh_fit_availability()

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
        self.notes_block.show_notes(None)  # drop stale result notes
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
        self._refresh_fit_availability()   # after the redisplay, as in `_load_sample`

    def _populate_axis_selectors(self) -> None:
        # Signals blocked while the boxes refill: their callers read the fits'
        # availability once, at the end, rather than per intermediate choice.
        frac = self._fraction
        with QtCore.QSignalBlocker(self.conc_combo), QtCore.QSignalBlocker(self.angle_combo):
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
        self._refresh_fit_availability()   # the dark count the typed blank is judged by

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
        label = dict((k, lab) for k, lab, _b in _CAL_FIELDS)[key]
        text = edit.text().strip()
        if text == '':
            if not allow_blank:
                # Declined, so the field shows the value still in use, as a refused
                # value does below.
                self._populate_calibration()
                self.status.setText(f'{label} cannot be blank.')
                return
            value = None
        else:
            try:
                value = float(text)
            except ValueError:
                self._populate_calibration()
                self.status.setText(f'{label}: "{text}" is not a number.')
                return
        try:
            self.controller.set_calibration_field(key, value, self.sample_id)
        except ValueError as exc:
            # Entry validation, the controller's: the field goes back to the stored
            # value and the status line says why.
            self._populate_calibration()
            self.status.setText(humanize_field_names(str(exc)))
            return
        if key in ('standard_wavelength_nm', 'standard_temperature_C',
                   'standard_refractive_index', 'dark_count_rate'):
            # The controller re-proposes the standard's refractive index from the
            # solvent library at the new condition (unless the user typed their own),
            # refills a cleared index from the library and a cleared dark count with 0;
            # repopulate so the value in use, and the index's provenance dot, show in
            # the field. Safe here: this slot fires on editingFinished, not per keystroke.
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
        except Exception as exc:  # noqa: BLE001 — surfaced in the label
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
        # The committed dark count decides whether a typed 90° blank can stand in.
        self._refresh_fit_availability()
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

        `_scale_causes`' stored-value counterpart: `SampleResult` carries the same flags
        under its own spellings. Named causes matter more here than on a fit
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
            # `_scale_causes`' words for a k_c applied with the standard's index blank; a
            # None (no cause recorded) keeps 'uncalibrated', as `SampleResult.scale_notes`.
            causes.append('no standard refractive index' if r.standard_index_known is False
                          else 'uncalibrated')
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
        self.angle_combo.setEnabled(key in ('single', 'ladder', 'calfree'))
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
        hint = (self._mie_solute_ri_summary() if is_mie
                else self._core_shell_ri_summary() if is_core_shell else '')
        # Shown only while it has a value to show: a missing index is named by the grayed
        # entry's tooltip and the status line, as every other missing input is.
        self.mie_solute_ri_hint.setText(hint)
        self.mie_solute_ri_hint.setVisible(bool(hint))
        if self._mie_solute_ri_hint_label is not None:
            self._mie_solute_ri_hint_label.setVisible(bool(hint))
        self._apply_run_gate()

    def _mie_solute_ri_summary(self) -> str:
        """Read-only reminder of the current sample's committed solute RI (edited on the
        Data tab, invariant 5) — the Mie fit forms m = n_solute/n_solvent from it. Names the
        value when set, and is empty when it is not (it is never guessed — invariant 3)."""
        sid = self.sample_id
        if sid is None:
            return ''
        n = self.controller.sls_solute_refractive_index(sid, self._current_fraction())
        return '' if n is None else f'Solute refractive index: {n:g} (from the Data tab).'

    def _core_shell_ri_summary(self) -> str:
        """Read-only reminder of the current sample's committed CORE and SHELL refractive
        indices (edited on the Data tab, invariant 5) — the core-shell fit forms the two
        contrasts m1 = n_core/n_solvent, m2 = n_shell/n_solvent from them. Names each one
        set, and is empty when neither is (neither is ever guessed — invariant 3)."""
        sid = self.sample_id
        if sid is None:
            return ''
        frac = self._current_fraction()
        n_core = self.controller.sls_solute_refractive_index(sid, frac)
        n_shell = self.controller.sls_shell_refractive_index(sid, frac)
        if n_core is not None and n_shell is not None:
            return f'Core refractive index: {n_core:g}, shell: {n_shell:g} (from the Data tab).'
        if n_core is not None:
            return f'Core refractive index: {n_core:g} (from the Data tab).'
        if n_shell is not None:
            return f'Shell refractive index: {n_shell:g} (from the Data tab).'
        return ''

    # ------------------------------------------------- what the sample can run ---
    def _selected_fit_key(self):
        """The selection's `SLS_FIT_REQUIREMENTS` key: the Method, or the P(q) model's
        own key when the Method is the form factor."""
        method = self.method_combo.currentData()
        if method == 'form_factor':
            return ('form_factor', self.model_combo.currentData())
        return method

    @QtCore.Slot()
    def _refresh_fit_availability(self) -> None:
        """Gray each Method and P(q) model entry the shown sample cannot run, its tooltip
        naming every input it lacks, then gate Run on the selection. Read from committed
        parameters and the mask (`Controller.sls_fit_availability`) at the Concentration
        and Angle boxes' choice; with no sample shown every entry is enabled. A grayed
        entry stays selected if it was: nothing is hidden or moved."""
        sid = self.sample_id
        self._availability = ({} if sid is None else self.controller.sls_fit_availability(
            sid, self._fraction, concentration=self.conc_combo.currentData(),
            angle=self.angle_combo.currentData()))
        role = QtCore.Qt.ItemDataRole.ToolTipRole
        for combo in (self.method_combo, self.model_combo):
            for i in range(combo.count()):
                data = combo.itemData(i)
                key = ('form_factor', data) if combo is self.model_combo else data
                item = combo.model().item(i)
                own = self._entry_tips.setdefault((id(combo), data), item.data(role))
                avail = self._availability.get(key)
                ok = avail is None or avail.ok
                item.setEnabled(ok)
                item.setData(own if ok else '\n'.join(
                    ['Needs:', *missing_input_lines(avail.missing)]), role)
        self._apply_run_gate()

    def _apply_run_gate(self) -> None:
        """Run follows the selected fit: enabled only when the shown sample can run it,
        and otherwise the status line says what is missing. A result on screen is never
        cleared here; the status line marks it out of date when its inputs have changed
        since it ran (R8.5)."""
        if self.sample_id is None or not self._runnable:
            return
        avail = self._availability.get(self._selected_fit_key())
        ok = avail is None or avail.ok
        self.run_button.setEnabled(ok)
        cached = self._last_run_by_sample.get((self.sample_id, self._fraction))
        stale = (self._ran and cached is not None and cached['sig']
                 != self.controller.sls_run_signature(self.sample_id, self._fraction))
        if ok:
            if self._gate_status and self.status.text() == self._gate_status:
                # The gate's reason goes; a fit on screen whose inputs changed keeps its
                # out-of-date mark, now with Run on to refresh it (R8.5).
                self.status.setText(_SLS_STALE_HINT if stale else '')
            self._gate_status = ''
            return
        label = (self.model_combo.currentText()
                 if self.method_combo.currentData() == 'form_factor'
                 else self.method_combo.currentText())
        lines = [f'{label} needs:', *missing_input_lines(avail.missing)]
        if stale:
            lines.append(_SLS_STALE_UNRUNNABLE)
        self._gate_status = '\n'.join(lines)
        self.status.setText(self._gate_status)

    def _reconcile_k_field_units(self) -> None:
        """Rewrite the Zimm-k field text into the ACTIVE q² display unit when it
        changed (a Settings ▸ Display units switch), mirroring how the axis-limit
        bar's typed limits re-render — so a user-typed k stays the same physical
        constant across a unit change instead of being reinterpreted against the
        new factor (which would rescale the grid). A no-op when the unit is
        unchanged, so it is safe to call on every (re)display. Converts relative
        to the field's PREVIOUS factor (`_k_field_qfac`), since by redisplay time
        `set_plot_units` has already swapped the global factor and the old one
        can no longer be recovered from `core.units`."""
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
        except Exception:  # noqa: BLE001 — no suggestion without a series
            return
        # Suggest fills the edit in the AXIS units (matching how a typed k is
        # read back); canonical at the default nm⁻² choice, factor exactly 1.0.
        # Record the unit the field is now written in, so a later unit change
        # reconciles from the right factor.
        qfac = display_factor_from('scattering_q2', 'nm⁻²')
        self.zimm_k_edit.setText(f'{zimm_spacing(rr) * qfac:.4g}')
        self._k_field_qfac = qfac

    @QtCore.Slot()
    def _on_run(self, *, reuse_blank_answer: bool = False) -> None:
        """Run the selected SLS analysis.

        `reuse_blank_answer` is passed by the two non-button callers (`_rerun` after a
        mask-list edit, and the plot's pick-to-mask handler). A PRESS OF RUN always asks
        about a missing solvent blank; a live refit reuses the answer given at that
        sample's last Run, so masking points does not re-prompt on every click.
        """
        if not self._runnable or self.sample_id is None:
            return
        # Run can look enabled mid-fit (a selection change re-applies the gate); a press
        # then must not touch the run epoch, or the fit in flight is dropped. Refused
        # here, before the blank and pick questions, whose answer would then go unused.
        if self._run_slot.hold(target=(self.sample_id, self._fraction), queue=False,
                               notice=lambda: busy_notice(self)):
            return
        # A mask refit reaches here without the button, so the gate binds here too: a
        # fit the sample cannot run is not dispatched, the status line says why, and the
        # result on screen stays (R8.5).
        self._refresh_fit_availability()
        selected = self._availability.get(self._selected_fit_key())
        if selected is not None and not selected.ok:
            return
        sid = self.sample_id
        if not self.controller.sample_has_measured_blank(sid):
            if not _confirm_unmeasured_blank(self, self.controller, sid,
                                             reuse=reuse_blank_answer):
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
        # `blankless` is the sample-level guard for a per-series operation; the two are
        # equivalent because `_regroup` fills the blank slot before parking any extra --
        # measured over five cases, not assumed (rule 12).
        replaced_picks = []
        blankless = not self.controller.sample_has_measured_blank(sid)
        if method in ('zimm', 'berry') and not blankless:
            # A blankless fit is not offered to Cross-Sample, so `run_zimm` returns before
            # the SampleResult write and this run replaces NOTHING -- the dialog below
            # would have promised a replacement and offered a Cancel protecting nothing.
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

        def _picks_replaced_by_this_run():
            """The picks this run actually replaced (`Controller.picks_a_run_replaced`):
            none after a compute-phase failure, those written before a present-phase one."""
            return self.controller.picks_a_run_replaced(sid, frac, replaced_picks)

        def fail(exc: BaseException, token) -> None:
            # A failure for a sample or fraction no longer shown must not act on the
            # view now shown (its Export belongs to the fit redisplayed there), so it
            # is told in a dialog naming the sample. One superseded by a newer run of
            # this same view is not shown: that run reports its own outcome. A fresh
            # one is flagged in the panel, not a modal (style guide R8.4) -- a mask
            # edit re-runs this live, and a modal would come up per click -- and the
            # previous fit's numbers go with its flag, so none stays on screen under
            # the failure.
            same_view = (sid, frac) == (self.sample_id, self._fraction)
            fresh = same_view and self._run_slot.current(token)
            self._apply_run_gate()   # the worker re-enabled Run; the selection decides
            who = sample_display_name(self.controller.workspace.samples[sid])
            text = (f'Could not run {method_text!r} on {who}: '
                    f'{humanize_field_names(str(exc))} Confirm parameters (Data tab) '
                    'and the calibration, then Update.')
            if fresh:
                self._clear_result_table()
                self.export_button.setEnabled(False)
                set_flag(self.flag_label, text)
            elif not same_view:
                QtWidgets.QMessageBox.critical(self, 'Analysis failed', text)
            gone = _picks_replaced_by_this_run()
            if gone:
                names = ', '.join(_QTY_DISPLAY[q] for q, _ in gone)
                lead = ('Analysis failed — see the warning below.' if fresh
                        else 'Analysis failed — see dialog.' if not same_view
                        else 'An earlier run failed.')
                self.status.setText(f'{lead} The run had already replaced your '
                                    f'picked {names} on {who}.')
            elif fresh:
                self.status.setText('Analysis failed — see the warning below.')

        def done(payload, token) -> None:
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
                who = sample_display_name(self.controller.workspace.samples[sid])
                disclosure = (f'{method_text} on {who} replaced your picked '
                              f'{names} (re-pick in Cross-Sample if needed).')
            elif blankless and method in ('zimm', 'berry'):
                # The mirror of the disclosure above, for the write that did NOT happen.
                # The dialog says this too, but a mask refit passes `reuse=True` and never
                # re-opens it, so a user can refit ten times without seeing it again --
                # and a silent non-write is how the SLS tab and the Cross-Sample tab come
                # to show different numbers for one sample with nothing explaining it.
                # Composed here with the replaced-pick line so both ride the same
                # not-view-gated path.
                who = sample_display_name(self.controller.workspace.samples[sid])
                disclosure = (f'{method_text} on {who} was not sent to Cross-Sample '
                              f'(no measured solvent blank); enter the value by hand '
                              f'there if you need it in a scaling plot.')
                # ...and name the OLDER value if one is still standing. Removing the blank
                # clears the computed triple, but a regroup that re-keys the blank away
                # does not, so Cross-Sample can still be serving an earlier run's number
                # while this tab shows a different, flagged one (round d0ce f3).
                stored = self.controller.workspace.samples[sid].result_for(frac)
                if any(getattr(stored, f'{q}_source', '') == 'computed'
                       for q in ('mw', 'rg', 'a2')):
                    disclosure += (' A value from an earlier run is still shown there '
                                   'for this sample.')
            if not self._run_slot.current(token):
                self._apply_run_gate()   # the worker re-enabled Run; the selection decides
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
            except Exception as exc:         # noqa: BLE001 — routed to fail(), the Analysis-failed path
                fail(exc, token)
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
            # After presentation's own status.clear(), so a selection changed while the
            # fit ran keeps its reason on the status line.
            self._apply_run_gate()
            # The fresh-arm write, after presentation's own status.clear(). Never
            # in _present_method or the cached payload: a redisplay replays the
            # payload verbatim, and a replayed "replaced your pick" would be a
            # stale claim about a past event.
            if disclosure:
                self.status.setText(disclosure)

        if self._run_slot.submit(
                lambda: self._compute_method(method, sid, frac, conc, angle, model),
                done, fail, target=(sid, frac), queue=False,
                busy_widgets=(self.run_button,),
                notice=lambda: self.status.setText(BUSY_NOTICE),
                description=f'{method_text} fit'):
            self.status.setText('Running in the background…')

    @property
    def _run_epoch(self) -> int:
        """The Run's staleness epoch, held by its slot."""
        return self._run_slot.epoch

    @_run_epoch.setter
    def _run_epoch(self, value: int) -> None:
        self._run_slot.epoch = value

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
            payload['res'] = c.run_debye(sid, conc, fraction=frac)
        elif method == 'guinier':
            payload['res'] = c.run_guinier(sid, conc, fraction=frac)
        elif method == 'single':
            payload['res'] = c.run_single_angle(sid, conc, angle, frac)
        elif method == 'ladder':
            payload['res'] = c.run_single_angle_ladder(sid, angle, frac)
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
        except Exception:                    # noqa: BLE001 — stale cached payload; cleared, status says so
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
        ellipsoid can reproduce the same P(q)). One results-table block per solution, and
        each solution a named source of the notes block, so no cause shows for the first
        alone; every solution is shown and none is silently preferred (the same
        structural-degeneracy contract as the DPLS shape display)."""
        c = self.controller
        first = results[0]
        self._stash_export(sid, 'form_factor_ellipsoid',
                           lambda p: c.export_form_factor(results, p))
        self._setup_axes()
        regime = 'apparent' if first.is_apparent else 'thermodynamic'
        self.ax.set_title(f'Form factor P(q): Ellipsoid ({regime})')

        if not first.converged:
            self.notes_block.show_notes(first)
            self._fill_result_table([_Row('Ellipsoid', 'no fit', first, ())])
            self._set_rest(())
            return

        plot_ellipsoid_form_factor(results, ax=self.ax)
        plural = len(results) > 1
        rows: List[_Row] = []
        sources = []
        for i, r in enumerate(results, 1):
            kind = 'prolate' if r.is_prolate else 'oblate'
            tag = f'S{i} ({kind})' if plural else kind.capitalize()
            sources.append((tag, r))
            dashed = bool(_scale_causes(r))
            # The ± follows identifiability alone: an imprecise or blankless ratio keeps it
            # (its marker says why), an unresolved one has none to show.
            if r.axial_ratio_resolved and r.axial_ratio_se is not None:
                p_val, p_se = format_pm(r.axial_ratio, r.axial_ratio_se), 'axial_ratio_se'
            else:
                p_val, p_se = format_fixed_sig(r.axial_ratio, sig), None
            # A solution's degeneracy clause rides its first row only.
            rows.append(_value_row(f'{tag}: axial ratio p', '', p_val, r, 'axial_ratio',
                                   p_se))
            rows.append(_value_row(f'{tag}: R<sub>g</sub>', 'nm',
                                   format_pm(r.rg_nm, r.rg_se), r, 'rg_nm', 'rg_se',
                                   flag='rg_reliable', exclude=('degenerate',)))
            rows.extend(_shape_solution_tail(tag, r, sig, plural=plural,
                                             apparent=first.is_apparent, dashed=dashed))
        self.notes_block.show_notes(sources if plural else first)
        self._fill_result_table(rows)
        _append_label_tip(self.result_table,
                          [i for i, row in enumerate(rows) if 'axial ratio' in row.label],
                          _AXIAL_RATIO_TIP)
        self._set_rest(())

    def _present_core_shell(self, sid: str, results, sig: int) -> None:
        """Display the core-shell P(q) fit -- a LIST of solutions (several core/outer radius
        pairs can reproduce the same P(q)). One results-table block per solution, and each
        solution a named source of the notes block; every solution is shown and none is
        silently preferred (the same structural-degeneracy contract as the ellipsoid / DPLS
        shape display)."""
        c = self.controller
        first = results[0]
        self._stash_export(sid, 'form_factor_core_shell',
                           lambda p: c.export_form_factor(results, p))
        self._setup_axes()
        regime = 'apparent' if first.is_apparent else 'thermodynamic'
        self.ax.set_title(f'Form factor P(q): Core–shell sphere ({regime})')

        if not first.converged:
            self.notes_block.show_notes(first)
            self._fill_result_table([_Row('Core–shell', 'no fit', first, ())])
            self._set_rest(())
            return

        plot_core_shell_form_factor(results, ax=self.ax)
        plural = len(results) > 1
        rows: List[_Row] = []
        sources = []
        for i, r in enumerate(results, 1):
            tag = f'S{i}' if plural else 'Fit'
            sources.append((tag, r))
            dashed = bool(_scale_causes(r))
            # A solution's degeneracy clause rides its first row only.
            rows.append(_value_row(f'{tag}: R<sub>core</sub>', 'nm',
                                   format_pm(r.r_core_nm, r.r_core_se), r, 'r_core_nm',
                                   'r_core_se'))
            rows.append(_value_row(f'{tag}: R<sub>outer</sub>', 'nm',
                                   format_pm(r.r_outer_nm, r.r_outer_se), r, 'r_outer_nm',
                                   'r_outer_se', exclude=('degenerate',)))
            rows.append(_value_row(f'{tag}: shell t', 'nm',
                                   format_fixed_sig(r.shell_thickness_nm, sig), r,
                                   'shell_thickness_nm', exclude=('degenerate',)))
            # The contrast-weighted scattering Rg, its value or n/a where the contrast
            # profile leaves it undefined; the marker names which cause took it down.
            rows.append(_value_row(f'{tag}: R<sub>g</sub>', 'nm',
                                   f'{format_fixed_sig(r.rg_nm, sig)} (contrast-weighted)',
                                   r, 'rg_nm', flag='rg_reliable', exclude=('degenerate',)))
            rows.extend(_shape_solution_tail(tag, r, sig, plural=plural,
                                             apparent=first.is_apparent, dashed=dashed))
        self.notes_block.show_notes(sources if plural else first)
        self._fill_result_table(rows)
        self._set_rest(())

    def _present_method(self, method: str, sid: str, conc, payload: Dict) -> None:
        """Main-thread phase: plot + fill tables from the worker's payload. Reads
        precomputed results (payload['rr'] / ['res']); makes no controller fit
        calls itself (the masked-point overlay reads the cached full series).

        Each arm fills the notes block with the result(s) the table shows BEFORE the
        table, so the cells it marks find their letters."""
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
            # Mw is apparent at one concentration, thermodynamic across concentrations
            # (the label's qualifier words say which); dashed when its scale is unknown.
            # Rg / R come from the angular shape and survive an unknown scale.
            dashed = bool(_scale_causes(res))
            rows = [
                _value_row('M<sub>w</sub>', 'g/mol',
                           _scaled_value(res, format_pm(res.mw_g_per_mol, res.mw_se)),
                           res, 'mw_g_per_mol', 'mw_se', dashed=dashed),
                _value_row('R<sub>g</sub>', 'nm', format_pm(res.rg_nm, res.rg_se), res,
                           'rg_nm', 'rg_se', flag='rg_reliable'),
            ]
            # Sphere and Mie are both solid spheres, so both report the geometric radius R
            # (the coil has none) -- matching the plot legend and the export column. R is a
            # fixed multiple of Rg, so it carries the same size verdict and the same notes.
            if res.model in (ff.SPHERE, ff.MIE):
                rows.append(_value_row('R', 'nm', format_pm(res.r_nm, res.r_se), res,
                                       'r_nm', 'r_se', flag='rg_reliable',
                                       mark=('r_nm', 'rg_nm')))
            if not res.is_apparent:
                rows.append(_value_row(
                    'A₂', 'mol·mL/g²',
                    _scaled_value(res, format_pm(res.a2_mol_mL_per_g2, res.a2_se)),
                    res, 'a2_mol_mL_per_g2', 'a2_se', dashed=dashed))
            rows.append(_diagnostic_row('qR<sub>g</sub>(max)',
                                        format_fixed_sig(res.qrg_max, sig), res, 'qrg_max'))
            rows.append(_Row('R²', format_fixed_sig(res.r_squared, sig)))
            self.notes_block.show_notes(res)
            self._fill_result_table(rows)
            self._set_rest(())
        elif method == 'debye':
            res = payload['res']
            self._stash_export(sid, 'debye', lambda p: c.export_debye(res, p))
            self._setup_axes()
            plot_debye(res, ax=self.ax)
            self.ax.set_title('Debye plot (single concentration)')
            mw_app = _scaled_value(res, format_pm(
                res.mw_apparent_g_per_mol, getattr(res, 'mw_apparent_se', None)))
            self.notes_block.show_notes(res)
            # ",app" is in every row label here (one concentration), so the word is not.
            self._fill_result_table([
                _value_row('M<sub>w,app</sub>', 'g/mol', mw_app, res,
                           'mw_apparent_g_per_mol', 'mw_apparent_se',
                           dashed=bool(_scale_causes(res)), exclude=('apparent',)),
                _value_row('R<sub>g,app</sub>', 'nm',
                           format_pm(res.rg_apparent_nm, getattr(res, 'rg_apparent_se', None)),
                           res, 'rg_apparent_nm', 'rg_apparent_se', flag='rg_reliable',
                           exclude=('apparent',)),
                # The regime the apparent-Rg verdict reads (the Zimm/Guinier rows' shape).
                _diagnostic_row('qR<sub>g</sub>(max)', format_fixed_sig(res.qrg_max, sig),
                                res, 'qrg_max'),
                _Row('R²', format_fixed_sig(res.r_squared, sig)),
            ])
            self._set_rest(())
        elif method == 'guinier':
            res = payload['res']
            self._stash_export(sid, 'guinier', lambda p: c.export_guinier(res, p))
            self._setup_axes()
            plot_guinier(res, ax=self.ax)
            self.ax.set_title('Guinier plot (single concentration)')
            self.notes_block.show_notes(res)
            self._fill_result_table([
                _value_row('R<sub>g,app</sub>', 'nm',
                           format_pm(res.rg_nm, getattr(res, 'rg_se', None)), res,
                           'rg_nm', 'rg_se', flag='rg_reliable', exclude=('apparent',)),
                _diagnostic_row('qR<sub>g</sub>(max)', format_fixed_sig(res.qrg_max, sig),
                                res, 'qrg_max'),
                _Row('R²', format_fixed_sig(res.r_squared, sig)),
            ])
            self._set_rest(())
        elif method == 'single':
            res = payload['res']
            self._stash_export(sid, 'single_angle',
                               lambda p: c.export_single_angle(res, p))
            self._clear_plot()
            # Single angle + single concentration = one datum, so Mw_app has no honest ±
            # — the documented fixed-sig fallback, never a fabricated ±. Dashed only when
            # its scale is unknown; an ill-conditioned (subnormal divisor) Mw is finite,
            # so it is still shown, with its marker.
            mw_app = _scaled_value(res, format_fixed_sig(
                res.mw_apparent_g_per_mol,
                self.controller.settings.no_uncertainty_sig_figs))
            self.notes_block.show_notes(res)
            self._fill_result_table([
                _value_row('M<sub>w,app</sub>', 'g/mol', mw_app, res,
                           'mw_apparent_g_per_mol', dashed=bool(_scale_causes(res)),
                           exclude=('apparent',)),
                _Row('Angle', f'{res.angle_deg:.0f}°'),
                _Row('c (mg/mL)', f'{res.concentration_g_per_mL * 1000:.3g}'),
            ])
            self._set_rest(())
        elif method == 'ladder':
            res = payload['res']
            self._stash_export(sid, 'single_angle_ladder',
                               lambda p: c.export_single_angle_ladder(res, p))
            self._setup_axes()
            plot_single_angle_ladder(res, ax=self.ax)
            self.ax.set_title(f'Single-angle ladder at {res.angle_deg:.0f}°')
            # Mw and the absolute A₂ carry both scales (dashed at the value when either
            # is unknown, as Zimm's are); the 2·A₂·Mw product cancels them and is shown
            # whatever the calibration state, as on the calibration-free tab.
            dashed = bool(_scale_causes(res))
            mw_value = _scaled_value(res, format_pm(res.mw_apparent_g_per_mol,
                                                    res.mw_apparent_se))
            a2_value = _scaled_value(res, format_pm(res.a2_apparent_mol_mL_per_g2,
                                                    res.a2_apparent_se))
            self.notes_block.show_notes(res)
            self._fill_result_table([
                _value_row('M<sub>w,app</sub>', 'g/mol', mw_value, res,
                           'mw_apparent_g_per_mol', 'mw_apparent_se', dashed=dashed,
                           exclude=('apparent',)),
                _value_row('A<sub>2,app</sub>', 'mol·mL/g²', a2_value, res,
                           'a2_apparent_mol_mL_per_g2', 'a2_apparent_se', dashed=dashed,
                           exclude=('apparent',)),
                _value_row('2·A₂·M<sub>w</sub>', 'scale-independent',
                           format_pm(res.two_a2_mw, res.two_a2_mw_se), res, 'two_a2_mw',
                           'two_a2_mw_se'),
                _Row('Angle', f'{res.angle_deg:.0f}°'),
                _Row('Points', str(res.n_conc)),
            ])
            # Each ± row's label carries its own words, the product's "± statistical only"
            # among them; a flag down is the notes block's line
            # (`tests/test_reliability_family.py::SILENT_FLAGS`).
            self._set_rest(())
            # Calculation nuance, on hover (rule 8's tooltip tier): the direction of
            # the ± on a short ladder, disclosed rather than left to be inferred, on the
            # labels of the rows whose ± it explains.
            _append_label_tip(self.result_table, (0, 1),
                              'Mw,app is a reciprocal of the fitted intercept; on a short '
                              'concentration ladder its SE is CONSERVATIVE — it meets or '
                              'exceeds the sampling spread and never under-reports '
                              '(Monte-Carlo-validated) — and tightens with more concentration '
                              'points or replicates. The A₂ ± is the slope SE halved. See the '
                              'Theory-and-Equations-Guide §4.3 and §6.1.')
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
            # remedies differ, so the cells must; neither absent state is marked, and the
            # refused one's scale cause is the block's line, read off the stored Mw.
            withheld = res.a2_mol_mL_per_g2 is None and self._mw_is_arbitrary_scale()
            if res.a2_mol_mL_per_g2 is not None:
                a2_row = _value_row('A₂', 'mol·mL/g²',
                                    format_pm(res.a2_mol_mL_per_g2, getattr(res, 'a2_se', None)),
                                    res, 'a2_mol_mL_per_g2', 'a2_se')
            else:
                a2_row = _value_row('A₂', 'mol·mL/g²',
                                    _UNCAL_A2 if withheld else 'set a manual Mw to get A₂',
                                    res, 'a2_mol_mL_per_g2', dashed=True)
            stored = (self.controller.workspace.samples[self.sample_id]
                      .result_for(self._fraction).scale_notes() if withheld else ())
            self.notes_block.show_notes(tuple(notes_in(res)) + tuple(stored))
            self._fill_result_table([
                _value_row('2·A₂·M<sub>w</sub>', 'scale-independent',
                           format_pm(res.two_a2_mw, getattr(res, 'two_a2_mw_se', None)),
                           res, 'two_a2_mw', 'two_a2_mw_se'),
                a2_row,
            ])
            if withheld:
                self.result_table.item(1, 1).setToolTip(_WITHHELD_A2_TOOLTIP)
            # The product row's label carries its "± statistical only" (and OLS); a flag
            # down is the notes block's line (`tests/test_reliability_family.py::SILENT_FLAGS`).
            self._set_rest(())
            if getattr(res, 'two_a2_mw_se', None) is not None:
                _append_label_tip(self.result_table, (0, 1),
                                  'No calibration or dn/dc enters 2·A₂·Mw (they cancel in '
                                  'the intensity ratio); the ± is a conservative statistical '
                                  'SE that never under-reports and tightens with more '
                                  'concentration points. The A₂ ± beneath it is that SE '
                                  'divided by 2·Mw with the Mw treated as EXACT, so it '
                                  'excludes the error on the Mw you supplied — which enters '
                                  'A₂ linearly. See the Theory-and-Equations-Guide §6.1.')
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
            self._clear_result_table()           # a per-c series, not a scalar result
            self.status.setText(f'Excess Rayleigh ratio for {len(rr)} concentrations.')
            # The one view that plots ΔR itself -- the quantity an assumed or absent blank
            # corrupts directly: the plotted series' gates about ΔR, each once (a missing
            # dn/dc is about the ordinate, which this view does not plot).
            self.notes_block.show_notes(tuple(
                n for r in rr if r.concentration_g_per_mL != 0
                for n in find_notes(r, about='excess_rayleigh_cm_inv')))
            self._set_rest(())
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
        # _point_coords lists (c, angle, x, y) for every point the plot can show, the
        # masked ones included, from the cached unmasked series — no fit. Keep only the
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

    @staticmethod
    def _fitted_ordinate(y) -> bool:
        """Whether a Kc/ΔR point is one the Debye and form-factor analyses keep, and so
        one their plots draw: finite and positive (a non-positive Kc/ΔR is left out
        with a note, ``analysis.sls``). The hit-test pin in the GUI smoke suite ties
        this predicate to what the plotters draw."""
        return bool(np.isfinite(y) and y > 0)

    def _point_coords(self, method: str, sid: str):
        """Every point the current method's plot can show, as (c, angle, x_data,
        y_data), for click hit-testing: each unmasked point its analysis keeps, which
        the plot draws (a form-factor model that found no fit draws none, and its
        points stay listed), and each masked point, which `_overlay_masked` draws
        hollow so a click can unmask it. Reads the cached unmasked series (as of the
        last run) and the mask — no fit.
        x_data is in the DRAWN axis units: the canonical q² (or q²+k·c composite)
        scaled by the active q² display factor, exactly as the plotters draw it
        (spec 04 stage 2; factor exactly 1.0 on the default nm⁻² choice)."""
        full = self._full_rr.get((sid, self._fraction))
        if full is None:
            return []
        qfac = display_factor_from('scattering_q2', 'nm⁻²')
        mask = self.controller.sls_mask(sid, self._fraction)
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
                        if not (self._fitted_ordinate(y) or mask.is_masked(cc, float(ang))):
                            continue
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
                    if self._fitted_ordinate(y) or (np.isfinite(y)
                                                    and mask.is_masked(cc, float(ang))):
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
        self._on_run(reuse_blank_answer=True)

    def _fill_result_table(self, rows) -> None:
        """Rebuild the per-method results table from `_Row`s (or plain (label, value)
        pairs). Each marked value takes the marker of every line the notes block shows
        about it, so the block is filled first; a value whose own flag is down with no
        line about it keeps its bare ⚠."""
        t = self.result_table
        t.setRowCount(len(rows))
        rows = [row if isinstance(row, _Row) else _Row(*row) for row in rows]
        caption = tuple(dict.fromkeys(w for row in rows for w in row.caption))
        self.pm_caption.setText('; '.join(caption))
        self.pm_caption.setToolTip('\n'.join(dict.fromkeys(
            line for row in rows if row.caption for line in row.label_tip.splitlines()
            if any(line.startswith(w) for w in row.caption))))
        for r, row in enumerate(rows):
            label = QtWidgets.QTableWidgetItem(row.label)
            if row.label_tip:
                label.setToolTip(row.label_tip)
            value = QtWidgets.QTableWidgetItem(row.value)
            mark =(self.notes_block.mark_item(value, row.source, row.attrs, whole=row.whole)
                    if row.source is not None else '')
            if not mark and row.flag and not bool(getattr(row.source, row.flag, True)):
                value.setText(f'{row.value} {MARKER_GLYPH}')
            t.setItem(r, 0, label)
            t.setItem(r, 1, value)
        # Cap the table at exactly its content height (real row height, not a per-row
        # guess) so the last row never clips into a scrollbar while it sits in the
        # roomy under-plot strip.
        row_h = t.verticalHeader().defaultSectionSize()
        t.setMaximumHeight(row_h * max(len(rows), 1) + 2 * t.frameWidth() + 2)
        t.resizeColumnsToContents()
        self.status.clear()

    def _clear_result_table(self) -> None:
        self.result_table.setRowCount(0)
        self.pm_caption.setText('')
        self.notes_block.show_notes(None)

    def _summarize_zimm(self, res) -> None:
        sig = self.controller.settings.no_uncertainty_sig_figs   # ±-less: R², two-route Mw
        mw_se = getattr(res, 'mw_se', None)
        rg_se = getattr(res, 'rg_se', None)
        a2_se = getattr(res, 'a2_se', None)
        # Mw AND absolute A₂ carry the scale (Mw_app = Mw/f, A₂_app = f·A₂): dash BOTH at
        # the value when it is unknown rather than show a meaningless number. Only Rg
        # (scale-free) survives here; the calibration-free 2·A₂·Mw product is a separate
        # method/table. An ill-conditioned (but calibrated) fit is NOT dashed — the value
        # is finite, so it stays (format_pm renders a non-physical NaN as 'n/a') with its
        # marker; the dash keys on the scale, not on the reliability flag.
        dashed = bool(_scale_causes(res))
        mw_value = _scaled_value(res, format_pm(res.mw_g_per_mol, mw_se))
        a2_value = _scaled_value(res, format_pm(res.a2_mol_mL_per_g2, a2_se))
        self.notes_block.show_notes(res)
        # A Berry Rg and Mw read past the Zimm limit say so in a word on their own labels
        # (`_RG_MW_QUALIFIERS`); a Zimm Rg and Mw past it are flagged, the line under the table.
        self._fill_result_table([
            _Row('Method', res.method.capitalize()),
            _value_row('M<sub>w</sub>', 'g/mol', mw_value, res, 'mw_g_per_mol', 'mw_se',
                       dashed=dashed),
            _value_row('R<sub>g</sub>', 'nm', format_pm(res.rg_nm, rg_se), res, 'rg_nm',
                       'rg_se', flag='rg_reliable'),
            _value_row('A₂', 'mol·mL/g²', a2_value, res, 'a2_mol_mL_per_g2', 'a2_se',
                       dashed=dashed, exclude=_RG_MW_QUALIFIERS),
            # The regime the Rg verdict reads, marked by a note about it where one exists.
            _diagnostic_row('qR<sub>g</sub>(max)', format_fixed_sig(res.qrg_max, sig), res,
                            'qrg_max'),
            _Row('R²', format_fixed_sig(res.r_squared, sig)),
        ])
        rest = []
        # Two-route consistency: Mw can be read off the c→0 line OR the q→0 line, and the
        # two should match; a large gap warns of curvature/extrapolation error (no ±
        # involved). `agree` is the RELATIVE DIFFERENCE between the two, so it is phrased as
        # "differ by X%" — "agreement X%" was ambiguous. The two absolute Mw print only
        # when BOTH SCALES ARE KNOWN: otherwise they carry the same arbitrary scale the
        # table has just dashed out, and would read as a molar mass many orders off. The
        # RATIO survives on both branches, being scale-independent. The Cabannes readout
        # in `_refresh_anisotropy_display` withholds its f·Mw on the same condition.
        agree = getattr(res, 'extrapolation_agreement_rel', None)
        mc0 = getattr(res, 'mw_from_c0_g_per_mol', None)
        mq0 = getattr(res, 'mw_from_q0_g_per_mol', None)
        route_disagrees = False
        if agree is not None and math.isfinite(agree) and mc0 is not None and mq0 is not None:
            # The notes block's line names the missing scale, so the withheld branch
            # states only why the two values are absent.
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
            rest.append(note)
        # Berry's free cross term as a multiple of the -b'd'/a' Berry's own equation predicts
        # to first order (the engine's `coef_q2c_ratio`, scale-free). A neutral readout,
        # never a flag -- the free column fits either sign by design.
        cross = getattr(res, 'coef_q2c_ratio', float('nan'))
        if math.isfinite(cross):
            rest.append(f"Berry q²c cross term: {format_fixed_sig(cross, sig)} × the value "
                        f"Berry's equation predicts to first order (−b′d′/a′)")
        # Red only for the >10 % route disagreement; the plain two-route note and the
        # cross-term readout are neutral.
        self._set_rest(rest, problem=route_disagrees)

    def _set_rest(self, parts: Sequence[str], *, problem: Optional[bool] = None) -> None:
        """The flag at the foot of the column, for what no line of the notes block
        carries, one fact per line; red when ``problem`` (default: whenever there is
        any)."""
        text = '\n'.join(p for p in parts if p)
        self._set_flag(text, problem=bool(text) if problem is None else problem)

    def _set_flag(self, text: str, *, problem: bool = True) -> None:
        """Set this tab's flag label via the shared two-tier renderer.

        ``problem=True`` → bold red ``error``, message led with ⚠; ``problem=False`` → a
        calm non-bold ``qualifier`` accent, message led with ⓘ. See
        :func:`gui.theme.set_flag` — the one place glyph + color are chosen so they can
        never disagree (style guide §5, R5.1)."""
        set_flag(self.flag_label, text, problem=problem)

    # ------------------------------------------------------------ figure ---
    def _setup_axes(self) -> None:
        if self.ax is not None:
            try:
                self.ax.set_xscale('linear')
            except Exception:  # noqa: BLE001 — stale axes on rebuild
                pass
        self.figure.clf()
        self.ax = self.figure.add_subplot(1, 1, 1)

    def _clear_plot(self, message: Optional[str] = None) -> None:
        if self.ax is not None:
            try:
                self.ax.set_xscale('linear')
            except Exception:  # noqa: BLE001 — stale axes on rebuild
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
