"""
The DLS Distribution sub-tab: NNLS, CONTIN and lognormal size (or Γ) distributions, with
the peak table and the α sweep.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from plotting.plots import (plot_distribution, plot_lcurve, plot_weighted_distribution,
                            annotate_decollided, display_factor as _disp_factor)
from analysis.optical_model import (WEIGHTINGS, WEIGHTING_LABELS, OPTICAL_MODELS,
                                    OPTICAL_MODEL_LABELS, INTENSITY, RDG, MIE, OpticalModelError)
from app.fit_requirements import DLS_UNIVERSAL_REQUIREMENTS
from core.notes import Note, find_notes, grouped, has_note, qualifiers, render
from core.naming import measurement_display_name, measurement_hover_card, sample_display_name
from gui.plot_controls import (make_split_panels, make_vertical_plot_stack, pin_under_column,
                               scrolling_notes, COLUMN_STACK_STRETCH, attach_residual_resizer,
                               set_plot_export_dressing, set_plot_export_name)
from gui.export_helper import confirm_overwrite, export_to_csv
from exporting.export import alpha_selection_note
from gui.field_labels import humanize_field_names
from gui.help import add_help_to_groupbox, HelpBadge
from gui.notes_block import NotesBlock, qualifier_words
from gui.plot_overlays import show_plot_placeholder, PLACEHOLDER_GRAY
from gui.distribution_fit_fields import DistributionFitFields
from gui.theme import ThemedLabel, set_flag
from gui.widgets import MeasurementPicker, make_plot_pane, marker_swatch, NAME_COL_MAX_W
from gui.worker import BACKGROUND_RUN_TOOLTIP, BUSY_NOTICE, RunSlot
from core.formatting import format_fixed_sig

from gui.dls.region import _region_fit_kwargs
from gui.dls.fits import (_DIST_METHODS, _METHOD_LINESTYLE, _fit_label, _failed_without_cause,
                          draw_fit_residuals)
from gui.dls.common import (_STALE_HINT, _widen_row_gap, _weighting_title, _selection_export_names,
                            _sample_label_of, _sample_prefixes, _PICKER_HELP, _GROUP_FIELD_CONC,
                            _GROUP_FIELD_ANGLE, _GROUP_PICKER_BULLETS, _left_out_lines,
                            _union_missing, _needs_lines)
from gui.dls.legend import (_method_legend_entries, _measurement_legend_entries, _draw_key,
                            export_legend_dressing)
from gui.dls.bases import _DeferredRepaint, _ResidualRatioMixin


#: The Distribution sub-tab's peak-table title and passive tooltip: every cell is the peak's
#: POSITION, its lobe's geometric mean, and its share of the weight.
_PEAK_RESULTS_TITLE = 'Peak results (geometric-mean Rh · % of weight)'


_PEAK_TABLE_TOOLTIP = (
    "Each cell is one peak: its geometric-mean Rh, the average of ln Rh over the peak "
    "(for a lognormal fit, its fitted median), and its share of the total weight. "
    "Overlapping peaks and a population past the Rh grid's end shift the mean; a peak "
    "whose top is the grid's first or last point is marked (Theory-and-Equations-Guide "
    "§3.4.1).")


def _nearest_log_node(grid: np.ndarray, x: float) -> int:
    """The index of the grid node nearest `x` on a log axis, where a peak label's height is
    read: the distribution's grid is log-spaced, so linear distance would pick the larger
    neighbour of a mean that sits closer, in ln, to the smaller one."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return int(np.nanargmin(np.abs(np.log(np.asarray(grid, dtype=float)) - math.log(x))))


def _ordinal_peak(i: int) -> str:
    """'1st peak', '2nd peak', … for a 0-based index (positional-peak row labels,
    matching the replicate-averaging peak language)."""
    n = i + 1
    suffix = ('th' if 11 <= (n % 100) <= 13
              else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th'))
    return f'{n}{suffix} peak'


# ===========================================================================
# Distribution sub-tab (NNLS / CONTIN / Lognormal)
# ===========================================================================

# The qualifier of a drawn distribution shown on the tab's own line rather than on a peak
# row's name: a sphere-model weighted view. The fits' residual weighting is a word in the peak
# table's title (`_weighting_title`).
_UNLABELLED_QUALIFIERS = frozenset({'model_weighted_view'})


class _DistributionTab(_DeferredRepaint, _ResidualRatioMixin, QtWidgets.QWidget):
    """Size / decay-rate distribution view over ONE or MANY ticked measurements,
    each computed by ONE or MANY chosen methods (Lognormal / NNLS / CONTIN). Every
    (measurement × method) curve is overlaid; the τ-window + baseline are shared with
    the Correlogram tab.

    Colour and point MARKER are the measurement, line STYLE is the method -- the same
    coding as the Correlogram tab, and the reason NNLS and CONTIN can now be told apart
    here (they were drawn in one colour and one solid line). This tab RUNS its three
    fits, which the controller holds; the Correlogram tab reads them through
    `stored_fits`, via the module that owns both, and is told they moved by
    `fitsChanged`.

    Each fit is kept in its own (measurement, method) slot with its own signature, as
    the Correlogram tab's are: a fit of one measurement reads nothing of the ticked set,
    so the ticks and the method boxes choose which stored fits are SHOWN and never
    discard one. A fit whose inputs have moved stays in its slot, is not
    drawn, and its run reads "press Run" (`_shown`)."""

    #: Emitted whenever this tab's stored fits change, carrying the method keys a Run
    #: just produced (empty on a tick change, so the layers list re-reads which fits the
    #: new ticked set can draw). The owning `DLSModule` routes it to the Correlogram
    #: tab's layers list, so a CONTIN Run here offers -- and ticks -- CONTIN over there
    #: without one widget reaching into the other.
    fitsChanged = QtCore.Signal(list)

    #: A hidden Distribution keeps its focus at a sidebar click and repaints when shown.
    defers_repaint = True

    def __init__(self, controller, region, selection, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.region = region                    # shared _AnalysisRegion
        self.selection = selection              # shared _OverlaySelection
        self.item_id: Optional[str] = None
        self._runnable = False
        # Every fit this tab has run, stamped with `_fit_signature` at dispatch, and the
        # message naming why a pair failed in the LAST Run, live in the controller under
        # ('distribution', iid, method) (`_stored`, `_failure`). What is drawn is
        # `_shown()`. The failure lines are replaced wholesale by each Run, as the
        # Correlogram tab's are; only the ticked pairs' are shown.

        # Per-draw reweighting cache: [(iid, method, WeightedDistributionResult or None)],
        # parallel to the fits the last _draw showed. None means intensity (no
        # reweighting) or a reweight error (the flag names it). Rebuilt every _draw;
        # used by export + the peak table.
        self._weighted: List[Tuple] = []
        self._reweight_errors: List[str] = []
        # iid -> `Controller.dls_fit_availability([iid])` for each ticked run, re-read by
        # `_refresh_fit_availability`; and each method box's and sphere-model entry's own
        # tooltip, put back when it stops being grayed.
        self._availability: Dict[str, Dict] = {}
        self._entry_tips: Dict[tuple, object] = {}
        self._placeholder = None                # empty-state Text artist (no distribution)
        # The Run's slot: a busy press is refused. Its epoch is bumped by a tick change
        # and read by a Run's `fail` ONLY, so a whole-Run failure landing after the
        # ticks moved never paints the status of the view now shown. `done` needs no
        # such guard: a finished fit lands in its own slot.
        self._run_slot = RunSlot(before_delivery=self.flush_view)
        self._build_ui()

    def _build_ui(self) -> None:
        # Distribution has the longest form labels ('CONTIN α selection:' etc.), so it
        # gets a slightly wider default + minimum than the other DLS tabs (labels
        # clipped at the old 340 px default). The Rh-grid triple-spinbox row measures
        # 384 px at the shipped size, so 385 keeps its two '?' badges inside the column
        # (they overhung by 10 and 6 px at 360).
        splitter, left, right = make_split_panels(self, left_min_width=385,
                                                  sizes=(403, 717))

        self.checklist = MeasurementPicker(
            self.controller, self.selection, kinds=('dls',),
            label_fn=measurement_display_name, header_fn=sample_display_name,
            tooltip_fn=measurement_hover_card,
            help_text=_PICKER_HELP, help_bullets=_GROUP_PICKER_BULLETS,
            group_fields=(_GROUP_FIELD_CONC, _GROUP_FIELD_ANGLE))
        self.checklist.selectionChanged.connect(self._on_selection_changed)

        box = QtWidgets.QGroupBox('Distribution')
        add_help_to_groupbox(box, 'Recover a full size distribution (not just one '
                             'average) from the correlogram.', bullets=[
                                 '<b>Lognormal</b>: assumes a single skewed peak — '
                                 'least free, most stable.',
                                 '<b>NNLS</b>: sharper but noisier; can split modes.',
                                 '<b>CONTIN</b> (default): smooth, regularized — the '
                                 'robust general choice.',
                                 'Their sizes are distribution-weighted, not the '
                                 'cumulant average.',
                                 '<b>Click</b> a ⚠ line or lettered peak for what to '
                                 'do.',
                                 '<b>Residual weighting</b>: the Biganzoli–Ferri '
                                 'variance (default, about twice as slow), or g₁² to '
                                 'reproduce a result reported under it.',
                                 '<b>CONTIN α selection</b>: robust GCV (default), the '
                                 'L-curve corner or the F-test, where a higher '
                                 '“probability to reject” gives a smoother fit.',
                                 '<b>GCV robustness γ</b>: 0.1 (default) is robust GCV, '
                                 '1 classical GCV (Theory-and-Equations-Guide §3.4.7).',
                                 'A method no ticked measurement can use is '
                                 '<b>grayed</b>, its tooltip naming what to enter.',
                                 'Kept fits and a failed lognormal: User-Manual §8.3.1.',
                             ])
        form = QtWidgets.QFormLayout(box)
        # Per-method checkboxes (replacing the old single "Overlay NNLS+CONTIN").
        # Adding a method later = one more entry in _DIST_METHODS.
        self.method_checks: Dict[str, QtWidgets.QCheckBox] = {}
        mrow = QtWidgets.QVBoxLayout(); mrow.setContentsMargins(0, 0, 0, 0)
        for label, key in _DIST_METHODS:
            cb = QtWidgets.QCheckBox(label)
            cb.setChecked(key == 'contin')        # CONTIN default
            self.method_checks[key] = cb
            mrow.addWidget(cb)
        mwidget = QtWidgets.QWidget(); mwidget.setLayout(mrow)
        form.addRow('Methods:', mwidget)
        self.axis_combo = QtWidgets.QComboBox()
        self.axis_combo.addItem('Rh (nm)', 'rh')
        self.axis_combo.addItem('Γ (decay rate, 1/s)', 'gamma')
        self.axis_combo.setToolTip('Plot the distribution against the size (Rh, via Stokes–Einstein — needs the committed viscosity and temperature) or against the raw decay rate Γ, which is what the correlogram measures; the Γ view lets you compare angles before the size conversion is trusted.')
        self.axis_combo.currentIndexChanged.connect(self._on_axis_changed)
        form.addRow('Distribution axis:', self.axis_combo)

        # Weighting selector (Phase 1b): re-express the SAME intensity distribution as
        # volume/surface/number. A re-presentation, not a re-fit — switching weighting
        # redraws from the cached intensity result (no recompute).
        self.weighting_combo = QtWidgets.QComboBox()
        for key in WEIGHTINGS:
            self.weighting_combo.addItem(WEIGHTING_LABELS[key], key)
        self.weighting_combo.currentIndexChanged.connect(self._on_weighting_changed)
        wrow = QtWidgets.QHBoxLayout(); wrow.setContentsMargins(0, 0, 0, 0)
        wrow.addWidget(self.weighting_combo, 1)
        wrow.addWidget(HelpBadge(
            'Volume / surface / number views re-weight the same intensity fit; they '
            'don’t improve or re-measure it.',
            bullets=[
                'Treat a large small-particle population in the number view as '
                '<b>suspect</b> unless the intensity data clearly supports it.',
                'A reweighted view carries no ± and is well-posed only for <b>spheres</b> '
                '(Theory-and-Equations-Guide §3.7).',
                'Pick the sphere scattering model to match the sample: <b>Rayleigh</b> '
                '(very small particles), <b>RDG</b> (low contrast), or <b>Mie</b> (the '
                'general high-contrast/large case, e.g. latex in water). RDG and Mie need '
                'the hand-entered <b>solute refractive index</b> (Data tab).',
            ]))
        wholder = QtWidgets.QWidget(); wholder.setLayout(wrow)
        form.addRow('Weighting:', wholder)
        # Sphere scattering model (revealed only for a non-intensity weighting).
        self.model_combo = QtWidgets.QComboBox()
        for key in OPTICAL_MODELS:
            self.model_combo.addItem(OPTICAL_MODEL_LABELS[key], key)
        self.model_combo.setToolTip(
            'How per-particle scattering is modeled to re-weight intensity.\n'
            '• Rayleigh point: very small particles; needs no refractive-index contrast.\n'
            '• RDG sphere: low contrast; needs the solute refractive index (Data tab).\n'
            '• Mie sphere: the general case, where RDG and Rayleigh fail (high contrast, '
            'larger particles); also needs the solute refractive index.\n'
            'Each law\'s validity bounds: Theory-and-Equations-Guide §3.7.')
        self.model_combo.currentIndexChanged.connect(self._on_weighting_changed)
        form.addRow('Sphere model:', self.model_combo)
        self._model_label = form.labelForField(self.model_combo)
        # Read-only solute-RI reminder (RDG only): editing happens on the Data tab
        # (invariant 5 — the Data tab owns parameter editing).
        self.solute_ri_hint = ThemedLabel('', role='hint', size=10)
        self.solute_ri_hint.setWordWrap(True)
        form.addRow('', self.solute_ri_hint)
        self._solute_ri_hint_label = form.labelForField(self.solute_ri_hint)
        # The fit inputs: the Rh grid, the residual weighting and CONTIN's α selector,
        # range, γ and F-test level -- the shared group the Settings tab and the Mode
        # Map hold too, seeded from Settings and overridable per run. Not
        # `weighting_combo` above, which re-expresses a finished intensity distribution
        # as volume/surface/number: that is a presentation of the answer, this group
        # holds inputs to it.
        self.fit_fields = DistributionFitFields(self)
        self.fit_fields.add_rows(form)
        self.reseed_from_settings()
        note = ThemedLabel('Delay window + baseline region are set on the '
                           'Correlogram tab (shared). Each ticked method runs on '
                           'every ticked measurement that has its inputs.',
                           role='hint', size=10)
        note.setWordWrap(True)
        form.addRow(note)
        # Run is pinned under the scrolled form and Export under the column (the §3
        # column template), so the form's length can never carry them out of view.
        self.run_button = QtWidgets.QPushButton('Run')
        self.run_button.setToolTip(BACKGROUND_RUN_TOOLTIP)
        self.run_button.clicked.connect(self._on_run)
        self.export_button = QtWidgets.QPushButton('Export CSV…')
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._on_export)
        # Peak results for the ticked curves, mirroring the Correlogram tab's glance
        # table: one ROW per (measurement · method), one column per positional peak.
        # Populated in _draw from the shown fits, emptied with the plot.
        results_section = QtWidgets.QWidget()
        rlay = QtWidgets.QVBoxLayout(results_section)
        rlay.setContentsMargins(0, 0, 0, 0)
        self.results_label = QtWidgets.QLabel(_PEAK_RESULTS_TITLE)
        # Wrapped: a title naming a view or the fits' weighting can be wider than the column.
        self.results_label.setWordWrap(True)
        rlay.addWidget(self.results_label)
        self.result_table = QtWidgets.QTableWidget(0, 0)
        self.result_table.setToolTip(_PEAK_TABLE_TOOLTIP)
        self.result_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.result_table.verticalHeader().setVisible(False)
        rlay.addWidget(self.result_table, 1)
        # Vertically resizable control column: drag the grips to size the
        # checklist / controls / results panes against each other.

        # Scrolled controls and a stated results floor, for the reason the Correlogram
        # tab's are: this tab passed the box's own `sizeHint` as the pane minimum
        # OUTRIGHT, so the splitter starved the pane below it and the peak table fell
        # off the bottom of the column entirely.
        dist_scroll = QtWidgets.QScrollArea()
        dist_scroll.setWidgetResizable(True)
        dist_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        dist_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        dist_scroll.setWidget(box)
        dist_pane = QtWidgets.QWidget()
        dist_layout = QtWidgets.QVBoxLayout(dist_pane)
        dist_layout.setContentsMargins(0, 0, 0, 0)
        dist_layout.setSpacing(4)
        dist_layout.addWidget(dist_scroll, 1)
        dist_layout.addWidget(self.run_button)
        # The notes of the shown fits and their reweighted views: one line per failed
        # precondition and a marker on each peak it is about. Pinned under the column
        # right below the peak table (R8.6).
        self.notes_block = NotesBlock(what='fits')
        self.notes_block.watch(self.result_table)
        self._block_fits: set = set()       # id() of each fit the block was fed
        # A sphere-model weighted view, on the neutral ⓘ tier, as a line of its own
        # (`_UNLABELLED_QUALIFIERS`).
        self.info_label = ThemedLabel('', size=11)
        self.info_label.setWordWrap(True)
        rlay.addWidget(scrolling_notes(self.info_label))
        vstack = make_vertical_plot_stack(
            [self.checklist, dist_pane, results_section], sizes=[200, 380, 250],
            min_heights=[90, 150, 200], stretch=COLUMN_STACK_STRETCH)
        left.addWidget(vstack, 1)
        self.status = QtWidgets.QLabel('')
        self.status.setWordWrap(True)
        # A red flag label for run-skips, mirroring the Correlogram tab so a skipped
        # measurement uses the same channel everywhere (style guide §5 R5.5) rather than
        # hiding on the muted status line. Set via the shared tiered renderer.
        self.flag_label = ThemedLabel('', role='error', bold=True)
        self.flag_label.setWordWrap(True)
        pin_under_column(splitter, self.export_button, self.status,
                         self.notes_block, self.flag_label)

        self.figure, self.canvas, nav_toolbar, self.axis_bar = make_plot_pane(
            self, self.controller, (5.3, 4.4), constrained_layout=True)
        # The distribution + residual do NOT share x (both carry their own x-label), so a
        # generous gap is needed for the grip line to clear the main plot's "Rh (nm)"
        # label (the indicator must not overlap the plots).
        _widen_row_gap(self.figure, hspace=0.20)
        # Main distribution panel + a residual panel below (like the Correlogram
        # tab). The residual x-axis is the delay time τ, NOT the Rh/Γ axis, so the
        # two do not share x. The residual is height-adjustable via a draggable handle
        # via that handle; a flat gridspec keeps constrained_layout happy.
        self._resid_ratio = 0.22
        self._resid_dragged = False              # set by the drag handle; see the floor
        a = (1.0 - self._resid_ratio) / 3.0
        self._gs = self.figure.add_gridspec(4, 1, height_ratios=[a, a, a, self._resid_ratio])
        self.ax = self.figure.add_subplot(self._gs[0:3, 0])
        self.resid_ax = self.figure.add_subplot(self._gs[3, 0])
        right.addWidget(nav_toolbar)
        # Bottom-panel selector: swap the lower panel between the fit residuals and
        # the CONTIN α-selection sweep (the diagnostic behind the automatic α). The
        # α-sweep is CONTIN-only; it draws nothing for NNLS/lognormal runs. Mirrors the
        # Utilities Diagnostic combo. The "Export α-sweep…" button rides alongside it so
        # the export sits next to the view it writes.
        brow = QtWidgets.QHBoxLayout(); brow.setContentsMargins(0, 0, 0, 0)
        brow.addWidget(QtWidgets.QLabel('Bottom panel:'))
        self.bottom_combo = QtWidgets.QComboBox()
        self.bottom_combo.addItem('Residuals', 'residuals')
        self.bottom_combo.addItem('α-sweep (CONTIN)', 'alpha')
        self.bottom_combo.setToolTip(
            'What the lower panel shows.\n'
            '• Residuals: data − reconstructed g₂−1 vs delay time, for the '
            'methods ticked on this sub-tab, each in its own line style; the parametric '
            'fits\' are on the Correlogram sub-tab.\n'
            '• α-sweep: how CONTIN chose its regularization α — the L-curve corner, '
            'the GCV minimum, or the F-test curve (Theory-and-Equations-Guide §3.4.5). '
            'CONTIN results only.')
        self.bottom_combo.currentIndexChanged.connect(self._on_bottom_changed)
        brow.addWidget(self.bottom_combo)
        brow.addWidget(HelpBadge(
            'Read the α-sweep to judge whether to trust CONTIN’s automatic α.',
            bullets=[
                '<b>L-curve</b>: a sharp <b>corner</b> is a clean trade-off; a rounded one '
                'means α is poorly determined.',
                '<b>GCV</b>: a distinct <b>minimum</b> is a confident pick; a flat valley '
                'is ambiguous.',
                '<b>F-test</b>: the curve climbs with α; the pick is the smoothest fit not '
                'significantly worse than the best.',
                'A <b>red halo</b> marks an α on a sweep edge, no L-curve corner or a '
                'one-point sweep; the F-test’s own two notes draw none and are lines under '
                'the peak table.',
                'One sweep per CONTIN curve, colored to match the distribution above '
                '(User-Manual §8.3.2).',
            ]))
        self.export_sweep_button = QtWidgets.QPushButton('Export α-sweep…')
        self.export_sweep_button.setEnabled(False)
        self.export_sweep_button.setToolTip(
            'Write the α-sweep (alphas, norms, and the GCV/F-test score curve) to an '
            'Origin CSV — one file per CONTIN curve.')
        self.export_sweep_button.clicked.connect(self._on_export_sweep)
        brow.addWidget(self.export_sweep_button)
        brow.addStretch(1)
        bholder = QtWidgets.QWidget(); bholder.setLayout(brow)
        right.addWidget(bholder)
        right.addWidget(self.canvas, 1)
        self._resid_resizer = attach_residual_resizer(
            self.canvas, self.figure, self.ax, self.resid_ax, self._on_resid_dragged)
        right.addWidget(self.axis_bar)
        self._update_weighting_controls_visibility()
        self._clear('Tick measurements + methods, then Run.')
        # Connected last, once the axes they redraw exist: a method box and every fit
        # field choose which stored fits are shown, so each moves the plot at once.
        for cb in self.method_checks.values():
            cb.toggled.connect(self._on_fit_input_moved)
        self.fit_fields.changed.connect(self._on_fit_input_moved)

    def reseed_from_settings(self) -> None:
        """(Re)seed the fit fields from the global Settings defaults. The per-run values
        the user sets here always win (run_distribution honors explicit kwargs over its
        settings setdefault)."""
        self.fit_fields.seed_from_settings(self.controller.settings)

    def reshow(self) -> None:
        """Re-read what the controller holds after a Settings apply, whose clear may have
        taken fits or failure lines this tab shows: the tab redraws and the Correlogram
        tab re-reads, as on a tick change."""
        self._draw()
        self.fitsChanged.emit([])

    def _grid_kwargs(self, method: str) -> Dict:
        """This tab's per-run fit kwargs for `method` -- the shared group's
        `fit_kwargs`. Riding here also puts them in `_fit_signature`, so moving a field
        marks a shown fit out of date."""
        return self.fit_fields.fit_kwargs(method)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # The shared region moves on the Correlogram tab with no signal to this one, so
        # the shown subset is re-read here: a fit the move made stale leaves the plot.
        # Through `set_measurement`, so a repaint a click left owing is paid here and the
        # flush on the sub-tab switch that follows finds none.
        self.set_measurement(self.item_id, self._runnable)

    # ---- what the ticked runs can fit ----
    def _refresh_fit_availability(self) -> None:
        """Judge every method, sphere model and (run, method) pair on the ticked runs,
        one run at a time as Run fits them (`Controller.dls_fit_availability`). A method
        box is disabled, its tooltip what is missing, only when no ticked run has the
        method's own inputs: a run that cannot be built at all is named on the status
        line instead. A sphere-model entry grays when no ticked run can be re-weighted
        under it. Run is on while one ticked pair can fit. A disabled box keeps its tick
        and a grayed model stays selected."""
        ws = self.controller.workspace.measurements
        ids = [i for i in self.selection.ids() if i in ws and ws[i].kind == 'dls']
        self._availability = {iid: self.controller.dls_fit_availability([iid])
                              for iid in ids}
        verdicts = list(self._availability.values())
        universal = {(r.code, r.phrase, r.param) for r in DLS_UNIVERSAL_REQUIREMENTS}
        for _label, key in _DIST_METHODS:
            fit, cb = ('distribution', key), self.method_checks[key]
            own = [tuple(m for m in v[fit].missing
                         if (m.code, m.phrase, m.param) not in universal)
                   for v in verdicts]
            usable = not own or any(not lacks for lacks in own)
            tip = self._entry_tips.setdefault(fit, cb.toolTip())
            cb.setEnabled(usable)
            cb.setToolTip(tip if usable
                          else f'Needs:\n{_needs_lines(_union_missing(fit, own))}')
        role = QtCore.Qt.ItemDataRole.ToolTipRole
        for i in range(self.model_combo.count()):
            view = ('sphere_model', self.model_combo.itemData(i))
            item = self.model_combo.model().item(i)
            tip = self._entry_tips.setdefault(view, item.data(role))
            usable = not verdicts or any(v[view].ok for v in verdicts)
            needs = _union_missing(view, [v[view].missing for v in verdicts])
            item.setEnabled(usable)
            item.setData(tip if usable else f'Needs:\n{_needs_lines(needs)}', role)
        self.run_button.setEnabled(bool(self._runnable_pairs()))
        self._update_weighting_controls_visibility()

    def _can_fit(self, iid: str, key: str) -> bool:
        """Whether the last `_refresh_fit_availability` found `iid` can run `key`."""
        verdict = self._availability.get(iid)
        return verdict is not None and verdict[('distribution', key)].ok

    def _runnable_pairs(self) -> List[Tuple[str, str]]:
        """The ticked (measurement, method) pairs a Run fits: those whose run has every
        input the method needs, in `_ticked_pairs` order."""
        return [(iid, key) for iid, key in self._ticked_pairs() if self._can_fit(iid, key)]

    def _left_out(self) -> List[Tuple[str, tuple]]:
        """Each ticked run some ticked method cannot fit, with every input those methods
        lack, for the status line."""
        lacks: Dict[str, Dict] = {}
        for iid, key in self._ticked_pairs():
            verdict = self._availability.get(iid)
            if verdict is not None and not verdict[('distribution', key)].ok:
                lacks.setdefault(iid, {}).update(
                    dict.fromkeys(verdict[('distribution', key)].missing))
        return [(iid, tuple(missing)) for iid, missing in lacks.items()]

    def _set_status(self, text: str, view_left_out=()) -> None:
        """The status line: each ticked run Run leaves out and what it lacks, then each
        run the weighted view leaves out (`view_left_out`, `(iid, missing)` pairs), then
        `text`."""
        lines = [*_left_out_lines(self.controller, self._left_out()),
                 *_left_out_lines(self.controller, view_left_out,
                                  what='is left out of this view:')]
        self.status.setText('\n'.join([*lines, text] if text else lines))

    # ---- selection ----
    def focus_state(self, item_id: Optional[str], runnable: bool) -> bool:
        """The state half (`_DeferredRepaint`): the focus alone."""
        self.item_id = item_id
        self._runnable = runnable
        return True

    def repaint_view(self) -> None:
        """The repaint half: the picker and the plot, from the controller as it is now."""
        # Decoupled from sidebar selection: the checklist
        # is the only thing that picks measurements for analysis.
        self.checklist.refresh()
        # The shell re-points here after a commit, so the shown subset is re-read: a fit
        # whose committed params moved leaves the plot and its run reads "press Run".
        self._draw()

    def _fit_signature(self, iid: str, key: str) -> tuple:
        """Everything a `key` distribution fit of `iid` reads beyond its raw correlogram:
        the measurement's committed params, the method, the kwargs the Run hands the
        engine (the shared window + this measurement's baseline over it, the Rh grid and,
        for CONTIN, the α range, the selector and its level or γ), and the settings seed
        the engine fills in itself (skip_initial_channels). Stamped on each stored fit at
        dispatch and compared on every redraw, so a fit fitted under a window, grid or
        commit since moved is flagged stale on its own (the two tabs share one region,
        so a Correlogram-tab edit reaches here). The region's applied parts ride in their
        own slot too, as on the Correlogram tab, so Reset flags a shown fit stale even
        where the kwargs happen not to move.

        No term for the ticked set: a fit of one measurement does not read it, and
        `RegionState.applied()` reads only the handles the user has set, so a tick
        change moves no fit's signature."""
        return (self.controller.committed_signature([iid]), key,
                tuple(sorted({**self._baseline_kwargs(iid, key),
                              **self._grid_kwargs(key)}.items())),
                self.controller.settings.skip_initial_channels,
                self.region.state().applied())

    def _fit_is_stale(self, iid: str, key: str) -> bool:
        """Whether the stored fit of `(iid, key)` was computed from inputs that have
        since moved. False when there is no such fit -- absent is not stale."""
        run = self._stored(iid, key)
        return run is not None and run.sig != self._fit_signature(iid, key)

    def _stored(self, iid: str, key: str):
        """The `key` fit of `iid` this tab ran (`Controller.shown_run`), else None."""
        return self.controller.shown_run(('distribution', iid, key))

    def _failure(self, iid: str, key: str) -> Optional[str]:
        """Why the last Run's `key` fit of `iid` was refused, else None."""
        return self.controller.failure_line(('distribution', iid, key))

    def _ticked_pairs(self) -> List[Tuple[str, str]]:
        """Every (measurement, method) pair the ticks ask to see: the ticked DLS
        measurements still in the workspace, in selection order, by the ticked method
        boxes, in `_DIST_METHODS` order."""
        ws = self.controller.workspace.measurements
        ids = [i for i in self.selection.ids() if i in ws and ws[i].kind == 'dls']
        return [(iid, key) for iid in ids for _label, key in self._ticked_methods()]

    def _shown(self) -> List[Tuple[str, str, object]]:
        """`[(iid, method, result)]` for every FRESH stored fit of a ticked pair -- the
        one subset the plot, the peak table, the α-sweep panel, both exports and the
        status and flag lines read, so none of them can speak for a fit the others do
        not show. A stale fit keeps its slot and is not shown."""
        out = []
        for iid, key in self._ticked_pairs():
            run = self._stored(iid, key)
            if run is not None and run.sig == self._fit_signature(iid, key):
                out.append((iid, key, run.result))
        return out

    def _unshown_note(self) -> str:
        """The status sentence naming each ticked pair that is not drawn: a stale fit
        takes the stale hint, a missing one asks for a Run. A pair that failed in the
        last Run is left to the red flag, which names it with its reason (the
        Correlogram tab's rule), and a missing or stale one Run cannot fit to the left-out
        lines (`_left_out`). '' when every ticked pair is shown."""
        ws = self.controller.workspace.measurements
        stale: Dict[str, List[str]] = {}
        missing: Dict[str, List[str]] = {}
        for iid, key in self._ticked_pairs():
            if self._failure(iid, key) is not None:
                continue
            v = self._stored(iid, key)
            if v is not None and not self._fit_is_stale(iid, key):
                continue
            if not self._can_fit(iid, key):
                continue                  # Run cannot refresh it: the left-out lines say why
            (stale if v is not None else missing).setdefault(
                measurement_display_name(ws[iid]), []).append(_fit_label(key))

        def named(runs: Dict[str, List[str]]) -> str:
            return '; '.join(f'{name} ({", ".join(keys)})' for name, keys in runs.items())

        parts = []
        if stale:
            parts.append(f'{named(stale)}: {_STALE_HINT}')
        if missing:
            parts.append(f'No fit yet for {named(missing)} — press Run.')
        return '   '.join(parts)

    def _ticked_failures(self) -> List[str]:
        """The last Run's failure messages for the pairs ticked now, in tick order."""
        return [msg for msg in (self._failure(*p) for p in self._ticked_pairs())
                if msg is not None]

    @QtCore.Slot()
    def _on_selection_changed(self) -> None:
        """The ticked set changed. Every stored fit is kept -- each is a fit of one
        measurement and reads nothing of the ticked set -- so the plot is redrawn from
        the store: unticking hides a run's curves and ticking it again shows them while
        they are fresh. A redraw only, on the axes built once (`ax.clear()`, never a new
        axes), and the Correlogram tab's layers list is told to re-read `stored_fits`.
        The ticks move from any co-plot tab, so a repaint this tab owes runs first."""
        self.flush_view()
        self._run_epoch += 1
        self._draw()
        self.fitsChanged.emit([])

    def _on_fit_input_moved(self, *_args) -> None:
        """A method box or a fit field moved: redraw from the store, so a fit whose
        input moved -- or whose method box is unticked -- leaves the plot and each
        ticked run without a current fit reads "press Run". Nothing leaves the store:
        ticking the method again, or moving the field back, shows those fits without a
        Run. The Correlogram tab's `_on_fit_input_moved` shape."""
        self._draw()

    @QtCore.Slot()
    def _on_axis_changed(self) -> None:
        if self._shown():
            self._draw()                          # re-plot the shown fits on the new axis

    @QtCore.Slot()
    def _on_weighting_changed(self) -> None:
        """Weighting or sphere-model changed: update the controls, then re-draw the SHOWN
        intensity fits under the new weighting (a re-presentation — never a re-fit)."""
        self._refresh_fit_availability()
        if self._shown():
            self._draw()

    def _active_weighting(self) -> Tuple[str, Optional[str]]:
        """(weighting, model) currently selected; model is None for the intensity view."""
        weighting = self.weighting_combo.currentData()
        if weighting == INTENSITY:
            return INTENSITY, None
        return weighting, self.model_combo.currentData()

    def _update_weighting_controls_visibility(self) -> None:
        """Reveal the sphere-model row only for a non-intensity weighting, and the
        solute-RI reminder for the RDG and Mie models (both form m = n_solute/n_medium)
        while a ticked measurement carries one: with none, the grayed entries' tooltips
        say what is missing."""
        weighting = self.weighting_combo.currentData()
        non_intensity = weighting != INTENSITY
        self.model_combo.setVisible(non_intensity)
        if self._model_label is not None:
            self._model_label.setVisible(non_intensity)
        needs_ri = non_intensity and self.model_combo.currentData() in (RDG, MIE)
        text = self._solute_ri_summary() if needs_ri else ''
        self.solute_ri_hint.setText(text)
        self.solute_ri_hint.setVisible(bool(text))
        if self._solute_ri_hint_label is not None:
            self._solute_ri_hint_label.setVisible(bool(text))

    def _solute_ri_summary(self) -> str:
        """A read-only reminder of the ticked measurements' committed solute RI (edited on
        the Data tab, invariant 5): the value or values set, '' when none is."""
        ws = self.controller.workspace.measurements
        ids = [i for i in self.selection.ids() if i in ws and ws[i].kind == 'dls']
        present = [v for v in (ws[iid].committed_params.get('solute_refractive_index')
                               for iid in ids) if v is not None]
        if not present:
            return ''
        uniq = sorted({round(float(v), 6) for v in present})
        shown = ', '.join(f'{v:g}' for v in uniq)
        return f'Solute refractive index: {shown} (from the Data tab).'

    def _ticked_methods(self) -> List[Tuple[str, str]]:
        return [(label, key) for label, key in _DIST_METHODS
                if self.method_checks[key].isChecked()]

    def _baseline_kwargs(self, iid: str, method: str) -> Dict:
        """What a `method` inversion of `iid` reads from the shared region -- the one
        reading (`_region_fit_kwargs`): the window once set, and a baseline value only
        once the span is placed, measured over its part inside the window. An untouched
        span sends none, and the engine estimates its own over the window's tail."""
        return _region_fit_kwargs(self.controller, self.region, iid, method)

    @QtCore.Slot()
    def _on_run(self) -> None:
        ws = self.controller.workspace.measurements
        ids = [i for i in self.selection.ids() if i in ws and ws[i].kind == 'dls']
        methods = self._ticked_methods()
        if not ids:
            self.status.setText('Tick at least one measurement.'); return
        if not methods:
            self.status.setText('Tick at least one distribution method.'); return
        if any(key == 'contin' for _label, key in methods):
            problem = self.fit_fields.alpha_problem()
            if problem:                       # never run an α the engine refuses
                self.status.setText(problem); return
        # Only the pairs whose run has the method's inputs: the rest are named on the
        # status line, and a failure recorded below is one the fit itself raised.
        self._refresh_fit_availability()
        pairs = self._runnable_pairs()
        if not pairs:
            return
        # Everything the fit needs is read from the widgets HERE, on the main
        # thread; the thunk below runs on the worker and touches only the
        # controller (the whole method is dispatched regardless of `key`, so a
        # future distribution method inherits background execution for free).
        jobs = [(iid, key, {**self._baseline_kwargs(iid, key), **self._grid_kwargs(key)})
                for iid, key in pairs]
        # Each fit's signature is taken HERE too, beside the kwargs it describes: a field
        # moved while the fit runs then leaves it stale on arrival, never fresh.
        sigs = {(iid, key): self._fit_signature(iid, key) for iid, key, _kw in jobs}
        labels = {iid: measurement_display_name(ws[iid]) for iid in ids}
        controller = self.controller

        def work():
            results, failures = [], []
            for iid, key, kw in jobs:
                try:
                    res = controller.run_distribution(iid, key, **kw)
                    results.append((iid, key, res))
                except Exception as exc:  # noqa: BLE001 — per (measurement, method), recorded; per (measurement, method)
                    failures.append((iid, key, f'{labels[iid]}·{key.upper()}: '
                                               f'{humanize_field_names(str(exc))}'))
            return results, failures

        def done(payload, _token) -> None:
            # Every finished fit lands in its own slot whatever is ticked now: it is a
            # valid fit of that (measurement, method), and only its display is gated
            # (`_shown`, ui_style_guide.md §4.1 item 2).
            results, failures = payload
            for iid, key, res in results:
                controller.record_shown_run(('distribution', iid, key), res,
                                            sigs[(iid, key)])
            for iid, key, _msg in failures:
                # Only THIS pair's slot: a failed NNLS must not drop the run's CONTIN.
                controller.forget_shown_run(('distribution', iid, key))
            controller.replace_run_failures(
                ('distribution',),
                {('distribution', iid, key): msg for iid, key, msg in failures})
            self._draw()
            # The Correlogram tab can draw these too: tell it they exist, and which
            # methods to tick (a fresh Run ticks its own method).
            self.fitsChanged.emit(
                list(dict.fromkeys(key for _iid, key, _res in results)))

        def fail(exc: BaseException, token) -> None:
            # A whole-Run failure landing after a tick change must not write onto the
            # view now shown: the epoch a tick change bumps, read here only.
            QtWidgets.QMessageBox.critical(
                self, 'Analysis failed',
                f'Could not run the distribution fit.\n\n'
                f'{humanize_field_names(str(exc))}')
            self._refresh_fit_availability()   # the worker re-enabled Run; the ticks decide
            if self._run_slot.current(token):
                self.status.setText('Analysis failed — see dialog.')

        if self._run_slot.submit(
                work, done, fail, target=None, queue=False,
                busy_widgets=(self.run_button,),
                notice=lambda: self.status.setText(BUSY_NOTICE),
                description='distribution fit', bump=False):
            self.status.setText('Fitting in the background…')

    @property
    def _run_epoch(self) -> int:
        """The Run's staleness epoch, held by its slot."""
        return self._run_slot.epoch

    @_run_epoch.setter
    def _run_epoch(self, value: int) -> None:
        self._run_slot.epoch = value

    @staticmethod
    def _as_distribution(result):
        return getattr(result, 'distribution', result)

    def stored_fits(self) -> Dict[Tuple[str, str], Dict]:
        """Every fit this tab has stored, keyed `(iid, method)`, each with its OWN
        staleness verdict (`_fit_is_stale`) -- the accessor the Correlogram tab reads
        through its owning module, which filters by its own ticks.

        Per fit, not per Run: each fit keeps its own slot and signature, so one run's
        fit can be fresh while another's is stale. Neither this tab's ticks nor its
        method boxes filter the answer; they choose only what THIS tab shows.
        """
        return {(iid, key): {'result': self._as_distribution(run.result),
                             'stale': self._fit_is_stale(iid, key)}
                for (_kind, iid, key), run in
                self.controller.shown_runs('distribution').items()}

    @QtCore.Slot()
    def _on_export(self) -> None:
        items = self._shown()
        if not items:
            return
        weighting, _model = self._active_weighting()
        weighted = weighting != INTENSITY
        # In a weighted view the plot forces the Rh axis (the axis combo is disabled but may
        # still hold a stale 'gamma' selection). Match it for the intensity FALLBACK export
        # too, so a reweight-errored curve never writes a Γ-axis (decay-rate) CSV into a file
        # offered as a weighted size distribution.
        axis = 'rh' if weighted else self.axis_combo.currentData()
        wd_by = {(iid, key): wd for iid, key, wd in self._weighted}

        name, suffixes = _selection_export_names(
            self.controller, [(iid, key) for iid, key, _res in items],
            'weighted_distribution' if weighted else 'distribution')

        def do_export(path: str):
            import os
            stem, ext = os.path.splitext(path)
            paths = [f'{stem}{sfx}{ext}' for sfx in suffixes]
            siblings = [] if len(items) == 1 else paths
            if not confirm_overwrite(self, siblings):
                return None
            written = []
            for (iid, key, res), p in zip(items, paths, strict=True):
                wd = wd_by.get((iid, key)) if weighted else None
                if wd is not None:
                    written.append(
                        self.controller.export_weighted_distribution(wd, p, item_id=iid))
                else:
                    # Intensity view, or a curve whose reweight errored: export the
                    # intensity distribution (never silently write nothing).
                    written.append(
                        self.controller.export_distribution(res, p, axis, item_id=iid))
            return written                     # summarised by `export_to_csv`

        status = export_to_csv(self, self.controller, name, do_export)
        if status:
            self.status.setText(status)

    def _draw(self) -> None:
        """Redraw the plot, the peak table, the bottom panel and the status and flag
        lines from the store's SHOWN subset (`_shown`) -- the one redraw every gesture
        ends in: a Run landing, a tick, a method box, a fit field, a view control, a
        re-point. On the axes built once; nothing shown empties them through `_clear`.
        Every gesture passes through here, so what the ticked runs can fit is re-read
        first."""
        self._refresh_fit_availability()
        shown = self._shown()
        self.export_button.setEnabled(bool(shown))
        if not shown:
            self._clear('Tick measurements + methods, then Run.'
                        if not self._ticked_pairs()
                        else 'No current fit for the ticked measurements — press Run.'
                        if self._runnable_pairs()
                        else 'No current fit for the ticked measurements.')
            self._set_status(self._unshown_note())
            # `_clear` drops the flag; a ticked pair's failure in the last Run is still
            # news, and with nothing drawn the flag is where it can still be read.
            skipped = self._ticked_failures()
            set_flag(self.flag_label, 'skipped: ' + '; '.join(skipped) if skipped else '',
                     problem=True)
            return
        if self._placeholder is not None:    # leaving the empty-state: drop the message
            self._placeholder.remove()
            self._placeholder = None
        self.ax.clear()
        self.resid_ax.clear()
        self.ax.set_visible(True)            # restore after an empty _clear() hid both
        self.resid_ax.set_visible(True)
        self.axis_bar.attach(self.ax)        # re-bind the axis controls (detached while empty)
        weighting, model = self._active_weighting()
        weighted = weighting != INTENSITY
        # A weighted view is a size re-expression → always plotted vs Rh; the Γ axis stays
        # meaningful only for the intensity view (so disable it while reweighted).
        axis = 'rh' if weighted else self.axis_combo.currentData()
        self.axis_combo.setEnabled(not weighted)
        # Plot saves suggest the same stem as the CSV export for the active view.
        set_plot_export_name(
            self.canvas,
            _selection_export_names(
                self.controller, [(iid, key) for iid, key, _res in shown],
                'weighted_distribution' if weighted else 'distribution')[0])
        ws = self.controller.workspace.measurements
        single = len(shown) == 1
        peak_items = []          # (x, y, text, color) collected across all curves
        # Legend material, in draw order: the name each ticked measurement is listed
        # under, the methods actually drawn, and which of a run's methods failed.
        legend_names: Dict[str, str] = {}
        legend_samples: Dict[str, str] = {}
        methods_drawn: List[str] = []
        failed_methods: Dict[str, List[str]] = {}
        drawn: List[Tuple] = []      # (iid, method, distribution) actually plotted
        self._weighted = []
        reweight_errors: List[str] = []
        view_left_out: Dict[str, tuple] = {}     # iid -> what the view's model lacks
        labelled_invalid = False        # label the shared 'outside model validity' entry once
        # As on the Correlogram tab: a legend carries the sample only when the ticked
        # set spans samples.
        prefixes = _sample_prefixes(self.controller,
                                    [iid for iid, _k, _r in shown])
        for iid, key, res in shown:
            d = self._as_distribution(res)
            color = self.selection.color_for(iid)
            # Colour + MARKER are the measurement, line STYLE is the method -- the
            # Correlogram tab's coding, so a dash pattern means the same thing on both.
            # The marker is what carries "which measurement" for a reader who cannot use
            # the colour (R10.1), now that line style has been taken by the method; it
            # is placed sparsely, since a marker per inversion-grid point is a band.
            marker = self.selection.marker_for(iid)
            ls = _METHOD_LINESTYLE.get(key, '-')
            name = measurement_display_name(ws[iid], sample=prefixes[iid])
            # The legend bookkeeping is at the BOTTOM of this loop, where the curve
            # is actually drawn: up here, a skipped curve still got an entry.
            # A failed lognormal fit draws the seed estimate, not a fit -- say so in
            # the legend (module docstring, owner-ruled: a SAVED image must not present
            # a seed curve as fitted). With one entry per measurement rather than one
            # per curve, the note has to name WHICH method failed, since the entry now
            # stands for every method drawn for that run.
            if not getattr(d, 'success', True):
                failed_methods.setdefault(iid, []).append(_fit_label(key))
            # Every curve is drawn but none is LISTED: the legend below carries one
            # entry per measurement plus one per method, not one per curve. Nine ticked
            # runs drew nine `name . METHOD` entries, which overran the axes on both
            # sides and covered the y-label (measured 2026-09-18).
            label = '_nolegend_'
            wd = None
            view = self._availability[iid][('sphere_model', model)] if weighted else None
            if view is not None and not view.ok:
                # A run the model cannot re-weight is named on the status line, never
                # handed to the re-weighting; its curve is skipped below.
                view_left_out.setdefault(iid, view.missing)
            elif weighted:
                try:
                    wd = self.controller.reweight_distribution(iid, d, weighting, model)
                except OpticalModelError as exc:
                    reweight_errors.append(f'{measurement_display_name(ws[iid])}: {exc}')
                except Exception as exc:  # noqa: BLE001 — surfaced in reweight_errors; defensive
                    reweight_errors.append(
                        f'{measurement_display_name(ws[iid])}: {humanize_field_names(str(exc))}')
            self._weighted.append((iid, key, wd))
            if wd is not None:
                # Label the shared out-of-range marker only on the FIRST flagged curve, so
                # an overlay's legend never repeats it -- without a second ax.legend() dedup
                # pass (avoiding extra Agg/Qt artist churn).
                inv_lbl = ('outside model validity'
                           if (not wd.all_valid and not labelled_invalid) else None)
                if not wd.all_valid:
                    labelled_invalid = True
                plot_weighted_distribution(
                    wd, ax=self.ax, label=label, color=color, fill=single,
                    linestyle=ls, marker=marker, invalid_label=inv_lbl)
                peak_items.extend(self._collect_weighted_peak_labels(wd, color))
            elif not weighted:
                plot_distribution(d, ax=self.ax, axis=axis, label=label,
                                  color=color, fill=single,
                                  linestyle=ls, marker=marker)
                peak_items.extend(self._collect_peak_labels(d, res, axis, color))
            else:
                # weighted-but-wd-None (a run left out of the view, or a reweight error):
                # the curve is skipped and the status line or the flag explains. Nothing
                # about it is listed, and it gets no residual.
                continue
            drawn.append((iid, key, d))
            legend_names.setdefault(iid, name)
            legend_samples.setdefault(
                iid, prefixes[iid] or _sample_label_of(self.controller, iid))
            methods_drawn.append(key)
        self._drawn = drawn
        self._reweight_errors = reweight_errors
        # One de-collided pass so peaks that coincide across curves stagger (capped at
        # 6 + "+N more") instead of piling into an unreadable blob.
        annotate_decollided(self.ax, peak_items)
        self._draw_legend(legend_names, methods_drawn, failed_methods,
                          groups=legend_samples)
        self.ax.set_title(
            f'DLS {WEIGHTING_LABELS[weighting].lower()}-weighted size distribution'
            if weighted else 'DLS size / rate distribution')
        self._draw_bottom()                  # residuals OR the CONTIN α-sweep
        self.canvas.draw_idle()
        self._floor_resid_ratio()            # after the layout is real, not before
        self.axis_bar.attach(self.ax)
        # Status now carries only operational notes; per-result peaks live in the
        # Summary tab (which is also where they persist).
        notes = []
        n_contin = sum(1 for _i, k, _r in shown if k == 'contin')
        if n_contin > 3:
            notes.append('CONTIN runs an α sweep per measurement — this can be slow.')
        # α-selection provenance: every CONTIN fit shown is fresh, and a fresh fit's
        # signature carries the α-selection fields as they stand now, so every one ran
        # under the one selector the control shows and one note names it (method
        # disclosure — a distribution is never ambiguous about how α was chosen). The
        # wording comes from the shared `alpha_selection_note` helper the CSV export also
        # uses, so the two surfaces can't drift as selectors are added. A user-supplied
        # α returns '' here (the α input control already shows it).
        contin_res = next((r for _i, k, r in shown if k == 'contin'), None)
        alpha_note = alpha_selection_note(
            getattr(contin_res, 'alpha_selection_method', None),
            getattr(contin_res, 'ftest_prob_reject', None),
            gcv_gamma=getattr(contin_res, 'gcv_gamma', None), style='gui')
        if alpha_note:
            notes.append(alpha_note)
        if self.controller.is_dirty():
            notes.append('(ran on last committed values)')
        unshown = self._unshown_note()       # the ticked runs this plot is NOT showing
        if unshown:
            notes.append(unshown)
        notes.append('Peak values are also in the Summary tab.')
        self._set_status('   '.join(notes), view_left_out.items())
        # The notes of every shown fit, of its peaks and of its reweighted view, one line
        # per failed precondition (a CONTIN alpha at a sweep edge, a peak whose top is the
        # Rh grid's end point, a reweighted bin outside the model range), before the peak
        # table so each marker's letter is its line's. A peak's verdict is its fit's in
        # either view, so its line stays while a reweighted table shows other lobes.
        names = {(iid, key): f'{measurement_display_name(ws[iid])} · {key.upper()}'
                 for iid, key, _r in shown}
        wd_by = {(iid, key): wd for iid, key, wd in self._weighted}
        fed = [(names[(iid, key)], res) for iid, key, res in shown]
        fed += [(names[(iid, key)], tuple(
                    n for *_v, peak_notes in self.controller.distribution_peaks(
                        res, with_notes=True) for n in peak_notes))
                for iid, key, res in shown]
        fed += [(names[p], wd) for p, wd in wd_by.items() if wd is not None and p in names]
        self.notes_block.show_notes(fed, n_total=len(shown))
        self._block_fits = {id(src) for _name, src in fed}
        # Operational flags only (ui_style_guide §5), on the red tier as on the
        # Correlogram tab (R5.5): a pair the last Run skipped, a reweighting that could
        # not run, and a shown fit that failed with no note to say why.
        flag_msgs = []
        failures = self._ticked_failures()     # the last Run's, for the pairs ticked now
        if failures:
            flag_msgs.append('skipped: ' + '; '.join(failures))
        if getattr(self, '_reweight_errors', None):
            flag_msgs.append('reweighting skipped — ' + '; '.join(self._reweight_errors))
        flag_msgs += [f'{measurement_display_name(ws[iid])}: {_fit_label(key)} fit failed'
                      for iid, key, res in shown
                      if _failed_without_cause(res, self._block_fits)]
        set_flag(self.flag_label, '; '.join(flag_msgs), problem=True)
        # The qualifiers of `_UNLABELLED_QUALIFIERS`, each on one line naming its fits, on
        # the neutral tier.
        weights = [(names[(iid, key)],
                    tuple(q for q in qualifiers(d) if q.code in _UNLABELLED_QUALIFIERS))
                   for iid, key, d in drawn]
        set_flag(self.info_label, '\n'.join(
            render(q, counted=True) for q, _names in grouped(weights, kinds=('qualifier',),
                                               what='fits', n_total=len(shown))),
            problem=False)
        self._refresh_results(shown)

    def _draw_legend(self, names, methods_drawn, failed, groups=None) -> None:
        """The on-screen KEY, and the dressing that gives the SAVED figure the full
        legend -- the Correlogram tab's split, with one difference forced by layout.

        THIS figure is a plain 4x1 stack, so it has no empty corner cell to put the key
        in; it goes below the panel, where four short entries cost about 30 px. The
        CONTENT and the coding are identical to the other tab's: measurement names on
        the written file, a method key on screen.

        Any artist the plotters labelled themselves -- the shared "outside model
        validity" marker -- is kept, ahead of the entries built here. A failed
        lognormal's note rides its measurement entry into the export legend, which is
        the surface the owner's ruling is about (a saved image must not present a seed
        curve as fitted); on screen the same fact is on the status flag.
        """
        if self.ax.get_legend() is not None:
            self.ax.get_legend().remove()
        plotter = list(zip(*self.ax.get_legend_handles_labels(), strict=True))             if self.ax.get_legend_handles_labels()[1] else []
        key_entries = plotter + _method_legend_entries(dict.fromkeys(methods_drawn))
        _draw_key(self.figure, key_entries)

        by_sample = {}
        for iid, name in names.items():
            note = failed.get(iid)
            by_sample.setdefault((groups or {}).get(iid) or '', []).append(
                (iid, name + (f' ({", ".join(note)} failed)' if note else '')))
        meas_entries, group_rows = _measurement_legend_entries(
            self.selection, by_sample)
        set_plot_export_dressing(
            self.canvas,
            export_legend_dressing(self.figure, meas_entries + key_entries,
                                   group_rows)
            if names else None)

    # ---- bottom panel: residuals <-> CONTIN α-sweep ----
    def _contin_results(self) -> list:
        """The shown CONTIN fits (in tick order) — the only ones with an α-sweep."""
        return [(iid, res) for iid, key, res in self._shown() if key == 'contin']

    def _draw_bottom(self) -> None:
        """Draw the lower panel per the combo: fit residuals, or the CONTIN α-sweep.
        Called from `_draw` on a cleared `resid_ax`; also updates the α-sweep export
        button's enabled state (α-sweep view AND ≥1 CONTIN result)."""
        view = self.bottom_combo.currentData()
        contin = self._contin_results()
        self.export_sweep_button.setEnabled(view == 'alpha' and bool(contin))
        if view == 'alpha':
            self._draw_alpha_sweep(contin)
        else:
            self._draw_residuals()

    def _draw_residuals(self) -> None:
        """Per-curve fit residuals (data − reconstructed g₂−1) vs delay time, through
        the one helper the Correlogram tab's residual panel also uses -- so the two
        panels cannot disagree about what a residual panel shows or how it is coded.

        THIS tab's ticked methods only: a parametric residual never appears here, and a
        distribution residual can appear here and on the Correlogram tab alike. Before
        the shared helper this drew every method of one measurement in one colour and
        one solid line, so NNLS and CONTIN could not be told apart; the method line
        style is what fixes that.
        """
        draw_fit_residuals(
            self.resid_ax,
            [(self.selection.color_for(iid), key, d)
             for iid, key, d in getattr(self, '_drawn', [])])
        # NO `_hide_shared_x` here. This tab's two panels do NOT share an x axis --
        # the distribution is drawn against Rh (or Γ) and the residual against delay
        # time τ, which this file's own construction comment states twice and
        # `_widen_row_gap(hspace=0.20)` exists for. Stripping the upper panel's axis
        # left a size distribution with no axis of its own sitting over a delay-time
        # axis on an unrelated range -- a peak position readable off the wrong
        # quantity, on screen and in the saved file. The helper is the CORRELOGRAM
        # tab's, where both panels really are τ.

    def _draw_alpha_sweep(self, contin: list) -> None:
        """Overlay each ticked CONTIN result's α-selection sweep in its curve color,
        method-aware (L-curve corner / GCV minimum / F-test curve). NNLS/lognormal have
        no sweep; if nothing to draw, show the CONTIN-only empty-state message instead.

        Every CONTIN result here is FRESH (`_shown`), and a fresh fit's signature
        carries the α-selection fields as they stand now, so every one ran under the one
        selector the control shows: they share one `alpha_selection_method` and
        therefore one ordinate — the overlay never mixes an L-curve
        (residual-vs-solution) with a GCV/F-test (score-vs-α) axis on this shared panel.
        A fit run under another selector is stale: it keeps its slot and is not drawn,
        here or above. That freshness rule is what holds this premise now that fits
        persist across Runs; a shown set that could mix selectors would need a guard
        here."""
        ws = self.controller.workspace.measurements
        drawn = 0
        for iid, res in contin:
            lcurve = getattr(res, 'lcurve', None)
            if lcurve is None:
                continue
            # The halo is a POSITIONAL mark, keyed on the states that say where the pick
            # sits -- the two edge flags and the L-curve's no-corner note (its one-point
            # sweep included), which reports the least-regularized end -- never on the
            # `alpha_reliable` roll-up above them: the F-test's two precondition notes
            # are about the sweep, not the pick's position, and draw nothing here (the
            # notes block under the peak table carries them). `plot_lcurve` keys the glyph by
            # the two edge names, so no-corner rides the floor mark; the wording each
            # state gets is its note's registry clause.
            ceiling = bool(getattr(res, 'alpha_at_ceiling', False))
            positional = (ceiling or bool(getattr(res, 'alpha_at_floor', False))
                          or has_note(res, 'alpha_no_corner'))
            try:
                plot_lcurve(
                    lcurve, ax=self.resid_ax,
                    method=getattr(res, 'alpha_selection_method', 'lcurve'),
                    color=self.selection.color_for(iid),
                    label=measurement_display_name(ws[iid]) if iid in ws else None,
                    mark_ceiling=ceiling,
                    mark_floor=positional and not ceiling,
                    gcv_gamma=getattr(res, 'gcv_gamma', None))
                drawn += 1
            except ValueError:
                # No sweep to show for this result: a user-fixed α (method 'user' —
                # no sweep, only a degenerate single point) or a requested ordinate
                # whose score curve is absent (e.g. 'gcv' with gcv is None). Neither is
                # reachable from this tab's sweep controls today, but skip rather than
                # draw a misleading one-point "corner"; drawn==0 → the empty-state.
                continue
        if drawn == 0:
            self.resid_ax.set_axis_off()
            self.resid_ax.text(
                0.5, 0.5, 'No α-sweep — CONTIN only',
                transform=self.resid_ax.transAxes, ha='center', va='center',
                color=PLACEHOLDER_GRAY, fontsize=9)

    @QtCore.Slot()
    def _on_bottom_changed(self) -> None:
        """Redraw only the lower panel when the Residuals ↔ α-sweep combo changes."""
        if not self._shown():
            return
        self.resid_ax.set_axis_on()          # a prior empty-state may have hidden it
        self.resid_ax.clear()
        self._draw_bottom()
        self.canvas.draw_idle()

    @QtCore.Slot()
    def _on_export_sweep(self) -> None:
        contin = self._contin_results()
        if not contin:
            return

        # No method token: the alpha sweep is CONTIN's own diagnostic, so the analysis
        # offers no choice to name.
        name, suffixes = _selection_export_names(
            self.controller, [(iid, None) for iid, _res in contin], 'alpha_sweep')

        def do_export(path: str):
            import os
            stem, ext = os.path.splitext(path)
            paths = [f'{stem}{sfx}{ext}' for sfx in suffixes]
            siblings = [] if len(contin) == 1 else paths
            if not confirm_overwrite(self, siblings):
                return None
            written = []
            for (iid, res), p in zip(contin, paths, strict=True):
                written.append(self.controller.export_lcurve(res, p, item_id=iid))
            return written                     # summarised by `export_to_csv`

        status = export_to_csv(self, self.controller, name, do_export)
        if status:
            self.status.setText(status)

    def _refresh_results(self, shown: List[Tuple[str, str, object]]) -> None:
        """Fill the peak-results panel from the `shown` fits — one colored row per
        (measurement · method), positional-peak columns. Mirrors the
        Correlogram results panel; reuses the same peak source as the plot labels."""
        ws = self.controller.workspace.measurements
        cols = [(iid, key, res) for iid, key, res in shown if iid in ws]
        # When a reweighted view is active, the peak table shows the SAME weighting as the
        # plot (rh, fraction-in-weighting); otherwise the intensity peaks. A curve whose
        # reweight errored (wd is None) shows no peaks.
        wd_by = {(iid, key): wd for iid, key, wd in self._weighted}
        weighting, _model = self._active_weighting()
        weighted = weighting != INTENSITY

        def _peaks_for(iid, key, res):
            # (Rh, fraction, the peak's own notes): an intensity peak carries its size
            # verdict (`DistributionPeak.notes`); a reweighted view's lobes carry none.
            if weighted:
                wd = wd_by.get((iid, key))
                return ([(rh, frac, ()) for rh, frac in
                         self.controller.weighted_distribution_peaks(wd)] if wd else [])
            return [(rh, frac, notes) for rh, _g, frac, notes
                    in self.controller.distribution_peaks(res, with_notes=True)]

        peaks_per = [_peaks_for(iid, key, res) for iid, key, res in cols]
        max_peaks = max((len(p) for p in peaks_per), default=0)
        n_rows = max(max_peaks, 1)
        table = self.result_table
        table.clear()
        # One ROW per (run · method), one COLUMN per positional peak -- transposed from
        # the column-per-run shape this table had, for the reason the Correlogram glance
        # table was rebuilt: this one grows a column per measurement AND per method, so
        # it failed by construction faster than that one did. The peak count is bounded
        # (three fit across the control column); the ticked set is not.
        table.setColumnCount(1 + n_rows)
        table.setRowCount(len(cols))
        table.horizontalHeader().setVisible(True)
        # Units live in the HEADER so a cell is '29.8 · 85%' and three peaks fit.
        table.setHorizontalHeaderLabels(
            ['Measurement', *[f'{_ordinal_peak(r)} (nm · %)' for r in range(n_rows)]])
        # The weighting names the whole table, not each column: it is one control's
        # value, and repeating it per column was spending the width the peaks need. A
        # reweighted view says what it is in its qualifier's words (a model-dependent
        # view under the chosen sphere model), the same for every row.
        views = [wd for wd in wd_by.values() if wd is not None]
        words = ((qualifier_words(views[0], exclude=('distribution_weighted',))
                  if views else [WEIGHTING_LABELS[weighting]]) if weighted else [])
        # The fits' residual weighting, when it is not the default, joins the title too.
        residual, clauses = _weighting_title([res for _iid, _key, res in cols])
        words += residual
        self.results_label.setText(
            f'{_PEAK_RESULTS_TITLE} — {", ".join(words)}' if words else _PEAK_RESULTS_TITLE)
        self.results_label.setToolTip(clauses)
        # Left-elided and capped like the Correlogram glance table, same reason: a file
        # stem's TAIL is what tells replicates apart (word wrap off, as there).
        table.setTextElideMode(QtCore.Qt.TextElideMode.ElideLeft)
        table.setWordWrap(False)
        # Distribution peaks are from an ill-posed inversion — no honest ±
        # — so the Rh uses the documented fixed-sig fallback, never a fabricated ±.
        sig = self.controller.settings.no_uncertainty_sig_figs
        for r, (iid, key, res) in enumerate(cols):
            name = QtWidgets.QTableWidgetItem(
                f'{measurement_display_name(ws[iid])} · {key.upper()}'
                if iid in ws else key.upper())
            name.setForeground(QtGui.QColor(self.selection.color_for(iid)))
            name.setIcon(marker_swatch(self.selection.color_for(iid),
                                       self.selection.marker_for(iid)))
            if iid in ws:
                name.setToolTip(measurement_hover_card(ws[iid]))
            table.setItem(r, 0, name)
            peaks = peaks_per[r]
            # Each peak is a value of the fit and of its reweighted view, and carries its
            # own size verdict: a failed precondition of any of them marks it (`_draw` fed
            # each to the notes block), the peak's own on its cell alone.
            wd = wd_by.get((iid, key))
            notes = (*find_notes(res), *(find_notes(wd) if wd is not None else ()))
            for c in range(1, 1 + n_rows):
                own: Tuple[Note, ...] = ()
                if c - 1 < len(peaks):
                    rh_nm, weight, own = peaks[c - 1]
                    txt = (f'{format_fixed_sig(rh_nm, sig)} · '
                           f'{format_fixed_sig(weight * 100, sig)}%')
                else:
                    txt = '—'
                it = QtWidgets.QTableWidgetItem(txt)
                table.setItem(r, c, it)
                if txt == '—' or not self.notes_block.mark_item(
                        it, (*notes, *own), ('weights', 'mean_rh_nm')):
                    it.setToolTip(txt)
        table.resizeColumnsToContents()
        if table.columnWidth(0) > NAME_COL_MAX_W:
            table.setColumnWidth(0, NAME_COL_MAX_W)
        for c in range(1, 1 + n_rows):
            table.horizontalHeader().setSectionResizeMode(
                c, QtWidgets.QHeaderView.ResizeMode.Stretch)
        # No max-height cap: the peak table sits in a stretch slot of the vertical
        # splitter, so it should fill (and be resizable into) that slot rather than
        # stay pinned to a row-count height (which stopped the results section from
        # expanding vertically). The 22 px/row estimate also clipped rows on
        # platforms with taller default rows.

    def _clear_results(self) -> None:
        self.result_table.clear()
        self.result_table.setRowCount(0)
        self.result_table.setColumnCount(0)
        # The title's words and their clauses were the cleared fits'.
        self.results_label.setText(_PEAK_RESULTS_TITLE)
        self.results_label.setToolTip('')

    def _collect_weighted_peak_labels(self, wd, color: str) -> list:
        """Peak labels (x, y, text, color) for a reweighted view — always vs Rh, fraction
        in the chosen weighting. Uses the controller's weighted peak finder (invariant 5).
        The annotation x is scaled to the active RADIUS display unit to match the plotted
        curve (drawn at rh*display_factor), so labels don't land off-axis when the unit ≠ nm."""
        rh = np.asarray(wd.rh_grid_nm, dtype=float)
        w = np.asarray(wd.weights, dtype=float)
        sig = self.controller.settings.no_uncertainty_sig_figs   # reweighted: ±-less
        rfac = _disp_factor('radius')
        items = []
        for rh_nm, _frac in self.controller.weighted_distribution_peaks(wd):
            if rh_nm is None or not math.isfinite(rh_nm) or rh.size == 0:
                continue
            # The label prints the peak's geometric mean and sits at it; its height is the
            # curve at the nearest node on the log axis, from the RAW grid.
            y = float(w[_nearest_log_node(rh, rh_nm)])
            items.append((rh_nm * rfac, y, f'{format_fixed_sig(rh_nm, sig)} nm', color))
        return items

    def _collect_peak_labels(self, dist, res, axis: str, color: str) -> list:
        """Collect this curve's peak labels as (x, y, text, color) tuples for the
        shared de-collision pass (GUI overlay; kept on saved images too, like the
        Correlogram τ markers). Uses the controller's peak finder so the analysis layer
        is not imported here. y is the curve's height at the peak, from the grid."""
        grid = np.asarray(dist.rh_grid_nm if axis == 'rh'
                          else dist.gamma_grid_s_inv, dtype=float)
        w = np.asarray(dist.weights, dtype=float)
        sig = self.controller.settings.no_uncertainty_sig_figs   # distribution peaks: ±-less
        # Scale the annotation x to the active display unit to match the plotted curve
        # (plot_distribution draws at x*display_factor); the y lookup stays on the RAW grid.
        fac = _disp_factor('radius' if axis == 'rh' else 'decay_rate')
        items = []
        # Each label prints the peak's geometric mean (the controller's peaks are at their
        # means) and sits at that abscissa, so the plot and the peak table show one number
        # per peak; the curve's own top is the node with the most weight. The height is the
        # curve at the nearest node on the log axis.
        for rh_nm, gamma_s_inv, _weight in self.controller.distribution_peaks(res):
            x = rh_nm if axis == 'rh' else gamma_s_inv
            if x is None or not math.isfinite(x) or grid.size == 0:
                continue
            y = float(w[_nearest_log_node(grid, x)])
            txt = (f'{format_fixed_sig(rh_nm, sig)} nm' if axis == 'rh'
                   else f'{format_fixed_sig(gamma_s_inv, sig)} s⁻¹')
            items.append((x * fac, y, txt, color))
        return items

    def _clear(self, message: str) -> None:
        if self._placeholder is not None:       # ax.clear() doesn't drop a figure.text
            self._placeholder.remove()
            self._placeholder = None
        self.ax.clear()
        self.resid_ax.clear()
        # A figure legend is NOT dropped by `ax.clear()` and `show_plot_placeholder`
        # only HIDES the axes, so without this the empty-state message renders over the
        # last run's legend, naming curves that are not drawn -- and a save of that
        # state writes it. The export dressing goes with it, for the same reason.
        self.figure.legends.clear()
        set_plot_export_dressing(self.canvas, None)
        self._drawn = []
        # Hide BOTH axes and show one centred grey message via the shared helper — the
        # clean empty-state look (a blank 0–1 box reads as broken). _draw restores them.
        self._placeholder = show_plot_placeholder(
            self.figure, self.canvas, message, axes=(self.ax, self.resid_ax))
        self.axis_bar.attach(None)              # nothing to drive while empty
        self.export_sweep_button.setEnabled(False)
        self._clear_results()
        self._weighted = []
        self._reweight_errors = []
        self.axis_combo.setEnabled(True)        # the weighted view disables it; restore
        # Drop any stale skipped-run flag: this is the reset choke point, and only
        # `_draw` sets the flag -- after this call when nothing is shown, re-stating just
        # the ticked pairs' failures -- so the red alarm cannot linger.
        self.flag_label.clear()
        self.info_label.clear()
        self.notes_block.clear()
        self._block_fits = set()
