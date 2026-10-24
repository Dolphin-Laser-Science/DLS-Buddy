"""
The DLS Correlogram sub-tab: parametric fits on the four-scale correlogram with a residual
panel and the result tables.
"""
from __future__ import annotations

import math
from collections import namedtuple
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtWidgets

from matplotlib.figure import Figure

from plotting.plots import (plot_correlogram_fit_line, plot_correlogram_scaled,
                            display_factor as _disp_factor)
from core import units as U
from core.naming import measurement_display_name, measurement_hover_card, sample_display_name
from gui.plot_controls import (LazyCanvas, make_split_panels, make_canvas_expanding,
                               make_vertical_plot_stack, pin_under_column, COLUMN_STACK_STRETCH,
                               attach_residual_resizer, set_plot_export_dressing,
                               set_plot_export_name, themed_navtoolbar)
from gui.export_helper import confirm_overwrite, export_to_csv
from gui.field_labels import humanize_field_names
from gui.help import add_help_to_groupbox
from gui.notes_block import NotesBlock
from gui.plot_overlays import show_plot_placeholder, PLACEHOLDER_GRAY
from gui.theme import ThemedLabel
from gui.widgets import MeasurementPicker, marker_swatch, NAME_COL_MAX_W, RichTextHeaderView
from gui.worker import BUSY_NOTICE, runner
from core.formatting import format_fixed_sig

from gui.dls.region import _region_fit_kwargs
from gui.dls.fits import (_CORR_METHODS, _FIT_METHODS, _CORR_RESULT_KINDS, _corr_run_key,
                          _METHOD_LINESTYLE, _fit_label, _failed_without_cause, draw_fit_residuals)
from gui.dls.common import (_STALE_HINT, _STALE_UNRUNNABLE, _widen_row_gap, _weighting_title,
                            _selection_export_names, _sample_label_of, _sample_prefixes,
                            _PICKER_HELP, _GROUP_FIELD_CONC, _GROUP_FIELD_ANGLE,
                            _GROUP_PICKER_BULLETS, _left_out_lines)
from gui.dls.tables import _vtable, _fill_vtable, _repoint_vtable, _reset_vtable, _RESULT_ROWS
from gui.dls.legend import (_marker_legend_entries, _method_legend_entries,
                            _measurement_legend_entries, _draw_key, export_legend_dressing)
from gui.dls.bases import _DeferredRepaint, _ResidualRatioMixin


# Scale modes for the four correlogram views, named Yscale-Xscale -> (xscale, yscale).
_SCALE_MODES = ['lin-log', 'lin-lin', 'log-lin', 'log-log']


_SCALE_XY = {
    'lin-log': ('log', 'linear'),     # the conventional DLS view
    'lin-lin': ('linear', 'linear'),
    'log-lin': ('linear', 'log'),
    'log-log': ('log', 'log'),
}


# Characteristic decay scales of a ticked correlogram set, in canonical SECONDS, used to
# frame the linear-τ views (and, later, to place a 1/e / inflection guide line — see
# _decay_frame). All taken over the SLOWEST ticked curve.
_DecayFrame = namedtuple('_DecayFrame', 'plateau tau_1e_s tau_end_s data_max_s')


def _decay_frame(curves) -> Optional['_DecayFrame']:
    """Characteristic decay scales, in canonical SECONDS, of an iterable of
    ``(delay_times_s, g2m1)`` correlograms, taken over the SLOWEST curve (largest 1/e
    time — the smallest angle → longest decay, so a frame built from it keeps every faster
    curve visible). Returns None when there is no usable curve. Qt-free so it is unit
    tested directly.

    Fields (see :class:`_DecayFrame`): ``plateau`` (peak g₂−1 of that curve, i.e. β),
    ``tau_1e_s`` (first lag where g₂−1 falls to plateau/e — the robust steep-region
    characteristic time, and exactly where a future 1/e guide line would sit),
    ``tau_end_s`` (last lag still above ~5 % of the plateau — the "mostly decayed" point),
    ``data_max_s`` (largest lag across ALL curves)."""
    best = None                    # (plateau, tau_1e, tau_end) of the slowest curve so far
    data_max = 0.0
    for tau, g2 in curves:
        tau = np.asarray(tau, dtype=float)
        g2 = np.asarray(g2, dtype=float)
        if tau.size == 0 or not np.isfinite(g2).any():
            continue
        data_max = max(data_max, float(np.nanmax(tau)))
        g0 = float(np.nanmax(g2))
        if not (math.isfinite(g0) and g0 > 0):
            continue
        # 1/e crossing sits in the STEEP part of the decay → robust to the noisy tail
        # (first downward crossing); the last 5 %-crossing marks where it is gone.
        below_1e = np.where(g2 <= g0 / math.e)[0]
        tau_1e = float(tau[below_1e[0]]) if below_1e.size else float(tau[-1])
        above_5 = np.where(g2 > 0.05 * g0)[0]
        tau_end = float(tau[above_5[-1]]) if above_5.size else tau_1e
        if best is None or tau_1e > best[1]:      # keep the slowest (largest τ_1e)
            best = (g0, tau_1e, tau_end)
    if best is None or not (math.isfinite(data_max) and data_max > 0):
        return None
    return _DecayFrame(best[0], best[1], best[2], data_max)


def _delay_window_row() -> Tuple[QtWidgets.QWidget, QtWidgets.QLineEdit,
                                 QtWidgets.QLineEdit, QtWidgets.QComboBox]:
    """Build the [min][max][unit] delay-window control (unit defaults to µs)."""
    row = QtWidgets.QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    mn = QtWidgets.QLineEdit(); mn.setPlaceholderText('min')
    mx = QtWidgets.QLineEdit(); mx.setPlaceholderText('max')
    unit = QtWidgets.QComboBox(); unit.addItems(U.unit_options('time'))
    for w in (mn, mx, unit):
        row.addWidget(w)
    holder = QtWidgets.QWidget(); holder.setLayout(row)
    return holder, mn, mx, unit


# ===========================================================================
# Correlogram sub-tab (parametric fits + 4-scale views + residuals + tables)
# ===========================================================================

# One results-row set per parametric method. A single results table is rebuilt
# with the selected method's rows, replacing the old
# always-visible Cumulant + KWW tables and the single/double-into-status line.
#
# Every Rh row names its kind in the label: one angle at one concentration is APPARENT
# (invariant 7), said where the number is read (ui_style_guide R5.3), and it is the whole
# qualifier -- WHICH average a cumulant fit reports (the scattered-intensity-weighted
# harmonic mean) is the Theory-and-Equations Guide's to define (CLAUDE.md invariant 7).

# Passive (nuance) tooltips for the cumulant order control and the μ₃ / μ₄ result rows.
# They carry the WHOLE caution: the higher moments ship bare, with no per-value validity
# glyph, because their unreliability is a constant property of the method rather than a
# per-run precondition (owner-ruled 2026-09-05; the ruling's home, its literature basis and
# the measured magnitudes are on `CumulantResult`). No citation in the text (ui_style_guide
# R6.1) -- the guide section is the pointer. Module constants so the GUI tier can assert them.
_CUMULANT_ORDER_TOOLTIP = (
    'Cumulant method only.\n'
    '1: the size (D) only.\n'
    '2: adds the PDI; below PDI ≈ 0.3 order 2 is at least as precise as order 4, '
    'except on clean data from PDI ≈ 0.2.\n'
    '3: adds μ₃, a diagnostic; on a broad sample it can report a badly wrong PDI as '
    'reliable.\n'
    '4: adds μ₄; better than order 2 in Γ at PDI ≈ 0.6, and in both Γ and the PDI on '
    'clean data at PDI ≈ 0.2–0.3.\n'
    'Precision as measured on this program\'s synthetic benchmarks; see the '
    'Theory-and-Equations-Guide §3.2.6.')


# The HEADLINE quantities of each method -- what the glance table shows, one row per
# ticked run. Every other quantity stays in the focused-run panel (`_RESULT_ROWS`).
#
# A glance column is named by the `_RESULT_ROWS` row it SHOWS, never by an index, for
# the reason `_RESULT_ROW_TOOLTIPS` is keyed by label. The header is the short form;
# "apparent" is said once on the table's title line.

# That resolution proves the LABEL exists and nothing more -- the index then subscripts
# `_result_values`, an independent hand-written list, and `_value_sources` hard-codes the
# same integers a third time. What forbids a drift between them is a pair of tests, not
# this resolution: `test_every_result_row_renders_the_quantity_its_label_names` and
# `test_a_precondition_note_lands_on_the_size_it_judges`. An earlier version of this
# comment claimed their safety before they existed.
_GLANCE_ROWS = {
    'cumulant': [('R<sub>h</sub> (nm)', 'R<sub>h</sub> (nm, apparent)'),
                 ('PDI', 'PDI')],
    'single':   [('R<sub>h</sub> (nm)', 'R<sub>h</sub> (nm, apparent)')],
    'double':   [('fast R<sub>h</sub> (nm)', 'fast R<sub>h</sub> (nm, apparent)'),
                 ('slow R<sub>h</sub> (nm)', 'slow R<sub>h</sub> (nm, apparent)')],
    'kww':      [('R<sub>h</sub>(τ<sub>c</sub>) (nm)',
                  'R<sub>h</sub>(τ<sub>c</sub>) (nm, apparent)'),
                 ('R<sub>h</sub>(⟨τ⟩) (nm)',
                  'R<sub>h</sub>(⟨τ⟩) (nm, apparent)')],
}


# Resolved to (header, index-into-`_result_values`) at import, so a headline naming a
# row that does not exist is an ImportError here and not a wrong number on screen.
# `.index` rather than `assert`: an assert disappears under -O, this cannot.
# How many glance rows are visible before the table scrolls. Six, because that is the
# replicate-set size the readability criterion is written against -- six ticked runs
# must be readable without scrolling, and a seventh must not cost the panel below.
_GLANCE_VISIBLE_ROWS = 6


# ... and the fewest it keeps when the section is squeezed. Three is enough to read a
# value, see that a second run differs, and know to scroll; below that the table stops
# being a summary at all.
_GLANCE_FLOOR_ROWS = 3


def _hide_shared_x(upper_ax):
    """Drop the x tick labels and x label from the UPPER axes of a fit+residual
    stack: the two share an x range and the residual carries the only axis, which is
    the standard convention. Both were printed twice until this (measured: removing
    the duplicate gives the residual ~6 px and the fit ~20 px).

    ONLY where the two panels genuinely share x. Applied to the DLS Distribution tab,
    whose panels are Rh (or Γ) over delay time τ, it strips the distribution's own
    axis and leaves it above an axis for a different quantity -- which is what it did
    until round `8fb4` caught it. The caller asserts the shared range; this helper
    cannot check it, since an axes carries no record of what its neighbour plots.
    """
    upper_ax.set_xlabel('')
    upper_ax.tick_params(labelbottom=False)


# Columns in the "Fits to show" grid -- see the measurement at its construction.
_LAYER_COLS = 4


_GLANCE_COLS = {
    key: [(header, _RESULT_ROWS[key].index(row)) for header, row in pairs]
    for key, pairs in _GLANCE_ROWS.items()
}


class _CorrelogramTab(_DeferredRepaint, _ResidualRatioMixin, QtWidgets.QWidget):
    """Parametric-fit view: four correlogram scales + residuals + result tables,
    over ONE or MANY co-plotted measurements (ticked in the checklist). All share
    one τ-window / baseline; each measurement gets a stable colour and a glance row.

    ANY stored fit can be drawn over the data, not only the one the Method box is
    showing: the "Fits to show" list offers all seven -- the four parametric fits this
    tab runs and the three distribution fits the Distribution tab runs -- so a cumulant
    and a CONTIN reconstruction can be compared on one correlogram. Colour and marker
    are the MEASUREMENT, line style is the METHOD, and the residual panel beneath shows
    the residuals of exactly the fits drawn above it, in the same coding."""

    #: The user moved the shared region -- a drag released, a field edited, or Reset.
    #: The owning `DLSModule` routes it to every "use the Correlogram tab's delay window
    #: and baseline" checkbox, which relabels itself and flags a checked result stale.
    regionChanged = QtCore.Signal()

    #: A hidden Correlogram keeps its focus at a sidebar click and repaints when shown.
    defers_repaint = True

    def __init__(self, controller, region, selection, parent=None, *,
                 distribution_fits=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.region = region                    # shared _AnalysisRegion (seconds)
        self.selection = selection              # shared _OverlaySelection
        self.item_id: Optional[str] = None
        self._runnable = False
        self._raw: Dict[str, Tuple] = {}        # iid -> (tau, g2m1) for checked DLS
        self._union_tau: Optional[Tuple[float, float]] = None
        # Each (measurement, method) fit this tab ran, with the signature it was run under,
        # and each refused Run's failure line, live in the controller under the fit's own
        # key (`_corr_run_key`): one slot per PAIR, so a correlogram carries several fits
        # at once, and a failure is read against the method it failed on, never the one
        # the box shows. The tab keeps no copy (`_shown_run`, `_failure`).

        # Zero-arg accessor for the fits the DISTRIBUTION tab draws, injected by the
        # owning DLSModule -- `{(iid, method): {'result','stale'}}`. A default that
        # answers nothing keeps this tab constructible on its own (its unit tests do),
        # and the indirection is why no widget here reaches into its sibling.
        self._distribution_fits = distribution_fits or (lambda: {})
        # The ticked runs the Method box's fit can run on, and each other ticked run with
        # what it lacks (`_refresh_fit_availability`): Run fits the first, the status
        # line names the second.
        self._fittable: List[str] = []
        self._left_out: List[Tuple[str, tuple]] = []
        self._scales = list(_SCALE_MODES)       # [main, side0, side1, side2]
        self._markers: Dict[str, object] = {}   # handle name -> Line2D on main_ax
        self._handle_glyphs: Dict[str, object] = {}  # handle name -> caret Line2D
        self._placeholder = None                # empty-state Text artist (nothing ticked)
        self._base_span = None                  # baseline shaded region inside the window
        self._base_span_outside: list = []      # ... and its greyed parts outside it
        self._drag: Optional[str] = None        # handle being dragged, or None
        self._suppress_fields = False
        self._build_ui()

    def _build_ui(self) -> None:
        splitter, left, right = make_split_panels(self, left_min_width=300,
                                                  sizes=(360, 740))

        self.checklist = MeasurementPicker(
            self.controller, self.selection, kinds=('dls',),
            label_fn=measurement_display_name, header_fn=sample_display_name,
            tooltip_fn=measurement_hover_card,
            help_text=_PICKER_HELP, help_bullets=_GROUP_PICKER_BULLETS,
            group_fields=(_GROUP_FIELD_CONC, _GROUP_FIELD_ANGLE))
        self.checklist.selectionChanged.connect(self._on_selection_changed)

        box = QtWidgets.QGroupBox('Parametric Fit')
        add_help_to_groupbox(box, 'Fit the correlogram to get a size (Rh).', bullets=[
            'Pick a <b>method</b>, set the <b>delay window</b> (the τ range fitted), '
            'then <b>Run</b>.',
            'Drag the markers or type values: green carets (top) move the window, '
            'purple (bottom) the <b>baseline region</b>.',
            'Neither applies until you move it; <b>Reset window + baseline</b> stops '
            'applying both.',
            '<b>Cumulant order</b>: 2 suits most samples; μ₃ and μ₄ are diagnostics, not '
            'measurements of the sample.',
            'A measurement with incomplete Data-tab parameters is left out; the status '
            'line says what to enter.',
            '<b>Click</b> a ⚠ line or lettered value for what to do.',
            'The Distribution sub-tab always reads this region, the others only with '
            'their region box ticked (User-Manual §8.1.2).',
        ])
        form = QtWidgets.QFormLayout(box)
        self.method_combo = QtWidgets.QComboBox()
        for label, key in _CORR_METHODS:
            self.method_combo.addItem(label, key)
        self.method_combo.setToolTip(
            'Cumulant: one average size and its spread (PDI), for a narrow, single '
            'population.\n'
            'Single/Double exponential: one or two distinct decay modes.\n'
            'Stretched Exponential (KWW): one broadened mode.\n'
            'For full size distributions use the Distribution sub-tab (NNLS / CONTIN).\n'
            'A Linear cumulant fit that bends back upward has left its range of validity; '
            'the sample needs a distribution method.\n'
            'What each method flags, how it reads the baseline region and how it weights '
            'its residuals: Theory-and-Equations-Guide §3.1–3.3.')
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        form.addRow('Method:', self.method_combo)
        self.order_spin = QtWidgets.QSpinBox()
        # 1-4, the engine's own range (`fit_cumulants`): order 4 is where the literature
        # stops, and the regime/noise caveats on orders 3-4 are this tooltip's job, since
        # the higher moments ship with no per-value flag (`CumulantResult.mu4_s_inv4`).
        self.order_spin.setRange(1, 4)
        self.order_spin.setValue(self.controller.settings.cumulant_order)
        self.order_spin.setToolTip(_CUMULANT_ORDER_TOOLTIP)
        form.addRow('Cumulant order:', self.order_spin)
        # The order is a fit input: changing it beside a shown fit flags that fit stale.
        self.order_spin.valueChanged.connect(lambda _v: self._on_fit_input_moved())
        win, self.tau_min, self.tau_max, self.tau_unit = _delay_window_row()
        form.addRow('Delay window:', win)
        # Baseline region (every DLS fit reads it): mean g2-1 over this τ range.
        brow = QtWidgets.QHBoxLayout(); brow.setContentsMargins(0, 0, 0, 0)
        self.base_lo = QtWidgets.QLineEdit(); self.base_lo.setPlaceholderText('low')
        self.base_hi = QtWidgets.QLineEdit(); self.base_hi.setPlaceholderText('high')
        brow.addWidget(self.base_lo); brow.addWidget(self.base_hi)
        bwidget = QtWidgets.QWidget(); bwidget.setLayout(brow)
        form.addRow('Baseline region:', bwidget)
        # Reset the window + baseline back to the defaults (full lag range; baseline
        # = last 25 %) — handy after dragging/zooming, and the one control that clears
        # both touched flags, so the fits go back to their own per-file estimates.
        self.reset_region_button = QtWidgets.QPushButton('Reset window + baseline')
        self.reset_region_button.setToolTip(
            'Put the delay window back to the full lag range and the baseline region '
            'back to the last quarter of the lags, and stop applying either: until you '
            'move a handle again each run keeps its own full lag range and each fit runs '
            'as if no baseline region had been placed (Theory-and-Equations-Guide '
            '§3.1.3).')
        self.reset_region_button.clicked.connect(self._on_reset_region)
        form.addRow(self.reset_region_button)
        # Two-way sync: editing a field moves its marker.
        for edit in (self.tau_min, self.tau_max, self.base_lo, self.base_hi):
            edit.editingFinished.connect(self._on_field_edit)
        self.tau_unit.currentTextChanged.connect(lambda _t: self._fields_from_region())
        self.run_button = QtWidgets.QPushButton('Run')
        self.run_button.clicked.connect(self._on_run)
        # NOT added to the form: it is pinned below the scroll area instead -- see
        # `fit_pane`. In the form it sat at content y=182 against a viewport measured
        # at 120/155/172 px (1280x760 / 1600x1020 / 1920x1080), i.e. below the fold at
        # every shipped window size, so the tab's primary action had to be scrolled to.

        # ---- "Fits to show": this tab's first LAYERS list (the locked GUI decision's
        # checkbox-layers pattern; the Distribution tab's `method_checks` is the nearest
        # existing one). One entry per method that can reconstruct a correlogram,
        # enabled once that fit exists for a ticked run.
        # It rides with the PLOT, not the left control column, and that is MEASURED: at
        # 1280x760 that column's three panes all sit at their floors (picker 200 px, its
        # own minimum; fit 120; results 300; of 636), so a fourth would overlap its
        # children (R3.2), and in the scrolled fit pane the box sat below the fold.
        layers_box = QtWidgets.QGroupBox('Fits to show')
        add_help_to_groupbox(
            layers_box,
            'Draw any stored fit over the data, not just the Method box’s.',
            bullets=[
                'An entry turns on once its fit <b>exists</b> for a ticked measurement: '
                'the parametric fits from this tab’s <b>Run</b>, <b>Lognormal / NNLS / '
                'CONTIN</b> from the <b>Distribution</b> sub-tab’s.',
                '<b>Color and point shape</b> are the measurement; <b>line style</b> is '
                'the method, as on the Distribution sub-tab.',
                'The <b>residual panel</b> shows the residuals of the fits ticked here.',
                'A fit whose inputs changed reads <b>(stale)</b> and is not drawn; press '
                '<b>Run</b> on its own sub-tab.',
                'The <b>results table</b> follows the <b>Method</b> box, not this list.',
                'The plot’s key names the methods; the figure <b>Save</b> writes also '
                'names the measurements.',
            ])
        # Two columns: seven checkboxes in one column take ~140 px of a control column
        # whose results section is already floored at 300 px.
        # Two rows of FOUR: measured on the real platform, the widest row is 434 px
        # against a 550 px canvas with every entry carrying its "(stale)" suffix, where
        # one row of seven needs 742 px and would be cut.
        lgrid = QtWidgets.QGridLayout(layers_box)
        lgrid.setContentsMargins(8, 2, 8, 2)
        lgrid.setVerticalSpacing(1)
        self.layer_checks: Dict[str, QtWidgets.QCheckBox] = {}
        for i, (label, key) in enumerate(_FIT_METHODS):
            cb = QtWidgets.QCheckBox(_fit_label(key))
            cb.setToolTip(
                f'Draw the stored {label} fit of every ticked measurement over the '
                'data, and its residuals in the panel beneath. Enabled once that fit '
                'exists; the line style shown is this method’s, on both DLS '
                'plot sub-tabs.')
            cb.setEnabled(False)               # until a fit of this method exists
            cb.stateChanged.connect(self._on_layer_toggled)
            self.layer_checks[key] = cb
            lgrid.addWidget(cb, i // _LAYER_COLS, i % _LAYER_COLS)
        lgrid.setColumnStretch(_LAYER_COLS, 1)     # keep the row left-packed
        # Guards the programmatic tick/enable pass below against re-entering `_redraw`
        # through `stateChanged` -- a sync that redraws that syncs.
        self._syncing_layers = False

        # One results area for the selected method: a single
        # table rebuilt with that method's rows, with Export directly beneath it.
        results_section = QtWidgets.QWidget()
        rlay = QtWidgets.QVBoxLayout(results_section)
        rlay.setContentsMargins(0, 0, 0, 0)
        self.results_label = QtWidgets.QLabel('Results')
        # Wrapped: a title naming the fits' weighting can be wider than the column.
        self.results_label.setWordWrap(True)
        rlay.addWidget(self.results_label)
        # GLANCE table -- one ROW per ticked run: its plot swatch, its name, and the
        # method's headline values (`_GLANCE_COLS`). One row per run rather than one
        # COLUMN per run, which is what this table used to be and why it failed by
        # construction: the control column is 360 px wide, so three ticked runs left
        # each value 70-80 px and every cell painted as '0…'. Measured once the one
        # name builder landed: a one-column-per-measurement header overran its column by
        # 228 px, i.e. a name cannot be held at any width in that shape.
        self.glance_table = QtWidgets.QTableWidget(0, 1)
        self.glance_table.setHorizontalHeader(RichTextHeaderView(self.glance_table))
        self.glance_table.verticalHeader().setVisible(False)
        self.glance_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        # Left-elided like the derived-run table: a file stem's TAIL is what tells six
        # replicates of one set apart, so the head is the part that may be dropped. Word
        # wrap off, or Qt wraps the name and right-elides its first line instead.
        self.glance_table.setTextElideMode(QtCore.Qt.TextElideMode.ElideLeft)
        self.glance_table.setWordWrap(False)
        # Stretch 1 against the detail panel's 0: when the section is short -- which it
        # is at the shipped window size, where all three panes sit at their minimums --
        # the SUMMARY must survive and the details must scroll, not the reverse. Without
        # this the panel held its full height and the glance table collapsed to its
        # header row, so the tab showed column titles and no values.
        rlay.addWidget(self.glance_table, 1)
        # FOCUSED-RUN panel -- every quantity of the fit that is not a headline, for the
        # one run the sidebar has focused. This is the widget this section held before
        # `_fill_tables` began rebuilding it as one column per measurement, so pointing
        # it back at a single run is partly a revert. A planned change moves these
        # quantities onto the Summary tab and takes this panel with them; until then
        # nothing loses its home.
        self.focus_label = QtWidgets.QLabel('')
        self.focus_label.setWordWrap(True)
        rlay.addWidget(self.focus_label)
        self.result_table = _vtable(_RESULT_ROWS['cumulant'])
        rlay.addWidget(self.result_table)
        # The notes of the fits both tables show: one line per failed precondition, the
        # runs named or counted, and a marker on each value a line is about. Pinned under
        # the column right below these results, in the flag's place (R8.6).
        self.notes_block = NotesBlock(what='runs')
        self.notes_block.watch(self.glance_table)
        self.notes_block.watch(self.result_table)
        self._block_fits: set = set()       # id() of each fit the block was fed
        # Export is pinned under the column with the status line (`pin_under_column`
        # below), not in this pane: when the column is taller than its viewport, the
        # last pane is the one that scrolls out of view.
        self.export_button = QtWidgets.QPushButton('Export CSV…')
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._on_export)
        rlay.addStretch(1)
        # Vertically resizable control column: drag the grips to size the
        # checklist / fit controls / results panes against each other.

        # The fit controls ride in a scroll area so their MINIMUM stops dictating the
        # column. `make_vertical_plot_stack` floors each pane at its own
        # `minimumSizeHint`; this box's form rows put that near 300 px, which the
        # splitter honours by starving the pane below -- leaving the results section
        # about 60 px, its title and a header row and no values.
        fit_scroll = QtWidgets.QScrollArea()
        fit_scroll.setWidgetResizable(True)
        fit_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        fit_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        fit_scroll.setWidget(box)
        # Run is pinned OUTSIDE the scrolled form (the §3 column template): the fit
        # controls may scroll, the action that runs them may not.
        fit_pane = QtWidgets.QWidget()
        fit_layout = QtWidgets.QVBoxLayout(fit_pane)
        fit_layout.setContentsMargins(0, 0, 0, 0)
        fit_layout.setSpacing(4)
        fit_layout.addWidget(fit_scroll, 1)
        fit_layout.addWidget(self.run_button)
        # The floor is passed to the stack, NOT set on the widget: Qt's `qSmartMinSize`
        # lets an explicit `setMinimumHeight` REPLACE the layout-derived minimum rather
        # than floor it, so a literal clearing the content at the shipped font stops
        # clearing it at a larger one (measured: 272 px at 9 pt, 320 px at 18 pt, which
        # 150 % text scaling reaches). `make_vertical_plot_stack` takes the max of the
        # two. A splitter handed less than a pane needs OVERLAPS its children (R3.2).
        vstack = make_vertical_plot_stack(
            [self.checklist, fit_pane, results_section], sizes=[200, 250, 300],
            min_heights=[80, 120, 300], stretch=COLUMN_STACK_STRETCH)
        left.addWidget(vstack, 1)

        # Export, the status line and the reliability flag sit UNDER the scrolled
        # control column, not inside it (style guide R8.6 and the §3 column template):
        # a flag is an always-visible label (§5), and at the shipped window size the
        # column is taller than its viewport, so whatever sits last scrolls out of view.
        # Pinning also takes the word-wrapping labels out of the scrolled layout, so the
        # scroll area sizes the column to its minimums rather than to the widgets'
        # natural heights.
        self.status = QtWidgets.QLabel('')
        self.status.setWordWrap(True)
        # Operational flags only: a skipped run, a shown fit that failed with no note to
        # say why (ui_style_guide §5). A fit's failed preconditions are the notes block's.
        self.flag_label = ThemedLabel('', role='error', bold=True)
        self.flag_label.setWordWrap(True)
        pin_under_column(splitter, self.export_button, self.status,
                         self.notes_block, self.flag_label)

        self.figure = Figure(figsize=(5.5, 4.6), constrained_layout=True)
        _widen_row_gap(self.figure)              # room for the residual grip line
        self.canvas = make_canvas_expanding(LazyCanvas(self.figure))
        # Flat 4×2 gridspec (constrained_layout handles a flat grid robustly; a nested
        # subgridspec trips its tick-bbox pass on degenerate log axes). The fit spans
        # the top 3 left rows, the residual the bottom-left row, the 3 side views the
        # top 3 right rows. The fit:residual split is adjustable (draggable residual):
        # the first three row heights stay equal (= a) so the
        # main + side views scale together, the 4th carries the residual's share. The
        # residual lives on this one canvas so it stays aligned under the fit.
        self._resid_ratio = 0.22                 # residual share of the left column
        self._resid_dragged = False              # set by the drag handle; see the floor
        a = (1.0 - self._resid_ratio) / 3.0
        self._gs = self.figure.add_gridspec(
            4, 2, width_ratios=[3, 1],
            height_ratios=[a, a, a, self._resid_ratio])
        self.main_ax = self.figure.add_subplot(self._gs[0:3, 0])
        self.resid_ax = self.figure.add_subplot(self._gs[3, 0])
        self.side_axes = [self.figure.add_subplot(self._gs[i, 1]) for i in range(3)]
        self.nav_toolbar = themed_navtoolbar(self.canvas, self, self.controller)
        self.canvas.mpl_connect('button_press_event', self._on_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('button_release_event', self._on_release)
        # Drag the gap between fit and residual to resize the residual (stays aligned).
        self._resid_resizer = attach_residual_resizer(
            self.canvas, self.figure, self.main_ax, self.resid_ax,
            self._on_resid_dragged)
        right.addWidget(self.nav_toolbar)
        right.addWidget(layers_box)
        right.addWidget(self.canvas, 1)
        # The "promote a side view" instruction lives here, not in the axis title: the
        # main axis is narrow (3/4 width minus the side stack), so a long title overflowed
        # into the y-label gutter and clipped its leading "C".
        promote_hint = ThemedLabel(
            'Double-click a side view to promote it to the main plot.',
            role='hint', size=10)
        promote_hint.setWordWrap(True)
        right.addWidget(promote_hint)

    def reseed_from_settings(self) -> None:
        self.order_spin.setValue(self.controller.settings.cumulant_order)

    def reshow(self) -> None:
        """Re-read what the controller holds after a Settings apply, whose clear may have
        taken fits or failure lines this tab shows: the tab then reads as after a sample
        switch."""
        self.export_button.setEnabled(self._any_cached())
        self._sync_layers()
        self._redraw()
        self._refresh_tables()
        self._update_status()

    def showEvent(self, event) -> None:                      # keep selection in sync
        super().showEvent(event)
        if self._repaint_pending:
            # A click left the whole repaint owing, Export included: paying it is this
            # refresh and more, so one repaint per show, not two.
            self.flush_view()
            return
        self.checklist.refresh()
        self._reload_raw()
        self._refresh_fit_availability()
        self._sync_layers()
        self._redraw()
        self._refresh_tables()
        self._update_status()

    # ---- selection ----
    def focus_state(self, item_id: Optional[str], runnable: bool) -> bool:
        """The state half (`_DeferredRepaint`): the focus, then the ticked runs' reload,
        which seeds the shared region the sub-tabs after this one read in the same click,
        so it runs at every focus, shown or not."""
        self.item_id = item_id
        self._runnable = runnable
        # Ahead of the checklist's refresh: the reload skips every id that refresh's prune
        # drops, so it reads the same runs.
        self._reload_raw()
        return True

    def repaint_view(self) -> None:
        """The repaint half: the picker, Run, Export, the plot, the tables and the
        status, from the stored focus and the controller as they are now."""
        # Sidebar selection only focuses/navigates; it no longer auto-ticks the
        # measurement for analysis. The "Measurements to plot" checklist is the
        # sole driver of what gets fit/overlaid.
        self.checklist.refresh()
        self._refresh_fit_availability()
        self.export_button.setEnabled(self._any_cached())
        self._sync_layers()
        self._redraw()
        self._refresh_tables()
        self._update_status()

    @QtCore.Slot()
    def _on_selection_changed(self) -> None:
        # The shared selection moves from any co-plot tab: a repaint owed runs first.
        self.flush_view()
        self._reload_raw()
        self._refresh_fit_availability()
        self.export_button.setEnabled(self._any_cached())
        self._sync_layers()
        self._redraw()
        self._refresh_tables()
        self._update_status()

    def _reload_raw(self) -> None:
        """Load the raw correlogram of every ticked DLS measurement and seed/extend
        the shared window from their UNION lag range."""
        self._raw = {}
        taus = []
        ws = self.controller.workspace.measurements
        for iid in self.selection.ids():
            m = ws.get(iid)
            if m is None or m.kind != 'dls':
                continue
            tau = np.asarray(m.raw['delay_times_s'], dtype=float)
            g2 = np.asarray(m.raw['correlogram'], dtype=float)
            self._raw[iid] = (tau, g2)
            taus.append(tau)
        if taus:
            allt = np.concatenate(taus)
            self._union_tau = (float(allt.min()), float(allt.max()))
            self.region.init_from_data(allt)      # untouched parts follow the union range
        else:
            self._union_tau = None
        self._fields_from_region()

    def _refresh_fit_availability(self) -> None:
        """Judge the Method box's fit on each ticked run, one run at a time as Run fits
        them (`Controller.dls_fit_availability`): Run is on while at least one can fit,
        and every other ticked run is kept, with what it lacks, for the status line. The
        checklist stays tickable for every run, since a ticked run's raw correlogram is
        drawn whether or not its parameters are complete."""
        key = ('correlogram', self.method_combo.currentData())
        verdicts = {iid: self.controller.dls_fit_availability([iid])[key]
                    for iid in self._raw}
        self._fittable = [iid for iid, a in verdicts.items() if a.ok]
        self._left_out = [(iid, a.missing) for iid, a in verdicts.items() if not a.ok]
        self.run_button.setEnabled(bool(self._fittable))

    def _any_cached(self) -> bool:
        """Whether the METHOD BOX's fit exists for any ticked run. The Export button and
        the status line follow the Method box, not the layers list: an export writes one
        method's numbers."""
        key = self.method_combo.currentData()
        return any(self._shown_run(i, key) is not None for i in self.selection.ids())

    def _shown_run(self, iid: str, key: str):
        """The `key` fit of `iid` this tab ran and draws (`Controller.shown_run`), else
        None."""
        return self.controller.shown_run(_corr_run_key(iid, key))

    def _failure(self, iid: str, key: str) -> Optional[str]:
        """Why this tab's last Run of the `key` fit of `iid` was refused, else None."""
        return self.controller.failure_line(_corr_run_key(iid, key))

    def _cache_is_stale(self, iid: str, key: str) -> bool:
        """Whether THIS tab's stored fit of `(iid, key)` was computed from inputs that
        have since moved. False when there is no such fit -- absent is not stale."""
        run = self._shown_run(iid, key)
        return run is not None and run.sig != self._run_signature(iid, key)

    # ---- the drawable fits: this tab's, plus the ones the Distribution tab owns ----
    def _stored_fits(self) -> Dict[Tuple[str, str], Dict]:
        """Every stored fit available for the TICKED set, keyed `(iid, method)`, each
        carrying the staleness verdict of the tab that RAN it -- per item 8's rule that
        staleness is judged by its owner.

        For the four parametric methods that is this tab's per-fit signature comparison.
        For the three distribution methods it is the Distribution tab's own verdict,
        which is per fit too: each of its fits has its own (measurement, method) slot
        with its own signature, as this tab's do, so one run's fit can be fresh beside
        another's stale one. Its ticks and method boxes do not filter what it hands over;
        this tab filters by its own ticked set, here."""
        ticked = set(self.selection.ids())
        out: Dict[Tuple[str, str], Dict] = {}
        for iid in dict.fromkeys(self.selection.ids()):
            for key in _CORR_RESULT_KINDS:
                run = self._shown_run(iid, key)
                if run is not None:
                    out[(iid, key)] = {'result': run.result,
                                       'stale': run.sig != self._run_signature(iid, key)}
        for (iid, key), v in self._distribution_fits().items():
            if iid in ticked:
                out[(iid, key)] = v
        return out

    def _shown_fits(self, stored: Optional[Dict] = None) -> List[Tuple[str, str, object]]:
        """`[(iid, method, result)]` for every fit the layers list has ticked, in
        measurement order then method order, skipping the stale ones (item 8: a fit its
        own tab marks stale is not drawn). The one ordering both the plot and the
        residual panel beneath it iterate, so the two cannot disagree about which curve
        is which."""
        stored = self._stored_fits() if stored is None else stored
        out = []
        for iid in self._raw:                    # ticked, in selection order
            for _label, key in _FIT_METHODS:
                cb = self.layer_checks.get(key)
                if cb is None or not cb.isChecked():
                    continue
                entry = stored.get((iid, key))
                if entry is None or entry['stale']:
                    continue
                out.append((iid, key, entry['result']))
        return out

    def _sync_layers(self, stored: Optional[Dict] = None, tick: Iterable[str] = ()) -> None:
        """Bring the layers list level with what is stored: enable a method once a fit
        of it exists for a ticked run, say so when the fits it would draw are stale, and
        tick the methods in `tick` (the ones a Run just produced).

        Signals are blocked over the pass: each programmatic `setChecked` would
        otherwise re-enter `_redraw` through `stateChanged`, once per box."""
        stored = self._stored_fits() if stored is None else stored
        self._syncing_layers = True
        try:
            for _label, key in _FIT_METHODS:
                cb = self.layer_checks[key]
                have = [v for (_i, k), v in stored.items() if k == key]
                cb.setEnabled(bool(have))
                # "(stale)" when EVERY stored fit of this method is stale: with one
                # fresh fit among them the tick still draws something, and the label
                # would be claiming otherwise.
                stale = bool(have) and all(v['stale'] for v in have)
                cb.setText(_fit_label(key) + (' (stale)' if stale else ''))
                if key in tick and have:
                    cb.setChecked(True)
        finally:
            self._syncing_layers = False

    @QtCore.Slot()
    def _on_layer_toggled(self) -> None:
        if self._syncing_layers:
            return
        self._redraw()

    @QtCore.Slot()
    def on_fits_changed(self, methods) -> None:
        """The Distribution tab finished a Run, or its ticks changed (it keeps every
        fit, so `stored_fits` is re-read either way): re-offer its methods here and tick
        the ones a Run just produced.

        Routed by the owning `DLSModule`, never by that widget reaching into this one
        (invariant 5's spirit at the widget layer): it owns both tabs and connects the
        signal to this slot.

        `_reload_raw` FIRST, as `set_measurement` and `_on_selection_changed` both do:
        removing a ticked run prunes the shared selection, and the Distribution tab --
        built first, so connected to `SelectionModel.changed` first -- emits from inside
        that prune, while `self._raw` still holds the deleted id. Redrawing then raised
        `KeyError` on `ws[iid]`; Qt swallowed it at the emit boundary, so the draw
        aborted half-done and only a console traceback said so. `_reload_raw` skips ids
        the workspace no longer has.

        A repaint this tab owes runs before any of that (`flush_view`).
        """
        self.flush_view()
        self._reload_raw()
        self._sync_layers(tick=set(methods or ()))
        self._redraw()
        # ... and the status, which is where a DRAWN-but-failed fit says so. Without
        # this the note only appeared after some other gesture refreshed the status,
        # i.e. never, for the one path that brings a failed lognormal to this tab.
        self._update_status()

    def _fit_kwargs(self, iid: str, key: Optional[str] = None) -> Dict:
        """What this method's fit of `iid` reads from the shared region -- the one
        reading (`_region_fit_kwargs`): the window and the baseline only once the user has
        set each, measured inside the window, as the one baseline input this method
        reads. `key` defaults to the method the combo is showing, so the staleness checks
        and the Run ask the same question.
        """
        return _region_fit_kwargs(self.controller, self.region, iid,
                                  key or self.method_combo.currentData())

    def _run_signature(self, iid: str, key: str) -> tuple:
        """Everything a parametric fit of `iid` reads beyond its raw correlogram: the
        measurement's committed params, the method, the cumulant order (cumulant only),
        the shared delay window and this measurement's baseline over the shared
        region, and the settings the engine seeds (skip_initial_channels; the cumulant
        method, fit cutoff and weighting, cumulant only). Stored with each cached fit and compared on every
        redisplay and region edit, so a shown fit is flagged stale exactly when a
        re-Run would read different inputs -- a dragged baseline handle included.

        The region's APPLIED parts ride here in their own slot as well as through the
        kwargs they gate, so that pressing Reset beside a shown fit flags it stale exactly
        as a drag does even in the corner where the kwargs happen not to move (a span that
        selects no channel on this correlogram emits no `baseline` either way)."""
        s = self.controller.settings
        cumulant = key == 'cumulant'
        return (self.controller.committed_signature([iid]), key,
                self.order_spin.value() if cumulant else None,
                tuple(sorted(self._fit_kwargs(iid, key).items())),
                self.region.state().applied(),
                s.skip_initial_channels, s.cumulant_method if cumulant else None,
                # Settings-seeded, so these ride here rather than beside the plot -- the
                # reason such a value needs no per-run widget to flag a shown fit stale.
                # `if cumulant` for the same reason `cumulant_method` is: the exponential
                # arms take the ENGINE default cutoff and weighting for their seeds, and
                # do not move when these settings do.
                s.cumulant_fit_cutoff if cumulant else None,
                s.cumulant_weighting if cumulant else None)

    def _fit_for(self, iid: str, key: Optional[str] = None):
        """This tab's stored result for `(iid, key)`, `key` defaulting to the method the
        combo is showing -- the tables and the status line, which follow the Method box.
        None when there is none. What the PLOT draws is `_shown_fits`, which follows the
        layers list and also reaches the Distribution tab's results."""
        run = self._shown_run(iid, key or self.method_combo.currentData())
        return run.result if run is not None else None

    # ---- actions ----
    def _on_fit_input_moved(self) -> None:
        """A fit INPUT moved beside a shown fit: re-label the layers list (which marks
        the now-stale entries) and refresh the status.

        One spelling of the response, because it was previously open-coded and four of
        the nine handlers that move an input had only half of it -- the curve was
        dropped from the plot while its checkbox still read as live."""
        self._sync_layers()
        self._redraw()
        self._update_status()

    @QtCore.Slot()
    def _on_method_changed(self) -> None:
        self.order_spin.setEnabled(self.method_combo.currentData() == 'cumulant')
        self._refresh_fit_availability()
        self.export_button.setEnabled(self._any_cached())
        self._redraw()
        self._refresh_tables()
        self._update_status()

    @QtCore.Slot()
    def _on_run(self) -> None:
        # Only the ticked runs that can fit: the others are named on the status line,
        # and a failure recorded below is one the fit itself raised.
        self._refresh_fit_availability()
        ids = list(self._fittable)
        if not ids:
            return
        # Cumulant/exp/KWW fits are fast enough to stay synchronous, but they
        # still write the shared controller.results dict — so they must not run
        # while a background fit is writing it too. Refuse rather
        # than race; the fit is a click away once the worker frees.
        if runner().is_busy:
            self.status.setText(BUSY_NOTICE)
            return
        key = self.method_combo.currentData()
        c = self.controller
        failures: Dict[tuple, str] = {}
        for iid in ids:
            try:
                kw = self._fit_kwargs(iid, key)   # window + region, per method
                if key == 'cumulant':
                    res = c.run_cumulants(iid, order=self.order_spin.value(), **kw)
                elif key == 'single':
                    res = c.run_single_exponential(iid, **kw)
                elif key == 'double':
                    res = c.run_double_exponential(iid, **kw)
                else:
                    res = c.run_kww(iid, **kw)
                c.record_shown_run(_corr_run_key(iid, key), res,
                                   self._run_signature(iid, key))
            except Exception as exc:  # noqa: BLE001 — per-measurement, failure recorded; per-measurement; never abort all
                failures[_corr_run_key(iid, key)] = str(exc)
                # Only THIS method's entry: a failed cumulant Run must not silently
                # drop the CONTIN reconstruction the user has ticked beside it.
                c.forget_shown_run(_corr_run_key(iid, key))
        # Every method's failure lines are replaced WHOLESALE by each Run: a line carries
        # no signature, so a kept one cannot say whether its cause has since been fixed.
        c.replace_run_failures(_CORR_RESULT_KINDS.values(), failures)
        self.export_button.setEnabled(self._any_cached())
        # A fresh Run ticks its own method, so the fit just computed is on screen
        # without a second gesture.
        self._sync_layers(tick={key})
        self._redraw()
        self._refresh_tables()
        self._update_status()

    @QtCore.Slot()
    def _on_export(self) -> None:
        key = self.method_combo.currentData()
        cached = [(iid, run.result) for iid, run in
                  ((iid, self._shown_run(iid, key)) for iid in self.selection.ids())
                  if run is not None]
        if not cached:
            self.status.setText('Run a fit first.')
            return

        name, suffixes = _selection_export_names(
            self.controller, [(iid, key) for iid, _res in cached], 'correlogram')

        def do_export(path: str):
            import os
            stem, ext = os.path.splitext(path)
            paths = [f'{stem}{sfx}{ext}' for sfx in suffixes]
            siblings = [] if len(cached) == 1 else paths
            if not confirm_overwrite(self, siblings):
                return None
            written = []
            for (iid, res), p in zip(cached, paths, strict=True):
                written.append(self.controller.export_correlogram_fit(iid, res, p))
            return written                     # summarised by `export_to_csv`

        status = export_to_csv(self, self.controller, name, do_export)
        if status:
            self.status.setText(status)

    # ---- region fields <-> shared region (seconds) ----
    _HANDLES = [('tau_min', 'tau_min_s'), ('tau_max', 'tau_max_s'),
                ('base_lo', 'base_lo_s'), ('base_hi', 'base_hi_s')]

    # Significant figures the four handle fields display. Named because the round-trip
    # BACK through it decides whether a field was edited (see `_displayed_handle`), so the
    # writer and the comparison must use one number.
    _HANDLE_SIG = 4

    def _fields_from_region(self) -> None:
        unit = self.tau_unit.currentText()
        self._suppress_fields = True
        for edit, (_, attr) in zip(
                (self.tau_min, self.tau_max, self.base_lo, self.base_hi),
                self._HANDLES, strict=True):
            x = getattr(self.region, attr)
            edit.setText('' if x is None
                         else f'{U.from_canonical("time", x, unit):.{self._HANDLE_SIG}g}')
        self._suppress_fields = False

    def _field_was_edited(self, edit, attr: str, unit: str, value) -> bool:
        """Did the user actually edit THIS field, or is it just being read back?

        `editingFinished` fires on one widget but the handler parses all four, and the
        fields render to `_HANDLE_SIG` significant figures, so parsing them all back does
        not round-trip: a stored 1.234567e-4 comes back as 1.235e-4. Writing that back
        silently truncated three handles the user never touched -- and, once the baseline
        region gained a touched state, marked it PLACED when the user had typed a
        delay-window bound.

        TWO independent tests, OR-ed, because neither alone covers the field:

        - Qt's own `isModified()`, which is True exactly when a human changed the text
          and False after any programmatic `setText`. This is the one signal that can
          tell "the user retyped the digits already shown" -- a real confirm-this-region
          gesture -- from "this box was never touched"; a value comparison cannot, since
          the two produce identical text.
        - the value comparison, for the programmatic path: a caller that sets the text
          and calls the handler (the headless tests, and any future code that drives the
          fields) leaves `isModified()` False, and would otherwise write nothing at all.
        """
        if edit is not None and edit.isModified():
            return True
        return value != self._displayed_handle(attr, unit)

    def _displayed_handle(self, attr: str, unit: str):
        """What `attr`'s field is currently SHOWING, read back in canonical seconds."""
        x = getattr(self.region, attr)
        if x is None:
            return None
        shown = f'{U.from_canonical("time", x, unit):.{self._HANDLE_SIG}g}'
        try:
            return U.to_canonical('time', float(shown), unit)
        except ValueError:
            return None

    def _on_field_edit(self) -> None:
        if self._suppress_fields:
            return
        unit = self.tau_unit.currentText()

        def parse(edit):
            t = edit.text().strip()
            if not t:
                return None
            try:
                return U.to_canonical('time', float(t), unit)
            except ValueError:
                return 'ERR'

        vals = {attr: parse(edit) for edit, (_, attr) in zip(
            (self.tau_min, self.tau_max, self.base_lo, self.base_hi), self._HANDLES, strict=True)}
        if 'ERR' in vals.values():
            self._fields_from_region()           # revert bad input
            return
        if (vals['tau_min_s'] is not None and vals['tau_max_s'] is not None
                and vals['tau_min_s'] >= vals['tau_max_s']):
            self._fields_from_region()
            return
        if (vals['base_lo_s'] is not None and vals['base_hi_s'] is not None
                and vals['base_lo_s'] >= vals['base_hi_s']):
            self._fields_from_region()
            return
        for edit, (_name, attr) in zip(
                (self.tau_min, self.tau_max, self.base_lo, self.base_hi),
                self._HANDLES, strict=True):
            v = vals[attr]
            if self._field_was_edited(edit, attr, unit, v):
                self.region.set_handle(attr, v)  # marks a MOVED handle's kind touched
            # Consumed either way: a modified flag left standing would make the NEXT
            # field's edit rewrite this handle at display precision, which is the
            # truncation the comparison above exists to prevent.
            edit.setModified(False)
        self._redraw()
        self._sync_layers()          # ... and the layers list says which are stale
        self._update_status()        # the window is a fit input: a shown fit may be stale
        self.regionChanged.emit()

    def _on_reset_region(self) -> None:
        """Restore the window + baseline to the from-data defaults."""
        self.region.force_reseed()
        self._reload_raw()           # re-seeds the region from the union lag range
        self._sync_layers()          # the reset staled every shown fit; say so
        self._redraw()
        self._update_status()
        self.regionChanged.emit()

    # ---- drag the markers on the main correlogram ----
    def _on_press(self, event) -> None:
        # The residual-resize handle gets first refusal on a press in its gap band,
        # so dragging the fit/residual divider never starts a marker drag.
        if self._resid_resizer.consumes_press(event):
            return
        # Double-click a side view → promote it to the main slot.
        if getattr(event, 'dblclick', False):
            for k, sax in enumerate(self.side_axes):
                if event.inaxes is sax:
                    self._scales[0], self._scales[k + 1] = (
                        self._scales[k + 1], self._scales[0])
                    self._redraw()
                    return
            return
        # Otherwise grab the nearest marker handle on the main axes.
        if (event.inaxes is not self.main_ax or event.xdata is None
                or getattr(self.nav_toolbar, 'mode', '')):
            return
        self._drag = self._nearest_handle(event)

    # Which handles live in which half of the plot: window markers carry
    # their carets at the top, baseline markers at the bottom, so a press is disambiguated
    # by y-band even when two markers share an x.
    _TOP_HANDLES = ('tau_min', 'tau_max')
    _BOTTOM_HANDLES = ('base_lo', 'base_hi')

    def _nearest_handle(self, event) -> Optional[str]:
        # The region stores τ in seconds, but the axis is drawn in display units
        # (µs by default), so handle positions are scaled by _disp_factor('time')
        # before being mapped to pixels.
        tfac = _disp_factor('time')
        if event.x is None or event.y is None:
            return None
        # Pick the candidate kind by which half of the axes the press is in
        # (top -> window, bottom -> baseline; see _TOP_HANDLES above).
        try:
            yf = self.main_ax.transAxes.inverted().transform((event.x, event.y))[1]
        except Exception:  # noqa: BLE001 — transform guard, no handle
            return None
        wanted = self._TOP_HANDLES if yf >= 0.5 else self._BOTTOM_HANDLES
        y_mid = sum(self.main_ax.get_ylim()) / 2.0
        best, best_px = None, 8.0                 # 8-pixel grab tolerance (within the band)
        for name, attr in self._HANDLES:
            if name not in wanted:
                continue
            x = getattr(self.region, attr)
            if x is None:
                continue
            try:
                px = self.main_ax.transData.transform((x * tfac, y_mid))[0]
            except Exception:  # noqa: BLE001 — transform guard, handle skipped
                continue
            if abs(px - event.x) < best_px:
                best, best_px = name, abs(px - event.x)
        return best

    def _on_motion(self, event) -> None:
        # (No hover-cursor cue here: the residual resizer manages the canvas cursor on
        # the same motion signal and would override it. The offset carets are the
        # affordance instead.)
        if (self._drag is None or event.inaxes is not self.main_ax
                or event.xdata is None):
            return
        # event.xdata is in display units (µs); the region + clamp work in seconds.
        tfac = _disp_factor('time')
        x = self._clamp_handle(self._drag, float(event.xdata) / tfac)
        attr = dict(self._HANDLES)[self._drag]
        self.region.set_handle(attr, x)         # marks a MOVED handle's kind touched
        line = self._markers.get(self._drag)
        if line is not None:
            line.set_xdata([x * tfac, x * tfac])   # draw in display units
        glyph = self._handle_glyphs.get(self._drag)
        if glyph is not None:
            glyph.set_xdata([x * tfac])            # caret tracks its line
        self.canvas.draw_idle()                    # move only the line + caret (cheap)

    def _clamp_handle(self, name: str, x: float) -> float:
        if self._union_tau is not None:           # clamp to the union of all ticked
            lo, hi = self._union_tau
            x = min(max(x, lo), hi)
        r = self.region
        if name == 'tau_min' and r.tau_max_s is not None:
            x = min(x, r.tau_max_s * (1 - 1e-9))
        elif name == 'tau_max' and r.tau_min_s is not None:
            x = max(x, r.tau_min_s * (1 + 1e-9))
        elif name == 'base_lo' and r.base_hi_s is not None:
            x = min(x, r.base_hi_s * (1 - 1e-9))
        elif name == 'base_hi' and r.base_lo_s is not None:
            x = max(x, r.base_lo_s * (1 + 1e-9))
        return x

    def _on_release(self, event) -> None:
        if self._drag is None:
            return
        self._drag = None
        self._fields_from_region()
        self._sync_layers()                      # ... and the layers list says so
        self._redraw()                           # recreate markers + baseline span
        self._update_status()                    # a dragged marker moved a fit input
        self.regionChanged.emit()

    # Marker styling + offset-handle geometry: window carets at the
    # top, baseline carets at the bottom (the disambiguation described above).
    # _BASE_HANDLE_Y is kept a touch above the bottom so the caret never lands in
    # the residual-resize gap band below main_ax.
    _WIN_COLOR, _BASE_COLOR = '#2ca02c', '#7e57c2'   # window green, baseline purple
    # The part of the baseline span outside the delay window, which no fit reads: a
    # neutral grey, hatched so it stays distinct from the purple fill in both themes.
    _OUTSIDE_COLOR = PLACEHOLDER_GRAY
    _WIN_HANDLE_Y, _BASE_HANDLE_Y = 0.96, 0.05

    def _draw_markers(self) -> None:
        """Draw the window (dotted, top carets) + baseline (dashed + shaded, bottom
        carets) markers on main_ax, each a full-height line plus an offset grab handle."""
        self._markers = {}
        self._handle_glyphs = {}
        self._base_span = None
        self._base_span_outside = []
        r = self.region
        # The region stores τ in canonical seconds, but the correlogram x-axis is
        # drawn in display units (µs by default, via plot_correlogram_scaled), so
        # every marker position is scaled by _disp_factor('time'). Without this the
        # markers land at the raw-seconds value on a µs axis (e.g. τ_max ≈ 1 s draws
        # at the "1 µs" tick, the "capped at 1 µs" bug).
        tfac = _disp_factor('time')
        # Carets use a blended transform: x in data units, y in axes fraction.
        htrans = self.main_ax.get_xaxis_transform()

        def _caret(x, marker, color, y):
            return self.main_ax.plot(
                [x], [y], marker=marker, color=color, markersize=10,
                markeredgewidth=0, transform=htrans, clip_on=False, zorder=6)[0]

        # Only the first line of each kind carries a legend label, so the legend
        # shows one "τ window" and one "baseline region" entry.
        if r.tau_min_s is not None:
            self._markers['tau_min'] = self.main_ax.axvline(
                r.tau_min_s * tfac, color=self._WIN_COLOR, ls=':', lw=1.4,
                label='τ window (fit range)')
            self._handle_glyphs['tau_min'] = _caret(
                r.tau_min_s * tfac, 'v', self._WIN_COLOR, self._WIN_HANDLE_Y)
        if r.tau_max_s is not None:
            self._markers['tau_max'] = self.main_ax.axvline(
                r.tau_max_s * tfac, color=self._WIN_COLOR, ls=':', lw=1.4)
            self._handle_glyphs['tau_max'] = _caret(
                r.tau_max_s * tfac, 'v', self._WIN_COLOR, self._WIN_HANDLE_Y)
        if r.base_lo_s is not None and r.base_hi_s is not None:
            lo, hi = sorted((r.base_lo_s * tfac, r.base_hi_s * tfac))
            # NOTHING OUTSIDE THE DELAY WINDOW COUNTS: a fit measures the span only over
            # its part inside the window (`app/analysis_region.py`), so the part outside
            # is drawn grey and the purple fill stops at the window edge. The handles
            # still move freely; this only says what the fits will read. The window drawn
            # is the handles' (an untouched window follows the ticked data's range).
            wlo = r.tau_min_s * tfac if r.tau_min_s is not None else lo
            whi = r.tau_max_s * tfac if r.tau_max_s is not None else hi
            in_lo, in_hi = max(lo, wlo), min(hi, whi)
            self._base_span = None
            if in_lo < in_hi:
                self._base_span = self.main_ax.axvspan(
                    in_lo, in_hi, color=self._BASE_COLOR, alpha=0.20,
                    label='baseline region')
            self._base_span_outside = [
                self.main_ax.axvspan(a, b, color=self._OUTSIDE_COLOR, alpha=0.25,
                                     hatch='//', linewidth=0, zorder=0.5)
                for a, b in ((lo, min(hi, wlo)), (max(lo, whi), hi)) if a < b]
            self._markers['base_lo'] = self.main_ax.axvline(
                r.base_lo_s * tfac, color=self._BASE_COLOR, ls='--', lw=1.2)
            self._handle_glyphs['base_lo'] = _caret(
                r.base_lo_s * tfac, '^', self._BASE_COLOR, self._BASE_HANDLE_Y)
            self._markers['base_hi'] = self.main_ax.axvline(
                r.base_hi_s * tfac, color=self._BASE_COLOR, ls='--', lw=1.2)
            self._handle_glyphs['base_hi'] = _caret(
                r.base_hi_s * tfac, '^', self._BASE_COLOR, self._BASE_HANDLE_Y)

    # A linear-τ view is framed to this multiple of the slowest curve's 1/e decay time,
    # so the decay sits roughly centered rather than crushed at the origin. Set
    # empirically (a single exp reaches ~5 % of its plateau at 3·τ_1e, so ×5 leaves a
    # little tail room); tuned on the real multi-angle set + a synthetic clean set.
    _LINEAR_FRAME_K = 5.0

    def _decay_characteristics(self) -> Optional[_DecayFrame]:
        """Decay scales of the currently-ticked correlograms (seconds), via the Qt-free
        :func:`_decay_frame`. Kept display-agnostic so both the linear-τ framing and any
        later 1/e guide-line overlay can reuse it."""
        return _decay_frame(self._raw.values())

    def _linear_x_clamp_display(self) -> Optional[float]:
        """Upper x-limit (DISPLAY units) for a LINEAR-τ correlogram view (lin-lin /
        log-lin), chosen so the decay sits roughly centered instead of crushed at the
        origin — the default full lag range (out to ~1e6 µs) buries a decay that
        completes decades earlier. Frame to ``_LINEAR_FRAME_K × τ_1e`` of the *slowest*
        ticked curve (its 1/e time sits at ~1/K of the axis, the mostly-decayed 5 % point
        near the middle), capped at the measured range. We deliberately key off τ_1e — the
        robust steep-region characteristic time — and NOT the 5 %-decayed point: one curve
        with a long shallow tail (a noisy aggregate mode) would otherwise blow the frame
        out and re-crush every other curve. That tail stays readable in the log views.
        Returns None when nothing is ticked, or when the ticked data has no usable
        positive-plateau decay to frame on (both fall through to full-range autoscale)."""
        dc = self._decay_characteristics()
        if dc is None:
            return None
        upper = min(self._LINEAR_FRAME_K * dc.tau_1e_s, dc.data_max_s)
        if not (math.isfinite(upper) and upper > 0):
            return None
        return upper * _disp_factor('time')

    def _redraw(self) -> None:
        for ax in (self.main_ax, self.resid_ax, *self.side_axes):
            ax.clear()
        if self._placeholder is not None:       # ax.clear() doesn't drop a figure.text
            self._placeholder.remove()
            self._placeholder = None
        if not self._raw:
            # Nothing ticked: hide EVERY axis (a blank main box in a sea of hidden
            # satellite/residual boxes reads as broken) and show one centred grey message
            # via the shared helper — the same clean empty-state look as the SLS tab.
            # Restored once a curve exists.
            self.figure.legends.clear()   # outlives ax.clear(); would float alone
            set_plot_export_dressing(self.canvas, None)   # ... and so does this
            self._placeholder = show_plot_placeholder(
                self.figure, self.canvas, 'Tick a measurement to plot',
                axes=(self.main_ax, self.resid_ax, *self.side_axes))
            return
        for ax in (self.main_ax, self.resid_ax, *self.side_axes):
            ax.set_visible(True)
        # Plot saves suggest the same stem as the CSV export for the active method.
        set_plot_export_name(
            self.canvas,
            _selection_export_names(
                self.controller,
                [(iid, self.method_combo.currentData()) for iid in self._raw],
                'correlogram')[0])
        ws = self.controller.workspace.measurements
        xs, ys = _SCALE_XY[self._scales[0]]
        x_clamp = self._linear_x_clamp_display()   # frames linear-τ views on the decay
        # A legend has no sample header above it, so it is one of the two surfaces that
        # prefixes the sample -- and only when the ticked set spans samples.
        prefixes = _sample_prefixes(self.controller, self._raw)
        # Everything drawn on this pass: one entry per (ticked measurement x ticked,
        # non-stale method). Computed ONCE and grouped by measurement, so the main view,
        # the three side views and the residual panel below all iterate the same list
        # and cannot disagree about which curve belongs to which fit.
        shown = self._shown_fits()
        fits_by_meas: Dict[str, List[Tuple[str, object]]] = {}
        for iid, key, res in shown:
            fits_by_meas.setdefault(iid, []).append((key, res))
        # main view: every ticked measurement's data ONCE, then each of its ticked fits.
        # Colour and marker are the MEASUREMENT (item 3); line style is the METHOD.
        for iid, (tau, g2) in self._raw.items():
            col = self.selection.color_for(iid)
            plot_correlogram_scaled(tau, g2, ax=self.main_ax, xscale=xs,
                                    yscale=ys, color=col,
                                    label=measurement_display_name(
                                        ws[iid], sample=prefixes[iid]),
                                    marker=self.selection.marker_for(iid))
            for key, res in fits_by_meas.get(iid, ()):
                plot_correlogram_fit_line(res.fit_tau_s, res.fitted_g2m1, ax=self.main_ax,
                                     xscale=xs, yscale=ys, color=col,
                                     linestyle=_METHOD_LINESTYLE[key])
        self.main_ax.set_title(f'Correlogram — {self._scales[0]}')
        self._draw_markers()                     # window + baseline handles
        if xs == 'linear':
            # Pin a linear delay axis to 0 (negative tau is meaningless) and frame the
            # right to the decay region so it sits centered, not crushed at the origin.
            # Done before the residual copies the x-lim below.
            right = x_clamp if x_clamp is not None else self.main_ax.get_xlim()[1]
            self.main_ax.set_xlim(0.0, right)
        self._draw_legend(shown)
        # residual panel: the residuals of exactly the fits drawn above, in the same
        # colour and line style, through the one helper the Distribution tab's panel
        # also uses. Drawn in the same DISPLAY units as the main axis so it lines up
        # under the fit -- it once drew raw seconds on a microsecond axis.
        draw_fit_residuals(
            self.resid_ax,
            [(self.selection.color_for(iid), key, res) for iid, key, res in shown],
            xscale=xs)
        self.resid_ax.set_xlim(self.main_ax.get_xlim())
        _hide_shared_x(self.main_ax)         # the residual carries the only x axis
        # side views: overlay all measurements at each scale
        for k, sax in enumerate(self.side_axes):
            name = self._scales[k + 1]
            sxs, sys = _SCALE_XY[name]
            for iid, (tau, g2) in self._raw.items():
                col = self.selection.color_for(iid)
                plot_correlogram_scaled(tau, g2, ax=sax, xscale=sxs, yscale=sys,
                                        compact=True, color=col,
                                        marker=self.selection.marker_for(iid))
                for key, res in fits_by_meas.get(iid, ()):
                    plot_correlogram_fit_line(res.fit_tau_s, res.fitted_g2m1, ax=sax,
                                         xscale=sxs, yscale=sys, color=col,
                                         linestyle=_METHOD_LINESTYLE[key],
                                         compact=True)
            if sxs == 'linear':                   # pin to 0; frame on the decay
                right = x_clamp if x_clamp is not None else sax.get_xlim()[1]
                sax.set_xlim(0.0, right)
            sax.set_title(name, fontsize=8)
        self.canvas.draw_idle()
        self._floor_resid_ratio()            # after the layout is real, not before

    def _draw_legend(self, shown) -> None:
        """The on-screen KEY -- one entry per METHOD drawn plus the window/baseline
        markers -- and the dressing that gives the SAVED figure the full legend.

        Not one entry per curve, and no longer one per measurement either. Before B8
        the data scatter and its fit line were each labelled, so nine ticked runs drew
        eighteen entries, clipped at eleven, covering the plot area. Fixing the count
        left thirteen full-length entries that no placement could carry: below the
        panel they took 154 px and left the residual 19 px of a 477 px canvas.

        The measurement names now go to the file, where a reader has no picker or
        glance table beside them; on screen those two already name every ticked run
        against its swatch. What stays here is `_measurement_legend_entries`' worth of
        nothing and a key that does not grow with the selection.

        The key sits in the figure's EMPTY bottom-right gridspec cell -- `[3, 1]`, the
        square under the side-view stack and beside the residual, which nothing has
        ever drawn in -- so it costs the plot no height at all. Measured on the nine-run
        frame: residual 19.0 -> 59.8 px, fit 154.6 -> 299.6 px. When the user drags the
        residual small enough that the cell cannot hold the key, it falls back to
        sitting below the panel.
        """
        ws = self.controller.workspace.measurements
        methods = list(dict.fromkeys(key for _i, key, _r in shown))
        key_entries = _method_legend_entries(methods) + _marker_legend_entries(
            self.main_ax)
        # A fit whose own `success` is False drew a SEED estimate, not a fit, and the
        # saved image must not present it as fitted (module docstring, owner-ruled).
        # This tab draws the Distribution tab's stored fits now, so it inherits that
        # duty; `_shown_fits` hands back the result objects, which carry `success`.
        failed = {}
        for iid, key, res in shown:
            if not getattr(res, 'success', True):
                failed.setdefault(iid, []).append(_fit_label(key))
        leg = _draw_key(self.figure, key_entries, bbox=self._key_cell())
        if leg is not None and not self._key_fits(leg):
            leg = _draw_key(self.figure, key_entries)      # below the panel instead

        # The saved figure additionally names every measurement, grouped by sample.
        prefixes = _sample_prefixes(self.controller, self._raw)
        by_sample = {}
        for iid in self._raw:
            sample = prefixes[iid] or _sample_label_of(self.controller, iid)
            note = failed.get(iid)
            by_sample.setdefault(sample, []).append(
                (iid, measurement_display_name(ws[iid])
                 + (f' ({", ".join(note)} failed)' if note else '')))
        meas_entries, groups = _measurement_legend_entries(self.selection, by_sample)
        set_plot_export_dressing(
            self.canvas,
            export_legend_dressing(self.figure, meas_entries + key_entries, groups)
            if meas_entries else None)

    def _key_cell(self):
        """The region the on-screen key sits in: the SIDE-VIEW column's x-range, at the
        `[3, 1]` cell's height -- or None before the first draw has positioned it.

        The x-range comes from the drawn side views, not from the gridspec cell. The
        cell is empty by construction, but with nothing to its right the residual axes
        SPILLS into it: measured 1015 px against a cell starting at 1003 px at
        1600x1020, and 1372 against 1351 at 1920x1080, so a key anchored at the cell's
        left edge printed over the residual plot (owner report, 2026-09-19). It did not
        at 1280x760, which is why the shipped screenshots did not show it. The side
        views never spill -- they have the residual's own row beneath them -- and they
        are the column the key visually belongs to.
        """
        try:
            cell = self._gs[3, 1].get_position(self.figure)
            side = self.side_axes[-1].get_position()
        except Exception:  # noqa: BLE001 — no layout yet: fall back to below the panel
            return None
        from matplotlib.transforms import Bbox
        return Bbox([[side.x0, cell.y0], [side.x1, cell.y1]])

    def _key_fits(self, leg) -> bool:
        """Whether `leg` is inside its cell. The cell shrinks with the residual's
        share, which the user can drag to a minimum, so this is a real branch."""
        cell = self._key_cell()
        if cell is None:
            return False
        try:
            ext = leg.get_window_extent(self.canvas.get_renderer())
        except Exception:  # noqa: BLE001 — unmeasurable: take the safe placement
            return False
        w = self.figure.get_figwidth() * self.figure.dpi
        h = self.figure.get_figheight() * self.figure.dpi
        return ext.width <= cell.width * w + 1 and ext.height <= cell.height * h + 1


    # ---- tables (a GLANCE row per ticked run + a panel for the focused one) ----
    def _refresh_tables(self) -> None:
        self._rebuild_tables()
        self._fill_tables()

    def _focused_id(self):
        """The run the detail panel describes: the sidebar's focused measurement when it
        is ticked, else the first ticked run.

        A panel that blanked whenever focus sat on an unticked run would withhold the
        detail quantities exactly when several runs are co-plotted -- which is when they
        are wanted -- so it falls back rather than emptying."""
        ids = [i for i in self.selection.ids()
               if i in self.controller.workspace.measurements]
        if self.item_id in ids:
            return self.item_id
        return ids[0] if ids else None

    def _rebuild_tables(self) -> None:
        ws = self.controller.workspace.measurements
        ids = [i for i in self.selection.ids() if i in ws]
        key = self.method_combo.currentData()
        method_label = {k: l for l, k in _CORR_METHODS}.get(key, 'Method')
        # "apparent" is said ONCE here instead of on every headline column: the glance
        # table is only as readable as its narrowest column, and the qualifier is a
        # property of the whole table (invariant 7 -- single-angle results are apparent).
        self._results_title = f'{method_label} results — apparent sizes'
        self.results_label.setText(self._results_title)
        self.results_label.setToolTip('')

        # `.get`, not a subscript: a method listed in `_CORR_METHODS` with no entry
        # here is a KeyError on every draw of the tab. The roster gate below makes that
        # state a red test instead, and this keeps it from being a crash meanwhile.
        cols = _GLANCE_COLS.get(key, [])
        g = self.glance_table
        g.clear()
        g.setColumnCount(1 + len(cols))
        g.setRowCount(len(ids))
        g.setHorizontalHeaderLabels(['Measurement', *[h for h, _idx in cols]])
        for r, iid in enumerate(ids):
            lm = ws[iid]
            name = QtWidgets.QTableWidgetItem(measurement_display_name(lm))
            # Colour AND marker shape, so a row means the same thing as its curve for a
            # reader who cannot use the colour (`marker_swatch`, WCAG 1.4.1).
            name.setIcon(marker_swatch(self.selection.color_for(iid),
                                       self.selection.marker_for(iid)))
            name.setToolTip(measurement_hover_card(lm))
            g.setItem(r, 0, name)
            for c in range(1, 1 + len(cols)):
                g.setItem(r, c, QtWidgets.QTableWidgetItem('—'))
        g.resizeColumnsToContents()
        # Cap the NAME column and give the rest to the values. Uncapped, one long file
        # stem takes the whole 360 px control column and pushes the headline values out
        # -- the defect this table exists to fix. The widget elides left, so the capped
        # name keeps the tail that tells replicates of one set apart.
        if g.columnWidth(0) > NAME_COL_MAX_W:
            g.setColumnWidth(0, NAME_COL_MAX_W)
        for c in range(1, 1 + len(cols)):
            g.horizontalHeader().setSectionResizeMode(
                c, QtWidgets.QHeaderView.ResizeMode.Stretch)
        # Height capped at `_GLANCE_VISIBLE_ROWS`, NOT grown to the row count. Grown,
        # nine ticked runs asked for ~320 px of a 330 px section and pushed the
        # focused-run panel, the export button and its own rows out of the pane -- the
        # user saw a header row and nothing under it. Beyond the cap it scrolls.
        row_h = g.verticalHeader().defaultSectionSize()
        chrome = g.horizontalHeader().height() + 2 * g.frameWidth() + 2
        shown = min(max(len(ids), 1), _GLANCE_VISIBLE_ROWS)
        g.setMaximumHeight(chrome + row_h * shown)
        # ... and a floor, so a squeezed section cannot shrink it to its header alone.
        g.setMinimumHeight(chrome + row_h * min(max(len(ids), 1), _GLANCE_FLOOR_ROWS))

        # Everything that is not a headline stays on this tab, for the focused run.
        _repoint_vtable(self.result_table, _RESULT_ROWS.get(key, []))

    def _mark(self, item, source, attrs, whole: bool) -> None:
        """Mark a value cell with the notes block's marker for the notes about it, its
        tooltip the block's popup; a cell with no note keeps its own full text as the
        tooltip, which keeps a value recoverable when a narrow column elides it."""
        if not self.notes_block.mark_item(item, source, attrs, whole=whole):
            item.setToolTip(item.text())

    def _fill_tables(self) -> None:
        key = self.method_combo.currentData()
        ws = self.controller.workspace.measurements
        ids = [i for i in self.selection.ids() if i in ws]
        sig = self._no_unc_sig()
        cols = _GLANCE_COLS.get(key, [])
        fits = [(iid, self._fit_for(iid)) for iid in ids]
        shown = [(measurement_display_name(ws[iid]), res) for iid, res in fits
                 if res is not None]
        # Before the cells, so each cell's marker letter is the line's.
        self.notes_block.show_notes(shown)
        self._block_fits = {id(res) for _name, res in shown}
        # The fits' residual weighting, when it is not the default, in the table's title.
        words, clauses = _weighting_title([res for _name, res in shown])
        title = getattr(self, '_results_title', self.results_label.text())
        self.results_label.setText(f'{title}, {", ".join(words)}' if words else title)
        self.results_label.setToolTip(clauses)
        for r, (_iid, res) in enumerate(fits):
            if res is None:
                continue
            # ONE computation feeds both views: the glance cells are indices into the
            # same `_result_values` list the panel below renders in full, so the two can
            # never print different numbers for one run.
            values = self._result_values(key, res, sig)
            sources = self._value_sources(key, res)
            for c, (_hdr, idx) in enumerate(cols, start=1):
                it = self.glance_table.item(r, c)
                if it is None:
                    continue
                it.setText(values[idx])
                # The marker sits on the value it is about, never on the name.
                self._mark(it, *sources[idx])

        focus = self._focused_id()
        if focus is None:
            self.focus_label.setText('')
            self.focus_label.setToolTip('')
        else:
            self.focus_label.setText(
                f'Details — {measurement_display_name(ws[focus])}')
            self.focus_label.setToolTip(measurement_hover_card(ws[focus]))
        res = self._fit_for(focus) if focus is not None else None
        if res is None:
            _reset_vtable(self.result_table)
            return
        _fill_vtable(self.result_table, self._result_values(key, res, sig))
        for r, source in enumerate(self._value_sources(key, res)):
            it = self.result_table.item(r, 1)
            if it is not None:
                self._mark(it, *source)

    def _no_unc_sig(self) -> int:
        """Sig figs for the ±-less result values (Settings → no-uncertainty precision).

        A single-correlogram fit has no honest ±, so ALL of its reported
        numbers — Rh, Γ, μ₂, β, τ, stretch, amplitude fractions — use this one fixed
        precision. Standing policy: a value with a ± is set by its ± and never comes
        here; a value without one is set by this knob (default 3)."""
        return self.controller.settings.no_uncertainty_sig_figs

    @staticmethod
    def _result_values(key: str, res, sig: int = 3) -> List[str]:
        """The result-table column values, in `_RESULT_ROWS[key]` order. Every reported
        number here is from a single correlogram (no honest ±), so all use
        the `sig` no-uncertainty precision; `order`/method/convergence are labels."""
        def q(v):
            return format_fixed_sig(v, sig)
        if key == 'cumulant':
            # A failed nonlinear fit now reports NaN outputs, not a linear-fit
            # substitute — label it '(failed)', and the Rh/Gamma cells read 'n/a'.
            method_lbl = res.method + ('' if res.success else ' (failed)')
            baseline_lbl = q(res.baseline)   # fitted B (nonlinear) or the pinned offset

            # One rule for every order-gated moment cell (PDI/μ₂ at order 2, μ₃ at 3, μ₄
            # at 4): below the order that fits the quantity it is UNMEASURED and says so,
            # so a blank is never read as a fitted zero; at or above it the value renders
            # through q(), which already turns a failed fit's None or NaN into 'n/a'.
            def moment_cell(value, needed_order):
                if res.order < needed_order:
                    return f'unmeasured (order {res.order})'
                return q(value)

            # PDI cell: the value with ✓ when it is valid; 'unmeasured (order 1)' for an
            # order-1 fit (no mu2 term), 'n/a' for a failed one. A PDI above the limit
            # takes the notes block's marker (`_fill_tables`); a negative one keeps its
            # own glyph, a verdict the fit carries with no note.
            if math.isfinite(res.pdi):
                pdi_cell = (f'{q(res.pdi)} ✓' if res.pdi_valid
                            else f'{q(res.pdi)} ⚠<0' if res.pdi < 0 else q(res.pdi))
            else:
                pdi_cell = moment_cell(res.pdi, 2)
            # Rh cell: the bare value; a failed precondition about it adds the notes
            # block's marker in `_fill_tables`.
            rh_cell = q(res.rh_nm)

            # μ₃ / μ₄ cells: the BARE value, no validity glyph -- the ruling and its reason
            # are on `CumulantResult`; the caution is the row label's tooltip.
            mu3_cell = moment_cell(res.mu3_s_inv3, 3)
            mu4_cell = moment_cell(res.mu4_s_inv4, 4)
            # mu2 through the SAME order gate as its siblings. It was the one moment
            # rendered raw, and below order 2 the engine returns 0.0 rather than None, so
            # the panel printed a bare '0' for a moment it never estimated -- while the
            # PDI cell directly above said "unmeasured (order 1)" about the same quantity.
            return [q(res.gamma_s_inv), rh_cell, pdi_cell,
                    moment_cell(res.mu2_s_inv2, 2), mu3_cell, mu4_cell,
                    str(res.order), method_lbl, baseline_lbl]
        # Parametric-fit Rh cells: the bare value, as on the cumulant; a NaN Rh renders
        # 'n/a' via q(), and a failed precondition about the size adds the notes block's
        # marker in `_fill_tables` -- PER MODE for single/double, whose modes are judged
        # independently (`_value_sources`).
        if key == 'single':
            return [q(res.mode.gamma_s_inv), q(res.mode.rh_nm),
                    q(res.beta), q(res.baseline), 'yes' if res.success else 'no']
        if key == 'double':
            return [q(res.mode1.rh_nm), q(res.mode1.amplitude_fraction),
                    q(res.mode2.rh_nm), q(res.mode2.amplitude_fraction),
                    q(res.baseline), 'yes' if res.success else 'no']
        return [q(res.stretch), q(res.tau_c_s), q(res.mean_tau_s),
                q(res.rh_from_tau_c_nm), q(res.rh_from_mean_tau_nm),
                q(res.baseline), 'yes' if res.success else 'no']

    @staticmethod
    def _value_sources(key: str, res) -> List[Tuple[object, Tuple[str, ...], bool]]:
        """What each value of `_result_values` is ABOUT, index-aligned with it and so
        with `_RESULT_ROWS[key]`: `(source, attrs, whole)`, the result whose notes the
        cell's marker reads, the attributes a note must name to sit on that cell, and
        whether a note about the whole fit does too (the sizes only).

        A note names the precondition of the value it sits ON (invariant 4), so the
        routing is per MODE for single/double, whose modes are judged independently and
        each carry their own notes about their own `rh_nm`; the cumulant and KWW have one
        verdict, read off the fit."""
        if key == 'cumulant':
            return [(res, ('gamma_s_inv',), False), (res, ('rh_nm',), True),
                    (res, ('pdi',), False), (res, ('mu2_s_inv2',), False),
                    (res, ('mu3_s_inv3',), False), (res, ('mu4_s_inv4',), False),
                    (res, ('order',), False), (res, ('method',), False),
                    (res, ('baseline',), False)]
        if key == 'single':
            return [(res.mode, ('gamma_s_inv',), False), (res.mode, ('rh_nm',), True),
                    (res, ('beta',), False), (res, ('baseline',), False),
                    (res, ('success',), False)]
        if key == 'double':
            return [(res.mode1, ('rh_nm',), True),
                    (res.mode1, ('amplitude_fraction',), False),
                    (res.mode2, ('rh_nm',), True),
                    (res.mode2, ('amplitude_fraction',), False),
                    (res, ('baseline',), False), (res, ('success',), False)]
        return [(res, ('stretch',), False), (res, ('tau_c_s',), False),
                (res, ('mean_tau_s',), False), (res, ('rh_from_tau_c_nm',), True),
                (res, ('rh_from_mean_tau_nm',), True), (res, ('baseline',), False),
                (res, ('success',), False)]

    # ---- status / flags ----
    def _update_status(self) -> None:
        key = self.method_combo.currentData()
        ws = self.controller.workspace.measurements
        notes = []
        if self.controller.is_dirty() and self._any_cached():
            notes.append('(ran on last committed values)')
        # Run refits only the ticked runs it can fit, so only their stale fits ask for it.
        stale = [iid for iid in self.selection.ids() if self._cache_is_stale(iid, key)]
        if any(iid in self._fittable for iid in stale):
            notes.append(_STALE_HINT)
        elif stale:
            notes.append(_STALE_UNRUNNABLE)
        # Before them, each ticked run Run leaves out and what to enter for it.
        lines = _left_out_lines(self.controller, self._left_out)
        text = '  '.join(notes)
        self.status.setText('\n'.join([*lines, text] if text else lines))
        # Operational flags only, one line per run, each led by its glyph: the run the
        # displayed method's Run skipped, and a shown fit -- in the tables or drawn from
        # the layers list -- that failed with no note on the block to say why.
        lines = []
        for iid in self.selection.ids():
            lm = ws.get(iid)
            failure = self._failure(iid, key) if lm is not None else None
            if failure is not None:
                lines.append(f'⚠ {measurement_display_name(lm)}: skipped — '
                             f'{humanize_field_names(failure)}')
        table = [(iid, key, self._fit_for(iid)) for iid in self.selection.ids()]
        failed = dict.fromkeys(
            (measurement_display_name(ws[iid]), _fit_label(k))
            for iid, k, res in [*table, *self._shown_fits()]
            if iid in ws and res is not None
            and _failed_without_cause(res, self._block_fits))
        lines += [f'⚠ {name}: {label} fit failed' for name, label in failed]
        self.flag_label.setText('\n'.join(lines))
