"""
The DLS Mode Map sub-tab: every extracted mode of every run in (q², Γ) space, with
user-picked mode families, their Γ-vs-q² fits and the recorded mode sets.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from plotting.plots import (plot_mode_map, plot_mode_family_fit, plot_amplitude_vs_q,
                            cycle as _plot_cycle, marker_cycle as _marker_cycle,
                            display_factor as _disp_factor, display_unit as _disp_unit)
from core.notes import WEIGHTING_PREFIX, Note, find_notes, render
from gui.plot_controls import (make_split_panels, make_vertical_plot_stack, pin_under_column,
                               scrolling_notes, COLUMN_STACK_STRETCH, set_plot_export_name)
from gui.export_helper import export_to_csv, sample_suggested_name
from gui.field_labels import humanize_field_names
from gui.help import add_help_to_groupbox, section_header
from gui.notes_block import NotesBlock
from gui.plot_overlays import reissue_legend_preserving_title, show_plot_placeholder
from gui.distribution_fit_fields import DistributionFitFields
from gui.theme import ThemedLabel, set_flag
from gui.widgets import make_plot_pane
from gui.worker import BACKGROUND_RUN_TOOLTIP, BUSY_NOTICE, RunSlot, runner
from core.formatting import format_fixed_sig, format_pm

from gui.dls.region import RegionCheckBox
from gui.dls.common import (_STALE_HINT, _STALE_UNRUNNABLE, _FOCUSED, _RESULTS_TITLE,
                            _title_weightings, _left_out_lines, _union_missing, _needs_lines)
from gui.dls.tables import _vtable, _label_row
from gui.dls.bases import _SampleRunTab


# Why 'Record mode set' cannot act yet, keyed by the controller's reason code
# (the _INELIGIBLE_REASON pattern: the wording lives at the display surface).
_RECORD_NOT_READY = {
    'no_map': 'Run the mode map first — a mode set records what the map '
              'currently shows.',
    'no_families': 'Add a family and pick map points into it first; the '
                   'recorded set snapshots the picked families.',
    'no_matches': 'No picked family currently matches a map point (the '
                  'extraction changed since the picks were made) — re-pick, '
                  'or re-run with the original source.',
}


# The mode-family fit's values read from its rate-versus-q² slope: D, the Rh from it, and
# the q⁴ form's coefficients. A VH family's word goes on each.
_SLOPE_ATTRS = frozenset({'d_m2_s', 'rh_nm', 'a_m2_s', 'b_m4_s'})


class _ModeMapTab(_SampleRunTab):
    """The mode map: every extracted mode of every run in (q², Γ) space, with
    user-picked mode families and per-family Γ-vs-q² fits.

    No association machinery: which points form a family is the USER's pick,
    made by clicking map points into the active family (the SLS mask
    click-to-toggle pattern; picks are value-keyed, so a re-extraction orphans
    a pick rather than re-aiming it). The through-origin verdict is a
    tolerance-carrying LABEL — a slow mode failing it is the interesting
    finding, not an error — and every family's Rh is apparent + mode-resolved.
    The map facets by polarization state, one state shown at a time, so a
    family can never span states.
    """

    selectionChanged = QtCore.Signal()           # ticked include-set changed
    defers_repaint = True        # repaints when shown (`_DeferredRepaint`)
    _PICK_RADIUS_PX = 12.0

    _SOURCES = [('Double exponential', 'double'), ('CONTIN', 'contin'),
                ('NNLS', 'nnls')]
    _STATE_LABELS = {'VV': 'VV (polarized)', 'VH': 'VH (depolarized)',
                     None: 'Unpolarized (VU / untagged)'}

    def __init__(self, controller, parent=None, *, distribution_fit_values=None,
                 region=None) -> None:
        super().__init__(controller, parent, region=region)
        # The Distribution sub-tab's current field values, read once per press of the
        # copy button -- an accessor handed in by the owning module, so neither tab
        # reaches into the other (the Correlogram tab's `distribution_fits` shape).
        self._distribution_fit_values = distribution_fit_values
        self._points: Dict[str, list] = {}       # sid -> metadata rows
        self._suppress_table = False
        self._suppress_family = False
        self._last_ckey: Optional[tuple] = None  # (sample, fraction) shown: the epoch's key
        # The Run's epoch and its deferred (sid, fraction), retried while the chosen
        # source can still run on that target.
        self._run_slot = RunSlot(available=self._deferred_run_available,
                                 rerun=lambda target: self._recompute(*target),
                                 before_delivery=self.flush_view)
        self._placeholder = None
        # The Run gate (`_update_run_enabled`): its lines for the status line (the ticked
        # runs the source leaves out, or the ask to tick one), the text it last wrote
        # there, so only its own text is cleared, and whether a ticked run can run; and
        # each source entry's own tooltip, put back when it stops being grayed.
        self._gate_lines: List[str] = []
        self._gate_status = ''
        self._gate_runnable = False
        self._entry_tips: Dict[str, object] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        # The Distribution tab's floor and default, for the reason stated there: the
        # Inversion Settings group carries the same Rh-grid triple-spinbox row.
        splitter, left, right = make_split_panels(self, left_min_width=385,
                                                  sizes=(403, 717))

        # ---- controls section ----
        controls = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(controls); cl.setContentsMargins(0, 0, 0, 0)
        cl.addWidget(section_header(
            'Mode Map', 'Extract every relaxation mode of every run, then click '
            'points into families to fit each mode separately.',
            bullets=[
                '<b>Run</b> extracts each ticked run\'s modes with the chosen source and '
                'plots them in (q², Γ) space.',
                'A source no ticked measurement can use is <b>grayed</b>, its tooltip '
                'naming what to enter.',
                'Add a family, select it, then click map points to assign them; clicking '
                'again removes, clicking another family\'s point moves it.',
                'Each family with points at ≥ 2 angles is fitted on the spot '
                '(Theory-and-Equations-Guide §3.5.4).',
                '<b>Hollow</b>: below the amplitude floor; <b>square</b>: undecayed, or '
                'its peak on the Rh grid\'s end; the legend names which.',
                'More: User-Manual §8.5.']))
        srow = QtWidgets.QWidget()
        sl = QtWidgets.QFormLayout(srow); sl.setContentsMargins(0, 0, 0, 0)
        self.source_combo = QtWidgets.QComboBox()
        for label, code in self._SOURCES:
            self.source_combo.addItem(label, code)
        self.source_combo.setToolTip(
            'Which fit supplies each run\'s modes: the two-mode double exponential '
            '(fast), or the resolved peaks of a CONTIN or NNLS distribution, inverted on '
            'the Inversion Settings below (slower per run, steeply so on a finer Rh '
            'grid). See the Theory-and-Equations-Guide §3.5.4.')
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        sl.addRow('Source:', self.source_combo)
        self.facet_combo = QtWidgets.QComboBox()
        self.facet_combo.setToolTip(
            'The polarization state shown (and pickable); states are never pooled on one '
            'map (Theory-and-Equations-Guide §3.6.1).')
        self.facet_combo.currentIndexChanged.connect(self._on_facet_changed)
        sl.addRow('Polarization:', self.facet_combo)
        cl.addWidget(srow)
        self.region_box = RegionCheckBox(self._region)
        self.region_box.toggled.connect(self._on_region_toggled)
        cl.addWidget(self.region_box)
        # The inversion's own fields -- the shared group the Distribution sub-tab and the
        # Settings tab hold -- shown for the distribution sources only: the parametric
        # double exponential reads none of them.
        self.inversion_box = QtWidgets.QGroupBox('Inversion Settings')
        add_help_to_groupbox(
            self.inversion_box,
            'What the CONTIN and NNLS sources invert each ticked run on.',
            bullets=[
                'They start from the Settings tab’s DLS Distribution Defaults and can be '
                'changed here; a change marks a shown map out of date.',
                'A Settings-tab Update that changes any setting, or Restore defaults, sets '
                'them back to the Settings values.',
                'The α rows are CONTIN’s alone and hide for NNLS.',
                '<b>Copy from the Distribution tab</b> fills every field once; a later '
                'change there does not follow.',
                'The α range, the F-test level and each point’s α, selector, γ and grid '
                'are recorded with the map and its export.',
            ])
        ibox_form = QtWidgets.QFormLayout(self.inversion_box)
        self.fit_fields = DistributionFitFields(self)
        self.fit_fields.add_rows(ibox_form)
        self.fit_fields.seed_from_settings(self.controller.settings)
        self.fit_fields.changed.connect(self._on_fit_input_moved)
        self.copy_fields_button = QtWidgets.QPushButton('Copy from the Distribution tab')
        self.copy_fields_button.setToolTip(
            'Fill every field above once with the Distribution sub-tab’s current '
            'values. A one-time copy: a later change on the Distribution sub-tab does '
            'not follow, and a Settings-tab Update that changes any setting sets the '
            'fields back to the Settings values.')
        self.copy_fields_button.clicked.connect(self._on_copy_fit_fields)
        self.copy_fields_button.setVisible(self._distribution_fit_values is not None)
        ibox_form.addRow(self.copy_fields_button)
        cl.addWidget(self.inversion_box)
        self._sync_source_controls()
        self.run_button = QtWidgets.QPushButton('Run')
        self.run_button.setToolTip(BACKGROUND_RUN_TOOLTIP)
        self.run_button.clicked.connect(self._on_run)
        # Run is pinned under the scrolled controls and Export under the column (the
        # §3 column template), so the Inversion Settings group a distribution source
        # shows can never carry them out of view.
        self.export_button = QtWidgets.QPushButton('Export CSV…')
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._on_export)
        self.bounds_button = QtWidgets.QPushButton('Integration bounds…')
        self.bounds_button.setEnabled(False)
        self.bounds_button.setToolTip(
            'Set manual Rh integration intervals per run for the CONTIN/NNLS '
            'sources, replacing the automatic lobe split where peaks overlap; for '
            'genuine overlap the double exponential\'s model amplitudes are the better '
            'route, since a manual split strains the decomposition\'s wide-separation '
            'condition (Theory-and-Equations-Guide §4.2). Bounds apply at the next Run '
            'and travel as provenance; the parametric source ignores them.')
        self.bounds_button.clicked.connect(self._on_bounds)
        cl.addWidget(self.bounds_button)
        self.record_set_button = QtWidgets.QPushButton('Record mode set…')
        self.record_set_button.setEnabled(False)
        self.record_set_button.setToolTip(
            'Record the picked families — their amplitudes, decay rates, '
            'regularization and integration bounds — as a mode set, which the SLS '
            'tab\'s Mode-Resolved sub-tab can partition any SLS intensity series of '
            'this sample by. A recorded set never follows later re-fits; record again '
            'for a fresh one.')
        self.record_set_button.clicked.connect(self._on_record_set)
        # Record mode set… acts on the families, so it sits in their pane below.
        self.status = QtWidgets.QLabel('')
        self.status.setWordWrap(True)

        # ---- measurements table (tick to include) ----
        tsec = QtWidgets.QWidget()
        tl = QtWidgets.QVBoxLayout(tsec); tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(section_header(
            'Measurements (tick to include in the map)',
            'Tick which runs are extracted when you press Run.',
            bullets=[
                'Only this sample\'s DLS measurements appear.',
                'Grayed rows can\'t be ticked — hover a row to see why.',
                'A replicate average next to its own source runs starts unticked, and '
                '<b>Select all</b> leaves it so; a family fit that picks its modes '
                'beside theirs says so.',
                'A run whose extraction fails contributes no points; the '
                'reason appears under the results.']))
        self.table_note = ThemedLabel('', role='hint', size=10, hide_when_empty=True)
        self.table_note.setWordWrap(True); tl.addWidget(self.table_note)
        helper = QtWidgets.QWidget()
        hl = QtWidgets.QHBoxLayout(helper); hl.setContentsMargins(0, 0, 0, 0)
        self.select_all_btn = QtWidgets.QPushButton('Select all')
        self.select_all_btn.clicked.connect(self._select_all)
        self.select_none_btn = QtWidgets.QPushButton('Select none')
        self.select_none_btn.clicked.connect(self._select_none)
        hl.addWidget(self.select_all_btn); hl.addWidget(self.select_none_btn)
        hl.addStretch(1)
        tl.addWidget(helper)
        self.table = QtWidgets.QTableWidget(0, 0)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(120)  # ~4 rows + header at the shipped 760 px height (R3.2)
        self.table.itemChanged.connect(self._on_table_changed)
        tl.addWidget(self.table, 1)

        # ---- families + results section ----
        fsec = QtWidgets.QWidget()
        fl = QtWidgets.QVBoxLayout(fsec); fl.setContentsMargins(0, 0, 0, 0)
        fl.addWidget(section_header(
            'Families', 'The selected family is the one your clicks assign '
            'points to.',
            bullets=[
                'Add creates a family; double-click renames it.',
                'The color/marker chip matches the family\'s map points.',
                'Role (fast/slow) feeds the amplitude report; at most one '
                'family holds each role.',
                'Delete removes the family and its results.',
                '<b>Click</b> a ⚠ line or lettered value for what to do.']))
        frow = QtWidgets.QWidget()
        fr = QtWidgets.QHBoxLayout(frow); fr.setContentsMargins(0, 0, 0, 0)
        self.add_family_btn = QtWidgets.QPushButton('Add family')
        self.add_family_btn.clicked.connect(self._on_add_family)
        self.del_family_btn = QtWidgets.QPushButton('Delete')
        self.del_family_btn.clicked.connect(self._on_del_family)
        self.role_combo = QtWidgets.QComboBox()
        self.role_combo.addItem('—', '')
        self.role_combo.addItem('fast', 'fast')
        self.role_combo.addItem('slow', 'slow')
        self.role_combo.setToolTip('Designate the selected family for the '
                                   'amplitude report.')
        self.role_combo.currentIndexChanged.connect(self._on_role_changed)
        fr.addWidget(self.add_family_btn); fr.addWidget(self.del_family_btn)
        fr.addWidget(QtWidgets.QLabel('Role:')); fr.addWidget(self.role_combo)
        fr.addStretch(1)
        fl.addWidget(frow)
        self.family_list = QtWidgets.QListWidget()
        self.family_list.setMinimumHeight(80)
        self.family_list.currentRowChanged.connect(self._on_family_selected)
        self.family_list.itemChanged.connect(self._on_family_renamed)
        fl.addWidget(self.family_list)
        # The selected family's results; the title names a residual weighting other than
        # the default (`_title_weightings`), so it wraps rather than widen the column.
        self.results_label = QtWidgets.QLabel(_RESULTS_TITLE)
        self.results_label.setWordWrap(True)
        fl.addWidget(self.results_label)
        self.result_table = _vtable([lbl for lbl, _ in self._result_spec()])
        fl.addWidget(self.result_table)
        # The notes of the map, the selected family's fit and the amplitude report: one
        # line per failed precondition, a marker on each value it is about; pinned under
        # the column right below the family results (R8.6). Their records leave the tab
        # but for the fallback weight's plain line (`gui/notes_block.py::SHOWN_RECORDS`).
        self.notes_block = NotesBlock(what='angles')
        self.notes_block.watch(self.result_table)
        fl.addWidget(self.record_set_button)
        # Pinned under the column for an operational flag; none is raised today.
        self.flag_label = ThemedLabel('', size=11)
        self.flag_label.setWordWrap(True)
        # The qualifiers the map relays from its runs, but the residual weighting, which is
        # a word on the family's Results rows (`_show_notes`).
        self.notes_label = ThemedLabel('', size=11)
        self.notes_label.setWordWrap(True)
        fl.addWidget(scrolling_notes(self.notes_label))

        # Scrolled, as the Distribution tab's controls are: the Inversion Settings group
        # makes the section taller than a short window can give it, and a floor at its
        # full height would starve the two panes beside it. The floor is the section's
        # height WITHOUT that group (the default double-exponential source hides it, and
        # a layout skips a hidden widget); Run is pinned under the scroll area, so it
        # stays in view whichever source is chosen.
        controls_scroll = QtWidgets.QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        controls_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        controls_scroll.setWidget(controls)
        floor = max(controls.sizeHint().height(), 120)
        ctrl_pane = QtWidgets.QWidget()
        ctrl_layout = QtWidgets.QVBoxLayout(ctrl_pane)
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        ctrl_layout.setSpacing(4)
        ctrl_layout.addWidget(controls_scroll, 1)
        ctrl_layout.addWidget(self.run_button)
        # Picker first, then the controls and Run, then the families and results (the
        # §3 column template).
        vstack = make_vertical_plot_stack(
            [tsec, ctrl_pane, fsec], sizes=[150, floor + 40, 200],
            min_heights=[130, floor, 120], stretch=COLUMN_STACK_STRETCH)
        left.addWidget(vstack, 1)
        pin_under_column(splitter, self.export_button, self.status,
                         self.notes_block, self.flag_label)

        # ---- right: view combo + canvas ----
        rw = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(rw); rl.setContentsMargins(0, 0, 0, 0)
        vrow = QtWidgets.QWidget()
        vl = QtWidgets.QHBoxLayout(vrow); vl.setContentsMargins(0, 0, 0, 0)
        vl.addWidget(QtWidgets.QLabel('View:'))
        self.view_combo = QtWidgets.QComboBox()
        self.view_combo.addItem('Mode map', 'map')
        self.view_combo.addItem('Amplitude vs q', 'amplitude')
        self.view_combo.setToolTip(
            'Amplitude vs q needs a "fast" and a "slow" family (Role above): '
            'it plots A_f/A_s against sin²(θ/2). No ± anywhere — single-run '
            'amplitudes carry no uncertainty.')
        self.view_combo.currentIndexChanged.connect(lambda _i: self._draw())
        vl.addWidget(self.view_combo); vl.addStretch(1)
        rl.addWidget(vrow)
        self.figure, self.canvas, self._nav, self.axis_bar = make_plot_pane(
            self, self.controller, (5.2, 4.3), constrained_layout=True)
        self.ax = self.figure.add_subplot(111)
        rl.addWidget(self._nav)
        rl.addWidget(self.canvas, 1)
        rl.addWidget(self.axis_bar)
        right.addWidget(rw, 1)
        self.canvas.mpl_connect('button_press_event', self._on_canvas_click)

    def _result_spec(self):
        df, du = _disp_factor('diffusion'), _disp_unit('diffusion')

        def pm(v, se):
            return format_pm(v * df, (se * df) if se is not None else None)
        sig = lambda x: format_fixed_sig(  # noqa: E731 — tiny local formatter
            x, self.controller.settings.no_uncertainty_sig_figs)
        return [
            (f'D ({du})', lambda r: pm(r.d_m2_s, r.d_se)),
            ('R<sub>h</sub> (nm)', lambda r: format_pm(r.rh_nm, r.rh_se)),
            ('R²', lambda r: sig(r.r_squared)),
            ('Through origin?', lambda r: '✓' if r.is_diffusive else '⚠ no'),
            (f'a ({du})', lambda r: pm(r.a_m2_s, r.a_se)),
            ('b (m⁴/s)', lambda r: format_pm(r.b_m4_s, r.b_se)),
            ('modes fitted', lambda r: str(r.angles_deg.size)),
        ]

    def _map_result(self, sid: str):
        return self.controller.results.get(('mode_map',) + self._ckey(sid))

    def _run_sig(self, ckey: tuple):
        """The signature this tab's Run of the map of `ckey`, a (sample, fraction), was
        dispatched under, kept beside the map in the controller; None when the stored map
        is not one this tab ran. The map itself is read live (`_map_result`)."""
        run = self.controller.shown_run(('mode_map', *ckey))
        return run.sig if run is not None and not run.superseded else None

    def _family_fits(self, sid: str) -> dict:
        return self.controller.results.get(
            ('mode_family',) + self._ckey(sid)) or {}

    def _focus_hook(self, sid: Optional[str]) -> None:
        """The Mode Map's part of the state half (`_SampleRunTab.focus_state`), run at
        every focus, shown or hidden: the re-point, the region choice, the rows and their
        ticks, and what a later focus reads back off the widgets, the facet and the family
        it keeps and the Run gate's status line."""
        ckey = self._ckey(sid) if sid is not None else None
        if ckey != self._last_ckey:      # (sample, fraction), like the Γ-q² tab
            self._run_epoch += 1
        self._last_ckey = ckey
        self.region_box.set_checked_silently(self._use_region.get(ckey, False))
        if not (self._runnable and sid is not None):
            self.family_list.clear()          # no family is kept across this focus
            self.run_button.setEnabled(False)
            self.status.clear()
            self._gate_lines, self._gate_status, self._gate_runnable = [], '', False
            self._gray_sources([])
            return
        self._points[sid] = self.controller.dls_sample_rows(
            sid, 'gamma_q2', self._fraction())
        fresh = {r['item_id'] for r in self._points[sid] if r['ok']}
        prev = self._included.get(ckey)       # per (sample, fraction), like the rows
        # First focus ticks the controller's default rows (`auto_tick`), as the Γ vs q²
        # tab does: a replicate average beside its own sources stays tickable, unticked.
        auto = {r['item_id'] for r in self._points[sid] if r['auto_tick']}
        self._included[ckey] = (prev & fresh) if prev is not None else auto
        # The facet and the family are kept from the widgets' last state, and the gate and
        # the stale mark each rewrite only their own text on the status line, so all four
        # run at every focus, the widgets the one record of the view.
        self._refresh_facets(sid)
        self._refresh_families(sid)
        self._update_run_enabled(sid)
        if self._map_result(sid) is not None:
            self._on_fit_input_moved()        # the stale mark, beside the gate's lines

    def repaint_view(self) -> None:
        """The repaint half: the table, the buttons, the table note, the results and the
        plot, from the stored focus and the rows the state half read."""
        sid = self._sample_id()
        if not (self._runnable and sid is not None):
            self.table.setRowCount(0)
            self.export_button.setEnabled(False)
            self.bounds_button.setEnabled(False)
            self.record_set_button.setEnabled(False)
            self.table_note.setText(
                'Select a DLS measurement in the sidebar to build this '
                'sample\'s mode map.')
            self._clear_results()
            self._clear('Select a DLS measurement.')
            return
        fresh = {r['item_id'] for r in self._points[sid] if r['ok']}
        self._refresh_table(sid)
        has_map = self._map_result(sid) is not None
        self.export_button.setEnabled(has_map)
        self.record_set_button.setEnabled(has_map)
        self.bounds_button.setEnabled(bool(fresh))
        if not self._points[sid]:
            self.table_note.setText('This sample has no DLS measurements.')
        elif not fresh:
            self.table_note.setText(
                'No measurements have confirmed parameters yet — set them in '
                'the Data tab and press Update, then return here.')
        else:
            self.table_note.clear()
        if has_map:
            self._fill_results(self._selected_family_fit(sid))
            self._draw()
        else:
            self._clear_results()
            self._clear('Tick measurements, then Run to extract this '
                        'sample\'s modes.')

    def _run_signature(self, ckey: tuple) -> tuple:
        """Everything a mode-map extraction for (sample, fraction) reads beyond its raw
        data: the committed params of the ticked measurements, the source, the settings
        the engine seeds (skip_initial_channels, for every source) and, for a
        distribution source, the Inversion Settings fields the Run hands it. Stored at
        dispatch, compared on redisplay and on every edit of those fields (the Γ-q²
        tab's shape)."""
        s = self.controller.settings
        source = self.source_combo.currentData()
        # `se_estimator` too: the Run ends by refitting the families, whose ± it routes,
        # and a switch clears those fits -- so a map shown across one needs a re-Run.
        seeds: tuple = (s.skip_initial_channels, s.se_estimator)
        if source != 'double':
            seeds += tuple(sorted(self._inversion_kwargs(source).items()))
        # The region choice (the Γ-q² tab's reasoning): toggling the box, or moving the
        # region while it is ticked, flags a shown map stale; an unticked one is left be.
        return (self.controller.committed_signature(self._included.get(ckey, set())),
                source, seeds, self.region_box.key(self._use_region.get(ckey, False)))

    def _region_moved(self) -> None:
        """The region choice is one of the map's fit inputs: re-check it as one."""
        self._on_fit_input_moved()

    # ---- inversion settings ----
    def _inversion_kwargs(self, source: str) -> Dict:
        """The kwargs a Run hands `run_mode_map` from this tab's own fields: none for the
        parametric source, which would refuse them."""
        return {} if source == 'double' else self.fit_fields.fit_kwargs(source)

    def _sync_source_controls(self) -> None:
        """Show the inversion fields for a distribution source only, and CONTIN's own
        rows for CONTIN only -- show/hide, never an inert field."""
        source = self.source_combo.currentData()
        self.inversion_box.setVisible(source != 'double')
        self.fit_fields.set_contin_rows_visible(source == 'contin')

    def _on_source_changed(self, _i) -> None:
        self._sync_source_controls()
        sid = self._sample_id()
        if self._runnable and sid is not None:
            self._update_run_enabled(sid)     # Run follows the chosen source
        self._on_fit_input_moved()

    def _on_fit_input_moved(self) -> None:
        """An input the next Run reads moved: say so beside a map drawn under the old
        inputs, after the Run gate's lines while it has any (`_show_gate`). Compared
        rather than assumed, so moving a field back clears the hint."""
        sid = self._sample_id()
        if sid is None:
            return
        if self._gate_lines:
            self._show_gate(sid)
            return
        if self._map_result(sid) is None:
            return
        ckey = self._ckey(sid)
        if self._run_sig(ckey) != self._run_signature(ckey):
            self.status.setText(_STALE_HINT)
        elif self.status.text() == _STALE_HINT:
            self.status.clear()

    def _on_copy_fit_fields(self) -> None:
        """Fill the fields ONCE from the Distribution sub-tab's current values. A copy,
        never a link: after it the map runs on what these fields show, so no value
        changes behind the Mode Map's back."""
        if self._distribution_fit_values is None:
            return
        self.fit_fields.seed(self._distribution_fit_values())
        copied = 'Copied the Distribution tab’s fit settings.'
        stale = self.status.text() == _STALE_HINT
        self.status.setText(f'{copied} {_STALE_HINT}' if stale else copied)

    def reseed_from_settings(self) -> None:
        """(Re)seed the Inversion Settings from the Settings defaults -- the cascade the
        shell runs after a Settings Update, as for the Distribution sub-tab."""
        self.fit_fields.seed_from_settings(self.controller.settings)

    # ---- measurements table ----
    def _refresh_table(self, sid: str) -> None:
        pts = self._points.get(sid, [])
        inc = self._included.get(self._ckey(sid), set())
        cu = _disp_unit('concentration')
        dfc = _disp_factor('concentration')
        Flag = QtCore.Qt.ItemFlag
        self._suppress_table = True
        try:
            t = self.table
            t.setColumnCount(3)
            t.setHorizontalHeaderLabels(['', 'Angle (°)', f'c ({cu})'])
            t.setRowCount(len(pts))
            for r, row in enumerate(pts):
                chk = QtWidgets.QTableWidgetItem()
                chk.setData(QtCore.Qt.ItemDataRole.UserRole, row['item_id'])
                chk.setFlags(Flag.ItemIsUserCheckable | Flag.ItemIsEnabled
                             if row['ok'] else Flag.ItemIsUserCheckable)
                chk.setCheckState(QtCore.Qt.CheckState.Checked
                                  if (row['ok'] and row['item_id'] in inc)
                                  else QtCore.Qt.CheckState.Unchecked)
                if row.get('reason') == 'unbuilt':
                    # The inputs the roster finds missing; a run only its loaded data
                    # keeps from building has none, and keeps the general sentence.
                    missing = row['missing']
                    chk.setToolTip(f'Needs:\n{_needs_lines(missing)}' if missing else
                                   'Parameters not confirmed yet — set them '
                                   'in the Data tab and press Update.')
                elif row.get('reason') == 'other_fraction':
                    chk.setToolTip('A different Mw fraction than the focused '
                                   'measurement — one map never mixes '
                                   'fractions.')
                else:
                    chk.setToolTip('Ticked runs are extracted when you Run.')
                t.setItem(r, 0, chk)
                ang = '—' if row['angle_deg'] is None else f"{row['angle_deg']:g}"
                conc = ('—' if row['concentration_g_per_mL'] is None
                        else f"{row['concentration_g_per_mL'] * dfc:g}")
                for c, v in enumerate([ang, conc], start=1):
                    it = QtWidgets.QTableWidgetItem(v)
                    it.setFlags(Flag.ItemIsEnabled)
                    t.setItem(r, c, it)
            t.resizeColumnsToContents()
        finally:
            self._suppress_table = False

    def _on_table_changed(self, item) -> None:
        if self._suppress_table or item.column() != 0:
            return
        sid = self._sample_id()
        if sid is None:
            return
        iid = item.data(QtCore.Qt.ItemDataRole.UserRole)
        inc = self._included.setdefault(self._ckey(sid), set())
        if item.checkState() == QtCore.Qt.CheckState.Checked:
            inc.add(iid)
        else:
            inc.discard(iid)
        self.selectionChanged.emit()
        self._run_epoch += 1
        if self._map_result(sid) is not None:
            self.status.setText('Selection changed — press Run to refresh.')
        self._update_run_enabled(sid)

    @QtCore.Slot()
    def _select_all(self) -> None:
        sid = self._sample_id()
        if sid is None:
            return
        self._included[self._ckey(sid)] = {r['item_id'] for r in self._points.get(sid, [])
                                           if r['auto_tick']}
        self._refresh_table(sid)
        self.selectionChanged.emit()
        self._run_epoch += 1
        self._update_run_enabled(sid)

    @QtCore.Slot()
    def _select_none(self) -> None:
        sid = self._sample_id()
        if sid is None:
            return
        self._included[self._ckey(sid)] = set()
        self._refresh_table(sid)
        self.selectionChanged.emit()
        self._run_epoch += 1
        self._update_run_enabled(sid)

    def _update_run_enabled(self, sid: str) -> bool:
        """Gate Run on the chosen source, judged on each ticked run as the map extracts
        them (`Controller.dls_fit_availability`): on while at least one ticked run can
        use it, the source entries none of them can use grayed, and the status line
        naming each ticked run the source leaves out with what it lacks. Run still hands
        every ticked run to the extraction, whose notes record the ones it leaves out.
        Returns whether a Run can extract anything."""
        ws = self.controller.workspace.measurements
        order = {r['item_id']: n for n, r in enumerate(self._points.get(sid, []))}
        ticked = sorted((i for i in self._included.get(self._ckey(sid), set()) if i in ws),
                        key=lambda i: (order.get(i, len(order)), i))
        verdicts = {iid: self.controller.dls_fit_availability([iid]) for iid in ticked}
        self._gray_sources(list(verdicts.values()))
        key = ('mode_map', self.source_combo.currentData())
        can_run = any(v[key].ok for v in verdicts.values())
        self._gate_runnable = bool(self._runnable) and can_run
        self.run_button.setEnabled(self._gate_runnable)
        self._gate_lines = (
            _left_out_lines(self.controller, [(iid, v[key].missing)
                                              for iid, v in verdicts.items()
                                              if not v[key].ok])
            if ticked else ['Tick at least one measurement to run.'])
        self._show_gate(sid)
        return can_run

    def _gray_sources(self, verdicts) -> None:
        """Gray each source entry none of the ticked runs can use (`verdicts`, one
        `dls_fit_availability` read per ticked run), its tooltip naming what they lack;
        with no run ticked every entry is offered. A grayed entry stays selected."""
        role = QtCore.Qt.ItemDataRole.ToolTipRole
        for i in range(self.source_combo.count()):
            source = self.source_combo.itemData(i)
            key = ('mode_map', source)
            item = self.source_combo.model().item(i)
            tip = self._entry_tips.setdefault(source, item.data(role))
            usable = not verdicts or any(v[key].ok for v in verdicts)
            needs = _union_missing(key, [v[key].missing for v in verdicts])
            item.setEnabled(usable)
            item.setData(tip if usable else f'Needs:\n{_needs_lines(needs)}', role)

    def _show_gate(self, sid: str) -> None:
        """Write the Run gate onto the status line: its lines, then the stale mark when a
        map is shown whose inputs have moved. With no lines, the gate's own text is
        cleared, and only while the status line still shows it, leaving a shown map's
        stale mark."""
        ckey = self._ckey(sid)
        stale = (self._map_result(sid) is not None
                 and self._run_sig(ckey) != self._run_signature(ckey))
        if self._gate_lines:
            mark = _STALE_HINT if self._gate_runnable else _STALE_UNRUNNABLE
            self._gate_status = '\n'.join([*self._gate_lines, *([mark] if stale else [])])
            self.status.setText(self._gate_status)
            return
        if self._gate_status and self.status.text() == self._gate_status:
            self.status.setText(_STALE_HINT if stale else '')
        self._gate_status = ''

    # ---- polarization facet ----
    def _refresh_facets(self, sid: str) -> None:
        """One state at a time, no 'all' — the facet IS the partition. The
        state set comes from the controller's public query (invariant 5:
        the GUI never re-derives the classification itself)."""
        ordered = (self.controller.mode_map_states(sid, self._fraction())
                   or [None])
        current = self.facet_combo.currentData()
        self.facet_combo.blockSignals(True)
        self.facet_combo.clear()
        for s in ordered:
            self.facet_combo.addItem(self._STATE_LABELS[s], s)
        if current in ordered:
            self.facet_combo.setCurrentIndex(ordered.index(current))
        self.facet_combo.setEnabled(len(ordered) > 1)
        self.facet_combo.blockSignals(False)

    def _facet_state(self):
        return self.facet_combo.currentData()

    def _on_facet_changed(self, _i) -> None:
        self._draw()

    # ---- families panel ----
    def _store(self, sid: str):
        return self.controller.mode_picks_for(sid, self._fraction())

    def _family_colour(self, index: int):
        colours = _plot_cycle()
        return colours[index % len(colours)]

    def _family_marker(self, index: int):
        markers = _marker_cycle()
        return markers[index % len(markers)]

    def _refresh_families(self, sid: str) -> None:
        store = self._store(sid)
        status = self.controller.mode_family_status(sid, self._fraction())
        sel_label = self._active_family_label()
        self._suppress_family = True
        try:
            self.family_list.clear()
            for i, fam in enumerate(store.families):
                n_picks, n_matched = status.get(fam.label, (0, 0))
                role = f'  [{fam.role}]' if fam.role else ''
                # A fully-orphaned family (picks, zero matches, a map present)
                # says so HERE -- with no fit there is no result to note on,
                # so the panel line is the disclosure surface.
                orphan = (' — matches nothing (re-run or re-pick)'
                          if (n_picks and not n_matched and sid
                              and self._map_result(sid) is not None) else '')
                it = QtWidgets.QListWidgetItem(
                    f'{fam.label}{role}  — {n_matched}/{n_picks} pick(s)'
                    f'{orphan}')
                it.setData(QtCore.Qt.ItemDataRole.UserRole, fam.label)
                it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
                it.setForeground(QtGui.QColor(self._family_colour(i)))
                self.family_list.addItem(it)
            if store.families:
                # By label, never by row: a rebuild after a sample / fraction switch
                # would otherwise select whichever family sits at the old ordinal
                # (the `_ModeResolvedTab._refresh` rule: preserve by id, never row).
                rows = [i for i, fam in enumerate(store.families) if fam.label == sel_label]
                self.family_list.setCurrentRow(rows[0] if rows else 0)
        finally:
            self._suppress_family = False
        self._sync_role_combo(sid)

    def _active_family_label(self) -> Optional[str]:
        it = self.family_list.currentItem()
        return it.data(QtCore.Qt.ItemDataRole.UserRole) if it else None

    def _sync_role_combo(self, sid: str) -> None:
        label = self._active_family_label()
        fam = self._store(sid).family(label) if label else None
        self.role_combo.blockSignals(True)
        self.role_combo.setCurrentIndex(
            {'': 0, 'fast': 1, 'slow': 2}.get(fam.role if fam else '', 0))
        self.role_combo.setEnabled(fam is not None)
        self.role_combo.blockSignals(False)

    @QtCore.Slot()
    def _on_add_family(self) -> None:
        sid = self._sample_id()
        if sid is None:
            return
        self.controller.add_mode_family(sid, self._fraction())
        self._refresh_families(sid)
        self.family_list.setCurrentRow(self.family_list.count() - 1)

    @QtCore.Slot()
    def _on_del_family(self) -> None:
        sid = self._sample_id()
        label = self._active_family_label()
        if sid is None or label is None:
            return
        self.controller.remove_mode_family(sid, self._fraction(), label)
        self._refresh_families(sid)
        self._fill_results(self._selected_family_fit(sid))
        self._draw()

    def _on_family_selected(self, _row: int) -> None:
        if self._suppress_family:
            return
        sid = self._sample_id()
        if sid is None:
            return
        self._sync_role_combo(sid)
        self._fill_results(self._selected_family_fit(sid))
        self._draw()

    def _on_family_renamed(self, item) -> None:
        if self._suppress_family:
            return
        sid = self._sample_id()
        old = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if sid is None or old is None:
            return
        # The edited text is the whole display line; strip the FIXED suffix
        # (role chip + pick counts) with a regex anchored at the END, so a
        # label that itself contains an em dash or brackets survives the
        # round-trip intact (never split on a delimiter the user's own text can contain).
        new = re.sub(r'(\s*\[(?:fast|slow)\])?\s+—\s+\d+/\d+ pick\(s\)'
                     r'( — matches nothing \(re-run or re-pick\))?$', '',
                     item.text()).strip()
        if not new or new == old:
            self._refresh_families(sid)
            return
        try:
            self.controller.rename_mode_family(sid, self._fraction(), old, new)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, 'Rename', str(exc))
        else:
            # The item still keys the OLD label; re-key it before `_refresh_families`
            # reads the selection back by label, or the rebuild lands on row 0 (round 1
            # f2). Suppressed: setData fires itemChanged into this very handler.
            self._suppress_family = True
            try:
                item.setData(QtCore.Qt.ItemDataRole.UserRole, new)
            finally:
                self._suppress_family = False
        self.controller.run_mode_family_fits(sid, self._fraction())
        self._refresh_families(sid)
        self._draw()

    def _on_role_changed(self, _i) -> None:
        sid = self._sample_id()
        label = self._active_family_label()
        if sid is None or label is None:
            return
        self.controller.set_mode_family_role(
            sid, self._fraction(), label, self.role_combo.currentData())
        self._refresh_families(sid)
        self._draw()

    def _selected_family_fit(self, sid: str):
        label = self._active_family_label()
        return self._family_fits(sid).get(label) if label else None

    # ---- run (worker-dispatched; the _recompute shape) ----
    @QtCore.Slot()
    def _on_run(self) -> None:
        if not self._runnable or self.item_id is None:
            return
        sid = self._sample_id()
        if sid is None:
            QtWidgets.QMessageBox.warning(
                self, 'No sample',
                'This measurement is not assigned to a sample.')
            return
        if self._update_run_enabled(sid):
            self._recompute(sid)

    def _recompute(self, sid: str, fraction=_FOCUSED) -> None:
        frac = self._fraction() if fraction is _FOCUSED else fraction
        ckey = (sid, frac)
        if self.source_combo.currentData() == 'contin':
            problem = self.fit_fields.alpha_problem()
            if problem:                       # never run an α the engine refuses
                self.status.setText(problem)
                return
        include_ids = set(self._included.get(ckey, set()))
        source = self.source_combo.currentData()
        # Read from the widgets HERE, on the main thread; the thunk below runs on the
        # worker and touches only the controller.
        inversion_kw = self._inversion_kwargs(source)
        region = self.region_box.region_for_run(self._use_region.get(ckey, False))
        sig = self._run_signature(ckey)
        # A deferred Run of a target no longer shown is extracted but never painted here.
        on_screen = ckey == self._last_ckey

        def done(res, token) -> None:
            self.controller.record_shown_run(('mode_map', *ckey), res, sig)
            self.run_button.setEnabled(self._gate_runnable)   # the worker re-enabled it
            if not self._run_slot.current(token):
                return
            self.export_button.setEnabled(True)
            self.record_set_button.setEnabled(True)
            self._refresh_facets(sid)
            self._refresh_families(sid)
            self._fill_results(self._selected_family_fit(sid))
            self._draw()
            self.status.clear()
            # The fields stay editable while the run is in flight, so compare the key the
            # run was dispatched under against the live one rather than assume a match.
            self._on_fit_input_moved()

        def fail(exc: BaseException, token) -> None:
            QtWidgets.QMessageBox.critical(
                self, 'Analysis failed',
                f'Could not build the mode map for sample {sid}.\n\n'
                f'{humanize_field_names(str(exc))}')
            self.run_button.setEnabled(self._gate_runnable)   # the worker re-enabled it
            if self._run_slot.current(token):  # the `done` guard: never paint a late
                self.status.setText('Analysis failed — see dialog.')   # failure elsewhere

        if self._run_slot.submit(
                lambda: self.controller.run_mode_map(
                    sid, fraction=frac, source=source, include_ids=include_ids,
                    region=region, **inversion_kw),
                done, fail, target=ckey, queue=True, busy_widgets=(self.run_button,),
                notice=lambda: self.status.setText(BUSY_NOTICE),
                description='Mode map extraction', paint=on_screen) and on_screen:
            self.status.setText('Extracting modes in the background…')

    def _deferred_run_available(self, target: tuple) -> bool:
        """Whether a Run deferred on `target`, a (sample, fraction), may still run: the
        chosen source is available on at least one of that target's ticked runs."""
        sid, _frac = target
        if not self._runnable or sid not in self.controller.workspace.samples:
            return False
        ws = self.controller.workspace.measurements
        key = ('mode_map', self.source_combo.currentData())
        return any(self.controller.dls_fit_availability([iid])[key].ok
                   for iid in self._included.get(target, set()) if iid in ws)

    # ---- click-to-pick (the SLS mask-editor pattern) ----
    def _point_coords(self, sid: str):
        """(angle, gamma, polarization_state, x_disp, y_disp) for every point
        of the CURRENT facet — the single source of both the click hit-test
        targets and the overlay positions (the D5 discipline). `gamma` is the
        PICK KEY, the node (`ModePoint.gamma_s_inv`); `y_disp` is where the point
        is plotted, its position (`Controller.mode_point_gamma`, a distribution lobe's
        geometric mean), so a click on a plotted point records the node."""
        res = self._map_result(sid)
        if res is None:
            return []
        state = self._facet_state()
        xf = _disp_factor('scattering_q2')
        yf = _disp_factor('decay_rate')
        return [(p.angle_deg, p.gamma_s_inv, p.polarization_state,
                 p.q2_m2 * xf, self.controller.mode_point_gamma(p) * yf)
                for p in res.points if p.polarization_state == state]

    def _on_canvas_click(self, event) -> None:
        sid = self._sample_id()
        if (sid is None or self._map_result(sid) is None
                or self.view_combo.currentData() != 'map'):
            return
        if event.inaxes is not self.ax or event.button != 1:
            return
        if getattr(self._nav, 'mode', ''):
            return                                    # pan/zoom active
        if runner().is_busy:
            self.status.setText(BUSY_NOTICE)
            return
        label = self._active_family_label()
        if label is None:
            self.status.setText('Add (or select) a family first — clicks '
                                'assign points to the selected family.')
            return
        coords = self._point_coords(sid)
        if not coords:
            return
        pts_disp = self.ax.transData.transform(
            [(x, y) for (_a, _g, _s, x, y) in coords])
        d = np.hypot(pts_disp[:, 0] - event.x, pts_disp[:, 1] - event.y)
        i = int(np.argmin(d))
        if d[i] > self._PICK_RADIUS_PX:
            return
        angle, gamma, state = coords[i][0], coords[i][1], coords[i][2]
        try:
            what = self.controller.toggle_mode_pick(
                sid, self._fraction(), label, angle, gamma, state)
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        try:
            self.controller.run_mode_family_fits(sid, self._fraction())
        except ValueError as exc:
            self.status.setText(str(exc))
        verb = {'added': 'added to', 'removed': 'removed from',
                'moved': 'moved to'}[what]
        self.status.setText(f'Point at {angle:g}° {verb} “{label}”.')
        self._refresh_families(sid)
        self._fill_results(self._selected_family_fit(sid))
        self._draw()

    # ---- results + drawing ----
    def _row_about(self) -> Dict[str, Tuple[Tuple[str, ...], bool]]:
        """What each Results row is ABOUT, keyed by its `_result_spec` label (never by
        position): the attributes a note or a qualifier must name to reach that row --
        its value and its ± -- and whether the whole fit's do too, the size row only
        (apparent, mode-resolved: invariant 7)."""
        du = _disp_unit('diffusion')
        return {f'D ({du})': (('d_m2_s', 'd_se'), False),
                'R<sub>h</sub> (nm)': (('rh_nm', 'rh_se'), True),
                'R²': (('r_squared',), False),
                'Through origin?': (('is_diffusive',), False),
                f'a ({du})': (('a_m2_s', 'a_se'), False),
                'b (m⁴/s)': (('b_m4_s', 'b_se'), False)}

    def _show_notes(self, sid, fit, rep=None) -> None:
        """Feed the notes block the map's, the selected family fit's and, in the amplitude
        view, the amplitude report's notes, as ONE source so a note relayed from the map's
        runs keeps the angles it names. Their records leave the tab but for the fallback
        weight's plain line (`gui/notes_block.py::SHOWN_RECORDS`). Any other qualifier the
        map relays keeps the neutral line below, but the residual weighting, which is a word
        on the family's Results rows (`_fill_results`)."""
        res = self._map_result(sid) if sid else None
        sources = [src for src in (res, fit, rep) if src is not None]
        self.notes_block.show_notes(tuple(n for src in sources for n in find_notes(src)))
        set_flag(self.notes_label, '\n'.join(dict.fromkeys(
            render(n, counted=True) for src in sources
            for n in find_notes(src, kind='qualifier')
            if not n.code.startswith(WEIGHTING_PREFIX))),
            problem=False)

    def _fill_results(self, fit) -> None:
        sid = self._sample_id()
        # The block first, so each marker's letter is its line's.
        self._show_notes(sid, fit)
        label = self._active_family_label()
        fam = self._store(sid).family(label) if (sid and label) else None
        about = self._row_about()
        # A VH family's slope carries rotation (the `vh_family` qualifier), so every row
        # read from the slope says so; the polarization is the family's, which no field of
        # the fit records.
        is_vh = fit is not None and fam is not None and fam.polarization_state == 'VH'
        # The fit records no weighting of its own: the title names the ones its picked
        # points ran under, read off those points, as the pooled lines read theirs.
        _title_weightings(
            self.results_label, _RESULTS_TITLE,
            self.controller.mode_family_weightings(sid, self._fraction(), label)
            if fit is not None and fam else ())
        for r, (row, fn) in enumerate(self._result_spec()):
            try:
                txt = fn(fit) if fit is not None else '—'
            except Exception:  # noqa: BLE001 — cell shows a dash
                txt = '—'
            attrs, whole = about.get(row, ((), False))
            vh = ((Note('vh_family'),) if is_vh and set(attrs) & _SLOPE_ATTRS else ())
            head = self.result_table.item(r, 0)
            if head is not None:
                _label_row(head, row, fit, attrs, whole=whole, extra=vh)
            it = self.result_table.item(r, 1)
            if it is not None:
                it.setText(txt)
                if fit is None or not self.notes_block.mark_item(it, fit, attrs,
                                                                 whole=whole):
                    it.setToolTip(txt)

    def _clear_results(self) -> None:
        _title_weightings(self.results_label, _RESULTS_TITLE, ())
        for r, (row, _fn) in enumerate(self._result_spec()):
            head, it = self.result_table.item(r, 0), self.result_table.item(r, 1)
            if head is not None:
                _label_row(head, row, None, (), whole=False)
            if it is not None:
                it.setText('—')
                it.setToolTip('')
        self.flag_label.clear()
        self.notes_block.clear()
        set_flag(self.notes_label, '', problem=False)

    def _draw(self) -> None:
        sid = self._sample_id()
        if sid is None or self._map_result(sid) is None:
            return
        if self._placeholder is not None:
            self._placeholder.remove()
            self._placeholder = None
        self.ax.clear()
        self.ax.set_visible(True)
        set_plot_export_name(self.canvas, sample_suggested_name(
            self.controller, sid, 'mode_map'))
        rep = None
        if self.view_combo.currentData() == 'amplitude':
            rep = self._draw_amplitude(sid)
        else:
            self._draw_map(sid)
        self._show_notes(sid, self._selected_family_fit(sid), rep)
        self.canvas.draw_idle()
        self.axis_bar.attach(self.ax)

    def _draw_map(self, sid: str) -> None:
        res = self._map_result(sid)
        state = self._facet_state()
        facet_pts = [p for p in res.points if p.polarization_state == state]
        plot_mode_map(facet_pts, ax=self.ax)
        store = self._store(sid)
        fits = self._family_fits(sid)
        xf = _disp_factor('scattering_q2')
        yf = _disp_factor('decay_rate')
        for i, fam in enumerate(store.families):
            # State EQUALITY on a BOUND family, exactly as the matcher and the
            # export label it (the `is not None` wildcard drew an unpolarized
            # family's fit on the VV/VH facets -- the display sibling of the
            # matcher's own wildcard defect).
            if not fam.bound or fam.polarization_state != state:
                continue
            fit = fits.get(fam.label)
            colour = self._family_colour(i)
            marker = self._family_marker(i)
            if fit is not None:
                plot_mode_family_fit(fit, ax=self.ax, color=colour,
                                     marker=marker, label=fam.label)
            else:
                # Matched on the node (the pick key), drawn at the plotted position.
                picked = [(p.q2_m2 * xf, self.controller.mode_point_gamma(p) * yf)
                          for p in facet_pts
                          if fam.has_pick(p.angle_deg, p.gamma_s_inv)]
                if picked:
                    xs, ys = zip(*picked, strict=True)
                    self.ax.scatter(xs, ys, s=44, color=colour, marker=marker,
                                    zorder=3, label=f'{fam.label} (unfitted)')
        reissue_legend_preserving_title(self.ax, fontsize=9)
        self.ax.set_title(f'Mode map — {self._STATE_LABELS[state]}')

    def _draw_amplitude(self, sid: str):
        """Draw the amplitude report and return it (None when it cannot be built)."""
        try:
            rep = self.controller.mode_amplitude_report(sid, self._fraction())
        except ValueError as exc:
            self._clear(str(exc))
            return None
        plot_amplitude_vs_q(rep, ax=self.ax)
        self.ax.set_title('Amplitude report — A_f/A_s vs sin²(θ/2)')
        return rep

    @QtCore.Slot()
    def _on_export(self) -> None:
        sid = self._sample_id()
        if sid is None or self._map_result(sid) is None:
            return
        frac = self._fraction()
        name = sample_suggested_name(self.controller, sid, 'mode_map')
        status = export_to_csv(
            self, self.controller, name,
            lambda p: self.controller.export_mode_map(sid, frac, p))
        if status:
            self.status.setText(status)

    # ---- recorded mode sets + manual integration bounds ----
    def _on_record_set(self) -> None:
        sid = self._sample_id()
        if sid is None:
            return
        frac = self._fraction()
        ready = self.controller.mode_set_readiness(sid, frac)
        if not ready['ok']:
            self.status.setText(_RECORD_NOT_READY.get(
                ready['reason_code'], 'The mode set cannot be recorded yet.'))
            return
        n = len(self.controller.mode_sets_for(sid, frac))
        label, ok = QtWidgets.QInputDialog.getText(
            self, 'Record mode set',
            'Label for the recorded set (a snapshot — later re-fits never '
            'change it):', text=f'Mode set {n + 1}')
        if not ok:
            return
        try:
            rset = self.controller.record_mode_set(
                sid, frac, label=label.strip() or None)
        except ValueError as exc:
            self.status.setText(humanize_field_names(str(exc)))
            return
        n_pts = sum(len(m.points) for m in rset.modes)
        msg = ('Recorded mode set “{0}” ({1} families, {2} points). '
               'Partition it on the SLS tab\'s Mode-Resolved sub-tab.'.format(
                   rset.label, len(rset.modes), n_pts))
        self.status.setText(msg)

    def _on_bounds(self) -> None:
        sid = self._sample_id()
        if sid is None:
            return
        frac = self._fraction()
        rows = [r for r in self._points.get(sid, []) if r['ok']]
        if not rows:
            return
        stored = self.controller.mode_bounds_for(sid, frac)
        auto = self.controller.mode_set_bounds_defaults(sid, frac)
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle('Manual integration bounds')
        lay = QtWidgets.QVBoxLayout(dlg)
        intro = QtWidgets.QLabel(
            'Rh intervals per run, as lo:hi pairs separated by “;” (e.g. '
            '5:60; 61:1000). Blank = the automatic lobe split. Bounds apply '
            'to the CONTIN/NNLS sources at the next Run.\n'
            'A manual split of overlapping peaks violates the '
            'wide-separation validity condition — the results will say so. '
            'For genuinely overlapping modes the double-exponential\'s model '
            'amplitudes are the better route.')
        intro.setWordWrap(True)
        lay.addWidget(intro)
        table = QtWidgets.QTableWidget(len(rows), 2)
        table.setHorizontalHeaderLabels(['Angle (°)', 'Bounds (nm)'])
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(True)
        for i, row in enumerate(rows):
            ang = ('—' if row['angle_deg'] is None
                   else f"{row['angle_deg']:g}")
            it = QtWidgets.QTableWidgetItem(ang)
            it.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
            table.setItem(i, 0, it)
            cur = stored.get(row['item_id'], [])
            text = '; '.join(f'{lo:g}:{hi:g}' for lo, hi in cur)
            cell = QtWidgets.QTableWidgetItem(text)
            lobes = auto.get(row['item_id'], [])
            if lobes:
                cell.setToolTip('Automatic lobes of the last run: '
                                + '; '.join(f'{lo:g}:{hi:g}'
                                            for lo, hi in lobes))
            table.setItem(i, 1, cell)
        table.resizeColumnsToContents()
        lay.addWidget(table, 1)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        # Phase 1: parse and validate EVERY row before anything is stored —
        # a later row's error must never leave earlier rows silently
        # committed.
        ops: list = []                       # (item_id, pairs-or-None)
        for i, row in enumerate(rows):
            text = (table.item(i, 1).text() or '').strip()
            if not text:
                ops.append((row['item_id'], None))
                continue
            try:
                pairs = [tuple(float(x) for x in seg.split(':'))
                         for seg in text.split(';') if seg.strip()]
                if any(len(p) != 2 for p in pairs):
                    raise ValueError(
                        'each interval must be one lo:hi pair')
                self.controller.validate_mode_bounds(pairs)
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(
                    self, 'Integration bounds',
                    'Row {0}: {1}. Nothing was changed.'.format(i + 1, exc))
                return
            ops.append((row['item_id'], pairs or None))
        # Phase 2: apply the fully validated set.
        n_set = n_cleared = 0
        for item_id, pairs in ops:
            if pairs is None:
                if item_id in stored:
                    self.controller.clear_mode_bounds(sid, frac, item_id)
                    n_cleared += 1
            else:
                self.controller.set_mode_bounds(sid, frac, item_id, pairs)
                n_set += 1
        if n_set or n_cleared:
            self.status.setText(
                'Manual bounds set for {0} run(s), cleared for {1}. '
                'Run the map again (CONTIN/NNLS source) to apply '
                'them.'.format(n_set, n_cleared))

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self.item_id is not None:
            self.set_measurement(self.item_id, self._runnable)

    def _clear(self, message: str) -> None:
        if self._placeholder is not None:
            self._placeholder.remove()
            self._placeholder = None
        self.ax.clear()
        self._placeholder = show_plot_placeholder(
            self.figure, self.canvas, message, axes=(self.ax,))
        self.axis_bar.attach(None)
