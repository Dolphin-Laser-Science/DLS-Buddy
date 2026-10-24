"""
The per-sample DLS analysis sub-tab behind Γ vs q² (over the sample's angles) and D vs c
(over its concentrations).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from plotting.plots import (plot_gamma_q2, plot_concentration_extrapolation,
                            display_factor as _disp_factor, display_unit as _disp_unit)
from app.controller import FIT_ELIGIBLE_QUALITIES
from core.data_models import IDENTITY_FORM_BY_FIELD
from core.notes import render
from core.naming import sample_display_name
from gui.plot_controls import (make_split_panels, make_vertical_plot_stack, pin_under_column,
                               set_plot_export_name)
from gui.export_helper import export_to_csv, sample_suggested_name
from gui.field_labels import humanize_field_names, missing_input_lines
from gui.help import section_header
from gui.notes_block import NotesBlock
from gui.plot_overlays import (draw_excluded_markers, reissue_legend_preserving_title,
                               show_plot_placeholder)
from gui.theme import ThemedLabel, color as theme_color
from gui.widgets import GroupTickBar, make_plot_pane, RichTextHeaderView
from gui.worker import BACKGROUND_RUN_TOOLTIP, BUSY_NOTICE, RunSlot
from core.formatting import format_fixed_sig, format_pm

from gui.dls.region import RegionCheckBox
from gui.dls.common import (_STALE_HINT, _STALE_UNRUNNABLE, _FOCUSED, _RESULTS_TITLE,
                            _ALL_WEIGHTINGS, _title_weightings, _GROUP_FIELD_CONC,
                            _GROUP_FIELD_ANGLE, _needs_lines)
from gui.dls.tables import _vtable, _label_row
from gui.dls.bases import _SampleRunTab


# "Same value?" for a group tick goes through THE shared field->form registry
# (core/data_models.IDENTITY_FORM_BY_FIELD), so the GUI's matching can never
# disagree with the distinctness gates for the same physical quantity.
_IDENTITY_FORMS = IDENTITY_FORM_BY_FIELD


class _SampleAnalysisTab(_SampleRunTab):
    """A single full-size sample-level plot: Gamma vs q^2, or D vs c."""

    selectionChanged = QtCore.Signal()           # ticked include-set changed (for the mirror)

    def __init__(self, controller, kind: str, parent=None, *, region=None) -> None:
        super().__init__(controller, parent, region=region)
        self.kind = kind                       # 'gamma_q2' or 'conc_extrap'
        # The tab's state is keyed (sample_id, fraction), NOT sample_id alone: a fit is
        # strictly per-fraction, so a sid-keyed one would redisplay one fraction's Rh
        # while the table grays those very rows under a different focus -- the
        # mis-attribution the literal-fraction partition exists to prevent, resurfacing
        # through the display. The fit itself, with the signature it ran under, lives in
        # the controller under (kind, sid, fraction) (`_shown_run`).
        self._points: Dict[str, list] = {}       # sample_id -> dls_sample_rows metadata
        self._suppress_table = False             # re-entrancy guard for checkbox edits
        self._last_ckey: Optional[tuple] = None  # (sample, fraction) shown: the epoch's key
        # The Run's epoch (bumped on a view or tick-set change) and its deferred
        # (sid, fraction), retried while the fit can still run on that target.
        self._run_slot = RunSlot(available=self._deferred_run_available,
                                 rerun=lambda target: self._recompute(*target),
                                 before_delivery=self.flush_view)
        self._placeholder = None                 # empty-state Text artist (no fit shown)
        self._build_ui()

    def _build_ui(self) -> None:
        splitter, left, right = make_split_panels(self, left_min_width=300,
                                                  sizes=(360, 740))

        # ---- controls section (the §3 column template: picker, then the region box
        # and Run, then results; Export and the status pinned under the column) ----
        controls = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(controls); cl.setContentsMargins(0, 0, 0, 0)
        self.region_box = RegionCheckBox(self._region)
        self.region_box.toggled.connect(self._on_region_toggled)
        cl.addWidget(self.region_box)
        self.run_button = QtWidgets.QPushButton('Run')
        self.run_button.setToolTip(BACKGROUND_RUN_TOOLTIP)
        self.run_button.clicked.connect(self._on_run)
        cl.addWidget(self.run_button)
        self.export_button = QtWidgets.QPushButton('Export CSV…')
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._on_export)
        self.status = QtWidgets.QLabel('')
        self.status.setWordWrap(True)

        # ---- per-measurement table section (tick to include) ----
        tsec = QtWidgets.QWidget()
        tl = QtWidgets.QVBoxLayout(tsec); tl.setContentsMargins(0, 0, 0, 0)
        # The Γ-q² fit varies ANGLE at a fixed concentration → group by concentration;
        # the D-c fit varies CONCENTRATION at a fixed angle → group by angle.
        # The group-by field is set with the GroupTickBar below.
        tl.addWidget(section_header(
            'Measurements (tick to include in the fit)',
            'Tick the measurements to fit, then press Run.',
            bullets=[
                # A conditional ELEMENT, never a conditional `bullets=` argument:
                # `extract_help.py::_bullets` matches on `ast.List`, so wrapping the list
                # in a concatenation makes it an `ast.BinOp` and the extractor silently
                # returns NOTHING, which drops every bullet from the generated companion
                # help doc while its sync test stays green.
                ('Fits Γ = D q² over the angles.' if self.kind == 'gamma_q2' else
                 'Extrapolates D to c → 0.'),
                'Each Γ is a fresh cumulant fit on the Settings values.',
                '<b>Run</b> needs ≥ 2 distinct angles or concentrations, one polarization '
                'only; the line under the column says what is missing.',
                'Grayed rows can\'t be ticked; hover a row for why.',
                'A replicate average beside its own sources starts unticked: it is not '
                'independent of them.',
                'A point with no usable decay rate is dropped, marked by a dotted line.',
                '<b>Click</b> a ⚠ line or lettered value for what to do.',
                'More: User-Manual §8.4, §8.6.']))
        # Empty/ineligible-state hint + NaN-temperature sibling note.
        self.table_note = ThemedLabel('', role='hint', size=10, hide_when_empty=True)
        self.table_note.setWordWrap(True); tl.addWidget(self.table_note)
        self.temp_note = ThemedLabel('', role='hint', size=10, hide_when_empty=True)
        self.temp_note.setWordWrap(True); tl.addWidget(self.temp_note)
        # Selection helpers: Select all / none + "tick all at X".
        helper = QtWidgets.QWidget()
        hl = QtWidgets.QHBoxLayout(helper); hl.setContentsMargins(0, 0, 0, 0)
        self.select_all_btn = QtWidgets.QPushButton('Select all')
        self.select_all_btn.setToolTip(
            'Tick every eligible measurement of this sample, except a replicate average '
            'next to its own source runs.')
        self.select_all_btn.clicked.connect(self._select_all)
        self.select_none_btn = QtWidgets.QPushButton('Select none')
        self.select_none_btn.setToolTip('Untick every measurement.')
        self.select_none_btn.clicked.connect(self._select_none)
        hl.addWidget(self.select_all_btn)
        hl.addWidget(self.select_none_btn)
        hl.addStretch(1)
        tl.addWidget(helper)
        # "Tick all at X" bulk selector via the shared GroupTickBar, on its own row so it
        # fits the narrow column: concentration in mg/mL for Γ-q² (angle varies), angle in
        # ° for D-c (concentration varies). The bar emits (key, value); we resolve this
        # sample's matching rows and tick them.
        self._group_field_spec = (_GROUP_FIELD_CONC if self.kind == 'gamma_q2'
                                  else _GROUP_FIELD_ANGLE)
        self.group_bar = GroupTickBar((self._group_field_spec,))
        self.group_bar.tickRequested.connect(self._on_group_tick)
        tl.addWidget(self.group_bar)
        self.table = QtWidgets.QTableWidget(0, 0)
        self.table.setHorizontalHeader(RichTextHeaderView(self.table))   # D_app header → rich text
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.itemChanged.connect(self._on_table_changed)
        tl.addWidget(self.table, 1)

        # ---- results section ----
        rsec = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(rsec); rl.setContentsMargins(0, 0, 0, 0)
        # The title names a residual weighting other than the default (`_title_weightings`),
        # so it wraps rather than widen the pane.
        self.results_label = QtWidgets.QLabel(_RESULTS_TITLE)
        self.results_label.setWordWrap(True)
        rl.addWidget(self.results_label)
        self.result_table = _vtable([lbl for lbl, _ in self._result_spec()])
        # A fixed row count, so the table's floor IS its content height (R3.2): the
        # splitter then cannot squeeze the last rows under the pane edge, whatever
        # the two wrap labels below it take (measured at 760 px: a 90-px pane floor
        # left the 4th and 5th rows cut).
        self.result_table.setMinimumHeight(self.result_table.maximumHeight())
        rl.addWidget(self.result_table)
        # The pooled fit's notes: one line per failed precondition, a marker on each value
        # it is about; pinned under the column right below the Results panel (R8.6). Its
        # records leave the tab but for the fallback weight's plain line
        # (`gui/notes_block.py::SHOWN_RECORDS`); its qualifiers are the rows' words.
        self.notes_block = NotesBlock(what='points')
        self.notes_block.watch(self.result_table)
        rl.addStretch(1)
        # Operational flags only, pinned under the column with the status (R8.6).
        self.flag_label = ThemedLabel('', role='error', bold=True)
        self.flag_label.setWordWrap(True)

        # Picker first, then the region box and Run, then results (the §3 column
        # template). The controls pane is two rows, so it takes no share of extra height.
        vstack = make_vertical_plot_stack(
            [tsec, controls, rsec], sizes=[260, 70, 250],
            min_heights=[90, controls.sizeHint().height(), 150],  # rsec: the 5-row results table (R3.2)
            stretch=(1, 0, 1))
        left.addWidget(vstack, 1)
        pin_under_column(splitter, self.export_button, self.status,
                         self.notes_block, self.flag_label)

        self.figure, self.canvas, self._nav, self.axis_bar = make_plot_pane(
            self, self.controller, (5.2, 4.3), constrained_layout=True)
        self.ax = self.figure.add_subplot(111)
        right.addWidget(self._nav)
        right.addWidget(self.canvas, 1)
        right.addWidget(self.axis_bar)

    def _result_spec(self):
        """(label, value-fn) per result row, for the results _vtable."""
        df, du = _disp_factor('diffusion'), _disp_unit('diffusion')

        def pm(v, se):
            return format_pm(v * df, (se * df) if se is not None else None)
        if self.kind == 'gamma_q2':
            return [
                (f'D ({du})', lambda r: pm(r.d_m2_s, r.d_se)),
                ('R<sub>h</sub> (nm)', lambda r: format_pm(r.rh_nm, r.rh_se)),
                ('R²', lambda r: format_fixed_sig(
                    r.r_squared, self.controller.settings.no_uncertainty_sig_figs)),
                ('Diffusive?', lambda r: '✓' if r.is_diffusive else '⚠ no'),
                ('angles fitted', lambda r: str(int(np.isfinite(
                    np.asarray(r.gamma_s_inv, dtype=float)).sum()))),
            ]
        return [
            (f'D₀ ({du})', lambda r: pm(r.d0_m2_s, r.d0_se)),
            ('R<sub>h,0</sub> (nm)', lambda r: format_pm(r.rh0_nm, r.rh0_se)),
            ('k<sub>D</sub> (mL/g)', lambda r: format_pm(r.kd_mL_per_g, r.kd_se)),
            ('R²', lambda r: format_fixed_sig(
                r.r_squared, self.controller.settings.no_uncertainty_sig_figs)),
            ('concs fitted', lambda r: str(r.n_concentrations)),
        ]

    #: A hidden tab keeps its focus at a sidebar click and repaints when it is shown
    #: (`_DeferredRepaint`); keep-last is the base's `focus_state`.
    defers_repaint = True

    def _focus_hook(self, sid: Optional[str]) -> None:
        """The state half's own part: the epoch's re-point and, on a runnable focus with
        a sample, its rows, the ticked set they merge into and the "tick all at X"
        values."""
        ckey = self._ckey(sid) if sid is not None else None
        # The epoch is keyed on (sample, fraction), never the sample alone: a
        # fraction switch within one sample is a re-point too, and a fit landing
        # after it must not paint the fraction no longer focused (the SLS tab's
        # fraction-change bump is the pattern).
        if ckey != self._last_ckey:
            self._run_epoch += 1     # re-pointed: drop any in-flight run's display
        self._last_ckey = ckey
        if not (self._runnable and sid is not None):
            return
        # Enumerate the sample's measurements as METADATA ONLY — no fitting on focus.
        # Γ/D appear only after an explicit Run.
        self._points[sid] = self.controller.dls_sample_rows(
            sid, self.kind, self._fraction())
        fresh = {r['item_id'] for r in self._points[sid] if r['ok']}
        # The ticked set is per (sample, fraction) like the fit stores: the rows are
        # per-fraction, so a sid-keyed set intersected with another fraction's rows
        # would empty on every fraction switch and never come back.
        prev = self._included.get(ckey)
        # Auto-tick the controller's default rows on first focus (ticking now only
        # selects — it computes nothing), so the common "use every point" case is one Run
        # away. `auto_tick`, not `ok`: a replicate average beside its own sources stays
        # tickable but is not ticked for the user. A kept tick is kept while the row is ok.
        auto = {r['item_id'] for r in self._points[sid] if r['auto_tick']}
        self._included[ckey] = (prev & fresh) if prev is not None else auto
        # Here, not in the repaint: a non-runnable focus only greys the bar, so the
        # values the last runnable focus wrote stay on show.
        self._populate_group_combo(sid)

    def repaint_view(self) -> None:
        """The repaint half: the view of the stored focus, its sample and fraction read
        from the controller now, right after the state half has run."""
        self.flag_label.clear()
        self.notes_block.clear()                        # drop stale result notes
        sid = self._sample_id()
        ckey = self._ckey(sid) if sid is not None else None
        self.region_box.set_checked_silently(self._use_region.get(ckey, False))
        if not (self._runnable and sid is not None):
            self.table.setRowCount(0)
            self.run_button.setEnabled(False)
            self.export_button.setEnabled(False)
            self._set_helpers_enabled(False)
            self.temp_note.clear()
            self.table_note.setText(
                'Select a DLS measurement in the sidebar to see this sample\'s '
                'measurements here.')
            self._clear_results()
            self._clear('Select a DLS measurement.')
            self.status.clear()
            return
        # The table shows this (sample, fraction)'s last-run per-point values (Γ/D +
        # graying, `_run_points_of`) — a run of another fraction stays in its own slot,
        # unseen here.
        self._refresh_table(sid)
        runnable_now = self._update_run_enabled(sid)
        self._set_helpers_enabled(True)
        run = self._shown_run(ckey)
        self.export_button.setEnabled(run is not None)
        # Empty-state / eligibility guidance.
        if not self._points[sid]:
            self.table_note.setText('This sample has no DLS measurements.')
        # A run with no concentration has confirmed parameters: its row says what it lacks.
        elif not any(r['ok'] or r['reason'] == 'no_concentration'
                     for r in self._points[sid]):
            self.table_note.setText(
                'No measurements have confirmed parameters yet — set them in the Data '
                'tab and press Update, then return here.')
        else:
            self.table_note.clear()
        n_pending = self.controller.dls_pending_temperature_siblings(sid)
        if n_pending:
            self.temp_note.setText(
                f'{n_pending} measurement(s) of this polymer/solvent have no confirmed '
                'temperature and are grouped separately — confirm it in the Data tab to '
                'include them here.')
        else:
            self.temp_note.clear()
        if run is not None:
            self._fill_results(run.result)
            self._draw(run.result)
            # With Run off, the gate's line already carries the mark without "press Run".
            if runnable_now and run.sig != self._run_signature(ckey):
                self.status.setText(_STALE_HINT)
        else:
            self._clear_results()
            self._clear('Tick measurements, then Run.')

    def _shown_run(self, ckey: tuple):
        """The fit of `ckey`, a (sample, fraction), this tab ran and draws
        (`Controller.shown_run`), else None."""
        return self.controller.shown_run((self.kind, *ckey))

    def _run_points_of(self, ckey: tuple) -> list:
        """The per-point values of the fit of `ckey` this tab draws -- every eligible
        measurement's Γ/D and quality, read off the run (`all_points`) -- else []."""
        run = self._shown_run(ckey)
        return list(getattr(run.result, 'all_points', None) or []) if run is not None else []

    def reshow(self) -> None:
        """Re-show this tab at its own focus after a Settings apply, whose clear may have
        taken the fit it shows, or when Export finds that fit gone. While the sidebar
        focuses another kind of measurement the tab keeps its last DLS sample, which the
        shell's redraw passes by."""
        self.set_measurement(self.item_id, self._runnable)

    def _run_signature(self, ckey: tuple) -> tuple:
        """Everything a Γ-q² / D-vs-c fit for (sample, fraction) reads beyond its raw
        data: the committed params of the TICKED measurements (their ids ride in the
        signature, so a tick change moves it) and the settings the engine seeds --
        skip_initial_channels, the cumulant method, the cumulant fit cutoff and the
        cumulant residual weighting (the Gamma-vs-q2 fit reads the user's values), the SE
        estimator (the ± is
        estimator-routed; the confirm in Settings promises a recompute). Stored with
        the cached fit at dispatch and compared on redisplay, so a cached fit is flagged
        stale exactly when a re-Run would read different inputs. The cumulant order joins
        them (every point fits at the Settings order), and so does the region choice --
        this (sample, fraction)'s box and, when ticked, the region's applied parts -- so
        toggling the box, or moving the region while it is ticked, flags a shown fit
        stale and an unticked one is left alone."""
        s = self.controller.settings
        return (self.controller.committed_signature(self._included.get(ckey, set())),
                s.skip_initial_channels, s.cumulant_method, s.cumulant_fit_cutoff,
                s.cumulant_weighting, s.cumulant_order, s.se_estimator,
                self.region_box.key(self._use_region.get(ckey, False)))

    # ---- the region choice ----
    def _on_region_input_moved(self) -> None:
        """Compare the shown fit's stored key against the live one -- stale when they
        differ, and the hint cleared when they agree again (a box ticked and unticked)."""
        sid = self._sample_id()
        if sid is None:
            return
        ckey = self._ckey(sid)
        run = self._shown_run(ckey)
        if run is None:
            return
        if not self._fit_availability(ckey).ok:
            self._update_run_enabled(sid)   # Run is off: its line carries the mark
            return
        if run.sig != self._run_signature(ckey):
            self.status.setText(_STALE_HINT)
        elif self.status.text() == _STALE_HINT:
            self.status.clear()

    # ---- per-measurement table (tick to include) ----
    def _refresh_table(self, sid: str) -> None:
        pts = self._points.get(sid, [])
        inc = self._included.get(self._ckey(sid), set())
        cu = _disp_unit('concentration')
        if self.kind == 'gamma_q2':
            headers = ['', 'Angle (°)', f'c ({cu})',
                       f'Γ ({_disp_unit("decay_rate")})',
                       f'D<sub>app</sub> ({_disp_unit("diffusion")})']
        else:
            headers = ['', f'c ({cu})', 'Angle (°)',
                       f'D<sub>app</sub> ({_disp_unit("diffusion")})']
        dfc, dfg, dfd = (_disp_factor('concentration'), _disp_factor('decay_rate'),
                         _disp_factor('diffusion'))
        # Per-point Γ/D + quality come from the last Run — '—'
        # until then, so nothing is computed on focus.
        run_by_id = {p['item_id']: p for p in self._run_points_of(self._ckey(sid))}
        Flag = QtCore.Qt.ItemFlag
        self._suppress_table = True
        try:
            t = self.table
            t.setColumnCount(len(headers))
            t.setHorizontalHeaderLabels(headers)
            t.setRowCount(len(pts))
            for r, row in enumerate(pts):
                chk = QtWidgets.QTableWidgetItem()
                chk.setData(QtCore.Qt.ItemDataRole.UserRole, row['item_id'])
                chk.setFlags(Flag.ItemIsUserCheckable | Flag.ItemIsEnabled
                             if row['ok'] else Flag.ItemIsUserCheckable)
                chk.setCheckState(QtCore.Qt.CheckState.Checked
                                  if (row['ok'] and row['item_id'] in inc)
                                  else QtCore.Qt.CheckState.Unchecked)
                chk.setToolTip(self._row_tooltip(row, run_by_id.get(row['item_id'])))
                t.setItem(r, 0, chk)
                ang = '—' if row['angle_deg'] is None else f"{row['angle_deg']:g}"
                conc = ('—' if row['concentration_g_per_mL'] is None
                        else f"{row['concentration_g_per_mL'] * dfc:g}")
                pt = run_by_id.get(row['item_id'])
                has_val = pt is not None and pt['quality'] in FIT_ELIGIBLE_QUALITIES
                sig = self.controller.settings.no_uncertainty_sig_figs   # per-angle Γ/D: ±-less
                gtxt = format_fixed_sig(pt['gamma_s_inv'] * dfg, sig) if has_val else '—'
                dtxt = format_fixed_sig(pt['d_app_m2_s'] * dfd, sig) if has_val else '—'
                vals = ([ang, conc, gtxt, dtxt] if self.kind == 'gamma_q2'
                        else [conc, ang, dtxt])
                for c, v in enumerate(vals, start=1):
                    it = QtWidgets.QTableWidgetItem(v)
                    it.setFlags(Flag.ItemIsEnabled)
                    t.setItem(r, c, it)
                self._tint_row(r, row['ok'] and row['item_id'] in inc)
            t.resizeColumnsToContents()
        finally:
            self._suppress_table = False

    def _row_tooltip(self, row: dict, pt: Optional[dict]) -> str:
        """Explain a row's eligibility / quality so the user knows why it is grayed or
        excluded."""
        reason = row.get('reason')
        if reason in ('unbuilt', 'no_concentration') and row.get('missing'):
            return f"Needs:\n{_needs_lines(row['missing'])}"
        if reason == 'unbuilt':
            return ('Parameters not confirmed yet — set them in the Data tab and press '
                    'Update to make this measurement eligible.')
        if reason == 'other_fraction':
            frac = row.get('mw_fraction')
            return (f'A different Mw fraction ({frac!r}) than the focused measurement — '
                    'a single fit must not mix fractions. Focus a measurement of this '
                    'fraction to include it.')
        if pt is not None:
            q = pt['quality']
            if q == 'fit_failed':
                why = pt.get('fit_error') or ''
                return ('Cumulant fit failed — excluded from the fit'
                        + (f': {humanize_field_names(why)}' if why else '.'))
            if q == 'nonpositive_q':
                return 'Non-physical scattering vector (q ≤ 0) — excluded from the fit.'
            if q in ('nonpositive_gamma', 'high_pdi', 'flagged'):
                # The point's own cumulant preconditions, in their registry clauses.
                notes = [render(n, counted=True) for n in pt.get('notes') or ()]
                if notes:
                    return '\n'.join(notes)
        return 'Eligible — ticked rows enter the fit when you press Run.'

    def _tint_row(self, r: int, on: bool) -> None:
        """Color a table row to match the shared idiom: ticked rows read
        `marker_selected` blue, like the sidebar mirror and the DLS picker."""
        col = theme_color(self.table, 'marker_selected') if on else QtGui.QBrush()
        for c in range(self.table.columnCount()):
            it = self.table.item(r, c)
            if it is not None:
                it.setForeground(col)

    def _on_table_changed(self, item) -> None:
        if self._suppress_table or item.column() != 0:
            return
        sid = self._sample_id()
        if sid is None:
            return
        iid = item.data(QtCore.Qt.ItemDataRole.UserRole)
        inc = self._included.setdefault(self._ckey(sid), set())
        checked = item.checkState() == QtCore.Qt.CheckState.Checked
        if checked:
            inc.add(iid)
        else:
            inc.discard(iid)
        self._suppress_table = True    # setForeground re-emits itemChanged — guard it
        self._tint_row(item.row(), checked)
        self._suppress_table = False
        self.selectionChanged.emit()  # repaint the sidebar mirror for the new subset
        self._run_epoch += 1         # ticked set changed: in-flight run is stale
        # Ticking only selects — it never refits or clears the plot.
        # A shown result just gets marked stale until the next Run.
        enough = self._update_run_enabled(sid)
        if self._shown_run(self._ckey(sid)) is not None and enough:
            self.status.setText('Selection changed — press Run to refresh.')

    # ---- selection helpers (Select all/none + tick-all-at-X) ----
    def _group_field(self) -> str:
        """The metadata key the 'tick all at X' selector groups by: concentration for
        the Γ-q² tab (angle varies), angle for the D-c tab (concentration varies)."""
        return self._group_field_spec[0]

    def _set_helpers_enabled(self, on: bool) -> None:
        self.select_all_btn.setEnabled(on)
        self.select_none_btn.setEnabled(on)
        self.group_bar.setEnabled(on)

    def _populate_group_combo(self, sid: str) -> None:
        """Fill the 'tick all at X' combo with the distinct values among ELIGIBLE rows
        (formatted with units by the GroupTickBar — mg/mL for concentration, ° for angle)."""
        field = self._group_field()
        vals = sorted({r[field] for r in self._points.get(sid, [])
                       if r['ok'] and r[field] is not None})
        self.group_bar.set_values(field, vals)

    def _apply_included(self, sid: str, ids: set) -> None:
        """Set the ticked include-set for a sample and refresh the dependent UI once
        (table, run-enabled, sidebar mirror) — the shared path for the bulk helpers."""
        self._included[self._ckey(sid)] = set(ids)
        self._refresh_table(sid)
        self.selectionChanged.emit()
        self._run_epoch += 1
        enough = self._update_run_enabled(sid)
        if self._shown_run(self._ckey(sid)) is not None and enough:
            self.status.setText('Selection changed — press Run to refresh.')

    @QtCore.Slot()
    def _select_all(self) -> None:
        sid = self._sample_id()
        if sid is None:
            return
        # The bulk helpers tick the controller's default rows (`auto_tick`), as first
        # focus does; an average beside its own sources is ticked by hand or not at all.
        self._apply_included(sid, {r['item_id'] for r in self._points.get(sid, [])
                                   if r['auto_tick']})

    @QtCore.Slot()
    def _select_none(self) -> None:
        sid = self._sample_id()
        if sid is None:
            return
        self._apply_included(sid, set())

    def _on_group_tick(self, field: str, target) -> None:
        """Add every default-ticked measurement sharing the chosen value to the ticked
        set (GroupTickBar.tickRequested handler)."""
        sid = self._sample_id()
        if sid is None or target is None:
            return
        form = _IDENTITY_FORMS[field]
        add = {r['item_id'] for r in self._points.get(sid, [])
               if r['auto_tick'] and r[field] is not None
               and form(r[field]) == form(target)}
        self._apply_included(sid, self._included.get(self._ckey(sid), set()) | add)

    def showEvent(self, event) -> None:
        """Re-populate from the current focus when the tab becomes visible, so
        switching to it shows content without a sidebar re-click."""
        super().showEvent(event)
        if self.item_id is not None:
            self.set_measurement(self.item_id, self._runnable)

    def _fit_availability(self, ckey: tuple):
        """The roster's verdict on this tab's fit over the ticked set of `ckey`, a
        (sample, fraction): the focused one or a deferred Run's target. Ticks of runs
        no longer in the workspace are not asked about."""
        known = self.controller.workspace.measurements
        ids = sorted(i for i in self._included.get(ckey, set()) if i in known)
        return self.controller.dls_fit_availability(ids)[self.kind]

    def _gate_run_button(self, sid: Optional[str]):
        """Enable Run only while the view shows a runnable sample whose ticked set the
        fit can take; returns the verdict, None when no sample is shown."""
        avail = None if sid is None else self._fit_availability(self._ckey(sid))
        self.run_button.setEnabled(bool(self._runnable) and avail is not None
                                   and avail.ok)
        return avail

    def _update_run_enabled(self, sid: str) -> bool:
        """Run follows the fit's availability over the ticked set; while it cannot run,
        the status line names what is missing (a fit on screen whose inputs changed
        keeps its out-of-date mark, without asking for Run), and otherwise it clears."""
        avail = self._gate_run_button(sid)
        if avail.ok:
            self.status.clear()
            return True
        name = 'Γ vs q²' if self.kind == 'gamma_q2' else 'D vs c'
        lines = [f'{name} needs:', *missing_input_lines(avail.missing, counted='ticked')]
        run = self._shown_run(self._ckey(sid))
        if run is not None and run.sig != self._run_signature(self._ckey(sid)):
            lines.append(_STALE_UNRUNNABLE)
        self.status.setText('\n'.join(lines))
        return False

    def _run_engine(self, sid: str, frac, include_ids, region=None):
        if self.kind == 'gamma_q2':
            return self.controller.run_gamma_q2(
                sid, fraction=frac, include_ids=include_ids, region=region)
        return self.controller.run_concentration_extrapolation(
            sid, fraction=frac, include_ids=include_ids, region=region)

    @QtCore.Slot()
    def _on_run(self) -> None:
        if not self._runnable or self.item_id is None:
            return
        sid = self._sample_id()
        if sid is None:
            QtWidgets.QMessageBox.warning(self, 'No sample',
                                          'This measurement is not assigned to a sample.')
            return
        if self._update_run_enabled(sid):
            self._recompute(sid)

    def _recompute(self, sid: str, fraction=_FOCUSED) -> None:
        """Fit over the ticked subset (on the worker thread — it builds + fits
        every included measurement) and refresh the plot + results table.
        `fraction` is the focused one, unless a deferred Run re-enters with the
        (sample, fraction) it was pressed on."""
        frac = self._fraction() if fraction is _FOCUSED else fraction
        # Key the stores by the RUN's (sample, fraction), captured now, never by
        # the focus at completion time — the focus may have moved while the
        # worker fitted this one.
        ckey = (sid, frac)
        include_ids = set(self._included.get(ckey, set()))
        # Read here, on the main thread: the region is a widget-owned value and a handle
        # may move while the worker fits.
        region = self.region_box.region_for_run(self._use_region.get(ckey, False))
        sig = self._run_signature(ckey)
        # A deferred Run whose target is no longer on screen: fit it (what the user
        # asked for), but never paint it over the view now shown.
        on_screen = ckey == self._last_ckey

        def done(res, token) -> None:
            # The fit lands in ITS OWN slot whatever is focused now (a refocus
            # shows it); only the display is gated on the view being the one this
            # run started from.
            self.controller.record_shown_run((self.kind, *ckey), res, sig)
            self._gate_run_button(self._sample_id())   # the worker re-enabled Run
            if not self._run_slot.current(token):
                return                      # view / tick set changed — stale VIEW
            # Fold the run's per-point values into the table + graying:
            # Γ/D now appear, and quality flags become hoverable.
            self._refresh_table(sid)
            self.export_button.setEnabled(True)
            self._fill_results(res)
            self._draw(res)
            self.status.clear()
            # The box and the region stay live while the run is in flight: compare the key
            # it ran under with the live one, as the Mode Map does on landing.
            self._on_region_input_moved()

        def fail(exc: BaseException, token) -> None:
            # The controller withdrew this run's previous result and row on the refusal
            # (`Controller._withdraw_on_refusal`), so what the tab drew of it goes too,
            # whatever is on screen; the view is redrawn empty whenever it shows this
            # run's (sample, fraction), a tick change since the Run included, as with no
            # fit. The status line keeps the `done` guard. The dialog names the run's own
            # sample.
            self.controller.forget_shown_run((self.kind, *ckey))
            self._gate_run_button(self._sample_id())   # the worker re-enabled Run
            sample = self.controller.workspace.samples.get(sid)
            name = sample_display_name(sample, frac) if sample is not None else sid
            QtWidgets.QMessageBox.critical(
                self, 'Analysis failed',
                f'Could not run this analysis on sample {name}.\n\n'
                f'{humanize_field_names(str(exc))}')
            if ckey == self._last_ckey:
                self._clear_results()
                self._clear('Tick measurements, then Run.')
                self.export_button.setEnabled(False)
                self._refresh_table(sid)
            if self._run_slot.current(token):
                self.status.setText('Analysis failed — see dialog.')

        description = ('Γ vs q² fit' if self.kind == 'gamma_q2'
                       else 'D vs c extrapolation')
        # A press while the worker is busy takes the pending slot and is retried at idle
        # against the (sample, fraction) it was pressed on, whatever is focused by then.
        if self._run_slot.submit(
                lambda: self._run_engine(sid, frac, include_ids, region),
                done, fail, target=ckey, queue=True, busy_widgets=(self.run_button,),
                notice=lambda: self.status.setText(BUSY_NOTICE),
                description=description, paint=on_screen) and on_screen:
            self.status.setText('Fitting in the background…')

    def _deferred_run_available(self, target: tuple) -> bool:
        """Whether a Run deferred on `target`, a (sample, fraction), may still run. The
        fit's availability is the precondition `_on_run` checks before dispatching: the
        retry reads the target's CURRENT ticked set, which a tick change while the worker
        was busy can leave one the fit refuses, whose ValueError would be a modal error."""
        sid, frac = target
        return (bool(self._runnable) and sid in self.controller.workspace.samples
                and self._fit_availability((sid, frac)).ok)

    def _row_about(self) -> Dict[str, Tuple[Tuple[str, ...], bool]]:
        """What each Results row is ABOUT, keyed by its `_result_spec` label (never by
        position): the result attributes a note or a qualifier must name to reach that
        row -- its value and its ± -- and whether the whole fit's do too, which only the
        size row takes (apparent, the cumulant average: invariant 7)."""
        du = _disp_unit('diffusion')
        if self.kind == 'gamma_q2':
            return {f'D ({du})': (('d_m2_s', 'd_se'), False),
                    'R<sub>h</sub> (nm)': (('rh_nm', 'rh_se'), True),
                    'R²': (('r_squared',), False),
                    'Diffusive?': (('is_diffusive',), False)}
        return {f'D₀ ({du})': (('d0_m2_s', 'd0_se'), False),
                'R<sub>h,0</sub> (nm)': (('rh0_nm', 'rh0_se'), True),
                'k<sub>D</sub> (mL/g)': (('kd_mL_per_g', 'kd_se'), False),
                'R²': (('r_squared',), False)}

    def _fill_results(self, res) -> None:
        # The block first, so each marker's letter is its line's.
        self.notes_block.show_notes(res)
        about = self._row_about()
        # The residual weighting each point's fit ran under, which the fit records beside
        # the token it asked for; the title names it, no row does.
        _title_weightings(self.results_label, _RESULTS_TITLE,
                          getattr(res, 'point_weightings', ()) or ())
        for r, (label, fn) in enumerate(self._result_spec()):
            try:
                txt = fn(res)
            except Exception:  # noqa: BLE001 — cell shows a dash
                txt = '—'
            attrs, whole = about.get(label, ((), False))
            head = self.result_table.item(r, 0)
            if head is not None:
                _label_row(head, label, res, attrs, whole=whole, exclude=_ALL_WEIGHTINGS)
            it = self.result_table.item(r, 1)
            if it is not None:
                it.setText(txt)
                if not self.notes_block.mark_item(it, res, attrs, whole=whole):
                    it.setToolTip(txt)

    def _clear_results(self) -> None:
        _title_weightings(self.results_label, _RESULTS_TITLE, ())
        for r, (label, _fn) in enumerate(self._result_spec()):
            head, it = self.result_table.item(r, 0), self.result_table.item(r, 1)
            if head is not None:
                _label_row(head, label, None, (), whole=False)
            if it is not None:
                it.setText('—')
                it.setToolTip('')
        self.flag_label.clear()
        self.notes_block.clear()

    @QtCore.Slot()
    def _on_export(self) -> None:
        sid = self._sample_id()
        run = None if sid is None else self._shown_run(self._ckey(sid))
        if run is None:
            # A regroup or removal made while another kind of measurement was focused
            # took the fit without redrawing this tab: draw its empty state now.
            self.reshow()
            return
        res = run.result
        if self.kind == 'gamma_q2':
            # 'D_vs_c' below: kept over the exporter's longer 'concentration_
            # extrapolation' stem — it names the plotted axes (deliberate).
            name = sample_suggested_name(self.controller, sid, 'gamma_q2')
            fn = self.controller.export_gamma_q2
        else:
            name = sample_suggested_name(self.controller, sid, 'D_vs_c')
            fn = self.controller.export_concentration_extrapolation
        status = export_to_csv(self, self.controller, name, lambda p: fn(res, p))
        if status:
            self.status.setText(status)

    def _draw(self, res) -> None:
        if self._placeholder is not None:        # leaving the empty-state: drop the message
            self._placeholder.remove()
            self._placeholder = None
        self.ax.clear()
        self.ax.set_visible(True)                # restore after an empty _clear() hid it
        self.axis_bar.attach(self.ax)            # re-bind the axis controls (detached while empty)
        sid = self._sample_id()
        # Plot saves suggest the same stem as the CSV export for this analysis.
        set_plot_export_name(self.canvas, sample_suggested_name(
            self.controller, sid,
            'gamma_q2' if self.kind == 'gamma_q2' else 'D_vs_c'))
        # No sentence of the tab's own about the fit: its failed preconditions are the
        # notes block's lines and markers, its qualifiers (apparent, the cumulant average,
        # the statistical ±) the Results rows' words (`_fill_results`). A line that is
        # not purely diffusive shows on the `Diffusive?` cell.
        if self.kind == 'gamma_q2':
            plot_gamma_q2(res, ax=self.ax)
            self.ax.set_title('Γ vs q² (multi-angle)')
        else:
            plot_concentration_extrapolation(res, ax=self.ax)
            self.ax.set_title('D vs c → infinite dilution')
        drew_excluded = False
        if sid is not None:
            drew_excluded = self._gray_points(sid)   # gray the unticked (excluded) ok points
        if drew_excluded:
            # The gray ×'s are added AFTER the analysis layer built the legend, so
            # re-issue it (preserving any title, e.g. the D-vs-c k_D title) to give the
            # 'excluded' marker an entry.
            reissue_legend_preserving_title(self.ax, fontsize=9)
        self.canvas.draw_idle()
        self.axis_bar.attach(self.ax)

    def _gray_points(self, sid: str) -> bool:
        """Gray × the computed points that are NOT ticked, from the run's per-point
        values (by item_id, so replicates are disambiguated). Returns True if any
        excluded point was drawn (so the caller can add a legend entry). Only the
        first marker is labeled so the legend shows one 'excluded' entry, not one
        per point."""
        inc = self._included.get(self._ckey(sid), set())
        if self.kind == 'gamma_q2':
            xf, yf, xk, yk = (_disp_factor('scattering_q2'),
                              _disp_factor('decay_rate'), 'q2_m2', 'gamma_s_inv')
        else:
            xf, yf, xk, yk = (_disp_factor('concentration'),
                              _disp_factor('diffusion'),
                              'concentration_g_per_mL', 'd_app_m2_s')
        xy = [(pt[xk] * xf, pt[yk] * yf)
              for pt in self._run_points_of(self._ckey(sid))
              if not (pt['quality'] not in FIT_ELIGIBLE_QUALITIES or pt['item_id'] in inc
                      or pt[xk] is None or not math.isfinite(pt[xk]))]
        return draw_excluded_markers(self.ax, xy, zorder=4)

    def _clear(self, message: str) -> None:
        if self._placeholder is not None:        # ax.clear() doesn't drop a figure.text
            self._placeholder.remove()
            self._placeholder = None
        self.ax.clear()
        # Hide the axis and show one centred grey message via the shared helper — the
        # clean empty-state look (a blank plot box reads as broken). _draw restores it.
        self._placeholder = show_plot_placeholder(
            self.figure, self.canvas, message, axes=(self.ax,))
        self.axis_bar.attach(None)               # nothing to drive while empty
