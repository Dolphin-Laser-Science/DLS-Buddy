"""
The DLS DDLS sub-tab: the depolarized fit over the sample's angles and its shape models.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from plotting.plots import plot_ddls, display_factor as _disp_factor
from core.notes import find_notes
from core.naming import sample_display_name
from gui.plot_controls import make_split_panels, pin_under_column, set_plot_export_name
from gui.export_helper import export_to_csv, sample_suggested_name
from gui.field_labels import humanize_field_names, missing_input_lines
from gui.help import add_help_to_groupbox, section_header
from gui.notes_block import NotesBlock, qualifier_words
from gui.plot_overlays import (draw_excluded_markers, reissue_legend_preserving_title,
                               show_plot_placeholder)
from gui.theme import ThemedLabel, color as theme_color, set_flag
from gui.widgets import make_plot_pane, RichTextDelegate
from gui.worker import BACKGROUND_RUN_TOOLTIP, BUSY_NOTICE, RunSlot
from core.formatting import format_fixed_sig, format_pm, pdg_erases_value

from gui.dls.region import RegionCheckBox
from gui.dls.common import _STALE_HINT, _STALE_UNRUNNABLE, _needs_lines
from gui.dls.tables import (_vtable, _qualified, _label_row, _fill_vtable, _reset_vtable,
                            _readonly_table)
from gui.dls.bases import _SampleRunTab


_ROD_FROM_BOX = object()   # `_DDLSTab._run_signature`: read the rod length from the box


# The ± fields behind the Shape Models table's two ± columns, over the three models
# (an attribute a model does not carry names nothing of it).
_SHAPE_DIMENSION_SES = ('length_se', 'diameter_se', 'radius_rot_se', 'radius_trans_se',
                        'semi_axis_polar_se', 'equatorial_radius_se')


_SHAPE_ASPECT_SES = ('aspect_ratio_se', 'axial_ratio_se')


class _DDLSTab(_SampleRunTab):
    """Depolarized DLS (DDLS): rotational diffusion D_r from a sample's VV/VH
    correlogram pairs. Sample-scoped, like the Γ vs q² tab. Tag each correlogram's
    polarization (VV/VH) in the Data tab; this pairs them by angle and extracts
    D_r = (Γ_VH − Γ_VV)/6, with D_t from the VV channel."""

    _RESULT_ROWS = ['D<sub>r</sub> (rad²/s)', 'τ<sub>rot</sub> (µs)', 'D<sub>t</sub> (m²/s)',
                    'R<sub>h,t</sub> (nm)', 'paired angles', 'single-exp (qL)']
    # What each Results row shows, keyed by its label: the attributes a note or a
    # qualifier must name to reach the row -- its value and its ±.
    _RESULT_ABOUT = {
        'D<sub>r</sub> (rad²/s)': ('d_r_rad2_s', 'd_r_se'),
        'τ<sub>rot</sub> (µs)': ('rotational_time_s',),
        'D<sub>t</sub> (m²/s)': ('d_t_m2_s', 'd_t_se'),
        'R<sub>h,t</sub> (nm)': ('rh_t_nm', 'rh_t_se'),
        'paired angles': ('n_angles',),
        'single-exp (qL)': ('single_exponential_valid',),
    }

    selectionChanged = QtCore.Signal()           # included angle set changed (mirror)
    defers_repaint = True     # hidden, a sidebar click runs only `focus_state`

    def __init__(self, controller, parent=None, *, region=None) -> None:
        super().__init__(controller, parent, region=region)
        # The status line the Run gate last wrote, so the gate clears only its own text.
        self._gate_status = ''
        # Each sample's fit, with the signature it ran under, its `info` and the
        # all-angle fit it drew excluded angles from, lives in the controller under
        # ('ddls', sid) (`_shown_run`).

        # sample_id -> (ddls_result, shapes dict): memoizes the shape-model inverse
        # (incl. the ellipsoid Monte-Carlo) so re-selecting a sample or exporting does
        # not re-run the MC on the GUI thread; keyed by result identity, so a re-run
        # invalidates it.
        self._shape_cache: Dict[str, tuple] = {}
        self._suppress_ddls = False             # re-entrancy guard for checkbox edits
        # The Run's epoch and the sample id of its deferred press, retried while DDLS
        # can still run on that sample.
        self._run_slot = RunSlot(available=self._deferred_run_available,
                                 rerun=lambda target: self._recompute(target),
                                 before_delivery=self.flush_view)
        self._rod_by_sid: Dict[str, Optional[float]] = {}   # rod length each sample's last fit used
        self._placeholder = None                # empty-state Text artist (no fit shown)
        self._build_ui()

    def _build_ui(self) -> None:
        splitter, left, right = make_split_panels(self, left_min_width=300,
                                                  sizes=(360, 740))

        # Paired angles with an include checkbox (tick to include in the D_r/D_t fit —
        # the same idiom as Γ vs q²), plus each correlogram's polarization + pairing.
        left.addWidget(section_header(
            'Paired Angles (tick to include in the fit)',
            'Tick which paired angles enter the D_r/D_t fit.',
            bullets=['DDLS pairs the sample\'s VV (polarized) and VH (depolarized) '
                     'correlograms by angle (Theory-and-Equations-Guide §3.6.1).',
                     'Tag each correlogram\'s polarization in the <b>Data</b> tab.',
                     'Only angles with BOTH a VV and a VH correlogram, each with its '
                     'parameters complete, can be paired.',
                     '<b>Run</b> is off while no paired angle is ticked or a Data-tab '
                     'value is missing; the line under the column says what to enter.',
                     'Repeats at one angle and polarization have their decay rates '
                     'averaged.',
                     'Untick an angle to drop it, e.g. an outlier.',
                     '<b>Click</b> a ⚠ line or lettered value for what to do.',
                     'More: User-Manual §8.7.']))
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(['', 'Angle (°)', 'Polarization', 'Paired'])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setMaximumHeight(170)
        # No row numbers and a content-sized check column, so the four columns fit the
        # 300 px column without a sideways scroll bar.
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemChanged.connect(self._on_angle_toggled)
        left.addWidget(self.table)
        # Outlier removal: untick an angle above to drop it; unticked angles are
        # grayed on the plot. "Include all" re-ticks every paired angle -- the table's
        # bulk-select, so it sits under the table.
        self.reset_excl_button = QtWidgets.QPushButton('Include all angles')
        self.reset_excl_button.clicked.connect(self._on_include_all)
        incl_row = QtWidgets.QHBoxLayout()
        incl_row.setContentsMargins(0, 0, 0, 0)
        incl_row.addWidget(self.reset_excl_button)
        incl_row.addStretch(1)
        left.addLayout(incl_row)

        rod_row = QtWidgets.QHBoxLayout()
        rod_row.setContentsMargins(0, 0, 0, 0)
        rod_row.addWidget(QtWidgets.QLabel('Rod length L (nm):'))
        self.rod_edit = QtWidgets.QLineEdit()
        self.rod_edit.setPlaceholderText('optional — for the qL < 3 check')
        self.rod_edit.setToolTip(
            'Rod length, used only to evaluate qL (the single-exponential validity '
            'guard); leave blank if unknown. Kept per sample: switching samples shows '
            'the rod length that sample\'s last fit used.')
        rod_row.addWidget(self.rod_edit)
        rod_w = QtWidgets.QWidget(); rod_w.setLayout(rod_row)
        left.addWidget(rod_w)
        self.region_box = RegionCheckBox(self._region)
        self.region_box.toggled.connect(self._on_region_toggled)
        left.addWidget(self.region_box)

        self.run_button = QtWidgets.QPushButton('Run')
        self.run_button.setToolTip(BACKGROUND_RUN_TOOLTIP)
        self.run_button.clicked.connect(self._on_run)
        left.addWidget(self.run_button)
        self.export_button = QtWidgets.QPushButton('Export CSV…')
        self.export_button.setEnabled(False)
        self.export_button.setToolTip(
            'Writes the paired angles, D_t / D_r and any shape models to an '
            'Origin-format CSV. A D_r flagged unphysical here is repeated in the '
            'Comments cell of every column that presents it, so the caveat travels '
            'with the file (Theory-and-Equations-Guide §3.6).')
        self.export_button.clicked.connect(self._on_export)

        left.addWidget(QtWidgets.QLabel('Results'))
        self.results = _vtable(self._RESULT_ROWS)
        left.addWidget(self.results)
        self.status = ThemedLabel('', role='muted', size=11)
        self.status.setWordWrap(True)
        self.flag_label = ThemedLabel('', role='error', bold=True)
        self.flag_label.setWordWrap(True)

        # Shape models (model-dependent dimensions). Framed + captioned so the
        # "assumed shape, not measured" caveat reads clearly vs the D_r/D_t observables.
        self.shape_box = QtWidgets.QGroupBox(
            'Shape Models (assumed geometry — not measured)')
        shape_layout = QtWidgets.QVBoxLayout(self.shape_box)
        self.shape_table = _readonly_table(
            ['Model', 'Dimensions', 'Aspect / ratio', 'OK?'], [])
        # The Dimensions cell (col 1) names quantities (R(D_r), R(D_t)=R_h); render it
        # rich. Its values never contain a literal '<', so the delegate's HTML path is safe.
        self.shape_table.setItemDelegateForColumn(1, RichTextDelegate(self.shape_table))
        shape_layout.addWidget(self.shape_table)
        self.shape_caveat = ThemedLabel('', role='hint', size=11)
        self.shape_caveat.setWordWrap(True)
        shape_layout.addWidget(self.shape_caveat)
        add_help_to_groupbox(
            self.shape_box,
            'Particle dimensions from D_t and D_r, assuming a shape: model-dependent, '
            'not a direct measurement.',
            bullets=[
                'Sphere: one radius from D_r, cross-checked against Rh (from D_t); '
                'the ratio flags whether a sphere is plausible.',
                'Rod: length and diameter of a rigid cylinder (trusted for aspect '
                'ratio 2–30).',
                'Ellipsoid: polar (a) and equatorial (b) semi-axes, prolate or '
                'oblate (trusted for 0.1–10); every ellipsoid that fits is listed, none '
                'preferred.',
                'p is the axial ratio; ✓/⚠ marks whether it is inside the model’s '
                'validated range (Theory-and-Equations-Guide §3.6).',
                'Every value here but the sphere’s R(D_r) is computed from D_t, so a '
                'failed precondition of D_t marks it with the same ⚠ and letter as the '
                'D_t result.',
            ])
        left.addWidget(self.shape_box)
        self.shape_box.setVisible(False)
        # The notes of the fit and its shape models: one line per failed precondition, a
        # marker on each value it is about; pinned under the column, below the result and
        # shape cells (R8.6).
        self.notes_block = NotesBlock(what='models')
        self.notes_block.watch(self.results)
        self.notes_block.watch(self.shape_table)
        left.addStretch(1)
        # Export, the status and the flag pinned under the column (§3 column template).
        pin_under_column(splitter, self.export_button, self.status,
                         self.notes_block, self.flag_label)

        self.figure, self.canvas, self._nav, self.axis_bar = make_plot_pane(
            self, self.controller, (5.2, 4.3), constrained_layout=True)
        self.ax = self.figure.add_subplot(111)
        right.addWidget(self._nav)
        right.addWidget(self.canvas, 1)
        right.addWidget(self.axis_bar)

    def _state_key(self, sid: str) -> str:
        """The sample alone: DDLS has no fraction (`run_ddls` takes none and this tab
        caches by sample). `_included` holds the sample's INCLUDED paired angles (tick to
        include, like Γ vs q²); the excluded set the engine gets is paired − included."""
        return sid

    def showEvent(self, event) -> None:
        """Re-read the workspace when the tab becomes visible, as the other sample-level
        sub-tabs do: runs added to the shown sample while it was out of sight are listed
        when it is opened (the owner's decision of 2026-10-22)."""
        super().showEvent(event)
        if self.item_id is not None:
            self.set_measurement(self.item_id, self._runnable)

    def _shown_run(self, sid: Optional[str]):
        """The DDLS fit of `sid` this tab ran and draws (`Controller.shown_run`), its
        extras `{'info', 'full'}`: the fit's `info` and the all-angle fit the excluded
        angles are drawn from. None when there is none."""
        return None if sid is None else self.controller.shown_run(('ddls', sid))

    def _focus_hook(self, sid: Optional[str]) -> None:
        """DDLS's part of the state half (`_SampleRunTab.focus_state`): on a re-point,
        the epoch and the rod box; on a runnable focus of a sample, the tick merge. DDLS
        is a single-sample tab that follows the sidebar focus, and `focus_state` keeps
        the sample on a non-DLS focus rather than blanking it: the intended navigation
        for a single-sample view (not a stopgap)."""
        # Against the sample followed before this focus: `focus_state` moves `_last_sid`
        # after this hook, and `_recompute`'s and `fail`'s branches read it.
        if sid != self._last_sid:
            self._run_epoch += 1     # re-pointed: drop any in-flight run's result
            # A rod length is a per-particle input: typed for one sample it must
            # not become the next sample's fit input (the manual-box family). The box
            # shows the rod THIS sample's last fit used, so an unchanged cached fit
            # does not read stale and a re-Run keeps its qL verdict.
            stored = self._rod_by_sid.get(sid) if sid is not None else None
            self.rod_edit.setText('' if stored is None else f'{stored:g}')
        if self._runnable and sid is not None:
            # At the focus, and again in `_refresh_table`: a sample's first merge ticks
            # every angle paired then, so a pair confirmed before the tab is shown joins
            # unticked, as when the tab was current.
            rows = self.controller.ddls_correlogram_summary(sid)
            paired = {float(r['angle_deg']) for r in rows if r['paired']}
            prev = self._included.get(sid)
            self._included[sid] = (prev & paired) if prev is not None else set(paired)

    def repaint_view(self) -> None:
        """The repaint half: the stored focus's sample, its paired angles and its fit or
        empty state, then the Run gate, from the controller as it is now."""
        self.flag_label.clear()
        sid = self._sample_id()
        self.region_box.set_checked_silently(self._use_region.get(sid, False))
        run = self._shown_run(sid)
        self.export_button.setEnabled(self._runnable and run is not None)
        if self._runnable and sid is not None:
            self._refresh_table(sid)
            if run is not None:
                self._draw(run.result, run.extras['info'])
                if run.sig != self._run_signature(sid):
                    self.status.setText(_STALE_HINT)
            else:
                self._clear_results()
                self._clear("Run DDLS to pair this sample's VV/VH correlograms.")
        else:
            self.table.setRowCount(0)
            self._clear_results()
            self._clear('Select a DLS measurement.' if not self._runnable
                        else 'No sample for this measurement.')
        # Last: the redisplay above rewrites the status line this gate writes to.
        self._apply_run_gate()

    def _fit_availability(self, sid: str):
        """The roster's verdict on DDLS for `sid`: every run the fit builds, with the
        paired angles the ticks leave out, as `run_ddls` takes them."""
        return self.controller.dls_fit_availability(
            self.controller.ddls_fit_runs(sid),
            exclude_angles=sorted(self._excluded_angles(sid)))['ddls']

    def _gate_run_button(self):
        """Enable Run only while a runnable measurement of a sample is shown and DDLS can
        run on it; returns the verdict, None when no sample is shown."""
        sid = self._sample_id()
        avail = (self._fit_availability(sid) if self._runnable and sid is not None
                 and sid in self.controller.workspace.samples else None)
        self.run_button.setEnabled(avail is not None and avail.ok)
        return avail

    def _apply_run_gate(self) -> bool:
        """Run follows DDLS's availability on the shown sample, and while it cannot run
        the status line names what is missing. A fit on screen is never cleared here;
        the status line marks it out of date when its inputs changed (R8.5), without
        asking for Run while Run is off. Returns whether Run is on."""
        avail = self._gate_run_button()
        sid = self._sample_id()
        run = self._shown_run(sid)
        stale = (avail is not None and run is not None
                 and run.sig != self._run_signature(sid))
        if avail is None or avail.ok:
            if self._gate_status and self.status.text() == self._gate_status:
                self.status.setText(_STALE_HINT if stale else '')
            self._gate_status = ''
            return avail is not None
        lines = ['DDLS needs:', *missing_input_lines(avail.missing, counted='ticked')]
        if stale:
            lines.append(_STALE_UNRUNNABLE)
        self._gate_status = '\n'.join(lines)
        self.status.setText(self._gate_status)
        return False

    def _clear_results(self) -> None:
        _reset_vtable(self.results)
        for r, row in enumerate(self._RESULT_ROWS):
            head, it = self.results.item(r, 0), self.results.item(r, 1)
            if head is not None:
                _label_row(head, row, None, (), whole=False)
            if it is not None:
                it.setToolTip('')
        self.status.clear()
        self.flag_label.clear()
        self.notes_block.clear()
        self.shape_table.setRowCount(0)
        self.shape_caveat.clear()
        self.shape_box.setVisible(False)

    def _paired_angles(self, sid: str) -> set:
        """The angles that have BOTH a VV and a VH correlogram (fittable)."""
        return {float(r['angle_deg'])
                for r in self.controller.ddls_correlogram_summary(sid) if r['paired']}

    def _excluded_angles(self, sid: str) -> set:
        """Angles to exclude from the fit = paired − included (engine takes excludes)."""
        inc = self._included.get(sid)
        if inc is None:
            return set()                          # not initialized yet → nothing excluded
        return self._paired_angles(sid) - inc

    def _refresh_table(self, sid: str) -> None:
        rows = self.controller.ddls_correlogram_summary(sid)
        paired = {float(r['angle_deg']) for r in rows if r['paired']}
        # Default: every paired angle included. Preserve prior ticks across refreshes
        # (intersect with the still-paired set), like the Γ vs q² tab.
        prev = self._included.get(sid)
        self._included[sid] = (prev & paired) if prev is not None else set(paired)
        inc = self._included[sid]
        self._suppress_ddls = True
        try:
            self.table.setRowCount(len(rows))
            for r, row in enumerate(rows):
                ang = float(row['angle_deg'])
                # Only a paired angle whose run builds is fittable → only it is checkable.
                fittable = row['paired'] and not row['missing']
                chk = QtWidgets.QTableWidgetItem()
                chk.setData(QtCore.Qt.ItemDataRole.UserRole, ang)
                if fittable:
                    chk.setFlags(QtCore.Qt.ItemFlag.ItemIsUserCheckable
                                 | QtCore.Qt.ItemFlag.ItemIsEnabled)
                    chk.setCheckState(QtCore.Qt.CheckState.Checked if ang in inc
                                      else QtCore.Qt.CheckState.Unchecked)
                else:
                    chk.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
                cells = (chk, QtWidgets.QTableWidgetItem(f"{ang:g}"),
                         QtWidgets.QTableWidgetItem(row['geometry'] or '—'),
                         QtWidgets.QTableWidgetItem('✓' if fittable else ''))
                needs = _needs_lines(row['missing'])
                for c, cell in enumerate(cells):
                    cell.setToolTip(f'Needs:\n{needs}' if row['missing'] else '')
                    self.table.setItem(r, c, cell)
                self._tint_ddls_row(r, fittable and ang in inc)
        finally:
            self._suppress_ddls = False

    def _tint_ddls_row(self, r: int, on: bool) -> None:
        col = theme_color(self.table, 'marker_selected') if on else QtGui.QBrush()
        for c in range(self.table.columnCount()):
            it = self.table.item(r, c)
            if it is not None:
                it.setForeground(col)

    @QtCore.Slot('QTableWidgetItem*')
    def _on_angle_toggled(self, item) -> None:
        if self._suppress_ddls or item.column() != 0:
            return
        sid = self._sample_id()
        if sid is None:
            return
        ang = item.data(QtCore.Qt.ItemDataRole.UserRole)
        inc = self._included.setdefault(sid, set())
        checked = item.checkState() == QtCore.Qt.CheckState.Checked
        if checked:
            inc.add(float(ang))
        else:
            inc.discard(float(ang))
        # An angle has two rows (VV + VH); keep both checkboxes + tints in sync.
        # (setCheckState/setForeground re-emit itemChanged, so guard the sync.)
        self._suppress_ddls = True
        for r in range(self.table.rowCount()):
            it0 = self.table.item(r, 0)
            if (it0 is not None
                    and it0.data(QtCore.Qt.ItemDataRole.UserRole) == ang
                    and it0.flags() & QtCore.Qt.ItemFlag.ItemIsUserCheckable):
                it0.setCheckState(QtCore.Qt.CheckState.Checked if checked
                                  else QtCore.Qt.CheckState.Unchecked)
                self._tint_ddls_row(r, checked)
        self._suppress_ddls = False
        self.selectionChanged.emit()          # repaint the sidebar mirror
        self._run_epoch += 1                  # ticked set changed: in-flight run is stale
        # A live refit on the new subset only while DDLS can run; else the gate's line.
        if self._apply_run_gate() and self._shown_run(sid) is not None:
            self._recompute(sid)

    @QtCore.Slot()
    def _on_run(self) -> None:
        if not self._runnable or self.item_id is None:
            return
        sid = self._sample_id()
        if sid is None:
            QtWidgets.QMessageBox.warning(self, 'No sample',
                                          'This measurement is not assigned to a sample.')
            return
        if not self._apply_run_gate():
            return                                # the status line says what is missing
        try:
            rod = self._rod_input()
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self, 'Invalid rod length',
                'Rod length must be a number in nm, or blank.')
            return
        self._rod_by_sid[sid] = rod
        self._run_epoch += 1     # rod length is a fit input: drop any in-flight fit
        self._recompute(sid)

    def _rod_input(self) -> Optional[float]:
        """The rod length typed in the box (nm), None when blank; ValueError on text
        that is not a number."""
        txt = self.rod_edit.text().strip()
        return float(txt) if txt else None

    def _run_signature(self, sid: str, rod=_ROD_FROM_BOX) -> tuple:
        """Everything a DDLS fit of `sid` reads beyond its raw correlograms: the
        committed params of the VV/VH pairs feeding it, the rod length, and the
        settings the engine seeds (skip_initial_channels, the cumulant method, the
        cumulant fit cutoff and weighting, the SE estimator behind the D_r/D_t ±). Stored at dispatch
        with the rod the fit reads (`rod=`), compared on redisplay against the box (the
        Γ-q² tab's shape).

        The settings half is not maintained by hand: every field
        `Controller.run_ddls` reads must appear here, and
        `test_every_run_signature_carries_the_settings_its_engine_path_reads` derives
        that set by scanning the controller rather than trusting this list."""
        s = self.controller.settings
        if rod is _ROD_FROM_BOX:
            try:
                rod = self._rod_input()
            except ValueError:
                rod = self.rod_edit.text().strip()
        return (self.controller.committed_signature(self._input_ids(sid)), rod,
                s.skip_initial_channels, s.cumulant_method, s.cumulant_fit_cutoff,
                s.cumulant_weighting, s.cumulant_order, s.se_estimator,
                self.region_box.key(self._use_region.get(sid, False)))

    # ---- the region choice (the Γ-q² tab's shape, held per sample) ----
    def _on_region_input_moved(self) -> None:
        """A region input moved: flag the shown fit stale when its stored key no longer
        matches, and clear the hint when it matches again. Never an automatic refit --
        unlike an angle tick, which refits the ticked subset -- because the region is
        another tab's input and the user decides when to re-read it."""
        sid = self._sample_id()
        if self._shown_run(sid) is None:
            return
        if not self._apply_run_gate():
            return                          # Run is off: its line carries the mark
        if self._shown_run(sid).sig != self._run_signature(sid):    # the rod box, as
            self.status.setText(_STALE_HINT)                        # repaint_view
        elif self.status.text() == _STALE_HINT:
            self.status.clear()

    def _recompute(self, sid: str) -> None:
        rod = self._rod_by_sid.get(sid)
        excl = self._excluded_angles(sid)         # paired − included
        controller = self.controller
        region = self.region_box.region_for_run(self._use_region.get(sid, False))
        sig = self._run_signature(sid, rod=rod)   # names the VV/VH ids feeding this fit
        on_screen = sid == self._last_sid   # else a deferred Run: never paint it here

        def work():
            # ONE thunk for both calls. Run order is load-bearing: run_ddls writes
            # results[('ddls', sid)] each call (last-write-wins), so the EXCLUDED fit runs
            # LAST and the controller's store ends up holding the fit shown in the table --
            # the one a caller of ddls_shape with no fit of its own reads. The un-excluded
            # `full` (run first, only when excluding) is kept solely to draw the
            # excluded-point markers; the fit it overwrote is put back at once, so the
            # tab's previous fit stays the one drawn until this Run lands or is refused.
            full = None
            if excl:
                previous = controller.held_run(('ddls', sid))
                full = controller.run_ddls(sid, rod_length_nm=rod, region=region)
                controller.restore_held_run(('ddls', sid), previous)
            res, info = controller.run_ddls(
                sid, rod_length_nm=rod, exclude_angles=excl, region=region)
            if full is None:
                full = (res, info)
            # The shape models (the ellipsoid inverse's Monte-Carlo is the heavy part) come
            # from this fit alone, so they run here rather than at the first draw. A failure
            # never fails the fit: the draw recomputes it and shows its caveat.
            try:
                shapes = controller.ddls_shape(sid, model='both', result=res)
            except Exception:  # noqa: BLE001 — surfaced by _draw_shape's own call
                shapes = None
            return (res, info), full, shapes

        def done(payload, token) -> None:
            # The fit lands in its own sample's slot whatever is focused now; only
            # the display is gated on the view being the one this run started from.
            (res, info), full, shapes = payload
            self.controller.record_shown_run(('ddls', sid), res, sig,
                                             {'info': info, 'full': full})
            if shapes is not None:
                self._shape_cache[sid] = (res, shapes)
            self._gate_run_button()           # the worker re-enabled Run
            if not self._run_slot.current(token):
                return                      # sample/exclusions changed — stale VIEW
            self.export_button.setEnabled(True)
            self._refresh_table(sid)
            self._draw(res, info)
            # The box and the region stay live while the run is in flight: compare the key
            # it ran under with the live one, as the Mode Map does on landing.
            self._on_region_input_moved()

        def fail(exc: BaseException, token) -> None:
            # A fit that cannot run is flagged in the panel, not a modal (style guide
            # R8.4): an angle untick re-runs this live, and a modal would come up per
            # click. The controller's message is relayed as written. The previous fit's
            # numbers go with its flag, so none stays on screen under the failure. A
            # failure for a sample no longer shown is told in a dialog naming it, never
            # painted over the view now shown; one superseded by a newer run of this
            # same sample is dropped, because that run reports its own outcome.
            msg = humanize_field_names(str(exc))
            self._gate_run_button()           # the worker re-enabled Run
            if sid != self._last_sid:
                who = self.controller.workspace.samples.get(sid)
                QtWidgets.QMessageBox.critical(
                    self, 'DDLS failed',
                    f'Could not run DDLS on {sample_display_name(who) if who else sid}.'
                    f'\n\n{msg}')
                return
            if not self._run_slot.current(token):
                return
            self._clear_results()
            self.export_button.setEnabled(False)
            set_flag(self.flag_label, f'DDLS could not run: {msg}')
            self.status.setText('DDLS failed — see the warning below.')

        # A press while the worker is busy takes the pending slot and is retried at idle
        # against the sample it was pressed on, whatever is focused by then.
        if self._run_slot.submit(
                work, done, fail, target=sid, queue=True,
                busy_widgets=(self.run_button,),
                notice=lambda: self.status.setText(BUSY_NOTICE),
                description='DDLS rotational-diffusion fit',
                paint=on_screen) and on_screen:
            self.status.setText('Fitting in the background…')

    def _deferred_run_available(self, target: str) -> bool:
        """Whether a Run deferred on sample `target` may still run: its angles may have
        been unticked while the worker was busy."""
        return (bool(self._runnable) and target in self.controller.workspace.samples
                and self._fit_availability(target).ok)

    def selected_item_ids(self) -> list:
        """The DLS measurements at the currently-included angles (sidebar-mirror
        contract): the mirror lights the VV/VH correlograms feeding the fit."""
        sid = self._sample_id()
        return [] if sid is None else self._input_ids(sid)

    def _input_ids(self, sid: str) -> list:
        """The VV/VH ids at `sid`'s included angles — the fit's inputs, for the
        sample named rather than the focused one (a deferred Run needs its own)."""
        inc = self._included.get(sid, set())
        out = []
        for lm in self.controller.workspace.sample_measurements(sid, 'dls'):
            ang = lm.committed_params.get('angle_deg')
            if ang is not None and float(ang) in inc:
                out.append(lm.item_id)
        return out

    @QtCore.Slot()
    def _on_include_all(self) -> None:
        sid = self._sample_id()
        if sid is None:
            return
        paired = self._paired_angles(sid)
        if self._included.get(sid) == paired:
            return                                # already all-included
        self._included[sid] = set(paired)
        self._refresh_table(sid)                 # re-tick every row
        self.selectionChanged.emit()
        self._run_epoch += 1     # included set changed: drop any in-flight fit
        if self._apply_run_gate() and self._shown_run(sid) is not None:
            self._recompute(sid)

    @QtCore.Slot()
    def _on_export(self) -> None:
        sid = self._sample_id()
        run = self._shown_run(sid)
        if run is None:
            # A regroup made while another kind of measurement was focused took the fit
            # without redrawing this tab: draw its empty state now.
            self.set_measurement(self.item_id, self._runnable)
            return
        res = run.result
        # Include the shape models when they are available (viscosity present,
        # physical D_r); export the observables alone otherwise. From the shown fit,
        # as the shape table is: its memoized models when they are that fit's.
        cached = self._shape_cache.get(sid)
        if cached is not None and cached[0] is res:
            shapes = cached[1]
        else:
            try:
                shapes = self.controller.ddls_shape(sid, model='both', result=res)
            except Exception:  # noqa: BLE001 — shape models optional in export
                shapes = None
        # export_to_csv shows the save dialog and writes the file; its returned
        # status is shown in the (non-error) results label, not the red flag line.
        status = export_to_csv(
            self, self.controller,
            sample_suggested_name(self.controller, sid, 'ddls'),
            lambda p: self.controller.export_ddls(res, p, shapes=shapes))
        if status:
            self.status.setText(status)

    def _draw(self, res, info) -> None:
        if self._placeholder is not None:       # leaving the empty-state: drop the message
            self._placeholder.remove()
            self._placeholder = None
        self.ax.clear()
        self.ax.set_visible(True)               # restore after an empty _clear() hid it
        self.axis_bar.attach(self.ax)           # re-bind the axis controls (detached while empty)
        plot_ddls(res, ax=self.ax)
        sid = self._sample_id()
        # Plot saves suggest the same stem as the CSV export for this analysis.
        set_plot_export_name(self.canvas,
                             sample_suggested_name(self.controller, sid, 'ddls'))
        excl = self._excluded_angles(sid) if sid else None
        drew_excluded = False
        run = self._shown_run(sid)
        if excl and run is not None:
            full = run.extras['full'][0]
            qf, gf = _disp_factor('scattering_q2'), _disp_factor('decay_rate')
            q2 = np.asarray(full.q2_m2, float) * qf
            ang = np.asarray(full.angles_deg, float)
            xy = [(x, y)
                  for arr in (full.gamma_vv_s_inv, full.gamma_vh_s_inv)
                  for x, y, a in zip(q2, np.asarray(arr, float) * gf, ang, strict=True)
                  if any(np.isclose(a, e) for e in excl)]
            drew_excluded = draw_excluded_markers(self.ax, xy, zorder=5)
        if drew_excluded:
            # Re-issue the legend so the gray ×'s (added after plot_ddls built it) show.
            # plot_ddls sets a titleless, framed legend at fontsize 8; match that.
            reissue_legend_preserving_title(self.ax, fontsize=8, frameon=True)
        self.canvas.draw_idle()
        self.axis_bar.attach(self.ax)

        sig = self.controller.settings.no_uncertainty_sig_figs
        # R_h,t with the ± that 1/D_t carries over from D_t's (the D_t row above it), at
        # the knob only when there is none, as the Shape Models table shows it. A withheld
        # R_h,t reads 'n/a', and the note behind it (no viscosity) is its marker and line.
        if math.isfinite(res.rh_t_nm) and res.rh_t_se is not None:
            rh = format_pm(res.rh_t_nm, res.rh_t_se)
        else:
            rh = format_fixed_sig(res.rh_t_nm, sig)
        paired = f'{len(info["paired_angles"])} · {res.method}'
        if info.get('n_replicate_angles'):
            paired += f' · {info["n_replicate_angles"]} replicate-avg'
        if res.single_exponential_valid is True:
            ql = 'qL < 3 ✓'
        elif res.single_exponential_valid is False:
            ql = 'qL ≥ 3'                    # the `ql_above_limit` marker follows it
        else:
            ql = '—'
        _fill_vtable(self.results, [
            format_pm(res.d_r_rad2_s, res.d_r_se),
            format_fixed_sig(res.rotational_time_s * 1e6, sig),
            format_pm(res.d_t_m2_s, res.d_t_se),
            rh, paired, ql])
        # The fit's failed preconditions are the notes block's lines and markers (the
        # shape models' join them in `_draw_shape`), its ± the rows' words.
        self.notes_block.show_notes(res)
        for r, row in enumerate(self._RESULT_ROWS):
            attrs = self._RESULT_ABOUT[row]
            head, it = self.results.item(r, 0), self.results.item(r, 1)
            if head is not None:
                _label_row(head, row, res, attrs, whole=False)
            if it is None:
                continue
            source, about = res, attrs
            if row == 'single-exp (qL)':
                # The qL note alone: it is about D_r, whose cell takes it with the rest.
                source, about = find_notes(res, 'ql_above_limit'), ('d_r_rad2_s',)
            if not self.notes_block.mark_item(it, source, about, whole=False):
                it.setToolTip('')               # a cell marked earlier keeps no popup
        self.status.clear()

        # Operational only: the angles that could not be paired, an inventory of what
        # never entered the fit, on the neutral tier (ui_style_guide §5).
        skipped = []
        if info['vv_only']:
            skipped.append(f"VV-only at {info['vv_only']}°")
        if info['vh_only']:
            skipped.append(f"VH-only at {info['vh_only']}°")
        if info['untagged']:
            skipped.append(f"{len(info['untagged'])} untagged")
        set_flag(self.flag_label, ('Skipped: ' + '; '.join(skipped) + '.') if skipped else '',
                 problem=False)

        self._draw_shape()

    def _draw_shape(self) -> None:
        """Compute and show the shape models of the DDLS fit this tab shows for the
        current sample -- its own run (`_shown_run`), never the controller's stored result,
        which a Settings clear empties while the tab keeps showing the fit, superseded and
        marked out of date."""
        sid = self._sample_id()
        try:
            # Memoize by the shown fit's identity: the ellipsoid inverse runs a Monte-Carlo
            # (heavy), and _draw_shape fires on every sample selection, not just Run.
            res = self._shown_run(sid).result
            cached = self._shape_cache.get(sid)
            if cached is not None and cached[0] is res:
                shapes = cached[1]
            else:
                shapes = self.controller.ddls_shape(sid, model='both', result=res)
                self._shape_cache[sid] = (res, shapes)
        except Exception as exc:  # noqa: BLE001 — surfaced in the caveat
            self.shape_table.setRowCount(0)
            self.shape_table.setToolTip('')      # drop any stale caveat hover
            self.shape_caveat.setText(f'Shape models unavailable: {exc}')
            self.shape_box.setVisible(True)
            return
        rod, sph = shapes['rod'], shapes['sphere']
        ellipsoids = shapes.get('ellipsoid') or []   # [] if the ellipsoid solve failed

        sig = self.controller.settings.no_uncertainty_sig_figs

        def pm_or_knob(v, se):
            # L/d/p/a/b/R(D_r) carry a Monte-Carlo/delta-method SE — WHEN it exists (both
            # D_t and D_r have one), show σ-driven so the ± sets the precision. When it
            # is None (a single-correlogram D_t, or <2 replicate angles), fall back to
            # the one no-uncertainty knob, uniformly with the ±-less cells below — not
            # format_pm's hard-coded no-SE default.

            # A sigma decades larger rounds the value itself to zero, which is how the
            # de-censored aspect-ratio MC erased what it measured. Both facts are real:
            # show them side by side (owner-ruled 2026-08-28). Display only -- the SE on
            # the result is untouched and the export writes both in full.
            if pdg_erases_value(v, se):
                # TWO sig figs on the spread, not one: `_pdg_decimals` keeps two whenever
                # sigma's leading digit is 1, so that a leading-1 sigma is not coarsened
                # (1.4e6 -> 1e6 understates it by 29 %). The displayed spread is subject
                # to the same must-not-under-report rule as the estimator (invariant 8).
                return f'{format_fixed_sig(v, sig)} ± {format_fixed_sig(se, 2)}'
            return format_pm(v, se) if se is not None else format_fixed_sig(v, sig)

        # A shape with no solution reads '—' and one outside its model's range an empty
        # OK? cell: the notes block's marker states each (`_mark_shape_row` below).
        if not rod.converged:
            rod_cells = ['Rod', '—', '', '✗']
        else:
            ok = '✓ in range' if rod.in_valid_range else ''
            rod_cells = [
                'Rod',
                f'L = {pm_or_knob(rod.length_nm, rod.length_se)} nm,  '
                f'd = {pm_or_knob(rod.diameter_nm, rod.diameter_se)} nm',
                f'p = {pm_or_knob(rod.aspect_ratio_p, rod.aspect_ratio_se)}', ok]
        # R(D_t) = Rh DOES carry an SE — propagated from d_t_se, the same ± the export
        # labels with its estimator — so it shows σ-driven like its neighbour; only the
        # sphericity ratio, which has none, falls back to the knob.
        sph_cells = [
            'Sphere',
            f'R(D<sub>r</sub>) = {pm_or_knob(sph.radius_rot_nm, sph.radius_rot_se)} nm  '
            f'vs  R(D<sub>t</sub>)=R<sub>h</sub> = '
            f'{pm_or_knob(sph.radius_trans_nm, sph.radius_trans_se)} nm',
            f'ratio {format_fixed_sig(sph.sphericity_ratio, sig)}',
            '✓' if sph.is_consistent else '✗']
        # Ellipsoid: one row per fitting solution (0-3; a given D_t/D_r admits up to
        # three -- 1 prolate + 2 oblate). The no-fit sentinel renders its own row.
        ell_rows = []
        for e in ellipsoids:
            if not e.converged:
                ell_rows.append(['Ellipsoid', '—', '', '✗'])
                continue
            kind = 'prolate' if e.is_prolate else 'oblate'
            ok = '✓ in range' if e.in_valid_range else ''
            ell_rows.append([
                f'Ellipsoid ({kind})',
                f'a = {pm_or_knob(e.semi_axis_polar_nm, e.semi_axis_polar_se)} nm,  '
                f'b = {pm_or_knob(e.equatorial_radius_nm, e.equatorial_radius_se)} nm',
                f'p = {pm_or_knob(e.axial_ratio_p, e.axial_ratio_se)}', ok])

        t = self.shape_table
        rows = [rod_cells, sph_cells, *ell_rows]
        models = [rod, sph, *ellipsoids]
        # The fit's notes and the models', one block: each failed precondition once. The
        # fit's own come first, so the markers `_draw` set on the Results keep their
        # letters.
        self.notes_block.show_notes(tuple(n for src in (res, *models)
                                          for n in find_notes(src)))
        # The ± columns say what their ± is, in the models' qualifier words.
        headers = ['Model', 'Dimensions', 'Aspect / ratio', 'OK?']
        for c, attrs in ((1, _SHAPE_DIMENSION_SES), (2, _SHAPE_ASPECT_SES)):
            words = list(dict.fromkeys(w for m in models
                                       for w in qualifier_words(m, attrs, whole=False)))
            headers[c] = _qualified(headers[c], words)
        t.setHorizontalHeaderLabels(headers)
        t.setRowCount(len(rows))
        # Every shape value but the sphere's R(D_r) is computed from D_t, so the notes the
        # D_t cell's marker reads mark them too.
        d_t_notes = tuple(n for n in find_notes(res, kind='precondition')
                          if 'd_t_m2_s' in n.about)
        for r, (cells, model) in enumerate(zip(rows, models, strict=True)):
            for c, txt in enumerate(cells):
                t.setItem(r, c, QtWidgets.QTableWidgetItem(txt))
            self._mark_shape_row(r, model, d_t_notes)
        t.resizeColumnsToContents()

        n_ell = sum(1 for e in ellipsoids if e.converged)
        if sph.is_consistent:
            verdict = 'Sphere consistent — a near-spherical particle.'
        else:
            verdict = ('Sphere inconsistent (ratio ≠ 1) → an anisotropic model '
                       '(rod or ellipsoid) is the relevant one.')
        if n_ell == 1:
            verdict += ' One ellipsoid of revolution reproduces these D_t, D_r.'
        elif n_ell >= 2:
            verdict += (f' The ellipsoid fit is degenerate — {n_ell} shapes '
                        f'(prolate/oblate) reproduce these D_t, D_r equally.')
        # This reading fires on every DDLS run, so it lives in a passive hover tooltip on
        # the table (auto-gated by Settings → Show tooltips), not an always-on line — the
        # ✓/✗ and ratio cells already carry the visible verdict.
        self.shape_table.setToolTip(verdict)
        self.shape_caveat.setText('')     # standalone caveat demoted to the tooltip above
        self.shape_box.setVisible(True)

    def _mark_shape_row(self, r: int, model, d_t_notes=()) -> None:
        """Mark one shape row's cells with the notes about them: the Dimensions cell a
        note about the whole model (no solution: its '—'), the aspect-ratio and OK?
        cells a note about the ratio (outside the model's range), and the D_t verdict
        `d_t_notes` on every cell computed from D_t: each cell of a solved model (the
        sphere's Dimensions cell ends on its R(D_t), where the marker sits) and the OK?
        verdict of one with no solution, whose other cells show no value. A cell with no
        note takes no tooltip of its own, so the table's reading shows on hover."""
        t = self.shape_table
        ratio = tuple(a for a in ('aspect_ratio_p', 'axial_ratio_p') if hasattr(model, a))
        solved = getattr(model, 'converged', True)
        for c, attrs, whole in ((1, (), True), (2, ratio, False), (3, ratio, False)):
            it = t.item(r, c)
            if it is None:
                continue
            d_t = d_t_notes if (solved or c == 3) else ()
            # D_t's notes first: their lines come before the models' on the block, so the
            # letters read in the block's order.
            self.notes_block.mark_item(it, (*d_t, *find_notes(model)),
                                       (*attrs, *(('d_t_m2_s',) if d_t else ())),
                                       whole=whole)

    def _clear(self, message: str) -> None:
        if self._placeholder is not None:       # ax.clear() doesn't drop a figure.text
            self._placeholder.remove()
            self._placeholder = None
        self.ax.clear()
        # Hide the axis and show one centred grey message via the shared helper — the
        # clean empty-state look (a blank plot box reads as broken). _draw restores it.
        self._placeholder = show_plot_placeholder(
            self.figure, self.canvas, message, axes=(self.ax,))
        self.axis_bar.attach(None)              # nothing to drive while empty
        self.shape_box.setVisible(False)
