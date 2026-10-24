"""
The SLS Mode-Resolved sub-tab: the sample's excess intensity partitioned by its recorded
mode sets, and the static fits over each mode's share.
"""
from __future__ import annotations

import html
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PySide6 import QtCore, QtWidgets

from core.notes import notes_in
from core.naming import sample_display_name
from plotting.plots import conc_mg_ml_label
from gui.plot_controls import make_split_panels, pin_under_column
from gui.export_helper import export_to_csv, sample_suggested_name
from gui.field_labels import has_honest_se, humanize_field_names
from gui.help import section_header
from gui.notes_block import NotesBlock, qualifier_words
from gui.plot_overlays import show_plot_placeholder
from gui.widgets import make_plot_pane
from gui.theme import ThemedLabel, set_flag
from gui.worker import RunSlot, busy_notice, runner
from core.formatting import format_fixed_sig, format_pm

from gui.sls.common import (_EMPTY, _scale_causes, _scaled_value, _RG_MW_QUALIFIERS,
                            _confirm_unmeasured_blank)
from gui.sls.rows import _has_pm


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
        self._fit: Optional[Any] = None                    # the fit the readout shows
        self._guinier: Optional[Any] = None                # last size fit
        self._guinier_conc: Optional[float] = None          # ... and the c it ran at
        self._fit_slot = RunSlot()                         # a busy Run is refused
        self.ax = None
        self._build_ui()
        self._refresh()

    # ---------------------------------------------------------------- UI ---
    def _build_ui(self) -> None:
        splitter, left, right = make_split_panels(self, left_min_width=420,
                                                  sizes=(470, 700))
        left.addWidget(section_header(
            'Mode-Resolved',
            'Split an SLS intensity series by the DLS modes\' amplitudes — '
            'the Sedlák partition — and analyze each mode\'s share; see the '
            'Theory-and-Equations-Guide §4.2.',
            bullets=[
                'Record a mode set on the DLS tab\'s <b>Mode Map</b>; it is listed '
                'here as a snapshot that a DLS refit never changes.',
                '<b>Partition</b> splits every SLS series of this sample, loaded or '
                'derived, that shares a concentration with the set.',
                'Pick a <b>Mode</b> and an <b>Analysis</b>, then <b>Run</b>: Amplitude '
                'Guinier gives the mode\'s apparent size; Guinier, Debye, Zimm and '
                'Berry run as on any series.',
                'Every result reads <i>apparent</i> and <i>mode-resolved</i>; a ⚠ line '
                'under the column names a failed precondition, and clicking it says '
                'what to do.']))

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
        # Both tables size from their floors and share the spare height, so their
        # preferred height, empty or not, never becomes the column's floor and carries
        # Run out of view (style guide R3.8).
        self.sets_table.setMinimumHeight(90)
        self.sets_table.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                      QtWidgets.QSizePolicy.Policy.Ignored)
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
            'that is strained. See the Theory-and-Equations-Guide §4.2 and §3.5.4.')
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
        self.part_table.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                      QtWidgets.QSizePolicy.Policy.Ignored)
        left.addWidget(self.part_table, 1)

        fit_form = QtWidgets.QFormLayout()
        self.mode_combo = QtWidgets.QComboBox()
        fit_form.addRow('Mode:', self.mode_combo)
        self.method_combo = QtWidgets.QComboBox()
        for label, code in self._METHODS:
            self.method_combo.addItem(label, code)
        self.method_combo.setToolTip(
            'Every result here is computed from ONE mode\'s share of the '
            'excess intensity and is apparent and mode-resolved. Its Kc/ΔR '
            'uses the TOTAL concentration and the chain dn/dc, an assumption the '
            'data cannot check for a mode that is not the chains, so an Mw '
            'from it is unanchored (Theory-and-Equations-Guide §4.2.2).')
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

        self.status = QtWidgets.QLabel('')
        self.status.setWordWrap(True)
        self.flag_label = ThemedLabel('', role='error', bold=True)
        self.flag_label.setWordWrap(True)
        # The failed preconditions of the fit and the partition the readout above shows,
        # pinned first under the column so they sit with it and never scroll away (R8.6).
        self.notes_block = NotesBlock()
        # Export, the status and the flag pinned under the column (§3 column template).
        # Partition and Run stay in the body, each after the inputs it acts on.
        pin_under_column(splitter, self.notes_block, self.export_button, self.status,
                         self.flag_label)

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
        self.figure, self.canvas, self.nav_toolbar, self.axis_bar = make_plot_pane(
            self, self.controller, (5.5, 4.6))
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
        self._fit = None
        self._guinier = None
        self._guinier_conc = None
        self.conc_combo.clear()
        self.part_table.setRowCount(0)
        self.part_table.setColumnCount(0)
        self.result_label.clear()
        self.status.clear()
        self.flag_label.clear()
        self.notes_block.show_notes(None)
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
        # The same blankless confirmation the Analysis tab asks (round d0ce f12). Partition
        # is this tab's entry point -- a mode-static fit refuses until one has run -- so
        # gating here covers both of its controller calls. `reuse=False`: a press of
        # Partition is a deliberate act, exactly like a press of Run.
        if not self.controller.sample_has_measured_blank(self.sample_id):
            if not _confirm_unmeasured_blank(self, self.controller, self.sample_id,
                                             reuse=False):
                return
        frac, s = sel
        try:
            self._parts = self.controller.run_mode_partition(
                self.sample_id, s.set_id, fraction=frac)
        except (ValueError, KeyError) as exc:
            set_flag(self.flag_label, humanize_field_names(str(exc)))
            self._parts = {}
            self._refresh_notes()
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
        self._refresh_notes()
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
        self._draw()

    def _refresh_notes(self) -> None:
        """Feed the notes block the fit the readout shows, then the partition the table
        shows, as one source: the fit's codes take the first letters, so the markers in
        its readout keep their letters when the Concentration selector moves."""
        c = self.conc_combo.currentData()
        part = self._parts.get(c) if c is not None else None
        notes = tuple(notes_in(self._fit)) if self._fit is not None else ()
        notes += tuple(notes_in(part)) if part is not None else ()
        self.notes_block.show_notes(notes or None)

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

        def done(res, _token) -> None:
            cur = self._selected()
            if (self.sample_id != sid or cur is None
                    or cur[1].set_id != set_id):
                return                       # superseded by a switch mid-run
            self._present_fit(res, mode, method, frac, set_id, conc=c)

        def fail(exc: BaseException, _token) -> None:
            cur = self._selected()
            if (self.sample_id != sid or cur is None
                    or cur[1].set_id != set_id):
                # Superseded by a switch mid-run (the `done` guard): the readout now
                # shows another set, so the failure is told, never painted over it.
                who = self.controller.workspace.samples.get(sid)
                QtWidgets.QMessageBox.warning(
                    self, 'Mode-resolved analysis failed',
                    f'Could not run {method!r} on {sample_display_name(who) if who else sid} '
                    f'(set {set_id}).\n\n{humanize_field_names(str(exc))}')
                return
            set_flag(self.flag_label, humanize_field_names(str(exc)))
            self.result_label.clear()
            self._fit = None
            self._refresh_notes()

        self._fit_slot.submit(work, done, fail, target=None, queue=False,
                              busy_widgets=(self.fit_button,),
                              notice=lambda: busy_notice(self),
                              description='mode-resolved analysis', paint=False)

    def _present_fit(self, res, mode: str, method: str,
                     frac: Optional[str], set_id: str,
                     conc: Optional[float] = None) -> None:
        """Show a completed mode-resolved fit. `conc` is the concentration a
        per-concentration method (amplitude Guinier, Guinier, Debye) ran at: the
        readout names it, and `_draw` flags the fit when the Concentration
        selector -- which re-fills the partition table beside it -- moves off it.

        Each row's label carries the fit's qualifier words (apparent, mode-resolved, the
        ± words beside a ± on screen); each value the notes block has a line about
        carries its marker, so the block is fed first."""
        self._parts = self.controller.results.get(
            ('mode_partition', self.sample_id, frac, set_id), self._parts)
        sig = self.controller.settings.no_uncertainty_sig_figs
        self._guinier_conc = conc          # every per-concentration method, not one
        self._fit = res
        self._refresh_notes()

        def _mark(attrs, whole: bool = True) -> str:
            m = self.notes_block.marker(res, attrs, whole=whole)
            return f' {m}' if m else ''

        if method == 'amplitude_guinier':
            self._guinier = res
            shown = ('rg_app_nm',) + (('rg_se',) if _has_pm(res.rg_app_nm, res.rg_se)
                                      else ())
            rows = [
                '{0}: <b>{1} nm</b>{2}'.format(
                    self._label('R<sub>g</sub>', res, shown, lead=(mode,)),
                    format_pm(res.rg_app_nm, res.rg_se), _mark(('rg_app_nm',))),
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
            # Mw/A₂ go through the tab's ONE dash-the-value body, exactly as the
            # Analysis sub-tab's cells do — WITH their ± via format_pm (the PDG
            # ±-sets-sig-figs policy). A ±-less value falls back to the no-uncertainty
            # sig-figs knob, never format_pm's own 4-sig-fig default. Rg is
            # slope/intercept, where the scale cancels — never dashed.
            def _pm(v: float, se) -> str:
                if has_honest_se(se):
                    return format_pm(v, se)
                return format_fixed_sig(v, sig)

            dashed = bool(_scale_causes(res))
            rows = []
            mw_attr = ('mw_g_per_mol' if hasattr(res, 'mw_g_per_mol')
                       else 'mw_apparent_g_per_mol')
            mw_se_attr = 'mw_se' if hasattr(res, 'mw_se') else 'mw_apparent_se'
            mw = getattr(res, mw_attr, float('nan'))
            mw_se = getattr(res, mw_se_attr, None)
            mw_txt = _scaled_value(
                res, ('{0} g/mol'.format(_pm(mw, mw_se))
                      if np.isfinite(mw) else 'n/a'))
            shown = (mw_attr,) + ((mw_se_attr,) if not dashed and _has_pm(mw, mw_se)
                                  else ())
            rows.append('{0}: {1}{2}'.format(
                self._label('M<sub>w</sub>', res, shown),
                mw_txt, '' if dashed else _mark((mw_attr,))))
            rg_attr = 'rg_nm' if hasattr(res, 'rg_nm') else 'rg_apparent_nm'
            rg_se_attr = 'rg_se' if hasattr(res, 'rg_se') else 'rg_apparent_se'
            rg = getattr(res, rg_attr, float('nan'))
            if np.isfinite(rg):
                rg_se = getattr(res, rg_se_attr, None)
                shown = (rg_attr,) + ((rg_se_attr,) if _has_pm(rg, rg_se) else ())
                rows.append('{0}: {1} nm{2}'.format(
                    self._label('R<sub>g</sub>', res, shown), _pm(rg, rg_se),
                    _mark((rg_attr,))))
            # The regime an Rg verdict read, with its limit -- the amplitude-Guinier rows'
            # shape, for every method whose result carries one (Guinier, Debye, Zimm/Berry);
            # a regime note marks the Rg row, and a Berry Rg and Mw past the Zimm limit say so
            # on their own labels (`_RG_MW_QUALIFIERS`).
            qrg = getattr(res, 'qrg_max', None)
            if qrg is not None and np.isfinite(qrg):
                rows.append('q·R<sub>g</sub> max = {0} (limit {1:g}){2}'.format(
                    format_fixed_sig(qrg, sig), res.qrg_max_valid,
                    _mark(('qrg_max',), whole=False)))
            a2 = getattr(res, 'a2_mol_mL_per_g2', None)
            if a2 is not None:
                a2_se = getattr(res, 'a2_se', None)
                shown = ('a2_mol_mL_per_g2',) + (
                    ('a2_se',) if not dashed and _has_pm(a2, a2_se) else ())
                rows.append('{0}: {1}{2}'.format(
                    self._label('A₂', res, shown, exclude=_RG_MW_QUALIFIERS),
                    _scaled_value(res, ('{0} mol·mL/g²'.format(_pm(a2, a2_se))
                                        if np.isfinite(a2) else 'n/a')),
                    '' if dashed else _mark(('a2_mol_mL_per_g2',))))
            if conc is not None:
                rows.append('fitted at c = {0}'.format(conc_mg_ml_label(conc)))
        # A flag down is the block's line (`tests/test_reliability_family.py::SILENT_FLAGS`);
        # the flag label keeps only a failed run's message, which a completed fit clears.
        self.flag_label.clear()
        self.result_label.setText('<br>'.join(rows) if rows else
                                  'No extractable value — see the notes.')
        self._draw()

    @staticmethod
    def _label(name: str, res, attrs: Sequence[str], *, lead: Sequence[str] = (),
               exclude: Sequence[str] = ()) -> str:
        """A readout row's label, ``name (lead, qualifier words)``: the words ``res``'s
        qualifiers give the attributes the row shows, escaped for this rich-text label;
        ``exclude`` drops codes another row's label carries."""
        words = [html.escape(w, quote=False)
                 for w in list(lead) + qualifier_words(res, attrs, exclude=exclude)]
        return f'{name} ({", ".join(words)})' if words else name

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
