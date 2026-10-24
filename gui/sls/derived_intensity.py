"""
The SLS Derived Intensity sub-tab: the selected sample's DLS runs turned into an SLS
intensity series.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from PySide6 import QtCore, QtWidgets

from core.naming import measurement_hover_card
from plotting.plots import conc_mg_ml_label
from gui.plot_controls import make_split_panels, pin_under_column
from gui.field_labels import humanize_field_names
from gui.help import section_header
from gui.notes_block import NotesBlock
from gui.plot_overlays import show_plot_placeholder
from gui.widgets import NAME_COL_MAX_W, make_plot_pane
from gui.theme import ThemedLabel, set_flag
from gui.worker import busy_notice, runner
from core.formatting import format_fixed_sig

from gui.sls.common import _EMPTY


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
    'replicate_average': ('An averaged correlogram is not a repeat measurement: its '
                          'count rate is its runs\' mean, not a measurement of its own, '
                          'so it cannot enter a static point. Tick its source runs '
                          'instead.'),
    'no_concentration': ('No concentration committed. Set one on the Data tab; a '
                         'static measurement needs one.'),
    'no_angle': ('No scattering angle. This sample\'s runs span several angles, so '
                 'the angle this trace was recorded at cannot be inferred: '
                 'double-click the Angle cell and enter it.'),
    'no_template_run': ('This sample has no DLS run for the trace to take its '
                        'identity (polymer, solvent, temperature, optics) from.'),
    'bad_angle': ('The angle entered is not a scattering angle: it must lie between '
                  '0° and 180°.'),
    'bad_concentration': ('The concentration entered is negative or not a number. '
                          'Use 0 for a solvent blank.'),
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

    UNATTACHED count-rate traces are listed too, un-ticked by default. A solvent
    blank is often measured without a correlogram (nothing to fit; the run exists
    to measure the solvent's intensity), so it reaches this route only through the
    trace it left. Such a row takes the sample's identity from the first ticked DLS
    run at derive time, its concentration defaults to 0 (a blank) and its angle is
    seeded when the sample's runs share one; both cells are editable on a trace row
    -- the one place this table is not read-only, because a trace carries neither.
    """

    workspaceChanged = QtCore.Signal()   # measurements were created

    # Short headers on purpose: spelled out, six columns overflowed the control
    # panel and pushed Status off the edge. Units follow ui_style_guide S9's
    # table convention (the degree sign, not the word). 'Run' is the name every
    # other surface shows (`core.naming`): without it six replicates at one angle
    # are six identical rows, and the run a count-rate-disagreement note names
    # cannot be found to untick. It is width-capped and left-elided (a file stem's
    # TAIL tells runs apart) so the seventh column cannot push Status off again.
    _COLS = ('Use', 'Measurement', 'Angle (°)', 'Count rate (cps)', 'Source',
             'c (mg/mL)', 'Status')
    _NAME_COL = 1
    _NAME_MAX_W = NAME_COL_MAX_W      # the shared cap; see gui/widgets.py

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
        # Unattached traces are OPT-IN: a trace belongs to no sample, so every
        # sample lists every one, and auto-ticking it would derive a stray blank
        # into whichever sample was selected. Ticked ids are remembered here.
        self._ticked_traces: set = set()
        # The user's own angle / concentration for a trace row, by trace id --
        # the two cells a trace carries no value for (see the class docstring).
        self._trace_params: Dict[str, Dict[str, Any]] = {}
        self._building = False
        self.ax = None
        self._build_ui()
        self._refresh()

    # ---------------------------------------------------------------- UI ---
    def _build_ui(self) -> None:
        # A wide default column, so the seven columns fit with the name capped and
        # Status stays on the panel at the shipped window size. The splitter scales these
        # sizes to the window, hence the proportion rather than the plot's usual 700.
        splitter, left, right = make_split_panels(self, left_min_width=420,
                                                  sizes=(530, 440))
        left.addWidget(section_header(
            'Derived Intensity',
            'Build an SLS intensity series from this sample\'s DLS runs '
            '(User-Manual §9.2).',
            bullets=[
                'Each run\'s mean count rate (set on the Data tab, read-only here) '
                'is its intensity at its angle.',
                'Runs sharing an angle and concentration are averaged; one '
                'measurement per concentration.',
                'Derived points are ordinary SLS measurements: analyze them on the '
                '<b>Analysis</b> sub-tab.',
                'Intensities are in cps: enter the calibrant intensity and any '
                'solvent blank in cps too.',
                'A derived point is a snapshot; <b>Derive</b> again to refresh it.',
                'A trace with no run (a blank recorded without a correlogram) is '
                'listed un-ticked; tick it to use its mean, and double-click its Angle '
                'or c cell to edit.',
            ]))

        self.header = QtWidgets.QLabel()
        self.header.setWordWrap(True)
        left.addWidget(self.header)

        self.table = QtWidgets.QTableWidget(0, len(self._COLS))
        self.table.setHorizontalHeaderLabels(list(self._COLS))
        self.table.verticalHeader().setVisible(False)
        # Editing is per CELL, not per table: every item is built non-editable
        # (`_refresh` clears ItemIsEditable) except a trace row's Angle and c
        # cells, so the read-only promise the help badge makes holds everywhere
        # a value has an authoritative home elsewhere.
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
            | QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed)
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.table.horizontalHeader().setStretchLastSection(True)
        # A capped name column elides rather than truncating the table: from the
        # LEFT, so the tail that distinguishes 'batch 1-3' from 'batch 1-6' survives.
        # Word wrap off, or Qt wraps the name and right-elides its first line instead.
        self.table.setTextElideMode(QtCore.Qt.TextElideMode.ElideLeft)
        self.table.setWordWrap(False)
        self.table.itemChanged.connect(self._on_cell_edited)
        left.addWidget(self.table, 1)

        tick_row = QtWidgets.QHBoxLayout()
        self.all_button = QtWidgets.QPushButton('Select all')
        self.all_button.setToolTip(
            'Tick every usable DLS run. Unattached traces stay as they are -- '
            'tick a trace yourself to use it.')
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
            'The intensities are count rates in cps, so the calibrant intensity '
            'and any solvent blank used with them must be in cps too; the program '
            'cannot detect a mismatch, and a kcps calibrant scales Mw by 1000. Nor '
            'can it check that the attenuation and laser power matched across runs, '
            'or that the detector stayed linear (Theory-and-Equations-Guide §4.1).')
        self.derive_button.clicked.connect(self._on_derive)

        self.status = QtWidgets.QLabel('')
        self.status.setWordWrap(True)
        self.flag_label = ThemedLabel('', role='error', bold=True)
        self.flag_label.setWordWrap(True)
        # The derived series' preconditions show here (the count-rate spread, flagless, is
        # the instance); every note, records included (the runs averaged, the unit),
        # travels in the derived point's stored record to the export Comments.
        self.notes_block = NotesBlock(what='concentrations')
        # Derive, the status, the notes and the flag pinned under the column (§3 column
        # template, R8.6).
        pin_under_column(splitter, self.derive_button, self.status, self.notes_block,
                         self.flag_label)

        self.figure, self.canvas, self.nav_toolbar, self.axis_bar = make_plot_pane(
            self, self.controller, (5.5, 4.6))
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
            self._ticked_traces.clear()
            self._trace_params.clear()
            self.status.clear()
            self.notes_block.show_notes(None)
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
        """Remember an exclusion (a run) or an inclusion (a trace) so the next
        refresh does not undo it."""
        if item_id in self.controller.workspace.traces:
            (self._ticked_traces.add if checked
             else self._ticked_traces.discard)(item_id)
        elif checked:
            self._excluded.discard(item_id)
        else:
            self._excluded.add(item_id)

    def _set_all(self, checked: bool) -> None:
        for row, info in enumerate(self._rows):
            widget = self.table.cellWidget(row, 0)
            if info.get('kind') == 'trace' and checked:
                continue                            # traces are opt-in only
            if widget is not None and widget.isEnabled():
                widget.setChecked(checked)          # drives _on_row_toggled
            elif not checked:
                self._excluded.add(info['item_id'])

    def _on_cell_edited(self, item) -> None:
        """A trace row's Angle or c cell was edited: keep the value and re-list."""
        if self._building:
            return
        row, col = item.row(), item.column()
        if row >= len(self._rows) or self._rows[row].get('kind') != 'trace':
            return
        key = {2: 'angle_deg', 5: 'concentration_g_per_mL'}.get(col)
        if key is None:
            return
        text = item.text().strip()
        params = self._trace_params.setdefault(self._rows[row]['item_id'], {})
        try:
            value = float(text) if text else None
        except ValueError:
            value = None
        if key == 'concentration_g_per_mL' and value is not None:
            value = value / 1000.0                  # the cell is in mg/mL
        params[key] = value
        # Rebuild after the editor has closed, not from inside its commit.
        QtCore.QTimer.singleShot(0, self._refresh)

    # --------------------------------------------------------- display ---
    def _refresh(self) -> None:
        self._rows = (self.controller.derived_intensity_candidates(
                          self.sample_id, self._trace_params)
                      if self.sample_id else [])
        self._building = True
        try:
            self._fill_table()
        finally:
            self._building = False
        self.table.resizeColumnsToContents()
        if self.table.columnWidth(self._NAME_COL) > self._NAME_MAX_W:
            self.table.setColumnWidth(self._NAME_COL, self._NAME_MAX_W)

        runs = [r for r in self._rows if r.get('kind') != 'trace']
        traces = [r for r in self._rows if r.get('kind') == 'trace']
        if not self.sample_id:
            self.header.setText('No sample selected.')
        elif not self._rows:
            self.header.setText(
                'This sample has no DLS runs, so there is nothing to derive from.')
        else:
            n_ok = sum(1 for r in runs if r['eligible'])
            text = ('{0} DLS run(s) in this sample; {1} can supply a static '
                    'point.'.format(len(runs), n_ok))
            if traces:
                text += (' {0} unattached trace(s) listed too (a blank recorded '
                         'without a correlogram, typically): tick one to use its '
                         'mean; its Angle and c cells are editable.'.format(
                             len(traces)))
            if any(not r['eligible'] for r in self._rows):
                text += (' A row marked "not usable" says what it needs in its '
                         'tooltip (turn on "Show tooltips on hover" under Settings '
                         'if none appears).')
            self.header.setText(text)
        self.derive_button.setEnabled(
            any(r['eligible'] for r in self._rows))
        self._draw()

    def _fill_table(self) -> None:
        """Rebuild the rows from `_rows` (called with itemChanged muted)."""
        self.table.setRowCount(len(self._rows))
        for row, info in enumerate(self._rows):
            is_trace = info.get('kind') == 'trace'
            box = QtWidgets.QCheckBox()
            box.setEnabled(bool(info['eligible']))
            if is_trace:
                box.setChecked(bool(info['eligible'])
                               and info['item_id'] in self._ticked_traces)
            else:
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
                info['label'],
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
            lm = (None if is_trace else
                  self.controller.workspace.measurements.get(info['item_id']))
            for col, text in enumerate(cells, start=1):
                item = QtWidgets.QTableWidgetItem(text)
                editable = is_trace and col in (2, 5)
                if not editable:
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                # The reason rides as a tooltip on every cell of the row rather
                # than as column text: spelled out in the cell it widened the
                # table past the panel and pushed Status off the edge entirely.
                # The name cell is the exception: it carries the identity card
                # every name in the program carries, its own full text first.
                if col == self._NAME_COL and lm is not None:
                    card = measurement_hover_card(lm)
                    item.setToolTip(f'{card}\n\n{reason}' if reason else card)
                elif reason:
                    item.setToolTip(reason)
                elif editable:
                    item.setToolTip(
                        'Double-click to edit: a trace carries no angle or '
                        'concentration of its own (c is in mg/mL; 0 = a blank).')
                self.table.setItem(row, col, item)

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
                item_ids, dn_dc_mL_per_g=dn_dc,
                trace_params={i: self._trace_params.get(i, {}) for i in item_ids
                              if i in self.controller.workspace.traces},
                template_sample_id=self.sample_id)
        except ValueError as exc:
            set_flag(self.flag_label, humanize_field_names(str(exc)))
            self.status.clear()
            return
        self.flag_label.clear()
        self._series = out['series']
        status = ('Created {0} SLS measurement(s) from {1} run(s). Calibrate and '
                  'analyze them on the Analysis sub-tab.'.format(
                      len(out['item_ids']), len(item_ids)))
        # The controller's own sentence for the earlier derived points this derive
        # replaced, its last note: an event of this press, so it rides the status line.
        if out.get('superseded') and out['notes']:
            status += ' ' + out['notes'][-1]
        self.status.setText(status)
        self.notes_block.show_notes(
            [(_EMPTY if s.concentration_g_per_mL is None
              else conc_mg_ml_label(s.concentration_g_per_mL), s) for s in out['series']])
        self._draw()
        self.workspaceChanged.emit()
