"""
gui/cross_module.py
===================

The Cross-Sample tab: the first AGGREGATE-scope module. Where the sample-scoped
tabs (Data/DLS/SLS) operate on the one measurement picked in the shell sidebar,
this tab reads results ACROSS samples: the ρ = Rg/Rh pairing and the log–log
scaling plots (Rg–Mw, A₂–Mw) are both built here.

Layout
------
* Left  — an include/exclude list of every sample with SLS data (the Cross-Sample
  universe: ρ also needs DLS, the scaling plots need Mw + Rg). All start included;
  click a sample to load it into the source panel; untick to exclude it from the
  views. This lives inside the tab (not the shell tree) so the module stays
  self-contained and headless-testable, and the shell's single-selection navigator
  is not overloaded with an aggregate selection model.
* Right — inner tabs: **ρ = Rg/Rh** (a table, one row per included sample that can
  pair ρ) and **Scaling** (log–log Rg–Mw and A₂–Mw plots over the included
  samples). Beneath them, a shared **source panel** for the selected sample: which
  Rg (SLS), Rh (DLS), and Mw (SLS) feed the analyses — each a labelled default the
  user can override or replace with a hand-entered value. ρ uses Rg/Rh; the scaling
  plots use Mw and Rg/A₂.

All numbers and provenance come from the controller (`compute_sample_rho`,
`sls_rg_candidates`, `dls_rh_candidates`, the auto/select/manual setters). No
analysis or physics here — the tab only displays and chooses.

Apparent vs thermodynamic is shown per row: ρ is "apparent" if either Rg or Rh is
an apparent (single-condition) value, never silently mixed with a thermodynamic
one.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from PySide6 import QtCore, QtWidgets

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from plotting.plots import plot_scaling
from gui.plot_controls import make_canvas_expanding
from gui.export_helper import export_to_csv
from gui.help import section_header
from gui.theme import ThemedLabel
from gui.widgets import roomy_tabs
from gui.worker import busy_notice, run_when_idle, runner
from analysis.utilities import interpret_scaling_exponent
from analysis.uncertainty import format_pm
from app import units as U


# ρ-table columns. Rg/Rh carry a unit that follows the global Display-units setting;
# `_columns()` inserts the active label, so this template holds only the fixed parts.
_COL_TEMPLATE = ['Sample', 'Rg', 'Rh', 'ρ = Rg/Rh', 'Shape', 'Type']
_MANUAL_LABEL = 'Manual entry…'


def _fmt(x: Optional[float], sig: int = 3) -> str:
    if x is None or not (isinstance(x, (int, float)) and math.isfinite(x)):
        return 'n/a'
    return f'{x:.{sig}g}'


def _sample_label(sample) -> str:
    poly, solv, temp = sample.polymer_name, sample.solvent_name, sample.temperature_K
    if poly and solv and temp is not None and not math.isnan(temp):
        return f'{poly} / {solv} @ {temp:g} K'
    return '(unconfirmed sample)'


class CrossSampleModule(QtWidgets.QWidget):
    """Aggregate ρ = Rg/Rh across samples, with per-sample source selection."""

    def __init__(self, controller, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self._included: Dict[str, bool] = {}     # sample_id -> included?
        # A "unit" is a (sample_id, fraction) pair: a Mw series is several units of
        # one sample. Rows/points/source-panel all operate on units.
        self._row_units: List[Tuple[str, Optional[str]]] = []
        self._current_unit: Optional[Tuple[str, Optional[str]]] = None
        self._size_quantity = 'rg'               # top scaling plot: 'rg' or 'rh'
        self._suppress = False                   # guard signal storms on rebuild
        # Hand-entered Rg/Rh/Mw typed while the worker was busy, keyed by which,
        # applied (re-checking) once it frees — a precious value is never dropped.
        self._pending_manual: Dict[str, tuple] = {}
        self._build_ui()
        self.refresh()

    # ------------------------------------------------------------------ UI ---
    def _build_ui(self) -> None:
        outer = QtWidgets.QHBoxLayout(self)

        # ---- left: include / exclude --------------------------------------
        left = QtWidgets.QVBoxLayout()
        outer.addLayout(left, 0)
        left.addWidget(section_header(
            'Samples (include / exclude)',
            'Compare results across samples:',
            bullets=[
                'Tick the samples to include in the ρ table and scaling plots.',
                '<b>ρ = Rg/Rh</b> hints at shape (≈0.78 sphere, ≈1.0 coil, &gt;1.7 '
                'rod) — needs both an SLS Rg and a DLS Rh per sample.',
                '<b>Scaling</b> fits Rg–Mw and A₂–Mw power laws across the included '
                'samples.',
                'Per-sample, choose which Rg / Rh / Mw value to use in the source '
                'panel below.',
            ]))
        self.sample_list = QtWidgets.QListWidget()
        self.sample_list.setMinimumWidth(240)
        self.sample_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.sample_list.itemSelectionChanged.connect(self._on_selection_changed)
        self.sample_list.currentItemChanged.connect(self._on_list_current)
        left.addWidget(self.sample_list, 1)
        sel_row = QtWidgets.QHBoxLayout()
        btn_sel_all = QtWidgets.QPushButton('Select all')
        btn_sel_none = QtWidgets.QPushButton('Select none')
        sel_row.addWidget(btn_sel_all)
        sel_row.addWidget(btn_sel_none)
        sel_row.addStretch(1)
        left.addLayout(sel_row)
        btn_sel_all.clicked.connect(self._select_all_samples)
        btn_sel_none.clicked.connect(self._select_none_samples)
        self.left_note = ThemedLabel('', role='hint', size=11)
        self.left_note.setWordWrap(True)
        left.addWidget(self.left_note)

        # ---- right: inner tabs (rho | scaling) + a shared source panel -----
        right = QtWidgets.QVBoxLayout()
        outer.addLayout(right, 1)

        self.inner = roomy_tabs(QtWidgets.QTabWidget())   # roomier tabs so labels don't clip (#3)
        self.inner.setMovable(True)              # drag to reorder sub-tabs (A4)
        right.addWidget(self.inner, 1)

        # rho tab: the rho table + interpretation
        rho_tab = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(rho_tab)
        self.table = QtWidgets.QTableWidget(0, len(_COL_TEMPLATE))
        self.table.setHorizontalHeaderLabels(self._columns())
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        rl.addWidget(self.table, 1)
        self.interp = ThemedLabel('', role='muted', size=11)
        self.interp.setWordWrap(True)
        rl.addWidget(self.interp)
        self.inner.addTab(rho_tab, 'ρ = Rg/Rh')

        # scaling tab: two log-log plots (size-Mw on top, A2-Mw below) + a button
        # that swaps the top plot's size axis between Rg and Rh.
        scaling_tab = QtWidgets.QWidget()
        sl = QtWidgets.QVBoxLayout(scaling_tab)
        swap_row = QtWidgets.QHBoxLayout()
        self.swap_size_button = QtWidgets.QPushButton('Show Rh–Mw')
        self.swap_size_button.setToolTip(
            'Swap the top scaling plot between Rg–Mw (SLS) and Rh–Mw (DLS).')
        self.swap_size_button.clicked.connect(self._on_swap_size)
        swap_row.addWidget(self.swap_size_button)
        self.scaling_export_button = QtWidgets.QPushButton('Export CSV…')
        self.scaling_export_button.setToolTip(
            'Export the current size–Mw and A₂–Mw scaling points + fits as CSV.')
        self.scaling_export_button.clicked.connect(self._on_export_scaling)
        swap_row.addWidget(self.scaling_export_button)
        swap_row.addStretch(1)
        sl.addLayout(swap_row)
        self.scaling_fig = Figure(figsize=(4.3, 4.8))
        self.scaling_canvas = make_canvas_expanding(FigureCanvas(self.scaling_fig))
        self.ax_rg = self.scaling_fig.add_subplot(2, 1, 1)
        self.ax_a2 = self.scaling_fig.add_subplot(2, 1, 2)
        sl.addWidget(self.scaling_canvas, 1)
        self.scaling_note = ThemedLabel('', role='muted', size=11)
        self.scaling_note.setWordWrap(True)
        sl.addWidget(self.scaling_note)
        self.inner.addTab(scaling_tab, 'Scaling (Rg–Mw, A₂–Mw)')
        self.inner.currentChanged.connect(lambda _i: self._refresh_scaling())

        right.addWidget(self._build_source_panel())

    def _build_source_panel(self) -> QtWidgets.QWidget:
        self.source_box = QtWidgets.QGroupBox('Source selection (selected sample)')
        grid = QtWidgets.QGridLayout(self.source_box)

        # Rg row: a combo of SLS candidates + Manual; a manual entry beside it.
        grid.addWidget(QtWidgets.QLabel('Rg (SLS):'), 0, 0)
        self.rg_combo = QtWidgets.QComboBox()
        self.rg_combo.activated.connect(lambda i: self._on_source_chosen('rg', i))
        grid.addWidget(self.rg_combo, 0, 1)
        self.rg_manual = QtWidgets.QLineEdit()
        self.rg_manual.setPlaceholderText(self._radius_unit())
        self.rg_manual.setFixedWidth(70)
        grid.addWidget(self.rg_manual, 0, 2)
        self.rg_manual_apparent = QtWidgets.QCheckBox('apparent')
        grid.addWidget(self.rg_manual_apparent, 0, 3)
        rg_set = QtWidgets.QPushButton('Set')
        rg_set.clicked.connect(lambda: self._on_manual('rg'))
        grid.addWidget(rg_set, 0, 4)

        # Rh row: a combo of DLS candidates + Manual; a manual entry beside it.
        grid.addWidget(QtWidgets.QLabel('Rh (DLS):'), 1, 0)
        self.rh_combo = QtWidgets.QComboBox()
        self.rh_combo.activated.connect(lambda i: self._on_source_chosen('rh', i))
        grid.addWidget(self.rh_combo, 1, 1)
        self.rh_manual = QtWidgets.QLineEdit()
        self.rh_manual.setPlaceholderText(self._radius_unit())
        self.rh_manual.setFixedWidth(70)
        grid.addWidget(self.rh_manual, 1, 2)
        self.rh_manual_apparent = QtWidgets.QCheckBox('apparent')
        grid.addWidget(self.rh_manual_apparent, 1, 3)
        rh_set = QtWidgets.QPushButton('Set')
        rh_set.clicked.connect(lambda: self._on_manual('rh'))
        grid.addWidget(rh_set, 1, 4)

        # Mw row (feeds the scaling plots): SLS candidates + Manual. No "apparent"
        # box -- a hand-entered Mw is treated as a trusted (calibrated) value.
        grid.addWidget(QtWidgets.QLabel('Mw (SLS):'), 2, 0)
        self.mw_combo = QtWidgets.QComboBox()
        self.mw_combo.activated.connect(lambda i: self._on_source_chosen('mw', i))
        grid.addWidget(self.mw_combo, 2, 1)
        self.mw_manual = QtWidgets.QLineEdit()
        self.mw_manual.setPlaceholderText(self._mw_unit())
        self.mw_manual.setFixedWidth(70)
        grid.addWidget(self.mw_manual, 2, 2)
        mw_set = QtWidgets.QPushButton('Set')
        mw_set.clicked.connect(lambda: self._on_manual('mw'))
        grid.addWidget(mw_set, 2, 4)

        grid.setColumnStretch(1, 1)
        self.source_box.setEnabled(False)
        return self.source_box

    # ------------------------------------------------------------- refresh ---
    def refresh(self) -> None:
        """Rebuild from the workspace. Called by the shell when this tab is shown
        or after a commit (the sample set or its results may have changed)."""
        # refresh() runs the labelled Rg/Rh/Mw/A2 auto-picks, which WRITE
        # SampleResult fields — so it must not run while a background fit is
        # writing them too (invariant 4). Defer until the worker frees.
        if runner().is_busy:
            run_when_idle(self.refresh)
            return
        universe = self.controller.samples_with_sls()
        # default: every SLS sample is included; remember prior choices.
        self._included = {sid: self._included.get(sid, True) for sid in universe}

        self._suppress = True
        self.sample_list.clear()
        by_id = {s.sample_id: s for s in self.controller.samples()}
        for sid in universe:
            item = QtWidgets.QListWidgetItem(_sample_label(by_id[sid]))
            item.setData(QtCore.Qt.ItemDataRole.UserRole, sid)
            self.sample_list.addItem(item)
            item.setSelected(self._included[sid])
        self._suppress = False

        if not universe:
            self.left_note.setText(
                'No sample has SLS data yet. The Cross-Sample analyses need SLS: '
                'ρ also needs DLS (for Rh); the scaling plots need Mw and Rg.')
        else:
            self.left_note.setText(
                f'{len(universe)} sample(s) with SLS. Untick to exclude; click a '
                'sample to edit its sources below.')

        # Default Rg / Rh / Mw for each included sample's fractions (labelled; never
        # clobbers a hand-entered value). Rh is a no-op for an SLS-only sample.
        for sid, inc in self._included.items():
            if inc:
                self._auto_select_all(sid)

        self._recompute()

        # point the source panel at a unit (keep the current one if still valid)
        cur_sid = self._current_unit[0] if self._current_unit else None
        if not self._included.get(cur_sid):
            self._current_unit = None
        target = cur_sid if self._included.get(cur_sid) else (
            universe[0] if universe else None)
        if target is not None:
            self._select_list_sample(target)
        else:
            self.source_box.setEnabled(False)
            self.interp.setText('')

    def _auto_select_all(self, sid: str) -> None:
        """Run the labelled default Rg/Rh/Mw/A2 picks for every fraction of a sample
        (the union of its SLS and DLS fraction labels)."""
        fracs = set(self.controller.sample_fractions(sid, 'sls'))
        fracs |= set(self.controller.sample_fractions(sid, 'dls'))
        for frac in fracs:
            self.controller.auto_select_rg(sid, frac)
            self.controller.auto_select_rh(sid, frac)
            self.controller.auto_select_mw(sid, frac)
            self.controller.auto_select_a2(sid, frac)

    def _recompute(self) -> None:
        """Refresh both computed views (ρ table + scaling plots)."""
        self._rebuild_table()
        self._refresh_scaling()

    # ----- display units (follow the global Settings "Display units" choice) -----
    def _radius_unit(self) -> str:
        """The active display unit for Rg/Rh (nm default), from the global setting."""
        return ((self.controller.settings.plot_units or {}).get('radius')
                or U.default_unit('radius'))

    def _mw_unit(self) -> str:
        """The active display unit for Mw (g/mol default), from the global setting."""
        return ((self.controller.settings.plot_units or {}).get('molar_mass')
                or U.default_unit('molar_mass'))

    def _columns(self) -> list:
        """ρ-table headers with the active Rg/Rh unit label."""
        ru = self._radius_unit()
        cols = list(_COL_TEMPLATE)
        cols[1], cols[2] = f'Rg ({ru})', f'Rh ({ru})'
        return cols

    def _disp_radius(self, x: Optional[float], runit: str) -> str:
        """Format a canonical (nm) radius in the active display unit; 'n/a' if missing."""
        if x is None or not (isinstance(x, (int, float)) and math.isfinite(x)):
            return 'n/a'
        return _fmt(U.from_canonical('radius', x, runit))

    def _unit_label(self, sid: str, fraction: Optional[str]) -> str:
        by_id = {s.sample_id: s for s in self.controller.samples()}
        base = _sample_label(by_id[sid])
        return f'{base} — {fraction}' if fraction is not None else base

    def _rebuild_table(self) -> None:
        self._suppress = True
        # Re-apply headers so a Display-units change updates the Rg/Rh unit labels.
        self.table.setHorizontalHeaderLabels(self._columns())
        runit = self._radius_unit()
        samples = self.controller.workspace.samples
        # One ρ row per (sample, fraction): only included samples that can pair ρ
        # (have both DLS and SLS), expanded over each sample's fraction labels.
        self._row_units = []
        for sid, inc in self._included.items():
            if inc and samples[sid].can_pair_rho:
                for frac in self.controller.sample_fractions(sid, 'sls'):
                    self._row_units.append((sid, frac))
        self.table.setRowCount(len(self._row_units))
        for row, (sid, frac) in enumerate(self._row_units):
            label = self._unit_label(sid, frac)
            try:
                rho = self.controller.compute_sample_rho(sid, frac)
                rho_text = (format_pm(rho.rho, rho.rho_se)
                            if rho.rho_se is not None else _fmt(rho.rho))
                est_note = (' [SE: classical OLS]'
                            if rho.se_estimator == 'ols' else '')
                rho_tip = (rho.interpretation if rho.rho_se is None else
                           rho.interpretation + '\n(± = statistical SE, from the '
                           'Rg and Rh regression fits; excludes systematics.'
                           + est_note + ')')
                cells = [label,
                         self._disp_radius(rho.rg_nm, runit),
                         self._disp_radius(rho.rh_nm, runit),
                         rho_text, rho.shape,
                         'apparent' if rho.is_apparent else 'thermodynamic']
                tips = [rho.interpretation, rho.rg_label, rho.rh_label,
                        rho_tip, rho.interpretation, '']
            except Exception as exc:                 # missing Rg/Rh, failed fit…
                cells = [label, 'n/a', 'n/a', '—', '—', '']
                tips = [str(exc)] * len(cells)
            for col, text in enumerate(cells):
                cell = QtWidgets.QTableWidgetItem(text)
                if tips[col]:
                    cell.setToolTip(tips[col])
                self.table.setItem(row, col, cell)
        self._suppress = False

    _SCALE_NAMES = {'rg': 'Rg–Mw', 'rh': 'Rh–Mw', 'a2': 'A₂–Mw'}
    _SCALE_NEEDS = {'rg': 'Rg', 'rh': 'Rh', 'a2': 'positive A₂'}

    def _on_swap_size(self) -> None:
        """Toggle the top scaling plot between Rg–Mw and Rh–Mw."""
        self._size_quantity = 'rh' if self._size_quantity == 'rg' else 'rg'
        self.swap_size_button.setText(
            'Show Rg–Mw' if self._size_quantity == 'rh' else 'Show Rh–Mw')
        self._refresh_scaling()

    def _refresh_scaling(self) -> None:
        included = [sid for sid, inc in self._included.items() if inc]
        notes = []
        # Top plot is the chosen size axis (Rg or Rh); bottom is always A2.
        for ax, q in ((self.ax_rg, self._size_quantity), (self.ax_a2, 'a2')):
            ax.clear()
            sd = self.controller.compute_scaling(included, q)
            plot_scaling(sd.fit, q, sd.labels, ax=ax)
            name = self._SCALE_NAMES[q]
            if sd.fit.fit_valid:
                # 'ν' is the size exponent (Rg/Rh ~ Mw^ν); for A₂ the log–log slope
                # is the raw exponent (A₂ ~ Mw^slope, negative in a good solvent).
                sym = 'slope' if q == 'a2' else 'ν'
                exp_text = format_pm(sd.fit.exponent, sd.fit.exponent_se)
                msg = f'{name}: {sym} = {exp_text}, {len(sd.mw)} points'
                if sd.n_excluded:
                    msg += f' ({sd.n_excluded} lacked Mw or a positive value)'
                # Interpretation hint, mirroring the ρ table (feedback B8).
                msg += '\n    → ' + interpret_scaling_exponent(sd.fit.exponent, q)
            else:
                msg = (f'{name}: need ≥2 points with Mw + {self._SCALE_NEEDS[q]} '
                       'to fit a slope')
            if sd.any_uncalibrated_mw:
                msg += ' — includes an UNCALIBRATED Mw (arbitrary scale)'
            notes.append(msg)
        self.scaling_fig.tight_layout()
        self.scaling_canvas.draw_idle()
        self.scaling_note.setText('\n'.join(notes))

    @QtCore.Slot()
    def _on_export_scaling(self) -> None:
        included = [sid for sid, inc in self._included.items() if inc]
        if not included:
            self.scaling_note.setText('Include at least one sample to export.')
            return

        def do_export(path: str) -> str:
            import os
            stem, ext = os.path.splitext(path)
            written = []
            for q in (self._size_quantity, 'a2'):
                sd = self.controller.compute_scaling(included, q)
                written.append(self.controller.export_scaling(
                    sd, f'{stem}_{q}{ext}'))
            return '; '.join(written)

        status = export_to_csv(self, 'scaling.csv', do_export)
        if status:
            self.scaling_note.setText(status)

    # ----------------------------------------------------------- callbacks ---
    def _select_all_samples(self) -> None:
        self.sample_list.selectAll()   # triggers _on_selection_changed

    def _select_none_samples(self) -> None:
        self.sample_list.clearSelection()   # triggers _on_selection_changed

    @QtCore.Slot()
    def _on_selection_changed(self) -> None:
        if self._suppress:
            return
        newly_included = []
        for i in range(self.sample_list.count()):
            it = self.sample_list.item(i)
            sid = it.data(QtCore.Qt.ItemDataRole.UserRole)
            now = it.isSelected()
            if now and not self._included.get(sid, False):
                newly_included.append(sid)
            self._included[sid] = now
        for sid in newly_included:
            self._auto_select_all(sid)
        self._recompute()

    @QtCore.Slot(QtWidgets.QListWidgetItem, QtWidgets.QListWidgetItem)
    def _on_list_current(self, current, _previous) -> None:
        if self._suppress or current is None:
            return
        sid = current.data(QtCore.Qt.ItemDataRole.UserRole)
        fracs = self.controller.sample_fractions(sid, 'sls')
        self._populate_source_panel(sid, fracs[0] if fracs else None)

    def _select_list_sample(self, sid: str,
                            fraction: Optional[str] = '__first__') -> None:
        """Highlight a sample in the list and load a unit into the source panel.
        With fraction left at the sentinel, the sample's first SLS fraction is used."""
        for i in range(self.sample_list.count()):
            it = self.sample_list.item(i)
            if it.data(QtCore.Qt.ItemDataRole.UserRole) == sid:
                self._suppress = True
                self.sample_list.setCurrentItem(it)
                self._suppress = False
                break
        if fraction == '__first__':
            fracs = self.controller.sample_fractions(sid, 'sls')
            fraction = fracs[0] if fracs else None
        self._populate_source_panel(sid, fraction)

    @QtCore.Slot()
    def _on_row_selected(self) -> None:
        if self._suppress:
            return
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        sid, frac = self._row_units[rows[0].row()]
        self._select_list_sample(sid, frac)

    def _populate_source_panel(self, sid: str, fraction: Optional[str]) -> None:
        self._current_unit = (sid, fraction)
        # Manual-entry placeholders follow the active Display-units choice.
        self.rg_manual.setPlaceholderText(self._radius_unit())
        self.rh_manual.setPlaceholderText(self._radius_unit())
        self.mw_manual.setPlaceholderText(self._mw_unit())
        r = self.controller.workspace.samples[sid].result_for(fraction)
        self._fill_combo(self.rg_combo, self.controller.sls_rg_candidates(sid, fraction),
                         current_label=r.rg_label, current_source=r.rg_source,
                         quantity='radius')
        self._fill_combo(self.rh_combo, self.controller.dls_rh_candidates(sid, fraction),
                         current_label=r.rh_label, current_source=r.rh_source,
                         quantity='radius')
        self._fill_combo(self.mw_combo, self.controller.sls_mw_candidates(sid, fraction),
                         current_label=r.mw_label, current_source=r.mw_source,
                         quantity='molar_mass')
        self.source_box.setEnabled(True)
        self.source_box.setTitle(f'Source selection — {self._unit_label(sid, fraction)}')
        try:
            rho = self.controller.compute_sample_rho(sid, fraction)
            flag = ('  [apparent ρ — at least one input is a single-condition '
                    'value]' if rho.is_apparent else '')
            self.interp.setText(f'{rho.interpretation}{flag}')
        except Exception as exc:
            self.interp.setText(str(exc))

    def _fill_combo(self, combo: QtWidgets.QComboBox, candidates, *,
                    current_label: str, current_source: str,
                    quantity: str = 'radius') -> None:
        """Rebuild a source-picker combo. Each item's UserRole data is a sentinel
        that `_on_source_chosen` dispatches on: a ResultCandidate object (a real
        result to select), the string 'USER' (keep the existing hand-entered
        value), or None (the trailing "Manual entry…" item). Signals are blocked
        during the rebuild so clearing/adding items does not fire currentIndexChanged
        (which would re-enter selection handling mid-rebuild). `quantity` picks the
        active display unit for the "User-entered" item (candidate labels carry no
        numeric value, so only that item needs converting)."""
        unit = (self._mw_unit() if quantity == 'molar_mass' else self._radius_unit())
        combo.blockSignals(True)
        combo.clear()
        select_index = -1
        for i, c in enumerate(candidates):
            combo.addItem(c.label)
            combo.setItemData(i, c, QtCore.Qt.ItemDataRole.UserRole)
            if current_source != 'user' and c.label == current_label:
                select_index = i
        # a hand-entered value shows as a distinct, pre-selected item (in display units)
        if current_source == 'user':
            val = self._user_value(combo)
            disp = (U.from_canonical(quantity, val, unit)
                    if isinstance(val, (int, float)) and math.isfinite(val) else val)
            combo.addItem(f'User-entered ({_fmt(disp)} {unit})')
            combo.setItemData(combo.count() - 1, 'USER', QtCore.Qt.ItemDataRole.UserRole)
            select_index = combo.count() - 1
        combo.addItem(_MANUAL_LABEL)
        combo.setItemData(combo.count() - 1, None, QtCore.Qt.ItemDataRole.UserRole)
        if select_index < 0 and combo.count() > 1:
            select_index = 0
        combo.setCurrentIndex(max(select_index, 0))
        combo.blockSignals(False)

    def _user_value(self, combo: QtWidgets.QComboBox) -> Optional[float]:
        if self._current_unit is None:
            return None
        sid, frac = self._current_unit
        r = self.controller.workspace.samples[sid].result_for(frac)
        if combo is self.rg_combo:
            return r.rg_nm
        if combo is self.mw_combo:
            return r.mw_g_per_mol
        return r.rh_nm

    def _combo(self, which: str) -> QtWidgets.QComboBox:
        return {'rg': self.rg_combo, 'rh': self.rh_combo, 'mw': self.mw_combo}[which]

    def _on_source_chosen(self, which: str, index: int) -> None:
        if self._current_unit is None:
            return
        sid, frac = self._current_unit
        data = self._combo(which).itemData(index, QtCore.Qt.ItemDataRole.UserRole)
        if data is None:                 # "Manual entry…" — let the user type + Set
            {'rg': self.rg_manual, 'rh': self.rh_manual,
             'mw': self.mw_manual}[which].setFocus()
            return
        if data == 'USER':               # keep the existing hand-entered value
            return
        if runner().is_busy:             # set_sample_* writes SampleResult a fit reads
            busy_notice(self)
            self._populate_source_panel(sid, frac)   # revert the combo to committed
            return
        setter = {'rg': self.controller.set_sample_rg,
                  'rh': self.controller.set_sample_rh,
                  'mw': self.controller.set_sample_mw}[which]
        setter(sid, data, frac)
        self._recompute()
        self._populate_source_panel(sid, frac)

    def _on_manual(self, which: str) -> None:
        if self._current_unit is None:
            return
        sid, frac = self._current_unit
        edit = {'rg': self.rg_manual, 'rh': self.rh_manual, 'mw': self.mw_manual}[which]
        text = edit.text().strip()
        if not text:
            return
        # The user types in the active display unit; store canonical (nm / g·mol⁻¹).
        quantity = 'molar_mass' if which == 'mw' else 'radius'
        unit = self._mw_unit() if which == 'mw' else self._radius_unit()
        try:
            value = U.to_canonical(quantity, float(text), unit)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, 'Invalid value', str(exc))
            return
        apparent = (None if which == 'mw' else
                    (self.rg_manual_apparent if which == 'rg'
                     else self.rh_manual_apparent).isChecked())
        if runner().is_busy:
            # set_manual_* writes SampleResult, which a background fit is reading —
            # but a hand-entered value is precious and must not be dropped. Queue
            # it and apply once the worker frees (re-checking, never concurrently).
            self._pending_manual[which] = (sid, frac, value, apparent)
            run_when_idle(self._flush_manual)
            self.interp.setText(
                'Entry queued — it will apply when the running analysis finishes.')
            return
        self._apply_manual(which, sid, frac, value, apparent)
        edit.clear()

    def _apply_manual(self, which, sid, frac, value, apparent) -> None:
        if which == 'mw':
            self.controller.set_manual_mw(sid, value, frac)
        else:
            setter = (self.controller.set_manual_rg if which == 'rg'
                      else self.controller.set_manual_rh)
            setter(sid, value, is_apparent=apparent, fraction=frac)
        self._recompute()
        self._populate_source_panel(sid, frac)

    def _flush_manual(self) -> None:
        """Apply queued hand-entered values once the worker frees. Re-checks busy
        (a job started mid-drain pushes it to the next completion)."""
        if not self._pending_manual:
            return
        if runner().is_busy:
            run_when_idle(self._flush_manual)
            return
        pending, self._pending_manual = self._pending_manual, {}
        edits = {'rg': self.rg_manual, 'rh': self.rh_manual, 'mw': self.mw_manual}
        for which, (sid, frac, value, apparent) in pending.items():
            self._apply_manual(which, sid, frac, value, apparent)
            edits[which].clear()
