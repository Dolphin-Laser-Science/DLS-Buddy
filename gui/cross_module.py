"""
gui/cross_module.py
===================

The Cross-Sample tab: the first AGGREGATE-scope module. Where the sample-scoped
tabs (Data/DLS/SLS) operate on the one measurement picked in the shell sidebar,
this tab reads results ACROSS samples: the ρ = Rg/Rh pairing and the log–log
scaling plots (Rg–Mw, A₂–Mw) are both built here.

Selection model (two distinct things, both in-tab — Item 10)
------------------------------------------------------------
This is the only AGGREGATE tab, so it needs two selections a sample-scoped tab
collapses into one, and both are made explicitly in the tab (never via the shell
tree, which only read-only *mirrors* the focused sample):

* **Membership** — the left include/exclude list: which samples enter the ρ table
  and scaling regressions. All SLS samples start included; untick to exclude.
* **Focus** — the source panel's own **Sample + Fraction** combos: which one
  sample/fraction the source rows edit. Explicit — not a side effect of clicking a
  list row or a ρ-table row (that hidden coupling was removed).

Layout
------
* Left  — the membership list (every sample with SLS data: ρ also needs DLS, the
  scaling plots need Mw + Rg).
* Right — inner tabs: **ρ = Rg/Rh** (read-only table, one row per included sample
  that can pair ρ) and **Scaling** (log–log Rg–Mw and A₂–Mw plots). Beneath them a
  shared **source panel**: Sample/Fraction focus combos, then Rg (SLS), Rh (DLS),
  Mw (SLS) and A₂ (SLS) source pickers. Each combo is a labelled default (best tier)
  the user can override; Rg/Rh/Mw also allow a hand-entered value. **A₂ is picker-
  only** — a solvent/T-specific coefficient with no external standard, and the very
  y-axis the A₂–Mw plot fits. Candidates are grouped by result type with the
  single-condition tail behind a "show all" toggle; Mw/A₂ carry an uncalibrated
  badge (scale-dependent; Rg is not).

All numbers and provenance come from the controller (`compute_sample_rho`,
`sls_rg_candidates`, `dls_rh_candidates`, `sls_a2_candidates`, the auto/select/manual
setters). No analysis or physics here — the tab only displays and chooses.

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
from gui.help import section_header, add_help_to_groupbox
from gui.theme import ThemedLabel
from gui.widgets import roomy_tabs
from gui.worker import busy_notice, run_when_idle, runner
from analysis.utilities import interpret_scaling_exponent, select_default_candidate
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


# Candidate grouping for the source-picker combos. Maps a candidate `kind` to
# (group heading, hidden-by-default). "Hidden" candidates are the long tail of
# single-condition apparents (per-angle cumulants, per-conc Debye/Guinier) that
# overflowed the flat dropdown — they hide behind the "show all" checkbox, but the
# currently-selected and default candidates are always shown regardless.
_KIND_GROUP = {
    'sls_zimm':              ('Extrapolated (thermodynamic)', False),
    'sls_berry':             ('Extrapolated (thermodynamic)', False),
    'dls_conc_extrap':       ('Extrapolated (thermodynamic)', False),
    'dls_gamma_q2':          ('Extrapolated (q→0)', False),
    'dls_replicate_avg':     ('Replicate-averaged', False),
    'dls_distribution_peak': ('Distribution peaks', False),
    'dls_cumulant':          ('Single-angle (apparent)', True),
    'sls_debye':             ('Single-condition (apparent)', True),
    'sls_guinier':           ('Single-condition (apparent)', True),
}
_HEADER_SENTINEL = '__HEADER__'


class CrossSampleModule(QtWidgets.QWidget):
    """Aggregate ρ = Rg/Rh across samples, with per-sample source selection."""

    # Emitted when the in-tab focused sample changes, so the shell can read-only
    # mirror it onto the Workspace tree (the tab selects in-tab, not via the tree).
    selectionChanged = QtCore.Signal()

    def __init__(self, controller, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self._included: Dict[str, bool] = {}     # sample_id -> included?
        # A "unit" is a (sample_id, fraction) pair: a Mw series is several units of
        # one sample. Rows/points/source-panel all operate on units.
        self._row_units: List[Tuple[str, Optional[str]]] = []
        self._current_unit: Optional[Tuple[str, Optional[str]]] = None
        self._size_quantity = 'rg'               # top scaling plot: 'rg' or 'rh'
        self._show_all_single = False            # reveal the single-condition tail
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
                'Tick to include in the views; choose the focused sample and its '
                'Rg / Rh / Mw / A₂ sources with the panel below (its Sample selector).',
            ]))
        self.sample_list = QtWidgets.QListWidget()
        self.sample_list.setMinimumWidth(240)
        self.sample_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        # The left list is MEMBERSHIP only (tick = included in the aggregate views).
        # Which sample the source panel edits ("focus") is chosen explicitly by the
        # in-tab Sample combo below — not by clicking here, and not by ρ-table rows.
        self.sample_list.itemSelectionChanged.connect(self._on_selection_changed)
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
        # The ρ table is READ-ONLY results — selecting a row no longer re-points the
        # source panel (that was the hidden, unintuitive coupling). Focus is the
        # Sample/Fraction combos only.
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
        self.source_box = QtWidgets.QGroupBox('Source selection')
        grid = QtWidgets.QGridLayout(self.source_box)

        # ---- header: explicit Sample + Fraction selectors (the focus, in-tab) -----
        header = QtWidgets.QHBoxLayout()
        header.addWidget(QtWidgets.QLabel('Sample:'))
        self.sample_combo = QtWidgets.QComboBox()
        self.sample_combo.setMinimumWidth(200)
        self.sample_combo.setToolTip(
            'Which included sample this panel edits. Explicit — the Workspace tree '
            'only mirrors this choice; it does not drive it.')
        self.sample_combo.currentIndexChanged.connect(self._on_sample_combo)
        header.addWidget(self.sample_combo, 1)
        self.fraction_label = QtWidgets.QLabel('Fraction:')
        header.addWidget(self.fraction_label)
        self.fraction_combo = QtWidgets.QComboBox()
        self.fraction_combo.setMinimumWidth(120)
        self.fraction_combo.setToolTip(
            'Which Mw fraction of this sample to edit (a sample can hold several '
            'fractions, each its own SLS series). Hidden when there is only one.')
        self.fraction_combo.currentIndexChanged.connect(self._on_fraction_combo)
        header.addWidget(self.fraction_combo)
        header.addStretch(1)
        grid.addLayout(header, 0, 0, 1, 5)

        # Rg row: a combo of SLS candidates + Manual; a manual entry beside it.
        grid.addWidget(QtWidgets.QLabel('Rg (SLS):'), 1, 0)
        self.rg_combo = QtWidgets.QComboBox()
        self.rg_combo.activated.connect(lambda i: self._on_source_chosen('rg', i))
        grid.addWidget(self.rg_combo, 1, 1)
        self.rg_manual = QtWidgets.QLineEdit()
        self.rg_manual.setPlaceholderText(self._radius_unit())
        self.rg_manual.setFixedWidth(70)
        grid.addWidget(self.rg_manual, 1, 2)
        self.rg_manual_apparent = QtWidgets.QCheckBox('apparent')
        grid.addWidget(self.rg_manual_apparent, 1, 3)
        rg_set = QtWidgets.QPushButton('Set')
        rg_set.clicked.connect(lambda: self._on_manual('rg'))
        grid.addWidget(rg_set, 1, 4)

        # Rh row: a combo of DLS candidates + Manual; a manual entry beside it.
        # (Disabled for an SLS-only sample — no DLS data to pick from.)
        self.rh_label = QtWidgets.QLabel('Rh (DLS):')
        grid.addWidget(self.rh_label, 2, 0)
        self.rh_combo = QtWidgets.QComboBox()
        self.rh_combo.activated.connect(lambda i: self._on_source_chosen('rh', i))
        grid.addWidget(self.rh_combo, 2, 1)
        self.rh_manual = QtWidgets.QLineEdit()
        self.rh_manual.setPlaceholderText(self._radius_unit())
        self.rh_manual.setFixedWidth(70)
        grid.addWidget(self.rh_manual, 2, 2)
        self.rh_manual_apparent = QtWidgets.QCheckBox('apparent')
        grid.addWidget(self.rh_manual_apparent, 2, 3)
        rh_set = QtWidgets.QPushButton('Set')
        rh_set.clicked.connect(lambda: self._on_manual('rh'))
        grid.addWidget(rh_set, 2, 4)
        self._rh_widgets = [self.rh_combo, self.rh_manual,
                            self.rh_manual_apparent, rh_set]

        # Mw row (feeds the scaling plots): SLS candidates + Manual. No "apparent"
        # box -- a hand-entered Mw is treated as a trusted (calibrated) value.
        # Column 3 carries a calibration badge (Mw is scale-dependent).
        grid.addWidget(QtWidgets.QLabel('Mw (SLS):'), 3, 0)
        self.mw_combo = QtWidgets.QComboBox()
        self.mw_combo.activated.connect(lambda i: self._on_source_chosen('mw', i))
        grid.addWidget(self.mw_combo, 3, 1)
        self.mw_manual = QtWidgets.QLineEdit()
        self.mw_manual.setPlaceholderText(self._mw_unit())
        self.mw_manual.setFixedWidth(70)
        grid.addWidget(self.mw_manual, 3, 2)
        self.mw_badge = ThemedLabel('', role='pending', size=11, bold=True)
        self.mw_badge.setToolTip(
            'Mw is scale-dependent: an uncalibrated fit gives an arbitrary-scale Mw.')
        grid.addWidget(self.mw_badge, 3, 3)
        mw_set = QtWidgets.QPushButton('Set')
        mw_set.clicked.connect(lambda: self._on_manual('mw'))
        grid.addWidget(mw_set, 3, 4)

        # A2 row (feeds the A2-Mw scaling plot): a picker over fit-derived Zimm/Berry
        # candidates ONLY — no manual entry. A2 is a solvent/T-specific interaction
        # coefficient (no external "standard" as Mw has) and is the y-axis the scaling
        # plot fits, so a hand-typed value would invite circularity. Scale-dependent,
        # so it carries a calibration badge like Mw.
        grid.addWidget(QtWidgets.QLabel('A₂ (SLS):'), 4, 0)
        self.a2_combo = QtWidgets.QComboBox()
        self.a2_combo.setToolTip(
            'A₂ from the Zimm/Berry double extrapolation (thermodynamic). '
            'Calibration-dependent — an uncalibrated A₂ is on an arbitrary scale '
            '(see the Advanced Guide, SLS section). No manual entry: A₂ is a '
            'solvent/temperature-specific interaction coefficient with no external '
            'standard, and it is the y-axis the A₂–Mw plot fits.')
        self.a2_combo.activated.connect(lambda i: self._on_source_chosen('a2', i))
        grid.addWidget(self.a2_combo, 4, 1)
        self.a2_unit_label = QtWidgets.QLabel('mol·mL/g²')
        grid.addWidget(self.a2_unit_label, 4, 2)
        self.a2_badge = ThemedLabel('', role='pending', size=11, bold=True)
        self.a2_badge.setToolTip(
            'A₂ is scale-dependent: an uncalibrated fit gives an arbitrary-scale A₂.')
        grid.addWidget(self.a2_badge, 4, 3)

        # "Show all" reveals the single-condition tail (per-angle cumulants, per-conc
        # Debye/Guinier) hidden by default to keep the pickers scannable.
        self.show_all_check = QtWidgets.QCheckBox('Show all single-condition results')
        self.show_all_check.setToolTip(
            'Reveal the per-angle / per-concentration apparent results hidden by '
            'default. The best-tier value stays selected either way.')
        self.show_all_check.toggled.connect(self._on_show_all_toggled)
        grid.addWidget(self.show_all_check, 5, 1, 1, 4)

        # Panel-level banner: fires when the chosen Mw and/or A2 is uncalibrated.
        self.cal_banner = ThemedLabel('', role='pending', size=11)
        self.cal_banner.setWordWrap(True)
        grid.addWidget(self.cal_banner, 6, 0, 1, 5)

        grid.setColumnStretch(1, 1)
        add_help_to_groupbox(
            self.source_box,
            'Choose which fitted value represents this sample in the ρ table and the '
            'scaling plots.',
            bullets=[
                '<b>Sample / Fraction</b>: pick the sample and Mw fraction to edit. '
                'Explicit here — the Workspace tree only mirrors this choice.',
                '<b>Rg / Rh / Mw / A₂</b>: pick the fit that feeds each analysis; the '
                'best-tier result is selected by default.',
                '<b>Show all single-condition results</b>: reveal the per-angle / '
                'per-concentration apparent values hidden by default.',
                '<b>⚠ badges</b>: Mw and A₂ are scale-dependent — an uncalibrated fit '
                'is on an arbitrary scale. Rg is scale-independent, so it never badges.',
                '<b>Manual entry</b> (Rg / Rh / Mw): type a trusted external value. A₂ '
                'has no manual entry — it is solvent/temperature-specific with no '
                'external standard.',
            ])
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
                f'{len(universe)} sample(s) with SLS. Untick to exclude from the '
                'views; pick the sample to edit in the panel below.')

        # Default Rg / Rh / Mw for each included sample's fractions (labelled; never
        # clobbers a hand-entered value). Rh is a no-op for an SLS-only sample.
        for sid, inc in self._included.items():
            if inc:
                self._auto_select_all(sid)

        self._recompute()

        # Rebuild the in-tab focus selector (the Sample combo) from the included
        # samples and focus one — keeping the current sample/fraction if still valid.
        self._populate_sample_combo()

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

    def _unit_label(self, sample, fraction: Optional[str]) -> str:
        base = _sample_label(sample)
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
            label = self._unit_label(samples[sid], frac)   # samples: the sid→Sample map
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
        # Ticking a sample runs _auto_select_all, which WRITES SampleResult fields
        # (the labelled Rg/Rh/Mw/A2 auto-picks) -- it must not race a background fit
        # writing them too (invariant 4), exactly as refresh() guards. Defer the whole
        # handler until the worker frees; the tick persists in the live widget, so
        # re-reading the selection at idle applies it correctly (deferred, not dropped).
        if runner().is_busy:
            run_when_idle(self._on_selection_changed)
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
        # Membership drives which samples the focus combo offers.
        self._populate_sample_combo()

    # -------------------------------------------------- in-tab focus combos ---
    def _populate_sample_combo(self) -> None:
        """Rebuild the Sample focus combo from the included samples, then focus one
        (keeping the current sample if it is still included, else the first)."""
        included = [sid for sid, inc in self._included.items() if inc]
        by_id = {s.sample_id: s for s in self.controller.samples()}
        cur_sid = self._current_unit[0] if self._current_unit else None
        self._suppress = True
        self.sample_combo.clear()
        for sid in included:
            self.sample_combo.addItem(_sample_label(by_id[sid]))
            self.sample_combo.setItemData(
                self.sample_combo.count() - 1, sid, QtCore.Qt.ItemDataRole.UserRole)
        self._suppress = False
        if not included:
            self._current_unit = None
            self.source_box.setEnabled(False)
            self.interp.setText('')
            self.selectionChanged.emit()             # clear the tree mirror
            return
        target = cur_sid if cur_sid in included else included[0]
        # Keep the current fraction only when the sample is unchanged.
        keep_frac = (self._current_unit[1]
                     if (self._current_unit and self._current_unit[0] == target)
                     else None)
        self._focus_sample(target, fraction=keep_frac)

    def _sample_combo_index(self, sid: str) -> int:
        for i in range(self.sample_combo.count()):
            if self.sample_combo.itemData(i, QtCore.Qt.ItemDataRole.UserRole) == sid:
                return i
        return -1

    def _populate_fraction_combo(self, sid: str,
                                 prefer: Optional[str] = None) -> Optional[str]:
        """Fill the Fraction combo with the sample's SLS fractions; return the chosen
        one. Hidden when the sample has a single unlabelled fraction (nothing to pick)."""
        fracs = list(self.controller.sample_fractions(sid, 'sls')) or [None]
        self._suppress = True
        self.fraction_combo.clear()
        for frac in fracs:
            self.fraction_combo.addItem('(unlabelled)' if frac is None else str(frac))
            self.fraction_combo.setItemData(
                self.fraction_combo.count() - 1, frac,
                QtCore.Qt.ItemDataRole.UserRole)
        chosen = prefer if prefer in fracs else fracs[0]
        self.fraction_combo.setCurrentIndex(max(fracs.index(chosen), 0))
        self._suppress = False
        single_unlabelled = (len(fracs) == 1 and fracs[0] is None)
        self.fraction_label.setVisible(not single_unlabelled)
        self.fraction_combo.setVisible(not single_unlabelled)
        return chosen

    def _focus_sample(self, sid: str, fraction: Optional[str] = None) -> None:
        """Point the source panel at (sid, fraction): sync both header combos, fill the
        rows, and notify the shell so the Workspace tree mirrors the focused sample."""
        self._suppress = True
        idx = self._sample_combo_index(sid)
        if idx >= 0:
            self.sample_combo.setCurrentIndex(idx)
        self._suppress = False
        frac = self._populate_fraction_combo(sid, prefer=fraction)
        self._current_unit = (sid, frac)
        self._populate_source_panel(sid, frac)
        self.selectionChanged.emit()

    @QtCore.Slot(int)
    def _on_sample_combo(self, _index: int) -> None:
        if self._suppress:
            return
        sid = self.sample_combo.currentData(QtCore.Qt.ItemDataRole.UserRole)
        if sid is None:
            return
        self._focus_sample(sid)                      # new sample → first fraction

    @QtCore.Slot(int)
    def _on_fraction_combo(self, _index: int) -> None:
        if self._suppress or self._current_unit is None:
            return
        sid = self._current_unit[0]
        frac = self.fraction_combo.currentData(QtCore.Qt.ItemDataRole.UserRole)
        self._current_unit = (sid, frac)
        self._populate_source_panel(sid, frac)
        self.selectionChanged.emit()

    @QtCore.Slot(bool)
    def _on_show_all_toggled(self, checked: bool) -> None:
        self._show_all_single = checked
        if self._current_unit is not None:
            sid, frac = self._current_unit
            self._populate_source_panel(sid, frac)

    def selected_item_ids(self) -> List[str]:
        """The focused sample's DLS+SLS measurement item ids, for the shell's read-only
        tree mirror (this tab selects in-tab, not via the tree)."""
        if self._current_unit is None:
            return []
        sid = self._current_unit[0]
        ids: List[str] = []
        for kind in ('dls', 'sls'):
            for lm in self.controller.workspace.sample_measurements(sid, kind):
                ids.append(lm.item_id)
        return ids

    def _populate_source_panel(self, sid: str, fraction: Optional[str]) -> None:
        self._current_unit = (sid, fraction)
        # Manual-entry placeholders follow the active Display-units choice.
        self.rg_manual.setPlaceholderText(self._radius_unit())
        self.rh_manual.setPlaceholderText(self._radius_unit())
        self.mw_manual.setPlaceholderText(self._mw_unit())
        r = self.controller.workspace.samples[sid].result_for(fraction)
        hidden = 0
        hidden += self._fill_candidate_combo(
            self.rg_combo, self.controller.sls_rg_candidates(sid, fraction),
            current_label=r.rg_label, current_source=r.rg_source, quantity='radius')
        hidden += self._fill_candidate_combo(
            self.mw_combo, self.controller.sls_mw_candidates(sid, fraction),
            current_label=r.mw_label, current_source=r.mw_source, quantity='molar_mass')
        hidden += self._fill_candidate_combo(
            self.a2_combo, self.controller.sls_a2_candidates(sid, fraction),
            current_label=r.a2_label, current_source=r.a2_source,
            quantity='molar_mass', allow_manual=False)

        # Rh row: disabled for an SLS-only sample (no DLS data to pick from).
        has_dls = bool(self.controller.workspace.sample_measurements(sid, 'dls'))
        for w in self._rh_widgets:
            w.setEnabled(has_dls)
        self.rh_label.setEnabled(has_dls)
        if has_dls:
            hidden += self._fill_candidate_combo(
                self.rh_combo, self.controller.dls_rh_candidates(sid, fraction),
                current_label=r.rh_label, current_source=r.rh_source, quantity='radius')
        else:
            self.rh_combo.blockSignals(True)
            self.rh_combo.clear()
            self.rh_combo.addItem('— no DLS data for this sample —')
            self.rh_combo.blockSignals(False)

        # Calibration surfacing: badge the scale-dependent rows (Mw, A2) when the
        # chosen candidate is uncalibrated; Rg is scale-independent (no badge).
        self._set_cal_badge(self.mw_badge, r.calibrated, r.mw_source, r.mw_g_per_mol)
        self._set_cal_badge(self.a2_badge, r.a2_calibrated, r.a2_source, r.a2_mol_mL_per_g2)
        self._update_cal_banner(r)

        # "Show all" reveals the hidden single-condition tail; annotate its count.
        base = 'Show all single-condition results'
        self.show_all_check.setText(
            base if (self._show_all_single or not hidden) else f'{base} ({hidden} hidden)')
        self.show_all_check.setEnabled(bool(hidden) or self._show_all_single)

        self.source_box.setEnabled(True)
        self.source_box.setTitle('Source selection')
        try:
            rho = self.controller.compute_sample_rho(sid, fraction)
            flag = ('  [apparent ρ — at least one input is a single-condition '
                    'value]' if rho.is_apparent else '')
            self.interp.setText(f'{rho.interpretation}{flag}')
        except Exception as exc:
            self.interp.setText(str(exc))

    def _set_cal_badge(self, badge: ThemedLabel, calibrated: Optional[bool],
                       source: str, value: Optional[float]) -> None:
        """Show a colourblind-safe (symbol + text) uncalibrated badge when a chosen,
        scale-dependent value (Mw/A2) is uncalibrated. Blank when calibrated, unknown,
        hand-entered (trusted), or absent."""
        show = (calibrated is False and source != 'user'
                and value is not None and math.isfinite(value))
        badge.setText('⚠ uncalibrated' if show else '')

    def _update_cal_banner(self, r) -> None:
        which = []
        if r.calibrated is False and r.mw_source != 'user' and r.mw_g_per_mol is not None:
            which.append('Mw')
        if (r.a2_calibrated is False and r.a2_source != 'user'
                and r.a2_mol_mL_per_g2 is not None):
            which.append('A₂')
        if which:
            self.cal_banner.setText(
                f'⚠ {" and ".join(which)} are UNCALIBRATED (arbitrary scale) — '
                'calibrate on the SLS tab, or select a calibrated fit.')
        else:
            self.cal_banner.setText('')

    def _fill_candidate_combo(self, combo: QtWidgets.QComboBox, candidates, *,
                              current_label: str, current_source: str,
                              quantity: str = 'radius',
                              allow_manual: bool = True) -> int:
        """Rebuild a source-picker combo, GROUPED by result type with the long tail of
        single-condition apparents hidden behind the "show all" checkbox.

        Each real item's UserRole data is a sentinel `_on_source_chosen` dispatches on:
        a ResultCandidate (select it), 'USER' (keep the hand-entered value), '__HEADER__'
        (a disabled group heading — ignored), or None (the trailing "Manual entry…").
        Signals are blocked during the rebuild so it doesn't re-enter selection handling.
        The current selection and the default candidate are ALWAYS shown, even if their
        group is hidden. Returns the number of candidates hidden (for the checkbox hint)."""
        default = select_default_candidate(list(candidates))
        combo.blockSignals(True)
        combo.clear()
        model = combo.model()
        select_index = -1
        default_index = -1
        hidden = 0

        # Decide visibility, then group the visible candidates preserving order.
        groups: Dict[str, list] = {}
        for c in candidates:
            title, hide_default = _KIND_GROUP.get(c.kind, ('Other', False))
            is_current = (current_source != 'user' and c.label == current_label)
            is_default = (default is not None and c is default)
            if hide_default and not self._show_all_single and not (is_current or is_default):
                hidden += 1
                continue
            groups.setdefault(title, []).append(c)

        for title, items in groups.items():
            combo.addItem(f'— {title} —')
            hi = combo.count() - 1
            combo.setItemData(hi, _HEADER_SENTINEL, QtCore.Qt.ItemDataRole.UserRole)
            model.item(hi).setEnabled(False)       # non-selectable heading
            for c in items:
                combo.addItem(f'    {c.label}')
                idx = combo.count() - 1
                combo.setItemData(idx, c, QtCore.Qt.ItemDataRole.UserRole)
                if current_source != 'user' and c.label == current_label:
                    select_index = idx
                if default is not None and c is default:
                    default_index = idx

        # a hand-entered value shows as a distinct, pre-selected item (in display units)
        if current_source == 'user':
            unit = (self._mw_unit() if quantity == 'molar_mass' else self._radius_unit())
            val = self._user_value(combo)
            disp = (U.from_canonical(quantity, val, unit)
                    if isinstance(val, (int, float)) and math.isfinite(val) else val)
            combo.addItem(f'User-entered ({_fmt(disp)} {unit})')
            combo.setItemData(combo.count() - 1, 'USER', QtCore.Qt.ItemDataRole.UserRole)
            select_index = combo.count() - 1
        if allow_manual:
            combo.addItem(_MANUAL_LABEL)
            combo.setItemData(combo.count() - 1, None, QtCore.Qt.ItemDataRole.UserRole)
        elif not groups and current_source != 'user':
            # no fit-derived candidates and no manual path (A2): a disabled hint
            combo.addItem('— no fit available (needs ≥2 concentrations) —')
            combo.setItemData(combo.count() - 1, _HEADER_SENTINEL,
                              QtCore.Qt.ItemDataRole.UserRole)
            model.item(combo.count() - 1).setEnabled(False)

        if select_index < 0:
            select_index = default_index
        if select_index < 0:                       # first selectable (skip headings)
            select_index = self._first_selectable(combo)
        if select_index >= 0:
            combo.setCurrentIndex(select_index)
        combo.blockSignals(False)
        return hidden

    @staticmethod
    def _first_selectable(combo: QtWidgets.QComboBox) -> int:
        """Index of the first non-heading item, or -1 if the combo is headings-only."""
        for i in range(combo.count()):
            if combo.itemData(i, QtCore.Qt.ItemDataRole.UserRole) != _HEADER_SENTINEL:
                return i
        return -1

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
        return {'rg': self.rg_combo, 'rh': self.rh_combo,
                'mw': self.mw_combo, 'a2': self.a2_combo}[which]

    def _on_source_chosen(self, which: str, index: int) -> None:
        if self._current_unit is None:
            return
        sid, frac = self._current_unit
        data = self._combo(which).itemData(index, QtCore.Qt.ItemDataRole.UserRole)
        if data == _HEADER_SENTINEL:     # a disabled group heading — ignore
            return
        if data is None:                 # "Manual entry…" — let the user type + Set
            edit = {'rg': self.rg_manual, 'rh': self.rh_manual,
                    'mw': self.mw_manual}.get(which)
            if edit is not None:         # A2 has no manual field
                edit.setFocus()
            return
        if data == 'USER':               # keep the existing hand-entered value
            return
        if runner().is_busy:             # set_sample_* writes SampleResult a fit reads
            busy_notice(self)
            self._populate_source_panel(sid, frac)   # revert the combo to committed
            return
        setter = {'rg': self.controller.set_sample_rg,
                  'rh': self.controller.set_sample_rh,
                  'mw': self.controller.set_sample_mw,
                  'a2': self.controller.set_sample_a2}[which]
        # explicit=True marks the pick 'picked' so a later passive refresh's
        # auto-select won't silently revert it to the best-tier default.
        setter(sid, data, frac, explicit=True)
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
