"""
gui/cross_module.py
===================

The Cross-Sample tab: the first AGGREGATE-scope module. Where the sample-scoped
tabs (Data/DLS/SLS) operate on the one measurement picked in the shell sidebar,
this tab reads results ACROSS samples: the ρ = Rg/Rh pairing and the log–log
scaling plots (Rg–Mw, A₂–Mw) are both built here.

Selection model (two distinct things; focus unified across tabs)
---------------------------------------------------------------------------
This is the only AGGREGATE tab, so it needs two selections a sample-scoped tab
collapses into one:

* **Membership** — the left include/exclude list: which samples enter the ρ table
  and scaling regressions. All SLS samples start included; untick to exclude. This is
  in-tab only (never driven by the sidebar).
* **Focus** — which one sample/fraction the source rows edit. The **Fraction** combo
  is in-tab; the **Sample** focus is now the shell's single **active sample**:
  it is driven by the in-tab Sample combo **or** by clicking a sample in the
  Workspace tree (`set_focused_sample`, called from the shell's `_focus_sample`) — the
  two are mirrors of one choice. (For a non-SLS active sample, which isn't in the Cross
  universe, the focus keeps its last value — Cross is aggregate over SLS-bearing
  samples.) Focus is still NOT a side effect of clicking a membership-list row or a
  ρ-table row (that hidden coupling was removed).

Layout
------
A **top row** over a **full-width bottom strip** (the source/interpretation panels
span under the list too, so the short membership list no longer stretches the whole
tab height):

* Top row, left  — the membership list (every sample with SLS data: ρ also needs
  DLS, the scaling plots need Mw + Rg).
* Top row, right — inner tabs: **ρ = Rg/Rh** (read-only table, one row per included
  sample that can pair ρ) and **Scaling** (log–log Rg–Mw and A₂–Mw plots).
* Bottom strip, left — a shared **Source Selection** panel: Sample/Fraction focus
  combos, then Rg (SLS), Rh (DLS), Mw (SLS) and A₂ (SLS) source pickers. Each combo
  is a labeled default (best tier) the user can override; Rg/Rh/Mw also allow a
  hand-entered value. **A₂ is picker-only** — a solvent/T-specific coefficient with
  no external standard, and the very y-axis the A₂–Mw plot fits. Candidates are
  grouped by result type with the single-condition tail behind a "show all" toggle;
  each row's ⚠ badge marks the lines in the panel's notes block about its chosen value
  (an arbitrary scale reaches Mw/A₂ only; Rg is scale-independent). The value combos
  elide when the panel is narrow (the full text stays in the dropdown popup).
* Bottom strip, right — an **Interpretation** context panel: a `QStackedWidget`
  (`_build_context_panel`) whose page follows the active inner tab — the ρ
  interpretation text on the ρ tab, the scaling-exponent note on the Scaling tab.
  (The interpretation labels used to live inside their sub-tabs.)

All numbers and provenance come from the controller (`compute_sample_rho`,
`sls_rg_candidates`, `dls_rh_candidates`, `sls_a2_candidates`, the auto/select/manual
setters). No analysis or physics here — the tab only displays and chooses.

Apparent vs thermodynamic is shown per row: ρ is "apparent" if either Rg or Rh is
an apparent (single-condition) value, never silently mixed with a thermodynamic
one. Each Source Selection row's label carries the words of the value it holds.
"""

from __future__ import annotations

import html
import math
from typing import Dict, List, Optional, Tuple

from PySide6 import QtCore, QtWidgets

from matplotlib.figure import Figure

from core.naming import sample_display_name
from plotting.plots import plot_scaling
from gui.plot_controls import LazyCanvas, make_canvas_expanding
from gui.export_helper import confirm_overwrite, export_to_csv
from gui.field_labels import CHAIN_TYPE_LABELS, has_honest_se
from gui.help import section_header, add_help_to_groupbox, HelpBadge
from core.note_registry import REGISTRY
from gui.notes_block import NotesBlock, qualifier_tooltip, qualifier_words
from gui.theme import ThemedLabel
from gui.widgets import roomy_tabs, RichTextHeaderView
from gui.worker import busy_notice, run_when_idle, runner
from core.workspace import DELIBERATE_TIERS, result_key_matches
from core.formatting import (format_fixed_sig, format_pm, pdg_erases_value,
                             format_value_at_uncertainty)
from core.notes import Note, render, vocabulary
from core import units as U


# Architecture-table columns. Rg/Rh carry a unit that follows the global Display-units
# setting; `_columns()` inserts the active label, so this template holds only the fixed
# parts. g and h are the branching contraction factors (from Rg and Rh); Chain type is
# a read-only mirror of the sample's Data-tab tag (editing lives on the Data tab).
_COL_TEMPLATE = ['Sample', 'Chain type', 'R<sub>g</sub>', 'R<sub>h</sub>',
                 'ρ = R<sub>g</sub>/R<sub>h</sub> (± statistical)', 'Shape',
                 'g', 'h', 'n (branch)', 'Type']
(_C_SAMPLE, _C_CHAIN, _C_RG, _C_RH, _C_RHO, _C_SHAPE,
 _C_G, _C_H, _C_N, _C_TYPE) = range(10)
# The branching-model selector: label -> the model kwarg compute_architecture takes.
# Regular star (arm number) is the default; the random models read a branching density n.
_BRANCHING_MODEL_CHOICES = [
    ('Regular star (arm number)', 'regular_star'),
    ('Random tri-functional (f=3)', 'random_tri'),
    ('Random tetra-functional (f=4)', 'random_tetra'),
]
_BRANCHING_MODEL_LABELS = {key: text for text, key in _BRANCHING_MODEL_CHOICES}
_CHAIN_DISPLAY = CHAIN_TYPE_LABELS   # stored -> shown

# The overview badge's ρ-band hint. Every band it names is one of the engine's bands in
# `analysis.utilities._rho_band`, and the pin test reads this string, so a re-cut of any
# edge goes red rather than drifting.
_RHO_HINT = (
    '<b>ρ = Rg/Rh</b> hints at architecture (≈0.78 sphere, ≈1.0 dendrimer / soft sphere, '
    '≈1.1–1.4 branched/star, ≈1.5–1.8 linear coil, ≈1.9–2.1 a broad or swollen coil, '
    'above ≈2.2 rod-like), a qualitative guide (Theory-and-Equations-Guide §5.1); it '
    'needs an SLS Rg and a DLS Rh per sample.')
_EMPTY = '—'                        # single not-applicable placeholder (+ a reason tooltip)
_MANUAL_LABEL = 'Manual entry…'


def _fmt(x: Optional[float], sig: int = 3) -> str:
    if x is None or not (isinstance(x, (int, float)) and math.isfinite(x)):
        return 'n/a'
    return f'{x:.{sig}g}'


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
    # A ladder is a whole concentration SERIES read at one angle, so it is neither
    # "single-condition" nor extrapolated — its own heading, named for the angle
    # contingency (invariant 7). Never hidden: it carries auto_selectable=False, which the
    # visibility test below already exempts, and hiding it would put a sample's only Mw/A2
    # option behind a checkbox (the degenerate-solution lesson, one kind over).
    'sls_ladder':            ('Single-angle ladder (apparent)', False),
}
_HEADER_SENTINEL = '__HEADER__'


def _with_words(text: str, words: List[str]) -> str:
    """A shown value with its qualifier words (`gui.notes_block.qualifier_words`),
    `40.5 (model-dependent)`: vocabulary, never an alarm (ui_style_guide §5). A missing
    value ('n/a', '—') takes none."""
    if not words or text in ('n/a', _EMPTY):
        return text
    return f'{text} ({", ".join(words)})'


# The Rg / Rh cells name what a value assumes (a shape, a degenerate pick); apparent
# against thermodynamic is the Type column's word, read off the row's ρ.
_CELL_WORDS_EXCLUDE = ('apparent',)


class CrossSampleModule(QtWidgets.QWidget):
    """Aggregate ρ = Rg/Rh across samples, with per-sample source selection."""

    # Emitted when the focused sample changes, so the shell read-only mirrors it onto
    # the Workspace tree.
    selectionChanged = QtCore.Signal()
    # Emitted when the USER picks a sample in the in-tab dropdown, so the shell makes it
    # the single active sample and fans it out to every tab + the sidebar.
    sampleFocusRequested = QtCore.Signal(str)

    def __init__(self, controller, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self._included: Dict[str, bool] = {}     # sample_id -> included?
        # Latest ArchitectureData (g/h laws + per-branched rows), cached so a table-row
        # click can show that row's g/h detail in the Interpretation panel without a
        # recompute. Chain-type tags now live on the samples (Data tab), read via
        # controller.sample_chain_type -- no panel-local role dict.
        self._arch = None
        # A "unit" is a (sample_id, fraction) pair: a Mw series is several units of
        # one sample. Rows/points/source-panel all operate on units.
        self._row_units: List[Tuple[str, Optional[str]]] = []
        self._current_unit: Optional[Tuple[str, Optional[str]]] = None
        # The (sid, fraction) the MANUAL-entry widgets are currently bound to. Tracked
        # separately from _current_unit (which _focus_sample/_on_fraction_combo set
        # BEFORE they repopulate the panel) so the panel can tell a real target change
        # from a same-target repopulate, and clear a stale typed value on the former.
        self._manual_unit: Optional[Tuple[str, Optional[str]]] = None
        self._size_quantity = 'rg'               # top scaling plot: 'rg' or 'rh'
        self._branching_model = 'regular_star'   # architecture g reading: star f_eff | random n
        self._show_all_single = False            # reveal the single-condition tail
        self._suppress = False                   # guard signal storms on rebuild
        # Hand-entered Rg/Rh/Mw typed while the worker was busy, applied (re-checking)
        # once it frees — a precious value is never dropped. Keyed by (which, sid, frac),
        # NOT by which alone: queuing the same quantity for two different samples while
        # busy must not silently drop the first.
        self._pending_manual: Dict[Tuple[str, str, Optional[str]], tuple] = {}
        self._build_ui()
        self.refresh()

    # ------------------------------------------------------------------ UI ---
    def _build_ui(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)

        # ===== top row: include/exclude list  |  workspace (inner sub-tabs) =====
        top_row = QtWidgets.QHBoxLayout()
        outer.addLayout(top_row, 1)

        # ---- left: include / exclude --------------------------------------
        left = QtWidgets.QVBoxLayout()
        top_row.addLayout(left, 0)
        left.addWidget(section_header(
            'Samples (Include / Exclude)',
            'Compare results across samples:',
            bullets=[
                'Tick the samples to include in the Architecture table and the scaling '
                'plots; pick each one\'s sources in the Source Selection panel below.',
                _RHO_HINT,
                '<b>Branching g / h</b>: tag samples <b>Linear</b> or <b>Branched</b> '
                'with the Chain type control on the Data tab.',
                '<b>Scaling</b> fits Rg–Mw and A₂–Mw power laws across the included '
                'samples.',
            ]))
        self.sample_list = QtWidgets.QListWidget()
        self.sample_list.setMinimumWidth(240)
        self.sample_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        # The left list is MEMBERSHIP only (tick = included in the aggregate views).
        # Which sample the source panel edits ("focus") is the shell's active sample —
        # driven by the in-tab Sample combo below OR a Workspace tree click,
        # never by clicking a row in THIS membership list or a ρ-table row.
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

        # ---- right (of top row): inner sub-tabs (rho | scaling) ------------
        # The interpretation text and the Source Selection panel used to live in
        # this column; they now sit in the full-width bottom strip below, freeing
        # the sample list from stretching the whole tab height.
        self.inner = roomy_tabs(QtWidgets.QTabWidget())   # roomier tabs so labels don't clip
        self.inner.setMovable(True)              # drag to reorder sub-tabs
        top_row.addWidget(self.inner, 1)

        # Architecture tab: ONE read-only table -- ρ = Rg/Rh, the shape guide, and the
        # branching contraction factors g (from Rg) and h (from Rh) side by side, with
        # each sample's Chain-type tag mirrored from the Data tab. Everything answers the
        # same question (is it branched, and how), so it lives on one row per sample.
        rho_tab = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(rho_tab)
        rl.addWidget(section_header(
            'Architecture — ρ, shape, and branching factors g / h',
            'One row per included sample and Mw fraction.',
            bullets=[
                'Tag samples <b>Linear</b> (≥2 at different Mw) and <b>Branched</b> with '
                'the <b>Chain type</b> control on the <b>Data tab</b>; g needs an Rg, h '
                'an Rh.',
                'A cell shows <b>—</b> where it can’t be computed; hover for why. Select '
                'a row for its readout below.',
                'An R<sub>g</sub> or R<sub>h</sub> that failed a check carries ⚠ and its '
                'line\'s letter; click the line under the table for the fix.',
                'A <b>±</b> on g or h appears only with replicate branched measurements.',
                'What ρ, its bands, g and h mean: Theory-and-Equations-Guide §5.1 and '
                '§5.3–5.4.',
            ]))
        # Branching-model selector: how the measured g is READ. Regular star -> an arm
        # number (f_eff); random tri/tetra -> a branching density n (Zimm & Stockmayer
        # Eqs. 47a/47b). It changes only the reading, never the g VALUE, and drives the
        # `n (branch)` column + the row-detail readout below.
        model_row = QtWidgets.QHBoxLayout()
        model_lbl = QtWidgets.QLabel('Branching model:')
        model_row.addWidget(model_lbl)
        self.branching_model_combo = QtWidgets.QComboBox()
        for text, _key in _BRANCHING_MODEL_CHOICES:
            self.branching_model_combo.addItem(text)
        self.branching_model_combo.setToolTip(
            'The reading is model-dependent: the same g gives a different n under a '
            'different model, and f_eff is a regular-star equivalent, not a literal arm '
            'count (Theory-and-Equations-Guide §5.3 and §5.3.1).')
        self.branching_model_combo.currentIndexChanged.connect(
            self._on_branching_model_changed)
        model_row.addWidget(self.branching_model_combo)
        model_row.addWidget(HelpBadge(
            'Choose the architecture a measured g is read against:',
            bullets=[
                '<b>Regular star</b>: an <b>effective arm number</b> f_eff.',
                '<b>Random tri-/tetra-functional</b>: a <b>branching density n</b>, the '
                'average branch units per molecule for 3- or 4-functional branch units, in '
                'the <b>n (branch)</b> column.',
                'g itself does not change; the readout names the model it assumed.',
            ]))
        model_row.addStretch(1)
        rl.addLayout(model_row)
        self.table = QtWidgets.QTableWidget(0, len(_COL_TEMPLATE))
        # Headers name quantities (R_g, R_h, ρ = R_g/R_h) — render them as rich text.
        self.table.setHorizontalHeader(RichTextHeaderView(self.table))
        self.table.setHorizontalHeaderLabels(self._columns())
        self.table.horizontalHeader().setStretchLastSection(True)
        # A sample name is 'polymer / solvent @ T K' and runs past the default section
        # width, so its column follows its contents rather than a fixed size -- the one
        # cell in this table whose text a reader needs whole to tell two rows apart.
        self.table.horizontalHeader().setSectionResizeMode(
            _C_SAMPLE, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        # The ρ header carries the always-visible "± statistical" qualifier and is the
        # one header longer than the default section width; size it to its text once.
        self.table.horizontalHeader().resizeSection(_C_RHO, 250)
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        # Fully read-only (Chain type is tagged on the Data tab). Selecting a row shows
        # that sample's g/h detail in the bottom Interpretation panel; it does NOT
        # re-point the source panel (focus is the Sample/Fraction combos only).
        self.table.itemSelectionChanged.connect(self._on_arch_row_selected)
        rl.addWidget(self.table, 1)
        # The preconditions of the values the table shows, one line per cause, pinned
        # under the table (R8.6); a marked cell opens its line's popup on a click.
        self.arch_block = NotesBlock(what='samples')
        self.arch_block.watch(self.table)
        rl.addWidget(self.arch_block, 0)
        # Linear-reference provenance (the fitted Rg-Mw / Rh-Mw laws) + Export.
        self.arch_caption = ThemedLabel(self._ARCH_HINT, role='muted', size=11)
        self.arch_caption.setWordWrap(True)
        self.arch_caption.setToolTip(
            'g and h compare a Branched sample with the linear Rg–Mw / Rh–Mw laws fit to '
            'the Linear samples, read at its Mw. Their ± comes from replicate branched '
            'measurements only and excludes the reference-law fit uncertainty '
            '(Theory-and-Equations-Guide §5.3–5.4, §6.4).')
        rl.addWidget(self.arch_caption, 0)
        exp_row = QtWidgets.QHBoxLayout()
        exp_row.addStretch(1)
        self.arch_export_button = QtWidgets.QPushButton('Export CSV…')
        self.arch_export_button.setToolTip(
            'Export the architecture table (ρ, shape, g, h) as one CSV.')
        self.arch_export_button.clicked.connect(self._on_export_architecture)
        exp_row.addWidget(self.arch_export_button)
        rl.addLayout(exp_row)
        self._rho_tab = rho_tab
        self.inner.addTab(rho_tab, 'Architecture')

        # scaling tab: two log-log plots (size-Mw on top, A2-Mw below) + a button
        # that swaps the top plot's size axis between Rg and Rh.
        scaling_tab = QtWidgets.QWidget()
        sl = QtWidgets.QVBoxLayout(scaling_tab)
        sl.addWidget(section_header(
            'Scaling — Rg–Mw and A₂–Mw power laws',
            'Log–log power-law fits across the included samples: a size exponent '
            'ν from Rg–Mw (or Rh–Mw) and the A₂–Mw slope.',
            bullets=[
                'One point per (sample, Mw fraction) that has an Mw (a '
                'hand-entered Mw wins) and a positive value — others are '
                'counted as excluded.',
                '<b>Branched</b> samples are plotted hollow and left out of the fit; '
                'the others fit the line.',
                '<b>Show Rh–Mw</b> swaps the top plot\'s size axis between Rg '
                '(SLS) and Rh (DLS).',
                'The readout below names excluded points and flags '
                'arbitrary-scale or model-dependent values.',
                'See the Theory-and-Equations-Guide §5.2.',
            ]))
        self.scaling_fig = Figure(figsize=(4.3, 4.8))
        self.scaling_canvas = make_canvas_expanding(LazyCanvas(self.scaling_fig))
        self.ax_rg = self.scaling_fig.add_subplot(2, 1, 1)
        self.ax_a2 = self.scaling_fig.add_subplot(2, 1, 2)
        sl.addWidget(self.scaling_canvas, 1)
        # Export row under the plot, at the bottom of the results area (R7.3) --
        # mirrors the Architecture tab's exp_row.
        swap_row = QtWidgets.QHBoxLayout()
        swap_row.addStretch(1)
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
        sl.addLayout(swap_row)
        # (scaling interpretation text now lives in the bottom-strip context panel.)
        self._scaling_tab = scaling_tab
        self.inner.addTab(scaling_tab, 'Scaling')

        # ===== bottom strip (full width): Source Selection | context panel =====
        # ~60/40 split. Source Selection is shared and identical on both sub-tabs;
        # the context panel swaps its interpretation page to follow the active one.
        bottom = QtWidgets.QHBoxLayout()
        outer.addLayout(bottom, 0)
        bottom.addWidget(self._build_source_panel(), 3)
        bottom.addWidget(self._build_context_panel(), 2)

        # Keep the context panel in sync with the visible sub-tab, and refresh the
        # scaling plots when the scaling tab is shown (robust to tab reordering).
        self.inner.currentChanged.connect(self._on_inner_tab_changed)

    def _build_context_panel(self) -> QtWidgets.QWidget:
        """The bottom-right context panel: interpretation text that follows the
        active sub-tab (ρ interpretation vs scaling-exponent interpretation), via
        a QStackedWidget keyed to the visible sub-tab. The ρ page also reserves
        the slot for the future Architecture / branching-g readout."""
        box = QtWidgets.QGroupBox('Interpretation')
        v = QtWidgets.QVBoxLayout(box)
        self.context_stack = QtWidgets.QStackedWidget()
        v.addWidget(self.context_stack)

        # page 0 — ρ interpretation (leaves room below for the Architecture block)
        rho_page = QtWidgets.QWidget()
        rp = QtWidgets.QVBoxLayout(rho_page)
        rp.setContentsMargins(0, 0, 0, 0)
        # A glanceable headline (the shape label, bold) over the detail sentence, so the
        # ρ readout reads at a glance; status/error messages clear the headline.
        self.interp_headline = ThemedLabel('', role='header', size=14, bold=True)
        self.interp_headline.setWordWrap(True)
        rp.addWidget(self.interp_headline)
        self.interp = ThemedLabel('', role='muted', size=12)
        self.interp.setWordWrap(True)
        rp.addWidget(self.interp)
        rp.addStretch(1)
        self.context_stack.addWidget(rho_page)

        # page 1 — scaling-exponent interpretation
        scaling_page = QtWidgets.QWidget()
        spv = QtWidgets.QVBoxLayout(scaling_page)
        spv.setContentsMargins(0, 0, 0, 0)
        # Mirror the ρ page: a bold headline (the exponent numbers, at a glance) over
        # the bulleted detail, at the same font sizes so the two context pages match.
        self.scaling_headline = ThemedLabel('', role='header', size=14, bold=True)
        self.scaling_headline.setWordWrap(True)
        spv.addWidget(self.scaling_headline)
        self.scaling_note = ThemedLabel('', role='muted', size=12)
        self.scaling_note.setWordWrap(True)
        spv.addWidget(self.scaling_note)
        # The plotted points' scale gates, one line per cause with its points named.
        self.scaling_block = NotesBlock(what='points')
        spv.addWidget(self.scaling_block)
        spv.addStretch(1)
        self.context_stack.addWidget(scaling_page)

        return box

    def _on_inner_tab_changed(self, _index: int) -> None:
        """Sync the bottom context panel to the visible sub-tab and refresh the
        scaling plots when the scaling tab is shown. Compares by tab identity
        (not raw index) so it stays correct if the user reorders the sub-tabs."""
        is_scaling = self.inner.currentWidget() is self._scaling_tab
        self.context_stack.setCurrentIndex(1 if is_scaling else 0)
        self._refresh_scaling()
        if not is_scaling:
            self._refresh_architecture()   # also self-heals a lingering export-status line

    def _set_interp(self, detail: str, headline: str = '', *,
                    from_row: bool = False) -> None:
        """Set the ρ context panel: an optional glanceable headline (the shape label,
        bold) over the detail text. Status / error / empty messages pass no headline.

        The panel has two writers -- the focused unit (every Source-panel action and
        refresh) and a clicked table row -- and describes exactly ONE of them at a time:
        a focus-driven write clears the row highlight, so a highlighted row is always the
        unit the text describes and never a different sample than the one on screen."""
        if not from_row:
            self.table.clearSelection()      # a no-op signal-wise when nothing is selected
        self.interp_headline.setText(headline)
        self.interp_headline.setVisible(bool(headline))
        self.interp.setText(detail)

    # ---- Architecture: branching contraction factors g (Rg) and h (Rh) ----------
    _ARCH_HINT = ('Tag samples Linear (≥2 at different Mw) and Branched with the Chain '
                  'type control on the Data tab to compute g and h.')

    def _refresh_architecture(self) -> None:
        """Rebuild the architecture table (which carries the g/h columns and the
        linear-reference caption). Deferred past a running fit like every other
        SampleResult read path -- a tab switch can fire any time."""
        if runner().is_busy:
            run_when_idle(self._refresh_architecture)
            return
        self._rebuild_table()

    def _on_branching_model_changed(self, index: int) -> None:
        """The branching-model selector changed: rebuild the table (which re-reads g as
        f_eff / n under the new model) and refresh the row-detail readout. The g VALUE is
        unchanged; only its interpretation and the `n (branch)` column change."""
        if self._suppress or not (0 <= index < len(_BRANCHING_MODEL_CHOICES)):
            return
        self._branching_model = _BRANCHING_MODEL_CHOICES[index][1]
        # Rebuild AND refresh the detail together, both deferred past a running fit. The
        # detail panel reads the cached self._arch (recomputed by the rebuild), so calling
        # it before a deferred rebuild would show the NEW model label against the OLD
        # cached n -- a mislabeled reading. Bundle them so they stay consistent.
        if runner().is_busy:
            run_when_idle(self._apply_branching_model)
            return
        self._apply_branching_model()

    def _apply_branching_model(self) -> None:
        """Rebuild the architecture table under the current model, then re-render the
        row-detail readout from the freshly-cached self._arch (keeps label and n in sync)."""
        self._rebuild_table()
        self._on_arch_row_selected()

    def _arch_data(self):
        """Compute (and cache) ArchitectureData over the included samples and set the
        caption from the fitted linear-reference laws. Returns the data (rows may be
        empty) or None on failure. Defensive: never breaks the tab."""
        included = [sid for sid, inc in self._included.items() if inc]
        try:
            data = self.controller.compute_architecture(
                included, branching_model=self._branching_model)
        except Exception as exc:  # noqa: BLE001 — surfaced in the caption; defensive: never break the tab
            self._arch = None
            self.arch_caption.setText(str(exc))
            return None
        self._arch = data
        self._set_arch_caption(data)
        return data

    def _set_arch_caption(self, data) -> None:
        """The linear-reference provenance line: the fitted Rg–Mw / Rh–Mw laws, or a
        guidance string when no linear reference exists yet."""
        parts = []
        if data.rg_law.fit_valid and data.rg_mw_range is not None:
            parts.append(f'Rg–Mw ν = {data.rg_law.exponent:.2f} '
                         f'(R² = {data.rg_law.r_squared:.3f}, {data.n_linear_rg} pts)')
        if data.rh_law.fit_valid and data.rh_mw_range is not None:
            parts.append(f'Rh–Mw ν = {data.rh_law.exponent:.2f} '
                         f'(R² = {data.rh_law.r_squared:.3f}, {data.n_linear_rh} pts)')
        if not parts:
            self.arch_caption.setText(data.message or self._ARCH_HINT)
            return
        # What the laws rest on is the readout's own vocabulary (a model-dependent or
        # degenerate contributing value, R5.6); a contributing Mw on an arbitrary scale is a
        # line of the block under the table, its cause and its samples named (data.notes).
        words = qualifier_words(data)
        self.arch_caption.setText('Linear reference: ' + ' · '.join(parts)
                                  + (f'  ({", ".join(words)})' if words else ''))
        self.arch_caption.setToolTip(qualifier_tooltip(data))

    def _gh_cell_text(self, res, value_attr: str) -> str:
        """Format a g or h cell: format_pm with the replicate ±, else fixed-sig."""
        sig = self.controller.settings.no_uncertainty_sig_figs
        v = getattr(res, value_attr)
        se = getattr(res, value_attr + '_se')
        return format_pm(v, se) if se is not None else format_fixed_sig(v, sig)

    def _n_cell(self, sid: str, g_res) -> Tuple[str, str]:
        """The `n (branch)` cell for the selected random model: the branching density from
        g. `—` under the regular-star model (n is not that model's reading), when g is
        missing (n needs Rg), or when g is out of the (0,1] domain (n undefined). The
        value carries no separate ± -- n inherits g's uncertainty (shown in the g cell)."""
        if self._branching_model == 'regular_star':
            return (_EMPTY, 'Branching density n — pick a Random tri-/tetra-functional '
                            'model above to read it (the regular star reads f_eff in g).')
        if g_res is None:
            return (_EMPTY, self._gh_missing_tip(sid, 'g'))     # n needs g (SLS Rg)
        n = g_res.branching_index
        if n is None or not math.isfinite(n):
            return (_EMPTY, 'n is undefined here: g ≥ 1 (the branched Rg is not smaller '
                            'than its linear reference at this Mw).')
        return (self._gh_cell_text(g_res, 'branching_index'), g_res.interpretation)

    def _rho_interp(self, sid: str, fraction: Optional[str]) -> Tuple[str, str]:
        """(detail, headline) for one unit's ρ reading -- THE body behind both writers of
        the Interpretation panel (the focused unit and a clicked table row), so the two
        can never describe the same unit differently."""
        try:
            rho = self.controller.compute_sample_rho(sid, fraction)
        except Exception as exc:  # noqa: BLE001 — surfaced as the detail text
            return str(exc), ''
        # The headline names what this ρ is (its qualifier words, e.g. 'apparent'); a
        # precondition of the ρ is the Architecture block's line, never this panel's.
        return rho.interpretation, _with_words(rho.shape, qualifier_words(rho))

    def _on_arch_row_selected(self) -> None:
        """Show the selected row's detail in the Interpretation panel: a branched row's
        g/h arm-number detail + the g-vs-h note, else that unit's ρ reading. A row click
        never re-points the Source panel; the highlight says which unit the panel
        describes, and a focus-driven write clears it (see _set_interp)."""
        if self._suppress:
            return
        model = self.table.selectionModel()
        rows = model.selectedRows() if model is not None else []
        if not rows:
            return
        r = rows[0].row()
        if not (0 <= r < len(self._row_units)):
            return
        sid, frac = self._row_units[r]
        res = self.controller.workspace.samples[sid].result_for(frac)
        arow = (self._arch.row_for(sid, res.effective_mw())
                if self._arch is not None else None)
        self.context_stack.setCurrentIndex(0)
        if arow is None:
            # Not a branched row (no g/h, or no architecture data at all) -> this unit's
            # ρ reading, so the previous row's text never stands on a new selection.
            self._set_interp(*self._rho_interp(sid, frac), from_row=True)
            return
        lines = []
        for label, gh, va in (('g', arow.g_result, 'g'), ('h', arow.h_result, 'h')):
            if gh is None:
                continue
            f_eff = gh.effective_arms
            fe = '—' if not math.isfinite(f_eff) else f'{f_eff:.1f}'
            lines.append(f'{label} = {self._gh_cell_text(gh, va)}  '
                         f'(regular-star f ≈ {fe})')
        # Under a random model, name the branching density n it reads from g (model-
        # dependent -- a different model gives a different n).
        g_res = arow.g_result
        if (self._branching_model != 'regular_star' and g_res is not None
                and g_res.branching_index is not None
                and math.isfinite(g_res.branching_index)):
            model = _BRANCHING_MODEL_LABELS[self._branching_model]
            words = qualifier_words(g_res, ('branching_index',), whole=False)
            lines.append(f'{model}: n ≈ {g_res.branching_index:.1f} branch units/molecule'
                         + (f' ({", ".join(words)})' if words else ''))
        if arow.g_h_comparison:
            lines.append(arow.g_h_comparison)
        primary = arow.g_result or arow.h_result
        if primary is not None and primary.mw_in_range is False:
            lines.append('ⓘ the branched Mw is out of the linear-reference range '
                         '(the law is extrapolated).')
        self._set_interp('\n'.join(lines), headline=arow.label, from_row=True)

    def _build_source_panel(self) -> QtWidgets.QWidget:
        self.source_box = QtWidgets.QGroupBox('Source Selection')
        grid = QtWidgets.QGridLayout(self.source_box)

        # ---- header: explicit Sample + Fraction selectors (the focus, in-tab) -----
        header = QtWidgets.QHBoxLayout()
        header.addWidget(QtWidgets.QLabel('Sample:'))
        self.sample_combo = QtWidgets.QComboBox()
        self.sample_combo.setMinimumWidth(200)
        self.sample_combo.setToolTip('The included sample this panel edits.')
        self.sample_combo.currentIndexChanged.connect(self._on_sample_combo)
        header.addWidget(self.sample_combo, 1)
        self.fraction_label = QtWidgets.QLabel('Fraction:')
        header.addWidget(self.fraction_label)
        self.fraction_combo = QtWidgets.QComboBox()
        self.fraction_combo.setMinimumWidth(120)
        self.fraction_combo.setToolTip(
            'Which Mw fraction of this sample to edit (a sample can hold several '
            'fractions, each its own SLS series and/or DLS measurements — every row of '
            'the table is listed here). Hidden when there is only one.')
        self.fraction_combo.currentIndexChanged.connect(self._on_fraction_combo)
        header.addWidget(self.fraction_combo)
        header.addStretch(1)
        grid.addLayout(header, 0, 0, 1, 6)

        # Each row's label names its quantity and carries the qualifier words of the value
        # the sample stores for it (`_show_row_words`).
        self.rg_row_label = QtWidgets.QLabel()
        self.rh_label = QtWidgets.QLabel()
        self.mw_row_label = QtWidgets.QLabel()
        self.a2_row_label = QtWidgets.QLabel()
        self._row_labels = {
            'rg': (self.rg_row_label, 'R<sub>g</sub> (SLS'),
            'rh': (self.rh_label, 'R<sub>h</sub> (DLS'),
            'mw': (self.mw_row_label, 'M<sub>w</sub> (SLS'),
            'a2': (self.a2_row_label, 'A₂ (SLS')}
        self._show_row_words(None)

        # Column 2 holds each row's badge, right after the combo whose chosen value it
        # marks: the marker of the lines in `source_block` about that value (`⚠ᵃ`, the
        # line's letter), the error tier (bold red ⚠) keeping colour and glyph together as
        # the SLS and DLS tabs mark a value (R5.2).
        self.rg_badge = ThemedLabel('', role='error', size=11, bold=True)
        self.rh_badge = ThemedLabel('', role='error', size=11, bold=True)
        self.mw_badge = ThemedLabel('', role='error', size=11, bold=True)
        self.a2_badge = ThemedLabel('', role='error', size=11, bold=True)

        # Rg row: a combo of SLS candidates + Manual; a manual entry beside it.
        grid.addWidget(self.rg_row_label, 1, 0)
        self.rg_combo = QtWidgets.QComboBox()
        self.rg_combo.activated.connect(lambda i: self._on_source_chosen('rg', i))
        grid.addWidget(self.rg_combo, 1, 1)
        grid.addWidget(self.rg_badge, 1, 2)
        self.rg_manual = QtWidgets.QLineEdit()
        self.rg_manual.setPlaceholderText(self._radius_unit())
        self.rg_manual.setFixedWidth(70)
        grid.addWidget(self.rg_manual, 1, 3)
        self.rg_manual_apparent = QtWidgets.QCheckBox('apparent')
        grid.addWidget(self.rg_manual_apparent, 1, 4)
        rg_set = QtWidgets.QPushButton('Set')
        rg_set.clicked.connect(lambda: self._on_manual('rg'))
        grid.addWidget(rg_set, 1, 5)

        # Rh row: a combo of DLS candidates + Manual; a manual entry beside it.
        # (Disabled for an SLS-only sample — no DLS data to pick from.)
        grid.addWidget(self.rh_label, 2, 0)
        self.rh_combo = QtWidgets.QComboBox()
        self.rh_combo.activated.connect(lambda i: self._on_source_chosen('rh', i))
        grid.addWidget(self.rh_combo, 2, 1)
        grid.addWidget(self.rh_badge, 2, 2)
        self.rh_manual = QtWidgets.QLineEdit()
        self.rh_manual.setPlaceholderText(self._radius_unit())
        self.rh_manual.setFixedWidth(70)
        grid.addWidget(self.rh_manual, 2, 3)
        self.rh_manual_apparent = QtWidgets.QCheckBox('apparent')
        grid.addWidget(self.rh_manual_apparent, 2, 4)
        rh_set = QtWidgets.QPushButton('Set')
        rh_set.clicked.connect(lambda: self._on_manual('rh'))
        grid.addWidget(rh_set, 2, 5)
        self._rh_widgets = [self.rh_combo, self.rh_manual,
                            self.rh_manual_apparent, rh_set]

        # Mw row (feeds the scaling plots): SLS candidates + Manual. No "apparent"
        # box -- a hand-entered Mw is treated as a trusted (calibrated) value.
        grid.addWidget(self.mw_row_label, 3, 0)
        self.mw_combo = QtWidgets.QComboBox()
        self.mw_combo.activated.connect(lambda i: self._on_source_chosen('mw', i))
        grid.addWidget(self.mw_combo, 3, 1)
        grid.addWidget(self.mw_badge, 3, 2)
        self.mw_manual = QtWidgets.QLineEdit()
        self.mw_manual.setPlaceholderText(self._mw_unit())
        self.mw_manual.setFixedWidth(70)
        grid.addWidget(self.mw_manual, 3, 3)
        mw_set = QtWidgets.QPushButton('Set')
        mw_set.clicked.connect(lambda: self._on_manual('mw'))
        grid.addWidget(mw_set, 3, 5)

        # A2 row (feeds the A2-Mw scaling plot): a picker over fit-derived candidates
        # ONLY (Zimm/Berry, and the apparent single-angle ladder) — no manual entry. A2 is a solvent/T-specific interaction
        # coefficient (no external "standard" as Mw has) and is the y-axis the scaling
        # plot fits, so a hand-typed value would invite circularity. Scale-dependent,
        # so it carries a calibration badge like Mw.
        grid.addWidget(self.a2_row_label, 4, 0)
        self.a2_combo = QtWidgets.QComboBox()
        self.a2_combo.setToolTip(
            'A₂ from the Zimm/Berry double extrapolation (thermodynamic) or a single-angle '
            'ladder (apparent, angle-contingent). Calibration-dependent: an uncalibrated '
            'A₂ is on an arbitrary scale (Theory-and-Equations-Guide §1.7.1).')
        self.a2_combo.activated.connect(lambda i: self._on_source_chosen('a2', i))
        grid.addWidget(self.a2_combo, 4, 1)
        grid.addWidget(self.a2_badge, 4, 2)
        self.a2_unit_label = QtWidgets.QLabel('mol·mL/g²')
        grid.addWidget(self.a2_unit_label, 4, 3)

        # "Show all" reveals the single-condition tail (per-angle cumulants, per-conc
        # Debye/Guinier) the pickers leave out to stay scannable; a value never selected
        # for you is always listed (`_fill_candidate_combo`).
        self.show_all_check = QtWidgets.QCheckBox('Show all single-condition results')
        self.show_all_check.setToolTip(
            'Reveal the per-angle / per-concentration apparent results the pickers leave '
            'out.')
        self.show_all_check.toggled.connect(self._on_show_all_toggled)
        grid.addWidget(self.show_all_check, 5, 1, 1, 5)

        # The chosen values' failed preconditions (`_show_value_notes`): one line per cause,
        # its remedy on a click; empty and hidden while the stored values failed none.
        self.source_block = NotesBlock()
        grid.addWidget(self.source_block, 6, 0, 1, 6)

        grid.setColumnStretch(1, 1)
        # Let the long-content value combos shrink and elide rather than dictate the
        # panel width: the bottom strip now shares its row with the context panel, so
        # a combo sized to its widest item ("Zimm extrapolation, 2 concentrations…")
        # would push the context panel off-screen. A modest floor keeps them readable.
        for _c in (self.rg_combo, self.rh_combo, self.mw_combo, self.a2_combo):
            _c.setSizeAdjustPolicy(
                QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            _c.setMinimumContentsLength(10)
        add_help_to_groupbox(
            self.source_box,
            'Pick the value each quantity uses (User-Manual §10.1.2–10.1.3):',
            bullets=[
                '<b>Rg, Rh, Mw, A₂</b>: the best-tier value is the default; a word beside '
                'the row\'s name says what the chosen value is.',
                '<b>Flagged candidates</b>: one whose fit failed a check shows the cause '
                'after its label, and once chosen is marked ⚠; click its line below for '
                'the fix.',
                '<b>Listed but never selected for you</b>, among others: an Rg, Mw or A₂ '
                "failing a check other than scale, a distribution peak at the Rh grid's "
                "end, a mode family's Rh.",
                '<b>Model-dependent</b>: a form-factor Rg or Mw carries the word '
                '<i>model-dependent</i> and is never the default over a slope-based value '
                'at the same tier.',
                '<b>Single-angle ladder</b>: an apparent Mw and A₂ per angle you ran, '
                'never selected for you, since the angle is your choice.',
                '<b>Show all single-condition results</b>: reveals the hidden per-angle '
                'and per-concentration values; one never selected for you is always '
                'listed.',
                '<b>Manual entry</b>: a trusted external Rg, Rh or Mw no run replaces; A₂ '
                'has none.',
            ])
        self.source_box.setEnabled(False)
        return self.source_box

    # ------------------------------------------------------------- refresh ---
    def refresh(self) -> None:
        """Rebuild from the workspace. Called by the shell when this tab is shown
        or after a commit (the sample set or its results may have changed)."""
        # refresh() runs the labeled Rg/Rh/Mw/A2 auto-picks, which WRITE
        # SampleResult fields — so it must not run while a background fit is
        # writing them too. Defer until the worker frees.
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
            item = QtWidgets.QListWidgetItem(sample_display_name(by_id[sid]))
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

        # Default Rg / Rh / Mw for each included sample's fractions (labeled; never
        # clobbers a hand-entered value). Rh is a no-op for an SLS-only sample.
        for sid, inc in self._included.items():
            if inc:
                self._auto_select_all(sid)

        self._recompute()

        # Rebuild the in-tab focus selector (the Sample combo) from the included
        # samples and focus one — keeping the current sample/fraction if still valid.
        self._populate_sample_combo()

    def _auto_select_all(self, sid: str) -> None:
        """Run the labeled default Rg/Rh/Mw/A2 picks for every fraction of a sample
        (the union of its SLS and DLS fraction labels)."""
        for frac in self.controller.sample_fractions_any(sid):
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
        """Architecture-table headers with the active Rg/Rh unit label."""
        ru = self._radius_unit()
        cols = list(_COL_TEMPLATE)
        cols[_C_RG], cols[_C_RH] = f'R<sub>g</sub> ({ru})', f'R<sub>h</sub> ({ru})'
        return cols

    def _disp_radius(self, x: Optional[float], se: Optional[float], runit: str) -> str:
        """Format a canonical (nm) radius in the active display unit, at the decimal
        place its σ supports; 'n/a' if missing.

        Honour the display unit FIRST, then round at the (converted) σ place — a radius
        unit change is a pure linear scale, so the σ converts with the value and the
        place stays correct. With no honest σ, fall back to the documented
        fixed-sig precision (Settings → no-uncertainty precision); we never fabricate a ±.

        A σ that would round the value itself to zero is shown beside it instead,
        ``4 ± 60`` (ui_style_guide §4), the one case the cell prints a ±; the callers
        give that ± its estimator's word.
        """
        if x is None or not (isinstance(x, (int, float)) and math.isfinite(x)):
            return 'n/a'
        v_disp = U.from_canonical('radius', x, runit)
        sig = self.controller.settings.no_uncertainty_sig_figs
        if has_honest_se(se):
            se_disp = U.from_canonical('radius', se, runit)
            if pdg_erases_value(v_disp, se_disp):
                # Two sig figs on the spread so a leading-1 σ is not coarsened (invariant
                # 8's must-not-under-report rule), as `gui/dls_module.py`'s shape cells.
                return f'{format_fixed_sig(v_disp, sig)} ± {format_fixed_sig(se_disp, 2)}'
            return format_value_at_uncertainty(v_disp, se_disp)
        return format_fixed_sig(v_disp, sig)

    def _unit_label(self, sample, fraction: Optional[str]) -> str:
        base = sample_display_name(sample)
        return f'{base} — {fraction}' if fraction is not None else base

    def _rebuild_table(self) -> None:
        self._suppress = True
        # Re-apply headers so a Display-units change updates the Rg/Rh unit labels.
        self.table.setHorizontalHeaderLabels(self._columns())
        runit = self._radius_unit()
        sig = self.controller.settings.no_uncertainty_sig_figs
        samples = self.controller.workspace.samples
        # The branching g/h laws + per-branched rows (also sets the caption). Cached on
        # self._arch for the row-select detail. Rows are per (sample, Mw).
        arch = self._arch_data()
        # One row per (sample, fraction) over EVERY included sample (broadened from the
        # old ρ-only gate): g needs only Rg, h needs only Rh, so an SLS-only or Rh-only
        # sample earns a row too -- its ρ/Shape/other-radius cells show — (not applicable).
        self._row_units = []
        for sid, inc in self._included.items():
            if not inc:
                continue
            for frac in self.controller.sample_fractions_any(sid):
                self._row_units.append((sid, frac))
        self.table.setRowCount(len(self._row_units))
        rho_cells = []                   # (row label, its ρ, the ρ cell) for the block
        radius_cells = []                # (the Rg or Rh cell, its stored notes, its field)
        for row, (sid, frac) in enumerate(self._row_units):
            label = self._unit_label(samples[sid], frac)   # samples: the sid→Sample map
            chain = _CHAIN_DISPLAY.get(self.controller.sample_chain_type(sid), _EMPTY)
            res = samples[sid].result_for(frac)
            try:
                rho = self.controller.compute_sample_rho(sid, frac)
            except Exception:  # noqa: BLE001 — row isolation, cell left blank
                rho = None
            if rho is not None:
                # The ± words ride IN the cell, so an OLS ± never shows unlabeled
                # (invariant 8 clause A); the header's "± statistical" holds for every
                # row, and the tooltip carries each word's clause.
                rho_text, rho_tip = format_fixed_sig(rho.rho, sig), rho.interpretation
                if rho.rho_se is not None:
                    rho_text = _with_words(
                        format_pm(rho.rho, rho.rho_se),
                        qualifier_words(rho, ('rho_se',), whole=False,
                                        exclude=('se_regression',)))
                    rho_tip += '\n' + qualifier_tooltip(rho, ('rho_se',), whole=False)
                rg_txt = self._disp_radius(rho.rg_nm, rho.rg_se, runit)
                rh_txt = self._disp_radius(rho.rh_nm, rho.rh_se, runit)
                rg_tip = rho.rg_label if rg_txt != 'n/a' else 'no Rg (SLS) for this sample'
                rh_tip = rho.rh_label if rh_txt != 'n/a' else 'no Rh (DLS) for this sample'
                cells = {
                    _C_RHO: (rho_text, rho_tip),
                    _C_SHAPE: (rho.shape, rho.interpretation),
                    _C_TYPE: (', '.join(qualifier_words(rho)) or 'thermodynamic',
                              qualifier_tooltip(rho)),
                }
            else:
                # No ρ (a radius is missing) -> ρ/Shape/Type blank, but show whichever
                # radius the sample DOES carry (an SLS-only sample still has its Rg, and
                # still gets g from `arch` below); only the missing radius shows —.
                have_rg = res.rg_nm is not None and math.isfinite(res.rg_nm)
                have_rh = res.rh_nm is not None and math.isfinite(res.rh_nm)
                rg_txt = self._disp_radius(res.rg_nm, res.rg_se, runit) if have_rg else _EMPTY
                rh_txt = self._disp_radius(res.rh_nm, res.rh_se, runit) if have_rh else _EMPTY
                rg_tip = (res.rg_label or '') if have_rg else 'no Rg (SLS) for this sample'
                rh_tip = (res.rh_label or '') if have_rh else 'no Rh (DLS) for this sample'
                _rho_tip = 'ρ needs both an SLS Rg and a DLS Rh'
                cells = {
                    _C_RHO: (_EMPTY, _rho_tip),
                    _C_SHAPE: (_EMPTY, _rho_tip),
                    _C_TYPE: (_EMPTY, _rho_tip),
                }
            # Each radius takes the qualifier words of the stored value it shows.
            for col, txt, tip, attr in ((_C_RG, rg_txt, rg_tip, 'rg_nm'),
                                        (_C_RH, rh_txt, rh_tip, 'rh_nm')):
                words = qualifier_words(res, (attr,), whole=False,
                                        exclude=_CELL_WORDS_EXCLUDE)
                clauses = qualifier_tooltip(res, (attr,), whole=False,
                                            exclude=_CELL_WORDS_EXCLUDE)
                # A ± shows only beside an erased value (`_disp_radius`). It reads
                # "± statistical only" (ui_style_guide §5), and an OLS one also names its
                # estimator (invariant 8 clause A), read off the per-quantity tag the
                # stored radius carries.
                se_attr = attr[:2] + '_se'
                if '±' in txt:
                    pm = [Note('se_regression', (se_attr,))]
                    if getattr(res, se_attr + '_estimator', None) == 'ols':
                        pm.append(Note('se_ols', (se_attr,)))
                    words = [vocabulary(n) for n in pm] + words
                    clauses = '\n'.join(t for t in [render(n, counted=True) for n in pm]
                                        + [clauses] if t)
                shown = _with_words(txt, words)
                cells[col] = (shown, '\n'.join(t for t in (tip, clauses) if t)
                              if shown != txt else tip)
            cells[_C_SAMPLE] = (label, '')
            cells[_C_CHAIN] = (chain, 'Tag on the Data tab (Chain type)'
                               if chain == _EMPTY else 'Set on the Data tab')
            # g / h from the architecture computation, joined by (sample, this Mw).
            arow = (arch.row_for(sid, samples[sid].result_for(frac).effective_mw())
                    if arch is not None else None)
            for col, gh, va in ((_C_G, arow.g_result if arow else None, 'g'),
                                (_C_H, arow.h_result if arow else None, 'h')):
                if gh is None:
                    cells[col] = (_EMPTY, self._gh_missing_tip(sid, va))
                else:
                    cells[col] = (self._gh_cell_text(gh, va), gh.interpretation)
            cells[_C_N] = self._n_cell(sid, arow.g_result if arow else None)
            for col in range(len(_COL_TEMPLATE)):
                text, tip = cells.get(col, (_EMPTY, ''))
                if col in (_C_RG, _C_RH) and text not in (_EMPTY, 'n/a'):
                    tip = f'{text}\n{tip}' if tip else text   # full value first: a cell elides
                item = QtWidgets.QTableWidgetItem(text)
                if tip:
                    item.setToolTip(tip)
                self.table.setItem(row, col, item)
                if col == _C_RHO and rho is not None:
                    rho_cells.append((label, rho, item))
                if col in (_C_RG, _C_RH) and text not in (_EMPTY, 'n/a'):
                    field = 'rg_nm' if col == _C_RG else 'rh_nm'
                    radius_cells.append((item, self._stored_value_notes(res, (field,)),
                                         field))
        # One source: the readout's relayed scale gates name their own samples (`where`).
        self.arch_block.show_notes(self._arch.notes if self._arch is not None else ())
        for _lbl, r, item in rho_cells:
            self.arch_block.mark_item(item, r, ('rho', 'rho_se'))
        # A radius shown marks its stored value's failed checks, as its source row does, the
        # lines after the readout's; ρ and the shape built from it carry no marker of them.
        self.arch_block.append_notes(tuple(n for _i, notes, _f in radius_cells for n in notes))
        for item, notes, field in radius_cells:
            tip = item.toolTip()
            if self.arch_block.mark_item(item, notes, (field,), whole=False) and tip:
                item.setToolTip(item.toolTip() + '<hr>'
                                + html.escape(tip).replace('\n', '<br>'))
        self._suppress = False

    def _gh_missing_tip(self, sid: str, which: str) -> str:
        """Why a g or h cell is blank for this sample."""
        radius = 'Rg (SLS)' if which == 'g' else 'Rh (DLS)'
        if self.controller.sample_chain_type(sid) != 'branched':
            return f'{which} is computed only for a Branched sample (tag it on the Data tab)'
        return (f'no {which}: needs this sample’s {radius} and a linear '
                f'{"Rg" if which == "g" else "Rh"}–Mw reference (≥2 Linear samples)')

    _SCALE_NAMES = {'rg': 'Rg–Mw', 'rh': 'Rh–Mw', 'a2': 'A₂–Mw'}
    _SCALE_NEEDS = {'rg': 'Rg', 'rh': 'Rh', 'a2': 'positive A₂'}
    def _on_swap_size(self) -> None:
        """Toggle the top scaling plot between Rg–Mw and Rh–Mw."""
        self._size_quantity = 'rh' if self._size_quantity == 'rg' else 'rg'
        self.swap_size_button.setText(
            'Show Rg–Mw' if self._size_quantity == 'rh' else 'Show Rh–Mw')
        self._refresh_scaling()

    def _refresh_scaling(self) -> None:
        # Reads SampleResult (compute_scaling) -- deferred past a running fit like
        # _refresh_architecture; the sub-tab switch and the swap button fire any time.
        if runner().is_busy:
            run_when_idle(self._refresh_scaling)
            return
        included = [sid for sid, inc in self._included.items() if inc]
        heads, notes, head_tips, scale_notes = [], [], [], []
        sig = self.controller.settings.no_uncertainty_sig_figs
        # Top plot is the chosen size axis (Rg or Rh); bottom is always A2.
        for ax, q in ((self.ax_rg, self._size_quantity), (self.ax_a2, 'a2')):
            ax.clear()
            sd = self.controller.compute_scaling(included, q)
            plot_scaling(sd.fit, q, sd.labels, ax=ax,
                         branched_x=sd.branched_mw, branched_y=sd.branched_y,
                         branched_labels=sd.branched_labels)
            name = self._SCALE_NAMES[q]
            # What the plotted VALUES are, not the fit: built BEFORE the branch and appended
            # to both, since an invalid fit still draws its points. The model-dependence and
            # degenerate words are the plot's qualifiers (R5.6); a point on an arbitrary
            # scale is a line of the block below, its points named (sd.notes).
            words = qualifier_words(sd)
            qual = f'  ({", ".join(words)})' if words else ''
            scale_notes.extend(sd.notes)
            if sd.fit.fit_valid:
                # 'ν' is the size exponent (Rg/Rh ~ Mw^ν); for A₂ the log–log slope
                # is the raw exponent (A₂ ~ Mw^slope, negative in a good solvent).
                sym = 'slope' if q == 'a2' else 'ν'
                # The two-point ladder has no residual dof, so exponent_se is None --
                # never format_pm's own fallback.
                exp_text = (format_pm(sd.fit.exponent, sd.fit.exponent_se)
                            if sd.fit.exponent_se is not None else
                            format_fixed_sig(sd.fit.exponent, sig))
                # A shown ± carries the fit's ± words (statistical only; OLS when the
                # estimator is the opt-in) on the headline, the one always-visible surface
                # this fit has (invariant 8 clause A); their clauses are its tooltip.
                if sd.fit.exponent_se is not None:
                    exp_text = _with_words(exp_text,
                                           qualifier_words(sd.fit, ('exponent_se',)))
                    head_tips.append(qualifier_tooltip(sd.fit, ('exponent_se',)))
                heads.append(f'{name}: {sym} = {exp_text}')          # glanceable
                bullet = f'• {name} ({len(sd.mw)} pts): ' \
                         + self.controller.interpret_exponent(sd.fit.exponent, q)
                if sd.branched_mw:
                    bullet += (f'  [{len(sd.branched_mw)} branched point(s) shown '
                               'hollow, excluded from the fit]')
                if sd.n_excluded:
                    bullet += f'  [{sd.n_excluded} lacked Mw or a positive value to plot]'
                notes.append(bullet + qual)
            else:
                msg = (f'• {name}: need ≥2 points with Mw + '
                       f'{self._SCALE_NEEDS[q]} to fit a slope')
                if sd.branched_mw:
                    msg += (f'  ({len(sd.branched_mw)} branched point(s) shown hollow — '
                            'tag ≥2 Linear samples to fit the line)')
                notes.append(msg + qual)
        self.scaling_fig.tight_layout()
        self.scaling_canvas.draw_idle()
        self.scaling_headline.setText(' · '.join(heads))
        self.scaling_headline.setToolTip('\n'.join(dict.fromkeys(
            line for tip in head_tips for line in tip.splitlines())))
        self.scaling_headline.setVisible(bool(heads))
        self.scaling_note.setText('\n'.join(notes))
        self.scaling_block.show_notes(tuple(scale_notes))

    @QtCore.Slot()
    def _on_export_scaling(self) -> None:
        # Recomputes at export time -- refused while a job is in flight, like every
        # computing sibling (the SLS exports); an export never reads a half-written fit.
        if runner().is_busy:
            busy_notice(self)
            return
        included = [sid for sid, inc in self._included.items() if inc]
        if not included:
            self.scaling_note.setText('Include at least one sample to export.')
            return

        def do_export(path: str):
            import os
            stem, ext = os.path.splitext(path)
            # BOTH written files are derived siblings the dialog never showed
            # (its own overwrite confirm covered only the literal chosen path).
            targets = [f'{stem}_{q}{ext}' for q in (self._size_quantity, 'a2')]
            if not confirm_overwrite(self, targets):
                return None
            written = []
            for q, p in zip((self._size_quantity, 'a2'), targets, strict=True):
                sd = self.controller.compute_scaling(included, q)
                written.append(self.controller.export_scaling(sd, p))
            return written                     # summarised by `export_to_csv`

        status = export_to_csv(self, self.controller, 'scaling.csv', do_export)
        if status:
            self.scaling_note.setText(status)

    @QtCore.Slot()
    def _on_export_architecture(self) -> None:
        if runner().is_busy:                     # recomputes ρ and g/h at export time
            busy_notice(self)
            return
        included = [sid for sid, inc in self._included.items() if inc]

        def do_export(path: str) -> str:
            return self.controller.export_architecture(
                included, path, branching_model=self._branching_model)

        status = export_to_csv(self, self.controller, 'architecture.csv', do_export)
        if status:
            self.arch_caption.setText(status)

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
        # (the labeled Rg/Rh/Mw/A2 auto-picks) -- it must not race a background fit
        # writing them too, exactly as refresh() guards. Defer the whole
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
            self.sample_combo.addItem(sample_display_name(by_id[sid]))
            self.sample_combo.setItemData(
                self.sample_combo.count() - 1, sid, QtCore.Qt.ItemDataRole.UserRole)
        self._suppress = False
        if not included:
            self._current_unit = None
            self.source_box.setEnabled(False)
            self._show_row_words(None)
            self._set_interp('')
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
        """Fill the Fraction combo with the sample's fractions -- the SLS ∪ DLS union
        the table rows enumerate, so every row is reachable from here -- and return the
        chosen one. Hidden when the sample has a single unlabeled fraction."""
        fracs = list(self.controller.sample_fractions_any(sid)) or [None]
        self._suppress = True
        self.fraction_combo.clear()
        for frac in fracs:
            self.fraction_combo.addItem('(unlabeled)' if frac is None else str(frac))
            self.fraction_combo.setItemData(
                self.fraction_combo.count() - 1, frac,
                QtCore.Qt.ItemDataRole.UserRole)
        chosen = prefer if prefer in fracs else fracs[0]
        self.fraction_combo.setCurrentIndex(max(fracs.index(chosen), 0))
        self._suppress = False
        single_unlabeled = (len(fracs) == 1 and fracs[0] is None)
        self.fraction_label.setVisible(not single_unlabeled)
        self.fraction_combo.setVisible(not single_unlabeled)
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
        # A user pick here drives the shell's single active sample; the shell
        # fans it out to every tab + the sidebar and calls back set_focused_sample.
        self.sampleFocusRequested.emit(sid)

    def set_focused_sample(self, sid: Optional[str]) -> None:
        """Focus `sid` in the source panel to follow the shell's active sample. Two
        out-of-universe cases, distinguished so the panel is never silently out of sync
        with the shell; membership is unchanged either way:
          - `sid` has no SLS data → not part of Cross (aggregate over SLS-bearing
            samples); keep the current panel (the documented no-op).
          - `sid` HAS SLS data but is currently excluded (unticked) → show an
            excluded-state notice rather than keep showing the PREVIOUS sample."""
        if not sid:
            return
        if self._sample_combo_index(sid) >= 0:
            self._focus_sample(sid)                  # included → focus, first fraction
        elif sid in self._included:                  # SLS-bearing but excluded
            self._show_excluded_notice(sid)
        # else: no SLS data — not a Cross sample; keep the current panel (no-op).

    def _show_excluded_notice(self, sid: str) -> None:
        """Source-panel empty-state for a sample the shell focused that is excluded
        from Cross-Sample (unticked in the membership list). Mirrors the no-samples
        empty-state (disable the box, drop the current unit so a later action can't
        write to a stale sample) but explains why."""
        self._current_unit = None
        self.source_box.setEnabled(False)
        self._show_row_words(None)
        sample = self.controller.workspace.samples.get(sid)
        label = sample_display_name(sample) if sample is not None else sid
        self._set_interp(
            f'{label} is excluded from Cross-Sample (unticked in the membership list). '
            'Re-tick it to pick sources or enter manual values.')
        self.selectionChanged.emit()                 # clear the tree mirror

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
        if self._current_unit is None:
            return
        # The repopulate reads SampleResult a fit may be writing: defer, re-reading the
        # live box at idle (deferred, never dropped -- the _on_selection_changed pattern).
        if runner().is_busy:
            run_when_idle(lambda: self._on_show_all_toggled(self.show_all_check.isChecked()))
            return
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
        # A typed-but-unsubmitted manual value is bound to the sample it was typed for.
        # When the panel switches to a DIFFERENT (sample, fraction), clear the manual entry
        # widgets so a later "Set" cannot write the stale text into the newly focused sample
        # as trusted 'user' provenance. Compare against _manual_unit (the panel's own last
        # target), NOT _current_unit — the callers set _current_unit before repopulating.
        # Clear only on a real target change: a repopulate for the SAME unit must not wipe
        # what the user is mid-typing. The manual widgets have no textChanged/toggled slots,
        # so clearing fires nothing.
        if self._manual_unit != (sid, fraction):
            for edit in (self.rg_manual, self.rh_manual, self.mw_manual):
                edit.clear()
            self.rg_manual_apparent.setChecked(False)
            self.rh_manual_apparent.setChecked(False)
        self._manual_unit = (sid, fraction)
        self._current_unit = (sid, fraction)
        # Manual-entry placeholders follow the active Display-units choice.
        self.rg_manual.setPlaceholderText(self._radius_unit())
        self.rh_manual.setPlaceholderText(self._radius_unit())
        self.mw_manual.setPlaceholderText(self._mw_unit())
        r = self.controller.workspace.samples[sid].result_for(fraction)
        hidden = 0
        hidden += self._fill_candidate_combo(
            self.rg_combo, self.controller.sls_rg_candidates(sid, fraction),
            current_result=r, result_quantity='rg', quantity='radius')
        hidden += self._fill_candidate_combo(
            self.mw_combo, self.controller.sls_mw_candidates(sid, fraction),
            current_result=r, result_quantity='mw', quantity='molar_mass')
        hidden += self._fill_candidate_combo(
            self.a2_combo, self.controller.sls_a2_candidates(sid, fraction),
            current_result=r, result_quantity='a2',
            quantity='molar_mass', allow_manual=False)

        # Rh row: disabled for an SLS-only sample (no DLS data to pick from).
        has_dls = bool(self.controller.workspace.sample_measurements(sid, 'dls'))
        for w in self._rh_widgets:
            w.setEnabled(has_dls)
        self.rh_label.setEnabled(has_dls)
        if has_dls:
            hidden += self._fill_candidate_combo(
                self.rh_combo, self.controller.dls_rh_candidates(sid, fraction),
                current_result=r, result_quantity='rh', quantity='radius')
        else:
            self.rh_combo.blockSignals(True)
            self.rh_combo.clear()
            self.rh_combo.addItem('— no DLS data for this sample —')
            self.rh_combo.blockSignals(False)

        self._show_value_notes(r)
        self._show_row_words(r)

        # "Show all" reveals the hidden single-condition tail; annotate its count.
        base = 'Show all single-condition results'
        self.show_all_check.setText(
            base if (self._show_all_single or not hidden) else f'{base} ({hidden} hidden)')
        self.show_all_check.setEnabled(bool(hidden) or self._show_all_single)

        self.source_box.setEnabled(True)
        self.source_box.setTitle('Source Selection')
        self._set_interp(*self._rho_interp(sid, fraction))

    def _show_value_notes(self, r) -> None:
        """The failed preconditions of the four values the sample stores, as every result
        surface shows a value's: one line per cause in `source_block`, and each row's badge
        the marker of the lines about its value (`⚠ᵃ`, `⚠ᵃᵇ` for two causes), its hover the
        lines' popup (ui_style_guide §5). The lines are the Mw and A2's scale gates
        (`SampleResult.scale_notes`, read through `is_arbitrary_scale`), then the notes the
        stored values carry from the candidates or runs that wrote them (`set_value_notes`);
        a legacy note from an older session marks no value and is not shown."""
        notes = (tuple(r.scale_notes())
                 + self._stored_value_notes(r, tuple(self._STORED_VALUE_FIELD.values())))
        self.source_block.show_notes(notes)
        for badge, quantity in ((self.rg_badge, 'rg'), (self.rh_badge, 'rh'),
                                (self.mw_badge, 'mw'), (self.a2_badge, 'a2')):
            value_field = (self._STORED_VALUE_FIELD[quantity],)
            codes = self.source_block.codes_for(notes, value_field, whole=False)
            badge.setText(self.source_block.marker(notes, value_field, whole=False))
            badge.setToolTip(self.source_block.popup_for(codes))

    def _show_row_words(self, r) -> None:
        """Each source row's label with the qualifier words of the value the sample stores
        for its quantity, `A₂ (SLS, apparent):`, their clauses the label's passive tooltip
        (ui_style_guide §5): what the chosen value is stays in view however far the combo
        beside it elides its label. A row with no stored value, or no panel focus (`r` None),
        shows the bare name."""
        for quantity, (label, name) in self._row_labels.items():
            field = self._STORED_VALUE_FIELD[quantity]
            value = getattr(r, field, None) if r is not None else None
            stored = isinstance(value, (int, float)) and math.isfinite(value)
            words = qualifier_words(r, (field,), whole=False) if stored else []
            label.setText(f'{name}{"".join(", " + w for w in words)}):')
            label.setToolTip(qualifier_tooltip(r, (field,), whole=False) if words else '')

    def _fill_candidate_combo(self, combo: QtWidgets.QComboBox, candidates, *,
                              current_result, result_quantity: str,
                              quantity: str = 'radius',
                              allow_manual: bool = True) -> int:
        """Rebuild a source-picker combo, GROUPED by result type with the long tail of
        single-condition apparents hidden behind the "show all" checkbox.

        Each real item's UserRole data is a sentinel `_on_source_chosen` dispatches on:
        a ResultCandidate (select it), 'USER' (keep the hand-entered value), '__HEADER__'
        (a disabled group heading — ignored), or None (the trailing "Manual entry…").
        Signals are blocked during the rebuild so it doesn't re-enter selection handling.
        The current selection and the default candidate are ALWAYS shown, even if their
        group is hidden. Returns the number of candidates hidden (for the checkbox hint).

        For a 'user' or 'picked' source the current selection is the sample's OWN item,
        carrying its number -- so the candidate it was picked FROM is an ordinary list
        entry and can be hidden behind "show all" like any other. That is the intended
        trade: the number the sample holds is always on screen, and no candidate item can
        claim to be the selection while its own value has moved on.

        `result_quantity` names which of the SampleResult's four quantities this combo is
        for ('rg' / 'mw' / 'a2' / 'rh') -- it selects the provenance key and the label, and
        is NOT the `quantity` argument beside it, which picks display units."""
        current_label = getattr(current_result, f'{result_quantity}_label', '')
        current_source = getattr(current_result, f'{result_quantity}_source', 'computed')
        default = self.controller.default_candidate(list(candidates))
        combo.blockSignals(True)
        combo.clear()
        model = combo.model()
        select_index = -1
        default_index = -1
        hidden = 0

        # ONE spelling of "is this candidate the current selection", used by both the
        # visibility test below and the selection assignment further down -- written twice,
        # only one of them gets widened. A 'user'/'picked' value is excluded: it is shown as
        # its OWN item, carrying its own number. Identity comes from the stored provenance
        # KEY, never the label -- the same value carries two spellings, so a prose test finds
        # only the one it was written against. This is the surface the key exists for: after
        # a reload the fit cache is empty, but the stored key and the re-derived candidates
        # both survive. Fails closed, so a pre-key session preselects nothing.
        def _is_current(c) -> bool:
            return (current_source not in DELIBERATE_TIERS
                    and result_key_matches(current_result, result_quantity,
                                           c.kind, c.source_id))

        # Decide visibility, then group the visible candidates preserving order.
        groups: Dict[str, list] = {}
        for c in candidates:
            if getattr(c, 'model_dependent', False):
                # Form-factor P(q) sizes group under their own model-dependent heading, split
                # by the (orthogonal) thermodynamic/apparent tier. Handled by the flag rather
                # than an entry per sls_ff_<model> kind, so a future model is grouped for free.
                # A DEGENERATE per-solution pick groups one tier stronger, "— degenerate", so the
                # user reads that the shape is one the data cannot distinguish (review-01 G3).
                is_degen = getattr(c, 'degenerate', False)
                suffix = ' — degenerate' if is_degen else ''
                title = (f'Extrapolated (model-dependent{suffix})' if not c.is_apparent
                         else f'Single-condition (model-dependent{suffix})')
                # Hide the apparent tail behind "show all" like the agnostic one -- EXCEPT
                # degenerate per-solution picks, which are the whole point of the feature
                # (offered as explicit choices) and are never the shown default
                # (auto_selectable=False), so hiding a single-condition degenerate fit would
                # make its solutions invisible until the user found the checkbox.
                # Degenerate picks stay visible. `and not is_degen` is REDUNDANT today,
                # both degenerate producers setting auto_selectable=not degenerate; the
                # collapse is gated on a producer-side test pinning that implication.
                hide_default = bool(c.is_apparent) and not is_degen
            else:
                title, hide_default = _KIND_GROUP.get(c.kind, ('Other', False))
            is_current = _is_current(c)
            is_default = (default is not None and c is default)
            # A candidate with auto_selectable=False can never BE the default, so for it
            # `hide_default` stops meaning "de-default" and starts meaning HIDE OUTRIGHT --
            # the opposite of "offered in the picker, out of the passive default", which is
            # what de-defaulting it was for. The degenerate branch above documents this
            # exact trap; stating the exemption ONCE here means every
            # hide_default kind inherits it, including sls_debye and sls_guinier the day one
            # of them is de-defaulted -- rather than a third hand-added predicate.
            if (hide_default and getattr(c, 'auto_selectable', True)
                    and not self._show_all_single and not (is_current or is_default)):
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
                # Full label on hover -- the combo elides it (R3.6).
                combo.setItemData(idx, f'    {c.label}'.strip(),
                                   QtCore.Qt.ItemDataRole.ToolTipRole)
                if _is_current(c):
                    select_index = idx
                if default is not None and c is default:
                    default_index = idx

        # A value the USER settled -- typed in, or picked from this list -- shows as a
        # distinct, pre-selected item carrying its own number (in display units). The 'USER'
        # sentinel means "keep what is stored", which `_on_source_chosen` already honours.
        # 'picked' shares this branch with 'user' rather than having its own: both are the
        # one case "the sample's value is not any candidate's current value". A picked value
        # that IS still its candidate's value reads identically, so nothing is lost in the
        # ordinary case, and the stale-label case cannot arise at all because no candidate
        # item claims to be the selection. The source is still named, after the number.
        if current_source in DELIBERATE_TIERS:
            # Value AND unit are keyed on `result_quantity`, the token that says which
            # quantity this combo is for. The previous form read the value by comparing the
            # combo against rg_combo/mw_combo and FELL THROUGH to rh_nm, so an explicitly
            # picked A2 showed the sample's Rh -- run through the molar-mass converter and
            # labelled g/mol. There is no fall-through left to be wrong.
            val = self._stored_value(current_result, result_quantity)
            if result_quantity == 'a2':
                # A2 has no display-unit setting anywhere (the panel's own label is the
                # fixed `mol·mL/g²`), so it is shown in its canonical unit, unconverted.
                unit, disp = 'mol·mL/g²', val
            else:
                unit = (self._mw_unit() if quantity == 'molar_mass'
                        else self._radius_unit())
                disp = (U.from_canonical(quantity, val, unit)
                        if isinstance(val, (int, float)) and math.isfinite(val) else val)
            if current_source == 'user':
                combo.addItem(f'User-entered ({_fmt(disp)} {unit})')
            else:
                combo.addItem(f'Picked ({_fmt(disp)} {unit})'
                              + (f' — {current_label}' if current_label else ''))
            combo.setItemData(combo.count() - 1, 'USER', QtCore.Qt.ItemDataRole.UserRole)
            # Full label on hover -- the combo elides it (R3.6).
            combo.setItemData(combo.count() - 1, combo.itemText(combo.count() - 1).strip(),
                               QtCore.Qt.ItemDataRole.ToolTipRole)
            select_index = combo.count() - 1
        if allow_manual:
            combo.addItem(_MANUAL_LABEL)
            combo.setItemData(combo.count() - 1, None, QtCore.Qt.ItemDataRole.UserRole)
        elif not groups and current_source not in DELIBERATE_TIERS:
            # no fit-derived candidates and no manual path: a disabled hint
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

    # Which SampleResult field holds each quantity's stored value. A table rather than an
    # if/elif chain because the chain's last arm was an unguarded fall-through: anything
    # that was not the Rg or Mw combo returned rh_nm, which made the A2 picker display the
    # sample's Rh. A missing key now raises instead of quietly answering with Rh.
    _STORED_VALUE_FIELD = {'rg': 'rg_nm', 'mw': 'mw_g_per_mol',
                           'a2': 'a2_mol_mL_per_g2', 'rh': 'rh_nm'}

    @staticmethod
    def _stored_value_notes(result, fields) -> tuple:
        """The failed preconditions `result` stores about the value fields `fields`, written
        by the candidate or run that stored each value (`set_value_notes`); a legacy note
        from an older session marks no value and is left out."""
        return tuple(n for n in result.notes
                     if set(n.about) & set(fields) and REGISTRY.get(n.code) is not None
                     and REGISTRY[n.code].kind == 'precondition')

    @classmethod
    def _stored_value(cls, result, result_quantity: str) -> Optional[float]:
        """The sample's stored value for one quantity, keyed on the same token the rest of
        `_fill_candidate_combo` reads its label, source and provenance key with."""
        return getattr(result, cls._STORED_VALUE_FIELD[result_quantity])

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
            self._pending_manual[(which, sid, frac)] = (value, apparent)
            run_when_idle(self._flush_manual)
            self._set_interp(
                'Entry queued — it will apply when the running analysis finishes.')
            return
        self._apply_manual(which, sid, frac, value, apparent)
        edit.clear()

    def _apply_manual(self, which, sid, frac, value, apparent, *,
                      repopulate: bool = True) -> None:
        # The setters refuse a non-positive or non-finite value. `to_canonical` above
        # does NOT -- '0', '-1' and '1e400' all convert cleanly -- so the refusal has to
        # be caught here or it escapes into a Qt slot. Caught in _apply_manual rather
        # than at the call site because BOTH paths reach it: the direct entry and
        # _flush_manual's replay of a value queued while a fit was running.
        try:
            if which == 'mw':
                self.controller.set_manual_mw(sid, value, frac)
            else:
                setter = (self.controller.set_manual_rg if which == 'rg'
                          else self.controller.set_manual_rh)
                setter(sid, value, is_apparent=apparent, fraction=frac)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, 'Invalid value', str(exc))
            return
        self._recompute()
        if repopulate:
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
        for (which, sid, frac), (value, apparent) in pending.items():
            self._apply_manual(which, sid, frac, value, apparent, repopulate=False)
            if self._current_unit == (sid, frac):
                edits[which].clear()      # else the box holds text typed for the NEW unit
        # Restore the panel to the sample actually in focus, not whichever queued
        # entry happened to apply last.
        if self._current_unit is not None:
            self._populate_source_panel(*self._current_unit)
