"""
gui/main_window.py
==================

The application shell. It owns the single `Controller`, a left-hand sidebar that
navigates the loaded measurements (auto-grouped into samples), and a `QTabWidget`
with the six agreed modules:

    Data | DLS | SLS | Cross-Sample | Utilities | Settings

Design (settled with the user)
------------------------------
* Sidebar = the live workspace: samples (auto-grouped by polymer/solvent/rounded
  temperature) with their measurements beneath. Loading and selection are global;
  picking a measurement points every module at it.
* The modules have different SCOPES, and the sidebar adapts:
    - sample-scoped (Data, DLS, SLS, Utilities): sidebar is the navigator;
    - aggregate (Cross-Sample): sidebar will become an include/exclude list;
    - global (Settings): sidebar is irrelevant and is disabled.
* Data owns parameter editing + commit; the analysis modules read committed
  parameters and never edit them. Committing can change a sample's identity, so
  the Data module signals the shell to re-group and refresh the sidebar.

All six modules are built and wired here (the old StubModule placeholders in
gui/stub_module.py are no longer used). The navigation and the
load -> confirm -> analyze loop run end to end through the controller.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from app.version import __version__
from app.controller import Controller
from core.workspace import _DLS_PARAM_KEYS, _SLS_PARAM_KEYS
from parsers.base_parser import ParseError
from parsers.brookhaven_dls import BrookhavenDLSParser
from parsers.brookhaven_sls import BrookhavenSLSParser
from parsers.alv_asc import ALVCorrelatorParser
from parsers.zetasizer_clipboard import ZetasizerClipboardParser
from parsers.zetasizer_export import ZetasizerExportParser
from parsers.generic_dls import GenericDLSParser
from parsers.generic_sls import GenericSLSParser

# Instrument-specific parsers tried, in order, by the format auto-detection on load.
# Each sniffs its own format and raises ParseError when the file is not its own, so
# the first that succeeds is the right one. The lenient *generic* parsers are NOT
# here (they would match almost anything); they are tried only as an explicit
# fallback after instrument detection fails -- the load button declares the data
# kind, and generic DLS prompts for its delay unit + data form.
_DLS_PARSERS = [ALVCorrelatorParser, ZetasizerClipboardParser,
                ZetasizerExportParser, BrookhavenDLSParser]
_SLS_PARSERS = [BrookhavenSLSParser]
_DELAY_UNITS = ['s', 'ms', 'us', 'ns']
_DATA_FORMS = [('g2(τ) − 1', 'g2m1'), ('g2(τ)', 'g2'), ('g1(τ)', 'g1')]

from gui.data_module import DataModule
from gui.dls_module import DLSModule, ask_average_method, show_average_summary
from gui.sls_module import SLSModule
from gui.cross_module import CrossSampleModule
from gui.utilities_module import UtilitiesModule
from gui.settings_module import SettingsModule
from gui.help import install_tooltip_gate, set_tooltips_enabled, section_header
from gui.theme import ThemedLabel, color as theme_color, retheme, is_dark
from gui.widgets import roomy_tabs, EmptyState
from gui.worker import runner, run_when_idle
from plotting.plots import (
    set_palette, set_plot_units, set_onscreen_plot_theme, apply_figure_theme)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg


class _WheelGuard(QtCore.QObject):
    """Application event filter: wheel events on QComboBox / QTabBar are
    suppressed so the scroll wheel only scrolls, never cycles options or tabs."""

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Type.Wheel:
            if isinstance(obj, (QtWidgets.QComboBox, QtWidgets.QTabBar)):
                event.ignore()
                return True
        return False


class MainWindow(QtWidgets.QMainWindow):
    """The shell: sidebar navigator + module tabs over one shared controller."""

    def __init__(self) -> None:
        super().__init__()
        self._wheel_guard = _WheelGuard(self)
        QtWidgets.QApplication.instance().installEventFilter(self._wheel_guard)
        self.setWindowTitle(f'DLS Buddy {__version__}')
        self.resize(1280, 760)

        self.controller = Controller()
        self.current_item: Optional[str] = None
        # The single ACTIVE SAMPLE: sample_of(current_item), the sample every
        # sample-scoped tab follows. Driveable from the tree or any tab's dropdown.
        self.active_sample_id: Optional[str] = None
        self._syncing_sample = False        # re-entrancy guard for _focus_sample
        # item_id -> its measurement leaf in the tree, kept so the selection mirror can
        # repaint rows without a full rebuild (rebuilt each _refresh_sidebar).
        self._leaf_items: Dict[str, QtWidgets.QTreeWidgetItem] = {}

        self._build_ui()
        self._apply_ui_density()
        self._maybe_reopen_session()     # restore the last session if the user opted in
        self._refresh_sidebar()
        if self.current_item is not None:
            self._set_current(self.current_item)
            self.cross_module.refresh()
        self._on_tab_changed(self.tabs.currentIndex())
        # If any saved setting was invalid and reverted to its default on load, tell the
        # user once -- so a hand-edit that didn't take is visible, not silently ignored.
        # Deferred one event-loop tick so the dialog is parented to the shown window.
        if self.controller.settings_load_problems:
            QtCore.QTimer.singleShot(0, self._report_settings_problems)

    def _report_settings_problems(self) -> None:
        """Surface the invalid-settings reversions collected at load (F-17/F-25). Mirrors the
        load-notes pattern in `_on_load_sls`: a non-blocking notice that names each field, its
        rejected value, and the default now in effect."""
        problems = self.controller.settings_load_problems
        if not problems:
            return
        QtWidgets.QMessageBox.warning(
            self, 'Some settings were reset',
            'Some saved settings were invalid and reverted to defaults. Set them again '
            'in the Settings tab if needed.\n\n  ' + '\n  '.join(problems))
        self.controller.settings_load_problems = []   # shown once per launch

    # ------------------------------------------------------------------ UI ---
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QHBoxLayout(central)

        # ---- sidebar (in a draggable splitter so its width is user-adjustable) --
        side_widget = QtWidgets.QWidget()
        side = QtWidgets.QVBoxLayout(side_widget)
        side.setContentsMargins(0, 0, 0, 0)

        self.load_button = QtWidgets.QPushButton('Load DLS correlogram…')
        self.load_button.clicked.connect(self._on_load_dls)
        side.addWidget(self.load_button)

        self.load_sls_button = QtWidgets.QPushButton('Load SLS intensities…')
        self.load_sls_button.clicked.connect(self._on_load_sls)
        side.addWidget(self.load_sls_button)

        side.addWidget(section_header(
            'Workspace', 'Your loaded data, grouped into samples:',
            bullets=[
                'Click a measurement to load it into the tabs.',
                'Click a <b>heading</b> (sample, or DLS/SLS) to select everything '
                'under it — handy for a one-shot parameter edit.',
                '<b>Right-click</b> a heading or selection to remove, move to '
                'another sample, or average replicates.',
                'Tick measurements in the DLS tab to overlay/fit several at once.',
            ]))
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setMinimumWidth(140)
        # Don't truncate long sample/measurement names: let the column grow to its
        # content and show a horizontal scrollbar when it exceeds the panel width
        # (the splitter also lets the user widen the panel). Full text on the tooltip.
        self.tree.setTextElideMode(QtCore.Qt.TextElideMode.ElideNone)
        self.tree.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        # Extended selection lets the user Ctrl/Shift-pick several measurements to
        # remove at once; the active measurement for the tabs follows the current
        # (focused) item.
        self.tree.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.itemSelectionChanged.connect(self._on_tree_selection)
        self.tree.itemClicked.connect(self._on_tree_clicked)
        self.tree.setContextMenuPolicy(
            QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_tree_context_menu)
        del_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key.Key_Delete), self.tree)
        del_shortcut.activated.connect(self._on_delete_key)
        side.addWidget(self.tree, 1)

        self.sidebar_note = ThemedLabel('', role='hint', size=11)
        self.sidebar_note.setWordWrap(True)
        side.addWidget(self.sidebar_note)

        # ---- module tabs ---------------------------------------------------
        self.tabs = roomy_tabs(QtWidgets.QTabWidget())   # roomier tabs so labels don't clip

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.splitter.addWidget(side_widget)
        self.splitter.addWidget(self.tabs)
        self.splitter.setStretchFactor(0, 0)   # sidebar keeps its width
        self.splitter.setStretchFactor(1, 1)   # tabs take the extra space
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setSizes([260, 900])
        outer.addWidget(self.splitter)

        self.data_module = DataModule(self.controller)
        self.dls_module = DLSModule(self.controller)
        self.sls_module = SLSModule(self.controller)
        self.data_module.committed.connect(self._on_committed)
        self.sls_module.committed.connect(self._on_committed)

        self.cross_module = CrossSampleModule(self.controller)
        self.utilities_module = UtilitiesModule(self.controller)
        self.utilities_module.workspaceChanged.connect(self._on_workspace_changed)
        self.settings_module = SettingsModule(self.controller)
        self.settings_module.applied.connect(self._on_settings_applied)
        # Sidebar selection mirror: repaint the tree's "selected in the active tab" marks
        # whenever a sample-scoped analysis tab's own selection changes.
        self.dls_module.selectionChanged.connect(self._on_selection_changed)
        self.sls_module.selectionChanged.connect(self._on_selection_changed)
        self.utilities_module.selectionChanged.connect(self._on_selection_changed)
        # Cross-Sample selects in-tab (its own Sample combo, not the tree), but it
        # read-only MIRRORS the focused sample onto the tree so the two stay coherent.
        self.cross_module.selectionChanged.connect(self._on_selection_changed)
        # Unified active sample: a user pick in ANY tab's sample dropdown
        # sets the single active sample, which the shell fans out to every tab + the
        # sidebar. One source of truth, driveable from the tree or any combo.
        self.sls_module.sampleFocusRequested.connect(self._focus_sample)
        self.utilities_module.sampleFocusRequested.connect(self._focus_sample)
        self.cross_module.sampleFocusRequested.connect(self._focus_sample)

        # Each tab is wrapped in a resizable scroll area: when the window is shrunk
        # below a module's natural height the content scrolls instead of pinning a
        # tall minimum on the whole window (so it can be resized down freely).
        self._tab_modules = [
            self.data_module, self.dls_module, self.sls_module,
            self.cross_module, self.utilities_module, self.settings_module]
        titles = ['Data', 'DLS', 'SLS', 'Cross-Sample', 'Utilities', 'Settings']
        # Map each tab's scroll-area wrapper back to its module, so the active
        # module is resolved by WIDGET, not tab index — tabs are user-reorderable
        # so index order is no longer fixed.
        self._module_by_wrapper: Dict[QtWidgets.QWidget, QtWidgets.QWidget] = {}
        # The data-scoped tabs show a shared empty-state CTA over their content when the
        # workspace has no measurements (style guide §8; owner-confirmed scope). Utilities
        # (its Synthetic Generator + Traces MAKE data from nothing) and Settings (global)
        # are excluded — they stay usable with an empty workspace.
        self._empty_tab_modules = {
            self.data_module, self.dls_module, self.sls_module, self.cross_module}
        # QStackedWidgets whose page 0 is the empty-state CTA and page 1 the module content,
        # flipped by _refresh_empty_state on every workspace change.
        self._empty_stacks: list[QtWidgets.QStackedWidget] = []
        for module, title in zip(self._tab_modules, titles, strict=True):
            wrapper = self._wrap_tab(module)
            self._module_by_wrapper[wrapper] = module
            self.tabs.addTab(wrapper, title)
        self.tabs.setMovable(True)               # drag to reorder main tabs
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self._apply_theme(self.controller.settings.theme)   # honor saved theme
        set_palette(self.controller.settings.plot_palette)  # and saved plot palette
        set_plot_units(self.controller.settings.plot_units)  # and saved plot units
        app = QtWidgets.QApplication.instance()
        if app is not None:
            install_tooltip_gate(app)                       # global tooltip on/off
        set_tooltips_enabled(self.controller.settings.show_tooltips)

        self.statusBar().showMessage('Load data to begin.')
        # Global busy affordance: while a background analysis runs, show a busy
        # cursor (BusyCursor, not WaitCursor — the window stays interactive) and
        # a status-bar message. One analysis in flight app-wide.
        runner().busy_changed.connect(self._on_busy_changed)

    @QtCore.Slot(bool, str)
    def _on_busy_changed(self, busy: bool, description: str) -> None:
        app = QtWidgets.QApplication.instance()
        if busy:
            app.setOverrideCursor(QtCore.Qt.CursorShape.BusyCursor)
            self.statusBar().showMessage(
                f'Running {description} in the background — the window stays '
                'usable; results appear when it finishes.')
        else:
            app.restoreOverrideCursor()
            self.statusBar().showMessage(f'Finished: {description}.', 4000)

    def _busy_guard(self) -> bool:
        """True (with a status notice) when an analysis is in flight. Workspace-
        mutating actions call this first and back off, because the worker reads
        committed parameters, settings, and workspace rows mid-computation."""
        if runner().is_busy:
            self.statusBar().showMessage(
                'Busy — wait for the running analysis to finish.', 3000)
            return True
        return False

    def _show_module(self, module: QtWidgets.QWidget) -> None:
        """Make the tab holding `module` current (by widget, since tabs reorder)."""
        for wrapper, mod in self._module_by_wrapper.items():
            if mod is module:
                self.tabs.setCurrentWidget(wrapper)
                return

    @staticmethod
    def _scrollable(widget: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
        """Wrap a module in a frameless, resizable scroll area so the window can be
        shrunk below the module's natural height (content scrolls)."""
        sa = QtWidgets.QScrollArea()
        sa.setWidgetResizable(True)            # child fills the viewport when it fits
        sa.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        sa.setWidget(widget)
        return sa

    def _wrap_tab(self, module: QtWidgets.QWidget) -> QtWidgets.QWidget:
        """Build the tab's top-level widget. For a data-scoped module (see
        `_empty_tab_modules`) this is a `QStackedWidget` whose page 0 is a shared
        empty-state CTA and page 1 the scroll-wrapped module, flipped by
        `_refresh_empty_state`. Other modules get the plain scroll wrapper."""
        content = self._scrollable(module)
        if module not in self._empty_tab_modules:
            return content
        stack = QtWidgets.QStackedWidget()
        stack.addWidget(EmptyState())          # page 0: onboarding CTA
        stack.addWidget(content)               # page 1: the module content
        self._empty_stacks.append(stack)
        return stack

    def _refresh_empty_state(self) -> None:
        """Show the empty-state CTA (page 0) on the data tabs when the workspace has no
        measurements, else the module content (page 1). Called from `_refresh_sidebar`
        (the single chokepoint hit by every workspace mutation)."""
        has_data = bool(self.controller.workspace.measurements)
        for stack in self._empty_stacks:
            stack.setCurrentIndex(1 if has_data else 0)

    # ------------------------------------------------------------- loading ---
    @staticmethod
    def _autodetect(path: str, parser_classes):
        """Try each instrument parser in turn; return the previews from the first
        that recognizes the file. Each parser sniffs its own format and raises
        ParseError otherwise, so exactly one succeeds. Returns None if none match."""
        for parser_cls in parser_classes:
            try:
                previews = parser_cls().parse(path)
            except ParseError:
                continue                 # not this format; try the next
            if previews:
                return previews
        return None

    def _try_generic_dls(self, path: str):
        """Generic two-column DLS fallback. Returns the previews (arrays converted),
        [] if the file is not a plain numeric table, or None if the user canceled
        the units / data-form prompt."""
        try:
            previews = GenericDLSParser().parse(path)
        except (ParseError, FileNotFoundError):
            return []
        if not previews:
            return []
        form = self._ask_generic_dls_form()
        if form is None:
            return None                      # canceled
        unit, data_form, baseline_B, beta = form
        for p in previews:
            p.delay_time_unit, p.data_form = unit, data_form
            p.baseline_B, p.beta = baseline_B, beta
            try:
                p.apply_data_conversion()
            except ParseError as exc:
                QtWidgets.QMessageBox.warning(self, 'Could not read generic file', str(exc))
                return None
        return previews

    def _ask_generic_dls_form(self):
        """Prompt for a generic DLS file's delay-time unit and data form (and beta /
        baseline if needed). Returns (unit, data_form, baseline_B, beta) or None."""
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle('Generic DLS file')
        form = QtWidgets.QFormLayout(dlg)
        info = QtWidgets.QLabel(
            'Reading this file as a plain two-column DLS correlogram. Set its '
            'delay-time unit and how column 2 is expressed.')
        info.setWordWrap(True)
        form.addRow(info)
        unit = QtWidgets.QComboBox()
        unit.addItems(_DELAY_UNITS)
        unit.setCurrentText('us')
        form.addRow('Delay-time unit', unit)
        dform = QtWidgets.QComboBox()
        dform.addItems([label for label, _ in _DATA_FORMS])
        form.addRow('Column 2 is', dform)
        beta = QtWidgets.QLineEdit('0.8')
        form.addRow('β (only if g1)', beta)
        base = QtWidgets.QLineEdit('')
        base.setPlaceholderText('only if g2')
        form.addRow('Baseline B (only if g2)', base)
        box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(dlg.accept)
        box.rejected.connect(dlg.reject)
        form.addRow(box)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        data_form = dict(_DATA_FORMS)[dform.currentText()]
        baseline_B = beta_v = None
        if data_form == 'g2':
            try:
                baseline_B = float(base.text())
            except ValueError:
                baseline_B = None            # apply_data_conversion will report it
        if data_form == 'g1':
            try:
                beta_v = float(beta.text())
            except ValueError:
                beta_v = 0.8
        return unit.currentText(), data_form, baseline_B, beta_v

    @QtCore.Slot()
    def _on_load_dls(self) -> None:
        if self._busy_guard():
            return
        # Multiple files may be selected at once; each is auto-detected independently.
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, 'Open DLS correlogram(s)', '',
            'Correlogram files (*.csv *.txt *.ASC *.asc *.dat);;All files (*.*)')
        if not paths:
            return
        first_id, n_files, n_meas, unreadable = None, 0, 0, []
        for path in paths:
            # Instrument-agnostic: auto-detect (ALV, Zetasizer, Brookhaven…), then
            # fall back to a plain two-column generic file (with a units prompt).
            previews = self._autodetect(path, _DLS_PARSERS)
            if not previews:
                previews = self._try_generic_dls(path)
            if previews is None:             # canceled the generic prompt -> skip
                continue
            if not previews:
                unreadable.append(os.path.basename(path))
                continue
            for preview in previews:
                raw = {
                    'delay_times_s': [float(x) for x in preview.delay_times_s],
                    'correlogram': [float(x) for x in preview.correlogram],
                }
                params = {k: getattr(preview, k, None) for k in _DLS_PARAM_KEYS}
                iid = self.controller.add_loaded('dls', raw, params, source_path=path)
                first_id = first_id or iid
            n_files += 1
            n_meas += len(previews)
        self._finish_load(first_id, n_files, n_meas, unreadable, 'DLS correlogram')

    @QtCore.Slot()
    def _on_load_sls(self) -> None:
        if self._busy_guard():
            return
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, 'Open SLS intensities', '',
            'SLS intensity files (*.csv *.txt *.dat);;All files (*.*)')
        if not paths:
            return
        first_id, n_files, n_meas, unreadable = None, 0, 0, []
        parse_notes = []                     # passive load-time ⓘ notices from parsers
        for path in paths:
            previews = self._autodetect(path, _SLS_PARSERS)
            if not previews:                 # fall back to a plain two-column table
                try:
                    previews = GenericSLSParser().parse(path)
                except (ParseError, FileNotFoundError):
                    previews = None
            if not previews:
                unreadable.append(os.path.basename(path))
                continue
            for p in previews:
                raw = {'angles_deg': [float(a) for a in p.angles_deg],
                       'intensities': [float(x) for x in p.intensities]}
                params = {k: getattr(p, k, None) for k in _SLS_PARAM_KEYS}
                iid = self.controller.add_loaded('sls', raw, params, source_path=path)
                first_id = first_id or iid
                for note in getattr(p, 'notes', ()):     # e.g. negative intensities
                    parse_notes.append(f'{os.path.basename(path)}: {note}')
            n_files += 1
            n_meas += len(previews)
        self._finish_load(first_id, n_files, n_meas, unreadable,
                          'SLS intensity export', warn_blanks=True)
        if parse_notes:
            QtWidgets.QMessageBox.information(
                self, 'Loaded with notes',
                'The data was loaded and is used as-entered. Notes:\n\n  '
                + '\n  '.join(parse_notes))

    def _finish_load(self, first_id, n_files: int, n_meas: int,
                     unreadable: list, kind_desc: str,
                     warn_blanks: bool = False) -> None:
        """Shared post-load: select the first new measurement, refresh, report.

        `warn_blanks` (SLS loads only) surfaces a multiple-solvent-blank collision:
        a DLS load can't create one, so it would only re-nag about a pre-existing
        collision."""
        if first_id is not None:
            self._set_current(first_id)
            self._refresh_sidebar()
            self._show_module(self.data_module)             # Data tab: confirm params first
            self.statusBar().showMessage(
                f'Loaded {n_meas} measurement(s) from {n_files} file(s). Confirm '
                'parameters in the Data tab and press Update; identity/optics set on '
                'one measurement apply to the whole sample.')
            if warn_blanks:
                self._warn_solvent_reference_collisions()
        if unreadable:
            files = '\n  '.join(unreadable)
            QtWidgets.QMessageBox.warning(
                self, 'Some files not loaded',
                f'Could not read {len(unreadable)} file(s) as a {kind_desc} in any '
                f'supported format:\n  {files}\n\nSee the user guide for the '
                'supported formats.')

    def _warn_solvent_reference_collisions(self) -> None:
        """If any sample loaded more than one c = 0 solvent blank, tell the user which
        blank is the active reference and that the extras are kept but unused (A5 --
        the data model has one reference slot; extras are never silently dropped)."""
        collisions = self.controller.workspace.solvent_reference_collisions()
        if not collisions:
            return
        n_extra = sum(len(v) for v in collisions.values())
        QtWidgets.QMessageBox.information(
            self, 'Multiple solvent blanks',
            f'{n_extra} extra solvent-blank (c = 0) series in '
            f'{len(collisions)} sample(s) were loaded beyond the one reference each '
            'sample uses. The FIRST-loaded blank is the active reference; the extras '
            'are kept (marked "extra blank (unused)" in the sidebar) but do not enter '
            'analysis. Remove or re-assign a blank if a different one should be the '
            'reference.')

    # ------------------------------------------------------------- sidebar ---
    def _refresh_sidebar(self) -> None:
        """Rebuild the samples/measurements tree from the workspace."""
        # Block signals during the rebuild so clear()/add don't fire
        # itemSelectionChanged and re-enter _set_current mid-rebuild.
        self.tree.blockSignals(True)
        self.tree.clear()
        self._leaf_items = {}
        for s in self.controller.samples():
            parent = QtWidgets.QTreeWidgetItem([self._sample_label(s)])
            parent.setToolTip(0, self._sample_label(s))   # full name if clipped
            parent.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
            # A role marker (a tuple) distinguishes structural nodes from measurement
            # leaves (whose role is the bare item_id string). This drives the
            # header-aware context menu and "click a header → select its children".
            parent.setData(0, QtCore.Qt.ItemDataRole.UserRole,
                           ('sample', s.sample_id))
            # Status annotation: an unconfirmed sample (identity not yet set in the Data
            # tab) reads amber, so it's obvious which samples still need confirming.
            if self._sample_label(s) == '(unconfirmed sample)':
                parent.setForeground(0, theme_color(self.tree, 'pending'))
            self.tree.addTopLevelItem(parent)
            parent.setExpanded(True)
            # Group the two measurement kinds under labeled, non-selectable headers
            # so DLS and SLS are visually delineated within a sample.
            if s.dls_item_ids:
                dls_group = self._group_header(parent, 'DLS',
                                               ('group', 'dls', s.sample_id))
                for iid in s.dls_item_ids:
                    self._leaf_items[iid] = self._add_meas_item(dls_group, iid)
            sls = ([(s.solvent_reference_item_id, 'solvent ref')]
                   if s.solvent_reference_item_id else [])
            sls += [(iid, '') for iid in s.sls_item_ids]
            # Extra c = 0 blanks beyond the single reference slot: show them so they
            # stay visible (never silently dropped -- A5), marked as unused.
            sls += [(iid, 'extra blank (unused)')
                    for iid in s.extra_solvent_reference_item_ids]
            if sls:
                sls_group = self._group_header(parent, 'SLS',
                                               ('group', 'sls', s.sample_id))
                for iid, marker in sls:
                    self._leaf_items[iid] = self._add_meas_item(sls_group, iid, marker)
        self._add_trace_nodes()
        self.tree.blockSignals(False)

        if self.current_item in self._leaf_items:
            self._leaf_items[self.current_item].setSelected(True)
        self._apply_selection_mirror()   # tint rows selected in the active tab
        self._refresh_empty_state()      # CTA vs content on the data tabs

    # ------------------------------------------------ selection mirror ---
    def _active_module(self) -> Optional[QtWidgets.QWidget]:
        """The module behind the current tab (tabs are reorderable, so resolve by
        widget, not index)."""
        return self._module_by_wrapper.get(self.tabs.currentWidget())

    def _leaf_base_color(self, item_id: str):
        """A measurement leaf's NON-selected foreground: the derived-average green for a
        replicate average (as `_add_meas_item` sets), otherwise the default palette."""
        lm = self.controller.workspace.measurements.get(item_id)
        is_avg = getattr(lm, 'derived_kind', None) == 'replicate_average'
        return theme_color(self.tree, 'marker_active') if is_avg else QtGui.QBrush()

    @QtCore.Slot()
    def _on_selection_changed(self) -> None:
        """A sample-scoped tab changed its own selection → repaint the mirror."""
        self._apply_selection_mirror()

    def _apply_selection_mirror(self) -> None:
        """Read-only reflect the ACTIVE tab's selected measurements onto the sidebar:
        selected leaves read `marker_selected` + bold, the rest revert to their base.
        A light per-row repaint (no tree rebuild) so expansion/scroll/selection survive.
        Cross-Sample reports its focused sample's measurements here (a read-only
        mirror); tabs with no selection concept (Data/Settings) simply clear the marks."""
        module = self._active_module()
        getter = getattr(module, 'selected_item_ids', None)
        selected = set(getter()) if callable(getter) else set()
        for iid, item in self._leaf_items.items():
            on = iid in selected
            item.setForeground(0, theme_color(self.tree, 'marker_selected')
                               if on else self._leaf_base_color(iid))
            font = item.font(0)
            font.setBold(on)
            item.setFont(0, font)

    @staticmethod
    def _sample_label(sample) -> str:
        poly, solv, temp = sample.polymer_name, sample.solvent_name, sample.temperature_K
        if poly and solv and temp is not None and not math.isnan(temp):
            return f'{poly} / {solv} @ {temp:g} K'
        return '(unconfirmed sample)'

    def _group_header(self, parent, text: str,
                      role) -> QtWidgets.QTreeWidgetItem:
        """A bold, non-selectable header row that groups a sample's measurements by
        kind (DLS / SLS) in the navigator. `role` is a tuple marker
        used by the context menu and click-to-select-children handling."""
        node = QtWidgets.QTreeWidgetItem([text])
        node.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)   # header, not selectable
        node.setData(0, QtCore.Qt.ItemDataRole.UserRole, role)
        font = node.font(0)
        font.setBold(True)
        node.setFont(0, font)
        node.setForeground(0, theme_color(self.tree, 'marker_group'))
        parent.addChild(node)
        node.setExpanded(True)
        return node

    def _add_meas_item(self, parent, item_id: str,
                       marker: str = '') -> QtWidgets.QTreeWidgetItem:
        """A measurement leaf under a DLS/SLS group. The kind lives on the group
        header, so the leaf shows the within-kind axes: angle, Mw fraction, source
        file (and a 'solvent ref' marker for the c = 0 reference)."""
        lm = self.controller.workspace.measurements[item_id]
        is_avg = getattr(lm, 'derived_kind', None) == 'replicate_average'
        # A derived (replicate-averaged) measurement has no source file; show a
        # synthetic name instead of the raw item id.
        label_override = lm.committed_params.get('label') or ''
        if label_override:
            base = label_override
        elif lm.source_path:
            base = os.path.basename(lm.source_path)
        elif is_avg:
            base = f'averaged correlogram ({item_id})'
        else:
            base = item_id
        angle = lm.working_params.get('angle_deg')
        frac = lm.working_params.get('mw_fraction')
        parts = []
        if marker:
            parts.append(marker)
        if is_avg:
            parts.append(f'avg ×{len(lm.derived_from or [])}')
        if angle:
            parts.append(f'{angle:g}°')
        if frac:
            parts.append(f'[{frac}]')           # delineate Mw fractions
        prefix = (' '.join(parts) + ' — ') if parts else ''
        text = f'{prefix}{base}'
        item = QtWidgets.QTreeWidgetItem([text])
        item.setToolTip(0, text)                       # full name if clipped
        item.setData(0, QtCore.Qt.ItemDataRole.UserRole, item_id)
        if is_avg:                                     # muted, to read as derived
            item.setForeground(0, theme_color(self.tree, 'marker_active'))
        parent.addChild(item)
        return item

    def _add_trace_nodes(self) -> None:
        """Add a flat top-level 'Traces' node listing the workspace's count-rate
        traces. Traces have no sample identity yet, so they
        live in one group rather than under a sample."""
        traces = self.controller.traces()
        if not traces:
            return
        node = QtWidgets.QTreeWidgetItem(['Traces'])
        node.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)      # header, not selectable
        node.setData(0, QtCore.Qt.ItemDataRole.UserRole, ('traces',))
        font = node.font(0)
        font.setBold(True)
        node.setFont(0, font)
        node.setForeground(0, theme_color(self.tree, 'marker_group'))
        self.tree.addTopLevelItem(node)
        node.setExpanded(True)
        for t in traces:
            label = t.sample_label or t.trace_id
            leaf = QtWidgets.QTreeWidgetItem([f'{label}  [{t.trace_id}]'])
            leaf.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)  # focused via itemClicked
            leaf.setData(0, QtCore.Qt.ItemDataRole.UserRole, ('trace', t.trace_id))
            leaf.setToolTip(0, leaf.text(0))
            node.addChild(leaf)

    def _descendant_trace_ids(self, item) -> list:
        """Every trace id at or below `item` (role ('trace', tid))."""
        out = []
        role = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if isinstance(role, tuple) and role[0] == 'trace':
            out.append(role[1])
        for i in range(item.childCount()):
            out += self._descendant_trace_ids(item.child(i))
        return out

    def load_trace_files(self) -> None:
        """Load count-rate trace(s) via the Utilities module's shared loader (which
        refreshes its list and emits workspaceChanged → the sidebar rebuilds)."""
        self.utilities_module.load_traces_dialog()

    def _remove_traces(self, tids: list) -> None:
        n = len(tids)
        resp = QtWidgets.QMessageBox.question(
            self, 'Remove trace',
            f'Remove {n} trace{"s" if n > 1 else ""} from the workspace? '
            'The source files on disk are not touched.')
        if resp != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.utilities_module.remove_traces(tids)   # emits workspaceChanged → refresh

    @QtCore.Slot()
    def _on_tree_selection(self) -> None:
        # The active measurement follows the CURRENT (focused) item so that
        # multi-selecting for removal doesn't scramble which one the tabs show.
        item = self.tree.currentItem()
        if item is None:
            return
        role = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if not isinstance(role, str):     # a header / sample / traces node
            return
        self._set_current(role)
        # Tell the Data tab the full highlighted set so an edit + commit can be
        # applied to every highlighted measurement at once.
        self.data_module.set_selected_ids(
            self._measurement_ids(self.tree.selectedItems()))

    @QtCore.Slot('QTreeWidgetItem*', int)
    def _on_tree_clicked(self, item, _col) -> None:
        """Left-click on a structural header (sample / DLS / SLS) selects all the
        measurements beneath it; a Traces node / trace leaf
        opens the Traces tab. Headers are not selectable themselves, so this
        rides the itemClicked signal rather than the selection-changed one."""
        role = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if not isinstance(role, tuple):
            return
        kind = role[0]
        if kind in ('sample', 'group'):
            self._select_descendants(item)
            # Clicking a sample / DLS / SLS header makes that sample the active sample
            # — the tree is now a real sample navigator for every tab. The
            # sample id is role[1] for 'sample', role[2] for a ('group', kind, sid) node.
            # select_leaf=False: _select_descendants already highlighted the whole sample
            # for a bulk Data edit; don't collapse that to the single representative leaf.
            # prefer_kind routes a DLS/SLS sub-header click to that kind's record.
            self._focus_sample(role[-1], select_leaf=False,
                               prefer_kind=(role[1] if kind == 'group' else None))
        elif kind == 'trace':
            self._show_module(self.utilities_module)
            self.utilities_module.focus_trace(role[1])
        elif kind == 'traces':
            self._show_module(self.utilities_module)

    def _select_descendants(self, header) -> None:
        """Select every measurement leaf under `header` and hand the set to the Data
        tab so a bulk edit + commit applies to all of them."""
        leaves = self._descendant_measurement_items(header)
        self.tree.blockSignals(True)
        self.tree.clearSelection()
        for it in leaves:
            it.setSelected(True)
        self.tree.blockSignals(False)
        self.data_module.set_selected_ids(
            [it.data(0, QtCore.Qt.ItemDataRole.UserRole) for it in leaves])

    def _descendant_measurement_items(self, item) -> list:
        """Every measurement-leaf QTreeWidgetItem at or below `item` (role is a str)."""
        out = []
        if isinstance(item.data(0, QtCore.Qt.ItemDataRole.UserRole), str):
            out.append(item)
        for i in range(item.childCount()):
            out += self._descendant_measurement_items(item.child(i))
        return out

    def _descendant_measurement_ids(self, item) -> list:
        return [it.data(0, QtCore.Qt.ItemDataRole.UserRole)
                for it in self._descendant_measurement_items(item)]

    @staticmethod
    def _measurement_ids(items) -> list:
        """The item_ids of any measurement items in `items` (skips headers and trace
        nodes, whose role is a tuple, not the bare item_id string)."""
        out = []
        for it in items:
            iid = it.data(0, QtCore.Qt.ItemDataRole.UserRole)
            if isinstance(iid, str):
                out.append(iid)
        return out

    @QtCore.Slot('QPoint')
    def _on_tree_context_menu(self, pos: QtCore.QPoint) -> None:
        clicked = self.tree.itemAt(pos)
        if clicked is None:
            return
        # A structural node (sample / DLS / SLS / Traces header) gets its own
        # remove-all / load menu; measurement leaves
        # keep the existing grouping/averaging/remove menu.
        role = clicked.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if isinstance(role, tuple):
            self._header_context_menu(clicked, role, pos)
            return
        selected = self.tree.selectedItems()
        # Act on the selection, unless the right-click landed outside it -- then
        # act on just the clicked item (standard file-manager behavior).
        if clicked not in selected:
            items = [clicked]
        else:
            items = selected
        ids = self._measurement_ids(items)
        if not ids:
            return
        n = len(ids)
        menu = QtWidgets.QMenu(self)

        # --- manual grouping: move into a new / existing sample, or back to auto ---
        act_new_sample = menu.addAction('Move to new sample')
        others = self._other_samples(ids)
        move_menu = menu.addMenu('Move to existing sample')
        move_menu.setEnabled(bool(others))
        sample_acts: Dict[object, str] = {}
        for s in others:
            sample_acts[move_menu.addAction(self._sample_label(s))] = s.sample_id
        act_auto = None
        if any(iid in self.controller.workspace.overrides for iid in ids):
            act_auto = menu.addAction('Return to auto-grouping')

        # --- DLS replicate averaging (needs >= 2 DLS measurements) ---
        act_avg_corr = act_avg_res = None
        if n >= 2 and self._all_dls(ids):
            menu.addSeparator()
            act_avg_corr = menu.addAction(
                f'Average correlation functions → new measurement ({n})')
            act_avg_res = menu.addAction(f'Average derived results… ({n})')

        menu.addSeparator()
        touched_sids = {self.controller.sample_id_of(i) for i in ids}
        touched_sids.discard(None)
        total_in_samples = sum(
            len(s.dls_item_ids) + len(s.sls_item_ids)
            + (1 if s.solvent_reference_item_id else 0)
            + len(s.extra_solvent_reference_item_ids)
            for sid in touched_sids
            for s in [self.controller.workspace.samples.get(sid)] if s
        )
        ns = len(touched_sids)
        act_remove_sample = menu.addAction(
            f'Remove sample{"s" if ns > 1 else ""}… ({total_in_samples} measurements)')
        act_remove = menu.addAction(
            f'Remove measurement{"s" if n > 1 else ""} ({n})')

        chosen = menu.exec(self.tree.viewport().mapToGlobal(pos))
        if chosen is None:
            return
        if chosen is act_new_sample:
            self._move_to_new_sample(ids)
        elif chosen in sample_acts:
            self._move_to_sample(ids, sample_acts[chosen])
        elif act_auto is not None and chosen is act_auto:
            self._return_to_auto(ids)
        elif act_avg_corr is not None and chosen is act_avg_corr:
            self._average_corr(ids)
        elif act_avg_res is not None and chosen is act_avg_res:
            self._average_results(ids)
        elif chosen is act_remove_sample:
            self._remove_sample_for(ids)
        elif chosen is act_remove:
            self._remove_measurements(ids)

    def _header_context_menu(self, item, role, pos: QtCore.QPoint) -> None:
        """Right-click menu for a structural node. The sample header removes the
        whole sample; a DLS/SLS header removes only that kind; the Traces node loads
        or clears traces."""
        menu = QtWidgets.QMenu(self)
        kind = role[0]
        if kind == 'traces':
            act_load = menu.addAction('Load trace…')
            tids = self._descendant_trace_ids(item)
            act_clear = menu.addAction(f'Remove all traces ({len(tids)})')
            act_clear.setEnabled(bool(tids))
            chosen = menu.exec(self.tree.viewport().mapToGlobal(pos))
            if chosen is act_load:
                self.load_trace_files()
            elif chosen is act_clear and tids:
                self._remove_traces(tids)
            return
        if kind == 'trace':
            act = menu.addAction('Remove trace')
            chosen = menu.exec(self.tree.viewport().mapToGlobal(pos))
            if chosen is act:
                self._remove_traces([role[1]])
            return
        ids = self._descendant_measurement_ids(item)
        if kind == 'sample':
            act = menu.addAction(
                f'Remove sample… ({len(ids)} measurements)')
            chosen = menu.exec(self.tree.viewport().mapToGlobal(pos))
            if chosen is act and ids:
                self._remove_sample_for(ids)        # removes the whole sample
        elif kind == 'group':
            label = 'DLS' if role[1] == 'dls' else 'SLS'
            act = menu.addAction(f'Remove all {label} measurements ({len(ids)})')
            chosen = menu.exec(self.tree.viewport().mapToGlobal(pos))
            if chosen is act and ids:
                self._remove_measurements(ids)      # only this kind, not the sample

    def _other_samples(self, ids: list) -> list:
        """Samples the selection could be moved into: every sample except the one
        the selection already sits entirely within (moving there would be a no-op)."""
        current = {self.controller.sample_id_of(i) for i in ids}
        return [s for s in self.controller.samples()
                if current != {s.sample_id}]

    def _all_dls(self, ids: list) -> bool:
        ms = self.controller.workspace.measurements
        return all(ms[i].kind == 'dls' for i in ids if i in ms)

    def _regroup_refresh(self) -> None:
        """Common tail after a grouping change: redraw sidebar, re-point tabs."""
        self._refresh_sidebar()
        self._set_current(self.current_item)
        self.cross_module.refresh()

    def _move_to_new_sample(self, ids: list) -> None:
        if self._busy_guard():
            return
        sid = self.controller.new_sample_id()
        for iid in ids:
            self.controller.assign_to_sample(iid, sid)
        self._regroup_refresh()

    def _move_to_sample(self, ids: list, sample_id: str) -> None:
        if self._busy_guard():
            return
        for iid in ids:
            self.controller.assign_to_sample(iid, sample_id)
        self._regroup_refresh()

    def _return_to_auto(self, ids: list) -> None:
        if self._busy_guard():
            return
        for iid in ids:
            self.controller.clear_override(iid)
        self._regroup_refresh()

    def _average_corr(self, ids: list) -> None:
        """Create a new averaged-correlogram measurement from the selection.
        This runs SYNCHRONOUSLY on the main thread: it only averages arrays (fast)
        and then mutates workspace.measurements + regroups — structural changes the
        main thread iterates for the sidebar, so they must not happen on the worker.
        The busy guard keeps it from overlapping a background fit."""
        if self._busy_guard():
            return
        try:
            new_id = self.controller.average_dls_correlograms(ids)
        except ValueError as e:
            QtWidgets.QMessageBox.warning(
                self, 'Cannot average correlograms', str(e))
            return
        self.current_item = new_id
        self._refresh_sidebar()
        self._set_current(new_id)
        self.cross_module.refresh()

    def _average_results(self, ids: list) -> None:
        """Fit each selected replicate, report mean +/- SD/sqrt(N), write Rh to
        the sample. The method prompt stays before dispatch (dialogs are
        main-thread); the N replicate fits run on the worker."""
        method = ask_average_method(self)
        if method is None:
            return
        controller = self.controller

        def done(summary: dict) -> None:
            show_average_summary(self, summary)
            self._refresh_sidebar()
            self.cross_module.refresh()

        def fail(exc: BaseException) -> None:
            if not isinstance(exc, ValueError):
                raise exc
            QtWidgets.QMessageBox.warning(
                self, 'Cannot average derived results', str(exc))

        if not runner().try_submit(
                lambda: controller.average_dls_results(ids, method),
                done, fail, description='replicate averaging'):
            self._busy_guard()

    @QtCore.Slot()
    def _on_delete_key(self) -> None:
        ids = self._measurement_ids(self.tree.selectedItems())
        if ids:
            self._remove_measurements(ids)

    def _remove_measurements(self, item_ids: list) -> None:
        if self._busy_guard():
            return
        n = len(item_ids)
        resp = QtWidgets.QMessageBox.question(
            self, 'Remove from workspace',
            f'Remove {n} measurement{"s" if n > 1 else ""} from the workspace? '
            'Their analysis results are cleared too. The source files on disk are '
            'not touched.')
        if resp != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.controller.remove_measurements(item_ids)
        remaining = list(self.controller.workspace.measurements.keys())
        if self.current_item not in remaining:
            self.current_item = remaining[0] if remaining else None
        self._refresh_sidebar()
        self._set_current(self.current_item)
        self.cross_module.refresh()

    def _remove_sample_for(self, ids: list) -> None:
        """Remove every measurement in the sample(s) that contain `ids`."""
        if self._busy_guard():
            return
        ws = self.controller.workspace
        sample_ids = {self.controller.sample_id_of(i) for i in ids}
        sample_ids.discard(None)
        all_meas: list = []
        labels: list = []
        for sid in sample_ids:
            s = ws.samples.get(sid)
            if s is None:
                continue
            labels.append(self._sample_label(s))
            all_meas += list(s.dls_item_ids) + list(s.sls_item_ids)
            if s.solvent_reference_item_id:
                all_meas.append(s.solvent_reference_item_id)
            all_meas += list(s.extra_solvent_reference_item_ids)
        if not all_meas:
            return
        label_str = ' + '.join(labels) if labels else 'sample'
        n = len(all_meas)
        resp = QtWidgets.QMessageBox.question(
            self, 'Remove sample',
            f'Remove "{label_str}" and all {n} of its measurements from the '
            'workspace? Analysis results are cleared too. Source files on disk '
            'are not touched.')
        if resp != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.controller.remove_measurements(all_meas)
        remaining = list(self.controller.workspace.measurements.keys())
        if self.current_item not in remaining:
            self.current_item = remaining[0] if remaining else None
        self._refresh_sidebar()
        self._set_current(self.current_item)
        self.cross_module.refresh()

    def _set_current(self, item_id: Optional[str]) -> None:
        """Focus a measurement (from the sidebar). Every sample-scoped tab follows the
        resulting active sample: Data/DLS to the measurement, SLS/Utilities/
        Cross to its sample — with a named empty-state where the sample has no data for
        that tab."""
        self.current_item = item_id
        self.active_sample_id = (self.controller.sample_id_of(item_id)
                                 if item_id is not None else None)
        self.data_module.set_measurement(item_id)
        self.dls_module.set_measurement(item_id)
        self.sls_module.set_measurement(item_id)
        self.utilities_module.set_measurement(item_id)
        self.cross_module.set_focused_sample(self.active_sample_id)

    def _representative_item(self, sid: Optional[str],
                             prefer_kind: Optional[str] = None) -> Optional[str]:
        """A measurement to focus for sample `sid` so the measurement-granular tabs
        (Data/DLS) have something to show. Defaults to DLS-first, but `prefer_kind`
        ('dls'/'sls') puts that kind first — so clicking a sample's SLS sub-header lands
        on its SLS record, not a DLS one (focused-review)."""
        if sid is None:
            return None
        s = self.controller.workspace.samples.get(sid)
        if s is None:
            return None
        dls = list(s.dls_item_ids)
        sls = list(s.sls_item_ids)
        if s.solvent_reference_item_id:
            sls.append(s.solvent_reference_item_id)
        order = (sls + dls) if prefer_kind == 'sls' else (dls + sls)
        return order[0] if order else None

    @QtCore.Slot(str)
    def _focus_sample(self, sid: str, *, select_leaf: bool = True,
                      prefer_kind: Optional[str] = None) -> None:
        """Make `sid` the single active sample and fan it out to every tab + the sidebar.
        The entry point for a sample/group header click and for any tab dropdown's
        `sampleFocusRequested` — one path, so the tree and every combo stay in lock-step.
        Guarded against re-entrancy (a fan-out that nudges a combo must not loop back).

        `select_leaf` (default True, the combo path) lights the representative leaf in the
        tree; a header click passes False because `_select_descendants` has already
        highlighted the whole sample for a bulk Data edit — don't collapse that to one.
        `prefer_kind` ('dls'/'sls') picks which kind the representative measurement is
        drawn from (a DLS/SLS sub-header click lands on that kind's record)."""
        if self._syncing_sample:
            return
        sid = sid or None
        self._syncing_sample = True
        try:
            rep = self._representative_item(sid, prefer_kind)
            self.current_item = rep
            self.active_sample_id = sid
            # Measurement-granular tabs follow the representative measurement.
            self.data_module.set_measurement(rep)
            self.dls_module.set_measurement(rep)
            # Sample-scoped tabs follow the sample (named empty-state if incompatible);
            # set_current_sample_id inside these does NOT emit, so no signal loop.
            self.sls_module.show_active_sample(sid)
            self.utilities_module.show_active_sample(sid)
            self.cross_module.set_focused_sample(sid)
            if select_leaf:
                self._select_leaf(rep)           # light the tree without a full rebuild
                self.data_module.set_selected_ids([rep] if rep else [])
            elif rep is not None:
                # Header click: _select_descendants already highlighted the whole sample;
                # keep that multi-selection but move the keyboard-nav "current" item onto
                # the representative leaf (NoUpdate = don't disturb the selection) so
                # `tree.currentItem()` doesn't lag `current_item` (focused-review).
                leaf = self._leaf_items.get(rep)
                if leaf is not None:
                    self.tree.blockSignals(True)
                    self.tree.setCurrentItem(
                        leaf, 0,
                        QtCore.QItemSelectionModel.SelectionFlag.NoUpdate)
                    self.tree.blockSignals(False)
            self._refresh_empty_state()
            self._apply_selection_mirror()
        finally:
            self._syncing_sample = False

    def _select_leaf(self, item_id: Optional[str]) -> None:
        """Select `item_id`'s leaf in the tree (and make it current) WITHOUT a full
        rebuild or firing `_on_tree_selection` — used when a combo/header drives the
        active sample."""
        self.tree.blockSignals(True)
        self.tree.clearSelection()
        leaf = self._leaf_items.get(item_id) if item_id is not None else None
        if leaf is not None:
            leaf.setSelected(True)
            self.tree.setCurrentItem(leaf)
        self.tree.blockSignals(False)

    def _refresh_cross(self) -> None:
        """Rebuild Cross-Sample, then point its focused sample at the active sample so it
        follows the sidebar/tabs."""
        self.cross_module.refresh()
        self.cross_module.set_focused_sample(self.active_sample_id)

    @QtCore.Slot()
    def _on_committed(self) -> None:
        # Committing may have changed a sample's identity -> re-group + redraw.
        self._refresh_sidebar()
        # Re-fan-out the current focus so the analysis tabs repopulate against the
        # freshly committed parameters — without this the Γ-q²/D-c measurement tables
        # stay empty until a sidebar re-click.
        self._set_current(self.current_item)
        self.cross_module.refresh()      # the pairable-sample set may have changed

    @QtCore.Slot()
    def _on_workspace_changed(self) -> None:
        """The synthetic generator injected data -> rebuild the navigator + Data tab
        (and select the first measurement if nothing is current)."""
        self.cross_module.refresh()
        if self.current_item is None:
            for s in self.controller.samples():
                ids = list(s.dls_item_ids) + list(s.sls_item_ids)
                if s.solvent_reference_item_id:
                    ids.append(s.solvent_reference_item_id)
                if ids:
                    self._set_current(ids[0])
                    break
        # One sidebar refresh, after any auto-selection, instead of one before AND
        # one after (the marker is painted from self.current_item either way).
        self._refresh_sidebar()

    # ------------------------------------------------------ sidebar scope ---
    @QtCore.Slot(int)
    def _on_tab_changed(self, index: int) -> None:
        """Adapt the sidebar to the active tab's scope."""
        # Tabs hold scroll-area wrappers and are reorderable, so resolve the active
        # module by the current WIDGET rather than assuming a fixed index order.
        w = self._module_by_wrapper.get(self.tabs.widget(index))
        # The global Settings tab needs no sample: disable the sidebar. (The Solvent
        # Explorer — also global — lives as a Utilities sub-tab; Utilities' sample-
        # scoped sidebar simply stays active while it is shown, harmlessly — the
        # Explorer ignores selection by design.)
        loading = w is not self.settings_module
        self.tree.setEnabled(loading)
        self.load_button.setEnabled(loading)
        self.load_sls_button.setEnabled(loading)
        if w is self.cross_module:
            # Deferred while a run is in flight: refresh() auto-selects labeled
            # Rg/Rh/Mw/A2 picks, which WRITES SampleResult fields the worker may
            # be reading — run it now, or once the current job completes. Then point
            # its focused sample at the active sample.
            run_when_idle(self._refresh_cross)
        if w is self.settings_module:
            self.sidebar_note.setText('Settings are global — no sample needed.')
        elif w is self.cross_module:
            self.sidebar_note.setText(
                'Cross-Sample is aggregate: tick samples to include in the views. '
                'Clicking a sample in the Workspace (or the in-tab dropdown) sets the '
                'focused sample here too.')
        else:
            self.sidebar_note.setText(
                'Click a sample or measurement to load it into every tab.')
        # Re-point the sidebar mirror at whichever tab is now active.
        self._apply_selection_mirror()

    # --------------------------------------------------------------- theme ---
    @staticmethod
    def _build_palette(dark: bool) -> QtGui.QPalette:
        """An EXPLICIT light or dark palette (never inherited from the OS), so the
        chosen theme overrides the system color scheme rather than tracking it."""
        pal = QtGui.QPalette()
        C = QtGui.QColor
        R = QtGui.QPalette.ColorRole
        G = QtGui.QPalette.ColorGroup
        if dark:
            base, alt, win, txt = C(35, 35, 38), C(45, 45, 48), C(53, 53, 53), C(220, 220, 220)
            disabled = C(120, 120, 120)
            mid, midlight, darkc = C(120, 120, 120), C(90, 90, 90), C(25, 25, 28)
        else:
            base, alt, win, txt = C(255, 255, 255), C(233, 233, 233), C(240, 240, 240), C(20, 20, 20)
            disabled = C(160, 160, 160)
            mid, midlight, darkc = C(150, 150, 150), C(200, 200, 200), C(120, 120, 120)
        pal.setColor(R.Window, win)
        pal.setColor(R.WindowText, txt)
        pal.setColor(R.Base, base)
        pal.setColor(R.AlternateBase, alt)
        pal.setColor(R.Text, txt)
        pal.setColor(R.Button, win)
        pal.setColor(R.ButtonText, txt)
        pal.setColor(R.ToolTipBase, base)
        pal.setColor(R.ToolTipText, txt)
        # Mid/Midlight/Dark were previously unset, so Fusion fell back to OS defaults
        # (e.g. a low-contrast mid-gray the "?" badge relied on). Set them per theme.
        pal.setColor(R.Mid, mid)
        pal.setColor(R.Midlight, midlight)
        pal.setColor(R.Dark, darkc)
        pal.setColor(R.Highlight, C(38, 110, 180))
        pal.setColor(R.HighlightedText, C(255, 255, 255))
        for role in (R.WindowText, R.Text, R.ButtonText):
            pal.setColor(G.Disabled, role, disabled)
        return pal

    def _apply_theme(self, theme: str) -> None:
        """Apply the theme globally (Fusion style + palette). 'light'/'dark' use an
        explicit palette that OVERRIDES the OS color scheme; 'system' follows it."""
        app = QtWidgets.QApplication.instance()
        if app is None:
            return
        app.setStyle('Fusion')
        # On Qt 6.8+ this forces the OS color scheme so even style-hint-aware
        # widgets honor the choice (and 'system' restores following the OS).
        hints = app.styleHints()
        if hasattr(hints, 'setColorScheme'):
            cs = QtCore.Qt.ColorScheme
            hints.setColorScheme({'light': cs.Light, 'dark': cs.Dark}.get(
                theme, cs.Unknown))
        if theme == 'dark':
            app.setPalette(self._build_palette(dark=True))
        elif theme == 'light':
            app.setPalette(self._build_palette(dark=False))
        else:  # 'system' — follow the OS / Qt default
            app.setPalette(app.style().standardPalette())
        # Deterministically re-apply token colors to themed labels + "?" badges (they
        # don't reliably auto-refresh from a stylesheet — see gui.theme.retheme).
        retheme(self)
        self._apply_plot_theme()

    def _apply_plot_theme(self) -> None:
        """Theme the ON-SCREEN plots to match the app theme when the user opted in and
        the theme is dark; otherwise keep them white (the default). Export always stays
        white (handled in the plot save path). rcParams drive future redraws; the
        one-shot walk re-themes canvases already on screen."""
        plot_dark = self.controller.settings.plot_match_theme and is_dark(self)
        set_onscreen_plot_theme(plot_dark)
        for canvas in self.findChildren(FigureCanvasQTAgg):
            apply_figure_theme(canvas.figure, plot_dark)
            canvas.draw_idle()

    # UI density -> point-size delta from the platform default.
    _DENSITY_DELTA = {'compact': -1, 'comfortable': 0, 'large': 2}
    # The platform-default base font size, captured ONCE per process (class-level, not
    # per-instance) before any density adjustment — so 'comfortable' == the untouched
    # default and compact/large are relative to it even if a second MainWindow is built.
    _base_font_pt: Optional[int] = None

    def _apply_ui_density(self) -> None:
        """Set the application-wide font size from the ui_density setting, relative to
        the platform default captured once at startup (so 'comfortable' leaves the
        default untouched). Applied app-wide via QApplication.setFont, at startup and on
        every settings Apply."""
        app = QtWidgets.QApplication.instance()
        if app is None:
            return
        font = app.font()
        if MainWindow._base_font_pt is None:
            pt = font.pointSize()
            MainWindow._base_font_pt = pt if pt > 0 else 9   # fallback for a pixel-size font
        delta = self._DENSITY_DELTA.get(self.controller.settings.ui_density, 0)
        font.setPointSize(max(6, MainWindow._base_font_pt + delta))
        app.setFont(font)

    # ---- last-session autosave / reopen ----
    @staticmethod
    def _managed_session_path() -> str:
        """The app-managed autosave file, beside settings.json (moves with the settings
        location — never a hard-coded/assumed drive path)."""
        from app.settings import SettingsState
        return str(SettingsState.default_path().parent / 'last_session.lsjson')

    def _maybe_reopen_session(self) -> None:
        """On startup, restore the last auto-saved session when the setting is on and the
        file loads. Fail-soft: any problem leaves an empty workspace (Batch-4 empty-state)."""
        s = self.controller.settings
        if not s.reopen_last_session:
            return
        path = s.last_session_path or self._managed_session_path()
        if not os.path.exists(path):
            return
        try:
            self.controller.load_session(path)
        except Exception:                          # noqa: BLE001 - corrupt/old file: start empty
            return
        ids = list(self.controller.workspace.measurements.keys())
        self.current_item = ids[0] if ids else None

    def closeEvent(self, event) -> None:
        """Auto-save the workspace on exit when reopen-last-session is on and there is
        data to save (never clobber the stored session with an empty one). Fail-soft —
        an autosave problem must never block closing."""
        try:
            s = self.controller.settings
            # Skip the autosave while a background fit is in flight: the
            # worker may still be writing SampleResult fields, so serializing now could
            # capture a torn snapshot. Fail-soft — the prior saved session simply stands.
            if (s.reopen_last_session and self.controller.workspace.measurements
                    and not runner().is_busy):
                path = self._managed_session_path()
                self.controller.save_session(path)
                s.last_session_path = path
                self.controller.settings.save()
        except Exception:                          # noqa: BLE001 - best effort on exit
            pass
        super().closeEvent(event)

    @QtCore.Slot()
    def _on_settings_applied(self) -> None:
        """Settings changed: re-theme, re-palette, and re-seed the module controls
        (the new defaults; existing results are untouched). The palette applies to
        plots drawn afterwards (the Utilities trace re-renders here)."""
        self._apply_theme(self.controller.settings.theme)
        self._apply_ui_density()                                # font size (5.5)
        set_palette(self.controller.settings.plot_palette)
        set_plot_units(self.controller.settings.plot_units)     # plot-axis units
        set_tooltips_enabled(self.controller.settings.show_tooltips)
        self._refresh_sidebar()      # re-pick tree marker colors for the new theme
        self.dls_module.reseed_from_settings()
        self.utilities_module.reseed_from_settings()
        # NB: the Solvent Explorer (a Utilities sub-tab) is intentionally NOT reseeded
        # here or by Utilities' reseed — default_solvent is a build-time seed ("seed,
        # never override"), so an unrelated Settings Apply must not yank the user's
        # active solvent/condition. It re-themes itself via its own changeEvent.
        self.cross_module.refresh()                             # redraw scaling axes
        self.sls_module.set_measurement(self.current_item)      # redraw SLS axes
        self.dls_module.set_measurement(self.current_item)      # redraw DLS overlays (new palette)
