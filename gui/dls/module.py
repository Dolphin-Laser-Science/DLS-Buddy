"""
The DLS tab (`DLSModule`): the inner sub-tabs and the protocols the tab calls them through.
"""
from __future__ import annotations

import html
from typing import Dict, List, Optional, Protocol, Tuple

from PySide6 import QtCore, QtWidgets

from plotting.plots import (cycle as _plot_cycle, marker_cycle as _marker_cycle,
                            linestyle_cycle as _linestyle_cycle)
from core.naming import measurement_display_name, measurement_hover_card
from gui.widgets import roomy_tabs, SelectionModel

from gui.dls.region import _AnalysisRegion
from gui.dls.averaging import AverageSources
from gui.dls.correlogram import _CorrelogramTab
from gui.dls.distribution import _DistributionTab
from gui.dls.sample_analysis import _SampleAnalysisTab
from gui.dls.mode_map import _ModeMapTab
from gui.dls.ddls import _DDLSTab
from gui.dls.summary import _SummaryTab


class _DLSSubTab(Protocol):
    """What `DLSModule` calls on every sub-tab (`_subtabs`): `_DeferredRepaint`'s
    halves and its flush (`gui/dls/bases.py`). The sample-level tabs'
    `selected_item_ids` is `_SampleRunTab`'s."""

    defers_repaint: bool
    _repaint_pending: bool

    def set_measurement(self, item_id: Optional[str], runnable: bool) -> None: ...

    def focus_state(self, item_id: Optional[str], runnable: bool) -> bool: ...

    def flush_view(self) -> None: ...


class _ReshownSubTab(_DLSSubTab, Protocol):
    """A sub-tab `DLSModule.reseed_from_settings` re-shows after a DLS clear."""

    def reshow(self) -> None: ...


class _ReseededSubTab(_DLSSubTab, Protocol):
    """A sub-tab with per-run fields `DLSModule.reseed_from_settings` reseeds."""

    def reseed_from_settings(self) -> None: ...


class DLSModule(QtWidgets.QWidget):
    """DLS analysis tab: a header + persistent sub-tabs. The Correlogram and
    Distribution tabs co-plot any ticked measurements (the former Overlay tab is
    folded in), sharing one analysis region and one measurement selection. A
    Summary tab reads the durable snapshot store written by every DLS run."""

    # Fires whenever the measurements this module has SELECTED for analysis change
    # (the shared overlay set, a sample-tab include change, or a sub-tab switch). The
    # shell connects it to repaint the sidebar's selection mirror.
    selectionChanged = QtCore.Signal()

    def __init__(self, controller, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.item_id: Optional[str] = None
        # The DLS clear generation the sub-tabs were last re-shown for
        # (`reseed_from_settings`).
        self._seen_dls_clear = controller.dls_clear_generation
        layout = QtWidgets.QVBoxLayout(self)
        self.header = QtWidgets.QLabel()
        self.header.setWordWrap(True)
        layout.addWidget(self.header)

        self.tabs = roomy_tabs(QtWidgets.QTabWidget())   # roomier tabs so labels don't clip
        # Shared DLS analysis region (fit window + baseline) AND the shared set of
        # measurements ticked for co-plotting — both used by the Correlogram and
        # Distribution tabs so the two stay consistent.
        self.region = _AnalysisRegion()
        # One shared selection model for the co-plotting tabs (Correlogram / Distribution
        # / Summary). The color cycle keeps each measurement a stable overlay color.
        # Pass the live `cycle` FUNCTION (not a snapshot) so a plot-palette switch
        # reaches the DLS overlay colours; markers/linestyles are palette-independent.
        self.selection = SelectionModel(color_cycle=_plot_cycle,
                                        marker_cycle=_marker_cycle(),
                                        linestyle_cycle=_linestyle_cycle())
        # The Distribution tab is built FIRST so the Correlogram tab can be handed its
        # accessor at construction. Only the construction order moves; the sub-tab order
        # on screen is set by `addTab` below and is unchanged.
        self.distribution_tab = _DistributionTab(controller, self.region, self.selection)
        self.correlogram_tab = _CorrelogramTab(
            controller, self.region, self.selection,
            distribution_fits=self.distribution_fits)
        # The four tabs that re-run a fit behind the scenes read the region only through
        # their own "use the Correlogram tab's delay window and baseline" box.
        self.gamma_q2_tab = _SampleAnalysisTab(controller, 'gamma_q2', region=self.region)
        self.mode_map_tab = _ModeMapTab(
            controller, distribution_fit_values=self.distribution_fit_values,
            region=self.region)
        self.conc_tab = _SampleAnalysisTab(controller, 'conc_extrap', region=self.region)
        self.ddls_tab = _DDLSTab(controller, region=self.region)
        self.summary_tab = _SummaryTab(controller, self.selection)
        self._subtabs: List[_DLSSubTab] = [
            self.correlogram_tab, self.distribution_tab, self.gamma_q2_tab,
            self.mode_map_tab, self.conc_tab, self.ddls_tab, self.summary_tab]
        self.tabs.addTab(self.correlogram_tab, 'Correlogram')
        self.tabs.addTab(self.distribution_tab, 'Distribution')
        self.tabs.addTab(self.gamma_q2_tab, 'Γ vs q²')
        self.tabs.addTab(self.mode_map_tab, 'Mode Map')   # generalizes Γ vs q² per mode
        self.tabs.addTab(self.conc_tab, 'D vs c')
        self.tabs.addTab(self.ddls_tab, 'DDLS')
        self.tabs.addTab(self.summary_tab, 'Summary')
        self.tabs.setMovable(True)               # drag to reorder sub-tabs
        layout.addWidget(self.tabs, 1)
        # Re-emit selection changes for the sidebar mirror: the shared overlay model,
        # each sample-scoped sub-tab's own include set, and a sub-tab switch (which
        # changes WHICH selection is active) all bubble up as selectionChanged.
        self.selection.changed.connect(self.selectionChanged)
        # Before the mirror's re-emit: Qt calls slots in connection order, and the mirror
        # reads the ticks the shown sub-tab's owed repaint brings level.
        self.tabs.currentChanged.connect(self._flush_shown_subtab)
        self.tabs.currentChanged.connect(lambda _i: self.selectionChanged.emit())
        # The module owns both co-plotting tabs, so IT carries the news that the
        # Distribution tab's stored fits moved -- neither widget reaches into the other.
        self.distribution_tab.fitsChanged.connect(
            self.correlogram_tab.on_fits_changed)
        for tab in (self.gamma_q2_tab, self.mode_map_tab, self.conc_tab,
                    self.ddls_tab):
            tab.selectionChanged.connect(self.selectionChanged)
            # ... and the news that the region moved, the same routing: each box
            # relabels itself and flags a checked result stale.
            self.correlogram_tab.regionChanged.connect(tab.on_region_changed)
        self.set_measurement(None)

    def distribution_fits(self) -> Dict[Tuple[str, str], Dict]:
        """The Distribution sub-tab's stored fits, keyed `(iid, method)`, each with that
        tab's own staleness verdict.

        The Correlogram tab's "Fits to show" list reads its three distribution entries
        through here, so the two sibling widgets never touch each other's state: the
        module that owns both is the only thing that knows they are siblings.
        """
        return self.distribution_tab.stored_fits()

    def distribution_fit_values(self) -> Dict:
        """The Distribution sub-tab's current fit-field values, for the Mode Map's
        one-time copy -- read through the module that owns both, like
        `distribution_fits`."""
        return self.distribution_tab.fit_fields.values()

    def average_prompt_sources(self) -> 'AverageSources':
        """What the "Average derived results…" prompt can copy or read, handed to it by
        the module that owns the tabs (the `distribution_fits` accessor shape): the
        shared region, the Correlogram tab's order box and the Distribution tab's fields.
        Read at the moment the user confirms, so the copy is of the values shown then."""
        fields = self.distribution_tab.fit_fields

        def distribution_kwargs(method: str) -> dict:
            # The copy reads the fields a CONTIN Run reads, so it refuses what that Run
            # refuses: `alpha_problem` is the one predicate for every consumer of the group.
            problem = fields.alpha_problem() if method == 'contin' else ''
            if problem:
                raise ValueError(problem)
            return fields.fit_kwargs(method)

        return AverageSources(
            region=self.region,
            cumulant_order=lambda: int(self.correlogram_tab.order_spin.value()),
            distribution_kwargs=distribution_kwargs)

    def selected_item_ids(self) -> list:
        """The measurements the ACTIVE sub-tab currently has selected (for the sidebar
        mirror). Co-plot tabs use the shared overlay set; the Γ-q²/D-vs-c tabs use their
        own per-sample include set; DDLS has none yet."""
        w = self.tabs.currentWidget()
        if w in (self.correlogram_tab, self.distribution_tab, self.summary_tab):
            ws = self.controller.workspace.measurements
            return [i for i in self.selection.ids() if i in ws and ws[i].kind == 'dls']
        if w in (self.gamma_q2_tab, self.mode_map_tab, self.conc_tab,
                 self.ddls_tab):
            return w.selected_item_ids()
        return []

    def reseed_from_settings(self) -> None:
        """After a Settings apply: first, when a skip, cumulant or order clear ran since
        the last re-show (`Controller.dls_clear_generation`), re-show each sub-tab that
        draws its fits from the controller, at its own focus, since the clear may have
        taken what it shows (the controller's clear takes each run's superseded copy and
        failure line with it, so setting the value back cannot redraw it as current); then
        reseed the per-run fields. An apply that ran no such clear leaves the sub-tabs as
        they are. The DDLS tab's fit is kept by design
        (`Controller._cumulant_dependent_cache_keys`), and the Mode Map's map is read
        live."""
        gen = self.controller.dls_clear_generation
        # Each sub-tab pays a repaint it owes before its re-show or reseed paints, since a
        # reseed's own field signals fire inside the call.
        if gen != self._seen_dls_clear:
            self._seen_dls_clear = gen
            reshown: Tuple[_ReshownSubTab, ...] = (
                self.correlogram_tab, self.distribution_tab, self.gamma_q2_tab,
                self.conc_tab)
            for tab in reshown:
                tab.flush_view()
                tab.reshow()
        reseeded: Tuple[_ReseededSubTab, ...] = (
            self.correlogram_tab, self.distribution_tab, self.mode_map_tab)
        for tab in reseeded:
            tab.flush_view()
            tab.reseed_from_settings()

    def focus_mode_map(self) -> None:
        """Bring the Mode Map sub-tab to the front (cross-tab navigation from
        the SLS tab's Mode-Resolved list — navigation, never duplication)."""
        self.tabs.setCurrentWidget(self.mode_map_tab)

    def _flush_shown_subtab(self, _index: int) -> None:
        """The sub-tab just made current pays the repaint a click left it owing."""
        tab = self.tabs.currentWidget()
        if tab is not None:
            tab.flush_view()

    def set_measurement(self, item_id: Optional[str]) -> None:
        self.item_id = item_id
        if item_id is None:
            self.header.setText('Pick a measurement in the Workspace list; '
                                'tick measurements in the sub-tabs to analyze.')
            self.header.setToolTip('')        # the card goes with the measurement
            runnable = False
        else:
            kind = self.controller.workspace.measurements[item_id].kind
            runnable = (kind == 'dls')
            # Named, not id'd: the one builder, and its hover card for the rest
            # (the id included, so a bug report still has it).
            lm = self.controller.workspace.measurements[item_id]
            name = measurement_display_name(lm)
            self.header.setToolTip(measurement_hover_card(lm))
            if runnable:
                self.header.setText(
                    f'DLS measurement: <b>{html.escape(name)}</b>')
            else:
                self.header.setText(
                    f'<b>{html.escape(name)}</b> is a {kind.upper()} '
                    'measurement — '
                    'open it in the SLS tab.')
        # The current sub-tab, and each that has not opted in, repaints now; a hidden
        # one that has keeps the focus and owes the repaint until it is shown
        # (`_DeferredRepaint`). In `_subtabs` order: the Correlogram's reload seeds the
        # shared region the tabs after it read.
        current = self.tabs.currentWidget()
        for tab in self._subtabs:
            if tab is current or not tab.defers_repaint:
                tab.set_measurement(item_id, runnable)
            elif tab.focus_state(item_id, runnable):
                tab._repaint_pending = True
