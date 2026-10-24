"""
The SLS tab (`SLSModule`): the shell holding the Analysis, Derived Intensity and
Mode-Resolved sub-tabs.
"""
from __future__ import annotations

from typing import Optional

from PySide6 import QtCore, QtWidgets

from gui.widgets import roomy_tabs

from gui.sls.analysis_tab import _AnalysisTab
from gui.sls.derived_intensity import _DerivedIntensityTab
from gui.sls.mode_resolved import _ModeResolvedTab


# ===========================================================================
# The SLS tab shell
# ===========================================================================
class SLSModule(QtWidgets.QWidget):
    """The SLS tab: analysis beside the surface that derives static points.

    A thin shell, like :class:`gui.dls_module.DLSModule` -- it owns the sub-tabs
    and forwards the small API the application shell drives them through. All the
    analysis lives in :class:`_AnalysisTab`.
    """

    committed = QtCore.Signal()
    selectionChanged = QtCore.Signal()
    sampleFocusRequested = QtCore.Signal(str)
    workspaceChanged = QtCore.Signal()
    modeMapFocusRequested = QtCore.Signal(str)   # -> DLS tab's Mode Map

    def __init__(self, controller, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = roomy_tabs(QtWidgets.QTabWidget())
        self.analysis = _AnalysisTab(controller)
        self.derived = _DerivedIntensityTab(controller)
        self.mode_resolved = _ModeResolvedTab(controller)
        self.tabs.addTab(self.analysis, 'Analysis')
        self.tabs.addTab(self.derived, 'Derived Intensity')
        self.tabs.addTab(self.mode_resolved, 'Mode-Resolved')
        self.tabs.setMovable(True)
        layout.addWidget(self.tabs, 1)

        self.analysis.committed.connect(self.committed)
        self.analysis.selectionChanged.connect(self.selectionChanged)
        self.analysis.sampleFocusRequested.connect(self.sampleFocusRequested)
        # A derivation creates measurements, so the shell must rebuild the sidebar
        # and the Data table -- and this tab's own candidate list is now stale too
        # (the new SLS point does not appear in it, but a re-derive should re-read
        # the runs it came from).
        self.derived.workspaceChanged.connect(self.workspaceChanged)
        self.derived.workspaceChanged.connect(self._on_derived)
        # Cross-tab navigation from the Mode-Resolved list to where mode sets
        # are DEFINED (never a duplicated view) -- the main window routes it.
        self.mode_resolved.modeMapFocusRequested.connect(
            self.modeMapFocusRequested)

    # -- the API the application shell drives --------------------------------
    @property
    def sample_id(self) -> Optional[str]:
        """The analyzed sample (the analysis sub-tab owns it)."""
        return self.analysis.sample_id

    def set_measurement(self, item_id: Optional[str]) -> None:
        for tab in (self.analysis, self.derived, self.mode_resolved):
            tab.set_measurement(item_id)

    def show_active_sample(self, sid: Optional[str], **kw) -> None:
        self.analysis.show_active_sample(sid, **kw)
        self.derived.show_active_sample(sid)
        self.mode_resolved.show_active_sample(sid)

    def selected_item_ids(self) -> list:
        return self.analysis.selected_item_ids()

    def _on_derived(self) -> None:
        """After a derivation, re-point the analysis sub-tab at the same sample so
        the new measurements are picked up without the user re-selecting."""
        self.analysis.show_active_sample(self.analysis.sample_id)
