"""
The DLS analysis region on screen: the delay window and baseline span a run reads
(`_AnalysisRegion`), the fit arguments it yields, and the per-tab checkbox that applies it.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from PySide6 import QtCore, QtWidgets

from app.analysis_region import RegionState, region_summary
from gui.help import HelpBadge
from gui.theme import ThemedLabel


class _AnalysisRegion:
    """The shared DLS analysis region (all in seconds), edited on the Correlogram
    tab (numeric fields + draggable markers): the fit WINDOW [tau_min, tau_max] plus a
    BASELINE span [base_lo, base_hi]. This object holds the handles and whether the
    user has moved each kind; what a fit reads from them is decided in ONE place,
    `app/analysis_region.py::region_fit_kwargs` (through `Controller.
    analysis_region_kwargs`), which the Correlogram and Distribution sub-tabs and every
    checked "use the Correlogram tab's delay window and baseline" path all call. That
    module's docstring states the rule; `state()` is the snapshot handed to it.

    NOTHING IS APPLIED UNTIL THE USER SETS IT (owner-ruled 2026-09-03 for the span,
    2026-09-22 for the window). `window_touched` / `baseline_touched` are False while the
    handles still hold the values `init_from_data` proposed from the ticked data, and an
    untouched part reaches no fit: each run keeps its own full lag range and its fit's
    own baseline estimate. The moment a handle of a kind is dragged or its field edited,
    that part applies, and `Reset window + baseline` clears both flags again. This is NOT
    policing: a region the user set is never refused, second-guessed or overridden
    (invariant 4 clause 2); only a region the user never set stops being applied. The
    measured hazard it closes is a region seeded from one run -- a baseline span, or a
    window on one lag axis -- reaching a run it was never placed on.

    It deliberately does NOT carry the engine inputs Settings seeds (the skipped
    channels, the cumulant method and order). Making them per-run controls here was tried
    and reverted: the Cross-Sample candidate builders re-derive the same fit from
    `controller.settings`, so a panel-local value promoted a DIFFERENT Rh than the
    panel showed (30.0 nm shown, 23.73 nm promoted, measured). The controller's seed
    homes are that one source of truth; the Correlogram tab's order box is the one
    per-run value, and the tier-1 candidate promotes the tab's own fit while it is fresh.
    """

    def __init__(self) -> None:
        self.tau_min_s: Optional[float] = None
        self.tau_max_s: Optional[float] = None
        self.base_lo_s: Optional[float] = None
        self.base_hi_s: Optional[float] = None
        # Has the USER placed each part, as opposed to `init_from_data` proposing it? The
        # Data tab's changed-since-commit highlight is the house pattern: a value the
        # program proposed and a value the user asserted are different states, and only
        # the second may steer a number.
        self.window_touched = False
        self.baseline_touched = False

    def init_from_data(self, tau: np.ndarray) -> None:
        """Propose the defaults from the ticked data: window = full range, baseline =
        last 25 % of the lags. An UNTOUCHED part follows the data on every call, so its
        handles draw the current selection's range; a part the user moved is theirs and
        stays put. Neither proposal reaches a fit (see the class docstring)."""
        if tau.size == 0:
            return
        lo, hi = float(tau.min()), float(tau.max())
        proposal = {'tau_min_s': lo, 'tau_max_s': hi,
                    'base_lo_s': float(np.quantile(tau, 0.75)), 'base_hi_s': hi}
        for attrs, touched in ((self._WINDOW_ATTRS, self.window_touched),
                               (self._BASELINE_ATTRS, self.baseline_touched)):
            for attr in attrs:
                # A part the user placed keeps its values even when typed before any data
                # was ticked (the flag, not the first call, decides); an end still empty
                # takes the proposal.
                if not touched or getattr(self, attr) is None:
                    setattr(self, attr, proposal[attr])

    def force_reseed(self) -> None:
        """Drop the placed values so the next init_from_data re-proposes both parts from
        the data (the Correlogram tab's 'Reset window + baseline'), and forget that the
        user ever moved either -- Reset is the one control that undoes a placement, so it
        is the one that clears the flags."""
        self.window_touched = False
        self.baseline_touched = False

    _WINDOW_ATTRS = ('tau_min_s', 'tau_max_s')
    _BASELINE_ATTRS = ('base_lo_s', 'base_hi_s')

    def set_handle(self, attr: str, value) -> None:
        """Move one handle, marking its kind touched when the handle actually moves.

        The one write path for both editors (drag and numeric field), so the two cannot
        disagree about what counts as a placement. The value comparison matters: the
        numeric-field editor rewrites all four handles on every edit, so a plain
        `setattr` there would mark the baseline touched when the user typed a
        delay-window bound and never went near the purple span -- and the window touched
        when they only moved the span.
        """
        if value != getattr(self, attr):
            if attr in self._WINDOW_ATTRS:
                self.window_touched = True
            elif attr in self._BASELINE_ATTRS:
                self.baseline_touched = True
        setattr(self, attr, value)

    def state(self) -> RegionState:
        """The Qt-free snapshot the one reading helper takes -- read on the main thread
        at dispatch, so a background fit never sees a handle move under it."""
        return RegionState(self.tau_min_s, self.tau_max_s, self.base_lo_s,
                           self.base_hi_s, window_touched=self.window_touched,
                           baseline_touched=self.baseline_touched)


def _region_fit_kwargs(controller, region: _AnalysisRegion, iid: str, method: str) -> Dict:
    """The GUI's thin call into the one reading (`Controller.analysis_region_kwargs`):
    what a `method` fit of measurement `iid` reads from the shared region. Every tab
    that fits under the region builds its Run and its staleness key from this, so a
    region cannot mean one thing on one tab and another elsewhere."""
    return controller.analysis_region_kwargs(iid, method, region.state())


class RegionCheckBox(QtWidgets.QWidget):
    """The opt-in "use the Correlogram tab's delay window and baseline" box -- ONE
    widget, one behaviour, on every surface that re-runs a fit behind the scenes: the
    Γ vs q², D vs c, Mode Map and DDLS sub-tabs and the "Average derived results…"
    prompt.

    - its label line says what the region holds now, or that nothing is set;
    - unticked (the default, held per (sample, fraction) by each tab and never saved)
      the run uses each measurement's own defaults, as the paths did before the box;
    - ticked, the run reads the region through the one reading every tab shares
      (`Controller.analysis_region_kwargs`), and the result's notes say so;
    - its `key()` joins the owning tab's staleness key, so toggling it -- or moving a
      handle while it is ticked -- flags a shown result stale, and leaves an unticked
      one alone.

    Opt-in, never automatic, because the region is ONE per DLS tab, not one per sample:
    a span placed while looking at one sample would otherwise reach another sample's
    runs unasked. A LIVE link to the other tab's region while ticked, unlike the Mode
    Map's one-time copy of the Distribution tab's fields (`ui_style_guide.md` §4 states
    the difference)."""

    toggled = QtCore.Signal(bool)

    LABEL = "Use the Correlogram tab's delay window and baseline"

    def __init__(self, region: _AnalysisRegion, parent=None) -> None:
        super().__init__(parent)
        self._region = region
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self.box = QtWidgets.QCheckBox(self.LABEL)
        self.box.setToolTip(
            'Ticked, each run is fitted as the Correlogram tab fits it: over the delay '
            'window set there, with a baseline measured over the part of the baseline '
            'region inside that window. Unticked, each run keeps its own full lag range '
            'and its fit\'s own baseline estimate; the result\'s notes say which it used.')
        self.box.toggled.connect(self.toggled)
        row.addWidget(self.box)
        row.addWidget(HelpBadge(
            'Fit these runs over the region set on the Correlogram tab.', bullets=[
                'Set the delay window or the baseline region on the '
                '<b>Correlogram</b> sub-tab, then tick this box and press <b>Run</b>.',
                'Only a window or region you have moved is used; the line under the box '
                'says what is set now.',
                'Held per sample and not saved with the session; each new sample starts '
                'unticked.',
                'Ticking it, or moving the region while it is ticked, marks a shown '
                'result out of date — press <b>Run</b> again.',
                'The result’s notes, its export and any Summary row it writes say it '
                'used the Correlogram tab’s region.',
            ]))
        row.addStretch(1)
        lay.addLayout(row)
        self.summary = ThemedLabel('', role='hint', size=10)
        self.summary.setWordWrap(True)
        lay.addWidget(self.summary)
        self.refresh()

    def refresh(self) -> None:
        """Re-read the region into the label line (the owning module calls this when the
        Correlogram tab reports a move)."""
        self.summary.setText(region_summary(self._region.state()))

    def isChecked(self) -> bool:
        return self.box.isChecked()

    def set_checked_silently(self, on: bool) -> None:
        """Show a held per-(sample, fraction) state without firing `toggled`."""
        was = self.box.blockSignals(True)
        try:
            self.box.setChecked(bool(on))
        finally:
            self.box.blockSignals(was)

    # The tabs hold the choice per (sample, fraction) -- the box shows only the one on
    # screen -- so both readers take the held choice rather than the box's own state; a
    # deferred Run of another (sample, fraction) reads its own.
    def region_for_run(self, checked: bool) -> Optional[RegionState]:
        """The snapshot a Run hands the controller: the region when `checked`, else None.
        Read on the main thread at dispatch."""
        return self._region.state() if checked else None

    def key(self, checked: bool) -> tuple:
        """The choice's part of a staleness key: the choice and, when `checked`, the
        region's APPLIED parts (`RegionState.applied`), so re-seeding an untouched handle
        never flags a result stale and every move of a set one does."""
        return (True, self._region.state().applied()) if checked else (False,)
