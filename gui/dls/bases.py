"""
The bases DLS sub-tabs build on: the repaint a hidden sub-tab may defer until it is shown
(`_DeferredRepaint`), the draggable residual panel's share of the plot
(`_ResidualRatioMixin`) and the per-sample run tab the Γ vs q², D vs c, Mode Map and DDLS
sub-tabs share (`_SampleRunTab`).
"""
from __future__ import annotations

from typing import Dict, Optional

from PySide6 import QtWidgets

from gui.dls.region import _AnalysisRegion


#: The residual panel's floor, in pixels of drawn axes. Below this the panel stops
#: showing a residual at all. It does NOT fire at the shipped window size once the
#: legend is a key (measured: 59.8 px); it is insurance for a smaller window, where
#: the fraction-of-the-stack split under-serves a panel whose own chrome is fixed.
_RESID_FLOOR_PX = 45.0


#: The floor's search: at most this many steps, each raising the residual's share by
#: `_RESID_FLOOR_STEP`, never past `_RESID_RATIO_MAX` of the stack.
_RESID_FLOOR_STEPS = 12


_RESID_FLOOR_STEP = 0.02


_RESID_RATIO_MAX = 0.40


class _DeferredRepaint:
    """A DLS sub-tab's `set_measurement` in two halves, so that a tab out of sight can
    keep its focus at a sidebar click and paint it only when it is shown.

    - `focus_state(item_id, runnable)`, the state half, holds everything whose outcome
      depends on the history of calls (the focus, the keep-last verdict, the ticked
      sets) and returns whether there is anything to paint;
    - `repaint_view()`, the repaint half, paints from the stored focus and the
      controller as they are when it runs.

    A tab that sets `defers_repaint` has its state half alone run by
    `DLSModule.set_measurement` while it is hidden, which then sets `_repaint_pending`.
    `flush_view` replays the tab's own `set_measurement` while that mark is set: on the
    tab's show, and first in every entry point that reaches the tab from outside it, so
    its paints keep the order an eager repaint gives them. The replay re-runs the state
    half against the workspace as it is then, and replays a tab that has not split its
    `set_measurement` too."""

    #: Whether `DLSModule.set_measurement` defers this tab's repaint while it is hidden;
    #: set by a tab whose `set_measurement` is this class's.
    defers_repaint = False
    #: A repaint is owed (`DLSModule.set_measurement` sets it, `flush_view` pays it).
    _repaint_pending = False
    #: The focus a replay passes back; Summary, which stores none, reads these.
    item_id: Optional[str] = None
    _runnable = False

    def set_measurement(self, item_id: Optional[str], runnable: bool) -> None:
        """Run the state half, then the repaint when it asks for one; the pending mark
        is cleared before the repaint, so a flush the repaint itself reaches is a no-op
        and a repaint that raises is not replayed at every later entry."""
        if self.focus_state(item_id, runnable):
            self._repaint_pending = False
            self.repaint_view()

    def focus_state(self, item_id: Optional[str], runnable: bool) -> bool:
        """The state half: store the focus; False when there is nothing to repaint."""
        raise NotImplementedError

    def repaint_view(self) -> None:
        """The repaint half, from the stored focus and the controller as they are now.
        Reached through `set_measurement`, which clears the pending mark first."""
        raise NotImplementedError

    def flush_view(self) -> None:
        """Pay a repaint the tab owes, by replaying its `set_measurement` at its stored
        focus (the call every `showEvent` makes); nothing when none is owed."""
        if self._repaint_pending:
            self._repaint_pending = False
            self.set_measurement(self.item_id, self._runnable)


class _ResidualRatioMixin:
    """The draggable residual panel's share of a four-row gridspec, for a tab holding
    `canvas`, `_gs` (the gridspec), `resid_ax`, `_resid_ratio` and `_resid_dragged`.
    The first three row heights stay equal, so the panels above scale together, and the
    fourth row takes the residual's share; one canvas keeps the residual aligned."""

    def _apply_resid_ratio(self, frac: float) -> None:
        """Resize the residual against the panels above it (the drag handle, or the
        floor); constrained_layout re-flows on the redraw."""
        self._resid_ratio = frac
        a = (1.0 - frac) / 3.0
        self._gs.set_height_ratios([a, a, a, frac])
        self.canvas.draw_idle()

    def _floor_resid_ratio(self) -> None:
        """Raise the residual's share until its drawn box clears `_RESID_FLOOR_PX`.

        The share is a FRACTION of a varying stack while the panel's own chrome is a
        fixed ~29 px, so the fraction under-serves the panel exactly when the stack is
        short. It does not fire at the shipped window size (measured 59.8 px there);
        it is insurance for a smaller one.

        Only while the user has not dragged the handle: an explicit drag is a decision
        and outranks the floor (`_ResidualResizer` owns `_resid_ratio` after that).
        """
        if self._resid_dragged:
            return
        for _ in range(_RESID_FLOOR_STEPS):
            try:
                box = self.resid_ax.get_window_extent(self.canvas.get_renderer())
            except Exception:  # noqa: BLE001 — not laid out yet; nothing to floor
                return
            if box.height >= _RESID_FLOOR_PX or self._resid_ratio >= _RESID_RATIO_MAX:
                return
            self._apply_resid_ratio(min(self._resid_ratio + _RESID_FLOOR_STEP,
                                        _RESID_RATIO_MAX))
            self.canvas.draw()

    def _on_resid_dragged(self, frac: float) -> None:
        """The drag handle moved: apply it, and stop the floor second-guessing it.
        An explicit drag is a decision and outranks `_floor_resid_ratio`."""
        self._resid_dragged = True
        self._apply_resid_ratio(frac)


# ===========================================================================
# The sample-level sub-tabs' base (Γ vs q², D vs c, the Mode Map, DDLS)
# ===========================================================================

class _SampleRunTab(_DeferredRepaint, QtWidgets.QWidget):
    """A sample-level sub-tab (Γ vs q², D vs c, the Mode Map, DDLS): one sample at a
    time, following the sidebar focus, with the measurements ticked to include, the
    "use the Correlogram tab's delay window and baseline" choice, and the Run's slot.

    The hooks where the tabs differ: `_state_key`, the key the ticked set and the
    region choice are held under, (sample, fraction) unless overridden (DDLS has no
    fraction and keys by sample); `_region_moved`, what a moved region re-checks; and
    `_focus_hook`, the tab's own part of the state half (`focus_state`). A subclass
    builds `region_box` and `_run_slot`, passing the slot `before_delivery=flush_view`;
    its `_recompute` is its own."""

    def __init__(self, controller, parent=None, *, region=None) -> None:
        super().__init__(parent)
        self.controller = controller
        # The DLS tab's shared region (owned by DLSModule), read only when this tab's
        # "use the Correlogram tab's delay window and baseline" box is ticked; a
        # standalone tab gets one of its own that nothing ever sets.
        self._region = region if region is not None else _AnalysisRegion()
        # `_state_key` -> is that box ticked. Never saved: each new key starts unticked.
        self._use_region: Dict[object, bool] = {}
        self.item_id: Optional[str] = None
        self._runnable = False
        self._included: Dict[object, set] = {}   # `_state_key` -> the ticked set
        self._last_sid: Optional[str] = None     # last DLS sample shown (keep-last)

    # ---- sample and fraction keying ----
    def _sample_id(self) -> Optional[str]:
        if self.item_id is None:
            return None
        return self.controller.sample_id_of(self.item_id)

    def _fraction(self) -> Optional[str]:
        lm = self.controller.workspace.measurements.get(self.item_id)
        return lm.committed_params.get('mw_fraction') if lm is not None else None

    def _ckey(self, sid: str) -> tuple:
        """Fit-store key: a fit is strictly per (sample, focused fraction)."""
        return (sid, self._fraction())

    def _state_key(self, sid: str):
        """The key `sid`'s ticked set and region choice are held under."""
        return self._ckey(sid)

    def selected_item_ids(self) -> list:
        """The ticked measurements for the current sample (sidebar-mirror contract)."""
        sid = self._sample_id()
        return sorted(self._included.get(self._state_key(sid), set())) if sid else []

    # ---- the state half ----
    def focus_state(self, item_id: Optional[str], runnable: bool) -> bool:
        """The sample-level tabs' state half. Focusing a non-DLS measurement keeps the sample
        the tab follows (keep-last navigation): nothing to repaint. Otherwise the focus
        is stored, the tab's own hook runs, and `_last_sid` becomes the sample a
        runnable focus names, so it means one thing, the last DLS sample this tab
        followed; the hook reads it before it moves."""
        if item_id is not None and not runnable and self._last_sid is not None:
            return False
        self.item_id = item_id
        self._runnable = runnable
        sid = self._sample_id()
        self._focus_hook(sid)
        if runnable and sid is not None:
            self._last_sid = sid
        return True

    def _focus_hook(self, sid: Optional[str]) -> None:
        """The tab's own part of the state half, for the stored focus's sample."""
        raise NotImplementedError

    # ---- the region choice ----
    def _on_region_toggled(self, on: bool) -> None:
        sid = self._sample_id()
        if sid is None:
            return
        self._use_region[self._state_key(sid)] = bool(on)
        self._region_moved()

    def on_region_changed(self) -> None:
        """The Correlogram tab moved the shared region: relabel the box, and re-check
        the shown result (stale only while this sample's box is ticked). A repaint the
        tab owes runs first."""
        self.flush_view()
        self.region_box.refresh()
        self._region_moved()

    def _region_moved(self) -> None:
        """A region input moved: re-check the shown result against it."""
        self._on_region_input_moved()

    # ---- the Run's slot ----
    def _flush_recompute(self) -> None:
        """The deferred retry, run now (`RunSlot.flush`)."""
        self._run_slot.flush()

    @property
    def _run_epoch(self) -> int:
        """The Run's staleness epoch, held by its slot."""
        return self._run_slot.epoch

    @_run_epoch.setter
    def _run_epoch(self, value: int) -> None:
        self._run_slot.epoch = value

    @property
    def _recompute_pending(self):
        """The target (`_state_key`) of a Run deferred for idle, else False."""
        return self._run_slot.pending

    @_recompute_pending.setter
    def _recompute_pending(self, value) -> None:
        self._run_slot.pending = value
