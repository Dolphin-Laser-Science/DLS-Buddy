"""
gui/widgets.py
==============

Small cross-module Qt widget helpers that don't belong to any one tab:

* the tab-bar room fix (`RoomyTabBar` / `roomy_tabs`) — usability feedback
  2026-06-30 item 3;
* the visible-grip splitter (`GripSplitter`);
* the **shared selection layer** (`SelectionModel` + `MeasurementPicker`) — the one
  measurement-picker idiom every analysis tab uses (real checkboxes, include-semantics,
  select-all/none, grouped by sample). Promoted from the DLS-only checklist so all tabs
  share one look and one model, and the sidebar can mirror "what's selected".

The tab-bar fix widens each tab's **size hint** rather than applying a `QTabBar::tab`
stylesheet — a stylesheet would override Fusion's native tab painting (flat, unstyled
tabs), whereas an overridden `tabSizeHint` only reserves a little more width and leaves
the painting alone.

Nothing here imports analysis/physics — pure Qt presentation (it does use the sibling
`gui.help` / `gui.theme` presentation helpers).
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence

from PySide6 import QtCore, QtGui, QtWidgets

from gui.help import section_header
from gui.theme import color

# Extra horizontal room added to every tab's size hint, in device-independent px.
# Tunable: bump it if a label still clips on a particular font/DPI.
TAB_EXTRA_PX = 14


def value_unit_row(value_widget: QtWidgets.QWidget,
                   unit_widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
    """A ``[value][unit]`` composite for a form row (the synthetic generator's and the
    Solvent Explorer's unit-aware inputs). Shared here so the two callers don't drift."""
    row = QtWidgets.QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(value_widget)
    row.addWidget(unit_widget)
    holder = QtWidgets.QWidget()
    holder.setLayout(row)
    return holder


class RoomyTabBar(QtWidgets.QTabBar):
    """A QTabBar that reserves a little extra width per tab so labels don't clip.

    Only the size hint is changed; Fusion still paints the tabs natively."""

    def tabSizeHint(self, index: int):
        size = super().tabSizeHint(index)
        size.setWidth(size.width() + TAB_EXTRA_PX)
        return size


def roomy_tabs(tab_widget: QtWidgets.QTabWidget) -> QtWidgets.QTabWidget:
    """Give `tab_widget` a :class:`RoomyTabBar` so its labels don't clip. Call this
    right after constructing the QTabWidget, before adding tabs. Returns the widget for
    convenience."""
    tab_widget.setTabBar(RoomyTabBar())
    return tab_widget


# ---------------------------------------------------------------------------
# Splitter with a visible "grip" handle (usability feedback 2026-06-30 #5/#9)
# ---------------------------------------------------------------------------
# Qt's default splitter handle is a near-invisible thin line, so users can't tell a
# divider is draggable. GripSplitter paints a small centered grip (three dots) on every
# handle, giving a consistent "drag me" cue to the controls↔plot divider, the stacked
# plots, and the resizable control columns alike.

class _GripHandle(QtWidgets.QSplitterHandle):
    """A splitter handle that paints three centered grip dots over the default handle.
    Dots run across the handle's short axis (a vertical column for a horizontal splitter,
    a horizontal row for a vertical one). The colour follows the theme (palette `Mid`)."""

    _DOT_RADIUS = 1.6
    _DOT_GAP = 5.0

    def paintEvent(self, ev) -> None:
        super().paintEvent(ev)               # keep the native handle look underneath
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(self.palette().color(QtGui.QPalette.ColorRole.Mid))
        r = self.rect()
        cx, cy = r.center().x(), r.center().y()
        for k in (-1, 0, 1):
            if self.orientation() == QtCore.Qt.Orientation.Horizontal:
                centre = QtCore.QPointF(cx, cy + k * self._DOT_GAP)   # vertical handle
            else:
                centre = QtCore.QPointF(cx + k * self._DOT_GAP, cy)   # horizontal handle
            p.drawEllipse(centre, self._DOT_RADIUS, self._DOT_RADIUS)
        p.end()

    def changeEvent(self, ev) -> None:
        if ev.type() == QtCore.QEvent.Type.PaletteChange:
            self.update()                    # recolour the dots on a theme switch
        super().changeEvent(ev)


class GripSplitter(QtWidgets.QSplitter):
    """A QSplitter whose handles show a visible grip so users can tell the divider is
    draggable. Otherwise behaves exactly like QSplitter."""

    def __init__(self, orientation, parent=None) -> None:
        super().__init__(orientation, parent)
        self.setHandleWidth(8)
        self.setChildrenCollapsible(False)

    def createHandle(self) -> QtWidgets.QSplitterHandle:
        return _GripHandle(self.orientation(), self)


# ---------------------------------------------------------------------------
# Shared measurement-selection layer (SelectionModel + MeasurementPicker)
# ---------------------------------------------------------------------------
# One idiom for "which measurements is this tab analysing?", used by every analysis
# tab: real checkboxes, INCLUDE-semantics (tick = use it), select-all/none, grouped by
# sample. The model is GUI-side only (not persisted — architecture invariant #5: the
# controller stays Qt-free/stateless about selection; analysis calls take the ticked
# ids inline via `include_ids`). The picker is a thin view over a model; two pickers can
# share one model (e.g. the DLS Correlogram + Distribution overlay) and stay in lock-step
# because both react to the model's `changed` signal.

class SelectionModel(QtCore.QObject):
    """An ordered, unique set of measurement ``item_id`` strings currently ticked for
    analysis. Emits ``changed`` on any mutation so a bound :class:`MeasurementPicker`
    (or the sidebar mirror) re-syncs without manual bookkeeping.

    Generalised from the DLS-only ``_OverlaySelection``. Insertion order is preserved so
    each measurement keeps a stable overlay colour (`colour_for`) across tabs. Pass the
    plotting colour cycle in via ``colour_cycle`` when overlay colours are needed (kept
    an argument so this module imports no plotting/physics)."""

    changed = QtCore.Signal()

    def __init__(self, parent: Optional[QtCore.QObject] = None, *,
                 colour_cycle: Optional[Sequence[str]] = None) -> None:
        super().__init__(parent)
        self._ids: list[str] = []                       # ordered, unique
        self._cycle = list(colour_cycle) if colour_cycle else None

    def ids(self) -> list[str]:
        return list(self._ids)

    def is_checked(self, iid: str) -> bool:
        return iid in self._ids

    def set(self, iid: str, on: bool) -> None:
        """Tick/untick one id; emits ``changed`` iff the set actually mutated."""
        if on and iid not in self._ids:
            self._ids.append(iid)
        elif not on and iid in self._ids:
            self._ids.remove(iid)
        else:
            return
        self.changed.emit()

    def set_many(self, iids: Iterable[str], on: bool) -> None:
        """Batch tick/untick; a single ``changed`` for the whole batch (select all/none)."""
        mutated = False
        for iid in iids:
            if on and iid not in self._ids:
                self._ids.append(iid); mutated = True
            elif not on and iid in self._ids:
                self._ids.remove(iid); mutated = True
        if mutated:
            self.changed.emit()

    def set_only(self, iid: Optional[str]) -> None:
        """Radio behaviour: make ``iid`` the sole ticked id (empty when None/''). Used by
        single-select pickers."""
        new = [iid] if iid else []
        if new != self._ids:
            self._ids = new
            self.changed.emit()

    def ensure(self, iid: Optional[str]) -> None:
        """Tick ``iid`` if absent (used to auto-include a focused pick)."""
        if iid and iid not in self._ids:
            self._ids.append(iid)
            self.changed.emit()

    def prune(self, live: Iterable[str]) -> None:
        """Drop ids no longer in the workspace (keeping the rest in order)."""
        live = set(live)
        kept = [i for i in self._ids if i in live]
        if kept != self._ids:
            self._ids = kept
            self.changed.emit()

    def clear(self) -> None:
        if self._ids:
            self._ids = []
            self.changed.emit()

    def colour_for(self, iid: str) -> Optional[str]:
        """A stable palette colour keyed by the id's position, or None if no colour cycle
        was supplied."""
        if not self._cycle:
            return None
        idx = self._ids.index(iid) if iid in self._ids else len(self._ids)
        return self._cycle[idx % len(self._cycle)]


class MeasurementPicker(QtWidgets.QWidget):
    """The shared "which measurements?" picker: a sample-grouped list of **real
    checkboxes** (`Qt.ItemIsUserCheckable`, include-semantics) backed by a
    :class:`SelectionModel`, with a "?" help badge and Select all / none.

    Physics-free: the caller injects ``label_fn(lm) -> str`` and ``header_fn(sample) ->
    str`` so DLS/SLS/Utilities each supply their own row/group labels. ``kinds`` filters
    which measurement kinds are eligible. ``single=True`` gives radio behaviour (ticking
    one unticks the rest) for the sample-scoped tabs. Emits ``selectionChanged`` whenever
    the ticked set changes (a re-emit of the model's ``changed``)."""

    selectionChanged = QtCore.Signal()

    def __init__(self, controller, model: SelectionModel, *,
                 kinds: Sequence[str] = ('dls',),
                 label_fn: Callable[[object], str],
                 header_fn: Callable[[object], str],
                 title: str = 'Measurements to plot',
                 help_text: str = '', help_bullets: Optional[Sequence[str]] = None,
                 single: bool = False,
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._model = model
        self._kinds = tuple(kinds)
        self._label_fn = label_fn
        self._header_fn = header_fn
        self._single = single
        self._leaf_by_id: dict = {}          # item_id -> QListWidgetItem (checkable leaves)

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        # Title + "?" badge (doc-rule #8: how-to-use help is the visible badge).
        v.addWidget(section_header(title, help_text, bullets=help_bullets))
        self._list = QtWidgets.QListWidget()
        # Min (not max) height so the list grows to fill its resizable splitter pane.
        self._list.setMinimumHeight(80)
        self._list.itemChanged.connect(self._on_item_changed)
        v.addWidget(self._list)
        # Select all / none (all is meaningless for a single-select picker → hidden then).
        btn_row = QtWidgets.QHBoxLayout()
        self._btn_all = QtWidgets.QPushButton('Select all')
        self._btn_none = QtWidgets.QPushButton('Select none')
        self._btn_all.clicked.connect(lambda: self._model.set_many(self._leaf_by_id, True))
        self._btn_none.clicked.connect(lambda: self._model.set_many(self._leaf_by_id, False))
        if single:
            self._btn_all.hide()
        btn_row.addWidget(self._btn_all)
        btn_row.addWidget(self._btn_none)
        btn_row.addStretch(1)
        v.addLayout(btn_row)

        # The model is the single source of truth: any mutation (this picker, a sibling
        # picker sharing the model, a prune) re-syncs the checkboxes here and re-emits.
        self._model.changed.connect(self._on_model_changed)

    # -- population ---------------------------------------------------------
    def refresh(self) -> None:
        """Rebuild the rows from the live workspace and re-tick them from the model.
        Prunes ids the workspace no longer has."""
        self._list.blockSignals(True)
        self._list.clear()
        self._leaf_by_id = {}
        live: set = set()
        groups: dict = {}                    # sid -> [lm, ...]  insertion-ordered
        for lm in self._controller.workspace.measurements.values():
            if lm.kind not in self._kinds:
                continue
            live.add(lm.item_id)
            sid = self._controller.sample_id_of(lm.item_id)
            groups.setdefault(sid, []).append(lm)
        for sid, ms in groups.items():
            sample = self._controller.workspace.samples.get(sid)
            hdr = QtWidgets.QListWidgetItem(
                self._header_fn(sample) if sample else '(unconfirmed)')
            hdr.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)   # header, not checkable
            font = hdr.font(); font.setBold(True); hdr.setFont(font)
            hdr.setForeground(color(self._list, 'marker_group'))
            self._list.addItem(hdr)
            for lm in ms:
                it = QtWidgets.QListWidgetItem('  ' + self._label_fn(lm))
                it.setData(QtCore.Qt.ItemDataRole.UserRole, lm.item_id)
                it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(QtCore.Qt.CheckState.Checked
                                 if self._model.is_checked(lm.item_id)
                                 else QtCore.Qt.CheckState.Unchecked)
                self._list.addItem(it)
                self._leaf_by_id[lm.item_id] = it
        self._list.blockSignals(False)
        self._model.prune(live)              # emits changed → _on_model_changed if it drops any
        self._apply_palette()                # tint the checked rows for the current theme

    def selected_item_ids(self) -> list[str]:
        """The ticked measurements that still exist as rows (model order preserved)."""
        return [i for i in self._model.ids() if i in self._leaf_by_id]

    # -- signal plumbing ----------------------------------------------------
    def _on_item_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        iid = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if iid is None:                      # a group header
            return
        on = item.checkState() == QtCore.Qt.CheckState.Checked
        if self._single and on:
            self._model.set_only(iid)        # radio: this becomes the sole pick
        else:
            self._model.set(iid, on)

    def _on_model_changed(self) -> None:
        self._sync_checks()
        self.selectionChanged.emit()

    def _sync_checks(self) -> None:
        """Push the model's ticked set onto the checkboxes (used when a sibling picker or
        a prune mutates the shared model). Guarded so it doesn't re-enter `itemChanged`."""
        self._list.blockSignals(True)
        for iid, it in self._leaf_by_id.items():
            want = (QtCore.Qt.CheckState.Checked if self._model.is_checked(iid)
                    else QtCore.Qt.CheckState.Unchecked)
            if it.checkState() != want:
                it.setCheckState(want)
        self._list.blockSignals(False)
        self._apply_palette()

    # -- theme --------------------------------------------------------------
    def _apply_palette(self) -> None:
        """Recolour group headers and tint the ticked rows with `marker_selected` for the
        current theme. Named `_apply_palette` so `gui.theme.retheme` re-runs it on a theme
        switch (alongside the per-widget PaletteChange below)."""
        for i in range(self._list.count()):
            it = self._list.item(i)
            iid = it.data(QtCore.Qt.ItemDataRole.UserRole)
            if iid is None:                  # header
                it.setForeground(color(self._list, 'marker_group'))
            elif it.checkState() == QtCore.Qt.CheckState.Checked:
                it.setForeground(color(self._list, 'marker_selected'))
            else:
                it.setForeground(QtGui.QBrush())   # reset → follow palette text colour

    def changeEvent(self, ev: QtCore.QEvent) -> None:
        if ev.type() == QtCore.QEvent.Type.PaletteChange:
            self._apply_palette()
        super().changeEvent(ev)


class SampleSelector(QtWidgets.QWidget):
    """A labelled dropdown of SAMPLES for the sample-scoped analysis tabs (SLS, I·sinθ),
    the sample-level analog of :class:`MeasurementPicker`. Lists the samples matching a
    ``predicate`` (e.g. "has SLS data"); emits ``sampleChanged(sample_id)`` (``''`` when
    none) on a user pick. The tab owns its sample here rather than inheriting the shell's
    focus — the sidebar merely navigates.

    ``predicate(sample) -> bool`` filters eligibility; ``label_fn(sample) -> str`` labels
    each row. A "?" help badge satisfies doc-rule #8."""

    sampleChanged = QtCore.Signal(str)          # sample_id, or '' when none

    def __init__(self, controller, *, predicate: Callable[[object], bool],
                 label_fn: Callable[[object], str], title: str = 'Sample',
                 help_text: str = '', help_bullets: Optional[Sequence[str]] = None,
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._predicate = predicate
        self._label_fn = label_fn
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(section_header(title, help_text, bullets=help_bullets))
        self._combo = QtWidgets.QComboBox()
        self._combo.currentIndexChanged.connect(self._on_changed)
        v.addWidget(self._combo)

    def refresh(self) -> None:
        """Rebuild the sample list, preserving the current pick if it still qualifies."""
        keep = self.current_sample_id()
        self._combo.blockSignals(True)
        self._combo.clear()
        sids = []
        for s in self._controller.samples():
            if self._predicate(s):
                self._combo.addItem(self._label_fn(s), s.sample_id)
                sids.append(s.sample_id)
        if not sids:
            self._combo.addItem('(no eligible samples)', None)
            self._combo.setEnabled(False)
        else:
            self._combo.setEnabled(True)
            if keep in sids:
                self._combo.setCurrentIndex(sids.index(keep))
        self._combo.blockSignals(False)

    def has_sample(self, sid: str) -> bool:
        return any(self._combo.itemData(i) == sid for i in range(self._combo.count()))

    def current_sample_id(self) -> Optional[str]:
        return self._combo.currentData()

    def set_current_sample_id(self, sid: Optional[str]) -> None:
        """Select `sid` programmatically WITHOUT emitting (a soft sync from the shell's
        focus — never fights a user pick)."""
        for i in range(self._combo.count()):
            if self._combo.itemData(i) == sid:
                self._combo.blockSignals(True)
                self._combo.setCurrentIndex(i)
                self._combo.blockSignals(False)
                return

    def _on_changed(self, _i: int) -> None:
        self.sampleChanged.emit(self.current_sample_id() or '')
