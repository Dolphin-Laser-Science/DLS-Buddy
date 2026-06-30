"""
gui/widgets.py
==============

Small cross-module Qt widget helpers that don't belong to any one tab.

Currently just the tab-bar room fix (usability feedback 2026-06-30 item 3): Qt's
Fusion style sizes each tab a pixel or two too tight for the system font/DPI, so the
last glyph of a label (e.g. the "s" of *Settings*) gets clipped. We widen the tab
**size hint** rather than apply a `QTabBar::tab` stylesheet — a stylesheet would
override Fusion's native tab painting (flat, unstyled tabs), whereas an overridden
`tabSizeHint` only reserves a little more width and leaves the painting alone.

Nothing here imports analysis/physics — pure Qt presentation.
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

# Extra horizontal room added to every tab's size hint, in device-independent px.
# Tunable: bump it if a label still clips on a particular font/DPI.
TAB_EXTRA_PX = 14


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
