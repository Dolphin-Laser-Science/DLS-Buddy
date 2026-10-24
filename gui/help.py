"""
gui/help.py
===========

A small, consistent in-program help system.

Two tiers, by design:

1. **How-to-use help is VISIBLE** — a circular "?" badge (`HelpBadge`) placed next to
   a section header. The badge is an obvious, clickable affordance; clicking it shows a
   concise popup. Use it for "what is this panel / how do I use it" guidance, and keep
   the text short (a sentence or a short bulleted list), pointing to the Theory-and-Equations-Guide
   for the underlying maths rather than re-deriving it inline.

2. **Calculation-nuance help is a passive tooltip** — an ordinary `setToolTip`. These
   are the quieter, hover-only notes. A global toggle (`set_tooltips_enabled`, wired to
   Settings → "Show tooltips") suppresses ALL passive tooltips for users who find them
   noisy; the visible "?" badges keep working on click regardless, so help is never
   fully hidden.

Nothing here imports analysis/physics — it is pure Qt presentation.
"""

from __future__ import annotations

import html
from typing import Optional, Sequence

from PySide6 import QtCore, QtGui, QtWidgets

from gui.theme import token

# Qt word-wraps a tooltip only when it is rich text, and its own rich-text width is too narrow
# to read (ui_style_guide R6.10); so a plain tooltip whose longest line is wider than this is
# shown as a block of this width, and a shorter one as it is.
TOOLTIP_WRAP_PX = 420


def wrapped_tooltip(text: str, font: QtGui.QFont) -> str:
    """`text` as the tooltip gate shows it: unchanged when it is empty, rich text, or no line
    of it is wider than `TOOLTIP_WRAP_PX` in `font`; otherwise escaped into a block of that
    width, its line breaks kept."""
    if not text or QtGui.Qt.mightBeRichText(text):
        return text
    metrics = QtGui.QFontMetrics(font)
    if max(metrics.horizontalAdvance(line) for line in text.split('\n')) <= TOOLTIP_WRAP_PX:
        return text
    body = html.escape(text, quote=False).replace('\n', '<br>')
    return f'<table width="{TOOLTIP_WRAP_PX}"><tr><td>{body}</td></tr></table>'


# ---------------------------------------------------------------------------
# Global "show passive tooltips" gate
# ---------------------------------------------------------------------------
def _hovered_tooltip(obj, event) -> "Optional[tuple]":
    """(text, widget, rect) of the tooltip Qt would show for this hover, read where Qt reads
    it: an item view's cell or header section, a menu action, else the widget's own. None
    where Qt would show nothing here (an empty tooltip propagates to the parent)."""
    if not isinstance(obj, QtWidgets.QWidget):
        return None
    pos = event.pos()
    view = obj.parentWidget()
    role = QtCore.Qt.ItemDataRole.ToolTipRole
    if isinstance(view, QtWidgets.QHeaderView) and obj is view.viewport():
        section = view.logicalIndexAt(pos)
        if section < 0 or view.model() is None:
            return None
        text = view.model().headerData(section, view.orientation(), role)
        start, size = view.sectionViewportPosition(section), view.sectionSize(section)
        rect = (QtCore.QRect(start, 0, size, obj.height())
                if view.orientation() == QtCore.Qt.Orientation.Horizontal
                else QtCore.QRect(0, start, obj.width(), size))
    elif isinstance(view, QtWidgets.QAbstractItemView) and obj is view.viewport():
        index = view.indexAt(pos)
        if not index.isValid():
            return None
        text, rect = index.data(role), view.visualRect(index)
    elif isinstance(obj, QtWidgets.QMenu):
        action = obj.actionAt(pos)
        # QAction.toolTip() falls back to the action's text; Qt shows only one set explicitly.
        if (action is None or not obj.toolTipsVisible()
                or action.toolTip() == action.text().replace('&', '')):
            return None
        text, rect = action.toolTip(), obj.actionGeometry(action)
    else:
        text, rect = obj.toolTip(), QtCore.QRect()
    return (text, obj, rect) if isinstance(text, str) and text else None


class _TooltipGate(QtCore.QObject):
    """An application-wide event filter over every hover tooltip. When disabled it swallows
    them, so a single setting turns every passive `setToolTip` on/off without touching the
    individual widgets; when enabled it shows a long plain tooltip wrapped
    (`wrapped_tooltip`) and leaves every other one to Qt. `HelpBadge` popups are shown
    programmatically on click, so they are unaffected by this gate."""

    enabled: bool = True

    def eventFilter(self, obj, event) -> bool:
        if event.type() != QtCore.QEvent.Type.ToolTip:
            return super().eventFilter(obj, event)
        if not _TooltipGate.enabled:
            return True          # consume → Qt never shows the tooltip
        hovered = _hovered_tooltip(obj, event)
        if hovered is not None:
            text, widget, rect = hovered
            shown = wrapped_tooltip(text, QtWidgets.QToolTip.font())
            if shown != text:
                QtWidgets.QToolTip.showText(event.globalPos(), shown, widget, rect)
                return True
        return super().eventFilter(obj, event)


_gate: Optional[_TooltipGate] = None


def install_tooltip_gate(app: QtWidgets.QApplication) -> None:
    """Install the global tooltip gate once on the application (idempotent)."""
    global _gate
    if _gate is None:
        _gate = _TooltipGate()
        app.installEventFilter(_gate)


def set_tooltips_enabled(on: bool) -> None:
    """Enable/disable all passive (hover) tooltips application-wide."""
    _TooltipGate.enabled = bool(on)


# ---------------------------------------------------------------------------
# The "?" help badge
# ---------------------------------------------------------------------------
def _bullets(lines: Sequence[str]) -> str:
    """Render a short list as compact HTML bullets (lists beat paragraphs)."""
    items = ''.join(f'<li>{ln}</li>' for ln in lines)
    return f'<ul style="margin:2px 0 0 0; -qt-list-indent:1;">{items}</ul>'


class HelpBadge(QtWidgets.QToolButton):
    """A small circular "?" button that shows a concise help popup.

    `text` may be plain or simple HTML. Pass `bullets=[...]` instead (optionally with a
    leading `text` intro) to render a short list. Clicking always shows the popup;
    hovering shows it too, but only while passive tooltips are enabled (so the global
    Settings toggle quietens the hover preview while the click affordance still works).
    """

    _QSS = (
        'QToolButton {{ border: 1px solid {fg}; border-radius: 8px; '
        'min-width: 16px; max-width: 16px; min-height: 16px; max-height: 16px; '
        'padding: 0; font-weight: bold; font-size: 10px; color: {fg}; }}'
        'QToolButton:hover {{ background: {fg}; color: {bg}; }}'
    )

    def __init__(self, text: str = '', *, bullets: Optional[Sequence[str]] = None,
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        html = text or ''
        if bullets:
            html = (html + _bullets(bullets)) if html else _bullets(bullets)
        self._html = html
        self.setText('?')
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setAutoRaise(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.setAccessibleName('Help')
        self._applying = False
        # A subtle, theme-following look. The color is re-applied on every theme switch
        # (see changeEvent) rather than captured once — the old code read the palette in
        # __init__ and froze, leaving the badge faint on the dark theme.
        self._apply_palette()
        # Hover preview via the normal tooltip system (so it honors the global gate).
        self.setToolTip(self._html)
        self.clicked.connect(self._popup)

    def _apply_palette(self) -> None:
        """(Re)compute the badge colors from the live theme and apply the QSS."""
        if self._applying:        # guard: setStyleSheet re-enters changeEvent (see ThemedLabel)
            return
        self._applying = True
        try:
            fg = token(self, 'badge')   # guaranteed-contrast token, not the old unset Mid role
            bg = self.palette().color(QtGui.QPalette.ColorRole.Base).name()
            self.setStyleSheet(self._QSS.format(fg=fg, bg=bg))
        finally:
            self._applying = False

    def changeEvent(self, ev: QtCore.QEvent) -> None:
        # A theme switch delivers PaletteChange; recolor so the badge never freezes.
        if ev.type() == QtCore.QEvent.Type.PaletteChange:
            self._apply_palette()
        super().changeEvent(ev)

    def _popup(self) -> None:
        # Programmatic show — independent of the passive-tooltip gate, so the badge is
        # always usable even when hover tooltips are switched off.
        QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), self._html, self)


# ---------------------------------------------------------------------------
# Section header + badge
# ---------------------------------------------------------------------------
def section_header(title: str, help_text: str = '', *,
                   bullets: Optional[Sequence[str]] = None,
                   bold: bool = True) -> QtWidgets.QWidget:
    """A row: a section title label followed by a "?" `HelpBadge`, then stretch.

    Drop it above a group of controls (or use `add_help_to_groupbox` to attach a badge
    to an existing QGroupBox's title row)."""
    w = QtWidgets.QWidget()
    row = QtWidgets.QHBoxLayout(w)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(5)
    label = QtWidgets.QLabel(f'<b>{title}</b>' if bold else title)
    row.addWidget(label)
    if help_text or bullets:
        row.addWidget(HelpBadge(help_text, bullets=bullets))
    row.addStretch(1)
    return w


def add_help_to_groupbox(box: QtWidgets.QGroupBox, help_text: str = '', *,
                         bullets: Optional[Sequence[str]] = None) -> HelpBadge:
    """Float a "?" `HelpBadge` right after a QGroupBox's title, the place
    `section_header` puts it (GUI style guide §6). Returns the badge. (Qt group-box
    titles can't host child widgets, so the badge is a free child positioned in the
    box's `resizeEvent` shim.) The title's rectangle comes from the style, so the badge
    follows the title's real width and font; a box too narrow for both keeps the badge
    inside its right edge."""
    badge = HelpBadge(help_text, bullets=bullets, parent=box)

    def _place() -> None:
        opt = QtWidgets.QStyleOptionGroupBox()
        box.initStyleOption(opt)
        title = box.style().subControlRect(
            QtWidgets.QStyle.ComplexControl.CC_GroupBox, opt,
            QtWidgets.QStyle.SubControl.SC_GroupBoxLabel, box)
        x = min(title.right() + 6, box.width() - badge.width() - 8)
        y = max(0, title.center().y() - badge.height() // 2)
        badge.move(x, y)

    # Reposition on resize without subclassing: wrap the existing resizeEvent.
    _orig = box.resizeEvent

    def _resize(ev):
        _orig(ev)
        _place()

    box.resizeEvent = _resize        # type: ignore[method-assign]
    badge.show()
    _place()
    return badge
