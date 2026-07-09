"""
gui/help.py
===========

A small, consistent in-program help system (owner feedback 2026-06-29, Section B).

Two tiers, by design:

1. **How-to-use help is VISIBLE** — a circular "?" badge (`HelpBadge`) placed next to
   a section header. The badge is an obvious, clickable affordance; clicking it shows a
   concise popup. Use it for "what is this panel / how do I use it" guidance, and keep
   the text short (a sentence or a short bulleted list), pointing to the Advanced Guide
   for the underlying maths rather than re-deriving it inline.

2. **Calculation-nuance help is a passive tooltip** — an ordinary `setToolTip`. These
   are the quieter, hover-only notes. A global toggle (`set_tooltips_enabled`, wired to
   Settings → "Show tooltips") suppresses ALL passive tooltips for users who find them
   noisy; the visible "?" badges keep working on click regardless, so help is never
   fully hidden.

Nothing here imports analysis/physics — it is pure Qt presentation.
"""

from __future__ import annotations

from typing import Optional, Sequence

from PySide6 import QtCore, QtGui, QtWidgets

from gui.theme import token


# ---------------------------------------------------------------------------
# Global "show passive tooltips" gate
# ---------------------------------------------------------------------------
class _TooltipGate(QtCore.QObject):
    """An application-wide event filter that swallows tooltip events when disabled,
    so a single setting turns every passive `setToolTip` on/off without touching the
    individual widgets. `HelpBadge` popups are shown programmatically on click, so
    they are unaffected by this gate."""

    enabled: bool = True

    def eventFilter(self, obj, event) -> bool:
        if (not _TooltipGate.enabled
                and event.type() == QtCore.QEvent.Type.ToolTip):
            return True          # consume → Qt never shows the tooltip
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


def tooltips_enabled() -> bool:
    return _TooltipGate.enabled


# ---------------------------------------------------------------------------
# The "?" help badge
# ---------------------------------------------------------------------------
def _bullets(lines: Sequence[str]) -> str:
    """Render a short list as compact HTML bullets (lists beat paragraphs, Section B)."""
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

    def setHelp(self, text: str = '', *, bullets: Optional[Sequence[str]] = None) -> None:
        html = text or ''
        if bullets:
            html = (html + _bullets(bullets)) if html else _bullets(bullets)
        self._html = html
        self.setToolTip(html)

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
    """Float a "?" `HelpBadge` at the top-right corner of a QGroupBox's frame, next to
    its title. Returns the badge. (Qt group-box titles can't host child widgets, so the
    badge is a free child positioned in the box's `resizeEvent` shim.)"""
    badge = HelpBadge(help_text, bullets=bullets, parent=box)

    def _place() -> None:
        badge.move(box.width() - badge.width() - 8, 2)

    # Reposition on resize without subclassing: wrap the existing resizeEvent.
    _orig = box.resizeEvent

    def _resize(ev):
        _orig(ev)
        _place()

    box.resizeEvent = _resize        # type: ignore[method-assign]
    badge.show()
    _place()
    return badge
