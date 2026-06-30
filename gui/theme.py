"""
gui/theme.py
============

A tiny, theme-aware **UI colour-token** layer (owner feedback 2026-06-30, items 1–2).

Why this exists
---------------
The app's window/base/text colours live in an explicit ``QPalette``
(``gui/main_window.py`` ``_build_palette``), and matplotlib line colours live in a
role→colour palette (``plotting/plots.py``). But the small *accent* colours scattered
through the widgets — section headers, muted notes, hint text, error flags, the pending
amber, tree markers — were inline hex literals (``color:#555`` …). Two problems:

1. They were tuned for a light background, so several are low-contrast on the dark theme.
2. A ``QLabel`` that sets *any* stylesheet **without** a ``color`` stops following the
   application palette and **freezes** to whatever palette was active when it was built
   (this is why the Settings headers rendered the dark theme's light-grey on the light
   theme — illegible).

This module fixes both by routing accent colours through **semantic role tokens** with a
per-theme value, and by giving the themed widgets a ``changeEvent`` hook so they
**re-apply on a theme switch** instead of freezing.

Design (kept future-theme-friendly — Note A)
--------------------------------------------
- Tokens are split into a ``LIGHT_TOKENS`` set and a ``DARK_TOKENS`` set, selected at
  paint time by ``is_dark(widget)`` (inferred from the *live* palette's Window
  lightness). That means ``'system'`` and any future theme that is fundamentally light
  or dark just work — no global theme-name state to keep in sync.
- ``ThemedLabel`` / ``themed_label`` for persistent labels; ``span`` for the inline-HTML
  status strings that are regenerated on each redraw.

Nothing here imports analysis/physics — pure Qt presentation.
"""

from __future__ import annotations

from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets


# ---------------------------------------------------------------------------
# Role → colour, per theme
# ---------------------------------------------------------------------------
# Roles (semantic, not literal):
#   header        — bold section-header text
#   muted         — secondary body text (the darker grey notes)
#   hint          — quieter, smaller hint text (the lighter grey notes)
#   error         — error / unreliable-result flags
#   pending       — "pending update" amber
#   marker_group  — tree group-header rows (DLS / SLS / Traces)
#   marker_active — derived / replicate-average tree leaves
#   badge         — the "?" help-badge outline + glyph
#
# Light values keep today's intent but stay clearly readable on white; dark values are
# chosen for contrast on the dark base (Window/Base are 35–53 grey).
LIGHT_TOKENS = {
    'header':        '#1a1a1a',
    'muted':         '#4d4d4d',
    'hint':          '#6e6e6e',
    'error':         '#cc0000',
    'pending':       '#b06000',
    'marker_group':  '#808080',
    'marker_active': '#008f63',
    'badge':         '#777777',
}

DARK_TOKENS = {
    'header':        '#e8e8e8',
    'muted':         '#b4b4b4',
    'hint':          '#9a9a9a',
    'error':         '#ff6b6b',
    'pending':       '#e0a050',
    'marker_group':  '#9a9a9a',
    'marker_active': '#3fd0a0',
    'badge':         '#bdbdbd',
}


# ---------------------------------------------------------------------------
# Theme detection + lookup
# ---------------------------------------------------------------------------
def is_dark(widget: Optional[QtWidgets.QWidget] = None) -> bool:
    """True if the active theme is dark, inferred from the **application** palette's
    Window lightness. Works for explicit light/dark AND 'system' (and any future theme
    that is fundamentally light or dark), so there is no theme-name global to keep in
    sync.

    The application palette is used (not the widget's) deliberately: the theme is global,
    and ``QApplication.setPalette`` updates it synchronously, whereas a child widget's
    resolved palette may lag during a theme switch. `widget` is only a fallback for the
    (unusual) case where no QApplication exists yet."""
    app = QtWidgets.QApplication.instance()
    if app is not None:
        pal = app.palette()
    elif widget is not None:
        pal = widget.palette()
    else:
        pal = QtGui.QPalette()
    return pal.color(QtGui.QPalette.ColorRole.Window).value() < 128


def token(widget: Optional[QtWidgets.QWidget], role: str) -> str:
    """The hex colour for `role` under the widget's current theme."""
    table = DARK_TOKENS if is_dark(widget) else LIGHT_TOKENS
    return table[role]


def color(widget: Optional[QtWidgets.QWidget], role: str) -> QtGui.QColor:
    """`token` as a QColor (for QPainter / setForeground sites)."""
    return QtGui.QColor(token(widget, role))


# ---------------------------------------------------------------------------
# Themed label
# ---------------------------------------------------------------------------
class ThemedLabel(QtWidgets.QLabel):
    """A QLabel whose text colour follows a theme **token** and re-applies itself on a
    theme switch (unlike a plain ``setStyleSheet('color:#…')`` label, which freezes).

    `role` picks the token; `bold`/`size`/`extra` add the usual styling without losing
    the theme-following behaviour. `size` is a px font-size; `extra` is raw extra QSS
    (e.g. ``'margin-top:8px;'``)."""

    def __init__(self, text: str = '', role: str = 'muted', *,
                 bold: bool = False, size: Optional[int] = None, extra: str = '',
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(text, parent)
        self._role = role
        self._bold = bold
        self._size = size
        self._extra = extra
        self._applying = False
        self._apply()

    def setRole(self, role: str) -> None:
        self._role = role
        self._apply()

    def _apply(self) -> None:
        # Re-entrancy guard: setStyleSheet re-polishes the widget, which can deliver
        # another change event back into changeEvent → _apply. Without this guard that
        # recurses without bound (and would crash on a real theme switch).
        if self._applying:
            return
        self._applying = True
        try:
            parts = [f'color:{token(self, self._role)};']
            if self._bold:
                parts.append('font-weight:bold;')
            if self._size:
                parts.append(f'font-size:{self._size}px;')
            if self._extra:
                parts.append(self._extra)
            # Scope to QLabel so the rule doesn't bleed to child widgets.
            self.setStyleSheet('QLabel { ' + ' '.join(parts) + ' }')
        finally:
            self._applying = False

    def changeEvent(self, ev: QtCore.QEvent) -> None:
        # A theme switch (app.setPalette) delivers PaletteChange to every widget; recolour
        # from the new palette. Re-applying a stylesheet raises StyleChange (not
        # PaletteChange), so this does not recurse.
        if ev.type() == QtCore.QEvent.Type.PaletteChange:
            self._apply()
        super().changeEvent(ev)


def themed_label(text: str = '', role: str = 'muted', *, bold: bool = False,
                 size: Optional[int] = None, extra: str = '',
                 parent: Optional[QtWidgets.QWidget] = None) -> ThemedLabel:
    """Terse factory for a :class:`ThemedLabel`."""
    return ThemedLabel(text, role, bold=bold, size=size, extra=extra, parent=parent)


# ---------------------------------------------------------------------------
# Inline-HTML span (for status strings rebuilt on each redraw)
# ---------------------------------------------------------------------------
def span(widget: Optional[QtWidgets.QWidget], role: str, text: str) -> str:
    """An HTML ``<span>`` coloured by `role` for the current theme. Use inside the
    dynamically-built rich-text status strings; they re-emit with a fresh token whenever
    the owning view redraws (which the shell triggers after a theme change)."""
    return f'<span style="color:{token(widget, role)}">{text}</span>'


# ---------------------------------------------------------------------------
# Deterministic re-theme of an existing widget tree
# ---------------------------------------------------------------------------
def retheme(root: QtWidgets.QWidget) -> None:
    """Re-apply token colours to every themed widget under `root`. Call this AFTER
    ``QApplication.setPalette`` on a theme switch: it does not rely on per-widget
    PaletteChange delivery (which a stylesheet'd widget can miss / receive before its
    own palette resolves). Covers :class:`ThemedLabel` and the help ``?`` badges (found
    by their ``_apply_palette`` method, to avoid importing ``gui.help`` here)."""
    for lb in root.findChildren(ThemedLabel):
        lb._apply()
    for w in root.findChildren(QtWidgets.QWidget):
        fn = getattr(w, '_apply_palette', None)
        if callable(fn):
            fn()
