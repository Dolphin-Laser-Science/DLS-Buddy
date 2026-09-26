"""
gui/widgets.py
==============

Small cross-module Qt widget helpers that don't belong to any one tab:

* the tab-bar room fix (`RoomyTabBar` / `roomy_tabs`) — a usability fix;
* the visible-grip splitter (`GripSplitter`);
* the **shared selection layer** (`SelectionModel` + `MeasurementPicker`) — the one
  measurement-picker idiom every analysis tab uses (real checkboxes, include-semantics,
  select-all/none, grouped by sample). Promoted from the DLS-only checklist so all tabs
  share one look and one model, and the sidebar can mirror "what's selected";
* the **bulk group-tick row** (`GroupTickBar`) — a "tick all at concentration / angle"
  helper shared by the Γ-q²/D-c tabs and the Distribution picker;
* the **offered-source picker** (`SourcePicker`) — the one affordance for "this input
  can come from more than one place; which one?".

The tab-bar fix widens each tab's **size hint** rather than applying a `QTabBar::tab`
stylesheet — a stylesheet would override Fusion's native tab painting (flat, unstyled
tabs), whereas an overridden `tabSizeHint` only reserves a little more width and leaves
the painting alone.

Nothing here imports analysis/physics — pure Qt presentation (it does use the sibling
`gui.help` / `gui.theme` presentation helpers).
"""

from __future__ import annotations

import math
import re
from typing import Callable, Iterable, Optional, Sequence

from PySide6 import QtCore, QtGui, QtWidgets

from core.data_models import IDENTITY_FORM_BY_FIELD
from core.naming import UNCONFIRMED_SAMPLE
from gui.help import section_header


def confirm_regroup_losses(parent, controller, losses) -> bool:
    """THE one "Replace stored results?" confirm, shared by every surface that
    pre-checks a regroup (Data-tab Update, SLS Update, the sidebar moves, Undo)
    so the same custody event can never ask differently by entry point. True
    when there is nothing to ask or the user proceeds; Cancel is the default
    and the caller applies nothing on it."""
    if not losses:
        return True
    resp = QtWidgets.QMessageBox.question(
        parent, 'Replace stored results?',
        controller.regroup_losses_message(losses),
        QtWidgets.QMessageBox.StandardButton.Yes
        | QtWidgets.QMessageBox.StandardButton.Cancel,
        QtWidgets.QMessageBox.StandardButton.Cancel)
    return resp == QtWidgets.QMessageBox.StandardButton.Yes

# A cell/header is rendered as rich text ONLY when it holds a recognised markup tag —
# never merely because it contains a '<'. This is a safety heuristic: a value like
# "qRg < 1.3" must take the plain path, not be mis-parsed as an (unclosed) tag and mangled.
# Keep this list in step with the tags the label strings actually use.
_HTML_TAG = re.compile(r'</?(?:sub|sup|b|i)>')
from gui.theme import color, ThemedLabel

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


# ---------------------------------------------------------------------------
# Passive load-time notices (the ⓘ channel)
# ---------------------------------------------------------------------------
# The title is a shared string on purpose: the DLS, SLS and trace loaders all render this
# box, and a user who has seen "Loaded with notes" once should recognise it whichever
# button they pressed (ui_style_guide §9 — one name per concept). The loader tests assert
# on it, so a rename is caught rather than diverging per tab.
PARSE_NOTES_TITLE = 'Loaded with notes'


def show_parse_notes(parent: QtWidgets.QWidget, notes: Sequence[str]) -> None:
    """Render passive parser notices as one ⓘ box. No-op when there are none.

    Lives here rather than in `main_window` because the trace loader (Utilities) does not
    go through `_finish_load` and would otherwise need its own copy -- the two-of-three
    wiring that leaves the third loader silently opted out (rule 10)."""
    if not notes:
        return
    QtWidgets.QMessageBox.information(
        parent, PARSE_NOTES_TITLE,
        'The data was loaded and is used as-entered. Notes:\n\n  '
        + '\n  '.join(notes))


# ---------------------------------------------------------------------------
# Empty-state call-to-action (style guide §8, R8.1)
# ---------------------------------------------------------------------------
# The first-launch workspace CTA: it names the two sidebar Load buttons a newcomer must
# find. Reused across the data-scoped tabs' content areas so an empty workspace onboards
# instead of showing full chrome over blank tables/plots (findings 4.1, 4.2).
LOAD_CTA = '← Load data to begin'


class EmptyState(QtWidgets.QWidget):
    """A centred, theme-aware call-to-action shown in a tab's content area (or any empty
    panel) when there is nothing to show yet (style guide §8, R8.1).

    Physics-free presentation only. One shared class, reused with tab-appropriate wording;
    the workspace first-launch CTA is :data:`LOAD_CTA`. The message follows the theme (via
    :class:`gui.theme.ThemedLabel`) so it re-colors on a light/dark switch."""

    def __init__(self, text: str = LOAD_CTA,
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        v = QtWidgets.QVBoxLayout(self)
        v.addStretch(1)
        self._label = ThemedLabel(text, role='muted', size=15)
        self._label.setWordWrap(True)
        self._label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self._label)
        v.addStretch(1)

    def setText(self, text: str) -> None:
        self._label.setText(text)


# ---------------------------------------------------------------------------
# Grouped dropdowns (the section-header idiom)
# ---------------------------------------------------------------------------

# A long dropdown whose entries fall into two kinds reads better as two labelled blocks than
# as one list with a suffix on every second row, where the suffix repeats the same word down
# the column. Qt has no first-class section header for a QComboBox, but a combo populated
# with addItem() is backed by a QStandardItemModel, so an ordinary row becomes one by
# clearing its enabled/selectable flags: Qt paints it in the disabled colour (correct in
# both themes) and keyboard navigation steps straight over it.

def add_combo_section_header(combo: QtWidgets.QComboBox, text: str) -> None:
    """Append a non-selectable, bold section header row to `combo`.

    Only valid on a combo populated through ``addItem`` (so its model is a
    ``QStandardItemModel``) and left NON-editable — an editable combo would let the
    user type the header text as if it were a value.

    Callers must make sure the current index does not land on a header; use
    :func:`first_selectable_index` after populating.
    """
    combo.addItem(text)
    item = combo.model().item(combo.count() - 1)
    item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEnabled
                  & ~QtCore.Qt.ItemFlag.ItemIsSelectable)
    font = item.font()
    font.setBold(True)
    item.setFont(font)


def first_selectable_index(combo: QtWidgets.QComboBox) -> int:
    """Index of the first row a user could actually choose, or -1 if there is none.

    Skips section headers and separators. A freshly populated combo defaults to
    index 0, which is a header whenever the list opens with one, so callers select
    this instead.
    """
    model = combo.model()
    for i in range(combo.count()):
        item = model.item(i)
        if item is not None and item.flags() & QtCore.Qt.ItemFlag.ItemIsSelectable:
            return i
    return -1


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
# Splitter with a visible "grip" handle (usability)
# ---------------------------------------------------------------------------
# Qt's default splitter handle is a near-invisible thin line, so users can't tell a
# divider is draggable. GripSplitter paints a small centered grip (three dots) on every
# handle, giving a consistent "drag me" cue to the controls↔plot divider, the stacked
# plots, and the resizable control columns alike.

class _GripHandle(QtWidgets.QSplitterHandle):
    """A splitter handle that paints three centered grip dots over the default handle.
    Dots run across the handle's short axis (a vertical column for a horizontal splitter,
    a horizontal row for a vertical one). The color follows the theme (palette `Mid`)."""

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
                center = QtCore.QPointF(cx, cy + k * self._DOT_GAP)   # vertical handle
            else:
                center = QtCore.QPointF(cx + k * self._DOT_GAP, cy)   # horizontal handle
            p.drawEllipse(center, self._DOT_RADIUS, self._DOT_RADIUS)
        p.end()

    def changeEvent(self, ev) -> None:
        if ev.type() == QtCore.QEvent.Type.PaletteChange:
            self.update()                    # recolor the dots on a theme switch
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
# Rich-text table cells (subscripts/superscripts in a QTableWidget)
# ---------------------------------------------------------------------------
# A QTableWidgetItem renders plain text only, so `M<sub>w</sub>` / `cm<sup>-1</sup>` show
# their raw tags -- and letters like w/g/c/r have no Unicode subscript, so HTML is the only
# way to render them as true subscripts. This delegate paints a cell's DisplayRole text as
# rich text so those quantities read correctly in a table. Presentation only; apply with
# `view.setItemDelegateForColumn(col, RichTextDelegate(view))`.

class RichTextDelegate(QtWidgets.QStyledItemDelegate):
    """Paint a table/list cell's text as HTML (for M<sub>w</sub>, k<sub>c</sub>, …)."""

    def paint(self, painter, option, index) -> None:
        opt = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        if not _HTML_TAG.search(opt.text):      # plain cell (no markup tag): default path
            super().paint(painter, option, index)
            return
        text = opt.text
        opt.text = ''                           # draw the background/selection ourselves,
        style = opt.widget.style() if opt.widget else QtWidgets.QApplication.style()
        style.drawControl(QtWidgets.QStyle.ControlElement.CE_ItemViewItem, opt, painter,
                          opt.widget)            # then paint the rich text on top
        doc = QtGui.QTextDocument()
        doc.setDefaultFont(opt.font)
        doc.setHtml(text)
        selected = bool(opt.state & QtWidgets.QStyle.StateFlag.State_Selected)
        ctx = QtGui.QAbstractTextDocumentLayout.PaintContext()
        role = (QtGui.QPalette.ColorRole.HighlightedText if selected
                else QtGui.QPalette.ColorRole.Text)
        ctx.palette.setColor(QtGui.QPalette.ColorRole.Text, opt.palette.color(role))
        rect = style.subElementRect(
            QtWidgets.QStyle.SubElement.SE_ItemViewItemText, opt, opt.widget)
        painter.save()
        painter.translate(rect.topLeft())
        painter.translate(0, max(0.0, (rect.height() - doc.size().height()) / 2.0))
        painter.setClipRect(rect.translated(-rect.topLeft()))
        doc.documentLayout().draw(painter, ctx)
        painter.restore()

    def sizeHint(self, option, index):
        opt = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        if not _HTML_TAG.search(opt.text):
            return super().sizeHint(option, index)
        doc = QtGui.QTextDocument()
        doc.setDefaultFont(opt.font)
        doc.setHtml(opt.text)
        return QtCore.QSize(int(doc.idealWidth()) + 4, int(doc.size().height()))


# The inset `RichTextHeaderView` paints its rich text at. It is spent on the LEFT
# only -- `paintSection` translates by it and then clips in the translated frame, so
# the paint budget is `rect.width() - 1 * inset`. The measurement below floors the
# chrome at TWO insets anyway, which is one more than the paint step needs: that is
# the safe direction (4 px of slack, measured), never an under-size.
_RICH_HEADER_INSET_PX = 4

# Width cap, in px, on a table column that holds a MEASUREMENT NAME, paired with
# `Qt.ElideLeft` on the table. One home because it is one job on three tables -- the
# Correlogram and Distribution glance tables and the SLS derived-run table -- all of
# which sit in a narrow control column where an uncapped file stem pushes the values
# it is meant to identify off the edge. Left-elision is the other half: a stem's TAIL
# is what tells six replicates of one set apart, so the head is what may be dropped.
NAME_COL_MAX_W = 150


def sidebar_measurement_order(controller) -> dict:
    """``{item_id: position}`` in the SIDEBAR's order.

    The order `gui/main_window.MainWindow._refresh_sidebar` paints: samples in order,
    then each
    sample's DLS runs, then its solvent reference, its SLS series and any extra
    blanks. Derived from the CONTROLLER rather than read off the tree widget, so a tab
    can reproduce it without reaching into another widget -- and without a tree
    existing at all, which is what makes it testable.

    Two tabs need it for different jobs and must agree: the Summary table orders its
    rows by it (they were sorted as TEXT, which put 100 deg before 50 deg) and the Data
    tab numbers a fanned-out label by it, so a file exported from one matches the row
    read in the other.
    """
    order: dict = {}
    for sample in controller.samples():
        ids = list(sample.dls_item_ids)
        if sample.solvent_reference_item_id:
            ids.append(sample.solvent_reference_item_id)
        ids += list(sample.sls_item_ids)
        ids += list(sample.extra_solvent_reference_item_ids)
        for iid in ids:
            order.setdefault(iid, len(order))
    return order


class RichTextHeaderView(QtWidgets.QHeaderView):
    """A horizontal table header that renders its section labels as HTML, so quantity
    names in a HEADER (R<sub>g</sub>, R<sub>h</sub>, ρ = R<sub>g</sub>/R<sub>h</sub>) show
    their subscripts — a plain QHeaderView paints header text without HTML. The
    RichTextDelegate only reaches data *cells*, so tables that name quantities in the
    header row need this. Apply with `table.setHorizontalHeader(RichTextHeaderView(table))`.
    Presentation only."""

    def __init__(self, parent=None) -> None:
        super().__init__(QtCore.Qt.Orientation.Horizontal, parent)
        self.setSectionsClickable(True)
        self.setHighlightSections(True)

    def _section_html(self, logical_index) -> str:
        """The section's DisplayRole string, '' when there is no model."""
        model = self.model()
        return '' if model is None else str(
            model.headerData(logical_index, QtCore.Qt.Orientation.Horizontal,
                             QtCore.Qt.ItemDataRole.DisplayRole) or '')

    def sectionSizeFromContents(self, logical_index):
        """Size a section from its RENDERED text, not from its markup.

        QHeaderView measures this hint by handing the section's DisplayRole string to
        the style as PLAIN text, so a label of 'ρ = R<sub>g</sub>/R<sub>h</sub>
        (± statistical)' asks for the width of the tags it never draws — roughly twice
        what it paints. Every column sized to its contents then demands far more room
        than the table has, which is what the header row cost the tables using this
        view. Measured here with the same QTextDocument `paintSection` draws with.

        The style's own chrome (the header margins, and the sort indicator when one is
        shown) is kept by subtracting what Qt measured the string as — its hint bolds
        the metrics — from the base hint, floored at the inset the paint step spends so
        an unfamiliar style cannot leave the rendered text clipped."""
        base = super().sectionSizeFromContents(logical_index)
        text = self._section_html(logical_index)
        if not _HTML_TAG.search(text):
            return base
        doc = QtGui.QTextDocument()
        doc.setDefaultFont(self.font())
        doc.setHtml(text)
        bold = QtGui.QFont(self.font())
        bold.setBold(True)
        as_plain = QtGui.QFontMetrics(bold).size(0, text).width()
        chrome = max(2 * _RICH_HEADER_INSET_PX, base.width() - as_plain)
        return QtCore.QSize(math.ceil(doc.idealWidth()) + chrome,
                            max(base.height(), math.ceil(doc.size().height())))

    def paintSection(self, painter, rect, logical_index) -> None:
        if not rect.isValid():
            return
        text = self._section_html(logical_index)
        if not _HTML_TAG.search(text):
            super().paintSection(painter, rect, logical_index)
            return
        # Paint the section chrome (background/border) with no text, then the rich text.
        opt = QtWidgets.QStyleOptionHeader()
        self.initStyleOption(opt)
        opt.rect = rect
        opt.section = logical_index
        opt.text = ''
        painter.save()
        self.style().drawControl(
            QtWidgets.QStyle.ControlElement.CE_Header, opt, painter, self)
        painter.restore()
        doc = QtGui.QTextDocument()
        doc.setDefaultFont(self.font())
        doc.setHtml(text)
        ctx = QtGui.QAbstractTextDocumentLayout.PaintContext()
        ctx.palette.setColor(
            QtGui.QPalette.ColorRole.Text,
            self.palette().color(QtGui.QPalette.ColorRole.ButtonText))
        painter.save()
        painter.translate(rect.left() + _RICH_HEADER_INSET_PX,
                          rect.top() + max(0.0, (rect.height() - doc.size().height()) / 2.0))
        painter.setClipRect(0, 0, rect.width() - _RICH_HEADER_INSET_PX, rect.height())
        doc.documentLayout().draw(painter, ctx)
        painter.restore()


# ---------------------------------------------------------------------------
# Shared measurement-selection layer (SelectionModel + MeasurementPicker)
# ---------------------------------------------------------------------------
# One idiom for "which measurements is this tab analyzing?", used by every analysis tab:
# real checkboxes, INCLUDE-semantics (tick = use it), select-all/none, grouped by sample.
# The model is GUI-side only and not persisted -- the controller stays Qt-free and stateless
# about selection, analysis calls taking the ticked ids inline via `include_ids`. The picker
# is a thin view; two pickers sharing one model stay in lock-step via its `changed` signal.

# Matplotlib's marker codes (`plotting.plots.marker_cycle`) as painted Qt shapes. The
# plot's non-colour channel is the SHAPE, so a swatch that showed colour alone would
# lose exactly what a colour-blind reader relies on (WCAG 1.4.1).
_SWATCH_POLYGONS = {
    's': ((-1, -1), (1, -1), (1, 1), (-1, 1)),
    '^': ((0, -1), (1, 0.8), (-1, 0.8)),
    'v': ((0, 1), (1, -0.8), (-1, -0.8)),
    'D': ((0, -1), (1, 0), (0, 1), (-1, 0)),
    'P': ((-1, -.38), (-.38, -.38), (-.38, -1), (.38, -1), (.38, -.38), (1, -.38),
          (1, .38), (.38, .38), (.38, 1), (-.38, 1), (-.38, .38), (-1, .38)),
}


def marker_swatch(color_hex: Optional[str], marker: Optional[str],
                  size: int = 12) -> QtGui.QIcon:
    """The icon that carries a co-plotted series' identity: its plot COLOUR and its
    plot MARKER SHAPE, so the row beside a name means the same thing as the curve.

    An empty icon when the COLOUR is missing -- an UNTICKED measurement has no colour
    of its own (`SelectionModel.color_for` keys on position in the ticked list), so it
    must not be given one here. A missing or unmapped marker falls to the disc.
    """
    if not color_hex:
        return QtGui.QIcon()
    pm = QtGui.QPixmap(size, size)
    pm.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pm)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    qc = QtGui.QColor(color_hex)
    painter.setBrush(QtGui.QBrush(qc))
    painter.setPen(QtGui.QPen(qc))
    r = (size - 3) / 2.0
    cx = cy = size / 2.0
    poly = _SWATCH_POLYGONS.get(marker or 'o')
    if marker == 'X':
        # A rotated filled plus: one shape definition, two markers.
        painter.translate(cx, cy)
        painter.rotate(45)
        painter.translate(-cx, -cy)
        poly = _SWATCH_POLYGONS['P']
    if poly is None:                          # 'o' and anything unmapped: a disc
        painter.drawEllipse(QtCore.QRectF(cx - r, cy - r, 2 * r, 2 * r))
    else:
        painter.drawPolygon(QtGui.QPolygonF(
            [QtCore.QPointF(cx + x * r, cy + y * r) for x, y in poly]))
    painter.end()
    return QtGui.QIcon(pm)


class SelectionModel(QtCore.QObject):
    """An ordered, unique set of measurement ``item_id`` strings currently ticked for
    analysis. Emits ``changed`` on any mutation so a bound :class:`MeasurementPicker`
    (or the sidebar mirror) re-syncs without manual bookkeeping.

    Generalized from the DLS-only ``_OverlaySelection``. Insertion order is preserved so
    each measurement keeps a stable overlay color (`color_for`) across tabs. Pass the
    plotting color cycle in via ``color_cycle`` when overlay colors are needed (kept
    an argument so this module imports no plotting/physics)."""

    changed = QtCore.Signal()

    def __init__(self, parent: Optional[QtCore.QObject] = None, *,
                 color_cycle=None, marker_cycle=None, linestyle_cycle=None) -> None:
        super().__init__(parent)
        self._ids: list[str] = []                       # ordered, unique
        # Each cycle may be a fixed sequence OR a zero-arg callable returning the current
        # sequence — pass a callable (e.g. plotting.plots.cycle) when the source can change
        # at runtime (a plot-palette switch rebinds the colour cycle), so this stays live
        # instead of freezing a copy. `_resolve` normalises the two forms at read time.
        self._cycle = color_cycle
        # Parallel marker / linestyle cycles (injected so this module imports no
        # plotting) so an overlaid series is distinguishable by SHAPE, not colour alone
        # (WCAG 1.4.1 / greyscale). Keyed by the same insertion index as
        # color_for, so each measurement keeps a stable (colour, marker, linestyle).
        self._markers = marker_cycle
        self._linestyles = linestyle_cycle

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
        """Radio behavior: make ``iid`` the sole ticked id (empty when None/''). Used by
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

    def _index_of(self, iid: str) -> int:
        return self._ids.index(iid) if iid in self._ids else len(self._ids)

    @staticmethod
    def _resolve(cyc):
        """A cycle, whether it was supplied as a fixed sequence or a live callable."""
        return cyc() if callable(cyc) else cyc

    def color_for(self, iid: str) -> Optional[str]:
        """A stable palette color keyed by the id's position, or None if no color cycle
        was supplied. Re-reads the (possibly callable) cycle each call so a live palette
        swap is reflected on the next redraw."""
        cyc = self._resolve(self._cycle)
        return cyc[self._index_of(iid) % len(cyc)] if cyc else None

    def marker_for(self, iid: str) -> Optional[str]:
        """A stable marker SHAPE keyed by the id's position (parallel to color_for), or
        None if no marker cycle was supplied."""
        cyc = self._resolve(self._markers)
        return cyc[self._index_of(iid) % len(cyc)] if cyc else None

    def linestyle_for(self, iid: str):
        """A stable linestyle keyed by the id's position, or None if none supplied."""
        cyc = self._resolve(self._linestyles)
        return cyc[self._index_of(iid) % len(cyc)] if cyc else None


class GroupTickBar(QtWidgets.QWidget):
    """A physics-free "tick all at X" helper row: one *(label, combo, Tick button)* group
    per field. Emits ``tickRequested(key, value)`` when a Tick button is pressed; the
    HOST resolves which measurements share that value and ticks them — this widget owns
    no selection state, so the same bar works over a table-backed selection
    (``_SampleAnalysisTab``) or a :class:`SelectionModel` (the Distribution picker).

    ``fields`` is a sequence of ``(key, noun, formatter)``: ``key`` labels the emitted
    signal, ``noun`` fills the label/tooltips ("Tick all at {noun}:"), and
    ``formatter(value) -> str`` renders each combo item WITH its unit (e.g.
    ``lambda c: f'{c*1000:g} mg/mL'``). The host supplies the distinct raw values per
    field via :meth:`set_values`; the raw value rides the combo item's ``UserRole`` and
    comes back verbatim in ``tickRequested`` (kept a Python object, not restringified)."""

    tickRequested = QtCore.Signal(str, object)

    def __init__(self, fields: Sequence[tuple],
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._combos: dict = {}          # key -> QComboBox
        self._buttons: dict = {}         # key -> QPushButton
        self._formatters: dict = {key: fmt for key, _noun, fmt in fields}
        # One (label, combo, Tick) row PER field, stacked vertically — a single HBox
        # would overflow the narrow control column once there are two fields.
        # A single-field bar is just one row.
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        for key, noun, _fmt in fields:
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(QtWidgets.QLabel(f'Tick all at {noun}:'))
            combo = QtWidgets.QComboBox()
            combo.setToolTip(
                f'Tick every eligible measurement sharing the chosen {noun}.')
            # Grow to its content but don't hog the row — keep the Tick button visible
            # in the narrow control column.
            combo.setSizeAdjustPolicy(
                QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
            self._combos[key] = combo
            row.addWidget(combo)
            btn = QtWidgets.QPushButton('Tick')
            btn.setToolTip(
                f'Add every eligible measurement at the selected {noun} to the '
                'ticked set.')
            btn.clicked.connect(lambda _checked=False, k=key: self._emit(k))
            self._buttons[key] = btn
            row.addWidget(btn)
            row.addStretch(1)
            outer.addLayout(row)

    def set_values(self, key: str, values: Iterable) -> None:
        """Fill one field's combo with the distinct raw ``values`` (formatted via the
        field's formatter). Disables that field's combo + button when there are none."""
        combo = self._combos.get(key)
        if combo is None:
            return
        fmt = self._formatters[key]
        combo.blockSignals(True)
        combo.clear()
        for v in values:
            combo.addItem(fmt(v), v)
        combo.blockSignals(False)
        has = combo.count() > 0
        combo.setEnabled(has)
        self._buttons[key].setEnabled(has)

    def item_texts(self, key: str) -> list:
        """The formatted (unit-bearing) display strings currently in a field's combo —
        e.g. ['1.4 mg/mL', '0.6 mg/mL']. Handy for verifying the unit rendering."""
        combo = self._combos.get(key)
        return [combo.itemText(i) for i in range(combo.count())] if combo else []

    def _emit(self, key: str) -> None:
        combo = self._combos[key]
        if combo.currentIndex() < 0:
            return
        self.tickRequested.emit(key, combo.currentData())


class MeasurementPicker(QtWidgets.QWidget):
    """The shared "which measurements?" picker: a sample-grouped list of **real
    checkboxes** (`Qt.ItemIsUserCheckable`, include-semantics) backed by a
    :class:`SelectionModel`, with a "?" help badge and Select all / none.

    Physics-free: the caller injects ``label_fn(lm) -> str``, ``header_fn(sample) ->
    str`` and (optionally) ``tooltip_fn(lm) -> str`` so the row text, the group text
    and the hover card are the caller's, never this widget's. ``kinds`` filters which
    measurement kinds are eligible. ``single=True`` gives radio behavior (ticking one
    unticks the rest) for the sample-scoped tabs. Emits ``selectionChanged`` whenever
    the ticked set changes (a re-emit of the model's ``changed``).

    A TICKED row carries its plot swatch (:func:`marker_swatch`) -- the colour and
    marker shape that measurement has in the co-plot -- so the list and the plot can
    be read against each other. An unticked row gets none: it has no colour yet."""

    selectionChanged = QtCore.Signal()

    def __init__(self, controller, model: SelectionModel, *,
                 kinds: Sequence[str] = ('dls',),
                 label_fn: Callable[[object], str],
                 header_fn: Callable[[object], str],
                 tooltip_fn: Optional[Callable[[object], str]] = None,
                 title: str = 'Measurements to Plot',
                 help_text: str = '', help_bullets: Optional[Sequence[str]] = None,
                 single: bool = False,
                 group_fields: Sequence[tuple] = (),
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._model = model
        self._kinds = tuple(kinds)
        self._label_fn = label_fn
        self._header_fn = header_fn
        self._tooltip_fn = tooltip_fn
        self._single = single
        self._group_fields = tuple(group_fields)
        self._leaf_by_id: dict = {}          # item_id -> QListWidgetItem (checkable leaves)

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        # Title + "?" badge (how-to-use help is the visible badge).
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

        # Optional bulk group-tick row: "tick all at concentration
        # / angle" over the currently-listed measurements. Emits a (key, value); we
        # resolve the matching leaves and tick them on the shared model.
        self._group_bar = GroupTickBar(self._group_fields) if self._group_fields else None
        if self._group_bar is not None:
            self._group_bar.tickRequested.connect(self._on_group_tick)
            v.addWidget(self._group_bar)

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
            # One string for it, whether the sample is unconfirmed or missing
            # entirely: to the reader both mean "this run's identity is not set",
            # and `core.naming` is where that wording lives.
            hdr = QtWidgets.QListWidgetItem(
                self._header_fn(sample) if sample else UNCONFIRMED_SAMPLE)
            hdr.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)   # header, not checkable
            font = hdr.font(); font.setBold(True); hdr.setFont(font)
            hdr.setForeground(color(self._list, 'marker_group'))
            self._list.addItem(hdr)
            for lm in ms:
                it = QtWidgets.QListWidgetItem('  ' + self._label_fn(lm))
                if self._tooltip_fn is not None:
                    it.setToolTip(self._tooltip_fn(lm))
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
        self._refresh_group_values()

    def _refresh_group_values(self) -> None:
        """Fill each group-tick combo with the distinct committed values among the
        currently-listed leaves (Distribution is cross-sample: the bar groups over
        every measurement on screen, matching the plotted committed data)."""
        if self._group_bar is None:
            return
        for key, _noun, _fmt in self._group_fields:
            vals = sorted({v for iid in self._leaf_by_id
                           if (v := self._group_value(iid, key)) is not None})
            self._group_bar.set_values(key, vals)

    def _group_value(self, iid: str, key: str):
        """The committed value of ``key`` for a leaf (what the plot uses), or None."""
        lm = self._controller.workspace.measurements.get(iid)
        return lm.committed_params.get(key) if lm is not None else None

    def _on_group_tick(self, key: str, value) -> None:
        """Tick every listed leaf whose committed ``key`` matches ``value`` (float-safe),
        adding to the current selection (never unticking). "Same value?" goes through
        the shared identity comparison form for the field (core/data_models.py), so
        the tick can never group differently from the engine's distinctness gates;
        a field with no identity form falls back to exact equality (the value came
        verbatim from the same committed set, so exact match is the safe default)."""
        form = IDENTITY_FORM_BY_FIELD.get(key, lambda v: v)
        try:
            target = form(float(value))
        except (TypeError, ValueError):
            return
        match = []
        for iid in self._leaf_by_id:
            v = self._group_value(iid, key)
            if v is not None and form(float(v)) == target:
                match.append(iid)
        self._model.set_many(match, True)

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
        """Recolor group headers, tint the ticked rows with `marker_selected` for the
        current theme, and put each ticked row's plot swatch beside its name. Named
        `_apply_palette` so `gui.theme.retheme` re-runs it on a theme switch (alongside
        the per-widget PaletteChange below) -- which is also what re-draws the swatches
        after a plot-palette switch."""
        for i in range(self._list.count()):
            it = self._list.item(i)
            iid = it.data(QtCore.Qt.ItemDataRole.UserRole)
            if iid is None:                  # header
                it.setForeground(color(self._list, 'marker_group'))
            elif it.checkState() == QtCore.Qt.CheckState.Checked:
                it.setForeground(color(self._list, 'marker_selected'))
                it.setIcon(marker_swatch(self._model.color_for(iid),
                                         self._model.marker_for(iid)))
            else:
                it.setForeground(QtGui.QBrush())   # reset → follow palette text color
                it.setIcon(QtGui.QIcon())          # unticked: no colour of its own

    def changeEvent(self, ev: QtCore.QEvent) -> None:
        if ev.type() == QtCore.QEvent.Type.PaletteChange:
            self._apply_palette()
        super().changeEvent(ev)


class SampleSelector(QtWidgets.QWidget):
    """A labeled dropdown of SAMPLES for the sample-scoped analysis tabs (SLS, I·sinθ),
    the sample-level analog of :class:`MeasurementPicker`. Lists the samples matching a
    ``predicate`` (e.g. "has SLS data"); emits ``sampleChanged(sample_id)`` (``''`` when
    none) on a user pick.

    The combo is a **mirror/override of the shell's single active sample**,
    not an independent model: a user pick drives the active sample (the shell fans it out to
    every tab + the sidebar); a sidebar focus is pushed back here via
    :meth:`set_current_sample_id` (no emit). When the active sample is one this tab can't
    display, :meth:`show_incompatible` puts a neutral placeholder here while the tab body
    shows a named empty-state.

    ``predicate(sample) -> bool`` filters eligibility; ``label_fn(sample) -> str`` labels
    each row. A "?" help badge carries the how-to-use help."""

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

    def show_incompatible(self, text: str) -> None:
        """Show a neutral, non-selectable placeholder (`text`) for an active sample this
        tab can't display — so the dropdown mirrors "the active sample has no data here"
        instead of a stale, still-eligible pick. Selected WITHOUT emitting;
        cleared by the next :meth:`refresh`. No-op when there are no eligible samples at
        all (the combo already reads "(no eligible samples)"). The combo stays enabled so
        the user can still open it and pick an eligible sample to leave the empty-state."""
        if not self._combo.isEnabled():          # no eligible samples → keep that message
            return
        self._combo.blockSignals(True)
        self._combo.addItem(text, None)
        idx = self._combo.count() - 1
        model = self._combo.model()
        item = model.item(idx) if hasattr(model, 'item') else None
        if item is not None:                     # grey it out in the drop-down list
            item.setEnabled(False)
        self._combo.setCurrentIndex(idx)
        self._combo.blockSignals(False)

    def _on_changed(self, _i: int) -> None:
        self.sampleChanged.emit(self.current_sample_id() or '')


# ---------------------------------------------------------------------------
# Offered-source picker
# ---------------------------------------------------------------------------
# Static, so the help inventory can resolve them (a count spliced into the sentence
# would make the tooltip invisible to that scan for no reader benefit).
_SOURCE_PICKER_TOOLTIP = (
    'Fill this value from one of the available sources. Nothing is written until '
    'you choose, and a value you typed is replaced only because you asked.')
_SOURCE_PICKER_EMPTY_TOOLTIP = (
    'Nothing else offers a value for this input, so there is nothing to choose '
    'between. Type a value instead.')


class SourcePicker(QtWidgets.QWidget):
    """Choose which of several offered sources fills one input.

    THE offer/commit affordance (architecture invariant 3). Several places in the
    program can propose a value for the same input -- a measurement's mean count
    rate may come from a linked count-rate trace, from that trace's outlier-clipped
    mean, or from the instrument file's own scalar -- and the program must offer
    them transparently rather than pick one silently. The shape is always:

        <thing>_offers(consumer_id)      enumerate candidates; WRITE NOTHING
        apply_<thing>_offer(id, source)  adopt one into the WORKING tier
        the choice travels as provenance on whatever is computed from it

    This widget is the user-facing half of that contract and knows nothing about
    what it is picking: it renders a button, asks for an offer, and hands the
    chosen source back. Deliberately general, because more than one input has this
    shape -- a solvent blank chosen from several c = 0 measurements is the same
    question, and should not grow a second, differently-behaved affordance.

    An offer is a mapping with ``source`` (the tag stored as provenance), ``value``
    and ``label`` (what the value is, in words). ``value_format`` renders the value
    for display; the default is str.
    """

    picked = QtCore.Signal(str)      # the chosen offer's `source` tag

    def __init__(self, text: str = 'From…', *, parent=None,
                 title: str = 'Choose a source',
                 prompt: str = 'Fill this value from:') -> None:
        super().__init__(parent)
        self._title = title
        self._prompt = prompt
        self._offers = []
        self._format = str
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.button = QtWidgets.QPushButton(text)
        self.button.clicked.connect(self._on_clicked)
        layout.addWidget(self.button)
        layout.addStretch(1)
        self.set_offers([])

    def set_offers(self, offers, *, value_format=None) -> None:
        """Show the candidates. With none, the button disables and says why."""
        self._offers = list(offers or [])
        if value_format is not None:
            self._format = value_format
        self.button.setEnabled(bool(self._offers))
        self.button.setToolTip(
            _SOURCE_PICKER_TOOLTIP if self._offers
            else _SOURCE_PICKER_EMPTY_TOOLTIP)

    def _entries(self):
        return ['{0} — {1}'.format(self._format(o['value']), o['label'])
                for o in self._offers]

    def _on_clicked(self) -> None:
        if not self._offers:
            return
        # editable=False is load-bearing: the dialog returns the TEXT, so a user who
        # could edit it could return a string matching no offer.
        text, ok = QtWidgets.QInputDialog.getItem(
            self, self._title, self._prompt, self._entries(), 0, False)
        if not ok or not text:
            return
        entries = self._entries()
        if text in entries:
            self.picked.emit(self._offers[entries.index(text)]['source'])
