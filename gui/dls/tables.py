"""
The DLS sub-tabs' result tables: the vertical label/value table, its rows and row tooltips,
and the read-only table the DDLS and Summary sub-tabs share.
"""
from __future__ import annotations

from typing import List

from PySide6 import QtWidgets

from core.notes import render, vocabulary
from gui.notes_block import qualifier_tooltip, qualifier_words
from gui.widgets import RichTextDelegate, RichTextHeaderView

from gui.dls.common import _SILENT_WEIGHTINGS


def _vtable(rows: List[str]) -> QtWidgets.QTableWidget:
    """A compact 2-column (parameter | value) table seeded with '—' values."""
    t = QtWidgets.QTableWidget(len(rows), 2)
    t.horizontalHeader().setVisible(False)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
    t.horizontalHeader().setSectionResizeMode(
        0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
    t.horizontalHeader().setSectionResizeMode(
        1, QtWidgets.QHeaderView.ResizeMode.Stretch)
    # Col 0 holds quantity names (R_h, D_r, k_D …); render it as rich text so their
    # subscripts show (a plain cell can't — no Unicode subscript for h/w/g/D).
    t.setItemDelegateForColumn(0, RichTextDelegate(t))
    for r, name in enumerate(rows):
        t.setItem(r, 0, QtWidgets.QTableWidgetItem(name))
        t.setItem(r, 1, QtWidgets.QTableWidgetItem('—'))
    # Cap at the REAL content height (the SLS `_fill_result_table` sizing), never a
    # 22-px-per-row literal: rows paint ~30 px at the shipped font, so the literal
    # clipped the last row of every 5-row table at any pane floor.
    row_h = t.verticalHeader().defaultSectionSize()
    t.setMaximumHeight(row_h * max(len(rows), 1) + 2 * t.frameWidth() + 2)
    return t


def _qualified(label: str, words) -> str:
    """A row or column label with its qualifier words (`gui.notes_block.qualifier_words`)
    inside its parentheses: 'R<sub>h</sub> (nm)' and ['apparent'] read
    'R<sub>h</sub> (nm, apparent)'; a label with none gains them."""
    if not words:
        return label
    joined = ', '.join(words)
    return f'{label[:-1]}, {joined})' if label.endswith(')') else f'{label} ({joined})'


def _label_row(item, label: str, result, attrs, *, whole: bool, extra=(),
               exclude=_SILENT_WEIGHTINGS) -> None:
    """Set a Results row's label cell: `label` with the qualifier words of `result` for
    the row's `attrs` (and the whole fit's when `whole`), its passive tooltip their
    clauses. `extra` are qualifier notes the result records no field for, read off what
    the surface shows (a VH mode family). `exclude` are the codes a row never names: the
    silent weightings, or every weighting where the table's title names them
    (`_title_weightings`). None as `result` restores the bare label."""
    if result is None:
        words, tips = [], []
    else:
        words = qualifier_words(result, attrs, whole=whole, exclude=exclude)
        tips = [t for t in qualifier_tooltip(result, attrs, whole=whole,
                                             exclude=exclude).split('\n') if t]
    words += [vocabulary(n) for n in extra]
    tips += [render(n, counted=True) for n in extra]
    item.setText(_qualified(label, words))
    item.setToolTip('\n'.join(tips))


def _fill_vtable(table, values) -> None:
    """Set the value column (col 1) of a `_vtable` from a list of strings."""
    for r, v in enumerate(values):
        it = table.item(r, 1)
        if it is not None:
            it.setText(v)


def _repoint_vtable(table, rows) -> None:
    """Re-point an existing `_vtable` at a different row set, in place.

    `_vtable` BUILDS a table; this reuses one when the method box changes the rows it
    should show. In place rather than swapping a fresh widget into the layout, so the
    pane's splitter share, its scroll position and every connection survive a method
    change. Re-applies the row-label tooltips and the content height cap exactly as
    `_vtable` does, because both are keyed to the row set that just changed.
    """
    table.setRowCount(len(rows))
    for r, name in enumerate(rows):
        label = QtWidgets.QTableWidgetItem(name)
        # The μ₃/μ₄ rows carry the in-program caution against over-reading a bare higher
        # moment; every other row's tooltip is its own label, for narrow-column elision.
        label.setToolTip(_RESULT_ROW_TOOLTIPS.get(name, name))
        table.setItem(r, 0, label)
        table.setItem(r, 1, QtWidgets.QTableWidgetItem('—'))
    row_h = table.verticalHeader().defaultSectionSize()
    table.setMaximumHeight(row_h * max(len(rows), 1) + 2 * table.frameWidth() + 2)
    # It may also SHRINK to three rows and scroll. A `_vtable` otherwise reports its
    # whole row count as its minimum, which in a short section is taken out of whatever
    # shares the space with it.
    table.setMinimumHeight(row_h * min(max(len(rows), 1), 3) + 2 * table.frameWidth() + 2)


def _reset_vtable(table) -> None:
    """Reset a `_vtable`'s value column to '—'."""
    for r in range(table.rowCount()):
        it = table.item(r, 1)
        if it is not None:
            it.setText('—')


_HIGHER_MOMENT_TOOLTIP = (
    'Fitted parameter, not a measured moment: at best (nonlinear method, broad sample) its '
    'uncertainty is comparable to its own value, and on the linear method or below '
    'PDI ≈ 0.3 it is noise — a 0 is the fit\'s bound, not a measurement. Use it to compare '
    'fits, not to characterize the sample; see the Theory-and-Equations-Guide §3.2.6.')


# Row-LABEL tooltips of the results table, keyed by the row's label so a reordering of
# `_RESULT_ROWS` cannot move a caution onto the wrong quantity. On the label cell, not the
# value cell: a value cell's tooltip carries that value's own reliability REASON (or, with
# nothing to explain, its full text for the elided-column case), and the two must not
# fight. Applied where the labels are BUILT (`_repoint_vtable`, which runs on every method
# change and would otherwise overwrite them); a row not listed here keeps its own label as
# the tooltip, the elided-header convention.
_RESULT_ROW_TOOLTIPS = {'μ₃ (s⁻³)': _HIGHER_MOMENT_TOOLTIP, 'μ₄ (s⁻⁴)': _HIGHER_MOMENT_TOOLTIP}


_RESULT_ROWS = {
    'cumulant': ['Γ (s⁻¹)', 'R<sub>h</sub> (nm, apparent)', 'PDI', 'μ₂ (s⁻²)',
                 'μ₃ (s⁻³)', 'μ₄ (s⁻⁴)', 'order', 'method', 'baseline B'],
    'single':   ['Γ (s⁻¹)', 'R<sub>h</sub> (nm, apparent)', 'β', 'baseline', 'converged'],
    'double':   ['fast R<sub>h</sub> (nm, apparent)', 'fast frac',
                 'slow R<sub>h</sub> (nm, apparent)', 'slow frac', 'baseline', 'converged'],
    'kww':      ['stretch s', 'τ<sub>c</sub> (s)', '⟨τ⟩ (s)',
                 'R<sub>h</sub>(τ<sub>c</sub>) (nm, apparent)',
                 'R<sub>h</sub>(⟨τ⟩) (nm, apparent)', 'baseline', 'converged'],
}


def _readonly_table(headers: List[str], rows: List[List[str]]) -> QtWidgets.QTableWidget:
    """A read-only QTableWidget filled from a header list + a list of string rows."""
    t = QtWidgets.QTableWidget(len(rows), len(headers))
    # Headers may name quantities (R_h …); render them as rich text (a plain header can't).
    t.setHorizontalHeader(RichTextHeaderView(t))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
    for r, cells in enumerate(rows):
        for c, text in enumerate(cells):
            t.setItem(r, c, QtWidgets.QTableWidgetItem(text))
    t.horizontalHeader().setSectionResizeMode(
        QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
    return t
