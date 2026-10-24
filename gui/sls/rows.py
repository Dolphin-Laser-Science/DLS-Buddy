"""
The SLS result table's rows: the value, diagnostic and shape-solution rows and the
qualifier words on their labels.
"""
from __future__ import annotations

import dataclasses
import html
import math
from typing import Any, List, Optional, Sequence, Tuple

from PySide6 import QtWidgets

from core.notes import Note, vocabulary
from gui.field_labels import has_honest_se
from gui.notes_block import qualifier_tooltip, qualifier_words
from core.formatting import format_fixed_sig, format_pm

from gui.sls.common import _scaled_value


def _has_pm(value, se) -> bool:
    """True when `format_pm` renders a ± for ``value`` and ``se`` (its own test), so a
    label names the ± words only beside a ± on screen."""
    return value is not None and math.isfinite(value) and has_honest_se(se)


# The ± qualifiers whose clause is the word (no short label): said once, on the caption
# above the Results table, rather than on every ± row, where they pushed the values out of
# a column at the shipped window size. The short words stay on the rows.
_CAPTION_CODES: Tuple[str, ...] = ('se_statistical', 'se_excludes_inversion')


_CAPTION_WORDS = frozenset(vocabulary(Note(code)) for code in _CAPTION_CODES)


def _qualified(base: str, unit: str, res, attrs: Sequence[str], *,
               exclude: Sequence[str] = ()) -> Tuple[str, str, Tuple[str, ...]]:
    """A row label with ``res``'s qualifier words for the attributes it shows,
    ``base (unit, word, word)``, the clauses behind the words (its passive tooltip), and
    the caption words (``_CAPTION_CODES``) the row's ± takes instead of its label."""
    words = qualifier_words(res, attrs, exclude=exclude)
    caption = tuple(w for w in words if w in _CAPTION_WORDS)
    words = [w for w in words if w not in _CAPTION_WORDS]
    if '<' in base:        # a rich-text label: the words are text, never markup
        words = [html.escape(w, quote=False) for w in words]
    inner = ', '.join(([unit] if unit else []) + words)
    label = f'{base} ({inner})' if inner else base
    return label, qualifier_tooltip(res, attrs, exclude=exclude), caption


@dataclasses.dataclass
class _Row:
    """One Results-table row: its label and value, and what marks the value.

    ``source`` is the result (or tuple of notes) whose notes mark the value cell, None
    for a cell no note is about (an input, a dashed value); ``attrs`` are the attributes
    the cell shows. ``flag`` names the cell's own verdict: down with no note to mark it
    (a silent condition), the cell keeps its bare ⚠."""
    label: str
    value: str
    source: Any = None
    attrs: Tuple[str, ...] = ()
    whole: bool = True
    flag: Optional[str] = None
    label_tip: str = ''
    caption: Tuple[str, ...] = ()


def _value_row(base: str, unit: str, text: str, res, attr: str,
               se_attr: Optional[str] = None, *, dashed: bool = False,
               flag: Optional[str] = None, mark: Sequence[str] = (),
               exclude: Sequence[str] = ()) -> _Row:
    """The row of one fitted value: its label qualified for what the cell shows (the ±
    words only beside a ± on screen), the value marked by the notes about ``attr`` (or
    ``mark``). A ``dashed`` value carries no ± and no marker: the dash names its cause."""
    shown: Tuple[str, ...] = (attr,)
    if (se_attr and not dashed
            and _has_pm(getattr(res, attr, None), getattr(res, se_attr, None))):
        shown += (se_attr,)
    label, tip, caption = _qualified(base, unit, res, shown, exclude=exclude)
    return _Row(label, text, None if dashed else res, tuple(mark) or (attr,),
                flag=flag, label_tip=tip, caption=caption)


def _append_label_tip(table: QtWidgets.QTableWidget, rows: Sequence[int], text: str) -> None:
    """Append calculation nuance (rule 8's tooltip tier) to the passive tooltip of the label
    cells of `rows`, after the clauses their qualifier words already carry."""
    for r in rows:
        item = table.item(r, 0)
        if item is not None:
            tip = item.toolTip()
            item.setToolTip(f'{tip}\n\n{text}' if tip else text)


def _diagnostic_row(label: str, text: str, res, attr: str) -> _Row:
    """A fit diagnostic (qRg(max)): no qualifier words, marked only by a note about it."""
    return _Row(label, text, res, (attr,), whole=False)


def _shape_solution_tail(tag: str, r, sig: int, *, plural: bool, apparent: bool,
                         dashed: bool) -> List[_Row]:
    """The rows closing one solution's block in a multi-solution form-factor display
    (ellipsoid, core-shell): Mw, A₂ (thermodynamic only), then qRg(max) and R².

    qRg(max) and R² are PER-SOLUTION values (each solution has its own Rg and its own
    fit quality), so with several solutions each block carries its own `tag`-labelled
    pair: an unlabeled trailing pair would show whichever solution lists first, and the
    list order is a presentation convention, not fit preference. With one solution the
    pair is unlabeled."""
    rows = [_value_row(f'{tag}: M<sub>w</sub>', 'g/mol',
                       _scaled_value(r, format_pm(r.mw_g_per_mol, r.mw_se)),
                       r, 'mw_g_per_mol', 'mw_se', dashed=dashed,
                       exclude=('degenerate',))]
    if not apparent:
        rows.append(_value_row(
            f'{tag}: A₂', 'mol·mL/g²',
            _scaled_value(r, format_pm(r.a2_mol_mL_per_g2, r.a2_se)),
            r, 'a2_mol_mL_per_g2', 'a2_se', dashed=dashed, exclude=('degenerate',)))
    prefix = f'{tag}: ' if plural else ''
    rows.append(_diagnostic_row(f'{prefix}qR<sub>g</sub>(max)',
                                format_fixed_sig(r.qrg_max, sig), r, 'qrg_max'))
    rows.append(_Row(f'{prefix}R²', format_fixed_sig(r.r_squared, sig)))
    return rows
