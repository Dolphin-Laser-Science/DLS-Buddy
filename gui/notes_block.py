"""The notes block: the one place a result surface shows its notes (ui_style_guide §5).

A surface that shows results owns one :class:`NotesBlock`, pinned under the widget that
shows the values (R8.6). It is fed the results the surface shows and renders, for each
distinct precondition code among them, one line under the surface,
``⚠ᵃ PDI 0.914 > 0.3, the cumulant limit (§3.2.1)``, the runs named or counted when the
surface shows several (``core.notes.grouped``, the owner's message rule). The value a note
is about carries the same marker, ``24.2 ⚠ᵃ`` (:meth:`NotesBlock.marker`,
:meth:`NotesBlock.mark_item`). A click on a line or a marked cell opens the popup the help
badge uses, with the clause, the remedy from ``gui/note_text.py`` and the guide section; a
hover opens it too, through the Show-tooltips gate, which a click never obeys.

Nothing here composes a sentence: every word is the registry's clause, the remedy table's
imperative or a qualifier's vocabulary (``core.notes.vocabulary``). A record never reaches
the block (it goes to the Summary cell's tooltip and the export), but for the records of
``SHOWN_RECORDS``, a residual weight the fit fell back to: each is one plain line after the
lettered ones, with no glyph, no letter, no alarm colour and no marker at any value, since
it fails no verdict, and it opens its popup on a click like any line. A qualifier is a word
on a row or column label (:func:`qualifier_words`), never a line. A sentence loaded from an
older session (a ``legacy`` note) is shown as a line, since its kind was never recorded.
"""
from __future__ import annotations

from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from core.note_registry import REGISTRY
from core.notes import (WHERE, WHERE_UNIT, Note, groups, notes_in, qualifiers, render,
                        vocabulary)
from gui.note_text import REMEDIES
from gui.theme import ThemedLabel

# The kinds a block shows as lettered lines; a qualifier never is, a record only below.
SHOWN_KINDS: Tuple[str, ...] = ('precondition', 'legacy')
# The records shown as a plain line, ui_style_guide §5's audience table: a fallback weight.
SHOWN_RECORDS: FrozenSet[str] = frozenset({'weight_fallback_uniform', 'weight_fallback_g1'})
MARKER_GLYPH = '⚠'
# The item data role a marked cell keeps its codes in, for the click popup.
CODES_ROLE = int(QtCore.Qt.ItemDataRole.UserRole) + 41
_SUPERSCRIPT = 'ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ'
GUIDE = 'Theory-and-Equations-Guide'


def letter(index: int) -> str:
    """The superscript letter of the ``index``-th cause on a surface (a, b, ... z, then
    two letters)."""
    n = len(_SUPERSCRIPT)
    if index < n:
        return _SUPERSCRIPT[index]
    return _SUPERSCRIPT[index // n - 1] + _SUPERSCRIPT[index % n]


def shown_notes(source: Any) -> Tuple[Note, ...]:
    """The notes of ``source`` (a result, or a tuple of notes) a block shows as lettered
    lines, the ones a marker can name."""
    return tuple(n for n in notes_in(source) if isinstance(n, Note)
                 and (n.code == 'legacy' or n.code in REGISTRY
                      and REGISTRY[n.code].kind in SHOWN_KINDS))


def shown_records(source: Any) -> Tuple[Note, ...]:
    """The records of ``source`` a block shows as plain lines (``SHOWN_RECORDS``)."""
    return tuple(n for n in notes_in(source)
                 if isinstance(n, Note) and n.code in SHOWN_RECORDS)


def _applies(q: Note, result: Any, attrs: Sequence[str], whole: bool) -> bool:
    """A qualifier speaks to a label showing ``attrs``: it is about one of them (a ± word
    only beside a reported ±, so a withheld one reads no estimator), or, when ``whole``,
    about the whole fit."""
    if not q.about:
        return whole
    shown = set(q.about) & set(attrs)
    return any(not a.endswith('_se') or getattr(result, a, None) is not None for a in shown)


def shown_qualifiers(result: Any, attrs: Sequence[str] = (), *, whole: bool = True,
                     exclude: Iterable[str] = ()) -> List[Note]:
    """The qualifiers of ``result`` a label showing ``attrs`` speaks: each about one of
    them, and, when ``whole``, each about the whole fit, in ``core.notes.qualifiers``'
    order. ``exclude`` drops codes a label states otherwise (a column already headed
    'apparent' for every row)."""
    skip = set(exclude)
    return [q for q in qualifiers(result)
            if q.code not in skip and _applies(q, result, attrs, whole)]


def qualifier_words(result: Any, attrs: Sequence[str] = (), *, whole: bool = True,
                    exclude: Iterable[str] = ()) -> List[str]:
    """The vocabulary of :func:`shown_qualifiers`, each word once."""
    words: List[str] = []
    for q in shown_qualifiers(result, attrs, whole=whole, exclude=exclude):
        word = vocabulary(q)
        if word not in words:
            words.append(word)
    return words


def qualifier_tooltip(result: Any, attrs: Sequence[str] = (), *, whole: bool = True,
                      exclude: Iterable[str] = ()) -> str:
    """The clauses behind :func:`qualifier_words`, one per line with its guide section:
    the passive tooltip of a qualified label."""
    lines: List[str] = []
    for q in shown_qualifiers(result, attrs, whole=whole, exclude=exclude):
        text = render(q, counted=True)
        if text not in lines:
            lines.append(text)
    return '\n'.join(lines)


def _line_key(note: Note) -> tuple:
    """What identifies the line a note is on: its code and its values, never the attributes
    it is about (a relayed copy names another result's field) nor where it was found."""
    return (note.code, tuple(sorted((k, v) for k, v in note.values.items()
                                    if k not in (WHERE, WHERE_UNIT))))


def _escape(text: str) -> str:
    return (text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))


class _Line(ThemedLabel):
    """One clickable line of the block: a failed precondition's on the alarm tier, a
    shown record's (``role='muted'``) on the calm one."""

    def __init__(self, text: str, html: str, parent: Optional[QtWidgets.QWidget] = None,
                 *, role: str = 'error'):
        super().__init__(text, role, parent=parent)
        self._html = html
        self.setWordWrap(True)
        self.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setToolTip(html)        # hover: the gated, passive route

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        # Programmatic show: a click opens the popup whatever the tooltip setting.
        QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), self._html, self)
        super().mousePressEvent(ev)


class NotesBlock(QtWidgets.QWidget):
    """The notes of the results one surface shows: one line per distinct precondition,
    a marker at each value a note is about, the remedy on click; then one plain line per
    distinct shown record (``SHOWN_RECORDS``), which no marker names.

    ``what`` is the noun a counted line names ('runs', 'angles', 'fits')."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, *, what: str = 'runs'):
        super().__init__(parent)
        self._what = what
        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 2, 0, 2)
        self._layout.setSpacing(1)
        # The lettered lines first, the shown records' plain lines after them, so a line
        # appended later keeps its letter's place above every record.
        self._lettered = QtWidgets.QVBoxLayout()
        self._plain = QtWidgets.QVBoxLayout()
        for sub in (self._lettered, self._plain):
            sub.setContentsMargins(0, 0, 0, 0)
            sub.setSpacing(1)
            self._layout.addLayout(sub)
        # One entry per line: its code, its letter, its popup; and the line each note
        # shown is on (copies of one code that differ in a limit are separate lines).
        self._codes: List[str] = []
        self._marks: List[str] = []
        self._html: List[str] = []
        self._lines: List[str] = []
        self._line_of: Dict[Note, int] = {}
        # The shown records' lines: their codes, popups, text and the copies they stand for.
        self._record_codes: List[str] = []
        self._record_html: List[str] = []
        self._record_lines: List[str] = []
        self._record_keys: set = set()
        self._watched: List[QtWidgets.QAbstractItemView] = []
        self.setVisible(False)

    # ---- feeding --------------------------------------------------------------------

    def show_notes(self, sources: Any, *, n_total: Optional[int] = None) -> None:
        """Show the notes of ``sources``: one result (or tuple of notes), or a sequence
        of ``(name, result or notes)`` pairs when the surface shows several. ``n_total``
        is the count a counted line is out of (default: the number of pairs)."""
        pairs = self._pairs(sources)
        self.clear()
        self._add_lines(groups([(name, shown_notes(src)) for name, src in pairs],
                               kinds=SHOWN_KINDS, what=self._what, n_total=n_total))
        self._add_records(groups([(name, shown_records(src)) for name, src in pairs],
                                 kinds=('record',), what=self._what, n_total=n_total))

    def append_notes(self, notes: Any) -> None:
        """Add the lines of ``notes`` (a tuple of notes the surface shows beside what
        :meth:`show_notes` was fed: a controller's relay about everything it shows, or the
        notes of a second table's rows) after the lines shown, their letters continuing,
        so a marker already written keeps its letter. A line that repeats one shown is
        not drawn twice."""
        fresh = tuple(n for n in shown_notes(notes) if _line_key(n) not in self._line_of)
        self._add_lines(groups([('', fresh)], kinds=SHOWN_KINDS, what=self._what))
        records = tuple(n for n in shown_records(notes)
                        if _line_key(n) not in self._record_keys)
        self._add_records(groups([('', records)], kinds=('record',), what=self._what))

    def _add_lines(self, lines) -> None:
        for note, names, members in lines:
            i = len(self._lines)
            mark = letter(i)
            line = f'{MARKER_GLYPH}{mark} {render(note, counted=True)}'
            html = self._popup_html(note, names, line)
            self._codes.append(note.code)
            self._marks.append(mark)
            self._html.append(html)
            self._lines.append(line)
            for member in members:
                self._line_of.setdefault(_line_key(member), i)
            self._lettered.addWidget(_Line(line, html, self))
        self.setVisible(bool(self._lines or self._record_lines))

    def _add_records(self, lines) -> None:
        """The shown records' plain lines: the clause alone, no glyph and no letter, on
        the calm tier; never in ``_line_of``, so no marker can name one."""
        for note, names, members in lines:
            line = render(note, counted=True)
            html = self._popup_html(note, names, line)
            self._record_codes.append(note.code)
            self._record_html.append(html)
            self._record_lines.append(line)
            self._record_keys.update(_line_key(m) for m in members)
            self._plain.addWidget(_Line(line, html, self, role='muted'))
        self.setVisible(bool(self._lines or self._record_lines))

    def clear(self) -> None:
        """Remove every line; markers already written stay with their cells."""
        for sub in (self._lettered, self._plain):
            while sub.count():
                item = sub.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    # Hidden at once: out of the layout, a line waiting for its deletion
                    # would still be drawn where it stood, under the lines that replace it.
                    widget.hide()
                    widget.deleteLater()
        for store in (self._codes, self._marks, self._html, self._lines,
                      self._record_codes, self._record_html, self._record_lines):
            store.clear()
        self._line_of.clear()
        self._record_keys.clear()
        self.setVisible(False)

    @staticmethod
    def _pairs(sources: Any) -> List[Tuple[str, Any]]:
        if sources is None:
            return []
        if isinstance(sources, (list, tuple)) and sources and all(
                isinstance(p, tuple) and len(p) == 2 and isinstance(p[0], str)
                for p in sources):
            return list(sources)
        return [('', sources)]

    def _popup_html(self, note: Note, names: Sequence[str], line: str) -> str:
        spec = REGISTRY.get(note.code)
        parts = [f'<b>{_escape(line)}</b>']
        if len(names) > 1 and any(names):
            parts.append(f'{self._what.capitalize()}: '
                         + _escape(', '.join(n for n in names if n)))
        remedy = REMEDIES.get(note.code)
        if remedy:
            parts.append(f'<i>{_escape(remedy)}</i>')
        if spec is not None and spec.section:
            parts.append(f'{GUIDE if spec.guide == "theory" else "User-Manual"} §{spec.section}')
        return '<br>'.join(parts)

    # ---- markers --------------------------------------------------------------------

    def _lines_for(self, source: Any, attrs: Sequence[str], whole: bool) -> List[int]:
        """The lines of this block that ``source``'s notes put at the value showing
        ``attrs``: each shown note about one of them, and, when ``whole``, each about the
        whole fit."""
        wanted = set(attrs)
        out: List[int] = []
        for n in shown_notes(source):
            i = self._line_of.get(_line_key(n))
            if i is None and self._codes.count(n.code) == 1:
                i = self._codes.index(n.code)     # one line of the code: it is that one
            if i is None or i in out:
                continue
            if (whole and not n.about) or wanted & set(n.about):
                out.append(i)
        return out

    def codes_for(self, source: Any, attrs: Sequence[str] = (), *,
                  whole: bool = True) -> Tuple[str, ...]:
        """The codes of the lines :meth:`marker` would letter, each once."""
        return tuple(dict.fromkeys(self._codes[i]
                                   for i in self._lines_for(source, attrs, whole)))

    def marker(self, source: Any, attrs: Sequence[str] = (), *, whole: bool = True) -> str:
        """``'⚠ᵃ'`` (``'⚠ᵃᵇ'`` for two causes) for the value showing ``attrs`` of
        ``source``, '' when no line on this block is about it."""
        lines = self._lines_for(source, attrs, whole)
        return (MARKER_GLYPH + ''.join(self._marks[i] for i in lines)) if lines else ''

    def popup_for(self, codes: Sequence[str]) -> str:
        """The popup of every line of these codes on this block, one after another."""
        return '<hr>'.join(html for c, html in zip((*self._codes, *self._record_codes),
                                                   (*self._html, *self._record_html),
                                                   strict=True) if c in codes)

    def mark_item(self, item: QtWidgets.QTableWidgetItem, source: Any,
                  attrs: Sequence[str] = (), *, whole: bool = True) -> str:
        """Append the marker to a table cell showing ``attrs`` of ``source``, give it the
        popup as its (gated) tooltip and keep its codes for the click; returns the marker."""
        codes = self.codes_for(source, attrs, whole=whole)
        if not codes:
            item.setData(CODES_ROLE, None)
            return ''
        mark = self.marker(source, attrs, whole=whole)
        text = item.text()
        if not text.endswith(mark):
            item.setText(f'{text} {mark}' if text else mark)
        item.setData(CODES_ROLE, list(codes))
        item.setToolTip(self.popup_for(codes))
        return mark

    def watch(self, view: QtWidgets.QAbstractItemView) -> None:
        """A click on a cell :meth:`mark_item` marked opens its popup."""
        if view in self._watched:
            return
        self._watched.append(view)
        view.clicked.connect(lambda index, v=view: self._clicked(v, index))

    def _clicked(self, view: QtWidgets.QAbstractItemView, index: QtCore.QModelIndex) -> None:
        codes = index.data(CODES_ROLE)
        if codes:
            html = self.popup_for(codes)
            if html:
                QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), html, view)

    # ---- reading (tests, and a surface that relays the lines) ---------------------------

    def lines(self) -> Tuple[str, ...]:
        """The lines shown, in order: the lettered ones, then the shown records'."""
        return (*self._lines, *self._record_lines)

    def codes(self) -> Tuple[str, ...]:
        """The codes of the lines shown, in order (a code twice when its copies differ
        in a limit)."""
        return (*self._codes, *self._record_codes)
