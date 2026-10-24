"""
The DLS Summary sub-tab: the workspace-wide results tables, per measurement and per sample,
with their detail columns and notes.
"""
from __future__ import annotations

import html
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from app.controller import DLS_DETAIL_SPECS
from core.notes import Note
from core.workspace import is_row_flag
from core.naming import measurement_display_name, measurement_hover_card, sample_display_name
from gui.plot_controls import make_split_panels, pin_under_column
from gui.export_helper import export_to_csv
from gui.field_labels import has_honest_se
from gui.notes_block import NotesBlock, qualifier_words
from gui.widgets import MeasurementPicker, marker_swatch, sidebar_measurement_order
from core.formatting import format_fixed_sig, format_value_at_uncertainty

from gui.dls.tables import _qualified, _RESULT_ROW_TOOLTIPS, _readonly_table
from gui.dls.bases import _DeferredRepaint


#: The Summary tab's distribution-peak column header tooltip. A row saved before the peak's
#: position became its geometric mean reloads as stored, the node (its exported `From` label
#: has no '(geometric mean)').
_SUMMARY_PEAK_TIP = (
    "Each peak's geometric-mean Rh, the average of ln Rh over the peak (for a lognormal "
    "fit, its fitted median), and its share of the intensity. A session saved by an earlier version shows the grid point with the most "
    "weight instead until the fit is run again (Theory-and-Equations-Guide §3.4.1).")


_SUMMARY_PICKER_BULLETS = [
    'Ticks pick which measurements the table shows when '
    '<b>“Ticked only”</b> is on.',
    'With “Ticked only” off, the table lists every DLS result.',
    '<b>“Show fit details”</b> adds every other quantity each parametric fit reports, '
    'a column per method you ran, each naming its fit.',
    'ⓘ on a cell marks how that number was obtained; hover to read it.',
    '⚠ with a letter marks a failed precondition, stated in the line of that letter '
    'under the per-measurement table; <b>click</b> either for what to do.',
    'More: User-Manual §8.8.',
]


# ===========================================================================
# Container: the DLS tab = header + inner sub-tabs
# ===========================================================================

# ===========================================================================
# Summary sub-tab (durable results table over the snapshot store)
# ===========================================================================

# How each sample-level source_kind is labeled in the table's "Source" column.
_SOURCE_LABELS = {
    'replicate_avg': 'Replicate avg',
    'gamma_q2': 'Γ vs q²',
    'conc_extrap': 'D vs c',
    'mode_family': 'Mode family',
}


# The ± column says what kind of ± it holds where it is read (ui_style_guide R5.3: the
# "statistical only" qualifier is never hidden), and a cell adds the estimator's word
# when the classical-OLS opt-in produced it (`_fill_sample`).
_SE_HEADER = '±SE (statistical)'


# The sample-level table's header tooltips: what the From column holds, and what a marker on
# a row's Rh or ± says.
_SAMPLE_FROM_TIP = (
    'Where the value came from: its runs, angles or concentrations and the method.')


_SAMPLE_RH_TIP = (
    'A ⚠ with a letter on a value or its ± marks a failed precondition of it, stated in the '
    'line of the same letter above this table.')


# ONE mark for a cell whose run simply has no such fit. Before this the same state
# rendered two ways in one table -- 'n/a' from `format_fixed_sig(None)` for the
# parametric columns, '—' from `_peak_cell([])` for the distribution ones.
#
# It is NOT a synonym for 'n/a': a stored NaN means the fit RAN and withheld its value
# (owner decision D2), and that still reads 'n/a'. `format_fixed_sig` maps missing and
# non-finite alike to 'n/a', so routing an absent value through it is what destroyed the
# distinction; `_num` keeps it.
_ABSENT = '—'


class _GroupRow(str):
    """A table row that is one SPANNING header, not a row of cells.

    A `str` subclass so the row still carries its text with no wrapper to unpack, and
    so `_replace_table` can tell it from a list of cells by type alone. Used for the
    Summary per-measurement table's sample bands -- the shape the measurement picker
    already uses -- in place of a Sample column repeated down every run.
    """


def _num(value, sig: int) -> str:
    """A Summary number cell: `_ABSENT` when the run has no such fit, else the value
    (which is 'n/a' when the fit ran and withheld it)."""
    return _ABSENT if value is None else format_fixed_sig(value, sig)


def _peak_cell(peaks: List[Tuple[float, float]], sig: int = 3) -> str:
    """Render a distribution method's peaks as 'Rh (Int %); Rh (Int %)'.

    NNLS/CONTIN/lognormal peaks are from an ill-posed inversion — no honest ±
    — so BOTH the Rh and its intensity weight use the one
    no-uncertainty knob (`sig`), never a separate fixed convention."""
    if not peaks:
        return '—'
    return '; '.join(
        f'{format_fixed_sig(rh, sig)} ({format_fixed_sig(frac * 100, sig)}%)'
        if frac is not None else format_fixed_sig(rh, sig)
        for rh, frac in peaks)


#: How deep a cumulant fit must go before a moment is a measurement rather than a blank,
#: keyed by the detail spec's STABLE STORAGE KEY. `_detail_cell` below is the home of what
#: the gate does and why it runs before the value is inspected at all.
_MOMENT_MIN_ORDER = {'mu2': 2, 'mu3': 3, 'mu4': 4}


#: How a detail column names the method it belongs to. Short, because it rides on every
#: detail header and the table is already wide.
_METHOD_SHORT = {'cumulant': 'cumulant', 'single': 'single exp',
                 'double': 'dbl exp', 'kww': 'KWW'}


def _detail_quantity(name: str, unit: str) -> str:
    """The quantity half of a detail header: the name, then its unit in parentheses.

    Deliberately the same spelling `_RESULT_ROWS` uses for the Correlogram panel's row
    labels, so `_RESULT_ROW_TOOLTIPS` keys both surfaces -- the owner-ruled μ₃/μ₄ caution
    is then one string with one home, copied onto this table rather than restated in it.
    `test_every_detail_quantity_is_spelled_as_the_correlogram_panel_spells_it` pins the
    parity for every quantity, not just the two that carry a caution today.
    """
    return f'{name} ({unit})' if unit else name


def _detail_header(name: str, unit: str, method: str) -> str:
    """A detail column's full header: the quantity, then the method that produced it.

    The method is named on EVERY detail column, never only where a name would collide.
    Several parametric arms reuse quantity names -- `baseline` and `converged` on three
    of them, `Γ` on two -- and the table is built to scroll sideways, so the headline
    column that would disambiguate is routinely off-screen when the duplicate is read.
    Measured at 1280 x 760 with all four arms run: 26 columns wanting 1548 px in a 602 px
    viewport, with two `baseline` columns visible at once and neither headline in view.

    Unconditionally rather than on collision, for the reason the measurement names
    already follow: a name that changes according to what else is on screen is a name a
    reader cannot trust or search for. A column means the same thing whatever is beside it.
    """
    return f'{_detail_quantity(name, unit)} · {_METHOD_SHORT.get(method, method)}'


def _detail_cell(value, key: str, details: Dict, sig: int) -> str:
    """One detail value, rendered. The stored value is the fit's own; painting is here.

    Three kinds arrive and each is shown as itself: a bool is the convergence answer
    ('yes'/'no', never 'True'), an int or str is printed as it stands (the order, the
    cumulant method), and every number goes through the no-uncertainty significant-figures
    knob -- these come from a single correlogram, so none carries a ± (invariant 8).

    A missing moment is the interesting case. Below the order that fits it there IS no
    value, and a bare blank -- or worse, a bare 0 -- would read as a fitted result, so the
    cell says which order the fit actually ran at.

    THE ORDER GATE RUNS FIRST, on the order rather than on the value's shape, and this is
    the normative home of that rule. The two absences do not look alike: µ₃ and µ₄ come
    back None below their order, but µ₂ comes back `0.0` -- a real float that is
    structurally absent rather than fitted -- so a gate that only asked "is it None?"
    printed a confident zero for the one moment a reader is most likely to act on, while
    the Correlogram panel's own cell said "unmeasured (order 1)" for the same fit. Asking
    the fit what order it ran at makes the value's shape stop mattering.

    The gate is keyed by storage key (`_MOMENT_MIN_ORDER`), so retitling a column cannot
    quietly disarm it.
    """
    need = _MOMENT_MIN_ORDER.get(key)
    order = details.get('order')
    if need is not None and isinstance(order, int) and order < need:
        return f'unmeasured (order {order})'
    if value is None:
        return _ABSENT
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, (int, str)):
        return str(value)
    return format_fixed_sig(value, sig)


def _with_detail_columns(specs: List, sig: int) -> List:
    """Expand each parametric method's headline into its full set of detail columns.

    The detail columns are read from the controller's one spec, so this function never
    names a quantity: it only decides WHERE the expansion goes -- directly after the
    method's existing headline columns, so a method's quantities stay together under one
    heading rather than being interleaved with another's.
    """
    out = []
    for spec in specs:
        out.append(spec)
        header, _populated, _cell, method, target, _tip = spec
        details = DLS_DETAIL_SPECS.get(method)
        # Only after the method's LAST headline column, so the details follow the whole
        # headline rather than splitting it (the cumulant has two: Rh and PDI).
        if not details or spec is not _last_headline_of(specs, method):
            continue
        for skey, name, _getter, unit, target in details:
            out.append((
                _detail_header(name, unit, method),
                (lambda m: lambda r: bool(r['details'].get(m)))(method),
                (lambda m, k: lambda r: _detail_cell(
                    r['details'].get(m, {}).get(k), k, r['details'].get(m, {}), sig)
                 )(method, skey),
                method, target,
                # The caution is COPIED, keyed by the QUANTITY -- one string, one home
                # (`_RESULT_ROW_TOOLTIPS`), serving this table and the Correlogram panel.
                _RESULT_ROW_TOOLTIPS.get(_detail_quantity(name, unit), '')))
    return out


def _last_headline_of(specs: List, method: str):
    """The final headline column belonging to `method`, or None."""
    owned = [s for s in specs if s[3] == method]
    return owned[-1] if owned else None


def _with_passive_notes(cell, method: str, target, is_headline: bool, shown):
    """Wrap a cell builder so a passive notice reaches the cell it is about.

    The ⓘ is the neutral qualifier tier (ui_style_guide section 5): a passive note says
    how the fit was obtained, never that a precondition failed, so it must not borrow the
    ⚠ channel. This cell's tooltip is the one place on an analysis tab a record reaches.
    Where a cell carries BOTH -- a flagged value that also rests on a bound -- it takes
    the ⓘ in front and the notes block's marker behind, and its tooltip the marker's
    popup and then the notices (`_SummaryTab._replace_table`).
    """
    def build(r):
        base = cell(r)
        notes = (r.get('notes', {}) or {}).get(method, {}) or {}
        mine = list(notes.get(target, ())) if target else []
        if is_headline:
            # Everything with nowhere else to go: whole-fit notices, and any note whose
            # quantity has no visible column.
            for key, sentences in notes.items():
                if not key or key not in shown:
                    mine.extend(sentences)
        if not mine:
            return base
        # The cell may already be (text, tooltip), (text, tooltip, ICON) -- the
        # measurement column carries its plot swatch in a third element -- or carry the
        # notes its marker reads in a fourth. Rebuilding it as a bare pair dropped that
        # icon, so a run that acquired a passive note silently lost its colour swatch: an
        # identity mark removed by an unrelated disclosure. The tail is preserved rather
        # than assumed empty.
        if isinstance(base, str):
            text, tip, tail = base, '', ()
        else:
            text, tip, tail = base[0], base[1], tuple(base[2:])
        joined = '\n'.join(([tip] if tip else []) + mine)
        return (f'ⓘ {text}', joined, *tail)
    return build


class _SummaryTab(_DeferredRepaint, QtWidgets.QWidget):
    """A workspace-wide, durable results table (the DLS Summary view).

    Two stacked tables read the controller's snapshot store: a wide per-measurement
    table (one row per measurement, methods side-by-side) and a compact sample-level
    Rh table (replicate averages + Γ-q² + D-c, with the apparent/thermodynamic
    distinction). Both persist across save/reload because the store does. The left
    panel embeds the SAME shared measurement checklist as the Correlogram and
    Distribution tabs, which here doubles as the table's 'Ticked only' filter."""

    #: A hidden Summary repaints when shown, not at every sidebar click.
    defers_repaint = True

    def __init__(self, controller, selection, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.selection = selection              # shared _OverlaySelection
        self._build_ui()

    def _build_ui(self) -> None:
        splitter, left, right = make_split_panels(self, left_min_width=300,
                                                  sizes=(360, 740))

        self.checklist = MeasurementPicker(
            self.controller, self.selection, kinds=('dls',),
            label_fn=measurement_display_name, header_fn=sample_display_name,
            tooltip_fn=measurement_hover_card, title='Measurements',
            help_text='Tick measurements to filter the Summary table.',
            help_bullets=_SUMMARY_PICKER_BULLETS)
        self.checklist.selectionChanged.connect(self._on_selection_changed)
        left.addWidget(self.checklist, 1)      # the picker takes the column's spare height

        self.ticked_only = QtWidgets.QCheckBox('Ticked only')
        self.ticked_only.setToolTip('Show only the measurements ticked above '
                                    '(otherwise every measurement that has results).')
        self.ticked_only.toggled.connect(self._refresh_tables)
        left.addWidget(self.ticked_only)

        # Off by default: the headline columns are what most readers want, and the full
        # set is wide. The tooltip is the passive nuance tier (how the numbers behave),
        # while the how-to-use sits in the picker's "?" badge above.
        self.show_details = QtWidgets.QCheckBox('Show fit details')
        self.show_details.setToolTip(
            'Add every other quantity each parametric fit reports — the decay rate, the '
            'moments, the fitted order and baseline, the amplitude fractions and the '
            'stretch. These come from a single correlogram, so none carries an '
            'uncertainty; see the Theory-and-Equations-Guide §3.')
        self.show_details.toggled.connect(self._refresh_tables)
        left.addWidget(self.show_details)

        self.export_button = QtWidgets.QPushButton('Export CSV…')
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._on_export)

        self.status = QtWidgets.QLabel('')
        self.status.setWordWrap(True)
        # Export and the status pinned at the column's foot (§3 column template).
        pin_under_column(splitter, self.export_button, self.status)

        right.addWidget(QtWidgets.QLabel('<b>Per-measurement results</b>'))
        self.per_meas_table = _readonly_table(['Sample', 'Measurement'], [])
        right.addWidget(self.per_meas_table, 1)
        # The flagged rows' notes under the table (R8.6): each distinct cause once, the
        # runs counted, a marker on each value it is about; the sample-level rows' after
        # them, their letters continuing, a marker on each row's Rh or ± cell.
        self.notes_block = NotesBlock(what='runs')
        self.notes_block.watch(self.per_meas_table)
        right.addWidget(self.notes_block)
        right.addWidget(QtWidgets.QLabel('<b>Sample-level R<sub>h</sub></b> '
                                         '(averages, Γ–q², D–c)'))
        self.sample_table = _readonly_table(
            ['Sample', 'Source', 'R<sub>h</sub> (nm)', _SE_HEADER, 'R<sub>h</sub> Type', 'From'], [])
        self.notes_block.watch(self.sample_table)
        right.addWidget(self.sample_table, 1)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # Through `set_measurement`, so a repaint a click left owing is paid here and the
        # flush on the sub-tab switch that follows finds none.
        self.set_measurement(self.item_id, self._runnable)

    def focus_state(self, item_id: Optional[str], runnable: bool) -> bool:
        """The state half (`_DeferredRepaint`): nothing, since this workspace-wide tab
        follows no focus; there is always a repaint to make."""
        return True

    def repaint_view(self) -> None:
        """The repaint half: the picker and both tables, from the controller as it is
        now."""
        # Workspace-wide tab: it doesn't depend on the sidebar pick, but a focus still
        # re-reads the workspace, so refresh here too.
        self.checklist.refresh()
        self._refresh_tables()

    @QtCore.Slot()
    def _on_selection_changed(self) -> None:
        """The shared ticks moved, from any co-plot tab: a repaint this tab owes runs
        first, then the tables follow the ticks."""
        self.flush_view()
        self._refresh_tables()

    @QtCore.Slot()
    def _refresh_tables(self) -> None:
        ids = self.selection.ids() if self.ticked_only.isChecked() else None
        meas = self.controller.dls_summary_measurement_rows(ids)
        self._fill_per_meas(meas)
        sample = self.controller.dls_summary_sample_rows()
        self._fill_sample(sample)
        self.export_button.setEnabled(bool(meas) or bool(sample))

    @staticmethod
    def _per_meas_columns(recs: List[Dict], sig: int, show_details: bool = False):
        """[(header, cell_fn, header_tooltip)] for the methods any LISTED run actually has.

        One spec per column, header and cell builder together, so a method cannot show
        up in the headers and not in the cells. Every method is on demand: before this,
        the double-exponential and lognormal columns appeared when populated and the
        other five were always there, so a cumulant-only workspace still paid for four
        columns of `_ABSENT` -- the asymmetry, not the on-demand rule, was the defect.

        With `show_details`, each parametric method that ran expands from its headline to
        the full set its detail spec names. The detail columns are built from that spec
        (`app.controller.DLS_DETAIL_SPECS`), never listed again here, so a quantity
        added there appears without a second edit -- the same one-spec rule that governs
        what the durable row saves.

        Every column carries the METHOD it belongs to and the result ATTRIBUTE it shows,
        which is how a passive note reaches the cell it explains rather than the fit as a
        whole. The headline column of a method is simply its first, and it is where a
        fit-level note -- or one whose target no visible column shows -- lands.
        """
        rh = 'R<sub>h</sub>'

        def flagged(text, value, method: str, r, attrs=None):
            # The cell carries the notes block's marker for the failed preconditions of
            # its method's fit (`flag_notes`, `_fill_per_meas`) about the value it shows
            # -- every one when `attrs` is None, the cell being the fit's one value. A
            # missing value takes no mark -- the house rule.
            notes = tuple(r.get('flag_notes', {}).get(method, ()))
            if not notes or value is None:
                return text
            whole = attrs is None
            if whole:
                attrs = tuple(dict.fromkeys(a for n in notes for a in n.about))
            return (text, '', None, (notes, attrs, whole))

        def peaks(method: str):
            return lambda r: flagged(_peak_cell(r['peaks'].get(method, []), sig),
                                     r['peaks'].get(method) or None, method, r)

        def scalar(key: str, method: str):
            return lambda r: flagged(_num(r[key], sig), r[key], method, r)

        # (header, populated, cell, method, note-target attribute, header tooltip).
        # The tooltip is CARRIED rather than looked up from the finished header: a detail
        # header now names its method too, so the caution's key is the quantity half and
        # splitting the header back apart would be a second spelling to keep in step.
        specs = [
            (f'Cumulant {rh}', lambda r: r['cumulant_rh'] is not None,
             scalar('cumulant_rh', 'cumulant'), 'cumulant', 'rh_nm', ''),
            ('PDI', lambda r: r['pdi'] is not None,
             lambda r: flagged(_num(r['pdi'], sig), r['pdi'], 'cumulant', r, ('pdi',)),
             'cumulant', 'pdi', ''),
            (f'Single {rh}', lambda r: r['single_rh'] is not None,
             scalar('single_rh', 'single'), 'single', 'rh_nm', ''),
            (f'Double-exp ({rh} fast / slow)', lambda r: r['double_fast'] is not None,
             lambda r: flagged(
                 f"{_num(r['double_fast'], sig)} / {_num(r['double_slow'], sig)}"
                 if r['double_fast'] is not None else _ABSENT,
                 r['double_fast'], 'double', r), 'double', None, ''),
            (f'KWW {rh}', lambda r: r['kww_rh'] is not None,
             scalar('kww_rh', 'kww'), 'kww', 'rh_nm', ''),
            (f'NNLS (geometric-mean {rh} / Int %)', lambda r: bool(r['peaks'].get('nnls')),
             peaks('nnls'), 'nnls', None, _SUMMARY_PEAK_TIP),
            (f'CONTIN (geometric-mean {rh} / Int %)',
             lambda r: bool(r['peaks'].get('contin')),
             peaks('contin'), 'contin', None, _SUMMARY_PEAK_TIP),
            (f'Lognormal (geometric-mean {rh} / Int %)',
             lambda r: bool(r['peaks'].get('lognormal')),
             peaks('lognormal'), 'lognormal', None, _SUMMARY_PEAK_TIP),
        ]
        if show_details:
            specs = _with_detail_columns(specs, sig)

        live = [s for s in specs if any(s[1](r) for r in recs)]
        # Which targets a VISIBLE column shows, per method. A note tagged anything else
        # -- '' for a whole-fit notice, or a quantity whose column is not on screen --
        # falls back to that method's headline, so no note is ever dropped for want of a
        # column, and none is guessed onto a neighbour that is not about it.
        shown = {}
        first_col_of = {}
        for idx, (_h, _p, _c, method, target, _tip) in enumerate(live):
            if target:
                shown.setdefault(method, set()).add(target)
            first_col_of.setdefault(method, idx)

        out = []
        for idx, (header, _populated, cell, method, target, tip) in enumerate(live):
            is_headline = first_col_of.get(method) == idx
            out.append((header,
                        _with_passive_notes(cell, method, target, is_headline,
                                            shown.get(method, frozenset())),
                        tip))
        return out

    def _fill_per_meas(self, recs: List[Dict]) -> None:
        # Every number here is from a single correlogram or an ill-posed inversion — no
        # honest ± — so all use the one no-uncertainty knob, never a
        # fabricated uncertainty.
        sig = self.controller.settings.no_uncertainty_sig_figs
        ws = self.controller.workspace.measurements
        order = sidebar_measurement_order(self.controller)
        recs = sorted(recs, key=lambda r: order.get(r.get('item_id') or '', len(order)))
        # Each row's flag notes per method, as `core.notes.Note` (a record given them
        # already keeps its own), then the block over every row shown: each cause once,
        # the runs counted -- before the cells, so each marker's letter is its line's.
        flags = self._flag_notes()
        recs = [r if 'flag_notes' in r else
                {**r, 'flag_notes': flags.get(r.get('item_id') or '', {})} for r in recs]
        self.notes_block.show_notes(
            [(r['measurement_label'], tuple(n for ns in r['flag_notes'].values() for n in ns))
             for r in recs], n_total=len(recs))
        cols = self._per_meas_columns(recs, sig, self.show_details.isChecked())
        headers = ['Measurement', *[h for h, _cell, _tip in cols]]
        # Header tooltips are index-aligned with `headers`, so the Measurement column
        # takes the leading '' rather than the first data column's caution.
        header_tips = ['', *[tip for _h, _cell, tip in cols]]

        rows: list = []
        current_sample = None
        for r in recs:
            # The sample is a GROUP HEADER row rather than a column repeated down every
            # run, the shape the measurement picker already uses. It was the widest
            # column in the table and said the same thing N times.
            if r['sample_label'] != current_sample:
                current_sample = r['sample_label']
                rows.append(_GroupRow(current_sample))
            iid = r.get('item_id') or ''
            lm = ws.get(iid)
            # The Measurement cell carries item 1's hover card, like every other
            # surface that names a run, and the run's plot swatch when it is TICKED.

            # Gated on the ticked set EXPLICITLY, the way `MeasurementPicker`'s painter
            # is. `color_for` does NOT answer None for an unticked id -- `_index_of`
            # returns `len(ids)` for a miss, so every unticked run drew the same real
            # colour, the one the NEXT ticked run will take. A swatch is an identity
            # claim, and with "Ticked only" off this table lists unticked runs.
            ticked = iid in set(self.selection.ids())
            name = (r['measurement_label'],
                    measurement_hover_card(lm) if lm is not None else '',
                    marker_swatch(self.selection.color_for(iid),
                                  self.selection.marker_for(iid))
                    if ticked else QtGui.QIcon())
            rows.append([name, *[cell(r) for _h, cell, _tip in cols]])
        self._replace_table(self.per_meas_table, headers, rows, header_tips,
                            block=self.notes_block)

    def _flag_notes(self) -> Dict[str, Dict[str, Tuple[Note, ...]]]:
        """{item_id: {method: the notes behind its stored row's flag}}: the precondition
        notes the Summary row's flag rests on, read off the stored rows by the one
        predicate the row's flag text and its export use (`core.workspace.is_row_flag`),
        every peak of a distribution fit together, each note once."""
        out: Dict[str, Dict[str, Tuple[Note, ...]]] = {}
        for row in self.controller.workspace.dls_result_rows.values():
            mine = out.setdefault(row.item_id, {})
            mine[row.method] = tuple(dict.fromkeys(
                (*mine.get(row.method, ()), *(n for n in row.notes if is_row_flag(n)))))
        return out

    def _fill_sample(self, recs: List[Dict]) -> None:
        headers = ['Sample', 'Source', 'Rh (nm)', _SE_HEADER, 'Rh Type', 'From']
        sig = self.controller.settings.no_uncertainty_sig_figs
        # The rows' own preconditions (`SampleRhRow.notes`) join the block after the
        # per-measurement lines, their letters continuing, before the cells are marked.
        self.notes_block.append_notes(tuple(n for r in recs for n in r.get('notes') or ()))
        rows = []
        for r in recs:
            se = r['rh_se']
            # With an honest SE, round Rh to the place its σ supports and
            # show the SE at that same place, so value and ± agree (45 ± 1, not 45.2 ± 1.03).
            # Without one, fall back to the documented fixed-sig precision — never a ±.
            # A classical-OLS ± (the Settings opt-in) says so IN the cell: the estimator
            # travels with the number it qualifies (invariant 8 clause A), in the same
            # words the Cross-Sample ρ cell and the export use.
            if has_honest_se(se):
                rh_cell = format_value_at_uncertainty(r['rh_nm'], se)
                # The column says the ± is statistical; a row's estimator is its word,
                # read off the row's fields by the one helper.
                words = qualifier_words(SimpleNamespace(**r), ('rh_se',), whole=False,
                                        exclude=('se_regression',))
                se_cell = _qualified(format_value_at_uncertainty(se, se), words)
            else:
                rh_cell = format_fixed_sig(r['rh_nm'], sig)
                se_cell = '—'
            # The row's verdict, read off its notes, marks the value and the ± each note is
            # about, as the per-measurement table marks its cells (ui_style_guide §5).
            notes = tuple(r.get('notes') or ())
            if notes:
                rh_cell = (rh_cell, '', None, (notes, ('rh_nm',), True))
                se_cell = (se_cell, '', None, (notes, ('rh_se',), False))
            rows.append([
                r['sample_label'],
                _SOURCE_LABELS.get(r['source_kind'], r['source_kind']),
                rh_cell, se_cell,
                r['rh_type_label'], r['from_label'],
            ])
        self._replace_table(self.sample_table, headers, rows,
                            ['', '', _SAMPLE_RH_TIP, '', '', _SAMPLE_FROM_TIP],
                            block=self.notes_block)

    @staticmethod
    def _replace_table(table: QtWidgets.QTableWidget, headers: List[str],
                       rows: List[List[str]],
                       header_tips: Optional[List[str]] = None, *,
                       block: Optional[NotesBlock] = None) -> None:
        """Rebuild `table`. A cell is a plain string, `(text, tooltip)`,
        `(text, tooltip, icon)` or `(text, tooltip, icon, (notes, attrs, whole))`, the
        last marked by `block` (`NotesBlock.mark_item`) with its popup first in the
        tooltip and the cell's own tooltip after it; a ROW may instead be a `_GroupRow`,
        painted as one spanning band.

        The tuple forms exist so a marked cell can explain itself where the mark appears
        (ui_style_guide §5: a soft, always-on interpretive caveat belongs in a passive
        tooltip) and so a run can carry its plot swatch beside its name. Plain strings
        still work unchanged -- `_fill_sample` passes them for every cell but a noted
        row's Rh and ± -- and that back-compatibility is asserted, not assumed.

        `header_tips`, when given, is index-aligned with `headers` and carries a caution
        that belongs to the QUANTITY rather than to any one run's value -- the μ₃/μ₄
        sentence. On the header, not the cells, for the reason `_RESULT_ROW_TOOLTIPS`
        gives: a value cell's tooltip is reserved for that value's own reason, and the
        two must not fight.
        """
        table.clear()
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        for c, tip in enumerate(header_tips or ()):
            item = table.horizontalHeaderItem(c)
            if item is not None and tip:
                item.setToolTip(tip)
        table.setRowCount(len(rows))
        for r, cells in enumerate(rows):
            if isinstance(cells, _GroupRow):
                item = QtWidgets.QTableWidgetItem(str(cells))
                font = item.font()
                font.setBold(True)
                item.setFont(font)
                table.setItem(r, 0, item)
                # One band across the whole width. Re-set every rebuild because
                # `clear()` drops the items but not the spans.
                table.setSpan(r, 0, 1, max(len(headers), 1))
                continue
            table.setSpan(r, 0, 1, 1)          # undo any span this row held before
            for c, cell in enumerate(cells):
                if isinstance(cell, tuple):
                    text = cell[0]
                    tip = cell[1] if len(cell) > 1 else ''
                    icon = cell[2] if len(cell) > 2 else None
                    mark = cell[3] if len(cell) > 3 else None
                else:
                    text, tip, icon, mark = cell, '', None, None
                item = QtWidgets.QTableWidgetItem(text)
                if icon is not None and not icon.isNull():
                    item.setIcon(icon)
                popup = ''
                if mark is not None and block is not None:
                    notes, attrs, whole = mark
                    if block.mark_item(item, notes, attrs, whole=whole):
                        popup = item.toolTip()
                if popup and tip:
                    tip = popup + '<hr>' + html.escape(tip).replace('\n', '<br>')
                if tip or popup:
                    item.setToolTip(tip or popup)
                table.setItem(r, c, item)
        table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents)

    @QtCore.Slot()
    def _on_export(self) -> None:
        ids = self.selection.ids() if self.ticked_only.isChecked() else None

        def do_export(path: str) -> str:
            return self.controller.export_dls_summary(path, ids)

        status = export_to_csv(self, self.controller, 'dls_summary.csv', do_export)
        if status:
            self.status.setText(status)
