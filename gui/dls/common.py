"""
Helpers the DLS sub-tabs share: the stale-fit hints, the weighting titles, the measurement
picker's help and selection names, and the lines naming what a run left out or needs.
"""
from __future__ import annotations

from typing import List, Tuple

from analysis.dls.cumulants import DEFAULT_RECORDED_WEIGHTINGS
from analysis.dls.distributions import DEFAULT_RECORDED_DISTRIBUTION_WEIGHTINGS
from app.fit_requirements import DLS_FIT_REQUIREMENTS
from core.note_registry import REGISTRY
from core.notes import WEIGHTING_PREFIX, Note, qualifiers, render, vocabulary
from core.naming import UNCONFIRMED_SAMPLE, measurement_display_name, sample_display_name
from gui.export_helper import measurement_export_names
from gui.field_labels import missing_input_lines
from gui.widgets import STALE_HINT, STALE_UNRUNNABLE


# Shown when a displayed fit was computed from inputs that have since changed: a
# committed parameter, a tab-owned input (the delay window / baseline / cumulant
# order, the ticked set) or a settings-seeded one (skip_initial_channels, the cumulant
# method, the cumulant fit cutoff, the SE estimator). Each tab's `_run_signature` names
# the inputs its fit reads and is stored with the cached fit; a redisplay compares the
# two -- the SLS tab's `sls_run_signature` shape. Complements the pre-commit '(ran on
# last committed values)' note: this covers the post-commit window, where `is_dirty()`
# has already gone false but the plot still reflects the old inputs.
_STALE_HINT = STALE_HINT


# The same mark for a fit kept on screen while Run is off, so it does not ask for Run
# (ui_style_guide.md § 4, "Unavailable analysis").
_STALE_UNRUNNABLE = STALE_UNRUNNABLE


#: `_recompute(sid, fraction=_FOCUSED)`: fit the focused fraction, unless a deferred
#: Run re-enters with the (sample, fraction) it was pressed on.
_FOCUSED = object()


#: The title over a pooled fit's Results table (Γ vs q², D vs c, a Mode Map family).
_RESULTS_TITLE = 'Results'


def _widen_row_gap(figure, hspace: float = 0.06) -> None:
    """Add a little extra vertical space between stacked subplots so the draggable
    residual grip line sits in clear space and doesn't touch a plot.
    Tolerant of matplotlib layout-engine API differences."""
    try:
        engine = figure.get_layout_engine()
        if engine is not None and hasattr(engine, 'set'):
            engine.set(hspace=hspace)
    except Exception:  # noqa: BLE001 — layout-engine API differences
        pass


# A residual weighting is a word in a DLS results table's title only when it is not the
# program's default (`ui_style_guide.md` § 5, "A residual weighting on screen"): the default
# is silent as it is in the export (`analysis/dls/cumulants.py`, whose sets these are), and
# the classical limit a fit falls to without a committed count rate is named.
_CLASSICAL_WEIGHTING = WEIGHTING_PREFIX + 'biganzoli-classical'


_SILENT_WEIGHTINGS = frozenset(
    WEIGHTING_PREFIX + w
    for w in DEFAULT_RECORDED_WEIGHTINGS | DEFAULT_RECORDED_DISTRIBUTION_WEIGHTINGS
) - {_CLASSICAL_WEIGHTING}


# Every residual-weighting code: a pooled fit's title reads what its points ran under, so no
# row names the token its fit asked for ('biganzoli' runs as its full form or its classical
# limit).
_ALL_WEIGHTINGS = frozenset(code for code in REGISTRY if code.startswith(WEIGHTING_PREFIX))


def _named_weightings(result) -> Tuple[Note, ...]:
    """The residual-weighting qualifiers of `result` a label names: those not silent."""
    return tuple(q for q in qualifiers(result)
                 if q.code.startswith(WEIGHTING_PREFIX) and q.code not in _SILENT_WEIGHTINGS)


def _counted_weightings(per, noun: str) -> Tuple[List[str], List[str]]:
    """`per` holds the named residual weightings of each fit or point: each one's word once,
    "(k of n <noun>)" after one that not all of them ran under, and its clause."""
    words: List[str] = []
    clauses: List[str] = []
    for q in dict.fromkeys(q for notes in per for q in notes):
        k = sum(q in notes for notes in per)
        words.append(vocabulary(q) if k == len(per)
                     else f'{vocabulary(q)} ({k} of {len(per)} {noun})')
        clauses.append(render(q, counted=True))
    return words, clauses


def _weighting_title(results) -> Tuple[List[str], str]:
    """A table of single fits names their residual weighting in its title, not on a run's
    capped name cell, whose tail tells replicates apart: `_counted_weightings` over the
    fits, the clauses joined for the title's tooltip."""
    words, clauses = _counted_weightings([_named_weightings(r) for r in results], 'fits')
    return words, '\n'.join(clauses)


def _point_weightings(forms) -> Tuple[List[str], List[str]]:
    """A pooled fit's residual weightings, read off what each point's fit ran under (one
    recorded form per point, '' for none): `_counted_weightings` over the points."""
    notes = [Note(WEIGHTING_PREFIX + f, (), {'weighting': f}) for f in forms if f]
    return _counted_weightings(
        [(q,) if q.code in REGISTRY and q.code not in _SILENT_WEIGHTINGS else ()
         for q in notes], 'points')


def _title_weightings(label, title: str, forms) -> None:
    """A pooled fit's results title: `title` with the words of `_point_weightings(forms)`
    after a dash, their clauses its tooltip. In the title, not on a row label, which they
    would widen past the pane and push the values out of view."""
    words, clauses = _point_weightings(forms)
    label.setText(f'{title} — {", ".join(words)}' if words else title)
    label.setToolTip('\n'.join(clauses))


# ===========================================================================
# Multi-measurement co-plotting (shared by the Correlogram & Distribution tabs)
# ===========================================================================

def _selection_sample_id(controller, iids):
    """The one sample every ticked id belongs to, or None when the set spans samples.

    ONE definition of "does this selection span samples?", shared by the export-name
    helper below and by the legends' sample prefix (`_sample_prefixes`).
    """
    sids = {controller.sample_id_of(iid) for iid in iids}
    return next(iter(sids)) if len(sids) == 1 else None


def _selection_export_names(controller, items, analysis: str, ext: str = 'csv'):
    """`(dialog name, per-file suffixes)` for a ticked-measurement export.

    `items` is `(item_id, method-or-None)` in fan-out order. The sample token rides
    the name when every ticked measurement belongs to ONE sample and is dropped
    otherwise, where a token would be ill-defined — but the common single-sample
    case must not propose the same filename for every sample (review finding).
    The suffixes are what the fan-out appends to whatever stem the user types; the
    dialog shows the whole name when exactly one measurement is ticked.
    """
    return measurement_export_names(
        controller, _selection_sample_id(controller, [iid for iid, _m in items]),
        items, analysis, ext)


def _sample_label_of(controller, iid) -> str:
    """This measurement's sample name, or `(unconfirmed sample)` when it is in none.

    `Controller.sample_id_of` is declared `Optional[str]` and a measurement in no
    sample is a listable, tickable state -- `MeasurementPicker.refresh` puts one under
    exactly that header. Both DLS legend builders subscripted `workspace.samples[...]`
    with its result, which is `samples[None]` -> KeyError inside a draw, swallowed by
    Qt at the slot boundary and leaving a half-drawn figure. `_sample_prefixes` two
    functions down already guards the same lookup; this is that guard, shared, so the
    two consumers cannot drift from it again.
    """
    sid = controller.sample_id_of(iid)
    sample = controller.workspace.samples.get(sid) if sid else None
    return sample_display_name(sample) if sample is not None else UNCONFIRMED_SAMPLE


def _sample_prefixes(controller, iids) -> dict:
    """Per-id sample prefix for a co-plotted set: the sample's name when the set SPANS
    samples, None when it does not.

    Builds no name of its own -- it decides ONE thing, whether the set spans samples,
    and hands `core.naming` the answer. The predicate is the one
    `_selection_export_names` uses for export names, so a legend and a filename can
    never disagree about it. A single-sample view names its sample once, above.
    """
    iids = list(iids)
    if _selection_sample_id(controller, iids) is not None:
        return {iid: None for iid in iids}
    samples = controller.workspace.samples
    out = {}
    for iid in iids:
        sid = controller.sample_id_of(iid)
        s = samples.get(sid) if sid else None
        out[iid] = sample_display_name(s) if s is not None else None
    return out


# The DLS overlay selection + checklist were promoted to the shared, framework-side
# `SelectionModel` + `MeasurementPicker` in `gui/widgets.py` (used by every analysis
# tab now, with real checkboxes and one visual idiom). `DLSModule` builds one shared
# `SelectionModel(color_cycle=cycle)` so overlay colors stay stable, and the
# Correlogram / Distribution / Summary tabs each embed a `MeasurementPicker` bound to it.
# `core.naming`'s `measurement_display_name` / `sample_display_name` are injected as the
# picker's label callables -- the one builder every surface in the program shares.

# Help shown on the "?" badge of the DLS measurement pickers (how-to-use).
_PICKER_HELP = 'Tick the measurements to analyze and overlay.'


_PICKER_BULLETS = [
    'Ticked measurements are fit and co-plotted together.',
    'Grouped by sample — you can tick across samples to compare.',
    '<b>Select all / none</b> toggles the whole list at once.',
    'Selecting in the Workspace sidebar only navigates; the tick boxes here '
    'decide what is analyzed.',
]


# Bulk "tick all at X" group fields for the DLS measurement pickers:
# concentration shown in mg/mL (stored g/mL), angle in degrees. Shared by
# the Distribution picker (both fields) and the Γ-q²/D-c tabs (one field each). Each is
# (committed-param key, noun, item-formatter-with-unit).
_GROUP_FIELD_CONC = ('concentration_g_per_mL', 'concentration',
                     lambda c: f'{c * 1000:g} mg/mL')


_GROUP_FIELD_ANGLE = ('angle_deg', 'angle', lambda a: f'{a:g}°')


# The cross-sample pickers (Correlogram + Distribution) offer both group-tick fields, so
# their help gains one extra bullet over the bare picker.
_GROUP_PICKER_BULLETS = _PICKER_BULLETS + [
    '<b>Tick all at concentration / angle</b> ticks every listed measurement '
    'sharing the chosen value (concentration in mg/mL, angle in °).',
]


def _left_out_lines(controller, runs_missing, what: str = 'is left out:') -> List[str]:
    """The status-line lines of a sub-tab that fits or extracts the ticked runs one at a
    time: for each `(item_id, missing)`, a ticked run the analysis leaves out, its display
    name and `what`, then one line per input it lacks (`missing_input_lines`). A run the
    workspace no longer holds is skipped. The one wording of the Correlogram,
    Distribution and Mode Map sub-tabs."""
    ws = controller.workspace.measurements
    lines: List[str] = []
    for iid, missing in runs_missing:
        lm = ws.get(iid)
        if lm is None:
            continue
        lines.append(f'{measurement_display_name(lm)} {what}')
        lines.extend(missing_input_lines(missing, counted='ticked'))
    return lines


def _union_missing(key, per_run) -> list:
    """Every input any run lacks for the `DLS_FIT_REQUIREMENTS` entry `key` (`per_run`,
    one `missing` tuple per ticked run), each once, in the order the roster declares
    them: the lines of an entry grayed because no ticked run, each judged on its own,
    can use it."""
    rank = {(r.code, r.phrase, r.param): n
            for n, r in enumerate(DLS_FIT_REQUIREMENTS[key])}
    found = dict.fromkeys(m for missing in per_run for m in missing)
    return sorted(found, key=lambda m: rank.get((m.code, m.phrase, m.param), len(rank)))


def _needs_lines(missing) -> str:
    """One line per missing input, as a grayed entry's tooltip lists them under
    'Needs:'."""
    return '\n'.join(missing_input_lines(missing, counted='ticked'))
