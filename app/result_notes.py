"""
app/result_notes.py
===================

The note, clause and suffix helpers the controller builds a result's labels and rows
from: the scale clauses of a scale-bearing label, the precondition notes behind a
flagged value and the label suffix they render to, the notes a durable copy or a pooled
relay carries beside its value, the left-out and replicate-verdict records, and a
Summary row's flag text and passive lines. None holds or reads the controller's state;
the controller imports them. Qt-free (invariant 5).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

from core.workspace import MeasurementResultRow, is_row_flag
from core.note_registry import REGISTRY as _NOTE_REGISTRY
from core.notes import (SCALE_CODES, WHERE, WHERE_COUNT, WHERE_UNIT, Note, RefusedInputs,
                        find_notes, qualifiers, render, value_range)
from analysis import dls as dls_engine


# The scale gates a scale-bearing label states in its parenthetical (`_scale_clauses`);
# the suffix of a flagged label (`_precondition_clauses`) leaves them to it.
_LABEL_SCALE_CODES = frozenset({'uncalibrated', 'no_standard_index', 'no_dn_dc'})


# The partition's per-mode preconditions (`analysis.sls.mode_point_notes`), which each
# per-mode fit mints for its own mode and `run_mode_static` therefore does not copy.
_PER_MODE_CODES = frozenset({'points_undecayed_used', 'points_low_amplitude_used',
                             'partition_angle_spoiled', 'points_grid_end_used',
                             'partition_angle_grid_end'})


def _scale_clauses(res) -> str:
    """A result label's scale clauses, '' when the scale is known: the clause of each
    `uncalibrated`, `no_standard_index` and `no_dn_dc` note the fit carries
    (`analysis.sls.scale_notes`), once each, in its export form, so the label states the
    cause in the registry's words.

    An Mw or A2 can be on an arbitrary scale for independent reasons -- no calibration
    constant (dR arbitrary), a constant applied with the standard's refractive index blank
    (the (n_s/n_cal)^2 step not formed), or no dn/dc (K = 1, so Kc/dR arbitrary) -- and
    they can hold at once. THE one body for every label that states it: the offered
    CANDIDATE labels and the labels the Run writes onto `SampleResult`, which the
    Cross-Sample source picker shows as the value's provenance. Those two drifted apart
    once, because a label is a COPY of the reason; reading the fit's own notes keeps the
    copy and the cause one computation."""
    texts = [render(n, surface='export') for n in find_notes(res)
             if n.code in _LABEL_SCALE_CODES]
    return ''.join(f'; {t}' for t in dict.fromkeys(texts))


def _precondition_notes(result, flag) -> Tuple[Note, ...]:
    """Each precondition note on `result` that fails `flag` (its registry row's `fails`),
    once per clause it renders to, in order. `flag` is one flag field or a tuple of them, a
    note failing any of them counted once. The scale gates a label states in its
    parenthetical are left out (`_scale_clauses`). THE list behind a label's clause and the
    notes a stored value carries (`_precondition_clauses`, `_value_notes`)."""
    flags = (flag,) if isinstance(flag, str) else tuple(flag)
    kept: Dict[str, Note] = {}
    for n in find_notes(result, kind='precondition'):
        if (n.code not in _LABEL_SCALE_CODES
                and any(f in _NOTE_REGISTRY[n.code].fails for f in flags)):
            kept.setdefault(render(n), n)
    return tuple(kept.values())


def _precondition_clauses(result, flag) -> str:
    """The clauses of `_precondition_notes`, joined by '; '; '' when there is none."""
    return '; '.join(render(n) for n in _precondition_notes(result, flag))


def _value_notes(notes) -> Tuple[Note, ...]:
    """`notes` re-aimed at a candidate's `value`, as `ResultCandidate.notes` carries them;
    a store re-aims them at the sample's field (`set_value_notes`)."""
    return tuple(replace(n, about=('value',)) for n in notes)


def _plotted_scale_notes(res, quantity: str) -> Tuple[Note, ...]:
    """The scale gates of the values `res` plots on a `quantity`-vs-Mw axis pair (the
    `_result_arbitrary_scale` pair: its Mw always, its A2 on the 'a2' axis), lifted onto the
    `ScalingData` field that holds each: `mw`, or `y` for the A2."""
    out = []
    for n in res.scale_notes():
        if 'mw_g_per_mol' in n.about:
            out.append(replace(n, about=('mw',)))
        elif quantity == 'a2' and 'a2_mol_mL_per_g2' in n.about:
            out.append(replace(n, about=('y',)))
    return tuple(out)


def _is_precondition(note: Note) -> bool:
    spec = _NOTE_REGISTRY.get(note.code)
    return spec is not None and spec.kind == 'precondition'


_CLASSICAL_WEIGHT = frozenset({'weighting.biganzoli-classical'})
# The fallback-weight records, how a fit was obtained when its sourced weight could not
# be formed and never a failed precondition: a pooled fit relays its inputs' by these.
_FALLBACK_WEIGHT_CODES = frozenset({'weight_fallback_uniform', 'weight_fallback_g1'})
# What a durable copy (a Summary row) or a pooled relay of a DLS fit carries beside its
# records, per source class: the qualifier codes relayed, and whether its non-scale
# preconditions are. An inversion relays its no-count-rate residual weight, which no label
# of the copy carries; a source whose own flag no relay counts relays its preconditions.
# A class not named here relays its records only.
_PASSIVE_RELAY: Dict[type, Tuple[frozenset, bool]] = {
    dls_engine.CumulantResult: (frozenset(), False),
    dls_engine.SingleExponentialResult: (frozenset(), False),
    dls_engine.DoubleExponentialResult: (frozenset(), False),
    dls_engine.KWWResult: (frozenset(), False),
    dls_engine.DistributionResult: (_CLASSICAL_WEIGHT, True),
    dls_engine.ContinResult: (_CLASSICAL_WEIGHT, False),
    dls_engine.ModeExtraction: (_CLASSICAL_WEIGHT, True),
}


def _passive_note_objects(res) -> Tuple[Note, ...]:
    """The notes a durable copy or a pooled relay of a fit carries beside its value rather
    than as its flag, as `core.notes.Note`: the qualifiers its class relays
    (`_PASSIVE_RELAY`), then its notes in order, each but a precondition, and a non-scale
    precondition too where its class relays them."""
    codes, preconditions = _PASSIVE_RELAY.get(type(res), (frozenset(), False))
    out: List[Note] = [q for q in qualifiers(res) if q.code in codes]
    for n in find_notes(res):
        if _is_precondition(n) and not (preconditions and n.code not in SCALE_CODES):
            continue
        out.append(n)
    return tuple(out)


def _lifted_notes(notes, about: Tuple[str, ...]) -> Tuple[Note, ...]:
    """`notes` as the result relaying them carries them: a note about the source's own
    attributes becomes one about `about` (the relaying result's field it explains), and
    a note about the whole source stays one about the whole result."""
    return tuple(replace(n, about=about) if n.about else n for n in notes)


def _relayed_notes(named, where_unit: str = '') -> Tuple[Note, ...]:
    """The notes of several sources, each distinct (code, values) once, the sources that
    carried it named in its `where` (`core.notes.render` prints them in front).

    `named` is [(source label, its notes)]: a run's name, or a point's coordinate in
    `where_unit`. First-seen order, a source once each; a note's `about` is the union of
    its copies'. A label of None names no source. The shape
    `analysis/dls/angular.py::_relayed_point_notes` relays a pooled fit's points in."""
    merged: Dict[tuple, list] = {}
    for label, notes in named:
        for n in notes:
            base = {k: v for k, v in n.values.items() if k not in (WHERE, WHERE_UNIT)}
            key = (n.code, tuple(sorted(base.items(), key=lambda kv: kv[0])))
            entry = merged.setdefault(key, [n, list(n.about), base, []])
            entry[1].extend(a for a in n.about if a not in entry[1])
            if label is not None and label not in entry[3]:
                entry[3].append(label)
    out = []
    for first, about, base, labels in merged.values():
        values = dict(base)
        if labels:
            values[WHERE] = tuple(labels)
            if where_unit:
                values[WHERE_UNIT] = where_unit
        out.append(replace(first, about=tuple(about), values=values))
    return tuple(out)


# Why a run was left out, by the code of its refusal's first note: the words the
# `run_left_out` record prints for each, its row's `cause` tokens, one per code a
# `RefusedInputs` in the engine carries.
_LEFT_OUT_CAUSES: Dict[str, str] = {
    'window_too_few_channels': 'too few channels in the window',
    'baseline_above_data': 'baseline ≥ the windowed peak',
    'weight_inputs_refused': 'residual weight not formed',
    'no_viscosity': 'no viscosity',
}


def _refusal_causes(exc: BaseException) -> Tuple[Note, ...]:
    """The notes a refusal names its cause by (`RefusedInputs.notes`), () for any other
    failure."""
    return tuple(exc.notes) if isinstance(exc, RefusedInputs) else ()


def _left_out_note(run: str, refusal: Sequence[Note] = ()) -> Note:
    """The `run_left_out` record of a run a fit could not take: its name, and the cause of
    the refusal's first note (`_LEFT_OUT_CAUSES`) beside it when the refusal named one."""
    cause = _LEFT_OUT_CAUSES.get(refusal[0].code) if refusal else None
    if cause is None:
        return Note('run_left_out', (), {'run': run})
    return Note('run_left_out', (), {'run': run, 'cause': cause})


def _counted_causes(flagged) -> Tuple[list, List[Tuple[int, Note]]]:
    """(the flagged fits, [(count, cause)]) over `flagged`, one tuple of precondition notes
    per fit: each cause (a code) once, in first-seen order, with the number of fits naming
    it and its values spanned (`core.notes.value_range`). The grouping behind the owner's
    message rule (`_counted_flag_notes`), in text and as a row's notes alike."""
    fits = [tuple(dict.fromkeys(f)) for f in flagged if f]
    groups: Dict[str, list] = {}               # code -> [fits carrying it, its notes]
    for notes in fits:
        for n in notes:
            groups.setdefault(n.code, [0, []])[1].append(n)
        for code in {n.code for n in notes}:
            groups[code][0] += 1
    causes = []
    for count, notes in groups.values():
        keys = list(dict.fromkeys(k for n in notes for k in n.values))
        causes.append((count, replace(notes[0], values={
            k: value_range([n.values[k] for n in notes if k in n.values]) for k in keys})))
    return fits, causes


def _counted_flag_notes(flagged, n_total: int, what: str) -> str:
    """``<k> of <n_total> <what> (<causes>)`` over `flagged`, one tuple of precondition
    notes per flagged fit, or '' when none is flagged.

    The owner's message rule for a verdict over combined results: each cause once, with
    its count and its value range -- a cause is a code, its value the span of the values
    its fits carry ("PDI 0.35–0.91 > 0.3, the cumulant limit"), and its count is printed
    when the fits are flagged for more than one cause. ONE grouping (`_counted_causes`)
    behind both replicate-averaging arms' verdict and the written average's Table-2 row
    (`_replicate_verdict_notes`), so the two cannot drift."""
    fits, causes = _counted_causes(flagged)
    if not fits:
        return ''
    shown = [f'{count} with {render(cause)}' if len(causes) > 1 else render(cause)
             for count, cause in causes]
    return f"{len(fits)} of {n_total} {what} ({'; '.join(shown)})"


def _replicate_verdict_notes(flagged, n_total: int) -> Tuple[Note, ...]:
    """A written replicate average's verdict as its Table-2 row carries it, () when no fit
    is flagged: the `replicates_flagged` count of the flagged fits out of the `n_total`
    averaged, then each cause once with its values spanned, the fits naming it counted in
    its `where` when there is more than one cause -- `_counted_flag_notes`' message rule as
    notes, in the shape a pooled line's row carries its verdict (a count, then the causes),
    so its labels render it as theirs do (`_row_suffix`). Counted from the fits, never from
    their display names, which repeat."""
    fits, causes = _counted_causes(flagged)
    if not fits:
        return ()
    out = [Note('replicates_flagged', ('rh_nm',), {'k': len(fits), 'n': int(n_total)})]
    for count, cause in causes:
        if len(causes) > 1:
            cause = replace(cause, values={**cause.values, WHERE: WHERE_COUNT.format(
                n=count, total=int(n_total), what='fits')})
        out.append(cause)
    return tuple(out)


def _row_flag_notes(source, about: Optional[Tuple[str, ...]] = None) -> Tuple[Note, ...]:
    """The precondition notes behind a flagged Summary value, () when it is not flagged:
    `source` (a fit, or one mode of it) is flagged when its `rh_reliable` is down or it
    did not succeed. `about`, when given, re-targets them to the field of the fit that
    holds `source` (a double exponential's `mode1` / `mode2`)."""
    unreliable = (not getattr(source, 'rh_reliable', True)
                  or getattr(source, 'success', True) is False)
    if not unreliable:
        return ()
    notes = find_notes(source, kind='precondition')
    return _lifted_notes(notes, about) if about is not None else notes


def _with_where(note: Note, where) -> Note:
    """`note` relayed from `where` (`core.notes.render` prints it in front)."""
    return replace(note, values={**note.values, WHERE: where})


def _row_notes(flag_notes, res) -> Tuple[Note, ...]:
    """A Summary row's notes: the precondition notes behind its flag, then the passive
    notes its fit's own surfaces show (`_passive_note_objects`), each once."""
    passive = tuple(n for n in _passive_note_objects(res) if not _is_precondition(n))
    return tuple(dict.fromkeys((*flag_notes, *passive)))


def _row_flag_text(row) -> str:
    """Why a Summary row's value is flagged, '' when it is not: each note behind its flag
    (`core.workspace.is_row_flag`) rendered, each once, joined by '; '. A Table-1
    `MeasurementResultRow` or a Table-2 `SampleRhRow`, each carrying its verdict as notes."""
    return '; '.join(dict.fromkeys(render(n) for n in row.notes if is_row_flag(n)))


def _row_passive_lines(row: MeasurementResultRow) -> Dict[str, List[str]]:
    """A Summary row's notes that are not behind its flag, rendered, keyed by the
    attribute each is about ('' for the whole fit)."""
    out: Dict[str, List[str]] = {}
    for n in row.notes:
        if not is_row_flag(n):
            out.setdefault(n.about[0] if n.about else '', []).append(render(n))
    return out


def _non_scale_notes(result, flag) -> Tuple[Note, ...]:
    """The notes behind each named flag that is down on `result`: () when every flag named
    is up (or the result carries none of them), else `_precondition_notes` of the down
    flags, which leaves the scale gates to the label's parenthetical. `flag` is one flag
    field or a tuple of them.

    THE one computation behind a flagged label's suffix (`_flag_suffix`), the notes the
    candidate carries (`_flag_notes`) and the SLS
    passive-default withhold, which reads that suffix
    (`Controller._sls_rg_candidates_uncached`), so a label never names a cause the
    selector or the Cross-Sample row ignored, nor the reverse."""
    flags = (flag,) if isinstance(flag, str) else tuple(flag)
    down = tuple(f for f in flags if not getattr(result, f, True))
    return _precondition_notes(result, down) if down else ()


def _non_scale_clauses(result, flag) -> str:
    """The clauses of `_non_scale_notes`, joined by '; '; '' when there is none."""
    return '; '.join(render(n) for n in _non_scale_notes(result, flag))


def _flag_notes(result, flag) -> Tuple[Note, ...]:
    """The notes behind `_flag_suffix(result, flag)`, aimed at the candidate's `value`:
    what the candidate carrying that suffix carries."""
    return _value_notes(_non_scale_notes(result, flag))


def _clause_suffix(clauses: str) -> str:
    """A label's suffix for `clauses`: ' — ' and the clauses, '' when there are none. The
    clause says what the measurement is, never a verdict word (ui_style_guide §5)."""
    return f" — {clauses}" if clauses else ''


def _flag_suffix(result, flag='rh_reliable') -> str:
    """The label suffix a flagged candidate carries into the picker, `SampleResult.rh_label`
    / `rg_label` and the export's source column: the clause of each precondition the fit
    names for that flag (`_non_scale_clauses`), so the reason is the one the DLS / SLS
    tab shows (one wording, one home -- the CONTIN sweep-edge candidate set the shape).
    `flag` names the result's own flag for the quantity offered: `rh_reliable` (the DLS
    tiers, a distribution peak among them), `rg_reliable` (the linear SLS family --
    Debye / Guinier / Zimm / Berry each carry one), or `mw_reliable` / `a2_reliable` on a
    scale-bearing label (an Mw or A2: `_guinier_suffix`, the Zimm / Berry candidates and
    the single-angle ladders), whose scale causes the label's parenthetical states
    (`_scale_clauses`); a tuple of flags states the clauses behind each one that is down,
    once each (a Gamma-vs-q^2 run's `rh_reliable` and its `is_diffusive`). Empty for an
    unflagged result, for a result that carries no such flag, and for a flag down with no
    note naming a cause other than the scale."""
    return _clause_suffix(_non_scale_clauses(result, flag))


# What a Gamma-vs-q^2 candidate's label states: its Rh verdict and its through-origin
# verdict, the latter warning without withholding (a non-diffusive run stays the passive
# default, the allow-but-warn ruling on `_dls_rh_promotable`).
_GAMMA_Q2_FLAGS = ('rh_reliable', 'is_diffusive')


def _rh_row_notes(res, rh_attr: str = 'rh_nm', se_attr: str = 'rh_se') -> Tuple[Note, ...]:
    """The precondition notes a Summary Table-2 row carries from the fit it records
    (`SampleRhRow.notes`): each failing one of the verdicts its Rh states
    (`_GAMMA_Q2_FLAGS`), and each failing no flag that is about the fit's Rh `rh_attr` or
    its ± `se_attr` (a replicate average fitted beside its own sources), re-aimed at the
    row's own `rh_nm` and `rh_se`."""
    out: List[Note] = []
    for n in find_notes(res, kind='precondition'):
        fails = _NOTE_REGISTRY[n.code].fails
        verdict = any(f in fails for f in _GAMMA_Q2_FLAGS)
        if not verdict and fails:
            continue
        on_rh, on_se = verdict or rh_attr in n.about, se_attr in n.about
        if on_rh or on_se:
            about = tuple(a for a, kept in (('rh_nm', on_rh), ('rh_se', on_se)) if kept)
            out.append(replace(n, about=about))
    return tuple(dict.fromkeys(out))


def _row_suffix(row, flags=_GAMMA_Q2_FLAGS) -> str:
    """`_flag_suffix` for a durable Table-2 row, which carries its verdict as notes rather
    than flags (`SampleRhRow.notes`): the clause of each of its precondition notes that
    fails one of `flags`, the suffix a Gamma-vs-q^2 candidate's label carries."""
    return _clause_suffix(_precondition_clauses(row, flags))


def _row_value_notes(row, flags=_GAMMA_Q2_FLAGS) -> Tuple[Note, ...]:
    """The notes behind `_row_suffix(row, flags)`, aimed at a candidate's `value`."""
    return _value_notes(_precondition_notes(row, flags))


def _guinier_suffix(result) -> str:
    """The suffix of a single-concentration Mw candidate, Guinier or Debye: the clause of
    each non-scale precondition its `mw_reliable` fails, which also keeps the value out of
    the passive default (`Controller._sls_rg_candidates_uncached`). A Guinier fit whose
    largest MEASURED q*Rg exceeds the validity limit carries the engine's own regime
    sentence -- the flag used to stop at the SLS panel while the value went on into
    rho = Rg/Rh. A fit with no extractable Rg is NOT such a fit: its intercept Mw is
    reliable and defaults normally (owner-ruled 2026-08-27), which is why this reads
    `mw_reliable` and not the regime flag.
    Empty for a fit inside the regime. The wording is the fit's own note's clause, which
    names the limit the verdict actually read (`qrg_max_valid`, echoed on the result), so
    no surface can print a bound the predicate did not use. A Debye fit, read by the same
    candidate loop, names its non-scale causes here as the Guinier and the Zimm labels name
    theirs, a design that is ill-conditioned or a measurement past the qRg limit among them
    (one treatment of the Mw candidates)."""
    # Reads `mw_reliable`, the Mw verdict itself -- not `guinier_valid`, which is also
    # False when no Rg could be extracted, a state that does NOT flag the intercept
    # (owner-ruled 2026-08-27; analysis/sls.py's guinier block names the reasoning). The
    # scale causes travel in the label's parenthetical (`_scale_clauses`), so a flag down
    # for them alone adds no suffix (`_flag_suffix`).
    return _flag_suffix(result, 'mw_reliable')
