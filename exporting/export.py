"""
exporting/export.py
===================

Origin-compatible CSV export for the analysis result objects produced by the
DLS and SLS modules.

Every analysis output can be written to a CSV that OriginLab imports directly,
with the four standard column-label rows it understands:

    Long Name, Units, Comments, Parameters

followed by the data rows. Columns may have different lengths (scalar results are
written as length-1 columns); shorter columns are padded with empty cells, which
Origin reads as missing values. The Parameters row carries per-column metadata --
for a wide-format Zimm table, for example, it encodes each column's concentration.

This is an I/O module: its functions write files and return the path written. It
contains no analysis logic; it only serializes result objects produced elsewhere.

Design notes
------------
- The low-level writer is `write_origin_csv(path, columns)`, taking a list of
  `OriginColumn(long_name, units, comments, data, parameter)`. Every per-result
  exporter builds a list of these and calls it, so the format lives in one place.
- Scalar results (Mw, Rg, A2, fit settings, calibration flags) are written as
  length-1 columns with their own Long Name and Units, so they import alongside
  the data with full labeling rather than as an opaque comment block.
- NaN/None are written as empty cells (Origin's missing-value convention).
- **One source, one column.** An export whose content traces to exactly ONE
  measurement carries a one-cell `Source file` column holding that measurement's
  bare file name (never a path, never the internal id) -- so a file pulled out of
  a batch still says which measurement it is. An export built from SEVERAL
  measurements (a Zimm set, a Gamma-q^2 fit, a replicate average) carries no such
  column: its sources are named in the Comments cells instead, leaving the data
  block's shape untouched. A per-measurement export OF a replicate average is a
  many-source export by that rule: the cell is empty and the contributors ride the
  Comments. `_source_file_column` is where the cell is built and the roster of
  which exporter is which is pinned in `tests/test_export_source_column_family.py`.
- Files are UTF-8 **with a byte-order mark**, so Excel decodes the degree sign
  and the Greek a Comments cell carries instead of reading them in the system
  code page. Read one back with `utf-8-sig`.

Change history
--------------
2026-06-13  export.py v1: core writer + exporters for the DLS result objects
            (cumulant/single/double/KWW correlogram fits, distributions,
            Gamma-q^2, concentration extrapolation) and SLS result objects
            (excess Rayleigh ratio, Debye, Zimm/Berry, calibration-free A2).
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np

# The cumulant's recorded weightings and its silent default set, imported rather than
# restated: which recorded forms the default request resolves to is the engine's to say.
# No cycle -- `analysis/` never imports `exporting/`.
from analysis.dls.cumulants import DEFAULT_RECORDED_WEIGHTINGS, RECORDED_WEIGHTINGS
# The distribution family's silent set, imported for the same reason; and the one scan a
# peak's size verdict comes from, so the per-fit peak cells state the lobe's own verdict
# rather than re-deriving it.
from analysis.dls.distributions import (DEFAULT_RECORDED_DISTRIBUTION_WEIGHTINGS,
                                        RECORDED_DISTRIBUTION_WEIGHTINGS,
                                        find_distribution_peaks)
# Which Gamma a mode point's derived quantities read, for the same reason: one home.
from analysis.dls.modes import plotted_gamma_s_inv
# A note's words are the registry's (`core/note_registry.py`); the export renders them
# and never mints one (`comment_for`).
from core.note_registry import REGISTRY
from core.notes import (SCALE_CODES, WEIGHTING_PREFIX, Note, has_note, notes_in, qualifiers,
                        render)


# ===========================================================================
# Core Origin-format writer
# ===========================================================================

@dataclass
class OriginColumn:
    """One column of an Origin-format CSV.

    long_name : str   -> Origin "Long Name" row (column title; axis label)
    units     : str   -> Origin "Units" row
    comments  : str   -> Origin "Comments" row
    data      : sequence -> the column values (any length; scalars use length 1)
    parameter : str   -> Origin "Parameters" row (per-column metadata, e.g. the
                         concentration of a wide-format Zimm column). Default ''.
    """
    long_name: str
    units: str
    comments: str
    data: Sequence[Any]
    parameter: str = ''


def _fmt(value: Any) -> str:
    """Format a single cell. Non-finite numbers become empty (Origin missing).

    A boolean renders 'yes' / 'no', never blank: `None` and a non-finite number
    already render as the empty cell, so a False flag written as '' would be
    indistinguishable from a flag that was never computed -- and a validity column
    (PDI valid, Guinier valid, Is diffusive) reports its failure precisely there.
    """
    if value is None:
        return ''
    # bool is an int subclass, so this branch must precede the int branch.
    if isinstance(value, (bool, np.bool_)):
        return 'yes' if value else 'no'
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return ''
        return f'{float(value):.10g}'
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def write_origin_csv(
    file_path: str,
    columns: Sequence[OriginColumn],
    delimiter: str = ',',
) -> str:
    """Write columns to an Origin-compatible CSV with the four label rows.

    Layout:
        row 1: Long Name
        row 2: Units
        row 3: Comments
        row 4: Parameters
        rows 5+: data (columns padded to equal length with empty cells)

    Parameters
    ----------
    file_path : str
        Destination path.
    columns : sequence of OriginColumn
    delimiter : str
        Column delimiter, ',' (default) or '\\t'.

    Returns
    -------
    str
        The path written.

    Raises
    ------
    ValueError
        If no columns are supplied.
    """
    if not columns:
        raise ValueError("write_origin_csv requires at least one column.")

    n_rows = max(len(c.data) for c in columns)
    # UTF-8 so non-ASCII cell content (e.g. "Γ vs q²", "°") writes on any platform
    # -- on Windows the default codec is cp1252 and would raise. WITH the byte-order
    # mark ('utf-8-sig'): Excel reads a mark-less CSV in the system code page, so a
    # degree sign or a Greek letter opened straight from disk came out as mojibake.
    # Origin is EXPECTED to skip the mark; that half is unverified, so it is stated as
    # an expectation. A reader that decodes as UTF-8 without expecting the mark sees a
    # stray U+FEFF on the first cell only -- read such a file back with 'utf-8-sig',
    # which also handles a file written before the mark.
    with open(file_path, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh, delimiter=delimiter)
        w.writerow([c.long_name for c in columns])
        w.writerow([c.units for c in columns])
        w.writerow([c.comments for c in columns])
        w.writerow([c.parameter for c in columns])
        for i in range(n_rows):
            row = []
            for c in columns:
                row.append(_fmt(c.data[i]) if i < len(c.data) else '')
            w.writerow(row)
    return file_path


def _scalar_column(long_name: str, units: str, value: Any,
                   comments: str = '') -> OriginColumn:
    """A length-1 column carrying a single scalar result."""
    return OriginColumn(long_name, units, comments, [value])


def _bare_file_name(value: Any) -> str:
    """A path's last component, whichever separator wrote it; '' for nothing.

    Both separators, because a session written on one platform is opened on
    another and `os.path.basename` splits only the one its host uses. Applied
    where the cell is WRITTEN, so the no-directory contract holds however the
    caller spelled the path it handed over.
    """
    return str(value or '').replace('\\', '/').rsplit('/', 1)[-1]


def _source_file_column(source_file: Any, note: str = '') -> OriginColumn:
    """The one-cell `Source file` column a single-measurement export carries.

    The bare file NAME, never a path and never the internal id: a reader holding
    one file out of a batch needs to know which measurement it is, and a path
    buries that in the part every sibling file shares.

    The cell is EMPTY when the measurement has no file of its own -- a replicate
    average is computed FROM measurements, not loaded -- and its contributors ride
    the Comments cell instead, the locked convention for provenance that has no
    column of its own (no extra rows, so the Origin import is unaffected).
    """
    bare = _bare_file_name(source_file)
    base = ('data file this export was computed from' if bare else
            'computed measurement: no source file of its own')
    return OriginColumn('Source file', '', _join_comments(base, note), [bare])


# ===========================================================================
# Comments cells: the clauses a column carries
# ===========================================================================
# A caveat in a Comments cell is a registry clause in its export form, read off the
# result's notes and qualifiers by code and `about`, never matched by wording. This
# module keeps two kinds of text of its own: what a column IS (its description), and the
# provenance strings a result carries, relayed as they stand.

def _registry_clause(code: str, **counts: int) -> str:
    """The export clause of a registry row, for a fact read off a typed field of a result
    that carries no note for it (a Rayleigh series' scale, a stand-in sample time, how many
    mode points sit on the Rh grid's end): `counts` fills a row whose placeholders are
    counts, printed as `render` prints an int. The words are the registry's; nothing is
    minted here."""
    return REGISTRY[code].clause.format(**{k: int(v) for k, v in counts.items()})


def _named(code: str, qualify: Sequence[str]) -> bool:
    """Is the qualifier `code` among `qualify`, by its code or a prefix ending in '.'?"""
    return any(code == q or (q.endswith('.') and code.startswith(q)) for q in qualify)


def _meets(note: Note, column_attrs: Sequence[str], whole: bool) -> bool:
    """Does `note` belong in the Comments cell of a column showing `column_attrs`?

    Yes when its `about` names one of the column's attributes; when the column carries
    the fit-wide notes (`whole`) and the note is about the whole fit (`about == ()`) or
    is a record of how the fit was obtained, which bears on every value the fit produced
    whatever parameter it names; and, for a precondition, when the column is gated by a
    verdict flag the precondition fails (a flag field among `column_attrs` that is in its
    registry row's `fails`)."""
    attrs = set(column_attrs)
    spec = REGISTRY.get(note.code)
    if whole and (not note.about or (spec is not None and spec.kind == 'record')):
        return True
    if attrs & set(note.about):
        return True
    return (spec is not None and spec.kind == 'precondition'
            and bool(attrs & set(spec.fails)))


def column_notes(result, column_attrs: Sequence[str] = (), *, whole: bool = False,
                 codes: Sequence[str] = (), qualify: Sequence[str] = ()) -> list:
    """The notes and qualifiers of `result` that a column's Comments cell carries.

    In order: every note of the result that `_meets` places on the column or whose code
    is in `codes` (a placement the note's `about` does not express), then every
    qualifier `core.notes.qualifiers` reads off the result whose code `qualify` names and
    which is about the column or the whole fit. A sentence a caller appended to a
    result's notes is about the whole fit."""
    out = []
    for note in notes_in(getattr(result, 'notes', ())):
        if isinstance(note, Note):
            if note.code in codes or _meets(note, column_attrs, whole):
                out.append(note)
        elif whole and str(note).strip():
            out.append(note)
    attrs = set(column_attrs)
    out += [q for q in qualifiers(result)
            if _named(q.code, qualify) and (not q.about or attrs & set(q.about))]
    return out


def comment_for(result, column_attrs: Sequence[str] = (), *, whole: bool = False,
                codes: Sequence[str] = (), qualify: Sequence[str] = ()) -> str:
    """The clauses `result` writes into the Comments cell of a column showing
    `column_attrs`: every note and qualifier `column_notes` places there, rendered for
    the export (`core.notes.render`, its values at three significant figures, no guide
    section), each once, joined by '; '; '' when there is none.

    `column_attrs` names the result's attributes the column shows, and the verdict flag
    that gates it where it has one; `whole` says the column carries the fit-wide notes
    (`_meets`); `codes` and `qualify` are described at `column_notes`."""
    texts = [render(n, surface='export') if isinstance(n, Note) else str(n).strip()
             for n in column_notes(result, column_attrs, whole=whole, codes=codes,
                                   qualify=qualify)]
    return _join_comments(*dict.fromkeys(texts))


def _flag_unexplained(result, flag: str) -> bool:
    """`flag` is down on `result` and none of its precondition notes fails it: a
    condition no note names yet, which the caller states with its own fallback."""
    value = getattr(result, flag, True)
    if value is None or bool(value):
        return False
    return not any(isinstance(n, Note) and n.code in REGISTRY
                   and flag in REGISTRY[n.code].fails
                   for n in notes_in(getattr(result, 'notes', ())))


#: The qualifier a value column names by default: a per-mode fit's value is one mode's
#: share of the intensity, never the sample's (invariant 7). Only a mode-resolved fit
#: mints it, so every other column is unchanged.
_VALUE_QUALIFIERS = ('mode_resolved',)


def _value_comment(result, column_attrs: Sequence[str], *, flag: str = '',
                   fallback: str = '', whole: bool = False,
                   codes: Sequence[str] = (),
                   qualify: Sequence[str] = _VALUE_QUALIFIERS) -> str:
    """A value column's clauses: `comment_for` over the column's attributes and the
    verdict `flag` gating it, then `fallback` when that flag is down and no note says
    why (a distribution result marked unconverged with no note: its values are the seed
    estimate)."""
    attrs = tuple(column_attrs) + ((flag,) if flag else ())
    return _join_comments(
        comment_for(result, attrs, whole=whole, codes=codes, qualify=qualify),
        fallback if flag and fallback and _flag_unexplained(result, flag) else '')


#: The qualifiers a ± column names: the statistical label of a scale-bearing fit's ±, a
#: per-mode fit's further exclusion of the inversion's systematics, and the opt-in
#: estimator (invariant 8). HC3 is the silent default.
_SE_QUALIFIERS = ('se_statistical', 'se_excludes_inversion', 'se_ols')


def _se_comment(result, column_attrs: Sequence[str], *descriptions: str,
                flag: str = '', fallback: str = '', whole: bool = False,
                qualify: Sequence[str] = _SE_QUALIFIERS) -> str:
    """A ± column's Comments cell: the clauses of the value it is the ± of (so a ± never
    reads cleaner than its number), the ± qualifiers, then the column's descriptions.
    `column_attrs` names the value and its ± field; joined, never `or`-ed, because the
    value's marker and the estimator are independent facts."""
    return _join_comments(
        _value_comment(result, column_attrs, flag=flag, fallback=fallback, whole=whole,
                       qualify=qualify),
        *descriptions)


# The recorded residual weightings a Comments cell names, by the family that records
# them; every other recorded form is what the program's default request resolves to and
# is silent, like 'calibrated' and HC3 (each default set's comment is the rule's home).
# The distribution family also names the default's infinite-count-rate form, because a
# distribution fitted without a committed mean count rate says so where its weights
# travel; the cumulant records that form in its `Cumulant weighting` value column only.
_CUMULANT_NAMED_WEIGHTINGS = tuple(sorted(
    WEIGHTING_PREFIX + w for w in RECORDED_WEIGHTINGS - DEFAULT_RECORDED_WEIGHTINGS))
_DISTRIBUTION_NAMED_WEIGHTINGS = tuple(sorted(
    WEIGHTING_PREFIX + w for w in (RECORDED_DISTRIBUTION_WEIGHTINGS
                                   - DEFAULT_RECORDED_DISTRIBUTION_WEIGHTINGS)
    | {'biganzoli-classical'}))


def _sample_time_comment(result) -> str:
    """The stand-in sample time a sourced weight was evaluated on, read off the typed
    `sample_time_provenance` ('axis'; silent for 'instrument', '' and a result without
    the field), in the registry's words: a number weighted on an inferred quantity never
    travels unlabelled."""
    return (_registry_clause('sample_time_from_axis')
            if getattr(result, 'sample_time_provenance', '') == 'axis' else '')


# ===========================================================================
# DLS exporters
# ===========================================================================

def export_dls_summary(records: Sequence[Dict[str, Any]], file_path: str,
                       delimiter: str = ',') -> str:
    """Write the DLS Summary store as a LONG/tidy CSV: one row per result or
    distribution peak (the wide on-screen table is purely a display; this is the
    machine-friendly shape). Numeric quantities get their own columns so Origin
    can plot them; text columns carry the provenance. `records` is a list of dicts
    with the keys built by Controller.export_dls_summary."""
    def col(key):
        return [r.get(key) for r in records]

    # An Rh flagged unreliable -- for ANY of its causes, not just the size range -- is
    # disclosed in the AFFECTED COLUMN's Comments cell, the locked export convention an
    # uncalibrated result also uses. This table is long/tidy, so a Comments cell is per
    # COLUMN and the note is an aggregate; rows are NAMED, capped at three, because a
    # flagged cell is not blank and `Method` does not tell flagged from healthy.
    # Aggregated per VALUE COLUMN, not once for `Rh`: a double-exponential row leaves `rh`
    # empty and carries its numbers in `rh_fast`/`rh_slow`, so one note on `Rh` annotated a
    # blank cell while the flagged `Rh_slow` exported bare.
    def _flag_comment(base: str, value_key: str) -> str:
        flagged = [r for r in records
                   if r.get('rh_note') and r.get(value_key) is not None]
        if not flagged:
            return base
        named = '; '.join(
            ' '.join(str(r.get(k) or '') for k in ('sample', 'measurement')).strip()
            or '(unnamed row)'
            for r in flagged[:3])
        if len(flagged) > 3:
            named += f' and {len(flagged) - 3} more'
        return (f'{base}; {len(flagged)} of {len(records)} rows carry a reliability flag '
                f'({named}) -- see the DLS panel or the per-fit export for each reason')

    rh_comment = _flag_comment('hydrodynamic radius', 'rh')
    # What each fit was run on, as values (the rule's home: `core.workspace.
    # MeasurementResultRow`'s docstring): the cumulant's amplitude cutoff, the leading
    # channels skipped and the residual weighting, read off the row's recorded fit inputs
    # (`Controller.export_dls_summary`). Blank where a method has none or a record predates
    # them, never a default the fit did not read.
    blank = (' (blank on sample-level rows, where the method has none, and on a row saved '
             'before fits recorded them)')
    fit_inputs = [
        OriginColumn('Fit cutoff', '', "the cumulant's amplitude cutoff, a fraction of the "
                     'peak g2-1 that bounds the retained range' + blank, col('fit_cutoff')),
        OriginColumn('Skipped channels', '', 'leading correlator channels dropped before '
                     'the fit' + blank, col('n_skipped')),
        OriginColumn('Weighting', '', 'residual weighting the fit ran under, its recorded '
                     'form' + blank, col('weighting')),
    ]

    columns = [
        OriginColumn('Sample', '', 'sample identity', col('sample')),
        OriginColumn('Measurement', '', 'name, with angle / concentration / fraction (blank '
                     'for sample-level rows)', col('measurement')),
        OriginColumn('Method', '', 'fit method or sample-level source', col('method')),
        OriginColumn('Rh', 'nm', rh_comment, col('rh')),
        OriginColumn('Rh_SE', 'nm', 'statistical SE (blank when none is honest)',
                     col('rh_se')),
        OriginColumn('SE estimator', '', 'covariance estimator behind Rh_SE for a '
                     'regression source; blank = HC3 default or a non-regression SE',
                     ['classical OLS' if r.get('se_estimator') == 'ols' else ''
                      for r in records]),
        OriginColumn('PDI', '', 'cumulant polydispersity index', col('pdi')),
        OriginColumn('Intensity fraction', '%', 'intensity-weighted peak area '
                     '(NOT mass/weight percent)', col('int_pct')),
        OriginColumn('Rh_fast', 'nm',
                     _flag_comment('double-exponential fast mode', 'rh_fast'),
                     col('rh_fast')),
        OriginColumn('Rh_slow', 'nm',
                     _flag_comment('double-exponential slow mode', 'rh_slow'),
                     col('rh_slow')),
        OriginColumn('Rh_Type', '', 'apparent / thermodynamic', col('rh_type')),
        OriginColumn('is_average', '', 'replicate-averaged sample Rh',
                     col('is_average')),
        OriginColumn('From', '', 'contributing measurements / method', col('from')),
        # Beside `From`, the two provenance columns together: which file a
        # per-measurement row came from, and which measurements a sample-level row
        # was built from. Blank on every row that is not one loaded measurement --
        # a sample-level row, and a replicate average, which has no file of its own.
        OriginColumn('Source file', '',
                     'data file this row was loaded from (blank for sample-level '
                     'and computed rows -- see From)',
                     [_bare_file_name(r.get('source_file')) for r in records]),
        # Last, so no column an Origin template already reads moves.
        *fit_inputs,
    ]
    return write_origin_csv(file_path, columns, delimiter)

def export_correlogram_fit(measurement, result, file_path: str,
                           delimiter: str = ',', *, source_file: str = '',
                           source_note: str = '') -> str:
    """Export a parametric DLS fit (cumulant / single / double / KWW).

    Writes the fitted delay times, the measured g2-1 over that window, the model
    curve, and the residuals, followed by the fit's scalar outputs as labeled
    length-1 columns. Works for any result object exposing fit_tau_s,
    fitted_g2m1, residuals (all four parametric DLS results do).

    Parameters
    ----------
    measurement : DLSMeasurement
        The source measurement (used to recover the measured g2-1 over the fit
        window). It carries no provenance -- the engine object is built from the
        committed parameters alone -- so the two `source_*` fields below are what
        the controller hands over.
    result : CumulantResult | SingleExponentialResult | DoubleExponentialResult |
             KWWResult
    file_path : str
    source_file : str
        The measurement's file name for the one-source `Source file` column.
    source_note : str
        What to say in that column's Comments cell when there is no file (a
        replicate average's contributors).

    Returns
    -------
    str
    """
    def _mode_comment(mode) -> str:
        """One ExponentialMode's Rh clauses, or '' when it is not flagged.

        Read off the MODE rather than the result: the result-level flag is the roll-up
        over its modes, so using it here would stamp one mode's reason onto the other
        mode's columns.

        Placement differs BY CAUSE, which is why the mode states its cause rather than
        leaving the exporter to infer one from the note text. A missing viscosity leaves
        the rate exact and only withholds the radius, so its note rides the Rh column
        alone; an undecayed mode's RATE is itself an extrapolation, so its note also rides
        Gamma and D (`_rate_comment`) -- matching export_mode_map, which puts the same
        verdict on Gamma and D_app.
        """
        return comment_for(mode, ('rh_nm', 'rh_reliable'))

    def _rate_comment(fit, *rate_attrs: str) -> str:
        """The clauses about a fit's rate columns, and its `decay_incomplete` note, whose
        `about` names the size only: an undecayed rate is itself an extrapolation."""
        return comment_for(fit, rate_attrs, codes=('decay_incomplete',))

    def _baseline_comment(result) -> str:
        """What the baseline column holds: the nonlinear cumulant's FITTED floating B,
        or the offset the caller pinned and the fit subtracted. A zero pinned offset is
        the silent default, like 'calibrated' -- the Comments cell speaks only when a
        baseline actually changed the numbers."""
        if getattr(result, 'method', None) == 'nonlinear':
            return ('fitted floating baseline (Theory-and-Equations-Guide §3.1.2)'
                    if getattr(result, 'success', True) else '')
        return ('pinned offset from the baseline region, subtracted before the fit'
                if getattr(result, 'baseline', 0.0) else '')

    # The fit's records (dead channels, a floating baseline resting on its bound, a moment
    # on its bound) and its notes about the whole fit ride every scalar the fit produced,
    # whatever parameter a record names (`whole`): one optimisation over one channel set
    # produced them all, so a note that changes how one number is read bears on the
    # others, and cannot be separated from them on the way out.
    def _fit(*attrs: str, codes: Sequence[str] = ()) -> str:
        """The clauses of one of the fit's own scalar columns: the notes about it, those
        `codes` add, the fit-wide notes and the fit's records."""
        return comment_for(result, attrs, whole=True, codes=codes)

    fit_note = _fit()
    tau = np.asarray(result.fit_tau_s, dtype=float)
    fit = np.asarray(result.fitted_g2m1, dtype=float)
    resid = np.asarray(result.residuals, dtype=float)
    data = fit + resid   # measured g2-1 over the fitted window

    columns = [
        OriginColumn('Delay time', 's', 'tau', tau),
        OriginColumn('g2-1 (data)', '', 'measured', data),
        OriginColumn('g2-1 (fit)', '', 'model', fit),
        OriginColumn('Residual', '', 'data - fit', resid),
    ]

    # Scalar outputs, dispatched by the fields each result type carries.
    cls = type(result).__name__
    # The amplitude cutoff the fit ran at travels as a value, in the cumulant's `Fit cutoff`
    # column, never as a sentence: the user owns the setting, and what a fit used is
    # disclosed as values (the rule's home: `core.workspace.MeasurementResultRow`'s
    # docstring). Which residual weighting built the fit, and whether its variance was
    # evaluated on a stand-in sample time, move every scalar below, so they ride each one
    # (silent on the sourced default, `_CUMULANT_NAMED_WEIGHTINGS`).
    weighting_comment = (
        _join_comments(comment_for(result, qualify=_CUMULANT_NAMED_WEIGHTINGS),
                       _sample_time_comment(result))
        if cls == 'CumulantResult' else '')
    # `fit_note` rides this one too, like every other scalar the fit produced: a dead-channel
    # or bound-active-baseline note is material on the number decision D3's shared support
    # exists to make comparable between the two arms.
    columns.append(_scalar_column('RMS error', '', result.rms_error,
                                  _join_comments(_fit('rms_error'), weighting_comment)))
    if cls == 'CumulantResult':
        def _moment(value, name: str, needed_order: int):
            """An order-gated moment as `(value, comment)` for its scalar column.

            Below the order that fits it the value is WITHHELD as well as commented --
            `_fmt` writes None and NaN alike as the empty cell -- so the blank is never
            misread on import as a failed fit or a measured zero. The order-1 PDI is the
            original case (no mu2 term, the Rh fully reliable) and mu3/mu4 follow it.

            The withholding is not redundant with the callers' own values: mu3 and mu4
            come back None below their order and so blank themselves, but mu2 comes back
            a real 0.0 -- structurally absent, not fitted -- so without this it exported a
            confident zero beside a PDI cell that correctly said "unmeasured", and a
            consumer reading the CSV has only those two columns to go on.
            """
            if result.order < needed_order:
                return None, (f'unmeasured: order-{result.order} cumulant has no '
                              f'{name} term')
            return value, ''

        def _scalar_note(*attrs: str, rate: bool = False) -> str:
            """A cumulant scalar's clauses: the notes about it (a rate column also takes
            the undecayed note), the fit-wide notes and the weighting."""
            return _join_comments(_fit(*attrs, codes=('decay_incomplete',) if rate else ()),
                                  weighting_comment)

        pdi_comment = _moment(result.pdi, 'mu2', 2)[1]
        mu2_val, mu2_comment = _moment(result.mu2_s_inv2, 'mu2', 2)
        # mu3 / mu4 travel BARE -- a value column and no companion verdict column (the
        # ruling and its reason are on `CumulantResult`). A failed fit carries None and
        # exports the empty cell with no reason, like every other NaN of that row. Row
        # count is unchanged (a new scalar column, per the locked export decision).
        mu3_val, mu3_comment = _moment(result.mu3_s_inv3, 'mu3', 3)
        mu4_val, mu4_comment = _moment(result.mu4_s_inv4, 'mu4', 4)
        columns += [
            _scalar_column('Cumulant order', '', result.order),
            _scalar_column('Cumulant method', '', getattr(result, 'method', 'linear')),
            _scalar_column('Fit cutoff', '', result.fit_cutoff),
            _scalar_column('Gamma', '1/s', result.gamma_s_inv,
                           _scalar_note('gamma_s_inv', rate=True)),
            # An out-of-range size travels WITH the number, in the affected column's
            # Comments cell only -- the same treatment an uncalibrated result gets, and for
            # the same reason: a flag that can change how a value is read must never be
            # separable from the value on the way out.
            _scalar_column('Rh', 'nm', result.rh_nm, _scalar_note('rh_nm', 'rh_reliable')),
            _scalar_column('D', 'm^2/s', result.d_m2_s, _scalar_note('d_m2_s', rate=True)),
            # The fit-wide note rides EVERY scalar the fit produced, not only the four
            # the size is read from: on the nonlinear arm B, beta, gamma, mu2, mu3 and mu4
            # come out of ONE `curve_fit` over one channel set, and on the linear arm out of one
            # log-polynomial regression, so a dead channel or a bound-active baseline bears
            # on the PDI exactly as it bears on Gamma. PDI is the more consequential of the
            # two: it is the cell a reader uses to judge whether the size is usable at all.

            # The weighting rides every scalar the fit PRODUCED, derived from the column
            # list rather than from the quantities a change happens to be about. Not every
            # column in this block: `Cumulant order`, `Cumulant method` and `Fit cutoff`
            # record what was ASKED FOR rather than what was measured, so they carry no
            # comment.
            _scalar_column('PDI', '', result.pdi,
                           _join_comments(pdi_comment, _scalar_note('pdi'))),
            _scalar_column('PDI valid', '', result.pdi_valid,
                           _join_comments(pdi_comment, _scalar_note('pdi'))),
            _scalar_column('mu2', '1/s^2', mu2_val,
                           _join_comments(mu2_comment, _scalar_note('mu2_s_inv2'))),
            _scalar_column('mu3', '1/s^3', mu3_val,
                           _join_comments(mu3_comment, _scalar_note('mu3_s_inv3'))),
            _scalar_column('mu4', '1/s^4', mu4_val,
                           _join_comments(mu4_comment, _scalar_note('mu4_s_inv4'))),
            _scalar_column('beta', '', result.beta, _scalar_note('beta')),
            _scalar_column('baseline B', '', getattr(result, 'baseline', 0.0),
                           _join_comments(_baseline_comment(result),
                                          _scalar_note('baseline'))),
            # The residual weighting the fit RAN UNDER, as a value -- 'biganzoli-full' when a
            # committed count rate entered the variance, 'biganzoli-classical' when none did,
            # or the opted-in form -- the same request-provenance shape as `Cumulant method`
            # above, and the one surface on which a run fitted without its instrument's
            # count rate says so, since the Comments cells stay silent on the default
            # (owner-ruled 2026-09-08, round 4 finding 41: a per-run fact, never a
            # warning). Last in the block, so no earlier column's position moves.
            _scalar_column('Cumulant weighting', '', getattr(result, 'weighting', '')),
        ]
    elif cls == 'SingleExponentialResult':
        # Same treatment as the cumulant Rh above, read off the MODE: the verdict belongs
        # to the mode, and a fit's modes can disagree (see the double branch below).
        mode = result.mode
        # `fit_note` (the fit-wide notes: today the dead-channel record) rides every
        # scalar this fit produced, exactly as on the cumulant branch above.
        columns += [
            _scalar_column('Gamma', '1/s', mode.gamma_s_inv,
                           _join_comments(_rate_comment(mode, 'gamma_s_inv'), fit_note)),
            _scalar_column('Rh', 'nm', mode.rh_nm,
                           _join_comments(_mode_comment(mode), fit_note)),
            _scalar_column('D', 'm^2/s', mode.d_m2_s,
                           _join_comments(_rate_comment(mode, 'd_m2_s'), fit_note)),
            _scalar_column('beta', '', result.beta, fit_note),
            _scalar_column('baseline', '', getattr(result, 'baseline', 0.0),
                           _join_comments(_baseline_comment(result), fit_note)),
        ]
    elif cls == 'DoubleExponentialResult':
        # PER MODE, not one comment across both. The routine real case is a measured fast
        # mode beside a slow one that never decayed; a shared comment either condemns the
        # good column or clears the bad one, and a flag that can change how a value is read
        # must never be separable from -- or misattached to -- the value on the way out.
        fast, slow = result.mode1, result.mode2
        # The per-mode verdict stays per mode; the fit-wide `fit_note` rides every scalar.
        columns += [
            _scalar_column('Gamma (fast)', '1/s', fast.gamma_s_inv,
                           _join_comments(_rate_comment(fast, 'gamma_s_inv'), fit_note)),
            _scalar_column('Rh (fast)', 'nm', fast.rh_nm,
                           _join_comments(_mode_comment(fast), fit_note)),
            _scalar_column('Amplitude (fast)', '', fast.amplitude_fraction,
                           _join_comments(comment_for(fast, ('amplitude_fraction',)),
                                          fit_note)),
            _scalar_column('Gamma (slow)', '1/s', slow.gamma_s_inv,
                           _join_comments(_rate_comment(slow, 'gamma_s_inv'), fit_note)),
            _scalar_column('Rh (slow)', 'nm', slow.rh_nm,
                           _join_comments(_mode_comment(slow), fit_note)),
            _scalar_column('Amplitude (slow)', '', slow.amplitude_fraction,
                           _join_comments(comment_for(slow, ('amplitude_fraction',)),
                                          fit_note)),
            _scalar_column('beta', '', result.beta, fit_note),
            _scalar_column('baseline', '', getattr(result, 'baseline', 0.0),
                           _join_comments(_baseline_comment(result), fit_note)),
        ]
    elif cls == 'KWWResult':
        # One flag covers both Rh channels: a KWW has one decay, reported two ways. The
        # relaxation times take the caveat only when the DECAY is what failed -- their
        # undecayed note names them; a missing viscosity leaves tau_c and <tau> exact.
        columns += [
            _scalar_column('tau_c', 's', result.tau_c_s, _fit('tau_c_s')),
            _scalar_column('Stretch', '', result.stretch, _fit('stretch')),
            _scalar_column('mean tau', 's', result.mean_tau_s, _fit('mean_tau_s')),
            _scalar_column('Rh (from <tau>)', 'nm', result.rh_from_mean_tau_nm,
                           _fit('rh_from_mean_tau_nm', 'rh_reliable')),
            _scalar_column('Rh (from tau_c)', 'nm', result.rh_from_tau_c_nm,
                           _fit('rh_from_tau_c_nm', 'rh_reliable')),
            _scalar_column('beta', '', result.beta, fit_note),
            _scalar_column('baseline', '', getattr(result, 'baseline', 0.0),
                           _join_comments(_baseline_comment(result), fit_note)),
        ]
    columns.append(_source_file_column(source_file, source_note))
    return write_origin_csv(file_path, columns, delimiter)


# Canonical vocabulary for how a CONTIN alpha was chosen -- ONE mapping behind both
# the CSV Comments cell (export_distribution) and the GUI status note
# (gui/dls_module). Historically these were parallel if/elif ladders kept in sync by
# hand, and the export copy was twice left behind when a new selector shipped:
# the 'gcv' default and the sweep-ceiling caution both reached the GUI first.
# Add a new selector here and BOTH surfaces pick it up.
_ALPHA_SELECTOR_NAMES = {
    'gcv': 'GCV',
    'ftest': 'F-test',
    'lcurve': 'L-curve corner',
    'user': 'user-supplied',
}
def gcv_gamma_tag(gcv_gamma: float) -> str:
    """The one spelling of a GCV result's robustness parameter on the record-side
    surfaces -- the distribution CSV's alpha comment, the alpha-sweep export's score
    column and a recorded mode set's provenance line: 'gamma=<value>', with
    ' (classical)' at exactly 1. `.15g`, never `:g`: a six-figure format prints a gamma
    within 1e-6 of 1 as 'gamma=1' beside the words 'robust GCV'. 15 significant figures
    keep every digit a double carries, so the record IS the value that ran."""
    return f'gamma={gcv_gamma:.15g}' + (' (classical)' if gcv_gamma == 1 else '')


def alpha_selection_note(selection_method: Optional[str],
                         ftest_prob_reject: Optional[float] = None, *,
                         gcv_gamma: Optional[float] = None,
                         edge_note: str = '', style: str = 'export') -> str:
    """Human string describing how a CONTIN alpha was chosen, for the given surface.

    One shared mapping behind both the CSV Comments cell (`export_distribution`,
    `style='export'`) and the GUI status note (`gui/dls_module`, `style='gui'`), so a
    new selector value is added in exactly one place instead of two parallel ladders.
    Returns '' for an unknown/None selector (NNLS/lognormal pass no method).

    Selector wording is kept verbatim to what each surface already shipped; the edge
    suffix follows the result's note (changed from an ASCII phrase to the shared
    constant when the note moved onto `ContinResult`):
      - export: "alpha by GCV, gamma=1 (classical)", "alpha by robust GCV, gamma=0.1",
                "alpha by F-test, p_reject=0.50", "alpha user-supplied";
                appends " (<edge_note>)" when the result carried a sweep-edge
                precondition (the export clauses of the notes failing
                `ContinResult.alpha_reliable`, read off the result by `export_distribution`
                -- so the export can never again forget the caution).
      - gui:    "CONTIN α by GCV (γ = 1, classical)", "CONTIN α by robust GCV (γ = 0.1)",
                "CONTIN α by F-test (p=0.50)". The GUI shows a
                user-supplied alpha in its input control and the edge caution on its
                own flag label, so neither is repeated here (returns '' for 'user';
                `edge_note` is ignored under `style='gui'`).

    Robust GCV is not a second selector: the recorded token stays 'gcv' and its gamma
    (`ContinResult.gcv_gamma`) is named beside it -- 'robust GCV' below 1, 'GCV' with
    the classical tag at 1. With no gamma (a GCV result recorded before it existed, or
    any other selector) the strings are the older "alpha by GCV" / "CONTIN α by GCV".
    """
    name = _ALPHA_SELECTOR_NAMES.get(selection_method)
    if name is None:
        return ''
    if style == 'gui' and selection_method == 'user':
        return ''
    has_gamma = selection_method == 'gcv' and gcv_gamma is not None
    if has_gamma and gcv_gamma < 1:
        name = 'robust GCV'
    lead = 'CONTIN α ' if style == 'gui' else 'alpha '
    note = lead + (name if selection_method == 'user' else 'by ' + name)
    if selection_method == 'ftest' and ftest_prob_reject is not None:
        note += (f' (p={ftest_prob_reject:.2f})' if style == 'gui'
                 else f', p_reject={ftest_prob_reject:.2f}')
    if has_gamma:
        classical = gcv_gamma == 1
        if style == 'gui':
            note += f' (γ = {gcv_gamma:.15g}' + (', classical)' if classical else ')')
        else:
            note += ', ' + gcv_gamma_tag(gcv_gamma)
    if style == 'export' and edge_note:
        note += f' ({edge_note})'
    return note


def export_distribution(result, file_path: str, axis: str = 'rh',
                        delimiter: str = ',', *,
                        source_file: str = '', source_note: str = '') -> str:
    """Export an NNLS, CONTIN or lognormal distribution.

    Accepts a `ContinResult` or a bare `DistributionResult` and unwraps the
    distribution itself. Writes the size/rate grid and the normalized weights. With
    axis='rh' the grid is hydrodynamic radius (nm); with axis='gamma' it is decay rate
    (1/s). Both grids are always included so either can be plotted. The reconstructed
    correlogram fit and residuals are written alongside, plus scalar summaries.

    How CONTIN's alpha was chosen (`alpha_selection_method`, `ftest_prob_reject`,
    `gcv_gamma`) and the sweep-edge preconditions (the notes failing `alpha_reliable`)
    are read OFF THE RESULT, so every caller writes the same alpha Comments cell and none
    can drop the note. A bare DistributionResult carries none of them and writes an empty
    alpha note.

    The residual weighting the fit ran under, the per-channel sample time that weight
    read and the form of its run duration are read OFF THE RESULT and written as three
    value columns. A weighting other than the program's default, and a sourced variance
    evaluated on the axis-derived sample time, are also named in the Comments of the
    Weight column and of every scalar the fit produced.

    `Peak Rh` and `Peak mean Rh` also carry the size verdict of the lobe they describe
    (`_peak_lobe_comment`): a lobe whose maximum is on the grid's end, a decay the window
    did not complete, a position outside the technique's size range.

    Parameters
    ----------
    result : ContinResult | DistributionResult
    file_path : str
    axis : str
        'rh' (default) or 'gamma' -- controls only the column order/labels; both
        grids are written regardless.
    source_file, source_note : str
        The one-source `Source file` cell and its Comments note (see
        `_source_file_column`).

    Returns
    -------
    str
    """
    dist = getattr(result, 'distribution', result)
    rh = np.asarray(dist.rh_grid_nm, dtype=float)
    gamma = np.asarray(dist.gamma_grid_s_inv, dtype=float)
    w = np.asarray(dist.weights, dtype=float)
    order = np.argsort(rh if axis == 'rh' else gamma)

    def _dist_comment(*attrs: str) -> str:
        """The clauses of the Weight column and of the summaries the inversion produced:
        the notes about them and about the whole fit (today the dead-channel record and a
        fallen-back weight), the named residual weighting, the stand-in sample time, and
        -- a failed nonlinear fit (success=False, lognormal only) meaning the weights ARE
        the seed estimate -- that caveat, which no note names."""
        return _join_comments(
            _value_comment(dist, attrs, flag='success', whole=True,
                           fallback='fit did not converge; values describe the seed '
                                    'estimate, not a fit',
                           qualify=_DISTRIBUTION_NAMED_WEIGHTINGS),
            _sample_time_comment(dist))

    columns = [
        OriginColumn('Rh', 'nm', 'hydrodynamic radius', rh[order]),
        OriginColumn('Gamma', '1/s', 'decay rate', gamma[order]),
        OriginColumn('Weight', '',
                     _join_comments('intensity fraction', _dist_comment('weights')),
                     w[order]),
    ]
    # Correlogram reconstruction (separate length from the grid).
    columns += [
        OriginColumn('Delay time', 's', 'tau', np.asarray(dist.fit_tau_s, dtype=float)),
        OriginColumn('g2-1 (fit)', '', 'reconstruction', np.asarray(dist.fitted_g2m1, dtype=float)),
        OriginColumn('Residual', '', 'data - fit', np.asarray(dist.residuals, dtype=float)),
    ]
    # CONTIN records how alpha was chosen in the alpha column's Comments cell, so a
    # distribution is never ambiguous about its regularization (mirrors the SLS
    # provenance-in-Comments convention). The selector, its level or γ and the
    # sweep-edge caution are read off the ContinResult here, not handed in by the
    # caller, so no caller can drop the caution; the wording comes from the shared
    # `alpha_selection_note` helper the GUI status note also uses, so the two can't
    # drift. A bare DistributionResult (NNLS/lognormal) carries no selector -> ''.
    alpha_note = alpha_selection_note(
        getattr(result, 'alpha_selection_method', None),
        getattr(result, 'ftest_prob_reject', None),
        gcv_gamma=getattr(result, 'gcv_gamma', None),
        edge_note=(comment_for(result, ('alpha_reliable',))
                   if hasattr(result, 'alpha_reliable') else ''), style='export')
    # The weighting and the stand-in sample time also move the alpha the sweep chose.
    alpha_weighting = _join_comments(
        comment_for(dist, qualify=_DISTRIBUTION_NAMED_WEIGHTINGS), _sample_time_comment(dist))
    # The two peak cells are one lobe's node and position, so they carry that lobe's size
    # verdict (`_peak_lobe_comment`); `Mean Rh` and `Mean Gamma` summarise the whole
    # distribution, no one lobe's.
    peak_verdict = _peak_lobe_comment(dist)
    columns += [
        _scalar_column('Method', '', dist.method, _dist_comment('method')),
        _scalar_column('alpha', '', dist.alpha,
                       comments=_join_comments(alpha_note, alpha_weighting)),
        _scalar_column('Peak Rh', 'nm', dist.peak_rh_nm,
                       _join_comments(peak_verdict, _dist_comment('peak_rh_nm'))),
        _scalar_column('Mean Rh', 'nm', dist.mean_rh_nm, _dist_comment('mean_rh_nm')),
        _scalar_column('Mean Gamma', '1/s', dist.mean_gamma_s_inv,
                       _dist_comment('mean_gamma_s_inv')),
        _scalar_column('beta', '', dist.beta),
        _scalar_column('beta estimated', '', dist.beta_estimated),
        _scalar_column('baseline', '', dist.baseline),
        _scalar_column('baseline estimated', '', dist.baseline_estimated),
        _scalar_column('RMS error', '', dist.rms_error),
        # What the residual weight WAS and what it READ, as values: the recorded form
        # ('biganzoli-full' with a committed count rate, 'biganzoli-classical' without,
        # 'g1-squared' asked for or fallen back to), the per-channel sample time
        # ('instrument' | 'axis') and the form of the variance's run duration ('finite' |
        # 'infinite'); the last two are empty where the weight read neither. The one
        # surface that says which default form ran, since the Comments cells stay silent
        # on the default. Last in the block, so no earlier column's position moves.
        _scalar_column('Residual weighting', '', getattr(dist, 'weighting', '')),
        _scalar_column('Sample time provenance', '',
                       getattr(dist, 'sample_time_provenance', '')),
        _scalar_column('Run duration form', '', getattr(dist, 'run_duration_form', '')),
        # The peak's POSITION beside `Peak Rh`'s node: the geometric mean of the lobe that
        # holds the largest-weight grid point (the lognormal arm's fitted median). Last in the
        # block, the rule the three columns above follow, so no earlier column moves.
        _scalar_column('Peak mean Rh', 'nm', getattr(dist, 'peak_mean_rh_nm', float('nan')),
                       _join_comments(_PEAK_MEAN_COMMENT, peak_verdict,
                                      _dist_comment('peak_mean_rh_nm'))),
        _source_file_column(source_file, source_note),
    ]
    return write_origin_csv(file_path, columns, delimiter)


def _peak_lobe_comment(dist) -> str:
    """The size verdict of the lobe holding a distribution's largest-weight node, the lobe
    its `Peak Rh` and `Peak mean Rh` describe: the clauses of the `DistributionPeak` whose
    lobe bounds contain `peak_rh_nm` (`find_distribution_peaks`, the one scan of a
    distribution's lobes and their verdicts: the grid end, decay completeness, the size
    range), '' when the node is not finite or its lobe is below the peak table's floor."""
    node = float(getattr(dist, 'peak_rh_nm', float('nan')))
    if not math.isfinite(node):
        return ''
    for pk in find_distribution_peaks(dist):
        if pk.rh_lo_nm <= node <= pk.rh_hi_nm:
            return comment_for(pk, ('rh_nm', 'mean_rh_nm', 'rh_reliable'))
    return ''


#: The `Peak mean Rh` column's Comments cell. A reweighted view's value is the geometric
#: mean of the reweighted lobe on every arm; the intensity export's lognormal value is its
#: fitted median, which the Weight column cannot reproduce, so that cell says so.
_WEIGHTED_PEAK_MEAN_COMMENT = 'geometric mean of the lobe holding the largest-weight grid point'
_PEAK_MEAN_COMMENT = _WEIGHTED_PEAK_MEAN_COMMENT + ' (lognormal: its fitted median exp(mu))'


def export_weighted_distribution(result, file_path: str,
                                 delimiter: str = ',', *,
                                 source_file: str = '',
                                 source_note: str = '') -> str:
    """Export a reweighted (volume/surface/number) distribution (WeightedDistributionResult).

    Writes the Rh grid, the reweighted weights, and a per-bin validity flag. The
    model-dependence qualifier (`model_weighted_view`) rides in the Weight column's
    Comments cell ONLY (mirroring the "uncalibrated, arbitrary scale" convention: no extra
    rows, so the Origin import is unaffected) -- the reweighted view is derived and
    model-dependent, never a measurement -- beside the result's notes about the weights.
    Carries no ± (invariant 8): reweighted distributions are report-only.
    """
    rh = np.asarray(result.rh_grid_nm, dtype=float)
    w = np.asarray(result.weights, dtype=float)
    valid = np.asarray(result.valid_mask, dtype=bool)
    order = np.argsort(rh)
    fail_note = ('' if getattr(result, 'source_success', True)
                 else 'source fit did not converge; values describe the seed estimate, '
                      'not a fit')
    label = str(result.weighting).capitalize()
    model = result.optical_model or 'intensity'
    weight_note = comment_for(result, ('weights',), qualify=('model_weighted_view',))
    columns = [
        OriginColumn('Rh', 'nm', 'hydrodynamic radius', rh[order]),
        OriginColumn(f'{label} weight', '',
                     _join_comments(weight_note, fail_note), w[order]),
        OriginColumn('In validity range', '', 'model valid for this size bin',
                     valid[order]),
    ]
    columns += [
        _scalar_column('Weighting', '', result.weighting),
        _scalar_column('Optical model', '', model),
        _scalar_column('Assumed shape', '', result.assumed_shape),
        _scalar_column('m (n_solute/n_medium)', '', result.m),
        _scalar_column('Source method', '', result.source_method, fail_note),
        _scalar_column('Peak Rh', 'nm', result.peak_rh_nm, fail_note),
        _scalar_column('Mean Rh', 'nm', result.mean_rh_nm, fail_note),
        _scalar_column('Bins out of range', '', result.n_invalid_bins,
                       comments=comment_for(result, ('n_invalid_bins', 'all_valid'))),
        _scalar_column('Peak mean Rh', 'nm',
                       getattr(result, 'peak_mean_rh_nm', float('nan')),
                       _join_comments(_WEIGHTED_PEAK_MEAN_COMMENT, fail_note)),
        _source_file_column(source_file, source_note),
    ]
    return write_origin_csv(file_path, columns, delimiter)


def export_lcurve(result, file_path: str, delimiter: str = ',', *,
                  source_file: str = '', source_note: str = '') -> str:
    """Export a CONTIN alpha-sweep for inspection.

    Accepts a `ContinResult` or a bare `LCurveResult` and unwraps the sweep itself (the
    shape `export_distribution` has). Always writes alpha, the weighted residual norm and
    the solution seminorm plus the chosen alpha/index; the solution norm ||x||^2
    (`x_norms`, the L-curve's size axis) follows the seminorm when the sweep carries it.
    The method-specific score curves -- the GCV-family score, the F-test cumulative F,
    and the Tikhonov effective DOF -- are appended only when the sweep carries them (they
    are populated per selection method: `gcv` for GCV, `ftest_fc` for the F-test,
    `dof_eff` for both), so the CSV matches whichever ordinate the diagnostic view showed.

    `LCurveResult.gcv` holds whichever GCV-family score was MINIMIZED -- the classical
    V(alpha) at gamma = 1, Lukas's robust score (RGCV) below it -- so the column's
    Comments cell names which, with the gamma, read off the ContinResult; a bare sweep
    carries no gamma and keeps the classical wording.
    """
    lcurve = getattr(result, 'lcurve', result)
    gcv_gamma = getattr(result, 'gcv_gamma', None)
    columns = [
        OriginColumn('alpha', '', 'regularization parameter',
                     np.asarray(lcurve.alphas, dtype=float)),
        OriginColumn('Residual norm', '', '||M^.5 (A x - y)||^2 (weighted)',
                     np.asarray(lcurve.residual_norms, dtype=float)),
        OriginColumn('Solution seminorm', '', '||L x||^2',
                     np.asarray(lcurve.solution_norms, dtype=float)),
    ]
    # The L-curve's size axis, ||x||^2 per alpha, beside the penalty seminorm; a sweep
    # saved before the field existed was selected on the seminorm and carries none.
    x_norms = getattr(lcurve, 'x_norms', None)
    if x_norms is not None:
        columns.append(OriginColumn('Solution norm', '', '||x||^2',
                                    np.asarray(x_norms, dtype=float)))
    # Method-specific score curves, present only for the method that ran.
    if getattr(lcurve, 'gcv', None) is not None:
        # The column keeps its header (an Origin import keys on it); WHICH score it holds
        # travels in the Comments cell, with the gamma at the same precision the
        # distribution export records it.
        if gcv_gamma is not None and gcv_gamma < 1:
            score_note = ('robust GCV score Vbar(alpha) = [gamma + (1 - gamma) mu2] '
                          f'V(alpha) (minimized), {gcv_gamma_tag(gcv_gamma)}')
        elif gcv_gamma is not None:
            score_note = f'V(alpha) (minimized), {gcv_gamma_tag(gcv_gamma)}'
        else:
            score_note = 'V(alpha) (minimized)'
        columns.append(OriginColumn('GCV score', '', score_note,
                                    np.asarray(lcurve.gcv, dtype=float)))
    if getattr(lcurve, 'ftest_fc', None) is not None:
        columns.append(OriginColumn('Cumulative F', '',
                                    'F-test probability to reject',
                                    np.asarray(lcurve.ftest_fc, dtype=float)))
    if getattr(lcurve, 'dof_eff', None) is not None:
        columns.append(OriginColumn('Effective DOF', '',
                                    'Tikhonov hat-trace per alpha',
                                    np.asarray(lcurve.dof_eff, dtype=float)))
    columns += [
        _scalar_column('Optimal alpha', '', lcurve.optimal_alpha),
        _scalar_column('Optimal index', '', lcurve.optimal_index),
        _source_file_column(source_file, source_note),
    ]
    return write_origin_csv(file_path, columns, delimiter)


def export_intensity_trace(trace, file_path: str, *, provenance: str = '',
                           disclosure: str = '', delimiter: str = ',') -> str:
    """Export a raw count-rate trace as Origin-style (time, count rate) columns.

    Where the trace came from, and any disclosure about the mean count rate
    committed alongside it, ride in the Comments cell of the affected column --
    the locked convention -- so the row count is untouched and an Origin import
    stays intact.
    """
    columns = [
        OriginColumn('Time', 's', 'elapsed time from the start of the run',
                     np.asarray(trace.times_s, dtype=float)),
        OriginColumn('Count rate', 'cps', _join_comments(provenance, disclosure),
                     np.asarray(trace.count_rates_cps, dtype=float)),
    ]
    return write_origin_csv(file_path, columns, delimiter)


def export_windowed_hampel(result, trace, file_path: str, *, provenance: str = '',
                           disclosure: str = '', global_flags=None,
                           delimiter: str = ',') -> str:
    """Export a windowed-Hampel decomposition as Origin-style numeric columns.

    The decomposition IS this diagnostic's deliverable, which is why its flags
    are columns here: the locked "flags are GUI-owned overlays, never baked into
    export data" decision forbids overlay flags contaminating ANOTHER analysis's
    numbers, not a diagnostic exporting its own.

    A 'no' cell and a blank cell mean different things in neighbouring columns,
    so each says which in its own Comments cell: the two boolean columns
    (`Flagged`, `Local verdict`) write 'yes' / 'no', while a blank in the rolling
    curves is "this layer had no verdict here".

    The global envelope's parameters ride alongside when supplied, so a reader
    can see both bands' settings without opening a second file.
    """
    columns = [
        OriginColumn('Time', 's', 'elapsed time from the start of the run',
                     np.asarray(result.times_s, dtype=float)),
        OriginColumn('Count rate', 'cps',
                     _join_comments(provenance, disclosure,
                                    'unaltered: this diagnostic flags, never filters'),
                     np.asarray(trace.count_rates_cps, dtype=float)),
        OriginColumn('Rolling median', 'cps',
                     'moving-window Hampel centre; blank where the layer abstained',
                     np.asarray(result.rolling_centre_cps, dtype=float)),
        OriginColumn('Rolling scale', 'cps',
                     f'{result.mad_to_sigma:g} x window MAD; '
                     f'blank where the layer abstained',
                     np.asarray(result.rolling_scale_cps, dtype=float)),
        OriginColumn('Flagged', '',
                     "impulse flag: 'yes' where flagged, 'no' where not",
                     np.asarray(result.flagged_mask, dtype=bool)),
        OriginColumn('Local verdict', '',
                     "'yes' where a full implosion-free window existed; 'no' "
                     "where this layer did not judge the sample",
                     np.asarray(result.local_verdict_mask, dtype=bool)),
        _scalar_column('k', '', result.k,
                       'robust band multiplier, shared with the global envelope'),
        _scalar_column('Window', 'points', result.window_pts,
                       'resolved from the trace length'
                       if result.window_was_resolved else 'user-set'),
        _scalar_column('Window K', 'points', result.window_half_width_k,
                       'window half-width; an impulse is at most K samples'),
        _scalar_column('Window floor', 'points', result.window_floor_pts),
        _scalar_column('Events', '', result.n_events,
                       'maximal runs of consecutive flagged samples'),
        _scalar_column('Longest event', 'points', result.longest_event_pts),
        _scalar_column('Judged', 'points', result.n_judged),
        _scalar_column('Fraction flagged', '', result.fraction_flagged,
                       'of the JUDGED samples, not of the whole trace'),
        _scalar_column('Abstained (edge)', '', result.fraction_abstained_edge,
                       'no full window fits at the first and last K samples'),
        _scalar_column('Abstained (implosion)', '',
                       result.fraction_abstained_implosion,
                       'window MAD imploded to zero; no scale exists there and '
                       'none is substituted'),
    ]
    if global_flags is not None:
        columns.extend([
            _scalar_column('Global k', '', global_flags.k,
                           'the whole-trace envelope, for comparison'),
            _scalar_column('Global flagged', 'points', global_flags.n_flagged),
            _scalar_column('Global scale', 'cps', global_flags.scale_cps,
                           'no usable scale: over half the samples are identical'
                           if global_flags.scale_is_degenerate else ''),
        ])
    return write_origin_csv(file_path, columns, delimiter)


# ---------------------------------------------------------------------------
# The remaining trace diagnostics, each exporting its own data
# ---------------------------------------------------------------------------
# One writer per view, all on the same locked conventions as the decomposition
# above: metadata rides in the four label rows and in length-1 parameter columns,
# never as extra data rows, so an Origin import stays intact. Each carries the
# caveat its own method needs -- the condition a curve is only valid under is
# part of the curve, and a file that travels without it can be read wrongly.

def export_welch_psd(result, file_path: str, *, provenance: str = '',
                     disclosure: str = '', delimiter: str = ',') -> str:
    """Export a Welch power spectral density as Origin-style (f, PSD) columns.

    The E.D.F. rides along because it is what says how much the estimate can be
    trusted point by point: it is the degrees of freedom of the approximating
    chi-square distribution, so the confidence factors beside it multiply the PSD
    to give the band. Without it the curve is a shape with no error bar.
    """
    columns = [
        OriginColumn('Frequency', 'Hz', 'one-sided; zero-frequency bin included',
                     np.asarray(result.frequencies_hz, dtype=float)),
        OriginColumn('PSD', 'cps^2/Hz',
                     _join_comments(provenance, disclosure,
                                    'Welch averaged modified periodogram; each '
                                    'segment mean-removed before windowing'),
                     np.asarray(result.psd, dtype=float)),
        _scalar_column('Shot-noise floor', 'cps^2/Hz', result.shot_floor_psd,
                       'flat 2 x mean count rate; assumes dead-time-free '
                       'Poisson detection'),
        _scalar_column('Mean count rate', 'cps', result.mean_cps),
        _scalar_column('Segment length', 'points', result.nperseg,
                       'resolved from the trace length'
                       if result.segment_was_resolved else 'user-set'),
        _scalar_column('Segment overlap', 'points', result.noverlap),
        _scalar_column('Segments', '', result.n_segments,
                       'K, the number of periodograms averaged'),
        _scalar_column('E.D.F.', '', result.edf,
                       'equivalent degrees of freedom of the approximating '
                       'chi-square distribution; halves at zero frequency and '
                       'at Nyquist'),
        _scalar_column('CI lower factor', '', result.ci_lower_factor,
                       f'multiply PSD by this for the '
                       f'{result.confidence_level:.0%} band'),
        _scalar_column('CI upper factor', '', result.ci_upper_factor),
        _scalar_column('Sample spacing', 's', result.dt_s),
    ]
    return write_origin_csv(file_path, columns, delimiter)


def export_periodogram(result, file_path: str, *, provenance: str = '',
                       disclosure: str = '', delimiter: str = ',') -> str:
    """Export a periodogram and its false-alarm level, Origin-style.

    Power is in units of the data variance, and the Comments say so, because the
    false-alarm level is only meaningful against that normalization. The
    null-assumption note (`null_not_white`) travels with the file for the same reason:
    a verdict computed for white noise on a trace that is not white is the one way this
    number misleads, and a reader opening the file later has no other way to know.
    """
    def null_note(attr: str) -> str:
        """A reading computed under the null: the result's note when the null is
        violated, else what the null assumes."""
        return comment_for(result, (attr,)) or 'null assumes white, Gaussian noise'

    columns = [
        OriginColumn('Frequency', 'Hz',
                     'zero and Nyquist bins excluded: neither follows the '
                     'exponential null',
                     np.asarray(result.frequencies_hz, dtype=float)),
        OriginColumn('Power', 'variance units',
                     _join_comments(provenance, disclosure,
                                    'classical periodogram normalized by the '
                                    'data variance'),
                     np.asarray(result.power, dtype=float)),
        _scalar_column('Peak frequency', 'Hz', result.peak_frequency_hz),
        _scalar_column('Peak power', 'variance units', result.peak_power),
        # Every quantity computed UNDER the null repeats the caveat, and the flag
        # itself gets a column. A CSV column is read alone -- an Origin user who
        # opens `Peak significant` never sees a note left on the column that
        # merely names the condition. The plot can replace the verdict outright;
        # a file cannot, so here the caveat travels on each number it governs.
        _scalar_column('False-alarm probability', '',
                       result.false_alarm_probability, null_note('false_alarm_probability')),
        _scalar_column('False-alarm level', 'variance units',
                       result.significance_level,
                       _join_comments(
                           f'power a peak must exceed at p = '
                           f'{result.requested_false_alarm:g}',
                           null_note('significance_level'))),
        _scalar_column('Peak significant', '', bool(result.peak_is_significant),
                       null_note('peak_is_significant')),
        _scalar_column('Null assumption violated', '',
                       bool(result.null_assumption_violated),
                       "'yes' when the blocking analysis found correlations, "
                       "which breaks the white-noise null the verdict above was "
                       "computed under"),
        _scalar_column('Independent frequencies', '', result.n_independent,
                       'N in the false-alarm formula'),
        _scalar_column('Variance', 'cps^2', result.variance_cps2,
                       'the normalizer'),
        _scalar_column('Sample spacing', 's', result.dt_s),
    ]
    return write_origin_csv(file_path, columns, delimiter)


def export_allan_deviation(result, file_path: str, *, provenance: str = '',
                           disclosure: str = '', delimiter: str = ',') -> str:
    """Export an Allan deviation curve as Origin-style (tau, sigma_y) columns.

    Two conditions travel with the numbers. The deviation is FRACTIONAL, so the
    mean it was divided by is here and the absolute form is one multiply away.
    And it is the zero-dead-time form, valid because the gates are contiguous
    (T = tau) -- the condition the definition is stated under, and one a reader
    cannot check from the columns alone.
    """
    columns = [
        OriginColumn('Averaging time', 's',
                     'tau = block size x sample spacing; contiguous gates '
                     '(T = tau, no dead time)',
                     np.asarray(result.tau_s, dtype=float)),
        OriginColumn('Allan deviation', '',
                     _join_comments(provenance, disclosure,
                                    'fractional: block means divided by the '
                                    'trace mean before differencing'),
                     np.asarray(result.allan_deviation, dtype=float)),
        OriginColumn('Pairs', '',
                     'm, the number of adjacent differences averaged at this tau',
                     np.asarray(result.allan_n_pairs, dtype=float)),
        OriginColumn('Block size', 'points', '',
                     np.asarray(result.block_sizes, dtype=float)),
        _scalar_column('Mean count rate', 'cps', result.mean_cps,
                       'the fractional normalizer'),
        _scalar_column('Sample spacing', 's', result.dt_s),
    ]
    return write_origin_csv(file_path, columns, delimiter)


def export_block_variance(result, file_path: str, *, provenance: str = '',
                          disclosure: str = '', delimiter: str = ',') -> str:
    """Export a blocking analysis as Origin-style (block size, SE) columns."""
    verdict = ('correlations detected' if result.correlations_detected
               else 'no correlations detected')
    columns = [
        OriginColumn('Block size', 'points', '',
                     np.asarray(result.block_sizes, dtype=float)),
        OriginColumn('Std error of the mean', 'cps',
                     _join_comments(provenance, disclosure,
                                    'std(block means) / sqrt(number of blocks)'),
                     np.asarray(result.standard_errors, dtype=float)),
        OriginColumn('Blocks', '', 'how many blocks at this size',
                     np.asarray(result.n_blocks, dtype=float)),
        _scalar_column('SE ratio', '', result.se_ratio,
                       f'{verdict} at threshold '
                       f'{result.correlation_threshold:g}'),
        _scalar_column('Correlation threshold', '', result.correlation_threshold),
        _scalar_column('Correlations detected', '',
                       bool(result.correlations_detected)),
        _scalar_column('Mean count rate', 'cps', result.mean_cps),
        _scalar_column('Sample spacing', 's', result.dt_s),
    ]
    return write_origin_csv(file_path, columns, delimiter)


def export_count_rate_histogram(result, file_path: str, *, provenance: str = '',
                                disclosure: str = '',
                                delimiter: str = ',') -> str:
    """Export a count-rate histogram and its fitted curves, Origin-style.

    Fitted curves are written only when they exist; a fit that was not requested
    or did not converge contributes no column at all, rather than a column of
    blanks that would read as a fit returning nothing.
    """
    columns = [
        OriginColumn('Bin center', 'cps', '',
                     np.asarray(result.bin_centers, dtype=float)),
        OriginColumn('Count', '',
                     _join_comments(provenance, disclosure,
                                    f'{result.n_bins} bins by the '
                                    f'{result.bin_method} rule'),
                     np.asarray(result.counts, dtype=float)),
    ]
    if result.gaussian_curve is not None:
        chi2 = result.gaussian_chi2_reduced
        columns.append(OriginColumn(
            'Gaussian fit', '',
            '' if chi2 is None else f'reduced chi-square {chi2:.4g}',
            np.asarray(result.gaussian_curve, dtype=float)))
    if result.poisson_curve is not None:
        chi2 = result.poisson_chi2_reduced
        columns.append(OriginColumn(
            'Poisson fit', '',
            '' if chi2 is None else f'reduced chi-square {chi2:.4g}',
            np.asarray(result.poisson_curve, dtype=float)))
    columns.extend([
        _scalar_column('Fano factor', '', result.fano_factor,
                       'variance / mean in count space; ~1 is shot-noise limited'),
        _scalar_column('Mean counts', '', result.mean_counts),
        _scalar_column('Variance counts', '', result.variance_counts),
        _scalar_column('CV', '', result.cv),
        _scalar_column('Shot-noise CV', '', result.shot_noise_cv,
                       'the Poisson expectation, 1/sqrt(mean counts)'),
        _scalar_column('Integration time', 's', result.integration_time_s,
                       'estimated from the time axis'
                       if result.integration_time_estimated else 'supplied'),
    ])
    return write_origin_csv(file_path, columns, delimiter)


def export_running_average(result, file_path: str, *, provenance: str = '',
                           disclosure: str = '', delimiter: str = ',') -> str:
    """Export a sliding-window running average, Origin-style."""
    columns = [
        OriginColumn('Time', 's', 'elapsed time from the start of the run',
                     np.asarray(result.times_s, dtype=float)),
        OriginColumn('Running mean', 'cps',
                     _join_comments(provenance, disclosure,
                                    f'sliding window of {result.window_description}'),
                     np.asarray(result.running_mean, dtype=float)),
        OriginColumn('Running std', 'cps', 'within-window standard deviation',
                     np.asarray(result.running_std, dtype=float)),
    ]
    return write_origin_csv(file_path, columns, delimiter)


def _blank_if_none(value):
    """A scalar cell that is EMPTY when the field does not apply, never 0 or 'None'.

    `cumulant_fit_cutoff` is None whenever the rates did not come from a cumulant, and a
    numeric default there would assert a setting the fit never read.
    """
    return '' if value is None else value


def export_gamma_q2(result, file_path: str, delimiter: str = ',') -> str:
    """Export a Gamma-vs-q^2 multi-angle analysis (GammaQ2Result).

    The pooled D and Rh carry whatever the analysis said about the set it was given --
    that the measurements do not all share one sample identity (the assumption Gamma vs
    q^2 rests on), that a point's weight fell back -- and the per-angle Gamma column what
    it said about the rates themselves (an angle dropped for an unusable decay rate, a
    per-point fit's record). Without it this exporter had no note channel at all, so a D
    fitted across two different samples exported looking like a measurement.

    The D and Rh columns ALSO carry the pooled fit's reliability verdict when it is
    flagged (the preconditions failing `rh_reliable`: a flagged per-angle fit inside the
    line, a pooled Rh outside the size range, no viscosity), on the Comments cell -- the
    same channel every other flagged value rides out on, since the export outlives the
    session.

    `Is diffusive` names the criteria it was judged against either way, and when the
    fit fails them the result's `not_diffusive` note rides it and D, Rh and their SEs,
    the values it is about: a Gamma that is not linear through the origin in q^2 makes
    the through-origin slope an apparent D. Each SE cell carries its value's marker, the
    estimator tag and the statistical label joined -- independent facts, none of which
    may displace another.
    """
    criteria = _diffusive_criteria(result)
    if result.is_diffusive:
        verdict_note = f'within stated tolerance: {criteria}'
    elif not math.isfinite(result.r_squared):
        # `analyze_gamma_q2` also returns is_diffusive=False when FEWER THAN TWO angles
        # yielded a usable rate: no line was fitted, so the criteria were never evaluated
        # and saying they failed would assert a test that never ran. The result's own
        # note says what happened (the precondition failing `is_diffusive`).
        verdict_note = (comment_for(result, ('is_diffusive',))
                        or 'no Gamma-vs-q^2 line was fitted, so the linearity criteria '
                           'were not evaluated')
    else:
        verdict_note = _failed_diffusive_comment(result, criteria)

    def pooled(*attrs: str) -> str:
        """The pooled D's and Rh's clauses: the notes about them and the whole fit, the
        preconditions failing the pooled `rh_reliable`, and `apparent`, since the line
        extrapolates over angle at one concentration (invariant 7)."""
        return _value_comment(result, attrs, flag='rh_reliable', whole=True,
                              qualify=('apparent',) + _VALUE_QUALIFIERS)

    def pooled_se(*attrs: str) -> str:
        return _se_comment(result, attrs, 'statistical (over angles)', flag='rh_reliable')

    columns = [
        OriginColumn('Angle', 'deg', '', np.asarray(result.angles_deg, dtype=float)),
        OriginColumn('q^2', 'm^-2', 'scattering vector squared',
                     np.asarray(result.q2_m2, dtype=float)),
        OriginColumn('Gamma', '1/s',
                     _join_comments('decay rate', comment_for(result, ('gamma_s_inv',))),
                     np.asarray(result.gamma_s_inv, dtype=float)),
        OriginColumn('D_app', 'm^2/s',
                     _join_comments('Gamma / q^2', comment_for(result, ('d_app_m2_s',))),
                     np.asarray(result.d_app_m2_s, dtype=float)),
        _scalar_column('D', 'm^2/s', result.d_m2_s,
                       _join_comments('through-origin slope', pooled('d_m2_s'))),
        _scalar_column('D SE', 'm^2/s', result.d_se, comments=pooled_se('d_m2_s', 'd_se')),
        _scalar_column('Rh', 'nm', result.rh_nm, comments=pooled('rh_nm')),
        _scalar_column('Rh SE', 'nm', result.rh_se, comments=pooled_se('rh_nm', 'rh_se')),
        _scalar_column('R^2', '', result.r_squared),
        _scalar_column('Intercept', '1/s', result.intercept_s_inv),
        _scalar_column('Intercept (relative)', '', result.intercept_relative),
        _scalar_column('D_app trend (relative)', '', result.d_app_trend_rel),
        _scalar_column('Is diffusive', '', result.is_diffusive, verdict_note),
        # The INPUTS the pooled D was fitted from, not just its outputs. A promotable Rh
        # must not travel without them (invariant 8 clause A's se_estimator obligation in
        # another material): two exports of the same angles at different cumulant cutoffs
        # were otherwise byte-identical in provenance. `gamma_source` gates the pair --
        # a single-exponential source reads no cumulant setting, so a value there would
        # claim one it never used.
        _scalar_column('Gamma source', '', getattr(result, 'gamma_source', '') or ''),
        _scalar_column('Cumulant method', '',
                       getattr(result, 'cumulant_method', None) or ''),
        _scalar_column('Cumulant fit cutoff', '',
                       _blank_if_none(getattr(result, 'cumulant_fit_cutoff', None))),
        _cumulant_weighting_row(result, ' deg'),
        *_cumulant_record_rows(result, ' deg'),
    ]
    return write_origin_csv(file_path, columns, delimiter)


def _diffusive_criteria(result) -> str:
    """The through-origin criteria a line's `is_diffusive` was judged against, with the
    thresholds the result carries."""
    return (f'R^2 >= {result.r2_threshold:g} and '
            f'|intercept|/mean(Gamma) <= {result.intercept_rel_threshold:g}')


def _failed_diffusive_comment(result, criteria: str) -> str:
    """A fitted line's failed `is_diffusive` cell: the preconditions failing the flag (the
    `not_diffusive` note, its measured R^2 and intercept), then the criteria it failed."""
    return _join_comments(comment_for(result, ('is_diffusive',)),
                          f'fails {criteria.replace(" and ", " or ")}')


def _cumulant_weighting_row(result, unit: str):
    """The `Cumulant weighting (requested)` provenance row of a pooled DLS export, beside
    `Cumulant method` and `Cumulant fit cutoff` and gated by the same `gamma_source` (blank
    under a single-exponential source, which reads no cumulant setting): the weighting the
    per-point cumulants were ASKED for, a request token. Named apart from the per-fit
    export's `Cumulant weighting` column, which records the form a fit ran under, so one
    column name never holds two vocabularies. Its Comments cell names the points whose default
    weight could not be formed and fell back to 'uniform', with the cause (the per-point
    `weight_fallback_uniform` record the result relays), so a pooled number built partly
    from unweighted points never travels unlabelled.
    One helper for both pooled exporters (rule 10: the two carry the row symmetrically)."""
    return _scalar_column('Cumulant weighting (requested)', '',
                          getattr(result, 'cumulant_weighting', None) or '',
                          comment_for(result, ('cumulant_weighting',),
                                      codes=('weight_fallback_uniform',)))


def _cumulant_record_rows(result, unit: str) -> list:
    """The two provenance rows a pooled DLS export closes on, after the request row:

    - `Cumulant order` -- the order every per-point cumulant ran at (the Settings order),
      beside the method, cutoff and weighting it moves the promotable Rh with; blank under
      a single-exponential source, which fits no cumulant;
    - `Cumulant weighting (recorded)` -- the weighting each per-point fit actually ran
      under (`point_weightings`), its distinct forms as the value, and in its Comments the
      points of each form when they differ and the points whose sourced weight was
      evaluated on a sample time derived from the delay axis -- the facts the per-fit
      export names on every scalar (`_sample_time_comment`), relayed here so a pooled
      number weighted on a stand-in never travels unlabelled.

    Appended after the existing rows, so every column an Origin template already reads
    keeps its place. One helper for both pooled exporters (rule 10)."""
    forms = tuple(getattr(result, 'point_weightings', ()) or ())
    provenance = tuple(getattr(result, 'point_sample_time_provenance', ()) or ())
    points = np.asarray(getattr(result, 'angles_deg', None)
                        if getattr(result, 'angles_deg', None) is not None
                        else getattr(result, 'concentrations_g_per_mL', ()), dtype=float)
    distinct = [f for f in dict.fromkeys(forms) if f]
    comments = []
    if len(distinct) > 1:
        comments.append('; '.join(
            f'{form} at ' + ', '.join(f'{x:.4g}' for x, f in zip(points, forms, strict=True)
                                      if f == form) + unit
            for form in distinct))
    axis = [f'{x:.4g}' for x, p in zip(points, provenance, strict=True) if p == 'axis']
    if axis:
        comments.append(f'{_registry_clause("sample_time_from_axis")} '
                        f'(at {", ".join(axis)}{unit})')
    return [
        _scalar_column('Cumulant order', '',
                       _blank_if_none(getattr(result, 'cumulant_order', None))),
        _scalar_column('Cumulant weighting (recorded)', '', ', '.join(distinct),
                       _join_comments(*comments)),
    ]


def export_mode_map(map_run, family_fits, amplitude, file_path: str, *,
                    point_families=None, family_states=None,
                    delimiter: str = ',') -> str:
    """Export a mode map, its per-family fits, and the amplitude report.

    One file, three blocks, per the locked conventions (flags never change row
    counts; caveats ride the affected columns' Comments cells):

      * per-point arrays -- every extracted mode of every run (`map_run`, a
        ModeMapRun), with `point_families` (an aligned sequence of family
        labels, '' = unassigned, supplied by the caller from the user's picks)
        as the Family column, then six per-point columns read off each point:
        the residual weighting its inversion ran under, the sample time and
        run-duration form that weight read, and the CONTIN alpha, its selector
        and the GCV gamma the amplitudes came from;
      * per-family scalars -- one labeled block per ModeFamilyFitResult in
        `family_fits` ({label: fit}, insertion-ordered). Every Rh is APPARENT
        and MODE-RESOLVED, and the through-origin verdict travels WITH its
        tolerances, so a number never outruns its label;
      * the fast/slow amplitude report (`amplitude`, an AmplitudeRatioResult
        or None), whose flatness verdict also names its threshold.

    Mode amplitudes and per-point decay rates are single-correlogram
    quantities and carry no SE columns (invariant 8); the per-family D/Rh/a/b
    SEs are multi-angle regression SEs, labeled statistical.
    """
    pts = list(map_run.points)
    fams = (list(point_families) if point_families is not None
            else [''] * len(pts))
    if len(fams) != len(pts):
        raise ValueError(
            f"point_families has {len(fams)} entries for {len(pts)} map "
            f"points; they must align."
        )
    n_low = sum(1 for p in pts if p.low_amplitude)
    amp_note = 'intensity fraction of the run (modes sum to ~1); NOT mass/weight percent'
    if n_low:
        floor = max(p.amplitude_floor for p in pts)
        amp_note = _join_comments(
            amp_note,
            f'{n_low} mode(s) below the {floor:.0%} amplitude floor: imprecise')
    # A separate judgement, so a separate note -- and it belongs to the RATE columns, not
    # to the amplitude one. `getattr`, because a map rebuilt from a session recorded before
    # this flag existed carries no verdict.
    n_undecayed = sum(1 for p in pts if getattr(p, 'decay_incomplete', False))
    rate_note = ''
    if n_undecayed:
        # The condition, not one test's threshold: a point is undecayed by its own rate or,
        # from an inversion, by its correlogram (`ModePoint.decay_incomplete`).
        rate_note = (
            f'{n_undecayed} mode(s) did not finish decaying inside their delay window '
            f'(Theory-and-Equations-Guide §3.1.4): those rates rest on a decay the '
            f'measurement did not complete')
    # The point verdict beside it, of the same note-gate shape: a mode whose lobe maximum
    # sits on the Rh grid's first or last node (`ModePoint.grid_end`), its rate that of a
    # population the grid does not contain, counted in the registry's words.
    n_grid_end = sum(1 for p in pts if getattr(p, 'grid_end', False))
    if n_grid_end:
        rate_note = _join_comments(
            rate_note, _registry_clause('grid_end_points', n=n_grid_end, total=len(pts)))
    columns = [
        # The map's notes are about its runs' extractions, every one of them: the map
        # column that describes every run carries them, whatever each names.
        OriginColumn('Angle', 'deg',
                     _join_comments('every extracted mode of every run',
                                    comment_for(map_run, whole=True)),
                     np.array([p.angle_deg for p in pts], dtype=float)),
        OriginColumn('q^2', 'm^-2', 'scattering vector squared',
                     np.array([p.q2_m2 for p in pts], dtype=float)),
        # The mode's POSITION, the Gamma its D_app was derived from, so each row's D_app
        # is its Gamma / q^2: a distribution mode's lobe geometric mean, not the node a
        # pick is keyed on (`plotted_gamma_s_inv`).
        OriginColumn('Gamma', '1/s',
                     _join_comments('mode decay rate (a distribution mode: the geometric '
                                    'mean of its lobe)', rate_note),
                     np.array([plotted_gamma_s_inv(p) for p in pts], dtype=float)),
        OriginColumn('D_app', 'm^2/s',
                     _join_comments('Gamma / q^2', rate_note),
                     np.array([p.d_m2_s for p in pts], dtype=float)),
        OriginColumn('Amplitude fraction', '', amp_note,
                     np.array([p.amplitude_fraction for p in pts], dtype=float)),
        OriginColumn('Mode', '', 'rank within its run (1 = fastest)',
                     np.array([p.mode_index + 1 for p in pts], dtype=float)),
        OriginColumn('Source', '', 'extraction method',
                     [p.source_method for p in pts]),
        OriginColumn('Family', '', "user-picked mode family ('' = unassigned)",
                     fams),
        # What built each point's amplitude, read OFF THE POINT and written per point:
        # the runs of one map can differ (one file states its sample times, the next
        # does not), and a per-column Comments cell cannot say which row is which.
        # `getattr` with an empty default: a point object without a field writes an
        # empty cell. They close the per-point block, so no earlier column moves.
        OriginColumn('Residual weighting', '',
                     "recorded form of the inversion's residual weight "
                     "('' = parametric source: no inversion)",
                     [getattr(p, 'weighting', '') for p in pts]),
        OriginColumn('Sample time provenance', '',
                     "per-channel sample time that weight read: instrument | axis "
                     "('' = it read none)",
                     [getattr(p, 'sample_time_provenance', '') for p in pts]),
        OriginColumn('Run duration form', '',
                     "form of that weight's run duration: finite | infinite "
                     "('' = it read none)",
                     [getattr(p, 'run_duration_form', '') for p in pts]),
        # The regularization each point's amplitudes came from, per point for the same
        # reason: the map's runs can land on different alphas, and the selector's own
        # parameter is named beside it. After the weighting block, so no earlier column
        # moves. The sweep's range and an F-test level are map-wide and ride the Angle
        # column's notes.
        OriginColumn('Alpha', '',
                     "CONTIN regularization alpha ('' = NNLS or parametric source: "
                     "no alpha)",
                     np.array([np.nan if getattr(p, 'alpha', None) is None
                               else float(p.alpha) for p in pts], dtype=float)),
        OriginColumn('Alpha selection', '',
                     "how CONTIN chose that alpha: gcv | lcurve | ftest | user "
                     "('' = no alpha)",
                     [getattr(p, 'alpha_selection_method', '') for p in pts]),
        OriginColumn('GCV gamma', '',
                     "robustness parameter of the GCV selector; 1 = classical GCV "
                     "('' = another selector or no alpha)",
                     np.array([np.nan if getattr(p, 'gcv_gamma', None) is None
                               else float(p.gcv_gamma) for p in pts], dtype=float)),
    ]

    for label, fit in family_fits.items():
        # A VH family's caveat prints beside the value it disqualifies (rule
        # 10's surfacing sweep): Gamma_VH carries the rotational term, so its
        # slope is not a translational D. `family_states` maps label -> the
        # family's polarization state ('VV' | 'VH' | None), caller-supplied.
        vh_note = ''
        if family_states is not None and family_states.get(label) == 'VH':
            vh_note = ('VH (depolarized): Gamma_VH = Gamma_VV + 6*D_r, so '
                       'the fitted slope is not a translational D and this '
                       'Rh is not a hydrodynamic radius')
        # The verdict names the criteria it was judged against on BOTH branches, as the
        # `export_gamma_q2` sibling does, and a failed one the family's `not_diffusive`
        # note. A mode family is exactly where the origin criteria are expected to fail (a
        # VH family's slope is not a translational D).
        criteria = _diffusive_criteria(fit)
        verdict_note = (f'within stated tolerance: {criteria}' if fit.is_diffusive
                        else _failed_diffusive_comment(fit, criteria))
        # `vh_note` rides the SE cells too, not only the values beside them: a reader
        # filtering on the SE columns must still see that the quantity is not a
        # translational D at all. The family's D and Rh, and their SEs, carry the
        # preconditions failing its `rh_reliable`, as the `export_gamma_q2` sibling's do.
        def se_note(*attrs, fit=fit, vh_note=vh_note, flag=''):
            return _join_comments(vh_note, _se_comment(fit, attrs, 'statistical (over angles)',
                                                       flag=flag, whole=True))
        columns += [
            _scalar_column(f'D ({label})', 'm^2/s', fit.d_m2_s,
                           _join_comments('through-origin slope; mode-resolved', vh_note,
                                          comment_for(fit, ('d_m2_s', 'rh_reliable'),
                                                      whole=True))),
            _scalar_column(f'D SE ({label})', 'm^2/s', fit.d_se,
                           se_note('d_m2_s', 'd_se', flag='rh_reliable')),
            _scalar_column(f'Rh ({label})', 'nm', fit.rh_nm,
                           _join_comments('apparent, mode-resolved '
                                          '(Stokes-Einstein on the family D)', vh_note,
                                          comment_for(fit, ('rh_nm', 'rh_reliable'),
                                                      whole=True))),
            _scalar_column(f'Rh SE ({label})', 'nm', fit.rh_se,
                           se_note('rh_nm', 'rh_se', flag='rh_reliable')),
            _scalar_column(f'R^2 ({label})', '', fit.r_squared),
            _scalar_column(f'Intercept ({label})', '1/s', fit.intercept_s_inv),
            _scalar_column(f'Intercept (relative) ({label})', '',
                           fit.intercept_relative),
            _scalar_column(f'Passes through origin ({label})', '',
                           fit.is_diffusive, verdict_note),
            _scalar_column(f'a ({label})', 'm^2/s', fit.a_m2_s,
                           _join_comments('Gamma = a q^2 + b q^4 (deviation form)',
                                          comment_for(fit, ('a_m2_s',)))),
            _scalar_column(f'a SE ({label})', 'm^2/s', fit.a_se, se_note('a_m2_s', 'a_se')),
            _scalar_column(f'b ({label})', 'm^4/s', fit.b_m4_s,
                           _join_comments('Gamma = a q^2 + b q^4 (deviation form)',
                                          comment_for(fit, ('b_m4_s',)))),
            _scalar_column(f'b SE ({label})', 'm^4/s', fit.b_se, se_note('b_m4_s', 'b_se')),
        ]

    if amplitude is not None:
        columns += [
            OriginColumn('Angle (amplitude report)', 'deg',
                         'angles shared by the fast and slow families',
                         np.asarray(amplitude.angles_deg, dtype=float)),
            OriginColumn('sin^2(theta/2)', '', '',
                         np.asarray(amplitude.sin2_half_theta, dtype=float)),
            OriginColumn('A_f', '', 'fast-mode amplitude fraction',
                         np.asarray(amplitude.fast_fraction, dtype=float)),
            OriginColumn('A_s', '', 'slow-mode amplitude fraction',
                         np.asarray(amplitude.slow_fraction, dtype=float)),
            OriginColumn('A_f/A_s', '',
                         _join_comments('no SE: single-run amplitudes',
                                        comment_for(amplitude, ('ratio_f_over_s',),
                                                    whole=True)),
                         np.asarray(amplitude.ratio_f_over_s, dtype=float)),
            _scalar_column(
                'Fast amplitude q-flat', '', amplitude.fast_flat,
                _join_comments(f'(max-min)/mean of A_f <= '
                               f'{amplitude.flatness_rel_threshold:g}'
                               f' (house diagnostic threshold)',
                               comment_for(amplitude, ('fast_flat',)))),
        ]

    return write_origin_csv(file_path, columns, delimiter)


def export_mode_partition(partition, guinier_fits, file_path: str, *,
                          delimiter: str = ',') -> str:
    """Export a mode-resolved intensity partition (+ amplitude-Guinier fits).

    Two blocks, per the locked conventions (flags never change row counts;
    caveats ride the affected columns' Comments cells):

      * per-angle arrays -- the total excess Rayleigh ratio, the picked modes'
        closure (sum of matched fractions), and each mode's partitioned series
        dR_i = A_i * dR (`partition`, a ModePartitionResult). Per-mode
        intensities are single-run ill-posed-amplitude quantities and carry NO
        SE columns (invariant 8); every per-mode column carries the partition
        provenance so a number cannot be read without its origin.
      * per-mode amplitude-Guinier scalars -- one labeled block per
        AmplitudeGuinierResult in `guinier_fits` ({label: fit},
        insertion-ordered; None/{} for none). Every Rg is APPARENT and
        MODE-RESOLVED, its resolution floor and validity limits travel WITH
        it, and its SE (when present) is the multi-angle regression SE,
        statistical only.
    """
    # The partition's notes about the whole set (the separation verdict among them) ride
    # the Angle and closure columns once each; those about the closure ride the closure
    # column, and those about the partitioned series every per-mode column.
    whole_note = comment_for(partition, whole=True)
    # q^2 and the scale flags live on the per-mode Rayleigh clones; every
    # clone shares the parent's, so the first is the column's.
    first = partition.per_mode[0].rayleigh
    columns = [
        OriginColumn('Angle', 'deg',
                     _join_comments('one concentration partitioned by mode '
                                    'amplitudes', whole_note),
                     np.asarray(partition.angles_deg, dtype=float)),
        OriginColumn('q^2', 'nm^-2', 'scattering vector squared',
                     np.asarray(first.q2_nm2, dtype=float)),
        OriginColumn('dR (total)', 'cm^-1',
                     _join_comments('unpartitioned excess Rayleigh ratio',
                                    _rayleigh_scale_comment([first], optical=False)),
                     np.asarray(partition.excess_total_cm_inv, dtype=float)),
        OriginColumn('Sum of picked fractions', '',
                     _join_comments(
                         'closure: 1 when the picked modes are exhaustive '
                         f'(disclosure threshold {partition.closure_tol:g}); '
                         'blank where withheld',
                         comment_for(partition, ('closure_fraction',), whole=True)),
                     np.asarray(partition.closure_fraction, dtype=float)),
    ]
    per_mode_note = comment_for(partition, ('per_mode',))
    for pm in partition.per_mode:
        role = f' ({pm.role})' if pm.role else ''
        columns.append(OriginColumn(
            f'dR [{pm.label}]', 'cm^-1',
            _join_comments(
                f'partitioned intensity{role}: mode amplitude x dR; '
                f'no SE (single-run inversion amplitudes)',
                _rayleigh_scale_comment([pm.rayleigh], optical=False),
                per_mode_note,
                str(pm.rayleigh.derived_provenance or ''),
                str(getattr(pm.rayleigh, 'blank_provenance', '') or '')),
            np.asarray(pm.rayleigh.excess_rayleigh_cm_inv, dtype=float)))

    for label, fit in (guinier_fits or {}).items():
        provenance = str(fit.derived_provenance or '')
        # The SE cell carries the same floor / reliability facts as the value cell
        # beside it: a reader filtering on the SE column must not see a clean ±
        # on a size the value column flags.
        columns += [
            _scalar_column(f'Rg app ({label})', 'nm', fit.rg_app_nm,
                           _join_comments('apparent, mode-resolved '
                                          '(amplitude-Guinier; '
                                          'Theory-and-Equations-Guide §4.2.1)',
                                          comment_for(fit, ('rg_app_nm', 'rg_reliable'),
                                                      whole=True,
                                                      qualify=('rg_below_resolution',)),
                                          provenance)),
            _scalar_column(f'Rg SE ({label})', 'nm', fit.rg_se,
                           _join_comments(
                               _se_comment(fit, ('rg_app_nm', 'rg_se'), flag='rg_reliable',
                                           whole=True,
                                           qualify=('rg_below_resolution',
                                                    'se_excludes_inversion', 'se_ols')),
                               provenance)),
            _scalar_column(f'A(0) ({label})', '', fit.amplitude_q0,
                           'extrapolated q->0 amplitude (1/intercept)'),
            _scalar_column(f'R^2 ({label})', '', fit.r_squared),
            _scalar_column(f'q*Rg max ({label})', '', fit.qrg_max,
                           f'validity limit {fit.qrg_max_valid:g}'),
            _scalar_column(f'Rg resolution floor ({label})', 'nm',
                           fit.rg_resolution_floor_nm,
                           f'from the fitted q-window at '
                           f'{fit.min_relative_change:g} relative change'),
        ]

    return write_origin_csv(file_path, columns, delimiter)


def export_ddls(result, file_path: str, *, shapes=None,
                delimiter: str = ',') -> str:
    """Export a depolarized DLS analysis (DDLSResult), optionally with shape models.

    Per-angle columns (angle, q^2, the VV/VH field decay rates, the per-angle D_r,
    and qL when a rod length was given) followed by the combined scalars (D_t, D_r
    with their SEs, Rh_t, the rotational time, ...). When `shapes` is supplied (the
    dict from controller.ddls_shape: 'rod' and/or 'sphere'), the model-derived
    dimensions are appended as labeled scalar columns whose Comments cell records
    that they assume a shape -- not a direct measurement.

    Every column whose VALUE IS A FUNCTION OF D_r carries the analysis's D_r verdict in
    its Comments cell when `d_r_reliable` is False -- the preconditions failing that flag,
    the way the SLS Mw/A2 columns carry theirs. That is the per-angle values, the mean, its
    SE, tau_rot = 1/(6 D_r), AND every shape-model dimension below, each of which is
    inverted from the (D_t, D_r) pair -- a column that merely PRINTS D_r is the narrower
    set, and stopping there is how a corrupted number stops looking corrupted. tau_rot
    shows why the aggregate is not the test: it is NaN only when the MEAN D_r is
    non-positive, so one negative angle inside a positive mean yields a finite, printable
    rotational time computed from a corrupted average -- and the shape models run on that
    same mean.

    The D_t verdict follows the same rule: when `d_t_reliable` is False its preconditions
    ride D_t, its SE, Rh_t and its SE, and every shape column that is a function of D_t --
    all of them but `Sphere R(D_r)` and its SE, which read D_r alone. Neither verdict
    reaches the other's own family: D_t is the through-origin slope of Gamma_VV vs q^2, so
    VH noise or an optically isotropic scatterer cannot corrupt it, while a VV rate in doubt
    puts both in doubt and fails both verdicts. (A swapped VV/VH pairing corrupts both
    channels, which is why the analysis's note names that cause: the split is about which
    number is in doubt for the causes a reader can act on, not a claim that D_t is immune.)

    Every shape-model SE says what produced it. The rod and ellipsoid ± are Monte-Carlo
    spreads over draws from a D_t whose SE came from the user's chosen covariance
    estimator, so those cells carry the estimator (the `se_ols` qualifier) and the draw
    census (the shape's `mc_census` record); `Sphere R(D_t)=Rh SE` is a pass-through of
    that same D_t SE and carries the estimator alone. `Sphere R(D_r) SE` derives from the
    replicate SEM of D_r, which no covariance estimator touches, so it carries none. The
    shape blocks' own preconditions are their flag columns, and carry no clause.
    """
    def d_r(*attrs: str) -> str:
        """A D_r-family column's clauses: the notes about the attributes it shows and
        the preconditions failing `d_r_reliable`."""
        return _value_comment(result, attrs, flag='d_r_reliable')

    def d_r_cell(attrs, description: str) -> str:
        """A D_r-family column's clauses, then its description unless the D_r verdict is
        down: the verdict outranks the description (an unphysical D_r makes the formula
        and the ± label beside it moot), a note that is not the verdict (the qL limit)
        never displaces it."""
        return _join_comments(d_r(*attrs), '' if d_r_comment else description)

    def mc_se(shape, se_attr: str) -> str:
        """A Monte-Carlo shape ±: its estimator and its draw census, or the bare
        label when the result carries no census (no SEs were given)."""
        return _join_comments(comment_for(shape, (se_attr,), qualify=('se_ols',)),
                              '' if has_note(shape, 'mc_census') else 'Monte-Carlo')

    d_r_comment = d_r()
    # The D_t verdict's clauses, for a column that is a function of D_t without showing it.
    d_t_comment = _value_comment(result, (), flag='d_t_reliable')
    # A shape column inverted from both: the two verdicts are independent, so joined.
    shape_comment = _join_comments(d_r_comment, d_t_comment)
    # All three SEs are withheld together on a single-angle run (invariant 8).
    se_withheld = (_DDLS_SINGLE_ANGLE_NO_SE
                   if getattr(result, 'method', '') == 'single-angle' else '')
    # The notes about the whole analysis ride the two decay-rate columns, the values they
    # qualify: which runs were fitted under the Correlogram tab's region, which were left
    # out of an angle's average. Never the D_t family, which a D_r verdict must not reach.
    rate_note = comment_for(result, whole=True)
    columns = [
        OriginColumn('Angle', 'deg', '', np.asarray(result.angles_deg, dtype=float)),
        OriginColumn('q^2', 'm^-2', 'scattering vector squared',
                     np.asarray(result.q2_m2, dtype=float)),
        OriginColumn('Gamma_VV', '1/s',
                     _join_comments('polarized field decay rate', rate_note),
                     np.asarray(result.gamma_vv_s_inv, dtype=float)),
        OriginColumn('Gamma_VH', '1/s',
                     _join_comments('depolarized field decay rate', rate_note),
                     np.asarray(result.gamma_vh_s_inv, dtype=float)),
        OriginColumn('D_r (per angle)', 'rad^2/s',
                     d_r_cell(('d_r_per_angle',), '(Gamma_VH - Gamma_VV)/6'),
                     np.asarray(result.d_r_per_angle, dtype=float)),
    ]
    if result.qL is not None:
        columns.append(OriginColumn('qL', '', 'single-exponential valid if < 3',
                                    np.asarray(result.qL, dtype=float)))
    columns += [
        _scalar_column('D_t', 'm^2/s', result.d_t_m2_s,
                       _join_comments('from VV, through-origin slope',
                                      comment_for(result, ('d_t_m2_s', 'd_t_reliable')))),
        _scalar_column('D_t SE', 'm^2/s', result.d_t_se,
                       comments=se_withheld or _se_comment(
                           result, ('d_t_m2_s', 'd_t_se'), 'statistical only')),
        _scalar_column('D_r', 'rad^2/s', result.d_r_rad2_s,
                       d_r_cell(('d_r_rad2_s',), 'mean of per-angle values')),
        # The D_r verdict outranks the withheld-SE note here: an unphysical D_r makes its
        # SE meaningless whether or not one was computed, and the single-angle case is
        # already stated on the two columns beside it.
        _scalar_column('D_r SE', 'rad^2/s', result.d_r_se,
                       d_r_cell(('d_r_rad2_s', 'd_r_se'), se_withheld or 'statistical only')),
        _scalar_column('Rh_t', 'nm', result.rh_t_nm,
                       _join_comments('Stokes radius from D_t',
                                      comment_for(result, ('rh_t_nm',)))),
        _scalar_column('Rh_t SE', 'nm', result.rh_t_se,
                       comments=se_withheld or _se_comment(
                           result, ('rh_t_nm', 'rh_t_se'), 'statistical (over angles)')),
        _scalar_column('tau_rot', 's', result.rotational_time_s,
                       d_r_cell(('rotational_time_s',), '1 / (6 D_r)')),
        _scalar_column('N angles', '', result.n_angles),
        _scalar_column('Method', '', result.method),
        _scalar_column('Single-exponential valid', '', result.single_exponential_valid),
    ]
    if shapes:
        rod = shapes.get('rod')
        if rod is not None:
            def rod_se_note(se_attr, rod=rod):
                return _join_comments(shape_comment, se_withheld or mc_se(rod, se_attr))
            columns += [
                _scalar_column('Rod L', 'nm', rod.length_nm,
                               _join_comments(shape_comment, 'MODEL: rigid cylinder '
                                              '(Theory-and-Equations-Guide §3.6.5), '
                                              'not measured')),
                _scalar_column('Rod L SE', 'nm', rod.length_se, rod_se_note('length_se')),
                _scalar_column('Rod d', 'nm', rod.diameter_nm, shape_comment),
                _scalar_column('Rod d SE', 'nm', rod.diameter_se,
                               rod_se_note('diameter_se')),
                _scalar_column('Rod aspect p', '', rod.aspect_ratio_p, shape_comment),
                _scalar_column('Rod aspect p SE', '', rod.aspect_ratio_se,
                               rod_se_note('aspect_ratio_se')),
                _scalar_column('Rod p in 2-30', '', rod.in_valid_range, shape_comment),
                _scalar_column('Rod converged', '', rod.converged, shape_comment),
            ]
        sphere = shapes.get('sphere')
        if sphere is not None:
            columns += [
                _scalar_column('Sphere R(D_r)', 'nm', sphere.radius_rot_nm,
                               _join_comments(d_r_comment, 'MODEL: sphere '
                                              '(Stokes-Einstein-Debye), not measured')),
                _scalar_column('Sphere R(D_r) SE', 'nm', sphere.radius_rot_se,
                               _join_comments(d_r_comment,
                                              se_withheld or 'statistical (over angles)')),
                _scalar_column('Sphere R(D_t)=Rh', 'nm', sphere.radius_trans_nm,
                               d_t_comment),
                _scalar_column('Sphere R(D_t)=Rh SE', 'nm', sphere.radius_trans_se,
                               _join_comments(d_t_comment, se_withheld or _se_comment(
                                   sphere, ('radius_trans_nm', 'radius_trans_se'),
                                   'statistical (over angles)', qualify=('se_ols',)))),
                _scalar_column('Sphericity ratio', '', sphere.sphericity_ratio,
                               _join_comments(shape_comment, 'R(D_r)/Rh; 1 = sphere')),
                _scalar_column('Sphere consistent', '', sphere.is_consistent,
                               shape_comment),
            ]
        ellipsoids = shapes.get('ellipsoid')
        if ellipsoids is not None:
            # A given D_t/D_r admits up to three ellipsoids (1 prolate + 2 oblate).
            # Emit a FIXED three-slot block (blank unused slots) so the column set is
            # input-independent, led by the count + degeneracy flag.
            fitted = [e for e in ellipsoids if e.converged]
            columns += [
                _scalar_column('Ellipsoid solutions', '', len(fitted),
                               _join_comments(shape_comment,
                                              'MODEL: ellipsoid of revolution (Perrin; '
                                              'Theory-and-Equations-Guide §3.6.6), '
                                              'not measured')),
                _scalar_column('Ellipsoid degenerate', '',
                               bool(fitted) and fitted[0].degenerate, shape_comment),
            ]
            for k in range(3):
                e = fitted[k] if k < len(fitted) else None
                lbl = f'Ellipsoid {k + 1}'
                def e_se_note(se_attr, e=e):
                    return _join_comments(shape_comment, se_withheld or mc_se(e, se_attr))
                columns += [
                    _scalar_column(f'{lbl} type', '',
                                   ('prolate' if e.is_prolate else 'oblate')
                                   if e is not None else None, shape_comment),
                    _scalar_column(f'{lbl} a (polar)', 'nm',
                                   e.semi_axis_polar_nm if e is not None else None,
                                   shape_comment),
                    _scalar_column(f'{lbl} a SE', 'nm',
                                   e.semi_axis_polar_se if e is not None else None,
                                   e_se_note('semi_axis_polar_se')),
                    _scalar_column(f'{lbl} b (equatorial)', 'nm',
                                   e.equatorial_radius_nm if e is not None else None,
                                   shape_comment),
                    _scalar_column(f'{lbl} b SE', 'nm',
                                   e.equatorial_radius_se if e is not None else None,
                                   e_se_note('equatorial_radius_se')),
                    _scalar_column(f'{lbl} p', '',
                                   e.axial_ratio_p if e is not None else None,
                                   shape_comment),
                    _scalar_column(f'{lbl} p in 0.1-10', '',
                                   e.in_valid_range if e is not None else None,
                                   shape_comment),
                ]
    return write_origin_csv(file_path, columns, delimiter)


def export_concentration_extrapolation(result, file_path: str,
                                       delimiter: str = ',') -> str:
    """Export a D-vs-c concentration extrapolation (ConcentrationExtrapolationResult).

    The kD column's Comments cell carries the preconditions failing `kd_reliable` (D0 <=
    0 sign flip or an ill-conditioned design), the way the SLS Mw/A2 exporters carry
    theirs; a healthy fit leaves it unmarked. The D0 and Rh0 columns carry those failing
    the Rh0 verdict (`rh_reliable`: a flagged per-point fit inside the line, an
    extrapolated Rh0 outside the size range, no viscosity). The result keeps one verdict
    per flag, so each column is gated on its own flag.

    A DROPPED concentration (no usable D -- a NaN in `d_values_m2_s`) is disclosed in the
    `D_app` and `D0` Comments cells (owner decision): the result's `unusable_rate_dropped`
    record, and, when too few remain to extrapolate at all, its `too_few_points`
    precondition on D0, Rh0 and kD. It has to be said here and not left to the GUI: the
    export outlives the session, and a blank `D_app` cell beside a `D0` fitted over a
    quietly smaller point set is exactly what the Comments cell exists for -- the same
    reasoning that puts "uncalibrated, arbitrary scale" there.

    Whatever the fit recorded about the whole set is relayed onto the `Concentration`
    column, which is the input those notes are about; what it recorded about the
    per-point values rides `D_app`."""
    d_app = np.asarray(result.d_values_m2_s, dtype=float)

    def rh0(*attrs: str, codes=()) -> str:
        return _value_comment(result, attrs, flag='rh_reliable', codes=codes)

    def kd(*attrs: str) -> str:
        return _value_comment(result, attrs, flag='kd_reliable')

    se_label = 'statistical (over concentrations)'
    columns = [
        OriginColumn('Concentration', 'g/mL',
                     comment_for(result, ('concentrations_g_per_mL',), whole=True),
                     np.asarray(result.concentrations_g_per_mL, dtype=float)),
        OriginColumn('D_app', 'm^2/s',
                     _join_comments('apparent diffusion coefficient',
                                    comment_for(result, ('d_values_m2_s',))), d_app),
        _scalar_column('D0', 'm^2/s', result.d0_m2_s,
                       _join_comments('c -> 0',
                                      rh0('d0_m2_s', codes=('unusable_rate_dropped',)))),
        _scalar_column('D0 SE', 'm^2/s', result.d0_se,
                       comments=_se_comment(result, ('d0_m2_s', 'd0_se'), se_label,
                                            flag='rh_reliable')),
        _scalar_column('Rh0', 'nm', result.rh0_nm, comments=rh0('rh0_nm')),
        _scalar_column('Rh0 SE', 'nm', result.rh0_se,
                       comments=_se_comment(result, ('rh0_nm', 'rh0_se'), se_label,
                                            flag='rh_reliable')),
        _scalar_column('kD', 'mL/g', result.kd_mL_per_g,
                       comments=kd('kd_mL_per_g') or 'diffusion interaction parameter'),
        _scalar_column('kD SE', 'mL/g', result.kd_se,
                       comments=_se_comment(result, ('kd_mL_per_g', 'kd_se'), se_label,
                                            flag='kd_reliable')),
        _scalar_column('Slope', 'm^2/s/(g/mL)', result.slope),
        _scalar_column('R^2', '', result.r_squared),
        # The same provenance pair its Gamma-vs-q^2 sibling carries -- both results hold
        # the fields, so exporting them from one exporter and not the other would leave
        # exactly the asymmetry rule 10 exists to close.
        _scalar_column('Gamma source', '', getattr(result, 'gamma_source', '') or ''),
        _scalar_column('Cumulant method', '',
                       getattr(result, 'cumulant_method', None) or ''),
        _scalar_column('Cumulant fit cutoff', '',
                       _blank_if_none(getattr(result, 'cumulant_fit_cutoff', None))),
        _cumulant_weighting_row(result, ' g/mL'),
        *_cumulant_record_rows(result, ' g/mL'),
    ]
    return write_origin_csv(file_path, columns, delimiter)


# ===========================================================================
# SLS exporters
# ===========================================================================

# The condition under which the calibration-free estimator's ABSOLUTE A2 exists at all.
# 2 A2 Mw is genuinely calibration-free, but A2 = (2 A2 Mw)/(2 Mw) inherits the scale of
# the Mw it divides by, so the controller supplies that Mw only when it is trustworthy and
# withholds A2 otherwise (see Controller.run_calibration_free_a2). Deliberately NOT the
# `uncalibrated` clause: that clause asserts a PRINTED number is on an arbitrary scale,
# and in the withheld case the cell is empty -- there is no number for it to be about.
_CALFREE_A2_MW_CONDITION = 'only if a calibrated or hand-entered Mw is supplied'

# The APPARENT default of a value fitted at one finite angle (the single-angle ladder, the
# calibration-free A2): it still carries P(q). Invariant 7, one wording for both files.
_SINGLE_ANGLE_APPARENT = 'APPARENT, single angle (contains P(q); not extrapolated to q -> 0)'

# Why a DDLS ± column is empty on a single-angle run. Invariant 8: a ± never ships from
# one correlogram (its lag channels are correlated -- Schaetzel 1990), so `analyze_ddls`
# withholds all three SEs when `method == 'single-angle'`. The cells must say that rather
# than keep their multi-angle default, which would assert a statistical SE that was
# deliberately not computed. Keyed on `method`, a structured field the file already
# exports, never on the result's prose.
_DDLS_SINGLE_ANGLE_NO_SE = ('no ± from a single angle: one correlogram per channel, so a '
                            '± needs repeats or multiple angles '
                            '(Theory-and-Equations-Guide §6)')

def _derived_note(results) -> str:
    """The derivation provenance these Rayleigh results carry, or ''.

    One string for the whole set, since the Comments cell describes a COLUMN: the one
    record every contributing result agrees on, stated alone; otherwise the set says it
    is mixed and then names every distinct record, so each derived point's runs and its
    count-rate spread survive the pooling (the way `_blank_note` names every blank). It
    agrees with the reduction a pooled fit's own provenance takes, enforced by
    `tests/test_derived_sls.py::test_a_pooled_fit_keeps_each_derived_points_runs_and_spread`.
    """
    results = list(results)
    notes = {str(getattr(r, 'derived_provenance', '') or '') for r in results}
    notes.discard('')
    if not notes:
        return ''
    # TWO counts, and they answer different questions: how many DISTINCT strings
    # there are, and how many results CARRY one. Deduplicating collapses the second,
    # so comparing the set's size against len(results) can only ever be true for a
    # single result -- which is why the wrong form reads as tested.
    carried = sum(1 for r in results
                  if str(getattr(r, 'derived_provenance', '') or ''))
    if len(notes) == 1 and carried == len(results):
        return notes.pop()
    return '; '.join(('some points derived from DLS count rates', *sorted(notes)))


def _blank_note(results) -> str:
    """Every distinct 'which solvent blank' record these Rayleigh results carry, or ''.

    Unlike `_derived_note` there is no mixed-set sentence: a blank note names a file,
    and a column built from two blanks is described by naming both. Empty when every
    point used the sample's reference blank (the silent default).
    """
    notes = sorted({str(getattr(r, 'blank_provenance', '') or '') for r in results}
                   - {''})
    return '; '.join(notes)


def _rayleigh_scale_comment(results, *, optical: bool = True) -> str:
    """The scale clauses of a column built from these Rayleigh results, '' on the
    silent calibrated default.

    The typed fields a Rayleigh result's scale notes are minted from are read here
    (a hand-built series carries the fields and no notes) and stated in the registry's
    words: `uncalibrated` when any result lacks an established absolute scale for want of
    a calibration constant, `no_standard_index` when any applied one with the standard's
    refractive index blank (`standard_index_known`), and `no_dn_dc` when any lacks a
    usable dn/dc -- the last only for a column that carries the optical constant K
    (`optical`): dn/dc does not enter dR, so the excess-Rayleigh column takes the
    calibration clauses alone. Independent facts, joined rather than ranked.
    """
    results = list(results)
    lacking = [r for r in results if not getattr(r, 'calibrated', True)]
    return _join_comments(
        _registry_clause('uncalibrated')
        if any(getattr(r, 'standard_index_known', True) for r in lacking) else '',
        '' if all(getattr(r, 'standard_index_known', True) for r in lacking)
        else _registry_clause('no_standard_index'),
        '' if not optical or all(getattr(r, 'dn_dc_supplied', True) for r in results)
        else _registry_clause('no_dn_dc'))


def _provenance_note(result) -> str:
    """Where a fit's points came from, when not an instrument's own static reading
    against the sample's reference blank: the derivation record (intensities computed
    from DLS count rates) and the solvent-blank record, joined. Both are records the
    fit copied from the Rayleigh results it consumed; '' in the common case.
    """
    return _join_comments(str(getattr(result, 'derived_provenance', '') or ''),
                          str(getattr(result, 'blank_provenance', '') or ''))


def _ordinate_scale_note(result, default: str, *, optical: bool = True) -> str:
    """Comments cell for a fit's own ordinate column.

    A Kc/dR ordinate carries BOTH scales -- the Rayleigh calibration (through dR) and
    the optical constant K (through dn/dc) -- so an unknown either way makes the
    printed numbers arbitrary, and the cell carries the fit's `uncalibrated`,
    `no_standard_index` (k_c applied with the standard's index blank, an arbitrary scale
    all the same) and `no_dn_dc` preconditions, which are about the scale-bearing values
    (Mw, A2) and not the ordinate that carries the scale, so they are named here by code.
    Without a marker a re-import to Origin reads the numbers as absolute. `optical=False`
    is for an ordinate that does not carry K (Guinier's ln(dR)): it takes the calibration
    clauses and the provenance, never the dn/dc one. One helper for every per-fit
    exporter, so they cannot drift apart from each other or from the Zimm/Rayleigh pair.
    """
    calibration = ('uncalibrated', 'no_standard_index')
    codes = calibration + ('no_dn_dc',) if optical else calibration
    return _join_comments(
        comment_for(result, codes=codes),
        # Independent of both scale causes and joined with them rather than ranked:
        # a derived series can also be uncalibrated.
        _provenance_note(result)) or default


def _join_comments(*parts: str) -> str:
    """Join the non-empty Comments fragments with '; '.

    Used where two INDEPENDENT facts can hold at once -- a dropped point and an
    uncalibrated scale are about different things, and `a or b` would silently withhold
    the second whenever the first fired.
    """
    return '; '.join(p for p in parts if p)


def export_debye(result, file_path: str, delimiter: str = ',') -> str:
    """Export a single-concentration Debye analysis (DebyeResult, apparent).

    The Mw column's Comments cell carries the preconditions failing `mw_reliable` --
    uncalibrated (arbitrary scale), no dn/dc, an unmeasured blank, an ill-conditioned
    design -- and the notes about Mw. The Rg column carries the size's own verdict (the
    preconditions failing `rg_reliable`): not scale-gated, so it names only a degenerate
    design, an unextractable size or an unmeasured blank."""
    mw = _value_comment(result, ('mw_apparent_g_per_mol',), flag='mw_reliable')
    rg = _value_comment(result, ('rg_apparent_nm',), flag='rg_reliable')

    # Anything the fit said about the whole fit -- the non-positive-ordinate drop. The
    # arrays below are the POST-drop ones held on the result, so this cell is the file's
    # only trace that an angle was removed (the Zimm sibling's rule).
    fit_note = comment_for(result, ('kc_over_dR',), whole=True)
    columns = [
        OriginColumn('q^2', 'nm^-2', '', np.asarray(result.q2_nm2, dtype=float)),
        OriginColumn('Kc/dR', 'mol/g',
                     _join_comments(fit_note, _ordinate_scale_note(result, 'Debye ordinate')),
                     np.asarray(result.kc_over_dR, dtype=float)),
        _scalar_column('Concentration', 'g/mL', result.concentration_g_per_mL),
        _scalar_column('Mw (apparent)', 'g/mol', result.mw_apparent_g_per_mol,
                       comments=mw or 'APPARENT, single concentration'),
        _scalar_column('Mw SE', 'g/mol', result.mw_apparent_se,
                       comments=_se_comment(result, ('mw_apparent_g_per_mol', 'mw_apparent_se'),
                                            flag='mw_reliable')),
        _scalar_column('Rg (apparent)', 'nm', result.rg_apparent_nm, comments=rg),
        _scalar_column('Rg SE', 'nm', result.rg_apparent_se,
                       comments=_se_comment(result, ('rg_apparent_nm', 'rg_apparent_se'),
                                            flag='rg_reliable')),
        _scalar_column('Intercept', 'mol/g', result.intercept_mol_per_g),
        _scalar_column('Slope', 'mol nm^2/g', result.slope),
        # The regime the apparent-Rg verdict read, with its limit (the Zimm/Guinier shape).
        _scalar_column('qRg (max)', '', result.qrg_max,
                       f'validity limit {result.qrg_max_valid:g}'),
        _scalar_column('R^2', '', result.r_squared),
        _scalar_column('Is apparent', '', result.is_apparent),
    ]
    return write_origin_csv(file_path, columns, delimiter)


def export_guinier(result, file_path: str, delimiter: str = ',') -> str:
    """Export a single-concentration Guinier analysis (GuinierResult, apparent).

    The Mw column carries the preconditions failing `mw_reliable` in its Comments cell
    (the uncalibrated marker, a degenerate design, a fit outside the Guinier regime). Rg
    comes from the slope and is not scale-gated; its own verdict (the preconditions
    failing `rg_reliable` -- a degenerate design or a fit outside the Guinier regime)
    lands on the Rg column, and the qRg (max) column names the validity limit that
    verdict was judged against.

    The ln(dR) ordinate is on the excess-Rayleigh scale, so an uncalibrated run offsets
    it by the unknown scale (its intercept, hence the apparent Mw, moves with it) and
    the column carries the calibration marker plus the points' provenance; it does not
    carry K, so the dn/dc marker stays off it (`_ordinate_scale_note(optical=False)`).
    """
    mw = _value_comment(result, ('mw_apparent_g_per_mol',), flag='mw_reliable')
    columns = [
        OriginColumn('q^2', 'nm^-2', '', np.asarray(result.q2_nm2, dtype=float)),
        OriginColumn('ln(dR)', '',
                     _join_comments(comment_for(result, ('ln_excess_rayleigh',), whole=True),
                                    _ordinate_scale_note(result, 'log excess Rayleigh ratio',
                                                         optical=False)),
                     np.asarray(result.ln_excess_rayleigh, dtype=float)),
        _scalar_column('Concentration', 'g/mL', result.concentration_g_per_mL),
        _scalar_column('Rg (apparent)', 'nm', result.rg_nm,
                       comments=_value_comment(result, ('rg_nm',), flag='rg_reliable')),
        _scalar_column('Rg SE', 'nm', result.rg_se,
                       comments=_se_comment(result, ('rg_nm', 'rg_se'), flag='rg_reliable')),
        _scalar_column('Mw (apparent)', 'g/mol', result.mw_apparent_g_per_mol,
                       comments=mw or 'APPARENT, single concentration'),
        _scalar_column('Mw SE', 'g/mol', result.mw_apparent_se,
                       comments=_se_comment(result, ('mw_apparent_g_per_mol', 'mw_apparent_se'),
                                            flag='mw_reliable')),
        _scalar_column('Intercept', '', result.intercept, 'ln(dR(0))'),
        _scalar_column('Slope', 'nm^2', result.slope, '-Rg^2/3'),
        _scalar_column('qRg (max)', '', result.qrg_max,
                       f'validity limit {result.qrg_max_valid:g}'),
        _scalar_column('Guinier valid', '', result.guinier_valid),
        _scalar_column('R^2', '', result.r_squared),
        _scalar_column('Is apparent', '', result.is_apparent),
    ]
    return write_origin_csv(file_path, columns, delimiter)


def export_single_angle(result, file_path: str, delimiter: str = ',') -> str:
    """Export a single-angle, single-concentration apparent Mw (SingleAngleResult).

    The Mw column carries the preconditions failing `mw_reliable` in its Comments cell
    (the uncalibrated marker, the value then being on an arbitrary scale; a subnormal or
    non-positive divisor), else the APPARENT default -- mirroring Debye/Guinier. There is
    no ordinate column here, so the points' provenance (a DLS-derived intensity, a
    non-default solvent blank) rides the Mw cell too: it is the one number the file has,
    and it was computed from them.
    """
    mw_comment = _join_comments(
        _value_comment(result, ('mw_apparent_g_per_mol',), flag='mw_reliable')
        or 'APPARENT, single angle + single concentration',
        _provenance_note(result))
    columns = [
        _scalar_column('Angle', 'deg', result.angle_deg),
        _scalar_column('q^2', 'nm^-2', result.q2_nm2),
        _scalar_column('Concentration', 'g/mL', result.concentration_g_per_mL),
        _scalar_column('Mw (apparent)', 'g/mol', result.mw_apparent_g_per_mol,
                       comments=mw_comment),
        _scalar_column('Is apparent', '', result.is_apparent),
    ]
    return write_origin_csv(file_path, columns, delimiter)


def export_single_angle_ladder(result, file_path: str, delimiter: str = ',') -> str:
    """Export a single-angle concentration-ladder fit (SingleAngleLadderResult, apparent).

    The Debye-plot-at-one-angle sibling of `export_debye`: the ordinate column carries
    both scale markers and the points' provenance (`_ordinate_scale_note`), the Mw and
    A2 columns carry the preconditions failing their OWN flags (uncalibrated / no dn/dc
    / ill-conditioned) else the APPARENT default, and the SE beneath each inherits them.
    The 2 A2 Mw product is scale-free (the calibration-free construction's quantity,
    read here off slope/intercept), so its cells carry the preconditions failing its own
    flag (a degenerate design, an unmeasured blank), else the same APPARENT default,
    never a scale clause.
    """
    def flagged(attr: str, flag: str, *, whole: bool = False) -> str:
        return _value_comment(result, (attr,), flag=flag, whole=whole)

    def flagged_se(attrs, flag: str, *descriptions: str, whole: bool = False) -> str:
        return _se_comment(result, attrs, *descriptions, flag=flag, whole=whole)

    apparent = _SINGLE_ANGLE_APPARENT
    # The non-positive-Kc/dR drop: the arrays held on the result are POST-drop, so the
    # cell is the file's only trace that a concentration was removed.
    fit_note = comment_for(result, ('concentrations_g_per_mL',), whole=True)
    columns = [
        OriginColumn('Concentration', 'g/mL', fit_note,
                     np.asarray(result.concentrations_g_per_mL, dtype=float)),
        OriginColumn('Kc/dR', 'mol/g',
                     _join_comments(comment_for(result, ('kc_over_dR',), whole=True),
                                    _ordinate_scale_note(result, 'single-angle ordinate')),
                     np.asarray(result.kc_over_dR, dtype=float)),
        _scalar_column('Angle', 'deg', result.angle_deg),
        _scalar_column('q^2', 'nm^-2', result.q2_nm2),
        _scalar_column('Mw (apparent)', 'g/mol', result.mw_apparent_g_per_mol,
                       comments=flagged('mw_apparent_g_per_mol', 'mw_reliable') or apparent),
        _scalar_column('Mw SE', 'g/mol', result.mw_apparent_se,
                       comments=flagged_se(('mw_apparent_g_per_mol', 'mw_apparent_se'),
                                           'mw_reliable')),
        _scalar_column('A2 (apparent)', 'mol mL/g^2', result.a2_apparent_mol_mL_per_g2,
                       comments=flagged('a2_apparent_mol_mL_per_g2', 'a2_reliable')
                       or apparent),
        _scalar_column('A2 SE', 'mol mL/g^2', result.a2_apparent_se,
                       comments=flagged_se(('a2_apparent_mol_mL_per_g2', 'a2_apparent_se'),
                                           'a2_reliable')),
        _scalar_column('2 A2 Mw', '', result.two_a2_mw,
                       flagged('two_a2_mw', 'two_a2_mw_reliable', whole=True)
                       or _join_comments('slope / intercept (calibration-free)', apparent)),
        _scalar_column('2 A2 Mw SE', '', result.two_a2_mw_se,
                       comments=flagged_se(('two_a2_mw', 'two_a2_mw_se'),
                                           'two_a2_mw_reliable',
                                           'statistical; the product cancels both scales',
                                           whole=True)),
        _scalar_column('Intercept', 'mol/g', result.intercept_mol_per_g),
        _scalar_column('Slope', 'mol mL/g^2', result.slope),
        _scalar_column('R^2', '', result.r_squared),
        _scalar_column('Is apparent', '', result.is_apparent),
    ]
    return write_origin_csv(file_path, columns, delimiter)


def export_rayleigh_series(results, file_path: str, delimiter: str = ',') -> str:
    """Export a set of excess Rayleigh ratios (one RayleighRatioResult per
    concentration) as a stacked long table: one row per (concentration, angle).

    The c = 0 solvent reference is skipped. If any concentration is uncalibrated,
    the dR / Kc/dR columns carry the "uncalibrated, arbitrary scale" note in their
    Comments header (no extra rows, so the Origin import is unaffected).
    """
    samples = [r for r in results if r.concentration_g_per_mL != 0]
    if not samples:
        raise ValueError("No non-zero-concentration results to export.")
    samples.sort(key=lambda r: r.concentration_g_per_mL)

    conc, angle, q, q2, dR, kc = [], [], [], [], [], []
    for r in samples:
        n = len(np.asarray(r.angles_deg, dtype=float))
        conc.extend([r.concentration_g_per_mL] * n)
        angle.extend(list(np.asarray(r.angles_deg, dtype=float)))
        q.extend(list(np.asarray(r.q_nm_inv, dtype=float)))
        q2.extend(list(np.asarray(r.q2_nm2, dtype=float)))
        dR.extend(list(np.asarray(r.excess_rayleigh_cm_inv, dtype=float)))
        kc.extend(list(np.asarray(r.kc_over_dR_mol_per_g, dtype=float)))

    # Where the intensities came from, when they were not an instrument's own static
    # reading. Independent of the two scale clauses and joined with them rather than
    # chosen between: a derived series can also be uncalibrated. dR does NOT carry K, so
    # an absent dn/dc leaves the excess-Rayleigh column on a perfectly good scale; only
    # the ordinate that multiplies by K takes it.
    derived_note = _join_comments(_derived_note(samples), _blank_note(samples))
    dR_note = (_join_comments(_rayleigh_scale_comment(samples, optical=False), derived_note)
               or 'dR(theta)')
    kc_note = (_join_comments(_rayleigh_scale_comment(samples), derived_note)
               or 'Zimm ordinate')
    columns = [
        OriginColumn('Concentration', 'g/mL', '', conc),
        OriginColumn('Angle', 'deg', '', angle),
        OriginColumn('q', 'nm^-1', 'scattering vector', q),
        OriginColumn('q^2', 'nm^-2', '', q2),
        OriginColumn('Excess Rayleigh ratio', 'cm^-1', dR_note, dR),
        OriginColumn('Kc/dR', 'mol/g', kc_note, kc),
    ]
    return write_origin_csv(file_path, columns, delimiter)


def _zimm_row_grid(samples):
    """Row abscissa for the wide Zimm table, and where each sample's points sit on it.

    Returns ``(q2_rows, placements)``: the sorted union of every sample's q^2 (one row
    each, so no ordinate is written against a q^2 it was not measured at), and per sample
    the row index of each of its points, in the sample's own array order.

    Two q^2 values count as the same angle when they agree to 12 DECIMALS ABSOLUTE. On a
    q^2 of order 1e-4 nm^-2 that is effectively an exact match (~1e-8 relative), which is
    what this grid wants: q^2 is computed from the same wavelength and refractive index
    for every angle of a sample, so a genuinely repeated angle is bit-identical while two
    distinct angles differ by orders of magnitude more. It is deliberately NOT a claim
    about the fit's own grid levels -- `zimm_analysis` rounds the same way to LIST its
    levels but decides membership with `np.isclose`, which is far looser at this
    magnitude. Nothing needs the two to agree: the grid-intercept columns carry their own
    abscissa (`q^2 (grid)`), so they cannot mis-pair with these rows however the levels
    were grouped. A sample with SEVERAL points at one q^2 (a repeated angle) gets a row
    per repeat: the union carries each key as many times as the most-repeated sample
    needs, so widening the grid can never drop a point.
    """
    keyed = []
    for r in samples:
        q2 = np.asarray(r.q2_nm2, dtype=float)
        keyed.append([(round(float(v), 12), float(v)) for v in q2])
    multiplicity = {}
    for pts in keyed:
        counts = {}
        for key, _value in pts:
            counts[key] = counts.get(key, 0) + 1
        for key, n in counts.items():
            multiplicity[key] = max(multiplicity.get(key, 0), n)
    # NaN sorts last under this key and never equals another NaN, so each non-finite q^2
    # keeps its own row rather than collapsing a real angle onto it.
    order = sorted(multiplicity, key=lambda k: (math.isnan(k), k))
    q2_rows, slots = [], {}
    for key in order:
        slots[key] = list(range(len(q2_rows), len(q2_rows) + multiplicity[key]))
        q2_rows.extend([key] * multiplicity[key])
    placements = []
    for pts in keyed:
        used, rows = {}, []
        for key, value in pts:
            i = used.get(key, 0)
            used[key] = i + 1
            row = slots[key][i]
            rows.append(row)
            q2_rows[row] = value          # the measured value, not the rounded key
        placements.append(rows)
    return q2_rows, placements


def export_zimm(rayleigh_results, zimm_result, file_path: str,
                delimiter: str = ',') -> str:
    """Export a Zimm/Berry dataset in wide format, plus the fitted Mw/Rg/A2.

    Produces a wide table: q^2 in the first column, then one Kc/dR column per
    concentration (the Parameters row encoding each concentration), matching the
    Origin Zimm-plot workflow. The thermodynamic results from `zimm_result` are
    appended as labeled scalar columns. The per-concentration and per-angle
    extrapolated intercepts are also written, for drawing the Zimm grid lines.

    If the run is uncalibrated, the Kc/dR ordinate columns and both intercept
    columns (Kc/dR-scale) carry the "uncalibrated, arbitrary scale" note in their
    Comments header, exactly as the sibling Rayleigh exporter does -- no extra rows,
    so the Origin import is unaffected. Rg still survives (slope-derived); its own
    verdict (the preconditions failing `rg_reliable` or `qrg_regime_valid` -- a degenerate
    design, the regime its Rg is read in) joins the Rg column's Comments beside the
    fit-wide notes.

    Parameters
    ----------
    rayleigh_results : sequence of RayleighRatioResult
        One per concentration (the c = 0 solvent reference is skipped). They need NOT
        share an angle set: the q^2 column is the union of theirs and each ordinate is
        written on the row of its own q^2, with a blank where a concentration has no
        point at that angle (`_zimm_row_grid`). A standard ladder, which does share its
        angles, exports exactly the table it always did.
    zimm_result : ZimmBerryResult
    file_path : str

    Returns
    -------
    str

    Raises
    ------
    ValueError
        If no non-zero-concentration results are supplied.
    """
    samples = [r for r in rayleigh_results if r.concentration_g_per_mL != 0]
    if not samples:
        raise ValueError("No non-zero-concentration results to export.")
    samples.sort(key=lambda r: r.concentration_g_per_mL)

    # The Kc/dR ordinate and both extrapolated intercepts are on the excess-Rayleigh
    # scale, so an uncalibrated run makes them arbitrary-scale -- stamp the same note
    # the sibling Rayleigh exporter uses (export_rayleigh_series), no extra rows.
    # Rg survives an uncalibrated Zimm (slope-derived), so the export is still useful;
    # the marker just prevents a re-import to Origin being mistaken for absolute scale.
    # Every column this note lands on is Kc/dR-derived, so it also carries K -- hence
    # BOTH scale causes are joined here, unlike export_rayleigh_series, whose plain
    # excess-Rayleigh column takes the calibration one alone.
    scale = _rayleigh_scale_comment(samples)
    scale_note = _join_comments(scale, _derived_note(samples))

    # Anything the fit itself said about the whole fit -- the drop of a non-positive Kc/dR
    # among it. It has to reach the file, because the ordinate columns below are the
    # PRE-drop arrays straight off the Rayleigh results: the export shows every point
    # beside a fit that used fewer. Independent of the scale note, hence joined rather
    # than ranked (`_join_comments`). A Berry Rg read past the limit the Zimm Rg is read at
    # is a qualifier about the whole fit, failing no value's flag, so it is named here to
    # ride every value cell beside the fit-wide notes, and the Rg SE: said wherever the
    # fit's values travel.
    berry = ('berry_past_zimm_limit',)
    fit_note = comment_for(zimm_result, whole=True, qualify=berry)

    # The abscissa is the UNION of every concentration's q^2, and each ordinate sits on
    # the row of the q^2 it was measured at. Taking it from the first sample instead put
    # every later concentration's ordinates against the FIRST one's grid, so a set whose
    # concentrations were scanned at different angles exported a wrong ordinate/abscissa
    # pairing -- a wrong VALUE in the file a user re-fits in Origin, while the in-app fit
    # (which flattens (q^2, c, y) triples) and the plot stayed right. The union
    # degenerates to today's single grid whenever the concentrations share their angles,
    # which is the standard Zimm ladder, so that file is unchanged.
    q2_rows, placements = _zimm_row_grid(samples)
    ragged_note = ('' if all(len(p) == len(q2_rows) for p in placements) else
                   'the concentrations do not share one q^2 grid: each ordinate is on the '
                   'row of the q^2 it was measured at, and a blank cell means no value '
                   'was written there -- that concentration has no point at that q^2, or '
                   'its ordinate is not a usable number (a masked or dropped point)')
    columns = [OriginColumn('q^2', 'nm^-2',
                            _join_comments('scattering vector squared', ragged_note),
                            q2_rows)]
    for r, rows_for_r in zip(samples, placements, strict=True):
        c_mg = r.concentration_g_per_mL * 1000.0
        y = np.asarray(r.kc_over_dR_mol_per_g, dtype=float)
        # None in a cell renders blank (`_fmt`), which is what a q^2 this concentration
        # never visited must show -- not a neighbour's number.
        cells = [None] * len(q2_rows)
        for value, row in zip(y, rows_for_r, strict=True):
            cells[row] = float(value)
        columns.append(OriginColumn(
            f'Kc/dR (c={c_mg:.4g} mg/mL)', 'mol/g',
            # The derivation and blank notes are PER COLUMN: each concentration names
            # the record of its own points and the blank it used, never a sibling's (the
            # intercepts below, fitted over all of them, name every distinct one).
            _join_comments(fit_note, scale,
                           str(getattr(r, 'derived_provenance', '') or ''),
                           str(getattr(r, 'blank_provenance', '') or '')) or 'Zimm ordinate',
            cells,
            parameter=f'{r.concentration_g_per_mL:.6g}',
        ))

    # Extrapolated grid intercepts (for plotting the c->0 and q->0 lines).
    # The two grid intercepts are per-level fits over the SAME post-drop arrays as the
    # headline fit, so a dropped point moves them and they carry the note with it.
    columns.append(OriginColumn(
        'Intercept vs c', 'mol/g' if zimm_result.method == 'zimm' else '(mol/g)^0.5',
        _join_comments(comment_for(zimm_result, ('intercept_per_concentration',), whole=True,
                                   qualify=berry),
                       _blank_note(samples),
                       scale_note or 'q->0 intercept per concentration'),
        np.asarray(zimm_result.intercept_per_concentration, dtype=float)))
    columns.append(OriginColumn(
        'Concentration (grid)', 'g/mL', 'for the q->0 line',
        np.asarray(zimm_result.concentrations_g_per_mL, dtype=float)))
    # The c->0 intercepts are fitted over the same blank-subtracted series as the q->0
    # ones, so they name the same blanks (the sibling column above already did).
    columns.append(OriginColumn(
        'Intercept vs q^2', 'mol/g' if zimm_result.method == 'zimm' else '(mol/g)^0.5',
        _join_comments(comment_for(zimm_result, ('intercept_per_angle',), whole=True,
                                   qualify=berry),
                       _blank_note(samples),
                       scale_note or 'c->0 intercept per angle'),
        np.asarray(zimm_result.intercept_per_angle, dtype=float)))
    columns.append(OriginColumn(
        'q^2 (grid)', 'nm^-2', 'for the c->0 line',
        np.asarray(zimm_result.q2_nm2, dtype=float)))

    # Thermodynamic results. Mw and A2 carry the preconditions failing their own flags --
    # the uncalibrated marker (arbitrary scale), the ill-conditioned marker (degenerate
    # design), a withheld non-finite value -- in their Comments cell; a healthy calibrated
    # run is the silent default. Rg also takes the regime it is read in
    # (`qrg_regime_valid`): Zimm flags its Rg past the limit, Berry only notes it.
    def thermo(attrs, flag: str) -> str:
        return _value_comment(zimm_result, attrs, flag=flag, whole=True,
                              qualify=_VALUE_QUALIFIERS + berry)

    def thermo_se(attrs, flag: str, qualify=()) -> str:
        return _se_comment(zimm_result, attrs, flag=flag,
                           qualify=_SE_QUALIFIERS + tuple(qualify))

    rg_attrs = ('rg_nm', 'qrg_regime_valid')
    columns += [
        _scalar_column('Method', '', zimm_result.method),
        _scalar_column('Mw', 'g/mol', zimm_result.mw_g_per_mol,
                       comments=thermo(('mw_g_per_mol',), 'mw_reliable') or 'thermodynamic'),
        _scalar_column('Mw SE', 'g/mol', zimm_result.mw_se,
                       comments=thermo_se(('mw_g_per_mol', 'mw_se'), 'mw_reliable')),
        _scalar_column('Rg', 'nm', zimm_result.rg_nm,
                       comments=thermo(rg_attrs, 'rg_reliable')),
        _scalar_column('Rg SE', 'nm', zimm_result.rg_se,
                       comments=thermo_se(rg_attrs + ('rg_se',), 'rg_reliable',
                                          qualify=berry)),
        _scalar_column('A2', 'mol mL/g^2', zimm_result.a2_mol_mL_per_g2,
                       comments=thermo(('a2_mol_mL_per_g2',), 'a2_reliable')),
        _scalar_column('A2 SE', 'mol mL/g^2', zimm_result.a2_se,
                       comments=thermo_se(('a2_mol_mL_per_g2', 'a2_se'), 'a2_reliable')),
        # The regime the Rg verdict read, with the limit it read (the Guinier export's
        # shape): Zimm flags its Rg past it, Berry only notes it (`zimm_analysis`).
        _scalar_column('qRg (max)', '', zimm_result.qrg_max,
                       _join_comments(f'validity limit {zimm_result.qrg_max_valid:g}'
                                      + ('' if zimm_result.method == 'zimm'
                                         else '; noted, not flagged, for Berry'),
                                      comment_for(zimm_result, ('qrg_max',),
                                                  qualify=berry))),
        _scalar_column('R^2', '', zimm_result.r_squared),
        _scalar_column('Is apparent', '', zimm_result.is_apparent),
    ]
    return write_origin_csv(file_path, columns, delimiter)


def _form_factor_valid_comment(r) -> str:
    """The `Form factor valid` Comments cell of a flagged form-factor fit, for all three
    form-factor exporters: the preconditions failing the flag; else the fit's own
    preconditions that take the whole fit down (each fails `mw_reliable`: an
    ill-conditioned, collapsed or unconverged fit), since the builders name ONE reason by
    precedence and such a fit is reported for that reason, never as past a minimum -- not a
    precondition of one other value (the axial ratio, a contrast-profile Rg); and the
    minimum sentence only when the fit names no such precondition."""
    named = comment_for(r, ('form_factor_valid',))
    if named:
        return named
    masking = tuple(n.code for n in notes_in(getattr(r, 'notes', ()))
                    if isinstance(n, Note) and n.code in REGISTRY
                    and REGISTRY[n.code].kind == 'precondition'
                    and n.code not in SCALE_CODES
                    and 'mw_reliable' in REGISTRY[n.code].fails)
    return ((comment_for(r, (), codes=masking) if masking else '')
            or 'fit past a form-factor minimum; Mw/A2 unreliable')


def export_form_factor(result, file_path: str, delimiter: str = ',') -> str:
    """Export a full-angular form-factor P(q) fit (FormFactorResult).

    A long table of the fitted points (q, Kc/dR, concentration) plus the fitted scalars:
    model, Mw, Rg (and, for a sphere, the geometric radius R), and A2 (thermodynamic only).
    Mw carries the preconditions failing `mw_reliable` (uncalibrated / ill-conditioned /
    past a minimum / a withheld non-finite value) in its Comments cell exactly as
    Debye/Zimm do; Rg and R survive an uncalibrated run (angular shape, scale cancels) so
    they carry no such marker. The apparent/thermodynamic distinction is written both as
    the `Is apparent` flag and in the Mw comment default.
    """
    mw_note = _value_comment(result, ('mw_g_per_mol',), flag='mw_reliable')
    a2_note = ('' if result.is_apparent
               else _value_comment(result, ('a2_mol_mL_per_g2',), flag='a2_reliable'))
    # An apparent fit resolves no A2, so its A2 SE inherits no A2 verdict.
    a2_se_note = (_se_comment(result, ('a2_se',)) if result.is_apparent
                  else _se_comment(result, ('a2_mol_mL_per_g2', 'a2_se'), flag='a2_reliable'))
    regime = 'APPARENT, single concentration' if result.is_apparent else 'thermodynamic'

    def size(*attrs: str) -> str:
        return _value_comment(result, attrs, flag='rg_reliable')

    def size_se(*attrs: str) -> str:
        return _se_comment(result, attrs, flag='rg_reliable')
    columns = [
        OriginColumn('q', 'nm^-1', 'scattering vector',
                     np.asarray(result.q_nm_inv, dtype=float)),
        OriginColumn('Kc/dR', 'mol/g',
                     _ordinate_scale_note(result, 'form-factor ordinate'),
                     np.asarray(result.kc_over_dR, dtype=float)),
        OriginColumn('Concentration', 'g/mL', '',
                     np.asarray(result.concentrations_g_per_mL, dtype=float)),
        _scalar_column('Model', '', result.model),
        _scalar_column('Mw', 'g/mol', result.mw_g_per_mol, comments=mw_note or regime),
        _scalar_column('Mw SE', 'g/mol', result.mw_se,
                       comments=_se_comment(result, ('mw_g_per_mol', 'mw_se'),
                                            flag='mw_reliable')),
        # The size columns carry their OWN verdict: `rg_reliable` is False when the fit
        # collapsed below the size floor, where the size is the defect rather than a
        # survivor of an unknown scale (round 9c44 f1 / 943d f1). R is a fixed multiple
        # of Rg, so it carries the same note.
        _scalar_column('Rg', 'nm', result.rg_nm, comments=size('rg_nm')),
        _scalar_column('Rg SE', 'nm', result.rg_se, comments=size_se('rg_nm', 'rg_se')),
        # Geometric sphere radius (nan for the coil model -> a blank cell).
        _scalar_column('R (sphere)', 'nm', result.r_nm, comments=size('r_nm', 'rg_nm')),
        _scalar_column('R SE', 'nm', result.r_se,
                       comments=size_se('r_nm', 'rg_nm', 'r_se')),
        _scalar_column('A2', 'mol mL/g^2', result.a2_mol_mL_per_g2, comments=a2_note),
        _scalar_column('A2 SE', 'mol mL/g^2', result.a2_se, comments=a2_se_note),
        _scalar_column('qRg (max)', '', result.qrg_max),
        # Model-specific reason on the invalid case: the preconditions failing the flag
        # carry the correct text (sphere first-minimum vs Mie deep-minimum), so don't
        # hard-code sphere wording for a flagged Mie fit.
        _scalar_column('Form factor valid', '', result.form_factor_valid,
                       comments='' if result.form_factor_valid
                       else _form_factor_valid_comment(result)),
        _scalar_column('R^2', '', result.r_squared),
        _scalar_column('Is apparent', '', result.is_apparent),
    ]
    # Mie provenance: the contrast m is a fixed user input, and the recovered size/Mw are
    # model-dependent on it -- record it (and the solute RI) so the label travels into export.
    if getattr(result, 'm', None) is not None:
        columns.append(_scalar_column(
            'm (n_solute/n_medium)', '', result.m,
            comments='Mie contrast (fixed user input); size and Mw are model-dependent on it'))
        columns.append(_scalar_column('Solute refractive index', '', result.n_solute))
    return write_origin_csv(file_path, columns, delimiter)


def export_ellipsoid_form_factor(results, file_path: str, delimiter: str = ',') -> str:
    """Export an ellipsoid-of-revolution P(q) fit (a LIST of EllipsoidFormFactorResult).

    A shape-from-scattering fit is multi-valued -- a prolate and an oblate ellipsoid can
    reproduce the same P(q) -- so `results` is the list of solutions from
    `ellipsoid_form_factor_analysis`. The shared fitted points (q, Kc/dR, concentration) are
    written once; each solution then adds a block of scalar columns prefixed ``S{i}
    (prolate|oblate)``. The degeneracy is reported STRUCTURALLY (one block per solution +
    the `Degenerate` flag and the `degenerate` qualifier on the Model cell), never folded
    into a +/-. Mw carries the uncalibrated / ill-conditioned marker, the axial ratio its
    own verdict (unresolved, resolved and imprecise, or blankless), `Axial ratio resolved`
    the identifiability alone, and `Form factor valid` carries the reason a
    fit past a form-factor minimum is unreliable in their Comments cells; Rg / the axial
    ratio survive an uncalibrated run (scale cancels).
    """
    first = results[0]
    columns = [
        OriginColumn('q', 'nm^-1', 'scattering vector',
                     np.asarray(first.q_nm_inv, dtype=float)),
        OriginColumn('Kc/dR', 'mol/g',
                     _ordinate_scale_note(first, 'form-factor ordinate'),
                     np.asarray(first.kc_over_dR, dtype=float)),
        OriginColumn('Concentration', 'g/mL', '',
                     np.asarray(first.concentrations_g_per_mL, dtype=float)),
    ]
    for i, r in enumerate(results, start=1):
        kind = 'prolate' if r.is_prolate else 'oblate'
        p = f'S{i} ({kind}) '
        mw_note = _value_comment(r, ('mw_g_per_mol',), flag='mw_reliable')
        regime = 'APPARENT, single concentration' if r.is_apparent else 'thermodynamic'
        # The axial ratio's own verdict, the preconditions failing `axial_ratio_reliable`:
        # not resolved at all, resolved and imprecise, or the blank not measured.
        ax_note = comment_for(r, ('axial_ratio', 'axial_ratio_reliable'))
        semi_axis_note = _value_comment(
            r, ('semi_axis_equatorial_nm', 'semi_axis_polar_nm', 'rg_nm', 'axial_ratio',
                'axial_ratio_reliable'), flag='rg_reliable')
        a2_note = ('' if r.is_apparent
                   else _value_comment(r, ('a2_mol_mL_per_g2',), flag='a2_reliable'))
        a2_se_note = (_se_comment(r, ('a2_se',)) if r.is_apparent
                      else _se_comment(r, ('a2_mol_mL_per_g2', 'a2_se'), flag='a2_reliable'))
        columns += [
            _scalar_column(p + 'Model', '', r.model,
                           comments=_join_comments('ellipsoid',
                                                   comment_for(r, qualify=('degenerate',)))),
            _scalar_column(p + 'Mw', 'g/mol', r.mw_g_per_mol, comments=mw_note or regime),
            _scalar_column(p + 'Mw SE', 'g/mol', r.mw_se,
                           comments=_se_comment(r, ('mw_g_per_mol', 'mw_se'),
                                                flag='mw_reliable')),
            _scalar_column(p + 'Rg', 'nm', r.rg_nm,
                           comments=_value_comment(r, ('rg_nm',), flag='rg_reliable')),
            _scalar_column(p + 'Rg SE', 'nm', r.rg_se,
                           comments=_se_comment(r, ('rg_nm', 'rg_se'), flag='rg_reliable')),
            _scalar_column(p + 'Axial ratio', '', r.axial_ratio,
                           comments=ax_note or 'polar/equatorial (>1 prolate, <1 oblate)'),
            _scalar_column(p + 'Axial ratio SE', '', r.axial_ratio_se,
                           comments=_se_comment(r, ('axial_ratio', 'axial_ratio_se',
                                                    'axial_ratio_reliable'))),
            # Computed from Rg and the axial ratio, so each names both values' clauses,
            # as the sphere's R names Rg's.
            _scalar_column(p + 'Semi-axis equatorial', 'nm', r.semi_axis_equatorial_nm,
                           comments=semi_axis_note),
            _scalar_column(p + 'Semi-axis polar', 'nm', r.semi_axis_polar_nm,
                           comments=semi_axis_note),
            _scalar_column(p + 'A2', 'mol mL/g^2', r.a2_mol_mL_per_g2, comments=a2_note),
            _scalar_column(p + 'A2 SE', 'mol mL/g^2', r.a2_se, comments=a2_se_note),
            _scalar_column(p + 'qRg (max)', '', r.qrg_max),
            # The sibling both other form-factor exporters carry, with the same conditioned
            # wording: the home of the validity reason, the preconditions failing the flag.
            _scalar_column(p + 'Form factor valid', '', r.form_factor_valid,
                           comments='' if r.form_factor_valid
                           else _form_factor_valid_comment(r)),
            # Identifiability alone, `axial_ratio_resolved` ("could the data tell the
            # shapes apart"), never the verdict beside it: a resolved ratio that is
            # imprecise or blankless reads 'yes' here and names that cause on the value's
            # own cells. A ratio not resolved names why, by the two codes that say so; the
            # blank fails the verdict too, but it is no answer to this column's question.
            _scalar_column(p + 'Axial ratio resolved', '', r.axial_ratio_resolved,
                           comments='' if r.axial_ratio_resolved
                           else comment_for(r, codes=('axial_ratio_unresolved',
                                                      'no_ff_solution'))),
            _scalar_column(p + 'Degenerate', '', r.degenerate),
            _scalar_column(p + 'R^2', '', r.r_squared),
            _scalar_column(p + 'Is apparent', '', r.is_apparent),
        ]
    return write_origin_csv(file_path, columns, delimiter)


def export_core_shell_form_factor(results, file_path: str, delimiter: str = ',') -> str:
    """Export a core-shell (coated-sphere) P(q) fit (a LIST of CoreShellFormFactorResult).

    Two radii from one angular curve are often under-determined, so `results` is the list of
    solutions from `core_shell_form_factor_analysis`. The shared fitted points (q, Kc/dR,
    concentration) are written once; each solution then adds a block of scalar columns prefixed
    ``S{i}``. The degeneracy is reported STRUCTURALLY (one block per solution + the `Degenerate`
    flag), never folded into a +/-. The two contrasts m1/m2 (fixed user inputs) and the core/
    shell refractive indices are recorded so the model-dependence travels into the export. The
    contrast-weighted Rg carries NO SE (structural degeneracy) and is blank when contrast-matched;
    the radii/Rg survive an uncalibrated run (angular shape, scale cancels), Mw/A2 do not.
    """
    first = results[0]
    columns = [
        OriginColumn('q', 'nm^-1', 'scattering vector',
                     np.asarray(first.q_nm_inv, dtype=float)),
        OriginColumn('Kc/dR', 'mol/g',
                     _ordinate_scale_note(first, 'form-factor ordinate'),
                     np.asarray(first.kc_over_dR, dtype=float)),
        OriginColumn('Concentration', 'g/mL', '',
                     np.asarray(first.concentrations_g_per_mL, dtype=float)),
    ]
    for i, r in enumerate(results, start=1):
        p = f'S{i} '
        mw_note = _value_comment(r, ('mw_g_per_mol',), flag='mw_reliable')
        regime = 'APPARENT, single concentration' if r.is_apparent else 'thermodynamic'
        # The size's own verdict, whichever of its causes failed it: an undefined Rg for
        # this contrast profile, a collapsed fit (r_outer below the size floor, whose Rg is
        # finite and must not be told the contrasts cancelled), an unmeasured blank. Only
        # a healthy Rg reads the column's description.
        rg_note = (_value_comment(r, ('rg_nm',), flag='rg_reliable')
                   or 'contrast-weighted; no SE (nonlinear combo of a degenerate radius pair)')
        a2_note = ('' if r.is_apparent
                   else _value_comment(r, ('a2_mol_mL_per_g2',), flag='a2_reliable'))
        a2_se_note = (_se_comment(r, ('a2_se',)) if r.is_apparent
                      else _se_comment(r, ('a2_mol_mL_per_g2', 'a2_se'), flag='a2_reliable'))
        columns += [
            _scalar_column(p + 'Model', '', r.model,
                           comments=_join_comments('core-shell',
                                                   comment_for(r, qualify=('degenerate',)))),
            _scalar_column(p + 'Mw', 'g/mol', r.mw_g_per_mol, comments=mw_note or regime),
            _scalar_column(p + 'Mw SE', 'g/mol', r.mw_se,
                           comments=_se_comment(r, ('mw_g_per_mol', 'mw_se'),
                                                flag='mw_reliable')),
            # The two radii carry the notes about them (the contrast-weighted Rg has its
            # own verdict, `rg_note`), and their SEs those, the estimator and the label.
            _scalar_column(p + 'R_core', 'nm', r.r_core_nm,
                           comments=comment_for(r, ('r_core_nm',))),
            _scalar_column(p + 'R_core SE', 'nm', r.r_core_se,
                           comments=_se_comment(r, ('r_core_nm', 'r_core_se'))),
            _scalar_column(p + 'R_outer', 'nm', r.r_outer_nm,
                           comments=comment_for(r, ('r_outer_nm',))),
            _scalar_column(p + 'R_outer SE', 'nm', r.r_outer_se,
                           comments=_se_comment(r, ('r_outer_nm', 'r_outer_se'))),
            _scalar_column(p + 'Shell thickness', 'nm', r.shell_thickness_nm,
                           comments=comment_for(r, ('shell_thickness_nm',))),
            _scalar_column(p + 'Rg (contrast-weighted)', 'nm', r.rg_nm, comments=rg_note),
            _scalar_column(p + 'A2', 'mol mL/g^2', r.a2_mol_mL_per_g2, comments=a2_note),
            _scalar_column(p + 'A2 SE', 'mol mL/g^2', r.a2_se, comments=a2_se_note),
            _scalar_column(p + 'qRg (max)', '', r.qrg_max),
            _scalar_column(p + 'Form factor valid', '', r.form_factor_valid,
                           comments='' if r.form_factor_valid
                           else _form_factor_valid_comment(r)),
            _scalar_column(p + 'Degenerate', '', r.degenerate),
            # Conditioned on the flag, like the `Form factor valid` sibling above and the
            # ellipsoid exporter's `Axial ratio resolved`: written unconditionally it would
            # deny a defined Rg beside the number, which the degenerate small-particle
            # regime (rg_undefined False, Rg ~ 27 nm) produces. The result's own note.
            _scalar_column(p + 'Rg undefined', '', r.rg_undefined,
                           comments=comment_for(r, codes=('rg_undefined',))),
            _scalar_column(p + 'm1 (core, n_core/n_medium)', '', r.m1,
                           comments='fixed user input; size and Mw are model-dependent on it'),
            _scalar_column(p + 'm2 (shell, n_shell/n_medium)', '', r.m2,
                           comments='fixed user input; size and Mw are model-dependent on it'),
            _scalar_column(p + 'Core refractive index', '', r.n_core),
            _scalar_column(p + 'Shell refractive index', '', r.n_shell),
            _scalar_column(p + 'R^2', '', r.r_squared),
            _scalar_column(p + 'Is apparent', '', r.is_apparent),
        ]
    return write_origin_csv(file_path, columns, delimiter)


def export_scaling(quantity: str, labels, mw, y, fit, file_path: str,
                   delimiter: str = ',', model_dependent: bool = False,
                   arbitrary_scale: bool = False, degenerate: bool = False,
                   branched_labels=(), branched_mw=(), branched_y=(),
                   n_excluded: int = 0, sources=(), branched_sources=()) -> str:
    """Export a cross-sample scaling plot (Rg-Mw / Rh-Mw / A2-Mw) and its power-law fit.

    Parameters
    ----------
    quantity : str
        'rg', 'rh', or 'a2' -- selects the y-column label/units.
    labels, mw, y : sequences
        The FITTED points (one per included sample/fraction), in order.
    branched_labels, branched_mw, branched_y : sequences
        The Branched-tagged points the plot draws hollow and holds out of the fit. They
        are plotted, so the file carries them -- as their own column group beside the
        fitted points (the `export_architecture` pattern: two kinds of row, two column
        groups, never one interleaved table), so a reader can tell which points fed the
        exponent and the fitted block keeps its row count.
    n_excluded : int
        Samples/fractions dropped before the fit (no Mw, or a non-positive value). The
        tab prints this count; the file carries it as a scalar column when non-zero.
    fit : ScalingResult
        The log-log power-law fit (duck-typed: exponent, exponent_se, prefactor,
        r_squared, n_points, fit_valid).
    file_path : str
    model_dependent : bool
        A plotted Mw (any quantity) or Rg (rg axis) came from a form-factor P(q) fit, so the
        exponent rests on a shape assumption -- flagged in a caveat cell, mirroring the
        arbitrary-scale caveat (model-dependence honesty axis).
    arbitrary_scale : bool
        A plotted VALUE is on an arbitrary scale -- an uncalibrated run, a missing dn/dc, or
        both. Which value depends on the axis: the Mw on every plot, and additionally the A2
        itself on the A2 plot (the caller derives it from `_result_arbitrary_scale`, the one
        definition). The emitted column is named for the CONDITION rather than for a quantity
        or for one of the two causes, since none of the three is the whole story.
    degenerate : bool
        A plotted value rests on a user-selected DEGENERATE form-factor solution (the data can't
        distinguish the shape) -- a stronger qualifier than model_dependent alone (review-01).
    sources, branched_sources : sequences of str
        Per-point provenance, parallel to the fitted and branched blocks: where each point's Mw
        came from, and where its plotted y value came from ("Mw: Zimm extrapolation ...;
        Rg: ..."). The `export_architecture` Rg source / Rh source pattern on the axes this file
        plots -- so an APPARENT value names the condition it was measured at (a single-angle
        ladder names its angle) rather than sitting in the file as a bare number. Optional: a
        caller that passes none writes the same columns it always did.
    """
    y_label, y_units = {
        'rg': ('Rg', 'nm'), 'rh': ('Rh', 'nm'),
        'a2': ('A2', 'mol mL/g^2'),
    }.get(quantity, ('y', ''))
    # For a size exponent it is nu (Rg/Rh ~ Mw^nu); for A2 the log-log slope is the
    # raw exponent (A2 ~ Mw^slope, negative in a good solvent).
    exp_name = 'slope' if quantity == 'a2' else 'nu'
    columns = [
        OriginColumn('Sample', '', 'label', list(labels)),
        OriginColumn('Mw', 'g/mol', 'effective (user Mw wins)',
                     np.asarray(mw, dtype=float)),
        OriginColumn(y_label, y_units, '', np.asarray(y, dtype=float)),
    ]
    if len(sources):
        columns.append(OriginColumn(
            'Source', '',
            'per-point provenance of the Mw and of the plotted value; an apparent value names '
            'its condition (a single-angle ladder names its angle)', list(sources)))
    columns += [
        _scalar_column('Quantity', '', quantity),
        _scalar_column(f'Exponent ({exp_name})', '', fit.exponent),
        _scalar_column('Exponent SE', '', fit.exponent_se,
                       comments=_se_comment(fit, ('exponent', 'exponent_se'),
                                            'statistical (log-log regression)',
                                            qualify=('se_ols',))),
        _scalar_column('Prefactor', '', fit.prefactor, 'y at Mw = 1'),
        _scalar_column('R^2', '', fit.r_squared),
        _scalar_column('n points', '', fit.n_points),
        _scalar_column('Fit valid', '', fit.fit_valid),
    ]
    if len(branched_mw):
        columns += [
            OriginColumn('Branched sample', '', 'plotted hollow, excluded from the fit',
                         list(branched_labels)),
            OriginColumn('Mw (branched)', 'g/mol', 'effective (user Mw wins)',
                         np.asarray(branched_mw, dtype=float)),
            OriginColumn(f'{y_label} (branched)', y_units, 'excluded from the fit',
                         np.asarray(branched_y, dtype=float)),
        ]
        if len(branched_sources):
            columns.append(OriginColumn(
                'Source (branched)', '', 'per-point provenance, as the fitted block',
                list(branched_sources)))
    if n_excluded:
        columns.append(_scalar_column(
            'Excluded', '', int(n_excluded),
            'samples/fractions dropped before the fit: no Mw, or no positive value of the '
            'plotted quantity (Rg, Rh or A2)'))
    if arbitrary_scale:
        columns.append(_scalar_column(
            'Arbitrary scale', '', True,
            'a plotted value is on an arbitrary scale -- an uncalibrated run, a missing '
            'dn/dc, or both; the Mw on any axis, or the A2 itself on the A2 axis; fine for '
            'the exponent if the scale is consistent across the plotted samples, but the '
            'plotted values and the prefactor are not absolute'))
    if model_dependent:
        columns.append(_scalar_column(
            'Model-dependent', '', True,
            'a plotted value (an Mw, or an Rg on the Rg axis) came from a form-factor P(q) fit '
            'and assumes a particle shape; where that point is fitted (not a hollow branched '
            'point) the exponent is contingent on the shape'))
    if degenerate:
        columns.append(_scalar_column(
            'Degenerate selection', '', True,
            'a plotted value rests on a user-selected form-factor solution among several the '
            'scattering data cannot distinguish; the exponent inherits that shape ambiguity'))
    return write_origin_csv(file_path, columns, delimiter)


def _opt_col(name, units, comments, values):
    """A column of Optional[float] values, None -> NaN (blank cell on import)."""
    return OriginColumn(name, units, comments,
                        np.asarray([float('nan') if v is None else v
                                    for v in values], dtype=float))


def export_architecture(rho_rows, gh_rows, rg_law, rh_law, file_path: str,
                        delimiter: str = ',', uncalibrated_mw: bool = False,
                        model_dependent: bool = False, degenerate: bool = False,
                        branching_model: str = 'regular_star') -> str:
    """Export the Cross-Sample Architecture tab: the ρ / Chain-type table + the
    branching contraction factors g (from Rg) and h (from Rh) + the linear-reference
    Rg-Mw and Rh-Mw power-law fits, in one Origin-format CSV.

    rho_rows : list of (label, rg_nm, rh_nm, rho, shape, chain_type, rg_source,
                        rh_source, rho_se, se_estimator)
        rg_source is the plain-language Rg provenance (e.g. "Zimm extrapolation …" or
        "Sphere P(q) fit … — model-dependent (sphere P(q))"), so the model-dependence
        of a form-factor Rg travels into the export (rule-10 display surface).
        rh_source is its Rh counterpart -- per-row provenance, mirroring Rg source.
        rho_se is the ratio-propagated ± (None where either radius carries none) and
        se_estimator the covariance estimator behind it ('hc3' | 'ols'), written PER
        ROW into the `rho SE estimator` column -- the tag is stored per SE, so rows
        can differ when the Settings choice was switched between computations.
    gh_rows  : list of (label, mw, g, g_se, f_eff_g, h, h_se, f_eff_h, mw_range_str, n)
    rg_law,
    rh_law   : (exponent, prefactor, r_squared, n_points, mw_lo, mw_hi) or None
    branching_model : 'regular_star' | 'random_tri' | 'random_tetra' -- the g reading;
        a random model adds the branching-density n column (Zimm & Stockmayer Eqs. 47a/47b).
    """
    _random_labels = {'random_tri': 'random tri-functional (f=3)',
                      'random_tetra': 'random tetra-functional (f=4)'}
    columns = []
    if rho_rows:
        (labels, rg, rh, rho, shape, chain,
         rg_src, rh_src, rho_se, rho_est) = zip(*rho_rows, strict=True)
        # The estimator travels PER ROW, as the screen shows it: compute_sample_rho
        # reads the stored per-SE tags (the setting may have been switched since a
        # given SE was computed), so no column-level label can speak for every row
        # (invariant 8 clause A) -- a companion column, the Rg / Rh source pattern.
        rho_se_note = ('ratio propagation from Rg SE and Rh SE; statistical (excl. '
                       'calibration/dn-dc); blank = a radius carries no SE; the '
                       'estimator is the next column')
        rho_est_cells = [('' if se is None else
                          'classical OLS' if e == 'ols' else 'HC3')
                         for se, e in zip(rho_se, rho_est, strict=True)]
        columns += [
            OriginColumn('Sample', '', 'architecture table', list(labels)),
            OriginColumn('Chain type', '', 'linear / branched tag', list(chain)),
            OriginColumn('Rg', 'nm', '', np.asarray(rg, dtype=float)),
            # Per-row Rg provenance -- a form-factor Rg reads "… — model-dependent (<model>
            # P(q))" here, so a shape-assuming size never travels unlabeled (invariant 7).
            OriginColumn('Rg source', '', 'provenance (form-factor Rg is model-dependent)',
                         list(rg_src)),
            OriginColumn('Rh', 'nm', '', np.asarray(rh, dtype=float)),
            # The Rh counterpart: per-row provenance, mirroring the Rg source column.
            OriginColumn('Rh source', '', 'provenance', list(rh_src)),
            OriginColumn('rho = Rg/Rh', '', 'shape guide', np.asarray(rho, dtype=float)),
            _opt_col('rho SE', '', rho_se_note, rho_se),
            OriginColumn('rho SE estimator', '',
                         'covariance estimator behind rho SE, per row: HC3 (the '
                         'default) or classical OLS (the opt-in selected when that SE '
                         'was computed); blank where there is no SE',
                         rho_est_cells),
            OriginColumn('Shape', '', 'architecture guide (not a hard label)', list(shape)),
        ]
    if gh_rows:
        gl, gmw, gv, gse, feg, hv, hse, feh, grng, gn = zip(*gh_rows, strict=True)
        columns += [
            OriginColumn('Branched sample', '', 'branching factors', list(gl)),
            OriginColumn('Mw (branched)', 'g/mol', '', np.asarray(gmw, dtype=float)),
            OriginColumn('g', '', 'Rg_branched^2 / Rg_linear^2', np.asarray(gv, dtype=float)),
            _opt_col('g SE', '', 'replicate SEM (blank = single measurement)', gse),
            _opt_col('f_eff (g)', '', 'regular-star-equivalent arms from g (guide)', feg),
        ]
        # Random-branching models add the branching density n (branch units/molecule),
        # a model-dependent reading of the SAME g; the default regular-star export omits it.
        if branching_model in _random_labels:
            columns.append(_opt_col(
                'n (branch units)', '',
                f'branching density from g, assuming {_random_labels[branching_model]} '
                '(model-dependent; Theory-and-Equations-Guide §5.3.1; no separate SE -- '
                'inherits g)', gn))
        columns += [
            OriginColumn('h', '', 'Rh_branched / Rh_linear', np.asarray(hv, dtype=float)),
            _opt_col('h SE', '', 'replicate SEM (blank = single measurement)', hse),
            _opt_col('f_eff (h)', '', 'regular-star-equivalent arms from h (guide)', feh),
            OriginColumn('Mw range', '', 'in / out of the linear-reference Mw range',
                         list(grng)),
        ]
    for name, law in (('Rg', rg_law), ('Rh', rh_law)):
        if law is None:
            continue
        exp, _pre, r2, n, lo, hi = law
        columns += [
            _scalar_column(f'Linear {name}-law nu', '', exp,
                           f'{name} ~ Mw^nu (linear reference)'),
            _scalar_column(f'Linear {name}-law R^2', '', r2),
            _scalar_column(f'Linear {name} points', '', n),
            _scalar_column(f'Linear {name} Mw min', 'g/mol', lo),
            _scalar_column(f'Linear {name} Mw max', 'g/mol', hi),
        ]
    if columns:
        columns.append(
            _scalar_column('Mw arbitrary scale', '', uncalibrated_mw,
                           'a consistent Mw scale is fine for g/h; a MIXED scale state '
                           'across the tagged samples is not'))
        if model_dependent:
            columns.append(_scalar_column(
                'Model-dependent', '', True,
                'a contributing Mw (or an Rg feeding the g factor / Rg-Mw law) came from a '
                'form-factor P(q) fit (assumes a particle shape) -- the g/h factors and '
                'reference law(s) are contingent on that shape'))
        if degenerate:
            columns.append(_scalar_column(
                'Degenerate selection', '', True,
                'a contributing Rg/Mw rests on a user-selected form-factor solution among several '
                'the scattering data cannot distinguish -- the g/h factors and reference law(s) '
                'inherit that shape ambiguity'))
    if not columns:
        raise ValueError('Nothing to export: no ρ rows and no branching g/h results.')
    return write_origin_csv(file_path, columns, delimiter)


def export_calibration_free_a2(result, file_path: str, delimiter: str = ',') -> str:
    """Export a calibration-free A2 analysis (CalibrationFreeA2Result).

    The 2 A2 Mw / A2 columns carry the preconditions failing `two_a2_mw_reliable` in their
    Comments cell -- the ill-conditioned marker when the Y-vs-c design is near-degenerate
    (near-equal concentrations), an unmeasured blank.

    There is no uncalibrated case for **2 A2 Mw**: that ratio is the calibration-free
    estimator, and the unknown scale cancels in it. That does NOT extend to the absolute
    A2 = (2 A2 Mw)/(2 Mw) and its SE beneath it, which are only as absolute as the Mw
    supplied -- so they come out EMPTY when no trustworthy Mw was available, and their
    Comments name the condition (`_CALFREE_A2_MW_CONDITION`) rather than leaving a bare
    blank. The calibration-free property belongs to that one column, never to the
    exporter as a whole.

    No scale note lands anywhere, but the two PROVENANCE records ride every cell built
    from Y(c): Y(c) is a ratio of solvent-subtracted intensities, so which blank was
    subtracted (and whether the intensities were derived rather than read) moves the
    number even though the scale cancels.

    Every value is APPARENT (invariant 7): Y(c) is formed at one finite angle, so the
    product is 2 A2 Mw P(q). The 2 A2 Mw and A2 cells carry the single-angle ladder's
    APPARENT default where no precondition of `two_a2_mw_reliable` fails, and the
    `Is apparent` column says so in every case.

    Its ± columns name the estimator. The product's ± says in its own description that
    both scales cancel, so it names no `se_statistical` qualifier; the A2 ± does, since A2
    divides the product by an Mw that carries both scales, which that ± treats as exact.
    """
    # The product's own verdict: the preconditions failing `two_a2_mw_reliable`.
    verdict = _value_comment(result, (), flag='two_a2_mw_reliable')
    # Where the intensities came from and which solvent blank they were referenced
    # against (both empty in the common case): neither cancels in Y(c) the way the two
    # scales do, so the cells built from Y(c) carry them -- the shape
    # `export_mode_partition`'s fit note takes.
    provenance = _join_comments(str(getattr(result, 'derived_provenance', '') or ''),
                                str(getattr(result, 'blank_provenance', '') or ''))

    def built_from_y(*attrs: str, **kw) -> str:
        """A cell built from Y(c): the notes about it and about the whole fit (today the
        drop of a non-positive excess Rayleigh ratio -- sharper here than in the Zimm
        sibling: the arrays below are the POST-drop ones held on the result, so the cell
        is the file's only trace that a concentration was removed), and the provenance."""
        return _join_comments(comment_for(result, attrs, whole=True, **kw), provenance)

    columns = [
        OriginColumn('Concentration', 'g/mL', built_from_y('concentrations_g_per_mL'),
                     np.asarray(result.concentrations_g_per_mL, dtype=float)),
        OriginColumn('Y', '', _join_comments('[I(c_ref)/c_ref] / [I(c)/c]',
                                             built_from_y('Y')),
                     np.asarray(result.Y, dtype=float)),
        _scalar_column('Angle', 'deg', result.angle_deg),
        _scalar_column('2 A2 Mw', '', result.two_a2_mw,
                       _join_comments(_value_comment(result, ('two_a2_mw',),
                                                     flag='two_a2_mw_reliable', whole=True)
                                      or _join_comments('slope / intercept',
                                                        _SINGLE_ANGLE_APPARENT),
                                      provenance)),
        _scalar_column('2 A2 Mw SE', '', result.two_a2_mw_se,
                       comments=_se_comment(result, ('two_a2_mw', 'two_a2_mw_se'), provenance,
                                            'statistical (calibration/dn-dc-free)',
                                            flag='two_a2_mw_reliable',
                                            whole=True, qualify=('se_ols',))),
        _scalar_column('A2', 'mol mL/g^2', result.a2_mol_mL_per_g2,
                       _join_comments(built_from_y('a2_mol_mL_per_g2'),
                                      verdict or _join_comments(_SINGLE_ANGLE_APPARENT,
                                                                _CALFREE_A2_MW_CONDITION))),
        _scalar_column('A2 SE', 'mol mL/g^2', result.a2_se,
                       comments=_se_comment(
                           result, ('a2_mol_mL_per_g2', 'a2_se'), provenance, verdict,
                           f'Mw treated as exact; {_CALFREE_A2_MW_CONDITION}',
                           whole=True, qualify=('se_statistical', 'se_ols'))),
        _scalar_column('Slope', '', result.slope),
        _scalar_column('Intercept', '', result.intercept),
        _scalar_column('R^2', '', result.r_squared),
        _scalar_column('Is apparent', '', result.is_apparent),
    ]
    return write_origin_csv(file_path, columns, delimiter)
