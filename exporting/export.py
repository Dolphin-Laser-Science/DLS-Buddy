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
    # -- on Windows the default codec is cp1252 and would raise. Pure-ASCII exports
    # are byte-identical (UTF-8 is a superset), so existing files are unaffected.
    with open(file_path, 'w', newline='', encoding='utf-8') as fh:
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

    columns = [
        OriginColumn('Sample', '', 'sample identity', col('sample')),
        OriginColumn('Measurement', '', 'angle / concentration / fraction (blank '
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
    ]
    return write_origin_csv(file_path, columns, delimiter)

def export_correlogram_fit(measurement, result, file_path: str,
                           delimiter: str = ',') -> str:
    """Export a parametric DLS fit (cumulant / single / double / KWW).

    Writes the fitted delay times, the measured g2-1 over that window, the model
    curve, and the residuals, followed by the fit's scalar outputs as labeled
    length-1 columns. Works for any result object exposing fit_tau_s,
    fitted_g2m1, residuals (all four parametric DLS results do).

    Parameters
    ----------
    measurement : DLSMeasurement
        The source measurement (used to recover the measured g2-1 over the fit
        window).
    result : CumulantResult | SingleExponentialResult | DoubleExponentialResult |
             KWWResult
    file_path : str

    Returns
    -------
    str
    """
    def _mode_comment(mode) -> str:
        """One ExponentialMode's reliability note, or '' when it is not flagged.

        Read off the MODE rather than the result: the result-level flag is the roll-up
        over its modes, so using it here would stamp one mode's reason onto the other
        mode's columns.

        Placement differs BY CAUSE, which is why the mode states its cause rather than
        leaving the exporter to infer one from the note text. A missing viscosity leaves
        the rate exact and only withholds the radius, so its note rides the Rh column
        alone; an undecayed mode's RATE is itself an extrapolation, so its note also rides
        Gamma and D -- matching export_mode_map, which puts the same verdict on Gamma and
        D_app. Before the mode carried `decay_incomplete` the exporter could not tell them
        apart and took the safe half only, so a flagged rate exported bare here while the
        map exported it caveated.
        """
        return ('' if getattr(mode, 'rh_reliable', True)
                else getattr(mode, 'reliability_note', ''))

    def _rate_comment(mode) -> str:
        """The subset of `_mode_comment` that also invalidates the rate and D."""
        return (getattr(mode, 'reliability_note', '')
                if getattr(mode, 'decay_incomplete', False) else '')

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
    columns.append(_scalar_column('RMS error', '', result.rms_error))
    if cls == 'CumulantResult':
        # An order-1 cumulant has no mu2 term, so PDI is UNMEASURED (NaN), not zero
        # at all. Mark that in the Comments so the blank/NaN PDI cell is never
        # misread on import as a failed fit -- the Rh is fully reliable.
        pdi_comment = ('unmeasured: order-1 cumulant has no mu2 term'
                       if result.order < 2 else '')
        # An out-of-range size travels WITH the number, in the affected column's Comments
        # cell only -- the same treatment an uncalibrated result gets, and for the same
        # reason: a flag that can change how a value is read must never be separable from
        # the value on the way out.
        rh_comment = ('' if getattr(result, 'rh_reliable', True)
                      else getattr(result, 'reliability_note', ''))
        columns += [
            _scalar_column('Cumulant order', '', result.order),
            _scalar_column('Cumulant method', '', getattr(result, 'method', 'linear')),
            _scalar_column('Gamma', '1/s', result.gamma_s_inv,
                           _rate_comment(result)),
            _scalar_column('Rh', 'nm', result.rh_nm, rh_comment),
            _scalar_column('D', 'm^2/s', result.d_m2_s,
                           _rate_comment(result)),
            _scalar_column('PDI', '', result.pdi, pdi_comment),
            _scalar_column('PDI valid', '', result.pdi_valid, pdi_comment),
            _scalar_column('mu2', '1/s^2', result.mu2_s_inv2),
            _scalar_column('beta', '', result.beta),
            _scalar_column('baseline B', '', getattr(result, 'baseline', 0.0)),
        ]
    elif cls == 'SingleExponentialResult':
        # Same treatment as the cumulant Rh above, read off the MODE: the verdict belongs
        # to the mode, and a fit's modes can disagree (see the double branch below).
        rh_comment = _mode_comment(result.mode)
        columns += [
            _scalar_column('Gamma', '1/s', result.mode.gamma_s_inv,
                           _rate_comment(result.mode)),
            _scalar_column('Rh', 'nm', result.mode.rh_nm, rh_comment),
            _scalar_column('D', 'm^2/s', result.mode.d_m2_s,
                           _rate_comment(result.mode)),
            _scalar_column('beta', '', result.beta),
        ]
    elif cls == 'DoubleExponentialResult':
        # PER MODE, not one comment across both. The routine real case is a measured fast
        # mode beside a slow one that never decayed; a shared comment either condemns the
        # good column or clears the bad one, and a flag that can change how a value is read
        # must never be separable from -- or misattached to -- the value on the way out.
        fast_comment = _mode_comment(result.mode1)
        slow_comment = _mode_comment(result.mode2)
        columns += [
            _scalar_column('Gamma (fast)', '1/s', result.mode1.gamma_s_inv,
                           _rate_comment(result.mode1)),
            _scalar_column('Rh (fast)', 'nm', result.mode1.rh_nm, fast_comment),
            _scalar_column('Amplitude (fast)', '', result.mode1.amplitude_fraction),
            _scalar_column('Gamma (slow)', '1/s', result.mode2.gamma_s_inv,
                           _rate_comment(result.mode2)),
            _scalar_column('Rh (slow)', 'nm', result.mode2.rh_nm, slow_comment),
            _scalar_column('Amplitude (slow)', '', result.mode2.amplitude_fraction),
            _scalar_column('beta', '', result.beta),
        ]
    elif cls == 'KWWResult':
        # One flag covers both Rh channels: a KWW has one decay, reported two ways. The
        # relaxation times take the caveat only when the DECAY is what failed -- see
        # _rate_comment; a missing viscosity leaves tau_c and <tau> exact.
        rh_comment = _mode_comment(result)
        rate_comment = _rate_comment(result)
        columns += [
            _scalar_column('tau_c', 's', result.tau_c_s, rate_comment),
            _scalar_column('Stretch', '', result.stretch),
            _scalar_column('mean tau', 's', result.mean_tau_s, rate_comment),
            _scalar_column('Rh (from <tau>)', 'nm', result.rh_from_mean_tau_nm,
                           rh_comment),
            _scalar_column('Rh (from tau_c)', 'nm', result.rh_from_tau_c_nm,
                           rh_comment),
            _scalar_column('beta', '', result.beta),
        ]
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
def alpha_selection_note(selection_method: Optional[str],
                         ftest_prob_reject: Optional[float] = None, *,
                         edge_note: str = '', style: str = 'export') -> str:
    """Human string describing how a CONTIN alpha was chosen, for the given surface.

    One shared mapping behind both the CSV Comments cell (`export_distribution`,
    `style='export'`) and the GUI status note (`gui/dls_module`, `style='gui'`), so a
    new selector value is added in exactly one place instead of two parallel ladders.
    Returns '' for an unknown/None selector (NNLS/lognormal pass no method).

    Selector wording is kept verbatim to what each surface already shipped; the edge
    suffix follows the result's note (changed from an ASCII phrase to the shared
    constant when the note moved onto `ContinResult`):
      - export: "alpha by GCV", "alpha by F-test, p_reject=0.50", "alpha user-supplied";
                appends " (<edge_note>)" when the result carried a sweep-edge
                reliability note (`ContinResult.reliability_note`, the one wording every
                surface shows -- so the export can never again forget the caution).
      - gui:    "CONTIN α by GCV", "CONTIN α by F-test (p=0.50)". The GUI shows a
                user-supplied alpha in its input control and the edge caution on its
                own flag label, so neither is repeated here (returns '' for 'user';
                `edge_note` is ignored under `style='gui'`).
    """
    name = _ALPHA_SELECTOR_NAMES.get(selection_method)
    if name is None:
        return ''
    if style == 'gui' and selection_method == 'user':
        return ''
    lead = 'CONTIN α ' if style == 'gui' else 'alpha '
    note = lead + (name if selection_method == 'user' else 'by ' + name)
    if selection_method == 'ftest' and ftest_prob_reject is not None:
        note += (f' (p={ftest_prob_reject:.2f})' if style == 'gui'
                 else f', p_reject={ftest_prob_reject:.2f}')
    if style == 'export' and edge_note:
        note += f' ({edge_note})'
    return note


def export_distribution(result, file_path: str, axis: str = 'rh',
                        delimiter: str = ',', *,
                        alpha_selection_method: Optional[str] = None,
                        ftest_prob_reject: Optional[float] = None,
                        alpha_edge_note: str = '') -> str:
    """Export an NNLS or CONTIN distribution (DistributionResult).

    Writes the size/rate grid and the normalized weights. With axis='rh' the
    grid is hydrodynamic radius (nm); with axis='gamma' it is decay rate (1/s).
    Both grids are always included so either can be plotted. The reconstructed
    correlogram fit and residuals are written alongside, plus scalar summaries.

    Parameters
    ----------
    result : DistributionResult
    file_path : str
    axis : str
        'rh' (default) or 'gamma' -- controls only the column order/labels; both
        grids are written regardless.

    Returns
    -------
    str
    """
    rh = np.asarray(result.rh_grid_nm, dtype=float)
    gamma = np.asarray(result.gamma_grid_s_inv, dtype=float)
    w = np.asarray(result.weights, dtype=float)
    order = np.argsort(rh if axis == 'rh' else gamma)

    # A failed nonlinear fit (success=False, lognormal only) means the weights ARE the
    # seed estimate -- the caveat must ride the Weight column a user actually plots,
    # not only the scalar summaries below (same fail_note on both).
    fail_note = ('' if getattr(result, 'success', True)
                 else 'fit did not converge; values describe the seed estimate, not a fit')
    columns = [
        OriginColumn('Rh', 'nm', 'hydrodynamic radius', rh[order]),
        OriginColumn('Gamma', '1/s', 'decay rate', gamma[order]),
        OriginColumn('Weight', '',
                     _join_comments('intensity fraction', fail_note), w[order]),
    ]
    # Correlogram reconstruction (separate length from the grid).
    columns += [
        OriginColumn('Delay time', 's', 'tau', np.asarray(result.fit_tau_s, dtype=float)),
        OriginColumn('g2-1 (fit)', '', 'reconstruction', np.asarray(result.fitted_g2m1, dtype=float)),
        OriginColumn('Residual', '', 'data - fit', np.asarray(result.residuals, dtype=float)),
    ]
    # CONTIN records how alpha was chosen in the alpha column's Comments cell, so a
    # distribution is never ambiguous about its regularization (mirrors the SLS
    # provenance-in-Comments convention). The wording -- and the sweep-ceiling
    # over-regularization caution -- come from the shared `alpha_selection_note`
    # helper the GUI status note also uses, so the two can't drift.
    # NNLS/lognormal pass no selection method -> ''.
    alpha_note = alpha_selection_note(
        alpha_selection_method, ftest_prob_reject,
        edge_note=alpha_edge_note, style='export')
    columns += [
        _scalar_column('Method', '', result.method, fail_note),
        _scalar_column('alpha', '', result.alpha, comments=alpha_note),
        _scalar_column('Peak Rh', 'nm', result.peak_rh_nm, fail_note),
        _scalar_column('Mean Rh', 'nm', result.mean_rh_nm, fail_note),
        _scalar_column('Mean Gamma', '1/s', result.mean_gamma_s_inv, fail_note),
        _scalar_column('beta', '', result.beta),
        _scalar_column('beta estimated', '', result.beta_estimated),
        _scalar_column('baseline', '', result.baseline),
        _scalar_column('baseline estimated', '', result.baseline_estimated),
        _scalar_column('RMS error', '', result.rms_error),
    ]
    return write_origin_csv(file_path, columns, delimiter)


def export_weighted_distribution(result, file_path: str,
                                 delimiter: str = ',') -> str:
    """Export a reweighted (volume/surface/number) distribution (WeightedDistributionResult).

    Writes the Rh grid, the reweighted weights, and a per-bin validity flag. The
    model-dependence caveat rides in the Weight column's Comments cell ONLY (mirroring the
    "uncalibrated, arbitrary scale" convention: no extra rows, so the Origin import is
    unaffected) -- the reweighted view is derived and model-dependent, never a measurement.
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
    m_txt = f", m = {result.m:.4g}" if result.m is not None else ""
    weight_note = (f"{label}-weighted: derived, model-dependent ({model} sphere{m_txt}); "
                   f"re-weights the same intensity fit and amplifies small-size noise "
                   f"(not a measurement)")
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
                       comments=('some size bins fall outside the model validity range'
                                 if not result.all_valid else '')),
    ]
    return write_origin_csv(file_path, columns, delimiter)


def export_lcurve(lcurve, file_path: str, delimiter: str = ',') -> str:
    """Export a CONTIN alpha-sweep (LCurveResult) for inspection.

    Always writes the L-curve axes (alpha, weighted residual norm, solution
    seminorm) plus the chosen alpha/index. The method-specific score curves --
    the GCV score, the F-test cumulative F, and the Tikhonov effective DOF -- are
    appended only when the sweep carries them (they are populated per selection
    method: `gcv` for GCV, `ftest_fc` for the F-test, `dof_eff` for both), so the
    CSV matches whichever ordinate the diagnostic view showed.
    """
    columns = [
        OriginColumn('alpha', '', 'regularization parameter',
                     np.asarray(lcurve.alphas, dtype=float)),
        OriginColumn('Residual norm', '', '||M^.5 (A x - y)||^2 (weighted)',
                     np.asarray(lcurve.residual_norms, dtype=float)),
        OriginColumn('Solution seminorm', '', '||L x||^2',
                     np.asarray(lcurve.solution_norms, dtype=float)),
    ]
    # Method-specific score curves, present only for the method that ran.
    if getattr(lcurve, 'gcv', None) is not None:
        columns.append(OriginColumn('GCV score', '', 'V(alpha) (minimized)',
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
    null-assumption note travels with the file for the same reason: a verdict
    computed for white noise on a trace that is not white is the one way this
    number misleads, and a reader opening the file later has no other way to know.
    """
    if result.null_assumption_violated:
        null_note = ('UNRELIABLE: this trace shows correlations, and the '
                     "false-alarm test's null assumes white noise")
    else:
        null_note = 'null assumes white, Gaussian noise'
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
                       result.false_alarm_probability, null_note),
        _scalar_column('False-alarm level', 'variance units',
                       result.significance_level,
                       _join_comments(
                           f'power a peak must exceed at p = '
                           f'{result.requested_false_alarm:g}', null_note)),
        _scalar_column('Peak significant', '', bool(result.peak_is_significant),
                       null_note),
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


def export_gamma_q2(result, file_path: str, delimiter: str = ',') -> str:
    """Export a Gamma-vs-q^2 multi-angle analysis (GammaQ2Result).

    The pooled D and Rh carry whatever the analysis said about the set it was given --
    that the measurements do not all share one sample identity (the assumption Gamma vs
    q^2 rests on), or that an angle was dropped for an unusable decay rate. Without it
    this exporter had no note channel at all, so a D fitted across two different samples
    exported looking like a measurement.

    The D and Rh columns ALSO carry the pooled fit's reliability verdict when it is
    flagged (`rh_reliable` False: a flagged per-angle fit inside the line, a pooled Rh
    outside the size range, no viscosity), on the Comments cell -- the same channel every
    other flagged value rides out on, since the export outlives the session.

    `Is diffusive` names the criteria it was judged against either way, and when the
    fit fails them the sentence also rides D, Rh and their SEs: a Gamma that is not
    linear through the origin in q^2 makes the through-origin slope an apparent D.
    Each SE cell carries its value's marker, the estimator tag and the statistical
    label joined -- independent facts, none of which may displace another.
    """
    fit_note = _analysis_notes(result)
    flag_note = ('' if getattr(result, 'rh_reliable', True)
                 else _reliability_comment(result, False))
    criteria = (f'R^2 >= {result.r2_threshold:g} and '
                f'|intercept|/mean(Gamma) <= {result.intercept_rel_threshold:g}')
    if result.is_diffusive:
        verdict_note = f'within stated tolerance: {criteria}'
        non_diffusive_note = ''
    elif not math.isfinite(result.r_squared):
        # `analyze_gamma_q2` also returns is_diffusive=False when FEWER THAN TWO angles
        # yielded a usable rate: no line was fitted, so the criteria were never evaluated
        # and saying they failed would assert a test that never ran. The result's own
        # notes and reliability note already say what happened, on these same cells.
        verdict_note = ('no Gamma-vs-q^2 line was fitted, so the linearity criteria '
                        'were not evaluated -- see the D and Rh cells')
        non_diffusive_note = ''
    else:
        non_diffusive_note = (
            f'not purely diffusive: fails {criteria.replace(" and ", " or ")} -- Gamma '
            f'is not linear through the origin in q^2, so D and Rh are apparent')
        verdict_note = non_diffusive_note
    se_note = _join_comments(flag_note, non_diffusive_note, _se_note(result),
                             'statistical (over angles)')
    columns = [
        OriginColumn('Angle', 'deg', '', np.asarray(result.angles_deg, dtype=float)),
        OriginColumn('q^2', 'm^-2', 'scattering vector squared',
                     np.asarray(result.q2_m2, dtype=float)),
        OriginColumn('Gamma', '1/s', 'decay rate',
                     np.asarray(result.gamma_s_inv, dtype=float)),
        OriginColumn('D_app', 'm^2/s', 'Gamma / q^2',
                     np.asarray(result.d_app_m2_s, dtype=float)),
        _scalar_column('D', 'm^2/s', result.d_m2_s,
                       _join_comments('through-origin slope', fit_note, flag_note,
                                      non_diffusive_note)),
        _scalar_column('D SE', 'm^2/s', result.d_se, comments=se_note),
        _scalar_column('Rh', 'nm', result.rh_nm,
                       comments=_join_comments(fit_note, flag_note, non_diffusive_note)),
        _scalar_column('Rh SE', 'nm', result.rh_se, comments=se_note),
        _scalar_column('R^2', '', result.r_squared),
        _scalar_column('Intercept', '1/s', result.intercept_s_inv),
        _scalar_column('Intercept (relative)', '', result.intercept_relative),
        _scalar_column('D_app trend (relative)', '', result.d_app_trend_rel),
        _scalar_column('Is diffusive', '', result.is_diffusive, verdict_note),
    ]
    return write_origin_csv(file_path, columns, delimiter)


def export_mode_map(map_run, family_fits, amplitude, file_path: str, *,
                    point_families=None, family_states=None,
                    delimiter: str = ',') -> str:
    """Export a mode map, its per-family fits, and the amplitude report.

    One file, three blocks, per the locked conventions (flags never change row
    counts; caveats ride the affected columns' Comments cells):

      * per-point arrays -- every extracted mode of every run (`map_run`, a
        ModeMapRun), with `point_families` (an aligned sequence of family
        labels, '' = unassigned, supplied by the caller from the user's picks)
        as the Family column;
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
    decay_note = ''
    if n_undecayed:
        floor = max(getattr(p, 'min_decay_completeness', float('nan')) for p in pts)
        decay_note = (
            f'{n_undecayed} mode(s) did not finish decaying inside their lag window '
            f'(Gamma * lag span below {floor:.3g}, ISO 22412:2017 A.1.2): those rates '
            f'are extrapolations past the end of the measurement')
    columns = [
        OriginColumn('Angle', 'deg',
                     _join_comments('every extracted mode of every run',
                                    _analysis_notes(map_run)),
                     np.array([p.angle_deg for p in pts], dtype=float)),
        OriginColumn('q^2', 'm^-2', 'scattering vector squared',
                     np.array([p.q2_m2 for p in pts], dtype=float)),
        OriginColumn('Gamma', '1/s',
                     _join_comments('mode decay rate', decay_note),
                     np.array([p.gamma_s_inv for p in pts], dtype=float)),
        OriginColumn('D_app', 'm^2/s',
                     _join_comments('Gamma / q^2', decay_note),
                     np.array([p.d_m2_s for p in pts], dtype=float)),
        OriginColumn('Amplitude fraction', '', amp_note,
                     np.array([p.amplitude_fraction for p in pts], dtype=float)),
        OriginColumn('Mode', '', 'rank within its run (1 = fastest)',
                     np.array([p.mode_index + 1 for p in pts], dtype=float)),
        OriginColumn('Source', '', 'extraction method',
                     [p.source_method for p in pts]),
        OriginColumn('Family', '', "user-picked mode family ('' = unassigned)",
                     fams),
    ]

    for label, fit in family_fits.items():
        fit_note = _analysis_notes(fit)
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
        # `export_gamma_q2` sibling does: written unconditionally it asserted the fit was
        # within tolerance beside a cell reading 'no'. A mode family is exactly where the
        # origin criteria are expected to fail (a VH family's slope is not a
        # translational D).
        criteria = (f'R^2 >= {fit.r2_threshold:g} and '
                    f'|intercept|/mean(Gamma) <= {fit.intercept_rel_threshold:g}')
        verdict_note = (
            f'within stated tolerance: {criteria}' if fit.is_diffusive else
            f'not purely diffusive: fails {criteria.replace(" and ", " or ")} -- this '
            f'family Gamma is not linear through the origin in q^2, so its D and Rh '
            f'are apparent')
        # `vh_note` rides the SE cells too, not only the values beside them: a reader
        # filtering on the SE columns must still see that the quantity is not a
        # translational D at all.
        se_note = _join_comments(vh_note, fit_note, _se_note(fit),
                                 'statistical (over angles)')
        columns += [
            _scalar_column(f'D ({label})', 'm^2/s', fit.d_m2_s,
                           _join_comments('through-origin slope; mode-resolved',
                                          vh_note, fit_note)),
            _scalar_column(f'D SE ({label})', 'm^2/s', fit.d_se, se_note),
            _scalar_column(f'Rh ({label})', 'nm', fit.rh_nm,
                           _join_comments('apparent, mode-resolved '
                                          '(Stokes-Einstein on the family D)',
                                          vh_note, fit_note)),
            _scalar_column(f'Rh SE ({label})', 'nm', fit.rh_se, se_note),
            _scalar_column(f'R^2 ({label})', '', fit.r_squared),
            _scalar_column(f'Intercept ({label})', '1/s', fit.intercept_s_inv),
            _scalar_column(f'Intercept (relative) ({label})', '',
                           fit.intercept_relative),
            _scalar_column(f'Passes through origin ({label})', '',
                           fit.is_diffusive, verdict_note),
            _scalar_column(f'a ({label})', 'm^2/s', fit.a_m2_s,
                           'Gamma = a q^2 + b q^4 (deviation form)'),
            _scalar_column(f'a SE ({label})', 'm^2/s', fit.a_se, se_note),
            _scalar_column(f'b ({label})', 'm^4/s', fit.b_m4_s,
                           'Gamma = a q^2 + b q^4 (deviation form)'),
            _scalar_column(f'b SE ({label})', 'm^4/s', fit.b_se, se_note),
        ]

    if amplitude is not None:
        amp_notes = _analysis_notes(amplitude)
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
            OriginColumn('A_f/A_s', '', _join_comments('no SE: single-run '
                                                       'amplitudes', amp_notes),
                         np.asarray(amplitude.ratio_f_over_s, dtype=float)),
            _scalar_column(
                'Fast amplitude q-flat', '', amplitude.fast_flat,
                f'(max-min)/mean of A_f <= {amplitude.flatness_rel_threshold:g}'
                f' (house diagnostic threshold)'),
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
    notes = _analysis_notes(partition)
    sep_note = ('' if partition.well_separated else
                f'mode separation below the factor-'
                f'{partition.separation_ratio_min:g} threshold: the partition '
                f'assumes well-separated modes')
    # q^2 and the scale flags live on the per-mode Rayleigh clones; every
    # clone shares the parent's, so the first is the column's.
    first = partition.per_mode[0].rayleigh
    columns = [
        OriginColumn('Angle', 'deg',
                     _join_comments('one concentration partitioned by mode '
                                    'amplitudes', sep_note, notes),
                     np.asarray(partition.angles_deg, dtype=float)),
        OriginColumn('q^2', 'nm^-2', 'scattering vector squared',
                     np.asarray(first.q2_nm2, dtype=float)),
        OriginColumn('dR (total)', 'cm^-1',
                     _join_comments('unpartitioned excess Rayleigh ratio',
                                    _scale_note(first.calibrated)),
                     np.asarray(partition.excess_total_cm_inv, dtype=float)),
        OriginColumn('Sum of picked fractions', '',
                     _join_comments(
                         'closure: 1 when the picked modes are exhaustive '
                         f'(disclosure threshold {partition.closure_tol:g}); '
                         'blank where withheld', notes),
                     np.asarray(partition.closure_fraction, dtype=float)),
    ]
    for pm in partition.per_mode:
        role = f' ({pm.role})' if pm.role else ''
        columns.append(OriginColumn(
            f'dR [{pm.label}]', 'cm^-1',
            _join_comments(
                f'partitioned intensity{role}: mode amplitude x dR; '
                f'no SE (single-run inversion amplitudes)',
                _scale_note(pm.rayleigh.calibrated),
                str(pm.rayleigh.derived_provenance or ''),
                str(getattr(pm.rayleigh, 'blank_provenance', '') or '')),
            np.asarray(pm.rayleigh.excess_rayleigh_cm_inv, dtype=float)))

    for label, fit in (guinier_fits or {}).items():
        fit_note = _join_comments(_analysis_notes(fit), fit.reliability_note,
                                  str(fit.derived_provenance or ''))
        floor_note = (f'below the ~{fit.rg_resolution_floor_nm:.0f} nm '
                      f'resolution floor of this q-range: relative changes '
                      f'only' if fit.below_resolution_floor else '')
        # The SE cell carries the same floor / reliability facts as the value cell
        # beside it: a reader filtering on the SE column must not see a clean ±
        # on a size the value column flags.
        se_note = _join_comments(
            floor_note, fit_note,
            'statistical (over angles); excludes the inversion systematics',
            _se_note(fit))
        columns += [
            _scalar_column(f'Rg app ({label})', 'nm', fit.rg_app_nm,
                           _join_comments('apparent, mode-resolved '
                                          '(amplitude-Guinier, Sedlak Eq. 7)',
                                          floor_note, fit_note)),
            _scalar_column(f'Rg SE ({label})', 'nm', fit.rg_se, se_note),
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

    Every column whose VALUE IS A FUNCTION OF D_r carries the analysis's unphysical-D_r
    verdict in its Comments cell when `d_r_reliable` is False, via the same
    `_reliability_comment` path the SLS Mw/A2 columns use. That is the per-angle values,
    the mean, its SE, tau_rot = 1/(6 D_r), AND every shape-model dimension below, each
    of which is inverted from the (D_t, D_r) pair -- a column that merely PRINTS D_r is
    the narrower set, and stopping there is how a corrupted number stops looking
    corrupted. tau_rot shows why the aggregate is not the test: it is NaN only when the
    MEAN D_r is non-positive, so one negative angle inside a positive mean yields a
    finite, printable rotational time computed from a corrupted average -- and the shape
    models run on that same mean.

    Two columns are deliberately NOT marked, because they read D_t alone:
    `Sphere R(D_t)=Rh` and its SE. Nor is the D_t family above -- D_t is the
    through-origin slope of Gamma_VV vs q^2, so it cannot be corrupted by VH noise or by
    an optically isotropic scatterer. (A swapped VV/VH pairing corrupts both channels,
    which is why the analysis's note names that cause: the split is about which number is
    in doubt for the causes a reader can act on, not a claim that D_t is immune.)

    Every shape-model SE says what produced it. The rod and ellipsoid ± are Monte-Carlo
    spreads over draws from a D_t whose SE came from the user's chosen covariance
    estimator, so those cells carry the estimator tag (`_se_note`) and the draw census
    (`_mc_note`); `Sphere R(D_t)=Rh SE` is a pass-through of that same D_t SE and carries
    the tag alone. `Sphere R(D_r) SE` derives from the replicate SEM of D_r, which no
    covariance estimator touches, so it carries none.
    """
    d_r_comment = _reliability_comment(result, result.d_r_reliable)
    # All three SEs are withheld together on a single-angle run (invariant 8).
    se_withheld = (_DDLS_SINGLE_ANGLE_NO_SE
                   if getattr(result, 'method', '') == 'single-angle' else '')
    columns = [
        OriginColumn('Angle', 'deg', '', np.asarray(result.angles_deg, dtype=float)),
        OriginColumn('q^2', 'm^-2', 'scattering vector squared',
                     np.asarray(result.q2_m2, dtype=float)),
        OriginColumn('Gamma_VV', '1/s', 'polarized field decay rate',
                     np.asarray(result.gamma_vv_s_inv, dtype=float)),
        OriginColumn('Gamma_VH', '1/s', 'depolarized field decay rate',
                     np.asarray(result.gamma_vh_s_inv, dtype=float)),
        OriginColumn('D_r (per angle)', 'rad^2/s',
                     d_r_comment or '(Gamma_VH - Gamma_VV)/6',
                     np.asarray(result.d_r_per_angle, dtype=float)),
    ]
    if result.qL is not None:
        columns.append(OriginColumn('qL', '', 'single-exponential valid if < 3',
                                    np.asarray(result.qL, dtype=float)))
    columns += [
        _scalar_column('D_t', 'm^2/s', result.d_t_m2_s, 'from VV, through-origin slope'),
        _scalar_column('D_t SE', 'm^2/s', result.d_t_se,
                       comments=se_withheld or _join_comments(
                           _se_note(result), 'statistical only')),
        _scalar_column('D_r', 'rad^2/s', result.d_r_rad2_s,
                       d_r_comment or 'mean of per-angle values'),
        # The D_r verdict outranks the withheld-SE note here: an unphysical D_r makes its
        # SE meaningless whether or not one was computed, and the single-angle case is
        # already stated on the two columns beside it.
        _scalar_column('D_r SE', 'rad^2/s', result.d_r_se,
                       d_r_comment or se_withheld or 'statistical only'),
        _scalar_column('Rh_t', 'nm', result.rh_t_nm, 'Stokes radius from D_t'),
        _scalar_column('Rh_t SE', 'nm', result.rh_t_se,
                       comments=se_withheld or _join_comments(
                           _se_note(result), 'statistical (over angles)')),
        _scalar_column('tau_rot', 's', result.rotational_time_s,
                       d_r_comment or '1 / (6 D_r)'),
        _scalar_column('N angles', '', result.n_angles),
        _scalar_column('Method', '', result.method),
        _scalar_column('Single-exponential valid', '', result.single_exponential_valid),
    ]
    if shapes:
        rod = shapes.get('rod')
        if rod is not None:
            rod_se_note = _join_comments(
                d_r_comment,
                se_withheld or _join_comments(_se_note(rod), _mc_note(rod)))
            columns += [
                _scalar_column('Rod L', 'nm', rod.length_nm,
                               _join_comments(d_r_comment, 'MODEL: rigid cylinder '
                                              '(Tirado 1984), not measured')),
                _scalar_column('Rod L SE', 'nm', rod.length_se, rod_se_note),
                _scalar_column('Rod d', 'nm', rod.diameter_nm, d_r_comment),
                _scalar_column('Rod d SE', 'nm', rod.diameter_se, rod_se_note),
                _scalar_column('Rod aspect p', '', rod.aspect_ratio_p, d_r_comment),
                _scalar_column('Rod aspect p SE', '', rod.aspect_ratio_se, rod_se_note),
                _scalar_column('Rod p in 2-30', '', rod.in_valid_range, d_r_comment),
                _scalar_column('Rod converged', '', rod.converged, d_r_comment),
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
                _scalar_column('Sphere R(D_t)=Rh', 'nm', sphere.radius_trans_nm),
                _scalar_column('Sphere R(D_t)=Rh SE', 'nm', sphere.radius_trans_se,
                               se_withheld or _join_comments(
                                   _se_note(sphere), 'statistical (over angles)')),
                _scalar_column('Sphericity ratio', '', sphere.sphericity_ratio,
                               _join_comments(d_r_comment, 'R(D_r)/Rh; 1 = sphere')),
                _scalar_column('Sphere consistent', '', sphere.is_consistent, d_r_comment),
            ]
        ellipsoids = shapes.get('ellipsoid')
        if ellipsoids is not None:
            # A given D_t/D_r admits up to three ellipsoids (1 prolate + 2 oblate).
            # Emit a FIXED three-slot block (blank unused slots) so the column set is
            # input-independent, led by the count + degeneracy flag.
            fitted = [e for e in ellipsoids if e.converged]
            columns += [
                _scalar_column('Ellipsoid solutions', '', len(fitted),
                               _join_comments(d_r_comment,
                                              'MODEL: ellipsoid of revolution (Perrin: '
                                              'Tanford 1961 translation + Koenig 1975 '
                                              'rotation), not measured')),
                _scalar_column('Ellipsoid degenerate', '',
                               bool(fitted) and fitted[0].degenerate, d_r_comment),
            ]
            for k in range(3):
                e = fitted[k] if k < len(fitted) else None
                lbl = f'Ellipsoid {k + 1}'
                e_se_note = _join_comments(
                    d_r_comment,
                    se_withheld or _join_comments(_se_note(e), _mc_note(e)))
                columns += [
                    _scalar_column(f'{lbl} type', '',
                                   ('prolate' if e.is_prolate else 'oblate')
                                   if e is not None else None, d_r_comment),
                    _scalar_column(f'{lbl} a (polar)', 'nm',
                                   e.semi_axis_polar_nm if e is not None else None,
                                   d_r_comment),
                    _scalar_column(f'{lbl} a SE', 'nm',
                                   e.semi_axis_polar_se if e is not None else None,
                                   e_se_note),
                    _scalar_column(f'{lbl} b (equatorial)', 'nm',
                                   e.equatorial_radius_nm if e is not None else None,
                                   d_r_comment),
                    _scalar_column(f'{lbl} b SE', 'nm',
                                   e.equatorial_radius_se if e is not None else None,
                                   e_se_note),
                    _scalar_column(f'{lbl} p', '',
                                   e.axial_ratio_p if e is not None else None,
                                   d_r_comment),
                    _scalar_column(f'{lbl} p in 0.1-10', '',
                                   e.in_valid_range if e is not None else None,
                                   d_r_comment),
                ]
    return write_origin_csv(file_path, columns, delimiter)


def export_concentration_extrapolation(result, file_path: str,
                                       delimiter: str = ',') -> str:
    """Export a D-vs-c concentration extrapolation (ConcentrationExtrapolationResult).

    The kD column's Comments cell carries the reliability note (D0 <= 0 sign flip or an
    ill-conditioned design) when kD is unreliable, mirroring the SLS Mw/A2 exporters via
    `_reliability_comment`; a healthy fit leaves it unmarked. The D0 and Rh0 columns
    carry the same note when the Rh0 verdict is flagged (`rh_reliable` False: a flagged
    per-point fit inside the line, an extrapolated Rh0 outside the size range, no
    viscosity). The result keeps ONE note for both flags, so each column is gated on
    its own flag rather than on the note's presence.

    A DROPPED concentration (no usable D -- a NaN in `d_values_m2_s`) is disclosed in the
    `D_app` and `D0` Comments cells (owner decision). It has to be said here and not
    left to the GUI's `notes` tier: the export outlives the session, and a blank `D_app`
    cell beside a `D0` fitted over a quietly smaller point set is exactly what the Comments
    cell exists for -- the same reasoning that puts "uncalibrated, arbitrary scale" there.

    Whatever the fit recorded about the set is relayed onto the `Concentration` column,
    which is the input those notes are about. The exporter's own drop wording below is kept
    rather than replaced by the analysis's: it counts against the concentrations actually
    written to the file and distinguishes the case where no fit ran at all, neither of
    which the analysis sentence can say."""
    kd_comment = ('' if result.kd_reliable
                  else _reliability_comment(result, result.kd_reliable))
    rh0_comment = ('' if getattr(result, 'rh_reliable', True)
                   else _reliability_comment(result, False))
    set_note = _analysis_notes(result)
    d_app = np.asarray(result.d_values_m2_s, dtype=float)
    n_dropped = int(np.count_nonzero(~np.isfinite(d_app)))
    # Two wordings, because `n_concentrations == 0` means NO fit ran: saying "fitted over
    # the remaining 0" would assert a fit that never happened -- the same cause-that-did-
    # not-occur defect this change removes from `reliability_note`.
    if not n_dropped:
        drop_note = ''
    elif result.n_concentrations:
        drop_note = (
            f'{n_dropped} of {d_app.size} concentration(s) dropped from the fit (no '
            f'usable D; blank D_app cells) -- D0/Rh0/kD are fitted over the remaining '
            f'{result.n_concentrations}')
    else:
        drop_note = (
            f'{n_dropped} of {d_app.size} concentration(s) had no usable D (blank D_app '
            f'cells), leaving too few to extrapolate -- D0/Rh0/kD are withheld')
    columns = [
        OriginColumn('Concentration', 'g/mL', set_note,
                     np.asarray(result.concentrations_g_per_mL, dtype=float)),
        OriginColumn('D_app', 'm^2/s',
                     drop_note or 'apparent diffusion coefficient', d_app),
        _scalar_column('D0', 'm^2/s', result.d0_m2_s,
                       _join_comments(f'c -> 0; {drop_note}' if drop_note else 'c -> 0',
                                      rh0_comment)),
        _scalar_column('D0 SE', 'm^2/s', result.d0_se,
                       comments=_join_comments(rh0_comment, _se_note(result),
                                               'statistical (over concentrations)')),
        _scalar_column('Rh0', 'nm', result.rh0_nm, comments=rh0_comment),
        _scalar_column('Rh0 SE', 'nm', result.rh0_se,
                       comments=_join_comments(rh0_comment, _se_note(result),
                                               'statistical (over concentrations)')),
        _scalar_column('kD', 'mL/g', result.kd_mL_per_g,
                       comments=kd_comment or 'diffusion interaction parameter'),
        _scalar_column('kD SE', 'mL/g', result.kd_se,
                       comments=_join_comments(kd_comment, _se_note(result),
                                               'statistical (over concentrations)')),
        _scalar_column('Slope', 'm^2/s/(g/mL)', result.slope),
        _scalar_column('R^2', '', result.r_squared),
    ]
    return write_origin_csv(file_path, columns, delimiter)


# ===========================================================================
# SLS exporters
# ===========================================================================

# The one wording for a scale-dependent (excess-Rayleigh) column that is on an
# arbitrary scale because the run was uncalibrated. Kept in one place so a reword
# can't drift across the Rayleigh/Zimm/reliability exporters (they were parallel
# copies -- the same class of divergence the shared CONTIN alpha note closed).
_UNCALIBRATED_SCALE_NOTE = 'uncalibrated, arbitrary scale'

# The same idea one factor over: no refractive-index increment was supplied, so the
# optical constant K was set to 1 and everything carrying it is on an arbitrary OPTICAL
# scale. Deliberately its own wording rather than a reuse of _UNCALIBRATED_SCALE_NOTE:
# the two absences are independent, can hold at once, and name different things for the
# reader to go fix -- a calibrant measurement versus a dn/dc value. Crucially they also
# cover different COLUMNS: dn/dc does not enter dR, so the excess-Rayleigh column is
# untouched by it while the Kc/dR ordinate is not.
_NO_DN_DC_SCALE_NOTE = 'no usable dn/dc, arbitrary optical scale'

# The condition under which the calibration-free estimator's ABSOLUTE A2 exists at all.
# 2 A2 Mw is genuinely calibration-free, but A2 = (2 A2 Mw)/(2 Mw) inherits the scale of
# the Mw it divides by, so the controller supplies that Mw only when it is trustworthy and
# withholds A2 otherwise (see Controller.run_calibration_free_a2). Deliberately NOT
# _UNCALIBRATED_SCALE_NOTE: that wording asserts a PRINTED number is on an arbitrary scale,
# and in the withheld case the cell is empty -- there is no number for it to be about.
# Reusing it would make one constant mean two things across exporters, the drift it exists
# to prevent.
_CALFREE_A2_MW_CONDITION = 'only if a calibrated or hand-entered Mw is supplied'

# Why a DDLS ± column is empty on a single-angle run. Invariant 8: a ± never ships from
# one correlogram (its lag channels are correlated -- Schaetzel 1990), so `analyze_ddls`
# withholds all three SEs when `method == 'single-angle'`. The cells must say that rather
# than keep their multi-angle default, which would assert a statistical SE that was
# deliberately not computed. Keyed on `method`, a structured field the file already
# exports, never on the result's prose.
_DDLS_SINGLE_ANGLE_NO_SE = ('no ± from a single angle: one correlogram per channel, so a '
                            '± needs repeats or multiple angles (ISO 22412)')


def _derived_note(results) -> str:
    """The derivation provenance shared by these Rayleigh results, or ''.

    One string for the whole set: the Comments cell describes a COLUMN, so a note
    is written only when every contributing result agrees on it. A mixed set --
    some loaded, some derived -- says so instead of implying the whole column came
    from one place.
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
    return 'some points derived from DLS count rates'


def _blank_note(results) -> str:
    """Every distinct 'which solvent blank' record these Rayleigh results carry, or ''.

    Unlike `_derived_note` there is no mixed-set sentence: a blank note names a file,
    and a column built from two blanks is described by naming both. Empty when every
    point used the sample's reference blank (the silent default).
    """
    notes = sorted({str(getattr(r, 'blank_provenance', '') or '') for r in results}
                   - {''})
    return '; '.join(notes)


def _scale_note(calibrated: bool) -> str:
    """Comments-cell marker for a RAYLEIGH-scale-dependent column: the uncalibrated note
    when `calibrated` is False, else '' (the silent calibrated default)."""
    return '' if calibrated else _UNCALIBRATED_SCALE_NOTE


def _optical_scale_note(dn_dc_supplied: bool) -> str:
    """Comments-cell marker for an OPTICAL-scale-dependent column (anything carrying K):
    the no-dn/dc note when `dn_dc_supplied` is False, else '' (the silent default).

    Its own selector rather than a second argument to `_scale_note`, because the two
    notes apply to overlapping but DIFFERENT column sets: every Kc/dR-derived column
    takes both, while the excess-Rayleigh column takes only the calibration one.
    Join them with `_join_comments` where both apply -- they are independent facts.
    """
    return '' if dn_dc_supplied else _NO_DN_DC_SCALE_NOTE


def _se_note(result) -> str:
    """Comments-cell label for a ± column: names the estimator only when it is the
    non-default classical OLS (silent for HC3, matching the 'calibrated is the silent
    default' convention). See the Theory-and-Equations-Guide §6.1.

    Join it with the column's other facts (`_join_comments`), never `or` it against
    them: the statistical label and the value's own marker hold whichever estimator
    ran, and an `or` would drop them exactly on the estimator that can under-report.
    """
    return 'SE: classical OLS' if getattr(result, 'se_estimator', 'hc3') == 'ols' else ''


def _mc_note(shape) -> str:
    """Comments-cell label for a Monte-Carlo-propagated shape-model ± (rod / ellipsoid).

    Names the draw census when the result carries one -- how many (D_t, D_r) draws
    were taken, how many the model reproduced, and the seed -- because the ± is the
    spread over the solvable draws, so the count is part of what the number means and a
    later change of draw count or seed would otherwise move a published ± silently.
    A result with no census (no SEs were given, or an older object) gets the bare label.
    """
    n_draws = int(getattr(shape, 'mc_n_draws', 0) or 0)
    if n_draws <= 0:
        return 'Monte-Carlo'
    return (f'Monte-Carlo ({int(getattr(shape, "mc_n_used", 0) or 0)} of {n_draws} '
            f'draws solvable, seed {getattr(shape, "mc_seed", None)})')


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
    printed numbers arbitrary. Without a marker a re-import to Origin reads them as
    absolute. One helper for every per-fit exporter, so they cannot drift apart from
    each other or from the Zimm/Rayleigh pair. `optical=False` is for an ordinate that
    does not carry K (Guinier's ln(dR)): it takes the calibration marker and the
    provenance, never the dn/dc one.
    """
    return _join_comments(
        _scale_note(getattr(result, 'calibrated', True)),
        (_optical_scale_note(getattr(result, 'dn_dc_supplied', True))
         if optical else ''),
        # Independent of both scale causes and joined with them rather than ranked:
        # a derived series can also be uncalibrated.
        _provenance_note(result)) or default


def _analysis_notes(result) -> str:
    """The result's own passive `notes` tier, joined for a Comments cell.

    The exporter RELAYS these rather than interpreting them. A note the analysis chose to
    emit is about the fit as a whole -- a point silently dropped before fitting, a pooled
    set that does not share sample identity -- so it belongs on the columns that fit rests
    on, and selecting among notes by matching their wording would make the export's
    behaviour a function of the analysis's prose (a reword would then disarm it silently).
    Accepts the tuple convention and the legacy `DDLSResult` string alike.
    """
    notes = getattr(result, 'notes', ()) or ()
    if isinstance(notes, str):
        return notes
    return ' '.join(str(n) for n in notes)


def _join_comments(*parts: str) -> str:
    """Join the non-empty Comments fragments with '; '.

    Used where two INDEPENDENT facts can hold at once -- a dropped point and an
    uncalibrated scale are about different things, and `a or b` would silently withhold
    the second whenever the first fired.
    """
    return '; '.join(p for p in parts if p)


def _size_comment(result) -> str:
    """Comments-cell note for a SIZE column (Rg / R) whose `rg_reliable` flag is False.

    Separate from `_reliability_comment` because the size is not scale-gated: Rg survives
    an uncalibrated run, so the "arbitrary scale" precedence that opens that helper is
    wrong here and would mask the real reason. Empty whenever the size stands -- including
    on every result type that carries no such flag.
    """
    if getattr(result, 'rg_reliable', True):
        return ''
    return (getattr(result, 'reliability_note', '')
            or 'the fitted size is not physically meaningful')


def _reliability_comment(result, reliable: bool = True, scale_free: bool = False) -> str:
    """Comments-cell note for a reliability-gated column (Mw / A2).

    `mw_reliable`/`a2_reliable` fold in SEVERAL independent gates — calibration, a sign
    guard (intercept/a/y > 0), numerical conditioning, and finiteness — so keying only on
    `calibrated`/`reliability_note` would let a value that is unreliable for one of the
    *other* reasons (a non-physical intercept, or a non-finite overflow) fall through and
    get the caller's healthy default note next to a NaN/inf. Pass the column's own flag
    (`result.mw_reliable` / `.a2_reliable`) so every False produces some explanatory text.

    Precedence: uncalibrated -> "arbitrary scale" (unchanged wording); else a specific
    `reliability_note` (ill-conditioned / subnormal / non-positive divisor); else, if the
    flag is False for another reason, a generic unreliable marker; else '' (the caller's
    default APPARENT/thermodynamic note fills in)."""
    # `scale_free`: the column's quantity cancels the optical scale (the calibration-
    # free 2 A2 Mw), so the two scale branches must not fire even though the result
    # RECORDS `calibrated` / `dn_dc_supplied` as provenance.
    if not scale_free and not getattr(result, 'calibrated', True):
        return _join_comments(_UNCALIBRATED_SCALE_NOTE,
                              _optical_scale_note(
                                  getattr(result, 'dn_dc_supplied', True)))
    if not scale_free and not getattr(result, 'dn_dc_supplied', True):
        return _NO_DN_DC_SCALE_NOTE
    note = getattr(result, 'reliability_note', '') or ''
    # Gated on the COLUMN's own flag (round 2 f7 / the 2026-08-27 review): a note is the
    # reason THIS column is unreliable, so a reason that leaves the column reliable -- an
    # Rg-only cause on an Mw cell -- must not travel here. Every cause that does gate
    # Mw/A2 (ill-conditioned, subnormal, non-positive divisor) also drops the flag, so
    # nothing that belongs on this cell is lost.
    if note and not reliable:
        return note
    if not reliable:
        return 'unreliable (non-physical or non-finite fit)'
    return ''


def export_debye(result, file_path: str, delimiter: str = ',') -> str:
    """Export a single-concentration Debye analysis (DebyeResult, apparent).

    The Mw column's Comments cell distinguishes uncalibrated (arbitrary scale) from an
    ill-conditioned design (numerically unreliable) — see `_reliability_comment`. The
    Rg column carries the size's own verdict (`_size_comment`, off `rg_reliable`): not
    scale-gated, so it names only a degenerate design or an unextractable size."""
    mw_comment = _reliability_comment(result, result.mw_reliable)
    # Anything the fit said -- the non-positive-ordinate drop. The arrays below are the
    # POST-drop ones held on the result, so this cell is the file's only trace that an
    # angle was removed (the Zimm sibling's rule).
    fit_note = _analysis_notes(result)
    columns = [
        OriginColumn('q^2', 'nm^-2', '', np.asarray(result.q2_nm2, dtype=float)),
        OriginColumn('Kc/dR', 'mol/g',
                     _join_comments(fit_note, _ordinate_scale_note(result, 'Debye ordinate')),
                     np.asarray(result.kc_over_dR, dtype=float)),
        _scalar_column('Concentration', 'g/mL', result.concentration_g_per_mL),
        _scalar_column('Mw (apparent)', 'g/mol', result.mw_apparent_g_per_mol,
                       comments=mw_comment or 'APPARENT, single concentration'),
        _scalar_column('Mw SE', 'g/mol', result.mw_apparent_se,
                       comments=_join_comments(mw_comment, _se_note(result),
                                               'statistical (excl. calibration/dn-dc)')),
        _scalar_column('Rg (apparent)', 'nm', result.rg_apparent_nm,
                       comments=_size_comment(result)),
        _scalar_column('Rg SE', 'nm', result.rg_apparent_se,
                       comments=_join_comments(_size_comment(result), _se_note(result),
                                               'statistical (excl. calibration/dn-dc)')),
        _scalar_column('Intercept', 'mol/g', result.intercept_mol_per_g),
        _scalar_column('Slope', 'mol nm^2/g', result.slope),
        _scalar_column('R^2', '', result.r_squared),
        _scalar_column('Is apparent', '', result.is_apparent),
    ]
    return write_origin_csv(file_path, columns, delimiter)


def export_guinier(result, file_path: str, delimiter: str = ',') -> str:
    """Export a single-concentration Guinier analysis (GuinierResult, apparent).

    The Mw column carries the uncalibrated marker in its Comments cell when the
    run was uncalibrated, or the ill-conditioned marker when the design is degenerate
    (`_reliability_comment`). Rg comes from the slope and is not scale-gated; its own
    verdict (`_size_comment`, off `rg_reliable` -- a degenerate design or a fit outside
    the Guinier regime) lands on the Rg column, and the qRg (max) column names the
    validity limit that verdict was judged against.

    The ln(dR) ordinate is on the excess-Rayleigh scale, so an uncalibrated run offsets
    it by the unknown scale (its intercept, hence the apparent Mw, moves with it) and
    the column carries the calibration marker plus the points' provenance; it does not
    carry K, so the dn/dc marker stays off it (`_ordinate_scale_note(optical=False)`).
    """
    mw_comment = _reliability_comment(result, result.mw_reliable)
    columns = [
        OriginColumn('q^2', 'nm^-2', '', np.asarray(result.q2_nm2, dtype=float)),
        OriginColumn('ln(dR)', '',
                     _join_comments(_analysis_notes(result),
                                    _ordinate_scale_note(result, 'log excess Rayleigh ratio',
                                                         optical=False)),
                     np.asarray(result.ln_excess_rayleigh, dtype=float)),
        _scalar_column('Concentration', 'g/mL', result.concentration_g_per_mL),
        _scalar_column('Rg (apparent)', 'nm', result.rg_nm,
                       comments=_size_comment(result)),
        _scalar_column('Rg SE', 'nm', result.rg_se,
                       comments=_join_comments(_size_comment(result), _se_note(result),
                                               'statistical (excl. calibration/dn-dc)')),
        _scalar_column('Mw (apparent)', 'g/mol', result.mw_apparent_g_per_mol,
                       comments=mw_comment or 'APPARENT, single concentration'),
        _scalar_column('Mw SE', 'g/mol', result.mw_apparent_se,
                       comments=_join_comments(mw_comment, _se_note(result),
                                               'statistical (excl. calibration/dn-dc)')),
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

    The Mw column carries the uncalibrated marker in its Comments cell when the run
    was uncalibrated (the value is then on an arbitrary scale), the ill-conditioned
    marker when the divisor is subnormal, else the APPARENT default — mirroring
    Debye/Guinier via `_reliability_comment`. There is no ordinate column here, so the
    points' provenance (a DLS-derived intensity, a non-default solvent blank) rides the
    Mw cell too: it is the one number the file has, and it was computed from them.
    """
    mw_comment = _join_comments(
        _reliability_comment(result, result.mw_reliable)
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

    scale_note = _scale_note(all(r.calibrated for r in samples))
    # dR does NOT carry K, so an absent dn/dc leaves the excess-Rayleigh column on a
    # perfectly good scale; only the ordinate that multiplies by K is affected.
    optical_note = _optical_scale_note(
        all(getattr(r, 'dn_dc_supplied', True) for r in samples))
    # Where the intensities came from, when they were not an instrument's own static
    # reading. Independent of the two scale notes and joined with them rather than
    # chosen between: a derived series can also be uncalibrated.
    derived_note = _join_comments(_derived_note(samples), _blank_note(samples))
    dR_note = _join_comments(scale_note, derived_note) or 'dR(theta)'
    kc_note = (_join_comments(scale_note, optical_note, derived_note)
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
    verdict (`_size_comment`, off `rg_reliable` -- a degenerate design) joins the Rg
    column's Comments beside the fit note.

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
    scale_note = _join_comments(
        _scale_note(all(r.calibrated for r in samples)),
        _optical_scale_note(all(getattr(r, 'dn_dc_supplied', True) for r in samples)),
        _derived_note(samples))

    # Anything the fit itself said -- today only Berry's "dropped N point(s) with a
    # non-positive Kc/dR". It has to reach the file, because the ordinate columns below
    # are the PRE-drop arrays straight off the Rayleigh results: the export shows every
    # point beside a fit that used fewer. Independent of the scale note, hence joined
    # rather than ranked (`_join_comments`).
    fit_note = _analysis_notes(zimm_result)

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
            # The blank note is PER COLUMN: each concentration names the blank it
            # used, never a sibling's (the intercepts below, fitted over all of
            # them, name every distinct one).
            _join_comments(fit_note, scale_note,
                           str(getattr(r, 'blank_provenance', '') or '')) or 'Zimm ordinate',
            cells,
            parameter=f'{r.concentration_g_per_mL:.6g}',
        ))

    # Extrapolated grid intercepts (for plotting the c->0 and q->0 lines).
    # The two grid intercepts are per-level fits over the SAME post-drop arrays as the
    # headline fit, so a dropped point moves them and they carry the note with it.
    columns.append(OriginColumn(
        'Intercept vs c', 'mol/g' if zimm_result.method == 'zimm' else '(mol/g)^0.5',
        _join_comments(fit_note, _blank_note(samples),
                       scale_note or 'q->0 intercept per concentration'),
        np.asarray(zimm_result.intercept_per_concentration, dtype=float)))
    columns.append(OriginColumn(
        'Concentration (grid)', 'g/mL', 'for the q->0 line',
        np.asarray(zimm_result.concentrations_g_per_mL, dtype=float)))
    # The c->0 intercepts are fitted over the same blank-subtracted series as the q->0
    # ones, so they name the same blanks (the sibling column above already did).
    columns.append(OriginColumn(
        'Intercept vs q^2', 'mol/g' if zimm_result.method == 'zimm' else '(mol/g)^0.5',
        _join_comments(fit_note, _blank_note(samples),
                       scale_note or 'c->0 intercept per angle'),
        np.asarray(zimm_result.intercept_per_angle, dtype=float)))
    columns.append(OriginColumn(
        'q^2 (grid)', 'nm^-2', 'for the c->0 line',
        np.asarray(zimm_result.q2_nm2, dtype=float)))

    # Thermodynamic results. Mw and A2 carry the uncalibrated marker (arbitrary scale),
    # the ill-conditioned marker (degenerate design), or a generic unreliable marker
    # (non-physical / non-finite) in their Comments cell; a healthy calibrated run is the
    # silent default. Pass each column's own flag so a value unreliable for a non-
    # conditioning reason is not mislabelled healthy (`_reliability_comment`).
    mw_note = _reliability_comment(zimm_result, zimm_result.mw_reliable)
    a2_note = _reliability_comment(zimm_result, zimm_result.a2_reliable)
    columns += [
        _scalar_column('Method', '', zimm_result.method),
        _scalar_column('Mw', 'g/mol', zimm_result.mw_g_per_mol,
                       comments=_join_comments(fit_note, mw_note) or 'thermodynamic'),
        _scalar_column('Mw SE', 'g/mol', zimm_result.mw_se,
                       comments=_join_comments(mw_note, _se_note(zimm_result),
                                               'statistical (excl. calibration/dn-dc)')),
        _scalar_column('Rg', 'nm', zimm_result.rg_nm,
                       comments=_join_comments(fit_note, _size_comment(zimm_result))),
        _scalar_column('Rg SE', 'nm', zimm_result.rg_se,
                       comments=_join_comments(_size_comment(zimm_result),
                                               _se_note(zimm_result),
                                               'statistical (excl. calibration/dn-dc)')),
        _scalar_column('A2', 'mol mL/g^2', zimm_result.a2_mol_mL_per_g2,
                       comments=_join_comments(fit_note, a2_note)),
        _scalar_column('A2 SE', 'mol mL/g^2', zimm_result.a2_se,
                       comments=_join_comments(a2_note, _se_note(zimm_result),
                                               'statistical (excl. calibration/dn-dc)')),
        _scalar_column('R^2', '', zimm_result.r_squared),
        _scalar_column('Is apparent', '', zimm_result.is_apparent),
    ]
    return write_origin_csv(file_path, columns, delimiter)


def export_form_factor(result, file_path: str, delimiter: str = ',') -> str:
    """Export a full-angular form-factor P(q) fit (FormFactorResult).

    A long table of the fitted points (q, Kc/dR, concentration) plus the fitted scalars:
    model, Mw, Rg (and, for a sphere, the geometric radius R), and A2 (thermodynamic only).
    Mw carries the uncalibrated / ill-conditioned / non-physical marker in its Comments cell
    exactly as Debye/Zimm do (`_reliability_comment`); Rg and R survive an uncalibrated run
    (angular shape, scale cancels) so they carry no such marker. The apparent/thermodynamic
    distinction is written both as the `Is apparent` flag and in the Mw comment default.
    """
    mw_note = _reliability_comment(result, result.mw_reliable)
    a2_note = ('' if result.is_apparent
               else _reliability_comment(result, result.a2_reliable))
    regime = 'APPARENT, single concentration' if result.is_apparent else 'thermodynamic'
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
                       comments=_join_comments(mw_note, _se_note(result),
                                               'statistical (excl. calibration/dn-dc)')),
        # The size columns carry their OWN verdict: `rg_reliable` is False when the fit
        # collapsed below the size floor, where the size is the defect rather than a
        # survivor of an unknown scale (round 9c44 f1 / 943d f1). R is a fixed multiple
        # of Rg, so it carries the same note.
        _scalar_column('Rg', 'nm', result.rg_nm, comments=_size_comment(result)),
        _scalar_column('Rg SE', 'nm', result.rg_se,
                       comments=_join_comments(_size_comment(result), _se_note(result),
                                               'statistical (excl. calibration/dn-dc)')),
        # Geometric sphere radius (nan for the coil model -> a blank cell).
        _scalar_column('R (sphere)', 'nm', result.r_nm, comments=_size_comment(result)),
        _scalar_column('R SE', 'nm', result.r_se,
                       comments=_join_comments(_size_comment(result), _se_note(result),
                                               'statistical (excl. calibration/dn-dc)')),
        _scalar_column('A2', 'mol mL/g^2', result.a2_mol_mL_per_g2, comments=a2_note),
        _scalar_column('A2 SE', 'mol mL/g^2', result.a2_se,
                       comments=_join_comments(a2_note, _se_note(result),
                                               'statistical (excl. calibration/dn-dc)')),
        _scalar_column('qRg (max)', '', result.qrg_max),
        # Model-specific reason on the invalid case: the result's own reliability_note carries
        # the correct text (sphere first-minimum vs Mie deep-minimum), so don't hard-code sphere
        # wording for a flagged Mie fit.
        _scalar_column('Form factor valid', '', result.form_factor_valid,
                       comments='' if result.form_factor_valid
                       else (result.reliability_note
                             or 'fit past a form-factor minimum; Mw/A2 unreliable')),
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
    the `Degenerate` flag), never folded into a +/-. Mw carries the uncalibrated / ill-
    conditioned marker, the axial ratio carries the unresolved-at-low-qRg marker, and
    `Form factor valid` carries the reason a fit past a form-factor minimum is unreliable
    (which `_reliability_comment` alone would hide behind the uncalibrated marker) in
    their Comments cells; Rg / the axial ratio survive an uncalibrated run (scale cancels).
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
        mw_note = _reliability_comment(r, r.mw_reliable)
        regime = 'APPARENT, single concentration' if r.is_apparent else 'thermodynamic'
        model_note = (f'ellipsoid; degenerate ({r.n_solutions} solutions)'
                      if r.degenerate else 'ellipsoid')
        ax_note = ('' if r.axial_ratio_reliable
                   else 'axial ratio unresolved (qRg too small to distinguish shape)')
        a2_note = '' if r.is_apparent else _reliability_comment(r, r.a2_reliable)
        columns += [
            _scalar_column(p + 'Model', '', r.model, comments=model_note),
            _scalar_column(p + 'Mw', 'g/mol', r.mw_g_per_mol, comments=mw_note or regime),
            _scalar_column(p + 'Mw SE', 'g/mol', r.mw_se,
                           comments=_join_comments(mw_note, _se_note(r),
                                                   'statistical (excl. calibration/dn-dc)')),
            _scalar_column(p + 'Rg', 'nm', r.rg_nm, comments=_size_comment(r)),
            _scalar_column(p + 'Rg SE', 'nm', r.rg_se,
                           comments=_join_comments(_size_comment(r), _se_note(r),
                                                   'statistical (excl. calibration/dn-dc)')),
            _scalar_column(p + 'Axial ratio', '', r.axial_ratio,
                           comments=ax_note or 'polar/equatorial (>1 prolate, <1 oblate)'),
            _scalar_column(p + 'Axial ratio SE', '', r.axial_ratio_se,
                           comments=_join_comments(ax_note, _se_note(r),
                                                   'statistical (excl. calibration/dn-dc)')),
            _scalar_column(p + 'Semi-axis equatorial', 'nm', r.semi_axis_equatorial_nm),
            _scalar_column(p + 'Semi-axis polar', 'nm', r.semi_axis_polar_nm),
            _scalar_column(p + 'A2', 'mol mL/g^2', r.a2_mol_mL_per_g2, comments=a2_note),
            _scalar_column(p + 'A2 SE', 'mol mL/g^2', r.a2_se,
                           comments=_join_comments(a2_note, _se_note(r),
                                                   'statistical (excl. calibration/dn-dc)')),
            _scalar_column(p + 'qRg (max)', '', r.qrg_max),
            # The sibling both other form-factor exporters carry, with the same conditioned
            # wording. It is the only home for the validity reason: `_reliability_comment`
            # returns the arbitrary-scale marker FIRST, so on an uncalibrated fit past the
            # first minimum the Mw cell says only "uncalibrated" and the reason would have
            # nowhere else to go.
            _scalar_column(p + 'Form factor valid', '', r.form_factor_valid,
                           comments='' if r.form_factor_valid
                           else (r.reliability_note
                                 or 'fit past a form-factor minimum; Mw/A2 unreliable')),
            _scalar_column(p + 'Axial ratio resolved', '', r.axial_ratio_reliable,
                           comments=ax_note),
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
        mw_note = _reliability_comment(r, r.mw_reliable)
        regime = 'APPARENT, single concentration' if r.is_apparent else 'thermodynamic'
        model_note = (f'core-shell; degenerate ({r.n_solutions} solutions)'
                      if r.degenerate else 'core-shell')
        # Three-way, because `rg_reliable` has three causes and only ONE of them is the
        # contrast profile: a collapsed fit (r_outer below the size floor) has a finite Rg
        # and must not be told the contrasts cancelled (round 943d f2, export side).
        if r.rg_undefined:
            rg_note = ('no defined scattering Rg for this contrast profile (contrasts cancel, '
                       'or radial moments oppose)')
        elif _size_comment(r):
            rg_note = _size_comment(r)
        else:
            rg_note = 'contrast-weighted; no SE (nonlinear combo of a degenerate radius pair)'
        a2_note = '' if r.is_apparent else _reliability_comment(r, r.a2_reliable)
        columns += [
            _scalar_column(p + 'Model', '', r.model, comments=model_note),
            _scalar_column(p + 'Mw', 'g/mol', r.mw_g_per_mol, comments=mw_note or regime),
            _scalar_column(p + 'Mw SE', 'g/mol', r.mw_se,
                           comments=_join_comments(mw_note, _se_note(r),
                                                   'statistical (excl. calibration/dn-dc)')),
            _scalar_column(p + 'R_core', 'nm', r.r_core_nm),
            # The two radii carry no per-column marker of their own (see `rg_note` for
            # the contrast-weighted Rg), so their SEs take the estimator tag + label alone.
            _scalar_column(p + 'R_core SE', 'nm', r.r_core_se,
                           comments=_join_comments(_se_note(r),
                                                   'statistical (excl. calibration/dn-dc)')),
            _scalar_column(p + 'R_outer', 'nm', r.r_outer_nm),
            _scalar_column(p + 'R_outer SE', 'nm', r.r_outer_se,
                           comments=_join_comments(_se_note(r),
                                                   'statistical (excl. calibration/dn-dc)')),
            _scalar_column(p + 'Shell thickness', 'nm', r.shell_thickness_nm),
            _scalar_column(p + 'Rg (contrast-weighted)', 'nm', r.rg_nm, comments=rg_note),
            _scalar_column(p + 'A2', 'mol mL/g^2', r.a2_mol_mL_per_g2, comments=a2_note),
            _scalar_column(p + 'A2 SE', 'mol mL/g^2', r.a2_se,
                           comments=_join_comments(a2_note, _se_note(r),
                                                   'statistical (excl. calibration/dn-dc)')),
            _scalar_column(p + 'qRg (max)', '', r.qrg_max),
            _scalar_column(p + 'Form factor valid', '', r.form_factor_valid,
                           comments='' if r.form_factor_valid
                           else (r.reliability_note
                                 or 'fit past a form-factor minimum; Mw/A2 unreliable')),
            _scalar_column(p + 'Degenerate', '', r.degenerate),
            # Conditioned on the flag, like the `Form factor valid` sibling above and the
            # ellipsoid exporter's `Axial ratio resolved`: written unconditionally it would
            # deny a defined Rg beside the number, which the degenerate small-particle
            # regime (rg_undefined False, Rg ~ 27 nm) produces.
            _scalar_column(p + 'Rg undefined', '', r.rg_undefined,
                           comments=('contrast profile has no defined scattering Rg'
                                     if r.rg_undefined else '')),
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
                   n_excluded: int = 0) -> str:
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
        _scalar_column('Quantity', '', quantity),
        _scalar_column(f'Exponent ({exp_name})', '', fit.exponent),
        _scalar_column('Exponent SE', '', fit.exponent_se,
                       comments=_join_comments(_se_note(fit),
                                               'statistical (log-log regression)')),
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
                '(Zimm & Stockmayer Eqs. 47a/47b; model-dependent; no separate SE -- '
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

    The 2 A2 Mw / A2 columns carry the ill-conditioned marker in their Comments cell when
    the Y-vs-c design is near-degenerate (near-equal concentrations). See
    `_reliability_comment`.

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
    """
    note = _reliability_comment(result, getattr(result, 'two_a2_mw_reliable', True),
                                scale_free=True)
    # Anything the fit said -- today only "dropped N point(s) with a non-positive excess
    # Rayleigh ratio". Sharper here than in the Zimm sibling: the arrays below are the
    # POST-drop ones held on the result, so the cell is the file's only trace that a
    # concentration was removed at all. Joined with where the intensities came from and
    # which solvent blank they were referenced against (both empty in the common case):
    # neither cancels in Y(c) the way the two scales do, so the cells built from Y(c)
    # carry them -- the shape `export_mode_partition`'s fit note takes.
    fit_note = _join_comments(
        _analysis_notes(result),
        str(getattr(result, 'derived_provenance', '') or ''),
        str(getattr(result, 'blank_provenance', '') or ''))
    columns = [
        OriginColumn('Concentration', 'g/mL', fit_note,
                     np.asarray(result.concentrations_g_per_mL, dtype=float)),
        OriginColumn('Y', '', _join_comments('[I(c_ref)/c_ref] / [I(c)/c]', fit_note),
                     np.asarray(result.Y, dtype=float)),
        _scalar_column('Angle', 'deg', result.angle_deg),
        _scalar_column('2 A2 Mw', '', result.two_a2_mw,
                       _join_comments(fit_note, note) or 'slope / intercept'),
        _scalar_column('2 A2 Mw SE', '', result.two_a2_mw_se,
                       comments=_join_comments(fit_note, note, _se_note(result),
                                               'statistical (calibration/dn-dc-free)')),
        _scalar_column('A2', 'mol mL/g^2', result.a2_mol_mL_per_g2,
                       _join_comments(fit_note, note or _CALFREE_A2_MW_CONDITION)),
        _scalar_column('A2 SE', 'mol mL/g^2', result.a2_se,
                       comments=_join_comments(
                           fit_note, note, _se_note(result),
                           f'statistical, Mw treated as exact; {_CALFREE_A2_MW_CONDITION}')),
        _scalar_column('Slope', '', result.slope),
        _scalar_column('Intercept', '', result.intercept),
        _scalar_column('R^2', '', result.r_squared),
    ]
    return write_origin_csv(file_path, columns, delimiter)
