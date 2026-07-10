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
    """Format a single cell. Non-finite numbers become empty (Origin missing)."""
    if value is None:
        return ''
    # bool is an int subclass, so this branch must precede the int branch;
    # matches the 'yes'/'' style used in export_dls_summary's is_average column.
    if isinstance(value, (bool, np.bool_)):
        return 'yes' if value else ''
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

    columns = [
        OriginColumn('Sample', '', 'sample identity', col('sample')),
        OriginColumn('Measurement', '', 'angle / concentration / fraction (blank '
                     'for sample-level rows)', col('measurement')),
        OriginColumn('Method', '', 'fit method or sample-level source', col('method')),
        OriginColumn('Rh', 'nm', 'hydrodynamic radius', col('rh')),
        OriginColumn('Rh_SE', 'nm', 'statistical SE (blank when none is honest)',
                     col('rh_se')),
        OriginColumn('SE estimator', '', 'covariance estimator behind Rh_SE for a '
                     'regression source; blank = HC3 default or a non-regression SE',
                     ['classical OLS' if r.get('se_estimator') == 'ols' else ''
                      for r in records]),
        OriginColumn('PDI', '', 'cumulant polydispersity index', col('pdi')),
        OriginColumn('Intensity fraction', '%', 'intensity-weighted peak area '
                     '(NOT mass/weight percent)', col('int_pct')),
        OriginColumn('Rh_fast', 'nm', 'double-exponential fast mode', col('rh_fast')),
        OriginColumn('Rh_slow', 'nm', 'double-exponential slow mode', col('rh_slow')),
        OriginColumn('Rh_Type', '', 'apparent / thermodynamic', col('rh_type')),
        OriginColumn('is_average', '', 'replicate-averaged sample Rh',
                     ['yes' if r.get('is_average') else '' for r in records]),
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
        columns += [
            _scalar_column('Cumulant order', '', result.order),
            _scalar_column('Cumulant method', '', getattr(result, 'method', 'linear')),
            _scalar_column('Gamma', '1/s', result.gamma_s_inv),
            _scalar_column('Rh', 'nm', result.rh_nm),
            _scalar_column('D', 'm^2/s', result.d_m2_s),
            _scalar_column('PDI', '', result.pdi),
            _scalar_column('PDI valid', '', result.pdi_valid),
            _scalar_column('mu2', '1/s^2', result.mu2_s_inv2),
            _scalar_column('beta', '', result.beta),
            _scalar_column('baseline B', '', getattr(result, 'baseline', 0.0)),
        ]
    elif cls == 'SingleExponentialResult':
        columns += [
            _scalar_column('Gamma', '1/s', result.mode.gamma_s_inv),
            _scalar_column('Rh', 'nm', result.mode.rh_nm),
            _scalar_column('D', 'm^2/s', result.mode.d_m2_s),
            _scalar_column('beta', '', result.beta),
        ]
    elif cls == 'DoubleExponentialResult':
        columns += [
            _scalar_column('Gamma (fast)', '1/s', result.mode1.gamma_s_inv),
            _scalar_column('Rh (fast)', 'nm', result.mode1.rh_nm),
            _scalar_column('Amplitude (fast)', '', result.mode1.amplitude_fraction),
            _scalar_column('Gamma (slow)', '1/s', result.mode2.gamma_s_inv),
            _scalar_column('Rh (slow)', 'nm', result.mode2.rh_nm),
            _scalar_column('Amplitude (slow)', '', result.mode2.amplitude_fraction),
            _scalar_column('beta', '', result.beta),
        ]
    elif cls == 'KWWResult':
        columns += [
            _scalar_column('tau_c', 's', result.tau_c_s),
            _scalar_column('Stretch', '', result.stretch),
            _scalar_column('mean tau', 's', result.mean_tau_s),
            _scalar_column('Rh (from <tau>)', 'nm', result.rh_from_mean_tau_nm),
            _scalar_column('Rh (from tau_c)', 'nm', result.rh_from_tau_c_nm),
            _scalar_column('beta', '', result.beta),
        ]
    return write_origin_csv(file_path, columns, delimiter)


def export_distribution(result, file_path: str, axis: str = 'rh',
                        delimiter: str = ',', *,
                        alpha_selection_method: Optional[str] = None,
                        ftest_prob_reject: Optional[float] = None) -> str:
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

    columns = [
        OriginColumn('Rh', 'nm', 'hydrodynamic radius', rh[order]),
        OriginColumn('Gamma', '1/s', 'decay rate', gamma[order]),
        OriginColumn('Weight', '', 'intensity fraction', w[order]),
    ]
    # Correlogram reconstruction (separate length from the grid).
    columns += [
        OriginColumn('Delay time', 's', 'tau', np.asarray(result.fit_tau_s, dtype=float)),
        OriginColumn('g2-1 (fit)', '', 'reconstruction', np.asarray(result.fitted_g2m1, dtype=float)),
        OriginColumn('Residual', '', 'data - fit', np.asarray(result.residuals, dtype=float)),
    ]
    # CONTIN records how alpha was chosen in the alpha column's Comments cell, so a
    # distribution is never ambiguous about its regularization (mirrors the SLS
    # provenance-in-Comments convention). NNLS/lognormal pass no selection method.
    if alpha_selection_method == 'ftest':
        alpha_note = (f'alpha by F-test, p_reject={ftest_prob_reject:.2f}'
                      if ftest_prob_reject is not None else 'alpha by F-test')
    elif alpha_selection_method == 'lcurve':
        alpha_note = 'alpha by L-curve corner'
    elif alpha_selection_method == 'user':
        alpha_note = 'alpha user-supplied'
    else:
        alpha_note = ''
    columns += [
        _scalar_column('Method', '', result.method),
        _scalar_column('alpha', '', result.alpha, comments=alpha_note),
        _scalar_column('Peak Rh', 'nm', result.peak_rh_nm),
        _scalar_column('Mean Rh', 'nm', result.mean_rh_nm),
        _scalar_column('Mean Gamma', '1/s', result.mean_gamma_s_inv),
        _scalar_column('beta', '', result.beta),
        _scalar_column('beta estimated', '', result.beta_estimated),
        _scalar_column('baseline', '', result.baseline),
        _scalar_column('baseline estimated', '', result.baseline_estimated),
        _scalar_column('RMS error', '', result.rms_error),
    ]
    return write_origin_csv(file_path, columns, delimiter)


def export_lcurve(lcurve, file_path: str, delimiter: str = ',') -> str:
    """Export a CONTIN L-curve sweep (LCurveResult) for inspection."""
    columns = [
        OriginColumn('alpha', '', 'regularization parameter',
                     np.asarray(lcurve.alphas, dtype=float)),
        OriginColumn('Residual norm', '', '||A x - y||^2',
                     np.asarray(lcurve.residual_norms, dtype=float)),
        OriginColumn('Solution norm', '', '||x||^2',
                     np.asarray(lcurve.solution_norms, dtype=float)),
        _scalar_column('Optimal alpha', '', lcurve.optimal_alpha),
        _scalar_column('Optimal index', '', lcurve.optimal_index),
    ]
    return write_origin_csv(file_path, columns, delimiter)


def export_gamma_q2(result, file_path: str, delimiter: str = ',') -> str:
    """Export a Gamma-vs-q^2 multi-angle analysis (GammaQ2Result)."""
    columns = [
        OriginColumn('Angle', 'deg', '', np.asarray(result.angles_deg, dtype=float)),
        OriginColumn('q^2', 'm^-2', 'scattering vector squared',
                     np.asarray(result.q2_m2, dtype=float)),
        OriginColumn('Gamma', '1/s', 'decay rate',
                     np.asarray(result.gamma_s_inv, dtype=float)),
        OriginColumn('D_app', 'm^2/s', 'Gamma / q^2',
                     np.asarray(result.d_app_m2_s, dtype=float)),
        _scalar_column('D', 'm^2/s', result.d_m2_s, 'through-origin slope'),
        _scalar_column('D SE', 'm^2/s', result.d_se,
                       comments=_se_note(result) or 'statistical (over angles)'),
        _scalar_column('Rh', 'nm', result.rh_nm),
        _scalar_column('Rh SE', 'nm', result.rh_se,
                       comments=_se_note(result) or 'statistical (over angles)'),
        _scalar_column('R^2', '', result.r_squared),
        _scalar_column('Intercept', '1/s', result.intercept_s_inv),
        _scalar_column('Intercept (relative)', '', result.intercept_relative),
        _scalar_column('D_app trend (relative)', '', result.d_app_trend_rel),
        _scalar_column('Is diffusive', '', result.is_diffusive),
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
    """
    columns = [
        OriginColumn('Angle', 'deg', '', np.asarray(result.angles_deg, dtype=float)),
        OriginColumn('q^2', 'm^-2', 'scattering vector squared',
                     np.asarray(result.q2_m2, dtype=float)),
        OriginColumn('Gamma_VV', '1/s', 'polarized field decay rate',
                     np.asarray(result.gamma_vv_s_inv, dtype=float)),
        OriginColumn('Gamma_VH', '1/s', 'depolarized field decay rate',
                     np.asarray(result.gamma_vh_s_inv, dtype=float)),
        OriginColumn('D_r (per angle)', 'rad^2/s', '(Gamma_VH - Gamma_VV)/6',
                     np.asarray(result.d_r_per_angle, dtype=float)),
    ]
    if result.qL is not None:
        columns.append(OriginColumn('qL', '', 'single-exponential valid if < 3',
                                    np.asarray(result.qL, dtype=float)))
    columns += [
        _scalar_column('D_t', 'm^2/s', result.d_t_m2_s, 'from VV, through-origin slope'),
        _scalar_column('D_t SE', 'm^2/s', result.d_t_se,
                       comments=_se_note(result) or 'statistical only'),
        _scalar_column('D_r', 'rad^2/s', result.d_r_rad2_s, 'mean of per-angle values'),
        _scalar_column('D_r SE', 'rad^2/s', result.d_r_se, 'statistical only'),
        _scalar_column('Rh_t', 'nm', result.rh_t_nm, 'Stokes radius from D_t'),
        _scalar_column('tau_rot', 's', result.rotational_time_s, '1 / (6 D_r)'),
        _scalar_column('N angles', '', result.n_angles),
        _scalar_column('Method', '', result.method),
        _scalar_column('Single-exponential valid', '', result.single_exponential_valid),
    ]
    if shapes:
        rod = shapes.get('rod')
        if rod is not None:
            columns += [
                _scalar_column('Rod L', 'nm', rod.length_nm,
                               'MODEL: rigid cylinder (Tirado 1984), not measured'),
                _scalar_column('Rod L SE', 'nm', rod.length_se, 'Monte-Carlo'),
                _scalar_column('Rod d', 'nm', rod.diameter_nm),
                _scalar_column('Rod d SE', 'nm', rod.diameter_se, 'Monte-Carlo'),
                _scalar_column('Rod aspect p', '', rod.aspect_ratio_p),
                _scalar_column('Rod p in 2-30', '', rod.in_valid_range),
                _scalar_column('Rod converged', '', rod.converged),
            ]
        sphere = shapes.get('sphere')
        if sphere is not None:
            columns += [
                _scalar_column('Sphere R(D_r)', 'nm', sphere.radius_rot_nm,
                               'MODEL: sphere (Stokes-Einstein-Debye), not measured'),
                _scalar_column('Sphere R(D_r) SE', 'nm', sphere.radius_rot_se),
                _scalar_column('Sphere R(D_t)=Rh', 'nm', sphere.radius_trans_nm),
                _scalar_column('Sphericity ratio', '', sphere.sphericity_ratio,
                               'R(D_r)/Rh; 1 = sphere'),
                _scalar_column('Sphere consistent', '', sphere.is_consistent),
            ]
    return write_origin_csv(file_path, columns, delimiter)


def export_concentration_extrapolation(result, file_path: str,
                                       delimiter: str = ',') -> str:
    """Export a D-vs-c concentration extrapolation (ConcentrationExtrapolationResult)."""
    columns = [
        OriginColumn('Concentration', 'g/mL', '',
                     np.asarray(result.concentrations_g_per_mL, dtype=float)),
        OriginColumn('D_app', 'm^2/s', 'apparent diffusion coefficient',
                     np.asarray(result.d_values_m2_s, dtype=float)),
        _scalar_column('D0', 'm^2/s', result.d0_m2_s, 'c -> 0'),
        _scalar_column('D0 SE', 'm^2/s', result.d0_se,
                       comments=_se_note(result) or 'statistical (over concentrations)'),
        _scalar_column('Rh0', 'nm', result.rh0_nm),
        _scalar_column('Rh0 SE', 'nm', result.rh0_se,
                       comments=_se_note(result) or 'statistical (over concentrations)'),
        _scalar_column('kD', 'mL/g', result.kd_mL_per_g),
        _scalar_column('kD SE', 'mL/g', result.kd_se,
                       comments=_se_note(result) or 'statistical (over concentrations)'),
        _scalar_column('Slope', 'm^2/s/(g/mL)', result.slope),
        _scalar_column('R^2', '', result.r_squared),
    ]
    return write_origin_csv(file_path, columns, delimiter)


# ===========================================================================
# SLS exporters
# ===========================================================================

def _se_note(result) -> str:
    """Comments-cell label for a ± column: names the estimator only when it is the
    non-default classical OLS (silent for HC3, matching the 'calibrated is the silent
    default' convention). See the Theory-and-Equations-Guide §15.1."""
    return 'SE: classical OLS' if getattr(result, 'se_estimator', 'hc3') == 'ols' else ''


def export_rayleigh_ratio(result, file_path: str, delimiter: str = ',') -> str:
    """Export an excess Rayleigh ratio (RayleighRatioResult).

    If the result is uncalibrated, the dR and Kc/dR columns carry an
    "uncalibrated, arbitrary scale" note in their Comments header row (no extra
    rows, so the Origin import is unaffected). A calibrated result is the default
    and is left unmarked.
    """
    scale_note = '' if result.calibrated else 'uncalibrated, arbitrary scale'
    columns = [
        OriginColumn('Angle', 'deg', '', np.asarray(result.angles_deg, dtype=float)),
        OriginColumn('q', 'nm^-1', 'scattering vector', np.asarray(result.q_nm_inv, dtype=float)),
        OriginColumn('q^2', 'nm^-2', '', np.asarray(result.q2_nm2, dtype=float)),
        OriginColumn('Excess Rayleigh ratio', 'cm^-1', scale_note or 'dR(theta)',
                     np.asarray(result.excess_rayleigh_cm_inv, dtype=float)),
        OriginColumn('Kc/dR', 'mol/g', scale_note or 'Zimm ordinate',
                     np.asarray(result.kc_over_dR_mol_per_g, dtype=float)),
        _scalar_column('Concentration', 'g/mL', result.concentration_g_per_mL),
        _scalar_column('Optical constant K', 'mol cm^2/g^2', result.optical_constant_K),
        _scalar_column('k_c used', 'cm^-1/intensity', result.k_c_used),
        _scalar_column('RI correction', '', result.ri_correction),
        _scalar_column('dn/dc', 'mL/g', result.dn_dc_mL_per_g),
        _scalar_column('Temperature', 'K', result.temperature_K),
    ]
    return write_origin_csv(file_path, columns, delimiter)


def export_debye(result, file_path: str, delimiter: str = ',') -> str:
    """Export a single-concentration Debye analysis (DebyeResult, apparent)."""
    mw_comment = '' if result.mw_reliable else 'uncalibrated, arbitrary scale'
    columns = [
        OriginColumn('q^2', 'nm^-2', '', np.asarray(result.q2_nm2, dtype=float)),
        OriginColumn('Kc/dR', 'mol/g', 'Debye ordinate',
                     np.asarray(result.kc_over_dR, dtype=float)),
        _scalar_column('Concentration', 'g/mL', result.concentration_g_per_mL),
        _scalar_column('Mw (apparent)', 'g/mol', result.mw_apparent_g_per_mol,
                       comments=mw_comment or 'APPARENT, single concentration'),
        _scalar_column('Mw SE', 'g/mol', result.mw_apparent_se,
                       comments=_se_note(result) or 'statistical (excl. calibration/dn-dc)'),
        _scalar_column('Rg (apparent)', 'nm', result.rg_apparent_nm),
        _scalar_column('Rg SE', 'nm', result.rg_apparent_se,
                       comments=_se_note(result) or 'statistical (excl. calibration/dn-dc)'),
        _scalar_column('Intercept', 'mol/g', result.intercept_mol_per_g),
        _scalar_column('Slope', 'mol nm^2/g', result.slope),
        _scalar_column('R^2', '', result.r_squared),
        _scalar_column('Is apparent', '', result.is_apparent),
    ]
    return write_origin_csv(file_path, columns, delimiter)


def export_guinier(result, file_path: str, delimiter: str = ',') -> str:
    """Export a single-concentration Guinier analysis (GuinierResult, apparent).

    The Mw column carries the uncalibrated marker in its Comments cell when the
    run was uncalibrated; Rg comes from the slope and stays reliable regardless.
    """
    mw_comment = '' if result.mw_reliable else 'uncalibrated, arbitrary scale'
    columns = [
        OriginColumn('q^2', 'nm^-2', '', np.asarray(result.q2_nm2, dtype=float)),
        OriginColumn('ln(dR)', '', 'log excess Rayleigh ratio',
                     np.asarray(result.ln_excess_rayleigh, dtype=float)),
        _scalar_column('Concentration', 'g/mL', result.concentration_g_per_mL),
        _scalar_column('Rg (apparent)', 'nm', result.rg_nm),
        _scalar_column('Rg SE', 'nm', result.rg_se,
                       comments=_se_note(result) or 'statistical (excl. calibration/dn-dc)'),
        _scalar_column('Mw (apparent)', 'g/mol', result.mw_apparent_g_per_mol,
                       comments=mw_comment or 'APPARENT, single concentration'),
        _scalar_column('Mw SE', 'g/mol', result.mw_apparent_se,
                       comments=_se_note(result) or 'statistical (excl. calibration/dn-dc)'),
        _scalar_column('Intercept', '', result.intercept, 'ln(dR(0))'),
        _scalar_column('Slope', 'nm^2', result.slope, '-Rg^2/3'),
        _scalar_column('qRg (max)', '', result.qrg_max),
        _scalar_column('Guinier valid', '', result.guinier_valid),
        _scalar_column('R^2', '', result.r_squared),
        _scalar_column('Is apparent', '', result.is_apparent),
    ]
    return write_origin_csv(file_path, columns, delimiter)


def export_single_angle(result, file_path: str, delimiter: str = ',') -> str:
    """Export a single-angle, single-concentration apparent Mw (SingleAngleResult).

    The Mw column carries the uncalibrated marker in its Comments cell when the run
    was uncalibrated (the value is then on an arbitrary scale), mirroring Debye/Guinier.
    """
    mw_comment = ('uncalibrated, arbitrary scale' if not result.mw_reliable else
                  'APPARENT, single angle + single concentration')
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

    scale_note = ('' if all(r.calibrated for r in samples)
                  else 'uncalibrated, arbitrary scale')
    columns = [
        OriginColumn('Concentration', 'g/mL', '', conc),
        OriginColumn('Angle', 'deg', '', angle),
        OriginColumn('q', 'nm^-1', 'scattering vector', q),
        OriginColumn('q^2', 'nm^-2', '', q2),
        OriginColumn('Excess Rayleigh ratio', 'cm^-1', scale_note or 'dR(theta)', dR),
        OriginColumn('Kc/dR', 'mol/g', scale_note or 'Zimm ordinate', kc),
    ]
    return write_origin_csv(file_path, columns, delimiter)


def export_zimm(rayleigh_results, zimm_result, file_path: str,
                delimiter: str = ',') -> str:
    """Export a Zimm/Berry dataset in wide format, plus the fitted Mw/Rg/A2.

    Produces a wide table: q^2 in the first column, then one Kc/dR column per
    concentration (the Parameters row encoding each concentration), matching the
    Origin Zimm-plot workflow. The thermodynamic results from `zimm_result` are
    appended as labeled scalar columns. The per-concentration and per-angle
    extrapolated intercepts are also written, for drawing the Zimm grid lines.

    Parameters
    ----------
    rayleigh_results : sequence of RayleighRatioResult
        One per concentration (the c = 0 solvent reference is skipped). Assumed to
        share the same angle set (standard Zimm experiment).
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

    # Shared abscissa from the first sample.
    q2 = np.asarray(samples[0].q2_nm2, dtype=float)
    columns = [OriginColumn('q^2', 'nm^-2', 'scattering vector squared', q2)]
    for r in samples:
        c_mg = r.concentration_g_per_mL * 1000.0
        columns.append(OriginColumn(
            f'Kc/dR (c={c_mg:.4g} mg/mL)', 'mol/g', 'Zimm ordinate',
            np.asarray(r.kc_over_dR_mol_per_g, dtype=float),
            parameter=f'{r.concentration_g_per_mL:.6g}',
        ))

    # Extrapolated grid intercepts (for plotting the c->0 and q->0 lines).
    columns.append(OriginColumn(
        'Intercept vs c', 'mol/g' if zimm_result.method == 'zimm' else '(mol/g)^0.5',
        'q->0 intercept per concentration',
        np.asarray(zimm_result.intercept_per_concentration, dtype=float)))
    columns.append(OriginColumn(
        'Concentration (grid)', 'g/mL', 'for the q->0 line',
        np.asarray(zimm_result.concentrations_g_per_mL, dtype=float)))
    columns.append(OriginColumn(
        'Intercept vs q^2', 'mol/g' if zimm_result.method == 'zimm' else '(mol/g)^0.5',
        'c->0 intercept per angle',
        np.asarray(zimm_result.intercept_per_angle, dtype=float)))
    columns.append(OriginColumn(
        'q^2 (grid)', 'nm^-2', 'for the c->0 line',
        np.asarray(zimm_result.q2_nm2, dtype=float)))

    # Thermodynamic results. Mw and A2 carry the uncalibrated marker in their
    # Comments cell when the run was uncalibrated; calibrated is the silent default.
    mw_note = '' if zimm_result.mw_reliable else 'uncalibrated, arbitrary scale'
    a2_note = '' if zimm_result.a2_reliable else 'uncalibrated, arbitrary scale'
    columns += [
        _scalar_column('Method', '', zimm_result.method),
        _scalar_column('Mw', 'g/mol', zimm_result.mw_g_per_mol,
                       comments=mw_note or 'thermodynamic'),
        _scalar_column('Mw SE', 'g/mol', zimm_result.mw_se,
                       comments=_se_note(zimm_result) or 'statistical (excl. calibration/dn-dc)'),
        _scalar_column('Rg', 'nm', zimm_result.rg_nm),
        _scalar_column('Rg SE', 'nm', zimm_result.rg_se,
                       comments=_se_note(zimm_result) or 'statistical (excl. calibration/dn-dc)'),
        _scalar_column('A2', 'mol mL/g^2', zimm_result.a2_mol_mL_per_g2,
                       comments=a2_note),
        _scalar_column('A2 SE', 'mol mL/g^2', zimm_result.a2_se,
                       comments=_se_note(zimm_result) or 'statistical (excl. calibration/dn-dc)'),
        _scalar_column('R^2', '', zimm_result.r_squared),
        _scalar_column('Is apparent', '', zimm_result.is_apparent),
    ]
    return write_origin_csv(file_path, columns, delimiter)


def export_scaling(quantity: str, labels, mw, y, fit, file_path: str,
                   delimiter: str = ',') -> str:
    """Export a cross-sample scaling plot (Rg-Mw / Rh-Mw / A2-Mw) and its power-law fit.

    Parameters
    ----------
    quantity : str
        'rg', 'rh', or 'a2' -- selects the y-column label/units.
    labels, mw, y : sequences
        The plotted points (one per included sample/fraction), in order.
    fit : ScalingResult
        The log-log power-law fit (duck-typed: exponent, exponent_se, prefactor,
        r_squared, n_points, fit_valid).
    file_path : str
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
                       comments=_se_note(fit) or 'statistical (log-log regression)'),
        _scalar_column('Prefactor', '', fit.prefactor, 'y at Mw = 1'),
        _scalar_column('R^2', '', fit.r_squared),
        _scalar_column('n points', '', fit.n_points),
        _scalar_column('Fit valid', '', fit.fit_valid),
    ]
    return write_origin_csv(file_path, columns, delimiter)


def export_calibration_free_a2(result, file_path: str, delimiter: str = ',') -> str:
    """Export a calibration-free A2 analysis (CalibrationFreeA2Result)."""
    columns = [
        OriginColumn('Concentration', 'g/mL', '',
                     np.asarray(result.concentrations_g_per_mL, dtype=float)),
        OriginColumn('Y', '', '[I(c_ref)/c_ref] / [I(c)/c]',
                     np.asarray(result.Y, dtype=float)),
        _scalar_column('Angle', 'deg', result.angle_deg),
        _scalar_column('2 A2 Mw', '', result.two_a2_mw, 'slope / intercept'),
        _scalar_column('2 A2 Mw SE', '', result.two_a2_mw_se,
                       comments=_se_note(result) or 'statistical (calibration/dn-dc-free)'),
        _scalar_column('A2', 'mol mL/g^2', result.a2_mol_mL_per_g2,
                       'only if Mw supplied'),
        _scalar_column('A2 SE', 'mol mL/g^2', result.a2_se,
                       comments=_se_note(result) or 'statistical (Mw treated as exact)'),
        _scalar_column('Slope', '', result.slope),
        _scalar_column('Intercept', '', result.intercept),
        _scalar_column('R^2', '', result.r_squared),
    ]
    return write_origin_csv(file_path, columns, delimiter)
