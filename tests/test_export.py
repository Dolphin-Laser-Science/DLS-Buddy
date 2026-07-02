"""Regression tests for the Origin-compatible CSV export layer (exporting/export.py).

Covers the low-level writer contract (the four label rows Long Name / Units /
Comments / Parameters, then data; ``_fmt`` cell rendering; the empty-columns
guard) and the per-analysis exporters, with particular attention to the
uncalibrated-comment injection invariant: an uncalibrated run marks ONLY the
affected columns' Comments cell with "uncalibrated, arbitrary scale" (no extra
rows, so the Origin import row count is unchanged), and Rg -- which survives an
uncalibrated run because the unknown scale cancels in the slope -- is never marked.

Inputs are built from the shared synthetic forward model + the real analysis
functions, so the exported objects are the genuine result dataclasses.
"""
from __future__ import annotations

import csv

import numpy as np
import pytest

from exporting import export as X
from fixtures.synthetic_dls import monomodal
from analysis import dls as E
from analysis import sls
from analysis import synthetic_dataset as synth
from core.data_models import SLSMeasurement

_UNCAL = "uncalibrated, arbitrary scale"


# ------------------------------------------------------------ helpers ---

def _read_csv(path):
    """Read an exported CSV back into (rows, {long_name: column_index})."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header = {name: i for i, name in enumerate(rows[0])}
    return rows, header


def _label_rows(rows):
    """The four Origin label rows: long names, units, comments, parameters."""
    return rows[0], rows[1], rows[2], rows[3]


def _build_uncalibrated_zimm():
    """A genuine uncalibrated Zimm dataset: per-concentration Rayleigh ratios
    (k_c = None) + the ZimmBerryResult from them. Rg is recoverable; Mw/A2 are not."""
    mw, rg, a2 = 1.0e6, 40.0, 9.0e-5
    angles = [40, 60, 80, 100, 120]
    concs = [2.0e-4, 4.0e-4, 6.0e-4, 8.0e-4]
    cal = synth.CalibrationSpec()
    sset = synth.build_sls_set(
        mw=mw, rg_nm=rg, a2_mol_mL_per_g2=a2, angles_deg=angles,
        concentrations_g_per_mL=concs, wavelength_nm=532.0,
        temperature_K=298.15, n_solvent=1.33, dn_dc=0.135, cal=cal,
        solvent_intensity_90=6000.0, polymer_name="PEG", solvent_name="water")

    def _meas(c):
        return SLSMeasurement(
            angles_deg=np.asarray(sset.angles_deg, dtype=float),
            intensities=np.asarray(sset.intensities[c], dtype=float),
            polymer_name="PEG", solvent_name="water", concentration_g_per_mL=c,
            temperature_K=298.15, wavelength_nm=532.0,
            solvent_refractive_index=1.33, dn_dc_mL_per_g=0.135)

    solvent = _meas(0.0)
    rr = [sls.compute_excess_rayleigh_ratio(
              _meas(c), solvent, calibration_constant=None,     # uncalibrated
              standard_refractive_index=cal.n_standard)
          for c in concs]
    zr = sls.zimm_analysis(rr, method="zimm")
    return rr, zr, rg


# ------------------------------------------------------- core writer ---

def test_origin_csv_label_rows_then_data(tmp_path):
    cols = [
        X.OriginColumn("Time", "s", "the clock", [0.0, 1.0, 2.0], parameter="p1"),
        X.OriginColumn("Signal", "V", "measured", [9.0, 8.0], parameter="p2"),
    ]
    out = tmp_path / "basic.csv"
    X.write_origin_csv(str(out), cols)
    rows, header = _read_csv(out)

    long_names, units, comments, params = _label_rows(rows)
    assert long_names == ["Time", "Signal"]
    assert units == ["s", "V"]
    assert comments == ["the clock", "measured"]
    assert params == ["p1", "p2"]
    # data starts at row 5 (index 4); shorter column padded with an empty cell.
    assert rows[4] == ["0", "9"]
    assert rows[5] == ["1", "8"]
    assert rows[6] == ["2", ""]          # Signal column ran out -> empty (missing)


def test_fmt_cell_rendering():
    assert X._fmt(None) == ""
    assert X._fmt(float("nan")) == ""
    assert X._fmt(float("inf")) == ""
    assert X._fmt(-float("inf")) == ""
    assert X._fmt(3) == "3"
    assert X._fmt(np.int64(7)) == "7"
    assert X._fmt(1.5) == "1.5"
    assert X._fmt("text") == "text"


def test_write_origin_csv_rejects_empty_columns(tmp_path):
    with pytest.raises(ValueError):
        X.write_origin_csv(str(tmp_path / "empty.csv"), [])


# ------------------------------------------ uncalibrated comment injection ---

def test_zimm_uncalibrated_marks_only_mw_and_a2(tmp_path):
    rr, zr, _rg = _build_uncalibrated_zimm()
    assert zr.mw_reliable is False
    assert zr.a2_reliable is False

    out = tmp_path / "zimm.csv"
    X.export_zimm(rr, zr, str(out))
    rows, header = _read_csv(out)
    _ln, _units, comments, _params = _label_rows(rows)

    # Mw and A2 columns carry the uncalibrated note in their Comments cell...
    assert comments[header["Mw"]] == _UNCAL
    assert comments[header["A2"]] == _UNCAL
    # ...but Rg is NOT marked (its slope-derived value survives an uncalibrated run).
    assert "Rg" in header
    assert _UNCAL not in comments[header["Rg"]]

    # the injection adds NO extra rows: 4 label rows + the wide-table data rows
    # (the abscissa is q^2, one row per angle) -- the note lives in a header cell.
    assert len(rows) == 4 + np.asarray(rr[0].q2_nm2).size


def test_zimm_calibrated_is_unmarked(tmp_path):
    # Same dataset but WITH calibration -> Mw/A2 reliable, no note anywhere.
    mw, rg, a2 = 1.0e6, 40.0, 9.0e-5
    angles = [40, 60, 80, 100, 120]
    concs = [2.0e-4, 4.0e-4, 6.0e-4, 8.0e-4]
    cal = synth.CalibrationSpec()
    sset = synth.build_sls_set(
        mw=mw, rg_nm=rg, a2_mol_mL_per_g2=a2, angles_deg=angles,
        concentrations_g_per_mL=concs, wavelength_nm=532.0, temperature_K=298.15,
        n_solvent=1.33, dn_dc=0.135, cal=cal, solvent_intensity_90=6000.0,
        polymer_name="PEG", solvent_name="water")

    def _meas(c):
        return SLSMeasurement(
            angles_deg=np.asarray(sset.angles_deg, dtype=float),
            intensities=np.asarray(sset.intensities[c], dtype=float),
            polymer_name="PEG", solvent_name="water", concentration_g_per_mL=c,
            temperature_K=298.15, wavelength_nm=532.0,
            solvent_refractive_index=1.33, dn_dc_mL_per_g=0.135)

    solvent = _meas(0.0)
    rr = [sls.compute_excess_rayleigh_ratio(
              _meas(c), solvent, calibration_constant=cal.k_c(),
              standard_refractive_index=cal.n_standard)
          for c in concs]
    zr = sls.zimm_analysis(rr, method="zimm")
    assert zr.mw_reliable is True

    out = tmp_path / "zimm_cal.csv"
    X.export_zimm(rr, zr, str(out))
    rows, header = _read_csv(out)
    comments = rows[2]
    assert comments[header["Mw"]] != _UNCAL
    assert comments[header["A2"]] != _UNCAL
    # a calibrated Zimm recovers Mw and Rg to the ground truth
    assert zr.mw_g_per_mol == pytest.approx(mw, rel=0.05)
    assert zr.rg_nm == pytest.approx(rg, rel=0.05)


def test_export_zimm_no_samples_raises(tmp_path):
    # export_zimm filters out the c = 0 reference; with no non-zero results it
    # has nothing to write and must raise.
    _rr, zr, _ = _build_uncalibrated_zimm()
    with pytest.raises(ValueError):
        X.export_zimm([], zr, str(tmp_path / "none.csv"))


# --------------------------------------------- correlogram-fit export ---

def test_export_correlogram_fit_cumulant(tmp_path):
    m = monomodal(30.0)
    res = E.fit_cumulants(m, order=2, method="nonlinear")
    out = tmp_path / "cumfit.csv"
    X.export_correlogram_fit(m, res, str(out))
    rows, header = _read_csv(out)

    for name in ("Delay time", "g2-1 (data)", "g2-1 (fit)", "Residual",
                 "Rh", "Gamma", "PDI", "Cumulant method"):
        assert name in header
    # the fitted-curve columns have the same length as the fit tau axis
    n = res.fit_tau_s.size
    assert len(rows) == 4 + n                # 4 label rows + n data rows
    # scalar Rh column, length-1, matches the result
    rh_col = header["Rh"]
    assert float(rows[4][rh_col]) == pytest.approx(res.rh_nm, rel=1e-6)


# ------------------------------------------------- distribution export ---

def test_export_distribution_nnls(tmp_path):
    m = monomodal(50.0)
    res = E.fit_nnls(m, rh_min_nm=1.0, rh_max_nm=1000.0, n_grid=60)
    out = tmp_path / "dist.csv"
    X.export_distribution(res, str(out))
    _rows, header = _read_csv(out)
    for name in ("Rh", "Gamma", "Weight", "Method", "Peak Rh", "Mean Rh"):
        assert name in header


# ---------------------------------------------------------- DDLS export ---

def test_export_ddls(tmp_path):
    # Build a small VV/VH rate set with a known D_t / D_r and export it.
    from analysis import depolarization as depol
    from fixtures.synthetic_dls import q_m

    d_t, d_r = 1.5e-11, 4.0e4
    angles = [40.0, 70.0, 100.0, 130.0]
    points = []
    for a in angles:
        q = q_m(a, 633.0, 1.33)
        g_vv = d_t * q * q
        g_vh = g_vv + 6.0 * d_r
        points.append(depol.DDLSRatePoint(
            angle_deg=a, q_m_inv=q, gamma_vv_s_inv=g_vv, gamma_vh_s_inv=g_vh))
    res = depol.analyze_ddls(points, temperature_K=298.15, viscosity_Pa_s=8.9e-4)

    out = tmp_path / "ddls.csv"
    X.export_ddls(res, str(out))
    _rows, header = _read_csv(out)
    for name in ("Angle", "q^2", "Gamma_VV", "Gamma_VH", "D_r (per angle)",
                 "D_t", "D_r", "Rh_t"):
        assert name in header
    # the forward D_t / D_r are recovered by the analysis
    assert res.d_r_rad2_s == pytest.approx(d_r, rel=1e-6)
    assert res.d_t_m2_s == pytest.approx(d_t, rel=1e-6)
