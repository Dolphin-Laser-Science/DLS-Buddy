"""Regression tests for analysis/sls.py (static light scattering).

analysis/sls.py had NO validator before this file. It is the SLS engine:
calibration constant, excess Rayleigh ratio, Zimm/Berry double extrapolation,
single-concentration Debye/Guinier, single-angle Mw, and the calibration-free
2*A2*Mw product.

Ground truth
------------
The synthetic tests use the program's OWN first-order Zimm forward model
(analysis.synthetic_dataset, via fixtures.synthetic_sls) so a clean recovery is a
closed round trip: Zimm reproduces the input Mw/Rg/A2 to numerical precision.
System = PEG in water, 532 nm, T = 298.15 K, n = 1.33, dn/dc = 0.135, calibrated
against a toluene VU standard (n = 1.496). Series (Mw g/mol, Rg nm, A2 mol*mL/g^2):
    PEG 100k  1e5 / 14 / 1.5e-4
    PEG 300k  3e5 / 27 / 1.2e-4
    PEG 1M    1e6 / 55 / 9.5e-5
    PEG 3M    3e6 / 105 / 7.0e-5

CLAUDE.md invariants exercised
------------------------------
  - K = 4 pi^2 n^2 (dn/dc)^2 / (Na lambda^4)   (== physics.optical_constant_K)
  - (n_solvent / n_standard)^2 correction applied when solvent != standard
  - apparent (single-c / single-angle) vs thermodynamic (fully extrapolated)
  - an UNCALIBRATED run flags Mw & A2 unreliable, but Rg SURVIVES (slope/intercept,
    the scale cancels) and so does the calibration-free 2*A2*Mw product
  - no ± is asserted here (single Zimm set); SEs are covered by their own module

Real-data end-to-end (@realdata): the committed Brookhaven PS(900k)/toluene Zimm
file recovers Mw ~ 1.01e6 g/mol and Rg ~ 40.5 nm through the same entry point the
controller uses (own k_c from the c = 0 toluene at 90 deg, R_VU geometry).

Notable finding: Berry does NOT reproduce the exact Zimm-form synthetic data (it
linearises sqrt(Kc/dR), which is not linear in q^2 for this generator); on PEG 3M
it underestimates (Mw ~ 2.4e6, Rg ~ 75 vs 3e6 / 105). That bias is asserted with
loose bounds and documented, not masked.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from analysis import sls
from physics import constants as C

from fixtures import synthetic_sls as S
from fixtures.data_paths import BROOKHAVEN_DIR, require


# ===========================================================================
# 1. Calibration constant
# ===========================================================================

def test_calibration_constant_value_and_geometry():
    # k_c = R_std / (I_net * sin theta). At 90 deg, sin = 1, so k_c = R_std / I.
    r_vu = C.rayleigh_ratio_toluene(532.0, 25.0, geometry="VU")
    kc = sls.compute_calibration_constant(1.0e5, 90.0, r_vu)
    assert kc == pytest.approx(r_vu / 1.0e5, rel=1e-12)

    # A CalibrationSpec's k_c() must agree with the engine (single source of truth).
    _, cal = S.make_from_truth("PEG 1M")
    assert cal.k_c() == pytest.approx(
        sls.compute_calibration_constant(cal.calibrant_intensity, 90.0, cal.rayleigh()),
        rel=1e-12)


def test_calibration_constant_off_ninety_uses_sin_theta():
    # Away from 90 deg the sin(theta) volume factor enters: k_c = R / (I sin theta).
    kc = sls.compute_calibration_constant(2.0e5, 30.0, 3.0e-5)
    assert kc == pytest.approx(3.0e-5 / (2.0e5 * math.sin(math.radians(30.0))), rel=1e-12)


def test_calibration_constant_rejects_nonpositive_net_intensity():
    with pytest.raises(ValueError):
        # net = 100 - 200 < 0
        sls.compute_calibration_constant(100.0, 90.0, 2.0e-5, dark_count_rate=200.0)


@pytest.mark.parametrize("angle", [0.0, 180.0, -10.0, 200.0])
def test_calibration_constant_rejects_bad_angle(angle):
    # The sin-theta scattering-volume factor degenerates at 0 and 180 deg, so
    # the calibrant angle must be strictly in (0, 180). 180 deg in particular is
    # a float trap: math.sin(math.radians(180)) ~ 1.2e-16 > 0, so the guard must
    # test the angle bounds, not `sin <= 0` (regression for that fix).
    with pytest.raises(ValueError):
        sls.compute_calibration_constant(1.0e5, angle, 2.0e-5)


# ===========================================================================
# 2. Excess Rayleigh ratio: calibration flag, K, ri-correction, angle guard
# ===========================================================================

def test_optical_constant_K_matches_physics():
    meas, cal = S.make_from_truth("PEG 1M")
    rr = S.rayleigh_results(meas, cal)[0]
    K_expected = C.optical_constant_K(S.N_SOLVENT, S.DN_DC, S.WAVELENGTH_NM)
    assert rr.optical_constant_K == pytest.approx(K_expected, rel=1e-12)


def test_calibrated_flag_and_kc_recorded():
    meas, cal = S.make_from_truth("PEG 1M")
    rr_cal = S.rayleigh_results(meas, cal, calibrated=True)[0]
    rr_unc = S.rayleigh_results(meas, cal, calibrated=False)[0]
    assert rr_cal.calibrated is True
    assert rr_cal.k_c_used == pytest.approx(cal.k_c(), rel=1e-12)
    assert rr_unc.calibrated is False
    assert rr_unc.k_c_used == 1.0            # arbitrary-scale fallback


def test_refractive_index_correction_applied():
    # water (1.33) vs toluene standard (1.496): f = (1.33/1.496)^2.
    meas, cal = S.make_from_truth("PEG 1M")
    sv = S.solvent_reference(meas)
    sample = next(m for m in meas if m.concentration_g_per_mL != 0)

    f = (S.N_SOLVENT / S.N_STANDARD) ** 2
    rr_with = sls.compute_excess_rayleigh_ratio(
        sample, sv, calibration_constant=cal.k_c(), standard_refractive_index=S.N_STANDARD)
    rr_without = sls.compute_excess_rayleigh_ratio(
        sample, sv, calibration_constant=cal.k_c(), standard_refractive_index=None)

    assert rr_with.ri_correction == pytest.approx(f, rel=1e-12)
    assert rr_without.ri_correction == 1.0
    # The dR array must actually be scaled by the correction factor.
    ratio = rr_with.excess_rayleigh_cm_inv / rr_without.excess_rayleigh_cm_inv
    assert np.allclose(ratio, f, rtol=1e-12)


def test_angle_mismatch_raises():
    meas, cal = S.make_from_truth("PEG 1M")
    sample = next(m for m in meas if m.concentration_g_per_mL != 0)
    from core.data_models import SLSMeasurement
    bad_solvent = SLSMeasurement(
        angles_deg=np.array([35.0, 45.0]), intensities=np.array([1.0, 2.0]),
        polymer_name="PEG", solvent_name="water", concentration_g_per_mL=0.0,
        temperature_K=S.TEMPERATURE_K, wavelength_nm=S.WAVELENGTH_NM,
        solvent_refractive_index=S.N_SOLVENT, dn_dc_mL_per_g=S.DN_DC)
    with pytest.raises(ValueError):
        sls.compute_excess_rayleigh_ratio(sample, bad_solvent, calibration_constant=cal.k_c())


# ===========================================================================
# 3. Zimm on clean synthetic PEG 1M (closed round trip -> tight)
# ===========================================================================

def test_zimm_peg_1m_round_trip():
    truth = S.SERIES["PEG 1M"]
    meas, cal = S.make_sls_set(truth["mw"], truth["rg"], truth["a2"])
    rr = S.rayleigh_results(meas, cal)
    z = sls.zimm_analysis(rr, method="zimm")

    assert z.mw_g_per_mol == pytest.approx(truth["mw"], rel=0.03)          # ~1e6
    assert z.rg_nm == pytest.approx(truth["rg"], abs=2.0)                   # ~55 nm
    assert z.a2_mol_mL_per_g2 == pytest.approx(truth["a2"], rel=0.05)       # ~9.5e-5
    # thermodynamic (fully extrapolated), calibrated -> everything reliable
    assert z.is_apparent is False
    assert z.calibrated is True and z.mw_reliable is True and z.a2_reliable is True
    assert z.n_concentrations == 5 and z.n_angles == len(S.DEFAULT_ANGLES)


# ===========================================================================
# 4. Berry on high-Mw PEG 3M
# ===========================================================================

def test_berry_peg_3m_high_mw():
    truth = S.SERIES["PEG 3M"]
    meas, cal = S.make_sls_set(truth["mw"], truth["rg"], truth["a2"])
    rr = S.rayleigh_results(meas, cal)
    b = sls.zimm_analysis(rr, method="berry")

    assert b.method == "berry"
    assert b.is_apparent is False and b.calibrated is True
    # Berry linearises sqrt(Kc/dR), which is NOT exact on this Zimm-form generator,
    # so it recovers the right order of magnitude but underestimates at large qRg
    # (empirically Mw ~ 2.4e6, Rg ~ 75 nm). Loose, honest bounds; see module docstring.
    assert b.mw_g_per_mol == pytest.approx(truth["mw"], rel=0.30)           # ~3e6 +/-30%
    assert b.rg_nm == pytest.approx(truth["rg"], rel=0.35)                  # ~105 nm +/-35%
    # documented bias direction: Berry < Zimm on this data
    z = sls.zimm_analysis(rr, method="zimm")
    assert b.mw_g_per_mol < z.mw_g_per_mol


# ===========================================================================
# 5. Full homologous series (parametrized): Zimm recovers each ground truth
# ===========================================================================

@pytest.mark.parametrize("name", list(S.SERIES.keys()))
def test_zimm_full_series_recovery(name):
    truth = S.SERIES[name]
    meas, cal = S.make_sls_set(truth["mw"], truth["rg"], truth["a2"])
    z = sls.zimm_analysis(S.rayleigh_results(meas, cal), method="zimm")
    assert z.mw_g_per_mol == pytest.approx(truth["mw"], rel=0.03)   # Mw within 3%
    assert z.rg_nm == pytest.approx(truth["rg"], rel=0.05)          # Rg within 5%
    assert z.a2_mol_mL_per_g2 == pytest.approx(truth["a2"], rel=0.10)  # A2 within 10%


# ===========================================================================
# 6. Debye single-concentration: apparent Mw/Rg, reliability tracks calibration
# ===========================================================================

def test_debye_single_concentration_apparent():
    truth = S.SERIES["PEG 1M"]
    meas, cal = S.make_sls_set(truth["mw"], truth["rg"], truth["a2"])
    rr_cal = S.rayleigh_results(meas, cal, calibrated=True)
    rr_unc = S.rayleigh_results(meas, cal, calibrated=False)

    d = sls.debye_analysis(rr_cal[0])          # lowest non-zero concentration
    assert d.is_apparent is True
    assert d.calibrated is True and d.mw_reliable is True
    assert math.isfinite(d.mw_apparent_g_per_mol) and d.mw_apparent_g_per_mol > 0
    # apparent Mw from a single c is 1/(1/Mw + 2 A2 c) < Mw; same order of magnitude.
    assert d.mw_apparent_g_per_mol == pytest.approx(truth["mw"], rel=0.15)
    # apparent Rg is slightly suppressed from the true Rg at finite c, but close.
    assert d.rg_apparent_nm == pytest.approx(truth["rg"], rel=0.15)

    # Uncalibrated: Mw is on an arbitrary scale (flagged), but Rg is IDENTICAL
    # (slope/intercept, the scale cancels) -> the CLAUDE.md invariant.
    d_unc = sls.debye_analysis(rr_unc[0])
    assert d_unc.calibrated is False and d_unc.mw_reliable is False
    assert d_unc.rg_apparent_nm == pytest.approx(d.rg_apparent_nm, rel=1e-9)


# ===========================================================================
# 7. Single-angle Mw; Guinier Rg = sqrt(-3*slope) and qRg validity
# ===========================================================================

def test_single_angle_mw():
    truth = S.SERIES["PEG 1M"]
    meas, cal = S.make_sls_set(truth["mw"], truth["rg"], truth["a2"])
    rr = S.rayleigh_results(meas, cal)[0]
    angle = float(S.DEFAULT_ANGLES[0])
    sa = sls.single_angle_mw(rr, angle)
    assert sa.is_apparent is True
    assert sa.angle_deg == pytest.approx(angle)
    assert math.isfinite(sa.mw_apparent_g_per_mol) and sa.mw_apparent_g_per_mol > 0
    # order-of-magnitude apparent Mw (contains P(q) and the 2 A2 c term).
    assert sa.mw_apparent_g_per_mol == pytest.approx(truth["mw"], rel=0.3)

    with pytest.raises(ValueError):
        sls.single_angle_mw(rr, 999.0)         # angle not measured


def test_guinier_small_rg_valid():
    truth = S.SERIES["PEG 100k"]               # small Rg -> in the Guinier regime
    meas, cal = S.make_sls_set(truth["mw"], truth["rg"], truth["a2"])
    rr = S.rayleigh_results(meas, cal)[0]
    g = sls.guinier_analysis(rr)
    # Rg = sqrt(-3*slope); for small qRg the Zimm form ~ the Guinier exponential.
    assert g.rg_nm == pytest.approx(math.sqrt(-3.0 * g.slope), rel=1e-12)
    assert g.rg_nm == pytest.approx(truth["rg"], rel=0.10)     # ~14 nm
    assert g.qrg_max <= 1.3 and g.guinier_valid is True
    assert g.is_apparent is True and g.mw_reliable is True


def test_guinier_large_rg_flagged_invalid():
    truth = S.SERIES["PEG 3M"]                 # large Rg -> qRg exceeds ~1.3
    meas, cal = S.make_sls_set(truth["mw"], truth["rg"], truth["a2"])
    rr = S.rayleigh_results(meas, cal)[0]
    g = sls.guinier_analysis(rr)
    assert g.qrg_max > 1.3
    assert g.guinier_valid is False


# ===========================================================================
# Curved form factor (Gap 2): Berry's advantage over Zimm on genuine P(q) data.
#
# The make_sls_set tests above use the exact first-order Zimm ordinate, where
# Zimm is exact and Berry only looks worse. make_curved_sls_set instead bakes in a
# real Debye-coil form factor P(q) = (2/x^2)(e^-x - 1 + x), x = (q Rg)^2 — the
# regime Berry's square-root linearisation was designed for. On this data Zimm's
# linear-in-q^2 fit is biased by the upward form-factor curvature, and Berry
# recovers Rg (and Mw) closer to truth. This documents the flip side of the
# exact-Zimm-data finding above.
# ===========================================================================

def test_berry_beats_zimm_on_curved_coil():
    mw_true, rg_true, a2 = 1.0e6, 60.0, 5.0e-5      # qRg_max ~ 1.8 over the default angles
    meas, cal = S.make_curved_sls_set(mw_true, rg_true, a2, model="debye")
    rr = S.rayleigh_results(meas, cal)
    z = sls.zimm_analysis(rr, method="zimm")
    b = sls.zimm_analysis(rr, method="berry")

    assert z.is_apparent is False and b.is_apparent is False
    assert z.calibrated is True and b.calibrated is True

    # Berry recovers Rg closer to truth than Zimm (the documented advantage).
    assert abs(b.rg_nm - rg_true) < abs(z.rg_nm - rg_true)
    assert b.rg_nm == pytest.approx(rg_true, rel=0.10)
    # Zimm's linear fit OVER-estimates Rg on an upward-curving coil form factor.
    assert z.rg_nm > rg_true

    # Berry's intercept-Mw is also closer to truth here than Zimm's.
    assert abs(b.mw_g_per_mol - mw_true) < abs(z.mw_g_per_mol - mw_true)
    assert b.mw_g_per_mol == pytest.approx(mw_true, rel=0.05)


def test_guinier_validity_flag_on_curved_data():
    mw_true, rg_true, a2 = 1.0e6, 60.0, 5.0e-5
    # Narrow low-angle window -> qRg small -> Guinier valid and Rg recovered.
    narrow, cal = S.make_curved_sls_set(mw_true, rg_true, a2, model="debye",
                                        angles=[20, 25, 30, 35, 40, 45])
    g_narrow = sls.guinier_analysis(S.rayleigh_results(narrow, cal)[0])
    assert g_narrow.qrg_max <= 1.3 and g_narrow.guinier_valid is True
    assert g_narrow.rg_nm == pytest.approx(rg_true, rel=0.10)

    # Wide window -> qRg exceeds the Guinier limit -> flagged invalid.
    wide, cal2 = S.make_curved_sls_set(mw_true, rg_true, a2, model="debye",
                                       angles=[30, 60, 90, 120, 150])
    g_wide = sls.guinier_analysis(S.rayleigh_results(wide, cal2)[0])
    assert g_wide.qrg_max > 1.3 and g_wide.guinier_valid is False


# ===========================================================================
# 8. Uncalibrated invariant: Mw & A2 flagged unreliable, Rg + 2*A2*Mw survive
# ===========================================================================

def test_uncalibrated_zimm_rg_survives_mw_flagged():
    truth = S.SERIES["PEG 1M"]
    meas, cal = S.make_sls_set(truth["mw"], truth["rg"], truth["a2"])
    rr_unc = S.rayleigh_results(meas, cal, calibrated=False)
    z = sls.zimm_analysis(rr_unc, method="zimm")

    # Mw and absolute A2 are unreliable on the arbitrary scale...
    assert z.calibrated is False
    assert z.mw_reliable is False and z.a2_reliable is False
    # ...but Rg SURVIVES: it comes from b/a (slope/intercept), scale cancels, so it
    # is finite, unflagged, and equal to the calibrated Rg / the truth.
    z_cal = sls.zimm_analysis(S.rayleigh_results(meas, cal, calibrated=True), method="zimm")
    assert math.isfinite(z.rg_nm)
    assert z.rg_nm == pytest.approx(truth["rg"], rel=0.05)
    assert z.rg_nm == pytest.approx(z_cal.rg_nm, rel=1e-9)


def test_calibration_free_product_invariant_to_calibration():
    truth = S.SERIES["PEG 1M"]
    meas, cal = S.make_sls_set(truth["mw"], truth["rg"], truth["a2"])
    angle = float(S.DEFAULT_ANGLES[0])         # lowest angle, closest to q -> 0
    cf_cal = sls.calibration_free_a2(
        S.rayleigh_results(meas, cal, calibrated=True), angle_deg=angle, mw_g_per_mol=truth["mw"])
    cf_unc = sls.calibration_free_a2(
        S.rayleigh_results(meas, cal, calibrated=False), angle_deg=angle, mw_g_per_mol=truth["mw"])

    # The 2*A2*Mw product uses only intensity RATIOS, so calibration cancels exactly.
    assert cf_cal.two_a2_mw == pytest.approx(cf_unc.two_a2_mw, rel=1e-9)
    assert math.isfinite(cf_cal.two_a2_mw) and cf_cal.two_a2_mw > 0
    # Order-of-magnitude of 2*A2*Mw (= 190 for PEG 1M); a small finite-angle bias
    # keeps it a little low, hence the 15% tolerance rather than exact.
    assert cf_cal.two_a2_mw == pytest.approx(2.0 * truth["a2"] * truth["mw"], rel=0.15)
    assert cf_cal.a2_mol_mL_per_g2 == pytest.approx(truth["a2"], rel=0.15)


# ===========================================================================
# 9. Guards: too-few concentrations / too-few angles raise
# ===========================================================================

def test_zimm_needs_two_concentrations():
    meas, cal = S.make_from_truth("PEG 1M")
    rr = S.rayleigh_results(meas, cal)
    with pytest.raises(ValueError):
        sls.zimm_analysis([rr[0]], method="zimm")     # single concentration


def test_zimm_needs_two_angles():
    # One angle per concentration -> not enough angular information.
    meas, cal = S.make_from_truth("PEG 1M", angles=[90.0])
    rr = S.rayleigh_results(meas, cal)
    with pytest.raises(ValueError):
        sls.zimm_analysis(rr, method="zimm")


def test_debye_needs_two_angles():
    meas, cal = S.make_from_truth("PEG 1M", angles=[90.0])
    rr = S.rayleigh_results(meas, cal)
    with pytest.raises(ValueError):
        sls.debye_analysis(rr[0])


def test_zimm_rejects_bad_method():
    meas, cal = S.make_from_truth("PEG 1M")
    rr = S.rayleigh_results(meas, cal)
    with pytest.raises(ValueError):
        sls.zimm_analysis(rr, method="guinier")


# ===========================================================================
# 10. REAL end-to-end: PS(900k)/toluene Zimm set -> bounded Mw and Rg
# ===========================================================================

# The Brookhaven file records an 'A' Dark Count Rate of 2150; the documented
# validation subtracts it, computing its own k_c from the c = 0
# toluene at 90 deg in R_VU geometry to reach Mw ~ 1.01e6.
_PS_DARK_COUNT = 2150.0


def _load_ps_900k():
    from parsers.brookhaven_sls import BrookhavenSLSParser
    path = require(BROOKHAVEN_DIR / "Zimm Plot - PS (900k) in Toluene Intensities.csv")
    previews = BrookhavenSLSParser().parse(str(path))
    for p in previews:
        p.polymer_name = "PS"
        p.temperature_K = 298.15               # file has no temperature field
    return [p.build() for p in previews]


@pytest.mark.realdata
def test_real_ps900k_zimm_bounded():
    meas = _load_ps_900k()
    assert len(meas) == 7                       # 6 concentrations + c = 0 reference

    solvent = next(m for m in meas if m.concentration_g_per_mL == 0)
    i90 = float(solvent.intensities[np.isclose(solvent.angles_deg, 90.0)][0])

    # Own k_c from the c = 0 toluene calibrant at 90 deg, VU geometry (BI-200SM).
    r_vu = C.rayleigh_ratio_toluene(532.0, 25.0, geometry="VU")
    k_c = sls.compute_calibration_constant(i90, 90.0, r_vu, dark_count_rate=_PS_DARK_COUNT)

    # PS is in toluene, so solvent == standard: (n_s/n_std)^2 = 1 exactly.
    rr = [
        sls.compute_excess_rayleigh_ratio(
            m, solvent, calibration_constant=k_c,
            standard_refractive_index=1.502, dark_count_rate=_PS_DARK_COUNT)
        for m in meas if m.concentration_g_per_mL != 0
    ]
    assert rr[0].ri_correction == pytest.approx(1.0, rel=1e-9)
    assert rr[0].calibrated is True

    z = sls.zimm_analysis(rr, method="zimm")
    assert z.is_apparent is False
    # Documented headline: Mw ~ 1.01e6 g/mol, Rg ~ 40.5 nm. Bounded ranges.
    assert 0.96e6 < z.mw_g_per_mol < 1.06e6
    assert 36.0 < z.rg_nm < 45.0


@pytest.mark.realdata
def test_real_ps900k_rg_survives_uncalibrated():
    # Rg must not depend on the absolute calibration: same value with k_c = None.
    meas = _load_ps_900k()
    solvent = next(m for m in meas if m.concentration_g_per_mL == 0)
    i90 = float(solvent.intensities[np.isclose(solvent.angles_deg, 90.0)][0])
    r_vu = C.rayleigh_ratio_toluene(532.0, 25.0, geometry="VU")
    k_c = sls.compute_calibration_constant(i90, 90.0, r_vu, dark_count_rate=_PS_DARK_COUNT)

    rr_cal = [sls.compute_excess_rayleigh_ratio(
        m, solvent, calibration_constant=k_c, standard_refractive_index=1.502,
        dark_count_rate=_PS_DARK_COUNT) for m in meas if m.concentration_g_per_mL != 0]
    rr_unc = [sls.compute_excess_rayleigh_ratio(
        m, solvent, calibration_constant=None, standard_refractive_index=1.502,
        dark_count_rate=_PS_DARK_COUNT) for m in meas if m.concentration_g_per_mL != 0]

    z_cal = sls.zimm_analysis(rr_cal, method="zimm")
    z_unc = sls.zimm_analysis(rr_unc, method="zimm")
    assert z_unc.mw_reliable is False
    assert z_unc.rg_nm == pytest.approx(z_cal.rg_nm, rel=1e-9)
    assert 36.0 < z_unc.rg_nm < 45.0
