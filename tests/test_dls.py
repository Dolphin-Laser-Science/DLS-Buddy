"""Regression tests for the DLS analysis package (``analysis/dls/``).

Ported from the hand-rolled validators:
  * the retired DLS-module validator, sections A-E (every public DLS function
    recovers known ground truth on the program's own forward model + the real
    ``test-data/Synthetic Clean`` ALV set), and
  * the numeric groups of the retired nonlinear-cumulant validator: A
    (default == explicit linear), B (nonlinear correctness + floating baseline),
    C (robustness), E (uncertainty: single fit exposes no SE + replicate SEM),
    F (pure-noise failure handling).

The controller / switch-guard / persistence groups (NLC G/H/I) live in the
workspace-tier tests, not here.

Tolerances and bounded ranges are copied verbatim from the source validators —
they were tuned deliberately against this forward model and the committed data.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from analysis import dls as E
from analysis.uncertainty import replicate_mean_se
from analysis.utilities import SyntheticPopulation, generate_synthetic_correlogram
from fixtures.data_paths import SYNTH_CLEAN_DIR, BROOKHAVEN_DIR, ALV_PS_TOLUENE_DIR
from fixtures.synthetic_dls import (
    bimodal,
    gamma_for_rh,
    load_alv,
    load_alv_as,
    make_measurement,
    monomodal,
    nearest_angle,
    q_m,
    rh_from_D,
    tau_grid,
)

KB = 1.380649e-23


# ============================================================ [A] exponentials

@pytest.mark.parametrize("rh", [10.0, 30.0, 80.0])
def test_single_exponential_recovers_rh_and_gamma(rh):
    r = E.fit_single_exponential(monomodal(rh))
    assert abs(r.mode.rh_nm - rh) < 0.6, f"got {r.mode.rh_nm:.3f}"
    g_true = gamma_for_rh(rh)
    assert abs(r.mode.gamma_s_inv - g_true) / g_true < 2e-3, (
        f"got {r.mode.gamma_s_inv:.4g} vs {g_true:.4g}")


def test_single_exponential_amplitude_and_success():
    r = E.fit_single_exponential(monomodal(30.0))
    assert r.mode.amplitude_fraction == 1.0
    assert r.success


def test_double_exponential_resolves_bimodal():
    d = E.fit_double_exponential(bimodal(20.0, 200.0, f_small=0.5))
    fast, slow = d.mode1, d.mode2   # mode1 is the faster (larger gamma) by contract
    assert fast.gamma_s_inv > slow.gamma_s_inv, (
        f"{fast.gamma_s_inv:.3g} vs {slow.gamma_s_inv:.3g}")
    assert abs(fast.rh_nm - 20.0) < 3.0, f"got {fast.rh_nm:.2f}"
    assert abs(slow.rh_nm - 200.0) < 40.0, f"got {slow.rh_nm:.2f}"
    assert abs(fast.amplitude_fraction + slow.amplitude_fraction - 1.0) < 1e-6


def test_kww_monodisperse_stretch_near_one():
    k = E.fit_kww(monomodal(30.0))
    assert 0.9 <= k.stretch <= 1.0, f"got {k.stretch:.3f}"
    assert abs(k.rh_from_tau_c_nm - 30.0) < 2.0, f"got {k.rh_from_tau_c_nm:.2f}"


def test_kww_polydisperse_stretch_below_one_and_below_monodisperse():
    # Polydisperse single mode via the richer generator (spread_cv = log-normal
    # width). A stretch below 1 AND below the monodisperse fit is the meaningful,
    # non-arbitrary signature of a spread of relaxation times.
    mono_k = E.fit_kww(monomodal(30.0))
    g = generate_synthetic_correlogram(
        [SyntheticPopulation(rh_nm=40.0, weight=1.0, spread_cv=0.6)],
        angle_deg=90.0, wavelength_nm=633.0, solvent_refractive_index=1.33,
        temperature_K=298.15, viscosity_Pa_s=8.9e-4, beta=0.9, noise_level=0.0,
        n_points=200, seed=7)
    m = make_measurement(g.signal, g.delay_times_s)
    kp = E.fit_kww(m)
    assert 0.0 < kp.stretch < 1.0, f"got {kp.stretch:.3f}"
    assert kp.stretch < mono_k.stretch, f"poly={kp.stretch:.3f} mono={mono_k.stretch:.3f}"
    assert math.isfinite(kp.mean_tau_s) and kp.mean_tau_s > 0


# ====================================================== [B] angular front-end

def test_gamma_q2_diffusive_recovers_rh():
    angles = [40.0, 60.0, 80.0, 100.0, 120.0]
    rh_true = 50.0
    ms = [monomodal(rh_true, angle_deg=a) for a in angles]
    gq = E.analyze_gamma_q2(ms)
    assert abs(gq.rh_nm - rh_true) < 1.0, f"got {gq.rh_nm:.3f}"
    assert gq.is_diffusive
    assert gq.r_squared > 0.98, f"R2={gq.r_squared:.5f}"
    assert abs(gq.intercept_relative) < 0.1, f"int_rel={gq.intercept_relative:.4f}"


def test_gamma_q2_non_diffusive_flagged():
    angles = [40.0, 60.0, 80.0, 100.0, 120.0]
    jumble = [50.0, 30.0, 65.0, 25.0, 70.0]
    msn = [monomodal(rh, angle_deg=a) for rh, a in zip(jumble, angles, strict=True)]
    gqn = E.analyze_gamma_q2(msn)
    assert not gqn.is_diffusive, (
        f"is_diffusive={gqn.is_diffusive}, R2={gqn.r_squared:.4f}")


def test_concentration_extrapolation_recovers_d0_and_kd():
    # Impose D(c) = D0 (1 + kD c).
    D0 = KB * 298.15 / (6 * np.pi * 8.9e-4 * 50.0e-9)    # D for Rh0 = 50 nm
    kD = 30.0                                            # mL/g (matches test-data doc)
    concs = [0.2e-3, 0.6e-3, 1.0e-3, 1.4e-3]            # g/mL
    msc = []
    for c in concs:
        Dc = D0 * (1 + kD * c)
        msc.append(monomodal(rh_from_D(Dc), angle_deg=90.0, conc=c))
    ex = E.extrapolate_diffusion_vs_concentration(msc)
    assert abs(ex.d0_m2_s - D0) / D0 < 0.02, f"got {ex.d0_m2_s:.4g} vs {D0:.4g}"
    assert abs(ex.rh0_nm - 50.0) < 1.0, f"got {ex.rh0_nm:.3f}"
    assert abs(ex.kd_mL_per_g - kD) < 4.0, f"got {ex.kd_mL_per_g:.2f}"
    assert ex.r_squared > 0.99, f"R2={ex.r_squared:.5f}"


def test_rh_gamma_converter_round_trip():
    q = q_m(90.0)
    g = E.rh_nm_to_gamma(50.0, q, 298.15, 8.9e-4)
    rh = E.gamma_to_rh_nm(g, q, 298.15, 8.9e-4)
    assert abs(rh - 50.0) < 1e-9, f"got {rh:.9f}"


# ====================================================== [C] distribution helpers

def test_nnls_bimodal_resolves_both_populations():
    # NNLS is unregularised -> "spiky": on a 10x-separated bimodal it resolves both
    # true populations (20 & 200 nm) but may add a spurious middle spike, so the
    # correct assertion is "resolves both modes", NOT an exact peak count.
    peaks = E.find_distribution_peaks(E.fit_nnls(bimodal(20.0, 200.0, f_small=0.5)))
    rhs = sorted(p.rh_nm for p in peaks)
    assert (len(peaks) >= 2 and any(12 < r < 35 for r in rhs)
            and any(130 < r < 320 for r in rhs)), f"peaks at {[round(r, 1) for r in rhs]}"
    assert sum(p.is_dominant for p in peaks) == 1
    assert abs(sum(p.weight_fraction for p in peaks) - 1.0) < 0.05, (
        f"sum={sum(p.weight_fraction for p in peaks):.3f}")


def test_contin_bimodal_resolves_large_mode():
    cpeaks = E.find_distribution_peaks(
        E.fit_contin(bimodal(20.0, 200.0, f_small=0.5)).distribution)
    crhs = sorted(p.rh_nm for p in cpeaks)
    assert len(cpeaks) >= 2, f"got {len(cpeaks)}"
    assert any(130 < r < 320 for r in crhs), f"peaks at {[round(r, 1) for r in crhs]}"


def _noisy_bimodal(rh_small, rh_large, f_small=0.5, noise=0.003, seed=1):
    """A two-population correlogram with realistic Gaussian noise on g2-1. The F-test
    is a STATISTICAL criterion: it needs a meaningful noise floor (on a noiseless fit
    the least-regularised solution is trivially 'best', so no smoothing is chosen).
    Real DLS always has noise; this mirrors that."""
    tau = tau_grid()
    g1 = (f_small * np.exp(-gamma_for_rh(rh_small) * tau)
          + (1 - f_small) * np.exp(-gamma_for_rh(rh_large) * tau))
    g2m1 = 0.9 * g1 ** 2 + np.random.RandomState(seed).normal(0.0, noise, tau.size)
    return make_measurement(g2m1, tau)


def test_contin_ftest_selection_picks_sensible_alpha():
    """CONTIN with the Provencher F-test picks a sensible, reproducible alpha that
    still resolves the bimodal, records its provenance, and moves monotonically with
    the probability-to-reject level (higher -> smoother/larger alpha)."""
    m = _noisy_bimodal(30.0, 200.0)
    r = E.fit_contin(m, alpha_method="ftest", ftest_prob_reject=0.5)
    # provenance recorded on the result
    assert r.alpha_selection_method == "ftest"
    assert r.ftest_prob_reject == 0.5
    assert r.lcurve.dof_eff is not None and r.lcurve.ftest_fc is not None
    # the F-test alpha comes from the sweep and is interior (not pinned to an end)
    assert r.lcurve.alphas[0] < r.lcurve.optimal_alpha < r.lcurve.alphas[-1]
    # it still resolves the large mode
    peaks = E.find_distribution_peaks(r.distribution)
    assert any(120 < p.rh_nm < 340 for p in peaks), (
        f"peaks at {[round(p.rh_nm, 1) for p in peaks]}")
    # direction: higher probability-to-reject -> smoother (>=) alpha, never rougher
    alphas = [E.fit_contin(m, alpha_method="ftest", ftest_prob_reject=p).lcurve.optimal_alpha
              for p in (0.1, 0.5, 0.9)]
    assert alphas[0] <= alphas[1] <= alphas[2], f"non-monotonic: {alphas}"
    assert alphas[2] > alphas[0], f"level had no effect: {alphas}"


def test_ftest_uses_fixed_reference_dof_not_per_alpha():
    """Provencher's F-statistic (Eqs. 3.23-3.24) uses NDF at the reference alpha_0
    (least-squares end), held FIXED across the sweep -- NOT the per-alpha DOF. On a
    kernel whose effective DOF falls steeply with alpha the two rules pick different
    solutions; `_ftest_corner` must follow the fixed-NDF_0 rule (else it selects too
    rough an alpha, as the 0.13.0 bug did)."""
    from scipy import special
    from analysis.dls.distributions import _ftest_corner

    ny = 200
    v = np.linspace(1.0, 1.25, 20)             # residual rises GENTLY (fc crosses 0.5 mid-sweep)
    dof = np.linspace(27.0, 5.0, 20)           # effective DOF falls steeply with alpha
    frac = np.clip((v - v[0]) / v[0], 0.0, None)

    # correct (Provencher): NDF fixed at the reference alpha_0 = argmin(V) = index 0
    ndf0 = dof[0]
    fc_fixed = special.fdtr(ndf0, ny - ndf0, frac * (ny - ndf0) / ndf0)
    idx_fixed = int(np.argmin(np.abs(fc_fixed - 0.5)))
    # buggy (per-alpha) rule, for contrast
    fc_per = special.fdtr(dof, ny - dof, frac * (ny - dof) / dof)
    idx_per = int(np.argmin(np.abs(fc_per - 0.5)))

    idx, fc = _ftest_corner(v, dof, n_data=ny, prob_reject=0.5)
    assert idx == idx_fixed, f"selector must use fixed NDF_0 (got {idx}, expected {idx_fixed})"
    assert idx != idx_per, "test design: the two rules should differ here"
    assert idx_fixed > idx_per, "fixed-NDF_0 must pick a smoother (larger-alpha) solution"
    assert np.allclose(fc, fc_fixed)


def test_contin_lcurve_default_unchanged_by_ftest_option():
    """The default (untouched) CONTIN path is bit-identical to before the F-test option
    existed: same chosen alpha and weights whether or not alpha_method is passed."""
    m = _noisy_bimodal(30.0, 200.0, seed=7)
    a = E.fit_contin(m)
    b = E.fit_contin(m, alpha_method="lcurve")
    assert a.alpha_selection_method == "lcurve"
    assert a.lcurve.optimal_alpha == b.lcurve.optimal_alpha
    assert np.array_equal(a.distribution.weights, b.distribution.weights)


def test_nnls_monodisperse_single_peak():
    mono_peaks = E.find_distribution_peaks(E.fit_nnls(monomodal(30.0)))
    assert len(mono_peaks) == 1, f"got {len(mono_peaks)}"


def test_distribution_axis_sorted_labelled_and_guarded():
    dist = E.fit_nnls(monomodal(30.0))
    xr, wr, lr = E.distribution_axis(dist, "rh")
    xg, wg, lg = E.distribution_axis(dist, "gamma")
    assert bool(np.all(np.diff(xr) > 0)) and "radius" in lr.lower()
    assert bool(np.all(np.diff(xg) > 0)) and "rate" in lg.lower()
    assert wr.size == dist.weights.size
    with pytest.raises(ValueError):
        E.distribution_axis(dist, "nope")


# ====================================================== [D] cross-method agreement

def test_cross_method_agreement_on_clean_correlogram():
    m = monomodal(37.0)
    est = {
        "cumulant-linear": E.fit_cumulants(m, method="linear").rh_nm,
        "cumulant-nonlinear": E.fit_cumulants(m, method="nonlinear").rh_nm,
        "single-exp": E.fit_single_exponential(m).mode.rh_nm,
        "nnls-peak": E.fit_nnls(m).peak_rh_nm,
        "contin-peak": E.fit_contin(m).distribution.peak_rh_nm,
        "lognormal-peak": E.fit_lognormal(m).peak_rh_nm,
    }
    for name, rh in est.items():
        assert abs(rh - 37.0) / 37.0 < 0.12, f"{name} got {rh:.2f}"
    spread = (max(est.values()) - min(est.values())) / 37.0
    assert spread < 0.15, (
        f"spread={spread*100:.1f}% over {{k: round(v,1) for k,v in est.items()}}")


# ====================================================== [E] real Synthetic-Clean

@pytest.mark.realdata
def test_real_peg1m_single_angle_and_multi_angle():
    peg1m = load_alv(SYNTH_CLEAN_DIR / "DLS - PEG 1M - 0.6 mg per mL.ASC")
    assert peg1m is not None, "PEG 1M file missing"
    m = nearest_angle(peg1m, 90.0)
    rc = E.fit_cumulants(m, method="nonlinear").rh_nm
    assert 32 < rc < 42, f"got {rc:.2f} @ {m.angle_deg:.0f}deg"
    pk = E.fit_contin(m).distribution.peak_rh_nm
    assert 28 < pk < 48, f"got {pk:.2f}"
    if len(peg1m) >= 2:
        gq = E.analyze_gamma_q2(peg1m, cumulant_method="nonlinear")
        assert 32 < gq.rh_nm < 42, f"got {gq.rh_nm:.2f}"
        assert gq.is_diffusive, f"R2={gq.r_squared:.4f}"


@pytest.mark.realdata
def test_real_peg300k_single_angle():
    peg300 = load_alv(SYNTH_CLEAN_DIR / "DLS - PEG 300k - 0.6 mg per mL.ASC")
    assert peg300 is not None, "PEG 300k file missing"
    m = nearest_angle(peg300, 90.0)
    rc = E.fit_cumulants(m, method="nonlinear").rh_nm
    assert 15 < rc < 21, f"got {rc:.2f}"


@pytest.mark.realdata
def test_real_peg100k_single_angle():
    # Added coverage the old validator lacked: PEG 100k ~ 9.3 nm (bounded 8-11 nm).
    peg100 = load_alv(SYNTH_CLEAN_DIR / "DLS - PEG 100k - 0.6 mg per mL.ASC")
    assert peg100 is not None, "PEG 100k file missing"
    m = nearest_angle(peg100, 90.0)
    rc = E.fit_cumulants(m, method="nonlinear").rh_nm
    assert 8 < rc < 11, f"got {rc:.2f}"


@pytest.mark.realdata
def test_real_peg3m_single_angle():
    # Added coverage the old validator lacked: PEG 3M ~ 68.6 nm (bounded 63-77 nm).
    peg3m = load_alv(SYNTH_CLEAN_DIR / "DLS - PEG 3M - 0.6 mg per mL.ASC")
    assert peg3m is not None, "PEG 3M file missing"
    m = nearest_angle(peg3m, 90.0)
    rc = E.fit_cumulants(m, method="nonlinear").rh_nm
    assert 63 < rc < 77, f"got {rc:.2f}"


@pytest.mark.realdata
def test_real_bimodal_nnls_resolves_two_populations():
    bim = load_alv(SYNTH_CLEAN_DIR / "DLS - Bimodal 20nm + 200nm.ASC")
    assert bim is not None, "bimodal file missing"
    m = nearest_angle(bim, 90.0)
    peaks = E.find_distribution_peaks(E.fit_nnls(m))
    rhs = sorted(p.rh_nm for p in peaks)
    has_small = any(10 < r < 35 for r in rhs)
    has_large = any(120 < r < 320 for r in rhs)
    assert len(peaks) >= 2 and has_small and has_large, (
        f"peaks at {[round(r, 1) for r in rhs]}")


@pytest.mark.realdata
def test_real_brookhaven_pvp_physical_bound():
    from core.data_models import DLSMeasurement
    from parsers.brookhaven_dls import BrookhavenDLSParser

    pvp_path = BROOKHAVEN_DIR / "Correlation Function - PVP (40k) in Water.csv"
    if not pvp_path.exists():
        pytest.skip("Brookhaven PVP file absent")
    previews = BrookhavenDLSParser().parse(str(pvp_path))
    pv = previews[0]
    m = DLSMeasurement(
        delay_times_s=pv.delay_times_s, correlogram=pv.correlogram,
        polymer_name="PVP", solvent_name="water", concentration_g_per_mL=None,
        temperature_K=298.15, angle_deg=90.0, wavelength_nm=632.8,
        solvent_refractive_index=1.33, viscosity_Pa_s=8.9e-4)
    rc = E.fit_cumulants(m, method="nonlinear").rh_nm
    assert math.isfinite(rc) and 1.0 < rc < 100.0, f"got {rc:.2f}"


# ============================ [NLC-A] non-breaking: default == explicit linear

@pytest.mark.parametrize("rh", [10.0, 30.0, 80.0])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_default_cumulant_method_is_linear_and_identical(rh, order):
    m = monomodal(rh)
    a = E.fit_cumulants(m, order=order)             # default
    b = E.fit_cumulants(m, order=order, method="linear")
    assert a.method == "linear"
    assert abs(a.gamma_s_inv - b.gamma_s_inv) == 0.0
    assert abs(a.rh_nm - b.rh_nm) == 0.0


# ============================ [NLC-B] nonlinear correctness + floating baseline

@pytest.mark.parametrize("rh", [10.0, 30.0, 80.0])
def test_nonlinear_cumulant_recovers_rh(rh):
    r = E.fit_cumulants(monomodal(rh), method="nonlinear")
    assert abs(r.rh_nm - rh) < 0.6 and r.success and r.method == "nonlinear", (
        f"Rh={r.rh_nm:.2f}")


@pytest.mark.parametrize("order", [1, 2, 3])
def test_nonlinear_cumulant_all_orders(order):
    r = E.fit_cumulants(monomodal(30.0), order=order, method="nonlinear")
    assert abs(r.rh_nm - 30.0) < 1.5, f"Rh={r.rh_nm:.2f}"


def test_nonlinear_recovers_floating_baseline_and_beats_linear():
    moff = monomodal(30.0, baseline=0.05)
    nl = E.fit_cumulants(moff, method="nonlinear")
    lin = E.fit_cumulants(moff, method="linear")
    assert abs(nl.baseline - 0.05) < 0.01, f"B={nl.baseline:.4f}"
    assert abs(nl.rh_nm - 30.0) < abs(lin.rh_nm - 30.0), (
        f"nl={nl.rh_nm:.2f} lin={lin.rh_nm:.2f}")


# ============================ [NLC-C] robustness vs linear

def test_nonlinear_matches_linear_on_clean_data():
    mclean = monomodal(30.0)
    nl = E.fit_cumulants(mclean, method="nonlinear").rh_nm
    lin = E.fit_cumulants(mclean, method="linear").rh_nm
    assert abs(nl - lin) < 0.5, f"nl={nl:.2f} lin={lin:.2f}"


def test_nonlinear_survives_afterpulse_spike():
    m = monomodal(30.0)
    g = m.correlogram.copy()
    g[0] = 0.9 * 2.0
    r = E.fit_cumulants(make_measurement(g, m.delay_times_s), method="nonlinear")
    assert r.rh_nm > 1.0, f"Rh={r.rh_nm:.3f} ok={r.success}"


# ============================ [NLC-E] uncertainty (invariant 8)

def test_single_nonlinear_fit_exposes_no_se_field():
    r = E.fit_cumulants(monomodal(30.0), method="nonlinear")
    has_se = any(hasattr(r, a) for a in ("rh_se", "se", "gamma_se"))
    assert not has_se


def test_nonlinear_replicate_averaging_yields_mean_and_sem():
    reps = [E.fit_cumulants(monomodal(30.0, noise=0.01, seed=s),
                            method="nonlinear").rh_nm for s in range(8)]
    st = replicate_mean_se(reps)
    assert st.sem is not None and st.sem > 0, f"mean={st.mean:.2f} sem={st.sem}"


# ============================ [NLC-F] failure handling

def test_pure_noise_nonlinear_no_exception_finite_flagged():
    rng = np.random.RandomState(1)
    g = rng.normal(0, 0.3, 200)            # pure noise, no decay
    r = E.fit_cumulants(make_measurement(g, tau_grid()), method="nonlinear")
    assert np.isfinite(r.rh_nm) and r.method == "nonlinear", (
        f"Rh={r.rh_nm} ok={r.success}")


# ============================ [PS] Real ALV-7004 data: PS 290k in toluene
#
# A curated clean subset (T~293.2 K, eta~0.59 cp) of the owner's ALV-7004/USB
# single-angle correlograms for polystyrene (nominal 290 kDa) in toluene. Used as
# a real-data hydrodynamic-radius anchor. Rh ~ 12 nm here is physically consistent
# with the literature Rg ~ 22 nm for PS in toluene via the good-solvent coil ratio
# rho = Rg/Rh ~ 1.8. Bounds are set from fitting the committed data, not certified
# values. SLS/Mw is deliberately NOT tested on this set (its absolute calibration
# is unreliable); only Rh (self-contained) is.
_PS_T = 293.23   # shared nominal temperature (files jitter at the sub-mK level)


def _ps_files(pattern):
    return sorted(ALV_PS_TOLUENE_DIR.glob(pattern))


@pytest.mark.realdata
def test_ps_toluene_gamma_q2_rh():
    files = _ps_files("*1.5 mg per mL*(avg)*.ASC")
    assert len(files) >= 7, f"expected the curated 1.5 mg/mL angular set, got {len(files)}"
    ms = [load_alv_as(f, "PS", "toluene", 0.0015, temperature_K=_PS_T)[0] for f in files]
    gq = E.analyze_gamma_q2(ms, cumulant_method="nonlinear")
    assert gq.is_diffusive, f"R2={gq.r_squared:.4f} int_rel={gq.intercept_relative:.4f}"
    assert gq.r_squared > 0.99
    assert abs(gq.intercept_relative) < 0.05
    assert 10.5 < gq.rh_nm < 14.0, f"Rh={gq.rh_nm:.2f} nm"


@pytest.mark.realdata
def test_ps_toluene_replicate_averaging():
    # Three true replicates at 90 deg, built at a shared nominal temperature so the
    # strict same-conditions check in average_replicate_correlograms accepts them.
    reps = [load_alv_as(f, "PS", "toluene", 0.0015, temperature_K=_PS_T)[0]
            for f in _ps_files("*090 deg (rep*.ASC")]
    assert len(reps) == 3
    avg = E.average_replicate_correlograms(reps)
    assert avg.mean_g2m1.size == reps[0].correlogram.size
    per_rep = [E.fit_cumulants(m, method="nonlinear").rh_nm for m in reps]
    st = replicate_mean_se(per_rep)
    assert st.sem is not None and st.sem > 0
    assert 10.5 < st.mean < 14.0, f"mean Rh={st.mean:.2f}"


@pytest.mark.realdata
def test_ps_toluene_rh_concentration_independent_in_dilute_regime():
    # 0.5 / 1.5 / 2 mg/mL at 90 deg. Over this dilute range kD*c is negligible, so
    # the apparent Rh is concentration-independent within scatter (the data does
    # NOT resolve a kD sign, so none is asserted).
    rhs = []
    for c, tok in [(0.0005, "0.5 mg per mL"), (0.0015, "1.5 mg per mL"), (0.002, "2 mg per mL")]:
        f = _ps_files(f"*{tok}*090 deg (avg)*.ASC")[0]
        m = load_alv_as(f, "PS", "toluene", c, temperature_K=_PS_T)[0]
        rhs.append(E.fit_cumulants(m, method="nonlinear").rh_nm)
    assert all(10.5 < r < 14.0 for r in rhs), f"Rh vs c = {[round(r, 2) for r in rhs]}"
