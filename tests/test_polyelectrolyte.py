"""Adversarial real-data test: a polyelectrolyte that violates simple DLS theory.

`test-data/ALV/NaPSS 40gL in Water/` is a curated clean subset of the owner's
ALV-7004/USB single-angle correlograms for **sodium poly(styrene sulfonate),
40 g/L in water, no added salt** — a semidilute polyelectrolyte. Such systems
famously break the monodisperse-diffusive assumptions of ordinary DLS: a very
fast "polyelectrolyte mode" (coupled polyion/counterion diffusion) plus a large
slow mode, a strongly q-dependent apparent size, and a broad/bimodal decay.

There is NO clean ground-truth size here, so this fixture does not assert a
"correct" Rh. Instead it asserts that the engine's **diagnostics fire on real
anomalous data** — the honest, valuable thing to regression-test:

  * cumulant PDI > 0.3 (``pdi_valid`` False) at every angle (the decay is not a
    single narrow exponential; PDI even exceeds 1, the unphysical-variance signal
    that the cumulant model is being pushed past its domain),
  * Gamma vs q^2 is flagged non-diffusive (fails R^2, through-origin, and D-trend
    criteria), and
  * the multi-component fits (double-exponential, NNLS) resolve two well-separated
    modes rather than one.

If a future change silently "diffusive-ified" this data or stopped flagging the
high PDI, these assertions break — which is the point.
"""
from __future__ import annotations

import pytest

from analysis import dls as E

from fixtures.data_paths import ALV_NAPSS_DIR
from fixtures.synthetic_dls import load_alv_as

_NAPSS_T = 295.3   # shared nominal temperature (files jitter at the sub-mK level)


def _load_napss():
    files = sorted(ALV_NAPSS_DIR.glob("*.ASC"))
    ms = [load_alv_as(f, "NaPSS", "water", 0.04, temperature_K=_NAPSS_T)[0]
          for f in files]
    return sorted(ms, key=lambda m: m.angle_deg)


@pytest.mark.realdata
def test_napss_gamma_q2_flagged_non_diffusive():
    ms = _load_napss()
    assert len(ms) >= 7, f"expected the curated NaPSS angular set, got {len(ms)}"
    gq = E.analyze_gamma_q2(ms, cumulant_method="nonlinear")
    # A real diffusive sample would pass all three; the polyelectrolyte fails them.
    assert gq.is_diffusive is False
    assert gq.r_squared < 0.98
    assert abs(gq.intercept_relative) > 0.1


@pytest.mark.realdata
def test_napss_cumulant_pdi_flagged_unreliable_at_all_angles():
    ms = _load_napss()
    for m in ms:
        r = E.fit_cumulants(m, method="nonlinear", order=2)
        assert r.pdi > 0.3, f"angle {m.angle_deg}: PDI={r.pdi:.2f} unexpectedly low"
        assert r.pdi_valid is False


@pytest.mark.realdata
def test_napss_multimodal_fast_and_slow_modes():
    ms = _load_napss()
    m90 = min(ms, key=lambda m: abs(m.angle_deg - 90.0))

    # Double-exponential: two well-separated modes (fast polyelectrolyte mode +
    # large slow mode). The absolute slow size is beyond DLS validity (it fits
    # long-lag drift/aggregates) — assert separation, not its value.
    d = E.fit_double_exponential(m90)
    fast, slow = sorted((d.mode1, d.mode2), key=lambda mode: mode.rh_nm)
    assert slow.rh_nm > 10.0 * fast.rh_nm, (
        f"modes not separated: {fast.rh_nm:.1f} vs {slow.rh_nm:.1f} nm")

    # NNLS resolves at least two peaks spanning small + large Rh.
    peaks = sorted(E.find_distribution_peaks(E.fit_nnls(m90)), key=lambda p: p.rh_nm)
    rhs = [p.rh_nm for p in peaks]
    assert len(peaks) >= 2, f"expected multimodal, got peaks at {rhs}"
    assert rhs[0] < 10.0 and rhs[-1] > 100.0, f"peaks at {rhs}"
