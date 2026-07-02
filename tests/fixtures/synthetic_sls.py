"""Forward-model SLS builders for the regression suite (test-only).

Wraps :func:`analysis.synthetic_dataset.build_sls_set` — the program's OWN
first-order Zimm forward model — into ready-to-analyse
:class:`core.data_models.SLSMeasurement` lists, together with the
:class:`~analysis.synthetic_dataset.CalibrationSpec` whose ``k_c`` reproduces the
absolute scale used to generate the intensities.

Because the intensities are generated with the exact Zimm equation

    Kc/ΔR = (1/Mw)(1 + q² Rg²/3) + 2 A2 c

that ``analysis.sls.zimm_analysis`` inverts (a closed round trip), a clean
(noise-free) Zimm recovery reproduces the input Mw/Rg/A2 to numerical precision.
Berry uses a different linearisation (sqrt of the ordinate) and is therefore
*not* exact on this Zimm-form data — that bias is asserted loosely and noted in
the tests, not masked.

Ground-truth system: PEG in water, 532 nm, T = 298.15 K, n_solvent = 1.33,
dn/dc = 0.135, calibrated against a toluene VU standard (n = 1.496).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple


import math

from analysis import synthetic_dataset as sd
from core.data_models import SLSMeasurement
from physics import constants as phys

# Angle / concentration grids and the homologous Mw series (Mw, Rg, Rh, A2).
DEFAULT_ANGLES: List[float] = [float(a) for a in sd.DEFAULT_SLS_ANGLES]
DEFAULT_CONCS_G: List[float] = [c * 1e-3 for c in sd.DEFAULT_SLS_CONCS_MG]
SERIES = sd.DEFAULT_SERIES

# Canonical optical parameters of the PEG-in-water test system.
WAVELENGTH_NM = sd.DEFAULT_WAVELENGTH_NM      # 532
TEMPERATURE_K = sd.DEFAULT_TEMPERATURE_K      # 298.15
N_SOLVENT = sd.DEFAULT_N_SOLVENT              # 1.33
DN_DC = sd.DEFAULT_DN_DC                       # 0.135
N_STANDARD = sd.DEFAULT_N_STANDARD            # 1.496 (toluene at 532 nm)


def make_sls_set(
    mw: float,
    rg_nm: float,
    a2_mol_mL_per_g2: float,
    *,
    angles: Optional[Sequence[float]] = None,
    concs_g: Optional[Sequence[float]] = None,
    wavelength_nm: float = WAVELENGTH_NM,
    temperature_K: float = TEMPERATURE_K,
    n_solvent: float = N_SOLVENT,
    dn_dc: float = DN_DC,
    n_standard: float = N_STANDARD,
    geometry: str = "VU",
    calibrant_intensity: float = sd.DEFAULT_CALIBRANT_INTENSITY,   # 1e5
    solvent_intensity_90: float = sd.DEFAULT_SOLVENT_INTENSITY_90,  # 6000
    noise: float = 0.0,
    seed: int = 1,
    polymer: str = "PEG",
    solvent: str = "water",
) -> Tuple[List[SLSMeasurement], "sd.CalibrationSpec"]:
    """Build a list of :class:`SLSMeasurement` (c = 0 first) + its CalibrationSpec.

    The returned measurements span ``angles`` at every concentration in
    ``concs_g`` (the c = 0 solvent reference is always present as the first
    element). ``cal.k_c()`` is the calibration constant that recovers the
    absolute scale; pass it to :func:`analysis.sls.compute_excess_rayleigh_ratio`
    with ``standard_refractive_index=cal.n_standard``.
    """
    angles = list(DEFAULT_ANGLES if angles is None else [float(a) for a in angles])
    concs_g = list(DEFAULT_CONCS_G if concs_g is None else [float(c) for c in concs_g])
    cal = sd.CalibrationSpec(
        wavelength_nm=wavelength_nm, temperature_C=25.0, geometry=geometry,
        n_standard=n_standard, calibrant_intensity=calibrant_intensity)
    sset = sd.build_sls_set(
        mw=mw, rg_nm=rg_nm, a2_mol_mL_per_g2=a2_mol_mL_per_g2,
        angles_deg=angles, concentrations_g_per_mL=concs_g,
        wavelength_nm=wavelength_nm, temperature_K=temperature_K,
        n_solvent=n_solvent, dn_dc=dn_dc, cal=cal,
        solvent_intensity_90=solvent_intensity_90, noise_level=noise, seed=seed,
        polymer_name=polymer, solvent_name=solvent)
    meas = [
        SLSMeasurement(
            angles_deg=sset.angles_deg.copy(),
            intensities=sset.intensities[c].copy(),
            polymer_name=polymer, solvent_name=solvent,
            concentration_g_per_mL=c, temperature_K=temperature_K,
            wavelength_nm=wavelength_nm, solvent_refractive_index=n_solvent,
            dn_dc_mL_per_g=dn_dc)
        for c in sset.concentrations_g_per_mL
    ]
    return meas, cal


def make_from_truth(name: str, **kw) -> Tuple[List[SLSMeasurement], "sd.CalibrationSpec"]:
    """``make_sls_set`` for a named ground-truth series entry (SERIES[name]).

    Maps the series dict keys (``mw``/``rg``/``a2``; ``rh`` is DLS-only, ignored)
    onto the builder's arguments. Extra keyword arguments (e.g. ``angles``) pass
    straight through.
    """
    p = SERIES[name]
    return make_sls_set(p["mw"], p["rg"], p["a2"], **kw)


# ===========================================================================
# Curved-form-factor SLS (Gap 2): genuine P(q), not the truncated 1 + q^2 Rg^2/3
# ===========================================================================
#
# make_sls_set above generates the EXACT first-order Zimm ordinate, which Zimm
# inverts perfectly and Berry (the sqrt linearisation) does not. That is only half
# the story: Berry's square-root form exists precisely because REAL high-Mw
# particles have a form factor P(q) that curves in q^2, and the sqrt straightens a
# coil's P(q)^-1 to higher order than the linear Zimm plot. This builder emits data
# with a genuine closed-form P(q) so the suite can show the flip side — Berry
# recovering Rg CLOSER to truth than Zimm at large qRg.
#
# Debye coil:  P(x) = (2/x^2)(e^-x - 1 + x),  x = (q Rg)^2   (independent, closed form)
# Sphere:      P(u) = [3 (sin u - u cos u)/u^3]^2, u = q R, Rg^2 = (3/5) R^2


def debye_form_factor(x: float) -> float:
    """Debye (Gaussian-coil) form factor P as a function of x = (q Rg)^2."""
    if x < 1e-6:
        return 1.0 - x / 3.0            # series limit; avoids 0/0 cancellation
    return (2.0 / x ** 2) * (math.exp(-x) - 1.0 + x)


def sphere_form_factor(u: float) -> float:
    """Solid-sphere form factor P as a function of u = q R (R the sphere radius)."""
    if u < 1e-6:
        return 1.0 - u ** 2 / 5.0
    return (3.0 * (math.sin(u) - u * math.cos(u)) / u ** 3) ** 2


def _curved_ordinate(angle_deg, c_g, mw, rg_nm, a2, *, wavelength_nm, n_solvent, model):
    """Kc/ΔR with a genuine form factor: 1/(Mw P(q)) + 2 A2 c."""
    q = phys.scattering_vector_q(angle_deg, wavelength_nm, n_solvent)   # nm^-1
    if model == "debye":
        P = debye_form_factor((q * rg_nm) ** 2)
    elif model == "sphere":
        R = rg_nm * math.sqrt(5.0 / 3.0)          # Rg^2 = (3/5) R^2
        P = sphere_form_factor(q * R)
    else:
        raise ValueError(f"model must be 'debye' or 'sphere', got {model!r}.")
    return 1.0 / (mw * P) + 2.0 * a2 * c_g


def make_curved_sls_set(
    mw: float,
    rg_nm: float,
    a2_mol_mL_per_g2: float,
    *,
    model: str = "debye",
    angles: Optional[Sequence[float]] = None,
    concs_g: Optional[Sequence[float]] = None,
    wavelength_nm: float = WAVELENGTH_NM,
    temperature_K: float = TEMPERATURE_K,
    n_solvent: float = N_SOLVENT,
    dn_dc: float = DN_DC,
    n_standard: float = N_STANDARD,
    geometry: str = "VU",
    calibrant_intensity: float = sd.DEFAULT_CALIBRANT_INTENSITY,
    solvent_intensity_90: float = sd.DEFAULT_SOLVENT_INTENSITY_90,
    polymer: str = "PEG",
    solvent: str = "water",
) -> Tuple[List[SLSMeasurement], "sd.CalibrationSpec"]:
    """Like make_sls_set, but intensities carry a genuine form factor P(q).

    The measured intensity is built by inverting the SAME analysis path the engine
    uses (ΔR = k_c · sinθ · (I_sample − I_solvent) · (n/n_std)^2), so the program
    recovers this curved ΔR — and then Zimm vs Berry differ because the ordinate is
    no longer linear in q^2. Ground truth stays known (mw, rg_nm, a2). Noise-free.
    """
    angles = list(DEFAULT_ANGLES if angles is None else [float(a) for a in angles])
    concs_g = list(DEFAULT_CONCS_G if concs_g is None else [float(c) for c in concs_g])
    cal = sd.CalibrationSpec(
        wavelength_nm=wavelength_nm, temperature_C=25.0, geometry=geometry,
        n_standard=n_standard, calibrant_intensity=calibrant_intensity)
    k_c = cal.k_c()
    K = phys.optical_constant_K(n_solvent, dn_dc, wavelength_nm)
    ri_corr = (n_solvent / n_standard) ** 2

    meas = []
    for c in concs_g:
        intensities = []
        for a in angles:
            base = sd.solvent_intensity(a, solvent_intensity_90)
            if c == 0:
                intensities.append(base)
                continue
            ordinate = _curved_ordinate(a, c, mw, rg_nm, a2_mol_mL_per_g2,
                                        wavelength_nm=wavelength_nm,
                                        n_solvent=n_solvent, model=model)
            dR = K * c / ordinate
            s = math.sin(math.radians(a))
            intensities.append(base + dR / (k_c * s * ri_corr))
        meas.append(SLSMeasurement(
            angles_deg=[float(a) for a in angles],
            intensities=intensities,
            polymer_name=polymer, solvent_name=solvent,
            concentration_g_per_mL=c, temperature_K=temperature_K,
            wavelength_nm=wavelength_nm, solvent_refractive_index=n_solvent,
            dn_dc_mL_per_g=dn_dc))
    return meas, cal


def solvent_reference(meas: Sequence[SLSMeasurement]) -> SLSMeasurement:
    """The c = 0 solvent-reference measurement in a set."""
    return next(m for m in meas if m.concentration_g_per_mL == 0)


def rayleigh_results(meas, cal, *, calibrated: bool = True):
    """Excess-Rayleigh-ratio results for every non-zero concentration.

    ``calibrated=False`` runs the uncalibrated path (calibration_constant=None,
    arbitrary scale) — used to test that Rg and the calibration-free product
    survive while Mw / A2 are flagged unreliable.
    """
    from analysis import sls

    sv = solvent_reference(meas)
    kc = cal.k_c() if calibrated else None
    return [
        sls.compute_excess_rayleigh_ratio(
            m, sv, calibration_constant=kc, standard_refractive_index=cal.n_standard)
        for m in meas if m.concentration_g_per_mL != 0
    ]
