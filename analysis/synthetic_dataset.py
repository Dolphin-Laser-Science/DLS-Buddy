"""
analysis/synthetic_dataset.py
=============================

Synthetic light-scattering **dataset** generation: the reusable engine behind
both the developer test-data generator and the in-app *Utilities ▸ Synthetic data*
generator.

What lives here
---------------
* **Forward models** (pure): the same optical-constant / Rayleigh / Zimm physics
  the analysis code uses, run *forwards* (parameters → intensities/correlograms),
  so that loading and analyzing the output recovers the ground truth.
* **Builders** (pure): assemble the artifacts the GUI offers — a single
  correlogram (delegated to :func:`analysis.utilities.generate_synthetic_correlogram`),
  a multi-angle DLS set, an SLS intensity set (a full Zimm grid or a single-angle
  / single-concentration slice), and a count-rate trace — each returned as a plain
  data object, never written or plotted here.
* **Writers** (file I/O): turn a built artifact into a *loadable instrument file*
  (ALV ``.ASC`` for DLS, Brookhaven ``.csv`` for SLS, a two-column CSV for a
  standalone trace). These are the file-I/O exception in the analysis package, in
  the same spirit as ``utilities.export_synthetic_correlogram_csv``; they are kept
  here so the controller (which may not import plotting) can save without the GUI.
* **Full-dataset orchestrator** :func:`generate_full_dataset`: regenerate a whole
  ``test-data/Synthetic *`` folder (the homologous Mw series + bimodal sample +
  trace + ``parameters.txt``). Picture rendering is intentionally *not* here (it
  needs the plotting layer); a separate developer wrapper adds it.

Deliberately **no plotting**, and the forward models are **re-derived here rather than
imported from the analysis engine** — so the generate → analyze round trip stays an
independent check — with a few documented exceptions where a second implementation
would add bugs rather than catch them (the Mie and core-shell form factors and the
core-shell contrast-weighted Rg, each import site saying why). The ground-truth
round-trip is validated by the test harness, not by this module analyzing its own
output; preview figures are drawn by the plotting layer from the GUI.

Seeding convention (every builder): one caller-supplied ``seed`` fans out into
independent streams by ADDITIVE offsets in disjoint bands, so that adding an angle,
a concentration or a blank later never moves the noise of anything already built --
    + round(100 * angle_deg)                 per-angle correlogram (0 .. 18 000); the matched
                                             traces of with_intensity_matched_traces too
    + 500 000                                the second run of a pair (VH; the blank)
    + 1 000 000 000 + k * 100 000            per-concentration correlograms, k = µg/mL
    + 2 000 000 000 + k * 100 000            per-concentration matched traces
    + 30 000                                 the calibrant trace
and the batch writers draw one seed per RUN from a profile-seeded stream (never the
literal profile seed for every run). ONE DOCUMENTED EXCEPTION: build_multi_angle_dls's
inline count-rate traces walk a single stream over the angle grid (they are ordered by
the grid, not per-angle seeded), so adding an angle moves the traces built after it.
Re-seeding them per angle would move every committed synthetic trace; the exception is
recorded rather than closed (2026-08-26 review).

The default constants describe the canonical test system (PEG in water, calibrated
against a toluene VU standard at 532 nm). They are defaults only: every builder
takes the physical parameters explicitly, so the in-app generator drives them from
user input and nothing system-specific is baked into a generated artifact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from physics import constants as phys
# The switch constants only -- a threshold, not a formula, so the models below stay
# independent transcriptions while the P > 1 cancellation window is closed everywhere.
from analysis.form_factor import _DEBYE_SERIES_X_MAX, _SPHERE_SERIES_U_MAX
from analysis.utilities import (
    SyntheticPopulation, SyntheticCorrelogramResult,
    generate_synthetic_correlogram,
)


# ===========================================================================
# Default test-data system (PEG in water vs a toluene VU standard at 532 nm)
# ===========================================================================
DEFAULT_WAVELENGTH_NM = 532.0
DEFAULT_TEMPERATURE_C = 25.0
DEFAULT_TEMPERATURE_K = 298.15
# The two refractive indices are the canonical test system's ROUND constants, not the
# solvent library's values at 532 nm / 25 C (water 1.3348, toluene 1.4996 -- the 1.496
# is toluene's sodium-D value). Every generated file writes its own n into its header,
# and the committed test-data/Synthetic corpora regenerate byte-for-byte from these,
# so an analysis of a generated set is self-consistent as long as the file's n is the
# one used; a user who accepts the library's proposal instead sees q shift by ~0.4 %
# and (n/n_std)^2 by ~0.7 %. Retuning them means regenerating every committed corpus
# (an owner decision, not a tidy-up), so they stay as they are and say what they are.
DEFAULT_N_SOLVENT = 1.33                   # water (round; library 1.3348 at 532 nm / 25 C)
DEFAULT_VISCOSITY_CP = 0.89
DEFAULT_VISCOSITY_PA_S = 0.89e-3
DEFAULT_N_STANDARD = 1.496                 # toluene n_D (library 1.4996 at 532 nm / 25 C)
DEFAULT_GEOMETRY = 'VU'                     # VU: vertical laser, no analyzer
DEFAULT_DN_DC = 0.135
DEFAULT_CALIBRANT_INTENSITY = 1.0e5        # toluene count rate at 90° (a.u.)
DEFAULT_SOLVENT_INTENSITY_90 = 6000.0      # water baseline at 90° (a.u.)
DEFAULT_KD_ML_G = 30.0                      # diffusion interaction kD (good solvent)

# Angle / concentration grids for the full test set.
DEFAULT_SLS_ANGLES = [35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145]
DEFAULT_DLS_ANGLES = [35, 50, 65, 80, 95, 110, 125, 140]
DEFAULT_SLS_CONCS_MG = [0.0, 0.2, 0.4, 0.6, 1.0, 1.4]    # mg/mL; the 0.0 is the ref

# Homologous Mw series for the full test set (Mw g/mol, Rg nm, Rh nm, A2 mol mL/g^2).
DEFAULT_SERIES = {
    'PEG 100k': dict(mw=1.0e5, rg=14.0, rh=9.5, a2=1.5e-4),
    'PEG 300k': dict(mw=3.0e5, rg=27.0, rh=18.0, a2=1.2e-4),
    'PEG 1M':   dict(mw=1.0e6, rg=55.0, rh=37.0, a2=9.5e-5),
    'PEG 3M':   dict(mw=3.0e6, rg=105.0, rh=70.0, a2=7.0e-5),
}
DEFAULT_DLS_CONCS_MG = {'PEG 1M': [0.2, 0.6, 1.4]}      # one sample gets a c-series
DEFAULT_DLS_SINGLE_MG = 0.6                              # the rest: one concentration

# The one series re-emitted in Brookhaven's SECOND SLS layout (single angular scan +
# its blank), so the corpus exercises `BrookhavenSLSScanParser` end to end. PEG 1M
# because Rg = 55 nm gives a scan with real angular dependence to fit; the
# concentration is one of DEFAULT_SLS_CONCS_MG so the column already exists and no
# extra random draw is taken.
DEFAULT_SLS_SCAN_SERIES = 'PEG 1M'
DEFAULT_SLS_SCAN_MG = 0.6

# Noise/realism profiles for the full test set.
PROFILES = {
    'Synthetic Clean': dict(corr_noise=0.005, sls_noise=0.005, trace_cv=0.03,
                            drift=0.0, spikes=0, seed=11),
    'Synthetic Messy': dict(corr_noise=0.030, sls_noise=0.025, trace_cv=0.06,
                            drift=0.12, spikes=4, seed=29),
}


# ===========================================================================
# Calibration specification
# ===========================================================================
@dataclass
class CalibrationSpec:
    """The SLS calibration used to scale generated excess intensities.

    ``k_c`` (= R_standard / I_calibrant) maps an intensity to a Rayleigh ratio,
    exactly as the analysis calibration does in reverse. ``include_in_files``
    controls only whether the *written* Brookhaven file carries calibration
    metadata: with it on (the default) the file documents the standard + constant,
    so a user who enters the matching calibrant gets an absolute Mw; with it off
    the file omits that metadata, so an analysis run is flagged uncalibrated
    (arbitrary-scale Mw). Either way the intensities themselves are generated with
    the same ``k_c`` — "uncalibrated" is a property of how the data is *used*, not
    of the numbers.
    """
    wavelength_nm: float = DEFAULT_WAVELENGTH_NM
    temperature_C: float = DEFAULT_TEMPERATURE_C
    geometry: str = DEFAULT_GEOMETRY
    n_standard: float = DEFAULT_N_STANDARD
    calibrant_intensity: float = DEFAULT_CALIBRANT_INTENSITY
    include_in_files: bool = True

    def rayleigh(self) -> float:
        """Rayleigh ratio of the toluene standard (cm^-1) for this geometry."""
        return phys.rayleigh_ratio_toluene(
            self.wavelength_nm, self.temperature_C, geometry=self.geometry)

    def k_c(self) -> float:
        """Calibration constant k_c = R_standard / I_calibrant."""
        return self.rayleigh() / self.calibrant_intensity


# ===========================================================================
# Forward models (pure)
# ===========================================================================
def solvent_intensity(angle_deg: float, i_ref_90: float) -> float:
    """Isotropic solvent baseline: I ∝ 1/sin θ (constant scattering volume × 1/sinθ)."""
    return i_ref_90 / math.sin(math.radians(angle_deg))


def _form_factor_P(model: str, q_nm_inv: float, rg_nm: float,
                   ellipsoid_axial_ratio: float = 1.0, *,
                   m: Optional[float] = None, wavelength_nm: Optional[float] = None,
                   n_solvent: Optional[float] = None,
                   r_core_nm: Optional[float] = None, r_outer_nm: Optional[float] = None,
                   m1: Optional[float] = None, m2: Optional[float] = None) -> float:
    """Single-particle form factor P(q), parameterized by Rg (sphere/coil are closed-form;
    the ellipsoid orientation average is evaluated numerically -- see below).

    The analytic models are deliberately inlined here (NOT imported from analysis.form_factor)
    so the generate->analyze round trip stays an independent check — the same convention the
    Cabannes/DDLS forward models follow. 'sphere' forms the geometric radius R = Rg·sqrt(5/3)
    and u = qR; 'debye' uses x = (q Rg)². 'ellipsoid' averages the sphere amplitude over
    orientation (Pedersen Eq. 61) with axial ratio `ellipsoid_axial_ratio` (eps =
    polar/equatorial), using an INDEPENDENT trapezoid rule in alpha (the engine uses
    Gauss-Legendre in u), so this stays a genuine cross-check. Series limits guard 0/0.

    'mie' (full Mie, needs the relative index `m` = n_solute/n_solvent + the optics) is the ONE
    model that does NOT inline an independent copy: a second hand-rolled Mie solver would be far
    more likely to introduce a bug than to catch one, and analysis/mie.py is already validated
    against the independent miepython oracle (tests/test_mie.py), so the round-trip independence
    for Mie is carried there. It uses analysis.mie via analysis.form_factor.mie_P.

    'core_shell' (coated sphere, needs the two radii `r_core_nm`/`r_outer_nm` and contrasts
    `m1`/`m2` + the optics) follows the SAME reasoned deviation as Mie -- it reuses the vendored
    Aden-Kerker/BHCOAT solver via analysis.form_factor.core_shell_P rather than a second inlined
    copy; its independence is carried by the PyMieScatt oracle in tests/test_mie.py. `rg_nm` is
    ignored for this model (the coated P(q) is set by the two radii, not a single Rg).
    """
    if model == 'sphere':
        u = q_nm_inv * rg_nm * math.sqrt(5.0 / 3.0)
        if abs(u) < _SPHERE_SERIES_U_MAX:
            u2 = u * u
            return 1.0 - u2 / 5.0 + 3.0 * u2 * u2 / 175.0
        return (3.0 * (math.sin(u) - u * math.cos(u)) / u ** 3) ** 2
    if model == 'debye':
        x = (q_nm_inv * rg_nm) ** 2
        if abs(x) < _DEBYE_SERIES_X_MAX:
            return 1.0 - x / 3.0 + x * x / 12.0
        return (2.0 / x ** 2) * (math.exp(-x) - 1.0 + x)
    if model == 'ellipsoid':
        eps = ellipsoid_axial_ratio
        R = rg_nm * math.sqrt(5.0 / (2.0 + eps * eps))       # equatorial radius
        alpha = np.linspace(0.0, math.pi / 2.0, 257)
        r = R * np.sqrt(np.sin(alpha) ** 2 + eps * eps * np.cos(alpha) ** 2)
        u = q_nm_inv * r
        small = np.abs(u) < _SPHERE_SERIES_U_MAX
        u_safe = np.where(small, 1.0, u)
        F2 = np.where(small, 1.0 - u ** 2 / 5.0 + 3.0 * u ** 4 / 175.0,
                      (3.0 * (np.sin(u_safe) - u_safe * np.cos(u_safe)) / u_safe ** 3) ** 2)
        return float(np.trapezoid(F2 * np.sin(alpha), alpha))
    if model == 'mie':
        if m is None or wavelength_nm is None or n_solvent is None:
            raise ValueError("form_factor_model='mie' needs m, wavelength_nm, and n_solvent.")
        from analysis.form_factor import mie_P
        return float(mie_P(np.array([q_nm_inv]), rg_nm, m,
                           wavelength_nm=wavelength_nm, n_medium=n_solvent)[0])
    if model == 'core_shell':
        if (r_core_nm is None or r_outer_nm is None or m1 is None or m2 is None
                or wavelength_nm is None or n_solvent is None):
            raise ValueError("form_factor_model='core_shell' needs r_core_nm, r_outer_nm, "
                             "m1, m2, wavelength_nm, and n_solvent.")
        from analysis.form_factor import core_shell_P
        return float(core_shell_P(np.array([q_nm_inv]), r_core_nm, r_outer_nm, m1, m2,
                                  wavelength_nm=wavelength_nm, n_medium=n_solvent)[0])
    raise ValueError(
        f"form_factor_model must be 'sphere', 'debye', 'ellipsoid', 'mie', or 'core_shell', "
        f"got {model!r}.")


def excess_rayleigh(angle_deg: float, c_g_per_mL: float, mw: float, rg_nm: float,
                    a2_mol_mL_per_g2: float, *, wavelength_nm: float,
                    n_solvent: float, dn_dc: float,
                    form_factor_model: str = 'guinier',
                    ellipsoid_axial_ratio: float = 1.0,
                    m: Optional[float] = None,
                    r_core_nm: Optional[float] = None, r_outer_nm: Optional[float] = None,
                    m1: Optional[float] = None, m2: Optional[float] = None) -> float:
    """Excess Rayleigh ratio ΔR (cm^-1) from the Zimm equation, run forwards.

    `form_factor_model` selects the angular dependence:

      - 'guinier' (default): the small-qRg Zimm expansion
            Kc/ΔR = (1/Mw)(1 + q² Rg²/3) + 2 A2 c
        — the historical generator behaviour (valid only at small qRg).
      - 'sphere' / 'debye' / 'ellipsoid' / 'mie': the FULL form factor
            Kc/ΔR = 1/(Mw · P(q; Rg)) + 2 A2 c
        so the set carries genuine large-qRg curvature the form-factor fit recovers
        (and Zimm/Berry do not). ⇒ ΔR = K c / (that ordinate). `ellipsoid_axial_ratio`
        (eps = polar/equatorial) is used only for 'ellipsoid'; `m` (= n_solute/n_solvent, the
        relative index) is REQUIRED only for 'mie' and emits genuine large-x Mie curvature.
    """
    q = phys.scattering_vector_q(angle_deg, wavelength_nm, n_solvent)   # nm^-1
    K = phys.optical_constant_K(n_solvent, dn_dc, wavelength_nm)
    if form_factor_model == 'guinier':
        inv_P_term = (1.0 / mw) * (1.0 + q * q * rg_nm * rg_nm / 3.0)
    else:
        inv_P_term = 1.0 / (mw * _form_factor_P(form_factor_model, q, rg_nm,
                                                ellipsoid_axial_ratio, m=m,
                                                wavelength_nm=wavelength_nm,
                                                n_solvent=n_solvent,
                                                r_core_nm=r_core_nm, r_outer_nm=r_outer_nm,
                                                m1=m1, m2=m2))
    ordinate = inv_P_term + 2.0 * a2_mol_mL_per_g2 * c_g_per_mL
    return K * c_g_per_mL / ordinate


def sample_intensity(angle_deg: float, c_g_per_mL: float, mw: float, rg_nm: float,
                     a2_mol_mL_per_g2: float, *, wavelength_nm: float,
                     n_solvent: float, dn_dc: float, k_c: float, n_standard: float,
                     i_ref_90: float, form_factor_model: str = 'guinier',
                     ellipsoid_axial_ratio: float = 1.0,
                     m: Optional[float] = None,
                     r_core_nm: Optional[float] = None, r_outer_nm: Optional[float] = None,
                     m1: Optional[float] = None, m2: Optional[float] = None) -> float:
    """Measured intensity = solvent baseline + the excess from ΔR, un-calibrated.

    Inverts the analysis path ΔR = k_c · sinθ · (I_sample − I_solvent) · (n/n_std)²,
    so the program recovers ΔR (and hence Mw/Rg/A2) from the generated intensity.
    `form_factor_model` (+ `ellipsoid_axial_ratio` / `m` / the core-shell radii+contrasts) is
    passed to `excess_rayleigh`.
    """
    base = solvent_intensity(angle_deg, i_ref_90)
    if c_g_per_mL == 0:
        return base
    dR = excess_rayleigh(angle_deg, c_g_per_mL, mw, rg_nm, a2_mol_mL_per_g2,
                         wavelength_nm=wavelength_nm, n_solvent=n_solvent, dn_dc=dn_dc,
                         form_factor_model=form_factor_model,
                         ellipsoid_axial_ratio=ellipsoid_axial_ratio, m=m,
                         r_core_nm=r_core_nm, r_outer_nm=r_outer_nm, m1=m1, m2=m2)
    s = math.sin(math.radians(angle_deg))
    return base + dR / (k_c * s * (n_solvent / n_standard) ** 2)


def rh_at_concentration(rh0_nm: float, c_g_per_mL: float,
                        kd_mL_g: float = DEFAULT_KD_ML_G) -> float:
    """Apparent Rh shrinking with concentration: D(c)=D0(1+kD c) ⇒ Rh = Rh0/(1+kD c)."""
    return rh0_nm / (1.0 + kd_mL_g * c_g_per_mL)


def _sphere_i1_inline(model: str, r_nm: float, q_nm_inv: float, *,
                      wavelength_nm: Optional[float] = None,
                      n_solvent: Optional[float] = None,
                      m: Optional[float] = None) -> float:
    """Relative per-particle sphere scattering I1(R, q), the forward direction of the
    Phase-1b/Phase-3 reweighting. The analytic models are inlined here (NOT imported from
    analysis.optical_model) so the generate->reweight->recover round trip stays an independent
    check. 'rayleigh': I1 ∝ R^6; 'rdg': I1 ∝ R^6 P(qR). Series limit guards the 0/0.

    'mie' (needs wavelength/n_solvent/m) is the exception that does NOT inline an independent
    copy -- a second Mie solver would be more likely to add a bug than catch one, and
    analysis/mie.py is validated against the miepython oracle, so the independence for Mie is
    carried there. It uses analysis.mie directly (|S1(x, m, theta)|^2)."""
    r6 = r_nm ** 6
    if model == 'rayleigh':
        return r6
    if model == 'rdg':
        u = q_nm_inv * r_nm
        if abs(u) < _SPHERE_SERIES_U_MAX:
            P = 1.0 - u * u / 5.0 + 3.0 * u * u * u * u / 175.0
        else:
            P = (3.0 * (math.sin(u) - u * math.cos(u)) / u ** 3) ** 2
        return r6 * P
    if model == 'mie':
        if m is None or wavelength_nm is None or n_solvent is None:
            raise ValueError("sphere model 'mie' needs wavelength_nm, n_solvent, and m.")
        from analysis import mie
        k = phys.optical_wavevector_k(wavelength_nm, n_solvent)   # nm^-1
        x = k * r_nm
        sin_half = min(1.0, max(0.0, q_nm_inv / (2.0 * k)))
        cos_theta = 1.0 - 2.0 * sin_half * sin_half
        return float(mie.mie_phase_vv(x, m, cos_theta))
    raise ValueError(f"sphere model must be 'rayleigh', 'rdg', or 'mie', got {model!r}.")


def intensity_populations_from_composition(
    specs: Sequence[Tuple[float, float]], weighting: str, *,
    angle_deg: float, wavelength_nm: float, n_solvent: float,
    model: str = 'rayleigh', m: Optional[float] = None) -> List[SyntheticPopulation]:
    """Forward-model INTENSITY-weighted populations from a known number/volume/surface
    composition, so the Phase-1b/Phase-3 reweighting inverse can be exercised end-to-end from
    a specified composition (the DLS-side generator-coverage gap, 2026-07-19 rule).

    ``specs`` is a list of ``(rh_nm, composition_weight)`` in the given ``weighting``
    ('number' | 'surface' | 'volume'). Each is converted to a NUMBER weight
    (N ∝ comp / R^p, p = 0/2/3) and then to an intensity/g1 amplitude via the per-particle
    sphere scattering ``I1(R, q)`` at this angle's q -- the exact inverse of
    ``analysis.optical_model.reweight_distribution``. The returned monodisperse
    SyntheticPopulation list drops straight into ``build_correlogram`` /
    ``build_multi_angle_dls`` (whose ``weight`` is an intensity amplitude), so the emitted
    correlogram's recovered intensity distribution reweights back to the input composition.

    Rayleigh's I1 ∝ R^6 is angle-independent; RDG's and Mie's carry the angular form factor, so
    their intensity weighting is angle-specific (reweight at the SAME angle to recover the
    composition). ``model='mie'`` (the general high-contrast/large-x case, e.g. latex in water
    that RDG refuses) REQUIRES the relative index ``m`` = n_solute/n_solvent.
    """
    p_exp = {'number': 0, 'surface': 2, 'volume': 3}.get(weighting)
    if p_exp is None:
        raise ValueError(
            f"weighting must be 'number', 'surface', or 'volume', got {weighting!r}.")
    q_nm_inv = phys.scattering_vector_q(angle_deg, wavelength_nm, n_solvent)   # nm^-1
    pops: List[SyntheticPopulation] = []
    for rh_nm, comp in specs:
        if not (math.isfinite(rh_nm) and rh_nm > 0
                and math.isfinite(comp) and comp > 0):
            raise ValueError(
                f"composition entry needs a positive finite rh_nm and weight, "
                f"got ({rh_nm!r}, {comp!r}).")
        number = comp / rh_nm ** p_exp
        intensity = number * _sphere_i1_inline(model, rh_nm, q_nm_inv,
                                               wavelength_nm=wavelength_nm,
                                               n_solvent=n_solvent, m=m)
        pops.append(SyntheticPopulation(rh_nm=float(rh_nm), weight=float(intensity)))
    return pops


# ===========================================================================
# Artifact data containers
# ===========================================================================
@dataclass
class SyntheticTrace:
    """A generated count-rate trace in canonical units (s, cps)."""
    times_s: np.ndarray
    count_rates_cps: np.ndarray
    label: str = ''


@dataclass
class MultiAngleDLS:
    """Correlograms (and a count-rate trace) at several angles, one ALV-file's worth."""
    angles_deg: List[float]
    delay_times_s: np.ndarray
    signals: Dict[float, np.ndarray]          # angle -> g2-1 column
    trace_times_s: np.ndarray
    trace_cps: Dict[float, np.ndarray]        # angle -> count rate (cps)
    mean_cr_kHz: Dict[float, float]
    # context (for writing + building measurements)
    label: str
    concentration_g_per_mL: float
    temperature_K: float
    viscosity_Pa_s: float
    n_solvent: float
    wavelength_nm: float
    output_form: str = 'g2m1'


@dataclass
class SyntheticSLSSet:
    """An SLS intensity set: intensities over an angle grid for several concentrations.

    ``concentrations_g_per_mL`` always includes 0.0 (the solvent reference) first.
    A full Zimm set has many angles × many concentrations; a single-concentration
    slice has many angles × one concentration; a single-angle slice has one angle ×
    many concentrations. The writer and the measurement-builder handle all three.
    """
    angles_deg: np.ndarray
    concentrations_g_per_mL: List[float]
    intensities: Dict[float, np.ndarray]      # concentration -> intensity over angles
    # context
    label: str
    polymer_name: str
    solvent_name: str
    temperature_K: float
    wavelength_nm: float
    n_solvent: float
    dn_dc: float
    cal: CalibrationSpec
    kind: str = 'zimm'                        # 'zimm' | 'single_concentration' | 'single_angle'
    # Hand-entered solute refractive index for a Mie set (n_solute = m * n_solvent); for a
    # core-shell set this is the CORE index (n_core = m1 * n_solvent). None for every other
    # model. Carried so an injected artifact reaches the workspace WITH the RI and is fittable
    # end-to-end (the Mie fit forms m = n_solute/n_solvent; core-shell forms m1 from it).
    solute_refractive_index: Optional[float] = None
    # Shell refractive index for a core-shell set (n_shell = m2 * n_solvent); None otherwise.
    # Carried so an injected core-shell artifact reaches the workspace with BOTH contrasts.
    shell_refractive_index: Optional[float] = None
    # Whether the WRITTEN file (and an injected measurement) carries the dn/dc. The
    # intensities are always generated with the real ``dn_dc`` -- they are physical --
    # but with this off the number is withheld, exactly as ``cal.include_in_files``
    # withholds the calibration block: the parser then leaves dn/dc None and the SLS
    # engine runs at K = 1 on a disclosed arbitrary optical scale (CLAUDE.md invariant
    # 3). It exists so that shipped path is reachable from synthetic ground truth
    # (owner-ruled 2026-08-26); dn/dc is still never proposed or defaulted by the program.
    dn_dc_in_files: bool = True

    @property
    def calibrated(self) -> bool:
        return self.cal.include_in_files


# ===========================================================================
# Builders (pure — return data, no I/O, no plotting)
# ===========================================================================
def build_correlogram(populations: Sequence[SyntheticPopulation], *, angle_deg: float,
                      wavelength_nm: float, solvent_refractive_index: float,
                      temperature_K: float, viscosity_Pa_s: float, beta: float = 0.8,
                      noise_level: float = 0.0, n_points: int = 200,
                      output_form: str = 'g2m1',
                      seed: Optional[int] = None,
                      **axis_kw) -> SyntheticCorrelogramResult:
    """A single correlogram (thin pass-through to the utilities generator).

    ``axis_kw`` forwards the lag-axis and noise-model parameters -- ``grid``,
    ``noise_model`` and the ``multitau_*`` set -- so a caller can ask
    for the multi-tau staircase a real correlator emits. Defaults are the generator's,
    i.e. the uniform log grid with constant noise.
    """
    return generate_synthetic_correlogram(
        populations, angle_deg=angle_deg, wavelength_nm=wavelength_nm,
        solvent_refractive_index=solvent_refractive_index, temperature_K=temperature_K,
        viscosity_Pa_s=viscosity_Pa_s, beta=beta, noise_level=noise_level,
        n_points=n_points, output_form=output_form, seed=seed, **axis_kw)


def build_count_rate_trace(*, duration_s: float = 120.0, dt_s: float = 0.1,
                           mean_kHz: float = 300.0, trace_cv: float = 0.03,
                           drift: float = 0.0, spikes: int = 0,
                           offset_start_s: Optional[float] = None,
                           offset_duration_s: float = 0.0,
                           offset_amplitude: float = 0.0,
                           sine_frequency_hz: float = 0.0,
                           sine_amplitude: float = 0.0,
                           seed: Optional[int] = None,
                           label: str = 'synthetic trace') -> SyntheticTrace:
    """A noisy count-rate trace: fractional-CV scatter + optional slow drift + dust spikes.

    The scatter is ``trace_cv * mean`` Gaussian, i.e. a FIXED relative width -- not
    shot noise, whose width would be sqrt(mean). At 300 kHz the two differ by ~30x.
    The distinction is load-bearing: a Poisson-shaped outlier envelope cannot see the
    spikes planted here, which is why the trace flagger uses a robust scale instead.

    LONG-RESIDENCE OFFSET (``offset_*``). A step that persists for
    ``offset_duration_s`` from ``offset_start_s``, raising the level by
    ``offset_amplitude * mean`` -- fractional, like ``drift``. Distinct from the
    point ``spikes`` above: it is the settled contamination a dust particle
    produces when it sits still in a viscous medium rather than passing through.
    It exists so the windowed Hampel layer's structural blind spot is
    demonstrable from ground truth -- an offset much longer than the analysis
    window must appear in that layer's recovered BASELINE and raise no impulse
    flags at all, while a shorter one flags. Planted before the spikes, so a
    spike landing inside the offset rides on the elevated level.

    PERIODIC COMPONENT (``sine_*``). A pure sinusoid of ``sine_frequency_hz``
    whose amplitude is ``sine_amplitude * mean`` -- fractional again, like
    ``drift`` and the offset. This is the ground truth for the frequency-domain
    diagnostics: the Welch spectrum must show a line at the planted frequency,
    and the periodogram's false-alarm test must call it significant while the
    same trace without it stays below the line. It is deliberately a clean
    sinusoid rather than a realistic vibration signature, because what is being
    validated is frequency RECOVERY against a number that is known exactly.

    Frequencies at or above Nyquist (1 / 2 dt_s) alias, exactly as they would on
    a real instrument; nothing here prevents that, since a diagnostic that hides
    aliasing would misteach the very thing this generator exists to show.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration_s, dt_s)
    cr = mean_kHz * (1.0 + trace_cv * rng.standard_normal(t.size))
    if drift and t.size and t[-1] > 0:                # t[-1]>0 guards a single-sample
        cr = cr + mean_kHz * drift * (t / t[-1])      # t[-1]>0 avoids a grid (t=[0.0]) -> 0/0
    if sine_frequency_hz and sine_amplitude:
        cr = cr + mean_kHz * sine_amplitude * np.sin(2.0 * np.pi
                                                     * sine_frequency_hz * t)
    if offset_start_s is not None and offset_duration_s > 0 and offset_amplitude:
        within = (t >= offset_start_s) & (t < offset_start_s + offset_duration_s)
        cr[within] += mean_kHz * offset_amplitude
    for _ in range(int(spikes)):
        cr[rng.integers(t.size)] *= rng.uniform(2.0, 3.5)
    cr = np.clip(cr, 1.0, None)
    return SyntheticTrace(times_s=t, count_rates_cps=cr * 1000.0, label=label)


def build_multi_angle_dls(populations_at_angle, *, angles_deg: Sequence[float],
                          wavelength_nm: float, solvent_refractive_index: float,
                          temperature_K: float, viscosity_Pa_s: float, beta: float = 0.85,
                          noise_level: float = 0.0, n_points: int = 200,
                          trace_cv: float = 0.03, drift: float = 0.0, spikes: int = 0,
                          label: str = '', concentration_g_per_mL: float = 0.0,
                          seed: Optional[int] = None, **axis_kw) -> MultiAngleDLS:
    """Correlograms at every angle (+ a count-rate trace per angle), for one sample.

    ``populations_at_angle`` is either a fixed list of SyntheticPopulation (same
    sizes at every angle) or a callable ``angle_deg -> list`` (rarely needed). The
    per-angle seed is offset by the angle in hundredths of a degree (the resolution a
    goniometer reads to) so the curves are independent for any two angles the GUI
    accepts -- an integer offset collapsed 90.0 and 90.4 onto one noise vector.

    The count-rate trace at each angle follows the SAME angular law as the module's
    static forward model (``solvent_intensity``: I proportional to 1/sin(theta), the
    scattering-volume factor), anchored at 350 kHz at 95 deg. Trace and static
    intensity are forward models of one observable, so a static series derived from
    these traces carries the solvent / scattering-volume angular shape (the 1/sin(theta)
    factor only -- no P(q): a derived static series from a bare multi-angle set is flat
    after the volume correction; the sample's own angular structure lives in the
    consistency tuple's matched traces).

    ``axis_kw`` forwards the lag-axis and noise-model parameters (see
    :func:`build_correlogram`). Every angle shares one axis, as a real instrument's
    angles do -- which is what lets the set be written to a single multi-angle .ASC.
    """
    angles = [float(a) for a in angles_deg]
    signals: Dict[float, np.ndarray] = {}
    delay = None
    for a in angles:
        pops = populations_at_angle(a) if callable(populations_at_angle) else populations_at_angle
        g = generate_synthetic_correlogram(
            pops, angle_deg=a, wavelength_nm=wavelength_nm,
            solvent_refractive_index=solvent_refractive_index, temperature_K=temperature_K,
            viscosity_Pa_s=viscosity_Pa_s, beta=beta, noise_level=noise_level,
            n_points=n_points, output_form='g2m1',
            seed=(None if seed is None else seed + int(round(100.0 * a))), **axis_kw)
        delay = g.delay_times_s
        signals[a] = g.signal

    # count-rate traces: more scattering at low angle; fractional-CV scatter (NOT shot
    # noise -- see build_count_rate_trace's docstring) + drift/spikes
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, 60.0, 0.05)
    trace_cps: Dict[float, np.ndarray] = {}
    mean_cr: Dict[float, float] = {}
    for a in angles:
        base = solvent_intensity(a, 350.0 * math.sin(math.radians(95.0)))    # kHz
        cr = base * (1.0 + trace_cv * rng.standard_normal(t.size))
        if drift and t.size and t[-1] > 0:           # guard t=[0.0] -> 0/0
            cr = cr + base * drift * (t / t[-1])
        for _ in range(int(spikes)):
            cr[rng.integers(t.size)] *= rng.uniform(2.0, 3.5)
        cps = np.clip(cr, 1.0, None) * 1000.0
        trace_cps[a] = cps
        mean_cr[a] = float(cps.mean() / 1000.0)        # kHz

    return MultiAngleDLS(
        angles_deg=angles, delay_times_s=delay, signals=signals,
        trace_times_s=t, trace_cps=trace_cps, mean_cr_kHz=mean_cr,
        label=label, concentration_g_per_mL=concentration_g_per_mL,
        temperature_K=temperature_K, viscosity_Pa_s=viscosity_Pa_s,
        n_solvent=solvent_refractive_index, wavelength_nm=wavelength_nm)


@dataclass
class TwoModeDLS:
    """A two-mode (fast + slow) multi-angle DLS set WITH its ground truth.

    The substrate for validating mode-resolved dynamics: the fast mode is a
    real diffusive population (Gamma_f = q^2 D_f); the slow mode is either
    diffusive too, or Q-INDEPENDENT (constant Gamma_s -- the relaxation-mode
    class), expressed through the effective-Rh bookkeeping trick
    (`vh_effective_population`'s precedent; the per-angle effective radius is
    never written out and reading it back as a size is meaningless).

    Ground truth is stored per angle so a recovery test asserts against what
    was actually generated -- and so a later linked "consistency tuple"
    (correlograms + trace + per-angle static intensities from ONE truth) can
    extend this record rather than invent a sibling.

    `crossing_angle_deg` is where Gamma_f(theta) = Gamma_s -- the regime where
    any rank-based mode association breaks, which is why it must be the TESTED
    case; NaN when the modes never cross (e.g. both diffusive).
    """
    data: MultiAngleDLS
    # ground truth
    rh_fast_nm: float
    d_fast_m2_s: float
    gamma_fast_s_inv: Dict[float, float]     # per angle
    gamma_slow_s_inv: Dict[float, float]     # per angle (constant if q-independent)
    a_fast_fraction: float                   # normalized g1 amplitude fractions
    a_slow_fraction: float
    slow_kind: str                           # 'diffusive' | 'q_independent'
    rh_slow_nm: float                        # NaN for a q-independent slow mode
    d_slow_m2_s: float                       # NaN for a q-independent slow mode
    crossing_angle_deg: float                # NaN when the modes never cross


def two_mode_populations(rh_fast_nm: float, gamma_slow_s_inv: float, *,
                         a_fast: float, a_slow: float,
                         wavelength_nm: float, n_solvent: float,
                         temperature_K: float, viscosity_Pa_s: float):
    """Callable ``angle_deg -> [fast, slow]`` populations for a q-independent slow mode.

    The fast population is a real Stokes-Einstein size; the slow one is the
    effective-Rh bookkeeping of a CONSTANT decay rate (chosen per angle so
    q^2 D_eff = Gamma_s), the `vh_effective_population` trick. The effective
    radius is never written out; a q-independent mode has no size. The slow
    population runs with `spread_cv = 0` by construction -- a width on a
    bookkeeping radius would model nothing.
    """
    def pops(angle_deg: float) -> List[SyntheticPopulation]:
        q = phys.scattering_vector_q_m(angle_deg, wavelength_nm, n_solvent)
        d_eff = gamma_slow_s_inv / (q * q)
        rh_eff_nm = phys.stokes_einstein_rh(d_eff, temperature_K, viscosity_Pa_s) * 1e9
        return [SyntheticPopulation(rh_nm=rh_fast_nm, weight=a_fast),
                SyntheticPopulation(rh_nm=rh_eff_nm, weight=a_slow)]
    return pops


def build_two_mode_dls(*, rh_fast_nm: float = 20.0, a_fast: float = 0.6,
                       a_slow: float = 0.4,
                       gamma_slow_s_inv: Optional[float] = None,
                       rh_slow_nm: Optional[float] = None,
                       crossing_angle_deg: Optional[float] = None,
                       angles_deg: Sequence[float], wavelength_nm: float,
                       n_solvent: float, temperature_K: float,
                       viscosity_Pa_s: float, beta: float = 0.85,
                       noise_level: float = 0.0, n_points: int = 200,
                       label: str = 'Two-mode DLS',
                       concentration_g_per_mL: float = 0.0,
                       seed: Optional[int] = None, **axis_kw) -> TwoModeDLS:
    """A fast+slow two-mode multi-angle set with known per-angle ground truth.

    The slow mode is specified by AT MOST one of:

      * ``rh_slow_nm``          -- a second diffusive population (no crossing);
      * ``gamma_slow_s_inv``    -- a q-independent mode at that constant rate;
      * ``crossing_angle_deg``  -- a q-independent mode whose rate is DERIVED
        so the fast mode's Gamma_f(theta) sweeps past it exactly there
        (Gamma_s = q(theta_c)^2 D_f).

    With none given, the default is a q-independent slow mode crossing at 82
    degrees -- the angle where Provencher & Stepanek's simulated fast and
    slow modes exactly overlap (their p. 2: at q = (q1*q5)^(1/2), i.e.
    theta = 82 deg, in a 60-140 deg set) [provencherstepanek1996] -- which
    sits mid-range for a typical goniometer's span, so the default dataset
    exercises exactly the case where rank-based association breaks.

    ``a_fast`` / ``a_slow`` are relative g1 amplitudes (the generator's
    population weights); the stored ground-truth fractions are normalized.
    """
    if not (math.isfinite(a_fast) and a_fast > 0
            and math.isfinite(a_slow) and a_slow > 0):
        raise ValueError(
            f"a_fast and a_slow must be positive finite g1 amplitudes, got "
            f"a_fast={a_fast!r}, a_slow={a_slow!r}."
        )
    n_specs = sum(v is not None
                  for v in (gamma_slow_s_inv, rh_slow_nm, crossing_angle_deg))
    if n_specs > 1:
        raise ValueError(
            "Give at most one of gamma_slow_s_inv, rh_slow_nm and "
            "crossing_angle_deg -- they are three ways to specify the ONE "
            "slow mode."
        )

    # D_f from the fast size (raises out of physics.constants for a bad size --
    # the operation-level guard, the build_ddls_set_from_diffusion precedent).
    d_fast = phys.stokes_einstein_diffusion_coefficient(
        rh_fast_nm * 1e-9, temperature_K, viscosity_Pa_s)

    angles = [float(a) for a in angles_deg]

    def _gamma_at(angle: float, d: float) -> float:
        q = phys.scattering_vector_q_m(angle, wavelength_nm, n_solvent)
        return d * (q * q)               # `*`, not `**` (the Python-scalar trap)

    if rh_slow_nm is not None:
        # Both modes diffusive: the plain two-population path; no crossing
        # (Gamma_f/Gamma_s is angle-independent).
        slow_kind = 'diffusive'
        d_slow = phys.stokes_einstein_diffusion_coefficient(
            rh_slow_nm * 1e-9, temperature_K, viscosity_Pa_s)
        pops = [SyntheticPopulation(rh_nm=rh_fast_nm, weight=a_fast),
                SyntheticPopulation(rh_nm=float(rh_slow_nm), weight=a_slow)]
        gamma_slow = {a: _gamma_at(a, d_slow) for a in angles}
        crossing = float('nan')
    else:
        slow_kind = 'q_independent'
        if gamma_slow_s_inv is None:
            theta_c = 82.0 if crossing_angle_deg is None else float(crossing_angle_deg)
            q_c = phys.scattering_vector_q_m(theta_c, wavelength_nm, n_solvent)
            gamma_slow_s_inv = d_fast * q_c * q_c
        gamma_slow_s_inv = float(gamma_slow_s_inv)
        d_slow = float('nan')
        rh_slow_nm = float('nan')
        pops = two_mode_populations(
            rh_fast_nm, gamma_slow_s_inv, a_fast=a_fast, a_slow=a_slow,
            wavelength_nm=wavelength_nm, n_solvent=n_solvent,
            temperature_K=temperature_K, viscosity_Pa_s=viscosity_Pa_s)
        gamma_slow = {a: gamma_slow_s_inv for a in angles}
        # Where Gamma_f(theta) = Gamma_s: sin(theta/2) = q_c * lambda/(4 pi n),
        # with q_c = sqrt(Gamma_s / D_f). NaN when outside the physical range.
        sin_half = (math.sqrt(gamma_slow_s_inv / d_fast)
                    * (wavelength_nm * 1e-9) / (4.0 * math.pi * n_solvent))
        crossing = (2.0 * math.degrees(math.asin(sin_half))
                    if 0.0 < sin_half <= 1.0 else float('nan'))

    data = build_multi_angle_dls(
        pops, angles_deg=angles, wavelength_nm=wavelength_nm,
        solvent_refractive_index=n_solvent, temperature_K=temperature_K,
        viscosity_Pa_s=viscosity_Pa_s, beta=beta, noise_level=noise_level,
        n_points=n_points, label=label,
        concentration_g_per_mL=concentration_g_per_mL, seed=seed, **axis_kw)

    gamma_fast = {a: _gamma_at(a, d_fast) for a in angles}
    total = a_fast + a_slow
    return TwoModeDLS(
        data=data, rh_fast_nm=float(rh_fast_nm), d_fast_m2_s=float(d_fast),
        gamma_fast_s_inv=gamma_fast, gamma_slow_s_inv=gamma_slow,
        a_fast_fraction=float(a_fast / total),
        a_slow_fraction=float(a_slow / total),
        slow_kind=slow_kind, rh_slow_nm=float(rh_slow_nm),
        d_slow_m2_s=float(d_slow), crossing_angle_deg=float(crossing))


@dataclass
class ConsistencyTuple:
    """Correlograms, count-rate traces and static intensities from ONE ground truth.

    The generator's other builders parameterise a correlogram set and a static set
    independently, so their numbers are unrelated: ``build_multi_angle_dls`` bases
    its trace on a hard-coded ``350 kHz * (sin 95 / sin theta)**0.5`` shape, while
    ``sample_intensity`` builds intensity as ``I_solvent(theta) + dR/(k_c sin theta
    (n/n_std)^2)``. That is fine for exercising each surface alone, and useless for
    the trace-derived static chain, whose whole claim is that the two ARE the same
    number.

    Here they are the same number by construction: the traces are built to have the
    static intensities as their means, so `sum` closure and Mw recovery are
    structural rather than asserted. ``dls`` is keyed by concentration and includes
    0.0 (the solvent blank), so injecting it lands the blank in the sample's
    reference slot through the ordinary c = 0 regrouping, with no special routing.

    Intensity IS a count rate in cps here, and that is a choice of unit rather than
    a physical claim: k_c = R_standard / I_calibrant, so scaling every intensity --
    sample, solvent AND calibrant -- by any positive factor leaves the excess
    Rayleigh ratio, and therefore Mw, exactly unchanged. Emitting the calibrant
    trace is what makes the convention self-evidencing: the calibrant intensity can
    be read off a trace like every other intensity in the set.
    """
    dls: Dict[float, MultiAngleDLS]                 # concentration -> set (incl. 0.0)
    sls: SyntheticSLSSet                            # the SAME intensities, statically
    calibrant_trace: Optional[SyntheticTrace]
    # ground truth
    mw: float
    rg_nm: float
    a2_mol_mL_per_g2: float
    rh_nm: float
    kd_mL_g: float
    angles_deg: List[float]
    concentrations_g_per_mL: List[float]
    static_intensity_cps: Dict[Tuple[float, float], float]   # (c, angle) -> truth
    cal: CalibrationSpec
    dn_dc: float
    # honesty / diagnostics
    exact_mean: bool
    trace_cv: float
    n_trace_points: int
    max_count_rate_cps: float


def with_intensity_matched_traces(
        dls: MultiAngleDLS, intensity_cps: Dict[float, float], *,
        duration_s: float = 60.0, dt_s: float = 0.05, trace_cv: float = 0.03,
        drift: float = 0.0, spikes: int = 0, exact_mean: bool = True,
        seed: Optional[int] = None) -> MultiAngleDLS:
    """A copy of `dls` whose per-angle trace means ARE `intensity_cps`.

    Additive by design: `build_multi_angle_dls` is called, not edited, and the set
    it returns is replaced rather than mutated. Its own trace stream is discarded,
    which cannot disturb the correlograms it already produced (its trace RNG is
    constructed after the correlogram loop). Unifying the two builders' noise loops
    would reorder their draws and move every committed synthetic trace (a known,
    deliberately unresolved asymmetry).

    The per-angle trace comes from the SHARED `build_count_rate_trace`, so this path
    cannot drift from the standalone one the way the multi-angle builder's inline
    copy did. With ``exact_mean`` the finished array is rescaled so its mean is the
    target to float rounding; the scatter, drift and spikes survive the rescale
    because it is a positive factor. Without it the mean carries the honest sampling
    error ``trace_cv / sqrt(n)``.

    NOTE ``drift`` with ``exact_mean=False`` biases the mean by ``drift/2`` -- a
    coherent multiplicative error on the derived intensity that no tolerance derived
    from ``trace_cv / sqrt(n)`` covers. Use drift with ``exact_mean=True``.
    """
    times = None
    trace_cps: Dict[float, np.ndarray] = {}
    mean_cr: Dict[float, float] = {}
    for angle in dls.angles_deg:
        target = float(intensity_cps[angle])
        if not np.isfinite(target) or target <= 0.0:
            raise ValueError(
                'with_intensity_matched_traces: the target intensity at '
                '{0:g} deg is {1!r} cps, which cannot be a count rate.'.format(
                    angle, target))
        trace = build_count_rate_trace(
            duration_s=duration_s, dt_s=dt_s, mean_kHz=target / 1000.0,
            trace_cv=trace_cv, drift=drift, spikes=spikes,
            seed=(None if seed is None else seed + int(round(100.0 * angle))))
        cps = np.asarray(trace.count_rates_cps, dtype=float)
        if exact_mean:
            # No guard on the divisor: build_count_rate_trace ends in
            # np.clip(cr, 1.0, None) * 1000.0, so every sample it returns is at
            # least 1000 cps and the mean is structurally positive. A predicate
            # here would be a second boundary policing a gap that does not exist
            # (CLAUDE.md rule 12) -- and one no test could drive. If that clip
            # ever goes, this divide is where it will be felt.
            cps = cps * (target / float(cps.mean()))
        times = np.asarray(trace.times_s, dtype=float)
        trace_cps[angle] = cps
        # Recomputed from the array that ships, never from the target: the ALV
        # header must describe the block written beside it.
        mean_cr[angle] = float(cps.mean() / 1000.0)
    return replace(dls, trace_times_s=times, trace_cps=trace_cps,
                   mean_cr_kHz=mean_cr)


def build_consistency_tuple(
        *, mw: float, rg_nm: float, a2_mol_mL_per_g2: float, rh_nm: float,
        angles_deg: Sequence[float], concentrations_g_per_mL: Sequence[float],
        wavelength_nm: float = DEFAULT_WAVELENGTH_NM,
        temperature_K: float = DEFAULT_TEMPERATURE_K,
        n_solvent: float = DEFAULT_N_SOLVENT,
        viscosity_Pa_s: float = DEFAULT_VISCOSITY_PA_S,
        dn_dc: float = DEFAULT_DN_DC,
        cal: Optional[CalibrationSpec] = None,
        solvent_intensity_90_cps: float = DEFAULT_SOLVENT_INTENSITY_90,
        kd_mL_g: float = DEFAULT_KD_ML_G,
        populations_at_angle=None,
        solvent_populations: Optional[Sequence] = None,
        form_factor_model: str = 'guinier',
        beta: float = 0.85, noise_level: float = 0.0, n_points: int = 200,
        trace_duration_s: float = 60.0, trace_dt_s: float = 0.05,
        trace_cv: float = 0.03, drift: float = 0.0, spikes: int = 0,
        exact_mean: bool = True, include_calibrant_trace: bool = True,
        polymer_name: str = '', solvent_name: str = '',
        label: str = 'Consistency tuple',
        seed: Optional[int] = None, **axis_kw) -> ConsistencyTuple:
    """Correlograms, traces and static intensities from one ground truth.

    The static set is built ONCE, by `build_sls_set`, and the traces are then made
    to have those intensities as their means. Nothing recomputes `sample_intensity`,
    so there is no second copy of the truth to disagree with the first.

    A note on the recipe: with the default calibrant and solvent levels, a PEG 1M or
    3M ladder emits count rates above the ~1.5 Mcps ceiling a real photon counter
    reaches, so those recipes generate traces no detector could produce. PEG 300k
    (mw=3e5, rg=27, a2=1.2e-4) keeps every point inside the ordinary working window.
    `max_count_rate_cps` on the returned record makes any excursion visible without
    refusing it -- flags, not filters.
    """
    cal = cal or CalibrationSpec(wavelength_nm=wavelength_nm)
    sls = build_sls_set(
        mw=mw, rg_nm=rg_nm, a2_mol_mL_per_g2=a2_mol_mL_per_g2,
        angles_deg=angles_deg, concentrations_g_per_mL=concentrations_g_per_mL,
        wavelength_nm=wavelength_nm, temperature_K=temperature_K,
        n_solvent=n_solvent, dn_dc=dn_dc, cal=cal,
        solvent_intensity_90=solvent_intensity_90_cps,
        noise_level=0.0, seed=seed, polymer_name=polymer_name,
        solvent_name=solvent_name, label=label, kind='zimm',
        form_factor_model=form_factor_model)

    angles = [float(a) for a in sls.angles_deg]
    truth: Dict[Tuple[float, float], float] = {}
    for conc in sls.concentrations_g_per_mL:
        for i, angle in enumerate(angles):
            truth[(float(conc), angle)] = float(sls.intensities[conc][i])

    if solvent_populations is None:
        # Bookkeeping, not a size claim: the blank must be loadable as a run, and a
        # correlogram needs some population. Override it to plant a real solvent
        # mode (the neat-reline mesoscale slow mode is the motivating case).
        solvent_populations = [SyntheticPopulation(rh_nm=1.0, weight=1.0)]

    def _pops(conc: float):
        if conc == 0.0:
            return solvent_populations
        if populations_at_angle is not None:
            return (lambda a: populations_at_angle(conc, a))
        return [SyntheticPopulation(
            rh_nm=rh_at_concentration(rh_nm, conc, kd_mL_g), weight=1.0)]

    dls: Dict[float, MultiAngleDLS] = {}
    for conc in sls.concentrations_g_per_mL:
        conc = float(conc)
        # Per-POINT seeding, not one stream walked over the grid: adding an angle or
        # a concentration later must not move the noise of the others -- for the
        # correlograms AND the traces (the bands are the module docstring's).
        k = int(round(conc * 1e6))
        corr_seed = None if seed is None else seed + 1_000_000_000 + k * 100_000
        point_seed = None if seed is None else seed + 2_000_000_000 + k * 100_000
        built = build_multi_angle_dls(
            _pops(conc), angles_deg=angles, wavelength_nm=wavelength_nm,
            solvent_refractive_index=n_solvent, temperature_K=temperature_K,
            viscosity_Pa_s=viscosity_Pa_s, beta=beta, noise_level=noise_level,
            n_points=n_points, label='{0} c={1:g}'.format(label, conc),
            concentration_g_per_mL=conc, seed=corr_seed, **axis_kw)
        dls[conc] = with_intensity_matched_traces(
            built, {a: truth[(conc, a)] for a in angles},
            duration_s=trace_duration_s, dt_s=trace_dt_s, trace_cv=trace_cv,
            drift=drift, spikes=spikes, exact_mean=exact_mean, seed=point_seed)

    calibrant_trace = None
    if include_calibrant_trace:
        t = build_count_rate_trace(
            duration_s=trace_duration_s, dt_s=trace_dt_s,
            mean_kHz=cal.calibrant_intensity / 1000.0, trace_cv=trace_cv,
            seed=(None if seed is None else seed + 30000),
            label='calibrant (toluene)')
        cps = np.asarray(t.count_rates_cps, dtype=float)
        if exact_mean and cps.mean() > 0:
            cps = cps * (cal.calibrant_intensity / float(cps.mean()))
        calibrant_trace = SyntheticTrace(times_s=np.asarray(t.times_s, dtype=float),
                                         count_rates_cps=cps, label=t.label)

    n_points_trace = int(next(iter(dls.values())).trace_times_s.size) if dls else 0
    return ConsistencyTuple(
        dls=dls, sls=sls, calibrant_trace=calibrant_trace,
        mw=mw, rg_nm=rg_nm, a2_mol_mL_per_g2=a2_mol_mL_per_g2, rh_nm=rh_nm,
        kd_mL_g=kd_mL_g, angles_deg=angles,
        concentrations_g_per_mL=[float(c) for c in sls.concentrations_g_per_mL],
        static_intensity_cps=truth, cal=cal, dn_dc=dn_dc,
        exact_mean=exact_mean, trace_cv=trace_cv,
        n_trace_points=n_points_trace,
        max_count_rate_cps=(max(truth.values()) if truth else 0.0))


@dataclass
class ModeResolvedTuple:
    """Two-mode correlograms, matched traces and per-angle static intensities
    from ONE ground truth -- the mode-resolved scattering oracle.

    `ConsistencyTuple`'s sibling, not an extension of it: that record's truth
    block is Zimm-shaped (one population's mw/rg/a2), while this one's truth is
    the PARTITION -- per-angle per-mode excess Rayleigh ratios whose fractions
    are the DLS amplitudes by construction. The static model is two-component:

      * the FAST mode's excess is angle-flat, dR_f(theta) = const (Sedlak's own
        phenomenology, and the flatness diagnostic's positive case);
      * the SLOW mode's follows the Guinier-type form
        dR_s(theta) = dR_s0 / (1 + Rg_s^2 q^2 / 3), so the per-angle amplitude
        fractions are q-DEPENDENT and carry exactly the known `slow_rg_nm` --
        the amplitude-Guinier fit recovers it in closed form at zero noise
        (1/dR_s is linear in q^2).

    The mode fractions are domain-free: the k_c sin(theta) (n/n_std)^2 factor
    converting excess counts to dR is shared by both modes at one angle, so
    A_i on raw excess counts equals A_i on Rayleigh ratios -- which is what
    lets one truth serve the DLS amplitudes AND the static partition.

    The slow mode's DYNAMICS are an independent truth choice (a domain-like
    scatterer can relax q-independently while its static envelope carries a
    size): by default q-independent, crossing the fast branch at 82 degrees --
    the association-breaking configuration stays the tested one.

    `dls` is keyed by concentration and includes 0.0 (the solvent blank), so
    injecting it lands the blank in the reference slot through the ordinary
    c = 0 regrouping. Traces are intensity-matched exactly as in
    `ConsistencyTuple` (same unit convention: intensity IS a count rate in cps;
    the calibrant trace makes the convention self-evidencing).
    """
    dls: Dict[float, MultiAngleDLS]                 # {0.0: blank, c: two-mode}
    sls: SyntheticSLSSet                            # the SAME intensities, statically
    calibrant_trace: Optional[SyntheticTrace]
    # per-angle ground truth (angle -> value)
    a_fast: Dict[float, float]                      # intensity fractions (sum to 1)
    a_slow: Dict[float, float]
    dr_fast_cm_inv: Dict[float, float]              # per-mode excess Rayleigh truth
    dr_slow_cm_inv: Dict[float, float]
    dr_total_cm_inv: Dict[float, float]             # = fast + slow by construction
    static_intensity_cps: Dict[Tuple[float, float], float]   # (c, angle) -> counts
    gamma_fast_s_inv: Dict[float, float]
    gamma_slow_s_inv: Dict[float, float]
    # scalar truth
    slow_rg_nm: float                               # what amplitude-Guinier recovers
    a_fast_q0: float                                # fast fraction at q -> 0
    rh_fast_nm: float
    d_fast_m2_s: float
    slow_kind: str                                  # 'q_independent' | 'diffusive'
    rh_slow_dynamic_nm: float                       # NaN for a q-independent slow mode
    crossing_angle_deg: float                       # NaN when the modes never cross
    concentration_g_per_mL: float
    angles_deg: List[float]
    cal: CalibrationSpec
    dn_dc: float
    # honesty / diagnostics
    exact_mean: bool
    trace_cv: float
    max_count_rate_cps: float


def build_mode_resolved_tuple(
        *, slow_rg_nm: float = 120.0, a_fast_q0: float = 0.3,
        excess_cps_90: float = 2.4e5,
        rh_fast_nm: float = 5.0,
        gamma_slow_s_inv: Optional[float] = None,
        rh_slow_nm: Optional[float] = None,
        crossing_angle_deg: Optional[float] = None,
        concentration_g_per_mL: float = 0.005,
        angles_deg: Sequence[float] = tuple(DEFAULT_DLS_ANGLES),
        wavelength_nm: float = DEFAULT_WAVELENGTH_NM,
        temperature_K: float = DEFAULT_TEMPERATURE_K,
        n_solvent: float = DEFAULT_N_SOLVENT,
        viscosity_Pa_s: float = DEFAULT_VISCOSITY_PA_S,
        dn_dc: float = DEFAULT_DN_DC,
        cal: Optional[CalibrationSpec] = None,
        solvent_intensity_90_cps: float = DEFAULT_SOLVENT_INTENSITY_90,
        beta: float = 0.85, noise_level: float = 0.0, n_points: int = 200,
        trace_duration_s: float = 60.0, trace_dt_s: float = 0.05,
        trace_cv: float = 0.03, exact_mean: bool = True,
        include_calibrant_trace: bool = True,
        polymer_name: str = '', solvent_name: str = '',
        label: str = 'Mode-resolved tuple',
        seed: Optional[int] = None, **axis_kw) -> ModeResolvedTuple:
    """Build the linked two-mode set with its partition ground truth.

    The static truth is set by three knobs: `slow_rg_nm` (the slow mode's
    Guinier size), `a_fast_q0` (the fast mode's intensity fraction at q -> 0)
    and `excess_cps_90` (the total excess count rate at 90 degrees, fixing the
    absolute scale). The slow mode's DYNAMICS mirror `build_two_mode_dls`'s
    contract: at most one of `gamma_slow_s_inv` / `rh_slow_nm` /
    `crossing_angle_deg`; with none given, q-independent crossing at 82
    degrees (the Provencher & Stepanek configuration -- rank-based
    association's breaking case stays the tested one).

    Deliberately a SIBLING of `build_consistency_tuple` sharing its low-level
    blocks: the two-component static model above is not expressible through
    `build_sls_set`'s single (mw, rg, a2) form, and forcing mode truth into
    `ConsistencyTuple` as Optionals would make every consumer None-check.
    """
    if not (math.isfinite(a_fast_q0) and 0.0 < a_fast_q0 < 1.0):
        raise ValueError(
            f"a_fast_q0 must be a fraction strictly between 0 and 1, got "
            f"{a_fast_q0!r}.")
    if not (math.isfinite(slow_rg_nm) and slow_rg_nm > 0):
        raise ValueError(
            f"slow_rg_nm must be a positive finite size, got {slow_rg_nm!r}.")
    if not (math.isfinite(excess_cps_90) and excess_cps_90 > 0):
        raise ValueError(
            f"excess_cps_90 must be a positive finite count rate, got "
            f"{excess_cps_90!r}.")
    n_specs = sum(v is not None
                  for v in (gamma_slow_s_inv, rh_slow_nm, crossing_angle_deg))
    if n_specs > 1:
        raise ValueError(
            "Give at most one of gamma_slow_s_inv, rh_slow_nm and "
            "crossing_angle_deg -- they are three ways to specify the ONE "
            "slow mode's dynamics."
        )

    cal = cal or CalibrationSpec(wavelength_nm=wavelength_nm)
    k_c = cal.k_c()
    ri2 = (n_solvent / cal.n_standard) ** 2
    angles = [float(a) for a in angles_deg]
    conc = float(concentration_g_per_mL)

    # --- static truth in the dR domain -------------------------------------
    # dR_f = F (flat); dR_s(theta) = S * P_s(theta), P_s = 1/(1 + Rg^2 q^2/3).
    # F/(F+S) = a_fast_q0 at q -> 0 fixes the ratio; the total excess counts at
    # 90 deg fix the scale (counts = dR / (k_c sin(theta) ri2), the
    # sample_intensity inversion).
    def _p_slow(angle: float) -> float:
        q = phys.scattering_vector_q(angle, wavelength_nm, n_solvent)   # nm^-1
        return 1.0 / (1.0 + slow_rg_nm * slow_rg_nm * q * q / 3.0)

    ratio_f_over_s = a_fast_q0 / (1.0 - a_fast_q0)          # F / S
    s90 = math.sin(math.radians(90.0))
    denom = (ratio_f_over_s + _p_slow(90.0))
    S = excess_cps_90 * k_c * s90 * ri2 / denom
    F = S * ratio_f_over_s

    dr_fast = {a: F for a in angles}
    dr_slow = {a: S * _p_slow(a) for a in angles}
    dr_total = {a: dr_fast[a] + dr_slow[a] for a in angles}
    a_fast_map = {a: dr_fast[a] / dr_total[a] for a in angles}
    a_slow_map = {a: dr_slow[a] / dr_total[a] for a in angles}

    truth: Dict[Tuple[float, float], float] = {}
    for a in angles:
        base = solvent_intensity(a, solvent_intensity_90_cps)
        s = math.sin(math.radians(a))
        truth[(0.0, a)] = base
        truth[(conc, a)] = base + dr_total[a] / (k_c * s * ri2)

    intensities = {
        0.0: np.array([truth[(0.0, a)] for a in angles], dtype=float),
        conc: np.array([truth[(conc, a)] for a in angles], dtype=float),
    }
    sls = SyntheticSLSSet(
        angles_deg=np.asarray(angles, dtype=float),
        concentrations_g_per_mL=[0.0, conc], intensities=intensities,
        label=label, polymer_name=polymer_name, solvent_name=solvent_name,
        temperature_K=temperature_K, wavelength_nm=wavelength_nm,
        n_solvent=n_solvent, dn_dc=dn_dc, cal=cal,
        kind='single_concentration')

    # --- dynamic truth (mirrors build_two_mode_dls's slow-mode contract) ----
    d_fast = phys.stokes_einstein_diffusion_coefficient(
        rh_fast_nm * 1e-9, temperature_K, viscosity_Pa_s)

    def _gamma_at(angle: float, d: float) -> float:
        q = phys.scattering_vector_q_m(angle, wavelength_nm, n_solvent)
        return d * (q * q)               # `*`, not `**` (the Python-scalar trap)

    if rh_slow_nm is not None:
        slow_kind = 'diffusive'
        d_slow = phys.stokes_einstein_diffusion_coefficient(
            rh_slow_nm * 1e-9, temperature_K, viscosity_Pa_s)
        gamma_slow = {a: _gamma_at(a, d_slow) for a in angles}
        crossing = float('nan')

        def _slow_rh_at(_angle: float) -> float:
            return float(rh_slow_nm)
    else:
        slow_kind = 'q_independent'
        if gamma_slow_s_inv is None:
            theta_c = 82.0 if crossing_angle_deg is None else float(crossing_angle_deg)
            q_c = phys.scattering_vector_q_m(theta_c, wavelength_nm, n_solvent)
            gamma_slow_s_inv = d_fast * q_c * q_c
        gamma_slow_s_inv = float(gamma_slow_s_inv)
        rh_slow_nm = float('nan')
        gamma_slow = {a: gamma_slow_s_inv for a in angles}
        sin_half = (math.sqrt(gamma_slow_s_inv / d_fast)
                    * (wavelength_nm * 1e-9) / (4.0 * math.pi * n_solvent))
        crossing = (2.0 * math.degrees(math.asin(sin_half))
                    if 0.0 < sin_half <= 1.0 else float('nan'))

        def _slow_rh_at(angle: float) -> float:
            # the effective-Rh bookkeeping of a constant rate (the
            # two_mode_populations trick); never a size claim
            q = phys.scattering_vector_q_m(angle, wavelength_nm, n_solvent)
            d_eff = gamma_slow_s_inv / (q * q)
            return phys.stokes_einstein_rh(
                d_eff, temperature_K, viscosity_Pa_s) * 1e9

    # Per-angle populations with PER-ANGLE weights -- the q-dependent slow
    # amplitude is this builder's whole point, so the constant-weight
    # `two_mode_populations` is not reused (it stays untouched).
    def _pops(angle: float) -> List[SyntheticPopulation]:
        return [SyntheticPopulation(rh_nm=rh_fast_nm, weight=a_fast_map[angle]),
                SyntheticPopulation(rh_nm=_slow_rh_at(angle),
                                    weight=a_slow_map[angle])]

    # The blank is an independent measurement: its correlograms take the module's
    # second-run band, never the sample's seed (sample-minus-blank must not cancel the
    # noise to zero). Traces follow the per-concentration band, k = 0 for the blank.
    k = int(round(conc * 1e6))
    dls: Dict[float, MultiAngleDLS] = {}
    built = build_multi_angle_dls(
        _pops, angles_deg=angles, wavelength_nm=wavelength_nm,
        solvent_refractive_index=n_solvent, temperature_K=temperature_K,
        viscosity_Pa_s=viscosity_Pa_s, beta=beta, noise_level=noise_level,
        n_points=n_points, label='{0} c={1:g}'.format(label, conc),
        concentration_g_per_mL=conc, seed=seed, **axis_kw)
    dls[conc] = with_intensity_matched_traces(
        built, {a: truth[(conc, a)] for a in angles},
        duration_s=trace_duration_s, dt_s=trace_dt_s, trace_cv=trace_cv,
        exact_mean=exact_mean,
        seed=(None if seed is None else seed + 2_000_000_000 + k * 100_000))
    blank = build_multi_angle_dls(
        # bookkeeping population: the blank must be loadable as a run
        [SyntheticPopulation(rh_nm=1.0, weight=1.0)],
        angles_deg=angles, wavelength_nm=wavelength_nm,
        solvent_refractive_index=n_solvent, temperature_K=temperature_K,
        viscosity_Pa_s=viscosity_Pa_s, beta=beta, noise_level=noise_level,
        n_points=n_points, label='{0} blank'.format(label),
        concentration_g_per_mL=0.0,
        seed=(None if seed is None else seed + 500_000), **axis_kw)
    dls[0.0] = with_intensity_matched_traces(
        blank, {a: truth[(0.0, a)] for a in angles},
        duration_s=trace_duration_s, dt_s=trace_dt_s, trace_cv=trace_cv,
        exact_mean=exact_mean,
        seed=(None if seed is None else seed + 2_000_000_000))

    calibrant_trace = None
    if include_calibrant_trace:
        t = build_count_rate_trace(
            duration_s=trace_duration_s, dt_s=trace_dt_s,
            mean_kHz=cal.calibrant_intensity / 1000.0, trace_cv=trace_cv,
            seed=(None if seed is None else seed + 30000),
            label='calibrant (toluene)')
        cps = np.asarray(t.count_rates_cps, dtype=float)
        if exact_mean and cps.mean() > 0:
            cps = cps * (cal.calibrant_intensity / float(cps.mean()))
        calibrant_trace = SyntheticTrace(
            times_s=np.asarray(t.times_s, dtype=float),
            count_rates_cps=cps, label=t.label)

    return ModeResolvedTuple(
        dls=dls, sls=sls, calibrant_trace=calibrant_trace,
        a_fast=a_fast_map, a_slow=a_slow_map,
        dr_fast_cm_inv=dr_fast, dr_slow_cm_inv=dr_slow,
        dr_total_cm_inv=dr_total, static_intensity_cps=truth,
        gamma_fast_s_inv={a: _gamma_at(a, d_fast) for a in angles},
        gamma_slow_s_inv=gamma_slow,
        slow_rg_nm=float(slow_rg_nm), a_fast_q0=float(a_fast_q0),
        rh_fast_nm=float(rh_fast_nm), d_fast_m2_s=float(d_fast),
        slow_kind=slow_kind, rh_slow_dynamic_nm=float(rh_slow_nm),
        crossing_angle_deg=float(crossing),
        concentration_g_per_mL=conc, angles_deg=angles, cal=cal, dn_dc=dn_dc,
        exact_mean=exact_mean, trace_cv=trace_cv,
        max_count_rate_cps=(max(truth.values()) if truth else 0.0))


def build_sls_set(*, mw: float, rg_nm: float, a2_mol_mL_per_g2: float,
                  angles_deg: Sequence[float], concentrations_g_per_mL: Sequence[float],
                  wavelength_nm: float, temperature_K: float, n_solvent: float,
                  dn_dc: float, cal: CalibrationSpec, solvent_intensity_90: float,
                  noise_level: float = 0.0, seed: Optional[int] = None,
                  polymer_name: str = '', solvent_name: str = '', label: str = '',
                  kind: str = 'zimm',
                  form_factor_model: str = 'guinier',
                  ellipsoid_axial_ratio: float = 1.0,
                  m: Optional[float] = None,
                  r_core_nm: Optional[float] = None, r_outer_nm: Optional[float] = None,
                  m1: Optional[float] = None, m2: Optional[float] = None,
                  dn_dc_in_files: bool = True) -> SyntheticSLSSet:
    """Build an SLS intensity set (a Zimm grid or a single-angle/-concentration slice).

    The solvent reference (c = 0) is always included as the first concentration; it
    is what the analysis subtracts to form the excess Rayleigh ratio.

    ``dn_dc_in_files=False`` builds a dn/dc-LESS set: the intensities still use
    ``dn_dc`` (the physics needs a contrast), but the written file and an injected
    measurement carry none, so the analysis takes the K = 1 arbitrary-scale path.

    `form_factor_model` selects the angular model of the generated intensities:
    'guinier' (default, the small-qRg Zimm expansion) or the full form factor
    'sphere' / 'debye' / 'ellipsoid' / 'mie' / 'core_shell' — the latter emitting genuine
    large-qRg (or, for Mie/core-shell, large-x/high-contrast) curvature that the Form factor P(q)
    fit recovers and Zimm/Berry cannot (see `excess_rayleigh`). `ellipsoid_axial_ratio` (eps =
    polar/equatorial) applies to 'ellipsoid'; `m` (= n_solute/n_solvent) is REQUIRED for 'mie';
    the core/outer radii `r_core_nm`/`r_outer_nm` and contrasts `m1`/`m2` (= n_core/n_solvent,
    n_shell/n_solvent) are REQUIRED for 'core_shell'.
    """
    k_c = cal.k_c()
    angles = np.asarray([float(a) for a in angles_deg], dtype=float)
    concs = sorted({float(c) for c in concentrations_g_per_mL})
    if not any(c == 0.0 for c in concs):
        concs = [0.0] + concs
    intensities: Dict[float, np.ndarray] = {}
    for c in concs:
        I = np.array([sample_intensity(
            a, c, mw, rg_nm, a2_mol_mL_per_g2, wavelength_nm=wavelength_nm,
            n_solvent=n_solvent, dn_dc=dn_dc, k_c=k_c, n_standard=cal.n_standard,
            i_ref_90=solvent_intensity_90,
            form_factor_model=form_factor_model,
            ellipsoid_axial_ratio=ellipsoid_axial_ratio, m=m,
            r_core_nm=r_core_nm, r_outer_nm=r_outer_nm, m1=m1, m2=m2)
            for a in angles], dtype=float)
        if noise_level and c > 0:
            # Per-POINT seeding (the trace path's rule): one stream per concentration,
            # keyed on the concentration value, so editing the ladder never moves the
            # noise of the points already on it.
            rng = np.random.default_rng(
                None if seed is None else seed + 1_000_000_000 + int(round(c * 1e6)) * 100_000)
            I = I * (1.0 + noise_level * rng.standard_normal(I.size))
        intensities[c] = np.clip(I, 1.0, None)
    # A Mie set records the solute RI (n_solute = m * n_solvent); a core-shell set records the
    # CORE (m1) and SHELL (m2) indices -- so an injected artifact is fittable end-to-end. None
    # for every other model.
    solute_ri = shell_ri = None
    if form_factor_model == 'mie' and m is not None:
        solute_ri = float(m) * n_solvent
    elif form_factor_model == 'core_shell' and m1 is not None and m2 is not None:
        solute_ri = float(m1) * n_solvent
        shell_ri = float(m2) * n_solvent
    return SyntheticSLSSet(
        angles_deg=angles, concentrations_g_per_mL=concs, intensities=intensities,
        label=label, polymer_name=polymer_name, solvent_name=solvent_name,
        temperature_K=temperature_K, wavelength_nm=wavelength_nm, n_solvent=n_solvent,
        dn_dc=dn_dc, cal=cal, kind=kind, solute_refractive_index=solute_ri,
        shell_refractive_index=shell_ri, dn_dc_in_files=bool(dn_dc_in_files))


# ===========================================================================
# Depolarized light scattering (DPLS / DDLS) builders
# ===========================================================================

# Static: a VV/VH intensity pair from a chosen depolarization ratio rho_v (vertical incident
# light). rho_v is the ground truth; delta^2 and the Cabannes f are what the static analysis
# must return for it. Their closed forms (Chu, *Laser Light Scattering* 2nd ed. 1991) are
# replicated in `_depolarized_derived` rather than imported, deliberately: it keeps this
# module's no-analysis-engine-import rule and leaves the generate->analyze round trip an
# independent check. (Their home is physics/constants.py, which this file DOES import.)

# Dynamic (DDLS): paired VV + VH multi-angle correlograms for a rod or an ellipsoid. The VV
# decay is q^2 D_t; the VH decay carries the extra rotational term (Zero & Pecora 1982;
# rotational term Pecora 1964). D_t and D_r come from the dimensions via Tirado & Garcia de
# la Torre 1984 / Tanford 1961 + Koenig 1975 (physics.constants). The VH correlogram runs
# through the SAME size-based generator via an angle-dependent EFFECTIVE radius chosen so its
# single exponential has exactly Gamma_VH -- bookkeeping only, never written out.
@dataclass
class DepolarizedPair:
    """A static VV/VH intensity pair (arbitrary units) + its ground-truth anisotropy.

    Only the ratio I_VH / I_VV = rho_v is physical; ``i_vv`` is an arbitrary reference
    level (the SLS Cabannes readout takes both intensities and forms the ratio). There
    is no loadable file for a static pair -- it surfaces as a ground-truth readout the
    user copies into the SLS anisotropy panel.
    """
    i_vv: float
    i_vh: float
    rho_v: float                       # I_VH / I_VV (vertical incident)
    rho_u: float                       # 2 rho_v / (1 + rho_v) (natural-light equiv.)
    optical_anisotropy_sq: float       # delta^2 = 5 rho_v / (3 - 4 rho_v)
    cabannes_isotropic_factor: float   # f = 1 - (4/3) rho_v
    dark_count: float = 0.0


def _depolarized_derived(rho_v: float) -> Tuple[float, float, float]:
    """``(rho_u, delta^2, f_iso)`` for a depolarization ratio -- the ONE home for the
    three closed forms, so the strict builder and the intensity form cannot drift apart.

    Each caller passes the quantity IT is responsible for: the intensity form passes the
    ratio it recovered, the strict builder passes the ratio its guard checked. That is the
    whole point of the split: gating these on a re-derived value while the guard checked a
    different one is rule 12's "guard sized to a different expression than the operation",
    and it silently NaN-s legal input.

    Range-gated, not pole-gated. The poles (rho_v = -1, 3/4) are exact zeros in the two
    denominators, but the NEIGHBOURHOOD is the hazard: past them the closed forms return
    finite, plausible, WRONG numbers (delta^2 goes negative, rho_u comes back positive),
    so both degrade to nan outside their domain rather than reporting nonsense as ground
    truth. Note the ranges differ -- rho_u's is CLOSED at 3/4, delta^2's is half-open.

    The forms and ranges are replicated here rather than imported: their home is
    ``physics/constants.py`` (``_RHO_V_DEPOLARIZATION_LIMIT`` plus the three closed
    forms), and hand-writing them keeps the generate -> analyze round trip an
    independent check instead of a tautology. Note this file DOES import that module as
    ``phys``, so the duplication is a deliberate choice, not an oversight.
    """
    rho_u = (2.0 * rho_v / (1.0 + rho_v)) if 0.0 <= rho_v <= 0.75 else float('nan')
    delta_sq = ((5.0 * rho_v / (3.0 - 4.0 * rho_v))
                if 0.0 <= rho_v < 0.75 else float('nan'))
    return rho_u, delta_sq, 1.0 - (4.0 / 3.0) * rho_v


def build_depolarized_pair_from_intensities(i_vv: float, i_vh: float, *,
                                            dark_count: float = 0.0) -> DepolarizedPair:
    """Static VV/VH pair straight from the two measured intensities.

    The primitive form, and the instrument-faithful one: a photometer reads I_VV and
    I_VH, and rho_v = (I_VH - dark)/(I_VV - dark) falls out of the pair, so there is no
    range left to check. NO physicality check -- the caller owns the physics. rho_v may
    come out negative (I_VH below the dark count: an over-subtracted baseline) or at or
    above the 3/4 depolarization limit (polarizer/analyzer leakage). Both are real
    instrument conditions, and both are generated on request. Only the RATIO is
    physical; ``i_vv`` is an arbitrary reference level, exactly as for the strict
    builder.

    Outside [0, 3/4] rho_u and delta^2 degrade to nan (see ``_depolarized_derived``).
    For FINITE input that is what the static analysis returns for the same pair, so
    ground truth and analysis agree field for field. The Cabannes factor is linear in
    rho_v, so for finite rho_v it is always reported, raw, and may sit outside [0, 1].

    The dark-subtracted I_VV must be POSITIVE and FINITE -- the same requirement
    ``depolarization_ratio`` enforces, and for the same reasons, so the generator and the
    analysis accept exactly the same pairs. Zero leaves rho_v undefined. NEGATIVE means
    the polarized channel read below its own dark count, so both intensities are noise
    and the ratio of two noise floors is not a depolarization (it comes out looking
    healthy: I_VV=1e3, I_VH=1.4e3, dark=2e3 gives rho_v = 0.6, delta^2 = 5.0, no flag).
    Non-finite launders a broken input into a clean answer -- an infinite I_VV gives
    rho_v = vh/inf = 0.0, a perfectly isotropic scatterer with no correction and no flag,
    the exact defect ``depolarization_ratio`` was hardened against (2026-07-31 spec-01).

    "The caller owns the physics" licenses relaxing PHYSICALITY, never arithmetic
    well-definedness. Note only the DENOMINATOR is guarded: a non-finite or
    below-dark I_VH is a real reading (a saturated or dropped VH channel, or the
    over-subtracted baseline this module exists to generate), and the range gates above
    already withhold the derived fields for it.
    """
    vv, vh, dark = float(i_vv), float(i_vh), float(dark_count)
    # One predicate on one computation: `vv - dark` is the quantity the division uses,
    # so guarding anything else would be sized to a different expression (rule 12).
    if not (math.isfinite(vv - dark) and vv - dark > 0.0):
        raise ValueError(
            f"dark-subtracted I_VV must be a positive finite number for rho_v to be "
            f"defined; got i_vv={i_vv!r} with dark_count={dark_count!r}.")
    rho_v = (vh - dark) / (vv - dark)
    rho_u, delta_sq, f_iso = _depolarized_derived(rho_v)
    return DepolarizedPair(i_vv=vv, i_vh=vh, rho_v=rho_v, rho_u=rho_u,
                           optical_anisotropy_sq=delta_sq,
                           cabannes_isotropic_factor=f_iso, dark_count=dark)


def build_depolarized_pair(*, rho_v: float,
                           i_vv_ref: float = DEFAULT_CALIBRANT_INTENSITY,
                           dark_count: float = 0.0) -> DepolarizedPair:
    """Static VV/VH intensity pair from a target depolarization ratio ``rho_v``.

    ``i_vh`` is built so ``(i_vh - dark)/(i_vv - dark)`` recovers ``rho_v``, so the SLS
    Cabannes analysis returns the given ``rho_v`` (and hence delta^2, f). ``rho_v`` must
    be in the physical range [0, 3/4) (delta^2 diverges at 3/4).

    This is the STRICT builder: the range check is the only thing it adds over
    ``build_depolarized_pair_from_intensities``, and it is the property the clean
    corpus's trustworthiness rests on. To build a pair with no range restriction -- a
    leaky polarizer at rho_v >= 3/4, an over-subtracted baseline at rho_v < 0 -- call
    the intensity form and own the physics yourself.

    The derived fields are computed from the CHECKED ``rho_v``, not from the ratio
    re-read off the emitted intensities, and the stored ``rho_v`` is the checked value.
    The two differ: ``i_vh = dark + rho_v*(i_vv - dark)`` rounds the small term to the
    spacing of ``dark``, so reading the ratio back carries an error of about
    ``ulp(dark)/(i_vv - dark)`` -- up to 2e-12 at dark_count = 99998.2, and enough at any
    dark_count to push an ``rho_v`` just under 3/4 over the pole. Gating the closed forms
    on the re-read value therefore NaN-ed input this function had just accepted (round 1,
    finding 1). So a caller who recomputes the ratio from ``i_vv``/``i_vh`` may see the
    last bits differ from ``rho_v``; the stored value is the one that was validated.
    """
    if not (0.0 <= rho_v < 0.75):
        raise ValueError(
            f"depolarization ratio rho_v must be in [0, 3/4); got {rho_v:g}.")
    i_vv = float(i_vv_ref)
    dark = float(dark_count)
    if not (math.isfinite(i_vv - dark) and i_vv - dark > 0.0):
        raise ValueError(
            f"dark-subtracted I_VV must be a positive finite number for rho_v to be "
            f"defined; got i_vv_ref={i_vv_ref!r} with dark_count={dark_count!r}.")
    rho_u, delta_sq, f_iso = _depolarized_derived(float(rho_v))
    return DepolarizedPair(i_vv=i_vv, i_vh=dark + rho_v * (i_vv - dark),
                           rho_v=float(rho_v), rho_u=rho_u,
                           optical_anisotropy_sq=delta_sq,
                           cabannes_isotropic_factor=f_iso, dark_count=dark)


def rod_diffusion(length_m: float, diameter_m: float, temperature_K: float,
                  viscosity_Pa_s: float):
    """Translational D_t (m^2/s), rotational D_r (rad^2/s), and (p, nu, delta_perp) of a
    rigid cylinder. Thin wrapper over physics.constants (Tirado & Garcia de la Torre
    1984). Raises outside the fitted axial-ratio range 2 < p < 30, where the end-effect
    polynomials are unsupported.
    """
    p = length_m / diameter_m
    if not phys.rod_aspect_ratio_valid(p):
        raise ValueError(
            f"axial ratio p = L/d = {p:.3g} is outside the Tirado (1984) validity "
            f"range 2 < p < 30; pick different L, d.")
    nu, delta_perp = phys.rod_end_corrections(p)
    d_t = phys.rod_translational_diffusion(length_m, diameter_m,
                                           temperature_K, viscosity_Pa_s)
    d_r = phys.rod_rotational_diffusion(length_m, diameter_m,
                                        temperature_K, viscosity_Pa_s)
    return d_t, d_r, p, nu, delta_perp


def ellipsoid_diffusion(polar_semi_axis_m: float, equatorial_radius_m: float,
                        temperature_K: float, viscosity_Pa_s: float):
    """Translational D_t (m^2/s), end-over-end rotational D_perp (rad^2/s), and axial
    ratio p = a/b of an ellipsoid of revolution. Thin wrapper over physics.constants
    (D_t Tanford 1961; D_perp Koenig 1975 -- the end-over-end tumbling DDLS observes).
    a = polar (symmetry) semi-axis, b = equatorial radius.
    """
    a, b = polar_semi_axis_m, equatorial_radius_m
    d_t = phys.ellipsoid_translational_diffusion(a, b, temperature_K, viscosity_Pa_s)
    d_r = phys.ellipsoid_rotational_diffusion(a, b, temperature_K, viscosity_Pa_s)
    return d_t, d_r, a / b


def vh_effective_population(d_t: float, d_r: float, *, wavelength_nm: float,
                           n_solvent: float, temperature_K: float,
                           viscosity_Pa_s: float):
    """Return a callable ``angle_deg -> [SyntheticPopulation]`` whose Stokes-Einstein
    decay equals Gamma_VH(angle) = q^2 D_t + 6 D_r, so the generated single exponential
    carries the rotational term. The effective radius is pure bookkeeping (never written
    out); reading it back as a "size" is meaningless -- DDLS works with decay rates.
    """
    def pops(angle_deg: float) -> List[SyntheticPopulation]:
        q = phys.scattering_vector_q_m(angle_deg, wavelength_nm, n_solvent)
        d_eff = d_t + 6.0 * d_r / (q * q)            # so q^2 * d_eff = Gamma_VH
        rh_eff_nm = phys.stokes_einstein_rh(d_eff, temperature_K, viscosity_Pa_s) * 1e9
        return [SyntheticPopulation(rh_nm=rh_eff_nm, weight=1.0)]
    return pops


@dataclass
class DDLSSet:
    """A paired VV + VH multi-angle correlogram set for DDLS (rotational diffusion).

    VV carries only translation (Gamma = q^2 D_t); VH carries the extra 6 D_r rotational
    term. Ground truth (shape, dimensions, D_t, D_r, axial ratio, Rh_t) is stored for the
    readout; the analysis recovers D_r = (Gamma_VH - Gamma_VV)/6 and D_t from the VV
    channel. ``dims_nm`` is (L, d) for a rod and (a, b) for an ellipsoid.
    """
    vv: MultiAngleDLS
    vh: MultiAngleDLS
    shape: str                         # 'rod' | 'ellipsoid'
    dims_nm: Tuple[float, float]
    d_t_m2_s: float
    d_r_rad2_s: float
    axial_ratio: float
    rh_t_nm: float


def build_ddls_set_from_diffusion(d_t_m2_s: float, d_r_rad2_s: float, *,
                                  angles_deg: Sequence[float], wavelength_nm: float,
                                  n_solvent: float, temperature_K: float,
                                  viscosity_Pa_s: float, beta: float = 0.9,
                                  vh_noise_level: float = 0.02, n_points: int = 200,
                                  seed: Optional[int] = None, label: str = 'DDLS',
                                  shape: str = 'unspecified',
                                  dims_nm: Optional[Tuple[float, float]] = None,
                                  axial_ratio: Optional[float] = None,
                                  concentration_g_per_mL: float = 0.0) -> DDLSSet:
    """Paired VV/VH multi-angle set from the two diffusion rates directly.

    The primitive form. DDLS measures RATES -- Gamma_VV = q^2 D_t and
    Gamma_VH = q^2 D_t + 6 D_r -- and every particle model in this module exists only to
    produce a (D_t, D_r) pair. Taking that pair itself generates data for models the
    library does not implement (a wormlike chain, a literature (D_t, D_r) table, a
    different end-correction scheme) and for rate pairs no rigid body produces at all.
    NO physicality check -- the caller owns the physics. D_r may be zero or negative
    (Gamma_VH at or below Gamma_VV, which a weak VH channel yields from noise alone),
    and the pair need not be reproducible by any rod or ellipsoid.

    Two arithmetic requirements survive, and they are Stokes-Einstein's rather than this
    function's: ``d_t_m2_s`` must be positive and finite (the VV channel is built from
    the Stokes radius it implies), and the VH effective radius needs
    d_t + 6 d_r / q^2 > 0 at EVERY requested angle -- so a negative D_r has to stay
    small against q^2 D_t at the LOWEST angle. Either violation raises out of
    ``physics.constants`` at the operation, not from a screen added here.

    ``shape``, ``dims_nm`` and ``axial_ratio`` are carried into the returned ground
    truth as LABELS only; nothing is derived from them. Left unset they read
    'unspecified' and nan, which is the honest answer when the caller started from rates
    and never had a geometry.

    ``concentration_g_per_mL`` is likewise a label -- DDLS reads rates, and nothing here
    depends on c. It is settable because the workspace groups a sample WITHOUT
    concentration (core.workspace.sample_group_key), so two sets at two concentrations
    land in one sample, and generating that pair is the only way to exercise the
    identity refusal in the controller's DDLS averaging. Default 0.0 = the historical
    value, so an existing call's artifact is unchanged.
    """
    dims = ((float('nan'), float('nan')) if dims_nm is None
            else (float(dims_nm[0]), float(dims_nm[1])))
    p = float('nan') if axial_ratio is None else float(axial_ratio)
    d_t, d_r = float(d_t_m2_s), float(d_r_rad2_s)
    rh_t_nm = phys.stokes_einstein_rh(d_t, temperature_K, viscosity_Pa_s) * 1e9
    common = dict(angles_deg=angles_deg, wavelength_nm=wavelength_nm,
                  solvent_refractive_index=n_solvent, temperature_K=temperature_K,
                  viscosity_Pa_s=viscosity_Pa_s, beta=beta, n_points=n_points,
                  concentration_g_per_mL=float(concentration_g_per_mL))
    vv = build_multi_angle_dls(
        [SyntheticPopulation(rh_nm=rh_t_nm, weight=1.0)], noise_level=0.0,
        label=f'{label} VV', seed=seed, **common)
    vh = build_multi_angle_dls(
        vh_effective_population(d_t, d_r, wavelength_nm=wavelength_nm,
                                n_solvent=n_solvent, temperature_K=temperature_K,
                                viscosity_Pa_s=viscosity_Pa_s),
        noise_level=vh_noise_level, label=f'{label} VH',
        seed=(None if seed is None else seed + 500_000), **common)
    return DDLSSet(vv=vv, vh=vh, shape=shape, dims_nm=dims,
                   d_t_m2_s=d_t, d_r_rad2_s=d_r, axial_ratio=p, rh_t_nm=rh_t_nm)


def build_ddls_set(*, shape: str, dims_nm: Tuple[float, float],
                   angles_deg: Sequence[float], wavelength_nm: float, n_solvent: float,
                   temperature_K: float, viscosity_Pa_s: float, beta: float = 0.9,
                   vh_noise_level: float = 0.02, n_points: int = 200,
                   seed: Optional[int] = None, label: str = 'DDLS',
                   concentration_g_per_mL: float = 0.0) -> DDLSSet:
    """Build a paired VV/VH multi-angle correlogram set for a rod or an ellipsoid.

    ``dims_nm`` is (L, d) for ``shape='rod'`` (Tirado) or (a, b) for
    ``shape='ellipsoid'`` (Perrin). The VV set is generated clean (it is the D_t ground
    truth); the VH set carries the rotational term and the (weak-signal) ``vh_noise_level``.

    This is the STRICT builder: the dimensions go through ``rod_diffusion`` /
    ``ellipsoid_diffusion``, so the rod's Tirado-fitted 2 < p < 30 range still raises
    here and the (D_t, D_r) handed on is by construction a real rigid body's. For a set
    built from rates no rod or ellipsoid produces, call
    ``build_ddls_set_from_diffusion`` and own the physics yourself.
    """
    x_nm, y_nm = float(dims_nm[0]), float(dims_nm[1])
    if shape == 'rod':
        d_t, d_r, p, _nu, _dp = rod_diffusion(
            x_nm * 1e-9, y_nm * 1e-9, temperature_K, viscosity_Pa_s)
    elif shape == 'ellipsoid':
        d_t, d_r, p = ellipsoid_diffusion(
            x_nm * 1e-9, y_nm * 1e-9, temperature_K, viscosity_Pa_s)
    else:
        raise ValueError(f"shape must be 'rod' or 'ellipsoid'; got {shape!r}.")
    return build_ddls_set_from_diffusion(
        d_t, d_r, angles_deg=angles_deg, wavelength_nm=wavelength_nm,
        n_solvent=n_solvent, temperature_K=temperature_K,
        viscosity_Pa_s=viscosity_Pa_s, beta=beta, vh_noise_level=vh_noise_level,
        n_points=n_points, seed=seed, label=label,
        shape=shape, dims_nm=(x_nm, y_nm), axial_ratio=p,
        concentration_g_per_mL=concentration_g_per_mL)


@dataclass
class DepolarizedArtifact:
    """The full "Depolarized (VV/VH)" artifact: a static intensity pair (drives the SLS
    Cabannes readout) + a dynamic VV/VH correlogram set (drives DDLS)."""
    pair: DepolarizedPair
    ddls: DDLSSet


def build_depolarized(*, rho_v: float, shape: str, dims_nm: Tuple[float, float],
                      angles_deg: Sequence[float], wavelength_nm: float, n_solvent: float,
                      temperature_K: float, viscosity_Pa_s: float, beta: float = 0.9,
                      vh_noise_level: float = 0.02, n_points: int = 200,
                      i_vv_ref: float = DEFAULT_CALIBRANT_INTENSITY,
                      seed: Optional[int] = None, label: str = 'DDLS',
                      concentration_g_per_mL: float = 0.0) -> DepolarizedArtifact:
    """Compose the static pair + the dynamic DDLS set into one artifact."""
    pair = build_depolarized_pair(rho_v=rho_v, i_vv_ref=i_vv_ref)
    ddls = build_ddls_set(
        shape=shape, dims_nm=dims_nm, angles_deg=angles_deg, wavelength_nm=wavelength_nm,
        n_solvent=n_solvent, temperature_K=temperature_K, viscosity_Pa_s=viscosity_Pa_s,
        beta=beta, vh_noise_level=vh_noise_level, n_points=n_points, seed=seed, label=label,
        concentration_g_per_mL=concentration_g_per_mL)
    return DepolarizedArtifact(pair=pair, ddls=ddls)


# ===========================================================================
# Branching-architecture series builder (Cross-Sample g / h / n feature)
# ===========================================================================
_ARCH_K, _ARCH_NU = 0.05, 0.5             # the exact linear law Rg = k * Mw^nu (nm, g/mol)


def g_star(f: int) -> float:
    """Regular f-arm star contraction factor g = (3f-2)/f^2 (Zimm & Stockmayer 1949 Eq. 15a)."""
    return (3 * f - 2) / f ** 2


def g_random_fraction(n: float, f: int) -> float:
    """Scattering g for a randomly-branched FRACTION of f-functional branch units carrying
    n branch units per molecule (Zimm & Stockmayer 1949 Eqs. 47a/47b = Burchard Eqs. 65/66).
    Only f=3 and f=4 have the tabulated closed form; other functionalities are unsupported."""
    if f == 3:
        return ((1.0 + n / 7.0) ** 0.5 + 4.0 * n / (9.0 * math.pi)) ** -0.5
    if f == 4:
        return ((1.0 + n / 6.0) ** 0.5 + 4.0 * n / (3.0 * math.pi)) ** -0.5
    raise ValueError(f'random-branching fraction g is tabulated only for f=3,4, got {f!r}')


def build_architecture_series() -> Dict[str, dict]:
    """``{name: dict(mw, rg, rh, a2, role, model, f, n, g)}`` -- a homologous LINEAR
    Rg-Mw series (Rg = 0.05 Mw^0.5), two regular stars, and two randomly-branched
    fractions (tri- and tetra-functional), to exercise the Cross-Sample Architecture tab.
    ``model``/``f``/``n`` document the ground-truth architecture behind each Branched
    sample's g; they do not enter the written instrument files. The branched Mw all sit
    inside the linear reference span [1e5, 9e5] so the g -> n/f_eff inversion is anchored.
    """
    def rg_lin(mw):
        return _ARCH_K * mw ** _ARCH_NU

    series: Dict[str, dict] = {}
    for mw, rho in [(1.0e5, 1.5), (4.0e5, 1.5), (9.0e5, 1.5)]:       # linear coil, rho=1.5
        rg = rg_lin(mw)
        series[f'PS linear {mw/1e3:.0f}k'] = dict(
            mw=mw, rg=round(rg, 3), rh=round(rg / rho, 3), a2=1.2e-4,
            role='linear', model=None, f=None, n=None, g=None)
    for mw, f, rho in [(2.0e5, 3, 1.40), (6.0e5, 4, 1.35)]:          # regular stars
        g = g_star(f)
        rg = rg_lin(mw) * math.sqrt(g)
        series[f'PS star {f}-arm'] = dict(
            mw=mw, rg=round(rg, 3), rh=round(rg / rho, 3), a2=1.0e-4,
            role='branched', model='regular_star', f=f, n=None, g=round(g, 4))
    for name, mw, f, n, rho in [
            ('PS random tri', 3.0e5, 3, 6, 1.42),                    # trifunctional, n=6
            ('PS random tetra', 7.0e5, 4, 4, 1.38)]:                 # tetrafunctional, n=4
        g = g_random_fraction(n, f)
        rg = rg_lin(mw) * math.sqrt(g)
        model = 'random_tri' if f == 3 else 'random_tetra'
        series[name] = dict(
            mw=mw, rg=round(rg, 3), rh=round(rg / rho, 3), a2=1.0e-4,
            role='branched', model=model, f=f, n=n, g=round(g, 4))
    return series


# ===========================================================================
# Writers (file I/O — the loadable instrument-file exception)
# ===========================================================================
def write_alv_asc(path: str, dls: MultiAngleDLS, *,
                  temperature_K: Optional[float] = None,
                  viscosity_cp: Optional[float] = None) -> str:
    """Write a MultiAngleDLS as an ALV ``.ASC`` (correlation block + count-rate block).

    Mirrors the ALV multi-angle format the project's ALV parser reads: the lag in
    the Correlation block is in **ms**, the Count Rate block in **kHz**.
    """
    angles = dls.angles_deg
    T_K = float(temperature_K if temperature_K is not None else dls.temperature_K)
    eta_cp = float(viscosity_cp if viscosity_cp is not None
                   else dls.viscosity_Pa_s * 1.0e3)
    delay = dls.delay_times_s
    t = dls.trace_times_s

    L = ['ALV-7012 CGS-12F Data, Synthetic (DLS Buddy test data)',
         'Date :\t"1/1/2026"', 'Time :\t"12:00:00 PM"',
         f'Samplename : \t"{dls.label}, c = {dls.concentration_g_per_mL * 1e3:g} mg/mL"']
    L += [f'SampMemo({i}) : \t""' for i in range(10)]
    L += [f'Temperature [K] :\t{T_K:.5f}', f'Viscosity [cp]  :\t{eta_cp:.5f}',
          f'Refractive Index:\t{dls.n_solvent:.5f}',
          f'Wavelength [nm] :\t{dls.wavelength_nm:.5f}']
    L += [f'Angle({i})[deg]     :\t{a:.5f}' for i, a in enumerate(angles, 1)]
    L += ['Duration [s]    :\t60.0', 'Runs            :\t1',
          'Mode            :\t"synthetic"']
    L += [f'MeanCR{i} [kHz]   :\t{dls.mean_cr_kHz[a]:.5f}' for i, a in enumerate(angles, 1)]
    L += [f'DC[{i}][kHz]      :\t0.00000' for i in range(1, len(angles) + 1)]
    L += ['', '"Correlation"']
    for k in range(delay.size):
        row = [f'{delay[k] * 1000:.5E}'] + [f'{dls.signals[a][k]:.5E}' for a in angles]
        L.append('  ' + '\t'.join(row))
    L.append('"Count Rate"')
    for k in range(t.size):
        row = [f'{t[k]:.5f}'] + [f'{dls.trace_cps[a][k] / 1000:.5f}' for a in angles]
        L.append('  ' + '\t'.join(row))
    with open(path, 'w', encoding='latin-1', newline='\n') as fh:
        fh.write('\n'.join(L) + '\n')
    return path


def write_brookhaven_sls(path: str, sls: SyntheticSLSSet) -> str:
    """Write a SyntheticSLSSet as a Brookhaven SLS ``.csv`` (key,value rows).

    Calibration metadata (constant, standard, Rayleigh) is written only when the
    set is calibrated (``cal.include_in_files``); an uncalibrated set omits it, so
    a user who loads it without entering a calibrant gets the uncalibrated flag. The
    dn/dc row is written only when ``sls.dn_dc_in_files``; a dn/dc-less set omits it,
    so the parser leaves dn/dc None and the analysis runs on the arbitrary optical scale.
    """
    angles = [float(a) for a in sls.angles_deg]
    concs_mg = [c * 1e3 for c in sls.concentrations_g_per_mL]
    rows: List[Tuple[str, str]] = [
        ('Date/Time', '1/1/2026 12:00:00 PM'),
        ('Sample ID', sls.label or f'{sls.polymer_name} in {sls.solvent_name}'),
        ('Operator ID', 'Synthetic'),
        ('Sample Liquid', sls.solvent_name or 'Water'),
        ('Refractive Index of Sample Liquid', f'{sls.n_solvent}'),
    ]
    if sls.dn_dc_in_files:
        rows.append(('Refractive Index Inc. (dn/dc) (mL/g)', f'{sls.dn_dc}'))
    if sls.calibrated:
        rows += [
            ('Calibration Constant', f'{sls.cal.k_c():.4E}'),
            ('Calibration Liquid', 'Toluene'),
            ('Refractive Index of Calibration Liquid', f'{sls.cal.n_standard}'),
            ('Rayleigh Ratio of Calibration Liquid', f'{sls.cal.rayleigh():.4E}'),
        ]
    rows += [('Wavelength (nm)', f'{sls.wavelength_nm:g}'),
             ("'A' Dark Count Rate", '0'),
             ('Number of Angles Measured', str(len(angles)))]
    rows += [(f'Angle {i} (degrees)', f'{a:g}') for i, a in enumerate(angles, 1)]
    rows += [('Number of Concentrations Measured', str(len(concs_mg)))]
    rows += [(f'Concentration {j} (mg/mL)', f'{c:g}') for j, c in enumerate(concs_mg, 1)]
    for j, c in enumerate(sls.concentrations_g_per_mL, 1):
        col = sls.intensities[c]
        for i in range(len(angles)):
            rows.append((f'Intensity - Concentration {j} - Angle {i + 1}', f'{col[i]:.0f}'))
    with open(path, 'w', encoding='latin-1', newline='\n') as fh:
        fh.write('\n'.join(f'{k},{v}' for k, v in rows) + '\n')
    return path


def write_brookhaven_sls_scan(path: str, sls: SyntheticSLSSet,
                              conc_g_per_mL: float) -> str:
    """Write ONE concentration of a SyntheticSLSSet as a Brookhaven angular-scan ``.csv``.

    The instrument's second SLS layout (``BrookhavenSLSScanParser``): a single
    acquisition plus the solvent blank measured in the same run, with no concentration
    table and no dn/dc. The set's ``c = 0`` column becomes ``Liquid Intensity`` and the
    column at ``conc_g_per_mL`` becomes ``Sample Intensity``.

    Deliberately omits the calibration block as well as dn/dc: layout B carries neither,
    so a file that shipped them would exercise a path no real export produces.

    Raises
    ------
    ValueError
        If the set has no ``c = 0`` blank, or no column at ``conc_g_per_mL`` -- either
        way the caller asked for a file that cannot represent this layout.
    """
    if 0.0 not in sls.intensities:
        raise ValueError('layout B needs the set\'s c = 0 blank; this set has none.')
    if conc_g_per_mL not in sls.intensities:
        raise ValueError(
            f'no intensity column at c = {conc_g_per_mL!r} g/mL; '
            f'available: {sorted(sls.intensities)}.')
    angles = [float(a) for a in sls.angles_deg]
    rows: List[Tuple[str, str]] = [
        ('Date/Time', '1/1/2026 12:00:00 PM'),
        ('Sample ID', sls.label or f'{sls.polymer_name} in {sls.solvent_name}'),
        ('Operator ID', 'Synthetic'),
        ('Sample Liquid', sls.solvent_name or 'Water'),
        ('Refractive Index of Sample Liquid', f'{sls.n_solvent}'),
        ('Wavelength (nm)', f'{sls.wavelength_nm:g}'),
        ("'A' Dark Count Rate", '0'),
        ('Number of Angles Measured', str(len(angles))),
    ]
    rows += [(f'Angle {i} (degrees)', f'{a:g}') for i, a in enumerate(angles, 1)]
    blank = sls.intensities[0.0]
    sample = sls.intensities[conc_g_per_mL]
    rows += [(f'Liquid Intensity - Angle {i + 1}', f'{blank[i]:.1f}')
             for i in range(len(angles))]
    rows += [(f'Sample Intensity - Angle {i + 1}', f'{sample[i]:.1f}')
             for i in range(len(angles))]
    with open(path, 'w', encoding='latin-1', newline='\n') as fh:
        fh.write('\n'.join(f'{k},{v}' for k, v in rows) + '\n')
    return path


def write_trace_csv(path: str, trace: SyntheticTrace, *, count_rate_unit: str = 'kcps') -> str:
    """Write a count-rate trace as a plain two-column CSV (time s, count rate).

    The default writes kcps (matching the bundled trace test file); load it via the
    generic trace parser, choosing time = s and the matching count-rate unit.
    """
    factor = {'cps': 1.0, 'kcps': 1e-3, 'Mcps': 1e-6}[count_rate_unit]
    cps = np.asarray(trace.count_rates_cps, dtype=float) * factor
    t = np.asarray(trace.times_s, dtype=float)
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(f'{ti:.2f},{ci:.3f}' for ti, ci in zip(t, cps, strict=True)) + '\n')
    return path


# ===========================================================================
# Full-dataset orchestrator (writes a whole test-data/Synthetic * folder)
# ===========================================================================
def generate_full_dataset(out_dir: str, profile_name: str, *,
                          series: Optional[Dict[str, dict]] = None) -> List[str]:
    """Regenerate a complete synthetic dataset folder (no pictures).

    Writes one SLS Zimm ``.csv`` per series sample, one SLS Guinier ``.csv`` re-emitting
    :data:`DEFAULT_SLS_SCAN_SERIES` in Brookhaven's OTHER layout (the single angular scan
    plus its blank), the DLS ``.ASC`` files (a concentration series for one sample, a
    single concentration for the rest), a bimodal DLS sample, a standalone trace, and
    ``parameters.txt``. Returns the list of file paths written. ``profile_name`` keys into
    :data:`PROFILES`. Picture rendering is added by a separate developer wrapper, which
    may import the plotting layer.
    """
    import os
    prof = PROFILES[profile_name]
    series = series or DEFAULT_SERIES
    rng = np.random.default_rng(prof['seed'])
    os.makedirs(out_dir, exist_ok=True)
    cal = CalibrationSpec()          # default calibrated test system
    written: List[str] = []

    # SLS Zimm sets (one file per sample)
    for name, p in series.items():
        sls = build_sls_set(
            mw=p['mw'], rg_nm=p['rg'], a2_mol_mL_per_g2=p['a2'],
            angles_deg=DEFAULT_SLS_ANGLES, concentrations_g_per_mL=[c * 1e-3 for c in DEFAULT_SLS_CONCS_MG],
            wavelength_nm=DEFAULT_WAVELENGTH_NM, temperature_K=DEFAULT_TEMPERATURE_K,
            n_solvent=DEFAULT_N_SOLVENT, dn_dc=DEFAULT_DN_DC, cal=cal,
            solvent_intensity_90=DEFAULT_SOLVENT_INTENSITY_90, noise_level=prof['sls_noise'],
            seed=rng.integers(1 << 31), polymer_name=name, solvent_name='water',
            label=f'{name} in Water')
        written.append(write_brookhaven_sls(
            os.path.join(out_dir, f'SLS Zimm - {name}.csv'), sls))
        # ... and one series again in the angular-scan layout, so the corpus covers
        # Brookhaven's second SLS format. Written from the SAME built `sls`, so no extra
        # rng.integers draw is taken and every other generated file stays byte-identical.
        if name == DEFAULT_SLS_SCAN_SERIES:
            written.append(write_brookhaven_sls_scan(
                os.path.join(out_dir, f'SLS Guinier - {name}.csv'), sls,
                DEFAULT_SLS_SCAN_MG * 1e-3))

    # DLS: a full concentration series for one sample, one concentration for the rest.
    # One seed per RUN, drawn from the profile stream exactly as the SLS half above
    # draws one per set: every run is an independent measurement (the literal profile
    # seed on every call gave four "different" samples bit-identical traces and a
    # concentration ladder with perfectly correlated correlogram noise).
    for name, p in series.items():
        concs = DEFAULT_DLS_CONCS_MG.get(name, [DEFAULT_DLS_SINGLE_MG])
        for c_mg in concs:
            c = c_mg * 1e-3
            pops = [SyntheticPopulation(rh_nm=rh_at_concentration(p['rh'], c), weight=1.0)]
            dls = build_multi_angle_dls(
                pops, angles_deg=DEFAULT_DLS_ANGLES, wavelength_nm=DEFAULT_WAVELENGTH_NM,
                solvent_refractive_index=DEFAULT_N_SOLVENT, temperature_K=DEFAULT_TEMPERATURE_K,
                viscosity_Pa_s=DEFAULT_VISCOSITY_PA_S, noise_level=prof['corr_noise'],
                trace_cv=prof['trace_cv'], drift=prof['drift'], spikes=prof['spikes'],
                label=f'{name} in Water', concentration_g_per_mL=c,
                seed=int(rng.integers(1 << 31)))
            written.append(write_alv_asc(
                os.path.join(out_dir, f'DLS - {name} - {c_mg:g} mg per mL.ASC'), dls,
                viscosity_cp=DEFAULT_VISCOSITY_CP))

    # bimodal DLS sample (CONTIN / NNLS / double-exp / KWW / peak picking)
    bimodal = build_multi_angle_dls(
        [SyntheticPopulation(rh_nm=20.0, weight=1.0),
         SyntheticPopulation(rh_nm=200.0, weight=2.0)],
        angles_deg=DEFAULT_DLS_ANGLES, wavelength_nm=DEFAULT_WAVELENGTH_NM,
        solvent_refractive_index=DEFAULT_N_SOLVENT, temperature_K=DEFAULT_TEMPERATURE_K,
        viscosity_Pa_s=DEFAULT_VISCOSITY_PA_S, noise_level=prof['corr_noise'],
        trace_cv=prof['trace_cv'], drift=prof['drift'], spikes=prof['spikes'],
        label='Bimodal latex mix in Water', concentration_g_per_mL=DEFAULT_DLS_SINGLE_MG * 1e-3,
        seed=int(rng.integers(1 << 31)))
    written.append(write_alv_asc(
        os.path.join(out_dir, 'DLS - Bimodal 20nm + 200nm.ASC'), bimodal,
        viscosity_cp=DEFAULT_VISCOSITY_CP))

    # standalone two-column trace (generic-trace load path)
    trace = build_count_rate_trace(
        mean_kHz=300.0, trace_cv=prof['trace_cv'], drift=prof['drift'],
        spikes=prof['spikes'], seed=int(rng.integers(1 << 31)))
    written.append(write_trace_csv(
        os.path.join(out_dir, 'Trace - count rate (kcps vs s).csv'), trace))

    written.append(write_parameters_txt(out_dir, profile_name, series))
    return written


def write_parameters_txt(out_dir: str, profile_name: str,
                         series: Optional[Dict[str, dict]] = None) -> str:
    """Write the human-readable parameters.txt (what to enter + ground truth)."""
    import os
    series = series or DEFAULT_SERIES
    messy = 'Messy' in profile_name
    cal = CalibrationSpec()
    lines = [
        f'DLS Buddy — {profile_name} test data', '=' * 50, '',
        'System: PEG in water, calibrated against a toluene standard.', '',
        'GLOBAL PARAMETERS (enter these at the Data-tab confirmation step):',
        '  Solvent                 : water',
        f'  Solvent refractive index: {DEFAULT_N_SOLVENT}',
        f'  Viscosity               : {DEFAULT_VISCOSITY_CP} cP   (= {DEFAULT_VISCOSITY_PA_S:g} Pa.s)',
        f'  Temperature             : {DEFAULT_TEMPERATURE_C} C  ({DEFAULT_TEMPERATURE_K} K)',
        f'  Wavelength              : {DEFAULT_WAVELENGTH_NM:g} nm',
        f'  dn/dc                   : {DEFAULT_DN_DC} mL/g', '',
        'SLS CALIBRATION (enter in the SLS tab calibration panel):',
        f'  Standard                : toluene   (geometry {cal.geometry})',
        f'  Standard refractive index: {cal.n_standard}',
        f'  Calibrant intensity     : {cal.calibrant_intensity:g}  at 90 deg',
        '  Dark count rate         : 0',
        f'  (the program then computes k_c = {cal.k_c():.4e})', '',
        'Notes:',
        '  - DLS files are ALV .ASC: all angles + a count-rate trace are in one file.',
        '    Enter polymer name, solvent (water), and the concentration (in the file',
        '    name) for each. Temperature/viscosity/n/wavelength are read from the file.',
        '  - SLS Zimm files are Brookhaven .csv (a full Zimm set: solvent reference at',
        '    c=0 plus 5 concentrations x 12 angles). Enter polymer name + temperature.',
        "  - The SLS Guinier file is the SAME data in Brookhaven's other .csv layout:",
        '    one angular scan plus the solvent blank measured beside it. It loads as TWO',
        '    measurements. That layout records no concentration and no dn/dc, so enter',
        '    polymer name + temperature AND:',
        f'        concentration = {DEFAULT_SLS_SCAN_MG:g} mg/mL   (for the sample, not the blank)',
        f'        dn/dc         = {DEFAULT_DN_DC} mL/g',
        '  - To pair rho = Rg/Rh for a sample, give its DLS and SLS files the SAME',
        '    polymer name + solvent + temperature so they group into one sample.',
        '  - The two-column trace loads via the Utilities > Traces tab (time = s,',
        '    count rate = kcps).',
    ]
    if messy:
        lines += ['', 'This is the MESSY set: ~2.5% intensity noise, noisier',
                  'correlograms, and traces with slow drift + occasional dust spikes.',
                  'Fits should still work but with more scatter (lower R^2, some',
                  'flagged outliers). Compare to the Clean set.']
    else:
        lines += ['', 'This is the CLEAN set: low noise (good, not perfect). Fits',
                  'should recover the ground-truth values below closely.']
    lines += ['', 'GROUND TRUTH (what the analyses should recover):', '',
              f'  {"Sample":<12}{"Mw (g/mol)":>14}{"Rg (nm)":>10}{"Rh (nm)":>10}{"A2 (mol mL/g^2)":>18}']
    for name, p in series.items():
        lines.append(f'  {name:<12}{p["mw"]:>14.2e}{p["rg"]:>10.1f}{p["rh"]:>10.1f}{p["a2"]:>18.2e}')
    lines += ['',
              '  Bimodal DLS sample: two populations at Rh = 20 nm and 200 nm',
              '    (intensity weights ~1 : 2). Use CONTIN or NNLS to resolve them.', '',
              '  rho = Rg/Rh ~ 1.5 for every series sample (the THETA-solvent linear-coil',
              '    value; a good-solvent coil sits nearer 1.78 -- the series pairs this rho',
              '    with a good-solvent size exponent, and the two are not made consistent).',
              '  Rg-Mw scaling exponent nu ~ 0.585; A2-Mw exponent ~ -0.2.',
              '  DLS shows a mild concentration dependence (kD ~ +30 mL/g), so the',
              '  D-vs-c extrapolation has a small positive slope (PEG 1M has 3 concs).']
    path = os.path.join(out_dir, 'parameters.txt')
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:   # LF, as committed
        fh.write('\n'.join(lines) + '\n')
    return path


# ===========================================================================
# Alpha-selection gap-case dataset (stresses CONTIN's regularization selector)
# ===========================================================================
# Correlograms the Clean/Messy homologous series lacks: a broad/polydisperse unimodal, a
# closely-spaced bimodal, and a well-separated bimodal at three noise levels. They exercise
# CONTIN alpha selection (GCV / L-curve / F-test) on the hard inversion cases -- a broad mode
# that legitimately needs heavier smoothing, and noisy bimodals a poor selector over-smooths
# into a single peak. (name, populations, noise level, seed, sample label.)
ALPHA_SELECTION_CASES = [
    ('DLS - Broad unimodal (Rh 50nm, CV 0.5)',
     [SyntheticPopulation(rh_nm=50.0, weight=1.0, spread_cv=0.5)],
     0.003, 101, 'Broad unimodal in Water'),
    ('DLS - Close bimodal (40nm + 90nm)',
     [SyntheticPopulation(rh_nm=40.0, weight=1.0),
      SyntheticPopulation(rh_nm=90.0, weight=1.0)],
     0.003, 102, 'Close bimodal in Water'),
    ('DLS - Wide bimodal (30nm + 200nm) noise 0.001',
     [SyntheticPopulation(rh_nm=30.0, weight=1.0),
      SyntheticPopulation(rh_nm=200.0, weight=1.0)],
     0.001, 111, 'Wide bimodal in Water, noise 0.001'),
    ('DLS - Wide bimodal (30nm + 200nm) noise 0.003',
     [SyntheticPopulation(rh_nm=30.0, weight=1.0),
      SyntheticPopulation(rh_nm=200.0, weight=1.0)],
     0.003, 113, 'Wide bimodal in Water, noise 0.003'),
    ('DLS - Wide bimodal (30nm + 200nm) noise 0.010',
     [SyntheticPopulation(rh_nm=30.0, weight=1.0),
      SyntheticPopulation(rh_nm=200.0, weight=1.0)],
     0.010, 120, 'Wide bimodal in Water, noise 0.010'),
]


def generate_alpha_selection_dataset(out_dir: str) -> List[str]:
    """Regenerate the alpha-selection gap-case set (test-data/Synthetic/Alpha-Selection/).

    Writes one loadable ALV ``.ASC`` per :data:`ALPHA_SELECTION_CASES` entry (same
    format and system as the Clean/Messy sets: PEG-in-water optics at 532 nm, all
    angles + a count-rate trace per file) plus a ``parameters.txt``. Fixed seeds, so
    re-running reproduces the same bytes. Returns the file paths written.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    written: List[str] = []
    for stem, pops, noise, seed, label in ALPHA_SELECTION_CASES:
        dls = build_multi_angle_dls(
            pops, angles_deg=DEFAULT_DLS_ANGLES, wavelength_nm=DEFAULT_WAVELENGTH_NM,
            solvent_refractive_index=DEFAULT_N_SOLVENT, temperature_K=DEFAULT_TEMPERATURE_K,
            viscosity_Pa_s=DEFAULT_VISCOSITY_PA_S, noise_level=noise, trace_cv=0.03,
            drift=0.0, spikes=0, label=label,
            concentration_g_per_mL=DEFAULT_DLS_SINGLE_MG * 1e-3, seed=seed)
        written.append(write_alv_asc(
            os.path.join(out_dir, stem + '.ASC'), dls, viscosity_cp=DEFAULT_VISCOSITY_CP))
    written.append(_write_alpha_selection_parameters(out_dir))
    return written


def _write_alpha_selection_parameters(out_dir: str) -> str:
    """Human-readable parameters.txt for the alpha-selection gap-case set."""
    import os
    lines = [
        'DLS Buddy — Synthetic Alpha-Selection test data', '=' * 50, '',
        'DLS correlograms that stress CONTIN\'s regularization-parameter (alpha)',
        'selection. Same optical system as the Clean/Messy sets (PEG-in-water,',
        '532 nm); load the .ASC files exactly as those (all angles are in one file;',
        'analyze the ~90 deg angle). Enter polymer name + solvent (water); temperature,',
        'viscosity, refractive index and wavelength are read from the file.', '',
        'GLOBAL PARAMETERS (Data-tab confirmation step):',
        f'  Solvent refractive index: {DEFAULT_N_SOLVENT}',
        f'  Viscosity               : {DEFAULT_VISCOSITY_CP} cP',
        f'  Temperature             : {DEFAULT_TEMPERATURE_C} C',
        f'  Wavelength              : {DEFAULT_WAVELENGTH_NM:g} nm', '',
        'CASES (ground truth — what the distribution methods should recover):', '',
        '  Broad unimodal            : single mode, median Rh = 50 nm, log-normal',
        '                              CV 0.5 (broad). Needs heavier smoothing; a good',
        '                              selector keeps ONE peak and does not under-smooth.',
        '  Close bimodal             : 40 + 90 nm (~2x separation, near the resolution',
        '                              limit) — the hard inversion case.',
        '  Wide bimodal (x3)         : 30 + 200 nm at noise 0.001 / 0.003 / 0.010.',
        '                              A good selector keeps TWO peaks at 0.001 and',
        '                              0.003; 0.010 is the margin (GCV resolves both',
        '                              at some angles and merges them at others);',
        '                              an over-smoothing one merges them into a single',
        '                              intermediate peak.', '',
        'Use CONTIN and compare the GCV (default), L-curve, and F-test selectors',
        '(DLS tab -> Distribution -> CONTIN alpha selection). GCV is the robust default;',
        'the L-curve tends to smooth more heavily on this kernel. See the',
        'Theory-and-Equations-Guide (CONTIN) for the methods.',
    ]
    path = os.path.join(out_dir, 'parameters.txt')
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:   # LF, as committed
        fh.write('\n'.join(lines) + '\n')
    return path


# ===========================================================================
# Branching-architecture dataset (writes a whole test-data/Synthetic/Architecture)
# ===========================================================================
def save_mode_resolved_dataset(tup: ModeResolvedTuple, out_dir: str, *,
                               viscosity_Pa_s: float = DEFAULT_VISCOSITY_PA_S,
                               recipe_note: str = '') -> List[str]:
    """Write a ModeResolvedTuple as a loadable folder (the Mode-Resolved set).

    Two multi-angle ALV ``.ASC`` files WITH count-rate traces (the two-mode
    sample and the c = 0 solvent blank), the calibrant trace CSV, and a
    ``parameters.txt`` recording the full ground truth. Deliberately NO
    pre-derived SLS file: static points are derived in-app from the traces, so
    the acceptance oracle stays the generator's truth rather than a check of
    our own derivation arithmetic.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    written: List[str] = []
    conc = tup.concentration_g_per_mL
    eta_cp = viscosity_Pa_s * 1.0e3
    written.append(write_alv_asc(
        os.path.join(out_dir, f'DLS two-mode c={conc * 1e3:g} mgml.ASC'),
        tup.dls[conc], viscosity_cp=eta_cp))
    written.append(write_alv_asc(
        os.path.join(out_dir, 'DLS solvent blank.ASC'),
        tup.dls[0.0], viscosity_cp=eta_cp))
    if tup.calibrant_trace is not None:
        written.append(write_trace_csv(
            os.path.join(out_dir, 'Calibrant trace (toluene).csv'),
            tup.calibrant_trace))

    d = tup.dls[conc]
    lines = [
        'Mode-Resolved synthetic set -- ground truth and how it was established',
        '=' * 70, '',
        'One linked truth, three faces: two-mode correlograms whose per-angle',
        'amplitude fractions ARE the static partition fractions; count-rate',
        'traces whose means ARE the static intensities (exact-mean matched);',
        'and the per-mode excess Rayleigh ratios below. Load both .ASC files,',
        'commit, run the mode map, pick the two families, record the mode set,',
        'derive the static series from the traces, and the partition must',
        'reproduce the table (the amplitude-Guinier recovers the slow Rg).', '',
        'Deliberately NO pre-derived SLS file: derive the static points in-app',
        'from the traces, so the check is against this truth, not against a',
        'stored copy of the derivation arithmetic.', '',
        'Recipe (build_mode_resolved_tuple): ' + (recipe_note or 'defaults'), '',
        'Scalar truth',
        '-' * 12,
        f'slow-mode Guinier size Rg_s     : {tup.slow_rg_nm:g} nm '
        f'(what the amplitude-Guinier fit recovers)',
        f'fast fraction at q -> 0         : {tup.a_fast_q0:g}',
        f'fast-mode Rh                    : {tup.rh_fast_nm:g} nm '
        f'(D_f = {tup.d_fast_m2_s:.6g} m^2/s)',
        f'slow-mode dynamics              : {tup.slow_kind}'
        + (f', crossing the fast branch at {tup.crossing_angle_deg:g} deg'
           if math.isfinite(tup.crossing_angle_deg) else ''),
        f'concentration                   : {conc * 1e3:g} mg/mL',
        f'wavelength / n / T / viscosity  : {d.wavelength_nm:g} nm / '
        f'{d.n_solvent:g} / {d.temperature_K:g} K / {viscosity_Pa_s:g} Pa*s',
        f'dn/dc (enter by hand)           : {tup.dn_dc:g} mL/g',
        f'calibrant intensity (toluene)   : {tup.cal.calibrant_intensity:g} cps '
        f'(k_c = {tup.cal.k_c():.6g}; geometry {tup.cal.geometry}, '
        f'{tup.cal.wavelength_nm:g} nm, {tup.cal.temperature_C:g} C)', '',
        'Per-angle truth (fractions are domain-free; dR in cm^-1, I in cps)',
        '-' * 68,
        'angle   A_fast   A_slow   dR_fast      dR_slow      dR_total     '
        'I_sample     I_solvent',
    ]
    for a in tup.angles_deg:
        lines.append(
            f'{a:5.1f}   {tup.a_fast[a]:.4f}   {tup.a_slow[a]:.4f}   '
            f'{tup.dr_fast_cm_inv[a]:.5e}  {tup.dr_slow_cm_inv[a]:.5e}  '
            f'{tup.dr_total_cm_inv[a]:.5e}  '
            f'{tup.static_intensity_cps[(conc, a)]:.6g}   '
            f'{tup.static_intensity_cps[(0.0, a)]:.6g}')
    lines += [
        '',
        'Gamma_fast per angle (1/s): '
        + ', '.join(f'{a:g}: {tup.gamma_fast_s_inv[a]:.5g}'
                    for a in tup.angles_deg),
        'Gamma_slow per angle (1/s): '
        + ', '.join(f'{a:g}: {tup.gamma_slow_s_inv[a]:.5g}'
                    for a in tup.angles_deg),
        '',
        f'exact_mean = {tup.exact_mean}; trace_cv = {tup.trace_cv:g}; '
        f'max count rate = {tup.max_count_rate_cps:.6g} cps',
    ]
    ppath = os.path.join(out_dir, 'parameters.txt')
    with open(ppath, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(lines) + '\n')
    written.append(ppath)
    return written


def generate_mode_resolved_dataset(out_dir: str, seed: int = 23) -> List[str]:
    """Regenerate the committed ``test-data/Synthetic/Mode-Resolved/`` set.

    Fixed recipe and seed, so re-running reproduces the same bytes: this function
    IS the committed set's recipe, and ``tests/test_synthetic_generator.py`` pins
    that the folder regenerates byte-identical from it. The in-app save path
    (``controller.save_synth_mode_resolved``) reaches ``build_mode_resolved_tuple``
    + ``save_mode_resolved_dataset`` directly with the user's own recipe, and no
    developer script wraps this one. The angle grid spans the 82-degree crossing
    (the association-breaking configuration stays the test-drivable one).
    """
    tup = build_mode_resolved_tuple(seed=seed, polymer_name='Two-mode demo',
                                    solvent_name='water',
                                    label='Mode-resolved demo')
    return save_mode_resolved_dataset(
        tup, out_dir,
        recipe_note=f'defaults, seed={seed} (slow_rg_nm=120, a_fast_q0=0.3, '
                    f'rh_fast_nm=5, crossing at 82 deg)')


def generate_architecture_dataset(out_dir: str, seed: int = 17) -> List[str]:
    """Regenerate the branching-architecture test set (SLS Zimm + multi-angle DLS per
    sample of :func:`build_architecture_series`, plus a ``parameters.txt``). Fixed seeds,
    so re-running reproduces the same bytes. Shared by the in-app Synthetic Generator and
    the developer test-data generator script. Returns the file
    paths written.
    """
    import os
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    cal = CalibrationSpec()
    series = build_architecture_series()
    written: List[str] = []
    for name, p in series.items():
        sls = build_sls_set(
            mw=p['mw'], rg_nm=p['rg'], a2_mol_mL_per_g2=p['a2'],
            angles_deg=DEFAULT_SLS_ANGLES,
            concentrations_g_per_mL=[c * 1e-3 for c in DEFAULT_SLS_CONCS_MG],
            wavelength_nm=DEFAULT_WAVELENGTH_NM, temperature_K=DEFAULT_TEMPERATURE_K,
            n_solvent=DEFAULT_N_SOLVENT, dn_dc=DEFAULT_DN_DC, cal=cal,
            solvent_intensity_90=DEFAULT_SOLVENT_INTENSITY_90, noise_level=0.005,
            seed=int(rng.integers(1 << 31)), polymer_name=name, solvent_name='water',
            label=f'{name} in water')
        written.append(write_brookhaven_sls(
            os.path.join(out_dir, f'SLS Zimm - {name}.csv'), sls))

        pops = [SyntheticPopulation(rh_nm=p['rh'], weight=1.0)]
        dls = build_multi_angle_dls(
            pops, angles_deg=DEFAULT_DLS_ANGLES, wavelength_nm=DEFAULT_WAVELENGTH_NM,
            solvent_refractive_index=DEFAULT_N_SOLVENT, temperature_K=DEFAULT_TEMPERATURE_K,
            viscosity_Pa_s=DEFAULT_VISCOSITY_PA_S, noise_level=0.005, trace_cv=0.03,
            drift=0.0, spikes=0, label=f'{name} in water',
            concentration_g_per_mL=DEFAULT_DLS_SINGLE_MG * 1e-3,
            seed=int(rng.integers(1 << 31)))       # one seed per run, like the SLS half
        written.append(write_alv_asc(
            os.path.join(out_dir, f'DLS - {name}.ASC'), dls,
            viscosity_cp=DEFAULT_VISCOSITY_CP))

    written.append(_write_architecture_parameters(out_dir, series, cal))
    return written


def _write_architecture_parameters(out_dir: str, series: Dict[str, dict],
                                   cal: CalibrationSpec) -> str:
    """Human-readable parameters.txt (ground truth + what to enter) for the set."""
    import os
    lines = [
        'DLS Buddy — Synthetic Architecture (branching contraction factor g) test data',
        '=' * 76, '',
        'A homologous LINEAR Rg-Mw series (Rg = 0.05 * Mw^0.5), two REGULAR STARS, and two',
        'RANDOMLY-BRANCHED FRACTIONS (tri- and tetra-functional), built to exercise the',
        'Cross-Sample Architecture tab: tag the linear samples Linear and a branched sample',
        'Branched (Chain type, Data tab), and g = (Rg_branched / Rg_linear)^2 is recovered.',
        'Its reading depends on the selected branching model:',
        '  * regular star            -> f_eff  (arm number, from g = (3f-2)/f^2)',
        '  * random tri / tetra      -> n      (branch units/molecule, from Eqs. 47a/47b)', '',
        'GLOBAL PARAMETERS (enter at the Data-tab confirmation step):',
        '  Solvent                 : water',
        f'  Solvent refractive index: {DEFAULT_N_SOLVENT}',
        f'  Viscosity               : {DEFAULT_VISCOSITY_CP} cP',
        f'  Temperature             : {DEFAULT_TEMPERATURE_C} C ({DEFAULT_TEMPERATURE_K} K)',
        f'  Wavelength              : {DEFAULT_WAVELENGTH_NM:g} nm',
        f'  dn/dc                   : {DEFAULT_DN_DC} mL/g', '',
        'SLS CALIBRATION (SLS tab calibration panel):',
        f'  Standard toluene ({cal.geometry}); calibrant intensity {cal.calibrant_intensity:g} at 90 deg',
        f'  (the program computes k_c = {cal.k_c():.4e})', '',
        'GROUND TRUTH (Role -> tag on the Architecture tab):',
        f'  {"sample":<18}{"role":<10}{"Mw":<12}{"Rg (nm)":<10}{"Rh (nm)":<10}'
        f'{"model":<14}{"f":<4}{"n":<4}{"g":<8}',
    ]
    _model_disp = {None: '', 'regular_star': 'regular star',
                   'random_tri': 'random tri', 'random_tetra': 'random tetra'}
    for name, p in series.items():
        f = '' if p['f'] is None else str(p['f'])
        n = '' if p.get('n') is None else str(p['n'])
        g = '' if p['g'] is None else f'{p["g"]:.4f}'
        lines.append(f'  {name:<18}{p["role"]:<10}{p["mw"]:<12.3g}'
                     f'{p["rg"]:<10.3f}{p["rh"]:<10.3f}'
                     f'{_model_disp[p.get("model")]:<14}{f:<4}{n:<4}{g:<8}')
    lines += ['',
              'Expected on the Architecture tab (linear series tagged Linear, one Branched):',
              '  PS star 3-arm    -> g ≈ 0.778;  regular star  -> f_eff ≈ 3',
              '  PS star 4-arm    -> g ≈ 0.625;  regular star  -> f_eff ≈ 4',
              '  PS random tri    -> g ≈ 0.672;  random tri    -> n ≈ 6 branch units/molecule',
              '  PS random tetra  -> g ≈ 0.578;  random tetra  -> n ≈ 4 branch units/molecule',
              '  (all four branched Mw sit inside the linear reference span 1e5-9e5)', '',
              'MODEL-DEPENDENCE (the headline caveat): the SAME g inverts to a DIFFERENT n',
              'under a different model. E.g. PS random tri (g ≈ 0.672) reads as n ≈ 6 under the',
              'tri model but n ≈ 2.4 under the tetra model, and f_eff ≈ 3.6 as a regular star —',
              'so the readout must always name the assumed model.', '']
    path = os.path.join(out_dir, 'parameters.txt')
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:   # LF, as committed
        fh.write('\n'.join(lines))
    return path


# ===========================================================================
# Form-factor SETS (SLS form-factor P(q) + paired DLS) for Cross-Sample
# ===========================================================================
# Each sample is an SLS form-factor set + a paired multi-angle DLS whose Rh is set
# from a textbook rho = Rg/Rh for that shape, so loading the pair, fitting the form
# factor for Rg and a cumulant/Gamma-q2 for Rh, and pairing them on the Cross-Sample
# tab reproduces a known rho shape verdict. The panel deliberately exercises the
# different axes of the cross-sample tagging/validation system (see parameters.txt).
_FF_SPHERE_RG_K = 0.4          # hard-sphere Rg = k * Mw^(1/3) (constant-density sphere), nm
_FF_SPHERE_RG_NU = 1.0 / 3.0
_FF_RHO_SPHERE = 0.775         # sqrt(3/5): hard sphere
_FF_RHO_SOFT = 1.0             # dendrimer / soft sphere / core-shell object
_FF_RHO_COIL = 1.5             # linear random coil (theta)
_FF_RHO_ROD = 2.5              # extended / rod-like


def build_form_factor_series() -> Dict[str, dict]:
    """``{name: dict(mw, rg, rh, a2, model, params, rho, verdict, role, calibrated,
    single_conc, degenerate)}`` — a panel of form-factor SLS sets, each paired with a DLS
    Rh chosen for that shape's textbook rho, to drive the Cross-Sample ρ / scaling /
    degenerate-selection surfaces from loadable data. ``rh`` is derived (rg / rho) so the
    ρ shape verdict is what's under test. ``params`` holds the model's shape parameters
    (axial ratio / core+outer radii + contrasts); ``degenerate`` is False, 'divergent'
    (ellipsoid: per-solution picks, no consensus) or 'consensus' (core-shell: Rg agree).
    """
    from analysis.form_factor import core_shell_contrast_weighted_rg
    series: Dict[str, dict] = {}

    # --- shape-verdict panel (one Mw each; rho set to the shape's textbook value) -------
    series['FF coil'] = dict(
        mw=8.0e5, rg=55.0, a2=8.0e-5, model='debye', params={},
        rho=_FF_RHO_COIL, verdict='random (Gaussian) coil', role=None,
        calibrated=True, single_conc=False, degenerate=False)
    series['FF ellipsoid'] = dict(
        mw=3.0e6, rg=120.0, a2=5.0e-5, model='ellipsoid',
        params=dict(ellipsoid_axial_ratio=3.0),
        rho=_FF_RHO_ROD, verdict='rod / extended', role=None,
        calibrated=True, single_conc=False, degenerate='divergent')
    # Rc35/Ro90 puts qRg high enough (~2) that the two-radius fit is degenerate, but the
    # solutions AGREE on the contrast-weighted Rg -> the "size determined, shape ambiguous"
    # consensus candidate is offered (Part C). (A smaller core-shell resolves to one solution.)
    _rc, _ro, _m1, _m2 = 35.0, 90.0, 1.20, 1.05
    _cs_rg = float(core_shell_contrast_weighted_rg(_rc, _ro, _m1, _m2))
    series['FF core-shell'] = dict(
        mw=4.0e6, rg=round(_cs_rg, 3), a2=6.0e-5, model='core_shell',
        params=dict(r_core_nm=_rc, r_outer_nm=_ro, m1=_m1, m2=_m2),
        rho=_FF_RHO_SOFT, verdict='dendrimer / soft sphere', role=None,
        calibrated=True, single_conc=False, degenerate='consensus')

    # --- model-dependent scaling series (homologous hard spheres, Rg ~ Mw^(1/3)) --------
    # Tagged Linear so the Cross-Sample scaling plot fits nu ~ 1/3 on a MODEL-DEPENDENT Rg
    # (distinct from the Architecture set's agnostic nu = 0.5).
    for mw in (5.0e5, 1.0e6, 2.0e6):
        rg = _FF_SPHERE_RG_K * mw ** _FF_SPHERE_RG_NU
        series[f'FF sphere {mw/1e6:g}M'] = dict(
            mw=mw, rg=round(rg, 3), a2=7.0e-5, model='sphere', params={},
            rho=_FF_RHO_SPHERE, verdict='hard sphere', role='linear',
            calibrated=True, single_conc=False, degenerate=False)

    # --- apparent variant (single concentration -> the form-factor Rg is apparent) ------
    series['FF sphere apparent'] = dict(
        mw=1.0e6, rg=40.0, a2=7.0e-5, model='sphere', params={},
        rho=_FF_RHO_SPHERE, verdict='hard sphere', role=None,
        calibrated=True, single_conc=True, degenerate=False)

    # --- uncalibrated variant (load in its own session / clear calibration) -------------
    series['FF sphere uncal'] = dict(
        mw=1.0e6, rg=40.0, a2=7.0e-5, model='sphere', params={},
        rho=_FF_RHO_SPHERE, verdict='hard sphere', role=None,
        calibrated=False, single_conc=False, degenerate=False)

    # --- large-Mie latex (the wrong-basin regime) ---------------------------------------
    # A latex sphere (m = 1.585/1.33 ~ 1.19) at Rg 250 nm puts qRg ~ 7.5, past the first Mie
    # minimum -- the regime where a single-seed Mie fit can land in a WRONG basin with Mw
    # 50-99% off while still reading reliable. Committed so that regime is reachable from
    # in-app ground truth and any grid/tolerance change is exercised end-to-end. Expect the
    # deep-minimum flag: Rg recovers exactly, Mw/A2 read ill-conditioned. Kept LAST -- the
    # per-sample SLS seeds are drawn sequentially from one RNG stream in dict order, so
    # appending (never inserting) keeps the earlier committed files byte-identical.
    series['FF Mie latex'] = dict(
        mw=6.0e5, rg=250.0, a2=2.0e-4, model='mie',
        params=dict(m=1.585 / DEFAULT_N_SOLVENT),
        rho=_FF_RHO_SPHERE, verdict='hard sphere', role=None,
        calibrated=True, single_conc=False, degenerate=False)

    for p in series.values():
        p['rh'] = round(p['rg'] / p['rho'], 3)          # DLS Rh set from the target rho
    return series


def generate_form_factor_dataset(out_dir: str, seed: int = 23) -> List[str]:
    """Regenerate the form-factor SET test data (an SLS form-factor CSV + a paired
    multi-angle DLS ASC per sample of :func:`build_form_factor_series`, plus a
    ``parameters.txt``). Fixed seeds -> reproducible bytes. Shared by the in-app Synthetic
    Generator and the developer test-data generator script.
    Returns the file paths written.
    """
    import os
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    series = build_form_factor_series()
    written: List[str] = []
    for name, p in series.items():
        cal = CalibrationSpec(include_in_files=bool(p['calibrated']))
        concs_mg = ([DEFAULT_DLS_SINGLE_MG] if p['single_conc']
                    else [c for c in DEFAULT_SLS_CONCS_MG if c > 0])
        sls = build_sls_set(
            mw=p['mw'], rg_nm=p['rg'], a2_mol_mL_per_g2=p['a2'],
            angles_deg=DEFAULT_SLS_ANGLES,
            concentrations_g_per_mL=[c * 1e-3 for c in concs_mg],
            wavelength_nm=DEFAULT_WAVELENGTH_NM, temperature_K=DEFAULT_TEMPERATURE_K,
            n_solvent=DEFAULT_N_SOLVENT, dn_dc=DEFAULT_DN_DC, cal=cal,
            solvent_intensity_90=DEFAULT_SOLVENT_INTENSITY_90, noise_level=0.005,
            seed=int(rng.integers(1 << 31)), polymer_name=name, solvent_name='water',
            label=f'{name} in water',
            kind=('single_concentration' if p['single_conc'] else 'zimm'),
            form_factor_model=p['model'], **p['params'])
        written.append(write_brookhaven_sls(
            os.path.join(out_dir, f'SLS form factor - {name}.csv'), sls))

        pops = [SyntheticPopulation(rh_nm=p['rh'], weight=1.0)]
        dls = build_multi_angle_dls(
            pops, angles_deg=DEFAULT_DLS_ANGLES, wavelength_nm=DEFAULT_WAVELENGTH_NM,
            solvent_refractive_index=DEFAULT_N_SOLVENT, temperature_K=DEFAULT_TEMPERATURE_K,
            viscosity_Pa_s=DEFAULT_VISCOSITY_PA_S, noise_level=0.005, trace_cv=0.03,
            drift=0.0, spikes=0, label=f'{name} in water',
            concentration_g_per_mL=DEFAULT_DLS_SINGLE_MG * 1e-3,
            seed=int(rng.integers(1 << 31)))       # one seed per run, like the SLS half
        written.append(write_alv_asc(
            os.path.join(out_dir, f'DLS - {name}.ASC'), dls,
            viscosity_cp=DEFAULT_VISCOSITY_CP))

    written.append(_write_form_factor_parameters(out_dir, series, CalibrationSpec()))
    return written


def _write_form_factor_parameters(out_dir: str, series: Dict[str, dict],
                                  cal: CalibrationSpec) -> str:
    """Human-readable parameters.txt (ground truth + what each sample exercises)."""
    import os
    _degen_disp = {False: 'no', 'divergent': 'yes (Rg-divergent)',
                   'consensus': 'yes (Rg-agree)'}
    lines = [
        'DLS Buddy — Synthetic Form-factor SETS (SLS P(q) + paired DLS) test data',
        '=' * 76, '',
        'Each sample is an SLS form-factor set PLUS a paired multi-angle DLS whose Rh is',
        'set from a textbook rho = Rg/Rh for that shape. Load a pair, fit the Form factor',
        'P(q) for Rg (SLS tab) and a cumulant / Gamma-q^2 for Rh (DLS tab), pick the',
        'form-factor Rg on the Cross-Sample tab, and rho = Rg/Rh reproduces the shape',
        'verdict below. The panel exercises the cross-sample tagging / validation axes:',
        '  * rho shape verdict          -> sphere / soft-sphere / coil / rod',
        '  * model-dependent flag        -> a form-factor Rg is flagged (neutral i) everywhere',
        '  * degenerate selection        -> ellipsoid (Rg-divergent) + core-shell (Rg-agree)',
        '  * model-dependent scaling     -> the FF sphere ladder fits nu ~ 1/3 (tag Linear)',
        '  * apparent vs thermodynamic   -> FF sphere apparent is single-concentration',
        '  * uncalibrated                -> FF sphere uncal (Mw arbitrary scale; Rg/rho survive)',
        '', 'GLOBAL PARAMETERS (enter at the Data-tab confirmation step):',
        '  Solvent                 : water',
        f'  Solvent refractive index: {DEFAULT_N_SOLVENT}',
        f'  Viscosity               : {DEFAULT_VISCOSITY_CP} cP',
        f'  Temperature             : {DEFAULT_TEMPERATURE_C} C ({DEFAULT_TEMPERATURE_K} K)',
        f'  Wavelength              : {DEFAULT_WAVELENGTH_NM:g} nm',
        f'  dn/dc                   : {DEFAULT_DN_DC} mL/g', '',
        'SLS CALIBRATION (SLS tab calibration panel; the calibrated samples carry it):',
        f'  Standard toluene ({cal.geometry}); calibrant intensity {cal.calibrant_intensity:g} at 90 deg',
        f'  (the program computes k_c = {cal.k_c():.4e})', '',
        'GROUND TRUTH:',
        f'  {"sample":<20}{"model":<12}{"Mw":<11}{"Rg (nm)":<9}{"Rh (nm)":<9}'
        f'{"rho":<6}{"verdict":<26}{"degenerate":<20}{"cal":<5}{"shape params"}',
    ]
    for name, p in series.items():
        # Shape params are a VARIABLE-length string, so they get their own TRAILING column --
        # never concatenated into the fixed-width `model` field (that would overflow the pad and
        # run the next column into it, exactly on the degenerate ellipsoid/core-shell rows).
        params = ''
        if p['model'] == 'ellipsoid':
            params = f"eps={p['params']['ellipsoid_axial_ratio']:g}"
        elif p['model'] == 'mie':
            params = f"m={p['params']['m']:g}"
        elif p['model'] == 'core_shell':
            params = (f"Rc={p['params']['r_core_nm']:g} Ro={p['params']['r_outer_nm']:g}"
                      f" m1={p['params']['m1']:g} m2={p['params']['m2']:g}")
        lines.append(
            f'  {name:<20}{p["model"]:<12}{p["mw"]:<11.3g}{p["rg"]:<9.3g}'
            f'{p["rh"]:<9.3g}{p["rho"]:<6.2f}{p["verdict"]:<26}'
            f'{_degen_disp[p["degenerate"]]:<20}{"yes" if p["calibrated"] else "NO":<5}{params}')
    lines += [
        '', 'WHAT TO EXPECT ON THE CROSS-SAMPLE TAB:',
        '  * FF coil        -> rho ~ 1.5  -> "random (Gaussian) coil"',
        '  * FF ellipsoid   -> rho ~ 2.5  -> "rod / extended"; the ellipsoid fit is DEGENERATE',
        '      and Rg-divergent, so each admissible solution is an explicit pick (no consensus).',
        '  * FF core-shell  -> rho ~ 1.0  -> "dendrimer / soft sphere"; DEGENERATE but the',
        '      solutions AGREE on Rg, so a "size determined, shape ambiguous" consensus is offered.',
        '  * FF Mie latex   -> rho ~ 0.78 "hard sphere" (a solid latex sphere); qRg ~ 7.5 sits',
        '      past the first Mie minimum, so Rg recovers while Mw/A2 carry the deep-minimum',
        '      ill-conditioning flag -- expected physics, not a defect.',
        '  * FF sphere 0.5M/1M/2M -> rho ~ 0.78 "hard sphere"; tagged Linear, the scaling plot',
        '      fits nu ~ 1/3 (Rg-Mw and Rh-Mw), carrying the MODEL-DEPENDENT flag.',
        '  * FF sphere apparent -> a single-concentration form-factor Rg -> flagged APPARENT.',
        '  * FF sphere uncal    -> load in its OWN session (or clear the SLS calibration): Mw and',
        '      A2 go arbitrary-scale (uncalibrated warning), but Rg and rho survive.',
        '', 'NOTE: SLS calibration is session-global, so do not load "FF sphere uncal" in the',
        'same session as the calibrated samples if you want to see its uncalibrated behaviour.', '']
    path = os.path.join(out_dir, 'parameters.txt')
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:   # LF, as committed
        fh.write('\n'.join(lines))
    return path
