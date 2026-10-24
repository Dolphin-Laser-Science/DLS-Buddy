"""
analysis/sls.py
===============

Static light scattering (SLS) analysis: molecular weight Mw, radius of gyration
Rg, and second virial coefficient A2 from angular/concentration intensity data.

Built in passes:

  Pass 1 (this file so far): calibration pipeline
    - compute_excess_rayleigh_ratio   raw intensities -> excess Rayleigh ratio

  Pass 2: single-concentration / apparent analyses (Debye plot, single-angle Mw)

  Pass 3: full Zimm and Berry double extrapolation (Mw, Rg, A2), and the
          calibration-free A2 method.

The basic SLS equation, the first-order Zimm equation (Russo et al. 2021, Eq. 13.3,
p. 501; its origin is Zimm 1948):

    K c / dR(theta, c) = (1 / Mw) [1 + q^2 Rg^2 / 3] + 2 A2 c + ...

the small-q form of the general Kc/dR = 1/(Mw P(q)) + 2 A2 c with
P(q) = 1 - q^2 Rg^2/3 + ... (Chu 1991, Eq. 2.1.8, p. 18).

with the optical constant (VV / vertical polarization, the factor-4 form)

    K = 4 pi^2 n^2 (dn/dc)^2 / (Na lambda^4)

and the scattering vector  q = (4 pi n / lambda) sin(theta / 2).

Calibration to an absolute excess Rayleigh ratio is platform-independent and
runs through ONE pipeline (compute_excess_rayleigh_ratio):

  A calibration constant k_c = R_std / I_std(90 deg) is formed from a single
     calibrant point (compute_calibration_constant: the measured toluene
     intensity at one angle, the authoritative Rayleigh ratio from
     physics/constants.py), and every excess Rayleigh ratio is then
         dR(theta) = k_c * sin(theta) * [I_sample - I_solvent] * (n_s/n_std)^2,
     applied unconditionally to every data source.

  The sin(theta) factor is the scattering-volume correction, paired with the one
     inside k_c (a k_c from a ~90 deg standard carries sin(90) = 1) -- it must
     never be applied a second time: forming k_c per angle as R_std/I_std(theta)
     with the sin(theta) volume correction and then dividing per angle would
     double-count it, so there is no per-angle standard-division route.
     compute_excess_rayleigh_ratio's docstring records why the step is
     unconditional and why a switch for it was retired.

     Provenance of the two halves, kept apart deliberately. The volume
     correction itself is Seery et al. 1989, which states it as PROSE and not
     as an equation (p. 1198: "the detection optics define a scattering volume
     which is inversely proportional to sine of the scattering angle. We
     correct for this by multiplying the scattering intensity by sin theta").
     The step above also leans on I_std(theta) sin(theta) = I_std(90), which
     Seery does not print: that equality is this module's own consequence of
     the correction plus the angle-independence of an optically isotropic
     scatterer under vertical incident polarization. Angle-independence is
     homed in Kerker 1969 (p. 34, p. 580) and Schaertl 2007 (p. 27), not in
     Seery -- who in fact measured the solvent at every angle (his Eq. 2,
     p. 1200, subtracts I_solvent(theta) angle by angle).

IMPORTANT calibration conventions (enforced here)
-------------------------------------------------
- The standard Rayleigh ratio defaults to the Sivokhin & Kazantsev (2021)
  temperature-corrected toluene value from physics/constants.py. For a
  calibration constant carrying an instrument's own (often outdated) toluene
  value, that value can be supplied so k_c is rescaled to the authoritative one.
  No instrument's stored Rayleigh ratio is trusted as the absolute reference.
- The (n_solvent / n_standard)^2 refractive-index correction is applied whenever
  the solvent and standard refractive indices differ. It is mandatory.
- The dark count is taken off the sample, the solvent reference and the calibrant
  alike. With a measured blank it cancels in the excess and survives only in k_c,
  hence in Mw and absolute A2; when no blank was measured it survives in the excess
  as well (`compute_excess_rayleigh_ratio`, "Where the dark count survives"). The
  sin(theta) volume correction is always applied, paired with the 1/sin(theta_cal)
  inside k_c.

References (in project knowledge)
---------------------------------
  Russo et al. 2021, Ch. 13, Eq. 13.3 (p. 501) and Chu 1991, Eq. 2.1.8 (p. 18): the
      Zimm equation, its origin Zimm 1948 (J. Chem. Phys. 16, 1093 and 1099)
  Berry 1966, p. 4552, and Podzimek 2011, Eqs. 2.28, 4.11, 4.13: the Berry construction
      (see zimm_analysis)
  Brookhaven BI-200SM manual, Section VIII (Rayleigh ratios / calibration)
  Brookhaven BIZPW (Zimm Plot) and Particle Explorer manuals: the vendor's Zimm-plot
      and export software, instrument references only -- never a method's source
  Takahashi et al. 2019, Anal. Sci. 35, 1045 (Rayleigh ratio)
  Seery et al. 1989 (sin(theta) volume correction, as prose, p. 1198)
  Kerker 1969 / Schaertl 2007 (angle-independent isotropic scattering)
  Sivokhin & Kazantsev 2021 (toluene Rayleigh ratio; via constants.py)

Design contract
---------------
Every function is PURE: it takes measurement objects (and parameters) and returns
a result object. No plotting, no file I/O, no mutation of inputs. Apparent
(single-angle or single-concentration) results are always clearly distinguished
from thermodynamic (fully extrapolated) ones.

Change history
--------------
2026-06-13  Pass 1: calibration pipeline. (sls.py v1)
            compute_excess_rayleigh_ratio, zimm_coordinate, result objects.
2026-08-05  Removed zimm_coordinate -- a two-tuple over fields every caller
            already read directly (.q2_nm2, .kc_over_dR_mol_per_g), with no
            caller anywhere in the tree.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from core.data_models import SLSMeasurement
from core.notes import Note, render
from analysis import uncertainty as unc
from analysis import form_factor as ff
from physics.constants import (
    scattering_vector_q,
    optical_constant_K,
    refractive_index_correction,
)


# ---------------------------------------------------------------------------
# Notes shared by the SLS fits (the model and its codes: core/notes.py, core/note_registry.py)
# ---------------------------------------------------------------------------

def _ill_conditioned_note(cond: float, limit: float, about: Tuple[str, ...]) -> Note:
    """The precondition note of a degenerate linear extrapolation design (see
    uncertainty.COND_LIMIT).

    A fit whose design X is near-singular (clustered angles, two near-equal
    concentrations, or an extreme intensity scale) yields a numerically unreliable
    1/coefficient Mw/A2, and a fragile coefficient-ratio Rg, even though each value is
    finite. `about` names the values of the carrying result that rest on the design.
    """
    return Note('ill_conditioned', about, {'cond': float(cond), 'limit': float(limit)})


def _non_positive_intercept_note(intercept: float, about: Tuple[str, ...]) -> Note:
    """The precondition note of the two fits whose 2 A2 Mw = slope/intercept shares the
    intercept's sign precondition with Mw = 1/intercept: a non-positive fitted
    intercept means the line the data describe has no physical Mw, so the product
    built on it is undefined too. The product is flagged and still reported (a zero
    intercept excepted); the apparent Mw beside it is NaN'd, since a negative or infinite
    reciprocal is not a molar mass (the single-angle ladder's `mw_app` branch)."""
    return Note('intercept_nonpositive', about,
                {'quantity': 'intercept', 'value': float(intercept), 'limit': 0.0})


def _no_rg_note(about: Tuple[str, ...], quantity: str, value: float, unit: str,
                required: str) -> Note:
    """The precondition note of a fit that extracts no Rg: the quantity the size is read
    from, its value and the sign the size needs. One code for every linear arm (Debye,
    Guinier, Zimm/Berry and the mode-resolved amplitude Guinier)."""
    return Note('no_rg', about, {'quantity': quantity, 'value': float(value),
                                 'unit': unit, 'required': required})


def _qrg_regime_note(about: Tuple[str, ...], qrg_max: float, limit: float,
                     regime: str) -> Note:
    """The precondition note of a fit read past its small-qRg regime: the largest fitted
    q times Rg against the limit the verdict read, named with the regime's form. `about`
    names what the limit judges, the size and, on the Debye and Zimm fits, the Mw."""
    return Note('qrg_above_limit', about,
                {'qrg': float(qrg_max), 'limit': float(limit), 'regime': regime})


def scale_notes(rayleigh_result, *, mw: Tuple[str, ...] = (), a2: Tuple[str, ...] = (),
                scale_free: Tuple[str, ...] = (),
                excess: Tuple[str, ...] = ()) -> Tuple[Note, ...]:
    """The scale gates of a fit built on `rayleigh_result`, as precondition notes.

    `rayleigh_result` is one `RayleighRatioResult` (or a stand-in with its fields), or the
    list or tuple of them the fit's own flags aggregate over (the same set, so a note and
    the flag it explains are read off one input): `uncalibrated` when any lacks an
    established absolute scale for want of a calibration constant, `no_standard_index`
    when any applied one with the standard's refractive index blank, so the (n_s/n_cal)^2
    step could not be formed (`standard_index_known`), `no_dn_dc` when any lacks a usable
    dn/dc, `blank_unmeasured` when any lacks a measured solvent blank.

    The two scale gates are about the fit's scale-bearing attributes, `mw` and `a2`, and
    are not minted when it names none (an estimator both scales cancel in). The blank is
    additive, so it is also about `scale_free`: the sizes and products the two scales
    cancel in and the blank does not (RayleighRatioResult's docstring is the ruling's home).
    `excess` names attributes on the Rayleigh-ratio scale alone, which the calibration and
    the blank reach and dn/dc does not (dn/dc enters K, not dR): the excess Rayleigh ratio
    itself, on the series `compute_excess_rayleigh_ratio` returns.
    """
    results = (list(rayleigh_result) if isinstance(rayleigh_result, (list, tuple))
               else [rayleigh_result])
    scaled = tuple(mw) + tuple(a2)
    calibration = scaled + tuple(excess)
    out = []
    # The calibration's cause, read off the flag the fits' verdicts read, so neither note can
    # stand beside a `calibrated` that is True: no k_c, or k_c with the index blank.
    lacking = [r for r in results if not bool(getattr(r, 'calibrated', True))]
    if calibration and any(bool(getattr(r, 'standard_index_known', True)) for r in lacking):
        out.append(Note('uncalibrated', calibration))
    if calibration and not all(bool(getattr(r, 'standard_index_known', True)) for r in lacking):
        out.append(Note('no_standard_index', calibration))
    if scaled and not all(bool(getattr(r, 'dn_dc_supplied', True)) for r in results):
        out.append(Note('no_dn_dc', scaled))
    if not all(bool(getattr(r, 'blank_measured', True)) for r in results):
        out.append(Note('blank_unmeasured', calibration + tuple(scale_free)))
    return tuple(out)


@dataclass(frozen=True)
class ModePointCounts:
    """One mode's counts over the points its partitioned share was built on
    (`mode_partition.partition_rayleigh`): how many were undecayed in their lag window, how
    many below their amplitude floor, and at how many it shares one decomposition with an
    undecayed mode. A mode's fraction is its share of the decomposition at its angle, so an
    undecayed mode's extrapolated amplitude sits in every fraction's denominator there (the
    per-angle rule); its own points count as undecayed, a decayed neighbour's as spoiled.

    The grid end follows the same rule (a recommendation the owner accepted, 2026-10-17;
    this docstring is its home): a mode whose lobe has its maximum on the Rh grid's first or last node stands
    for a population the grid does not contain, so its own points count as `grid_end` and
    a neighbour's that are not at the grid's end, at the angles they share one
    decomposition with it, as `grid_end_spoiled`. The two rules are independent: a point
    both undecayed and at the grid's end counts under each, and a neighbour of both kinds
    of mode counts under both spoilers."""
    label: str
    points: int
    undecayed: int = 0
    low_amplitude: int = 0
    spoiled: int = 0
    grid_end: int = 0
    grid_end_spoiled: int = 0


def mode_point_notes(sources, *, about: Tuple[str, ...]) -> Tuple[Note, ...]:
    """The per-mode preconditions of a fit built on partitioned series, as notes about
    `about`, the fit's values on that mode's share.

    `sources` is one `RayleighRatioResult` (its `mode_counts`), one `ModePointCounts`, or
    a list or tuple of either, every series the fit consumed; the counts are summed over
    them. A fitter folds `not notes` into each verdict the notes' registry rows fail, from
    this one computation. `()` on a series that is no mode's share."""
    items = list(sources) if isinstance(sources, (list, tuple)) else [sources]
    counts = [c for c in (s if isinstance(s, ModePointCounts)
                          else getattr(s, 'mode_counts', None) for s in items)
              if c is not None]
    if not counts:
        return ()
    label = counts[0].label
    total = int(sum(c.points for c in counts))
    undecayed = int(sum(c.undecayed for c in counts))
    low = int(sum(c.low_amplitude for c in counts))
    spoiled = int(sum(c.spoiled for c in counts))
    at_end = int(sum(c.grid_end for c in counts))
    end_spoiled = int(sum(c.grid_end_spoiled for c in counts))
    out = []
    if undecayed:
        out.append(Note('points_undecayed_used', about,
                        {'label': label, 'n': undecayed, 'total': total}))
    if spoiled:
        out.append(Note('partition_angle_spoiled', about, {'n': spoiled, 'total': total}))
    if at_end:
        out.append(Note('points_grid_end_used', about,
                        {'label': label, 'n': at_end, 'total': total}))
    if end_spoiled:
        out.append(Note('partition_angle_grid_end', about,
                        {'n': end_spoiled, 'total': total}))
    if low:
        out.append(Note('points_low_amplitude_used', about,
                        {'label': label, 'n': low, 'total': total}))
    return tuple(out)


def _ordinate_scale(y: np.ndarray) -> float:
    """Representative magnitude of a Kc/dR ordinate, for normalizing a form-factor fit.

    Every form-factor fit normalizes its ordinate before optimizing: an uncalibrated (or
    dn/dc-absent) run puts Kc/dR on an arbitrary scale, so the raw values can sit near
    1e-16 where the residual minimum drowns in float roundoff. Dividing by a
    representative magnitude makes the optimization well-scaled at any scale; the factor
    multiplies only 1/Mw and A2 and leaves the angular sizes untouched, which is exactly
    the physics that Rg survives an unknown scale (invariant 6).

    THE one body of that scale, for all three form-factor entry points. A scale cannot
    be invented from data that has none: a degenerate ordinate (every value zero)
    normalized to some substitute and fitted returns a fabricated Rg and Mw that read as
    reliable, while Guinier returns the correct Rg on the same input. So this REFUSES. That is the same class of refusal as `guinier_analysis` demanding two
    angles with dR > 0 and `zimm_analysis` demanding two concentrations: the input cannot
    support the fit, which is a different thing from a fit that ran and did not converge.

    Non-finite points are the caller's business, not this function's — every caller has
    already dropped them (the shared finite mask before the angle-count guard), so an
    all-NaN ordinate never reaches here; it is refused one step earlier for having no
    fittable angles. What reaches here and must be refused is a finite ordinate whose
    every value is zero, or whose median underflows to zero.

    Raises
    ------
    ValueError
        If the ordinate carries no usable magnitude.
    """
    scale = float(np.median(np.abs(y)))
    if not (scale > 0 and math.isfinite(scale)):
        raise ValueError(
            "Form-factor fit: the Kc/dR ordinate is degenerate — its representative "
            f"magnitude is {scale!r}, so there is no scale to fit against. Every value "
            "is zero (or underflows to zero), which happens when the excess scattering "
            "is indistinguishable from the solvent blank, or when the optical constant K "
            "collapsed. Check the solvent reference, the calibration constant, and the "
            "refractive-index increment."
        )
    return scale


# ===========================================================================
# Result objects
# ===========================================================================

@dataclass
class RayleighRatioResult:
    """Excess Rayleigh ratio for one concentration across angles.

    Carries the excess Rayleigh ratio dR(theta), the scattering vectors, the
    optical constant K, and the Zimm ordinate Kc/dR.

    TWO independent scale flags, and they answer different questions:

    * `calibrated` -- False when the absolute scale of dR was not established:
      no calibration constant was available (dR on an arbitrary scale), OR the
      calibration standard's refractive index was blank, so the mandatory
      (n_s/n_cal)^2 correction could not be applied and dR is off by that unknown
      factor. `standard_index_known` says which, typed; `calibration_detail` says it in
      words; `k_c_used` records the constant that WAS applied (1.0 only in the first case).
    * `dn_dc_supplied` -- False when the sample carried no refractive-index
      increment, so K was set to 1 and the Kc/dR ORDINATE is on an arbitrary
      optical scale. dR is unaffected: dn/dc enters K, not the Rayleigh ratio.

    Either one leaves the scale-DEPENDENT quantities (Mw, absolute A2) unreliable
    while the scale-INDEPENDENT ones (every Rg, the form-factor sizes, and the
    calibration-free 2 A2 Mw product) remain valid -- the downstream results carry
    the corresponding flags. They can hold independently and are reported
    independently; neither subsumes the other.

    A THIRD flag, `blank_measured`, and why Rg does NOT survive it
    -------------------------------------------------------------
    `blank_measured` is False when no c = 0 solvent measurement existed for this
    series and the user confirmed the run anyway. It is deliberately NOT a third
    scale flag, and it is the one absence that takes Rg down with it.

    An unknown calibration constant, and an absent dn/dc, are MULTIPLICATIVE: they
    scale every point by the same unknown factor, which cancels out of a slope/
    intercept ratio. That is exactly why Rg, the form-factor sizes and the
    calibration-free 2*A2*Mw product survive both.

    A blank that was not measured is ADDITIVE, and that alone is enough. An additive
    term does not divide out of a slope-to-intercept ratio the way a multiplicative one
    does, so it corrupts the q^2 dependence -- Rg, not just Mw. Measured on a planted
    Rg = 40.0 nm: a constant offset of 10 % of the signal reads back as 36.3 nm on the
    Guinier route. Hence `blank_measured` is ANDed into EVERY reliability verdict
    downstream, `rg_reliable` and `two_a2_mw_reliable` included.

    Be precise about WHEN it also varies with angle, because the obvious statement is
    wrong. With no subtraction at all the residue left in dR is exactly CONSTANT in
    angle: the pipeline's sin(theta) volume correction cancels the solvent's own
    1/sin(theta). The same holds for a typed value that is simply too big or too small.
    Angle-dependence is the EXTRA damage done when the 1/sin(theta) assumption itself
    fails -- a solvent that does not scatter like a simple liquid -- and it is not needed
    for any of the reasoning above.

    This docstring is the normative home of that ruling (owner-ruled). See CLAUDE.md's
    flag-not-refuse clause for why the value is still reported rather than withheld.
    """
    # identity / context
    concentration_g_per_mL: float
    temperature_K: float
    wavelength_nm: float
    solvent_refractive_index: float
    dn_dc_mL_per_g: Optional[float]    # None -> not supplied; K was set to 1
    # per-angle arrays (sorted by angle ascending)
    angles_deg: np.ndarray
    q_nm_inv: np.ndarray
    q2_nm2: np.ndarray                 # q^2 in nm^-2 (Zimm abscissa, q-space)
    excess_rayleigh_cm_inv: np.ndarray # dR(theta) in cm^-1 (arbitrary scale if not calibrated)
    kc_over_dR_mol_per_g: np.ndarray   # Kc / dR  (the Zimm ordinate)
    # optical constant and calibration bookkeeping
    optical_constant_K: float
    calibrated: bool                   # False -> dR's absolute scale not established (no
                                       # k_c, or no standard n for the (n_s/n_cal)^2 step);
                                       # its dn/dc sibling is `dn_dc_supplied` below
    k_c_used: float                    # calibration constant applied (1.0 if none given)
    ri_correction: float               # (n_solvent / n_standard)^2 (1.0 if skipped)
    dark_count_subtracted: float
    sin_theta_applied: bool
    # Which scale absence `calibrated` / `dn_dc_supplied` record, in words, for a reader of
    # this object: provenance, not a note. The fits built on this result state each absence
    # as a precondition note of their own (`scale_notes`).
    calibration_detail: str = ''
    # The `calibrated` sibling for the OPTICAL scale: False -> no dn/dc was supplied, so
    # K = 1 and the Kc/dR ordinate is on an arbitrary optical scale. Independent of
    # `calibrated` (which is about dR, and dn/dc does not enter dR). It sits down here
    # only because it needs a default and `calibrated` does not; read the two together.
    # Defaulted True so a hand-built result reads as "dn/dc was known" -- the same
    # not-a-scale-verdict direction as workspace.is_arbitrary_scale's None edge.
    dn_dc_supplied: bool = True
    # How the sample intensities were obtained, when they were not read from an
    # instrument's own static file -- e.g. averaged from DLS runs' count rates.
    # A RECORD carried for disclosure, exactly as `se_estimator` carries which
    # covariance produced a SE: it enters no computation, and it travels into the
    # affected columns' Comments so a number cannot be read without its origin.
    # Empty for an ordinary loaded measurement, which needs no such note.
    derived_provenance: str = ''
    # Which solvent blank was subtracted, when it is NOT the sample's reference blank
    # (the silent default) -- a second RECORD beside derived_provenance, kept in its own
    # field so the derivation reducers above never see it: their mixed-set sentence is
    # a claim about DLS-derived points and must stay one. Empty for the common case.
    blank_provenance: str = ''
    # Was the solvent blank MEASURED at every angle, as the canonical method subtracts
    # it (Seery et al. 1989 Eq. 2, p. 1200)? False when the series had no c = 0
    # measurement and the user confirmed a run anyway. WHICH of the two blankless cases
    # happened -- a blank computed from a typed 90 deg intensity, or no subtraction at
    # all -- is carried by `blank_provenance`'s sentence, not by a third state here.
    # Why it gates every verdict INCLUDING the sizes, unlike the two scale flags: see
    # the class docstring. Defaulted True, so a direct engine caller that says nothing
    # keeps today's behaviour (`dn_dc_supplied`'s not-a-verdict direction).
    blank_measured: bool = True
    # True on a per-mode clone the Sedlák partition builds (`mode_partition.partition_rayleigh`):
    # dR is then one mode's share of the excess intensity, and every fit built on it carries
    # the flag forward so the mode-resolved values are named as such (invariant 7).
    mode_resolved: bool = False
    # False when a calibration constant was applied with the calibration standard's
    # refractive index blank, so `calibrated` is False for that cause alone (the scale gates
    # name it `no_standard_index`). True otherwise, an absent k_c included: the index
    # corrects a calibration, so without one the scale is arbitrary whatever the index is.
    standard_index_known: bool = True
    # On a per-mode clone (`mode_partition.partition_rayleigh`): that mode's counts over
    # the points its share was built on; None on any other series. A fit that reads them
    # names and folds them through `mode_point_notes`.
    mode_counts: Optional[ModePointCounts] = None
    # The three gates above as precondition notes (`scale_notes`), for a surface that shows
    # this series itself: no calibration and an unmeasured blank are about dR and the
    # ordinate, no usable dn/dc about the ordinate alone. Minted by
    # `compute_excess_rayleigh_ratio` from the flags they restate; a clone (`replace` of
    # its arrays) keeps both. The fits built on this series mint their own.
    notes: Tuple[Note, ...] = ()


# ===========================================================================
# Calibration: one unified model (compute our own k_c; never trust vendor values)
# ===========================================================================

def compute_calibration_constant(
    calibrant_intensity: float,
    calibrant_angle_deg: float,
    standard_rayleigh_ratio: float,
    dark_count_rate: float = 0.0,
) -> float:
    """Compute the calibration constant k_c from a single calibrant measurement.

    The instrument is calibrated by measuring a standard liquid (usually toluene)
    of known Rayleigh ratio, typically once at 90 deg. The constant that converts
    measured intensity to absolute Rayleigh ratio is

        k_c = R_standard / (I_calibrant_net * sin(theta_cal))

    where I_calibrant_net is dark-subtracted. (At 90 deg, sin = 1.) This single
    value is then used for every sample, regardless of which instrument produced
    the data -- one calibration model for all sources. Vendor-stored calibration
    constants are never used; compute your own here.

    The standard's Rayleigh ratio must be supplied in the SAME scattering geometry
    in which the calibrant was measured (e.g. R_VU for a no-analyzer instrument).
    Get it from physics.constants.rayleigh_ratio_toluene(wavelength, T, geometry).

    Parameters
    ----------
    calibrant_intensity : float
        Measured scattered intensity of the pure standard liquid.
    calibrant_angle_deg : float
        Angle at which the calibrant was measured (usually 90).
    standard_rayleigh_ratio : float
        Rayleigh ratio of the standard in cm^-1, in the measurement geometry.
    dark_count_rate : float
        Dark count subtracted from the calibrant intensity. Default 0.

    Returns
    -------
    float
        The calibration constant k_c (cm^-1 per unit intensity).

    Raises
    ------
    ValueError
        If the net intensity is not a positive finite number, or the calibrant angle is not
        strictly between 0 and 180 deg (where the sin-theta scattering-volume
        factor degenerates to zero).
    """
    I_net = calibrant_intensity - dark_count_rate
    if not (math.isfinite(I_net) and I_net > 0):
        raise ValueError(
            f"Net calibrant intensity must be a positive finite number, got {I_net} "
            f"(intensity {calibrant_intensity} - dark {dark_count_rate})."
        )
    # Guard on the angle directly, not on sin(theta): at exactly 180 deg
    # math.sin(math.radians(180)) is ~1.2e-16 (float round-off), not 0, so a
    # `sin <= 0` test would let the degenerate back-scatter geometry through.
    if not (0.0 < calibrant_angle_deg < 180.0):
        raise ValueError(
            f"calibrant_angle must be strictly between 0 and 180 deg (the "
            f"sin-theta factor degenerates to 0 at 0 and 180 deg), got "
            f"{calibrant_angle_deg} deg."
        )
    s = math.sin(math.radians(calibrant_angle_deg))
    return standard_rayleigh_ratio / (I_net * s)


def compute_excess_rayleigh_ratio(
    sample: SLSMeasurement,
    solvent_reference: SLSMeasurement,
    calibration_constant: Optional[float] = None,
    standard_refractive_index: Optional[float] = None,
    dark_count_rate: float = 0.0,
    transmittance: Optional[Sequence[float]] = None,
    derived_provenance: str = '',
    blank_provenance: str = '',
    blank_measured: bool = True,
) -> RayleighRatioResult:
    """Convert raw intensities to an excess Rayleigh ratio dR(theta).

    One unified pipeline for all data sources:
      1. dark-count subtraction:   I_net = I_raw - dark    (sample and solvent alike;
                                                            see below)
      2. solvent subtraction:      I_excess = I_net,sample - I_net,solvent
      3. volume correction:        x sin(theta)            (always; see below)
      4. absorbance correction:    / transmittance         (if transmittance given;
                                                            not reachable from the app)
      5. calibration:              x k_c x (n_s/n_cal)^2
    with dR(theta) = k_c * sin(theta) * I_excess * (n_s/n_cal)^2.

    The calibration constant k_c is the single unifying quantity. Compute it from
    a calibrant measurement with `compute_calibration_constant`, or pass a value
    you determined independently. A vendor-stored constant is NEVER used here.

    Where the dark count survives
    -----------------------------
    Step 1 takes the same scalar d off the sample and the solvent reference, so against
    a MEASURED blank it cancels in step 2 and I_excess does not depend on it. It
    survives only in k_c = R_std / ((I_cal - d) sin(theta_cal)), when k_c was computed
    with it (`compute_calibration_constant`): a constant scale on dR, so it moves Mw and
    absolute A2 and leaves Rg, the form-factor sizes and 2 A2 Mw as they are. When no
    blank was measured it survives in the excess as well, because the reference is
    built from it. A typed 90 deg blank is d + (I90 - d) / sin(theta)
    (`app.controller.Controller._typed_blank_for`), which leaves d (1 - sin(theta)) in
    I_excess sin(theta); with no blank at all the reference is d at every angle, so the
    sample alone loses it, leaving -d sin(theta). Both terms are angle-dependent, so
    there d moves the angular shape of dR, and Rg with it, not only its scale.

    An absent dn/dc, and why K = 1 rather than a refusal
    ---------------------------------------------------
    `sample.dn_dc_mL_per_g` may be None -- for many real samples no increment exists
    (no measured value has ever been published for the polymer) or the quantity is not
    defined at all (particle dispersions). It enters only the optical constant K, which
    multiplies the WHOLE Kc/dR ordinate, so it cancels in every coefficient ratio: each
    Rg, the form-factor radii and axial ratio, and the calibration-free 2 A2 Mw are all
    unchanged by it. Guinier's Rg does not merely cancel it, it never sees it -- that fit
    reads dR, and dn/dc does not enter dR.

    So an absent dn/dc sets K = 1 and reports `dn_dc_supplied = False` plus a note, the
    same treatment an absent calibration constant gets. The alternative -- refusing, by
    leaving K undefined -- would make the ordinate all-NaN and kill every fit built on
    it, including the form-factor fits that the particle samples with no defined dn/dc
    are precisely the ones needing. The cost of K = 1 is one more reliability gate to
    carry, which is cheap; the cost of refusing is the analysis the user came for.

    The same path catches a dn/dc that is present but UNUSABLE. K is proportional to
    (dn/dc)^2 (and to n^2 and lambda^-4), so a perfectly legal finite nonzero increment
    can still underflow K to zero or overflow it to infinity -- measured, dn/dc = 1e-160
    alone gives K = 0.0 exactly, which would zero the whole ordinate and destroy Rg, a
    SCALE-FREE output, through a scale-only parameter. A predicate on dn/dc could not
    close that: the edge depends on n and lambda too, so it could be mis-sized three
    ways. Checking K itself closes all three with one computation, which is why the
    measurement's own entry guard rejects only the one value that is never a measurement
    (zero) and leaves the range question here.

    The resulting Mw is absurd by roughly seven orders of magnitude, and that is
    protective rather than dangerous: it is flagged, noted, and obviously wrong at a
    glance, where the `dn/dc = 0` workaround this replaced produced a plausible-looking
    Mw flagged reliable.

    Why step 3 is unconditional, and must stay that way
    --------------------------------------------------
    It used to be switchable (`apply_sin_theta`, since removed) -- residue of a retired
    two-route calibration model, and a trap. The sin(theta) here is
    STRUCTURALLY PAIRED with the one inside k_c: `compute_calibration_constant`
    returns R_std / (I_cal,net * sin(theta_cal)), so k_c already carries a
    1 / sin(theta_cal). Skipping the multiplication here leaves that factor with
    nothing to cancel it.

    The damage is quiet. Calibration is normally done at 90 deg, where sin = 1, so the
    k_c side looks innocent and the whole error lands on the sample side: dR wrong by a
    factor of sin(theta) at every angle except 90. That is ANGLE-DEPENDENT, so it
    corrupts the q^2 dependence -- **Rg** comes out wrong, not just Mw -- with nothing
    to flag it. A geometry that genuinely needs different treatment (flat cell, fibre
    optic) needs a designed calibration path of its own, not a boolean here.
    Pinned by `test_dR_recovers_a_known_R_and_is_invariant_to_the_calibrant_angle`.

    If `calibration_constant` is None, the analysis still runs but is flagged
    UNCALIBRATED: dR is placed on an arbitrary scale (k_c = 1). Scale-independent
    results (Rg; the calibration-free 2 A2 Mw) remain valid; Mw and absolute A2
    do not, and the downstream results say so.

    Parameters
    ----------
    sample : SLSMeasurement
        Solution at one concentration, across angles.
    solvent_reference : SLSMeasurement
        Pure-solvent measurement (concentration 0) at the SAME angles.
    calibration_constant : float, optional
        k_c (cm^-1 per unit intensity). None -> uncalibrated (flagged).
    standard_refractive_index : float, optional
        Refractive index of the calibration standard, for the (n_s/n_cal)^2
        correction -- a MANDATORY step of the calibration whenever the solvent
        differs from the standard (CLAUDE.md, physics enforced in code). If omitted
        the correction cannot be applied: the factor is 1, a note is recorded, and
        the result is flagged `calibrated=False` exactly as an absent k_c is -- the
        absolute scale is not established either way (invariant 4: a missing
        required physical input fails the precondition; the value is still computed
        and reported, never withheld). Before 2026-08-26 this absence only wrote the
        note, and every downstream Mw/A2 still read reliable (+26 % measured for
        water against a toluene standard). With k_c applied, `standard_index_known`
        records that the index was the cause, and the scale gates name it.
    dark_count_rate : float
        Taken off the sample and the solvent reference alike (step 1), so it cancels
        against a measured blank; see "Where the dark count survives". Default 0.
    transmittance : sequence of float, optional
        Per-angle ROUND-TRIP internal transmittance for absorbing samples: the fraction
        of light surviving the incident path into the scattering volume TIMES the
        fraction surviving the exit path at that angle. dR is divided by it ONCE, so
        the single number supplied per angle must already be that product. Omit for
        non-absorbing samples.

        **This is not a single-pass spectrophotometric reading.** A cuvette T measured
        straight through covers the incident leg only; supplying it here under-corrects
        by roughly the exit leg's own factor -- and the same quantity divides straight
        into Mw and A2 (below), so the error is not cosmetic. Whoever wires this up owns
        deriving the round-trip factor from the cell geometry.

        **Supported here, but not currently reachable from the application** -- no
        parser reads a transmittance and no GUI control supplies one, so nothing
        passes this today. Kept because it is a real correction and this program's
        target solvents include coloured deep eutectics. Forward-ready, in the sense
        `analyzer_geometry` is in core/workspace.py.

        For whoever wires it up, four things:

        (1) NO SOURCE OF RECORD. The correction is implemented here but is cited
        nowhere -- no primary-source reference anywhere in the project's citation
        records. Under the primary-source gate that makes it
        UNVERIFIED, so acquiring the authoritative reference is a prerequisite for
        wiring it up -- not a follow-up. Do not reconstruct the correction from a
        plausible restatement; the gate exists because that has produced a real bug here
        before.

        (2) Transmittance is a per-sample MEASURED quantity, so invariant 3 applies
        with full force. It must arrive as a user-supplied, provenance-tagged value
        through an explicit commit -- never defaulted, guessed, or derived from a
        solvent library. It multiplies straight into Mw and A2, which is exactly the
        rule dn/dc lives under, and for the same reason.

        (3) RANGE IS UNGUARDED. Physically 0 < T <= 1, but only the array's LENGTH is
        checked. `T = 0` yields `inf`, a negative T flips dR's sign, and a T > 1
        amplifies rather than corrects -- all silently, none flagged. A wire-up owes
        the value guard, not just the field.

        (4) KNOWN INCONSISTENCY, measured 2026-08-05, unreachable today: this array is
        indexed against the pipeline's ANGLE-SORTED arrays, while `intensities` is
        reordered to match (`s_order`). So a caller who lists angles out of ascending
        order and writes transmittance parallel to their own angle list gets it
        silently misapplied to the wrong angles -- a real, non-obvious trap the moment
        anything starts supplying this. Give it in ascending-angle order until that is
        settled. The one-line remedy (reorder with `s_order`, like every sibling array)
        deserves its own decision rather than riding along in an unrelated change.

    Returns
    -------
    RayleighRatioResult

    Raises
    ------
    ValueError
        If sample and solvent angles differ, or transmittance has the wrong length.
    """
    if (sample.angles_deg.shape != solvent_reference.angles_deg.shape
            or not np.allclose(np.sort(sample.angles_deg),
                               np.sort(solvent_reference.angles_deg))):
        raise ValueError(
            "Sample and solvent_reference must be measured at the same angles."
        )

    s_order = np.argsort(sample.angles_deg)
    r_order = np.argsort(solvent_reference.angles_deg)
    angles = sample.angles_deg[s_order]
    I_sample = sample.intensities[s_order] - dark_count_rate
    I_solvent = solvent_reference.intensities[r_order] - dark_count_rate
    I_excess = I_sample - I_solvent

    # Unconditional, and paired with the 1/sin(theta_cal) inside k_c -- see the
    # docstring. `sin_theta_applied` is therefore always True; it is retained rather than
    # dropped so the result-field family stays uniform.
    I_excess = I_excess * np.sin(np.radians(angles))
    sin_applied = True

    if transmittance is not None:
        t = np.asarray(transmittance, dtype=float)
        if t.shape != angles.shape:
            raise ValueError(
                f"transmittance must have one value per angle ({angles.size}), "
                f"got {t.size}."
            )
        I_excess = I_excess / t

    # Calibration constant (computed elsewhere) or uncalibrated fallback.
    note = ''
    if calibration_constant is None:
        k_c = 1.0
        calibrated = False
        note = ('UNCALIBRATED: no calibration constant supplied; dR is on an '
                'arbitrary scale. Mw and absolute A2 are unreliable; Rg and the '
                'calibration-free 2 A2 Mw remain valid.')
    else:
        k_c = float(calibration_constant)
        calibrated = True

    # (n_s/n_cal)^2 correction when the standard's refractive index is known.
    if standard_refractive_index is not None:
        ri_corr = refractive_index_correction(sample.solvent_refractive_index,
                                              standard_refractive_index)
    else:
        ri_corr = 1.0
        # No standard refractive index -> the (n_s/n_cal)^2 correction was genuinely
        # skipped: a missing REQUIRED input of the calibration, so it drops `calibrated`
        # the way an absent k_c does (the one scale gate every downstream
        # mw_reliable/a2_reliable reads) -- flag, never withhold. Not gated on the
        # solvent's actual n ("is it toluene?"): without the standard's n we cannot know
        # whether the solvent matches it, and a hard-coded n would be a solvent-specific
        # magic number (invariants 3 & 4). k_c, when given, is still applied -- dR is
        # then as close as the inputs allow, and `k_c_used` records it.
        calibrated = False
        extra = (' NO STANDARD REFRACTIVE INDEX: the (n_s/n_cal)^2 correction could '
                 'not be applied, so the absolute scale of dR is not established and '
                 'Mw / absolute A2 are unreliable; supply standard_refractive_index '
                 '(Rg and the calibration-free 2 A2 Mw are unaffected).')
        note = (note + extra) if note else extra.strip()

    dR = k_c * I_excess * ri_corr   # cm^-1 (arbitrary scale if uncalibrated)

    # Optical constant, or the arbitrary-optical-scale fallback. Mirrors the k_c branch
    # above deliberately: an absent dn/dc and an absent calibration constant are the SAME
    # kind of absence -- an unknown positive factor on the whole Kc/dR ordinate -- and two
    # absences of one kind must not behave differently (rule 10). K = 1 is a disclosed
    # scale placeholder, not a proposed physical value (invariant 3); see this function's
    # docstring for why the flag carries the caveat rather than a refusal.
    K = (None if sample.dn_dc_mL_per_g is None
         else optical_constant_K(sample.solvent_refractive_index,
                                 sample.dn_dc_mL_per_g, sample.wavelength_nm))
    # Guard the RESULT, not the input: K goes as (dn/dc)^2, n^2 and lambda^-4, so a legal,
    # finite, nonzero dn/dc can still drive it to 0 or inf. An unusable K is then treated
    # exactly as an absent one -- the optical scale is unknown either way -- and the note
    # `dn_dc_supplied` covers both, and every user-facing surface says "no USABLE dn/dc"
    # rather than "no dn/dc" so the sentence is true either way -- a supplied increment
    # that collapses K is not a usable one. The note below records which of the two it
    # was, for a reader of the result object; it is not a display surface.
    # See this function's docstring for why the check is on K rather than on dn/dc.
    if K is None or not (math.isfinite(K) and K > 0):
        collapsed = K is not None
        K = 1.0
        dn_dc_supplied = False
        extra_dn_dc = (
            (' UNUSABLE dn/dc: the supplied refractive-index increment gives an optical '
             'constant K that is not a usable positive number (it underflowed to zero or '
             'overflowed), so K was set to 1.'
             if collapsed else
             ' NO dn/dc SUPPLIED: the optical constant K was set to 1,')
            + ' Kc/dR is therefore on an arbitrary optical scale: Mw and absolute A2 are '
              'unreliable, while Rg, the form-factor sizes and the calibration-free '
              '2 A2 Mw remain valid.')
        note = (note + extra_dn_dc) if note else extra_dn_dc.strip()
    else:
        dn_dc_supplied = True
    c = sample.concentration_g_per_mL
    with np.errstate(divide='ignore', invalid='ignore'):
        kc_over_dR = np.where(np.isfinite(dR) & (dR != 0), K * c / dR, np.nan)

    q = np.array([
        scattering_vector_q(float(a), sample.wavelength_nm,
                            sample.solvent_refractive_index)
        for a in angles
    ], dtype=float)

    result = RayleighRatioResult(
        concentration_g_per_mL=c,
        temperature_K=sample.temperature_K,
        wavelength_nm=sample.wavelength_nm,
        solvent_refractive_index=sample.solvent_refractive_index,
        dn_dc_mL_per_g=sample.dn_dc_mL_per_g,
        angles_deg=angles,
        q_nm_inv=q,
        q2_nm2=q ** 2,
        excess_rayleigh_cm_inv=dR,
        kc_over_dR_mol_per_g=kc_over_dR,
        optical_constant_K=K,
        calibrated=calibrated,
        dn_dc_supplied=dn_dc_supplied,
        k_c_used=k_c,
        ri_correction=ri_corr,
        dark_count_subtracted=float(dark_count_rate),
        sin_theta_applied=sin_applied,
        calibration_detail=note,
        derived_provenance=str(derived_provenance or ''),
        blank_provenance=str(blank_provenance or ''),
        # Copied through, never read here: this function's arithmetic is identical
        # whichever way the blank was obtained, which is the point -- a typed blank
        # arrives as an ordinary SLSMeasurement and the pipeline gains no branch.
        blank_measured=bool(blank_measured),
        standard_index_known=not (calibration_constant is not None
                                  and standard_refractive_index is None),
    )
    # The ordinate Kc/dR carries both scales, as an Mw does; dR the calibration alone.
    return replace(result, notes=scale_notes(result, mw=('kc_over_dR_mol_per_g',),
                                             excess=('excess_rayleigh_cm_inv',)))


# ===========================================================================
# Pass 2: single-concentration and single-angle (apparent) analyses
# ===========================================================================
#
# These operate on one concentration. They yield APPARENT quantities -- not
# extrapolated to infinite dilution -- and every result says so explicitly, so an
# apparent Mw is never mistaken for the thermodynamic Mw.


# The largest q*Rg at which a small-qRg angular expansion is read as holding: the default of
# every qRg regime verdict in the SLS family, one default for one physics condition (rule 10;
# `tests/test_sls.py::test_the_qrg_verdicts_and_the_setting_share_one_default` pins its
# readers, and the controller seeds each call per run from `settings.guinier_qrg_max`, pinned
# by `tests/test_workspace.py::test_every_controller_qrg_verdict_call_is_seeded_from_the_setting`).
# The figure the cited source prints for the initial-slope Rg and for Guinier's law alike:
# qRg < 1 (Theory-and-Equations-Guide §4.4, §4.6). A general criterion (invariant 4).
QRG_MAX_VALID_DEFAULT = 1.0


@dataclass
class DebyeResult:
    """Single-concentration Debye/Guinier analysis: Kc/dR vs q^2.

    All quantities are APPARENT (at this finite concentration): the intercept is
    1/Mw + 2 A2 c, not 1/Mw. Rg is the apparent radius of gyration. A full Zimm or
    Berry analysis over several concentrations is needed for thermodynamic values.

    `notes` holds every note of the fit as `core.notes.Note` (its precondition failures and
    its records), as on every SLS fit result below.
    """
    concentration_g_per_mL: float
    q2_nm2: np.ndarray
    kc_over_dR: np.ndarray
    # linear fit  Kc/dR = intercept + slope * q^2
    intercept_mol_per_g: float        # = 1/Mw_app + 2 A2 c
    slope: float                      # = (1/Mw)(Rg^2/3) at this c
    mw_apparent_g_per_mol: float      # 1 / intercept
    rg_apparent_nm: float             # sqrt(3 * slope / intercept)
    r_squared: float
    is_apparent: bool                 # always True here
    n_angles: int
    calibrated: bool                  # was the Rayleigh ratio calibrated?
    mw_reliable: bool                 # False if uncalibrated (Rg stays reliable)
    mw_apparent_se: Optional[float] = None   # statistical (regression) SEs --
    rg_apparent_se: Optional[float] = None   # exclude calibration/dn-dc systematics
    se_estimator: str = 'hc3'                 # covariance estimator behind the SEs
    # numerical-health of the linear design (Kc/dR vs q^2). well_conditioned=False ->
    # near-degenerate angles -> mw_reliable is suppressed (value still shown).
    well_conditioned: bool = True
    cond: float = float('nan')                # condition number of the design
    # The 1/coefficient family flag for the SIZE (rule 10; the shape
    # `mode_partition.AmplitudeGuinierResult` set): Rg^2 = 3*slope/intercept divides
    # by a fitted coefficient of the SAME near-singular solve `well_conditioned` judges,
    # so a degenerate design (or no extractable size) drops this flag -- the value still
    # reports (flag, don't NaN). Scale-free: neither `calibrated` nor `dn_dc_supplied`
    # enters it. It also carries the qRg regime below, as the Zimm Rg's does.
    rg_reliable: bool = True
    # The qRg regime of the first-order angular term (`debye_analysis`), the
    # `ZimmBerryResult` shape: the largest fitted q times the apparent Rg, the limit it was
    # judged against (echoed), and whether it held -- False also when no Rg was extracted.
    # A measured qRg past the limit drops `mw_reliable` as well; no Rg does not.
    qrg_max: float = float('nan')
    qrg_max_valid: float = QRG_MAX_VALID_DEFAULT
    qrg_regime_valid: bool = True
    # Every note of the fit (preconditions and records), as Notes; the non-positive Kc/dR
    # drop is also emitted via warnings.warn, its message the note's export clause.
    notes: Tuple[Note, ...] = ()
    # `calibrated`'s sibling for the OPTICAL scale (see RayleighRatioResult): False ->
    # no dn/dc was supplied, so K was 1 and Mw/A2 here are on an arbitrary optical
    # scale. The angular sizes are untouched -- K multiplies the whole Kc/dR ordinate
    # and cancels in every coefficient ratio.
    dn_dc_supplied: bool = True
    # Carried forward from the Rayleigh result(s) this fit consumed, so a number
    # cannot be read without its origin whichever exporter writes it. A record; it
    # enters no computation. Empty for an ordinary loaded measurement.
    derived_provenance: str = ''
    # Which solvent blank the contributing points were referenced against, when
    # it is not the sample reference; copied from the Rayleigh result(s) exactly as
    # derived_provenance is, and exported beside it. '' for the common case.
    blank_provenance: str = ''
    # Carried from the Rayleigh result(s) this fit consumed: False -> the blank was not
    # measured at every angle (typed 90 deg value, or no subtraction). ANDed into every
    # reliability verdict on this result, because an additive blank does not cancel the
    # way an unknown SCALE does. See RayleighRatioResult.blank_measured.
    blank_measured: bool = True
    # Carried from the Rayleigh result(s) this fit consumed: False -> k_c was applied with
    # the standard's refractive index blank, so `calibrated` is False for that cause alone
    # (the scale gates name it). See RayleighRatioResult.standard_index_known.
    standard_index_known: bool = True
    # Carried from the Rayleigh result: True when it is one mode's share of the excess
    # intensity (RayleighRatioResult.mode_resolved), so the values here are mode-resolved.
    mode_resolved: bool = False


@dataclass
class SingleAngleResult:
    """Single-angle, single-concentration apparent molecular weight.

    Mw_app = dR / (K c) at one angle and concentration -- no angular or
    concentration extrapolation. With no angular information there is no Rg, and
    the value still contains the form factor P(q) and the 2 A2 c term. Strictly an
    order-of-magnitude / apparent quantity.
    """
    concentration_g_per_mL: float
    angle_deg: float
    q2_nm2: float
    mw_apparent_g_per_mol: float
    is_apparent: bool                 # always True here
    calibrated: bool = True           # was the Rayleigh ratio calibrated?
    mw_reliable: bool = True          # False if uncalibrated (Mw_app on an arbitrary scale)
    # numerical-health of the 1/y reciprocal (no design matrix here): a subnormal
    # divisor y = Kc/dR loses catastrophic precision in 1/y -> mw_reliable suppressed.
    well_conditioned: bool = True
    cond: float = float('nan')                # NaN: no design matrix for a single datum
    # Every note of the fit, as Notes (this method has no drop/warn site of its own, so
    # only preconditions land here). See DebyeResult.notes.
    notes: Tuple[Note, ...] = ()
    # `calibrated`'s sibling for the OPTICAL scale (see RayleighRatioResult): False ->
    # no dn/dc was supplied, so K was 1 and Mw/A2 here are on an arbitrary optical
    # scale. The angular sizes are untouched -- K multiplies the whole Kc/dR ordinate
    # and cancels in every coefficient ratio.
    dn_dc_supplied: bool = True
    # Carried forward from the Rayleigh result(s) this fit consumed, so a number
    # cannot be read without its origin whichever exporter writes it. A record; it
    # enters no computation. Empty for an ordinary loaded measurement.
    derived_provenance: str = ''
    # Which solvent blank the contributing points were referenced against, when
    # it is not the sample reference; copied from the Rayleigh result(s) exactly as
    # derived_provenance is, and exported beside it. '' for the common case.
    blank_provenance: str = ''
    # Carried from the Rayleigh result(s) this fit consumed: False -> the blank was not
    # measured at every angle (typed 90 deg value, or no subtraction). ANDed into every
    # reliability verdict on this result, because an additive blank does not cancel the
    # way an unknown SCALE does. See RayleighRatioResult.blank_measured.
    blank_measured: bool = True
    # Carried from the Rayleigh result(s) this fit consumed: False -> k_c was applied with
    # the standard's refractive index blank, so `calibrated` is False for that cause alone
    # (the scale gates name it). See RayleighRatioResult.standard_index_known.
    standard_index_known: bool = True


@dataclass
class GuinierResult:
    """Single-concentration Guinier analysis: ln(dR) vs q^2.

    The Guinier approximation dR(q) = dR(0) exp(-q^2 Rg^2 / 3) linearizes as
    ln(dR) = ln(dR(0)) - (Rg^2/3) q^2, so a straight-line fit gives Rg from the
    SLOPE (Rg = sqrt(-3*slope)) and an apparent Mw from the INTERCEPT
    (dR(0) = K c Mw). Like Debye, Rg is scale-independent -- it comes from the
    slope of a log-intensity plot, so it is reliable even when the run is
    uncalibrated; only the intercept-derived Mw needs calibration.

    Valid only in the Guinier regime, q*Rg < 1 by default (`QRG_MAX_VALID_DEFAULT`);
    `guinier_valid` flags this from
    the largest q used, against the limit echoed in `qrg_max_valid` (a per-run
    control seeded from Settings -- recorded on the result so every surface names the
    bound the verdict actually read). A False `guinier_valid` always drops `rg_reliable`
    and names itself in a precondition note (invariant 4). It drops `mw_reliable` only
    when qRg was MEASURED past the limit: the other cause of a False verdict is a qRg
    that could not be measured at all (a non-negative slope, so no Rg), and an unmeasured
    qRg never exceeded the bound the intercept's precondition names -- see
    `regime_ok_for_mw`. Both values are still reported either way. Single
    concentration -> the values are apparent.
    """
    concentration_g_per_mL: float
    q2_nm2: np.ndarray
    ln_excess_rayleigh: np.ndarray    # ln(dR) at the fitted angles
    slope: float                      # d ln(dR) / d q^2  (= -Rg^2/3), nm^2
    intercept: float                  # ln(dR(0))
    rg_nm: float                      # sqrt(-3*slope); nan if slope >= 0
    mw_apparent_g_per_mol: float      # exp(intercept) / (K c); nan if K or c <= 0
    qrg_max: float                    # q*Rg at the largest fitted angle
    guinier_valid: bool               # qrg_max <= qrg_max_valid
    r_squared: float
    n_angles: int
    is_apparent: bool                 # always True here
    calibrated: bool                  # was the Rayleigh ratio calibrated?
    mw_reliable: bool                 # False if uncalibrated (Rg stays reliable)
    rg_se: Optional[float] = None            # statistical (regression) SEs --
    mw_apparent_se: Optional[float] = None   # exclude calibration/dn-dc systematics
    se_estimator: str = 'hc3'                 # covariance estimator behind the SEs
    # numerical-health of the linear design (ln(dR) vs q^2). well_conditioned=False ->
    # near-degenerate angles -> mw_reliable is suppressed (value still shown).
    well_conditioned: bool = True
    cond: float = float('nan')                # condition number of the design
    # The qRg limit `guinier_valid` was judged against -- echoed, as its sibling
    # `mode_partition.AmplitudeGuinierResult.qrg_max_valid` is, so a display can never
    # print a bound other than the one the predicate read (Settings seed it; the
    # default here matches `guinier_analysis`'s).
    qrg_max_valid: float = QRG_MAX_VALID_DEFAULT
    # The SIZE's own flag (rule 10; see DebyeResult.rg_reliable): False on a degenerate
    # design, no extractable size, or a fit outside the Guinier regime -- the precondition
    # `guinier_valid` reports. Scale-free.
    rg_reliable: bool = True
    # Every note of the fit, as Notes -- the drop of angles whose excess Rayleigh ratio is
    # not positive (ln(dR) needs dR > 0) among them. See DebyeResult.notes.
    notes: Tuple[Note, ...] = ()
    # `calibrated`'s sibling for the OPTICAL scale (see RayleighRatioResult): False ->
    # no dn/dc was supplied, so K was 1 and Mw/A2 here are on an arbitrary optical
    # scale. The angular sizes are untouched -- K multiplies the whole Kc/dR ordinate
    # and cancels in every coefficient ratio.
    dn_dc_supplied: bool = True
    # Carried forward from the Rayleigh result(s) this fit consumed, so a number
    # cannot be read without its origin whichever exporter writes it. A record; it
    # enters no computation. Empty for an ordinary loaded measurement.
    derived_provenance: str = ''
    # Which solvent blank the contributing points were referenced against, when
    # it is not the sample reference; copied from the Rayleigh result(s) exactly as
    # derived_provenance is, and exported beside it. '' for the common case.
    blank_provenance: str = ''
    # Carried from the Rayleigh result(s) this fit consumed: False -> the blank was not
    # measured at every angle (typed 90 deg value, or no subtraction). ANDed into every
    # reliability verdict on this result, because an additive blank does not cancel the
    # way an unknown SCALE does. See RayleighRatioResult.blank_measured.
    blank_measured: bool = True
    # Carried from the Rayleigh result(s) this fit consumed: False -> k_c was applied with
    # the standard's refractive index blank, so `calibrated` is False for that cause alone
    # (the scale gates name it). See RayleighRatioResult.standard_index_known.
    standard_index_known: bool = True
    # Carried from the Rayleigh result (see DebyeResult.mode_resolved).
    mode_resolved: bool = False


def _nonpositive_ordinate_note(n_drop: int, ordinate: str,
                               about: Tuple[str, ...] = ()) -> Note:
    """One note for every SLS arm that drops non-positive points.

    A negative excess Rayleigh ratio (hence a negative Kc/dR) is the ordinary
    signature of an over-subtracted solvent blank -- common at high angle in the
    low-contrast systems this program targets -- and no linear model here has a
    meaning for it: a solution cannot scatter less than its own solvent. Every arm
    (Debye, Guinier, Zimm, Berry, the ladder, the calibration-free A2 and the three
    form-factor models) drops such points and records it on the result's `notes`, and
    via warnings.warn with the note's export clause as the message, where before only
    Berry did (the square root forced it) and the others fitted them silently -- Debye
    reported an 11.5x wrong apparent Mw as reliable on one measured input.
    """
    return Note('nonpositive_points_dropped', about,
                {'n': int(n_drop), 'ordinate': ordinate})


def _warn_note(note: Note, stacklevel: int) -> None:
    """Emit a note as a RuntimeWarning for a headless caller, its message the note's
    export clause, so the warning and the note keep one source."""
    warnings.warn(render(note, surface='export'), RuntimeWarning, stacklevel=stacklevel + 1)


def _drop_nonpositive_ordinate(label: str, q2, c, y, notes: list):
    """Drop the non-positive Kc/dR points of a flattened design and disclose it -- the
    linear arms' rule applied to the form-factor arms, which fit the same ordinate:
    1/(Mw P(q)) + 2 A2 c has no meaning at or below zero either. Keep-both: the note on
    the result, the warning for a headless caller. Returns the surviving (q2, c, y)."""
    good = y > 0
    n_drop = int((~good).sum())
    if n_drop == 0:
        return q2, c, y
    if not good.any():
        # Nothing would survive. An all-ZERO set (an underflowed ordinate) is handed on
        # whole so _ordinate_scale names the degeneracy; a set with a NEGATIVE point
        # would pass that guard (it reads |y|) and be fitted, so it is refused here,
        # naming the cause like the linear arms' post-drop refusals do.
        if not (y < 0).any():
            return q2, c, y
        raise ValueError(
            f"{label} analysis: every one of the {n_drop} Kc/dR point(s) is "
            "non-positive; a solution cannot scatter less than its own solvent. "
            "Check the solvent blank."
        )
    note = _nonpositive_ordinate_note(n_drop, 'Kc/ΔR')
    notes.append(note)
    _warn_note(note, stacklevel=4)   # through _form_factor_inputs to the entry point's caller
    return q2[good], c[good], y[good]


def _shared_derived_provenance(rayleigh_results) -> str:
    """The derivation provenance the whole set agrees on, or a mixed-set statement and
    every distinct record.

    A fit's provenance describes every point that went into it, so one record is stated
    alone only when every contributing result carries the SAME one. Otherwise the set
    says it is mixed and then names every distinct record, joined, so each derived
    point's runs and its count-rate spread survive the pooling (the way
    `_shared_blank_provenance` names every blank). Two counts are needed and they answer
    different questions -- how many DISTINCT strings there are, and how many results
    CARRY one -- because deduplicating collapses the second.

    Takes a SEQUENCE only. A single result reads its own field directly at the call
    site -- one input shape, so there is no type test here to mis-size (the first
    draft keyed on isinstance(RayleighRatioResult), which a duck-typed stand-in
    walked straight past into list()).
    """
    items = list(rayleigh_results)
    seen = {str(getattr(r, 'derived_provenance', '') or '') for r in items}
    seen.discard('')
    if not seen:
        return ''
    carried = sum(1 for r in items
                  if str(getattr(r, 'derived_provenance', '') or ''))
    if len(seen) == 1 and carried == len(items):
        return seen.pop()
    return '; '.join(('some points derived from DLS count rates', *sorted(seen)))


def _shared_blank_provenance(rayleigh_results) -> str:
    """Every distinct 'which solvent blank' record the set carries, joined; '' when
    every point used the sample's reference blank. No mixed-set sentence: a blank note
    names a file, and a fit over two blanks is described by naming both."""
    notes = sorted({str(getattr(r, 'blank_provenance', '') or '')
                    for r in rayleigh_results} - {''})
    return '; '.join(notes)


def debye_analysis(rayleigh_result: RayleighRatioResult,
                   estimator: str = 'hc3',
                   cond_limit: float = unc.COND_LIMIT,
                   qrg_max_valid: float = QRG_MAX_VALID_DEFAULT) -> DebyeResult:
    """Single-concentration Debye plot: linear fit of Kc/dR vs q^2.

    Gives the apparent molecular weight (from the intercept) and apparent radius
    of gyration (from the slope-to-intercept ratio) at one concentration. Results
    are flagged apparent: the intercept is 1/Mw + 2 A2 c, not 1/Mw.

    Inputs not prescribed by the source: the conditioning ceiling ``cond_limit`` (default
    ``COND_LIMIT``, a numerical-health bound), the two-angle minimum (the line's mathematical
    minimum), the drop of a non-positive Kc/dR (stated on ``notes``) and the weighting
    (equal weights on every retained point); prescribed: the qRg regime ``qrg_max_valid``
    reads (Russo et al. 2021, pp. 517-518; Takahashi et al. 2019, pp. 1045-1046).

    Parameters
    ----------
    rayleigh_result : RayleighRatioResult
        One concentration's calibrated data (>= 2 angles).
    qrg_max_valid : float
        Upper q*Rg for the first-order angular term the apparent Rg and Mw rest on, the
        same term the Zimm plot reads (default `QRG_MAX_VALID_DEFAULT`; the controller seeds
        it from the one Setting). Past it ``rg_reliable`` and ``mw_reliable`` drop and one
        ``qrg_above_limit`` note about both names the failed precondition: the sources put
        the size under qRg < 1 (Russo et al. 2021, pp. 517-518) and the molecular weight
        too, in the Rayleigh region Rgq < 1 (Takahashi et al. 2019, p. 1045; p. 1046 places
        the Zimm plot's M at Rgq << 1). A fit with no extractable Rg has no qRg to judge, so
        its Mw verdict is not touched by the limit.

    Returns
    -------
    DebyeResult

    Raises
    ------
    ValueError
        If fewer than two finite angles are available.
    """
    q2 = rayleigh_result.q2_nm2
    y = rayleigh_result.kc_over_dR_mol_per_g
    finite = np.isfinite(q2) & np.isfinite(y)
    # Finite AND positive: Kc/dR <= 0 means dR <= 0, an over-subtracted blank. Such a
    # point has no place in a 1/Mw + ... model and, left in, dominates the least-squares
    # line (the Berry arm has always dropped this class; Debye fitted it silently).
    # Disclosed on `notes` + a warning, never a refusal (invariant 4 flag-not-refuse).
    good = finite & (y > 0)
    notes: list = []
    n_drop = int((finite & ~good).sum())
    if n_drop:
        drop = _nonpositive_ordinate_note(n_drop, 'Kc/ΔR')
        notes.append(drop)
        _warn_note(drop, stacklevel=2)
    q2, y = q2[good], y[good]
    if q2.size < 2:
        raise ValueError(
            "Debye analysis needs at least two angles with a finite, positive Kc/dR. "
            "Check the solvent blank."
        )

    fit = unc.linear_fit(q2, y, estimator, cond_limit)  # cov order [intercept, slope]
    slope, intercept = fit.slope, fit.intercept
    r2 = fit.r_squared

    # Mw_app = 1/intercept. The intercept (Kc/dR at q->0 = 1/Mw + 2 A2 c) is
    # physically positive; a noisy NON-POSITIVE intercept would give a negative or
    # infinite Mw, which is impossible. Guard it to NaN (mirroring the Rg ratio>0
    # guard) rather than present an unphysical negative Mw as a result.
    mw_app = (1.0 / intercept
              if (math.isfinite(intercept) and intercept > 0) else float('nan'))
    # Rg^2 = 3 * slope / intercept  (apparent). Also requires a PHYSICAL (positive)
    # intercept: with both slope and intercept negative, ratio>0 would otherwise report
    # an Rg extracted from a fit whose intercept was just judged unphysical (voiding Mw)
    # — guard the whole ratio on the same intercept>0, not just ratio>0.
    ratio = (slope / intercept
             if (math.isfinite(intercept) and intercept != 0) else float('nan'))
    rg_app = (math.sqrt(3.0 * ratio)
              if (math.isfinite(ratio) and ratio > 0
                  and math.isfinite(intercept) and intercept > 0) else float('nan'))

    # Statistical SEs by first-order propagation through the (intercept, slope) cov.
    mw_se = rg_se = None
    if math.isfinite(fit.intercept_se):
        if intercept > 0:                      # only when mw_app is a finite result
            mw_se = unc.se_or_none(unc.propagate(
                [-1.0 / unc.power_or_nan(intercept, 2), 0.0], fit.cov))
        if math.isfinite(rg_app) and rg_app > 0 and slope != 0 and intercept != 0:
            jac = [-rg_app / (2.0 * intercept), rg_app / (2.0 * slope)]
            rg_se = unc.se_or_none(unc.propagate(jac, fit.cov))

    # Numerical health of the Kc/dR-vs-q^2 design: a near-degenerate angle set makes
    # 1/intercept fragile even when calibrated and intercept>0 -> suppress mw_reliable
    # and carry the reason (value still shown; Rg from the slope-ratio stays reported).
    reasons = list(scale_notes(rayleigh_result, mw=('mw_apparent_g_per_mol',),
                               scale_free=('rg_apparent_nm',)))
    if not fit.well_conditioned:
        reasons.append(_ill_conditioned_note(
            fit.cond, cond_limit, ('mw_apparent_g_per_mol', 'rg_apparent_nm')))
    # The apparent Mw is withheld for its cause, the intercept, and otherwise for its own
    # non-finite reciprocal; the no-Rg note below names the size alone.
    if not (math.isfinite(intercept) and intercept > 0):
        reasons.append(_non_positive_intercept_note(intercept, ('mw_apparent_g_per_mol',)))
    elif not math.isfinite(mw_app):
        reasons.append(Note('mw_not_finite', ('mw_apparent_g_per_mol',)))
    partition = mode_point_notes(rayleigh_result,
                                 about=('mw_apparent_g_per_mol', 'rg_apparent_nm'))
    reasons += partition
    if not (math.isfinite(rg_app) and rg_app > 0):
        # The size's own failed precondition, named (invariant 4): there is no Rg to
        # report -- the slope/intercept ratio is not positive (an angular dependence
        # inside the noise, or an over-corrected blank), or the intercept the ratio
        # divides by is not. Mw from the intercept is untouched by it.
        if math.isfinite(intercept) and intercept > 0:
            reasons.append(_no_rg_note(('rg_apparent_nm',), 'slope/intercept', ratio,
                                       'nm²', 'positive'))
        else:
            reasons.append(_no_rg_note(('rg_apparent_nm',), 'intercept', intercept,
                                       'mol/g', 'positive'))
    # The qRg regime over the points fitted, keyed as `zimm_analysis` keys it: a NaN qRg
    # (no Rg) is said above, and +inf falls on the flagging side (rule 12). `regime_valid`
    # is the size's verdict, False with no Rg as well; `regime_failed` is a MEASURED qRg past
    # the limit, the one predicate behind the note and the Mw's verdict alike, since no Rg
    # is no failed precondition of the intercept.
    qrg_max = (float(math.sqrt(float(np.max(q2))) * rg_app) if math.isfinite(rg_app)
               else float('nan'))
    regime_valid = bool(math.isfinite(qrg_max) and qrg_max <= qrg_max_valid)
    regime_failed = (not math.isnan(qrg_max)) and not (qrg_max <= qrg_max_valid)
    if regime_failed:
        reasons.append(_qrg_regime_note(('rg_apparent_nm', 'mw_apparent_g_per_mol'),
                                        qrg_max, qrg_max_valid, 'Debye'))
    notes += reasons
    # The blank gate, read defensively like the provenance records beside it, so a
    # hand-built RayleighRatioResult from before this field existed reads as measured.
    blank_ok = bool(getattr(rayleigh_result, 'blank_measured', True))

    return DebyeResult(
        concentration_g_per_mL=rayleigh_result.concentration_g_per_mL,
        q2_nm2=q2, kc_over_dR=y,
        intercept_mol_per_g=float(intercept), slope=float(slope),
        mw_apparent_g_per_mol=float(mw_app), rg_apparent_nm=float(rg_app),
        r_squared=float(r2), is_apparent=True, n_angles=int(q2.size),
        calibrated=rayleigh_result.calibrated,
        dn_dc_supplied=rayleigh_result.dn_dc_supplied,
        derived_provenance=str(getattr(rayleigh_result, 'derived_provenance', '') or ''),
        blank_provenance=str(getattr(rayleigh_result, 'blank_provenance', '') or ''),
        blank_measured=blank_ok,
        standard_index_known=bool(getattr(rayleigh_result, 'standard_index_known', True)),
        # intercept > 0 as well as calibrated: mw_app was already NaN'd for a
        # non-positive intercept above, so it must not read reliable either (keeps
        # the sign guard and the reliability flag consistent). well_conditioned adds
        # the numerical-degeneracy gate (invariant 4: scale-invariant, not a magnitude);
        # math.isfinite closes the float-overflow edge (consistent with Guinier/Zimm).
        # dn_dc_supplied is the second SCALE gate beside `calibrated` (K = 1 puts 1/Mw_app
        # on an arbitrary optical scale); Rg, a slope/intercept ratio, takes neither. A
        # measured qRg past the limit fails the intercept as it fails the slope.
        mw_reliable=(rayleigh_result.calibrated
                     and rayleigh_result.dn_dc_supplied and blank_ok
                     and bool(math.isfinite(intercept) and intercept > 0)
                     and fit.well_conditioned and math.isfinite(mw_app)
                     and not partition and not regime_failed),
        # The size shares the design's conditioning verdict (it is a ratio of the same
        # two coefficients) but neither scale gate; NaN'd sizes are not reliable. It
        # DOES take the blank gate: an absent or assumed blank is ADDITIVE, so it
        # survives the ratio the scales cancel in.
        rg_reliable=bool(math.isfinite(rg_app) and rg_app > 0
                         and fit.well_conditioned and blank_ok and regime_valid
                         and not partition),
        qrg_max=qrg_max, qrg_max_valid=float(qrg_max_valid),
        qrg_regime_valid=regime_valid,
        mw_apparent_se=mw_se, rg_apparent_se=rg_se,
        se_estimator=estimator,
        well_conditioned=fit.well_conditioned, cond=fit.cond,
        notes=tuple(notes),
        mode_resolved=bool(getattr(rayleigh_result, 'mode_resolved', False)),
    )


def single_angle_mw(rayleigh_result: RayleighRatioResult, angle_deg: float) -> SingleAngleResult:
    """Apparent Mw from one angle of one concentration: Mw_app = dR/(Kc).

    No extrapolation in either angle or concentration. Use only as a rough check;
    the value contains both the form factor and the virial term.

    Parameters
    ----------
    rayleigh_result : RayleighRatioResult
    angle_deg : float
        Which measured angle to use (must match one in the result).

    Returns
    -------
    SingleAngleResult

    Raises
    ------
    ValueError
        If the angle is not present, or concentration is zero.
    """
    idx = np.where(np.isclose(rayleigh_result.angles_deg, angle_deg))[0]
    if idx.size == 0:
        raise ValueError(
            f"angle {angle_deg} deg is not among the measured angles "
            f"{rayleigh_result.angles_deg.tolist()}."
        )
    i = int(idx[0])
    c = rayleigh_result.concentration_g_per_mL
    if c == 0:
        raise ValueError("Cannot compute Mw at zero concentration.")
    y = rayleigh_result.kc_over_dR_mol_per_g[i]   # = K c / dR
    # Mw = 1/y requires a POSITIVE y (Kc/dR): a non-positive value (over-subtracted
    # solvent -> dR < 0) would give a negative/infinite Mw. Guard the sign, not just
    # y != 0, and don't mark such a result reliable (the 1/coefficient sign rule,
    # matching the Zimm/Debye intercept>0 guards).
    y_ok = bool(math.isfinite(y) and y > 0)   # Python bool (y is a numpy scalar; keep it native)
    mw_app = 1.0 / y if y_ok else float('nan')
    # Numerical-health of the reciprocal (no design matrix for a single datum): a
    # SUBNORMAL divisor y (below the smallest normal float, ~2.2e-308 -- e.g. from an
    # extreme intensity scale) loses catastrophic precision in 1/y and yields a
    # huge-but-finite garbage Mw. This is a float-representation criterion, scale-
    # invariant, NOT a magnitude ceiling on Mw (invariant 4). Flag it and suppress
    # reliability while still showing the number.
    tiny = float(np.finfo(float).tiny)
    well_cond = bool(y_ok and y >= tiny and math.isfinite(mw_app))
    # Every not-well_cond branch carries a note (the "well_conditioned=False => note
    # present" contract the export/GUI surfacing relies on): a non-positive divisor
    # (over-subtracted solvent -> Mw undefined) vs. a subnormal divisor (finite but
    # numerically unreliable). A healthy result has only its scale notes, if any.
    notes = list(scale_notes(rayleigh_result, mw=('mw_apparent_g_per_mol',)))
    if well_cond:
        pass
    elif not y_ok:
        notes.append(Note('kc_dr_nonpositive', ('mw_apparent_g_per_mol',),
                          {'value': float(y), 'limit': 0.0}))
    else:
        notes.append(Note('subnormal_divisor', ('mw_apparent_g_per_mol',),
                          {'value': float(y), 'limit': tiny}))
    return SingleAngleResult(
        concentration_g_per_mL=c, angle_deg=float(rayleigh_result.angles_deg[i]),
        q2_nm2=float(rayleigh_result.q2_nm2[i]),
        mw_apparent_g_per_mol=float(mw_app), is_apparent=True,
        calibrated=rayleigh_result.calibrated,
        dn_dc_supplied=rayleigh_result.dn_dc_supplied,
        derived_provenance=str(getattr(rayleigh_result, 'derived_provenance', '') or ''),
        blank_provenance=str(getattr(rayleigh_result, 'blank_provenance', '') or ''),
        blank_measured=bool(getattr(rayleigh_result, 'blank_measured', True)),
        standard_index_known=bool(getattr(rayleigh_result, 'standard_index_known', True)),
        # Mw_app = dR/(K c) here, so it is the one apparent quantity that consumes dn/dc
        # DIRECTLY rather than through a ratio -- there is no angular information for a
        # scale to cancel in. Both scale gates apply, and so does the blank gate: dR at
        # this one angle still carries whatever the blank left behind.
        mw_reliable=(rayleigh_result.calibrated
                     and rayleigh_result.dn_dc_supplied
                     and bool(getattr(rayleigh_result, 'blank_measured', True))
                     and well_cond),
        well_conditioned=well_cond, notes=tuple(notes),
    )


def single_angle_ladder(
    rayleigh_results: Sequence[RayleighRatioResult],
    angle_deg: float,
    estimator: str = 'hc3',
    cond_limit: float = unc.COND_LIMIT,
) -> SingleAngleLadderResult:
    """Apparent Mw and A2 from a concentration series at ONE angle.
    (Theory-and-Equations-Guide, "One angle over a concentration series" -- the fixed-angle Debye plot)

    The classical Debye plot at fixed angle [Debye 1947, Eq. (5)]: a straight line
    Kc/dR_theta = 1/Mw_app + 2 A2_app c over the ladder, Mw from the intercept and
    A2 from the slope. This is the analysis a single-angle instrument -- or the
    Derived Intensity route, which materialises exactly one angle per
    concentration -- can reach; `zimm_analysis` needs two angles and refuses.

    Everything is APPARENT (invariant 7): there is no q -> 0 extrapolation, so the
    intercept is 1/(Mw P(q)) and the slope is angle-contingent. The canonical method
    is Debye's own (1947); the fit is the house `linear_fit` (HC3 by default), the
    reciprocal-intercept SE the delta method through the fitted covariance -- a
    deliberate house choice benchmarked conservative on a short ladder, never a
    private estimator.

    Parameters
    ----------
    rayleigh_results
        One result per concentration (c = 0 entries are skipped); each must carry
        `angle_deg` among its angles to contribute.
    angle_deg
        The one angle to read the ordinate at.

    Raises
    ------
    ValueError
        If fewer than two non-zero concentrations carry the angle, or dropping the
        non-positive Kc/dR points leaves fewer than two.

    Warns
    -----
    RuntimeWarning
        When a non-positive Kc/dR point is dropped before the fit.
    """
    pts = []
    for r in rayleigh_results:
        c = r.concentration_g_per_mL
        if c == 0:
            continue
        idx = np.where(np.isclose(r.angles_deg, angle_deg))[0]
        if idx.size == 0:
            continue
        i = int(idx[0])
        pts.append((float(c), float(r.kc_over_dR_mol_per_g[i]),
                    float(r.q2_nm2[i]), r))
    if len(pts) < 2:
        raise ValueError(
            f"A single-angle ladder needs at least two non-zero concentrations "
            f"measured at {angle_deg} deg; found {len(pts)}."
        )
    pts.sort(key=lambda t: t[0])
    conc = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    q2s = np.array([p[2] for p in pts])
    sources = [p[3] for p in pts]

    # Finite AND positive, the linear arms' rule: Kc/dR <= 0 is an over-subtracted
    # blank and has no place on a 1/Mw + 2 A2 c line. Disclosed, never a refusal,
    # unless nothing usable is left. A NON-finite point is the user's own mask
    # (masked_rayleigh writes NaN) and is dropped silently, exactly as Debye and
    # Guinier drop it: the note counts only the finite non-positive points, so it
    # never tells a user their mask is an over-subtracted blank.
    notes: list = []
    finite = np.isfinite(y)
    good = finite & (y > 0)
    n_drop = int((finite & ~good).sum())
    if n_drop:
        drop = _nonpositive_ordinate_note(n_drop, 'Kc/ΔR')
        notes.append(drop)
        _warn_note(drop, stacklevel=2)
    if not good.all():
        conc, y, q2s = conc[good], y[good], q2s[good]
        sources = [src for src, keep in zip(sources, good, strict=True) if keep]
        if conc.size < 2:
            raise ValueError(
                "A single-angle ladder needs at least two concentrations with a "
                f"finite, positive Kc/dR at {angle_deg} deg; after dropping the masked "
                f"(non-finite) and non-positive points only {conc.size} remain. Check "
                "the mask and the solvent blank."
            )
    # The angle's q^2 as the FITTED points carry it (read after the drop, so it never
    # names a concentration that left the fit).
    q2 = float(q2s[0])

    fit = unc.linear_fit(conc, y, estimator, cond_limit)   # cov order [intercept, slope]
    slope, intercept = fit.slope, fit.intercept

    # Mw_app = 1/intercept needs a PHYSICAL (positive) intercept -- the same sign
    # guard the Debye and Zimm intercepts carry; a noisy non-positive intercept gives
    # a negative or infinite Mw, which is not a molar mass. NaN'd, and not reliable.
    intercept_ok = bool(math.isfinite(intercept) and intercept > 0)
    mw_app = 1.0 / intercept if intercept_ok else float('nan')
    a2_app = slope / 2.0
    two_a2_mw = (float(slope / intercept)
                 if (math.isfinite(intercept) and intercept != 0) else float('nan'))

    mw_se = a2_se = two_se = None
    if math.isfinite(fit.intercept_se):
        if intercept_ok:
            # The reciprocal: d(1/a)/da = -1/a^2, through the covariance so the
            # (zero) slope entry is explicit -- the delta method, first order.
            mw_se = unc.se_or_none(unc.propagate(
                [-1.0 / unc.power_or_nan(intercept, 2), 0.0], fit.cov))
        # A2 = slope/2: linear in one coefficient, so its SE is the pass-through.
        a2_se = unc.se_or_none(fit.slope_se / 2.0)
        if intercept != 0:
            jac = [-slope / unc.power_or_nan(intercept, 2), 1.0 / intercept]
            two_se = unc.se_or_none(unc.propagate(jac, fit.cov))

    scaled = ('mw_apparent_g_per_mol', 'a2_apparent_mol_mL_per_g2')
    reasons = list(scale_notes(sources, mw=scaled[:1], a2=scaled[1:],
                               scale_free=('two_a2_mw',)))
    if not fit.well_conditioned:
        reasons.append(_ill_conditioned_note(fit.cond, cond_limit,
                                             scaled + ('two_a2_mw',)))
    if not intercept_ok:
        reasons.append(_non_positive_intercept_note(intercept, scaled + ('two_a2_mw',)))
    else:
        # A value withheld past the intercept's sign: its own reciprocal or ratio overflowed.
        if not math.isfinite(mw_app):
            reasons.append(Note('mw_not_finite', scaled[:1]))
        if not math.isfinite(two_a2_mw):
            reasons.append(Note('two_a2_mw_not_finite', ('two_a2_mw',)))
    if not math.isfinite(a2_app):
        reasons.append(Note('a2_not_finite', scaled[1:]))
    notes += reasons
    all_calibrated = all(bool(getattr(r, 'calibrated', True)) for r in sources)
    all_dn_dc = all(bool(getattr(r, 'dn_dc_supplied', True)) for r in sources)
    all_blank = all(bool(getattr(r, 'blank_measured', True)) for r in sources)

    return SingleAngleLadderResult(
        angle_deg=float(angle_deg), q2_nm2=q2,
        concentrations_g_per_mL=conc, kc_over_dR=y,
        intercept_mol_per_g=float(intercept), slope=float(slope),
        mw_apparent_g_per_mol=float(mw_app),
        a2_apparent_mol_mL_per_g2=float(a2_app),
        two_a2_mw=two_a2_mw, r_squared=float(fit.r_squared),
        is_apparent=True, n_conc=int(conc.size),
        calibrated=all_calibrated, dn_dc_supplied=all_dn_dc,
        blank_measured=all_blank,
        standard_index_known=all(bool(getattr(r, 'standard_index_known', True))
                                 for r in sources),
        # Both SCALE gates (Kc/dR carries k_c through dR and K through dn/dc), the
        # blank gate, the sign guard, the design's health and finiteness -- the
        # ZimmBerryResult predicate, one angle down.
        mw_reliable=(all_calibrated and all_dn_dc and all_blank and intercept_ok
                     and fit.well_conditioned and math.isfinite(mw_app)),
        a2_reliable=(all_calibrated and all_dn_dc and all_blank and intercept_ok
                     and fit.well_conditioned and math.isfinite(a2_app)),
        # The product cancels both scales, but it is slope/INTERCEPT: the intercept's
        # sign is its precondition as much as Mw's (a negative intercept gave a
        # wrong-signed product flagged reliable), beside the design's health. The
        # blank gate DOES reach it -- an additive term survives the ratio.
        two_a2_mw_reliable=(intercept_ok and fit.well_conditioned and all_blank
                            and math.isfinite(two_a2_mw)),
        mw_apparent_se=mw_se, a2_apparent_se=a2_se, two_a2_mw_se=two_se,
        se_estimator=estimator,
        well_conditioned=fit.well_conditioned, cond=fit.cond,
        notes=tuple(notes),
        derived_provenance=_shared_derived_provenance(sources),
        blank_provenance=_shared_blank_provenance(sources),
    )


def guinier_analysis(rayleigh_result: RayleighRatioResult,
                     qrg_max_valid: float = QRG_MAX_VALID_DEFAULT,
                     estimator: str = 'hc3',
                     cond_limit: float = unc.COND_LIMIT) -> GuinierResult:
    """Single-concentration Guinier plot: linear fit of ln(dR) vs q^2.

    Fits ln(dR) = ln(dR(0)) - (Rg^2/3) q^2. Rg comes from the slope
    (Rg = sqrt(-3*slope)) and is scale-independent (reliable even uncalibrated);
    the apparent Mw comes from the intercept (dR(0) = K c Mw) and needs
    calibration. Only angles with a finite, POSITIVE excess Rayleigh ratio are
    used (ln requires dR > 0).

    The Guinier approximation is due to Guinier & Fournet (1955; originally
    Guinier 1939); Russo et al. (2021, Ch. 13, Guinier's law, Eq. 13.41) derive the
    plot, and Chu (1991, p. 18) prints the small-q expansion P = 1 - q^2 Rg^2/3 + ...
    it rests on, not the plot.

    Parameters
    ----------
    rayleigh_result : RayleighRatioResult
        One concentration's data (>= 2 angles with dR > 0).
    qrg_max_valid : float
        Upper q*Rg for the Guinier approximation (default `QRG_MAX_VALID_DEFAULT`, a
        general criterion, not system-specific). `guinier_valid` reflects this.

    Returns
    -------
    GuinierResult

    Raises
    ------
    ValueError
        If fewer than two angles have a finite, positive excess Rayleigh ratio.
    """
    q2 = rayleigh_result.q2_nm2
    q = rayleigh_result.q_nm_inv
    dR = rayleigh_result.excess_rayleigh_cm_inv
    finite = np.isfinite(q2) & np.isfinite(dR)
    good = finite & (dR > 0)
    # The drop is disclosed, not silent: three angles vanishing from the Rg fit with
    # nothing on any surface saying so was measured on an over-subtracted blank.
    notes: list = []
    n_drop = int((finite & ~good).sum())
    if n_drop:
        drop = _nonpositive_ordinate_note(n_drop, 'excess Rayleigh ratio')
        notes.append(drop)
        _warn_note(drop, stacklevel=2)
    q2g, qg, dRg = q2[good], q[good], dR[good]
    if q2g.size < 2:
        raise ValueError(
            "Guinier analysis needs at least two angles with a finite, positive "
            "excess Rayleigh ratio (ln(dR) requires dR > 0). Check the solvent blank."
        )

    y = np.log(dRg)
    fit = unc.linear_fit(q2g, y, estimator, cond_limit)  # cov order [intercept, slope]
    slope, intercept = fit.slope, fit.intercept
    r2 = fit.r_squared

    # Rg from the slope: slope = -Rg^2/3  (q^2 in nm^-2 -> Rg in nm).
    rg = math.sqrt(-3.0 * slope) if slope < 0 else float('nan')

    # A RISING profile flags no Mw (owner ruling, 2026-08-28): no SLS Mw verdict in this
    # module gates on the slope -- Debye, Zimm/Berry and the form-factor fits all ask
    # about the INTERCEPT -- so flagging here would make Guinier the only one answering a
    # question about the trend. Its one note is the no-Rg precondition below, which
    # carries the slope's signed value (a rise and a flat profile read the same predicate).

    # Apparent Mw from the intercept: dR(0) = exp(intercept) = K c Mw. Use np.exp, which
    # SATURATES to +inf for a large intercept (>~709.78), where math.exp would raise a
    # bare OverflowError and crash the analysis. An infinite dR0 -> infinite Mw is then
    # caught by the math.isfinite gate on mw_reliable below (adjacent fix #2).
    K = rayleigh_result.optical_constant_K
    c = rayleigh_result.concentration_g_per_mL
    with np.errstate(over='ignore'):       # a large intercept saturates dR0 to +inf on purpose
        dR0 = float(np.exp(intercept))
    mw_app = (dR0 / (K * c)
              if (math.isfinite(K) and K > 0 and math.isfinite(c) and c > 0)
              else float('nan'))

    # Statistical SEs: Rg from the slope alone; Mw from the intercept (dMw/dint = Mw).
    rg_se = mw_se = None
    if math.isfinite(fit.slope_se):
        if math.isfinite(rg) and rg > 0:
            rg_se = unc.se_or_none((3.0 / (2.0 * rg)) * fit.slope_se)
        if math.isfinite(mw_app):
            mw_se = unc.se_or_none(abs(mw_app) * fit.intercept_se)

    qrg_max = float(qg.max() * rg) if math.isfinite(rg) else float('nan')
    valid = math.isfinite(qrg_max) and qrg_max <= qrg_max_valid

    # Numerical health of the ln(dR)-vs-q^2 design + a finiteness guard on Mw. Mw here
    # is exp(intercept)/(Kc): a large intercept (>~709) overflows to inf, so the flag
    # must require a finite Mw (previously mw_reliable was unconditional on `calibrated`,
    # weaker than Debye/Zimm -- adjacent fix #2). well_conditioned adds the design gate.
    # A fit outside the Guinier regime is the second failed precondition, and it fails
    # for the whole line (slope AND intercept), so it names itself here and gates BOTH
    # flags below -- one wording, read by the SLS tab, the export and the Cross-Sample
    # candidate label alike. Joined with the conditioning note when both fire.
    reasons = list(scale_notes(rayleigh_result, mw=('mw_apparent_g_per_mol',),
                               scale_free=('rg_nm',)))
    if not fit.well_conditioned:
        reasons.append(_ill_conditioned_note(fit.cond, cond_limit,
                                             ('mw_apparent_g_per_mol', 'rg_nm')))
    if math.isfinite(qrg_max) and not valid:
        # Its own code, the Guinier limit: past it this fit's intercept Mw fails too, as
        # the Debye and Zimm intercepts fail past theirs (`qrg_above_limit`).
        reasons.append(Note('qrg_above_guinier_limit', ('rg_nm', 'mw_apparent_g_per_mol'),
                            {'qrg': float(qrg_max), 'limit': float(qrg_max_valid)}))
    elif not math.isfinite(qrg_max):
        # `valid` is also False when no Rg could be extracted, and that cause names
        # itself: the note states what the predicate READ -- a slope that is not
        # negative -- never why, since the same predicate matches a measurably RISING
        # profile too. The intercept Mw is not flagged for it: qRg could not be
        # measured, so the regime limit was never exceeded.
        reasons.append(_no_rg_note(('rg_nm',), 'slope', slope, 'nm²', 'negative'))
    # `valid` is False for two OPPOSITE reasons and only one is a failed precondition of
    # the INTERCEPT: a MEASURED qRg past the limit strains the parabola and biases it; a
    # NaN qRg means the angular dependence is unmeasurably small, the regime the form
    # assumes (owner-ruled 2026-08-27). Rg is withheld either way.
    # `isnan`, not `not isfinite`: +inf is a qRg past every limit -- the most strained
    # regime there is -- so it must fall on the FLAGGING side (rule 12, both tails; not
    # constructible here, since an infinite qRg needs a non-finite slope whose design
    # fails `well_conditioned` first).
    regime_ok_for_mw = math.isnan(qrg_max) or bool(valid)
    # exp(intercept) saturates to inf past ~709.78, which withholds the apparent Mw.
    if not math.isfinite(mw_app):
        reasons.append(Note('mw_not_finite', ('mw_apparent_g_per_mol',)))
    partition = mode_point_notes(rayleigh_result, about=('mw_apparent_g_per_mol', 'rg_nm'))
    reasons += partition
    notes += reasons
    blank_ok = bool(getattr(rayleigh_result, 'blank_measured', True))

    return GuinierResult(
        concentration_g_per_mL=c, q2_nm2=q2g, ln_excess_rayleigh=y,
        slope=float(slope), intercept=float(intercept), rg_nm=float(rg),
        mw_apparent_g_per_mol=float(mw_app), qrg_max=qrg_max,
        guinier_valid=bool(valid), r_squared=float(r2), n_angles=int(q2g.size),
        is_apparent=True, calibrated=rayleigh_result.calibrated,
        dn_dc_supplied=rayleigh_result.dn_dc_supplied,
        derived_provenance=str(getattr(rayleigh_result, 'derived_provenance', '') or ''),
        blank_provenance=str(getattr(rayleigh_result, 'blank_provenance', '') or ''),
        blank_measured=blank_ok,
        standard_index_known=bool(getattr(rayleigh_result, 'standard_index_known', True)),
        # Rg here comes from the SLOPE of ln(dR) vs q^2 and never touches K at all, so it
        # is bit-identical whether or not dn/dc was supplied. Only the intercept-derived
        # Mw = exp(intercept)/(K c) rides on the optical scale -- hence the scale gates
        # on mw_reliable and none on rg_reliable. `rg_reliable` carries the regime verdict
        # whole; `mw_reliable` carries only its MEASURED half (see `regime_ok_for_mw`) --
        # a qRg past the limit is a failed precondition of the intercept, an unmeasurable
        # qRg is not. Neither is a fit-quality score (invariant 4). `blank_ok` is the one
        # absence BOTH take: it is additive, so it bends the slope too.
        mw_reliable=(rayleigh_result.calibrated and rayleigh_result.dn_dc_supplied
                     and blank_ok
                     and math.isfinite(mw_app) and fit.well_conditioned
                     and regime_ok_for_mw and not partition),
        rg_reliable=bool(math.isfinite(rg) and rg > 0 and fit.well_conditioned
                         and valid and blank_ok and not partition),
        qrg_max_valid=float(qrg_max_valid),
        rg_se=rg_se, mw_apparent_se=mw_se,
        se_estimator=estimator,
        well_conditioned=fit.well_conditioned, cond=fit.cond,
        notes=tuple(notes),
        mode_resolved=bool(getattr(rayleigh_result, 'mode_resolved', False)),
    )


# ===========================================================================
# Pass 3: full Zimm / Berry double extrapolation + calibration-free A2
# ===========================================================================
#
# Both models are linear in their parameters, so one multilinear regression -- on (1, q^2, c)
# for Zimm, on (1, q^2, c, q^2 c) for Berry -- resolves Mw, Rg and A2 together, the "global"
# fit. Model forms and inversions are on `zimm_analysis`; the derivations are in the
# Theory-and-Equations-Guide.


@dataclass
class ZimmBerryResult:
    """Thermodynamic Mw, Rg, and A2 from a Zimm or Berry double extrapolation."""
    method: str                       # 'zimm' or 'berry'
    mw_g_per_mol: float
    rg_nm: float                      # nan if the angular term is non-physical
    a2_mol_mL_per_g2: float
    # global multilinear fit  ord = a + b q^2 + d c  (ord = Kc/dR), or for Berry
    # sqrt(Kc/dR) = a' + b' q^2 + d' c + e' q^2 c -- the cross term is `coef_q2c` below
    coef_intercept: float             # a (or a')
    coef_q2: float                    # b (or b')
    coef_c: float                     # d (or d')
    r_squared: float
    n_points: int
    n_concentrations: int
    n_angles: int
    is_apparent: bool                 # False: thermodynamic (fully extrapolated)
    calibrated: bool                  # was the Rayleigh ratio calibrated?
    mw_reliable: bool                 # False if uncalibrated
    a2_reliable: bool                 # False if uncalibrated (absolute A2 needs Mw)
    # per-concentration q->0 intercepts (for the Zimm grid plot / cross-check)
    concentrations_g_per_mL: np.ndarray
    intercept_per_concentration: np.ndarray   # ordinate at q^2 -> 0, each c
    # per-angle c->0 intercepts
    q2_nm2: np.ndarray
    intercept_per_angle: np.ndarray           # ordinate at c -> 0, each angle
    # Berry's fitted q^2 c cross term e', free (see `zimm_analysis`). It vanishes on both
    # limit lines, so no reported quantity reads it. NaN for Zimm, and for a Berry fit whose
    # grid could not resolve it (the three-column fallback, disclosed on `notes`).
    coef_q2c: float = float('nan')
    # `coef_q2c` as a multiple of the -b'd'/a' Berry's own equation predicts to first order:
    # 1 on that form, -1 on a product of the two lines. Both carry the ordinate's scale, so
    # the ratio survives an unknown one. NaN wherever `coef_q2c` is, or the prediction is 0.
    coef_q2c_ratio: float = float('nan')
    # statistical (regression) standard errors -- exclude calibration/dn-dc
    # systematics. None when too few points to estimate them.
    mw_se: Optional[float] = None
    rg_se: Optional[float] = None
    a2_se: Optional[float] = None
    # consistency cross-check: Mw from the two single-route extrapolations (the
    # q->0 intercepts extrapolated to c->0, and the c->0 intercepts to q->0) should
    # agree; extrapolation_agreement_rel is their relative difference.
    mw_from_c0_g_per_mol: Optional[float] = None
    mw_from_q0_g_per_mol: Optional[float] = None
    extrapolation_agreement_rel: Optional[float] = None
    se_estimator: str = 'hc3'                 # covariance estimator behind the SEs
    # numerical-health of the global design X = [1, q^2, c] (Berry: [1, q^2, c, q^2 c], or
    # the three columns after a fallback -- the design actually fitted). well_conditioned=
    # False -> near-degenerate (clustered angles / near-equal concentrations) ->
    # mw_reliable and a2_reliable are suppressed even when calibrated (value still shown).
    well_conditioned: bool = True
    cond: float = float('nan')                # condition number of the design
    # The SIZE's own flag (rule 10; see DebyeResult.rg_reliable): Rg^2 = 3b/a (6b'/a')
    # is a ratio of two coefficients of the same near-singular solve, so it shares the
    # conditioning verdict and none of the scale gates. For Zimm it also carries the qRg
    # regime below; for Berry the regime is a qualifier. Value still reports.
    rg_reliable: bool = True
    # The qRg regime (`zimm_analysis`): the largest fitted q times the reported Rg, the
    # limit it was judged against (echoed, as `GuinierResult.qrg_max_valid` is), and
    # whether it held -- False also when no Rg was extracted, as `guinier_valid` is. A Zimm
    # fit past the limit names it in a precondition note and drops its Mw with its Rg; a
    # Berry fit holds no limit of its own, so its verdict is True wherever qRg(max) is finite
    # and a Berry Rg and Mw read past the Zimm limit are stated as a qualifier
    # (`core.notes.qualifiers`, `berry_past_zimm_limit`).
    qrg_max: float = float('nan')
    qrg_max_valid: float = QRG_MAX_VALID_DEFAULT
    qrg_regime_valid: bool = True
    # Every note of the fit, as Notes: its preconditions, the drop of non-positive Kc/dR
    # points (both methods; also emitted via warnings.warn, its message the note's export
    # clause), and a concentration or q^2 level whose grid intercept is a single ordinate
    # rather than a fit.
    notes: Tuple[Note, ...] = ()
    # `calibrated`'s sibling for the OPTICAL scale (see RayleighRatioResult): False ->
    # no dn/dc was supplied, so K was 1 and Mw/A2 here are on an arbitrary optical
    # scale. The angular sizes are untouched -- K multiplies the whole Kc/dR ordinate
    # and cancels in every coefficient ratio.
    dn_dc_supplied: bool = True
    # Carried forward from the Rayleigh result(s) this fit consumed, so a number
    # cannot be read without its origin whichever exporter writes it. A record; it
    # enters no computation. Empty for an ordinary loaded measurement.
    derived_provenance: str = ''
    # Which solvent blank the contributing points were referenced against, when
    # it is not the sample reference; copied from the Rayleigh result(s) exactly as
    # derived_provenance is, and exported beside it. '' for the common case.
    blank_provenance: str = ''
    # Carried from the Rayleigh result(s) this fit consumed: False -> the blank was not
    # measured at every angle (typed 90 deg value, or no subtraction). ANDed into every
    # reliability verdict on this result, because an additive blank does not cancel the
    # way an unknown SCALE does. See RayleighRatioResult.blank_measured.
    blank_measured: bool = True
    # Carried from the Rayleigh result(s) this fit consumed: False -> k_c was applied with
    # the standard's refractive index blank, so `calibrated` is False for that cause alone
    # (the scale gates name it). See RayleighRatioResult.standard_index_known.
    standard_index_known: bool = True
    # Carried from the Rayleigh results: True when they are one mode's share of the excess
    # intensity (see DebyeResult.mode_resolved).
    mode_resolved: bool = False


@dataclass
class CalibrationFreeA2Result:
    """A2 (times Mw) from intensity ratios alone -- no absolute calibration.

    Uses Y(c) = [I_excess(c_ref)/c_ref] / [I_excess(c)/c] at a fixed angle, which
    is linear in c with slope/intercept = 2 A2 Mw P(q), the product 2 A2 Mw at q -> 0
    (`calibration_free_a2`'s rearrangement). Because only intensity RATIOS
    enter, the optical constant and Rayleigh-ratio calibration cancel -- valuable
    when absolute calibration is uncertain (e.g. low-dn/dc systems). If Mw is
    supplied, A2 itself is returned.
    """
    angle_deg: float
    concentrations_g_per_mL: np.ndarray
    Y: np.ndarray
    slope: float
    intercept: float
    two_a2_mw: float                  # slope / intercept = 2 A2 Mw P(q), 2 A2 Mw at q -> 0  (mol*mL/g ... * g/mol)
    a2_mol_mL_per_g2: Optional[float] # if mw provided
    r_squared: float
    # These SEs are free of calibration/dn-dc systematics *by construction*:
    # 2*A2*Mw = slope/intercept is calibration- and dn/dc-independent (the point of
    # this estimator), so unlike the Zimm/Debye SEs there is no such systematic to exclude.
    two_a2_mw_se: Optional[float] = None   # statistical SE (slope/intercept covariance)
    a2_se: Optional[float] = None          # if mw provided (mw treated as exact)
    se_estimator: str = 'hc3'              # covariance estimator behind the SEs
    # numerical-health of the Y-vs-c design [1, c]: near-equal concentrations make it
    # near-collinear -> slope/intercept fragile. two_a2_mw_reliable is suppressed (value
    # still shown). There is no `calibrated` gate -- this estimator is calibration-free.
    well_conditioned: bool = True
    cond: float = float('nan')             # condition number of the design
    two_a2_mw_reliable: bool = True        # False if the design is ill-conditioned
    # Every note of the fit, as Notes (e.g. dropped non-positive excess-Rayleigh points,
    # also emitted via warnings.warn, its message the note's export clause).
    notes: Tuple[Note, ...] = ()
    # APPARENT, always (invariant 7): Y(c) is formed at ONE finite angle, and there
    # slope/intercept = 2 A2 Mw P(q) -- the angle-contingent product, not its q -> 0
    # limit. Structural, like DebyeResult's; carried so the export and the
    # apparent/thermodynamic family gate can read it.
    is_apparent: bool = True
    # The provenance quartet every other fit built from RayleighRatioResults carries
    # (rule 10). `calibrated` / `dn_dc_supplied` are RECORDS here, not gates: both scales
    # cancel in the ratio Y(c), which is the point of the estimator, so neither enters
    # `two_a2_mw_reliable`. They travel so a reader of the result knows what the inputs
    # were (an absolute A2 = product / 2Mw does NOT cancel them -- the controller decides
    # that with the sample's Mw). Aggregated over the contributing concentrations the
    # way `zimm_analysis` aggregates them.
    calibrated: bool = True
    dn_dc_supplied: bool = True
    # The THIRD absence, and the one that GATES `two_a2_mw_reliable` where the two above
    # only record. An unknown k_c and an absent dn/dc are MULTIPLICATIVE, so they divide
    # out of Y(c) = (Kc/dR)_ref / (Kc/dR) exactly -- that cancellation IS the estimator.
    # An unmeasured blank leaves an ADDITIVE term in each dR instead, which does not
    # cancel in a ratio: it biases slope and intercept differently, so the product is
    # wrong by an amount nothing here can bound. Normative home:
    # RayleighRatioResult.blank_measured.
    blank_measured: bool = True
    # Carried from the Rayleigh result(s) this fit consumed: False -> k_c was applied with
    # the standard's refractive index blank, so `calibrated` is False for that cause alone
    # (the scale gates name it). See RayleighRatioResult.standard_index_known.
    standard_index_known: bool = True
    # Where the intensities came from, and which solvent blank each concentration was
    # referenced against, when not an instrument reading against the sample reference.
    # These do NOT cancel in Y(c): a different blank moves every excess ratio, hence the
    # slope/intercept -- so a 2 A2 Mw cannot be read without them. Empty for the common
    # case; reduced by `_shared_derived_provenance` / `_shared_blank_provenance`.
    derived_provenance: str = ''
    blank_provenance: str = ''


@dataclass
class SingleAngleLadderResult:
    """Single-angle concentration series -- the classical Debye plot at one angle.

    Kc/dR_theta = 1/Mw_app + 2 A2_app c over the ladder (Theory-and-Equations-Guide,
    "One angle over a concentration series", the fixed-angle form of the Debye
    intercept equation): Mw from the
    intercept, A2 from the slope. APPARENT, always (invariant 7): at one angle the
    q -> 0 extrapolation was never done, so the intercept is 1/(Mw P(q)) -- the
    apparent Mw is Mw P(q) <= Mw. A Zimm or Berry set over several angles is the
    route to the thermodynamic values.

    The calibration-free product 2 A2 Mw = slope/intercept is carried as well: it
    is what survives an uncalibrated run (CLAUDE.md, physics), exactly as in the
    calibration-free construction, with the same reciprocal-tail caveat.
    """
    angle_deg: float
    q2_nm2: float
    concentrations_g_per_mL: np.ndarray
    kc_over_dR: np.ndarray
    # linear fit  Kc/dR = intercept + slope * c
    intercept_mol_per_g: float        # = 1/Mw_app
    slope: float                      # = 2 A2_app  (mol mL / g^2)
    mw_apparent_g_per_mol: float      # 1 / intercept
    a2_apparent_mol_mL_per_g2: float  # slope / 2
    two_a2_mw: float                  # slope / intercept, calibration-free
    r_squared: float
    is_apparent: bool                 # always True here
    n_conc: int
    calibrated: bool                  # was the Rayleigh ratio calibrated?
    mw_reliable: bool                 # False if uncalibrated / no dn/dc / degenerate
    # The absolute A2 needs both scales too -- and the line's intercept sign: a
    # non-positive 1/Mw is a failed model, A2's precondition as much as Mw's
    # (ZimmBerryResult.a2_reliable's rule; A2 = slope/2 is that same line's slope).
    a2_reliable: bool
    two_a2_mw_reliable: bool          # the product needs neither; design health only
    # Statistical (regression) SEs -- exclude calibration/dn-dc systematics. The Mw SE
    # is a RECIPROCAL of the fitted intercept (Monte-Carlo-validated conservative on
    # the short ladder: tests/test_uncertainty.py); the A2 SE is the linear
    # pass-through slope_se/2 and inherits that validation; the product's SE is the
    # RATIO slope/intercept through the same covariance, gated in the same end-to-end
    # Monte-Carlo test (tests/test_sls.py) as the Mw SE.
    mw_apparent_se: Optional[float] = None
    a2_apparent_se: Optional[float] = None
    two_a2_mw_se: Optional[float] = None
    se_estimator: str = 'hc3'                 # covariance estimator behind the SEs
    # numerical health of the [1, c] design: near-equal concentrations -> fragile
    # 1/intercept -> mw_reliable / a2_reliable suppressed (value still shown).
    well_conditioned: bool = True
    cond: float = float('nan')                # condition number of the design
    # Every note of the fit, as Notes (its preconditions, the non-positive Kc/dR drop).
    notes: Tuple[Note, ...] = ()
    # `calibrated`'s sibling for the OPTICAL scale (see RayleighRatioResult).
    dn_dc_supplied: bool = True
    # The provenance pair every fit built from RayleighRatioResults carries.
    derived_provenance: str = ''
    blank_provenance: str = ''
    # Carried from the Rayleigh result(s) this fit consumed: False -> the blank was not
    # measured at every angle (typed 90 deg value, or no subtraction). ANDed into every
    # reliability verdict on this result, because an additive blank does not cancel the
    # way an unknown SCALE does. See RayleighRatioResult.blank_measured.
    blank_measured: bool = True
    # Carried from the Rayleigh result(s) this fit consumed: False -> k_c was applied with
    # the standard's refractive index blank, so `calibrated` is False for that cause alone
    # (the scale gates name it). See RayleighRatioResult.standard_index_known.
    standard_index_known: bool = True


def _collect_zimm_points(rayleigh_results: Sequence[RayleighRatioResult]):
    """Flatten a set of per-concentration results into (q2, c, Kc/dR) arrays."""
    q2_all, c_all, y_all = [], [], []
    for r in rayleigh_results:
        if r.concentration_g_per_mL == 0:
            continue   # skip the solvent reference
        # The concentration is filtered alongside the two arrays, not assumed finite.
        # It is a scalar broadcast across the level, so a non-finite one drops the whole
        # level -- which is the point: it would otherwise reach the per-angle c->0 fit as
        # a non-finite ABSCISSA (np.polyfit raises there), and a NaN level additionally
        # survives np.unique while matching no row, so `ordinate[m][0]` below raised
        # IndexError on an empty mask. Two crashes closed at the one place the design is
        # assembled, rather than at each fit downstream (2026-07-31 spec-01).
        if not math.isfinite(r.concentration_g_per_mL):
            continue
        good = np.isfinite(r.q2_nm2) & np.isfinite(r.kc_over_dR_mol_per_g)
        q2_all.append(r.q2_nm2[good])
        c_all.append(np.full(int(good.sum()), r.concentration_g_per_mL))
        y_all.append(r.kc_over_dR_mol_per_g[good])
    if not q2_all:
        raise ValueError("No non-zero-concentration data to analyze.")
    return (np.concatenate(q2_all), np.concatenate(c_all), np.concatenate(y_all))


def _q2_levels(q2):
    """The distinct angles of a flattened design: q^2 (nm^-2) rounded to 12 decimal places,
    so one angle measured at several concentrations is one level."""
    return np.unique(np.round(q2, 12))


def _aggregate_scale_flags(rayleigh_results) -> Tuple[bool, bool, bool]:
    """(all_calibrated, all_dn_dc_supplied, all_blank_measured) over a fit's Rayleigh
    results: each True only when every non-zero-concentration result carries it."""
    # One set for every flag, the non-zero concentrations: the solvent blank carries no
    # dn/dc of its own and none is ever read from it, and a series with no measured c = 0
    # point contributes no r.concentration_g_per_mL == 0 entry at all, so the skip is what
    # makes the blank flag an honest `all` rather than a vacuous one. The blank flag is not
    # a scale: it gates every verdict (the RayleighRatioResult docstring says why).
    all_calibrated = all(r.calibrated for r in rayleigh_results
                         if r.concentration_g_per_mL != 0)
    all_dn_dc_supplied = all(r.dn_dc_supplied for r in rayleigh_results
                             if r.concentration_g_per_mL != 0)
    all_blank_measured = all(bool(getattr(r, 'blank_measured', True))
                             for r in rayleigh_results
                             if r.concentration_g_per_mL != 0)
    return all_calibrated, all_dn_dc_supplied, all_blank_measured


def _form_factor_inputs(rayleigh_results):
    """The form-factor fits' flattened design, (q2, c, y, q, n_conc, n_ang, thermodynamic,
    drop_notes): the points `_collect_zimm_points` keeps, less the non-positive Kc/dR ones
    (recorded on `drop_notes`; `_drop_nonpositive_ordinate` hands an all-zero ordinate on
    whole), with q the root of q^2 for the form factor's q*Rg argument.
    Two or more concentrations make the fit thermodynamic. No angle floor is applied here:
    the floor is the model's, set by how many shape parameters it fits."""
    q2, c, y = _collect_zimm_points(rayleigh_results)
    drop_notes: list = []
    q2, c, y = _drop_nonpositive_ordinate('Form-factor', q2, c, y, drop_notes)
    q = np.sqrt(q2)
    n_conc = int(np.unique(c).size)
    n_ang = int(_q2_levels(q2).size)
    thermodynamic = n_conc >= 2
    return q2, c, y, q, n_conc, n_ang, thermodynamic, drop_notes


def zimm_analysis(
    rayleigh_results: Sequence[RayleighRatioResult],
    method: str = 'zimm',
    estimator: str = 'hc3',
    cond_limit: float = unc.COND_LIMIT,
    qrg_max_valid: float = QRG_MAX_VALID_DEFAULT,
) -> ZimmBerryResult:
    """Full Zimm or Berry double extrapolation for Mw, Rg, and A2.

    Performs the global first-order fit (multilinear regression of the ordinate
    on 1, q^2, and c, and for Berry q^2 c) over all angles and concentrations, and also
    returns the per-concentration and per-angle intercepts used to draw the classic Zimm
    grid (for visualization and as a cross-check on the global fit).

    The first-order Zimm equation ``Kc/dR = (1/Mw)(1 + q^2 Rg^2/3) + 2 A2 c`` (Russo et al.
    2021, Eq. 13.3, p. 501) is LINEAR in the parameters (a, b, d) of the model
    ``a + b q^2 + d c``, with ``a = 1/Mw``, ``b = (1/Mw)(Rg^2/3)``, ``d = 2 A2`` -- so one
    multilinear regression of Kc/dR on (1, q^2, c) over all points gives ``Mw = 1/a``,
    ``Rg = sqrt(3 b / a)``, ``A2 = d / 2``; Russo et al. fit the two lines one by one or
    the whole equation at once (pp. 500, 504), and this is the second.

    Berry takes the square root of Berry's own equation (Berry 1966, p. 4552:
    ``Kc/R = [M P(u,0)]^-1 (1 + 2 Gamma2 P(u,0) c)``, the virial term inside P -- the page's
    brace of Q and dP/dc taken as unity, as Berry finds it, and the c^2 terms dropped; Gamma2
    = A2 M is not printed and follows by matching the Zimm equation's 2 A2 c at P = 1) and fits
    ``sqrt(Kc/dR) = a' + b' q^2 + d' c + e' q^2 c``. Berry prints the two limit lines of
    that surface, his square-root plots of ``sqrt(Kc/dR)`` against c at q -> 0 and against
    q^2 at c -> 0 (pp. 4552-4553); the four-column surface through both is this program's
    construction, not a printed form. The cross term vanishes on both lines,
    so ``a' = 1/sqrt(Mw)``, ``b' = a' Rg^2/6``, ``d' = A2 sqrt(Mw)`` and ``Mw = 1/a'^2``,
    ``Rg = sqrt(6 b'/a')``, ``A2 = d' a'`` -- rearrangements of that equation at its limits,
    as Podzimek 2011 prints them (Eq. 2.28, p. 54; Eqs. 4.11 and 4.13, p. 221). On a
    complete angle x concentration grid this least squares IS Berry's line-by-line
    procedure in one solve: the bilinear basis separates over a product grid. The cross
    term e' is FREE, a named choice: Berry's equation predicts e' = -b'd'/a' to first order
    (its root carries sqrt(P) on the virial term), a product of the two lines gives
    +b'd'/a', and the free column recovers either; it is recorded as ``coef_q2c``. Without
    the column the global surface would be additive, a form no source prints, and biased on
    any product-form data. Where the four-column design is rank-deficient or above
    ``cond_limit`` (a ragged grid), the fit falls back to the three columns and says so on
    ``notes``, never on a flag.

    Neither construction is exact at finite qRg on Berry's own equation with a real form
    factor (on a Debye coil at qRg_max 1.37: four-column Berry Rg -1.9 %, Zimm +7.8 %, the
    direction Podzimek 2011 p. 230 reports for the first-order Zimm Rg). The sources place
    Zimm at Rgq << 1 and Berry at Rgq ~ 1 (Takahashi et al. 2019, p. 1046), the
    initial-slope Rg at qRg < 1 (Russo et al. 2021, pp. 517-518) and the molecular weight
    in the Rayleigh region, Rgq < 1 (Takahashi et al. 2019, p. 1045), the figure
    `QRG_MAX_VALID_DEFAULT` takes.

    Inputs not prescribed by the source: the conditioning ceiling ``cond_limit`` (default
    ``COND_LIMIT``, a numerical-health bound), the angle identity (q^2 rounded to 12
    decimal places), the two-angle and two-concentration minimums (the design's mathematical
    minimums), the drop of a non-positive Kc/dR (stated on ``notes``), the agreement check
    between the two single-route Mw values, the weighting (equal weights on every retained
    point), the Berry four-column surface with its cross term free and its three-column
    fallback on a ragged grid (this program's construction, above); prescribed: the
    first-order Zimm equation (Russo et al. 2021 Eq. 13.3, p. 501; Chu 1991 Eq. 2.1.8,
    p. 18), fitted whole rather than line by line (Russo et al. 2021, pp. 500 and 504), the
    Berry square-root ordinate (Berry 1966, p. 4552) and its inversions by rearrangement
    (Podzimek 2011 Eqs. 2.28, 4.11 and 4.13), and the qRg regime ``qrg_max_valid`` reads
    (Russo et al. 2021, p. 517; Takahashi et al. 2019, pp. 1045-1046).

    The ladder deliberately carries NO identity machinery (product decision,
    owner-ruled 2026-08-13). The one hazard identity fields could not catch is an
    untagged multi-Mw series -- two ladders of different molar mass pooled into one
    fit whose 1/Mw intercept is a meaningless mix -- and Mw is an OUTPUT of this
    fit, not an input field, so no comparison here can detect it. The honest
    ceiling is the passive pooling hint the SLS tab already shows (duplicate
    concentrations / extra blanks), plus the user's Mw-fraction tags, which
    partition the series upstream (combination doctrine rule 2,
    core/data_models.py). Building identity into RayleighRatioResult was
    considered and rejected: cost on every SLS path, no detection power.

    Parameters
    ----------
    rayleigh_results : sequence of RayleighRatioResult
        One per concentration (the solvent reference, c = 0, is ignored). Need at
        least two concentrations and two angles to resolve all three quantities.
    method : str
        'zimm' (ordinate = Kc/dR; default) or 'berry' (ordinate = sqrt(Kc/dR), the
        construction the sources place at qRg ~ 1).
    cond_limit : float
        Numerical-health ceiling for the design's (column-normalized) condition
        number; above it the fit is flagged near-degenerate and mw_reliable/a2_reliable
        are suppressed (see uncertainty.COND_LIMIT). Overridable; default COND_LIMIT.
    qrg_max_valid : float
        Upper q*Rg for the first-order angular term (default `QRG_MAX_VALID_DEFAULT`, shared
        with every qRg verdict; the controller seeds it from the one Setting). Both methods
        report ``qrg_max`` -- the largest fitted q times the reported Rg -- and echo the
        limit. Past it a Zimm fit drops ``rg_reliable`` and ``mw_reliable`` and names the
        failed precondition in one note about both (``qrg_above_limit``): the sources put
        the Zimm plot's M, not only its Rg, at Rgq << 1 (Takahashi et al. 2019, p. 1046),
        and give the number where they choose samples for M, the Rayleigh region Rgq < 1
        (p. 1045). A fit
        with no extractable Rg has no qRg to judge and keeps its Mw verdict. A Berry fit
        drops nothing, the sources placing Berry at Rgq ~ 1 and printing no limit for it,
        and its Rg and Mw read past the Zimm limit are a qualifier read off ``qrg_max`` and
        ``qrg_max_valid`` (``berry_past_zimm_limit``). Neither drops A2: the regime is stated
        on the c -> 0 equations (Takahashi's Eqs. 4 and 5 are limits at C -> 0).

    Returns
    -------
    ZimmBerryResult

    Raises
    ------
    ValueError
        If method is invalid or there are too few concentrations/angles.
    """
    if method not in ('zimm', 'berry'):
        raise ValueError(f"method must be 'zimm' or 'berry', got {method!r}.")

    q2, c, kc_over_dR = _collect_zimm_points(rayleigh_results)
    all_calibrated, all_dn_dc_supplied, all_blank_measured = _aggregate_scale_flags(
        rayleigh_results)
    n_conc = np.unique(c).size
    n_ang = _q2_levels(q2).size
    if n_conc < 2:
        raise ValueError(
            f"Zimm/Berry analysis needs at least two concentrations, got {n_conc}."
        )
    if n_ang < 2:
        raise ValueError(
            f"Zimm/Berry analysis needs at least two angles, got {n_ang}."
        )

    notes: list = []       # every note of the result, as Notes
    label = method.capitalize()
    # A non-positive Kc/dR (noise or an over-subtracted solvent reference, especially
    # at high angle) has no place in EITHER model: Berry cannot take its square root,
    # and Zimm's 1/Mw + ... line has no meaning for a solution scattering less than its
    # solvent -- left in, such a point dominates the global least squares. Both methods
    # drop the class and say so (Berry always did; Zimm fitted it silently -- rule 10,
    # one drop site for the family). `~(x > 0)` catches NaN (which compares False) but
    # NOT +inf, which passes as "positive" and survives sqrt(inf) = inf straight into
    # the global fit; the Guinier sibling already uses the finite form, so match it.
    nonpos = ~(np.isfinite(kc_over_dR) & (kc_over_dR > 0))
    if np.any(nonpos):
        n_drop = int(nonpos.sum())
        # Single source of the message -> both the note and the warning.
        drop = _nonpositive_ordinate_note(n_drop, 'Kc/ΔR')
        notes.append(drop)
        _warn_note(drop, stacklevel=2)
        q2, c, kc_over_dR = q2[~nonpos], c[~nonpos], kc_over_dR[~nonpos]
        n_conc = np.unique(c).size
        n_ang = _q2_levels(q2).size
        if n_conc < 2 or n_ang < 2:
            raise ValueError(
                f"{label} analysis: after dropping non-positive Kc/dR points, "
                f"only {n_conc} concentration(s) and {n_ang} angle(s) remain "
                "(need at least two of each). Check the solvent blank."
            )
    ordinate = np.sqrt(kc_over_dR) if method == 'berry' else kc_over_dR

    # Global multilinear fit: ordinate = a + b q^2 + d c, with its 3x3 covariance -- and for
    # Berry the free cross column e q^2 c as well (see the docstring), 4x4. The cross term
    # vanishes on both limit lines, so a, b, d keep their meaning and their inversions.
    A3 = np.column_stack([np.ones_like(q2), q2, c])
    mf = (unc.multilinear_fit(np.column_stack([A3, q2 * c]), ordinate, estimator, cond_limit)
          if method == 'berry' else unc.multilinear_fit(A3, ordinate, estimator, cond_limit))
    if method == 'berry' and (mf.rank_deficient or not mf.well_conditioned):
        # A ragged grid can leave the cross column unresolvable. The three-column fit is
        # Berry's limit lines without the cross term, so falling back to it is a record
        # on `notes`, never a flag; its own conditioning then decides the flags below.
        notes.append(Note('berry_cross_term_dropped', (), {
            'reason': 'rank-deficient' if mf.rank_deficient else 'ill-conditioned'}))
        mf = unc.multilinear_fit(A3, ordinate, estimator, cond_limit)
    a, b, d = float(mf.coeffs[0]), float(mf.coeffs[1]), float(mf.coeffs[2])
    e = float(mf.coeffs[3]) if len(mf.coeffs) == 4 else float('nan')
    # Berry's first-order prediction -b'd'/a' (see the docstring). Each divisor is checked
    # finite as well as nonzero, so an inf never becomes a plausible 0.0 ratio; `*` on
    # floats saturates rather than raises (rule 12).
    e_pred = -b * d / a if (math.isfinite(a) and a != 0) else float('nan')
    e_ratio = (e / e_pred if (math.isfinite(e_pred) and e_pred != 0) else float('nan'))
    cov = mf.cov                          # order (a, b, d[, e])
    pad = [0.0] * (len(mf.coeffs) - 3)    # Mw, Rg, A2 never read e: a zero Jacobian entry
    r2 = mf.r_squared

    # The ordinate intercept a (Kc/dR extrapolated to q->0, c->0) is physically
    # POSITIVE: a = 1/Mw (Zimm) or 1/sqrt(Mw) (Berry). A noisy NON-POSITIVE intercept
    # would give a negative or infinite Mw, which is impossible -- guard it to NaN
    # (mirroring the Debye path's intercept>0 guard) rather than present an unphysical
    # negative Mw. Rg shares the guard: with a<=0 the whole fit is unphysical, so an
    # Rg extracted from its slope/intercept ratio is not meaningful either.
    if method == 'zimm':
        mw = 1.0 / a if (math.isfinite(a) and a > 0) else float('nan')
        rg = (math.sqrt(3.0 * b / a)
              if (math.isfinite(a) and a > 0
                  and math.isfinite(b / a) and b / a > 0) else float('nan'))
        a2 = d / 2.0                      # independent of a (no sign guard needed)
    else:  # berry
        # Mw = 1/a^2: require a>0 AND a^2 not underflowed to 0.0 (an extreme scale can
        # drive a to a denormal that passes a>0 yet squares to 0.0 -> 1/0.0 would raise
        # ZeroDivisionError). A2 = d'*a' is sign-dependent on a, so it must be NaN'd on a
        # non-physical intercept too (else a garbage A2 rides out with a2_reliable set).
        # Named `a_sq`, NOT `_a2`: `a2` one line below is the second virial coefficient,
        # and two same-scope floats one underscore apart is a one-character slip away from
        # a silently wrong Mw or A2 -- in the worst possible place.
        a_sq = unc.power_or_nan(a, 2)   # NaN unless a^2 is a usable divisor (both tails)
        mw = (1.0 / a_sq
              if (math.isfinite(a) and a > 0 and math.isfinite(a_sq)) else float('nan'))
        rg = (math.sqrt(6.0 * b / a)
              if (math.isfinite(a) and a > 0
                  and math.isfinite(b / a) and b / a > 0) else float('nan'))
        a2 = d * a if (math.isfinite(a) and a > 0) else float('nan')   # A2 = d'/sqrt(Mw) = d'*a'

    # Statistical SEs by first-order propagation through Cov(a,b,d[,e]). Rg uses the same
    # Jacobian form for Zimm (Rg^2=3b/a) and Berry (Rg^2=6b/a): dRg/da=-Rg/2a, dRg/db=+Rg/2b.
    # Mw and A2 differ between the constructions; none reads Berry's e (`pad`'s zero entry).
    # Gated on a>0 (so no SE travels with a non-physical Mw) AND on the Mw Jacobian's
    # denominator being a usable one: at an extreme intensity scale a^2 / a^3 can UNDERFLOW
    # to 0.0 even though a != 0, which would make -1/a^2 raise a bare ZeroDivisionError. The
    # denominator comes from the shared `power_or_nan`, whose NaN reaches the withhold
    # through `propagate`.
    mw_se = rg_se = a2_se = None
    if np.all(np.isfinite(cov)) and a > 0:
        if method == 'zimm':
            mw_se = unc.se_or_none(unc.propagate(
                [-1.0 / unc.power_or_nan(a, 2), 0.0, 0.0] + pad, cov))
            a2_se = unc.se_or_none(unc.propagate([0.0, 0.0, 0.5] + pad, cov))
        else:  # berry: Mw=1/a^2, A2=d*a
            mw_se = unc.se_or_none(unc.propagate(
                [-2.0 / unc.power_or_nan(a, 3), 0.0, 0.0] + pad, cov))
            a2_se = unc.se_or_none(unc.propagate([d, 0.0, a] + pad, cov))
        if math.isfinite(rg) and rg > 0 and b != 0:
            rg_se = unc.se_or_none(
                unc.propagate([-rg / (2.0 * a), rg / (2.0 * b), 0.0] + pad, cov))

    # Per-concentration q->0 intercepts and per-angle c->0 intercepts (for the grid).
    # A level with a single usable point gets no per-level line: its lone ordinate IS
    # the grid intercept (no extrapolation at that level; the global fit above is
    # unaffected). Disclosed on `notes`, never refused -- the user's point selection
    # (a mask, a single-angle run at one concentration) is theirs to make (invariant 4
    # clause 2; owner-ruled 2026-08-26, CLAUDE.md invariant 4).
    conc_levels = np.unique(c)
    inter_per_c = []
    single_point_conc = []
    for cl in conc_levels:
        m = c == cl
        if _q2_levels(q2[m]).size >= 2:
            s_i, i_i = unc.polyfit_or_nan(q2[m], ordinate[m], 1)
        else:
            # One usable ANGLE at this level (one point, or several at one q^2 -- a
            # repeated run), so there is no line: its ordinate stands in (the mean of
            # the repeats), and the level is disclosed below.
            i_i = float(np.mean(ordinate[m]))
            single_point_conc.append(float(cl))
        inter_per_c.append(i_i)
    q2_levels = _q2_levels(q2)
    inter_per_q = []
    single_point_q2 = []
    for ql in q2_levels:
        m = np.isclose(q2, ql)
        if np.unique(c[m]).size >= 2:
            s_i, i_i = unc.polyfit_or_nan(c[m], ordinate[m], 1)
        else:
            i_i = float(np.mean(ordinate[m]))
            single_point_q2.append(float(ql))
        inter_per_q.append(i_i)
    if single_point_conc:
        notes.append(Note('grid_level_single_angle', ('intercept_per_concentration',),
                          {'levels': tuple(single_point_conc)}))
    if single_point_q2:
        notes.append(Note('grid_level_single_concentration', ('intercept_per_angle',),
                          {'levels': tuple(single_point_q2)}))

    # Consistency cross-check: the two single-route extrapolations of the ordinate
    # intercept to (q->0, c->0) should agree. Route 1: q->0 intercepts (per c)
    # extrapolated to c->0. Route 2: c->0 intercepts (per angle) extrapolated to q->0.
    def _mw_from_intercept(a0: float) -> Optional[float]:
        # a0 <= 0 (not just == 0): a non-positive intercept gives a non-physical Mw, so
        # the agreement diagnostic must not be built on it. The Berry square goes through
        # the shared power_or_nan, mirroring the main Mw computation -- and named a0_sq,
        # not _a02, for the same reason a_sq is not _a2 above.
        if not math.isfinite(a0) or a0 <= 0:
            return None
        if method == 'zimm':
            return 1.0 / a0
        a0_sq = unc.power_or_nan(a0, 2)
        return 1.0 / a0_sq if math.isfinite(a0_sq) else None

    mw_q0 = mw_c0 = agree = None
    # NB both size checks below are ALWAYS TRUE here, kept only as defence in depth:
    # conc_levels/q2_levels come from the same arrays as n_conc/n_ang, which this function
    # already requires to be >= 2. So mw_q0/mw_c0 are always assigned and the agreement
    # cross-check cannot silently degrade to None -- do NOT read them as evidence that a
    # single-concentration or single-angle Zimm is possible; it is not. (The PER-LEVEL
    # `>= 2` guards further down are a different thing and are genuinely reachable: they
    # handle ragged data, one concentration measured at a single angle or vice versa.)
    if conc_levels.size >= 2:
        mw_q0 = _mw_from_intercept(
            unc.linear_fit(conc_levels, np.array(inter_per_c)).intercept)
    if q2_levels.size >= 2:
        mw_c0 = _mw_from_intercept(
            unc.linear_fit(q2_levels, np.array(inter_per_q)).intercept)
    if mw_q0 and mw_c0 and (mw_q0 + mw_c0) != 0:
        agree = abs(mw_q0 - mw_c0) / (0.5 * abs(mw_q0 + mw_c0))

    # Numerical health of the global design actually fitted (X = [1, q^2, c], or Berry's
    # four columns): clustered angles or two
    # near-equal concentrations (or an extreme intensity scale) make it near-singular,
    # so 1/a (Mw) and d*a/d/2 (A2) are fragile even when calibrated and a>0. Suppress
    # both reliability flags and carry the reason (values still shown; Rg from the
    # slope-ratio stays reported). cond(X) is scale-invariant (invariant 4).
    scaled = ('mw_g_per_mol', 'a2_mol_mL_per_g2')
    reasons = list(scale_notes(
        [r for r in rayleigh_results if r.concentration_g_per_mL != 0],
        mw=scaled[:1], a2=scaled[1:], scale_free=('rg_nm',)))
    if not mf.well_conditioned:
        reasons.append(_ill_conditioned_note(mf.cond, cond_limit, scaled + ('rg_nm',)))
    # Mw and A2 are withheld for their cause, the intercept, and otherwise for their own
    # non-finite values; a Zimm A2 = d/2 reads no intercept, so its own may stand alone.
    a_ok = bool(math.isfinite(a) and a > 0)
    if not a_ok:
        reasons.append(_non_positive_intercept_note(a, scaled))
    elif not math.isfinite(mw):
        reasons.append(Note('mw_not_finite', scaled[:1]))
    if not math.isfinite(a2) and (a_ok or method == 'zimm'):
        reasons.append(Note('a2_not_finite', scaled[1:]))
    partition = mode_point_notes(
        [r for r in rayleigh_results if r.concentration_g_per_mL != 0],
        about=('mw_g_per_mol', 'rg_nm', 'a2_mol_mL_per_g2'))
    reasons += partition
    if not (math.isfinite(rg) and rg > 0):
        # No size, and Mw and A2 untouched by it: the q^2-to-intercept coefficient ratio
        # is not positive, or the intercept it divides by is not.
        if math.isfinite(a) and a > 0:
            reasons.append(_no_rg_note(('rg_nm',), 'slope/intercept', b / a, 'nm²',
                                       'positive'))
        else:
            reasons.append(_no_rg_note(('rg_nm',), 'intercept', a,
                                       'mol/g' if method == 'zimm' else '(mol/g)^½',
                                       'positive'))
    # The qRg regime over the points actually fitted (post-drop). `regime_failed` keys on
    # `isnan`, not `not isfinite`, as `guinier_analysis` does: a NaN qRg (no Rg) is said
    # above, and +inf is a qRg past every limit (rule 12, both tails). `zimm_past_limit` is
    # the one predicate behind the note and the Zimm Mw's verdict, so a NaN qRg flags no Mw.
    # Berry is judged by no limit until the program holds one of its own; past the Zimm
    # limit its Rg is described, never flagged (`core.notes.qualifiers`), and its Mw kept.
    qrg_max = (float(math.sqrt(float(np.max(q2))) * rg) if math.isfinite(rg)
               else float('nan'))
    regime_valid = bool(math.isfinite(qrg_max)
                        and (method == 'berry' or qrg_max <= qrg_max_valid))
    regime_failed = (not math.isnan(qrg_max)) and not (qrg_max <= qrg_max_valid)
    zimm_past_limit = regime_failed and method == 'zimm'
    if zimm_past_limit:
        reasons.append(_qrg_regime_note(('rg_nm', 'mw_g_per_mol'), qrg_max, qrg_max_valid,
                                        'Zimm'))
    notes += reasons

    return ZimmBerryResult(
        method=method, mw_g_per_mol=float(mw), rg_nm=float(rg),
        a2_mol_mL_per_g2=float(a2),
        coef_intercept=a, coef_q2=b, coef_c=d, coef_q2c=e, coef_q2c_ratio=e_ratio,
        r_squared=float(r2),
        n_points=int(q2.size), n_concentrations=int(n_conc), n_angles=int(n_ang),
        is_apparent=False,
        calibrated=all_calibrated,
        dn_dc_supplied=all_dn_dc_supplied,
        blank_measured=all_blank_measured,
        standard_index_known=all(bool(getattr(r, 'standard_index_known', True))
                                 for r in rayleigh_results
                                 if r.concentration_g_per_mL != 0),
        derived_provenance=_shared_derived_provenance(rayleigh_results),
        blank_provenance=_shared_blank_provenance(rayleigh_results),
        # Two SCALE gates (all_calibrated; all_dn_dc_supplied -- K = 1 puts the intercept
        # a = 1/Mw on an arbitrary optical scale), and Rg is ungated by both. a > 0: a
        # non-positive ordinate intercept makes the whole extrapolation non-physical
        # (negative/infinite Mw; for Berry a sign-flipped A2 = d'*a'), so neither Mw nor A2
        # may read reliable -- both are NaN'd above. well_conditioned adds the thin-design
        # gate; math.isfinite closes the float-overflow edge (a underflowing to 0 NaNs Mw).
        # None is a magnitude threshold on Mw (invariant 4; a huge Mw from bad input is GIGO).
        # A Zimm fit past the qRg limit fails its Mw; A2 (c -> 0) never reads the limit.
        mw_reliable=(all_calibrated and all_dn_dc_supplied and all_blank_measured
                     and a_ok and mf.well_conditioned and math.isfinite(mw)
                     and not partition and not zimm_past_limit),
        a2_reliable=(all_calibrated and all_dn_dc_supplied and all_blank_measured
                     and a_ok and mf.well_conditioned and math.isfinite(a2)
                     and not partition),
        # The size: ungated by SCALE, gated by the design it shares with Mw/A2 and by
        # all_blank_measured -- which gates all three, Rg included, because an unmeasured
        # blank is ADDITIVE and survives the q^2-to-intercept ratio the scales cancel in.
        rg_reliable=bool(math.isfinite(rg) and rg > 0 and mf.well_conditioned
                         and all_blank_measured and regime_valid and not partition),
        qrg_max=qrg_max, qrg_max_valid=float(qrg_max_valid),
        qrg_regime_valid=regime_valid,
        concentrations_g_per_mL=conc_levels,
        intercept_per_concentration=np.array(inter_per_c),
        q2_nm2=q2_levels, intercept_per_angle=np.array(inter_per_q),
        mw_se=mw_se, rg_se=rg_se, a2_se=a2_se,
        mw_from_c0_g_per_mol=mw_c0, mw_from_q0_g_per_mol=mw_q0,
        extrapolation_agreement_rel=agree,
        se_estimator=estimator,
        well_conditioned=mf.well_conditioned, cond=mf.cond,
        notes=tuple(notes),
        mode_resolved=any(bool(getattr(r, 'mode_resolved', False))
                          for r in rayleigh_results if r.concentration_g_per_mL != 0),
    )


# ===========================================================================
# Full-angular form-factor P(q) fit (sphere / Gaussian coil)
# ===========================================================================
#
# Where the Zimm/Guinier expansion above uses only the small-qRg limit P(q) ~ 1 - q^2 Rg^2/3,
# this fit uses the FULL form factor (analysis/form_factor.py), so it stays valid to larger
# qRg. See the fit function below for the model and its apparent/thermodynamic split.

# Largest qRg at which a solid-sphere fit is trustworthy: the form factor's FIRST MINIMUM,
# u = qR = 4.49341 (the first root of tan u = u), converted via Rg = R*sqrt(3/5). Past it the
# amplitude (Mw) and the zero location trade off, so Mw can be badly wrong even at R^2 = 1 --
# and R^2 does not flag it. A sphere fit reaching past this has mw/a2 reliability suppressed.
# A general, documented physics criterion (invariant 4), not a system-specific threshold.
# Derivation and sources: Theory-and-Equations-Guide, "The sphere's first minimum"
# (Pedersen 1997 Eq. 55; Guinier & Fournet 1955 Fig. 6). The Debye coil P(q) is monotonic, so
# it has no such limit.
SPHERE_FIRST_MINIMUM_QRG = 4.49340945790906 * math.sqrt(3.0 / 5.0)   # ~= 3.4806

# Mie form-factor fit reliability floor. The Mie sphere is normally the CORRECT model past the
# analytic sphere's first minimum, BUT where the fitted P(q) dips to a deep minimum inside the
# angular range, that angle's ordinate 1/(Mw P) is a small difference of large intensities --
# noise-dominated -- and the Mw/A2 fit is ill-conditioned there, a wrong basin passing both
# `well_conditioned` and R^2. Reliable when the fitted min P over the angles stays >= this
# floor: P ~ 0.22 at qRg ~ 2 recovers to a few %, P ~ 2e-3 past the first minimum is 30%+
# wrong. 0.05 reads as "scattering < 5% of forward at some angle". A documented default
# (invariant 4), of `mie_p_min`. Rg is still reported; only Mw/A2 reliability is suppressed.
MIE_FIT_P_MIN = 0.05

# Smallest physically meaningful linear size (nm) for a form-factor fit -- the LOWER tail of
# the same reliability question the two thresholds above ask from above. Below it the fit has
# not measured a particle. Stated as an Rg, and set so it means the SAME PHYSICAL SIZE as the
# DLS floor: CUMULANT_MIN_RH_NM = 0.2 nm is a RADIUS, and a sphere of radius 0.2 nm has
# Rg = 0.2*sqrt(3/5) = 0.155 nm. A documented physics criterion (invariant 4), not tuned to
# any system, and the default of each form-factor fit's `form_factor_min_size_nm`; see
# _size_above_floor for what it binds on each result and why.
FORM_FACTOR_MIN_SIZE_NM = 0.155


def _size_above_floor(size_nm: float, floor_nm: float = FORM_FACTOR_MIN_SIZE_NM) -> bool:
    """True when a fitted form-factor size clears the size floor ``floor_nm``
    (``FORM_FACTOR_MIN_SIZE_NM`` by default).

    **Why a fit is flagged here: NON-PHYSICALITY, not information loss.** The load-bearing
    claim is only that the fitted body is smaller than any particle a light-scattering
    measurement could have seen -- 0.155 nm as an Rg is a 0.4 nm sphere diameter, narrower
    than a water molecule. It is deliberately NOT "P(q) is flat here, so Mw is meaningless":
    P > 0.995 over the whole angular range for any Rg below ~3 nm, and there Mw is the
    ordinary, correct Debye/Zimm intercept, which this predicate rightly leaves alone.

    **Why neither existing criterion catches it.** That is where P -> 1 does the work: as the
    size goes to zero the shape term drops out of the ordinate, so at ``rg -> 0`` the Mie
    kernel returns all ones -- ``min P = 1`` and ``qRg_max = 0`` -- and a collapsed fit passes
    both upper-tail tests. The floor exists to close that detection gap, not to describe why
    the number is wrong.

    **What it binds.** Rg on the coil/sphere/Mie and ellipsoid results; the OUTER RADIUS on
    the core-shell one, whose contrast-weighted Rg is undefined by design when the contrasts
    cancel. At 0.155 nm either reading is sub-molecular, which is the only claim it makes.

    **Flag, not refusal.** The DLS sibling ``CUMULANT_MIN_RH_NM`` (``analysis/dls/cumulants.py``)
    REJECTS below its floor (owner decision D2); here the fit is FLAGGED and every fitted
    value is still reported, matching the two criteria this one joins. The two floors agree on
    the physical size -- see FORM_FACTOR_MIN_SIZE_NM for the sqrt(3/5) that converts them.

    Shared by all three form-factor builders so the floor cannot drift apart between them
    (rule 10). A non-finite size returns True -- there is no size to judge, and the caller's
    own ``converged`` / ``rg_undefined`` flags already say so.
    """
    return not (math.isfinite(size_nm) and size_nm < floor_nm)


def _sub_molecular_note(label: str, size_nm: float, about: Tuple[str, ...],
                        floor_nm: float = FORM_FACTOR_MIN_SIZE_NM) -> Note:
    """The precondition note of a fit that collapsed below the size floor: smaller than a
    solvent molecule, so not a particle this measurement could have seen.

    One code for all three builders (rule 10); `label` names the quantity that tripped it,
    which differs by result -- see FORM_FACTOR_MIN_SIZE_NM.
    """
    return Note('sub_molecular_size', about,
                {'quantity': label, 'size': float(size_nm), 'floor': float(floor_nm)})


def _nonphysical_notes(inv_mw: float, mw: float, sizes, about: Tuple[str, ...]) -> List[Note]:
    """The precondition notes of a form-factor fit whose solver succeeded on a non-physical
    parameter, one per quantity: 1/Mw not a positive finite number, else Mw not finite (a
    reciprocal that overflowed), and each `(name, value)` of `sizes` not positive and
    finite. One helper for the three builders (rule 10)."""
    out = []
    if not (math.isfinite(inv_mw) and inv_mw > 0):
        out.append(Note('solution_nonphysical', about, {'quantity': '1/Mw'}))
    elif not math.isfinite(mw):
        out.append(Note('solution_nonphysical', about, {'quantity': 'Mw'}))
    for name, value in sizes:
        if not (math.isfinite(value) and value > 0):
            out.append(Note('solution_nonphysical', about, {'quantity': name}))
    return out


def _ff_ill_conditioned_note(cond: float, limit: float, about: Tuple[str, ...]) -> Note:
    """The precondition note of a form-factor fit whose Jacobian is near-singular at the
    optimum (too few or clustered angles): its 1/Mw and A2 are numerically fragile. The
    size is not gated on it (each builder's `rg_reliable`), so the code is not the linear
    designs' `ill_conditioned`, whose ratio sizes share the design."""
    return Note('jacobian_ill_conditioned', about,
                {'cond': float(cond), 'limit': float(limit)})


@dataclass
class FormFactorResult:
    """Full-angular form-factor P(q) fit for Mw, Rg (and, for a sphere, R) and A2.

    `model` is 'sphere' or 'debye' (Gaussian coil). The fit is APPARENT at one
    concentration (`is_apparent=True`; the 2 A2 c term folds into an apparent Mw, as in
    the Debye/Guinier plots) and THERMODYNAMIC across concentrations (`is_apparent=False`,
    A2 resolved). Rg and the sphere radius R come from the angular shape and survive an
    uncalibrated run; Mw / A2 need calibration.
    """
    model: str                        # 'sphere' | 'debye' | 'mie'
    is_apparent: bool                 # True: single concentration; False: extrapolated
    converged: bool                   # least-squares success and physical parameters
    # fitted quantities
    mw_g_per_mol: float               # apparent at single c; thermodynamic across c
    rg_nm: float                      # radius of gyration (all models)
    r_nm: float                       # geometric sphere radius (sphere/Mie; nan for coil)
    a2_mol_mL_per_g2: Optional[float] # thermodynamic only (None when apparent)
    # fit context
    q_nm_inv: np.ndarray              # scattering vectors used (all points, sorted input)
    kc_over_dR: np.ndarray            # ordinate data fit
    concentrations_g_per_mL: np.ndarray  # per-point concentration
    n_points: int
    n_concentrations: int
    n_angles: int
    qrg_max: float                    # max q*Rg over the fitted points (regime indicator)
    r_squared: float
    calibrated: bool                  # was the Rayleigh ratio calibrated?
    mw_reliable: bool                 # False if uncalibrated / non-physical / ill-conditioned
    # Absolute A2 needs calibration and its OWN finiteness gate -- like Mw it can come out
    # non-finite from the (unbounded) fit even when Mw is fine, so it does NOT inherit
    # mw_reliable (mirrors ZimmBerryResult's separate a2_reliable). False when apparent
    # (no A2 is resolved at a single concentration).
    a2_reliable: bool = False
    # False when a SPHERE fit extends past the first form-factor minimum (qrg_max >
    # SPHERE_FIRST_MINIMUM_QRG), where the single-sphere fit is ill-conditioned and Mw can be
    # wrong even at R^2 = 1 -> mw/a2 reliability suppressed; when a MIE fit spans a deep
    # minimum (min P < MIE_FIT_P_MIN); or -- for EVERY model, the monotonic Debye coil
    # included -- when the fitted Rg falls below FORM_FACTOR_MIN_SIZE_NM, where P(q) has
    # collapsed to 1 and the shape term carries no information. Documented physics criteria
    # (invariant 4), not tuned.
    form_factor_valid: bool = True
    # statistical (regression) SEs -- exclude calibration/dn-dc systematics; None when
    # too few points or a degenerate Jacobian. Monte-Carlo-validated (invariant 8).
    mw_se: Optional[float] = None
    rg_se: Optional[float] = None
    r_se: Optional[float] = None
    a2_se: Optional[float] = None
    se_estimator: str = 'hc3'
    # False when the fitted size is sub-molecular (below FORM_FACTOR_MIN_SIZE_NM), where the
    # SIZE is the defect -- not the amplitude. Carried separately from form_factor_valid
    # because that flag's two upper-tail reasons leave Rg sound (past a minimum it is the
    # amplitude that trades off), so gating Rg on it would wrongly withhold a good size.
    # `_ff_rg_offered` reads this by getattr, as the DLS `rh_reliable` siblings are read.
    rg_reliable: bool = True
    # numerical-health of the fit Jacobian (parameter identifiability). well_conditioned
    # =False -> near-degenerate (too few/clustered angles) -> mw_reliable suppressed.
    well_conditioned: bool = True
    cond: float = float('nan')
    # Every note of the fit, as Notes (its preconditions -- non-convergence among them --
    # and the non-positive Kc/dR drop).
    notes: Tuple[Note, ...] = ()
    # `calibrated`'s sibling for the OPTICAL scale (see RayleighRatioResult): False ->
    # no dn/dc was supplied, so K was 1 and Mw/A2 here are on an arbitrary optical
    # scale. The angular sizes are untouched -- K multiplies the whole Kc/dR ordinate
    # and cancels in every coefficient ratio.
    dn_dc_supplied: bool = True
    # Carried forward from the Rayleigh result(s) this fit consumed, so a number
    # cannot be read without its origin whichever exporter writes it. A record; it
    # enters no computation. Empty for an ordinary loaded measurement.
    derived_provenance: str = ''
    # Which solvent blank the contributing points were referenced against, when
    # it is not the sample reference; copied from the Rayleigh result(s) exactly as
    # derived_provenance is, and exported beside it. '' for the common case.
    blank_provenance: str = ''
    # Carried from the Rayleigh result(s) this fit consumed: False -> the blank was not
    # measured at every angle (typed 90 deg value, or no subtraction). ANDed into every
    # reliability verdict on this result, because an additive blank does not cancel the
    # way an unknown SCALE does. See RayleighRatioResult.blank_measured.
    blank_measured: bool = True
    # Carried from the Rayleigh result(s) this fit consumed: False -> k_c was applied with
    # the standard's refractive index blank, so `calibrated` is False for that cause alone
    # (the scale gates name it). See RayleighRatioResult.standard_index_known.
    standard_index_known: bool = True
    # Mie-model provenance (None for the analytic sphere/coil models). The relative index
    # m = n_solute/n_medium is a FIXED user-supplied input to the Mie fit (never fitted --
    # invariant 3), echoed here so it travels into display + export like the DLS reweighting.
    # wavelength_nm + n_medium also let a plot re-evaluate the Mie P(q) overlay (mie_P needs
    # the optical context to map q<->angle and Rg<->size parameter).
    m: Optional[float] = None
    n_solute: Optional[float] = None
    n_medium: Optional[float] = None
    wavelength_nm: Optional[float] = None
    # The thresholds the verdicts above read, echoed as `qrg_max_valid` is: the Mie
    # deep-minimum floor on the fitted P(q) and the size floor.
    mie_p_min: float = MIE_FIT_P_MIN
    form_factor_min_size_nm: float = FORM_FACTOR_MIN_SIZE_NM


# Shared least_squares STOPPING tolerances (ftol/xtol/gtol) for every form-factor fit. A
# stopping criterion, not a physics/basin choice: loosening cannot move a basin, only stop
# polishing sooner. Tighter than scipy's 1e-8 default so a clean forward-model round trip
# converges to numerical precision instead of stalling on its plateau; 1e-10 is the loosest
# value at which every regression gate still passes and clean round-trip errors stay at
# MACHINE precision (TRF's quadratic local convergence overshoots the plateau worry), while
# noisy-data fits run appreciably faster than at 1e-14. Keep the three equal, and re-measure
# the sweep if the fit models change.
_FF_LS_TOL = 1e-10


def _profile_rg_starts(P_of_rg, q, yn, c, thermodynamic, n_grid=180, n_keep=4,
                       qrg_hi=50.0, *, P_of_rg_batch=None):
    """Robust Rg seeds for a MULTI-LOBED form factor (sphere/Mie), by profiling the residual
    over a dense geometric Rg grid.

    At a FIXED Rg the ordinate ``inv_Mw/P(q;Rg) [+ 2 A2 c]`` is LINEAR in (inv_Mw[, A2]), so
    the best-fit residual at that Rg is one exact linear solve -- cheap. Scanning Rg from
    ``qRg ~ 0.3`` (nearly flat) to ``qRg ~ qrg_hi`` (deep into a large particle's oscillations)
    and returning the ``n_keep`` lowest-residual grid points gives least_squares starts that
    reach the global basin even when the P(q) oscillates fast (the failure the q-scaled
    heuristic missed for large Mie particles). Rg values whose optimal inv_Mw is non-physical
    (<= 0) are rejected. ``qrg_hi = 50`` covers Rg up to ~1.6 um (532 nm / water / 150 deg) --
    beyond typical SLS; the grid density (n_grid over the log range) keeps a few points per Mie
    oscillation across it. A particle larger still would escape the grid, but such a fit sits
    deep past the form-factor minima where the deep-minimum reliability gate flags Mw/A2 anyway.
    """
    q = np.asarray(q, dtype=float)
    q_max = float(np.max(q)) if q.size else 0.0
    if not (math.isfinite(q_max) and q_max > 0):
        return []
    rg_grid = np.geomspace(0.3 / q_max, qrg_hi / q_max, n_grid)
    a2_col = 2.0 * np.asarray(c, dtype=float)
    # Optional batched evaluator (perf spec 2026-07-22): one call for the whole grid's P
    # matrix instead of n_grid scalar recurrences. Numerically interchangeable with the
    # per-node path (the batched Mie recurrence reproduces each node's scalar series to
    # ~machine precision, and the SELECTED seeds are identical grid floats -- asserted in
    # tests); everything downstream of P is unchanged.
    P_rows = None
    if P_of_rg_batch is not None:
        P_rows = np.asarray(P_of_rg_batch(rg_grid), dtype=float)
    costs = np.full(rg_grid.size, np.inf)
    for i, rg in enumerate(rg_grid):
        P = P_rows[i] if P_rows is not None else np.asarray(P_of_rg(rg), dtype=float)
        # P is FLOORED (not the whole node dropped) where the form factor grazes a null
        # at a measured angle -- the same semantics as _profile_core_shell_starts, for
        # the same two reasons (family rule 10; owner-approved 2026-07-23): (a) a sphere
        # P(q) has EXACT zeros, so a single grazing angle would discard a competitive
        # (possibly true-basin) Rg seed outright; (b) keep-vs-drop was a knife-edge
        # decision on a ulp-scale quantity (batch and scalar P differ at ~1e-14 near the
        # 1e-12 threshold -- code-review 2026-07-23), whereas flooring deprioritizes a
        # grazing node smoothly via its large-but-finite residual.
        inv_p = 1.0 / np.maximum(P, 1e-12)
        if not np.all(np.isfinite(inv_p)):
            continue                            # non-finite P (should not occur) -> skip node
        X = np.column_stack([inv_p, a2_col]) if thermodynamic else inv_p[:, None]
        beta, *_ = np.linalg.lstsq(X, yn, rcond=None)
        if not (math.isfinite(beta[0]) and beta[0] > 0):   # non-physical/non-finite amplitude
            continue
        resid = X @ beta - yn
        costs[i] = float(resid @ resid)
    # Stable, like the core-shell sibling's `scored.sort`: exactly-equal costs (the
    # floored 1/P saturates over a run of large-Rg nodes) then keep grid order.
    order = np.argsort(costs, kind='stable')
    return [float(rg_grid[i]) for i in order[:n_keep] if math.isfinite(costs[i])]


def _echoed_optics(rayleigh_results) -> Tuple[float, float]:
    """The (wavelength_nm, solvent_refractive_index) pair echoed onto a form-factor
    result — and, for Mie/core-shell, fed to the P(q) kernel's optical context.

    Normally the set is optically uniform and this is just the shared pair. The data
    model permits per-measurement optics (e.g. runs merged from two instruments); then
    the pair with the LARGEST k = n/λ is returned, so the plot layer's back-scatter
    cap 2k stays ≥ every measured q (each point's q ≤ 2k of its own optics) and the
    drawn model curve can never be truncated below a measured datum (code review
    2026-07-24, spec-05 response — the old first-entry read could hand the plot a k
    belonging to a different measurement than the one defining q_max). Pairs are
    sorted before the max so ties break deterministically.
    """
    pairs = sorted({(float(r.wavelength_nm), float(r.solvent_refractive_index))
                    for r in rayleigh_results})
    return max(pairs, key=lambda p: p[1] / p[0])


def form_factor_analysis(
    rayleigh_results,
    model: str = ff.SPHERE,
    estimator: str = 'hc3',
    cond_limit: float = unc.COND_LIMIT,
    m: Optional[float] = None,
    *,
    mie_p_min: float = MIE_FIT_P_MIN,
    form_factor_min_size_nm: float = FORM_FACTOR_MIN_SIZE_NM,
) -> FormFactorResult:
    """Fit the full form factor P(q) to Kc/dR across angles (and concentrations).

    The model is ``Kc/dR(q, c) = 1/(Mw P(q; Rg)) + 2 A2 c``. It is nonlinear in Rg (P is),
    so the parameters are found by least squares seeded from the linear Zimm/Debye
    estimate. With ONE concentration the ``+2 A2 c`` term is not separable from 1/Mw at
    q -> 0, so that fit is APPARENT (2 parameters: 1/Mw_app, Rg) exactly like the
    Debye/Guinier apparent plots; with two or more concentrations the full 3-parameter
    model is fit and the result is THERMODYNAMIC. Rg (and the sphere radius R) are
    scale-independent and survive an uncalibrated run; Mw / A2 inherit the usual
    calibration gate.

    Inputs not prescribed by the source: the deep-minimum reliability floor
    ``MIE_FIT_P_MIN`` (0.05; no source prints a floor on P(q)), the size floor
    ``FORM_FACTOR_MIN_SIZE_NM`` (0.155 nm, the owner-decided DLS floor
    ``CUMULANT_MIN_RH_NM`` converted to an Rg), the sphere model's limit
    ``SPHERE_FIRST_MINIMUM_QRG`` (the form factor's first minimum, a mathematical value;
    suppressing Mw/A2 reliability past it is the program's choice), and the unweighted least
    squares on the normalised ordinate, where Pedersen 1997 Eq. 1 (p. 174) minimises a
    chi-square weighted by each point's standard deviation (the floor is this program's
    stand-in for that weighting near a minimum); prescribed: the model
    Kc/dR = 1/(Mw P) + 2 A2 c (Chu 1991 Eq. 2.1.8, p. 18). Not yet read against a source:
    the seed from the linear Zimm/Debye estimate, the Rg-grid multistart, the bounds and
    the solver tolerances.

    Parameters
    ----------
    rayleigh_results : RayleighRatioResult or sequence of them
        One per concentration (a c = 0 solvent reference is ignored). One non-zero
        concentration -> apparent fit (1/Mw_app, Rg); two or more -> thermodynamic fit
        (1/Mw, Rg, A2).
    model : str
        'sphere' (solid sphere; returns geometric R and Rg), 'debye' (Gaussian coil;
        returns Rg), or 'mie' (full Mie sphere -- the general large-x / high-contrast case;
        returns geometric R and Rg, and REQUIRES ``m``). See analysis/form_factor.py.
    estimator : str
        Covariance estimator for the SEs, 'hc3' (default, never under-reports) or 'ols'
        (opt-in, invariant 8 clause A).
    cond_limit : float
        Numerical-health ceiling on the fit Jacobian's condition number.
    m : float, optional
        Relative refractive index ``n_solute/n_medium`` -- REQUIRED for model='mie' (it enters
        the Mie scattering arithmetic), unused otherwise. A FIXED user-supplied input, never
        fitted (invariant 3).
    mie_p_min, form_factor_min_size_nm : float
        The Mie deep-minimum floor on the fitted P(q) and the size floor (defaults
        ``MIE_FIT_P_MIN`` and ``FORM_FACTOR_MIN_SIZE_NM``), echoed on the result.

    Returns
    -------
    FormFactorResult

    Raises
    ------
    ValueError
        For an unknown model, model='mie' without a valid ``m``, or fewer than two angles
        (a two-parameter shape fit needs at least two distinct q-points to constrain Rg).
    """
    if model == ff.ELLIPSOID:
        raise ValueError(
            "the ellipsoid model is not fit by form_factor_analysis (it returns a single "
            "result, but an ellipsoid P(q) fit is multi-valued); call "
            "ellipsoid_form_factor_analysis, which returns a LIST of solutions."
        )
    if model not in ff._RG_ONLY_MODELS and model != ff.MIE:
        raise ValueError(
            f"unknown form-factor model {model!r}; expected one of "
            f"{ff._RG_ONLY_MODELS + (ff.MIE,)}."
        )
    if model == ff.MIE and not (m is not None and math.isfinite(m) and m > 0):
        raise ValueError(
            "the Mie form-factor fit needs the relative refractive index m = "
            "n_solute/n_medium (a positive finite number, from the hand-entered solute "
            "refractive index); it is unset."
        )
    if isinstance(rayleigh_results, RayleighRatioResult):
        rayleigh_results = [rayleigh_results]

    q2, c, y, q, n_conc, n_ang, thermodynamic, drop_notes = _form_factor_inputs(
        rayleigh_results)
    # Parameters: p = 2 (apparent 1/Mw, Rg) or 3 (thermodynamic +A2). Need >= 2 distinct
    # angles to constrain the shape Rg (two q-points, as Debye/Zimm need for their slope),
    # and >= p total points for identifiability. With >= 2 angles the apparent case has
    # n_points = n_ang >= 2 = p, and the thermodynamic case (n_conc >= 2) has >= 4 >= 3
    # points -- so a single angle floor of 2 covers both. (An SE still needs n - p >= 2;
    # a barely-determined fit just reports the point estimate with no SE, like a 2-angle
    # Debye.) This matches zimm_analysis, which accepts 2 angles x >= 2 concentrations.
    if n_ang < 2:
        raise ValueError(
            "Form-factor P(q) fit needs at least two angles to resolve the shape; "
            f"got {n_ang}."
        )
    all_calibrated, all_dn_dc_supplied, all_blank_measured = _aggregate_scale_flags(
        rayleigh_results)
    # Optical context: the Mie P(q) needs it in the arithmetic (size parameter + angle),
    # and it is echoed on EVERY result so the plot layer can cap its model grid at
    # back-scatter q = 2k (spec-05 review response: previously Mie-only, which left the
    # sphere/coil overlay uncappable). n_solute is Mie provenance.
    wl_nm, n_med = _echoed_optics(rayleigh_results)
    n_sol = (m * n_med) if (model == ff.MIE and m is not None) else None

    # Fit a NORMALIZED ordinate yn = y / y_scale so the optimization is well-scaled at
    # any ordinate scale; the factor multiplies only 1/Mw and A2 and leaves Rg untouched.
    # Rescaled back to physical units after the fit. Why it refuses rather than falling
    # back to 1.0: see _ordinate_scale.
    y_scale = _ordinate_scale(y)
    yn = y / y_scale

    # ---- seed from the linear Zimm/Debye estimate (robust starting point) ----
    if thermodynamic:
        A = np.column_stack([np.ones_like(q2), q2, c])
        seed = unc.multilinear_fit(A, yn, estimator, cond_limit).coeffs
        inv_mw0, b0, a2_0 = float(seed[0]), float(seed[1]), float(seed[2]) / 2.0
    else:
        lf = unc.linear_fit(q2, yn, estimator, cond_limit)
        inv_mw0, b0, a2_0 = float(lf.intercept), float(lf.slope), 0.0
    # Rg^2 = 3 b / a from the linear seed (Zimm relation); fall back to a mild default if
    # the seed is non-physical (noise), so least_squares always starts inside the domain.
    if math.isfinite(inv_mw0) and inv_mw0 > 0 and b0 / inv_mw0 > 0:
        rg0 = math.sqrt(3.0 * b0 / inv_mw0)
    else:
        rg0 = 10.0
    if not (inv_mw0 > 0 and math.isfinite(inv_mw0)):
        pos = yn[yn > 0]
        inv_mw0 = float(pos.min()) if pos.size else 1.0

    # ---- nonlinear least squares on the FULL (normalized) form-factor ordinate ----
    def _P(rg):
        if model == ff.MIE:
            return ff.mie_P(q, rg, m, wavelength_nm=wl_nm, n_medium=n_med)
        return ff.form_factor_P(model, q, rg)

    def residuals(theta):
        P = _P(theta[1])                                   # theta = (inv_mw, rg[, a2])
        a2 = theta[2] if thermodynamic else None
        return ff.model_ordinate(P, theta[0], a2, c) - yn

    if thermodynamic:
        lb = [0.0, 0.0, -np.inf]
        ub = [np.inf, np.inf, np.inf]
    else:
        lb = [0.0, 0.0]
        ub = [np.inf, np.inf]

    def _make_theta0(rg_start):
        return ([inv_mw0, rg_start, a2_0] if thermodynamic else [inv_mw0, rg_start])

    # MULTI-START over Rg. The sphere AND Mie P(q) are MULTI-LOBED (minima enter the angular
    # range around qRg ~ 4.5 for the sphere, and Mie oscillates ever faster with size), so a
    # single start from the linear Rg seed can land in a WRONG local minimum and return a
    # badly-wrong Mw/Rg that still "converges", with no goodness-of-fit signal to catch it --
    # and Mie stays VALID past the minimum at high contrast, so a wrong-basin Mie fit would
    # sail through every reliability gate. The ordinate is LINEAR in (inv_Mw[, A2]) at FIXED
    # Rg, so profile the residual over a dense geometric Rg grid and seed least_squares from
    # the best grid points. The monotonic Debye coil keeps the cheap q-scaled starts.
    q_typ = float(np.median(q[q > 0])) if np.any(q > 0) else 1.0
    if model in (ff.SPHERE, ff.MIE):
        # Mie seeds via the batched grid evaluator (one lockstep recurrence sweep instead
        # of 180 scalar ones -- results-neutral, see _profile_rg_starts); the analytic
        # sphere P is cheap numpy already, so it keeps the per-node path.
        _P_batch = None
        if model == ff.MIE:
            def _P_batch(rg_arr):
                return ff.mie_P_batch(q, rg_arr, m, wavelength_nm=wl_nm, n_medium=n_med)
        rg_starts = _profile_rg_starts(_P, q, yn, c, thermodynamic,
                                       P_of_rg_batch=_P_batch) + [rg0]
    else:
        # Debye coil (monotonic P, no lobes): the cheap q-scaled spread suffices.
        rg_starts = [rg0] + [tgt / q_typ for tgt in (1.0, 2.0, 3.0, 4.0)]
    cands = []
    for rg_start in rg_starts:
        if not (rg_start > 0 and math.isfinite(rg_start)):
            continue
        # x_scale='jac' auto-scales the parameter-magnitude spread; tolerances: _FF_LS_TOL.
        cands.append(least_squares(residuals, _make_theta0(rg_start), bounds=(lb, ub),
                                   method='trf', x_scale='jac',
                                   ftol=_FF_LS_TOL, xtol=_FF_LS_TOL, gtol=_FF_LS_TOL))
    # Keep the lowest-cost start; a non-finite cost sorts to +inf so a NaN first candidate
    # can never win over a finite one (a plain `cand.cost < best.cost` would, since any
    # comparison with NaN is False).
    res = min(cands, key=lambda c: c.cost if math.isfinite(c.cost) else float('inf'))

    # ---- rescale the fitted parameters back to physical (Kc/dR) units ----
    # Physical param = D @ theta_n with D = diag([y_scale, 1, y_scale]) (Rg is untouched);
    # its covariance transforms as D cov_n D (a linear reparameterization).
    inv_mw = float(res.x[0]) * y_scale
    rg = float(res.x[1])
    a2 = float(res.x[2]) * y_scale if thermodynamic else None
    mw = 1.0 / inv_mw if (math.isfinite(inv_mw) and inv_mw > 0) else float('nan')
    # Sphere and Mie are both solid spheres, so R = Rg*sqrt(5/3) is meaningful for both; the
    # Gaussian coil has no geometric radius (nan).
    r_geom = (ff.sphere_radius_from_rg(rg) if model in (ff.SPHERE, ff.MIE)
              else float('nan'))
    converged = bool(res.success and math.isfinite(inv_mw) and inv_mw > 0
                     and math.isfinite(rg) and rg > 0 and math.isfinite(mw))

    # ---- covariance -> SEs (HC3 sandwich on the fit Jacobian; delta method) ----
    cov_n, cond, well_cond = unc.covariance_from_jacobian(
        res.jac, res.fun, estimator, cond_limit)
    scale_vec = np.array([y_scale, 1.0, y_scale][:cov_n.shape[0]])
    cov = cov_n * np.outer(scale_vec, scale_vec)
    mw_se = rg_se = r_se = a2_se = None
    if np.all(np.isfinite(cov)) and converged:
        p = cov.shape[0]
        # Mw = 1/inv_mw (nonlinear reciprocal -> MC-validated, invariant 8).
        jac_mw = np.zeros(p); jac_mw[0] = -1.0 / unc.power_or_nan(inv_mw, 2)
        mw_se = unc.se_or_none(unc.propagate(jac_mw, cov))
        # Rg is a direct fit parameter; R = Rg*sqrt(5/3) is a linear scaling of it (sphere
        # and Mie -- both solid spheres; the coil has no geometric radius).
        jac_rg = np.zeros(p); jac_rg[1] = 1.0
        rg_se = unc.se_or_none(unc.propagate(jac_rg, cov))
        if model in (ff.SPHERE, ff.MIE) and rg_se is not None:
            r_se = rg_se * ff._RG_TO_R
        if thermodynamic:
            jac_a2 = np.zeros(p); jac_a2[2] = 1.0
            a2_se = unc.se_or_none(unc.propagate(jac_a2, cov))

    # ---- flags + regime indicator ----
    # R^2 on the NORMALIZED ordinate the fit actually minimized (res.fun and yn share the
    # y_scale, so the ratio — and R^2 — is scale-invariant either way).
    ss_res = float(np.sum(res.fun ** 2))
    ss_tot = float(np.sum((yn - yn.mean()) ** 2))
    r2 = unc.r_squared_or_nan(ss_res, ss_tot)
    qrg_max = float(np.max(q) * rg) if (math.isfinite(rg) and rg > 0) else float('nan')

    # Reliability of the shape fit past a form-factor minimum. Sphere: flag when the largest
    # qRg passes the first minimum (Mw/zero-location trade off; R^2 does not catch it). Mie:
    # flag when the FITTED P(q) dips to a deep minimum in the angular range -- there the
    # ordinate 1/(Mw P) is noise-dominated and the Mw/A2 fit is ill-conditioned; that one
    # criterion covers both the low-contrast (m -> 1 -> RDG-like zero) and high-contrast
    # (deep first Mie minimum) failure modes flagged in review. The size floor is the LOWER
    # tail of the same predicate and applies to EVERY model, the monotonic Debye coil
    # included: a sub-molecular Rg is not a shape result whatever P(q) does. Rg still reported.
    past_first_min = math.isfinite(qrg_max) and qrg_max > SPHERE_FIRST_MINIMUM_QRG
    above_floor = _size_above_floor(rg, form_factor_min_size_nm)
    if model == ff.SPHERE:
        form_factor_valid = above_floor and not past_first_min
    elif model == ff.MIE:
        mie_min_p = (float(np.min(_P(rg)))
                     if (math.isfinite(rg) and rg > 0 and converged) else float('nan'))
        form_factor_valid = above_floor and not (math.isfinite(mie_min_p)
                                                 and mie_min_p < mie_p_min)
    else:
        form_factor_valid = above_floor   # Debye coil (monotonic P) — no minimum above it

    scaled = ('mw_g_per_mol', 'a2_mol_mL_per_g2')
    notes: list = list(drop_notes)
    notes += scale_notes([r for r in rayleigh_results if r.concentration_g_per_mL != 0],
                         mw=scaled[:1], a2=scaled[1:], scale_free=('rg_nm',))
    if not res.success:
        not_converged = Note('ff_not_converged', scaled + ('rg_nm',))
        notes.append(not_converged)
        _warn_note(not_converged, stacklevel=2)
    elif not converged:
        notes += _nonphysical_notes(inv_mw, mw, (('Rg', rg),), scaled + ('rg_nm',))
    if thermodynamic and a2 is not None and not math.isfinite(a2):
        notes.append(Note('a2_not_finite', scaled[1:]))
    # The design's conditioning is its own criterion and is named beside the shape-fit
    # reasons, never instead. Between those two the size floor outranks the minimum: a
    # collapsed fit trips NEITHER minimum, so describing it as "past the first minimum"
    # would state the opposite of what happened (qRg_max -> 0).
    if not well_cond:
        notes.append(_ff_ill_conditioned_note(cond, cond_limit, scaled))
    if not above_floor:
        notes.append(_sub_molecular_note('Rg', rg, scaled + ('rg_nm',),
                                         form_factor_min_size_nm))
    elif not form_factor_valid and model == ff.MIE:
        notes.append(Note('deep_minimum', scaled, {'limit': float(mie_p_min)}))
    elif not form_factor_valid:
        notes.append(Note('past_first_minimum', scaled,
                          {'qrg': float(qrg_max), 'limit': float(SPHERE_FIRST_MINIMUM_QRG)}))

    return FormFactorResult(
        model=model, is_apparent=not thermodynamic, converged=converged,
        mw_g_per_mol=float(mw), rg_nm=float(rg), r_nm=float(r_geom),
        a2_mol_mL_per_g2=a2,
        q_nm_inv=q, kc_over_dR=y, concentrations_g_per_mL=c,
        n_points=int(q.size), n_concentrations=n_conc, n_angles=n_ang,
        qrg_max=qrg_max, r_squared=float(r2), form_factor_valid=form_factor_valid,
        calibrated=all_calibrated,
        dn_dc_supplied=all_dn_dc_supplied, blank_measured=all_blank_measured,
        standard_index_known=all(bool(getattr(r, 'standard_index_known', True))
                                 for r in rayleigh_results
                                 if r.concentration_g_per_mL != 0),
        derived_provenance=_shared_derived_provenance(rayleigh_results),
        blank_provenance=_shared_blank_provenance(rayleigh_results),
        # Rg/R survive uncalibrated AND dn/dc-absent runs alike (the ordinate scale cancels
        # -- the same normalization the y_scale rescaling below already encodes); Mw needs
        # BOTH scales known, a physical (converged, inv_mw>0) fit, a well-conditioned
        # Jacobian (invariant 4/6), and a fit within the sphere form factor's first
        # minimum (past it Mw is ill-conditioned).
        # `blank_measured` joins them, and unlike the two scales it also gates the SIZE:
        # an unmeasured blank is ADDITIVE, so it does not cancel in the ordinate
        # normalization the sizes rely on (angle-dependence is an extra, not the reason).
        mw_reliable=(all_calibrated and all_dn_dc_supplied and all_blank_measured
                     and converged and well_cond and form_factor_valid),
        # Absolute A2 (thermodynamic only) shares Mw's calibration/conditioning/validity gates
        # but carries its OWN finiteness check -- the unbounded virial parameter can diverge
        # while Mw stays finite (mirrors ZimmBerryResult.a2_reliable). Always False when
        # apparent (single c resolves no A2).
        a2_reliable=(thermodynamic and all_calibrated and all_dn_dc_supplied
                     and all_blank_measured and converged and well_cond
                     and form_factor_valid and a2 is not None and math.isfinite(a2)),
        # The floor's tail is the one that condemns the SIZE, so it -- not the whole
        # form_factor_valid -- gates the Rg offer (finding 1, round 9c44).
        rg_reliable=(converged and above_floor and all_blank_measured),
        mw_se=mw_se, rg_se=rg_se, r_se=r_se, a2_se=a2_se, se_estimator=estimator,
        well_conditioned=well_cond, cond=cond,
        notes=tuple(notes),
        # m/n_solute are Mie-only provenance (None for the analytic models; m is a fixed
        # user input, not fitted). The wavelength/medium pair is echoed for EVERY model so
        # the plot layer can form k = 2πn/λ and cap the drawn grid at q = 2k (spec-05
        # review response — the sphere/coil overlay was the uncapped odd sibling out).
        m=(m if model == ff.MIE else None),
        n_solute=n_sol, n_medium=float(n_med),
        wavelength_nm=float(wl_nm),
        mie_p_min=float(mie_p_min), form_factor_min_size_nm=float(form_factor_min_size_nm),
    )


# --------------------------------------------------------------------------------------
# Ellipsoid-of-revolution form-factor fit (Phase 2). Unlike the sphere/coil fit above, a
# shape-from-scattering inverse is NOT single-valued: at low-to-moderate qRg a prolate and
# an oblate ellipsoid (and a sphere) reproduce the same P(q). So this returns a LIST of
# EllipsoidFormFactorResult -- one per fitting solution -- exactly like the DPLS ellipsoid
# inverse (analysis/depolarization.ellipsoid_dimensions_from_diffusion): every comparably-
# fitting solution is reported, `degenerate` is set when more than one fits, none is
# silently preferred, and a no-fit is a single `converged=False` sentinel (never []).

# Axial-ratio (epsilon = polar/equatorial) bounds for the fit. Wide but finite; extreme
# ratios beyond these are not resolvable from a light-scattering P(q) anyway.
ELLIPSOID_EPS_MIN = 0.05
ELLIPSOID_EPS_MAX = 20.0
# Two distinct solutions count as the SAME (dedup) when their axial ratios agree to within
# this fraction in log-epsilon AND their Rg agree to this relative tolerance -- the shared
# near-sphere limit (eps ~ 1) collapses to one solution.
_ELLIPSOID_CLUSTER_TOL = 0.05
# A solution joins the reported (degenerate) set when its RMS residual exceeds the best
# solution's by less than this fraction of the DATA RMS -- i.e. its extra misfit is a
# negligible part of the signal. This is regime-aware: at low qRg prolate and oblate both
# fit essentially perfectly (kept, degenerate); at high qRg a wrong branch's misfit exceeds
# the tolerance (dropped, resolved). A documented general criterion (invariant 4), not tuned.
_ELLIPSOID_DEGENERACY_TOL = 0.02
# The axial ratio is UNRESOLVED (report Rg, flag eps) when the fit's FRACTIONAL sensitivity
# to eps is negligible next to its fractional sensitivity to Rg -- eps barely moves the
# residual, exactly the low-qRg limit where P(q) -> 1 - (qRg)^2/3 for every shape. The
# measure d(resid)/d(ln eps) / d(resid)/d(ln rg) is dimensionless and depends only on the
# qRg regime (P depends on q, Rg only through q*Rg), NOT on absolute size. Default 0.1: the
# axial ratio is "resolved" once eps moves the fit at least ~10% as strongly (fractionally)
# as Rg does -- a boundary near qRg ~ 1.5, general and noise-independent. A relative-SE
# ceiling backs it up for the noisy borderline. The defaults of the fit's parameters.
_ELLIPSOID_AXIAL_SENS_TOL = 0.1
_ELLIPSOID_AXIAL_MAX_REL_SE = 0.5


@dataclass
class EllipsoidFormFactorResult:
    """One ellipsoid-of-revolution P(q) fit solution (Pedersen 1997 Eq. 61).

    `ellipsoid_form_factor_analysis` returns a LIST of these -- one per solution that fits
    the angular Kc/dR comparably (a prolate and an oblate ellipsoid can reproduce the same
    P(q)); `degenerate` is True on every member when more than one fits. Size is the radius
    of gyration `rg_nm` (comparable to the other form-factor models); `axial_ratio`
    (epsilon = polar/equatorial: >1 prolate, <1 oblate) is the shape, with the equatorial
    radius R and polar semi-axis eps*R derived for display. The fit is APPARENT at one
    concentration (`is_apparent=True`) and THERMODYNAMIC across concentrations. Rg and the
    axial ratio come from the angular shape and survive an uncalibrated run; Mw / A2 need
    calibration.

    At low qRg the axial ratio is unidentifiable (all shapes share the Guinier limit):
    `axial_ratio_resolved` and `axial_ratio_reliable` are then False, the axial-ratio SE is
    withheld, and Rg/Mw are still reported and gated only on the CORE (inv_Mw, Rg[, A2])
    conditioning -- so a well-determined Rg is not held hostage to an unresolvable shape. An
    identified axial ratio whose relative SE is past the precision limit is resolved and
    not reliable: its SE is kept, and its precondition names the relative SE. An
    unmeasured blank fails `axial_ratio_reliable` as it fails every reliability verdict
    here, and leaves `axial_ratio_resolved` and the SE as they are.
    """
    model: str                        # always 'ellipsoid' (uniform with FormFactorResult)
    is_apparent: bool
    converged: bool
    degenerate: bool                  # >1 solution fits this data comparably
    is_prolate: bool                  # axial_ratio > 1
    # fitted quantities
    mw_g_per_mol: float               # apparent at single c; thermodynamic across c
    rg_nm: float                      # radius of gyration
    axial_ratio: float                # epsilon = polar/equatorial semi-axis
    semi_axis_equatorial_nm: float    # R (derived from Rg, eps)
    semi_axis_polar_nm: float         # eps * R (derived)
    a2_mol_mL_per_g2: Optional[float] # thermodynamic only (None when apparent)
    # fit context
    q_nm_inv: np.ndarray
    kc_over_dR: np.ndarray
    concentrations_g_per_mL: np.ndarray
    n_points: int
    n_concentrations: int
    n_angles: int
    qrg_max: float                    # max q*Rg over the fitted points (regime indicator)
    r_squared: float
    calibrated: bool
    mw_reliable: bool                 # False if uncalibrated / non-physical / ill-conditioned
    a2_reliable: bool = False
    # The axial ratio's trust verdict, resolved AND precise AND the blank measured: False
    # when the ratio is unresolved (qRg too small to distinguish shape), resolved and
    # imprecise (its relative SE past `axial_max_rel_se`), or the blank was not measured
    # (`blank_measured`, as on every reliability verdict here); each cause names itself.
    axial_ratio_reliable: bool = True
    # The identifiability half of the verdict above: True when the fit's sensitivity to
    # the axial ratio clears `axial_sens_tol` on a well-conditioned full design, so the ±
    # is computed and shown. A resolved ratio fails the verdict only when imprecise
    # (`axial_ratio_imprecise`) or blankless, its ± shown; an unresolved one fails it with
    # the ± withheld. The blank never moves this field.
    axial_ratio_resolved: bool = True
    # Conservative proxy: past the sphere first-minimum qRg the single-particle fit is
    # ill-conditioned (Mw/zero-location trade off); the orientation-averaged ellipsoid P
    # smears the minimum, so this is a conservative (early) flag -- Mw/A2 reliability
    # suppressed past it, Rg still reported. Also False below FORM_FACTOR_MIN_SIZE_NM, the
    # same predicate's lower tail, where P(q) has collapsed to 1.
    form_factor_valid: bool = True
    # statistical (regression) SEs -- exclude calibration/dn-dc systematics; None when too
    # few points, a degenerate Jacobian, or (axial ratio) unresolved. MC-validated (inv 8).
    mw_se: Optional[float] = None
    rg_se: Optional[float] = None
    axial_ratio_se: Optional[float] = None
    a2_se: Optional[float] = None
    # False when the fitted Rg is sub-molecular -- see FormFactorResult.rg_reliable.
    rg_reliable: bool = True
    se_estimator: str = 'hc3'
    # Conditioning of the CORE (inv_Mw, Rg[, A2]) design -- deliberately excludes the eps
    # direction so Mw/Rg reliability is independent of axial-ratio identifiability.
    well_conditioned: bool = True
    cond: float = float('nan')
    n_solutions: int = 1              # size of the degenerate set this solution belongs to
    # Every note of the solution, as Notes; its degeneracy is a qualifier read off
    # `degenerate` / `n_solutions` / `is_prolate` (core.notes.qualifiers), never a note.
    notes: Tuple[Note, ...] = ()
    # `calibrated`'s sibling for the OPTICAL scale (see RayleighRatioResult): False ->
    # no dn/dc was supplied, so K was 1 and Mw/A2 here are on an arbitrary optical
    # scale. The angular sizes are untouched -- K multiplies the whole Kc/dR ordinate
    # and cancels in every coefficient ratio.
    dn_dc_supplied: bool = True
    # Carried forward from the Rayleigh result(s) this fit consumed, so a number
    # cannot be read without its origin whichever exporter writes it. A record; it
    # enters no computation. Empty for an ordinary loaded measurement.
    derived_provenance: str = ''
    # Which solvent blank the contributing points were referenced against, when
    # it is not the sample reference; copied from the Rayleigh result(s) exactly as
    # derived_provenance is, and exported beside it. '' for the common case.
    blank_provenance: str = ''
    # Carried from the Rayleigh result(s) this fit consumed: False -> the blank was not
    # measured at every angle (typed 90 deg value, or no subtraction). ANDed into every
    # reliability verdict on this result, because an additive blank does not cancel the
    # way an unknown SCALE does. See RayleighRatioResult.blank_measured.
    blank_measured: bool = True
    # Carried from the Rayleigh result(s) this fit consumed: False -> k_c was applied with
    # the standard's refractive index blank, so `calibrated` is False for that cause alone
    # (the scale gates name it). See RayleighRatioResult.standard_index_known.
    standard_index_known: bool = True
    # Optical context (fixed user inputs), echoed like the Mie/core-shell siblings so the
    # plot layer can form k = 2*pi*n/lambda and cap the model grid at back-scatter q = 2k
    # (spec engineering-05). Always populated by the fit (the ellipsoid has one model, so
    # no Mie-style conditional); Optional only for hand-built stand-ins.
    n_medium: Optional[float] = None
    wavelength_nm: Optional[float] = None
    # The thresholds the verdicts above read, echoed as `qrg_max_valid` is: the axial-ratio
    # sensitivity and precision limits and the size floor.
    axial_sens_tol: float = _ELLIPSOID_AXIAL_SENS_TOL
    axial_max_rel_se: float = _ELLIPSOID_AXIAL_MAX_REL_SE
    form_factor_min_size_nm: float = FORM_FACTOR_MIN_SIZE_NM


#: The no-fit precondition of the ellipsoid fit: on its sentinel, and its warning's message.
_ELLIPSOID_NO_FIT_NOTE = Note('no_ff_solution', (), {'model': 'ellipsoid of revolution'})


def _ellipsoid_no_fit(q, y, c, n_conc, n_ang, thermodynamic, calibrated, estimator,
                      n_med, wl_nm, *, dn_dc_supplied, blank_measured=True, derived_provenance='',
        blank_provenance='', notes=(), standard_index_known=True, thresholds=None):
    """The single converged=False sentinel returned when no ellipsoid fits (never []).
    `notes` is the drop and scale notes of the input; the no-fit note joins them."""
    return EllipsoidFormFactorResult(
        model=ff.ELLIPSOID, is_apparent=not thermodynamic, converged=False,
        degenerate=False, is_prolate=False,
        # Nothing was fitted, so the nan Rg is not a size to offer -- matching the
        # core-shell no-fit sentinel, which set this in the same change (round 943d f4).
        rg_reliable=False,
        mw_g_per_mol=float('nan'), rg_nm=float('nan'), axial_ratio=float('nan'),
        semi_axis_equatorial_nm=float('nan'), semi_axis_polar_nm=float('nan'),
        a2_mol_mL_per_g2=None,
        q_nm_inv=q, kc_over_dR=y, concentrations_g_per_mL=c,
        n_points=int(q.size), n_concentrations=n_conc, n_angles=n_ang,
        qrg_max=float('nan'), r_squared=float('nan'), calibrated=calibrated,
        dn_dc_supplied=dn_dc_supplied,
        blank_measured=blank_measured,
        derived_provenance=derived_provenance,
        blank_provenance=blank_provenance,
        mw_reliable=False, a2_reliable=False, axial_ratio_reliable=False,
        axial_ratio_resolved=False, standard_index_known=standard_index_known,
        se_estimator=estimator, well_conditioned=False,
        n_solutions=0, notes=tuple(notes) + (_ELLIPSOID_NO_FIT_NOTE,),
        n_medium=float(n_med), wavelength_nm=float(wl_nm), **(thresholds or {}))


def ellipsoid_form_factor_analysis(
    rayleigh_results,
    estimator: str = 'hc3',
    cond_limit: float = unc.COND_LIMIT,
    degeneracy_tol: float = _ELLIPSOID_DEGENERACY_TOL,
    *,
    axial_sens_tol: float = _ELLIPSOID_AXIAL_SENS_TOL,
    axial_max_rel_se: float = _ELLIPSOID_AXIAL_MAX_REL_SE,
    form_factor_min_size_nm: float = FORM_FACTOR_MIN_SIZE_NM,
) -> List[EllipsoidFormFactorResult]:
    """Fit the ellipsoid-of-revolution form factor P(q) to Kc/dR across angles (and c).

    Returns a LIST of EllipsoidFormFactorResult -- one per axial-ratio solution that fits
    the data comparably (a prolate and an oblate ellipsoid, and at limited q a sphere, can
    reproduce the same P(q)); `degenerate` is set on every member when more than one fits.
    The list is ordered by ASCENDING (axial ratio, Rg) -- a deterministic presentation
    convention, not fit preference: accepted solutions are statistically indistinguishable,
    so cost cannot honestly rank them. A no-fit yields a single `converged=False` sentinel,
    never an empty list. Mirrors the DPLS ellipsoid inverse
    (analysis/depolarization.ellipsoid_dimensions_from_diffusion).

    The ordinate is Kc/dR = 1/(Mw P(q; Rg, eps)) + 2 A2 c, with P the orientation-averaged
    ellipsoid form factor (analysis/form_factor.ellipsoid_P, Pedersen 1997 Eq. 61). One
    non-zero concentration -> apparent (1/Mw_app, Rg, eps); two or more -> thermodynamic
    (1/Mw, Rg, eps, A2). The fit is a multi-start nonlinear least squares seeded across the
    oblate/sphere/prolate axial-ratio range; distinct low-cost minima are clustered and
    every solution whose extra misfit is below `degeneracy_tol` of the data RMS is reported.

    Parameters
    ----------
    rayleigh_results : RayleighRatioResult or sequence of them
        One per concentration (a c = 0 solvent reference is ignored).
    estimator : str
        Covariance estimator for the SEs, 'hc3' (default) or 'ols' (invariant 8 clause A).
    cond_limit : float
        Numerical-health ceiling on the core-design condition number.
    degeneracy_tol : float
        A solution joins the reported set when its RMS residual exceeds the best solution's
        by less than this fraction of the data RMS (default 2 %). Documented, overridable.
    axial_sens_tol, axial_max_rel_se, form_factor_min_size_nm : float
        The axial ratio's identifiability threshold (its fractional sensitivity against
        Rg's) and precision limit (its relative SE), and the size floor; defaults
        ``_ELLIPSOID_AXIAL_SENS_TOL``, ``_ELLIPSOID_AXIAL_MAX_REL_SE`` and
        ``FORM_FACTOR_MIN_SIZE_NM``, echoed on every result.

    Raises
    ------
    ValueError
        For fewer than three distinct angles (a three-parameter shape fit -- 1/Mw, Rg, eps
        -- needs at least three angular points to constrain the axial ratio).
    """
    if isinstance(rayleigh_results, RayleighRatioResult):
        rayleigh_results = [rayleigh_results]

    q2, c, y, q, n_conc, n_ang, thermodynamic, drop_notes = _form_factor_inputs(
        rayleigh_results)
    # The ellipsoid adds an axial-ratio parameter over the sphere/coil fit, so it needs
    # angular CURVATURE to constrain the shape: at least three distinct angles (two only fix
    # a slope -> Rg, blind to eps). This is stricter than the >= 2 angles the Rg-only sphere/
    # coil fit accepts, by design.
    if n_ang < 3:
        raise ValueError(
            "Ellipsoid form-factor P(q) fit needs at least three angles to resolve the "
            f"axial ratio; got {n_ang}."
        )
    all_calibrated, all_dn_dc_supplied, all_blank_measured = _aggregate_scale_flags(
        rayleigh_results)
    # Optical context for the result (display/plot provenance; the Pedersen P(q) itself
    # never needs it) — the cap-safe extraction shared with the Mie and core-shell paths.
    wl_nm, n_med = _echoed_optics(rayleigh_results)

    # Normalize the ordinate (as in form_factor_analysis) so Rg/eps are scale-invariant and
    # the fit is stable at any scale; the factor multiplies only 1/Mw and A2.
    y_scale = _ordinate_scale(y)
    yn = y / y_scale

    # ---- linear Zimm/Debye seed for (1/Mw, Rg) ----
    if thermodynamic:
        A = np.column_stack([np.ones_like(q2), q2, c])
        seed = unc.multilinear_fit(A, yn, estimator, cond_limit).coeffs
        inv_mw0, b0, a2_0 = float(seed[0]), float(seed[1]), float(seed[2]) / 2.0
    else:
        lf = unc.linear_fit(q2, yn, estimator, cond_limit)
        inv_mw0, b0, a2_0 = float(lf.intercept), float(lf.slope), 0.0
    if math.isfinite(inv_mw0) and inv_mw0 > 0 and b0 / inv_mw0 > 0:
        rg0 = math.sqrt(3.0 * b0 / inv_mw0)
    else:
        rg0 = 10.0
    if not (inv_mw0 > 0 and math.isfinite(inv_mw0)):
        pos = yn[yn > 0]
        inv_mw0 = float(pos.min()) if pos.size else 1.0

    eps_idx = 2                                     # column of epsilon in theta / Jacobian

    def residuals(theta):
        P = ff.ellipsoid_P(q, theta[1], theta[2])          # theta = (inv_mw, rg, eps[, a2])
        a2 = theta[3] if thermodynamic else None
        return ff.model_ordinate(P, theta[0], a2, c) - yn

    if thermodynamic:
        lb = [0.0, 0.0, ELLIPSOID_EPS_MIN, -np.inf]
        ub = [np.inf, np.inf, ELLIPSOID_EPS_MAX, np.inf]
    else:
        lb = [0.0, 0.0, ELLIPSOID_EPS_MIN]
        ub = [np.inf, np.inf, ELLIPSOID_EPS_MAX]

    def _theta0(rg_start, eps_start):
        base = [inv_mw0, rg_start, eps_start]
        return base + [a2_0] if thermodynamic else base

    # MULTI-START across the axial-ratio range (oblate / sphere / prolate) x a small Rg
    # spread, so both the prolate and the oblate basin are found. The eps seeds straddle
    # eps = 1; the Rg seeds are the linear seed plus a couple q-scaled values.
    q_typ = float(np.median(q[q > 0])) if np.any(q > 0) else 1.0
    eps_seeds = (0.4, 0.7, 1.0, 1.6, 3.0)
    rg_seeds = [rg0] + [tgt / q_typ for tgt in (1.5, 3.0)]
    solutions = []                                  # (cost, res) for physical converged fits
    for rg_start in rg_seeds:
        if not (rg_start > 0 and math.isfinite(rg_start)):
            continue
        for eps_start in eps_seeds:
            # x_scale='jac' auto-scales the parameter spread; tolerances: _FF_LS_TOL.
            res = least_squares(residuals, _theta0(rg_start, eps_start), bounds=(lb, ub),
                                method='trf', x_scale='jac',
                                ftol=_FF_LS_TOL, xtol=_FF_LS_TOL, gtol=_FF_LS_TOL)
            inv_mw, rg, eps = float(res.x[0]), float(res.x[1]), float(res.x[2])
            if (res.success and inv_mw > 0 and rg > 0 and math.isfinite(res.cost)
                    and ELLIPSOID_EPS_MIN < eps < ELLIPSOID_EPS_MAX):
                solutions.append((float(res.cost), res))

    scaled = ('mw_g_per_mol', 'a2_mol_mL_per_g2')
    input_notes = tuple(drop_notes) + scale_notes(
        [r for r in rayleigh_results if r.concentration_g_per_mL != 0],
        mw=scaled[:1], a2=scaled[1:], scale_free=('rg_nm', 'axial_ratio'))
    all_index_known = all(bool(getattr(r, 'standard_index_known', True))
                          for r in rayleigh_results if r.concentration_g_per_mL != 0)
    thresholds = dict(axial_sens_tol=float(axial_sens_tol),
                      axial_max_rel_se=float(axial_max_rel_se),
                      form_factor_min_size_nm=float(form_factor_min_size_nm))
    if not solutions:
        # Emit the same RuntimeWarning the sphere/coil form_factor_analysis raises on a
        # non-converged fit, so a warning-based QC gate sees the ellipsoid no-fit too
        # (fix-the-family: the sentinel must not be a silent stderr gap for this sibling).
        _warn_note(_ELLIPSOID_NO_FIT_NOTE, stacklevel=2)
        return [_ellipsoid_no_fit(q, y, c, n_conc, n_ang, thermodynamic,
                                  all_calibrated, estimator, n_med, wl_nm,
                                  dn_dc_supplied=all_dn_dc_supplied, blank_measured=all_blank_measured,
            derived_provenance=_shared_derived_provenance(rayleigh_results),
            blank_provenance=_shared_blank_provenance(rayleigh_results), notes=input_notes,
            standard_index_known=all_index_known, thresholds=thresholds)]

    # ---- cluster distinct minima (lowest cost per basin), then keep the comparable set ----
    solutions.sort(key=lambda t: t[0])
    clusters = []                                   # each: (cost, res, rg, eps)
    for cost, res in solutions:
        rg, eps = float(res.x[1]), float(res.x[2])
        dup = False
        for _cost_c, _res_c, rg_c, eps_c in clusters:
            if (abs(math.log(eps) - math.log(eps_c)) < _ELLIPSOID_CLUSTER_TOL
                    and abs(rg - rg_c) <= _ELLIPSOID_CLUSTER_TOL * rg_c):
                dup = True
                break
        if not dup:
            clusters.append((cost, res, rg, eps))

    n_pts = int(q.size)
    ss_tot = float(np.sum((yn - yn.mean()) ** 2))
    data_rms = (math.sqrt(ss_tot / n_pts)
                if (math.isfinite(ss_tot) and ss_tot > 0) else 1.0)
    best_rms = math.sqrt(2.0 * clusters[0][0] / n_pts)  # least_squares cost = 0.5 sum res^2
    accepted = [cl for cl in clusters
                if math.sqrt(2.0 * cl[0] / n_pts) - best_rms <= degeneracy_tol * data_rms]
    degenerate = len(accepted) > 1
    # Presentation order (owner decision, 2026-07-23): ascending (axial ratio, Rg) -- a
    # DETERMINISTIC physical key, never fit preference. Accepted degenerate solutions are
    # statistically indistinguishable BY CONSTRUCTION (within degeneracy_tol of the best),
    # so ordering them by raw cost let ulp-level noise (BLAS build, solver tolerance)
    # decide which solution lists first -- irreproducible across machines/releases
    # (code-review 2026-07-23). ACCEPTANCE above stays cost-based (the robust 2%-RMS
    # band); only the listing order of the accepted set is conventional.
    accepted.sort(key=lambda cl: (cl[3], cl[2]))

    results = []
    for _cost, res, _rg, _eps in accepted:
        results.append(_build_ellipsoid_result(
            res, q, y, c, yn, y_scale, thermodynamic, n_conc, n_ang, n_pts,
            all_calibrated, estimator, cond_limit, eps_idx, degenerate, len(accepted),
            n_med, wl_nm, dn_dc_supplied=all_dn_dc_supplied, blank_measured=all_blank_measured,
            derived_provenance=_shared_derived_provenance(rayleigh_results),
            blank_provenance=_shared_blank_provenance(rayleigh_results), drop_notes=input_notes,
            standard_index_known=all_index_known, thresholds=thresholds))
    return results


def _build_ellipsoid_result(res, q, y, c, yn, y_scale, thermodynamic, n_conc, n_ang,
                            n_pts, calibrated, estimator, cond_limit, eps_idx,
                            degenerate, n_solutions, n_med, wl_nm, *, dn_dc_supplied, blank_measured=True, derived_provenance='',
        blank_provenance='', drop_notes=(), standard_index_known=True, thresholds=None):
    """Assemble one EllipsoidFormFactorResult (params, SEs, flags) from a converged fit.
    `thresholds` holds the axial-ratio and size-floor thresholds the verdicts read (the
    defaults where None), echoed on the result.

    Covariance when the axial ratio is UNRESOLVED. eps is profiled out by slicing the
    Jacobian to the core columns, so every dof guard inside the covariance sees p_core --
    but eps WAS a fitted parameter and the fit's honest residual dof is n - p_full. Two
    consequences, one per estimator:

      * BOTH withhold when the full fit has under 2 residual dof (`_hc3_cov`'s own rule).
        HC3's internal guard cannot apply it here because it was handed the sliced design,
        and this withhold once sat in the OLS branch alone on the premise that HC3 already
        withheld -- so the DEFAULT estimator emitted a Mw/Rg SE on a four-angle apparent
        fit exactly where the opt-in refused.
      * OLS divides RSS by (n - p_core), so its variance is rescaled to n - p_full. HC3 is
        residual-based with no such divisor and needs no rescale.

    The DOF half is what that closes. The LEVERAGE half stays open: HC3's h_ii still come
    from the core design while the residuals are the full fit's, so its sandwich is
    under-inflated in this branch. Measured conservative (1.16x/1.18x of the sampling SD)
    in the one regime tried, so a defensibility gap rather than a measured under-report --
    recorded as a carried residual against this method rather than fixed here.
    """
    thresholds = {'axial_sens_tol': _ELLIPSOID_AXIAL_SENS_TOL,
                  'axial_max_rel_se': _ELLIPSOID_AXIAL_MAX_REL_SE,
                  'form_factor_min_size_nm': FORM_FACTOR_MIN_SIZE_NM, **(thresholds or {})}
    inv_mw = float(res.x[0]) * y_scale
    rg = float(res.x[1])
    eps = float(res.x[2])
    a2 = float(res.x[3]) * y_scale if thermodynamic else None
    mw = 1.0 / inv_mw if (math.isfinite(inv_mw) and inv_mw > 0) else float('nan')
    r_eq, r_pol = ff.ellipsoid_semi_axes_from_rg(rg, eps)
    is_prolate = eps > 1.0
    converged = bool(res.success and math.isfinite(inv_mw) and inv_mw > 0
                     and math.isfinite(rg) and rg > 0 and math.isfinite(mw))

    # ---- axial-ratio identifiability: is the eps column of the Jacobian meaningful? ----
    # Compare the FRACTIONAL sensitivities d(resid)/d(ln eps) = eps*||J_eps|| and
    # d(resid)/d(ln rg) = rg*||J_rg||. Both are dimensionless (the normalized residual and
    # ln are unitless), so their ratio depends on the qRg REGIME, not on the absolute
    # particle size. A raw ||J_eps||/||J_rg|| would carry units of nm (eps is dimensionless,
    # rg is in nm), keying the resolved/unresolved verdict on Rg rather than qRg -- a
    # scale-dependence caught in code review (2026-07-20) and forbidden by invariant 4.
    J = np.asarray(res.jac, dtype=float)
    col = np.linalg.norm(J, axis=0)
    frac_eps = eps * col[eps_idx]                   # d(resid)/d(ln eps)
    frac_rg = rg * col[1]                            # d(resid)/d(ln rg)
    eps_sens = frac_eps / (frac_rg + 1e-300)         # dimensionless, Rg-independent

    # ---- covariance for the SEs ----
    # Full covariance (all params) when eps is resolved; otherwise a CORE covariance over
    # (inv_Mw, Rg[, A2]) with the eps column removed, so Mw/Rg reliability is independent of
    # axial-ratio identifiability. Physical param = D @ theta_n, D = diag([y_scale, 1, 1,
    # y_scale]) (Rg, eps untouched); covariance transforms as D cov_n D.
    core_cols = [0, 1] + ([3] if thermodynamic else [])
    cov_full, cond_full, wc_full = unc.covariance_from_jacobian(
        J, res.fun, estimator, cond_limit)
    eps_resolved = bool(eps_sens >= thresholds['axial_sens_tol'] and wc_full
                        and np.all(np.isfinite(cov_full)))

    mw_se = rg_se = axial_ratio_se = a2_se = None
    if eps_resolved:
        scale_vec = np.array([y_scale, 1.0, 1.0, y_scale][:cov_full.shape[0]])
        cov = cov_full * np.outer(scale_vec, scale_vec)
        cond, well_cond = cond_full, wc_full
        if converged:
            p = cov.shape[0]
            j = np.zeros(p)
            j[0] = -1.0 / unc.power_or_nan(inv_mw, 2)       # Mw = 1/inv_mw (reciprocal)
            mw_se = unc.se_or_none(unc.propagate(j, cov))
            j = np.zeros(p); j[1] = 1.0                      # Rg direct
            rg_se = unc.se_or_none(unc.propagate(j, cov))
            j = np.zeros(p); j[eps_idx] = 1.0                # axial ratio direct
            axial_ratio_se = unc.se_or_none(unc.propagate(j, cov))
            if thermodynamic:
                j = np.zeros(p); j[3] = 1.0
                a2_se = unc.se_or_none(unc.propagate(j, cov))
        # A resolved-but-imprecise axial ratio (large relative SE) is still flagged, its ±
        # shown; its own precondition below names the relative SE against the limit.
        rel_se = (axial_ratio_se / eps
                  if (axial_ratio_se is not None and math.isfinite(eps) and eps > 0)
                  else float('nan'))
        axial_ratio_reliable = bool(rel_se <= thresholds['axial_max_rel_se'])
    else:
        # eps unresolved -> profile it out: SEs for (inv_Mw, Rg[, A2]) only, eps SE withheld.
        cov_c, cond, well_cond = unc.covariance_from_jacobian(
            J[:, core_cols], res.fun, estimator, cond_limit)
        # eps consumed a degree of freedom the SLICED design cannot see, so the dof both
        # estimators must judge is n - p_full (see this function's docstring: the withhold
        # applies to both, the rescale to OLS alone, and the leverage half stays open).
        p_full, p_core = J.shape[1], len(core_cols)
        dof_full = n_pts - p_full
        if dof_full < 2:
            cov_c = np.full_like(cov_c, float('nan'))
        elif str(estimator).lower() == 'ols':
            cov_c = cov_c * ((n_pts - p_core) / dof_full)
        scale_vec = np.array([y_scale, 1.0, y_scale][:cov_c.shape[0]])
        cov = cov_c * np.outer(scale_vec, scale_vec)
        if converged and np.all(np.isfinite(cov)):
            p = cov.shape[0]
            j = np.zeros(p); j[0] = -1.0 / unc.power_or_nan(inv_mw, 2)
            mw_se = unc.se_or_none(unc.propagate(j, cov))
            j = np.zeros(p); j[1] = 1.0
            rg_se = unc.se_or_none(unc.propagate(j, cov))
            if thermodynamic:
                j = np.zeros(p); j[2] = 1.0
                a2_se = unc.se_or_none(unc.propagate(j, cov))
        rel_se = float('nan')
        axial_ratio_reliable = False

    # ---- flags + regime indicator ----
    ss_res = float(np.sum(res.fun ** 2))
    ss_tot = float(np.sum((yn - yn.mean()) ** 2))
    r2 = unc.r_squared_or_nan(ss_res, ss_tot)
    qrg_max = float(np.max(q) * rg) if (math.isfinite(rg) and rg > 0) else float('nan')
    # Conservative validity proxy: past the sphere first-minimum qRg the single-particle fit
    # is ill-conditioned. The orientation average smears the ellipsoid minimum, so flagging
    # at the sphere threshold is conservative (early). Suppresses Mw/A2 reliability past it.
    # The size floor is the same predicate's lower tail -- see FORM_FACTOR_MIN_SIZE_NM.
    above_floor = _size_above_floor(rg, thresholds['form_factor_min_size_nm'])
    form_factor_valid = above_floor and (not math.isfinite(qrg_max)
                                         or qrg_max <= SPHERE_FIRST_MINIMUM_QRG)

    # The axial ratio fails its own verdict, never Mw's, for one of two causes: it is not
    # identified at all (the ± withheld), or it is identified and imprecise (the ± shown).
    # The degeneracy is a qualifier read off `degenerate` / `n_solutions` / `is_prolate`,
    # never a note. (Only res.success solutions are ever assembled here -- non-converged
    # fits are filtered before clustering -- so a solution that is not `converged` reached
    # a non-physical parameter, named below.)
    scaled = ('mw_g_per_mol', 'a2_mol_mL_per_g2')
    notes: list = list(drop_notes)
    if not eps_resolved:
        notes.append(Note('axial_ratio_unresolved', ('axial_ratio',), {'qrg': float(qrg_max)}))
    elif not axial_ratio_reliable:
        notes.append(Note('axial_ratio_imprecise', ('axial_ratio',),
                          {'rel_se': float(rel_se),
                           'limit': float(thresholds['axial_max_rel_se'])}))
    if not converged:
        notes += _nonphysical_notes(inv_mw, mw, (('Rg', rg),), scaled + ('rg_nm',))
    if thermodynamic and a2 is not None and not math.isfinite(a2):
        notes.append(Note('a2_not_finite', scaled[1:]))

    # The design's conditioning beside the shape-fit reasons, never instead; between
    # those, the floor outranks the first minimum -- see the sphere/Mie chain for why.
    if not well_cond:
        notes.append(_ff_ill_conditioned_note(cond, cond_limit, scaled))
    if not above_floor:
        notes.append(_sub_molecular_note('Rg', rg, scaled + ('rg_nm',),
                                         thresholds['form_factor_min_size_nm']))
    elif not form_factor_valid:
        notes.append(Note('past_first_minimum', scaled,
                          {'qrg': float(qrg_max), 'limit': float(SPHERE_FIRST_MINIMUM_QRG)}))

    return EllipsoidFormFactorResult(
        model=ff.ELLIPSOID, is_apparent=not thermodynamic, converged=converged,
        degenerate=degenerate, is_prolate=is_prolate,
        mw_g_per_mol=float(mw), rg_nm=float(rg), axial_ratio=float(eps),
        semi_axis_equatorial_nm=float(r_eq), semi_axis_polar_nm=float(r_pol),
        a2_mol_mL_per_g2=a2,
        q_nm_inv=q, kc_over_dR=y, concentrations_g_per_mL=c,
        n_points=n_pts, n_concentrations=n_conc, n_angles=n_ang,
        qrg_max=qrg_max, r_squared=float(r2), calibrated=calibrated,
        dn_dc_supplied=dn_dc_supplied,
        blank_measured=blank_measured,
        standard_index_known=standard_index_known,
        derived_provenance=derived_provenance,
        blank_provenance=blank_provenance,
        # Rg/eps survive an uncalibrated OR dn/dc-absent run alike (the ordinate scale
        # cancels); Mw needs BOTH scales known, a physical fit,
        # a well-conditioned CORE (inv_Mw, Rg[, A2]) design, and a fit within the sphere
        # first-minimum proxy. Deliberately NOT gated on axial-ratio identifiability.
        # `blank_measured` joins them, and unlike the two scales it also gates the SIZE:
        # an unmeasured blank is ADDITIVE, so it does not cancel in the ordinate
        # normalization the sizes rely on (angle-dependence is an extra, not the reason).
        mw_reliable=(calibrated and dn_dc_supplied and blank_measured
                     and converged and well_cond and form_factor_valid),
        a2_reliable=(thermodynamic and calibrated and dn_dc_supplied and blank_measured
                     and converged and well_cond
                     and form_factor_valid and a2 is not None and math.isfinite(a2)),
        # The trust verdict takes the blank like every reliability verdict here (the rule's
        # home is RayleighRatioResult's docstring); identifiability, which the +- withhold
        # and every "unresolved" word read, is `axial_ratio_resolved`: the blank leaves it.
        axial_ratio_reliable=(axial_ratio_reliable and blank_measured),
        axial_ratio_resolved=eps_resolved,
        form_factor_valid=form_factor_valid,
        rg_reliable=(converged and above_floor and blank_measured),  # finding 1 -- see the field's comment
        mw_se=mw_se, rg_se=rg_se, axial_ratio_se=axial_ratio_se, a2_se=a2_se,
        se_estimator=estimator, well_conditioned=well_cond, cond=cond,
        n_solutions=n_solutions, notes=tuple(notes),
        n_medium=float(n_med), wavelength_nm=float(wl_nm),
        axial_sens_tol=float(thresholds['axial_sens_tol']),
        axial_max_rel_se=float(thresholds['axial_max_rel_se']),
        form_factor_min_size_nm=float(thresholds['form_factor_min_size_nm']),
    )


# --------------------------------------------------------------------------------------
# Core-shell (coated-sphere) form-factor fit. Like the ellipsoid, this returns a LIST of
# solutions: a coated sphere has TWO geometric radii (core + outer) and two FIXED
# user-supplied contrasts, and two radii from ONE angular P(q) are frequently
# under-determined -- several (R_core, R_outer) pairs reproduce the same curve. Every
# comparably-fitting solution is reported, `degenerate` is set when more than one fits, none
# is silently preferred, and a no-fit is a single `converged=False` sentinel (never []). The
# optics come from analysis/mie.py via analysis.form_factor.core_shell_P.

# Deep-form-factor-minimum reliability floor for the FITTED coated P(q) over the angles, the
# same physical criterion (and value) as the homogeneous-Mie MIE_FIT_P_MIN: at an angle where
# P(q) dips below this, the ordinate 1/(Mw P) is a noise-dominated small difference of large
# intensities and the Mw/A2 fit is ill-conditioned there (Rg is still reported -- the shape is
# constrained by the whole curve). A documented default (invariant 4) of
# `core_shell_form_factor_analysis`'s `core_shell_p_min` -- the same shape as MIE_FIT_P_MIN.
CORE_SHELL_FIT_P_MIN = 0.05
# Two solutions are the SAME basin (dedup) when both radii agree to this relative tolerance.
_CORE_SHELL_CLUSTER_TOL = 0.05
# A solution joins the reported (degenerate) set when its RMS residual exceeds the best
# solution's by less than this fraction of the DATA RMS (its extra misfit is a negligible
# part of the signal). A documented general criterion (invariant 4), not tuned.
_CORE_SHELL_DEGENERACY_TOL = 0.02
# Multi-start seed grid (Phase-3 lesson: the coated P(q) oscillates fast, so a dense profile
# scan -- not a few q-scaled guesses -- is needed to find the global basin). Outer radius over
# qR_outer in [0.3, 50] (covers sub-nm to ~1.6 um at 532 nm / water / 150 deg), core as a
# fraction of the outer radius. n_grid over the log-outer range keeps a few points per coated
# oscillation across it; a particle larger still sits deep past the form-factor minima where
# the deep-minimum reliability gate flags Mw/A2 anyway (as for the Mie sphere).
_CORE_SHELL_QROUTER_LO = 0.3
_CORE_SHELL_QROUTER_HI = 50.0
_CORE_SHELL_N_OUTER = 90
_CORE_SHELL_CORE_FRACTIONS = (0.15, 0.3, 0.45, 0.6, 0.75, 0.9)
# Positive floor on the fitted core radius (nm): keeps the optimizer off the r_core = 0
# discontinuity in core_shell_P (its x > 0 guard), far below any physical core.
_CORE_SHELL_RMIN_NM = 1.0e-6


@dataclass
class CoreShellFormFactorResult:
    """One coated-sphere (core-shell) P(q) fit solution (Aden-Kerker / Bohren & Huffman 1983
    Eq. 8.2).

    `core_shell_form_factor_analysis` returns a LIST of these -- one per (R_core, R_outer)
    solution that fits the angular Kc/dR comparably; `degenerate` is True on every member when
    more than one fits. The two contrasts `m1` (core) and `m2` (shell) are FIXED user inputs,
    never fitted (invariant 3). The reported size `rg_nm` is the **contrast-weighted** radius of
    gyration (the scattering Rg the coated P(q) shows in its Guinier limit; comparable to the
    Zimm/sphere Rg), which is `nan` for a contrast-matched object. The fit is APPARENT at one
    concentration (`is_apparent=True`) and THERMODYNAMIC across concentrations. Rg and the radii
    come from the angular shape and survive an uncalibrated run; Mw / A2 need calibration.

    Uncertainty (invariant 8): the fitted-radius SEs `r_core_se`/`r_outer_se` are direct/linear
    HC3 pass-throughs of the fit covariance, and `a2_se` is likewise a linear pass-through -- all
    three inherit the covariance's validation. `mw_se` is the NONLINEAR reciprocal-Mw SE and is
    the one Monte-Carlo-validated against the sampling SD in tests/test_sls.py (both the
    thermodynamic and apparent regimes). There is deliberately **no Rg SE** (`rg_se` is always
    None): the contrast-weighted Rg is a nonlinear combination of two radii that are themselves
    often structurally degenerate, so its uncertainty is reported STRUCTURALLY by the solution
    list, not as a symmetric +/-.
    """
    model: str                        # always 'core_shell' (uniform with FormFactorResult)
    is_apparent: bool
    converged: bool
    degenerate: bool                  # >1 (R_core, R_outer) solution fits comparably
    rg_undefined: bool                # the contrast-weighted Rg is not physically defined:
                                      # near-zero net contrast (matched) OR an opposite-sign
                                      # core/shell radial moment (Rg^2 <= 0, hollow-like)
    # fitted quantities
    mw_g_per_mol: float               # apparent at single c; thermodynamic across c
    rg_nm: float                      # contrast-weighted radius of gyration (nan if matched)
    r_core_nm: float                  # fitted core radius
    r_outer_nm: float                 # fitted outer radius (= core + shell thickness)
    shell_thickness_nm: float         # r_outer - r_core (derived)
    a2_mol_mL_per_g2: Optional[float] # thermodynamic only (None when apparent)
    # fit context
    q_nm_inv: np.ndarray
    kc_over_dR: np.ndarray
    concentrations_g_per_mL: np.ndarray
    n_points: int
    n_concentrations: int
    n_angles: int
    qrg_max: float                    # max q*Rg over the fitted points (regime indicator)
    r_squared: float
    calibrated: bool
    mw_reliable: bool                 # False if uncalibrated / non-physical / ill-conditioned
    a2_reliable: bool = False
    # False for THREE causes, and a reader must distinguish them: the contrast-weighted Rg
    # is undefined (contrast-matched/hollow -- see `rg_undefined`), the fit did not converge,
    # or the fitted OUTER RADIUS is below FORM_FACTOR_MIN_SIZE_NM. Only the first implies a
    # nan Rg, so a display that maps False onto "contrast-matched" alone mislabels a
    # collapsed fit (round 943d f2). Rg survives an uncalibrated run, so calibration does
    # NOT gate it.
    rg_reliable: bool = True
    # False when the fitted coated P(q) spans a deep form-factor minimum (min P < the floor) --
    # Mw/A2 ill-conditioned there, Rg still reported -- or when the fitted OUTER RADIUS falls
    # below FORM_FACTOR_MIN_SIZE_NM, the same predicate's lower tail, where P(q) has collapsed
    # to 1. The floor binds r_outer rather than rg_nm here because this result's Rg is a
    # contrast-weighted moment that is undefined by design when the contrasts cancel.
    # Documented physics criteria (invariant 4).
    form_factor_valid: bool = True
    # statistical (regression) SEs -- exclude calibration/dn-dc systematics; None when too few
    # points or a degenerate/ill-conditioned Jacobian. r_core/r_outer SEs are direct/linear
    # pass-throughs; mw/a2 SEs are MC-validated (invariant 8). rg_se is ALWAYS None (see above).
    mw_se: Optional[float] = None
    r_core_se: Optional[float] = None
    r_outer_se: Optional[float] = None
    a2_se: Optional[float] = None
    rg_se: Optional[float] = None     # always None -- structural degeneracy, not a +/-
    se_estimator: str = 'hc3'
    well_conditioned: bool = True
    cond: float = float('nan')
    n_solutions: int = 1              # size of the degenerate set this solution belongs to
    # Every note of the solution, as Notes; its degeneracy is a qualifier read off
    # `degenerate` / `n_solutions` (core.notes.qualifiers), never a note.
    notes: Tuple[Note, ...] = ()
    # `calibrated`'s sibling for the OPTICAL scale (see RayleighRatioResult): False ->
    # no dn/dc was supplied, so K was 1 and Mw/A2 here are on an arbitrary optical
    # scale. The angular sizes are untouched -- K multiplies the whole Kc/dR ordinate
    # and cancels in every coefficient ratio.
    dn_dc_supplied: bool = True
    # Carried forward from the Rayleigh result(s) this fit consumed, so a number
    # cannot be read without its origin whichever exporter writes it. A record; it
    # enters no computation. Empty for an ordinary loaded measurement.
    derived_provenance: str = ''
    # Which solvent blank the contributing points were referenced against, when
    # it is not the sample reference; copied from the Rayleigh result(s) exactly as
    # derived_provenance is, and exported beside it. '' for the common case.
    blank_provenance: str = ''
    # Carried from the Rayleigh result(s) this fit consumed: False -> the blank was not
    # measured at every angle (typed 90 deg value, or no subtraction). ANDed into every
    # reliability verdict on this result, because an additive blank does not cancel the
    # way an unknown SCALE does. See RayleighRatioResult.blank_measured.
    blank_measured: bool = True
    # Carried from the Rayleigh result(s) this fit consumed: False -> k_c was applied with
    # the standard's refractive index blank, so `calibrated` is False for that cause alone
    # (the scale gates name it). See RayleighRatioResult.standard_index_known.
    standard_index_known: bool = True
    # Contrast provenance (fixed user inputs), echoed for display + export like the Mie fields.
    m1: Optional[float] = None        # core relative index n_core/n_medium
    m2: Optional[float] = None        # shell relative index n_shell/n_medium
    n_core: Optional[float] = None
    n_shell: Optional[float] = None
    n_medium: Optional[float] = None
    wavelength_nm: Optional[float] = None
    # The thresholds the verdicts above read, echoed as `qrg_max_valid` is: the deep-minimum
    # floor on the fitted P(q), the size floor and the contrast-match tolerance of the Rg.
    core_shell_p_min: float = CORE_SHELL_FIT_P_MIN
    form_factor_min_size_nm: float = FORM_FACTOR_MIN_SIZE_NM
    core_shell_contrast_match_rel: float = ff._CORE_SHELL_CONTRAST_MATCH_REL


#: The no-fit precondition of the core-shell fit: on its sentinel, and its warning's message.
_CORE_SHELL_NO_FIT_NOTE = Note('no_ff_solution', (), {'model': 'core-shell sphere'})


def _core_shell_no_fit(q, y, c, n_conc, n_ang, thermodynamic, calibrated, estimator,
                       m1, m2, n_core, n_shell, n_med, wl_nm, *, dn_dc_supplied, blank_measured=True, derived_provenance='',
        blank_provenance='', notes=(), standard_index_known=True, thresholds=None):
    """The single converged=False sentinel returned when no coated sphere fits (never []).
    `notes` is the drop and scale notes of the input; the no-fit note joins them."""
    return CoreShellFormFactorResult(
        model=ff.CORE_SHELL, is_apparent=not thermodynamic, converged=False,
        degenerate=False, rg_undefined=False,
        mw_g_per_mol=float('nan'), rg_nm=float('nan'), r_core_nm=float('nan'),
        r_outer_nm=float('nan'), shell_thickness_nm=float('nan'), a2_mol_mL_per_g2=None,
        q_nm_inv=q, kc_over_dR=y, concentrations_g_per_mL=c,
        n_points=int(q.size), n_concentrations=n_conc, n_angles=n_ang,
        qrg_max=float('nan'), r_squared=float('nan'), calibrated=calibrated,
        dn_dc_supplied=dn_dc_supplied,
        blank_measured=blank_measured,
        derived_provenance=derived_provenance,
        blank_provenance=blank_provenance,
        # form_factor_valid stays True: nothing was fitted, so there is no size for the
        # FORM_FACTOR_MIN_SIZE_NM floor to judge, and every *_reliable flag is already False.
        mw_reliable=False, a2_reliable=False, rg_reliable=False, form_factor_valid=True,
        standard_index_known=standard_index_known,
        se_estimator=estimator, well_conditioned=False,
        n_solutions=0, notes=tuple(notes) + (_CORE_SHELL_NO_FIT_NOTE,),
        m1=m1, m2=m2, n_core=n_core, n_shell=n_shell, n_medium=n_med, wavelength_nm=wl_nm,
        **(thresholds or {}))


def _profile_core_shell_starts(P_of, q, yn, c, thermodynamic, n_keep=6, *,
                               P_of_batch=None):
    """Robust (R_core, R_outer) seeds for the coated P(q), by profiling the residual over a
    dense 2-D grid of (outer radius, core fraction).

    At FIXED (R_core, R_outer) the ordinate ``inv_Mw/P [+ 2 A2 c]`` is LINEAR in (inv_Mw[, A2]),
    so the best-fit residual at each grid node is one exact linear solve -- cheap. Scanning the
    outer radius geometrically over qR_outer in [lo, hi] and the core as a fraction of it, then
    returning the ``n_keep`` lowest-residual nodes, gives least_squares starts that reach the
    global basin even though the coated P(q) oscillates fast (the Phase-3 wrong-basin lesson).

    Returns ``(r_core, r_outer, inv_mw0)`` per kept node: the node's own best-fit ``inv_Mw``
    (the linear solve already paid for it) warm-starts least_squares far better than a single
    global min-ordinate guess when a large 2*A2*c offset shifts the true inv_Mw off it.

    P is FLOORED (not the whole node dropped) where the coated form factor grazes a null at a
    measured angle: a single near-zero angle would otherwise discard the entire (R_core, R_outer)
    seed, so a large/high-contrast particle whose P(q) genuinely crosses a minimum inside the
    q-range could lose its true-basin seeds and be mis-reported as a no-fit. Flooring keeps the
    node with a large-but-finite residual there, so it is correctly deprioritized if the data
    does NOT show that null and retained if it does (the deep-minimum reliability gate then
    handles the Mw/A2 conditioning downstream).
    """
    q = np.asarray(q, dtype=float)
    q_max = float(np.max(q)) if q.size else 0.0
    if not (math.isfinite(q_max) and q_max > 0):
        return []
    r_outer_grid = np.geomspace(_CORE_SHELL_QROUTER_LO / q_max,
                                _CORE_SHELL_QROUTER_HI / q_max, _CORE_SHELL_N_OUTER)
    a2_col = 2.0 * np.asarray(c, dtype=float)
    # Optional batched evaluator (perf spec 2026-07-22): the whole 2-D grid's P matrix in
    # one lockstep BHCOAT sweep instead of N_OUTER x len(FRACTIONS) scalar recurrences.
    # The node arrays are built in the SAME enumeration order as the double loop below
    # (outer over r_outer, inner over the core fractions, identical float products), so
    # the stable sort + n_keep truncation select identical seeds -- asserted in tests.
    P_rows = None
    if P_of_batch is not None:
        rc_nodes = np.array([frac * r_outer for r_outer in r_outer_grid
                             for frac in _CORE_SHELL_CORE_FRACTIONS])
        ro_nodes = np.repeat(r_outer_grid, len(_CORE_SHELL_CORE_FRACTIONS))
        P_rows = np.asarray(P_of_batch(rc_nodes, ro_nodes), dtype=float)
    node_idx = -1
    scored = []                                     # (cost, r_core, r_outer, inv_mw0)
    for r_outer in r_outer_grid:
        for frac in _CORE_SHELL_CORE_FRACTIONS:
            node_idx += 1
            r_core = frac * r_outer
            P = (P_rows[node_idx] if P_rows is not None
                 else np.asarray(P_of(r_core, r_outer), dtype=float))
            inv_p = 1.0 / np.maximum(P, 1e-12)      # floor P (grazing a null) instead of dropping
            if not np.all(np.isfinite(inv_p)):
                continue                            # non-finite P (should not occur) -> skip node
            X = np.column_stack([inv_p, a2_col]) if thermodynamic else inv_p[:, None]
            beta, *_ = np.linalg.lstsq(X, yn, rcond=None)
            if not (math.isfinite(beta[0]) and beta[0] > 0):   # non-physical/non-finite ampl.
                continue
            resid = X @ beta - yn
            scored.append((float(resid @ resid), float(r_core), float(r_outer), float(beta[0])))
    scored.sort(key=lambda t: t[0])
    return [(rc, ro, inv_mw0) for _cost, rc, ro, inv_mw0 in scored[:n_keep]]


def core_shell_form_factor_analysis(
    rayleigh_results,
    m1: float,
    m2: float,
    estimator: str = 'hc3',
    cond_limit: float = unc.COND_LIMIT,
    degeneracy_tol: float = _CORE_SHELL_DEGENERACY_TOL,
    *,
    core_shell_p_min: float = CORE_SHELL_FIT_P_MIN,
    form_factor_min_size_nm: float = FORM_FACTOR_MIN_SIZE_NM,
    core_shell_contrast_match_rel: float = ff._CORE_SHELL_CONTRAST_MATCH_REL,
) -> List[CoreShellFormFactorResult]:
    """Fit the coated-sphere (core-shell) form factor P(q) to Kc/dR across angles (and c).

    Returns a LIST of CoreShellFormFactorResult -- one per (R_core, R_outer) solution that fits
    the data comparably (two radii from one angular curve are often under-determined); every
    member has `degenerate=True` when more than one fits. The list is ordered by ASCENDING
    (R_outer, R_core) -- a deterministic presentation convention, not fit preference:
    accepted solutions are statistically indistinguishable, so cost cannot honestly rank
    them. A no-fit yields a single `converged=False` sentinel, never an empty list. Mirrors
    the ellipsoid inverse (ellipsoid_form_factor_analysis).

    The ordinate is Kc/dR = 1/(Mw P(q; R_core, R_outer)) + 2 A2 c, with P the coated form factor
    (analysis/form_factor.core_shell_P; Bohren & Huffman 1983 Eq. 8.2). The relative refractive
    indices `m1` (core) and `m2` (shell), formed from the hand-entered core/shell refractive
    indices and the solvent index, are FIXED inputs -- never fitted (invariant 3). The fit is a
    2-D multi-start nonlinear least squares over (R_core, shell thickness t >= 0), seeded from a
    dense profile scan; distinct low-cost minima are clustered and every solution whose extra
    misfit is below `degeneracy_tol` of the data RMS is reported.

    Parameters
    ----------
    rayleigh_results : RayleighRatioResult or sequence of them
        One per concentration (a c = 0 solvent reference is ignored).
    m1, m2 : float
        Core and shell relative refractive indices (n_core/n_medium, n_shell/n_medium),
        positive and finite. Fixed user inputs.
    estimator : str
        Covariance estimator for the SEs, 'hc3' (default) or 'ols' (invariant 8 clause A).
    cond_limit : float
        Numerical-health ceiling on the fit Jacobian's condition number.
    degeneracy_tol : float
        A solution joins the reported set when its RMS residual exceeds the best solution's by
        less than this fraction of the data RMS (default 2 %). Documented, overridable.
    core_shell_p_min, form_factor_min_size_nm, core_shell_contrast_match_rel : float
        The deep-minimum floor on the fitted P(q), the size floor on the outer radius and
        the net-contrast tolerance below which the contrast-weighted Rg is undefined;
        defaults ``CORE_SHELL_FIT_P_MIN``, ``FORM_FACTOR_MIN_SIZE_NM`` and
        ``form_factor._CORE_SHELL_CONTRAST_MATCH_REL``, echoed on every result.

    Raises
    ------
    ValueError
        For a non-positive/non-finite m1 or m2, or fewer than three distinct angles (two radii
        plus 1/Mw need angular curvature -- at least three angular points to constrain them).
    """
    for name, mm in (("m1", m1), ("m2", m2)):
        if not (mm is not None and math.isfinite(mm) and mm > 0):
            raise ValueError(
                f"the core-shell fit needs a positive finite {name} "
                f"(= n_{'core' if name == 'm1' else 'shell'}/n_medium, from the hand-entered "
                f"refractive index); got {mm!r}."
            )
    if isinstance(rayleigh_results, RayleighRatioResult):
        rayleigh_results = [rayleigh_results]

    q2, c, y, q, n_conc, n_ang, thermodynamic, drop_notes = _form_factor_inputs(
        rayleigh_results)
    # Two radii + 1/Mw is a three-parameter shape fit; it needs angular CURVATURE, so at least
    # three distinct angles (two only fix a slope -> one size). Stricter than the >= 2 the
    # Rg-only sphere/coil fit accepts, matching the ellipsoid's three-angle floor.
    if n_ang < 3:
        raise ValueError(
            "Core-shell form-factor P(q) fit needs at least three angles to resolve the two "
            f"radii; got {n_ang}."
        )
    all_calibrated, all_dn_dc_supplied, all_blank_measured = _aggregate_scale_flags(
        rayleigh_results)
    wl_nm, n_med = _echoed_optics(rayleigh_results)
    n_core = m1 * n_med
    n_shell = m2 * n_med

    # Normalize the ordinate so the radii are scale-invariant and the fit is stable at any
    # scale; the factor multiplies only 1/Mw and A2 (as in form_factor_analysis).
    y_scale = _ordinate_scale(y)
    yn = y / y_scale

    def _P(r_core, r_outer):
        return ff.core_shell_P(q, r_core, r_outer, m1, m2,
                               wavelength_nm=wl_nm, n_medium=n_med)

    # ---- linear Zimm seed for A2 (inv_Mw is profiled out per grid node) ----
    if thermodynamic:
        A = np.column_stack([np.ones_like(q2), q2, c])
        seed = unc.multilinear_fit(A, yn, estimator, cond_limit).coeffs
        a2_0 = float(seed[2]) / 2.0
    else:
        a2_0 = 0.0

    def residuals(theta):
        P = _P(theta[1], theta[1] + theta[2])          # theta = (inv_mw, r_core, t[, a2])
        a2 = theta[3] if thermodynamic else None
        return ff.model_ordinate(P, theta[0], a2, c) - yn

    # R_core lower bound is a tiny POSITIVE floor, not 0: at exactly r_core = 0 the guard in
    # core_shell_P returns the flat point limit P = 1 (to keep the coated solver's x > 0), a
    # discontinuity from the correct homogeneous-outer-sphere curve at r_core -> 0+. The floor
    # (far below any physical core) keeps the trust-region optimizer inside the smooth domain.
    if thermodynamic:
        lb = [0.0, _CORE_SHELL_RMIN_NM, 0.0, -np.inf]
        ub = [np.inf, np.inf, np.inf, np.inf]
    else:
        lb = [0.0, _CORE_SHELL_RMIN_NM, 0.0]
        ub = [np.inf, np.inf, np.inf]

    def _theta0(r_core, r_outer, inv_mw0):
        base = [inv_mw0, r_core, max(r_outer - r_core, 0.0)]
        return base + [a2_0] if thermodynamic else base

    solutions = []                                  # (cost, res) for physical converged fits
    def _P_batch(rc_arr, ro_arr):
        return ff.core_shell_P_batch(q, rc_arr, ro_arr, m1, m2,
                                     wavelength_nm=wl_nm, n_medium=n_med)

    for r_core, r_outer, inv_mw0 in _profile_core_shell_starts(_P, q, yn, c, thermodynamic,
                                                               P_of_batch=_P_batch):
        if not (math.isfinite(r_core) and r_core > 0 and r_outer > r_core
                and math.isfinite(inv_mw0) and inv_mw0 > 0):
            continue
        # Warm-start inv_Mw from the node's own linear solve (not a global min-ordinate guess).
        # x_scale='jac' auto-scales the parameter spread; tolerances: _FF_LS_TOL.
        res = least_squares(residuals, _theta0(r_core, r_outer, inv_mw0),
                            bounds=(lb, ub), method='trf', x_scale='jac',
                            ftol=_FF_LS_TOL, xtol=_FF_LS_TOL, gtol=_FF_LS_TOL)
        inv_mw, rc, t = float(res.x[0]), float(res.x[1]), float(res.x[2])
        if (res.success and math.isfinite(inv_mw) and inv_mw > 0
                and math.isfinite(rc) and rc > 0 and t >= 0 and math.isfinite(res.cost)):
            solutions.append((float(res.cost), res))

    scaled = ('mw_g_per_mol', 'a2_mol_mL_per_g2')
    # The blank is about every size the angular shape gives -- the radii and the shell
    # they bound beside the Rg -- as it is about the ellipsoid's axial ratio.
    input_notes = tuple(drop_notes) + scale_notes(
        [r for r in rayleigh_results if r.concentration_g_per_mL != 0],
        mw=scaled[:1], a2=scaled[1:],
        scale_free=('rg_nm', 'r_core_nm', 'r_outer_nm', 'shell_thickness_nm'))
    all_index_known = all(bool(getattr(r, 'standard_index_known', True))
                          for r in rayleigh_results if r.concentration_g_per_mL != 0)
    thresholds = dict(core_shell_p_min=float(core_shell_p_min),
                      form_factor_min_size_nm=float(form_factor_min_size_nm),
                      core_shell_contrast_match_rel=float(core_shell_contrast_match_rel))
    if not solutions:
        _warn_note(_CORE_SHELL_NO_FIT_NOTE, stacklevel=2)
        return [_core_shell_no_fit(q, y, c, n_conc, n_ang, thermodynamic, all_calibrated,
                                   estimator, m1, m2, n_core, n_shell, n_med, wl_nm,
                                   dn_dc_supplied=all_dn_dc_supplied, blank_measured=all_blank_measured,
            derived_provenance=_shared_derived_provenance(rayleigh_results),
            blank_provenance=_shared_blank_provenance(rayleigh_results), notes=input_notes,
            standard_index_known=all_index_known, thresholds=thresholds)]

    # ---- cluster distinct minima (lowest cost per basin), keep the comparable set ----
    solutions.sort(key=lambda t: t[0])
    clusters = []                                   # (cost, res, r_core, r_outer)
    for cost, res in solutions:
        rc, ro = float(res.x[1]), float(res.x[1]) + float(res.x[2])
        dup = False
        for _c, _r, rc_c, ro_c in clusters:
            if (abs(rc - rc_c) <= _CORE_SHELL_CLUSTER_TOL * rc_c
                    and abs(ro - ro_c) <= _CORE_SHELL_CLUSTER_TOL * ro_c):
                dup = True
                break
        if not dup:
            clusters.append((cost, res, rc, ro))

    n_pts = int(q.size)
    ss_tot = float(np.sum((yn - yn.mean()) ** 2))
    data_rms = (math.sqrt(ss_tot / n_pts)
                if (math.isfinite(ss_tot) and ss_tot > 0) else 1.0)
    best_rms = math.sqrt(2.0 * clusters[0][0] / n_pts)      # least_squares cost = 0.5 sum res^2
    accepted = [cl for cl in clusters
                if math.sqrt(2.0 * cl[0] / n_pts) - best_rms <= degeneracy_tol * data_rms]
    degenerate = len(accepted) > 1
    # Presentation order (owner decision, 2026-07-23): ascending (r_outer, r_core) -- the
    # ellipsoid sibling's deterministic-physical-key convention, for the same reason
    # (accepted degenerate solutions are statistically indistinguishable; raw-cost order
    # was ulp-noise-decided at ties). Acceptance above stays cost-based.
    accepted.sort(key=lambda cl: (cl[3], cl[2]))

    return [
        _build_core_shell_result(
            res, q, y, c, yn, y_scale, thermodynamic, n_conc, n_ang, n_pts, all_calibrated,
            estimator, cond_limit, degenerate, len(accepted), _P, m1, m2, n_core, n_shell,
            n_med, wl_nm, dn_dc_supplied=all_dn_dc_supplied, blank_measured=all_blank_measured,
            derived_provenance=_shared_derived_provenance(rayleigh_results),
            blank_provenance=_shared_blank_provenance(rayleigh_results), drop_notes=input_notes,
            standard_index_known=all_index_known, thresholds=thresholds)
        for _cost, res, _rc, _ro in accepted
    ]


def _build_core_shell_result(res, q, y, c, yn, y_scale, thermodynamic, n_conc, n_ang, n_pts,
                             calibrated, estimator, cond_limit, degenerate, n_solutions, P_of,
                             m1, m2, n_core, n_shell, n_med, wl_nm, *, dn_dc_supplied, blank_measured=True, derived_provenance='',
        blank_provenance='', drop_notes=(), standard_index_known=True, thresholds=None):
    """Assemble one CoreShellFormFactorResult (params, SEs, flags) from a converged fit.
    `thresholds` holds the deep-minimum, size-floor and contrast-match thresholds the
    verdicts read (the defaults where None), echoed on the result."""
    thresholds = {'core_shell_p_min': CORE_SHELL_FIT_P_MIN,
                  'form_factor_min_size_nm': FORM_FACTOR_MIN_SIZE_NM,
                  'core_shell_contrast_match_rel': ff._CORE_SHELL_CONTRAST_MATCH_REL,
                  **(thresholds or {})}
    inv_mw = float(res.x[0]) * y_scale
    r_core = float(res.x[1])
    t = float(res.x[2])
    r_outer = r_core + t
    a2 = float(res.x[3]) * y_scale if thermodynamic else None
    mw = 1.0 / inv_mw if (math.isfinite(inv_mw) and inv_mw > 0) else float('nan')
    rg = ff.core_shell_contrast_weighted_rg(
        r_core, r_outer, m1, m2, contrast_match_rel=thresholds['core_shell_contrast_match_rel'])
    # rg is nan for EITHER a near-zero net contrast (contrast-matched) OR an opposite-sign
    # core/shell radial moment (Rg^2 <= 0, a hollow-like profile with a large net contrast) --
    # in both the scattering Rg is not physically defined, so the wording below stays true of
    # both (a "contrast-matched" label would mislabel the second case).
    rg_undefined = not math.isfinite(rg)
    converged = bool(res.success and math.isfinite(inv_mw) and inv_mw > 0
                     and math.isfinite(r_core) and r_core > 0 and math.isfinite(mw))

    # ---- covariance -> SEs (HC3 sandwich on the fit Jacobian; delta method) ----
    # Physical param = D @ theta_n, D = diag([y_scale, 1, 1, y_scale]) (radii untouched). SEs
    # are emitted whenever the fit converged with a finite covariance -- NOT gated on well_cond
    # (matching the sphere/Mie/ellipsoid builders): an ill-conditioned fit still shows its ±
    # next to the mw_reliable=False flag, rather than a bare dash. Withholding never under-
    # reports, but emitting keeps the display uniform with the rest of the family (rule 10).
    cov_n, cond, well_cond = unc.covariance_from_jacobian(
        res.jac, res.fun, estimator, cond_limit)
    scale_vec = np.array([y_scale, 1.0, 1.0, y_scale][:cov_n.shape[0]])
    cov = cov_n * np.outer(scale_vec, scale_vec)
    mw_se = r_core_se = r_outer_se = a2_se = None
    if np.all(np.isfinite(cov)) and converged:
        p = cov.shape[0]
        j = np.zeros(p)
        j[0] = -1.0 / unc.power_or_nan(inv_mw, 2)           # Mw = 1/inv_mw (reciprocal; MC-gated)
        mw_se = unc.se_or_none(unc.propagate(j, cov))
        j = np.zeros(p); j[1] = 1.0                          # R_core direct
        r_core_se = unc.se_or_none(unc.propagate(j, cov))
        j = np.zeros(p); j[1] = 1.0; j[2] = 1.0              # R_outer = R_core + t (linear)
        r_outer_se = unc.se_or_none(unc.propagate(j, cov))
        if thermodynamic:
            j = np.zeros(p); j[3] = 1.0
            a2_se = unc.se_or_none(unc.propagate(j, cov))

    # ---- flags + regime indicator ----
    ss_res = float(np.sum(res.fun ** 2))
    ss_tot = float(np.sum((yn - yn.mean()) ** 2))
    r2 = unc.r_squared_or_nan(ss_res, ss_tot)
    qrg_max = float(np.max(q) * rg) if (math.isfinite(rg) and rg > 0) else float('nan')
    # Deep-minimum reliability: flag when the fitted coated P(q) dips below the floor within the
    # angular range (same criterion as the Mie sphere). Rg still reported.
    min_p = float(np.min(P_of(r_core, r_outer))) if converged else float('nan')
    # The size floor binds the OUTER RADIUS here, not Rg: this result's Rg is a contrast-weighted
    # moment that is undefined by design when the contrasts cancel (rg_undefined), so it is not a
    # size to judge. r_outer is the particle's outer radius and is finite on any converged fit.
    above_floor = _size_above_floor(r_outer, thresholds['form_factor_min_size_nm'])
    form_factor_valid = above_floor and not (math.isfinite(min_p)
                                             and min_p < thresholds['core_shell_p_min'])

    # The undefined Rg fails the size's verdict alone; the degeneracy is a qualifier read
    # off `degenerate` / `n_solutions`, never a note.
    scaled = ('mw_g_per_mol', 'a2_mol_mL_per_g2')
    notes: list = list(drop_notes)
    if rg_undefined:
        notes.append(Note('rg_undefined', ('rg_nm',)))
    # Only res.success solutions are assembled, so one that is not `converged` reached a
    # non-physical parameter.
    if not converged:
        notes += _nonphysical_notes(inv_mw, mw, (('R_core', r_core),), scaled + ('rg_nm',))
    if thermodynamic and a2 is not None and not math.isfinite(a2):
        notes.append(Note('a2_not_finite', scaled[1:]))

    # The design's conditioning beside the shape-fit reasons, never instead; between
    # those, the floor outranks the deep minimum -- see the sphere/Mie chain for why.
    if not well_cond:
        notes.append(_ff_ill_conditioned_note(cond, cond_limit, scaled))
    if not above_floor:
        notes.append(_sub_molecular_note('outer radius', r_outer,
                                         scaled + ('rg_nm', 'r_outer_nm'),
                                         thresholds['form_factor_min_size_nm']))
    elif not form_factor_valid:
        notes.append(Note('deep_minimum', scaled,
                          {'limit': float(thresholds['core_shell_p_min'])}))

    return CoreShellFormFactorResult(
        model=ff.CORE_SHELL, is_apparent=not thermodynamic, converged=converged,
        degenerate=degenerate, rg_undefined=rg_undefined,
        mw_g_per_mol=float(mw), rg_nm=float(rg), r_core_nm=float(r_core),
        r_outer_nm=float(r_outer), shell_thickness_nm=float(t), a2_mol_mL_per_g2=a2,
        q_nm_inv=q, kc_over_dR=y, concentrations_g_per_mL=c,
        n_points=n_pts, n_concentrations=n_conc, n_angles=n_ang,
        qrg_max=qrg_max, r_squared=float(r2), calibrated=calibrated,
        dn_dc_supplied=dn_dc_supplied,
        blank_measured=blank_measured,
        standard_index_known=standard_index_known,
        derived_provenance=derived_provenance,
        blank_provenance=blank_provenance,
        # Radii/Rg survive an uncalibrated OR dn/dc-absent run alike (the ordinate scale
        # cancels); Mw needs BOTH scales known, a physical fit, a
        # well-conditioned design, and a fit within a deep form-factor minimum.
        # `blank_measured` joins them, and unlike the two scales it also gates the SIZE:
        # an unmeasured blank is ADDITIVE, so it does not cancel in the ordinate
        # normalization the sizes rely on (angle-dependence is an extra, not the reason).
        mw_reliable=(calibrated and dn_dc_supplied and blank_measured
                     and converged and well_cond and form_factor_valid),
        a2_reliable=(thermodynamic and calibrated and dn_dc_supplied and blank_measured
                     and converged and well_cond
                     and form_factor_valid and a2 is not None and math.isfinite(a2)),
        # `above_floor` is on r_outer here (rg is a contrast-weighted moment, undefined by
        # design when the contrasts cancel), so a collapsed coated sphere withholds its Rg too.
        rg_reliable=(converged and math.isfinite(rg) and above_floor and blank_measured),
        form_factor_valid=form_factor_valid,
        mw_se=mw_se, r_core_se=r_core_se, r_outer_se=r_outer_se, a2_se=a2_se, rg_se=None,
        se_estimator=estimator, well_conditioned=well_cond, cond=cond,
        n_solutions=n_solutions, notes=tuple(notes),
        m1=float(m1), m2=float(m2), n_core=float(n_core), n_shell=float(n_shell),
        n_medium=float(n_med), wavelength_nm=float(wl_nm),
        core_shell_p_min=float(thresholds['core_shell_p_min']),
        form_factor_min_size_nm=float(thresholds['form_factor_min_size_nm']),
        core_shell_contrast_match_rel=float(thresholds['core_shell_contrast_match_rel']),
    )


# Agree-on-Rg tolerance for the "size determined, shape ambiguous" consensus of a degenerate
# form-factor fit: its per-solution Rg values agree when their relative spread is within
# _FF_CONSENSUS_RG_REL OR their absolute spread is within _FF_CONSENSUS_RG_ABS_NM (the floor keeps
# the relative test from being needlessly harsh at small sizes). Owner-set defaults (2026-07-21):
# core-shell 36.8/37.3/37.3/37.3 -> agree; ellipsoid 101/129/175 -> do not. The consensus reports
# the MEAN of the solution Rg values. Documented in the Advanced Guide.
_FF_CONSENSUS_RG_REL = 0.10
_FF_CONSENSUS_RG_ABS_NM = 1.0


def rg_values_agree(rgs, *, rel_tol: float = _FF_CONSENSUS_RG_REL,
                    abs_tol_nm: float = _FF_CONSENSUS_RG_ABS_NM) -> bool:
    """Whether a degenerate fit's per-solution Rg values agree closely enough to call the SIZE
    determined even though the shape/radii are not. True when the peak spread is within
    `rel_tol` of the mean OR within `abs_tol_nm` absolute (defaults _FF_CONSENSUS_RG_REL,
    _FF_CONSENSUS_RG_ABS_NM). Needs >= 2 finite positive values; a single value is trivially
    'determined' (True)."""
    vals = [float(r) for r in rgs if r is not None and np.isfinite(r) and r > 0]
    if len(vals) < 2:
        return len(vals) == 1
    spread = max(vals) - min(vals)
    mean = sum(vals) / len(vals)
    return spread <= abs_tol_nm or (mean > 0 and spread / mean <= rel_tol)


def form_factor_consensus_rg(rgs, *, rel_tol: float = _FF_CONSENSUS_RG_REL,
                             abs_tol_nm: float = _FF_CONSENSUS_RG_ABS_NM) -> Optional[float]:
    """The consensus Rg (nm) of a degenerate form-factor fit: the mean of the per-solution Rg
    values `rgs` when there is more than one and they agree (`rg_values_agree` at `rel_tol`,
    `abs_tol_nm`), else None. The caller passes the Rg of the degenerate solutions it offers."""
    rgs = list(rgs)
    if len(rgs) > 1 and rg_values_agree(rgs, rel_tol=rel_tol, abs_tol_nm=abs_tol_nm):
        return float(sum(rgs) / len(rgs))
    return None


def calibration_free_a2(
    sample_results,
    angle_deg: float,
    reference_index: int = 0,
    mw_g_per_mol: Optional[float] = None,
    estimator: str = 'hc3',
    cond_limit: float = unc.COND_LIMIT,
) -> CalibrationFreeA2Result:
    """A2 from intensity ratios at a fixed angle, without absolute calibration.

    A rearrangement of the Zimm equation's general form, Kc/dR = 1/(Mw P(q)) + 2 A2 c
    (Chu 1991, Eq. 2.1.8, printed p. 18). At a fixed angle it makes the excess intensity
    I_excess proportional to c Mw P /(1 + 2 A2 Mw P c), so forming

        Y(c) = [I_excess(c_ref)/c_ref] / [I_excess(c)/c] = (1 + 2 A2 Mw P c) / (1 + 2 A2 Mw P c_ref)

    gives a line in c whose slope/intercept = 2 A2 Mw P(q): the product 2 A2 Mw in the
    q -> 0 limit, apparent at a finite angle. The optical constant and the Rayleigh-ratio
    calibration cancel in the ratio, and so does c_ref: the reference only scales every
    Y(c) by one constant, so any concentration serves as it. This is useful when absolute
    calibration is uncertain, e.g. low-dn/dc systems.

    Inputs not prescribed by the source: the angle (the caller's; the source's product is
    its q -> 0 limit), the reference concentration (any; it cancels), the two-concentration
    minimum (the line's mathematical minimum), the drop of a point the mask hid (silent, the
    user's own choice) and of a non-positive excess (stated on ``notes``), the conditioning
    ceiling ``cond_limit`` (default ``COND_LIMIT``, a numerical-health bound) and the
    weighting (equal weights on every retained point).

    Parameters
    ----------
    sample_results : sequence
        Objects exposing ``concentration_g_per_mL``, ``angles_deg`` and
        ``excess_rayleigh_cm_inv`` (e.g. RayleighRatioResult). The excess Rayleigh
        ratio at the chosen angle is the excess-intensity proxy; the calibration
        constants cancel in Y, so their absolute scale is irrelevant.
    angle_deg : float
        Angle at which to form the ratios (ideally the lowest available, closest
        to q -> 0).
    reference_index : int
        Which concentration (after sorting ascending, excluding c = 0, any whose point
        at the angle the mask hid and any whose excess there is non-positive) is the
        reference c_ref. Default 0 (the lowest shown non-zero concentration with a
        positive excess). Every choice gives the same slope/intercept.
    mw_g_per_mol : float, optional
        If provided, A2 is returned (= (slope/intercept) / (2 Mw)).

    Returns
    -------
    CalibrationFreeA2Result

    Raises
    ------
    ValueError
        If fewer than two non-zero concentrations have the requested angle, or fewer
        than two once the points the mask hid are dropped, or once the non-positive
        excess points are dropped; or if ``reference_index`` is out of range for the
        concentrations left.

    Warns
    -----
    RuntimeWarning
        If any point with a non-positive excess Rayleigh ratio is dropped before the
        reference is chosen and Y(c) is formed.
    """
    # Build (c, excess) at the requested angle from each result, keeping the result
    # itself beside each point so the provenance of what actually entered Y(c) can be
    # reduced afterwards (rule 10: the quartet every sibling fit carries).
    pts = []
    for r in sample_results:
        c = r.concentration_g_per_mL
        if c == 0:
            continue
        idx = np.where(np.isclose(r.angles_deg, angle_deg))[0]
        if idx.size == 0:
            continue
        excess = float(r.excess_rayleigh_cm_inv[int(idx[0])])
        pts.append((c, excess, r))
    if len(pts) < 2:
        raise ValueError(
            f"Need at least two non-zero concentrations measured at "
            f"{angle_deg} deg; found {len(pts)}."
        )
    pts.sort(key=lambda t: t[0])
    # A non-finite excess carries no measurement; a point the mask hid is its ordinary
    # source (`masked_rayleigh` writes NaN). It leaves silently before the reference is
    # chosen, as `single_angle_ladder` drops it, so a hidden point never refuses the fit as
    # its reference and is never named as a non-positive excess below. Which remaining
    # concentration becomes the reference does not matter: it scales every Y(c) by one
    # constant (the docstring's rearrangement).
    pts = [p for p in pts if math.isfinite(p[1])]
    if len(pts) < 2:
        raise ValueError(
            f"calibration-free A2: after dropping the non-finite points (such as those the "
            f"mask hid), only {len(pts)} concentration(s) remain at {angle_deg} deg (need "
            "at least two). Check the mask and the dark count rate."
        )

    # Drop the non-positive excess points, also before the reference is chosen, so a
    # measured one gets one treatment wherever it sits, the reference included. A
    # negative or zero excess (noise floor or an over-subtracted solvent reference --
    # common in exactly the low-contrast systems this calibration-free estimator
    # targets) would give a negative or infinite Y and silently corrupt the
    # slope/intercept. Mirror the Berry non-positive-Kc/dR drop above. Unlike a hidden
    # point it is a measured value, so the note and its warning say it left.
    notes: list = []       # every note of the result, as Notes
    # Every point here is finite (the hidden ones left above), so the note counts only the
    # measured non-positive excesses; `isfinite` stays in the predicate as a structural
    # guard, so a +inf could never pass as positive if that drop changes.
    n_drop = sum(1 for p in pts if not (math.isfinite(p[1]) and p[1] > 0))
    if n_drop:
        # Single source of the message -> both the note and the warning.
        drop = _nonpositive_ordinate_note(n_drop, 'excess Rayleigh ratio', ('two_a2_mw',))
        notes.append(drop)
        _warn_note(drop, stacklevel=2)
        pts = [p for p in pts if math.isfinite(p[1]) and p[1] > 0]
        if len(pts) < 2:
            raise ValueError(
                "calibration-free A2: after dropping non-positive excess points, "
                f"only {len(pts)} concentration(s) remain (need at least two). "
                "Check the solvent blank."
            )
    conc = np.array([p[0] for p in pts])
    excess = np.array([p[1] for p in pts])
    sources = [p[2] for p in pts]

    if not (0 <= reference_index < conc.size):
        raise ValueError(
            f"reference_index {reference_index} out of range for {conc.size} "
            f"concentrations."
        )
    c_ref = conc[reference_index]
    e_ref = excess[reference_index]
    # The reference sets the numerator of every ratio, so it must be finite and positive.
    # Both drops above guarantee it; the negated form is what fires if either changes.
    if not (math.isfinite(e_ref) and e_ref > 0):
        raise ValueError(
            f"calibration-free A2: the reference concentration "
            f"(index {reference_index}, c = {c_ref:g} g/mL) has a non-positive or "
            f"non-finite excess Rayleigh ratio ({e_ref:g}); its ratio is undefined."
        )

    # Y(c) = (e_ref / c_ref) / (excess / c)
    Y = (e_ref / c_ref) / (excess / conc)
    fit = unc.linear_fit(conc, Y, estimator, cond_limit)  # cov order [intercept, slope]
    slope, intercept = fit.slope, fit.intercept
    r2 = fit.r_squared

    two_a2_mw = (float(slope / intercept)
                 if (math.isfinite(intercept) and intercept != 0) else float('nan'))
    a2 = (two_a2_mw / (2.0 * mw_g_per_mol)) if mw_g_per_mol else None

    # SE of slope/intercept (2A2Mw P(q), 2A2Mw at q -> 0): slope & intercept are correlated (same fit),
    # so propagate through the 2x2 covariance, not as an independent ratio. On a short
    # concentration ladder this HC3 SE is CONSERVATIVE (meets-or-exceeds the sampling
    # spread, ~1.3x at n=5; never under-reports) because HC3 up-weights the high-leverage
    # endpoints; it tightens toward exact as the ladder lengthens. See advanced_guide
    # 15.1/11.4 and tests/test_sls.test_calibration_free_a2_se_conservative.
    two_a2_mw_se = a2_se = None
    if intercept != 0 and math.isfinite(fit.intercept_se):
        jac = [-slope / unc.power_or_nan(intercept, 2),
               1.0 / intercept]                            # order [intercept, slope]
        two_a2_mw_se = unc.se_or_none(unc.propagate(jac, fit.cov))
        if two_a2_mw_se is not None and mw_g_per_mol:
            a2_se = two_a2_mw_se / (2.0 * mw_g_per_mol)     # Mw treated as exact

    # Numerical health of the Y-vs-c design [1, c]: near-equal concentrations make it
    # near-collinear, so slope/intercept (= 2 A2 Mw) is fragile even though the intercept
    # is ~1 by construction (Y(c_ref)=1, not the a->0 failure mode of the other methods).
    # Suppress two_a2_mw_reliable and carry the reason; the value is still returned.
    # And the intercept's sign (the single-angle ladder's rule 10 sibling): Y(c_ref) = 1
    # by construction, so a non-positive fitted intercept is a line the data do not
    # describe, and the product built on it is undefined.
    intercept_ok = bool(math.isfinite(intercept) and intercept > 0)
    all_blank = all(bool(getattr(s, 'blank_measured', True)) for s in sources)
    # Both scales cancel in Y(c), so only the blank is a precondition here: the two scale
    # flags below are records of the inputs, not gates (see the dataclass).
    product = ('two_a2_mw', 'a2_mol_mL_per_g2')
    reasons = list(scale_notes(sources, scale_free=product))
    if not fit.well_conditioned:
        reasons.append(_ill_conditioned_note(fit.cond, cond_limit, product))
    if not intercept_ok:
        reasons.append(_non_positive_intercept_note(intercept, product))
    elif not math.isfinite(two_a2_mw):
        reasons.append(Note('two_a2_mw_not_finite', ('two_a2_mw',)))
    notes += reasons

    return CalibrationFreeA2Result(
        angle_deg=float(angle_deg), concentrations_g_per_mL=conc, Y=Y,
        slope=float(slope), intercept=float(intercept), two_a2_mw=two_a2_mw,
        a2_mol_mL_per_g2=a2, r_squared=float(r2),
        two_a2_mw_se=two_a2_mw_se, a2_se=a2_se,
        se_estimator=estimator,
        well_conditioned=fit.well_conditioned, cond=fit.cond,
        # `and math.isfinite(two_a2_mw)`: the sibling reliability flags (Debye, Zimm/Berry,
        # Guinier) all carry it, and this one did not -- cond(X) is computed from the
        # concentration design alone, so a non-finite value would ride out flagged
        # reliable. This is the quantity CLAUDE.md singles out as surviving an
        # uncalibrated run, so it is the last one that should be able to (rule 10).
        # `all_blank`: the one absence this estimator does NOT shrug off -- see the
        # dataclass field for why an additive term survives the Y(c) ratio.
        two_a2_mw_reliable=(intercept_ok and fit.well_conditioned and all_blank
                            and math.isfinite(two_a2_mw)),
        notes=tuple(notes),
        is_apparent=True,
        # Records, not gates (see the dataclass): aggregated over the points that
        # entered Y(c), the way `zimm_analysis` aggregates them. getattr with the
        # "known" default because the inputs are duck-typed (see `sample_results`).
        calibrated=all(bool(getattr(s, 'calibrated', True)) for s in sources),
        dn_dc_supplied=all(bool(getattr(s, 'dn_dc_supplied', True)) for s in sources),
        blank_measured=all_blank,
        standard_index_known=all(bool(getattr(s, 'standard_index_known', True))
                                 for s in sources),
        derived_provenance=_shared_derived_provenance(sources),
        blank_provenance=_shared_blank_provenance(sources),
    )
