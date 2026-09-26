"""
core/data_models.py
===================

The shared internal data structures for the light scattering analysis platform.

This is the most important file in the project. Every parser produces these
objects, and every analysis function consumes them. The structures defined here
are the "common internal format" that decouples the parser layer from everything
downstream: a parser's only job is to turn a file on disk into one of these
objects, and an analysis function's only job is to turn these objects into
results. Neither side needs to know anything about the other.

Canonical internal units (decided deliberately; conversions happen at load time
inside the parsers, never here and never in analysis code):

    delay time      -> seconds            (s)
    temperature     -> kelvin             (K)
    concentration   -> grams per mL       (g/mL)
    wavelength      -> nanometres         (nm)   [q formula stays in nm/nm]
    angle           -> degrees            (deg)  [converted to radians in physics]
    viscosity       -> pascal-seconds     (Pa.s)
    count rate      -> counts per second  (cps)

Four primary objects live here:

    SampleKey      - the identity tuple used to associate DLS and SLS data
                     measured on the same sample.
    IntensityTrace - a count-rate-vs-time record. A first-class object:
                     loadable and analysable on its own, with or without an
                     associated correlogram.
    DLSMeasurement - one DLS correlogram measured at one angle.
    SLSMeasurement - one SLS angular series measured at one concentration.

The combination doctrine -- may these measurements be combined? (owner-ruled 2026-08-13)
-----------------------------------------------------------------------------------------

Membership is a proposal; combination is a contract; results follow the sample,
not its id. The workspace groups measurements into a sample as a bookkeeping
proposal (the auto-key plus the user's manual override) -- membership never by
itself licenses combining measurements in one fit. What an analysis may combine
is governed by two rules; every site that decides it points here and names the
rule it implements:

Rule 1 (partition). A fit or average never spans a user-set partition tag
(mw_fraction, polarization state). Enforced structurally -- a mixed set is
never eligible -- with the engine refusal as backstop, naming the field and the
remedy. Refusal exists ONLY here, and its criterion is the category error: the
pooled number would estimate nothing (Gamma_VH = Gamma_VV + 6 D_r; a mean over
two Mw fractions describes neither). A merely DEGRADED number -- one that
estimates a real quantity badly -- is never refused; it computes and discloses
under rule 2, the same flag-don't-block treatment PDI > 0.3 and an out-of-range
Rh already get.

Rule 1's listed tags are EXAMPLES, not the boundary (owner-ruled 2026-08-19).
The criterion is the category error -- the combined object has no referent --
and two things it is NOT.

It is not about trustworthiness. A number with a real referent that is merely
degraded, apparent, unanchored, or on an arbitrary scale is never refused,
however strongly its caveat is worded. A per-mode Zimm on a domain-type mode is
the worked case: its Mw rests on an unverifiable contrast assumption and its
disclosure says so in the strongest words the guide uses -- and it still
computes, ungated, because an apparent Mw for a real mode is a poor estimate of
something, not an estimate of nothing.

It is not about a fit's own axes. A Zimm regresses across concentration by
construction (the c coefficient IS 2 A2), and sample_group_key drops
concentration on purpose because a ladder is one sample. What cannot happen is a
single measurement OBJECT claiming two: an SLSMeasurement holds one
concentration_g_per_mL, so collating two into it writes a false record rather
than a poor estimate. That is the object-vs-fit divergence sample_group_key
already calls intentional, stated here because this is where a site author
reasons about combination.

So the refusing surfaces are wider than the two tags: check_replicate_identity
compares eight fields, and every one of them is a field whose disagreement makes
the AVERAGE describe no real condition.

A replicate average is not a repeat, so it never enters another average
(owner-ruled 2026-09-13). The replicate +/- is honest only because its inputs are
independent repeats of one condition; an average counted beside its own sources adds
no scatter but raises N, and beside other runs it counts as one repeat while carrying
a fraction of one run's variance. Refused in every averaging action -- the replicate
averages and the derived static point, where it is also listed ineligible -- on the item's
own tag, whatever else is selected; fitted on its own it is an ordinary measurement.
Reopen trigger, recorded: one real selection wrongly refused in ordinary use.

Rule 2 (disclosure). Everything else combines, and anything surprising about
the combination is stated where it travels: on the result itself when there is
one (notes -> export Comments), on the row or label when it is eligibility
(grayed + reason), as a passive hint when the program cannot verify the hazard
(an untagged multi-Mw series: Mw is an output, not an identity field). Silence
is never a mode where a user is present or a durable value is written -- and a
passive path (candidates, auto-select) never writes a durable value rule 1
would refuse. Identity fields (polymer, solvent, temperature -- compared only
through the shared comparison forms in this file) sit under THIS rule, not
rule 1: a within-sample identity mix is constructible only by the user's own
manual override, the hybrid grouping model makes that the user's call, and the
note travels to exports.

Dialogs guard the destruction of stored values; they never license
computations. A confirm can legitimate deleting the user's own stored result
(a regroup that would drop it); consent cannot make a meaningless number
meaningful, so there is no ask-and-proceed on any fit or averaging path.

Standing surfaces (a classification aid for the next site author, not policy):
refuse (the replicate-identity check, the controller's mixed-polarization
refusal, the DDLS cross-bucket check, an average offered to an average) is rule 1's
backstop; warn-and-continue
(the Gamma-vs-q2 identity note), filter-and-disclose (the DLS tab's grayed
other-fraction rows) and the passive hint (the SLS pooling hint) are rule 2 at
three surfaces; silent grouping is not a mode -- the sidebar IS the workspace
proposal's disclosure surface, and levels self-derived from one array are not
identity decisions at all; silent filtering survives only where no user is
addressable (the Rh candidate tiers), and the offered label still discloses.

Implementation note:
    These are @dataclasses (Optional[X] fields default to None where a
    measurement can exist without them). We pass eq=False because the objects
    hold NumPy arrays: the auto-generated __eq__ would compare arrays with ==,
    which returns an array rather than a bool and raises. Identity and
    association are handled by SampleKey instead.

Change history
--------------
2026-06-12  Initial implementation. (data_models v1)
            Defines SampleKey, IntensityTrace, DLSMeasurement, SLSMeasurement.
            Canonical units: s, K, g/mL, nm, deg, Pa.s, cps.
            DLSMeasurement renamed from ScatteringMeasurement for symmetry
            with SLSMeasurement.
            IntensityTrace field renamed count_rates_kcps -> count_rates_cps;
            Brookhaven parser will multiply kcps values by 1000 on load.
            Viscosity stored as Pa.s; user-facing mPa.s option handled at
            the parser / confirmation layer (multiply by 1e-3 on load).
2026-06-12  Added solvent_name. (data_models v2)
            solvent_name is now a required field on DLSMeasurement and
            SLSMeasurement and a fourth field in SampleKey, so the same
            polymer in different solvents no longer collides. Solvent names
            are normalized against a controlled vocabulary
            (normalize_solvent_name); unrecognized names are used as-is with
            a one-time warning. IntensityTrace does not carry a solvent.
2026-06-19  Added analyzer_geometry to SLSMeasurement. (data_models v3)
            Optional polarization geometry ('VV'/'VH'/'VU' or None), the bridge
            for static depolarized light scattering: a VV and a VH series sharing
            sample identity + angle pair into the Cabannes correction and the
            depolarization ratio (analysis/depolarization.py). Optional with a None
            default -> non-breaking; all existing polarized workflows are unchanged.
            DLSMeasurement gains the same field when the dynamic DPLS phase begins.
2026-06-19  Added analyzer_geometry to DLSMeasurement. (data_models v4)
            The same optional 'VV'/'VH'/'VU' field on the correlogram object, for
            the DYNAMIC depolarized analysis: a VV and a VH correlogram at one angle
            pair into the rotational diffusion coefficient D_r = (Gamma_VH -
            Gamma_VV)/6 (analysis/depolarization.analyze_ddls). Non-breaking (None
            default); workspace.build() forwards it.
2026-08-06  Solvent identity: the program no longer guesses. (data_models v5)
            The controlled vocabulary (_SOLVENT_ALIASES and its normalize /
            is_recognized / warn helpers) is DELETED. A solvent name is stored
            verbatim -- whatever was picked, typed, or recorded by the instrument --
            and compared only after strip + casefold, in key construction. Nothing
            is mapped: "H2O" is its own sample, not water. Polymer names now get
            that same rule (previously case-preserving), so the two identity fields
            behave identically. Temperature moves from six significant figures to
            the shared 0.1 K bucket in round_temperature_for_identity(), which
            core/workspace.sample_group_key uses too -- one computation, so the
            two notions of sample identity cannot drift apart again.
2026-08-13  The combination doctrine stated once. (data_models v6)
            "May these measurements be combined?" now has one answer -- the
            two-rule doctrine above. Every identity-deciding consumer carries a
            one-line pointer naming the rule it implements instead of deriving
            its own; the doctrine text lives only here.
2026-09-10  The fit's instrument inputs travel on the measurement. (data_models v7)
            DLSMeasurement gains two optional fields: `sample_times_s`, the
            per-channel sample time a parser records when its file states the
            correlator's sampling structure (the Brookhaven .dat; None for every
            format that states none), and `mean_count_rate_cps`, the run's mean
            photon count rate, set by `LoadedMeasurement.build()` from the COMMITTED
            parameter -- so the count rate reaches the engine exactly as viscosity
            does, on the object, and no caller threads it. `usable_rate()` is the
            one predicate for "is this a count rate", shared by every reader.
2026-09-11  The run duration joins them. (data_models v8)
            DLSMeasurement gains `duration_s`, the run duration T a parser records
            when its file states one for the correlogram (the ALV `Duration` of a
            one-run acquisition), in `raw` beside the sample times and never an
            editable parameter. The exact-in-Gamma residual weight's 1/(M - k) reads
            it; without one the weight takes its T -> infinity form.
2026-09-13  Rule 1 gains the replicate average. (doctrine text only)
            A replicate average never enters another average: it is not a repeat,
            and counting it as one under-reports the replicate +/-. No field change.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import NamedTuple, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Sample identity / join key
# ---------------------------------------------------------------------------

# How many significant figures to keep when building a SampleKey from floating
# point concentration and temperature values. See make_sample_key() below for
# why this exists. Six sig figs is generous enough never to merge two genuinely
# different samples, while absorbing the tiny rounding noise that unit
# conversions introduce (e.g. converting mg/mL to g/mL).
SAMPLE_KEY_SIG_FIGS = 6


class SampleKey(NamedTuple):
    """
    Identifies a physical sample for the purpose of associating datasets.

    This is the join key that will eventually let the program pair an Rh value
    from a DLS measurement with an Rg value from an SLS measurement on the same
    sample, to compute rho = Rg / Rh -- without any future refactor.

    It is a NamedTuple, which means two SampleKeys are equal (and hash equal,
    so they work as dictionary keys) when all four fields match. That is
    exactly the behavior a join key needs.

    Fields: (polymer, solvent, concentration, temperature).
    Solvent is part of sample identity because the same polymer at the same
    concentration and temperature in two different solvents is two physically
    different samples (different chain conformation, different solvent quality).
    Including solvent lets you compare, e.g., PVP in water vs PVP in reline
    without their keys colliding. Solvent and polymer names are compared after
    strip + casefold and nothing more, so "water", "Water" and " WATER " are one
    key while "H2O" is its own -- the program does not guess that an abbreviation
    means a particular solvent (see make_sample_key).

    This granularity is deliberately more than the rho = Rg/Rh use case strictly
    needs (which pairs extrapolated c->0 results), because more granularity is
    always recoverable -- you can match on a subset of fields -- whereas lost
    granularity cannot be reconstructed.

    IMPORTANT: build these with make_sample_key(), not by calling SampleKey(...)
    directly. The factory rounds the floating-point fields so that values which
    are "the same number" in physical terms compare as equal, and applies the
    strip + casefold rule to the two name fields.
    """
    polymer_name: str
    solvent_name: str
    concentration_g_per_mL: Optional[float]   # None for a DLS-only measurement
    temperature_K: float


# ---------------------------------------------------------------------------
# Identity-field comparison
# ---------------------------------------------------------------------------

# These are the comparison primitives BENEATH the combination doctrine (module
# docstring above): they answer "do these fields match?", never "may these be
# combined?" -- that question is the doctrine's, and its consumers point there.

# The one rule, applied to BOTH identity name fields (polymer and solvent) on every path --
# picked from the library, typed by hand, or recorded by the instrument:
#
#     store   verbatim, exactly as it arrived
#     compare strip + casefold, in key construction only
#     map     never

# There is deliberately no controlled vocabulary folding "H2O" or "MilliQ" into canonical
# names. The program does not guess what a name means: casefolding is not a guess, because
# "Water" and "water" are the same characters, but deciding "H2O" means water is. Instrument
# metadata makes the point -- a Zetasizer's dispersant cell is free text typed by the
# operator, and "My Phosphate Buffer" is an ordinary value for it. So "H2O" is its own
# sample and the user supplies its refractive index and viscosity by hand (invariant 3).
# A picked value being canonical BY CONSTRUCTION also leaves no normalization for two call
# sites to disagree about. See core/workspace.sample_group_key.


def casefold_identity_name(raw_name: str) -> str:
    """The comparison form of an identity name: stripped and casefolded, nothing else.

    Used for BOTH polymer_name and solvent_name, by both notions of sample
    identity (:func:`make_sample_key` here and ``core.workspace.sample_group_key``).
    Callers store the original string; only keys see this form.

    ``str.casefold`` rather than ``str.lower`` because it is the Unicode-correct
    caseless-matching operation -- German "STRASSE" and "Straße" compare equal,
    which ``lower`` gets wrong. For the ASCII solvent names in the library the two
    agree, so this costs nothing and is right for a name the user typed.
    """
    return raw_name.strip().casefold()


# Temperature is bucketed rather than compared exactly, because no experiment holds a set
# point to better than about 0.05 C and a finer key splits one sample on thermometer noise.
# One decimal place (0.1 K) is deliberately COARSER than the six significant figures
# concentration uses: concentration is a number you dilute to, temperature one you approach.

# The bucket is placed in CELSIUS, not Kelvin, and that is load-bearing. Set points are
# chosen in Celsius and the conversion adds exactly 0.15, so 20, 25, 30, 35 and 50 C land
# EXACTLY on 0.1 K bin edges -- bucketing in Kelvin would split a sample at 25.00 C over a
# microkelvin of noise, the modal case rather than a rare one. Rounding the Celsius value
# puts each at a bin CENTRE, 0.05 C clear of either edge. Not tuning to a lab or instrument
# (invariant 4): it is the unit the number was chosen in. Known limitation, accepted: any
# rounding bucket splits at ITS edges, so 25.04 and 25.06 C land either side of one. The
# remedy is the manual regrouping that already ships, NOT a clustering algorithm.
TEMPERATURE_GROUP_DECIMALS = 1
_KELVIN_AT_ZERO_CELSIUS = 273.15


def round_temperature_for_identity(temperature_K: float) -> float:
    """Bucket a temperature to 0.1 C for sample identity, returned in Kelvin.

    THE single computation behind both notions of sample identity, so there is no
    second predicate that can be sized differently from the first (CLAUDE.md rule
    12). ``core.workspace.sample_group_key`` imports this rather than rounding on
    its own; that function handles its own None-tolerance before calling here.

    Two temperatures in the same bucket return the same float, which is all a key
    needs -- the value is a bucket label, never a temperature to compute with.
    """
    celsius = float(temperature_K) - _KELVIN_AT_ZERO_CELSIUS
    return round(celsius, TEMPERATURE_GROUP_DECIMALS) + _KELVIN_AT_ZERO_CELSIUS


def _round_to_sig_figs(value: float, sig_figs: int) -> float:
    """Round a float to a fixed number of significant figures (not decimals).

    We use significant figures rather than decimal places because canonical
    units span very different magnitudes: concentration is a small number like
    0.0003638 g/mL, while temperature is a few hundred K. A fixed number of
    decimal places that suited one would ruin the other.
    """
    if value == 0.0:
        return 0.0
    # math.floor(log10(|value|)) gives the position of the leading digit.
    decimals = sig_figs - 1 - math.floor(math.log10(abs(value)))
    return round(value, decimals)


def round_concentration_for_identity(concentration_g_per_mL: float) -> float:
    """Bucket a concentration for identity / distinctness comparisons, in g/mL.

    THE one concentration comparison form -- the same SAMPLE_KEY_SIG_FIGS
    rounding :func:`make_sample_key` applies, promoted so the eligibility and
    distinctness sites (distinct-concentration count gates, duplicate-ladder
    detection, the GUI's tick-all-at matching) cannot size their own decimals
    differently from the key that groups the data. Two concentrations that are
    one value to the sample key are one value to every consumer.

    A non-finite value passes through unchanged, exactly as the site-local
    ``round(c, 12)`` this form replaced did: a comparison form compares, it
    never validates -- a nan/inf must surface at the input that produced it,
    not as a crash in a passive display hint reading committed parameters.
    """
    c = float(concentration_g_per_mL)
    if not math.isfinite(c):
        return c
    return _round_to_sig_figs(c, SAMPLE_KEY_SIG_FIGS)


# Angle identity bucket: two runs are "at the same angle" when their angles agree
# to 3 decimal places. Angles arrive as fixed instrument text with no per-run
# drift mechanism, so this is a float-formatting guard, not a tolerance. Angle is
# not a SampleKey field (it is an axis within a sample), which is why this form
# lives beside the key primitives rather than inside them.
ANGLE_IDENTITY_DECIMALS = 3


def round_angle_for_identity(angle_deg: float) -> float:
    """Bucket a scattering angle for identity comparisons ("same angle?").

    THE one angle comparison form, shared by the DDLS VV/VH pairing, the Rh
    candidate tiers' angle grouping, and the GUI's tick-all-at-angle matching --
    so no consumer buckets the same physical question at its own precision.
    """
    return round(float(angle_deg), ANGLE_IDENTITY_DECIMALS)


def polarization_state(geometry: Optional[str]) -> Optional[str]:
    """THE one mapping from an analyzer-geometry label to a polarization state.

    Only 'VV' and 'VH' select a polarization; 'VU' records no analyzer in the beam
    and an absent tag means the user has not said -- one unpolarized class, never
    distinct tokens (combination doctrine rule 1, this module's docstring). Shared
    by the interactive fits' mixed-polarization refusal, the Rh candidate tiers' VH
    exclusion, the DDLS pairing, and the derived-static-series collation, so no two
    sites can classify one label differently.

    Lives here rather than in a consumer because it is a comparison primitive like
    the forms above, and its consumers now span app/ and analysis/.
    """
    return geometry if geometry in ('VV', 'VH') else None


# THE one field-name -> identity-comparison-form registry, so no GUI surface can
# carry its own copy of the mapping and drift from a sibling's (the mapping-level
# version of the per-site drift the forms above exist to end). A field absent
# here has no identity form -- compare it exactly or not at all.
IDENTITY_FORM_BY_FIELD = {
    'angle_deg': round_angle_for_identity,
    'concentration_g_per_mL': round_concentration_for_identity,
}


def make_sample_key(
    polymer_name: str,
    solvent_name: str,
    concentration_g_per_mL: Optional[float],
    temperature_K: float,
) -> SampleKey:
    """Canonical constructor for SampleKey. Always use this, never SampleKey().

    Comparison forms applied:
      - polymer_name and solvent_name: stripped + casefolded, nothing mapped
        (:func:`casefold_identity_name`). Both name fields get the SAME rule --
        they are the same kind of thing, and treating them differently is what
        let the two notions of sample identity drift apart.
      - temperature: bucketed to 0.1 K by :func:`round_temperature_for_identity`,
        the shared computation ``core.workspace.sample_group_key`` also uses.
      - concentration: rounded to SAMPLE_KEY_SIG_FIGS significant figures so
        floating-point noise (e.g. a mg/mL -> g/mL conversion) cannot break a join.
    """
    name = casefold_identity_name(polymer_name)
    solvent = casefold_identity_name(solvent_name)
    # Concentration may be absent for a DLS-only measurement (size/diffusion
    # analysis does not use it). Keep None in the key so such measurements still
    # join by (polymer, solvent, temperature) without colliding with a real c.
    conc = (None if concentration_g_per_mL is None
            else round_concentration_for_identity(concentration_g_per_mL))
    temp = round_temperature_for_identity(temperature_K)
    return SampleKey(name, solvent, conc, temp)


# ---------------------------------------------------------------------------
# Small validation helpers (private to this module)
# ---------------------------------------------------------------------------
#
# Centralizing these means each dataclass validates itself the same way, with
# clear error messages, instead of repeating checks inline. Failing loudly at
# construction time -- the moment bad data enters the system -- is far easier
# to debug than a confusing error deep inside an analysis function later.

def _as_1d_float_array(value, field_name: str) -> np.ndarray:
    """Coerce input to a 1-D float NumPy array, or raise a clear error."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1:
        raise ValueError(
            f"{field_name} must be one-dimensional, but has shape {arr.shape}."
        )
    if arr.size == 0:
        raise ValueError(f"{field_name} must not be empty.")
    return arr


def _require_same_length(a, b, name_a: str, name_b: str) -> None:
    if len(a) != len(b):
        raise ValueError(
            f"{name_a} and {name_b} must have the same length "
            f"({len(a)} != {len(b)})."
        )


# The two scalar guards below test FINITENESS explicitly. Neither comparison does it for
# them, and the asymmetry is easy to misread: `not (v > 0)` rejects nan but ADMITS +inf,
# while `v < 0` admits BOTH nan and +inf. A non-finite physical scalar that survives here
# reaches the engine and crashes it far from the cause -- an infinite refractive index
# gives q = inf, and np.polyfit answers that with a bare "SVD did not converge". Tokens
# like `inf`, `nan` and `1e400` all parse straight through the GUI's float(text), so this
# is a typo away (2026-07-31 spec-01; mirrors the _require_finite_array guard on the
# measured arrays and the explicit dn/dc check on SLSMeasurement).

def _require_positive(value, field_name: str) -> None:
    if value is None or not (math.isfinite(value) and value > 0):
        raise ValueError(
            f"{field_name} must be a positive finite number, got {value!r}."
        )


def _require_non_negative(value, field_name: str) -> None:
    if value is None or not (math.isfinite(value) and value >= 0):
        raise ValueError(
            f"{field_name} must be a finite number, zero or positive, got {value!r}."
        )


def _require_finite_or_none(value, field_name: str) -> None:
    """Reject a non-finite scalar, but accept None and accept zero.

    For optional fields where the required property is that the number IS a number --
    a display-only vendor constant, say -- rather than that it is physically positive.
    """
    if value is not None and not math.isfinite(value):
        raise ValueError(
            f"{field_name} must be a finite number, got {value!r}."
        )


def _require_scattering_angle(value, field_name: str) -> None:
    if value is None or not (0 < value < 180):
        raise ValueError(
            f"{field_name} must be strictly between 0 and 180 degrees, "
            f"got {value!r}."
        )


def _require_non_empty_string(value, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{field_name} must be a non-empty string, got {value!r}."
        )


def _require_finite_array(arr: np.ndarray, field_name: str) -> None:
    """Reject an array holding any nan/inf value.

    Tokens like nan/inf/1e400 parse through float() into non-finite values; a
    correlogram or delay axis containing them is meaningless and must never reach
    the analysis engine (mirrors the SLSMeasurement dn/dc math.isfinite guard).
    """
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            f"{field_name} must contain only finite numbers; found nan/inf value(s)."
        )


def _require_increasing_lag_axis(arr: np.ndarray, field_name: str) -> None:
    """Reject a delay/lag axis that is not strictly increasing from a non-negative start.

    A correlator's lag axis is strictly increasing by construction; non-monotonic,
    duplicate, or negative lags are physically impossible and would corrupt every
    downstream fit (which assumes an ordered axis). A leading zero-lag channel is
    allowed -- some correlators report tau = 0 as the first channel -- so only a
    NEGATIVE first value is rejected, not a zero one.
    """
    if arr[0] < 0 or not np.all(np.diff(arr) > 0):
        raise ValueError(
            f"{field_name} must be strictly increasing from a non-negative start "
            f"(a correlator lag axis); got non-monotonic, duplicate, or "
            f"negative values."
        )


def _require_non_negative_array(arr: np.ndarray, field_name: str) -> None:
    """Reject an array holding any negative value (e.g. a count rate)."""
    if np.any(arr < 0):
        raise ValueError(
            f"{field_name} must be zero or positive; found negative value(s)."
        )


def _require_positive_array(arr: np.ndarray, field_name: str) -> None:
    """Reject an array holding any zero or negative value (e.g. a sample time)."""
    if np.any(arr <= 0):
        raise ValueError(
            f"{field_name} must be strictly positive; found zero or negative value(s)."
        )


def usable_rate(value) -> Optional[float]:
    """A mean photon count rate as the engine takes it, or None -- the ONE predicate.

    The rule, in order: `bool`, `str` and `bytes` are refused by TYPE (``float(True)``
    and ``float('1e5')`` are numbers a caller never meant); then ``float()`` admits every
    real numeric type, numpy scalars and 0-d arrays included; then the value must be
    finite and strictly positive. A zero is not a rate -- an all-zero trace produces
    one, and the fit must fall to its classical limit and say so rather than weight
    on a placeholder. Two-sided by construction: an underflowed 0.0 and a saturated
    inf both come back None.

    Every reader and writer of the committed count rate goes through here, so "is this
    a usable rate" has one answer; the roster is pinned by
    `tests/test_workspace.py::test_every_reader_of_the_committed_count_rate_agrees`.
    """
    if value is None or isinstance(value, (bool, str, bytes)):
        return None
    try:
        rate = float(value)
    except (TypeError, ValueError):
        return None
    return rate if math.isfinite(rate) and rate > 0.0 else None


# Plausibility bands for auto-filled header metadata. These are DELIBERATELY wide,
# general sanity bounds (not solvent- or instrument-specific tuning, cf. invariants
# 3 & 4): a value outside them is almost certainly a unit slip or typo in the source
# file (e.g. T = 2980 K for 298.0, n = 13.3 for 1.33). The check only WARNS -- the
# value is user-owned and freely overridable -- it never rejects.
_PLAUSIBLE_TEMPERATURE_K = (250.0, 370.0)      # solution light scattering
_PLAUSIBLE_REFRACTIVE_INDEX = (1.2, 1.8)       # common liquids


def implausible_metadata_reasons(temperature_K, refractive_index) -> list:
    """Short, human-readable reasons the given auto-fill metadata looks physically
    implausible (a likely unit slip), or ``[]`` if both are within their bands.

    Pure and side-effect-free: the SINGLE source of truth for the plausibility bands,
    shared by the load-time warning (:func:`_warn_if_implausible_metadata`) and the
    Data-tab's passive note, so the two never drift. Each reason is glyph-less and
    self-contained, so a caller can lead it with its own framing (a UserWarning
    preamble, or a GUI ``ⓘ`` note).
    """
    reasons = []
    lo, hi = _PLAUSIBLE_TEMPERATURE_K
    if temperature_K is not None and not (lo <= temperature_K <= hi):
        reasons.append(
            f"temperature {temperature_K:g} K is outside the usual {lo:g}-{hi:g} K "
            f"for solution light scattering")
    lo, hi = _PLAUSIBLE_REFRACTIVE_INDEX
    if refractive_index is not None and not (lo <= refractive_index <= hi):
        reasons.append(
            f"refractive index {refractive_index:g} is outside the usual "
            f"{lo:g}-{hi:g} for common liquids")
    return reasons


def _warn_if_implausible_metadata(temperature_K, refractive_index) -> None:
    """Emit a non-fatal UserWarning if T or n is physically implausible.

    Called from the measurement __post_init__ so it fires when a measurement is built
    from parsed/auto-filled header metadata. A warning, not a guard: the user owns
    these values and may legitimately override them (invariant 3). The GUI surfaces
    the same check as a passive note via :func:`implausible_metadata_reasons`.
    """
    reasons = implausible_metadata_reasons(temperature_K, refractive_index)
    if reasons:
        warnings.warn(
            "Implausible metadata (likely a unit slip): " + "; ".join(reasons)
            + ". The value is used exactly as entered.",
            UserWarning, stacklevel=3,
        )


# Polarization/analyzer geometry of a measurement, as (incident, analyzer):
#   VV  vertical incident, vertical analyzer     -- ordinary polarized intensity
#   VH  vertical incident, horizontal analyzer   -- depolarized intensity (DPLS)
#   VU  vertical incident, no analyzer           -- vertical laser, no analyzer
# None means the geometry was not recorded (legacy / unspecified). This vocabulary
# matches the Rayleigh-geometry codes in physics/constants.py; it is duplicated here
# (not imported) to keep core/ free of any physics/ dependency. The depolarization
# analysis pairs a VV measurement with a VH measurement by sample identity + angle.
ANALYZER_GEOMETRIES = ('VV', 'VH', 'VU')


def _normalize_analyzer_geometry(value, field_name: str):
    """Return the canonical upper-case geometry code, or None. Raise if invalid."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            f"{field_name} must be a string (one of {ANALYZER_GEOMETRIES}) or None, "
            f"got {value!r}."
        )
    code = value.strip().upper()
    if code not in ANALYZER_GEOMETRIES:
        raise ValueError(
            f"{field_name} must be one of {ANALYZER_GEOMETRIES} or None, got {value!r}."
        )
    return code


# Polymer chain architecture of a SAMPLE, for the Cross-Sample branching readouts:
#   'linear'    a linear reference (defines the linear Rg-Mw / Rh-Mw law)
#   'branched'  a branched target (gets a contraction factor g and/or h)
# None means untagged -- treated as linear for the Scaling-plot fit, but NOT used as an
# architecture linear reference, which requires an explicit tag. A sample-level LABEL, never
# a grouping key: linear and branched are already different polymers. Stored per-measurement
# in committed_params, never on the pure measurement/analysis objects.
CHAIN_TYPES = ('linear', 'branched')


def _normalize_chain_type(value, field_name: str = 'chain_type'):
    """Return the canonical lower-case chain-type code, or None. Raise if invalid."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            f"{field_name} must be a string (one of {CHAIN_TYPES}) or None, "
            f"got {value!r}."
        )
    code = value.strip().lower()
    if code == '':
        return None
    if code not in CHAIN_TYPES:
        raise ValueError(
            f"{field_name} must be one of {CHAIN_TYPES} or None, got {value!r}."
        )
    return code


# ---------------------------------------------------------------------------
# IntensityTrace  (count rate vs time)
# ---------------------------------------------------------------------------

@dataclass(eq=False, repr=False)
class IntensityTrace:
    """A count-rate-vs-time record from a scattering experiment.

    This is a first-class, independent object. It can be loaded from a count
    rate file and analyzed on its own (drift, stationarity, dust spikes,
    basic statistics) with no correlogram present at all. When it does belong
    to a correlogram, the association is optional and made via measurement_id.

    Fields
    ------
    times_s
        Elapsed time of each sample, in seconds from the start of the run.
        Parsers convert native timestamp formats (e.g. HH:MM:SS.sss) into
        elapsed seconds.
    count_rates_cps
        Photon count rate at each time point, in counts per second (cps).
        The Brookhaven export is in kcps; the parser multiplies by 1000
        before constructing this object.
    measurement_id
        Optional back-reference to an associated DLSMeasurement. None if
        this trace stands alone.
    sample_label
        Human-readable label carried from the source file, if present.
    source_file
        Path to the file this trace was loaded from (provenance).
    """

    times_s: np.ndarray
    count_rates_cps: np.ndarray

    measurement_id: Optional[str] = None
    sample_label: Optional[str] = None
    source_file: Optional[str] = None

    def __post_init__(self) -> None:
        # Clean and validate inputs (runs right after the dataclass __init__).
        self.times_s = _as_1d_float_array(self.times_s, "times_s")
        self.count_rates_cps = _as_1d_float_array(
            self.count_rates_cps, "count_rates_cps"
        )
        _require_same_length(
            self.times_s, self.count_rates_cps, "times_s", "count_rates_cps"
        )
        # Reject nan/inf on BOTH arrays before any physical check. _require_non_negative_array
        # below uses np.any(arr < 0), which admits nan AND +inf (nan comparisons are False,
        # +inf >= 0), so it is NOT a finite guard -- the correlogram sibling (DLSMeasurement)
        # finite-checks its arrays and this one must too (rule-10 fail-loud symmetry).
        _require_finite_array(self.times_s, "times_s")
        _require_finite_array(self.count_rates_cps, "count_rates_cps")
        # Raw photon count rates are physically non-negative; a negative value is a
        # corrupt/over-subtracted trace, not real data. Reject it here rather than
        # let it bias downstream statistics (mean, baseline, blocking).
        _require_non_negative_array(self.count_rates_cps, "count_rates_cps")

    def __repr__(self) -> str:
        # A compact repr keeps the console and debugger readable
        # instead of dumping the full arrays.
        label = self.sample_label if self.sample_label is not None else "unlabeled"
        return (
            f"IntensityTrace(label={label!r}, "
            f"n_points={self.times_s.size}, "
            f"duration_s={self.times_s[-1]:.1f})"
        )


# ---------------------------------------------------------------------------
# DLSMeasurement  (one correlogram at one angle)
# ---------------------------------------------------------------------------

@dataclass(eq=False, repr=False)
class DLSMeasurement:
    """One dynamic light scattering correlogram, measured at a single angle.

    The stored correlogram is always g2(tau) - 1, the baseline-subtracted
    intensity autocorrelation function. Parsers are responsible for converting
    whatever a file contains (raw g1, un-subtracted g2, etc.) into this single
    canonical form before constructing the object, so that all analysis code
    can assume one representation.

    Required data
    -------------
    delay_times_s, correlogram
        The measured curve. Equal-length 1-D arrays. delay_times_s is in
        seconds; correlogram is g2(tau) - 1 (dimensionless).

    Required identity  (these four together define the SampleKey)
    -------------------------------------------------------------
    polymer_name, solvent_name, concentration_g_per_mL, temperature_K

    Required optics / geometry  (needed to compute the scattering vector q)
    -----------------------------------------------------------------------
    angle_deg, wavelength_nm, solvent_refractive_index

    Optional
    --------
    viscosity_Pa_s
        Solvent viscosity at the measurement temperature, in Pa.s.
        Required only for the Stokes-Einstein step (D -> Rh). Cumulant
        fitting and Gamma/q^2 analysis do not need it, so a measurement
        may exist without it. The Stokes-Einstein function raises a clear
        error if it is called without this value.
        User-facing entry in mPa.s is supported at the parser / parameter
        confirmation layer; the value is multiplied by 1e-3 before storage.
    sample_label, instrument_name, source_file
        Provenance carried from the source file where available.
    sample_times_s
        Per-channel sample time Delta_t_k, in seconds, same length as the delay
        axis: the time the correlator accumulated each channel over, which the
        nonlinear cumulant's sourced residual weight (Schaetzel 1990 Eq. 49) needs
        per channel. Written by a parser whose file STATES the sampling structure
        (the Brookhaven .dat: one sampling time per correlator section plus the
        high-speed section's fixed table) and None otherwise, in which case the
        engine derives a stand-in from the delay axis and records which was used.
        An instrument fact like the lags themselves: never user-edited.
    mean_count_rate_cps
        The run's mean photon count rate, in counts per second -- the one header
        value the same weight reads (its shot-noise groups, n_bar_k = <c.r.> *
        Delta_t_k). User-owned and provenance-tagged like viscosity: it reaches
        this object only from the COMMITTED parameters through
        `LoadedMeasurement.build()`, None when nothing usable was committed (the
        classical infinite-count-rate limit, which the fit result discloses).
        Must be a usable rate when present -- see `usable_rate`.
    duration_s
        The run duration T, in seconds: how long the correlator accumulated this
        correlogram. The exact-in-Gamma form of the same weight (Biganzoli & Ferri
        2018 Eq. 9) averages lag k over M - k products, 1/(M_k - k) = Delta_t_k /
        (T - tau_k), so T stays in the relative shape unless it is long against the
        last lag. Written by a parser whose file STATES it for this correlogram (the
        ALV `Duration` of a one-run acquisition; a Brookhaven header whose stated run
        length covers every channel that collected counts) and None otherwise, in which case the
        weight takes its T -> infinity form and the fit result records which. An
        instrument fact like the sample times: never user-edited. Finite and positive
        when present.

    A count-rate trace recorded beside this correlogram is NOT held here. Traces
    live in one place -- the workspace's own trace store, keyed by trace_id, with
    an optional measurement_id back-reference -- so that a solvent or calibrant
    trace, which belongs to no measurement, is representable by the same surface.
    """

    # --- measured data ---
    delay_times_s: np.ndarray
    correlogram: np.ndarray          # g2(tau) - 1

    # --- sample identity ---
    polymer_name: str
    solvent_name: str
    # Concentration is OPTIONAL for DLS: cumulant/single/double/KWW/NNLS/CONTIN
    # and Gamma/q^2 size analysis never use it. It is still required for the
    # multi-concentration diffusion extrapolation, which raises a clear error if
    # asked to run without it. None is allowed; a supplied value must be >= 0.
    concentration_g_per_mL: Optional[float]
    temperature_K: float

    # --- optics / geometry ---
    angle_deg: float
    wavelength_nm: float
    solvent_refractive_index: float

    # --- optional ---
    viscosity_Pa_s: Optional[float] = None
    # Solute (particle) refractive index, hand-entered per sample. Consumed ONLY by the
    # out-of-Rayleigh (RDG) intensity->volume/number reweighting (analysis/optical_model.py)
    # and the planned Phase-3 Mie treatment, which form m = n_solute/solvent_refractive_index.
    # Treated like dn/dc: NEVER proposed, defaulted, or guessed (invariant 3) -- it is
    # material-specific and there is no particle library. Blank (None) is valid for pure
    # DLS/cumulant runs and for Rayleigh-limit reweighting (which needs no contrast).
    solute_refractive_index: Optional[float] = None
    # Polarization/analyzer geometry of this correlogram, as (incident, analyzer):
    # 'VV' (polarized), 'VH' (depolarized), 'VU' (no analyzer), or None if not
    # recorded. The depolarized dynamic analysis pairs a VV with a VH correlogram
    # (same sample identity + angle) to extract the rotational diffusion coefficient
    # D_r = (Gamma_VH - Gamma_VV)/6. None leaves all existing (polarized) DLS
    # workflows unchanged.
    analyzer_geometry: Optional[str] = None
    # Free-text molecular-weight fraction label (e.g. "250k", "1M"). A WITHIN-sample
    # axis used to partition a Mw series into independent analyses; NOT part of the
    # SampleKey. None = unfractioned (the common single-sample case).
    mw_fraction: Optional[str] = None
    sample_label: Optional[str] = None
    instrument_name: Optional[str] = None
    source_file: Optional[str] = None
    # --- the fit's instrument inputs (the class docstring's "Optional" section) ---
    # Per-channel sample time (s), parser-written when the file states its sampling
    # structure; None means "derive a stand-in from the axis and say so". Kept in
    # `LoadedMeasurement.raw` beside the lags, never in the editable parameters.
    sample_times_s: Optional[np.ndarray] = None
    # Mean count rate (cps) from the COMMITTED parameter via `LoadedMeasurement.build()`
    # -- the one sanctioned path -- so every fit on this object reads the same rate.
    mean_count_rate_cps: Optional[float] = None
    # Run duration T (s), parser-written when the file states it for this correlogram;
    # None means "the weight's T -> infinity form, and say so". Kept in
    # `LoadedMeasurement.raw` beside the sample times, never in the editable parameters.
    duration_s: Optional[float] = None

    def __post_init__(self) -> None:
        self.delay_times_s = _as_1d_float_array(self.delay_times_s, "delay_times_s")
        self.correlogram = _as_1d_float_array(self.correlogram, "correlogram")
        _require_same_length(
            self.delay_times_s, self.correlogram, "delay_times_s", "correlogram"
        )
        _require_finite_array(self.delay_times_s, "delay_times_s")
        _require_finite_array(self.correlogram, "correlogram")
        _require_increasing_lag_axis(self.delay_times_s, "delay_times_s")
        _require_non_empty_string(self.polymer_name, "polymer_name")
        _require_non_empty_string(self.solvent_name, "solvent_name")
        _require_positive(self.temperature_K, "temperature_K")
        _require_positive(self.wavelength_nm, "wavelength_nm")
        _require_positive(self.solvent_refractive_index, "solvent_refractive_index")
        _require_scattering_angle(self.angle_deg, "angle_deg")
        if self.concentration_g_per_mL is not None:
            _require_non_negative(
                self.concentration_g_per_mL, "concentration_g_per_mL")
        if self.viscosity_Pa_s is not None:
            _require_positive(self.viscosity_Pa_s, "viscosity_Pa_s")
        if self.solute_refractive_index is not None:
            _require_positive(self.solute_refractive_index, "solute_refractive_index")
        self.analyzer_geometry = _normalize_analyzer_geometry(
            self.analyzer_geometry, "analyzer_geometry")
        if self.sample_times_s is not None:
            # Validated like the axis it pairs with: one finite, strictly positive
            # sample time per channel.
            self.sample_times_s = _as_1d_float_array(self.sample_times_s, "sample_times_s")
            _require_same_length(
                self.delay_times_s, self.sample_times_s, "delay_times_s", "sample_times_s")
            _require_finite_array(self.sample_times_s, "sample_times_s")
            _require_positive_array(self.sample_times_s, "sample_times_s")
        if self.mean_count_rate_cps is not None:
            rate = usable_rate(self.mean_count_rate_cps)
            if rate is None:
                raise ValueError(
                    f"mean_count_rate_cps must be a finite positive photon count rate in "
                    f"counts per second, got {self.mean_count_rate_cps!r}; pass None for "
                    f"the classical infinite-count-rate limit rather than a placeholder.")
            self.mean_count_rate_cps = rate
        if self.duration_s is not None:
            # Refused rather than dropped: a duration the file states but Eq. (9) cannot
            # use is an error in the file or the caller, never a silent T -> infinity.
            duration = float(self.duration_s)
            if not (math.isfinite(duration) and duration > 0.0):
                raise ValueError(
                    f"duration_s must be a finite positive run duration in seconds, got "
                    f"{self.duration_s!r}; pass None when the file states none.")
            self.duration_s = duration
        _warn_if_implausible_metadata(
            self.temperature_K, self.solvent_refractive_index)

    @property
    def sample_key(self) -> SampleKey:
        """The identity of this measurement, derived from its identity fields.

        Computed on demand rather than stored, so it can never drift out of
        sync with polymer_name / solvent_name / concentration / temperature.
        """
        return make_sample_key(
            self.polymer_name, self.solvent_name,
            self.concentration_g_per_mL, self.temperature_K,
        )

    def __repr__(self) -> str:
        c = self.concentration_g_per_mL
        c_str = 'n/a' if c is None else f'{c:g} g/mL'
        return (
            f"DLSMeasurement(polymer={self.polymer_name!r}, "
            f"solvent={self.solvent_name!r}, "
            f"c={c_str}, "
            f"T={self.temperature_K:g} K, "
            f"angle={self.angle_deg:g} deg, "
            f"n_points={self.delay_times_s.size})"
        )


# ---------------------------------------------------------------------------
# SLSMeasurement  (one angular series at one concentration)
# ---------------------------------------------------------------------------

@dataclass(eq=False, repr=False)
class SLSMeasurement:
    """One static light scattering angular series at a single concentration.

    One object = one concentration, measured across one or more angles. A full
    Zimm-plot dataset is a list of these (one per concentration); the pure
    solvent reference is the object whose concentration is 0. A single-angle
    measurement is just an angular series of length one -- no special case
    anywhere in the analysis code.

    Required data
    -------------
    angles_deg, intensities
        Equal-length 1-D arrays: scattering angles and measured intensity at
        each. Intensities are stored as raw instrument values; conversion to
        excess Rayleigh ratios happens in the SLS analysis module, which needs
        the solvent reference and calibration standard to do so correctly.

    Required identity
    -----------------
    polymer_name, solvent_name, concentration_g_per_mL, temperature_K

    Required optics
    ---------------
    wavelength_nm, solvent_refractive_index

    Optional optics
    ---------------
    dn_dc_mL_per_g -- may be absent (None). It enters ONLY the optical constant
    K, which multiplies the whole Kc/dR ordinate and therefore cancels in every
    coefficient ratio: every Rg, the form-factor radii and axial ratio, and the
    calibration-free 2*A2*Mw are unchanged by it. An absent dn/dc is the same
    kind of absence as an absent calibration constant -- an unknown factor on the
    ordinate -- so it is handled the same way: K = 1 (an arbitrary optical scale),
    the result carries dn_dc_supplied = False and a note, and the outputs that DO
    ride on the scale (absolute Mw, absolute A2) are flagged unreliable. This is
    not a proposed value (invariant 3): nothing is guessed, and the substitution
    travels with the result that discloses it.

    A SUPPLIED value must be finite and NONZERO. Zero is refused because K is
    proportional to (dn/dc)^2, so it collapses the entire Kc/dR ordinate; the
    form-factor fits then normalised that degenerate ordinate and returned a
    fabricated Rg and Mw flagged reliable. Blank is the honest entry for a user who
    has no value to give, which is why zero is refused rather than accommodated
    (a typed zero is a claim, not an absence). Negative increments stay legal:
    the sign cancels in the square, and a solute less refractive than its solvent
    genuinely has one.

    Optional calibration reference  (informational; NOT used in analysis)
    ----------------------------------------------------------------------
    The Rayleigh ratio used in analysis is always the program-computed,
    temperature-corrected toluene value from physics/constants.py (Takahashi
    et al. 2019 at 532 nm; Sivokhin & Kazantsev 2021 at 660 nm), not the value
    embedded in the instrument file. These fields record what the file reported
    for traceability and comparison.
    """

    # --- measured data ---
    angles_deg: np.ndarray
    intensities: np.ndarray

    # --- sample identity ---
    polymer_name: str
    solvent_name: str
    concentration_g_per_mL: float
    temperature_K: float

    # --- optics ---
    wavelength_nm: float
    solvent_refractive_index: float
    # Optional: absent (None) is legal and disclosed, NOT defaulted -- see the class
    # docstring's "Optional optics". This is the last field without a default, so
    # giving it one needs no reordering; every field below already has one.
    dn_dc_mL_per_g: Optional[float] = None

    # --- optional ---
    # Solute (particle) refractive index, hand-entered per sample. Consumed ONLY by the
    # full-Mie form-factor P(q) fit, which forms m = n_solute/solvent_refractive_index -- the
    # contrast entering the Mie scattering arithmetic. Treated like dn/dc: NEVER proposed,
    # defaulted, or guessed (invariant 3), being material-specific with no particle library.
    # Blank (None) is valid for every non-Mie SLS run. Mirrors
    # DLSMeasurement.solute_refractive_index; for the core-shell fit it is the CORE index.
    solute_refractive_index: Optional[float] = None

    # Shell (coating) refractive index for the core-shell form-factor P(q) fit
    # (analysis/sls.core_shell_form_factor_analysis), which forms m1 = n_core/solvent_RI and
    # m2 = n_shell/solvent_RI -- the two contrasts of the coated sphere. Here
    # solute_refractive_index is reused as the CORE index and this is the SHELL index. Treated
    # like solute_refractive_index / dn_dc: NEVER proposed, defaulted, or guessed (invariant 3)
    # -- material-specific, no particle library. Blank (None) is valid for every non-core-shell
    # SLS run (only the core-shell model consumes it).
    shell_refractive_index: Optional[float] = None

    # --- optional calibration reference (informational only) ---
    calibration_constant: Optional[float] = None
    standard_name: Optional[str] = None
    standard_rayleigh_ratio_file: Optional[float] = None
    standard_refractive_index: Optional[float] = None

    # --- optional polarization geometry ---
    # Analyzer/incident polarization of this intensity series, as (incident,
    # analyzer): 'VV' (polarized), 'VH' (depolarized), 'VU' (no analyzer), or None
    # if not recorded. Used by the depolarization analysis to pair a VV series with
    # a VH series (same sample identity + angle) for the Cabannes correction and the
    # depolarization ratio. None leaves all existing (polarized) workflows unchanged.
    analyzer_geometry: Optional[str] = None

    # --- optional provenance ---
    # Free-text molecular-weight fraction label (e.g. "250k", "1M"). A WITHIN-sample
    # axis that partitions a Mw series into independent Zimm fits; NOT part of the
    # SampleKey. The c = 0 solvent reference is shared across fractions.
    mw_fraction: Optional[str] = None
    sample_label: Optional[str] = None
    instrument_name: Optional[str] = None
    source_file: Optional[str] = None

    def __post_init__(self) -> None:
        self.angles_deg = _as_1d_float_array(self.angles_deg, "angles_deg")
        self.intensities = _as_1d_float_array(self.intensities, "intensities")
        _require_same_length(
            self.angles_deg, self.intensities, "angles_deg", "intensities"
        )
        # Reject nan/inf intensities, mirroring DLSMeasurement's correlogram guard.
        # (angles_deg is finite-checked as a side effect of the per-angle range loop
        # below; intensities has no such check, so guard it explicitly -- rule-10.)
        _require_finite_array(self.intensities, "intensities")
        for angle in self.angles_deg:
            _require_scattering_angle(angle, "each value in angles_deg")
        _require_non_empty_string(self.polymer_name, "polymer_name")
        _require_non_empty_string(self.solvent_name, "solvent_name")
        _require_positive(self.temperature_K, "temperature_K")
        _require_positive(self.wavelength_nm, "wavelength_nm")
        _require_positive(self.solvent_refractive_index, "solvent_refractive_index")
        _require_non_negative(self.concentration_g_per_mL, "concentration_g_per_mL")
        # dn/dc: ABSENT (None) is legal and disclosed -- see the class docstring's
        # "Optional optics" for why, and what the analyses then report. What is refused
        # is a SUPPLIED value that cannot be a real increment: non-finite, or exactly
        # zero (K is proportional to (dn/dc)^2, so zero collapses the whole Kc/dR
        # ordinate and no molecular weight can come from it). NEGATIVE stays legal --
        # the sign cancels in the square, and a solute less refractive than its solvent
        # genuinely has one -- so the predicate is NONZERO, never positive. Sizing it to
        # positivity would refuse real samples.
        if self.dn_dc_mL_per_g is not None:
            if not math.isfinite(self.dn_dc_mL_per_g):
                raise ValueError(
                    f"dn_dc_mL_per_g must be a finite number or left blank, "
                    f"got {self.dn_dc_mL_per_g!r}."
                )
            if self.dn_dc_mL_per_g == 0:
                raise ValueError(
                    "dn_dc_mL_per_g must not be zero: the optical constant K is "
                    "proportional to (dn/dc)^2, so zero collapses the whole Kc/dR "
                    "ordinate and no molecular weight can come from it. Leave dn/dc "
                    "blank instead -- Rg, the form-factor sizes and the "
                    "calibration-free 2*A2*Mw are all still computed, and Mw/A2 are "
                    "reported on a disclosed arbitrary scale."
                )
        if self.solute_refractive_index is not None:
            _require_positive(self.solute_refractive_index, "solute_refractive_index")
        if self.shell_refractive_index is not None:
            _require_positive(self.shell_refractive_index, "shell_refractive_index")
        # The vendor calibration record, as read from the instrument file. These are
        # DISPLAY-ONLY: the program computes its own k_c from a calibrant point, and the
        # value the SLS engine uses arrives as a function argument from the calibration
        # panel, not from this field. So the property required here is FINITENESS, not
        # positivity -- a stale or uncalibrated instrument legitimately writes 0, and
        # rejecting that would make the whole file unloadable over a number nothing reads.
        # Guarded at all because an unvalidated physical scalar is a hole; the predicate is
        # matched to the property, since over-inclusion is safe in COVERAGE, not strictness.
        for _name in ("calibration_constant", "standard_rayleigh_ratio_file",
                      "standard_refractive_index"):
            _require_finite_or_none(getattr(self, _name), _name)
        self.analyzer_geometry = _normalize_analyzer_geometry(
            self.analyzer_geometry, "analyzer_geometry"
        )
        _warn_if_implausible_metadata(
            self.temperature_K, self.solvent_refractive_index)

    @property
    def sample_key(self) -> SampleKey:
        return make_sample_key(
            self.polymer_name, self.solvent_name,
            self.concentration_g_per_mL, self.temperature_K,
        )

    def __repr__(self) -> str:
        return (
            f"SLSMeasurement(polymer={self.polymer_name!r}, "
            f"solvent={self.solvent_name!r}, "
            f"c={self.concentration_g_per_mL:g} g/mL, "
            f"T={self.temperature_K:g} K, "
            f"n_angles={self.angles_deg.size})"
        )
