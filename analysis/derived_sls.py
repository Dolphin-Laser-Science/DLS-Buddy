"""
analysis/derived_sls.py
=======================

Turn a set of DLS runs' mean count rates into ONE static (SLS) angular series.

This is the engine half of "trace-derived static points": a DLS run's average
scattered intensity is the static observable, so a goniometer that recorded only
correlograms can still yield a full static analysis. Sedlak and Amis (1992,
J. Chem. Phys. 96:817) did exactly this on a Brookhaven BI-200SM -- "Measurements
of integrated intensities (static light scattering) were done using the same setup
by counting the total number of counts per a defined time interval" (p. 819) --
so the practice is standard, not a house invention.

What this module does NOT do
----------------------------
It never reads a trace, a workspace or a file, and it never decides WHICH mean
count rate a run carries. It is handed the number the user committed, together
with the tag saying where that number came from ('trace:mean',
'trace:sigma_clipped', 'vendor', 'user'), and it collates. Choosing the source is
the user's, through the Data tab (invariant 3); recording the choice is this
module's, because a derived number must travel with its provenance.

Two different combination questions, two different rules
--------------------------------------------------------
The combination doctrine (core/data_models.py) governs both, differently:

* Runs at the SAME angle are AVERAGED into one number. That is a combination of
  the strict kind, so it goes through the shared
  :func:`analysis.dls.replicate.check_replicate_identity` -- the one answer to
  "are these the same measurement, repeated?" -- rather than a second copy of an
  identity check here (doctrine rule 1's backstop).
* Runs at DIFFERENT angles are COLLATED into a series. Nothing is averaged, so a
  disagreement in an inherited optical constant is a disclosure, not a refusal:
  it becomes a note and the first run's value is used (doctrine rule 2).

Two refusals, and why each meets rule 1's criterion -- the combined object has no
referent -- rather than merely being degraded:

* A mixed POLARIZATION state, which is one of rule 1's named partition tags. VV
  and VH are different measurements; a series spanning both describes neither.
* A mixed CONCENTRATION, which is not a partition tag and does not need to be:
  the tags are examples and the criterion governs. An ``SLSMeasurement`` holds one
  ``concentration_g_per_mL``, so collating two into it writes a false record, not a
  worse number. Note the direction carefully -- a Zimm SPANS concentrations by
  construction and a sample holds a whole ladder; what is refused is one object
  claiming two. The caller groups by concentration, so this is a backstop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.data_models import (DLSMeasurement, casefold_identity_name,
                              polarization_state, round_angle_for_identity,
                              round_concentration_for_identity,
                              round_temperature_for_identity)
from analysis.dls.replicate import check_replicate_identity

__all__ = ['DerivedIntensityInput', 'DerivedPoint', 'DerivedStaticSeries',
           'derive_static_series', 'require_one_polarization_state']


@dataclass(frozen=True)
class DerivedIntensityInput:
    """One DLS run offered as a static point.

    ``mean_count_rate_cps`` is the value COMMITTED on that run, and ``source`` is
    its committed provenance tag. Both are passed in rather than derived here so
    that this module cannot disagree with the Data tab about either.
    """
    item_id: str
    measurement: DLSMeasurement
    mean_count_rate_cps: float
    source: str


@dataclass(frozen=True)
class DerivedPoint:
    """One angle of the derived series, with the runs that produced it."""
    angle_deg: float
    intensity_cps: float
    source_item_ids: Tuple[str, ...]
    source_tags: Tuple[str, ...]

    @property
    def is_replicate_mean(self) -> bool:
        """True when more than one run was averaged into this point."""
        return len(self.source_item_ids) > 1


@dataclass(frozen=True)
class DerivedStaticSeries:
    """A static angular series assembled from DLS runs' mean count rates.

    ``intensities`` are count rates in cps. That is a legal SLS intensity unit:
    the pipeline needs only that every intensity -- sample, solvent and calibrant
    -- share one unit, because the calibration constant k_c = R_standard /
    I_calibrant cancels it exactly. Nothing downstream can CHECK that the other
    intensities share it, so ``notes`` always carries one sentence naming the unit
    and what a calibrant entered in kcps would cost.
    """
    angles_deg: np.ndarray
    intensities: np.ndarray
    concentration_g_per_mL: Optional[float]
    polymer_name: str
    solvent_name: str
    temperature_K: float
    wavelength_nm: float
    solvent_refractive_index: float
    analyzer_geometry: Optional[str]
    mw_fraction: Optional[str]
    points: Tuple[DerivedPoint, ...] = ()
    # Defaults to the empty tuple like every other result dataclass carrying notes,
    # so the warnings bridge can read the family uniformly (tests/test_notes_bridge.py).
    notes: Tuple[str, ...] = ()

    @property
    def source_item_ids(self) -> Tuple[str, ...]:
        """Every run that contributed, in angle order."""
        return tuple(i for p in self.points for i in p.source_item_ids)

    @property
    def source_tags(self) -> Tuple[str, ...]:
        """The distinct provenance tags across every contributing run."""
        seen: List[str] = []
        for point in self.points:
            for tag in point.source_tags:
                if tag not in seen:
                    seen.append(tag)
        return tuple(seen)


def require_one_polarization_state(measurements) -> None:
    """Raise unless every measurement shares one polarization state.

    Doctrine rule 1 asks for TWO things: the partition is "enforced structurally --
    a mixed set is never eligible -- with the engine refusal as backstop". This is
    the one predicate behind both, exported so the caller can run it over the whole
    SELECTION before it groups. That matters: grouping by concentration first makes
    every group internally uniform, so a per-group check can never see a selection
    that mixes VV at one concentration with VH at another -- and those derive into
    two measurements that pool into one Zimm.

    Comparison is through the shared `polarization_state`, so this site cannot
    classify a label differently from the fits, the Rh tiers or the DDLS pairing.
    """
    items = list(measurements)
    states = {polarization_state(m.analyzer_geometry) for m in items}
    if len(states) > 1:
        labels = sorted({str(m.analyzer_geometry) for m in items})
        raise ValueError(
            "These runs do not share one polarization state (analyzer geometry: "
            "{0}). VV and VH are different measurements, so a single static "
            "series cannot span them, and neither can a set of series analyzed "
            "together. Derive each geometry separately.".format(
                ', '.join(labels))
        )


def _require_usable_rate(entry: DerivedIntensityInput) -> float:
    """The entry's mean count rate as a usable float, or raise.

    Checked on the VALUE rather than on whatever produced it: a photon rate can be
    neither negative nor non-finite, and both float tails arrive here the same way
    (a saturated product reads inf, an underflowed one reads 0.0 -- the second is
    representable but cannot carry a scattering series, so it is refused too).
    """
    value = float(entry.mean_count_rate_cps)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(
            "Run {0!r} has a mean count rate of {1!r} cps, which cannot be a "
            "scattered intensity. Commit a positive mean count rate on the Data "
            "tab, or leave this run out of the derivation.".format(
                entry.item_id, value)
        )
    return value


def derive_static_series(
    entries: Sequence[DerivedIntensityInput],
    *, rtol: float = 1e-6,
) -> DerivedStaticSeries:
    """Collate DLS runs' committed mean count rates into one static series.

    Parameters
    ----------
    entries
        The runs to derive from, in any order. Runs sharing an angle are averaged
        through the shared replicate-identity check; the result is sorted by angle.
    rtol
        Relative tolerance handed to :func:`check_replicate_identity` for the
        same-angle averaging. Documented and overridable rather than a magic
        number (invariant 4).

    Raises
    ------
    ValueError
        On an empty set, a mixed polarization state, a mixed concentration, or a
        mean count rate that cannot be an intensity. See the module docstring for
        why each of these is a refusal rather than a note.
    """
    if not entries:
        raise ValueError('No runs to derive a static series from.')

    entries = list(entries)
    rates = {e.item_id: _require_usable_rate(e) for e in entries}

    # --- refusal 1: the polarization partition (doctrine rule 1) ---
    require_one_polarization_state(e.measurement for e in entries)

    # --- refusal 2: one concentration per SLS measurement ---
    concs = {(None if e.measurement.concentration_g_per_mL is None
              else round_concentration_for_identity(
                  e.measurement.concentration_g_per_mL))
             for e in entries}
    if len(concs) > 1:
        shown = sorted(str(c) for c in concs)
        raise ValueError(
            "These runs carry more than one concentration ({0} g/mL). A static "
            "measurement holds exactly one, so derive each concentration "
            "separately.".format(', '.join(shown))
        )

    first = entries[0].measurement
    notes: List[str] = []

    # --- disclosure: inherited identity and optical constants (doctrine rule 2) ---
    # Nothing here is averaged across angles, so a disagreement is reported and the
    # first run's value is used -- the same "garbage in, garbage out, and the
    # parameters are the user's" stance the replicate average takes for the fields
    # it deliberately does not compare.
    for label, getter, form in (
        ('polymer name', lambda m: m.polymer_name, casefold_identity_name),
        ('solvent name', lambda m: m.solvent_name, casefold_identity_name),
        ('temperature', lambda m: m.temperature_K, round_temperature_for_identity),
        ('wavelength', lambda m: m.wavelength_nm, lambda v: round(float(v), 6)),
        ('solvent refractive index',
         lambda m: m.solvent_refractive_index, lambda v: round(float(v), 9)),
        ('molecular-weight fraction', lambda m: m.mw_fraction, lambda v: v),
        ('analyzer geometry', lambda m: m.analyzer_geometry, lambda v: v),
    ):
        seen = {form(getter(e.measurement)) if getter(e.measurement) is not None
                else None for e in entries}
        if len(seen) > 1:
            notes.append(
                "The collated runs do not all share one {0}; the derived series "
                "uses the first run's value ({1!r}). Correct the run parameters "
                "if that is not what you meant.".format(label, getter(first))
            )

    # --- collate: one point per angle bucket ---
    by_angle: Dict[float, List[DerivedIntensityInput]] = {}
    for entry in entries:
        by_angle.setdefault(
            round_angle_for_identity(entry.measurement.angle_deg), []).append(entry)

    points: List[DerivedPoint] = []
    for angle in sorted(by_angle):
        group = by_angle[angle]
        if len(group) > 1:
            # Averaging, not collating -- so the strict shared check governs, and a
            # non-replicate set raises rather than being silently meaned. Recording
            # every member id is what keeps this order-independent: a first-wins
            # rule would make the result depend on load order -- the same defect
            # shape a first-loaded solvent blank has been measured to produce
            # elsewhere in the program.
            check_replicate_identity([e.measurement for e in group], rtol=rtol)
            notes.append(
                '{0} runs at {1:g} deg were averaged into one point.'.format(
                    len(group), angle))
        intensity = float(np.mean([rates[e.item_id] for e in group]))
        points.append(DerivedPoint(
            angle_deg=float(angle),
            intensity_cps=intensity,
            source_item_ids=tuple(e.item_id for e in group),
            source_tags=tuple(e.source for e in group),
        ))

    # --- disclosure: the unit, on every series (doctrine rule 2) ---
    # A file-loaded SLS series keeps its own file's unit, and one session calibrant
    # intensity is applied to it and to this one alike, so a unit mismatch is
    # invisible to the engine -- it can only be stated.
    notes.append(
        'Derived intensities are count rates in cps (a kcps or kHz reading was '
        'converted on load). k_c cancels the unit only when the calibrant '
        'intensity and every SLS file in the same session are in cps too; a '
        'calibrant entered in kcps would scale Mw by 1000.'
    )

    return DerivedStaticSeries(
        angles_deg=np.array([p.angle_deg for p in points], dtype=float),
        intensities=np.array([p.intensity_cps for p in points], dtype=float),
        concentration_g_per_mL=first.concentration_g_per_mL,
        polymer_name=first.polymer_name,
        solvent_name=first.solvent_name,
        temperature_K=first.temperature_K,
        wavelength_nm=first.wavelength_nm,
        solvent_refractive_index=first.solvent_refractive_index,
        analyzer_geometry=first.analyzer_geometry,
        mw_fraction=first.mw_fraction,
        points=tuple(points),
        notes=tuple(notes),
    )
