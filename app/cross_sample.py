"""
app/cross_sample.py
===================

The data shapes the controller hands the Cross-Sample tab: one sample's
rho = Rg/Rh with its provenance (`SampleRho`), a scaling plot's points and power-law fit
(`ScalingData`), and the branching readout's rows and table (`ArchitectureRow`,
`ArchitectureData`). The controller imports and builds them. Qt-free (invariant 5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from core.notes import Note
from analysis import utilities as util
from analysis.utilities import ScalingResult


@dataclass
class SampleRho:
    """The shape parameter rho = Rg/Rh for one sample, with full provenance.

    Bundles the underlying RhoResult value with the plain-language source labels
    and the apparent/thermodynamic status of each input. `is_apparent` is True if
    EITHER Rg or Rh is apparent -- rho is only a thermodynamic shape parameter
    when both inputs are infinite-dilution values (Rg from a Zimm/Berry
    extrapolation, Rh from a c->0 extrapolation).
    """
    sample_id: str
    rho: float
    rg_nm: float
    rh_nm: float
    rg_label: str
    rh_label: str
    rg_source: str               # 'computed' | 'picked' | 'user'
    rh_source: str               # 'computed' | 'picked' | 'user'
    is_apparent: bool
    interpretation: str
    shape: str = ''              # concise architecture label (e.g. 'random coil')
    rho_se: Optional[float] = None  # statistical SE (only when both Rg and Rh have one)
    se_estimator: Optional[str] = None  # covariance estimator behind rho_se (Rg/Rh
    #                                     regression SEs); None when rho_se is None
    rg_se: Optional[float] = None   # regression SE of Rg (None if Rg has no honest ±),
    rh_se: Optional[float] = None   # and of Rh — carried so the display can round each
    #                                 value to the place its own σ supports (invariant 8)
    rg_model_dependent: bool = False  # True = Rg came from a form-factor P(q) fit (assumes a
    #                                   shape) -> the display flags it (neutral ⓘ); orthogonal to
    #                                   is_apparent. Rh is always model-agnostic (DLS).
    rg_model: str = ''              # the assumed shape when rg_model_dependent (e.g. 'sphere')
    rg_degenerate: bool = False     # True = Rg rests on a user-selected DEGENERATE form-factor
    #                                 solution (the data can't distinguish the shape); always implies
    #                                 rg_model_dependent, disclosed one tier stronger (review-01; rule 10)
    # The underlying RhoResult's notes (`core.notes.Note`; e.g. the Rg and Rh sample keys
    # disagree). Structurally empty in the cross-sample path (Rg and Rh come from ONE
    # SampleResult, so no keys are compared) — carried for a uniform surface + future
    # cross-sample pairing. The same fact is also emitted via warnings.warn (keep-both).
    notes: Tuple[Note, ...] = ()


@dataclass
class ScalingData:
    """Points + power-law fit for a cross-sample scaling plot (Rg-Mw or A2-Mw).

    One point per included sample: its effective Mw (a user Mw wins) against Rg or
    A2. The fit is a log-log regression -> the exponent (Rg ~ Mw^nu, A2 ~ Mw^-a).
    Parallel lists (sample_ids/labels/mw/y) are the plotted points, in order.
    """
    quantity: str                    # 'rg' or 'a2'
    fit: ScalingResult
    sample_ids: List[str]
    labels: List[str]
    mw: List[float]
    y: List[float]
    # A plotted VALUE is on an arbitrary scale: the Mw (every plot's X axis) or, on the a2
    # axis, the A2 itself -- the one definition lives in `_result_arbitrary_scale`. Named for
    # the CONDITION, not for Mw: unlike its ArchitectureData namesake below, this flag has two
    # possible sources, so the two dataclasses deliberately STOP mirroring here.
    any_arbitrary_scale: bool
    n_excluded: int                  # samples dropped (no Mw, or non-positive y)
    # Branched-tagged points: plotted (so the contraction is visible) but held OUT of
    # the linear power-law fit (a branched chain would bias the linear scaling law).
    # Parallel lists, same shape as the fitted points above.
    branched_mw: List[float] = field(default_factory=list)
    branched_y: List[float] = field(default_factory=list)
    branched_labels: List[str] = field(default_factory=list)
    # A plotted point drew a MODEL-DEPENDENT source: a form-factor Mw (any quantity) or, for
    # the rg axis, a form-factor Rg. The exponent then rests on a shape assumption -> flagged
    # (neutral) in the scaling note + export, mirroring any_arbitrary_scale (honesty axis).
    any_model_dependent: bool = False
    # A plotted point drew a DEGENERATE form-factor source (a user-selected solution the data can't
    # distinguish) -> the exponent rests on an ambiguity the user resolved, flagged one tier stronger
    # than model_dependent alone (review-01; rule-10 sibling of any_model_dependent).
    any_degenerate: bool = False
    # Per-point provenance, parallel to the lists above: the plain-language label of the Mw
    # and of the plotted y value, as `_scaling_point_source` composes them. Carried so the
    # scaling EXPORT can name where each point came from, the `export_architecture` Rg
    # source / Rh source pattern applied to the axes this file actually plots -- which is
    # what makes an apparent single-angle-ladder Mw or A2 name its angle in the file rather
    # than only on screen.
    sources: List[str] = field(default_factory=list)
    branched_sources: List[str] = field(default_factory=list)
    # The scale gates of every plotted point whose value `any_arbitrary_scale` judges
    # arbitrary, relayed from its stored sample result (`SampleResult.scale_notes`) with the
    # points named: about `mw` for an Mw, about `y` for an A2 on the a2 axis.
    notes: Tuple[Note, ...] = ()


@dataclass
class ArchitectureRow:
    """One branched (sample, Mw) architecture readout: the scattering factor g (from
    Rg, if the sample carries one) and/or the hydrodynamic factor h (from Rh). Either
    may be None when that radius is absent -- g needs Rg, h needs Rh."""
    sample_id: str
    label: str
    mw: float
    g_result: Optional['util.ArchitectureResult'] = None
    h_result: Optional['util.HydrodynamicResult'] = None
    # The g-vs-h reading (util.compare_g_h): an interpretation of the two factors, not a note.
    g_h_comparison: str = ''


@dataclass
class ArchitectureData:
    """Branching contraction factors g (Rg) and h (Rh) over the tagged samples + the
    linear reference laws.

    `rg_law` / `rh_law` are the linear-only Rg-Mw and Rh-Mw power laws fit to the
    samples tagged Linear (chain_type; route 2), each independent (a sample can feed
    one and not the other). `rows` is one ArchitectureRow per branched (sample, Mw).
    `message` is a non-empty guidance string when nothing is computable yet (too few
    linear points, or no branched sample tagged).
    """
    rg_law: 'ScalingResult'
    rh_law: 'ScalingResult'
    rg_mw_range: Optional[tuple]     # (min, max) Mw of the linear Rg series; None if no fit
    rh_mw_range: Optional[tuple]     # (min, max) Mw of the linear Rh series; None if no fit
    n_linear_rg: int                 # linear Rg points fitted
    n_linear_rh: int                 # linear Rh points fitted
    rows: List['ArchitectureRow']    # per branched (sample, Mw)
    # Mw-specific by nature, and deliberately NOT renamed alongside ScalingData's
    # `any_arbitrary_scale`: the g/h factors and the reference laws have no A2 axis, so Mw is
    # the only scale-bearing input here and naming it is the more precise caveat.
    any_uncalibrated_mw: bool = False  # a contributing Mw came from an uncalibrated run
    # A contributing Mw (or, for the Rg law/g factor, an Rg) came from a form-factor P(q) fit,
    # so a g/h factor or reference law rests on a particle-shape assumption -> flagged (neutral)
    # in the architecture export, mirroring any_uncalibrated_mw (model-dependence axis).
    any_model_dependent: bool = False
    # A contributing Rg/Mw rests on a user-selected DEGENERATE form-factor solution -> the g/h factor
    # or reference law rests on an ambiguity the user resolved (review-01; rule-10 sibling).
    any_degenerate: bool = False
    message: str = ''
    # The scale gates of every contributing Mw `any_uncalibrated_mw` judges arbitrary,
    # relayed from its stored sample result (`SampleResult.scale_notes`) with the points
    # named, about the whole readout (the laws and every g/h rest on those Mw).
    notes: Tuple[Note, ...] = ()

    def row_for(self, sample_id: str, mw: Optional[float]) -> Optional['ArchitectureRow']:
        """The ArchitectureRow matching (sample_id, Mw), or None -- lets the GUI join
        g/h onto its per-(sample, fraction) ρ rows. O(1) via a lazily-built index (the
        GUI calls this once per table row, so a linear scan would be quadratic)."""
        if mw is None or not np.isfinite(mw):
            return None
        idx = self.__dict__.get('_row_index')
        if idx is None:
            idx = {(r.sample_id, round(r.mw, 6)): r for r in self.rows}
            self.__dict__['_row_index'] = idx
        return idx.get((sample_id, round(float(mw), 6)))
