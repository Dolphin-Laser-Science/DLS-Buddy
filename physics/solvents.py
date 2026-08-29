"""Solvent physical-property library: refractive index n(lambda, T) and dynamic
viscosity eta(T) from a chosen solvent, temperature, and wavelength.

Pure module -- no Qt, no file I/O, no plotting, no input mutation. It mirrors the
``physics/constants.py`` pattern (``rayleigh_ratio_toluene``): a module-level,
source-provenanced data table plus pure evaluator functions that **raise on
out-of-range input rather than extrapolating**.

Role in the platform (architecture invariant #3): this library only *offers*
convenience defaults. The values it returns become part of an analysis solely
through the user's explicit commit, are provenance-tagged, and are always
overridable. No solvent constant is ever baked into an analysis function, and
**dn/dc is never proposed here** (the central low-contrast vulnerability stays
strictly hand-entered).

Unified forms (both fit offline from the primary literature by the
maintainer's coefficient-generator script ``gen_solvent_coeffs.py`` -- see
that script for the source-by-source correlations and the fit procedure):

    refractive index   n(lambda, T) = A + B/lam_um^2 + C/lam_um^4
                                        + dn_dT * (T - T_ref)          (lam in um)
    viscosity          eta(T)       = A_vft * exp(B_vft / (T - T0))    (Pa s)

Confidence tiers (``'primary'`` vs ``'estimate'``) are labels only: the runtime
evaluates both identically. The tier gates which front-end offers the solvent and
drives display badges; it never changes the math.

A display-only uncertainty accompanies each property. Per architecture invariant
#8 it is **not** propagated into any analysis standard error -- n/eta uncertainty
is exactly the calibration-class systematic that the reported regression SE
excludes. It is surfaced for the user's judgement, not carried into a +/-.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from core.data_models import casefold_identity_name


# ===========================================================================
# Record
# ===========================================================================

@dataclass(frozen=True)
class SolventProps:
    """Sourced coefficient record for one solvent, keyed by canonical name.

    The refractive-index block is always present; the viscosity block is optional
    (a record may carry n but no eta), signaled by ``eta_vft_A_Pa_s is None``.
    Per-solvent source provenance lives in the code comment beside each row in
    ``_SOLVENT_PROPS`` (mirroring ``constants.py:_TOLUENE_RVV_25C``), with one
    machine-readable exception: ``eta_source_citekey`` pins the viscosity
    source so tests can assert against it (see the field's comment). No
    citation ever reaches the UI.
    """
    tier: str                         # 'primary' | 'estimate'

    # ---- refractive index: Cauchy in lambda (um) + linear dn/dT ----
    n_cauchy_A: float
    n_cauchy_B_um2: float
    n_cauchy_C_um4: float
    dn_dT_per_K: float
    n_ref_temp_K: float
    n_lambda_min_nm: float
    n_lambda_max_nm: float
    n_temp_min_K: float
    n_temp_max_K: float
    n_uncertainty: float              # absolute, display-only
    n_fit_residual: float             # max |unified - source| over the box
    # Grade of the dn/dT slope only (the dispersion shape is always primary):
    # 'measured' -> dn/dT from the same primary source; 'bulk' -> the primary
    # dispersion is single-temperature, so dn/dT is taken from a bulk/handbook
    # compilation to reach the standard-lab box, and carries a visible flag.
    n_dndt_grade: str = 'measured'    # 'measured' | 'bulk'

    # ---- viscosity: Vogel-Fulcher-Tammann (optional block) ----
    eta_vft_A_Pa_s: Optional[float] = None
    eta_vft_B_K: Optional[float] = None
    eta_vft_T0_K: Optional[float] = None
    eta_temp_min_K: Optional[float] = None
    eta_temp_max_K: Optional[float] = None
    eta_uncertainty_rel: Optional[float] = None   # relative, display-only
    eta_fit_residual_rel: Optional[float] = None  # max relative fit residual
    # Confidence of the viscosity source, best -> weakest:
    #   'reference' -- a critically-evaluated reference correlation (IUPAC/Assael);
    #   'measured'  -- a dedicated single-lab measurement (e.g. the pure-component
    #                  endpoint of a binary-mixture study);
    #   'bulk'      -- a handbook/bulk compilation (CRC, Riddick).
    eta_source_grade: Optional[str] = None
    # Citekey of the shipped viscosity source (e.g. 'santos2006') -- provenance
    # metadata ONLY: never a physics input, never shown in the UI. It exists so
    # the CoolProp oracle census can machine-pin OUR side of its independence
    # labels (tests/test_solvent_coolprop_oracle.py) -- a shipped-source swap
    # then fails loudly instead of silently degrading a label. Where one
    # record's fitted points genuinely span two compilations the citekey is the
    # '+'-joined pair (cs2/dcm: 'crc95+riddick1970'; every part must resolve to a
    # known citekey). Viscosity only: the RI side blends sources (dispersion vs
    # dn/dT) on several records, so a single RI citekey would misattribute; its
    # provenance stays in the row comment. None = no viscosity block.
    eta_source_citekey: Optional[str] = None

    # ---- per-condition uncertainty descriptor (display-only) ----
    # Components of sigma_n(lambda, T) and sigma_eta_rel(T) -- see the Advanced
    # Guide sec. 5 for the model. u_src = the source's stated accuracy floor;
    # the node tuples are piecewise-linear UPPER envelopes ((x, r), ...) of the
    # unified-fit residual (clamped flat beyond the end nodes); u_slope = the
    # documented dn/dT-slope allowance. The accessors additionally add the
    # module-level inter-laboratory floors (N_INTERLAB_FLOOR /
    # ETA_INTERLAB_FLOOR_REL below), which are not per-record fields. None ->
    # that component is absent (a record without a descriptor falls back to
    # the flat box-wide scalar).
    # Like every library uncertainty these are shown, NEVER propagated into an
    # analysis SE (invariant #8), and the box-wide scalars above are their
    # derived maxima over the validity box (rounded up to a display step).
    n_u_src: Optional[float] = None
    n_u_slope_per_K: Optional[float] = None
    n_r_lambda_nodes: Optional[tuple] = None    # (lambda_nm, abs residual)
    n_r_temp_nodes: Optional[tuple] = None      # (T_K, abs residual)
    eta_u_src_rel: Optional[float] = None
    eta_r_temp_nodes: Optional[tuple] = None    # (T_K, relative residual)

    @property
    def has_viscosity(self) -> bool:
        return self.eta_vft_A_Pa_s is not None


# ===========================================================================
# GENERATED coefficient table -- do not hand-edit.
# Produced by the maintainer's gen_solvent_coeffs.py from the primary
# literature; regenerate (never hand-edit) and re-confirm the anchors/residuals
# in the solvent regression tests after any source change. Per-solvent source
# provenance is the code comment on each row (plus the machine-readable
# eta_source_citekey field for the viscosity source).
# ===========================================================================

_SOLVENT_PROPS: Dict[str, SolventProps] = {
    # Water. RI: Harvey et al., JPCRD 27, 761 (1998) Eq. A1 (IAPWS R9-97)
    #   + IAPWS-95 density at 0.1 MPa (Wagner & Pruss 2002; CoolProp 8.0.0
    #   dev-time, validated on the printed check values), Cauchy + const
    #   dn/dT fit over 400-800 nm x 15-60 C; Daimon 2007 cross-check.
    #   700-800 nm truth carries Harvey grade 1e-3 (disclosed; floor covers).
    #   eta: Patek et al., JPCRD 38, 21 (2009) Eq. 7 (direct, 0.1 MPa).
    'water': SolventProps(
        tier='primary',
        n_cauchy_A=1.3233300072517413, n_cauchy_B_um2=0.003744328663041001, n_cauchy_C_um4=-8.46937987265691e-05,
        dn_dT_per_K=-0.0001387373145243162, n_ref_temp_K=293.15,
        n_lambda_min_nm=400.0, n_lambda_max_nm=800.0,
        n_temp_min_K=288.15, n_temp_max_K=333.15,
        n_uncertainty=0.0016, n_fit_residual=6.677e-04,
        eta_vft_A_Pa_s=3.208763e-05, eta_vft_B_K=480.6716, eta_vft_T0_K=153.5189,
        eta_temp_min_K=273.15, eta_temp_max_K=343.15,
        eta_uncertainty_rel=0.011, eta_fit_residual_rel=4.536e-03,
        eta_source_grade='reference', eta_source_citekey='patek2009',
        n_u_src=0.00015, n_u_slope_per_K=1.110e-05,
        n_r_lambda_nodes=(
            (400, 5.326e-04), (450, 5.355e-04), (500, 5.355e-04),
            (550, 4.819e-04), (600, 3.861e-04), (650, 3.914e-04),
            (700, 4.970e-04), (750, 6.678e-04), (800, 6.678e-04),
        ),
        n_r_temp_nodes=(
            (288.15, 6.678e-04), (293.775, 6.678e-04), (299.4, 4.359e-04),
            (305.025, 4.491e-04), (310.65, 4.491e-04), (316.275, 4.386e-04),
            (321.9, 3.431e-04), (327.525, 5.355e-04), (333.15, 5.355e-04),
        ),
        eta_u_src_rel=0.01,
        eta_r_temp_nodes=(
            (273.15, 4.537e-03), (281.9, 4.537e-03), (290.65, 2.155e-03),
            (299.4, 1.932e-03), (308.15, 1.696e-03), (316.9, 1.710e-03),
            (325.65, 1.710e-03), (334.4, 2.959e-03), (343.15, 2.959e-03),
        ),
    ),
    # Toluene. RI: Samoc, J. Appl. Phys. 94, 6167 (2003), Cauchy form b at
    #   20 C; dn/dT = -5.273e-4/K (Samoc, 632.8 nm). eta: Santos et al.,
    #   JPCRD 35, 1 (2006) Eq. 5 (direct, saturation line).
    'toluene': SolventProps(
        tier='primary',
        n_cauchy_A=1.4747749999999995, n_cauchy_B_um2=0.006990310000000583, n_cauchy_C_um4=0.00021775999999988985,
        dn_dT_per_K=-0.0005273, n_ref_temp_K=293.15,
        n_lambda_min_nm=405.0, n_lambda_max_nm=830.0,
        n_temp_min_K=288.15, n_temp_max_K=313.15,
        n_uncertainty=0.0014, n_fit_residual=8.882e-16,
        eta_vft_A_Pa_s=1.899410e-05, eta_vft_B_K=938.2867, eta_vft_T0_K=20.0568,
        eta_temp_min_K=260.0, eta_temp_max_K=370.0,
        eta_uncertainty_rel=0.01, eta_fit_residual_rel=8.765e-03,
        eta_source_grade='reference', eta_source_citekey='santos2006',
        n_u_src=0.0002, n_u_slope_per_K=2.637e-05,
        n_r_lambda_nodes=(
            (405, 1.111e-15), (458.125, 1.111e-15), (511.25, 4.441e-16),
            (564.375, 4.441e-16), (617.5, 4.441e-16), (670.625, 4.441e-16),
            (723.75, 4.441e-16), (776.875, 4.441e-16), (830, 4.441e-16),
        ),
        eta_u_src_rel=0.00909318,
        eta_r_temp_nodes=(
            (260, 3.817e-03), (273.75, 4.161e-03), (287.5, 4.161e-03),
            (301.25, 3.581e-03), (315, 2.888e-03), (328.75, 3.530e-03),
            (342.5, 3.530e-03), (356.25, 3.519e-03), (370, 2.554e-03),
        ),
    ),
    # Ethanol. RI: Moreels et al., Appl. Opt. 23, 3010 (1984), measured
    #   dispersion + dn/dT (20 & 25 C Pulfrich). eta: Sotiriadou et al.,
    #   Int. J. Thermophys. 44, 40 (2023) Eq. 13 (direct, 0.1 MPa).
    'ethanol': SolventProps(
        tier='primary',
        n_cauchy_A=1.3527024315476344, n_cauchy_B_um2=0.003044160497979229, n_cauchy_C_um4=1.671302471312284e-05,
        dn_dT_per_K=-0.00035120000000010275, n_ref_temp_K=293.15,
        n_lambda_min_nm=476.0, n_lambda_max_nm=633.0,
        n_temp_min_K=288.15, n_temp_max_K=313.15,
        n_uncertainty=0.0013, n_fit_residual=3.398e-05,
        eta_vft_A_Pa_s=2.178149e-07, eta_vft_B_K=3850.0657, eta_vft_T0_K=-154.1331,
        eta_temp_min_K=253.15, eta_temp_max_K=343.15,
        eta_uncertainty_rel=0.03, eta_fit_residual_rel=2.943e-03,
        eta_source_grade='reference', eta_source_citekey='sotiriadou2023',
        n_u_src=0.0003, n_u_slope_per_K=1.756e-05,
        n_r_lambda_nodes=(
            (476.5, 3.399e-05), (488, 3.399e-05), (496.5, 3.399e-05),
            (514.5, 1.878e-05), (632.8, 1.878e-05),
        ),
        n_r_temp_nodes=(
            (293.15, 3.399e-05), (298.15, 3.399e-05),
        ),
        eta_u_src_rel=0.0298551,
        eta_r_temp_nodes=(
            (253.15, 2.944e-03), (264.4, 2.944e-03), (275.65, 1.366e-03),
            (286.9, 1.351e-03), (298.15, 1.059e-03), (309.4, 1.178e-03),
            (320.65, 1.178e-03), (331.9, 2.218e-03), (343.15, 2.218e-03),
        ),
    ),
    # Glycerol. RI: Jakubczyk et al., Sci. Data 10, 894 (2023), Sellmeier
    #   + measured dn/dT(lambda), band-averaged. eta: Ferreira et al., J.
    #   Chem. Thermodyn. 113, 162 (2017), our VFT fit to their measured
    #   room-T data (their published VFT is low-T, ~9% low at 25 C).
    'glycerol': SolventProps(
        tier='primary',
        n_cauchy_A=1.4614682369866907, n_cauchy_B_um2=0.004189576889686445, n_cauchy_C_um4=-1.7582581227478834e-05,
        dn_dT_per_K=-0.000255635638865972, n_ref_temp_K=293.15,
        n_lambda_min_nm=400.0, n_lambda_max_nm=800.0,
        n_temp_min_K=274.15, n_temp_max_K=318.15,
        n_uncertainty=0.0015, n_fit_residual=4.032e-04,
        eta_vft_A_Pa_s=1.318176e-06, eta_vft_B_K=2171.0842, eta_vft_T0_K=137.0831,
        eta_temp_min_K=288.15, eta_temp_max_K=324.15,
        eta_uncertainty_rel=0.06, eta_fit_residual_rel=3.000e-02,
        eta_source_grade='reference', eta_source_citekey='ferreira2017',
        n_u_src=0.0003, n_u_slope_per_K=2.046e-05,
        n_r_lambda_nodes=(
            (400, 4.033e-04), (450, 4.033e-04), (500, 2.033e-04),
            (550, 1.174e-04), (600, 1.255e-04), (650, 1.326e-04),
            (700, 1.437e-04), (750, 2.758e-04), (800, 2.758e-04),
        ),
        n_r_temp_nodes=(
            (274.15, 4.033e-04), (279.65, 4.033e-04), (285.15, 3.310e-04),
            (290.65, 2.587e-04), (296.15, 1.864e-04), (301.65, 1.409e-04),
            (307.15, 1.630e-04), (312.65, 2.082e-04), (318.15, 2.082e-04),
        ),
        eta_u_src_rel=0.0543069,
        eta_r_temp_nodes=(
            (292.89, 2.551e-02), (298.1, 2.551e-02), (302.59, 2.551e-02),
            (312.81, 1.592e-02), (323.88, 2.078e-03),
        ),
    ),
    # Benzene. RI: Moreels et al., Appl. Opt. 23, 3010 (1984), measured
    #   dispersion + dn/dT (20 & 25 C). eta: Avgeri et al., JPCRD 43,
    #   033103 (2014), recommended saturation table (VFT fit).
    'benzene': SolventProps(
        tier='primary',
        n_cauchy_A=1.4810366360119385, n_cauchy_B_um2=0.006042236548531766, n_cauchy_C_um4=0.00035674662759029203,
        dn_dT_per_K=-0.0006988000000000187, n_ref_temp_K=293.15,
        n_lambda_min_nm=476.0, n_lambda_max_nm=633.0,
        n_temp_min_K=288.15, n_temp_max_K=313.15,
        n_uncertainty=0.0016, n_fit_residual=4.334e-04,
        eta_vft_A_Pa_s=2.232002e-05, eta_vft_B_K=770.5117, eta_vft_T0_K=64.3463,
        eta_temp_min_K=288.15, eta_temp_max_K=340.0,
        eta_uncertainty_rel=0.02, eta_fit_residual_rel=1.364e-03,
        eta_source_grade='reference', eta_source_citekey='avgeri2014',
        n_u_src=0.0004, n_u_slope_per_K=3.494e-05,
        n_r_lambda_nodes=(
            (476.5, 1.944e-04), (488, 1.944e-04), (496.5, 1.408e-04),
            (514.5, 4.334e-04), (632.8, 4.334e-04),
        ),
        n_r_temp_nodes=(
            (293.15, 4.334e-04), (298.15, 4.334e-04),
        ),
        eta_u_src_rel=0.0199534,
        eta_r_temp_nodes=(
            (280, 1.161e-03), (300, 1.364e-03), (320, 1.364e-03),
            (340, 1.364e-03),
        ),
    ),
    # Cyclohexane. RI: Moreels et al., Appl. Opt. 23, 3010 (1984),
    #   measured dispersion + dn/dT. eta: Tariq et al., JPCRD 43,
    #   033101 (2014), recommended 0.1 MPa table (VFT fit).
    'cyclohexane': SolventProps(
        tier='primary',
        n_cauchy_A=1.4155997005019536, n_cauchy_B_um2=0.003633988494486867, n_cauchy_C_um4=4.111047166304371e-05,
        dn_dT_per_K=-0.0004444000000000378, n_ref_temp_K=293.15,
        n_lambda_min_nm=476.0, n_lambda_max_nm=633.0,
        n_temp_min_K=288.15, n_temp_max_K=313.15,
        n_uncertainty=0.0013, n_fit_residual=3.217e-05,
        eta_vft_A_Pa_s=1.348277e-05, eta_vft_B_K=1035.2561, eta_vft_T0_K=51.1805,
        eta_temp_min_K=288.15, eta_temp_max_K=350.0,
        eta_uncertainty_rel=0.02, eta_fit_residual_rel=9.518e-04,
        eta_source_grade='reference', eta_source_citekey='tariq2014',
        n_u_src=0.0002, n_u_slope_per_K=2.222e-05,
        n_r_lambda_nodes=(
            (476.5, 2.778e-05), (488, 3.218e-05), (496.5, 3.218e-05),
            (514.5, 3.218e-05), (632.8, 3.117e-05),
        ),
        n_r_temp_nodes=(
            (293.15, 3.218e-05), (298.15, 3.218e-05),
        ),
        eta_u_src_rel=0.0199773,
        eta_r_temp_nodes=(
            (290, 8.580e-04), (300, 8.580e-04), (310, 8.580e-04),
            (320, 9.480e-04), (330, 9.480e-04), (340, 9.518e-04),
            (350, 9.518e-04),
        ),
    ),
    # n-Hexane. RI: Kozma et al., JOSA B 22, 1479 (2005), Cauchy at 22 C;
    #   dn/dT BULK (Riddick, -5.42e-4/K). eta: Michailidou et al., JPCRD
    #   42, 033104 (2013), recommended saturation table (VFT fit).
    'hexane': SolventProps(
        tier='primary',
        n_cauchy_A=1.3655575826650064, n_cauchy_B_um2=0.0034102941620269703, n_cauchy_C_um4=1.600929406298142e-05,
        dn_dT_per_K=-0.000542, n_ref_temp_K=295.15,
        n_lambda_min_nm=400.0, n_lambda_max_nm=640.0,
        n_temp_min_K=288.15, n_temp_max_K=313.15,
        n_uncertainty=0.002, n_fit_residual=6.335e-06,
        n_dndt_grade='bulk',
        eta_vft_A_Pa_s=8.952677e-06, eta_vft_B_K=1218.6086, eta_vft_T0_K=-50.0000,
        eta_temp_min_K=270.0, eta_temp_max_K=340.0,
        eta_uncertainty_rel=0.021, eta_fit_residual_rel=5.371e-03,
        eta_source_grade='reference', eta_source_citekey='michailidou2013',
        n_u_src=0.0006, n_u_slope_per_K=8.130e-05,
        n_r_lambda_nodes=(
            (400, 6.335e-06), (430, 6.335e-06), (460, 3.797e-06),
            (490, 2.765e-06), (520, 2.874e-06), (550, 2.874e-06),
            (580, 2.840e-06), (610, 4.514e-06), (640, 4.514e-06),
        ),
        eta_u_src_rel=0.02,
        eta_r_temp_nodes=(
            (270, 4.159e-03), (280, 4.159e-03), (290, 3.400e-03),
            (300, 3.502e-03), (310, 3.502e-03), (320, 3.502e-03),
            (330, 5.371e-03), (340, 5.371e-03),
        ),
    ),
    # Methanol. RI: Moutzouris et al., Appl. Phys. B 116, 617 (2014),
    #   ext-Cauchy at 300 K; dn/dT -4.0e-4/K (El-Kashef 2000, measured).
    #   eta: Xiang et al., JPCRD 35, 1597 (2006), recommended sat table.
    'methanol': SolventProps(
        tier='primary',
        n_cauchy_A=1.3191738503809085, n_cauchy_B_um2=0.0025677347692291646, n_cauchy_C_um4=4.736189945955989e-05,
        dn_dT_per_K=-0.0004, n_ref_temp_K=300.0,
        n_lambda_min_nm=450.0, n_lambda_max_nm=700.0,
        n_temp_min_K=288.15, n_temp_max_K=323.15,
        n_uncertainty=0.0015, n_fit_residual=1.499e-05,
        eta_vft_A_Pa_s=4.937586e-06, eta_vft_B_K=1522.3696, eta_vft_T0_K=-25.6764,
        eta_temp_min_K=273.15, eta_temp_max_K=343.15,
        eta_uncertainty_rel=0.021, eta_fit_residual_rel=4.323e-04,
        eta_source_grade='reference', eta_source_citekey='xiang2006',
        n_u_src=0.0002, n_u_slope_per_K=3.200e-05,
        n_r_lambda_nodes=(
            (450, 1.089e-05), (481.25, 1.089e-05), (512.5, 7.241e-06),
            (543.75, 6.919e-06), (575, 7.389e-06), (606.25, 7.817e-06),
            (637.5, 7.817e-06), (668.75, 1.500e-05), (700, 1.500e-05),
        ),
        eta_u_src_rel=0.02,
        eta_r_temp_nodes=(
            (270, 1.779e-04), (280, 2.608e-04), (290, 2.608e-04),
            (300, 2.608e-04), (310, 2.593e-04), (320, 3.689e-04),
            (330, 3.689e-04), (340, 4.323e-04), (345, 4.323e-04),
        ),
    ),
    # Acetone. RI: Moreels et al., Appl. Opt. 23, 3010 (1984), measured
    #   dispersion + dn/dT (primary sources Cooper 1983 / Moreels 1984).
    #   eta: Sotiriadou et al., IJT 46, 3 (2024), 3-point sat table (VFT).
    'acetone': SolventProps(
        tier='primary',
        n_cauchy_A=1.3500335264629626, n_cauchy_B_um2=0.00292724945858236, n_cauchy_C_um4=8.061338972911811e-05,
        dn_dT_per_K=-0.0004484000000000723, n_ref_temp_K=293.15,
        n_lambda_min_nm=476.0, n_lambda_max_nm=633.0,
        n_temp_min_K=288.15, n_temp_max_K=313.15,
        n_uncertainty=0.0014, n_fit_residual=2.642e-05,
        eta_vft_A_Pa_s=1.051350e-05, eta_vft_B_K=1172.3153, eta_vft_T0_K=-50.0000,
        eta_temp_min_K=255.0, eta_temp_max_K=320.0,
        eta_uncertainty_rel=0.058, eta_fit_residual_rel=2.000e-02,
        eta_source_grade='reference', eta_source_citekey='sotiriadou2024acetone',
        n_u_src=0.0003, n_u_slope_per_K=2.242e-05,
        n_r_lambda_nodes=(
            (476.5, 2.311e-05), (488, 2.311e-05), (496.5, 2.311e-05),
            (514.5, 2.643e-05), (632.8, 2.643e-05),
        ),
        n_r_temp_nodes=(
            (293.15, 2.643e-05), (298.15, 2.643e-05),
        ),
        eta_u_src_rel=0.055,
        eta_r_temp_nodes=(
            (250, 1.557e-02), (350, 1.557e-02),
        ),
    ),
    # Carbon tetrachloride [estimate]. RI: Moreels 1984 (measured). eta:
    #   CRC 95th bulk table (VFT). CCl4 has a strong tetrahedral-symmetry
    #   depolarization -- a good SLS calibrant candidate.
    'ccl4': SolventProps(
        tier='estimate',
        n_cauchy_A=1.4470358189788315, n_cauchy_B_um2=0.0045805855873710275, n_cauchy_C_um4=5.199773869387237e-05,
        dn_dT_per_K=-0.0005696000000000826, n_ref_temp_K=293.15,
        n_lambda_min_nm=476.0, n_lambda_max_nm=633.0,
        n_temp_min_K=288.15, n_temp_max_K=313.15,
        n_uncertainty=0.0014, n_fit_residual=3.910e-05,
        eta_vft_A_Pa_s=7.115792e-06, eta_vft_B_K=1688.1023, eta_vft_T0_K=-50.0000,
        eta_temp_min_K=253.0, eta_temp_max_K=340.0,
        eta_uncertainty_rel=0.06, eta_fit_residual_rel=2.000e-02,
        eta_source_grade='bulk', eta_source_citekey='crc95',
        n_u_src=0.0002, n_u_slope_per_K=2.848e-05,
        n_r_lambda_nodes=(
            (476.5, 2.280e-05), (488, 2.280e-05), (496.5, 2.280e-05),
            (514.5, 3.910e-05), (632.8, 3.910e-05),
        ),
        n_r_temp_nodes=(
            (293.15, 3.910e-05), (298.15, 3.910e-05),
        ),
        eta_u_src_rel=0.05,
    ),
    # Ethylene glycol. RI: Jakubczyk 2023 (Sellmeier + measured
    #   dn/dT, band-averaged; measured 1-45 C). eta: Mebelli et al.,
    #   IJT 42, 116 (2021) reference correlation (saturation table,
    #   VFT). Both authoritative -> PRIMARY tier.
    'ethylene glycol': SolventProps(
        tier='primary',
        n_cauchy_A=1.4210992176225008, n_cauchy_B_um2=0.003815192042837697, n_cauchy_C_um4=-1.806423889449309e-05,
        dn_dT_per_K=-0.0002832777924418773, n_ref_temp_K=293.15,
        n_lambda_min_nm=400.0, n_lambda_max_nm=800.0,
        n_temp_min_K=274.15, n_temp_max_K=318.15,
        n_uncertainty=0.0015, n_fit_residual=4.093e-04,
        eta_vft_A_Pa_s=2.578858e-05, eta_vft_B_K=1013.4917, eta_vft_T0_K=141.7273,
        eta_temp_min_K=273.15, eta_temp_max_K=345.0,
        eta_uncertainty_rel=0.05, eta_fit_residual_rel=1.000e-02,
        eta_source_grade='reference', eta_source_citekey='mebelli2021',
        n_u_src=0.0003, n_u_slope_per_K=2.267e-05,
        n_r_lambda_nodes=(
            (400, 4.093e-04), (450, 4.093e-04), (500, 2.075e-04),
            (550, 1.167e-04), (600, 1.192e-04), (650, 1.334e-04),
            (700, 1.557e-04), (750, 2.718e-04), (800, 2.718e-04),
        ),
        n_r_temp_nodes=(
            (274.15, 4.093e-04), (279.65, 4.093e-04), (285.15, 3.275e-04),
            (290.65, 2.457e-04), (296.15, 1.640e-04), (301.65, 1.162e-04),
            (307.15, 1.650e-04), (312.65, 2.229e-04), (318.15, 2.229e-04),
        ),
        eta_u_src_rel=0.049,
        eta_r_temp_nodes=(
            (265, 2.134e-03), (290, 2.134e-03), (310, 2.134e-03),
            (330, 1.696e-03), (350, 1.696e-03),
        ),
    ),
    # DMSO [estimate]. RI: Li et al., Infrared Phys. Technol. 125, 104313
    #   (2022); freezes at 18.5 C -> box from 19 C. eta: Grande et al., JCED
    #   54 (2009), measured pure-component 298-318 K (VFT), stated 5e-4 mPa s
    #   (sec. 2 p. 652; located 07-29 -- face value, floor dominates). dn/dT
    #   -4.21e-4/K MEASURED (Akmarov, J. Appl. Spectrosc. 80, 610 (2013)).
    'dmso': SolventProps(
        tier='estimate',
        n_cauchy_A=1.4580943482744348, n_cauchy_B_um2=0.007472730011510397, n_cauchy_C_um4=1.2397072968739486e-05,
        dn_dT_per_K=-0.000421, n_ref_temp_K=296.15,
        n_lambda_min_nm=450.0, n_lambda_max_nm=700.0,
        n_temp_min_K=292.15, n_temp_max_K=313.15,
        n_uncertainty=0.0021, n_fit_residual=2.006e-05,
        eta_vft_A_Pa_s=2.713552e-06, eta_vft_B_K=2299.7607, eta_vft_T0_K=-50.0000,
        eta_temp_min_K=298.15, eta_temp_max_K=318.15,
        eta_uncertainty_rel=0.033, eta_fit_residual_rel=2.000e-02,
        eta_source_grade='measured', eta_source_citekey='grande2009',
        n_u_src=0.0015, n_u_slope_per_K=3.368e-05,
        n_r_lambda_nodes=(
            (450, 2.007e-05), (481.25, 2.007e-05), (512.5, 1.175e-05),
            (543.75, 8.985e-06), (575, 9.034e-06), (606.25, 9.034e-06),
            (637.5, 8.987e-06), (668.75, 1.443e-05), (700, 1.443e-05),
        ),
        eta_u_src_rel=0.0004,
        eta_r_temp_nodes=(
            (298.15, 5.032e-03), (303.15, 5.032e-03), (308.15, 5.267e-03),
            (313.15, 5.504e-03), (318.15, 5.504e-03),
        ),
    ),
    # DMF [estimate]. RI: Li 2022 (4-Cauchy overshoots the D-line ~3e-3 ->
    #   wide n_unc). eta: Watanabe et al., JCED 61, 4215 (2016), SI Table
    #   S4, Ur 2%, served rows 293-323 K (VFT; range clip + Nikam 2005
    #   supersession documented at _ESTIMATE_VISC). dn/dT -4.6e-4/K bulk.
    'dmf': SolventProps(
        tier='estimate',
        n_cauchy_A=1.4092423717835432, n_cauchy_B_um2=0.009103965262542445, n_cauchy_C_um4=-0.0003242900170044999,
        dn_dT_per_K=-0.00046, n_ref_temp_K=296.15,
        n_lambda_min_nm=450.0, n_lambda_max_nm=700.0,
        n_temp_min_K=288.15, n_temp_max_K=313.15,
        n_uncertainty=0.0041, n_fit_residual=6.219e-05,
        n_dndt_grade='bulk',
        eta_vft_A_Pa_s=1.533251e-05, eta_vft_B_K=1378.9013, eta_vft_T0_K=-50.0000,
        eta_temp_min_K=293.15, eta_temp_max_K=323.15,
        eta_uncertainty_rel=0.039, eta_fit_residual_rel=2.000e-02,
        eta_source_grade='measured', eta_source_citekey='watanabe2016',
        n_u_src=0.0037, n_u_slope_per_K=6.900e-05,
        n_r_lambda_nodes=(
            (450, 6.220e-05), (481.25, 6.220e-05), (512.5, 3.640e-05),
            (543.75, 2.785e-05), (575, 2.801e-05), (606.25, 2.801e-05),
            (637.5, 2.786e-05), (668.75, 4.472e-05), (700, 4.472e-05),
        ),
        eta_u_src_rel=0.02,
        eta_r_temp_nodes=(
            (293.15, 9.043e-03), (298.15, 9.043e-03), (303.15, 6.386e-03),
            (313.15, 5.310e-03), (323.15, 3.661e-03),
        ),
    ),
    # Dichloromethane [estimate]. RI: Li 2022 (A2>0/A3<0 signs are correct).
    #   eta: CRC bulk + Riddick 15 C point (VFT); DCM boils at 39.6 C ->
    #   box capped at 38 C. dn/dT -5.5e-4/K bulk (Riddick).
    'dcm': SolventProps(
        tier='estimate',
        n_cauchy_A=1.4096964317530076, n_cauchy_B_um2=0.0051683816644690515, n_cauchy_C_um4=9.907639869782044e-05,
        dn_dT_per_K=-0.00055, n_ref_temp_K=296.15,
        n_lambda_min_nm=450.0, n_lambda_max_nm=700.0,
        n_temp_min_K=288.15, n_temp_max_K=311.15,
        n_uncertainty=0.0022, n_fit_residual=1.205e-07,
        n_dndt_grade='bulk',
        eta_vft_A_Pa_s=2.065087e-04, eta_vft_B_K=64.4115, eta_vft_T0_K=205.2182,
        eta_temp_min_K=253.15, eta_temp_max_K=311.15,
        eta_uncertainty_rel=0.06, eta_fit_residual_rel=2.000e-02,
        eta_source_grade='bulk', eta_source_citekey='crc95+riddick1970',
        n_u_src=0.0013, n_u_slope_per_K=8.250e-05,
        n_r_lambda_nodes=(
            (450, 1.205e-07), (481.25, 1.205e-07), (512.5, 7.049e-08),
            (543.75, 5.394e-08), (575, 5.424e-08), (606.25, 5.424e-08),
            (637.5, 5.396e-08), (668.75, 8.660e-08), (700, 8.660e-08),
        ),
        eta_u_src_rel=0.05,
    ),
    # Ethyl acetate [estimate]. RI: Li 2022. eta: Djojoputro &
    #   Ismadji, JCED 50, 727 (2005), measured 293-343 K, stated
    #   0.004 mPa s -> <=1.5% (VFT; supersedes CRC bulk). dn/dT
    #   -4.9e-4/K bulk (Riddick).
    'ethyl acetate': SolventProps(
        tier='estimate',
        n_cauchy_A=1.3588404430949712, n_cauchy_B_um2=0.005135900630671554, n_cauchy_C_um4=-2.4631943919642754e-05,
        dn_dT_per_K=-0.00049, n_ref_temp_K=296.15,
        n_lambda_min_nm=450.0, n_lambda_max_nm=700.0,
        n_temp_min_K=288.15, n_temp_max_K=313.15,
        n_uncertainty=0.0021, n_fit_residual=4.741e-06,
        n_dndt_grade='bulk',
        eta_vft_A_Pa_s=1.059309e-05, eta_vft_B_K=1282.3061, eta_vft_T0_K=-50.0000,
        eta_temp_min_K=293.15, eta_temp_max_K=343.15,
        eta_uncertainty_rel=0.036, eta_fit_residual_rel=2.000e-02,
        eta_source_grade='measured', eta_source_citekey='djojoputro2005',
        n_u_src=0.001, n_u_slope_per_K=7.350e-05,
        n_r_lambda_nodes=(
            (450, 4.741e-06), (481.25, 4.741e-06), (512.5, 2.775e-06),
            (543.75, 2.123e-06), (575, 2.135e-06), (606.25, 2.135e-06),
            (637.5, 2.124e-06), (668.75, 3.409e-06), (700, 3.409e-06),
        ),
        eta_u_src_rel=0.015,
        eta_r_temp_nodes=(
            (293.15, 3.506e-03), (298.15, 3.506e-03), (303.15, 2.296e-03),
            (308.15, 3.132e-03), (313.15, 3.132e-03), (318.15, 3.132e-03),
            (323.15, 3.125e-03), (328.15, 2.452e-03), (333.15, 1.372e-03),
            (338.15, 5.065e-03), (343.15, 5.065e-03),
        ),
    ),
    # THF [estimate]. RI: Li 2022. eta: Sotiriadou et al., IJT 45, 123 (2024)
    #   recommended saturation table (VFT; the authors label it preliminary,
    #   6% -> estimate tier). dn/dT -4.4e-4/K bulk (Riddick).
    'thf': SolventProps(
        tier='estimate',
        n_cauchy_A=1.3921461241008641, n_cauchy_B_um2=0.005433516682806349, n_cauchy_C_um4=-2.271226832771013e-05,
        dn_dT_per_K=-0.00044, n_ref_temp_K=296.15,
        n_lambda_min_nm=450.0, n_lambda_max_nm=700.0,
        n_temp_min_K=288.15, n_temp_max_K=313.15,
        n_uncertainty=0.0018, n_fit_residual=4.933e-06,
        n_dndt_grade='bulk',
        eta_vft_A_Pa_s=1.196807e-05, eta_vft_B_K=1270.3315, eta_vft_T0_K=-50.0000,
        eta_temp_min_K=255.0, eta_temp_max_K=335.0,
        eta_uncertainty_rel=0.061, eta_fit_residual_rel=2.000e-02,
        eta_source_grade='reference', eta_source_citekey='sotiriadou2024thf',
        n_u_src=0.0006, n_u_slope_per_K=6.600e-05,
        n_r_lambda_nodes=(
            (450, 4.933e-06), (481.25, 4.933e-06), (512.5, 2.887e-06),
            (543.75, 2.209e-06), (575, 2.222e-06), (606.25, 2.222e-06),
            (637.5, 2.210e-06), (668.75, 3.547e-06), (700, 3.547e-06),
        ),
        eta_u_src_rel=0.06,
        eta_r_temp_nodes=(
            (250, 3.023e-03), (350, 3.023e-03),
        ),
    ),
    # Chloroform [estimate]. RI: Samoc 2003 form b (20 C). eta: CRC 95th
    #   bulk (VFT). dn/dT -5.9e-4/K bulk (Riddick; Samoc x-check -5.98e-4).
    'chloroform': SolventProps(
        tier='estimate',
        n_cauchy_A=1.4317372754780762, n_cauchy_B_um2=0.0052585346624343604, n_cauchy_C_um4=-8.724697828428173e-05,
        dn_dT_per_K=-0.00059, n_ref_temp_K=293.15,
        n_lambda_min_nm=476.0, n_lambda_max_nm=700.0,
        n_temp_min_K=288.15, n_temp_max_K=313.15,
        n_uncertainty=0.0022, n_fit_residual=8.051e-06,
        n_dndt_grade='bulk',
        eta_vft_A_Pa_s=3.185222e-05, eta_vft_B_K=799.7440, eta_vft_T0_K=15.0438,
        eta_temp_min_K=270.0, eta_temp_max_K=323.15,
        eta_uncertainty_rel=0.06, eta_fit_residual_rel=2.000e-02,
        eta_source_grade='bulk', eta_source_citekey='crc95',
        n_u_src=0.0003, n_u_slope_per_K=8.850e-05,
        n_r_lambda_nodes=(
            (476, 8.051e-06), (504, 8.051e-06), (532, 4.897e-06),
            (560, 4.027e-06), (588, 3.880e-06), (616, 3.884e-06),
            (644, 3.884e-06), (672, 6.193e-06), (700, 6.193e-06),
        ),
        eta_u_src_rel=0.05,
    ),
    # Carbon disulfide [estimate]. RI: Samoc 2003 form b (20 C, high
    #   dispersion). eta: CRC bulk + Riddick 15 C point (VFT). dn/dT
    #   -7.8e-4/K bulk (Riddick; Samoc x-check -7.91e-4).
    'cs2': SolventProps(
        tier='estimate',
        n_cauchy_A=1.5825714219445437, n_cauchy_B_um2=0.013941344123834853, n_cauchy_C_um4=0.0006055412873567766,
        dn_dT_per_K=-0.00078, n_ref_temp_K=293.15,
        n_lambda_min_nm=476.0, n_lambda_max_nm=700.0,
        n_temp_min_K=288.15, n_temp_max_K=313.15,
        n_uncertainty=0.0027, n_fit_residual=7.165e-05,
        n_dndt_grade='bulk',
        eta_vft_A_Pa_s=2.735066e-05, eta_vft_B_K=889.6147, eta_vft_T0_K=-50.0000,
        eta_temp_min_K=270.0, eta_temp_max_K=300.0,
        eta_uncertainty_rel=0.06, eta_fit_residual_rel=2.000e-02,
        eta_source_grade='bulk', eta_source_citekey='crc95+riddick1970',
        n_u_src=0.0004, n_u_slope_per_K=1.170e-04,
        n_r_lambda_nodes=(
            (476, 7.166e-05), (504, 7.166e-05), (532, 4.271e-05),
            (560, 3.257e-05), (588, 3.102e-05), (616, 3.102e-05),
            (644, 3.086e-05), (672, 4.688e-05), (700, 4.688e-05),
        ),
        eta_u_src_rel=0.05,
    ),
    # Acetonitrile [estimate]. RI: Moutzouris 2014 (n^2 at 300 K) +
    #   its -3.4e-4/K estimate. eta: Grande et al., JCED 54 (2009),
    #   measured pure-component 298-318 K (VFT), stated 5e-4 mPa s
    #   (sec. 2 p. 652; face value, floor dominates). Zhang 2018 was
    #   weighed and reverted -- ThermoML population outlier at 318 K.
    'acetonitrile': SolventProps(
        tier='estimate',
        n_cauchy_A=1.334530152490182, n_cauchy_B_um2=0.0013452287122629697, n_cauchy_C_um4=0.00023313865005241243,
        dn_dT_per_K=-0.00034, n_ref_temp_K=300.0,
        n_lambda_min_nm=450.0, n_lambda_max_nm=700.0,
        n_temp_min_K=288.15, n_temp_max_K=313.15,
        n_uncertainty=0.0021, n_fit_residual=4.219e-06,
        n_dndt_grade='bulk',
        eta_vft_A_Pa_s=6.195471e-05, eta_vft_B_K=317.3786, eta_vft_T0_K=112.3950,
        eta_temp_min_K=298.15, eta_temp_max_K=318.15,
        eta_uncertainty_rel=0.033, eta_fit_residual_rel=2.000e-02,
        eta_source_grade='measured', eta_source_citekey='grande2009',
        n_u_src=0.0015, n_u_slope_per_K=5.100e-05,
        n_r_lambda_nodes=(
            (450, 3.717e-06), (481.25, 3.717e-06), (512.5, 2.326e-06),
            (543.75, 2.111e-06), (575, 2.213e-06), (606.25, 2.273e-06),
            (637.5, 2.273e-06), (668.75, 4.219e-06), (700, 4.219e-06),
        ),
        eta_u_src_rel=0.002,
        eta_r_temp_nodes=(
            (298.15, 2.804e-03), (303.15, 2.804e-03), (308.15, 2.804e-03),
            (313.15, 1.827e-03), (318.15, 6.250e-04),
        ),
    ),
}


# ===========================================================================
# Helpers
# ===========================================================================

_C_TO_K = 273.15


def _lookup(name: str) -> SolventProps:
    """Return the record for ``name``, matched caselessly, or raise a helpful ValueError.

    EXACT match after strip + casefold -- no alias vocabulary, because the program
    does not guess what a solvent name means (see core.data_models). A name picked
    from the library always hits, since every table key below is already a lowercase
    canonical name. A typed or instrument-supplied name generally does NOT, and that
    is the correct outcome: the caller falls back to asking the user for refractive
    index and viscosity by hand rather than proposing a value for a solvent it only
    guessed at (architecture invariant 3).
    """
    rec = _SOLVENT_PROPS.get(casefold_identity_name(name))
    if rec is None:
        available = ', '.join(available_solvents())
        raise ValueError(
            f"No solvent-property data for {name!r}. "
            f"Available solvents: {available}."
        )
    return rec


# ===========================================================================
# Pure evaluators
# ===========================================================================

def _check_n_box(rec: SolventProps, name: str,
                 wavelength_nm: float, temperature_K: float) -> None:
    """Shared refractive-index range guard (raise, never extrapolate)."""
    if not (rec.n_lambda_min_nm <= wavelength_nm <= rec.n_lambda_max_nm):
        raise ValueError(
            f"Wavelength {wavelength_nm} nm is outside the refractive-index "
            f"validity range for {name!r} "
            f"({rec.n_lambda_min_nm:g}-{rec.n_lambda_max_nm:g} nm)."
        )
    if not (rec.n_temp_min_K <= temperature_K <= rec.n_temp_max_K):
        raise ValueError(
            f"Temperature {temperature_K - _C_TO_K:g} C is outside the "
            f"refractive-index validity range for {name!r} "
            f"({rec.n_temp_min_K - _C_TO_K:g}-{rec.n_temp_max_K - _C_TO_K:g} C)."
        )


def _check_eta_box(rec: SolventProps, name: str, temperature_K: float) -> None:
    """Shared viscosity availability + range guard (raise, never extrapolate)."""
    if not rec.has_viscosity:
        raise ValueError(f"Viscosity data is not available for {name!r}.")
    if not (rec.eta_temp_min_K <= temperature_K <= rec.eta_temp_max_K):
        raise ValueError(
            f"Temperature {temperature_K - _C_TO_K:g} C is outside the viscosity "
            f"validity range for {name!r} "
            f"({rec.eta_temp_min_K - _C_TO_K:g}-{rec.eta_temp_max_K - _C_TO_K:g} C)."
        )


def refractive_index_solvent(
    name: str, wavelength_nm: float, temperature_C: float
) -> float:
    """Refractive index of ``name`` at the given vacuum wavelength and temperature.

    Evaluates the unified Cauchy + linear-dn/dT form. Raises ``ValueError`` if the
    solvent is unknown or if the wavelength/temperature falls outside the record's
    validity box (never silently extrapolates).

    Two sources of record, one per half: the Cauchy dispersion is Samoc 2003 Eq. 3
    (p. 6170) truncated at three terms -- the truncation that paper itself performs
    and prints for toluene -- and the linear (dn/dT)(T - T_ref) term is Jakubczyk
    et al. 2023 Eq. 1 (p. 4). Per-solvent coefficients are refit from each solvent's
    own authoritative correlation, which is often neither of those two; the unified
    form is the house choice (Theory-and-Equations-Guide S5).
    """
    rec = _lookup(name)
    temperature_K = temperature_C + _C_TO_K
    _check_n_box(rec, name, wavelength_nm, temperature_K)
    lam_um2 = (wavelength_nm / 1000.0) ** 2
    return (rec.n_cauchy_A
            + rec.n_cauchy_B_um2 / lam_um2
            + rec.n_cauchy_C_um4 / lam_um2 ** 2
            + rec.dn_dT_per_K * (temperature_K - rec.n_ref_temp_K))


def viscosity_solvent(name: str, temperature_C: float) -> float:
    """Dynamic viscosity of ``name`` in Pa s at the given temperature.

    Evaluates the unified VFT form. Raises ``ValueError`` if the solvent is
    unknown, carries no viscosity block, or if the temperature is outside the
    record's validity box (never silently extrapolates).
    """
    rec = _lookup(name)
    temperature_K = temperature_C + _C_TO_K
    _check_eta_box(rec, name, temperature_K)
    return rec.eta_vft_A_Pa_s * math.exp(
        rec.eta_vft_B_K / (temperature_K - rec.eta_vft_T0_K))


# ===========================================================================
# Per-condition display uncertainty -- shown, NEVER propagated
# ===========================================================================

# Inter-laboratory reproducibility floors. A single lab's
# stated uncertainty is an *internal* estimate: it cannot contain the
# systematics that lab did not recognise. The observed inter-laboratory spread
# measures the unrecognised-systematic content of the measurement class, and
# our sources are members of that class -- so each band carries a floor sized
# from that spread, added in quadrature. FROZEN, disclosed constants: they were
# measured once from versioned snapshots and must NOT move when the
# refractiveindex.info or ThermoML snapshots update (re-freezing is a
# deliberate, documented decision, never an automatic re-read).
#
# N_INTERLAB_FLOOR: absolute sigma_n allowance, applied to every RI record (the
# library holds no critically-evaluated RI correlation -- all 19 records are
# single-lab papers, so the floor applies uniformly). Provenance: the median
# inter-author disagreement of published n(lambda, T) curves -- p50 = 1.20e-3
# over 204 qualifying nodes, 10 materials, authors brought to a common T with
# our own dn/dT (re-measured 2026-07-27; the same estimator as
# tests/test_ri_database_check.py:_DB_TYPICAL_SPREAD). The median of a
# max-min author envelope adopted as a +/- term is conservative by design
# (invariant #8 Clause A: the default display must never under-report).
# NB the twin constant: _DB_TYPICAL_SPREAD holds the same measured number under
# the OPPOSITE update policy -- it is re-derived on every snapshot bump, while
# this floor stays frozen. Divergence between the two after a bump is EXPECTED,
# not an inconsistency; re-freezing this floor to a new measurement is an
# owner-gated decision (it moves every displayed band and forces regeneration).
N_INTERLAB_FLOOR = 1.20e-3

# ETA_INTERLAB_FLOOR_REL: relative sigma_eta allowance for records whose
# viscosity source is NOT a critically-evaluated reference correlation
# (eta_source_grade 'measured'/'bulk'). 'reference' sources are exempt: an
# evaluated correlation has already done the inter-laboratory analysis, and
# widening it would double-count the evaluators' completed work. Provenance:
# the median inter-article p10-p90 spread of published dynamic-viscosity
# measurements over the six measurable non-reference records is 6.35 %, i.e.
# +/-3.2 % (ThermoML v2020-09-30). Adopted k:
# the native p10-p90 half-width, ~1.28 sigma under normality -- stated rather
# than inherited, and the populations are skewed so the conversion is
# indicative. Median, not mean (chloroform's 24 % width is an outlier of the
# set). CS2 and DCM are permanently floor-unmeasurable (0 articles / 1 point
# in the archive); they still receive this floor via their 'bulk' grade.
ETA_INTERLAB_FLOOR_REL = 0.032


def _eta_interlab_floor(grade: Optional[str]) -> float:
    """The grade-dispatched eta inter-laboratory floor -- THE single home of the
    exemption rule: a critically-evaluated ``'reference'`` correlation gets no
    floor; every other grade (``'measured'``/``'bulk'``, and ``None`` for a
    hand-built record, which by definition is not an evaluated correlation)
    gets ``ETA_INTERLAB_FLOOR_REL``. The runtime accessor and the generator's
    box-max/report code all dispatch through here."""
    return 0.0 if grade == 'reference' else ETA_INTERLAB_FLOOR_REL


def _floored_box_n(rec: SolventProps) -> float:
    """The box-wide n scalar with the never-under-report lift -- the SINGLE
    home of the composition (2026-07-30 review: previously four hand-synced
    max() copies): a regenerated scalar already contains its floor and passes
    through unchanged; a hand-built scalar below the floor is lifted to it.
    Called by the sigma fallback and by ``solvent_property_info``."""
    return max(rec.n_uncertainty, N_INTERLAB_FLOOR)


def _floored_box_eta(rec: SolventProps) -> float:
    """Eta twin of ``_floored_box_n``, dispatching the grade exemption
    through ``_eta_interlab_floor``."""
    return max(rec.eta_uncertainty_rel, _eta_interlab_floor(rec.eta_source_grade))


def _sigma_n_prefloor_terms(u_src: float, r_lam: float, r_temp: float,
                            slope_term: float) -> float:
    """The INTERNAL (pre-floor) sigma_n quadrature -- the single home of the
    model's term list. The runtime accessors and the generator's box-max
    mirror (``gen_solvent_coeffs._sigma_n_from_desc``) all compose from this,
    so a term added here propagates everywhere or nowhere."""
    return math.sqrt(u_src ** 2 + r_lam ** 2 + r_temp ** 2 + slope_term ** 2)


def _sigma_eta_prefloor_terms(u_src_rel: float, r_temp: float) -> float:
    """The INTERNAL (pre-floor) sigma_eta quadrature (see
    ``_sigma_n_prefloor_terms`` -- same single-home rationale)."""
    return math.sqrt(u_src_rel ** 2 + r_temp ** 2)


def _envelope_value(nodes: Optional[tuple], x: float) -> float:
    """Evaluate a piecewise-linear upper-envelope node tuple at ``x``.

    Clamped flat beyond the end nodes (the honest continuation where the source
    supplies no further truth); an absent envelope contributes nothing.
    """
    if not nodes:
        return 0.0
    if x <= nodes[0][0]:
        return nodes[0][1]
    if x >= nodes[-1][0]:
        return nodes[-1][1]
    for (x0, r0), (x1, r1) in zip(nodes, nodes[1:], strict=False):
        if x0 <= x <= x1:
            if x1 <= x0:
                return max(r0, r1)
            return r0 + (r1 - r0) * (x - x0) / (x1 - x0)
    return nodes[-1][1]


def solvent_uncertainty_n(
    name: str, wavelength_nm: float, temperature_C: float
) -> float:
    """Per-condition ABSOLUTE display uncertainty sigma_n(lambda, T).

    ``hypot`` of the INTERNAL component (``solvent_uncertainty_n_prefloor``:
    source accuracy, unified-fit residual envelopes in lambda and T, propagated
    dn/dT-slope allowance) with the uniform inter-laboratory floor
    ``N_INTERLAB_FLOOR`` (Theory-and-Equations-Guide sec. 5). Same range guards
    as ``refractive_index_solvent`` (raises outside the box, never
    extrapolates). Display-only per invariant #8 -- never propagated into any
    analysis SE. The box-wide ``n_uncertainty`` is this model's maximum over
    the validity box (floor included by the generator), so the per-condition
    value never exceeds it. A record without a descriptor falls back to
    ``max(box-wide scalar, floor)``: a regenerated scalar already contains the
    floor (max() then returns it unchanged -- no double count), while a
    hand-built scalar below the floor is LIFTED to it, so the documented
    uniform floor holds by construction rather than by docstring.
    """
    rec = _lookup(name)
    temperature_K = temperature_C + _C_TO_K
    _check_n_box(rec, name, wavelength_nm, temperature_K)
    if rec.n_u_src is None:
        return _floored_box_n(rec)
    return math.hypot(
        _sigma_n_prefloor_terms(
            rec.n_u_src,
            _envelope_value(rec.n_r_lambda_nodes, wavelength_nm),
            _envelope_value(rec.n_r_temp_nodes, temperature_K),
            rec.n_u_slope_per_K * (temperature_K - rec.n_ref_temp_K)),
        N_INTERLAB_FLOOR)


def solvent_uncertainty_n_prefloor(
    name: str, wavelength_nm: float, temperature_C: float
) -> float:
    """The INTERNAL (pre-inter-lab-floor) component of ``solvent_uncertainty_n``.

    NOT a display quantity: it exists for gates that already carry their own
    inter-laboratory term and must not count it twice --
    ``tests/test_ri_database_check.py``'s Group-2 tolerance (internal band +
    its measured ``_DB_TYPICAL_SPREAD``) is the consumer. Keeping it here, next
    to the floor, means a change to the floor's composition (a lambda shape, a
    grade exemption) changes both halves in one place instead of silently
    breaking an arithmetic inversion in the test. No eta counterpart is needed:
    the viscosity oracle (CoolProp) deliberately gates on the full displayed
    band -- a band-containment claim -- not on the internal component.
    Descriptor branch is exact; the descriptor-less fallback strips the floor
    from the ceil-rounded scalar in quadrature, which is approximate -- gates
    needing exactness must assert the descriptor is present.
    """
    rec = _lookup(name)
    temperature_K = temperature_C + _C_TO_K
    _check_n_box(rec, name, wavelength_nm, temperature_K)
    if rec.n_u_src is None:
        return math.sqrt(max(0.0, rec.n_uncertainty ** 2 - N_INTERLAB_FLOOR ** 2))
    return _sigma_n_prefloor_terms(
        rec.n_u_src,
        _envelope_value(rec.n_r_lambda_nodes, wavelength_nm),
        _envelope_value(rec.n_r_temp_nodes, temperature_K),
        rec.n_u_slope_per_K * (temperature_K - rec.n_ref_temp_K))


def solvent_uncertainty_eta(name: str, temperature_C: float) -> float:
    """Per-condition RELATIVE display uncertainty sigma_eta_rel(T).

    ``hypot`` of the INTERNAL component (source's stated relative uncertainty
    floor + unified-VFT residual envelope in T) with the grade-dispatched
    inter-laboratory floor ``_eta_interlab_floor`` -- ``'reference'`` exempt
    (a critically-evaluated correlation has already absorbed the inter-lab
    analysis). A bulk/handbook source supplies no temperature shape, so its
    band stays honestly FLAT (at the record's floor plus the inter-lab
    allowance). Same guards as ``viscosity_solvent``. Display-only per
    invariant #8; the box-wide ``eta_uncertainty_rel`` is this model's maximum
    over the box (floor included by the generator). A record without a
    descriptor falls back to ``max(box-wide scalar, floor)`` -- a regenerated
    scalar already contains the floor (returned unchanged), a hand-built one
    below it is LIFTED to it, exactly as on the n side.
    """
    rec = _lookup(name)
    temperature_K = temperature_C + _C_TO_K
    _check_eta_box(rec, name, temperature_K)
    floor = _eta_interlab_floor(rec.eta_source_grade)
    if rec.eta_u_src_rel is None:
        return _floored_box_eta(rec)
    return math.hypot(
        _sigma_eta_prefloor_terms(
            rec.eta_u_src_rel,
            _envelope_value(rec.eta_r_temp_nodes, temperature_K)),
        floor)


# ===========================================================================
# Metadata accessors (used by both front-ends; carry no citations)
# ===========================================================================

def available_solvents(tier: Optional[str] = None) -> List[str]:
    """Canonical names with property data, optionally filtered by tier.

    ``tier=None`` returns all; ``'primary'`` returns the authoritative set for the
    Data-tab dropdown; ``'estimate'`` returns the weaker-data set.
    """
    names = sorted(_SOLVENT_PROPS)
    if tier is None:
        return names
    return [n for n in names if _SOLVENT_PROPS[n].tier == tier]


def solvent_tier(name: str) -> Optional[str]:
    """Tier of ``name`` (``'primary'``/``'estimate'``), or None if unknown."""
    rec = _SOLVENT_PROPS.get(casefold_identity_name(name))
    return rec.tier if rec is not None else None


def solvent_property_info(name: str) -> dict:
    """Display metadata for ``name``: tier, validity boxes, uncertainties, grade.

    Drives tooltips/badges/caveats without re-deriving values. Contains **no**
    paper names or citations. Raises ``ValueError`` for an unknown solvent.
    """
    rec = _lookup(name)
    # The box-max scalars published here share the never-under-report floor
    # guarantee of the sigma accessors (same max(scalar, floor) composition,
    # eta dispatched through the grade rule) -- this is the GUI's "(range max)"
    # caption path, the third publishing surface next to solvent_uncertainty_n
    # / _eta. Latent until a record ships an unfloored scalar, but the
    # guarantee must not depend on the generator always flooring first.
    info = {
        'tier': rec.tier,
        'n_lambda_min_nm': rec.n_lambda_min_nm,
        'n_lambda_max_nm': rec.n_lambda_max_nm,
        'n_temp_min_C': rec.n_temp_min_K - _C_TO_K,
        'n_temp_max_C': rec.n_temp_max_K - _C_TO_K,
        'n_uncertainty': _floored_box_n(rec),
        'n_dndt_grade': rec.n_dndt_grade,
        'has_viscosity': rec.has_viscosity,
    }
    if rec.has_viscosity:
        info.update({
            'eta_temp_min_C': rec.eta_temp_min_K - _C_TO_K,
            'eta_temp_max_C': rec.eta_temp_max_K - _C_TO_K,
            'eta_uncertainty_rel': _floored_box_eta(rec),
            'eta_source_grade': rec.eta_source_grade,
        })
    return info
