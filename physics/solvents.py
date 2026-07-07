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

from core.data_models import normalize_solvent_name


# ===========================================================================
# Record
# ===========================================================================

@dataclass(frozen=True)
class SolventProps:
    """Sourced coefficient record for one solvent, keyed by canonical name.

    The refractive-index block is always present; the viscosity block is optional
    (a record may carry n but no eta), signalled by ``eta_vft_A_Pa_s is None``.
    Per-solvent source provenance lives in the code comment beside each row in
    ``_SOLVENT_PROPS`` (mirroring ``constants.py:_TOLUENE_RVV_25C``); there are
    deliberately **no citation fields** on the record and none reach the UI.
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

    # ---- per-condition uncertainty descriptor (display-only; Spec 3) ----
    # Components of sigma_n(lambda, T) and sigma_eta_rel(T) -- see the Advanced
    # Guide sec. 12 for the model. u_src = the source's stated accuracy floor;
    # the node tuples are piecewise-linear UPPER envelopes ((x, r), ...) of the
    # unified-fit residual (clamped flat beyond the end nodes); u_slope = the
    # documented dn/dT-slope allowance. None -> that component is absent (a
    # record without a descriptor falls back to the flat box-wide scalar).
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
# provenance is the code comment on each row.
# ===========================================================================

_SOLVENT_PROPS: Dict[str, SolventProps] = {
    # Water. RI: Daimon & Masumura, Appl. Opt. 46, 3811 (2007), 4-term
    #   Sellmeier (20 C column) refit to Cauchy; dn/dT from the 19<->24 C
    #   spread. eta: Patek et al., JPCRD 38, 21 (2009) Eq. 7 (direct, 0.1 MPa).
    'water': SolventProps(
        tier='primary',
        n_cauchy_A=1.3230781735379857, n_cauchy_B_um2=0.0037969070122122616, n_cauchy_C_um4=-8.780016466781657e-05,
        dn_dT_per_K=-9.500716838736989e-05, n_ref_temp_K=293.15,
        n_lambda_min_nm=400.0, n_lambda_max_nm=800.0,
        n_temp_min_K=288.15, n_temp_max_K=318.15,
        n_uncertainty=0.0009, n_fit_residual=1.975e-04,
        eta_vft_A_Pa_s=3.208763e-05, eta_vft_B_K=480.6716, eta_vft_T0_K=153.5189,
        eta_temp_min_K=273.15, eta_temp_max_K=343.15,
        eta_uncertainty_rel=0.011, eta_fit_residual_rel=4.536e-03,
        eta_source_grade='reference',
        n_u_src=0.00015, n_u_slope_per_K=3.326e-05,
        n_r_lambda_nodes=(
            (400, 1.776e-04), (450, 1.776e-04), (500, 1.057e-04),
            (550, 7.806e-05), (600, 9.889e-05), (650, 9.889e-05),
            (700, 9.837e-05), (750, 1.976e-04), (800, 1.976e-04),
        ),
        n_r_temp_nodes=(
            (292.15, 1.976e-04), (293.15, 1.976e-04), (294.65, 1.877e-04),
            (297.15, 1.795e-04),
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
        n_uncertainty=0.0006, n_fit_residual=8.882e-16,
        eta_vft_A_Pa_s=1.899410e-05, eta_vft_B_K=938.2867, eta_vft_T0_K=20.0568,
        eta_temp_min_K=260.0, eta_temp_max_K=370.0,
        eta_uncertainty_rel=0.01, eta_fit_residual_rel=8.765e-03,
        eta_source_grade='reference',
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
        n_uncertainty=0.0005, n_fit_residual=3.398e-05,
        eta_vft_A_Pa_s=2.178149e-07, eta_vft_B_K=3850.0657, eta_vft_T0_K=-154.1331,
        eta_temp_min_K=253.15, eta_temp_max_K=343.15,
        eta_uncertainty_rel=0.03, eta_fit_residual_rel=2.943e-03,
        eta_source_grade='reference',
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
        n_uncertainty=0.0008, n_fit_residual=4.032e-04,
        eta_vft_A_Pa_s=1.318176e-06, eta_vft_B_K=2171.0842, eta_vft_T0_K=137.0831,
        eta_temp_min_K=288.15, eta_temp_max_K=324.15,
        eta_uncertainty_rel=0.06, eta_fit_residual_rel=3.000e-02,
        eta_source_grade='reference',
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
        n_uncertainty=0.0011, n_fit_residual=4.334e-04,
        eta_vft_A_Pa_s=2.232002e-05, eta_vft_B_K=770.5117, eta_vft_T0_K=64.3463,
        eta_temp_min_K=288.15, eta_temp_max_K=340.0,
        eta_uncertainty_rel=0.02, eta_fit_residual_rel=1.364e-03,
        eta_source_grade='reference',
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
        n_uncertainty=0.0005, n_fit_residual=3.217e-05,
        eta_vft_A_Pa_s=1.348277e-05, eta_vft_B_K=1035.2561, eta_vft_T0_K=51.1805,
        eta_temp_min_K=288.15, eta_temp_max_K=350.0,
        eta_uncertainty_rel=0.02, eta_fit_residual_rel=9.518e-04,
        eta_source_grade='reference',
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
        n_uncertainty=0.0016, n_fit_residual=6.335e-06,
        n_dndt_grade='bulk',
        eta_vft_A_Pa_s=8.952677e-06, eta_vft_B_K=1218.6086, eta_vft_T0_K=-50.0000,
        eta_temp_min_K=270.0, eta_temp_max_K=340.0,
        eta_uncertainty_rel=0.021, eta_fit_residual_rel=5.371e-03,
        eta_source_grade='reference',
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
        n_uncertainty=0.0008, n_fit_residual=1.499e-05,
        eta_vft_A_Pa_s=4.937586e-06, eta_vft_B_K=1522.3696, eta_vft_T0_K=-25.6764,
        eta_temp_min_K=273.15, eta_temp_max_K=343.15,
        eta_uncertainty_rel=0.021, eta_fit_residual_rel=4.323e-04,
        eta_source_grade='reference',
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
        n_uncertainty=0.0006, n_fit_residual=2.642e-05,
        eta_vft_A_Pa_s=1.051350e-05, eta_vft_B_K=1172.3153, eta_vft_T0_K=-50.0000,
        eta_temp_min_K=255.0, eta_temp_max_K=320.0,
        eta_uncertainty_rel=0.058, eta_fit_residual_rel=2.000e-02,
        eta_source_grade='reference',
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
    #   depolarisation -- a good SLS calibrant candidate.
    'ccl4': SolventProps(
        tier='estimate',
        n_cauchy_A=1.4470358189788315, n_cauchy_B_um2=0.0045805855873710275, n_cauchy_C_um4=5.199773869387237e-05,
        dn_dT_per_K=-0.0005696000000000826, n_ref_temp_K=293.15,
        n_lambda_min_nm=476.0, n_lambda_max_nm=633.0,
        n_temp_min_K=288.15, n_temp_max_K=313.15,
        n_uncertainty=0.0007, n_fit_residual=3.910e-05,
        eta_vft_A_Pa_s=7.115792e-06, eta_vft_B_K=1688.1023, eta_vft_T0_K=-50.0000,
        eta_temp_min_K=253.0, eta_temp_max_K=340.0,
        eta_uncertainty_rel=0.05, eta_fit_residual_rel=2.000e-02,
        eta_source_grade='bulk',
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
        n_uncertainty=0.0008, n_fit_residual=4.093e-04,
        eta_vft_A_Pa_s=2.578858e-05, eta_vft_B_K=1013.4917, eta_vft_T0_K=141.7273,
        eta_temp_min_K=273.15, eta_temp_max_K=345.0,
        eta_uncertainty_rel=0.05, eta_fit_residual_rel=1.000e-02,
        eta_source_grade='reference',
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
    #   54 (2009), measured pure-component 298-318 K (VFT). dn/dT -4.21e-4/K
    #   MEASURED (Akmarov et al., J. Appl. Spectrosc. 80, 610 (2013); sign neg).
    'dmso': SolventProps(
        tier='estimate',
        n_cauchy_A=1.4580943482744348, n_cauchy_B_um2=0.007472730011510397, n_cauchy_C_um4=1.2397072968739486e-05,
        dn_dT_per_K=-0.000421, n_ref_temp_K=296.15,
        n_lambda_min_nm=450.0, n_lambda_max_nm=700.0,
        n_temp_min_K=292.15, n_temp_max_K=313.15,
        n_uncertainty=0.0017, n_fit_residual=2.006e-05,
        eta_vft_A_Pa_s=2.713552e-06, eta_vft_B_K=2299.7607, eta_vft_T0_K=-50.0000,
        eta_temp_min_K=293.15, eta_temp_max_K=320.0,
        eta_uncertainty_rel=0.03, eta_fit_residual_rel=2.000e-02,
        eta_source_grade='measured',
        n_u_src=0.0015, n_u_slope_per_K=3.368e-05,
        n_r_lambda_nodes=(
            (450, 2.007e-05), (481.25, 2.007e-05), (512.5, 1.175e-05),
            (543.75, 8.985e-06), (575, 9.034e-06), (606.25, 9.034e-06),
            (637.5, 8.987e-06), (668.75, 1.443e-05), (700, 1.443e-05),
        ),
        eta_u_src_rel=0.03,
    ),
    # DMF [estimate]. RI: Li 2022 (4-Cauchy overshoots the D-line ~3e-3 ->
    #   wide n_unc). eta: Nikam & Kharat, JCED 50, 455 (2005), measured
    #   pure-component 298-313 K (VFT). dn/dT -4.6e-4/K bulk (Riddick).
    'dmf': SolventProps(
        tier='estimate',
        n_cauchy_A=1.4092423717835432, n_cauchy_B_um2=0.009103965262542445, n_cauchy_C_um4=-0.0003242900170044999,
        dn_dT_per_K=-0.00046, n_ref_temp_K=296.15,
        n_lambda_min_nm=450.0, n_lambda_max_nm=700.0,
        n_temp_min_K=288.15, n_temp_max_K=313.15,
        n_uncertainty=0.0039, n_fit_residual=6.219e-05,
        n_dndt_grade='bulk',
        eta_vft_A_Pa_s=6.656722e-05, eta_vft_B_K=485.8765, eta_vft_T0_K=103.0585,
        eta_temp_min_K=288.0, eta_temp_max_K=318.0,
        eta_uncertainty_rel=0.03, eta_fit_residual_rel=2.000e-02,
        eta_source_grade='measured',
        n_u_src=0.0037, n_u_slope_per_K=6.900e-05,
        n_r_lambda_nodes=(
            (450, 6.220e-05), (481.25, 6.220e-05), (512.5, 3.640e-05),
            (543.75, 2.785e-05), (575, 2.801e-05), (606.25, 2.801e-05),
            (637.5, 2.786e-05), (668.75, 4.472e-05), (700, 4.472e-05),
        ),
        eta_u_src_rel=0.03,
    ),
    # Dichloromethane [estimate]. RI: Li 2022 (A2>0/A3<0 signs are correct).
    #   eta: CRC bulk (VFT); DCM boils at 39.6 C -> box capped at 38 C.
    #   dn/dT -5.5e-4/K bulk (Riddick).
    'dcm': SolventProps(
        tier='estimate',
        n_cauchy_A=1.4096964317530076, n_cauchy_B_um2=0.0051683816644690515, n_cauchy_C_um4=9.907639869782044e-05,
        dn_dT_per_K=-0.00055, n_ref_temp_K=296.15,
        n_lambda_min_nm=450.0, n_lambda_max_nm=700.0,
        n_temp_min_K=288.15, n_temp_max_K=311.15,
        n_uncertainty=0.0018, n_fit_residual=1.205e-07,
        n_dndt_grade='bulk',
        eta_vft_A_Pa_s=2.065087e-04, eta_vft_B_K=64.4115, eta_vft_T0_K=205.2182,
        eta_temp_min_K=253.15, eta_temp_max_K=311.15,
        eta_uncertainty_rel=0.05, eta_fit_residual_rel=2.000e-02,
        eta_source_grade='bulk',
        n_u_src=0.0013, n_u_slope_per_K=8.250e-05,
        n_r_lambda_nodes=(
            (450, 1.205e-07), (481.25, 1.205e-07), (512.5, 7.049e-08),
            (543.75, 5.394e-08), (575, 5.424e-08), (606.25, 5.424e-08),
            (637.5, 5.396e-08), (668.75, 8.660e-08), (700, 8.660e-08),
        ),
        eta_u_src_rel=0.05,
    ),
    # Ethyl acetate [estimate]. RI: Li 2022. eta: CRC bulk (VFT).
    #   dn/dT -4.9e-4/K bulk (Riddick).
    'ethyl acetate': SolventProps(
        tier='estimate',
        n_cauchy_A=1.3588404430949712, n_cauchy_B_um2=0.005135900630671554, n_cauchy_C_um4=-2.4631943919642754e-05,
        dn_dT_per_K=-0.00049, n_ref_temp_K=296.15,
        n_lambda_min_nm=450.0, n_lambda_max_nm=700.0,
        n_temp_min_K=288.15, n_temp_max_K=313.15,
        n_uncertainty=0.0017, n_fit_residual=4.741e-06,
        n_dndt_grade='bulk',
        eta_vft_A_Pa_s=1.437050e-05, eta_vft_B_K=1000.5685, eta_vft_T0_K=2.3159,
        eta_temp_min_K=270.0, eta_temp_max_K=323.15,
        eta_uncertainty_rel=0.05, eta_fit_residual_rel=2.000e-02,
        eta_source_grade='bulk',
        n_u_src=0.001, n_u_slope_per_K=7.350e-05,
        n_r_lambda_nodes=(
            (450, 4.741e-06), (481.25, 4.741e-06), (512.5, 2.775e-06),
            (543.75, 2.123e-06), (575, 2.135e-06), (606.25, 2.135e-06),
            (637.5, 2.124e-06), (668.75, 3.409e-06), (700, 3.409e-06),
        ),
        eta_u_src_rel=0.05,
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
        n_uncertainty=0.0013, n_fit_residual=4.933e-06,
        n_dndt_grade='bulk',
        eta_vft_A_Pa_s=1.196807e-05, eta_vft_B_K=1270.3315, eta_vft_T0_K=-50.0000,
        eta_temp_min_K=255.0, eta_temp_max_K=335.0,
        eta_uncertainty_rel=0.061, eta_fit_residual_rel=2.000e-02,
        eta_source_grade='reference',
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
        n_uncertainty=0.0018, n_fit_residual=8.051e-06,
        n_dndt_grade='bulk',
        eta_vft_A_Pa_s=3.185222e-05, eta_vft_B_K=799.7440, eta_vft_T0_K=15.0438,
        eta_temp_min_K=270.0, eta_temp_max_K=323.15,
        eta_uncertainty_rel=0.05, eta_fit_residual_rel=2.000e-02,
        eta_source_grade='bulk',
        n_u_src=0.0003, n_u_slope_per_K=8.850e-05,
        n_r_lambda_nodes=(
            (476, 8.051e-06), (504, 8.051e-06), (532, 4.897e-06),
            (560, 4.027e-06), (588, 3.880e-06), (616, 3.884e-06),
            (644, 3.884e-06), (672, 6.193e-06), (700, 6.193e-06),
        ),
        eta_u_src_rel=0.05,
    ),
    # Carbon disulfide [estimate]. RI: Samoc 2003 form b (20 C, high
    #   dispersion). eta: CRC bulk (VFT). dn/dT -7.8e-4/K bulk (Riddick;
    #   Samoc x-check -7.91e-4).
    'cs2': SolventProps(
        tier='estimate',
        n_cauchy_A=1.5825714219445437, n_cauchy_B_um2=0.013941344123834853, n_cauchy_C_um4=0.0006055412873567766,
        dn_dT_per_K=-0.00078, n_ref_temp_K=293.15,
        n_lambda_min_nm=476.0, n_lambda_max_nm=700.0,
        n_temp_min_K=288.15, n_temp_max_K=313.15,
        n_uncertainty=0.0024, n_fit_residual=7.165e-05,
        n_dndt_grade='bulk',
        eta_vft_A_Pa_s=2.735066e-05, eta_vft_B_K=889.6147, eta_vft_T0_K=-50.0000,
        eta_temp_min_K=270.0, eta_temp_max_K=300.0,
        eta_uncertainty_rel=0.05, eta_fit_residual_rel=2.000e-02,
        eta_source_grade='bulk',
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
    #   measured pure-component 298-318 K (VFT; supersedes the earlier
    #   2-point Riddick/CRC data -- CRC 25 C ran ~8% high).
    'acetonitrile': SolventProps(
        tier='estimate',
        n_cauchy_A=1.334530152490182, n_cauchy_B_um2=0.0013452287122629697, n_cauchy_C_um4=0.00023313865005241243,
        dn_dT_per_K=-0.00034, n_ref_temp_K=300.0,
        n_lambda_min_nm=450.0, n_lambda_max_nm=700.0,
        n_temp_min_K=288.15, n_temp_max_K=313.15,
        n_uncertainty=0.0017, n_fit_residual=4.219e-06,
        n_dndt_grade='bulk',
        eta_vft_A_Pa_s=6.195471e-05, eta_vft_B_K=317.3786, eta_vft_T0_K=112.3950,
        eta_temp_min_K=288.0, eta_temp_max_K=318.0,
        eta_uncertainty_rel=0.03, eta_fit_residual_rel=2.000e-02,
        eta_source_grade='measured',
        n_u_src=0.0015, n_u_slope_per_K=5.100e-05,
        n_r_lambda_nodes=(
            (450, 3.717e-06), (481.25, 3.717e-06), (512.5, 2.326e-06),
            (543.75, 2.111e-06), (575, 2.213e-06), (606.25, 2.273e-06),
            (637.5, 2.273e-06), (668.75, 4.219e-06), (700, 4.219e-06),
        ),
        eta_u_src_rel=0.03,
    ),
}


# ===========================================================================
# Helpers
# ===========================================================================

_C_TO_K = 273.15


def _lookup(name: str) -> SolventProps:
    """Canonicalise ``name`` and return its record, or raise a helpful ValueError."""
    canonical = normalize_solvent_name(name)
    rec = _SOLVENT_PROPS.get(canonical)
    if rec is None:
        available = ', '.join(available_solvents())
        raise ValueError(
            f"No solvent-property data for {name!r} (canonical {canonical!r}). "
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
# Per-condition display uncertainty (Spec 3) -- shown, NEVER propagated
# ===========================================================================

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

    The quadrature of the record's descriptor components (Advanced Guide sec. 12):
    the source's stated accuracy floor, the unified-fit residual envelopes in
    lambda and T, and the propagated dn/dT-slope allowance. Same range guards as
    ``refractive_index_solvent`` (raises outside the box, never extrapolates).
    Display-only per invariant #8 -- never propagated into any analysis SE. The
    box-wide ``n_uncertainty`` is this model's maximum over the validity box, so
    the per-condition value never exceeds it. A record without a descriptor
    falls back to that flat box-wide scalar.
    """
    rec = _lookup(name)
    temperature_K = temperature_C + _C_TO_K
    _check_n_box(rec, name, wavelength_nm, temperature_K)
    if rec.n_u_src is None:
        return rec.n_uncertainty
    return math.sqrt(
        rec.n_u_src ** 2
        + _envelope_value(rec.n_r_lambda_nodes, wavelength_nm) ** 2
        + _envelope_value(rec.n_r_temp_nodes, temperature_K) ** 2
        + (rec.n_u_slope_per_K * (temperature_K - rec.n_ref_temp_K)) ** 2)


def solvent_uncertainty_eta(name: str, temperature_C: float) -> float:
    """Per-condition RELATIVE display uncertainty sigma_eta_rel(T).

    Quadrature of the source's stated relative uncertainty floor and the
    unified-VFT residual envelope in T. A bulk/handbook source supplies no
    temperature shape, so its band is honestly FLAT at the record's floor.
    Same guards as ``viscosity_solvent``. Display-only per invariant #8; the
    box-wide ``eta_uncertainty_rel`` is this model's maximum over the box.
    """
    rec = _lookup(name)
    temperature_K = temperature_C + _C_TO_K
    _check_eta_box(rec, name, temperature_K)
    if rec.eta_u_src_rel is None:
        return rec.eta_uncertainty_rel
    return math.sqrt(
        rec.eta_u_src_rel ** 2
        + _envelope_value(rec.eta_r_temp_nodes, temperature_K) ** 2)


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
    rec = _SOLVENT_PROPS.get(normalize_solvent_name(name))
    return rec.tier if rec is not None else None


def solvent_property_info(name: str) -> dict:
    """Display metadata for ``name``: tier, validity boxes, uncertainties, grade.

    Drives tooltips/badges/caveats without re-deriving values. Contains **no**
    paper names or citations. Raises ``ValueError`` for an unknown solvent.
    """
    rec = _lookup(name)
    info = {
        'tier': rec.tier,
        'n_lambda_min_nm': rec.n_lambda_min_nm,
        'n_lambda_max_nm': rec.n_lambda_max_nm,
        'n_temp_min_C': rec.n_temp_min_K - _C_TO_K,
        'n_temp_max_C': rec.n_temp_max_K - _C_TO_K,
        'n_uncertainty': rec.n_uncertainty,
        'n_dndt_grade': rec.n_dndt_grade,
        'has_viscosity': rec.has_viscosity,
    }
    if rec.has_viscosity:
        info.update({
            'eta_temp_min_C': rec.eta_temp_min_K - _C_TO_K,
            'eta_temp_max_C': rec.eta_temp_max_K - _C_TO_K,
            'eta_uncertainty_rel': rec.eta_uncertainty_rel,
            'eta_source_grade': rec.eta_source_grade,
        })
    return info
