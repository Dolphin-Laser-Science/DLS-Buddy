"""
analysis/dls
============

Dynamic light scattering (DLS) correlation-function analysis.

This package recovers particle dynamics from a measured intensity autocorrelation
function g2(tau) - 1. It provides six analysis methods in two families:

  Parametric models (fit a fixed functional form):
    - fit_cumulants            1st / 2nd / 3rd order cumulant expansion
    - fit_single_exponential   single decay mode
    - fit_double_exponential   two discrete decay modes
    - fit_kww                  Kohlrausch-Williams-Watts (stretched exponential)

  Distribution methods (recover a full decay-rate distribution):
    - fit_nnls                 non-negative least squares (no smoothing)
    - fit_contin               regularized (smoothed) inversion, L-curve alpha
    - fit_lognormal            single-mode lognormal (parametric distribution)

Shared front-end: multi-angle Gamma/q^2, concentration extrapolation, replicate
averaging.

Package layout (a split of the former single dls.py module; the public
API and import surface are unchanged -- everything below is re-exported here, so
`from analysis.dls import fit_cumulants` keeps working):

    _common.py        shared helpers (q, viscosity, tau-window, rh-convert, rms,
                      distribution baseline/beta/grid)
    cumulants.py      fit_cumulants (linear Koppel 1972 + nonlinear Frisken 2001)
    exponentials.py   single / double / KWW fits
    distributions.py  NNLS / CONTIN / lognormal + peaks + Rh<->Gamma axis helper
    angular.py        Gamma-q^2, concentration extrapolation, Rh<->Gamma converters
    replicate.py      true-replicate correlogram averaging

Formulation and conventions
---------------------------
The field autocorrelation is g1(tau) = sum_i A_i exp(-Gamma_i tau), where the
decay rate of a mode is Gamma = D q^2, D is the translational diffusion
coefficient, and q is the scattering vector. The MEASURED quantity is related by
the Siegert relation:

    g2(tau) - 1 = beta |g1(tau)|^2

where beta is the coherence factor (intercept). The PARAMETRIC fits (cumulants,
single/double/KWW exponentials) fit g2(tau) - 1 directly (beta floating), rather
than linearizing to g1 by taking a square root, because the noise lives on g2 - 1
and a naive square-root transform distorts that noise and fails where g2 - 1 dips
negative in the baseline. The DISTRIBUTION methods (NNLS/CONTIN/lognormal) DO invert
the field ACF g1 -- the accepted-standard Laplace inversion -- but recover it with a
SIGN-PRESERVING square root (so the baseline noise stays zero-mean, not rectified
into a pedestal) and statistically WEIGHTED residuals; see analysis/dls/distributions.py.

A consequence of the Siegert relation is a FACTOR OF 2 in the decay exponent of the
PARAMETRIC fits: for a single mode, g2 - 1 = beta exp(-2 Gamma tau), and those fits
carry the factor of 2 explicitly. The distribution methods instead invert g1 with a
single-Gamma kernel exp(-Gamma tau) (no factor of 2). Either way we always report
Gamma as the physical g1 decay rate, from which D = Gamma / q^2 and the hydrodynamic
radius follows by Stokes-Einstein, Rh = kB T / (6 pi eta D).

The cumulant relation (Koppel 1972; Frisken 2001; Salazar et al. 2023 Eqs. 8-9)
in this convention is:

    ln(g2 - 1) = ln(beta) - 2 Gamma tau + mu2 tau^2 - (mu3 / 3) tau^3 + ...

so a weighted polynomial fit of ln(g2 - 1) versus tau yields beta, Gamma, and the
moments directly. PDI = mu2 / Gamma^2; the cumulant expansion is considered
reliable only for PDI <= 0.3.

References (in project knowledge)
---------------------------------
  Provencher 1982, Comput. Phys. Commun. 27, 213 and 229  (CONTIN)
  Salazar et al. 2023, the user-friendly DLS GUI paper, + its DLS_GUI.py code.
    Used here as a cross-check, NOT as an authority: their SLSQP sum-constrained
    solver and some implementation details are deliberately not followed (see the
    distribution methods). Where the GUI and the Provencher papers or
    first-principles formulation diverge, this package follows the latter.
  Brookhaven Particle Explorer and TurboCorr DLSW manuals
  Koppel 1972; Frisken 2001 (cumulants)

Design contract
---------------
Every function is PURE: it takes a DLSMeasurement (and parameters) and returns a
result object. No function plots, writes files, or mutates inputs. Result objects
carry plot-ready arrays (fitted curve, residuals) so plotting/plots.py can render
without recomputation.
"""

from __future__ import annotations

from analysis.dls._common import _apply_tau_window
from analysis.dls.cumulants import (
    CUMULANT_MIN_RH_NM,
    CUMULANT_PDI_VALIDITY_LIMIT,
    CumulantResult,
    fit_cumulants,
)
from analysis.dls.exponentials import (
    DoubleExponentialResult,
    ExponentialMode,
    KWWResult,
    SingleExponentialResult,
    fit_double_exponential,
    fit_kww,
    fit_single_exponential,
)
from analysis.dls.distributions import (
    ContinResult,
    DistributionPeak,
    DistributionResult,
    LCurveResult,
    distribution_axis,
    find_distribution_peaks,
    fit_contin,
    fit_lognormal,
    fit_nnls,
)
from analysis.dls.angular import (
    ConcentrationExtrapolationResult,
    GammaQ2Result,
    analyze_gamma_q2,
    extrapolate_diffusion_vs_concentration,
    gamma_per_measurement,
    gamma_to_rh_nm,
    rh_nm_to_gamma,
)
from analysis.dls.replicate import (
    AveragedCorrelogramResult,
    average_replicate_correlograms,
)

__all__ = [
    # constants
    'CUMULANT_MIN_RH_NM',
    'CUMULANT_PDI_VALIDITY_LIMIT',
    # cumulants
    'CumulantResult',
    'fit_cumulants',
    # exponentials
    'ExponentialMode',
    'SingleExponentialResult',
    'DoubleExponentialResult',
    'KWWResult',
    'fit_single_exponential',
    'fit_double_exponential',
    'fit_kww',
    # distributions
    'DistributionResult',
    'LCurveResult',
    'ContinResult',
    'DistributionPeak',
    'fit_nnls',
    'fit_lognormal',
    'fit_contin',
    'distribution_axis',
    'find_distribution_peaks',
    # angular / front-end
    'GammaQ2Result',
    'ConcentrationExtrapolationResult',
    'gamma_to_rh_nm',
    'rh_nm_to_gamma',
    'analyze_gamma_q2',
    'extrapolate_diffusion_vs_concentration',
    'gamma_per_measurement',
    # replicate averaging
    'AveragedCorrelogramResult',
    'average_replicate_correlograms',
    # private helper used by the headless DLS validators
    '_apply_tau_window',
]
